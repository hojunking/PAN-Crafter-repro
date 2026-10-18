"""Isolated frozen-A Teacher continuation and matched Stage 2 tail engines.

No old trainer is patched.  Teacher continuation retains the source Teacher's
Adam moments, original H50 scheduler position and RR sampler.  Tail forks retain
the parent's Student moments/sampler, but explicitly restart a common low-LR
schedule.  No tail constructs a Teacher or a conditional-q calibration table.
All checkpoint imports are SHA-pinned trusted local experiment artifacts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import math
from pathlib import Path

import torch
from torch import nn

from directshift48.method import (DirectPANAligner, DirectTrainingSources, ScharrAbsolute,
                                  epsilon_loss, sample_epsilon)
from donor30.artifacts import _cpu, _same, _save_immutable, capture_rng, restore_rng
from donor30.masking import masked_mean, mask_manifest
from donor30.plan import LOSS_MASK_POLICY
from pa.warp import warp_pan, warp_support_mask
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_method.data import require_sha256
from panda_method.edge_data import EdgeDataSources
from panda_method.reconstruction import Reconstructor, build_reconstructor
from panda_stage2_condq.donor import LoadedDonor, load_donor, _role_state
from panda_stage2_condq.method import ConditionalQStage2, native_support
from panda_stage2_condq.sources import RegisteredRRTrain, validate_training_batch
from panda_stage2_condq.spec import Stage2Spec, MASK2_POLICY
from panda_stage2_condq.stream import StreamSpec, TrainStream
from panda_stage2_condq.training import (StudentTrainer, TrainingSpec,
    configure_precision, _finite_tree, _norm)
from ta2.evaluation import file_sha


SCHEMA = "PANDA30R_COMPLETE_CHECKPOINT_V1"
TRAINER_SCHEMA = "PANDA30R_NATIVE_TRAINER_V1"
KINDS = ("mature_teacher", "tail_gt", "tail_fc2_eps", "tail_fc2_abs")
HEAD_UPDATES = 5000


def _registered(sources):
    registered = RegisteredRRTrain(sources) if isinstance(sources, EdgeDataSources) else sources
    if not isinstance(registered, RegisteredRRTrain):
        raise TypeError("Pinned RegisteredRRTrain or EdgeDataSources required")
    registered.verify()
    return registered


def _read(path, sha):
    require_sha256(sha, "explicit trusted local checkpoint SHA256")
    path = Path(path).resolve(strict=True)
    if not path.is_file() or file_sha(path) != sha:
        raise ValueError("Checkpoint SHA mismatch")
    stat = path.stat()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if path.stat() != stat or file_sha(path) != sha:
        raise ValueError("Checkpoint changed during import")
    if not isinstance(payload, dict):
        raise ValueError("Structured local checkpoint required")
    return payload


def _validate_config(cfg):
    if cfg.get("engine_kind") not in KINDS or cfg.get("dataset") != "WV3":
        raise ValueError("Explicit native WV3 engine required")
    for key in ("run_id", "campaign_id", "server"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ValueError("Explicit campaign/run/server identity required")
    for key in ("attempt", "seed", "start_update", "planned_horizon"):
        if type(cfg.get(key)) is not int or cfg[key] < (1 if key in ("attempt", "planned_horizon") else 0):
            raise ValueError("Invalid native run counter: " + key)
    training = TrainingSpec(**cfg["training_spec"])
    if (training.weight_decay != .01 or training.betas != (.9, .999)
            or training.adam_eps != 1e-8 or training.min_lr != 0):
        raise ValueError("Unchanged FP32 AdamW .01/(.9,.999)/1e-8 required")
    if cfg["engine_kind"] == "mature_teacher":
        if (not 0 < cfg["start_update"] < 50000
                or cfg["planned_horizon"] != 50000 - cfg["start_update"]
                or training.total_updates != 50000 or training.warmup_updates != 100):
            raise ValueError("Teacher continuation must preserve original global H50/100-warmup schedule")
    elif (cfg["start_update"] not in (20000, 40000)
            or cfg["planned_horizon"] != 50000 - cfg["start_update"]
            or training.total_updates != cfg["planned_horizon"]
            or training.warmup_updates != 0 or training.lr != 1e-5):
        raise ValueError("Tail must fork exactly 20K/40K with common 1e-5 cosine to 50K")
    active = cfg["engine_kind"] in ("tail_fc2_eps", "tail_fc2_abs")
    declared = {"lr_A":1e-6 if active else 0., "lambda_rec":1.,
        "lambda_epsilon":.003 if active else 0.,
        "lambda_abs":.2 if cfg["engine_kind"] == "tail_fc2_abs" else 0.,
        "adapt_updates":HEAD_UPDATES if active else 0,
        "epsilon_radius_rr":3., "epsilon_distribution":"UNIFORM_AREA_DISK"}
    for key,value in declared.items():
        if key in cfg and cfg[key] != value:
            raise ValueError("Declared native setting differs from executed objective/schedule: " + key)
    return training


def _state_hash(stage):
    roles = {"A": state_dict_sha256(stage.aligner)}
    if stage.teacher is not None:
        roles["T"] = state_dict_sha256(stage.teacher)
    if stage.student is not None:
        roles["S"] = state_dict_sha256(stage.student)
    return canonical_sha256(roles)


class NativeAdaptation(nn.Module):
    """Only existing reconstruction, Scharr and R3 epsilon objectives.

    Inference never changes ``requires_grad``; in particular it does not use
    StudentInference, whose constructor freezes a shared live Aligner.
    """
    def __init__(self, aligner, reconstruction, cfg, *, donor_binding, parent_sha):
        super().__init__()
        _validate_config(cfg)
        if not isinstance(aligner, DirectPANAligner) or not isinstance(reconstruction, Reconstructor):
            raise TypeError("Existing DirectPANAligner and Reconstructor required")
        self.engine_kind = cfg["engine_kind"]
        self.spec = Stage2Spec("WV3", "gt_only", cfg["seed"], 0., 0., 0., 1e-6,
                               mask_policy=MASK2_POLICY)
        if reconstruction.spec != self.spec.reconstruction_spec or aligner.ms_bands != 8:
            raise ValueError("Exact W112/D123/P0/BICUBIC_MS architecture required")
        require_sha256(parent_sha, "parent checkpoint SHA256")
        self.aligner = aligner
        self.teacher = reconstruction if self.engine_kind == "mature_teacher" else None
        self.student = None if self.engine_kind == "mature_teacher" else reconstruction
        self.spatial = ScharrAbsolute("WV3")
        self._binding = deepcopy(donor_binding)
        self.sources = DirectTrainingSources("WV3", self._binding["source_manifest_sha256"])
        self.parent_sha = parent_sha
        self.start_update, self.total_updates = cfg["start_update"], cfg["planned_horizon"]
        self.epsilon_seed = cfg.get("epsilon_seed", cfg.get("stream_seeds", {}).get("rr_epsilon", cfg["seed"]))
        if type(self.epsilon_seed) is not int or not 0 <= self.epsilon_seed < 2**63:
            raise ValueError("Explicit dedicated CPU epsilon seed required")
        self._epsilon_rng = torch.Generator(device="cpu").manual_seed(self.epsilon_seed)
        self.updates, self.forward_calls = 0, 0
        self._head_final_sha = None
        self._frozen_body_sha = self._body_hash()
        self.set_progress(0)
        self._contract = deepcopy(self.pipeline_manifest())

    @property
    def reconstruction(self):
        return self.teacher if self.teacher is not None else self.student

    @property
    def donor_binding(self):
        return deepcopy(self._binding)

    @property
    def head_active(self):
        return self.engine_kind in ("tail_fc2_eps", "tail_fc2_abs") and self.updates < HEAD_UPDATES

    def _body_hash(self):
        return state_dict_sha256({k:v for k,v in self.aligner.state_dict().items()
            if not (self.engine_kind in ("tail_fc2_eps", "tail_fc2_abs") and k.startswith("fc2."))})

    def _head_hash(self):
        return state_dict_sha256(self.aligner.fc2)

    def set_progress(self, update):
        if type(update) is not int or not 0 <= update <= self.total_updates:
            raise ValueError("Native local update outside schedule")
        self.updates = update
        self.aligner.requires_grad_(False).eval()
        if self.head_active:
            self.aligner.fc2.requires_grad_(True)
        elif self.engine_kind in ("tail_fc2_eps", "tail_fc2_abs") and self._head_final_sha is None:
            self._head_final_sha = self._head_hash()
        for p in self.aligner.parameters():
            if not p.requires_grad:
                p.grad = None
        self.reconstruction.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.aligner.eval()  # GN/LN have no running stats; no flags changed.
        return self

    def pipeline_manifest(self):
        return {"revision": "PANDA30R_NATIVE_ADAPTATION_V1", "engine_kind": self.engine_kind,
            "sensor": "WV3", "aligner": self.aligner.manifest(),
            "reconstruction": self.reconstruction.spec.manifest(),
            "start_update": self.start_update, "local_horizon": self.total_updates,
            "warp": "single_original_PAN_bicubic_border_FP32_align_corners_false_current_pan_px",
            "native_lms": "PROVIDED_STORED_LMS", "mask_policy": MASK2_POLICY,
            "loss_mask_policy": LOSS_MASK_POLICY,
            "loss_mask": mask_manifest(), "spatial_roi": 8,
            "teacher_or_calibration_in_tail": False, "KD_q_tail": "disabled_all_tail_controls",
            "active_aligner_names": ["fc2.weight", "fc2.bias"] if self.engine_kind in
                ("tail_fc2_eps", "tail_fc2_abs") else [],
            "active_aligner_local_updates": HEAD_UPDATES if self.engine_kind in
                ("tail_fc2_eps", "tail_fc2_abs") else 0,
            "lambda_epsilon_active": .003 if self.engine_kind in ("tail_fc2_eps", "tail_fc2_abs") else 0.,
            "lambda_absolute_active": .2 if self.engine_kind == "tail_fc2_abs" else 0.,
            "epsilon": "RR_only_area_uniform_R3_native_target_detached",
            "evaluation": "unchanged_unmasked_official_RR20_FR20"}

    def get_extra_state(self):
        return {"pipeline": self.pipeline_manifest(), "donor_binding": deepcopy(self._binding),
            "parent_sha": self.parent_sha, "epsilon_seed": self.epsilon_seed,
            "epsilon_rng": self._epsilon_rng.get_state().clone(), "updates": self.updates,
            "forward_calls": self.forward_calls, "frozen_body_sha": self._frozen_body_sha,
            "head_final_sha": self._head_final_sha}

    def set_extra_state(self, state):
        for key in ("pipeline", "donor_binding", "parent_sha", "epsilon_seed", "frozen_body_sha"):
            if not _same(state.get(key), self.get_extra_state()[key]):
                raise ValueError("Native stage provenance mismatch: " + key)
        if type(state.get("forward_calls")) is not int or state["forward_calls"] < 0:
            raise ValueError("Invalid native forward counter")
        generator = torch.Generator(device="cpu")
        generator.set_state(state["epsilon_rng"].cpu())
        if state["head_final_sha"] is not None:
            require_sha256(state["head_final_sha"], "frozen final head SHA")
        self._head_final_sha = state["head_final_sha"]
        self._epsilon_rng, self.forward_calls = generator, state["forward_calls"]
        self.set_progress(state["updates"])

    def assert_contract(self, *, full_audit=True):
        if self.pipeline_manifest() != self._contract:
            raise ValueError("Native scientific contract changed")
        for name, p in self.aligner.named_parameters():
            expected = self.head_active and name.startswith("fc2.")
            if p.requires_grad != expected or (not expected and p.grad is not None):
                raise ValueError("Only active final fc2 may receive Aligner gradients")
        if not all(p.requires_grad for p in self.reconstruction.parameters()):
            raise ValueError("All reconstruction parameters must remain trainable")
        if full_audit:
            if self._body_hash() != self._frozen_body_sha:
                raise ValueError("Frozen Aligner weights changed")
            if self._head_final_sha is not None and self._head_hash() != self._head_final_sha:
                raise ValueError("Final fc2 changed after its 5K freeze boundary")

    def forward(self, rr):
        validate_training_batch(rr, self._binding)
        self.assert_contract(full_audit=False)
        before = self._epsilon_rng.get_state().clone()
        try:
            with torch.autocast(device_type=rr.pan.device.type, enabled=False):
                correction = self.aligner(rr.pan, rr.native_lms)
                support = native_support(rr.pan, correction)
                aligned = warp_pan(rr.pan, correction)  # exactly one native warp
                prediction = self.reconstruction(aligned, rr.ms, rr.native_lms)
                mask = warp_support_mask(*rr.pan.shape[-2:], correction.detach())
                rec = masked_mean((prediction["y"] - rr.gt.detach()).abs(), mask)
                absolute = epsilon = rec.new_zeros(())
                if self.head_active:
                    if self.engine_kind == "tail_fc2_abs":
                        absolute = self.spatial.absolute_loss(aligned, rr.gt.detach(), roi_margin=8)
                    with torch.no_grad():
                        eps = sample_epsilon(len(rr.pan), self._epsilon_rng).to(rr.pan.device)
                        shifted_pan = warp_pan(rr.pan, eps)
                        shifted_support = warp_support_mask(*rr.pan.shape[-2:], eps)
                        if not bool(shifted_support[..., 8:-8, 8:-8].all()):
                            raise ValueError("Epsilon PAN lacks full clean ROI8 Aligner input support")
                    shifted_u = self.aligner(shifted_pan, rr.native_lms)
                    epsilon = epsilon_loss(shifted_u, correction, eps)
                loss = rec + .2 * absolute + .003 * epsilon
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite native objective")
        except Exception:
            self._epsilon_rng.set_state(before)
            raise
        self.forward_calls += 1
        return {"loss": loss, "rec": rec, "abs_rr": absolute, "eps_rr": epsilon,
            "weighted_abs": .2*absolute, "weighted_eps_rr": .003*epsilon,
            "loss_mask": mask, "correction": correction,
            "native": {**prediction, "u": correction, "correction": correction}, **support}

    @torch.no_grad()
    def native_prediction(self, batch, *, return_aligner_details=False):
        batch = replace(batch, gt=None)
        batch.validate(8)
        if (batch.lineage.sensor != "WV3" or batch.lineage.reference != "PROVIDED_STORED_LMS"
                or batch.lineage.grid_name != "PAN_" + batch.lineage.resolution):
            raise ValueError("Native WV3/PAN-grid/stored LMS inference required")
        with torch.autocast(device_type=batch.pan.device.type, enabled=False):
            result = self.aligner(batch.pan, batch.native_lms, return_details=return_aligner_details)
            correction, details = result if return_aligner_details else (result, None)
            aligned = warp_pan(batch.pan, correction)
            predicted = self.reconstruction(aligned, batch.ms, batch.native_lms)
        return {**predicted, "u": correction, "correction": correction,
                **({"aligner_details": details} if details is not None else {})}

    def inference_model(self):
        return _Inference(self.aligner, self.reconstruction)


class _Inference(nn.Module):
    def __init__(self, aligner, reconstruction):
        super().__init__()
        self.aligner, self.student = aligner, reconstruction

    @torch.no_grad()
    def forward(self, batch):
        batch = replace(batch, gt=None)
        batch.validate(8)
        if (batch.lineage.sensor != "WV3" or batch.lineage.reference != "PROVIDED_STORED_LMS"
                or batch.lineage.grid_name != "PAN_" + batch.lineage.resolution):
            raise ValueError("Native WV3/PAN-grid/stored LMS inference required")
        with torch.autocast(device_type=batch.pan.device.type, enabled=False):
            u = self.aligner(batch.pan, batch.native_lms)
            out = self.student(warp_pan(batch.pan, u), batch.ms, batch.native_lms)
        return {**out, "u":u, "correction":u}


class NativeTrainer:
    def __init__(self, stage, cfg, *, evaluator_pin=None):
        configure_precision()
        self.training = _validate_config(cfg)
        if not isinstance(stage, NativeAdaptation) or stage.engine_kind != cfg["engine_kind"]:
            raise TypeError("Matching isolated native stage required")
        self.stage, self.config = stage, deepcopy(cfg)
        self.engine_kind = stage.engine_kind
        self.updates, self.poisoned = 0, False
        self.evaluator_pin = deepcopy(evaluator_pin)
        from panda_stage2_condq.selection import resolve_policy
        self.selection_policy = resolve_policy("WV3", .959)
        role = "T" if self.engine_kind == "mature_teacher" else "S"
        groups = [{"params": list(stage.reconstruction.parameters()), "role":role, "lr":self._rec_lr(0)}]
        if self.engine_kind in ("tail_fc2_eps", "tail_fc2_abs"):
            groups.append({"params":list(stage.aligner.fc2.parameters()), "role":"A", "lr":1e-6})
        self.optimizer = torch.optim.AdamW(groups, lr=self.training.lr,
            betas=self.training.betas, eps=self.training.adam_eps, weight_decay=self.training.weight_decay)
        self._names = self._binding()
        self._initial_contract = self.contract()
        self._imported = False

    def _rec_lr(self, update):
        if self.engine_kind == "mature_teacher":
            return self.training.rate(self.config["start_update"] + update)
        return self.training.rate(update) if update else self.training.lr

    @staticmethod
    def _head_lr(update):
        return .5e-6 * (1 + math.cos(math.pi * min(update, HEAD_UPDATES)/HEAD_UPDATES))

    def _binding(self):
        names = {id(p):name for name,p in self.stage.named_parameters()}
        actual = [id(p) for g in self.optimizer.param_groups for p in g["params"]]
        expected = {id(p) for p in self.stage.reconstruction.parameters()}
        if self.engine_kind in ("tail_fc2_eps", "tail_fc2_abs"):
            expected |= {id(p) for p in self.stage.aligner.fc2.parameters()}
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Native optimizer has missing, duplicated or forbidden parameters")
        return [[names[id(p)] for p in g["params"]] for g in self.optimizer.param_groups]

    def contract(self):
        return {"schema":TRAINER_SCHEMA, "config":deepcopy(self.config),
            "pipeline":self.stage.pipeline_manifest(), "parent_sha":self.stage.parent_sha,
            "donor_binding":self.stage.donor_binding, "training":asdict(self.training),
            "optimizer_names":deepcopy(self._names), "eval_protocol":deepcopy(self.evaluator_pin),
            "selection":self.selection_policy.manifest(),
            "resume": "source_reconstruction_moments_and_stream_preserved",
            "precision":"FP32_no_AMP_no_TF32"}

    def _check_optimizer(self, state, update, *, full_audit=True):
        reference = self.optimizer.state_dict()
        if len(state.get("param_groups", [])) != len(reference["param_groups"]):
            raise ValueError("Optimizer group count changed")
        expected_ids = set()
        for group, ref, live in zip(state["param_groups"], reference["param_groups"], self.optimizer.param_groups):
            role = group.get("role")
            steps = min(update, HEAD_UPDATES) if role == "A" else self.config["start_update"] + update
            rate = self._head_lr(update) if role == "A" else self._rec_lr(update)
            if (group.get("role") != ref["role"] or group.get("params") != ref["params"]
                    or group.get("lr") != rate or tuple(group.get("betas", ())) != self.training.betas
                    or group.get("eps") != self.training.adam_eps
                    or group.get("weight_decay") != self.training.weight_decay
                    or any(group.get(k) != ref.get(k) for k in
                        ("amsgrad", "maximize", "capturable", "differentiable", "foreach", "fused"))):
                raise ValueError("Native AdamW hyperparameters/order/LR changed")
            if steps:
                expected_ids.update(group["params"])
            for pid, parameter in zip(group["params"], live["params"]):
                if not steps:
                    continue
                item = state.get("state", {}).get(pid, {})
                if (set(item) != {"step", "exp_avg", "exp_avg_sq"}
                        or not isinstance(item["step"], torch.Tensor) or item["step"].numel() != 1
                        or float(item["step"]) != steps):
                    raise ValueError("Complete imported/native Adam moments and exact step counters required")
                for key in ("exp_avg", "exp_avg_sq"):
                    if (not isinstance(item[key], torch.Tensor) or item[key].shape != parameter.shape
                            or item[key].dtype != parameter.dtype):
                        raise ValueError("AdamW moment tensor shape/dtype changed")
                if full_audit and (not bool(torch.isfinite(item["exp_avg"]).all())
                        or not bool(torch.isfinite(item["exp_avg_sq"]).all())
                        or bool((item["exp_avg_sq"] < 0).any())):
                    raise ValueError("AdamW moments nonfinite or negative second moment")
        if set(state.get("state", {})) != expected_ids:
            raise ValueError("Incomplete or unexpected native optimizer moments")

    def assert_contract(self, *, full_audit=True):
        if self.poisoned:
            raise RuntimeError("Poisoned native trainer; restore complete checkpoint")
        if (not self._imported or self.contract() != self._initial_contract or self._binding() != self._names
                or self.updates != self.stage.updates):
            raise ValueError("Native source/config/update contract changed or source not imported")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark:
            raise ValueError("Native precision policy changed")
        self.stage.assert_contract(full_audit=full_audit)
        self._check_optimizer(self.optimizer.state_dict(), self.updates, full_audit=full_audit)

    def step_accumulated(self, batches):
        audit = self.updates == 0 or (self.updates+1) % self.training.full_audit_interval == 0
        self.assert_contract(full_audit=audit)
        if self.updates >= self.config["planned_horizon"]:
            raise ValueError("Native schedule completed")
        batches = list(batches)
        total = sum(len(b.pan) for b in batches)
        if (not batches or total != self.training.effective_batch_size
                or any(len(b.pan) != self.training.microbatch_size for b in batches)):
            raise ValueError("Pinned effective/microbatch sizes required")
        saved_extra = deepcopy(self.stage.get_extra_state())
        self.stage.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics, corrections = {}, []
        try:
            for batch in batches:
                out = self.stage(batch)
                weight = len(batch.pan)/total
                (out["loss"] * weight).backward()
                for name in ("loss", "rec", "abs_rr", "eps_rr", "weighted_abs", "weighted_eps_rr", "warp_support_fraction"):
                    metrics[name] = metrics.get(name, 0.) + weight*float(out[name].detach())
                corrections.extend(out["correction"].detach().cpu().tolist())
            self.stage.assert_contract(full_audit=audit)
            norms = {g["role"]:_norm(p.grad for p in g["params"]) for g in self.optimizer.param_groups}
            for group in self.optimizer.param_groups:
                if group["role"] != "A" or self.stage.head_active:
                    if any(p.grad is None for p in group["params"]):
                        raise ValueError("An active optimized parameter received no gradient")
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            self.stage.set_extra_state(saved_extra)
            raise
        try:
            for group in self.optimizer.param_groups:
                group["lr"] = self._head_lr(self.updates+1) if group["role"] == "A" else self._rec_lr(self.updates+1)
            self.optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for g in self.optimizer.param_groups for p in g["params"]):
                raise FloatingPointError("Nonfinite weights after native update")
            self.updates += 1
            self.stage.set_progress(self.updates)
            self.assert_contract(full_audit=audit)
        except Exception:
            self.poisoned = True
            raise
        return {"update":self.updates, "lifetime_update":self.config["start_update"]+self.updates,
            **metrics, "gradient":norms, "correction_dy_dx":corrections,
            "learning_rates":{g["role"]:g["lr"] for g in self.optimizer.param_groups},
            "aligner_head_active_next_update":self.stage.head_active,
            "mask_policy":MASK2_POLICY, "spatial_support_ok":True}

    def state_dict(self):
        self.assert_contract()
        return _cpu({"schema":TRAINER_SCHEMA, "contract":self.contract(), "updates":self.updates,
            "stage":self.stage.state_dict(), "optimizer":self.optimizer.state_dict(),
            "scheduler":{"local_update":self.updates, "lifetime_update":self.config["start_update"]+self.updates}})

    def load_state_dict(self, state):
        if state.get("schema") != TRAINER_SCHEMA or state.get("contract") != self._initial_contract:
            raise ValueError("Native resume scientific/provenance contract differs")
        update = state.get("updates")
        if (type(update) is not int or not 0 <= update <= self.config["planned_horizon"]
                or state.get("scheduler") != {"local_update":update,"lifetime_update":self.config["start_update"]+update}
                or state["stage"]["_extra_state"].get("updates") != update
                or state["stage"]["_extra_state"].get("forward_calls") != update*
                    (self.training.effective_batch_size//self.training.microbatch_size)):
            raise ValueError("Native optimizer/model/scheduler/forward counters differ")
        _finite_tree(state["stage"])
        expected_stage = self.stage.state_dict()
        if set(state["stage"]) != set(expected_stage):
            raise ValueError("Native checkpoint parameter/buffer keys differ")
        for name,expected in expected_stage.items():
            incoming = state["stage"][name]
            if isinstance(expected,torch.Tensor) and (not isinstance(incoming,torch.Tensor)
                    or incoming.shape != expected.shape or incoming.dtype != expected.dtype):
                raise ValueError("Native checkpoint parameter/buffer shape or dtype differs: "+name)
        self._check_optimizer(state["optimizer"], update)
        try:
            self.stage.load_state_dict(state["stage"], strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.updates, self._imported = update, True
            self.assert_contract()
        except Exception:
            self.poisoned = True
            raise


def _import_moments(trainer, source, group_index, expected_names):
    groups = source["optimizer"]["param_groups"]
    old = groups[group_index]
    names = source.get("optimizer_parameter_names", source.get("contract", {}).get("optimizer_parameters"))
    if names is None or names[group_index] != expected_names:
        raise ValueError("Source reconstruction optimizer names/order mismatch")
    if (tuple(old.get("betas", ())) != trainer.training.betas
            or old.get("eps") != trainer.training.adam_eps
            or old.get("weight_decay") != trainer.training.weight_decay
            or old.get("amsgrad") is not False or old.get("maximize") is not False):
        raise ValueError("Source AdamW policy differs")
    state = trainer.optimizer.state_dict()
    if any(old.get(k) != state["param_groups"][0].get(k)
           for k in ("capturable","differentiable","foreach","fused")):
        raise ValueError("Source AdamW execution policy differs")
    new_ids = state["param_groups"][0]["params"]
    if len(old["params"]) != len(new_ids):
        raise ValueError("Source optimizer parameter count differs")
    for old_id, new_id in zip(old["params"], new_ids):
        if old_id not in source["optimizer"]["state"]:
            raise ValueError("Source AdamW moments missing; no silent fresh optimizer")
        state["state"][new_id] = deepcopy(source["optimizer"]["state"][old_id])
    trainer._check_optimizer(state, 0)
    trainer.optimizer.load_state_dict(state)
    trainer._imported = True
    trainer.assert_contract()


def _check_stream_progress(stream, lifetime_update, batch):
    state = stream.state_dict()
    if state["epoch"]*stream.count + state["cursor"] != lifetime_update*batch:
        raise ValueError("RR sampler and optimizer lifetime update do not share one boundary")


def build_mature_trainer(cfg, sources, donor_path, donor_sha, *, device="cpu", evaluator_pin=None):
    training = _validate_config(cfg)
    if cfg["engine_kind"] != "mature_teacher":
        raise ValueError("Teacher continuation config required")
    registered = _registered(sources)
    donor = load_donor(donor_path, donor_sha, registered, device=device)
    payload = _read(donor_path, donor_sha)
    source, source_cfg = payload["trainer"], payload["config"]
    base = source_cfg["baseline_resolved"]
    if (source["updates"] != cfg["start_update"]
            or donor.binding.get("donor_source_mask_policy") != LOSS_MASK_POLICY
            or source_cfg["planned_horizon"] != 50000 or base["warmup_updates0"] != 100
            or source_cfg["lr_U"] != training.lr or base["min_lr"] != training.min_lr
            or source["effective_batch_size"] != training.effective_batch_size
            or source["microbatch_size"] != training.microbatch_size):
        raise ValueError("Continuation must preserve exact source MASK2 H50 schedule/batch/update")
    stage = NativeAdaptation(donor.aligner, donor.teacher, cfg,
        donor_binding=donor.binding, parent_sha=donor_sha).to(device)
    trainer = NativeTrainer(stage, cfg, evaluator_pin=evaluator_pin)
    expected = ["teacher."+name for name,_ in stage.teacher.named_parameters()]
    candidates = [i for i,names in enumerate(source["optimizer_parameter_names"]) if names == expected]
    if len(candidates) != 1:
        raise ValueError("Unique source Teacher optimizer group required")
    if source["optimizer"]["param_groups"][candidates[0]]["lr"] != training.rate(cfg["start_update"]):
        raise ValueError("Source Teacher optimizer LR is not its declared original scheduler boundary")
    _import_moments(trainer, source, candidates[0], expected)
    augmentation, seeds = base["augmentation"], source_cfg["stream_seeds"]
    stream = TrainStream(registered, StreamSpec(seeds["rr_data"], seeds["rr_augmentation"],
        **augmentation), microbatch_size=training.microbatch_size,
        effective_batch_size=training.effective_batch_size, device=device)
    old = payload["stream"]
    if (old.get("count") != stream.count or old.get("sampling") != "SHUFFLED_EPOCH_NO_DROP"
            or old.get("augmentation_scope") != "ONE_SHARED_MICROBATCH_TRANSFORM"):
        raise ValueError("Source complete RR sampler is required")
    converted = {k:deepcopy(old[k]) for k in ("order","cursor","epoch","rng","augmentation_rng")}
    converted["contract"] = stream.state_dict()["contract"]
    stream.load_state_dict(converted)
    _check_stream_progress(stream, cfg["start_update"], training.effective_batch_size)
    restore_rng(payload["rng"])
    return trainer, stream


def build_tail_trainer(cfg, sources, parent_path, parent_sha, *, device="cpu", evaluator_pin=None):
    training = _validate_config(cfg)
    if cfg["engine_kind"] not in KINDS[1:]:
        raise ValueError("Explicit tail config required")
    registered, payload = _registered(sources), _read(parent_path, parent_sha)
    if payload.get("schema") not in (SCHEMA, "PANDA_S2_CONDQ_CHECKPOINT_V1"):
        raise ValueError("Full new conditional-q parent checkpoint required")
    source = payload.get("trainer", {})
    if (source.get("schema") != "PANDA_S2_CONDQ_STUDENT_TRAINER_V1"
            or source.get("updates") != cfg["start_update"]):
        raise ValueError("Tail must fork from specified completed conditional-q update")
    source_training = TrainingSpec(**source["contract"]["training"])
    if (source_training.total_updates != 50000
            or source_training.effective_batch_size != training.effective_batch_size
            or source_training.microbatch_size != training.microbatch_size
            or source.get("scheduler") != {"completed_updates":cfg["start_update"],
                                            "spec":asdict(source_training)}
            or len(source["optimizer"]["param_groups"]) != 1
            or source["optimizer"]["param_groups"][0]["lr"] != source_training.rate(cfg["start_update"])):
        raise ValueError("Parent must share H50 and exact batch contract")
    extra = source["stage"]["_extra_state"]
    binding = extra["donor_binding"]
    expected_student_seed = cfg.get("stream_seeds",{}).get("S_init",cfg["seed"])
    if (not _same(extra,source["contract"]["stage_contract"])
            or extra["pipeline"].get("seed") != expected_student_seed):
        raise ValueError("Parent full stage contract or paired Student seed differs")
    if (binding["sensor"] != "WV3" or binding["source_manifest_sha256"] != registered.inventory.source_manifest_sha256
            or extra["pipeline"]["mask_policy"] != MASK2_POLICY):
        raise ValueError("Tail parent source/MASK2 binding mismatch")
    a_state, s_state = _role_state(source["stage"], "aligner."), _role_state(source["stage"], "student.")
    if state_dict_sha256(a_state) != binding["aligner_state_sha256"]:
        raise ValueError("Parent frozen Aligner hash mismatch")
    roles = {"A":state_dict_sha256(a_state),"S":state_dict_sha256(s_state)}
    if any(k.startswith("teacher.") for k in source["stage"]):
        roles["T"] = state_dict_sha256(_role_state(source["stage"],"teacher."))
        if roles["T"] != binding["teacher_state_sha256"]:
            raise ValueError("Parent frozen Teacher hash mismatch")
    if payload["schema"] == SCHEMA and (payload.get("engine_kind") != "stage2_condq"
            or payload.get("model_state_sha256") != canonical_sha256(roles)
            or payload.get("config",{}).get("start_update") != 0):
        raise ValueError("Parent complete checkpoint paired model identity differs")
    if payload["schema"] == SCHEMA:
        parent_cfg = payload["config"]
        expected_parent = {"run_id":cfg.get("parent_run"),"case":cfg.get("parent_case"),
                           "replica":cfg.get("parent_replica"),"server":cfg.get("server"),
                           "campaign_id":cfg.get("campaign_id")}
        if any(value is not None and parent_cfg.get(key) != value for key,value in expected_parent.items()):
            raise ValueError("Parent checkpoint belongs to another registered run/server/campaign")
    spec = Stage2Spec("WV3", "gt_only", cfg["seed"], 0., 0., 0., 1e-6, mask_policy=MASK2_POLICY)
    with torch.random.fork_rng(devices=[]):
        aligner = DirectPANAligner(8)
        aligner.load_state_dict(a_state, strict=True)
        student = build_reconstructor(spec.reconstruction_spec, seed=cfg["seed"], role="S")
        student.load_state_dict(s_state, strict=True)
    stage = NativeAdaptation(aligner, student, cfg, donor_binding=binding, parent_sha=parent_sha).to(device)
    trainer = NativeTrainer(stage, cfg, evaluator_pin=evaluator_pin)
    _import_moments(trainer, source, 0, [name for name,_ in student.named_parameters()])
    state = payload.get("stream") or source.get("stream_state")
    if not state:
        raise ValueError("Full parent RR stream required; no fresh-stream tail fallback")
    stream = TrainStream(registered, StreamSpec(**state["contract"]["stream"]),
        microbatch_size=training.microbatch_size, effective_batch_size=training.effective_batch_size, device=device)
    stream.load_state_dict(state)
    _check_stream_progress(stream, cfg["start_update"], training.effective_batch_size)
    from panda_stage2_condq.training import restore_rng as restore_student_rng
    restore_student_rng(source["rng"])
    return trainer, stream


def build_fresh_stage2(cfg, donor, stats, *, evaluator_pin=None, device="cpu"):
    """Existing fresh conditional-q method/trainer, with no scientific override."""
    spec = Stage2Spec(**cfg["stage2_spec"])
    if spec.mask_policy != MASK2_POLICY or cfg.get("engine_kind") != "stage2_condq":
        raise ValueError("Fresh campaign Stage 2 requires explicit MASK2 conditional-q config")
    stage = ConditionalQStage2(spec, donor.aligner,
        None if spec.variant == "gt_only" else donor.teacher,
        donor_binding=donor.binding, stats=stats).to(device)
    trainer = StudentTrainer(stage, TrainingSpec(**cfg["training_spec"]),
        evaluation_pin=evaluator_pin, hqnr_target=.959)
    trainer.config, trainer.engine_kind = deepcopy(cfg), "stage2_condq"
    return trainer


def advance(trainer, stream):
    """No data/epsilon consumption on pre-optimizer failure; poison on partial step."""
    before = deepcopy(stream.state_dict())
    try:
        return trainer.step_accumulated(stream.next_update())
    except Exception:
        if not trainer.poisoned:
            stream.load_state_dict(before)
        raise


def _pins(code_pin, evaluator_pin):
    if not isinstance(code_pin, dict) or not code_pin or not isinstance(evaluator_pin, dict) or not evaluator_pin:
        raise ValueError("Explicit code and evaluator pins required")
    for sha in code_pin.values():
        require_sha256(sha, "code file SHA256")
    return {"code":deepcopy(code_pin),"evaluator":deepcopy(evaluator_pin)}


def save_checkpoint(path, trainer, stream, *, code_pin, evaluator_pin, evaluation=None):
    """Atomic complete state for native engines or unchanged StudentTrainer."""
    trainer.assert_contract()
    pins = _pins(code_pin, evaluator_pin)
    if trainer.evaluator_pin != evaluator_pin:
        raise ValueError("Checkpoint evaluator differs from trainer pin")
    cfg = trainer.config
    _check_stream_progress(stream, cfg.get("start_update", 0)+trainer.updates,
                           trainer.training.effective_batch_size)
    if evaluation is not None:
        identity = {k:cfg[k] for k in ("run_id","attempt","seed","dataset")}
        identity.update(completed_update=trainer.updates, model_state_sha256=_state_hash(trainer.stage))
        from panda40.stage1 import _check_evaluation
        _check_evaluation(evaluation, identity)
    state = (trainer.state_dict(stream_state=stream.state_dict()) if isinstance(trainer, StudentTrainer)
             else trainer.state_dict())
    payload = {"schema":SCHEMA, "engine_kind":trainer.engine_kind, "config":deepcopy(cfg),
        "pins":pins, "trainer":_cpu(state), "stream":_cpu(stream.state_dict()),
        "rng":_cpu(capture_rng()), "model_state_sha256":_state_hash(trainer.stage),
        "evaluation":deepcopy(evaluation), "resume_only":evaluation is None}
    path = Path(path).absolute()
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise FileExistsError("Immutable checkpoint destination exists or is symlinked")
    path.parent.mkdir(parents=True,exist_ok=True)
    _save_immutable(path,payload)
    saved = _read(path,file_sha(path))
    if not _same(saved,payload):
        raise ValueError("Checkpoint serialization roundtrip differs")
    return {"checkpoint_path":str(path),"file_sha256":file_sha(path),
        "model_state_sha256":payload["model_state_sha256"],"completed_update":trainer.updates,
        "weights_saved":True,"checkpoint_verified":True,
        "verification":"COMPLETE_STATE_SERIALIZED_ROUNDTRIP", "resume_only":evaluation is None}


def load_checkpoint(path, sha, trainer, stream, *, code_pin, evaluator_pin):
    payload = _read(path,sha)
    if (payload.get("schema") != SCHEMA or payload.get("config") != trainer.config
            or payload.get("engine_kind") != trainer.engine_kind
            or payload.get("pins") != _pins(code_pin,evaluator_pin)):
        raise ValueError("Complete resume run/config/code/evaluator contract mismatch")
    # Validate the sampler without leaving it changed when model checks fail.
    previous = deepcopy(stream.state_dict())
    try:
        stream.load_state_dict(payload["stream"])
        _check_stream_progress(stream, trainer.config.get("start_update",0)+payload["trainer"]["updates"],
                               trainer.training.effective_batch_size)
        trainer.load_state_dict(payload["trainer"])
        if _state_hash(trainer.stage) != payload["model_state_sha256"]:
            raise ValueError("Restored weights disagree with checkpoint identity")
        restore_rng(payload["rng"])
    except Exception:
        trainer.poisoned = True
        stream.load_state_dict(previous)
        raise
    return payload


def load_mature_donor(path, sha, sources, *, include_teacher=True, device="cpu"):
    """Explicit donor adapter; never pretends the new trainer is legacy Stage 1."""
    registered, payload = _registered(sources), _read(path,sha)
    if type(include_teacher) is not bool:
        raise ValueError("Explicit include_teacher boolean required")
    if (payload.get("schema") != SCHEMA or payload.get("engine_kind") != "mature_teacher"
            or payload.get("trainer",{}).get("schema") != TRAINER_SCHEMA):
        raise ValueError("Explicit mature-Teacher checkpoint required")
    cfg, state = payload["config"], payload["trainer"]
    _validate_config(cfg)
    if state["updates"] != cfg["planned_horizon"] or cfg["start_update"]+state["updates"] != 50000:
        raise ValueError("Only completed lifetime50K mature Teacher may become donor")
    extra = state["stage"]["_extra_state"]
    if (state.get("contract",{}).get("config") != cfg
            or state["contract"].get("pipeline") != extra["pipeline"]
            or state.get("scheduler") != {"local_update":state["updates"],"lifetime_update":50000}
            or extra.get("updates") != state["updates"]
            or extra["pipeline"].get("engine_kind") != "mature_teacher"
            or extra["pipeline"].get("loss_mask_policy") != LOSS_MASK_POLICY
            or extra["pipeline"].get("active_aligner_names") != []
            or extra["pipeline"].get("lambda_absolute_active") != 0
            or extra["pipeline"].get("lambda_epsilon_active") != 0):
        raise ValueError("Mature donor training/geometry/counter contract is incomplete")
    require_sha256(extra.get("parent_sha"),"mature parent checkpoint SHA")
    original = extra["donor_binding"]
    inventory = registered.inventory
    for key,value in (("sensor",inventory.sensor),("source_manifest_sha256",inventory.source_manifest_sha256),
            ("catalog_sha256",inventory.catalog_sha256),("source_h5_sha256",inventory.source_h5_sha256),
            ("sample_ids_sha256",inventory.sample_ids_sha256),("sample_count",len(inventory.sample_ids))):
        if original.get(key) != value:
            raise ValueError("Mature donor source lineage differs: "+key)
    a_state, t_state = _role_state(state["stage"],"aligner."), _role_state(state["stage"],"teacher.")
    hashes = {"A":state_dict_sha256(a_state),"T":state_dict_sha256(t_state)}
    if (hashes["A"] != original["aligner_state_sha256"]
            or hashes["A"] != extra["frozen_body_sha"]
            or canonical_sha256(hashes) != payload["model_state_sha256"]):
        raise ValueError("Mature donor changed frozen A or paired A/T identity")
    spec = Stage2Spec("WV3","gt_only",cfg["seed"],0.,0.,0.,1e-6,mask_policy=MASK2_POLICY)
    with torch.random.fork_rng(devices=[]):
        aligner = DirectPANAligner(8)
        if (extra["pipeline"]["aligner"] != aligner.manifest()
                or canonical_sha256(extra["pipeline"]["reconstruction"]) != canonical_sha256(spec.reconstruction_spec.manifest())):
            raise ValueError("Mature donor geometry/reconstruction manifest changed")
        aligner.load_state_dict(a_state,strict=True)
        teacher = build_reconstructor(spec.reconstruction_spec,seed=0,role="T") if include_teacher else None
        if teacher is not None:
            teacher.load_state_dict(t_state,strict=True)
    for module in (aligner,teacher):
        if module is not None:
            module.requires_grad_(False).eval().to(device)
    binding = {**deepcopy(original), "aligner_state_sha256":hashes["A"],"teacher_state_sha256":hashes["T"],
        "donor_checkpoint_id":sha,"donor_checkpoint_sha256":sha,"donor_run_id":cfg["run_id"],
        "donor_step":50000,"donor_source_mask_policy":LOSS_MASK_POLICY}
    loaded = LoadedDonor(aligner,teacher,binding,{"checkpoint_path":str(Path(path).resolve()),
        "checkpoint_sha256":sha,"run_id":cfg["run_id"],"completed_update":50000,
        "same_checkpoint_A_T":True,"source_pipeline":extra["pipeline"],
        "source_checkpoint_sha256":extra["parent_sha"],"teacher_constructed":include_teacher,
        "selection":"EXPLICIT_COMPLETED_FROZEN_A_TEACHER_CONTINUATION"})
    loaded.assert_frozen()
    return loaded


class _DonorReplayStage(nn.Module):
    """Read-only adapter for independently replaying either imported A/T pair."""
    engine_kind = "donor_replay"

    def __init__(self, donor):
        super().__init__()
        self.aligner, self.teacher, self.student = donor.aligner, donor.teacher, None
        self._donor = donor
        self.sources = DirectTrainingSources("WV3", donor.binding["source_manifest_sha256"])
        self._pipeline = deepcopy(donor.checkpoint_metadata["source_pipeline"])
        self._pipeline.update(engine_kind=self.engine_kind, loss_mask_policy=LOSS_MASK_POLICY,
            replay_only=True, source_checkpoint_sha256=donor.binding["donor_checkpoint_sha256"])

    @property
    def reconstruction(self):
        return self.teacher

    native_prediction = NativeAdaptation.native_prediction

    def pipeline_manifest(self):
        return deepcopy(self._pipeline)

    def get_extra_state(self):
        return {"pipeline":self.pipeline_manifest(),"donor_binding":deepcopy(self._donor.binding)}

    def set_extra_state(self,state):
        if state != self.get_extra_state():
            raise ValueError("Read-only donor replay identity changed")

    def train(self,mode=True):
        super().train(False)
        return self


class _DonorReplay:
    engine_kind = "donor_replay"
    optimizer = None
    poisoned = False
    updates = 0

    def __init__(self,donor,cfg,evaluator_pin):
        self.stage = _DonorReplayStage(donor)
        self.config = deepcopy(cfg)
        self.evaluator_pin = deepcopy(evaluator_pin)
        if cfg.get("engine_kind") != "donor_replay" or cfg.get("start_update") != donor.binding["donor_step"]:
            raise ValueError("Replay config must explicitly retain source lifetime step")

    def assert_contract(self,*,full_audit=True):
        self.stage._donor.assert_frozen()


def build_donor_replay(cfg,sources,path,sha,*,device="cpu",evaluator_pin=None,mature=False):
    """Replay original output without re-entering a legacy training resume path."""
    importer = load_mature_donor if mature else load_donor
    donor = importer(path,sha,sources,device=device)
    if donor.binding.get("donor_source_mask_policy") != LOSS_MASK_POLICY:
        raise ValueError("Only explicit MASK2 Direct A/T sources admitted")
    result = _DonorReplay(donor,cfg,evaluator_pin)
    result.assert_contract()
    return result
