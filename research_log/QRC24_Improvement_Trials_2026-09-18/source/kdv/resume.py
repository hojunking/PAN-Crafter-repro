"""exact resume (QEDGE9 감사 F04; 계획 §8.2 'Resume: cache/hash/permutation 및 다음 batch/update 경로 동일').

main.py 의 재개는 optimizer/scheduler/scaler/RNG 만 복원하고 DataLoader(RandomSampler, num_workers) 의 배치 순서는 복원하지 않는 '근사 재개' 다.
여기서는 epoch 시작 시점의 **전역 torch RNG 상태**(RandomSampler 의 permutation 과 worker base_seed 가 여기서 나온다) 와 epoch 시작 step 을 accelerate checkpoint 에 넣고,
재개할 때 그 상태를 복원해 같은 iterator 를 만든 뒤 이미 소비한 batch 수만큼 건너뛴다(worker 가 같은 항목을 같은 순서로 읽으므로 worker 의 augmentation RNG 도 같은 만큼 진행한다).
그 뒤 전역 RNG 는 checkpoint 시점 상태(accelerate 가 복원한 것)로 되돌려 main process 의 이후 소비도 연속 실행과 같게 한다. epsilon/TRI 는 전용 generator(RNGState) 가 이미 checkpoint 에 있다.
kdv.exact_resume: true 인 run 만 (기존 checkpoint 와 custom object 수가 다르므로 옛 run 에는 켜지 않는다)."""
import torch


class ExactResumeMismatch(RuntimeError):
    """checkpoint 의 epoch 상태와 global_step/loader 가 맞지 않는다 — 같은 run id 로 처음부터 자동 재실행하지 않는다 (trainer 가 exit 4 로 멈춘다; QRECON24 감사 F01)."""


class EpochState:
    """accelerate register_for_checkpointing 용: epoch 시작 시점의 전역 torch RNG 상태 + epoch 시작 global_step."""

    def __init__(self):
        self.epoch_rng = None; self.epoch_start_step = None; self.n_batches = None

    def set(self, rng_state, start_step, n_batches):
        self.epoch_rng = rng_state.clone(); self.epoch_start_step = int(start_step); self.n_batches = int(n_batches)

    def state_dict(self):
        return dict(epoch_rng=self.epoch_rng, epoch_start_step=self.epoch_start_step, n_batches=self.n_batches)

    def load_state_dict(self, sd):
        self.epoch_rng = sd.get("epoch_rng"); self.epoch_start_step = sd.get("epoch_start_step"); self.n_batches = sd.get("n_batches")


def begin_epoch(loader, es, global_step, resume_pending):
    """epoch 의 iterator 를 만든다. 반환 (iterator, skip, info).
    resume_pending 이고 es 에 epoch 상태가 있으면: skip = global_step − epoch_start_step (0 ≤ skip < n_batches 이어야 한다), epoch 시작 RNG 복원 → iterator 생성 → skip 개 batch 소비 → 전역 RNG 를 checkpoint 시점 상태로 복귀.
    아니면: 이번 epoch 의 시작 상태를 es 에 기록하고 새 iterator."""
    n = len(loader); info = dict(exact=False, skipped=0, epoch_start_step=None, boundary=False)
    if resume_pending and es is not None and es.epoch_rng is not None and es.epoch_start_step is not None:
        skip = int(global_step) - int(es.epoch_start_step)
        if int(es.n_batches or n) != n or not (0 <= skip <= n):
            raise ExactResumeMismatch(f"exact resume: checkpoint 의 epoch 상태(start {es.epoch_start_step}, n_batches {es.n_batches}) 와 global_step {global_step}/loader {n} 가 맞지 않는다 — 자동 fresh 재실행 없이 멈춘다(사람이 판단)")
        if skip == n:                                                       # QRECON24 감사 F01: epoch 끝 checkpoint(save_epoch; global_step = epoch_start + n) — 그 epoch 는 끝났다.
            es.set(torch.get_rng_state(), global_step, n)                  # 연속 실행과 같이 **다음 epoch** 를 checkpoint 시점(= epoch 끝) 의 전역 RNG 에서 새로 시작한다 (재생·skip 없음)
            info.update(exact=True, skipped=0, epoch_start_step=int(global_step), boundary=True); return iter(loader), 0, info
        post = torch.get_rng_state(); torch.set_rng_state(es.epoch_rng)
        it = iter(loader)                                                   # 같은 permutation · 같은 worker base_seed
        for _ in range(skip):
            next(it)                                                        # 소비한 batch 만큼 건너뛴다 (worker RNG 도 같은 만큼 진행)
        torch.set_rng_state(post)                                           # main process 전역 RNG 는 checkpoint 시점으로
        info.update(exact=True, skipped=skip, epoch_start_step=int(es.epoch_start_step)); return it, skip, info
    if es is not None:
        es.set(torch.get_rng_state(), global_step, n)
    info["epoch_start_step"] = int(global_step)
    return iter(loader), 0, info
