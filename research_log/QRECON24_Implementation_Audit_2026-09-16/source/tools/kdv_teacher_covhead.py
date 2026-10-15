#!/usr/bin/env python
"""G5 Teacher covariance head 를 따로 만들어 둔다 (train_kdv 가 없으면 자동으로 같은 함수를 부르므로 선택 사항). 계획 §11.6.

    python tools/kdv_teacher_covhead.py --teacher work_dir/<run> --tag best_hqnr --source eq_closure [--epochs 3]
결과: work_dir/_kdv_calibration/<teacher_id>/covhead_<source>.pt (+ .json manifest). teacher_id 는 --teacher-id (기본 run 이름).
"""
import argparse, json, os, sys, yaml, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--teacher", required=True); ap.add_argument("--tag", default="best_hqnr"); ap.add_argument("--source", choices=["eq_closure", "geo_curvature"], default="eq_closure")
    ap.add_argument("--teacher-id", default=None); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--n-patches", type=int, default=3072); ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    from main import import_class
    from kdv.teacher_assets import load_run_model, freeze
    from kdv.registry import resolve
    from kdv.calibration import calibration_batches, calibrate_covariance, calibrate_covhead
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(os.path.join(ROOT, a.teacher, "meta", "config.yaml")))
    teacher, man = load_run_model(a.teacher, a.tag, import_class(cfg["model"])); freeze(teacher); teacher.to(dev)
    spec = resolve(dict(recipe="A1", input_protocol="I-A", aligner_policy="A-FT", donor=dict(source="x"), teacher=dict(run=a.teacher, id="tmp"), rec=dict(case="R3"),
                        geom_kd=dict(mode="G5", covariance_source=a.source, outer_weight=1.0)))
    class _A:  # calibration_batches 가 args 객체를 받는다
        feeder = cfg["feeder"]; train_feeder_args = cfg["train_feeder_args"]
    batches, bman = calibration_batches(_A, a.n_patches, a.seed, int(cfg.get("batch_size", 48)))
    cov = calibrate_covariance(teacher, batches, spec, dev)
    print("covariance:", json.dumps({k: cov[k] for k in ("status", "k0") if k in cov}))
    if cov["status"] != "OK":
        print("!! covariance calibration 실패 — head 를 만들지 않는다"); sys.exit(4)
    sd, hman = calibrate_covhead(teacher, batches, spec, cov, dev, epochs=a.epochs)
    tid = a.teacher_id or os.path.basename(a.teacher.rstrip("/")); od = os.path.join(ROOT, "work_dir", "_kdv_calibration", tid); os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f"covhead_{a.source}.pt"); torch.save(sd, out)
    json.dump(dict(teacher=man, covariance=cov, head=hman, calibration_set=bman), open(out.replace(".pt", ".json"), "w"), indent=1)
    print("saved", out, "loss/epoch", [round(x, 4) for x in hman["loss_per_epoch"]])


if __name__ == "__main__":
    main()
