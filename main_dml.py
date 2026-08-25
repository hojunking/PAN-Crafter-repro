"""Teacher–Teacher DML 진입점. research_log/s2_teacher_teacher_mutual_learning_plan.md

  python main_dml.py --config config/dml_m0.yaml     # 대조군 (lambda=0)
  python main_dml.py --config config/dml_m1.yaml     # mutual

M0 와 M1 은 **완전히 같은 초기 상태**에서 출발해야 한다 (계획 4절).
seed 로 결정론적으로 초기화하고, 초기 가중치를 meta/init/ 에 저장해 감사 가능하게 둔다.
차이는 mutual_lambda 하나뿐이다.
"""
import argparse, os, shutil, sys, yaml, torch, numpy as np, random
from main import (import_class, str2bool, init_seed, load_data, load_model, get_parser)
from utils import Report, MetricsCSV
from train import Trainer
from train_dml import DMLRunner


def build_peer(args, data_loader, seed, sub):
    """peer 하나를 Trainer 로 감싼다. work_dir 을 나눠 체크포인트·mat 이 섞이지 않게 한다."""
    init_seed(seed)                       # 가중치 초기화를 seed 로 고정
    model = load_model(args)
    import copy
    a = copy.deepcopy(args)
    a.work_dir = os.path.join(args.work_dir, sub)
    os.makedirs(a.work_dir, exist_ok=True)
    return Trainer(a, data_loader, model)


def main():
    p = get_parser()
    p.add_argument('--seed-b', type=int, default=2026, help='peer B 초기화 seed')
    p.add_argument('--mutual-lambda', type=float, default=0.0, help='lambda_max. 0 이면 대조군 M0')
    p.add_argument('--mutual-warmup', type=int, default=2500)
    p.add_argument('--mutual-ramp-end', type=int, default=5000)
    p.add_argument('--mutual-diag-iter', type=int, default=500)
    a = p.parse_args()
    if a.config:
        with open(a.config) as f:
            cfg = yaml.safe_load(f)
        key = vars(a).keys()
        for k in cfg:
            if k not in key:
                print(f'WRONG ARG: {k}'); assert k in key
        p.set_defaults(**cfg)
        a = p.parse_args()

    os.makedirs(a.work_dir, exist_ok=True)
    train_log = Report(a.work_dir, type='train')
    test_log = Report(a.work_dir, type='test')
    train_log.write(f'[DML] mutual_lambda={a.mutual_lambda}  warmup={a.mutual_warmup}  '
                    f'ramp_end={a.mutual_ramp_end}  seedA={a.seed}  seedB={a.seed_b}')

    # dataloader 는 하나만 만들어 두 peer 가 **같은 배치·같은 증강**을 보게 한다 (계획 5절)
    init_seed(a.seed)
    data_loader = load_data(a)

    peerA = build_peer(a, data_loader, a.seed, 'peerA')
    peerB = build_peer(a, data_loader, a.seed_b, 'peerB')
    init_seed(a.seed)                     # 배치 순서는 두 실행에서 동일해야 한다

    # 초기 상태 스냅샷 (계획 4절 — 공정한 paired initialization 의 감사 근거)
    init_dir = os.path.join(a.work_dir, 'meta', 'init'); os.makedirs(init_dir, exist_ok=True)
    for nm, tr in (('peerA', peerA), ('peerB', peerB)):
        torch.save(tr.model.state_dict(), os.path.join(init_dir, f'{nm}.pt'))
    train_log.write(f'[DML] 초기 가중치 저장: {init_dir}')

    runner = DMLRunner(a, peerA, peerB, data_loader['train'])
    # MetricsCSV 는 work_dir 을 받아 그 안에 metrics.csv 를 만든다.
    # peer 별 work_dir 이 나뉘어 있으므로 각자에 하나씩 둔다.
    metrics = {'peerA': MetricsCSV(peerA.args.work_dir),
               'peerB': MetricsCSV(peerB.args.work_dir)}

    total_epoch = a.num_iter // len(data_loader['train']) + 1
    gs = 0
    best = {'peerA': (1e9, 0), 'peerB': (1e9, 0)}
    for epoch in range(total_epoch):
        train_log.write(f'========= Epoch {epoch + 1} of {total_epoch} =========')
        gs = runner.train(train_log, gs)
        if (epoch + 1) % a.eval_epoch == 0:
            for nm, tr in (('peerA', peerA), ('peerB', peerB)):
                test_log.write(f'--- {nm} ---')
                v = tr.validate(test_log, epoch + 1)
                tr.test_reduced(test_log, epoch + 1)
                tr.test_full(test_log, epoch + 1)
                metrics[nm].append(epoch + 1, gs, tr.last_reduced_metrics,
                                   tr.last_full_metrics, tr.last_val_metrics)
                if v < best[nm][0]:
                    best[nm] = (v, epoch + 1); tr.save_best_model_val()
                test_log.write(f'{nm} best val ERGAS {best[nm][0]:.6f} @epoch {best[nm][1]}')
            runner.save_diag(os.path.join(a.work_dir, 'diagnostics.csv'))
        if gs >= a.num_iter:
            break

    runner.save_diag(os.path.join(a.work_dir, 'diagnostics.csv'))
    for nm, tr in (('peerA', peerA), ('peerB', peerB)):
        tr.test_reduced_save(tag='best_val'); tr.test_full_save(tag='best_val')
        test_log.write(f'[export] {nm} -> {tr.args.work_dir}/results/')
    test_log.write(f'DONE  A {best["peerA"]}  B {best["peerB"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
