"""Teacher–Teacher DML 진입점. research_log/s2_teacher_teacher_mutual_learning_plan.md

  python main_dml.py --config config/dml_m0.yaml     # 대조군 (lambda=0)
  python main_dml.py --config config/dml_m1.yaml     # mutual

M0 와 M1 은 **완전히 같은 초기 상태**에서 출발해야 한다 (계획 4절).
seed 로 결정론적으로 초기화하고, 초기 가중치를 meta/init/ 에 저장해 감사 가능하게 둔다.
차이는 mutual_lambda 하나뿐이다.
"""
import argparse, json, os, shutil, sys, yaml, torch, numpy as np, random
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
    p.add_argument('--init-from', type=str, default=None,
                   help='짝 실행의 meta/init 경로. M0 가 만든 번들을 M1 이 로드해 '
                        '동일 초기 상태를 보장한다 (계획 4절)')
    p.add_argument('--ckpt-iter', type=int, default=5000,
                   help='이 스텝마다 두 peer 를 함께 저장한다. 9시간 런이 '
                        '중간에 죽어도 재개할 수 있게 한다. 0 이면 끄기')
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
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'      # main.py 와 동일 — gpu 인자 적용
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    init_seed(a.seed)
    data_loader = load_data(a)

    peerA = build_peer(a, data_loader, a.seed, 'peerA')
    peerB = build_peer(a, data_loader, a.seed_b, 'peerB')
    init_seed(a.seed)                     # 배치 순서는 두 실행에서 동일해야 한다

    # 계획 4절 — 공정한 paired initialization.
    # model weight 만으로는 부족하다. optimizer/scheduler/RNG 까지 accelerator.save_state
    # 로 통째로 저장하고, --init-from 으로 M0/M1 이 **같은 번들을 로드**하게 한다.
    # 그래야 "차이는 mutual loss 뿐" 이 seed 재현이 아니라 실제 동일성으로 보장된다.
    # 학습 시점 config 를 peer 별 meta 로 스냅샷한다. 이것이 없으면 _upload_dml.sh 가
    # '업로드 시점'의 config 로 meta 를 합성·동결해, config 를 고쳐 재실행한 경우 시트가
    # 낡은 설정으로 라벨링된다 (자가검증 F11).
    if a.config and os.path.exists(a.config):
        import shutil as _sh
        for nm, tr in (('peerA', peerA), ('peerB', peerB)):
            md = os.path.join(tr.args.work_dir, 'meta'); os.makedirs(md, exist_ok=True)
            if nm == 'peerB':
                txt = open(a.config).read()
                txt = __import__('re').sub(r'^seed: .*$', f'seed: {a.seed_b}', txt, count=1,
                                           flags=__import__('re').M)
                open(os.path.join(md, 'config.yaml'), 'w').write(txt)
            else:
                _sh.copy(a.config, os.path.join(md, 'config.yaml'))

    init_root = os.path.join(a.work_dir, 'meta', 'init')
    if a.init_from:
        for nm, tr in (('peerA', peerA), ('peerB', peerB)):
            src = os.path.join(a.init_from, nm)
            tr.accelerator.load_state(src)
            train_log.write(f'[DML] 초기 상태 로드: {nm} <- {src}')
        torch.set_rng_state(torch.load(os.path.join(a.init_from, 'rng_cpu.pt')))
        np.random.set_state(np.load(os.path.join(a.init_from, 'rng_np.npy'),
                                    allow_pickle=True).item()['state'])
        train_log.write(f'[DML] RNG 상태 복원 완료 — M0/M1 이 동일 지점에서 출발한다')
    else:
        for nm, tr in (('peerA', peerA), ('peerB', peerB)):
            tr.accelerator.save_state(os.path.join(init_root, nm))
        torch.save(torch.get_rng_state(), os.path.join(init_root, 'rng_cpu.pt'))
        np.save(os.path.join(init_root, 'rng_np.npy'),
                {'state': np.random.get_state()}, allow_pickle=True)
        train_log.write(f'[DML] 초기 상태 저장(model+optimizer+scheduler+RNG): {init_root}')
        train_log.write(f'[DML] 짝 실행은 --init-from {init_root} 로 같은 지점에서 시작할 것')

    runner = DMLRunner(a, peerA, peerB, data_loader['train'])
    # MetricsCSV 는 work_dir 을 받아 그 안에 metrics.csv 를 만든다.
    # peer 별 work_dir 이 나뉘어 있으므로 각자에 하나씩 둔다.
    metrics = {'peerA': MetricsCSV(peerA.args.work_dir),
               'peerB': MetricsCSV(peerB.args.work_dir)}

    total_epoch = a.num_iter // len(data_loader['train']) + 1
    gs, start_epoch, last_ckpt = 0, 0, 0
    best = {'peerA': (1e9, 0), 'peerB': (1e9, 0)}

    # 재개 — 9시간 런이 중간에 죽어도 처음부터 다시 돌리지 않는다.
    # 배치 순서까지 복원되지는 않으므로 "근사 재개" 다 (KNOWN_ISSUES C-1 과 같은 한계).
    # --resume 은 main.py get_parser() 에서 상속받는다 (중복 등록 금지).
    # DML 에서는 pair checkpoint 디렉터리를 가리킨다.
    if a.resume:
        for nm, tr in (('peerA', peerA), ('peerB', peerB)):
            tr.accelerator.load_state(os.path.join(a.resume, nm))
        st = json.load(open(os.path.join(a.resume, 'state.json')))
        gs, start_epoch = st['global_step'], st['epoch']
        last_ckpt = gs
        dcsv = os.path.join(a.work_dir, 'diagnostics.csv')
        if os.path.exists(dcsv):     # save_diag 가 'w' 모드라 기존 이력을 미리 실어 둔다
            import csv as _csv
            runner.diag.extend(list(_csv.DictReader(open(dcsv))))
        best = {k: tuple(v) for k, v in st['best'].items()}
        bs_path = os.path.join(a.work_dir, 'best_state.json')
        if os.path.exists(bs_path):          # eval 마다 갱신되는 쪽이 최신이다
            best = {k: tuple(v) for k, v in json.load(open(bs_path)).items()}
        train_log.write(f'[DML] 재개: step {gs}, epoch {start_epoch}, best {best} '
                        f'(배치 순서는 복원되지 않는 근사 재개)')

    def save_pair(tag):
        d = os.path.join(a.work_dir, tag)
        for nm, tr in (('peerA', peerA), ('peerB', peerB)):
            tr.accelerator.save_state(os.path.join(d, nm))
        json.dump({'global_step': gs, 'epoch': epoch + 1,
                   'best': {k: list(v) for k, v in best.items()}},
                  open(os.path.join(d, 'state.json'), 'w'), indent=1)
        train_log.write(f'[DML] pair checkpoint 저장: {tag} (step {gs})')

    for epoch in range(start_epoch, total_epoch):
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
                    # main.py 와 같은 계약 — best 갱신 즉시 영속화. pair ckpt(5K 간격)에만
                    # 의존하면 crash-resume 시 stale best 로 되돌아가, 근사 재개의 재평가가
                    # 디스크의 더 좋은 best_val 을 더 나쁜 가중치로 덮어쓸 수 있다.
                    json.dump({k: list(vv) for k, vv in best.items()},
                              open(os.path.join(a.work_dir, 'best_state.json'), 'w'))
                test_log.write(f'{nm} best val ERGAS {best[nm][0]:.6f} @epoch {best[nm][1]}')
            runner.save_diag(os.path.join(a.work_dir, 'diagnostics.csv'))
        if a.ckpt_iter and gs // a.ckpt_iter > last_ckpt // a.ckpt_iter:
            save_pair('checkpoint_pair'); last_ckpt = gs
        if gs >= a.num_iter:
            break

    runner.save_diag(os.path.join(a.work_dir, 'diagnostics.csv'))

    # main.py:222-229 와 같은 계약 — **선택된 checkpoint 를 로드한 뒤에** 내보낸다.
    # 이걸 빠뜨리면 현재(최종) 가중치가 best_val 이름으로 저장되어, 다른 실행의
    # best_val mat 과 비교 불가능한 값이 된다. best 가 없으면 아예 내보내지 않는다.
    for nm, tr in (('peerA', peerA), ('peerB', peerB)):
        ckpt = os.path.join(tr.args.work_dir, 'best_val')
        if not os.path.isdir(ckpt):
            test_log.write(f'[export] {nm} 건너뜀 — best_val checkpoint 가 없다')
            continue
        tr.accelerator.load_state(ckpt)
        tr.test_reduced_save(tag='best_val')
        tr.test_full_save(tag='best_val')
        test_log.write(f'[export] {nm} best_val(@epoch {best[nm][1]}) -> {tr.args.work_dir}/results/')
    test_log.write(f'DONE  A {best["peerA"]}  B {best["peerB"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
