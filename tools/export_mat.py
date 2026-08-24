"""임의의 accelerate checkpoint 에서 DLPan-Toolbox 입력용 .mat 을 내보낸다.

배포본은 best 가 갱신될 때마다 고정 파일명으로 .mat 을 덮어써서, 학습이 끝난 뒤
다른 후보를 MATLAB 으로 재평가할 수 없었다(KNOWN_ISSUES.md D-1). 학습 루프에서는
metrics.csv 와 checkpoint 만 남기고, .mat 은 이 스크립트로 필요한 시점에 만든다.

usage:
  python tools/export_mat.py --config config/pancrafter_wv3.yaml \
                             --ckpt work_dir/wv3_baseline/best_reduced
  python tools/export_mat.py --config config/pancrafter_wv3.yaml \
                             --ckpt work_dir/wv3_baseline/epoch-100 --tag ep100

출력: <work_dir>/results/reduced_<tag>.mat, <work_dir>/results/full_<tag>.mat
      (--tag 생략 시 checkpoint 디렉터리 이름을 그대로 쓴다)
"""

import os
import sys
import argparse

import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_parser, import_class, load_model, init_seed  # noqa: E402
from train import Trainer  # noqa: E402


def build_args(config_path, work_dir=None):
    parser = get_parser()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    known = vars(parser.parse_args([])).keys()
    for k in cfg:
        assert k in known, f'WRONG ARG: {k}'
    parser.set_defaults(**cfg)
    args = parser.parse_args([])
    if work_dir:
        args.work_dir = work_dir
    return args


def load_test_only(args):
    """test 로더만 만든다. train h5 는 수 GB 라 export 에는 불필요하다."""
    Feeder = import_class(args.feeder)
    reduced = torch.utils.data.DataLoader(
        dataset=Feeder(**args.test_reduced_feeder_args),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_worker, drop_last=False)
    full = torch.utils.data.DataLoader(
        dataset=Feeder(**args.test_full_feeder_args),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_worker, drop_last=False)
    # Trainer 가 4개 키를 요구하므로 train/val 자리는 reduced 로 채운다 (사용되지 않음).
    return dict(train=reduced, val=reduced, test_reduced=reduced, test_full=full)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True, help='accelerate save_state 디렉터리')
    ap.add_argument('--tag', default=None, help='출력 파일명 접미사 (기본: checkpoint 디렉터리 이름)')
    ap.add_argument('--work-dir', default=None, help='출력 위치 (기본: config 의 work_dir)')
    ap.add_argument('--gpu', type=int, default=None)
    opt = ap.parse_args()

    args = build_args(opt.config, opt.work_dir)
    if opt.gpu is not None:
        args.gpu = opt.gpu
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    init_seed(args.seed)

    tag = opt.tag or os.path.basename(opt.ckpt.rstrip('/'))
    ckpt = os.path.abspath(opt.ckpt)
    assert os.path.isdir(ckpt), f'checkpoint 디렉터리가 없다: {ckpt}'

    trainer = Trainer(args=args, data_loader=load_test_only(args), model=load_model(args))
    trainer.accelerator.load_state(ckpt)
    print(f'[export] ckpt={ckpt}\n[export] tag={tag}\n[export] work_dir={args.work_dir}')

    trainer.test_reduced_save(tag=tag)
    trainer.test_full_save(tag=tag)

    out = os.path.join(args.work_dir, 'results')
    print(f'[export] 완료: {out}/reduced_{tag}.mat, {out}/full_{tag}.mat')
    print('[export] DLPan-Toolbox(MATLAB) 로 평가해 최종 수치를 확정할 것.')


if __name__ == '__main__':
    main()
