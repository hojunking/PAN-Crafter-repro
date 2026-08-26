# --------------------------------------------------------
# PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening
# Copyright (c) 2025 Jeonghyeok Do, Sungpyo Kim, Geunhyuk Youk, Jaehyup Lee†, and Munchurl Kim†
#
# This code is released under the MIT License (see LICENSE file for details).
#
# This software is licensed for **non-commercial research and educational use only**.
# For commercial use, please contact: mkimee@kaist.ac.kr
# --------------------------------------------------------

import os
import sys
import traceback
import argparse
import yaml

import torch
import numpy as np
import random

from train import Trainer
from utils import Report, MetricsCSV

class YamlAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        yaml_dict = yaml.safe_load(values)
        setattr(namespace, self.dest, yaml_dict)

def get_parser():
    parser = argparse.ArgumentParser(description='PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening')
    parser.add_argument('--work-dir', default='./work_dir', help='the work folder for storing results')
    parser.add_argument('--config', default='./config/test.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=2025, help='random seed for pytorch')
    parser.add_argument('--log-iter', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-iter', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=0, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-epoch', type=int, default=5, help='the interval for evaluating models (#iteration)')

    # feeder
    parser.add_argument('--feeder', default='feeders.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=4, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=YamlAction, default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--val-feeder-args', action=YamlAction, default=dict(), help='the arguments of data loader for validation')
    parser.add_argument('--test-reduced-feeder-args', action=YamlAction, default=dict(), help='the arguments of data loader for test (reduced)')
    parser.add_argument('--test-full-feeder-args', action=YamlAction, default=dict(), help='the arguments of data loader for test (full)')

    # data
    parser.add_argument('--num-bands', type=int, default=4, help='the number of bands')
    parser.add_argument('--max-pixel', type=float, default=1.0, help='maximum pixel value')

    # model
    parser.add_argument('--model', default='model', help='the model will be used')
    parser.add_argument('--model-args', action=YamlAction, default=dict(), help='the arguments of model')

    # loss
    parser.add_argument('--res', type=str2bool, default=True, help='residual connection')
    parser.add_argument('--w-off', type=float, default=1.0, help='switcher threshold')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1'])

    # optim
    parser.add_argument('--gpu', type=int, default=0, help='the index of GPUs for training or testing')
    parser.add_argument('--optimizer', default='AdamW', help='type of optimizer')
    parser.add_argument('--lr-scheduler', default='cosine', help='type of learning rate scheduler')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--num-iter', type=int, default=1, help='the number of total iteration')
    parser.add_argument('--num-warmup', type=int, default=1, help='the number of warmup iteration')
    parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1, help='test batch size')
    parser.add_argument('--mixed-precision', type=str, default=None, choices=['no', 'fp16', 'bf16'])

    # test
    parser.add_argument('--pretrained-path', type=str, default='/model.safetensors', help='path for test')
    parser.add_argument('--residual-base', type=str, default='bicubic', choices=['bicubic','lms'],
                        help="잔차 기준선. 'bicubic'은 배포본 동작(bicubic(ms,x4)), 'lms'는 데이터셋 제공 보간")
    parser.add_argument('--fr-select-indices', type=str, default='12-19',
                        help="공식 HQNR 선택에 쓰는 FR index 범위 (논문 대조 프로토콜)")
    parser.add_argument('--mars', type=str, default='dual', choices=['dual', 'ms'],
                        help="MARs mode. 'dual'=배포본(MS+PAN 복제 학습), 'ms'=단일 mode (M1 실험)")
    parser.add_argument('--select-on', type=str, default='test', choices=['test', 'val', 'hqnr'],
                        help="best 체크포인트 선택 기준. 'test'는 배포본 동작(테스트셋으로 고름, "
                             "낙관 편향). 'val'은 검증셋으로 고르고 테스트는 마지막에 한 번만 본다")
    parser.add_argument('--val-batch-size', type=int, default=16,
                        help='검증셋 평가 배치. 지표는 표본별로 계산하므로 결과에 영향 없다')
    parser.add_argument('--resume', type=str, default=None,
                        help='이어서 학습할 accelerate checkpoint 디렉터리 (예: work_dir/.../checkpoint-20000)')
    return parser


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(args):
    Feeder = import_class(args.feeder)
    data_loader = dict()
    data_loader['train'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.train_feeder_args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        drop_last=True)
    data_loader['val'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.val_feeder_args),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        drop_last=False)
    data_loader['test_reduced'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.test_reduced_feeder_args),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        drop_last=False)
    data_loader['test_full'] = torch.utils.data.DataLoader(
        dataset=Feeder(**args.test_full_feeder_args),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        drop_last=False)
    return data_loader


def load_model(args):
    Model = import_class(args.model)
    model = Model(**args.model_args)
    return model


def train(args):
    global_step = 0
    train_log = Report(args.work_dir, type='train')
    test_log = Report(args.work_dir, type='test')
    metrics_csv = MetricsCSV(args.work_dir)          # KNOWN_ISSUES.md D-1
    data_loader = load_data(args)
    model = load_model(args)

    trainer = Trainer(args=args, data_loader=data_loader, model=model)

    best_ergas, best_ds, best_val_ergas = 9999, 1, 9999
    best_hqnr, best_epoch_hqnr = -1.0, 0
    best_epoch_reduced = best_epoch_full = best_epoch_val = 0
    last_epoch = 0
    total_epoch = args.num_iter // len(data_loader['train']) + 1

    test_log.write(f"[select_on={args.select_on}] " + (
        "검증셋으로 best 를 고르고 테스트셋은 곡선 기록용으로만 본다."
        if args.select_on == 'val' else
        "배포본 동작: 테스트셋으로 best 를 고른다 (낙관 편향 있음, KNOWN_ISSUES.md E-1)."))

    # KNOWN_ISSUES.md C-1: optimizer/scheduler 가 accelerator 에 등록되어 있으므로
    # checkpoint 가 완전하다. dataloader 는 prepare 하지 않으므로 배치 순서까지는
    # 복원되지 않는 '근사 재개' 다. 엄밀한 재현이 목적이면 처음부터 다시 돌릴 것.
    best_state_path = os.path.join(args.work_dir, 'best_state.json')
    if args.resume:
        trainer.accelerator.load_state(args.resume)
        # best 추적값을 복구한다. 없으면 재시작 후 첫 평가가 과거 best 를 덮어쓴다.
        if os.path.exists(best_state_path):
            import json as _json
            bs = _json.load(open(best_state_path))
            best_hqnr = bs.get('best_hqnr', best_hqnr)
            best_epoch_hqnr = bs.get('best_epoch_hqnr', best_epoch_hqnr)
            best_val_ergas = bs.get('best_val_ergas', best_val_ergas)
            best_epoch_val = bs.get('best_epoch_val', best_epoch_val)
            train_log.write(f'[resume] best_state 복구: {bs}')
        name = os.path.basename(args.resume.rstrip('/'))
        n = int(name.split('-')[-1])
        if name.startswith('epoch-'):
            # save_checkpoint 는 epoch 번호로 저장한다. step 으로 오독하면
            # last_epoch=0 이 되어 스케줄이 통째로 어긋난다.
            last_epoch = n
            global_step = n * len(data_loader['train'])
        else:                                  # checkpoint-<global_step> 형식
            global_step = n
            last_epoch = global_step // len(data_loader['train'])
        train_log.write(f'[resume] {args.resume} 에서 재개 — global_step={global_step}, epoch={last_epoch} '
                        f'(배치 순서는 복원되지 않는 근사 재개)')

    for epoch in range(last_epoch, total_epoch):
        train_log.write(f'========= Epoch {epoch + 1} of {total_epoch} =========')
        global_step = trainer.train(train_log, global_step)

        # KNOWN_ISSUES.md C-2: save_epoch 이 0 이면 (epoch+1) % 0 으로 ZeroDivisionError 가 났다.
        if args.save_epoch > 0 and (epoch + 1) % args.save_epoch == 0:
            trainer.save_checkpoint(epoch + 1)

        if (epoch + 1) % args.eval_epoch == 0:
            val_ergas = trainer.validate(test_log, epoch + 1) if args.select_on == 'val' else None
            ergas = trainer.test_reduced(test_log, epoch + 1)
            ds, hqnr = trainer.test_full(test_log, epoch + 1)
            scc = trainer.last_reduced_metrics.get('scc', float('nan'))
            # 핵심 3지표는 어느 모드에서든 반드시 로그에 남긴다.
            # HQNR 은 공식 DLPan 프로토콜(D_lambda_K + block-UQI D_s, index 12-19)이다.
            core = f'[핵심] HQNR(공식 {args.fr_select_indices}): {hqnr:.6f}\tSCC: {scc:.6f}\tERGAS: {ergas:.6f}'
            test_log.write(core)
            train_log.write(core)

            metrics_csv.append(epoch + 1, global_step,
                               trainer.last_reduced_metrics, trainer.last_full_metrics,
                               trainer.last_val_metrics)

            if args.select_on == 'hqnr':
                # HQNR=(1-D_l)(1-D_s) 기준. FR 검증 split 이 없어 FR 테스트셋으로 고른다 —
                # no-reference 지표지만 선택 편향은 존재한다. 문서에 명시했다.
                if hqnr > best_hqnr:
                    best_hqnr = hqnr
                    best_epoch_hqnr = epoch + 1
                    trainer.save_best_model_hqnr()
                    import json as _json
                    _json.dump({'best_hqnr': best_hqnr, 'best_epoch_hqnr': best_epoch_hqnr},
                               open(best_state_path, 'w'))
                test_log.write(f'Best HQNR: {best_hqnr:.6f}\tBest Epoch (hqnr): {best_epoch_hqnr}')
            elif args.select_on == 'val':
                # 선택에 테스트 지표를 쓰지 않는다. 위 test_* 값은 곡선 기록용이다.
                if val_ergas < best_val_ergas:
                    best_val_ergas = val_ergas
                    best_epoch_val = epoch + 1
                    trainer.save_best_model_val()
                    import json as _json
                    _json.dump({'best_val_ergas': best_val_ergas, 'best_epoch_val': best_epoch_val},
                               open(best_state_path, 'w'))
                test_log.write(f'Best val ERGAS: {best_val_ergas:.6f}\tBest Epoch (val): {best_epoch_val}')
            else:
                if ergas < best_ergas:
                    best_ergas = ergas
                    best_epoch_reduced = epoch + 1
                    trainer.save_best_model_reduced()
                if ds < best_ds:
                    best_ds = ds
                    best_epoch_full = epoch + 1
                    trainer.save_best_model_full()
                test_log.write(f'Best ERGAS: {best_ergas:.6f}\tBest Epoch (Reduced): {best_epoch_reduced}\t'
                               f'Best D_s: {best_ds:.6f}\tBest Epoch (Full): {best_epoch_full}')

    # KNOWN_ISSUES.md D-1: 학습 중에는 .mat 을 쓰지 않는다. 배포본은 best 가 갱신될 때마다
    # 고정 파일명으로 덮어써서 후보를 사후 재평가할 수 없었다. 학습이 끝난 뒤 선택된
    # checkpoint 에서만 내보낸다. 임의 checkpoint 는 tools/export_mat.py 로 처리한다.
    tags = {'val': ['best_val'], 'hqnr': ['best_hqnr'],
            'test': ['best_reduced', 'best_full']}[args.select_on]
    for tag in tags:
        ckpt = os.path.join(args.work_dir, tag)
        if not os.path.isdir(ckpt):
            continue
        trainer.accelerator.load_state(ckpt)
        trainer.test_reduced_save(tag=tag)
        trainer.test_full_save(tag=tag)
        test_log.write(f'[export] {tag} -> results/reduced_{tag}.mat, results/full_{tag}.mat')

    if args.select_on == 'hqnr':
        test_log.write(f'DONE\tselect_on=hqnr\tBest HQNR: {best_hqnr:.6f} @epoch {best_epoch_hqnr}')
    elif args.select_on == 'val':
        test_log.write(f'DONE\tselect_on=val\tBest val ERGAS: {best_val_ergas:.6f} @epoch {best_epoch_val}')
    else:
        test_log.write(f'DONE\tselect_on=test\tBest ERGAS: {best_ergas:.6f} @epoch {best_epoch_reduced}\t'
                       f'Best D_s: {best_ds:.6f} @epoch {best_epoch_full}')
    test_log.write('최종 수치는 results/*.mat 을 tools/eval_dlpan.py / eval_dlpan_fr.py 로 평가할 것.')


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_args = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_args.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_args)

    args = parser.parse_args()
    init_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.phase == 'train':
        train(args)
    else:
        raise ValueError('Unknown phase: {}'.format(args.phase))
