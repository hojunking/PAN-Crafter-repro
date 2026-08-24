"""work_dir/*/metrics.csv 를 읽어 학습 곡선을 그리고 실험 간 요약을 낸다.

배포본은 eval 결과를 test_log.txt 에 텍스트로만 남겨 사후 비교가 어려웠다.
이 스크립트는 baseline / fixed 처럼 여러 실험을 한 장에 겹쳐 그린다.

usage:
  python tools/plot_metrics.py work_dir/wv3_baseline work_dir/wv3_fixed \
                               --out results_log/assets/curve_wv3.png

주의: 여기 그려지는 값은 utils.py 의 파이썬 구현이다. 후보를 좁히는 용도이고,
최종 수치는 results/*.mat 을 DLPan-Toolbox(MATLAB) 로 평가해 확정한다
(KNOWN_ISSUES.md D-1 ~ D-5).
"""

import os
import csv
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# (컬럼, 표시명, 좋은 방향)
PANELS = [
    ('ergas',        'ERGAS (reduced)',    'lower'),
    ('psnr',         'PSNR (reduced)',     'higher'),
    ('sam',          'SAM (reduced)',      'lower'),
    ('q4_first4',    'Q4-first4 (reduced)', 'higher'),
    ('d_s',          'D_s (full)',         'lower'),
    ('qnr',          'QNR (full)',         'higher'),
]


def read(run_dir):
    path = os.path.join(run_dir, 'metrics.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} 가 없다. 아직 eval 이 한 번도 돌지 않았을 수 있다.')
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items() if v != ''})
            except ValueError:
                continue
    rows.sort(key=lambda r: r['epoch'])
    return rows


def best(rows, col, direction):
    vals = [(r[col], int(r['epoch'])) for r in rows if col in r]
    if not vals:
        return None, None
    return (min(vals) if direction == 'lower' else max(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+', help='work_dir 아래 실험 디렉터리들')
    ap.add_argument('--out', default=None, help='저장 경로 (.png)')
    opt = ap.parse_args()

    runs = [(os.path.basename(d.rstrip('/')), read(d)) for d in opt.run_dirs]

    print(f'{"실험":<20s} {"eval수":>6s} ' + ' '.join(f'{n:>22s}' for _, n, _ in PANELS))
    for name, rows in runs:
        cells = []
        for col, _, direction in PANELS:
            v, ep = best(rows, col, direction)
            cells.append('        -             ' if v is None else f'{v:>14.6f} @ep{ep:<4d}')
        print(f'{name:<20s} {len(rows):>6d} ' + ' '.join(cells))
    print('\n(각 열은 그 실험의 최고값과 그때의 epoch. 파이썬 구현 지표이므로 후보 선별용이다.)')

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (col, title, direction) in zip(axes.ravel(), PANELS):
        for name, rows in runs:
            xs = [r['epoch'] for r in rows if col in r]
            ys = [r[col] for r in rows if col in r]
            if xs:
                ax.plot(xs, ys, marker='o', ms=3, lw=1.2, label=name)
        ax.set_title(f'{title}  ({direction} is better)')
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend()
    # 그림 안 문자열은 ASCII 로 둔다. matplotlib 기본 폰트에 한글 글리프가 없어 두부가 된다.
    fig.suptitle('PAN-Crafter - python-side metrics (final numbers come from DLPan-Toolbox)')
    fig.tight_layout()

    out = opt.out or 'metrics_curve.png'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
