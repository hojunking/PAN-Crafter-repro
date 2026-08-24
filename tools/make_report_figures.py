"""2026-08-14 WV3 재현 보고서용 그림 생성.

  python tools/make_report_figures.py

출력: results_log/assets/{curve_wv3,qual_wv3_reduced,lpan_mismatch}.png
그림 안 문자열은 ASCII 로 둔다 (matplotlib 기본 폰트에 한글 글리프가 없다).
"""
import os, csv
import numpy as np
import h5py, cv2
from scipy.io import loadmat
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results_log/assets')
os.makedirs(OUT, exist_ok=True)
# 그림 안 문자열은 ASCII 로 둔다 (matplotlib 기본 폰트에 한글 글리프가 없다).
RUNS = [('baseline / sel=test', 'wv3_baseline', '#2563eb'),
        ('baseline / sel=val', 'wv3_baseline_valsel', '#60a5fa'),
        ('fixed / sel=test', 'wv3_fixed', '#dc2626'),
        ('fixed / sel=val', 'wv3_fixed_valsel', '#f87171')]

# DLPan 프로토콜 실측값 (tools/eval_dlpan.py / eval_dlpan_fr.py). 설정당 2회.
MEASURED = {
    'wv3_ergas':  {'baseline': [2.1633, 2.1643], 'fixed': [2.1765, 2.1804], 'paper': 2.040},
    'wv3_hqnr':   {'baseline': [0.9486, 0.9475], 'fixed': [0.9496, 0.9508], 'paper': 0.958},
    'wv2_ergas':  {'baseline': [4.3162, 4.3251], 'fixed': [4.3364, 4.3362], 'paper': 4.169},
    'wv2_hqnr':   {'baseline': [0.9125, 0.9155], 'fixed': [0.9305, 0.9345], 'paper': 0.942},
}


def load_csv(name):
    p = os.path.join(ROOT, 'work_dir', name, 'metrics.csv')
    return [{k: float(v) for k, v in r.items() if v != ''} for r in csv.DictReader(open(p))]


# ---------- 1. 학습 곡선 ----------
def fig_curves():
    panels = [('ergas', 'ERGAS (reduced, lower better)'), ('psnr', 'PSNR dB (reduced, higher)'),
              ('sam', 'SAM (reduced, lower)'), ('ssim', 'SSIM (reduced, higher)'),
              ('d_s', 'D_s (full-res, lower)'), ('qnr', 'QNR (full-res, higher)')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (col, title) in zip(axes.ravel(), panels):
        for lab, run, c in RUNS:
            rows = load_csv(run)
            ax.plot([r['epoch'] for r in rows], [r[col] for r in rows],
                    marker='o', ms=2.5, lw=1.3, color=c, label=lab)
        ax.set_title(title, fontsize=10); ax.set_xlabel('epoch'); ax.grid(alpha=.3)
        if col in ('d_s', 'qnr'):
            ax.set_facecolor('#fff5f5')
            ax.text(.5, .5, 'lpan mismatch\n(see lpan_mismatch.png)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color='#b91c1c', alpha=.55)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('PAN-Crafter WV3 - baseline vs fixed (python-side metrics; final numbers from DLPan-Toolbox)',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/curve_wv3.png', dpi=130); plt.close(fig)
    print('  curve_wv3.png')


# ---------- 2. 정성 비교 ----------
def to_rgb(x, lo, hi):                      # x: [C,H,W] 0..2047
    idx = [4, 2, 1] if x.shape[0] == 8 else [2, 1, 0]
    r = np.stack([x[i] for i in idx], -1)
    return np.clip((r - lo) / (hi - lo), 0, 1)


def fig_qualitative(scenes=(2, 11)):
    d = loadmat(os.path.join(ROOT, 'work_dir/wv3_baseline/results/reduced_best_reduced.mat'))
    df = loadmat(os.path.join(ROOT, 'work_dir/wv3_fixed/results/reduced_best_reduced.mat'))
    n = len(scenes)
    fig, axes = plt.subplots(n, 5, figsize=(16, 3.35 * n))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(scenes):
        gt = d['gt'][s]
        lo, hi = np.percentile(gt, 1), np.percentile(gt, 99)
        lms = cv2.resize(d['ms'][s].transpose(1, 2, 0), (256, 256),
                         interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
        pan = d['pan'][s][0]
        pl, ph = np.percentile(pan, 1), np.percentile(pan, 99)
        cells = [(np.clip((pan - pl) / (ph - pl), 0, 1), 'PAN (256x256, 1ch)', 'gray'),
                 (to_rgb(lms, lo, hi), 'LRMS bicubic x4 (input)', None),
                 (to_rgb(d['sr'][s], lo, hi), 'PAN-Crafter baseline', None),
                 (to_rgb(df['sr'][s], lo, hi), 'PAN-Crafter fixed', None),
                 (to_rgb(gt, lo, hi), 'Ground truth', None)]
        for c, (img, title, cm) in enumerate(cells):
            axes[r, c].imshow(img, cmap=cm); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0: axes[r, c].set_title(title, fontsize=10)
        axes[r, 0].set_ylabel(f'scene #{s}', fontsize=10)
    fig.suptitle('WV3 reduced-resolution, best_reduced (epoch 225). Same 1-99 percentile stretch per row.',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/qual_wv3_reduced.png', dpi=130); plt.close(fig)
    print('  qual_wv3_reduced.png')


# ---------- 3. lpan 불일치 증거 ----------
def fig_lpan():
    K = cv2.getGaussianKernel(41, 1.98); K = K @ K.T
    dn = lambda p: cv2.filter2D(p, -1, K, borderType=cv2.BORDER_REPLICATE)[2::4, 2::4]
    D = os.path.join(ROOT, 'data/PanCollection')
    rows = [('WV3', 'wv3', 'WV3'), ('QB', 'qb', 'QB'), ('GF2', 'gf2', 'GF2')]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5))
    for r, (nm, s, D2) in enumerate(rows):
        with h5py.File(f'{D}/{D2}/full_examples_h5/test_{s}_OrigScale_multiExm1.h5') as f:
            pan = f['pan'][0, 0].astype(np.float64)
        with h5py.File(f'{D}/{D2}/full_examples_h5/test_{s}_OrigScale_multiExm1_pan.h5') as f:
            lp = f['lpan'][0, 0].astype(np.float64)
        pred = dn(pan)
        corr = np.corrcoef(pred.ravel(), lp.ravel())[0, 1]
        for c, (img, t) in enumerate([(pan, f'{nm}  PAN 512x512 (in .h5)'),
                                      (pred, 'expected lpan = Gauss(1.98)+/4'),
                                      (lp, f'distributed lpan   corr={corr:.3f}')]):
            lo, hi = np.percentile(img, 1), np.percentile(img, 99)
            axes[r, c].imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap='gray')
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, c].set_title(t, fontsize=9,
                                 color='#b91c1c' if (c == 2 and corr < .9) else 'black')
    fig.suptitle('pan_h5.zip: full-resolution lpan. GF2 matches (corr 1.000); WV3/QB do not (corr ~0.01)',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/lpan_mismatch.png', dpi=130); plt.close(fig)
    print('  lpan_mismatch.png')


def fig_tradeoff():
    """in-domain 과 zero-shot 에서 baseline/fixed 의 우열이 뒤집히는 것을 보인다."""
    panels = [('wv3_ergas', 'WV3 in-domain\nERGAS (lower better)', 'lower'),
              ('wv3_hqnr',  'WV3 in-domain\nHQNR (higher better)', 'higher'),
              ('wv2_ergas', 'WV2 zero-shot\nERGAS (lower better)', 'lower'),
              ('wv2_hqnr',  'WV2 zero-shot\nHQNR (higher better)', 'higher')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (key, title, direction) in zip(axes, panels):
        d = MEASURED[key]
        for i, (name, c) in enumerate([('baseline', '#2563eb'), ('fixed', '#dc2626')]):
            v = d[name]
            ax.bar([i], [np.mean(v)], 0.55, color=c, alpha=.85,
                   yerr=[[np.mean(v)-min(v)], [max(v)-np.mean(v)]], capsize=6,
                   error_kw=dict(lw=1.2))
            for x in v:
                ax.plot([i], [x], 'k.', ms=6, zorder=3)
        ax.axhline(d['paper'], ls='--', lw=1.4, color='#16a34a')
        ax.text(1.45, d['paper'], ' paper', color='#16a34a', va='center', fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['baseline', 'fixed'])
        ax.set_title(title, fontsize=10)
        lo = min(min(d['baseline']), min(d['fixed']), d['paper'])
        hi = max(max(d['baseline']), max(d['fixed']), d['paper'])
        m = (hi - lo) * 0.35 + 1e-6
        ax.set_ylim(lo - m, hi + m); ax.grid(axis='y', alpha=.3)
    fig.suptitle('Fixing the released code (A-1/A-2) is neutral-to-worse in-domain '
                 'but clearly better on the unseen satellite. n=2 runs per config (dots).',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(f'{OUT}/tradeoff_indomain_vs_zeroshot.png', dpi=130); plt.close(fig)
    print('  tradeoff_indomain_vs_zeroshot.png')


if __name__ == '__main__':
    fig_curves(); fig_qualitative(); fig_lpan(); fig_tradeoff()
    print(f'저장 위치: {OUT}')
