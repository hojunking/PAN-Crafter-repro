"""Preserve historical H/E publication and representatives, disclose the new PSNR goal."""
import json
import math
from .registry import TARGET


def three_metric_pass(h, e, p):
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (h, e, p)):
        raise ValueError('Full-precision finite H/E/PSNR required')
    return h >= TARGET['HQNR'] and e <= TARGET['ERGAS'] and p >= TARGET['PSNR']


def annotate(value, R):
    value['additional_goal'] = dict(TARGET)
    I = R.INDEX
    for index, row in enumerate(value['rows']):
        note = json.loads(row[I['Original description / notes']])
        pub = note['publication']
        passed = three_metric_pass(*(row[I[R.METRICS[k]]] for k in ('HQNR', 'ERGAS', 'PSNR')))
        pub.update(psnr_min=45.08, three_metric_target_pass=passed,
                   publication_joint_pass_means='HQNR_AND_ERGAS; historical posting rule retained')
        label = row[I['Selection@Step']].replace('조건 충족', 'H/E 충족')
        row[I['Selection@Step']] = label + (' · 3목표 충족' if passed else ' · 3목표 미달')
        row[I['Original description / notes']] = R.canonical_json(note)
        short = json.loads(value['notes'][index])
        short['publication'] = pub
        value['notes'][index] = R.canonical_json(short)
        if max(len(row[I['Original description / notes']]), len(value['notes'][index])) >= 45000:
            raise ValueError('Publication annotation exceeds safe cell size')
    value['view_hash'] = R.digest({'rows': value['rows'], 'notes': value['notes']})
    return value


def install():
    from panda_sensor_focus import best_publication as B
    original = B.project
    B.POLICY = 'PANDA_GF2_PSNR4508_HE_PASS_REPRESENTATIVE_PLUS_THREE_METRIC_ANNOTATION_V1'
    B.project = lambda root, server, require_cost=False: annotate(original(root, server, require_cost=require_cost), B.R)
    from panda_sensor_focus.best_publication import header
    previous = header.desired('s3')
    header.OLD_D1['GF2'] = previous['D1']
    header.OLD_A3 = previous['A3']
    header.desired = lambda server: {
        'D1': 'GF2 목표: HQNR ≥ 0.9645 AND ERGAS ≤ 0.522 AND PSNR ≥ 45.08 · 같은 모델 상태의 FR20 + RR20',
        'A3': '게시: H/E 통과 step 전부 + run별 대표(HQNR 충족 중 ERGAS 최소, 없으면 HQNR 최고); PSNR 포함 3목표 별도 표시'}
