"""KDV case registry · resolver · run naming (plan §4.2, §5.4, §6.3, §9.1, §11.5, §13.1, §22).

resolver 는 잘못된 조합을 warning 이 아니라 오류로 막는다:
  A-FR + alignment_kd≠G0 · Teacher 없는 R0–R3 / T·FIX·WH·AD 통계 / G1 · I-N 반경 없음 · offset loss 를 I-A 나 frozen aligner 에 · N recipe 를 I-A 에.
frozen aligner(A-FR) 에서는 geometry/offset 항이 Student 에 gradient 를 주지 않으므로 optimizer loss 에서 제외하고 진단만 남긴다 (§10.3)."""
REC_CASES = {'N0': 'gt', 'R0': 'fixed_kd', 'R1': 'hard_only', 'R2': 'teacher_error', 'R3': 'adaptive'}
STAT_KINDS = {'OFF': None, 'IV': 'image_var', 'GV': 'grad_var', 'GC': 'grad_cov', 'SC': 'spectral_cov', 'EDGE': 'edge'}
STAT_MODES = ('H', 'T', 'FIX', 'WH', 'AD')
POLICIES = ('A-FR', 'A-FT', 'A-SC', 'A-ID')                 # A-FTW 는 phase(warm freeze→unfreeze) 로 표현한다
PROTOCOLS = ('I-A', 'I-N', 'I-NATIVE-TRANSFER')
GEOM_KD = {'G0': 'none', 'G1': 'mean_kd_scalar_k0'}       # G2–G5 · G-EQ · G-STRUCT: PENDING (covariance_geometry 미구현) — resolver 가 막는다
GEOM_PENDING = ('G2', 'G3', 'G4', 'G5', 'G-EQ', 'G-STRUCT')
RECIPES = ('A1', 'A2', 'A3', 'N1', 'N2_SG', 'N3_NOSG', 'T112DFR', 'NOALIGN')     # T112DFR = T112_DONORFROZEN_REC (§3.3)
POLICY_TAG = {'A-FR': 'AFR', 'A-FT': 'AFT', 'A-SC': 'ASC', 'A-ID': 'AID'}
PROTOCOL_TAG = {'I-A': 'IA', 'I-N': 'IN', 'I-NATIVE-TRANSFER': 'INT'}
RECIPE_AUX = {'A1': {}, 'A2': dict(edge_weight=0.1), 'A3': dict(geometry_weight=0.01), 'N1': {}, 'N2_SG': dict(offset_weight=0.01, offset_stop_reference=True),
              'N3_NOSG': dict(offset_weight=0.01, offset_stop_reference=False), 'T112DFR': {}, 'NOALIGN': {}}


def _bad(msg):
    raise ValueError(f"[kdv resolver] {msg}")


def resolve(k):
    """config 의 kdv dict → 정규화된 spec dict. 잘못된 조합은 ValueError."""
    k = k or {}
    recipe = k.get('recipe'); protocol = k.get('input_protocol', 'I-A'); policy = k.get('aligner_policy')
    if recipe not in RECIPES:
        _bad(f"recipe {recipe!r} ∉ {RECIPES}")
    if protocol not in PROTOCOLS:
        _bad(f"input_protocol {protocol!r} ∉ {PROTOCOLS}")
    if policy not in POLICIES:
        _bad(f"aligner_policy {policy!r} ∉ {POLICIES}")
    rec = dict(k.get('rec') or {}); rec_case = rec.get('case', 'N0')
    if rec_case not in REC_CASES:
        _bad(f"rec.case {rec_case!r} ∉ {list(REC_CASES)}")
    st = dict(k.get('stat') or {}); stat_enabled = bool(st.get('enabled', False))
    stat_key = st.get('kind', 'OFF') if stat_enabled else 'OFF'
    if stat_key not in STAT_KINDS:
        _bad(f"stat.kind {stat_key!r} ∉ {list(STAT_KINDS)}")
    stat_mode = st.get('mode', 'H') if stat_enabled else None
    if stat_enabled and stat_mode not in STAT_MODES:
        _bad(f"stat.mode {stat_mode!r} ∉ {STAT_MODES}")
    if stat_key == 'EDGE' and stat_mode != 'H':
        _bad("STAT-EDGE(signed Scharr) 는 GT 대조(H) 만 정의한다 (§13.2 Q15)")
    window = int(st.get('window', 5)) if stat_enabled else None
    if stat_enabled and stat_key != 'EDGE' and (window < 3 or window % 2 == 0):
        _bad(f"stat.window {window} 는 3 이상 홀수")
    g = dict(k.get('geom_kd') or {}); geom = g.get('mode', 'G0')
    if geom in GEOM_PENDING:
        _bad(f"alignment KD {geom} 은 covariance 산출(§11.2–11.6)이 아직 구현되지 않았다 — PENDING. G0/G1 만 가능")
    if geom not in GEOM_KD:
        _bad(f"geom_kd.mode {geom!r} ∉ {list(GEOM_KD)}")
    teacher = dict(k.get('teacher') or {}); has_teacher = bool(teacher.get('run'))
    needs_teacher = (rec_case != 'N0') or (stat_enabled and stat_mode != 'H') or geom != 'G0' or bool(teacher.get('force_forward', False))
    if needs_teacher and not has_teacher:
        _bad(f"rec {rec_case} / stat {stat_key}-{stat_mode} / geom {geom} 은 Teacher 가 필요한데 kdv.teacher.run 이 없다")
    if has_teacher and not teacher.get('id'):
        _bad("kdv.teacher.id (예: T112_v01) 가 필요하다")
    donor = dict(k.get('donor') or {})
    if policy in ('A-FR', 'A-FT') and not donor.get('source'):
        _bad(f"{policy} 는 kdv.donor.source(aligner donor checkpoint) 가 필요하다")
    if policy in ('A-SC', 'A-ID') and donor.get('source'):
        _bad(f"{policy} 에 donor 를 주지 않는다 (A-SC 는 독립 초기화, A-ID 는 aligner 없음)")
    if policy == 'A-FR' and geom != 'G0':
        _bad("A-FR(frozen aligner) 에 alignment KD 를 넣지 않는다 — NOT_APPLICABLE (§0.4, §10.3)")
    if policy == 'A-ID' and recipe != 'NOALIGN':
        _bad("A-ID 는 recipe NOALIGN 으로 적는다")
    if recipe == 'NOALIGN' and policy != 'A-ID':
        _bad("recipe NOALIGN 은 A-ID 전용")
    if recipe in ('N1', 'N2_SG', 'N3_NOSG') and protocol == 'I-A':
        _bad("N donor 에 I-A 를 붙이지 않는다 — I-N(corruption 유지) 또는 I-NATIVE-TRANSFER(명시적 domain-transfer 대조) (§5.4)")
    corr = dict(k.get('corruption') or {}); radius = float(corr.get('radius_hr', 0.0) or 0.0)
    if protocol == 'I-N' and radius <= 0:
        _bad("I-N 은 corruption.radius_hr > 0 이 필요하다 (출처·단위 확인, §5.4)")
    if protocol != 'I-N' and radius > 0:
        _bad(f"{protocol} 에 corruption.radius_hr 를 주지 않는다")
    aux = dict(RECIPE_AUX[recipe]); aux.update({kk: v for kk, v in (k.get('aux') or {}).items() if v is not None})
    edge_w = float(aux.get('edge_weight', 0.0)); geo_w = float(aux.get('geometry_weight', 0.0)); off_w = float(aux.get('offset_weight', 0.0))
    trainable = policy in ('A-FT', 'A-SC')
    disabled = []
    if off_w > 0 and protocol != 'I-N':
        _bad("offset consistency 는 I-N 에서만 정의된다")
    if geo_w > 0 and policy == 'A-ID':
        _bad("A-ID 에 PAN–GT geometry 항을 넣을 수 없다 (aligner 없음)")
    if geo_w > 0 and not trainable:
        disabled.append('geometry'); geo_w_eff = 0.0
    else:
        geo_w_eff = geo_w
    if off_w > 0 and not trainable:
        disabled.append('offset'); off_w_eff = 0.0
    else:
        off_w_eff = off_w
    if geom == 'G1' and not trainable:
        _bad("G1 mean-KD 는 trainable aligner(A-FT/A-SC) 에서만")
    lam_gkd = float(g.get('outer_weight', 0.0)); k0 = float(g.get('k0', 1.0))
    if geom == 'G1' and lam_gkd <= 0:
        _bad("G1 은 geom_kd.outer_weight > 0 이 필요하다")
    return dict(recipe=recipe, protocol=protocol, policy=policy, rec_case=rec_case, rec_mode=REC_CASES[rec_case],
                stat_enabled=stat_enabled, stat_key=stat_key, stat_kind=STAT_KINDS[stat_key], stat_mode=stat_mode, stat_window=window,
                geom=geom, geom_outer_weight=lam_gkd, geom_k0=k0, needs_teacher=needs_teacher, has_teacher=has_teacher, teacher_id=teacher.get('id'),
                aligner_trainable=trainable, edge_weight=edge_w, geometry_weight=geo_w, geometry_weight_effective=geo_w_eff,
                offset_weight=off_w, offset_weight_effective=off_w_eff, offset_stop_reference=bool(aux.get('offset_stop_reference', True)),
                aux_ramp_updates=int(aux.get('ramp_updates', 5000)), geometry_sigma_hr=float(aux.get('geometry_sigma_hr', 2.0)),
                geometry_margin_hr=int(aux.get('geometry_margin_hr', 11)), disabled_terms=disabled, radius_hr=radius)


def stat_tag(spec):
    return 'OFF' if not spec['stat_enabled'] else f"{spec['stat_key']}{spec['stat_mode']}"


def run_name(spec, seed, version='v01', prefix='S2W112'):
    """S2W112_<recipe>_<input>_<aligner>_<rec>_<stat>_<geomKD>_s<seed>_<version> (§13.1)."""
    return f"{prefix}_{spec['recipe']}_{PROTOCOL_TAG[spec['protocol']]}_{POLICY_TAG[spec['policy']]}_{spec['rec_case']}_{stat_tag(spec)}_{spec['geom'].replace('-', '')}_s{int(seed)}_{version}"


def describe(spec, k=None):
    """시트·문서용 서술 (약명 단독 금지 규약)."""
    k = k or {}
    pol = {'A-FR': 'aligner frozen(donor 복사, 학습 없음)', 'A-FT': 'aligner donor 초기화 후 학습', 'A-SC': 'aligner 독립 초기화 학습', 'A-ID': 'aligner 없음(raw PAN, sampler 없음)'}[spec['policy']]
    rec = {'N0': 'L1(GT)', 'R0': 'L1 + β·|S−T|', 'R1': '(1+α d_T)·L1', 'R2': '(1+α d_T)·L1 + β(1−d_T)|S−T|', 'R3': '(1+α d_T)·L1 + β(1−d_T)a_T|S−T| (adaptive)'}[spec['rec_case']]
    st = 'stat OFF' if not spec['stat_enabled'] else f"stat {spec['stat_kind']} w{spec['stat_window']} mode {spec['stat_mode']}"
    t = f" Teacher {spec['teacher_id']}" if spec['has_teacher'] else ' (no Teacher)'
    return f"{spec['protocol']} · {pol} · rec {rec} · {st} · geomKD {spec['geom']}{t}"
