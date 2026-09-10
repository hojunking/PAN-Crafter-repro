"""KDV case registry · resolver · run naming (plan §4.2, §5.4, §6.3, §9.1, §11.5, §13.1, §22).

resolver 는 잘못된 조합을 warning 이 아니라 오류로 막는다:
  A-FR + alignment_kd≠G0 · Teacher 없는 R0–R3 / T·FIX·WH·AD 통계 / G1–G5 · G2–G5 에 covariance_source 없음 · I-N 반경 없음 · offset loss 를 I-A 나 frozen aligner 에 · N recipe 를 I-A 에.
frozen aligner(A-FR) 에서는 geometry/offset 항이 Student 에 gradient 를 주지 않으므로 optimizer loss 에서 제외하고 진단만 남긴다 (§10.3)."""
REC_CASES = {'N0': 'gt', 'R0': 'fixed_kd', 'R1': 'hard_only', 'R2': 'teacher_error', 'R3': 'adaptive'}
STAT_KINDS = {'OFF': None, 'IV': 'image_var', 'GV': 'grad_var', 'GC': 'grad_cov', 'SC': 'spectral_cov', 'EDGE': 'edge'}
STAT_MODES = ('H', 'T', 'FIX', 'WH', 'AD')
POLICIES = ('A-FR', 'A-FT', 'A-SC', 'A-ID')                 # A-FTW 는 phase(warm freeze→unfreeze) 로 표현한다
PROTOCOLS = ('I-A', 'I-N', 'I-NATIVE-TRANSFER')
GEOM_KD = {'G0': 'none', 'G1': 'mean_kd_scalar_k0', 'G2': 'scalar_trace_precision', 'G3': 'full_precision', 'G4': 'diagonal_precision', 'G5': 'gaussian_distribution_kd', 'G-STRUCT': 'gt_structure_tensor'}
COV_SOURCES = ('none', 'eq_closure', 'geo_curvature', 'struct')      # Π_T 출처 (§11.2 geo · §11.4 eq · §11.7 struct) — 모두 frozen Teacher 에서 계산
GEOM_PENDING = ('G-CORR', 'G-XVIEW')                                  # corrupted correction 까지의 KD (§12) — 미구현
SOURCE_TAG = {'none': '', 'eq_closure': 'EQ', 'geo_curvature': 'GEO', 'struct': ''}
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
    g = dict(k.get('geom_kd') or {}); geom = g.get('mode', 'G0'); cov_src = g.get('covariance_source', 'none')
    if geom in GEOM_PENDING:
        _bad(f"alignment KD {geom} 은 미구현 (§12 G-CORR/G-XVIEW)")
    if geom not in GEOM_KD:
        _bad(f"geom_kd.mode {geom!r} ∉ {list(GEOM_KD)}")
    if cov_src not in COV_SOURCES:
        _bad(f"geom_kd.covariance_source {cov_src!r} ∉ {COV_SOURCES}")
    if geom == 'G-STRUCT':
        if cov_src not in ('none', 'struct'):
            _bad("G-STRUCT 의 가중은 GT 구조 텐서(struct) 뿐 — 다른 covariance_source 를 주지 않는다")
        cov_src = 'struct'
    if geom == 'G0' and cov_src != 'none':
        _bad("G0 에 covariance_source 를 주지 않는다")
    if geom in ('G2', 'G3', 'G4', 'G5') and cov_src not in ('eq_closure', 'geo_curvature'):
        _bad(f"{geom} 은 covariance_source eq_closure|geo_curvature 가 필요하다 (§11.5)")
    if geom == 'G1' and cov_src == 'struct':
        _bad("G1 의 k0 는 Π_T 출처(eq_closure|geo_curvature) 에서 calibration 하거나 k0 를 명시한다")
    if geom == 'G1' and cov_src == 'none' and g.get('k0') is None:
        _bad("G1: covariance_source 없이 쓰려면 geom_kd.k0 를 명시한다 (calibration 없는 단위 계수)")
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
    if geom != 'G0' and not trainable:
        _bad(f"{geom} alignment KD 는 trainable aligner(A-FT/A-SC) 에서만 (A-FR 은 NOT_APPLICABLE, §10.3)")
    ow = g.get('outer_weight', 0.0); r_gkd = float(g.get('r_gkd', 0.1)); k0 = (None if g.get('k0') is None else float(g.get('k0')))
    lam_gkd = None if ow == 'calibrate' else float(ow or 0.0)
    if geom != 'G0' and lam_gkd is not None and lam_gkd <= 0:
        _bad(f"{geom} 은 geom_kd.outer_weight > 0 (또는 'calibrate': λ_GKD = r_gkd / k0) 이 필요하다")
    if geom != 'G0' and lam_gkd is None and r_gkd <= 0:
        _bad("geom_kd.outer_weight: calibrate 는 r_gkd > 0 이 필요하다")
    probes = dict(K=int((g.get('probes') or {}).get('K', 16)), radius_hr=float((g.get('probes') or {}).get('radius_hr', 1.0)), guard_margin=int((g.get('probes') or {}).get('guard_margin', 4)))
    if probes['radius_hr'] <= 0 or probes['radius_hr'] > 2.5 or probes['K'] < 2:
        _bad(f"probes.radius_hr {probes['radius_hr']} 는 (0, 2.5] (64² patch 의 guard support), K ≥ 2")     # 검토 지적: radius ≥ 3 이면 전 sample invalid → KD 가 조용히 0
    if float(g.get('sigma_min', 0.05)) <= 0 or float((g.get('geo') or {}).get('sigma_min', 0.05)) <= 0:
        _bad("geom_kd.sigma_min / geo.sigma_min 은 > 0")
    geo = dict(h=float((g.get('geo') or {}).get('h', 0.05)), gamma=(g.get('geo') or {}).get('gamma', 'calibrate'), sigma_min=float((g.get('geo') or {}).get('sigma_min', 0.05)),
               tau_rel=float((g.get('geo') or {}).get('tau_rel', 1e-3)), tau_abs=(g.get('geo') or {}).get('tau_abs', 'calibrate'), s2_bounds=(g.get('geo') or {}).get('s2_bounds', 'calibrate'),
               max_fd_autograd_rel_err=float((g.get('geo') or {}).get('max_fd_autograd_rel_err', 0.1)), max_h_consistency=float((g.get('geo') or {}).get('max_h_consistency', 0.2)))
    eq_sigma_min = float(g.get('sigma_min', 0.05))
    # ---- TRI-A/B/C (addendum) — soft 만 바꾼다; hard·parent 계수 불변
    tri = dict(k.get('tri') or {}); ta = dict(tri.get('a') or {}); tb = dict(tri.get('b') or {}); tc = dict(tri.get('c') or {})
    a_mode = ta.get('mode', 'off'); b_mode = tb.get('mode', 'off'); c_mode = tc.get('mode', 'off')
    from kdv.tri import A_MODES, B_MODES, C_MODES, C_PHI, C_SIGMA_CONTROLS
    if a_mode not in A_MODES or b_mode not in B_MODES or c_mode not in C_MODES:
        _bad(f"tri modes a={a_mode} b={b_mode} c={c_mode} ∉ {A_MODES}/{B_MODES}/{C_MODES}")
    tri_on = (a_mode != 'off') or (b_mode != 'off') or (c_mode != 'off')
    if tri_on and not has_teacher:
        _bad("TRI-A/B/C 는 Teacher 가 필요하다")
    if a_mode != 'off' and rec_case not in ('R0', 'R2', 'R3'):
        _bad("TRI-A 는 soft 항이 있는 rec(R0/R2/R3) 위에서만 (N0/R1 은 soft 가 없다)")
    if b_mode != 'off' and not (stat_enabled and stat_mode in ('AD', 'FIX', 'T')):
        _bad("TRI-B 는 통계 soft 항이 있는 stat(AD/FIX/T) 위에서만 (statistics=OFF·H·WH 에 B 를 켜지 않는다)")
    c_phi = tc.get('phi', 'identity'); c_src = tc.get('covariance_source', 'none'); c_sig_ctrl = tc.get('sigma_control', 'none')
    if c_phi not in C_PHI or c_sig_ctrl not in C_SIGMA_CONTROLS:
        _bad(f"tri.c phi {c_phi} / sigma_control {c_sig_ctrl}")
    if c_mode != 'off':
        if c_phi == 'identity' and rec_case not in ('R0', 'R2', 'R3'):
            _bad("TRI-C(identity) 는 rec soft 가 있어야 한다")
        if c_phi == 'stat' and not (stat_enabled and stat_mode in ('AD', 'FIX', 'T')):
            _bad("TRI-C(stat) 는 통계 soft 가 있어야 한다")
        if c_mode in ('diag', 'full', 'qscalar') and c_src not in ('eq_closure', 'geo_curvature'):
            _bad(f"TRI-C-{c_mode.upper()} 는 covariance_source eq_closure|geo_curvature 가 필요하다 (누락을 covariance 0 으로 처리하지 않는다, §6.5)")
        if c_mode in ('sens', 'qiso') and c_src != 'none':
            _bad(f"TRI-C-{c_mode.upper()} 는 covariance 를 쓰지 않는다 — covariance_source 를 주지 않는다")
        if c_sig_ctrl != 'none' and c_mode not in ('diag', 'full', 'qscalar'):
            _bad("sigma_control 은 covariance 를 쓰는 C 모드에서만")
    tri_spec = dict(enabled=tri_on, a_mode=a_mode, a_eps=float(ta.get('eps', 1e-6)), b_mode=b_mode, b_eps=(tb.get('eps', 'calibrate')), c_mode=c_mode, c_phi=c_phi, c_src=c_src,
                    c_h=float(tc.get('h', 0.05)), c_s_c=tc.get('s_c', 'tau'), c_s_sens=tc.get('s_sens', 'calibrate'), c_lambda_q=tc.get('lambda_q', 'calibrate'), c_sigma_control=c_sig_ctrl,
                    shuffle_seed=int(tri.get('shuffle_seed', 4321)))
    return dict(recipe=recipe, protocol=protocol, policy=policy, rec_case=rec_case, rec_mode=REC_CASES[rec_case], tri=tri_spec,
                stat_enabled=stat_enabled, stat_key=stat_key, stat_kind=STAT_KINDS[stat_key], stat_mode=stat_mode, stat_window=window,
                geom=geom, geom_outer_weight=lam_gkd, geom_r_gkd=r_gkd, geom_k0=k0, cov_source=cov_src, probes=probes, geo=geo, eq_sigma_min=eq_sigma_min,
                covhead_epochs=int(g.get('covhead_epochs', 3)), needs_teacher=needs_teacher, has_teacher=has_teacher, teacher_id=teacher.get('id'),
                aligner_trainable=trainable, edge_weight=edge_w, geometry_weight=geo_w, geometry_weight_effective=geo_w_eff,
                offset_weight=off_w, offset_weight_effective=off_w_eff, offset_stop_reference=bool(aux.get('offset_stop_reference', True)),
                aux_ramp_updates=int(aux.get('ramp_updates', 5000)), geometry_sigma_hr=float(aux.get('geometry_sigma_hr', 2.0)),
                geometry_margin_hr=int(aux.get('geometry_margin_hr', 11)), disabled_terms=disabled, radius_hr=radius)


def stat_tag(spec):
    return 'OFF' if not spec['stat_enabled'] else f"{spec['stat_key']}{spec['stat_mode']}"


def geom_tag(spec):
    """이름 토큰: G0 · G1K(명시 k0) · G1EQ/G2EQ/G3EQ/G4EQ/G5EQ · G3GEO … · GSTRUCT."""
    g = spec['geom']
    if g == 'G0':
        return 'G0'
    if g == 'G-STRUCT':
        return 'GSTRUCT'
    if g == 'G1' and spec['cov_source'] == 'none':
        return 'G1K'
    return g + SOURCE_TAG[spec['cov_source']]


def arch_prefix(model_args, server='S2'):
    """이름 접두 = 서버 + 골격: S2W112D123 (계획 §13.1 의 S2W112 에 depth 를 붙였다 — 2026-09-10 depth [1,2,3] 결정 뒤 D124 와 구분)."""
    return f"{server}W{int(model_args['hidden_size'])}D{''.join(str(d) for d in model_args['depth'])}"


def tri_tag(spec):
    """TRI 토큰 (addendum §0.1: 앞단 A1/A2/A3 와 혼동을 막는 TRI 접두): TRI_A<mode>_B<mode>_C<mode><src>. 전부 off 면 없음."""
    t = spec.get('tri') or {}
    if not t.get('enabled'):
        return ''
    from kdv.tri import TOKEN
    c = TOKEN[t['c_mode']] + ({'eq_closure': 'EQ', 'geo_curvature': 'GEO'}.get(t['c_src'], '') if t['c_mode'] != 'off' else '')
    if t['c_mode'] != 'off' and t['c_phi'] == 'stat':
        c += 'V'
    if t['c_mode'] != 'off' and t['c_sigma_control'] != 'none':
        c += {'rotate': 'ROT', 'shuffle': 'SHUF'}[t['c_sigma_control']]
    return f"TRI_A{TOKEN[t['a_mode']]}_B{TOKEN[t['b_mode']]}_C{c}"


def run_name(spec, seed, version='v01', prefix='S2W112D123'):
    """<prefix>_<recipe>_<input>_<aligner>_<rec>_<stat>_<geomKD>[_TRI_A*_B*_C*]_s<seed>_<version> (§13.1 + depth + TRI)."""
    tt = tri_tag(spec)
    return f"{prefix}_{spec['recipe']}_{PROTOCOL_TAG[spec['protocol']]}_{POLICY_TAG[spec['policy']]}_{spec['rec_case']}_{stat_tag(spec)}_{geom_tag(spec)}{('_' + tt) if tt else ''}_s{int(seed)}_{version}"


def describe(spec, k=None):
    """시트·문서용 서술 (약명 단독 금지 규약)."""
    k = k or {}
    pol = {'A-FR': 'aligner frozen(donor 복사, 학습 없음)', 'A-FT': 'aligner donor 초기화 후 학습', 'A-SC': 'aligner 독립 초기화 학습', 'A-ID': 'aligner 없음(raw PAN, sampler 없음)'}[spec['policy']]
    rec = {'N0': 'L1(GT)', 'R0': 'L1 + β·|S−T|', 'R1': '(1+α d_T)·L1', 'R2': '(1+α d_T)·L1 + β(1−d_T)|S−T|', 'R3': '(1+α d_T)·L1 + β(1−d_T)a_T|S−T| (adaptive)'}[spec['rec_case']]
    st = 'stat OFF' if not spec['stat_enabled'] else f"stat {spec['stat_kind']} w{spec['stat_window']} mode {spec['stat_mode']}"
    t = f" Teacher {spec['teacher_id']}" if spec['has_teacher'] else ' (no Teacher)'
    gk = {'G0': 'geomKD 없음', 'G1': 'G1 mean-KD(k0·I)', 'G2': 'G2 scalar(tr Π/2)', 'G3': 'G3 full precision', 'G4': 'G4 diag precision', 'G5': 'G5 Gaussian KL(cov head)', 'G-STRUCT': 'G-STRUCT GT 구조텐서 가중'}[spec['geom']]
    src = {'none': '', 'eq_closure': ' Π from probe closure(G-EQ)', 'geo_curvature': ' Π from A3 geometry curvature', 'struct': ''}[spec['cov_source']]
    tri = spec.get('tri') or {}
    tr = ''
    if tri.get('enabled'):
        tr = f" · TRI A={tri['a_mode']} B={tri['b_mode']} C={tri['c_mode']}" + (f"({tri['c_src']},{tri['c_phi']})" if tri['c_mode'] != 'off' else '')
    return f"{spec['protocol']} · {pol} · rec {rec} · {st} · {gk}{src}{t}{tr}"
