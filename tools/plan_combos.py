"""단일 제거 실험 결과를 읽고, 조합(중복) 실험 config 를 자동 생성한다.

규칙:
  1. 각 제거 원자(atom)의 ERGAS 열화를 A3 기준선 대비로 계산
  2. 열화가 작은 순으로 정렬
  3. 상위(=값싼) 원자들의 쌍 → 삼중 → 누적 조합 순으로 config 생성
     (쌍이 가장 정보량이 높다: 열화가 더해지는지 곱해지는지를 본다)
  4. 상호 배타 조합은 제외 (panbr 은 kpan/hf 를 포함하므로 함께 쓰지 않는다)

출력: config/combo_*.yaml + work_dir/combo_order.txt (실행 순서)
"""
import os, sys, json, glob, itertools, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_TAG = "A3base"          # work_dir/abl_A3base = sweep_W128D2222A3 심볼릭 링크

# 원자: (태그, config 패치, 설명, 충돌하는 원자들)
ATOMS = {
    "btl":    (dict(drop_loc=["4"]),                  "bottleneck CM3A 제거",      set()),
    "enc4":   (dict(drop_loc=["3e"]),                 "encoder H/4 CM3A 제거",     set()),
    "dec4":   (dict(drop_loc=["3d"]),                 "decoder H/4 CM3A 제거",     set()),
    "kpan":   (dict(cm3a_use_k_pan=False),            "k_pan conv 제거",           {"panbr"}),
    "panbr":  (dict(cm3a_pan_branch=False),           "PAN 브랜치 전체 제거",        {"kpan", "hf"}),
    "hf":     (dict(cm3a_v_pan_hf=False),             "고주파 채널 제거",            {"panbr"}),
    "gate":   (dict(attn_mode_gate=False),            "AttnBlock mode gate 제거",  set()),
    "resmod": (dict(res_mode_mod=False),              "ResBlock mode 변조 제거",    set()),
}
# 단일 실험 디렉터리명 → 원자
SINGLE = {"A1btl":"btl", "A2enc":"enc4", "A3dec":"dec4", "B1kpan":"kpan",
          "B2panbr":"panbr", "B3hf":"hf", "C1gate":"gate", "C2resmod":"resmod"}


def read_ergas(tag):
    p = f"{ROOT}/work_dir/abl_{tag}/eval_summary.json"
    if not os.path.exists(p): return None
    try: return json.load(open(p))["ergas"]
    except Exception: return None


def main():
    base = read_ergas(BASE_TAG)
    if base is None:
        print("기준선(A3base) 결과가 없다. 중단."); return 1
    deg = {}
    for d, a in SINGLE.items():
        e = read_ergas(d)
        if e is not None: deg[a] = (e - base) / base * 100
    if len(deg) < 3:
        print(f"완료된 단일 실험이 {len(deg)}개뿐. 조합 계획 불가."); return 1

    print(f"기준선 A3 ERGAS {base:.4f}\n원자별 열화 (작을수록 제거 용이)")
    order = sorted(deg, key=lambda k: deg[k])
    for a in order: print(f"  {a:<8}{deg[a]:>+7.2f}%   {ATOMS[a][1]}")

    combos = []
    def ok(cs):
        for x, y in itertools.combinations(cs, 2):
            if y in ATOMS[x][2] or x in ATOMS[y][2]: return False
        # CM3A 위치를 전부 없애면 CM3A 관련 플래그는 의미가 없다
        drops = {d for c in cs for d in ATOMS[c][0].get("drop_loc", [])}
        if drops >= {"4", "3e", "3d"} and any(c in ("kpan", "panbr", "hf") for c in cs): return False
        return True

    cheap = order[:5]
    for pair in itertools.combinations(cheap, 2):
        if ok(pair): combos.append(pair)
    for tri in itertools.combinations(order[:4], 3):
        if ok(tri): combos.append(tri)
    for k in (4, 5):
        c = tuple(order[:k])
        if ok(c) and c not in combos: combos.append(c)
    # 예상 열화 합이 작은 순 (가장 유망한 조합부터)
    combos.sort(key=lambda c: sum(deg[a] for a in c))

    made = []
    base_cfg = open(f"{ROOT}/config/sweep_W128D2222A3.yaml").read()
    for cs in combos:
        tag = "_".join(cs)
        loc = ["3e", "4", "3d"]
        extra = {}
        for c in cs:
            patch = ATOMS[c][0]
            for d in patch.get("drop_loc", []):
                if d in loc: loc.remove(d)
            for k, v in patch.items():
                if k != "drop_loc": extra[k] = v
        t = base_cfg.replace(f"work_dir: {ROOT}/work_dir/sweep_W128D2222A3",
                             f"work_dir: {ROOT}/work_dir/combo_{tag}")
        newloc = '  cm3a_locations: [' + ", ".join(f'"{x}"' for x in loc) + ']' if loc else '  cm3a_locations: []'
        lines = [newloc] + [f"  {k}: {v}" for k, v in extra.items()]
        t = t.replace('  cm3a_locations: ["3e", "4", "3d"]', "\n".join(lines))
        t = (f"# 조합 제거 실험 (자동 생성). 제거: {', '.join(ATOMS[c][1] for c in cs)}\n"
             f"# 단일 실험 열화 합계 예상 {sum(deg[a] for a in cs):+.2f}% — 실제와 비교해 가산성 확인\n") + t
        open(f"{ROOT}/config/combo_{tag}.yaml", "w").write(t)
        made.append(tag)

    open(f"{ROOT}/work_dir/combo_order.txt", "w").write("\n".join(made) + "\n")
    print(f"\n조합 config {len(made)}개 생성 (실행 순서 = 예상 열화 오름차순)")
    for i, m in enumerate(made, 1):
        cs = m.split("_")
        print(f"  {i:2d}. {m:<28} 예상 {sum(deg[a] for a in cs):+6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
