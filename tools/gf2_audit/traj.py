import csv, sys, os
runs = sys.argv[1:]
cols = ['epoch','global_step','hqnr_official','d_lambda_official','d_s_official','ergas','sam','scc','psnr','qnr','d_lambda','d_s','hqnr_alt','fscc_official']
for r in runs:
    p = f'work_dir/{r}/metrics.csv'
    if not os.path.exists(p):
        print('MISSING', p); continue
    rows = list(csv.DictReader(open(p)))
    have = [c for c in cols if c in rows[0]]
    print('=====', r, 'n_rows', len(rows))
    print('\t'.join(have))
    for row in rows:
        out=[]
        for c in have:
            v=row.get(c,'')
            try: out.append(f'{float(v):.4f}' if c not in('epoch','global_step') else v)
            except: out.append(v or '-')
        print('\t'.join(out))
    # summary
    def f(row,c):
        try: return float(row[c])
        except: return float('nan')
    import math
    hq=[(f(x,'hqnr_official'),x['epoch'],f(x,'d_lambda_official'),f(x,'d_s_official'),f(x,'ergas')) for x in rows if not math.isnan(f(x,'hqnr_official'))]
    if hq:
        b=max(hq); print('BEST hqnr_official', b)
        print('LAST', hq[-1])
        ds=[x[3] for x in hq]; print('D_s min/median/max over evals', min(ds), sorted(ds)[len(ds)//2], max(ds))
        dl=[x[2] for x in hq]; print('D_l min/median/max over evals', min(dl), sorted(dl)[len(dl)//2], max(dl))
