import json, sys, os
for r in sys.argv[1:]:
    p=f'work_dir/{r}/results/fr_mat20.json'
    if not os.path.exists(p): print('=====',r,'NO fr_mat20.json'); continue
    d=json.load(open(p))
    print('=====',r)
    keys=[k for k in d if not k.startswith('per_scene')]
    for k in keys:
        v=d[k]
        if isinstance(v,float): v=round(v,5)
        if isinstance(v,dict):
            print(f'  {k}: '+json.dumps({kk:(round(vv,5) if isinstance(vv,float) else vv) for kk,vv in v.items() if not str(kk).startswith("per_scene")},ensure_ascii=False)[:600])
        else:
            print(f'  {k}: {v}')
