#!/usr/bin/env python3
"""Build DESIGN-ONLY ABLR2X case manifests. Does not launch or edit live queues.

The pinned legacy first-five-sweep registry stays byte-for-byte unchanged in
legacy/. New manifests are separate overlays and require runner integration.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTENSION_ID = 'ABLR2X_S123_C17_20260923_v1'
LEGACY_ID = 'PANDA_ABL_S1WV3_S2QB_ADAPTIVE_LOOP_20260921_v2'
GF2_ID = 'PANDA_ABL_GF2_S3_ADAPTIVE_LOOP_20260923_v1'
COMMIT = '6dde5ea81d4d841b406835ac8d2903e5eeec8535'
LANES = {
 's1': dict(sensor='WV3', bands=8, max_dn=2047, t_base=781000, s_base=791000, train_n=9714, val_n=1080, sheet='WV3-s1'),
 's2': dict(sensor='QB', bands=4, max_dn=2047, t_base=881000, s_base=891000, train_n=17139, val_n=None, sheet='QB-s2'),
 's3': dict(sensor='GF2', bands=4, max_dn=1023, t_base=981000, s_base=991000, train_n=19809, val_n=2201, sheet='GF2-s3(5090)'),
}
NAMES = {
 'TPLUS':'Teacher: HRMS reconstruction with relative-shift consistency',
 'TZERO':'Teacher: HRMS reconstruction without relative-shift consistency',
 'C00':'Baseline: PAN and MS reconstruction without alignment or Teacher',
 'C01':'Add low- and high-frequency PAN inputs',
 'C02':'Add a Student PAN Aligner trained from scratch',
 'C03':'Initialize the Student Aligner from the consistency-trained Teacher',
 'C04':'Add uniform Teacher prediction supervision',
 'C05':'Use reconstruction-error-guided hard and selective soft fitting',
 'C06':'Add GT edge supervision with constant geometry weighting',
 'C07':'Full PANDA: reconstruction- and geometry-reliability-guided fitting',
 'C08':'Full model without low- and high-frequency PAN inputs',
 'C09':'Full model without Student PAN alignment',
 'C10':'Full model without any Teacher guidance',
 'C11':'Full model using a Teacher trained without shift consistency',
 'C12':'Full model without extra hard-target reweighting',
 'C13':'Full model without Teacher prediction supervision',
 'C14':'Full model without GT edge supervision',
 'C15':'Full model using mean reliability instead of sample-wise geometry weights',
 'C16':'Full model with a frozen Student Aligner',
 'C17':'Initialize the Student Aligner from a Teacher trained without shift consistency',
}

def read_csv(path: Path) -> list[dict[str,str]]:
 with path.open(encoding='utf-8-sig', newline='') as f:
  return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
 path.parent.mkdir(parents=True,exist_ok=True)
 if fields is None:
  fields=list(rows[0]) if rows else []
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise'); w.writeheader(); w.writerows(rows)

def write_json(path: Path, obj: object) -> None:
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def legacy_inputs() -> tuple[list[dict],dict,list[dict],dict]:
 d=ROOT/'legacy'
 rows=read_csv(d/'ABLR2_BOOT5_190_TrainingCases_2026-09-21.csv')
 cats=read_csv(d/'ABLR2_ComponentCatalog_2026-09-21.csv')
 graph=read_csv(d/'ABLR2_ComparisonGraph_2026-09-21.csv')
 registry=json.loads((d/'ABLR2_DesignRegistry_2026-09-21.json').read_text())
 return rows,{x['case_id']:x for x in cats},graph,registry

def teacher_for(rows: list[dict], server: str, sweep: str, kind: str) -> dict:
 found=[r for r in rows if (r['server'],r['sweep_id'],r['case_id'])==(server,sweep,kind)]
 if len(found)!=1: raise ValueError('Expected one exact local Teacher')
 return found[0]

def add_c17(anchor: dict, zero_teacher: dict) -> dict:
 r=copy.deepcopy(anchor)
 r['case_id']='C17'; r['run_id']=r['run_id'].replace('_C03_','_C17_')
 r['teacher_kind']='TZERO'; r['reference_id']=zero_teacher['reference_id']
 # Fractional rank describes an insertion WITHOUT renumbering legacy ranks.
 # Production binder uses insert_after_run_id rather than accepting it blindly.
 r['queue_rank']=str(float(anchor['queue_rank'])+0.5)
 return r

def gf2_legacy_cases(legacy: list[dict]) -> list[dict]:
 out=[]
 for old in [x for x in legacy if x['server']=='s1']:
  r=copy.deepcopy(old); p=int(r['sweep_id'][1:]); ts=981000+p; ss=991000+p
  def convert(v: str) -> str:
   return (v.replace(LEGACY_ID,GF2_ID).replace('WV3','GF2').replace('s1','s3')
     .replace(str(781000+p),str(ts)).replace(str(791000+p),str(ss)))
  r={k:convert(v) for k,v in r.items()}
  r['plan_id']=GF2_ID; r['server']='s3'; r['sensor']='GF2'; r['bands']='4'; r['max_dn']='1023'
  r['stored_input_channels']='5' if r['role']=='T' else '7'
  if r['teacher_seed']: r['teacher_seed']=str(ts)
  if r['student_seed']: r['student_seed']=str(ss)
  out.append(r)
 return out

def annotate(rows: list[dict], allrows: list[dict]) -> list[dict]:
 bykey={(r['server'],r['sweep_id'],r['case_id']):r for r in allrows}
 out=[]
 for raw in rows:
  r=copy.deepcopy(raw); key=(r['server'],r['sweep_id']); cid=r['case_id']
  teacher=bykey.get((*key,r['teacher_kind'])) if r['requires_teacher']=='True' or r['role']=='T' else None
  anchor=bykey.get((*key,'C03')) if cid=='C17' else None
  r.update(extension_id=EXTENSION_ID,
    admission_action=('REGISTER_NEW_GF2_ONLY' if r['server']=='s3' else
                      'APPEND_IF_ABSENT' if cid=='C17' else 'PRESERVE_EXISTING_NO_RESTART'),
    requires_runner_extension='True',runtime_bound='False',
    teacher_run_id=teacher['run_id'] if teacher else '',
    teacher_endpoint_step='50000' if teacher else '',
    anchor_run_id=anchor['run_id'] if anchor else '',
    insert_after_run_id=anchor['run_id'] if anchor else '',
    normalization=f"2*DN/{r['max_dn']}-1",
    rr_q='Q8' if r['sensor']=='WV3' else 'Q4',
    panmix_enabled='False',
    display_label=f"{r['sweep_id']} | {cid} | {NAMES[cid]}",
    evidence_role='DESIGN_ONLY_NOT_LAUNCHED')
  out.append(r)
 return out

def build() -> dict:
 legacy,cat,graph,oldreg=legacy_inputs()
 assert len(legacy)==190
 additions=[]
 for a in [r for r in legacy if r['case_id']=='C03']:
  additions.append(add_c17(a,teacher_for(legacy,a['server'],a['sweep_id'],'TZERO')))
 gf=gf2_legacy_cases(legacy)
 for a in [r for r in gf if r['case_id']=='C03']:
  gf.append(add_c17(a,teacher_for(gf,'s3',a['sweep_id'],'TZERO')))
 allrows=legacy+additions+gf
 allrows.sort(key=lambda r:(r['server'],int(r['sweep_id'][1:]),float(r['queue_rank'])))
 annotated=annotate(allrows,allrows)
 # Persist all legacy rows unchanged; the overlay is a new identity, not a rewrite.
 first17=[copy.deepcopy(cat[f'C{i:02}']) for i in range(17)]
 c17=copy.deepcopy(cat['C03']); c17.update(case_id='C17',name='C03-TZERO: alignment-prior-only consistency ablation',
   teacher='TZERO',group='ALIGNMENT_PRIOR_ONLY',compare_to='C03;C02',
   purpose='Change only the source of initial Student A: matched TZERO endpoint instead of TPLUS',
   source_case_id='C03')
 catalog=first17+[c17]
 enriched_graph=[]
 for g in graph:
  enriched_graph.append(dict(g,queue_enabled='True',interpretation='INHERITED_CONTROLLED_OR_GROUP_COMPARISON'))
 enriched_graph += [
  dict(relation_id='A17',parent='C17',child='C03',kind='ALIGNMENT_PRIOR_ONLY',queue_enabled='True',
       interpretation='Shift-consistency-trained versus unregularized initial A; output Teacher, e, q and edge absent'),
  dict(relation_id='A17_SCRATCH',parent='C02',child='C17',kind='DIAGNOSTIC_ONLY',queue_enabled='False',
       interpretation='Unregularized pretrained A versus scratch A; descriptive, not another automatic recheck'),
 ]
 registry={
  'schema':'ABLR2X_DESIGN_ONLY_v1','not_a_launcher':True,'extension_id':EXTENSION_ID,
  'baseline_commit_observed':COMMIT,'prior_design_id':LEGACY_ID,'gf2_design_id':GF2_ID,
  'lanes':LANES,'protected_servers':['s4','s5'],
  'main_cases':[f'C{i:02}' for i in range(18)],'teacher_cases':['TPLUS','TZERO'],
  'first_five_reference_universe':300,'legacy_unchanged_rows':190,'new_definition_rows':110,
  's1_s2_new_c17_rows':10,'s3_new_rows':100,'jobs_per_expanded_sweep':20,
  'updates_per_job':50000,'primary_selection':oldreg['primary_checkpoint'],
  'secondary_selection':'EXACT50K','native_pan_only':True,'panmix_enabled':False,
  'loop':copy.deepcopy(oldreg['loop']),
  'thresholds':copy.deepcopy(oldreg['thresholds']),
  'runtime_bindings':{'actual_state':None,'dataset_manifests':None,'teacher_checkpoint_shas':None,
                      'pinned_runtime_release':None,'until_stop_authorization_receipt':None},
  'implementation_required':['C17 overlay and validators','s3/GF2 lane and DN1023 sensor parameters',
     'GF2 canonical RR/FR evaluation adapter','state-preserving pinned-runtime migration',
     'continuous until-stop authorization mode replacing finite lease admission for new jobs'],
  'first_five_s3_seed_policy':'TS981001..981005 / SS991001..991005',
  'future_seed_policy':'INHERIT_PERFORMANCE_INDEPENDENT_SHA256_LEDGER_WITH_GF2_SALT',
 }
 registry['loop'].update(service_mode='UNTIL_OPERATOR_STOP',max_campaign_cycles=None,
     lease_hours=None,require_initial_operator_lease=False,
     require_explicit_until_stop_authorization=True,shared_lock=False,cross_server_barrier=False,
     retain_local_duplicate_guard=True,stop_for_resource_safety=True)
 registry['loop']['refresh']['student_cases_per_sweep']=18
 registry['thresholds']['calibration_graph']='LEGACY_17_RELATIONS_ONLY_TO_PRESERVE_EXISTING_THRESHOLDS'
 return dict(legacy=legacy,all=annotated,catalog=catalog,graph=enriched_graph,registry=registry)

def validate(b: dict) -> dict:
 allrows=b['all']; ids=[r['run_id'] for r in allrows]
 assert len(allrows)==300 and len(set(ids))==300
 old_byid={r['run_id']:r for r in b['legacy']}
 for r in allrows:
  if r['run_id'] in old_byid:
   assert all(r[k]==v for k,v in old_byid[r['run_id']].items())
  assert r['server'] in LANES and r['sensor']==LANES[r['server']]['sensor']
  assert int(r['max_dn'])==LANES[r['server']]['max_dn']
  assert r['panmix_enabled']=='False'
  assert '\n' not in r['display_label'] and '\r' not in r['display_label']
  assert not any(token in r['display_label'] for token in ('SS791','SS891','SS991','TS781','TS881','TS981'))
  assert int(r['num_updates'])==50000 and int(r['stored_input_channels'])==int(r['bands'])+(1 if r['role']=='T' else 3)
  if r['role']=='S' and r['requires_teacher']=='False': assert not r['teacher_run_id'] and not r['reference_id']
 for srv in LANES:
  rr=[r for r in allrows if r['server']==srv]
  assert len(rr)==100
  for p in range(1,6):
   block=[r for r in rr if r['sweep_id']==f'P{p:02}']; assert len(block)==20
   cases={r['case_id']:r for r in block}
   assert set(cases)=={'TPLUS','TZERO',*[f'C{i:02}' for i in range(18)]}
   a,z=cases['C03'],cases['C17']
   for k in ['student_seed','seed_group','input_layout','stored_input_channels','width','depth','num_updates',
             'u_peak_lr','a_peak_lr','alpha','beta','lambda_edge','requires_calibration']:
    assert a[k]==z[k],(srv,p,k)
   assert z['teacher_kind']=='TZERO' and z['teacher_run_id']==cases['TZERO']['run_id']
   assert z['anchor_run_id']==a['run_id'] and z['requires_calibration']=='False'
   assert cases['TPLUS']['teacher_seed']==cases['TZERO']['teacher_seed']
   ordered=sorted(block,key=lambda r:float(r['queue_rank']))
   i=next(i for i,r in enumerate(ordered) if r['case_id']=='C03')
   assert ordered[i+1]['case_id']=='C17'
 cat={r['case_id']:r for r in b['catalog']}
 for k in ['layout','mask_L','mask_H','aligner','hard_mode','alpha','soft_mode','beta','soft_trust',
           'soft_advantage','lambda_edge','q_edge','q_aligner','updates','u_peak_lr','a_peak_lr',
           'teacher_predictions_used','teacher_q_used','teacher_A_clone_used']:
  assert cat['C17'][k]==cat['C03'][k],k
 assert len(b['graph'])==19
 return dict(status='PASS_DESIGN_VALIDATION_ONLY',first_five_reference_rows=300,
             preserved_legacy_rows=190,s1_s2_append_rows=10,s3_new_rows=100,
             unique_run_ids=300,student_configurations=18,comparison_relations=19,
             auto_recheck_relations=18,server_count=3,
             train_launched=False,remote_queue_written=False,gpu_tested=False)

def main() -> None:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,default=ROOT/'registries')
 args=p.parse_args(); out=args.out.resolve(); b=build(); check=validate(b)
 allrows=b['all']; c17=[r for r in allrows if r['server'] in ('s1','s2') and r['case_id']=='C17']; gf=[r for r in allrows if r['server']=='s3']
 write_csv(out/'ABLR2X_First5_ReferenceUniverse_300.csv',allrows)
 write_csv(out/'ABLR2X_S1S2_Append_C17_10.csv',c17)
 write_csv(out/'ABLR2X_GF2_S3_BOOT5_100.csv',gf)
 write_csv(out/'ABLR2X_NewDefinitions_110.csv',c17+gf)
 write_csv(out/'ABLR2X_ComponentCatalog_18.csv',b['catalog'])
 write_csv(out/'ABLR2X_ComparisonGraph_19.csv',b['graph'])
 write_json(out/'ABLR2X_DesignRegistry.json',b['registry'])
 write_json(out/'validation.json',check)
 print(json.dumps(check,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
