"""Independent local 40h budgets; s4 screening is descriptive, never a gate."""
from dataclasses import dataclass
from datetime import datetime,timedelta,timezone
import math
from gfp40.plan import (CAMPAIGN_ID,UPGRADE_ORDER,block_for,
    registry_sha256,DIAGNOSTIC_HOURS,verify_lane)

def utc(value):
    if isinstance(value,str):value=datetime.fromisoformat(value.replace('Z','+00:00'))
    if not isinstance(value,datetime) or value.tzinfo is None:raise ValueError('Timezone-aware local t0 required')
    return value.astimezone(timezone.utc)

@dataclass(frozen=True)
class CampaignWindow:
    t0_utc:object
    server:object=None
    def __post_init__(self):
        object.__setattr__(self,'t0_utc',utc(self.t0_utc))
        if self.server is not None:verify_lane(self.server)
    @property
    def admission_cutoff_utc(self):return self.t0_utc+timedelta(hours=34)
    @property
    def train_finish_utc(self):return self.t0_utc+timedelta(hours=37)
    @property
    def deadline_utc(self):return self.t0_utc+timedelta(hours=40)
    def to_dict(self):return dict(campaign_id=CAMPAIGN_ID,server=self.server,clock_policy='PER_SERVER_ACTUAL_START_40H',
        **{k:getattr(self,k).isoformat() for k in
        ('t0_utc','admission_cutoff_utc','train_finish_utc','deadline_utc')})
    @classmethod
    def from_dict(cls,value):
        result=cls(value['t0_utc'],value.get('server'))
        if result.to_dict()!=value:raise ValueError('Local P40 clock was changed or uses obsolete shared policy')
        return result

def nonnegative(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('Invalid measured/reserved cost')
    return float(value)

def estimate_case(case,observations=()):
    values=[nonnegative(v['wall_hours']) for v in observations if v.get('complete') is True
        and v.get('server')==case.server and v.get('updates')==case.updates
        and v.get('arm')==case.arm and v.get('family')==case.calibration_family
        and v.get('includes_eval_io_diagnostics') is True]
    return max(case.reservation_hours,max(values[-5:],default=0.)*1.15)

def estimate_block(block,observations=(),completed=()):
    if isinstance(block,str):block=block_for(block)
    if not set(completed)<=set(block.run_ids):raise ValueError('Foreign completed case')
    return sum(estimate_case(c,observations) for c in block.cases if c.run_id not in completed)

def admission(window,now,block,*,observations=(),completed=(),receipt=None,
        outstanding_hours=0.,debt_hours=0.,cache_hours=0.,diagnostic_hours=DIAGNOSTIC_HOURS):
    if isinstance(block,str):block=block_for(block)
    if window.server is not None and window.server!=block.server_id:
        raise ValueError('Admission cannot use another server\'s clock')
    now=utc(now)
    if now<window.t0_utc:raise ValueError('Cannot hide preparation before t0')
    if receipt:
        if (receipt.get('campaign_id')!=CAMPAIGN_ID or receipt.get('registry_sha256')!=registry_sha256()
            or receipt.get('window')!=window.to_dict() or receipt.get('block_id')!=block.block_id
            or receipt.get('run_ids')!=list(block.run_ids) or receipt.get('allowed') is not True
            or receipt.get('reason')!='ADMITTED' or not window.t0_utc<=utc(receipt['at_utc'])<window.admission_cutoff_utc
            or utc(receipt['at_utc'])+timedelta(hours=nonnegative(receipt['required_hours']))>window.train_finish_utc):
            raise ValueError('Exact original whole-block admission required')
    elif completed:raise ValueError('Started block lacks admission')
    remaining=estimate_block(block,observations,completed)
    required=remaining+sum(nonnegative(v) for v in (outstanding_hours,debt_hours,cache_hours,diagnostic_hours))
    if now>=window.train_finish_utc:reason='OPTIMIZER_CLOSED'
    elif not receipt and now>=window.admission_cutoff_utc:reason='NOT_ADMITTED_CUTOFF'
    elif not receipt and now+timedelta(hours=required)>window.train_finish_utc:reason='NOT_ADMITTED_BUDGET'
    else:reason='CONTINUE_ADMITTED_BLOCK' if receipt else 'ADMITTED'
    return dict(campaign_id=CAMPAIGN_ID,registry_sha256=registry_sha256(),window=window.to_dict(),
        server=block.server_id,block_id=block.block_id,run_ids=list(block.run_ids),at_utc=now.isoformat(),
        remaining_hours=remaining,outstanding_hours=outstanding_hours,debt_hours=debt_hours,
        cache_hours=cache_hours,diagnostic_hours=diagnostic_hours,required_hours=required,
        projected_overrun=now+timedelta(hours=required)>window.train_finish_utc,
        allowed=reason in ('ADMITTED','CONTINUE_ADMITTED_BLOCK'),reason=reason)

SCREEN_PAIRS={f:tuple((f'B{start+4*i:02d}',f'B{start+4*i+1:02d}') if j==0 else
    (f'B{start+4*i+3:02d}',f'B{start+4*i+2:02d}') for j in range(2))
    for i,f in enumerate(('G025','G050','G0125','P025','P075')) for start in (3,)}

def metric_vector(record):
    v=dict(H=record['fr']['hqnr'],E=record['rr']['ergas'],Ds=record['fr']['d_s'],Dl=record['fr']['d_lambda'])
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in v.values()) or v['E']<=0:
        raise ValueError('Nonfinite or invalid native endpoint metrics')
    return v

def effect(ctrl,mix,selection='EXACT_FINAL'):
    a,b=metric_vector(ctrl['selections'][selection]),metric_vector(mix['selections'][selection])
    return dict(delta_h=b['H']-a['H'],relative_e=b['E']/a['E']-1,delta_ds=b['Ds']-a['Ds'],delta_dlambda=b['Dl']-a['Dl'])

def complete_rows(rows,ids):
    return all(k in rows and rows[k].get('complete') is True and rows[k].get('integrity_verified') is True for k in ids)

def contrast(rows,pairs):
    if not complete_rows(rows,[v for p in pairs for v in p]):return dict(complete=False)
    values=[dict(control=c,alternative=m,**effect(rows[c],rows[m]),
                 val_delta_h=effect(rows[c],rows[m],'RR_VAL_SELECTED')['delta_h']) for c,m in pairs]
    mean=lambda key:sum(v[key] for v in values)/len(values)
    sensitive=any(v['val_delta_h']<-.001 for v in values)
    return dict(complete=True,effects=values,mean_delta_h=mean('delta_h'),min_delta_h=min(v['delta_h'] for v in values),
        mean_relative_e=mean('relative_e'),mean_delta_ds=mean('delta_ds'),checkpoint_sensitive=sensitive,
        spatial_signal=bool(len(values)==2 and all(v['delta_h']>0 and v['delta_ds']<0 and v['relative_e']<=.01
            and v['delta_dlambda']<=.001 for v in values) and mean('delta_h')>=.002 and mean('relative_e')<=.005 and not sensitive))

def select_family(rows):
    """Descriptive s4 screen only; never changes the fixed confirmation recipe."""
    inputs={f:contrast(rows,pairs) for f,pairs in SCREEN_PAIRS.items()}
    incumbent=SCREEN_PAIRS['G025'];upgrades={};eligible=[]
    for f in UPGRADE_ORDER:
        vs=contrast(rows,tuple((base[1],alt[1]) for base,alt in zip(incumbent,SCREEN_PAIRS[f])))
        good=bool(inputs[f].get('spatial_signal') and vs.get('complete') and vs['mean_delta_h']>=.001
            and vs['mean_relative_e']<=.005 and all(v['delta_h']>=-.0005 and v['delta_ds']<=.0005
                and v['relative_e']<=.01 and v['delta_dlambda']<=.001 for v in vs['effects']))
        upgrades[f]=dict(incumbent_comparison=vs,eligible=good)
        if good:eligible.append(f)
    chosen='G025'
    if eligible:
        chosen=eligible[0]
        for f in eligible[1:]:
            a,b=upgrades[f]['incumbent_comparison'],upgrades[chosen]['incumbent_comparison']
            delta=a['min_delta_h']-b['min_delta_h']
            if delta>1e-8 or abs(delta)<=1e-8 and a['mean_relative_e']<b['mean_relative_e']-1e-8:chosen=f
    all_complete=complete_rows(rows,[f'B{i:02d}' for i in range(1,23)])
    reason='VALIDATED_UPGRADE' if chosen!='G025' else 'NO_VALIDATED_UPGRADE' if all_complete else 'PARTIAL_SCREEN_FALLBACK'
    return dict(family=chosen,reason=reason,input_effects=inputs,upgrades=upgrades,screen_complete=all_complete)

# Only unstarted blocks can be shed; confirmation is protected until these go.
DROP_ORDER={
    's3':('S3_CONFIRM_3','A_LONG60_2'),
    's4':('S4_CONFIRM_3','B_P075_TWO_PARENTS','B_P025_TWO_PARENTS','B_G0125_TWO_PARENTS'),
    's5':('S5_CONFIRM_3','C_HIGH_TWO_PARENTS','C_SCHEDULE_2')}

def trim_unstarted(server,blocks,available_hours,observations=(),started=()):
    """Budget-only shedding, never performance-conditioned seed cancellation."""
    keep=list(blocks);dropped=[]
    order=DROP_ORDER[server]+(f'S{server[1]}_CONFIRM_2',)
    total=lambda:sum(estimate_block(b,observations) for b in keep)
    for name in order:
        if total()<=available_hours:break
        matches=[b for b in keep if b.block_id==name and name not in started]
        if matches:keep.remove(matches[0]);dropped.append(name)
    return keep,dropped
