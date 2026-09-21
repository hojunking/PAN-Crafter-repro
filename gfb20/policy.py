"""Shared clock, whole-block admission, and preregistered descriptive decisions."""
from dataclasses import dataclass
from datetime import datetime,timedelta,timezone
import math
from gfb20.plan import CAMPAIGN_ID,REGISTRY,block_for,registry_sha256


def utc(value):
    if isinstance(value,str): value=datetime.fromisoformat(value.replace('Z','+00:00'))
    if not isinstance(value,datetime) or value.tzinfo is None:
        raise ValueError('An explicit timezone-aware common t0 is required')
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class CampaignWindow:
    t0_utc: object
    def __post_init__(self): object.__setattr__(self,'t0_utc',utc(self.t0_utc))
    @property
    def admission_cutoff_utc(self): return self.t0_utc+timedelta(hours=16)
    @property
    def train_finish_utc(self): return self.t0_utc+timedelta(hours=18)
    @property
    def training_deadline_utc(self): return self.train_finish_utc
    @property
    def deadline_utc(self): return self.t0_utc+timedelta(hours=20)
    def to_dict(self):
        return dict(campaign_id=CAMPAIGN_ID,t0_utc=self.t0_utc.isoformat(),
            admission_cutoff_utc=self.admission_cutoff_utc.isoformat(),
            train_finish_utc=self.train_finish_utc.isoformat(),deadline_utc=self.deadline_utc.isoformat())
    @classmethod
    def from_dict(cls,value):
        obj=cls(value['t0_utc'])
        if obj.to_dict()!=value: raise ValueError('GFB20 shared clock changed or is incomplete')
        return obj


def nonnegative(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('Cost must be finite and nonnegative')
    return float(value)


def estimate_case(case, observations=()):
    matched=[nonnegative(o['wall_hours']) for o in observations if o.get('complete') is True
        and o.get('server')==case.server and o.get('updates')==case.updates
        and o.get('input_views')==case.input_views and o.get('includes_eval_io_diagnostics') is True]
    return max(case.reservation_hours,max(matched[-5:],default=0.)*1.15)


def estimate_block(block,observations=(),completed=()):
    if isinstance(block,str): block=block_for(block)
    if not set(completed)<=set(block.run_ids): raise ValueError('Completion belongs to another block')
    return sum(estimate_case(c,observations) for c in block.cases if c.run_id not in completed)


def validate_admission(receipt,block,window):
    if isinstance(block,str): block=block_for(block)
    if (receipt.get('campaign_id')!=CAMPAIGN_ID or receipt.get('block_id')!=block.block_id
        or receipt.get('server')!=block.server_id or receipt.get('run_ids')!=list(block.run_ids)
        or receipt.get('registry_sha256')!=registry_sha256() or receipt.get('window')!=window.to_dict()
        or receipt.get('allowed') is not True or receipt.get('reason')!='ADMITTED'):
        raise ValueError('Continuation requires original whole-block admission')
    start=utc(receipt['at_utc'])
    if not window.t0_utc<=start<window.admission_cutoff_utc:
        raise ValueError('Original admission outside allowed window')
    if start+timedelta(hours=nonnegative(receipt['required_hours']))>=window.train_finish_utc:
        raise ValueError('Original whole block could not fit')


def admission(window,now,block,*,remaining_hours=None,debt_hours=0.,observations=(),
              completed=(),receipt=None,integrity=True):
    if isinstance(window,dict): window=CampaignWindow.from_dict(window)
    if isinstance(block,str): block=block_for(block)
    now=utc(now)
    if now<window.t0_utc: raise ValueError('Common clock has not started')
    minimum=estimate_block(block,observations,completed)
    remaining=minimum if remaining_hours is None else nonnegative(remaining_hours)
    if remaining<minimum: raise ValueError('Cannot underreserve the unfinished whole block')
    if receipt: validate_admission(receipt,block,window)
    elif completed: raise ValueError('Partial block lacks original admission')
    required=remaining+nonnegative(debt_hours)
    if not integrity: reason='BLOCKED_INTEGRITY'
    elif now>=window.train_finish_utc: reason='OPTIMIZER_CLOSED'
    elif not receipt and now>=window.admission_cutoff_utc: reason='NOT_ADMITTED_CUTOFF'
    elif now+timedelta(hours=required)>=window.train_finish_utc: reason='NOT_ADMITTED_BUDGET'
    else: reason='CONTINUE_ADMITTED_BLOCK' if receipt else 'ADMITTED'
    return dict(campaign_id=CAMPAIGN_ID,server=block.server_id,block_id=block.block_id,
        run_ids=list(block.run_ids),registry_sha256=registry_sha256(),window=window.to_dict(),
        at_utc=now.isoformat(),remaining_hours=remaining,debt_hours=debt_hours,required_hours=required,
        allowed=reason in ('ADMITTED','CONTINUE_ADMITTED_BLOCK'),reason=reason)


CONTRASTS = {
    'S3_H010_R100':(('A01','A02'),('A05','A06')),
    'S3_H010_OLD':(('A04','A03'),),
    'S3_STREAM':(('A08','A07'),('A09','A10')),
    'S3_FRESH_REPLAY':(('A12','A13'),('A15','A16')),
    'S4_K1':(('B01','B02'),('B06','B05')),
    'S4_E100':(('B01','B03'),('B06','B04')),
    'S4_FRESH_TRANSFER':(('B07','B08'),('B10','B09')),
    'S5_INPUT':(('C02','C03'),('C05','C04')),
    'S5_SCALE':(('C01','C02'),('C06','C05')),
    'S5_TOTAL':(('C01','C03'),('C06','C04')),
    'S5_FRESH_TRANSFER':(('C07','C08'),('C10','C09'))}


def metric_vector(selection):
    values=dict(H=selection['fr']['hqnr'],E=selection['rr']['ergas'],
                Ds=selection['fr']['d_s'],Dl=selection['fr']['d_lambda'])
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values.values()) or values['E']<=0:
        raise ValueError('Invalid paired official metrics')
    return values


def paired_effect(control,alternative,selection='EXACT_FINAL'):
    a,b=metric_vector(control['selections'][selection]),metric_vector(alternative['selections'][selection])
    return dict(delta_h=b['H']-a['H'],relative_e=b['E']/a['E']-1,
                delta_ds=b['Ds']-a['Ds'],delta_dlambda=b['Dl']-a['Dl'])


def summarize_contrast(rows,pairs):
    complete=all(c in rows and a in rows and rows[c].get('complete') is True
                 and rows[a].get('complete') is True for c,a in pairs)
    if not complete: return dict(complete=False,status='INCOMPLETE_OR_FAILED',pairs=[list(p) for p in pairs])
    effects=[]
    for c,a in pairs:
        exact=paired_effect(rows[c],rows[a])
        val=paired_effect(rows[c],rows[a],'RR_VAL_SELECTED')
        effects.append(dict(control=c,alternative=a,**exact,val_delta_h=val['delta_h']))
    mean=lambda k: sum(e[k] for e in effects)/len(effects)
    sensitive=any(e['val_delta_h']<-.001 for e in effects)
    spatial=(len(effects)==2 and all(e['delta_h']>0 and e['delta_ds']<0
        and e['relative_e']<=.01 and e['delta_dlambda']<=.001 for e in effects)
        and mean('delta_h')>=.002 and mean('relative_e')<=.005)
    rr=(len(effects)==2 and mean('relative_e')<=-.003
        and all(e['delta_h']>=-.001 for e in effects) and mean('delta_ds')<=.0005)
    return dict(complete=True,effects=effects,mean_delta_h=mean('delta_h'),min_delta_h=min(e['delta_h'] for e in effects),
        mean_relative_e=mean('relative_e'),mean_delta_ds=mean('delta_ds'),checkpoint_sensitive=sensitive,
        spatial_signal=spatial,rr_preserving_signal=rr,
        status='CHECKPOINT_SENSITIVE' if sensitive else ('SPATIAL_SIGNAL' if spatial else 'NO_SPATIAL_SIGNAL'))


def low_active_mass(diagnostics):
    required=(0,5000,20000)
    if any(str(k) not in diagnostics and k not in diagnostics for k in required): return None
    vals=[diagnostics.get(str(k),diagnostics.get(k)) for k in required]
    return all(v['weighted_soft_hard_ratio']<1e-3 and v['advantage_positive_fraction']<1e-4 for v in vals)


def promotion(server,rows):
    contrasts={name:summarize_contrast(rows,pairs) for name,pairs in CONTRASTS.items() if name.startswith('S'+server[1])}
    result=dict(server=server,contrasts=contrasts,test_aware=True,statistical_significance=False,
                fresh_signal=False,fresh_allowed_after_negative=False)
    if server=='s3':
        core=contrasts['S3_H010_R100']
        result['ready']=core['complete']
        result['fresh_signal']=bool(core['complete'] and not core['checkpoint_sensitive']
                                  and (core['spatial_signal'] or core['rr_preserving_signal']))
    elif server=='s4':
        values={p:contrasts['S4_'+p] for p in ('K1','E100')}
        result['ready']=all(v['complete'] for v in values.values())
        if not result['ready']: return result
        qualified=[p for p,v in values.items() if v['spatial_signal'] and not v['checkpoint_sensitive']]
        chosen='E100'
        if len(qualified)==1: chosen=qualified[0]
        elif len(qualified)==2:
            k,e=values['K1'],values['E100']
            delta=k['min_delta_h']-e['min_delta_h']
            if abs(delta)>1e-8: chosen='K1' if delta>0 else 'E100'
            elif abs(k['mean_relative_e']-e['mean_relative_e'])>1e-8:
                chosen='K1' if k['mean_relative_e']<e['mean_relative_e'] else 'E100'
            else: chosen='K1'
        result.update(selected_profile=chosen,fresh_signal=bool(qualified),
                      fresh_allowed_after_negative=not bool(qualified))
    elif server=='s5':
        inp,total=contrasts['S5_INPUT'],contrasts['S5_TOTAL']
        result['ready']=inp['complete'] and total['complete']
        if result['ready']:
            signal=(inp['spatial_signal'] and not inp['checkpoint_sensitive']
                and total['mean_delta_h']>0 and all(e['relative_e']<=.01 for e in total['effects']))
            result.update(fresh_signal=signal,fresh_allowed_after_negative=not signal)
    else: raise ValueError('Protected/unknown server')
    result['status']=('INCOMPLETE_CORE' if not result['ready'] else 'PROMOTED_SIGNAL' if result['fresh_signal']
        else 'FRESH_PATH_TEST_AFTER_NEGATIVE_FT' if result['fresh_allowed_after_negative'] else 'NO_REPLAY_SIGNAL')
    return result


def block_eligible(block,decision,*,mixed_calibration_hours=0.):
    if isinstance(block,str): block=block_for(block)
    if block.stage in ('CORE','STREAM_CHECK'): return True,'REGISTERED_CORE_OR_STREAM'
    if not decision.get('ready'): return False,decision.get('status','INCOMPLETE_CORE')
    if block.server_id=='s3' and not decision.get('fresh_signal'): return False,'NO_R100_REPLAY_SIGNAL'
    if block.block_id=='C_FRESH2' and mixed_calibration_hours>2: return False,'MIXED_CACHE_OVER_2H_DROP_SECOND_FRESH'
    return True,decision['status']
