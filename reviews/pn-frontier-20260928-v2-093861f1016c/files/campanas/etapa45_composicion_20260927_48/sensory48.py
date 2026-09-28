"""Prescribed ORN target intervention, not a calibrated peripheral transducer.

Only selected ORN targets change. Incoming modulation of those targets is
bypassed explicitly; initial states, tau, outgoing filters and all other target
laws are retained. Never writes a motor target or instantaneous neural state.
"""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import types
import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ('sham', 'dm1', 'profile', 'permuted')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def inputs(folder=HERE):
    folder = Path(folder)
    perm = json.loads((folder/'PERMUTATION_RESULT.json').read_text())
    prepared = json.loads((folder/'PREFLIGHT_RESULT.json').read_text())
    anatomy = json.loads((folder/'ANATOMIA_SELECCIONADA.json').read_text())
    rows = sorted(perm['records'], key=lambda r:r['id'])
    known = {r['id']:(r['type'],r['side']) for r in anatomy['records'] if r['side'] in ('L','R')}
    need(len(rows) == len(known) == 694, 'Wrong target population')
    ids = np.asarray([r['id'] for r in rows], dtype=np.int64)
    need(len(np.unique(ids)) == len(ids), 'Duplicate target')
    need(all(known[r['id']] == (r['type'],r['side']) for r in rows), 'Target identity differs')
    baseline = np.asarray([r['baseline_Hz'] for r in rows], dtype=np.float64)
    lookup = {(r['side'],'ORN_'+r['glomerulus']):r for r in prepared['profiles']
              if r['pattern'] == 'selective_DM1_matched'}
    delta = dict(sham=np.zeros(len(rows)),
        dm1=np.asarray([lookup[r['side'],r['type']]['increment_Hz'] for r in rows]),
        profile=np.asarray([float(Fraction(r['profile_increment_rational'])) for r in rows]),
        permuted=np.asarray([float(Fraction(r['permuted_increment_rational'])) for r in rows]))
    need(all(baseline[j] == lookup[r['side'],r['type']]['common_baseline_Hz']
             for j,r in enumerate(rows)), 'Basal policy differs')
    for side in ('L','R'):
        selected=[r for r in rows if r['side']==side]
        original=[Fraction(r['profile_increment_rational']) for r in selected]
        control=[Fraction(r['permuted_increment_rational']) for r in selected]
        need(Counter(original)==Counter(control), 'Changed increment histogram')
        need(float(sum(original)) == prepared['sides'][side]['target_weighted_increment'], 'Changed nominal quantity')
    need(np.isfinite(baseline).all() and np.all(baseline>=0), 'Invalid basal input')
    for arm,d in delta.items():
        need(d.shape==baseline.shape and np.isfinite(d).all() and np.all(d>=0), 'Invalid increment '+arm)
    return dict(ids=ids, baseline=baseline, delta=delta,
        types=np.asarray([r['type'] for r in rows]), sides=np.asarray([r['side'] for r in rows]))


def nominal(spec, arm, k, plan):
    need(arm in ARMS, 'Unknown input arm')
    need(type(k) is int and 1<=k<=plan['duration_ms'], 'Invalid consumed interval')
    active = plan['odor_on_ms'] < k <= plan['odor_off_ms']
    return spec['baseline'] + (spec['delta'][arm] if active else 0.)


CUDA = r'''
extern "C" __global__ void prescribed_orn(
 int n, const long long* rows, const double* nominal, const double* caps,
 double* target, double* observed_rate, double* observed_target) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;
 if(j>=n)return;
 double value=nominal[j]/caps[j];
 target[rows[j]]=value;
 observed_rate[j]=nominal[j]; observed_target[j]=target[rows[j]];
}
extern "C" __global__ void audit_consumed_orn(
 int n, const long long* rows, const double* nominal, const double* caps,
 const double* target, unsigned long long* errors, unsigned long long* calls,
 double* observed_rate, double* observed_target) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;
 if(j>=n)return;
 if(j==0)atomicAdd(calls,1ULL);
 double expected=nominal[j]/caps[j];
 if(target[rows[j]]!=expected)atomicAdd(errors,1ULL);
 observed_rate[j]=nominal[j];observed_target[j]=target[rows[j]];
}
'''


class TerminalInput:
    def __init__(self, brain, plan, arm, spec=None):
        import cupy as cp
        self.cp=cp;self.brain=brain;self.plan=plan;self.arm=arm
        self.spec=inputs() if spec is None else spec
        ids=self.spec['ids']
        self.rows=np.searchsorted(brain.brain.node_ids, ids).astype(np.int64)
        need(np.all(self.rows < len(brain.brain.node_ids)) and
             np.array_equal(brain.brain.node_ids[self.rows],ids), 'Missing ORN rows')
        need(not np.any(brain.visual_mask[self.rows]), 'ORN target overlaps visual law')
        self.caps=np.asarray(brain.caps[self.rows],dtype=np.float64)
        need(np.isfinite(self.caps).all() and np.all(self.caps>0), 'Invalid normalization caps')
        for d in self.spec['delta'].values():
            need(np.all(self.spec['baseline']+d<=self.caps), 'Rate exceeds cap; no silent clipping')
        self.device_rows=cp.asarray(self.rows)
        self.device_caps=cp.asarray(self.caps)
        self.device_rates=cp.asarray(self.spec['baseline'])
        self.seen_rates=cp.full(len(ids),np.nan,dtype=cp.float64)
        self.seen_target=cp.full(len(ids),np.nan,dtype=cp.float64)
        self.kernel=cp.RawKernel(CUDA,'prescribed_orn',options=('--fmad=false','--prec-div=true'))
        self.audit_kernel=cp.RawKernel(CUDA,'audit_consumed_orn',options=('--fmad=false','--prec-div=true'))
        self.errors=cp.zeros(1,dtype=cp.uint64);self.calls=cp.zeros(1,dtype=cp.uint64)
        self.original=brain.coefficients_gpu
        self.was_local='coefficients_gpu' in brain.__dict__
        self.k=0;self.current=self.spec['baseline'].copy()

        def coefficients(owner,state,drive,light):
            target,rate=self.original(state,drive,light)
            self.kernel(((len(ids)+127)//128,),(128,),
                (np.int32(len(ids)),self.device_rows,self.device_rates,self.device_caps,
                 target,self.seen_rates,self.seen_target))
            return target,rate
        brain.coefficients_gpu=types.MethodType(coefficients,brain)

    def consume(self,k):
        need(k==self.k+1, 'Input interval skipped or replayed')
        self.current=nominal(self.spec,self.arm,k,self.plan)
        self.device_rates.set(np.ascontiguousarray(self.current))
        self.errors.fill(0);self.calls.fill(0)
        # The runtime also waits for the default stream before its captured
        # private stream. Make ownership explicit at this external boundary.
        self.cp.cuda.get_current_stream().synchronize()
        self.k=k
        return self.current.copy()

    def observed(self):
        rates=self.seen_rates.get();target=self.seen_target.get()
        need(int(self.calls.get()[0])>0, 'No RHS consumption witness')
        need(int(self.errors.get()[0])==0, 'A final RHS consumed another ORN target')
        need(np.array_equal(rates,self.current), 'Captured graph consumed stale nominal input')
        need(np.array_equal(target,self.current/self.caps), 'Captured ORN target differs')
        return rates,target

    def audit(self,target):
        """Called by the RHS after the final target writer, for every stage."""
        n=len(self.rows)
        self.audit_kernel(((n+127)//128,),(128,),
            (np.int32(n),self.device_rows,self.device_rates,self.device_caps,target,
             self.errors,self.calls,self.seen_rates,self.seen_target))

    def state(self):
        return dict(schema='prescribed_orn_target_v1',arm=self.arm,consumed_ms=self.k,
            ids=self.spec['ids'],caps=self.caps,nominal_Hz=self.current,
            model_intervention=True,incoming_target_modulation_bypassed=True,
            instantaneous_state_clamped=False,tau_preserved=True,
            interpretation='Nominal q target; not calibrated odor, spike train or physiological release.')

    def close(self):
        if self.was_local:self.brain.coefficients_gpu=self.original
        else:self.brain.__dict__.pop('coefficients_gpu',None)
