"""Bounds proposed by ChatGPT; this implementation was written independently.

For an odd nondecreasing |f|<=1, f on nonnegative sample amplitudes is
a positive mixture of threshold steps plus unused mass on zero. Thus the
weighted integral extrema are the extrema of signed threshold tail sums.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import time
import numpy as np


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def monotone_bound(x, weights_ns, ceiling=5.):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(weights_ns)
    need(x.ndim == w.ndim == 1 and x.shape == w.shape, 'Shape mismatch')
    need(np.isfinite(x).all(), 'Nonfinite state')
    need(w.dtype.kind in 'iu', 'Weights must be signed integer durations')
    need(sum(abs(int(v)) for v in w) <= np.iinfo(np.int64).max, 'Duration overflow')
    need(np.isfinite(ceiling) and ceiling > 0, 'Invalid command envelope')
    active = x != 0
    a = np.abs(x[active]); signed = np.sign(x[active]).astype(np.int64)*w[active].astype(np.int64)
    if not len(a):
        return dict(lower_deg=0., upper_deg=0., unique_amplitudes=0), (a, signed)
    order = np.argsort(-a, kind='stable')
    a = a[order]; cumulative = np.cumsum(signed[order], dtype=np.int64)
    ends = np.r_[a[:-1] != a[1:], True]
    sums = cumulative[ends]
    lo = min(0, int(np.min(sums))); hi = max(0, int(np.max(sums)))
    return dict(lower_deg=ceiling*1e-9*lo, upper_deg=ceiling*1e-9*hi,
                unique_amplitudes=int(ends.sum())), (a[ends], sums)


def exhaustive_threshold_bound(x, w, ceiling=5.):
    # Deliberately independent O(n^2) formula; small recorded tapes make it cheap.
    vals = [0]
    for a in sorted(set(abs(float(v)) for v in x if v != 0)):
        vals.append(sum(int(t)*(1 if v > 0 else -1) for v,t in zip(x,w) if abs(v) >= a))
    return ceiling*1e-9*min(vals), ceiling*1e-9*max(vals)


def selftest():
    x = np.array([-.8, .2, -.2, .5, .8])
    w = np.array([2,3,4,5,6], dtype=np.int64)*1000000
    b,_ = monotone_bound(x,w)
    slow = exhaustive_threshold_bound(x,w)
    need((b['lower_deg'],b['upper_deg']) == slow, 'Threshold enumeration mismatch')
    amps = sorted(set(abs(x)))
    for values in itertools.combinations_with_replacement([0.,.25,.5,.75,1.], len(amps)):
        f = dict(zip(amps,values))
        integral = 5e-9*math.fsum(float(t)*np.sign(v)*f[abs(v)] for v,t in zip(x,w))
        need(b['lower_deg']-1e-12 <= integral <= b['upper_deg']+1e-12, 'Monotone function escaped bound')
    tied,_ = monotone_bound(np.array([-1.,1.]), np.array([1000000,1000000]))
    need(tied['lower_deg'] == tied['upper_deg'] == 0., 'Ties must move together')
    shared,_ = monotone_bound(np.r_[x,x],np.r_[w,-w])
    need(shared['lower_deg'] == shared['upper_deg'] == 0., 'Identical shared-reader difference')
    zeros,_ = monotone_bound(np.zeros(2),np.array([1,2]))
    need(zeros['lower_deg'] == zeros['upper_deg'] == 0., 'Odd f(0)')
    rejected=0
    for a,t in ((np.array([np.nan]),np.array([1])),(np.array([1.]),np.array([.1])),(np.ones(2),np.ones(1,dtype=int))):
        try:monotone_bound(a,t)
        except ValueError:rejected+=1
    need(rejected == 3, 'Invalid input escaped')
    return dict(ties=True,monotone_enumeration=True,shared_reader=True,zero=True,invalid_inputs_rejected=rejected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,required=True)
    args=parser.parse_args();start=time.perf_counter();root=Path(__file__).resolve().parent
    p=json.loads((root/'PLAN.json').read_text());bplan=json.loads((root/'BOUND_PLAN.json').read_text())
    need(not args.out.exists(),'Unique output required');args.out.mkdir(parents=True)
    tests=selftest(); signals=[]; times=[]; bounds={}; raw={}
    for name in bplan['inputs']:
        row=p['inputs'][name];path=Path(row['path']);need(sha(path)==row['sha256'],'Input identity changed')
        with np.load(path,allow_pickle=False) as z:
            q=z['DN_q_usada'][40:];base=z['DN_baseline'][40:]
            x=(q[:,2]-base[:,2])-(q[:,3]-base[:,3])
            ticks=np.diff(z['CNS_time_ns'][39:])
            need(len(ticks)==len(x) and np.all(ticks==1000000),'Unexpected clock')
            need(np.isfinite(q).all() and np.isfinite(base).all(),'Nonfinite state')
            signals.append(x);times.append(ticks)
            result,curves=monotone_bound(x,ticks,bplan['ceiling_deg_s'])
            slow=exhaustive_threshold_bound(x,ticks,bplan['ceiling_deg_s'])
            need(np.allclose([result['lower_deg'],result['upper_deg']],slow,rtol=0,atol=1e-12),'Independent real bound mismatch')
            result['current_command_deg']=float(np.rad2deg(z['command_yaw_rate_rad_s'][40:]).sum()*.001)
            need(result['lower_deg']-1e-12 <= result['current_command_deg'] <= result['upper_deg']+1e-12,'Archived current reader violates bound')
            bounds[name]=result;raw[name+'_amplitudes']=curves[0];raw[name+'_tail_ns']=curves[1]
    need(np.array_equal(times[0],times[1]),'Unmatched paired clocks')
    x=np.r_[signals[1],signals[0]];w=np.r_[times[1],-times[0]]
    shared,curves=monotone_bound(x,w,bplan['ceiling_deg_s'])
    slow=exhaustive_threshold_bound(x,w,bplan['ceiling_deg_s'])
    need(np.allclose([shared['lower_deg'],shared['upper_deg']],slow,rtol=0,atol=1e-12),'Paired bound mismatch')
    raw['shared_amplitudes']=curves[0];raw['shared_tail_ns']=curves[1]
    shared['current_command_difference_deg']=bounds['long_equal']['current_command_deg']-bounds['long_identity']['current_command_deg']
    need(shared['lower_deg']-1e-12 <= shared['current_command_difference_deg'] <= shared['upper_deg']+1e-12,'Current shared reader violates bound')
    np.savez_compressed(args.out/'tails.npz',**raw)
    out=dict(individual=bounds,shared_equal_minus_original=shared,tests=tests,
        source_sha256=sha(Path(__file__)),plan_sha256=sha(root/'BOUND_PLAN.json'),
        input_sha256={n:p['inputs'][n]['sha256'] for n in bplan['inputs']},
        classification='CONFIRMED_ALGEBRA_ON_FIXED_SIGNALS_ONLY',
        no_body_bound=True,no_gain_selected=True,new_organism_runs=0,stage4_admitted=False,stage5_admitted=False)
    (args.out/'RESULT.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    runtime=dict(wall_s=time.perf_counter()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    need(runtime['wall_s']<bplan['budget']['cpu_wall_s'] and runtime['peak_rss_gib']<bplan['budget']['max_rss_gib'],'Resource budget exceeded')
    (args.out/'RUNTIME.json').write_text(json.dumps(runtime,indent=2)+'\n')
    print(json.dumps(dict(bounds=out['individual'],shared=shared,runtime=runtime)))


if __name__=='__main__':
    main()
