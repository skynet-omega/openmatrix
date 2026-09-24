"""Bounded arithmetic-precision preflight on the real CSR, outside the organism."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import cupy as cp
import numpy as np

from csr_encoding_probe import EXPECTED, sha_array, source_kernel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=Path("/home/daroch/AXIOMA_FLYWIRE/matrix")
OUT=HERE/"fast_fp32_probe_01"
FAST=HERE/"fast_fp32_coefficient.cu"

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    started=time.perf_counter()
    obj=None
    result={"schema":"fp32_fast_csr_preflight_v1","status":"STARTED",
            "organism_trial_ms":0,"fast_mode_only":True,"stage_admission":False}
    try:
        sys.path[:0]=[str(OLD/"work/motor14_20260922"),
                       str(OLD/"work/motor13_20260922"),
                       str(ROOT/"motor_nuevo/pipeline_review_20260922")]
        from motor_runtime import load
        obj,_,_,_,_,_=load(OUT/"preparation_inputs")
        h=obj.core.hybrid;W=h.brain.W;n=int(W.shape[0])
        w=np.asarray(h.weights64)
        if n!=166700 or W.nnz!=25582938 or sha_array(w)!=EXPECTED["weights"] or sha_array(W.indices)!=EXPECTED["indices"]:
            raise ValueError("Graph identity changed")
        visual_host=np.asarray(h.visual_mask,dtype=np.bool_)
        connected=bool(h.visual_output_connected)
        if connected:
            effective_weight_edges=int(W.nnz)
        else:
            visual_rows=np.repeat(visual_host,np.diff(W.indptr))
            effective_weight_edges=int(np.count_nonzero(visual_rows | ~visual_host[W.indices]))
        result.update(loop_edges_per_evaluation=int(W.nnz),
                      effective_weight_edges_per_evaluation=effective_weight_edges,
                      visual_output_connected=connected,
                      logical_index_and_effective_weight_bytes_per_evaluation=int(4*W.nnz+8*effective_weight_edges))
        base_source=source_kernel()
        fast_source=FAST.read_text()
        options=("--std=c++11","--fmad=false","--prec-div=true","--prec-sqrt=true")
        baseline=cp.RawKernel(base_source,"coefficient",options=options)
        fast=cp.RawKernel(fast_source,"coefficient_fast",options=options)
        ptr=cp.asarray(W.indptr,dtype=cp.int64);idx=cp.asarray(W.indices,dtype=cp.int32)
        common64=(cp.asarray(w),cp.asarray(h.caps),cp.asarray(h.visual_mask),
                  cp.asarray(h.tau),cp.asarray(h.rate_gain),cp.asarray(h.rate_theta))
        common32=tuple(common64[i] if i==2 else cp.asarray(cp.asnumpy(v).astype(np.float32))
                       for i,v in enumerate(common64))
        initial=np.asarray(h.state[h.transmission_start:h.transmission_start+n],dtype=np.float64)
        if initial.shape!=(n,) or not np.isfinite(initial).all():
            raise ValueError("Initial release shape/finiteness")
        coordinate=np.arange(n,dtype=np.float64)
        cases={"real":initial,
               "smooth_1pct":initial*(1.0+0.01*np.sin(coordinate*0.013)),
               "near_threshold_1pct":initial*0.01}
        versions={}
        samples={}
        for name,release_host in cases.items():
            if not np.isfinite(release_host).all():
                raise ValueError("Nonfinite input")
            r64=cp.asarray(release_host);r32=cp.asarray(release_host.astype(np.float32))
            drive64=cp.zeros(n,dtype=cp.float64);photo64=cp.zeros(n,dtype=cp.float64)
            drive32=cp.zeros(n,dtype=cp.float32);photo32=cp.zeros(n,dtype=cp.float32)
            args64=(np.int32(n),ptr,idx,common64[0],r64,common64[1],common64[2],
                    common64[3],common64[4],common64[5],drive64,photo64,
                    np.float64(h.parameters["conductance_per_stored_weight"]),
                    np.bool_(h.visual_output_connected))
            args32=(np.int32(n),ptr,idx,common32[0],r32,common32[1],common32[2],
                    common32[3],common32[4],common32[5],drive32,photo32,
                    np.float32(h.parameters["conductance_per_stored_weight"]),
                    np.bool_(h.visual_output_connected))
            out64=(cp.empty(n,dtype=cp.float64),cp.empty(n,dtype=cp.float64))
            out32=(cp.empty(n,dtype=cp.float64),cp.empty(n,dtype=cp.float64))
            grid=((n*32+255)//256,)
            baseline(grid,(256,),args64+out64)
            fast(grid,(256,),args32+out32)
            t64,r64o=(cp.asnumpy(v) for v in out64)
            t32,r32o=(cp.asnumpy(v) for v in out32)
            if not all(np.isfinite(v).all() for v in (t64,r64o,t32,r32o)):
                raise ValueError("Nonfinite operator output")
            dt=np.abs(t64-t32);dr=np.abs(r64o-r32o)
            worst=int(np.argmax(dt));worstr=int(np.argmax(dr/(1+np.abs(r64o))))
            versions[name]={"target_max_abs":float(dt[worst]),"target_worst_row":worst,
                            "target_ref_at_worst":float(t64[worst]),
                            "rate_max_scaled":float((dr/(1+np.abs(r64o)))[worstr]),
                            "rate_worst_row":worstr,
                            "target_sign_or_zero_changed":int(np.count_nonzero((t64>0)!=(t32>0)))}
            if name=="real":
                for _ in range(3):
                    baseline(grid,(256,),args64+out64);fast(grid,(256,),args32+out32)
                cp.cuda.get_current_stream().synchronize()
                samples={"baseline":[],"fast":[]}
                for i in range(12):
                    for mode in (("baseline","fast") if i%2==0 else ("fast","baseline")):
                        kernel,args,out=(baseline,args64,out64) if mode=="baseline" else (fast,args32,out32)
                        before,after=cp.cuda.Event(),cp.cuda.Event()
                        before.record();kernel(grid,(256,),args+out);after.record();after.synchronize()
                        samples[mode].append(float(cp.cuda.get_elapsed_time(before,after)))
        median={k:statistics.median(v) for k,v in samples.items()}
        target_error=max(v["target_max_abs"] for v in versions.values())
        rate_error=max(v["rate_max_scaled"] for v in versions.values())
        speedup=median["baseline"]/median["fast"]
        error_gate=target_error<=0.00005 and rate_error<=0.0001
        speed_gate=speedup>=2.0
        result.update(status="COMPLETE_LOCAL_PROMISING" if error_gate and speed_gate else "COMPLETE_NEGATIVE",
                      source_sha256=EXPECTED["source"],fast_kernel_sha256=hashlib.sha256(FAST.read_bytes()).hexdigest(),
                      cases=versions,kernel_ms_samples=samples,kernel_ms_median=median,
                      fast_kernel_speedup=speedup,max_target_abs_error=target_error,
                      max_rate_scaled_error=rate_error,error_gate=error_gate,speed_gate=speed_gate,
                      limits="Only three frozen coefficient inputs and an isolated CSR kernel; no event, long trajectory, specialized models or body comparison")
    except BaseException as e:
        result.update(status="FAILED_RETAINED",error_type=type(e).__name__,
                      error_message=str(e)[:300])
    finally:
        if obj is not None:
            try: obj.close()
            except BaseException as e: result.update(status="FAILED_CLEANUP",cleanup_error_type=type(e).__name__)
        result["wall_s"]=time.perf_counter()-started
        (OUT/"RESULT.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k!="kernel_ms_samples"},indent=2))
    if result["status"].startswith("FAILED"):
        raise SystemExit(2)

if __name__=="__main__":
    main()
