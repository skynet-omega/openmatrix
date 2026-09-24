"""Measure weighted source changes on an already-completed real 1-ms run."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FLYWIRE=Path("/home/daroch/AXIOMA_FLYWIRE/matrix")
OUT=HERE/"rho_1ms_probe_01"
PLAN=HERE/"RHO_1MS_PLAN_01.json"
FINAL=HERE/"fusion_base_1ms_01/final_state/session.npz"
EXPECTED={
 "initial":"7d3d18ecdda76c2f99af6354bd5eaa7a420991d985b0828641bea9cab7af7687",
 "weights":"7fb2ee43cb45f9c5a366bad29a36d6b7f6771dff50d3c2699a74aaaf6418c999",
 "indices":"c0a2d02fbe1ab449a2cdf7377024361a625f201266f7d6415c3c8584dc67a2db",
}

def digest_array(value):
 return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()

def digest_file(path):
 h=hashlib.sha256()
 with path.open("rb") as handle:
  for block in iter(lambda:handle.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir()
 began=time.perf_counter();obj=None
 receipt={"schema":"graded_source_weighted_change_result_v1","status":"STARTED","new_simulated_ms":0,"stage_admission":False}
 try:
  plan=json.loads(PLAN.read_text())
  if digest_file(FINAL)!=plan["input_sha256"]:raise ValueError("Final snapshot identity changed")
  sys.path[:0]=[str(FLYWIRE/"work/motor14_20260922"),
                 str(FLYWIRE/"work/motor13_20260922"),
                 str(ROOT/"motor_nuevo/pipeline_review_20260922")]
  from motor_runtime import load
  obj,*_=load(OUT/"preparation_inputs")
  h=obj.core.hybrid; csr=h.brain.W;n=int(csr.shape[0])
  if n!=166700 or csr.nnz!=25582938:raise ValueError("Graph dimensions changed")
  if digest_array(h.weights64)!=EXPECTED["weights"] or digest_array(csr.indices)!=EXPECTED["indices"]:
   raise ValueError("Effective graph identity changed")
  start=np.asarray(h.state[h.transmission_start:h.transmission_start+n],dtype=np.float64)
  if digest_array(start)!=EXPECTED["initial"]:raise ValueError("Initial transmission differs from prior paired probe")
  with np.load(FINAL,allow_pickle=False) as saved:
   full=np.asarray(saved["array_261"],dtype=np.float64)
  if full.shape!=h.state.shape or not np.isfinite(full).all():raise ValueError("Final state shape/finiteness")
  stop=full[h.transmission_start:h.transmission_start+n]
  delta=np.abs(stop-start)
  outdegree=np.bincount(np.asarray(csr.indices,dtype=np.int32),minlength=n)
  if int(outdegree.sum())!=int(csr.nnz):raise ValueError("Outdegree mass mismatch")
  rows=[]
  for threshold in plan["diagnostic_thresholds_abs"]:
   changed=delta>threshold
   rows.append({"abs_threshold":threshold,"changed_sources":int(np.count_nonzero(changed)),
                "source_fraction":float(np.mean(changed)),
                "outgoing_edges_from_changed_sources":int(outdegree[changed].sum()),
                "weighted_rho":float(outdegree[changed].sum()/csr.nnz)})
  receipt.update(status="COMPLETE",neuron_count=n,edge_count=int(csr.nnz),
                 start_sha256=digest_array(start),final_state_sha256=digest_array(full),
                 delta_max=float(np.max(delta)),delta_median=float(np.median(delta)),
                 thresholds=rows,interpretation_limit=plan["interpretation_limit"])
 except BaseException as exc:
  receipt.update(status="FAILED_RETAINED",error_type=type(exc).__name__,error_message=str(exc)[:300])
 finally:
  if obj is not None:
   try:obj.close()
   except BaseException as exc:receipt.update(status="FAILED_CLEANUP",cleanup_error_type=type(exc).__name__)
  receipt["wall_seconds"]=time.perf_counter()-began
  (OUT/"RESULT.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
  print(json.dumps(receipt,indent=2))
 if receipt["status"].startswith("FAILED"):raise SystemExit(2)

if __name__=="__main__":main()
