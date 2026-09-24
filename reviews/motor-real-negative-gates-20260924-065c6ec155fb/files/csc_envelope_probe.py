"""Read-only error-envelope diagnostic on a real completed 1-ms organism run."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=Path("/home/daroch/AXIOMA_FLYWIRE/matrix")
OUT=HERE/"csc_envelope_probe_01"
FINAL=HERE/"fusion_base_1ms_01/final_state/session.npz"
PLAN=HERE/"CSC_ENVELOPE_PLAN_01.json"
EXPECTED_START="7d3d18ecdda76c2f99af6354bd5eaa7a420991d985b0828641bea9cab7af7687"

def hash_array(value):return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
def hash_file(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for part in iter(lambda:stream.read(1024*1024),b""):h.update(part)
 return h.hexdigest()

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();started=time.perf_counter();obj=None
 r={"schema":"graded_csc_omission_envelope_result_v1","status":"STARTED","new_simulated_ms":0,"stage_admission":False}
 try:
  plan=json.loads(PLAN.read_text())
  if hash_file(FINAL)!=plan["input_sha256"]:raise ValueError("Final snapshot changed")
  sys.path[:0]=[str(OLD/"work/motor14_20260922"),str(OLD/"work/motor13_20260922"),str(ROOT/"motor_nuevo/pipeline_review_20260922")]
  from motor_runtime import load
  obj,*_=load(OUT/"preparation_inputs")
  h=obj.core.hybrid; W=h.brain.W;n=int(W.shape[0]);w=np.asarray(h.weights64,dtype=np.float64)
  if n!=166700 or W.nnz!=25582938 or hash_array(w)!=plan["graph_weights_sha256"]:raise ValueError("Graph identity mismatch")
  if not bool(h.visual_output_connected):raise ValueError("This envelope assumes all graph edges are enabled")
  start=np.asarray(h.state[h.transmission_start:h.transmission_start+n],dtype=np.float64)
  if hash_array(start)!=EXPECTED_START:raise ValueError("Initial state changed")
  with np.load(FINAL,allow_pickle=False) as saved: full=np.asarray(saved["array_261"],dtype=np.float64)
  if full.shape!=h.state.shape or not np.isfinite(full).all():raise ValueError("Final state shape/finiteness")
  delta=np.abs(full[h.transmission_start:h.transmission_start+n]-start)
  caps=np.asarray(h.caps,dtype=np.float64)
  visual_scale=abs(float(h.parameters["conductance_per_stored_weight"]))
  source_gain=np.maximum(np.abs(caps),visual_scale)
  abs_operator=csr_matrix((np.abs(w),W.indices,W.indptr),shape=W.shape)
  rows=[]
  for threshold in plan["thresholds_abs"]:
   omitted=(delta<=threshold)
   upper=np.asarray(abs_operator.dot(np.where(omitted,delta*source_gain,0.)),dtype=np.float64)
   if not np.isfinite(upper).all():raise ValueError("Nonfinite envelope")
   rows.append({"threshold":threshold,"omitted_sources":int(np.count_nonzero(omitted)),
                "input_bound_max":float(upper.max()),"input_bound_p99":float(np.percentile(upper,99)),
                "input_bound_median":float(np.median(upper)),"rows_bound_over_1e_minus_6":int(np.count_nonzero(upper>1e-6)),
                "rows_bound_over_1e_minus_4":int(np.count_nonzero(upper>1e-4))})
  r.update(status="COMPLETE",neuron_count=n,edge_count=int(W.nnz),visual_scale=visual_scale,
           source_gain_max=float(source_gain.max()),rows=rows,interpretation=plan["decision"])
 except BaseException as e:r.update(status="FAILED_RETAINED",error_type=type(e).__name__,error_message=str(e)[:300])
 finally:
  if obj is not None:
   try:obj.close()
   except BaseException as e:r.update(status="FAILED_CLEANUP",cleanup_error_type=type(e).__name__)
  r["wall_seconds"]=time.perf_counter()-started
  (OUT/"RESULT.json").write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
  print(json.dumps(r,indent=2))
 if r["status"].startswith("FAILED"):raise SystemExit(2)

if __name__=="__main__":main()
