"""Format-dependent lower bound for the CURRENT repeated full-CSR coefficient pass.

Not a universal physical lower bound for alternative algorithms or compression.
"""
import hashlib
import json
from pathlib import Path

import cupy as cp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTRACT = ROOT / "motor_nuevo/abc_20260922/contract.json"
SOURCE = ROOT / "motor_nuevo/native_hybrid_20260922/legacy_sources/gpu_visual_brain.py"
RESULT = HERE / "BANDWIDTH_FLOOR_01.json"

def main():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    contract = json.loads(CONTRACT.read_text())
    edges = int(contract["stored_edges"])
    if edges != 25582938:
        raise ValueError("Graph edge count changed")
    source = SOURCE.read_text()
    if "for(long long e=ptr[row]+lane; e<ptr[row+1]; e+=32)" not in source:
        raise ValueError("Full CSR scan source no longer matches")
    # Paired real-epoch receipt, not a theoretical step count.
    run = json.loads((HERE/"fusion_base_1ms_01/RESULT.json").read_text())
    accepted = int(run["runtime"]["CNS"]["accepted"])
    rejected = int(run["runtime"]["CNS"]["rejected"])
    if accepted != 191 or rejected != 0:
        raise ValueError("Real trial count changed")
    evals = 6 * (accepted + rejected)
    p = cp.cuda.runtime.getDeviceProperties(0)
    clock_khz = int(p["memoryClockRate"])
    bus_bits = int(p["memoryBusWidth"])
    l2_bytes = int(p["l2CacheSize"])
    bandwidth_bytes_s = clock_khz * 1000 * 2 * (bus_bits / 8)
    bytes_weights_indices = edges * (8 + 4)
    bytes_total_nominal = bytes_weights_indices * evals
    # Optimistic L2 floor: up to the whole L2 remains useful across full passes.
    bytes_total_after_l2_ideal = max(0,bytes_weights_indices-l2_bytes)*evals
    target_s_per_ms = 60/1000
    result = {
        "schema":"current_full_csr_format_bandwidth_floor_v1",
        "claim_scope":"Conditional on the existing CSR kernel scanning every stored edge in all six coefficient evaluations per adaptive trial; not a bound on new methods",
        "graph_contract_sha256":hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "coefficient_source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "stored_edges":edges,"real_trials_per_ms":accepted+rejected,
        "coefficients_per_trial":6,"coefficient_evaluations_per_ms":evals,
        "index_bytes":4,"weight_bytes":8,
        "graph_stream_bytes_per_evaluation":bytes_weights_indices,
        "graph_stream_bytes_per_ms_nominal":bytes_total_nominal,
        "device":p["name"].decode(),
        "memory_clock_khz_reported":clock_khz,
        "memory_bus_bits":bus_bits,"L2_bytes":l2_bytes,
        "theoretical_peak_bytes_per_s":bandwidth_bytes_s,
        "ideal_nominal_stream_seconds_per_ms":bytes_total_nominal/bandwidth_bytes_s,
        "ideal_after_full_L2_reuse_seconds_per_ms":bytes_total_after_l2_ideal/bandwidth_bytes_s,
        "target_wall_seconds_per_ms":target_s_per_ms,
        "nominal_stream_vs_target_ratio":bytes_total_nominal/bandwidth_bytes_s/target_s_per_ms,
        "ideal_L2_stream_vs_target_ratio":bytes_total_after_l2_ideal/bandwidth_bytes_s/target_s_per_ms,
        "exclusions":["recurrent state, target/rate writes, specialized PN/KC, body, compute, launch costs, bandwidth not at peak"],
        "reasoning":"The existing kernel scans all CSR entries per coefficient evaluation. The edge arrays exceed L2. A fused host loop with unchanged six full passes cannot meet the target under this FP64+int32 representation even at peak streaming bandwidth."
    }
    RESULT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
