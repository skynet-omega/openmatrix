"""Conservative static write-order map for neuron target/rate coefficients."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import signal
import time

import numpy as np


HERE = Path(__file__).resolve().parent
CAPTURE = HERE.parent / "multirate_real_20260924_01/capture_01"


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


# Listed from deepest base outward. "add" and "transform" consume earlier
# target values; a later touch does not necessarily erase an earlier layer.
# Conditional enablement is deliberately ignored.
WRITERS = (
    ("measured_t4", "_GpuMeasuredT4VisualBrain", "_mi9_cuda/rows", "add", None),
    ("cyborg_retinal_port", "_GpuCyborgRetinalPortBrain", "_retinal_port_cuda_rows", "set", "set"),
    ("pvlp_adaptation", "_GpuPvlpAdaptationBrain", "_pvlp_cuda_rows", "set", None),
    ("orn_pn", "_GpuProstheticOlfactoryBrain", "_orn_pn_cuda/pn_rows", "set", None),
    ("regional", "_GpuKcGammaRegionalBrain", "_regional_cuda/rows", "set", None),
    ("t4_gaba", "_GpuT4GabaBrain", "_gaba_cuda/rows", "add", None),
    ("retinal_transduction", "_GpuRetinalTransductionBrain", "_retinal_cuda/rows", "set", "set"),
    ("cvn7", "_GpuCvn7ConductanceBrain", "_cvn7_cuda_rows", "set", "set"),
    ("lamina_boundary", "_GpuLaminaBoundaryBrain", "_boundary_cuda/rows", "transform", "transform"),
    ("pnkc_receptor", "_GpuPnkcReceptorBrain", "_pnkc_cuda/rows", "set", None),
    ("kcgamma_output", "_GpuKcGammaOutputBrain", "_output_cuda/rows", "set", None),
    ("kc_apl_dynamic", "_GpuKcAplDynamicBrain", "_dynamic_gpu_rows", "set", "set"),
    ("orn_terminal", "_OrnPeripheralTerminalMixin", "_orn_terminal_cuda/rows", "set", None),
)


def run(out: Path) -> dict:
    start = time.monotonic()
    need(not out.exists(), "Output must be unique")
    plan_path = HERE / "PRECEDENCE_REPAIR_PLAN_08.json"
    repair = json.loads(plan_path.read_text())
    need(repair["schema"] == "owner_precedence_repair_v2", "Wrong repair plan")
    need(sha(HERE / "precedence_01/RESULT.json") == repair["invalid_v1_result_sha256"]
         and sha(HERE / "owner_precedence_screen_v1.py") == repair["v1_source_sha256"],
         "Previous failure changed")
    semantic_root = HERE.parent / "native_hybrid_20260922/legacy_sources"
    need({name: sha(semantic_root / name) for name in repair["verified_semantic_sources_sha256"]}
         == repair["verified_semantic_sources_sha256"], "Owner semantics source changed")
    plan = json.loads((HERE / "PRECEDENCE_PLAN_07.json").read_text())
    need(plan["schema"] == "owner_write_precedence_screen_v1", "Wrong plan")
    paths = {
        "gpu_coefficient_layout.py": CAPTURE / "executed_sources/legacy_sources__gpu_coefficient_layout.py",
        "EFFECTIVE_ARRAY_PATHS.json": CAPTURE / "EFFECTIVE_ARRAY_PATHS.json",
        "effective_gpu_arrays.npz": CAPTURE / "effective_gpu_arrays.npz",
    }
    need({k: sha(v) for k, v in paths.items()} == plan["frozen_inputs_sha256"], "Source changed")
    source = paths["gpu_coefficient_layout.py"].read_text()
    functions = re.findall(r"^def\s+(_\w+)\(self,", source, flags=re.MULTILINE)
    need(len(functions) == 17 and functions[-1] == "_GpuSynapticVisualBrain",
         "Unexpected wrapper layout")
    deep_order = list(reversed(functions))
    expected_names = [entry[1] for entry in WRITERS]
    need(all(name in deep_order for name in expected_names)
         and [name for name in deep_order if name in expected_names] == expected_names,
         "Writer order changed")
    need([entry[0] for entry in WRITERS] == plan["ordered_writers_after_base"],
         "Writer labels changed")
    mapping = json.loads(paths["EFFECTIVE_ARRAY_PATHS.json"].read_text())
    keys = {entry[2] for entry in WRITERS}
    keys.add("cuda/indptr")
    need(all(key in mapping for key in keys), "Missing selected owner rows")
    with np.load(paths["effective_gpu_arrays.npz"], allow_pickle=False) as archive:
        arrays = {key: archive[mapping[key]] for key in keys}
    ptr = arrays["cuda/indptr"]
    n = len(ptr) - 1
    need(n == plan["frozen_checks"]["neurons"], "Wrong neuron count")
    target_count = np.zeros(n, dtype=np.uint8)
    rate_count = np.zeros(n, dtype=np.uint8)
    last_target = np.full(n, -1, dtype=np.int16)
    last_rate = np.full(n, -1, dtype=np.int16)
    masks = []
    writers = []
    for index, (label, function, key, target_op, rate_op) in enumerate(WRITERS):
        rows = arrays[key]
        need(rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer)
             and np.all((rows >= 0) & (rows < n)), "Invalid owner rows: " + key)
        mask = np.zeros(n, dtype=np.bool_)
        mask[rows] = True
        masks.append(mask)
        if target_op is not None:
            target_count[mask] += 1
            last_target[mask] = index
        if rate_op is not None:
            rate_count[mask] += 1
            last_rate[mask] = index
        writers.append({"label": label, "wrapper": function, "row_key": key,
                        "potential_rows": int(mask.sum()), "target_operation": target_op,
                        "rate_operation": rate_op})
    intersections = []
    for i, first in enumerate(masks):
        for j in range(i + 1, len(masks)):
            overlap = int(np.count_nonzero(first & masks[j]))
            if overlap:
                intersections.append({"earlier": WRITERS[i][0], "later": WRITERS[j][0],
                                      "rows": overlap})
    pair_hist = {}
    for t, r in zip(last_target, last_rate):
        if t != -1 or r != -1:
            tl = "base" if t == -1 else WRITERS[int(t)][0]
            rl = "base" if r == -1 else WRITERS[int(r)][0]
            pair_hist[(tl, rl)] = pair_hist.get((tl, rl), 0) + 1
    result = {
        "schema": "owner_write_precedence_screen_result_v2",
        "status": "COMPLETE_CONSERVATIVE_STATIC",
        "plan_sha256": sha(plan_path),
        "original_plan_sha256": sha(HERE / "PRECEDENCE_PLAN_07.json"),
        "script_sha256": sha(Path(__file__)),
        "frozen_inputs_sha256": plan["frozen_inputs_sha256"],
        "neurons": n,
        "owner_writers": writers,
        "potential_target_override_union": int(np.count_nonzero(target_count)),
        "potential_rate_override_union": int(np.count_nonzero(rate_count)),
        "rows_with_multiple_potential_target_writers": int(np.count_nonzero(target_count >= 2)),
        "rows_with_multiple_potential_rate_writers": int(np.count_nonzero(rate_count >= 2)),
        "rows_with_different_last_target_and_rate_touch": int(np.count_nonzero(
            (last_target != last_rate) & ((last_target != -1) | (last_rate != -1)))),
        "intersections": sorted(intersections, key=lambda row: -row["rows"]),
        "last_touch_pair_histogram": [
            {"target": key[0], "rate": key[1], "rows": value}
            for key, value in sorted(pair_hist.items(), key=lambda item: -item[1])
        ],
        "interpretation_limit": "Potential row touches from source order, regardless of conditional enablement. Add and transform operations retain earlier values; last touch is not sole causal owner. Base target/rate and source substitutions still need live candidate verification.",
        "stage_admission": False,
        "wall_s": time.monotonic() - start,
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    budget = repair["budget"]
    need(result["wall_s"] <= budget["wall_seconds_max"]
         and result["maxrss_kib"] <= budget["ram_gib_max"] * 1024**2,
         "Budget exceeded")
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    need(len(encoded.encode()) <= budget["output_bytes_max"], "Output budget exceeded")
    out.mkdir(parents=True, exist_ok=False)
    (out / "RESULT.json").write_text(encoded)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("30-second CPU budget")))
    signal.alarm(30)
    result = run(args.out)
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ("intersections", "last_touch_pair_histogram", "owner_writers")},
                     ensure_ascii=False, allow_nan=False), flush=True)
