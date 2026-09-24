"""One bounded real-graph CUDA experiment; Python owns loading and timing only."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import cupy as cp
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FLYWIRE = Path("/home/daroch/AXIOMA_FLYWIRE/matrix")
CONTRACT = HERE / "CONTRACT.json"
KERNEL = HERE / "batched_csr.cu"
N_EXPECTED = 166700
E_EXPECTED = 25582938


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_array(value):
    return digest_bytes(np.ascontiguousarray(value).view(np.uint8))


def launch_scalar(kernel, common, n, rhs_count):
    for index in range(rhs_count):
        kernel(((n * 32 + 255) // 256,), (256,), common + (np.int32(index),))


def launch_batched(kernel, common, n):
    kernel(((n * 32 + 255) // 256,), (256,), common)


def measure_arm(launch):
    start, finish = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    launch()
    finish.record()
    finish.synchronize()
    return float(cp.cuda.get_elapsed_time(start, finish))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    outdir = args.out.resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    result = {
        "schema": "native_unified_batched_csr_primitive_result_v1",
        "status": "STARTED",
        "organism_loads": 0,
        "simulated_ms": 0,
        "kernel_only": True,
        "scientific_scope": "Eight candidate RHS on real effective CSR; no recurrent, organism or stage speed claim",
    }
    organism = None
    try:
        plan = json.loads(CONTRACT.read_text(encoding="utf-8"))
        if plan["schema"] != "native_unified_batched_csr_primitive_v1":
            raise ValueError("Contract schema mismatch")
        if plan["input"]["rhs_count"] != 8 or plan["budget"]["timed_repetitions_per_arm"] != 12:
            raise ValueError("Unexpected frozen counts")
        source_hashes = {
            "contract_sha256": digest_bytes(CONTRACT.read_bytes()),
            "kernel_sha256": digest_bytes(KERNEL.read_bytes()),
            "harness_sha256": digest_bytes(Path(__file__).read_bytes()),
        }
        sys.path[:0] = [
            str(FLYWIRE / "work/motor14_20260922"),
            str(FLYWIRE / "work/motor13_20260922"),
            str(ROOT / "motor_nuevo/pipeline_review_20260922"),
        ]
        from motor_runtime import load

        organism, _, _, _, _, _ = load(outdir / "preparation_inputs")
        result["organism_loads"] = 1
        hybrid = organism.core.hybrid
        csr = hybrid.brain.W
        n = int(csr.shape[0])
        weight = np.ascontiguousarray(hybrid.weights64, dtype=np.float64)
        column = np.ascontiguousarray(csr.indices, dtype=np.int32)
        rowptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
        if n != N_EXPECTED or int(csr.nnz) != E_EXPECTED:
            raise ValueError("Real graph dimensions mismatch")
        if weight.shape != (E_EXPECTED,) or column.shape != (E_EXPECTED,) or rowptr.shape != (n + 1,):
            raise ValueError("CSR buffer shapes mismatch")
        source_hashes.update({
            "weights_sha256": digest_array(weight),
            "indices_sha256": digest_array(column),
            "indptr_i64_sha256": digest_array(rowptr),
        })
        result["source_hashes"] = source_hashes
        if source_hashes["weights_sha256"] != plan["input"]["weights_sha256"]:
            raise ValueError("Frozen effective weights SHA256 mismatch")
        if source_hashes["indices_sha256"] != plan["input"]["indices_sha256"]:
            raise ValueError("Frozen CSR indices SHA256 mismatch")
        if rowptr[0] != 0 or rowptr[-1] != E_EXPECTED or np.any(rowptr[1:] < rowptr[:-1]):
            raise ValueError("Invalid CSR row pointer")
        if np.any(column < 0) or np.any(column >= n):
            raise ValueError("Invalid CSR column index")
        if not np.isfinite(weight).all():
            raise ValueError("Nonfinite effective weights")
        degrees = np.diff(rowptr)
        result["degree_stats"] = {
            "min": int(degrees.min()),
            "p50": float(np.percentile(degrees, 50)),
            "p90": float(np.percentile(degrees, 90)),
            "p99": float(np.percentile(degrees, 99)),
            "p99_9": float(np.percentile(degrees, 99.9)),
            "max": int(degrees.max()),
        }
        base = np.asarray(hybrid.state[hybrid.transmission_start:hybrid.transmission_start + n],
                          dtype=np.float64)
        if base.shape != (n,) or not np.isfinite(base).all():
            raise ValueError("Nonfinite or wrong-size real-state source")
        rng = np.random.default_rng(20260924)
        rhs = np.empty((8, n), dtype=np.float64)
        rhs[0] = base
        for index in range(1, 8):
            rhs[index] = base + rng.standard_normal(n) * (index * 1e-4)
        if not np.isfinite(rhs).all():
            raise ValueError("Nonfinite candidate RHS")
        source_hashes["state_rhs0_sha256"] = digest_array(base)
        source_hashes["eight_rhs_sha256"] = digest_array(rhs)
        result["rhs_generation"] = "RHS0 real transmission state; RHS1..7 fixed seed 20260924 normal perturbations, amplitude index*1e-4; illustrative only"
        result["neuron_count"] = n
        result["edge_count"] = E_EXPECTED
        result["rhs_count"] = 8

        module = cp.RawModule(code=KERNEL.read_text(encoding="utf-8"),
                              options=("--std=c++14", "--fmad=false", "--prec-div=true", "--prec-sqrt=true"),
                              name_expressions=("csr_scalar_8x_reference", "csr_batched_8"))
        scalar = module.get_function("csr_scalar_8x_reference")
        batched = module.get_function("csr_batched_8")
        gpu_rowptr = cp.asarray(rowptr)
        gpu_column = cp.asarray(column)
        gpu_weight = cp.asarray(weight)
        gpu_rhs = cp.asarray(rhs)
        out_scalar = cp.empty((8, n), dtype=cp.float64)
        out_batched = cp.empty((8, n), dtype=cp.float64)
        scalar_args = (np.int32(n), gpu_rowptr, gpu_column, gpu_weight, gpu_rhs, out_scalar)
        batched_args = (np.int32(n), gpu_rowptr, gpu_column, gpu_weight, gpu_rhs, out_batched)
        scalar_run = lambda: launch_scalar(scalar, scalar_args, n, 8)
        batched_run = lambda: launch_batched(batched, batched_args, n)
        for _ in range(plan["budget"]["warmups_per_arm"]):
            scalar_run()
            batched_run()
        cp.cuda.get_current_stream().synchronize()
        reference = cp.asnumpy(out_scalar)
        candidate = cp.asnumpy(out_batched)
        equal_bits = np.equal(reference.view(np.uint64), candidate.view(np.uint64))
        result["bitwise_equal_per_rhs"] = [bool(np.all(equal_bits[k])) for k in range(8)]
        result["different_values_per_rhs"] = [int(np.count_nonzero(~equal_bits[k])) for k in range(8)]
        result["nonfinite_reference"] = int(np.count_nonzero(~np.isfinite(reference)))
        result["nonfinite_batched"] = int(np.count_nonzero(~np.isfinite(candidate)))
        result["reference_output_sha256"] = digest_array(reference)
        result["batched_output_sha256"] = digest_array(candidate)
        if (not all(result["bitwise_equal_per_rhs"]) or result["nonfinite_reference"]
                or result["nonfinite_batched"]):
            result["status"] = "NUMERIC_GATE_FAILED"
            raise ValueError("Bitwise or finite-output gate failed; timing suppressed")
        if time.perf_counter() - began >= plan["budget"]["wall_seconds_max"]:
            result["status"] = "BUDGET_EXHAUSTED"
            raise TimeoutError("Wall budget exhausted before timing")
        samples = {"eight_scalar_ms": [], "one_batched_ms": []}
        for repetition in range(plan["budget"]["timed_repetitions_per_arm"]):
            order = (("eight_scalar_ms", scalar_run), ("one_batched_ms", batched_run))
            if repetition & 1:
                order = order[::-1]
            for name, launch in order:
                samples[name].append(measure_arm(launch))
        result["kernel_ms_samples"] = samples
        result["kernel_ms_median"] = {name: statistics.median(values) for name, values in samples.items()}
        result["speedup_scalar8_over_batched"] = (
            result["kernel_ms_median"]["eight_scalar_ms"] /
            result["kernel_ms_median"]["one_batched_ms"]
        )
        result["status"] = ("PRIMITIVE_PASSED" if result["speedup_scalar8_over_batched"] >=
                            plan["gates"]["minimum_kernel_only_speedup_to_continue_waveform_research"]
                            else "PRIMITIVE_SPEED_GATE_FAILED")
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:350]
        if result["status"] == "STARTED":
            result["status"] = "FAILED_RETAINED"
    finally:
        if organism is not None:
            try:
                organism.close()
            except BaseException as exc:
                result["cleanup_error_type"] = type(exc).__name__
                result["status"] = "FAILED_CLEANUP"
        result["wall_seconds"] = time.perf_counter() - began
        (outdir / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "kernel_ms_samples"}, indent=2))
    if result["status"] != "PRIMITIVE_PASSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
