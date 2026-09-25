"""One real 1-ms organism replay; transient CUDA edge lesion only after it commits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EPOCH = ROOT / "motor_nuevo/epoch_cost_20260923"
CAPTURE = ROOT / "motor_nuevo/multirate_real_20260924_01/capture_01"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_positions(ptr, rows):
    return np.concatenate([np.arange(ptr[r], ptr[r + 1], dtype=np.int64)
                           for r in rows])


def run(out):
    started = time.monotonic()
    out = out.resolve()
    need(__debug__, "Historical loader requires normal Python")
    need(not out.exists(), "Output must be unique")
    plan_path = HERE / "KC_APL_LESION_PLAN_15.json"
    plan = json.loads(plan_path.read_text())
    need(plan["schema"] == "kc_apl_base_edge_lesion_plan_v1", "Wrong plan")
    actual = {name: sha(ROOT / name) for name in plan["inputs_sha256"]}
    need(actual == plan["inputs_sha256"], "Frozen source/capture changed")
    mapping = json.loads((CAPTURE / "EFFECTIVE_ARRAY_PATHS.json").read_text())
    with np.load(CAPTURE / "effective_gpu_arrays.npz", allow_pickle=False) as z:
        ptr = z[mapping["cuda/indptr"]].copy()
        weights_capture = z[mapping["cuda/weights"]].copy()
        dynamic = z[mapping["_dynamic_gpu_rows"]].copy()
    need(len(dynamic) == 4064 and len(np.unique(dynamic)) == len(dynamic)
         and len(ptr) == 166701 and ptr[-1] == len(weights_capture),
         "Unexpected captured base layout")
    positions = selected_positions(ptr, np.sort(dynamic))
    need(len(positions) == 836605, "Expected dynamic incoming edge count changed")
    with np.load(CAPTURE / "trace_state.npz", allow_pickle=False) as z:
        states = z["values"][plan["states"]].copy()
    with np.load(CAPTURE / "trace_target.npz", allow_pickle=False) as z:
        target_ref = z["values"][plan["states"]].copy()
    with np.load(CAPTURE / "trace_rate.npz", allow_pickle=False) as z:
        rate_ref = z["values"][plan["states"]].copy()
    with np.load(CAPTURE / "trace_clock.npz", allow_pickle=False) as z:
        times = z["query_s"][plan["states"]].copy()
    need(states.shape == target_ref.shape == rate_ref.shape == (2, 359373)
         and np.isfinite(states).all() and np.isfinite(times).all(), "Captured query layout")
    import cupy as cp
    sys.path[:0] = [str(EPOCH), str(HERE), str(ROOT / "motor_nuevo/multirate_real_20260924_01")]
    import run_set
    from runtime_session import RuntimeSession

    old_load, old_init = run_set.load, RuntimeSession.__init__
    context = {"installed": 0, "tested": 0, "epochs_seen": 0}
    free_initial, total_vram = cp.cuda.runtime.memGetInfo()
    report = {
        "schema": "kc_apl_base_edge_lesion_result_v1",
        "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
        "inputs_sha256": actual, "status": "STARTED",
        "organism_ms_requested": 1, "states": plan["states"],
        "dynamic_rows": len(dynamic), "selected_base_edges": len(positions),
        "gpu_total_bytes": int(total_vram),
    }

    def loaded(*args, **kwargs):
        value = old_load(*args, **kwargs)
        context["organism"] = value[0]
        return value

    def installed(session, *args, **kwargs):
        old_init(session, *args, **kwargs)
        context["installed"] += 1
        adapter = session.adapter
        b = adapter.brain
        original_build = adapter.build

        def build(drive, light):
            original_build(drive, light)
            from effective_oracle import EffectiveOracle
            context["oracle"] = EffectiveOracle(adapter)

        adapter.build = build
        physical_step = session.events.step

        def step(brain, ns, drive, light):
            epoch = context["epochs_seen"]
            context["epochs_seen"] += 1
            result = physical_step(brain, ns, drive, light)
            if epoch != 1:
                return result
            need(ns == 125000 and context["tested"] == 0, "Wrong accepted block")
            context["tested"] += 1
            oracle = context["oracle"]
            weights = b.cuda["weights"]
            need(bool(cp.array_equal(b.cuda["indptr"], cp.asarray(ptr))),
                 "Prepared CSR row pointers differ")
            device_positions = cp.asarray(positions)
            original = weights[device_positions].copy()
            need(bool(cp.array_equal(original, cp.asarray(weights_capture[positions]))),
                 "Prepared selected weights differ")
            nonzero = int(cp.count_nonzero(original).get())
            need(nonzero > 0, "Trivial all-zero lesion")
            report["selected_nonzero_weights"] = nonzero
            baselines = []
            baseline_checks = []
            for i, t in enumerate(times):
                x = cp.asarray(states[i], dtype=cp.float64)
                z, a, r = oracle.query(x, float(t))
                oracle.stream.synchronize()
                expected_a, expected_r = cp.asarray(target_ref[i]), cp.asarray(rate_ref[i])
                exact = (bool(cp.array_equal(z, x)), bool(cp.array_equal(a, expected_a)),
                         bool(cp.array_equal(r, expected_r)))
                baseline_checks.append({
                    "state": int(plan["states"][i]), "time_s": float(t),
                    "projected_exact": exact[0], "target_exact": exact[1],
                    "rate_exact": exact[2],
                })
                baselines.append((a.copy(), r.copy()))
            report["baseline_checks"] = baseline_checks
            need(all(all((row["projected_exact"], row["target_exact"], row["rate_exact"]))
                     for row in baseline_checks), "Baseline differs from frozen consumed trace")
            lesion_checks = []
            restoration_checks = []
            try:
                with oracle.stream:
                    weights[device_positions] = 0.
                oracle.stream.synchronize()
                need(int(cp.count_nonzero(weights[device_positions]).get()) == 0,
                     "Lesion did not zero selected weights")
                for i, t in enumerate(times):
                    x = cp.asarray(states[i], dtype=cp.float64)
                    _, a, r = oracle.query(x, float(t))
                    oracle.stream.synchronize()
                    aa, rr = baselines[i]
                    changed_a = cp.flatnonzero(a != aa)
                    changed_r = cp.flatnonzero(r != rr)
                    lesion_checks.append({
                        "state": int(plan["states"][i]),
                        "target_exact": bool(cp.array_equal(a, aa)),
                        "rate_exact": bool(cp.array_equal(r, rr)),
                        "target_changed_coordinates": int(len(changed_a)),
                        "rate_changed_coordinates": int(len(changed_r)),
                        "target_first_changed": cp.asnumpy(changed_a[:10]).tolist(),
                        "rate_first_changed": cp.asnumpy(changed_r[:10]).tolist(),
                        "target_max_abs": float(cp.max(cp.abs(a - aa))),
                        "rate_max_abs": float(cp.max(cp.abs(r - rr))),
                    })
            finally:
                with oracle.stream:
                    weights[device_positions] = original
                oracle.stream.synchronize()
                report["weights_restored_exact"] = bool(
                    cp.array_equal(weights[device_positions], original))
                for i, t in enumerate(times):
                    x = cp.asarray(states[i], dtype=cp.float64)
                    _, a, r = oracle.query(x, float(t))
                    oracle.stream.synchronize()
                    aa, rr = baselines[i]
                    restoration_checks.append({
                        "state": int(plan["states"][i]),
                        "target_exact": bool(cp.array_equal(a, aa)),
                        "rate_exact": bool(cp.array_equal(r, rr)),
                    })
            report["lesion_checks"] = lesion_checks
            report["restoration_checks"] = restoration_checks
            report["baseline_exact"] = True
            report["restoration_exact"] = bool(
                report["weights_restored_exact"]
                and all(row["target_exact"] and row["rate_exact"]
                        for row in restoration_checks))
            report["lesion_full_target_rate_exact"] = bool(
                all(row["target_exact"] and row["rate_exact"] for row in lesion_checks))
            report["max_abs_tolerance"] = 0
            report["nonfinite_zero"] = bool(
                all(np.isfinite(row["target_max_abs"])
                    and np.isfinite(row["rate_max_abs"]) for row in lesion_checks))
            report["oracle_queries"] = oracle.calls
            return result

        session.events.step = step

    run_set.load = loaded
    RuntimeSession.__init__ = installed
    old_argv = sys.argv
    sys.argv = ["run_set.py", "--out", str(out), "--odor", "sham",
                "--engine", "causal_cuda", "--ms", "1", "--observe", "off",
                "--cuda-profile", "off", "--profile", "off", "--kc-capture", "off"]
    code = None
    error = None
    try:
        code = run_set.main()
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc),
                 "traceback": traceback.format_exc()}
    finally:
        sys.argv = old_argv
        run_set.load = old_load
        RuntimeSession.__init__ = old_init
    report["runner_exit_code"] = code
    report["error"] = error
    report["runtime_installed_count"] = context["installed"]
    report["epoch_steps_seen"] = context["epochs_seen"]
    report["test_executions"] = context["tested"]
    report["wall_s"] = time.monotonic() - started
    report["maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    free_after, _ = cp.cuda.runtime.memGetInfo()
    report["gpu_free_after_bytes"] = int(free_after)
    report["gpu_allocation_delta_upper_bytes"] = int(max(0, free_initial - free_after))
    report["disk_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) if out.exists() else 0
    budget = plan["budget"]
    budget_ok = (
        report["wall_s"] <= budget["wall_seconds_max"]
        and report["maxrss_kib"] <= budget["ram_gib_max"] * 1024**2
        and report["disk_bytes"] <= budget["disk_bytes_max"]
        and report["gpu_allocation_delta_upper_bytes"] <= budget["gpu_vram_gib_max"] * 1024**3)
    gates = plan["gates_frozen"]
    gate_ok = all(report.get(k) == v for k, v in gates.items())
    report["budget_ok"] = bool(budget_ok)
    report["frozen_gates_ok"] = bool(gate_ok)
    report["status"] = ("PASS_LIMITED" if code == 0 and error is None
                        and context["tested"] == 1 and budget_ok and gate_ok
                        else "INCOMPLETE_OR_FALSIFIED")
    report["scope"] = "Two saved states, full target/rate, same prepared oracle; no omitted-edge kernel timing or long organism behavior."
    need(out.exists(), "Runner did not create output")
    (out / "LESION_RESULT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(TimeoutError("Prospective 120s wall limit")))
    signal.alarm(120)
    value = run(args.out)
    print(json.dumps({k: value[k] for k in ("status", "runner_exit_code", "wall_s",
                                            "budget_ok", "frozen_gates_ok",
                                            "lesion_checks") if k in value},
                     ensure_ascii=False, allow_nan=False))
