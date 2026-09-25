"""Stream-ordered real-organism A/B/A lesion with archived full arrays."""
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
MULTIRATE = ROOT / "motor_nuevo/multirate_real_20260924_01"
SLOTS = 128


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def run(out):
    start = time.monotonic()
    out = out.resolve()
    need(__debug__ and not out.exists(), "Normal Python and unique output required")
    plan_path = HERE / "KC_APL_STREAM_REPAIR_PLAN_24.json"
    plan = json.loads(plan_path.read_text())
    need(plan["schema"] == "kc_apl_stream_repair_plan_v1", "Wrong plan")
    need(sha(Path(__file__)) == plan["script_sha256"], "Script changed after freeze")
    actual = {name: sha(ROOT / name) for name in plan["frozen_inputs_sha256"]}
    need(actual == plan["frozen_inputs_sha256"], "Frozen source changed")
    mapping = json.loads((CAPTURE / "EFFECTIVE_ARRAY_PATHS.json").read_text())
    with np.load(CAPTURE / "effective_gpu_arrays.npz", allow_pickle=False) as z:
        ptr = z[mapping["cuda/indptr"]].copy()
        weights_capture = z[mapping["cuda/weights"]].copy()
        dynamic = np.sort(z[mapping["_dynamic_gpu_rows"]].copy())
    need(len(dynamic) == 4064 and len(np.unique(dynamic)) == 4064
         and len(ptr) == 166701 and ptr[-1] == len(weights_capture),
         "Captured base layout")
    positions = np.concatenate([
        np.arange(ptr[row], ptr[row + 1], dtype=np.int64) for row in dynamic])
    need(len(positions) == 836605, "Expected incoming edges changed")

    import cupy as cp
    sys.path[:0] = [str(EPOCH), str(HERE), str(MULTIRATE)]
    import run_set
    from runtime_session import RuntimeSession
    from capture_effective_block import TRACE

    report = {
        "schema": "kc_apl_stream_repair_result_v1",
        "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
        "inputs_sha256": actual, "status": "STARTED",
        "simulation_ms_requested": 1, "dynamic_rows": len(dynamic),
        "incoming_base_edges": len(positions), "prior_capture_queries": 60,
        "prior_capture_events": 7,
    }
    context = {"installed": 0, "steps": 0, "tested": 0}
    free_before, total_vram = cp.cuda.runtime.memGetInfo()
    old_load, old_init = run_set.load, RuntimeSession.__init__

    def loaded(*args, **kwargs):
        value = old_load(*args, **kwargs)
        return value

    def installed(session, *args, **kwargs):
        old_init(session, *args, **kwargs)
        context["installed"] += 1
        adapter = session.adapter
        b = adapter.brain
        n_state = len(b.state)
        need(n_state == 359373, "Full state layout")
        import graph_core
        graph_class = graph_core.NativeGraph
        old_coeff = graph_class.coeff
        context["graph_class"] = graph_class
        context["old_coeff"] = old_coeff
        module = cp.RawModule(code=TRACE, options=("--std=c++11", "--fmad=false"))
        marker = module.get_function("mark")
        capture = module.get_function("capture")
        tag = cp.asarray([-1], dtype=cp.int64)
        count = cp.zeros(1, dtype=cp.uint64)
        ts, starts, steps, fractions = [cp.full(SLOTS, np.nan, dtype=cp.float64)
                                        for _ in range(4)]
        xs, aa, rr = [cp.empty((SLOTS, n_state), dtype=cp.float64)
                      for _ in range(3)]
        context["buffers"] = (tag, count, ts, starts, steps, fractions, xs, aa, rr)

        def traced(self, y, frac):
            z = self.project(y, self.clock, frac) if self.project else y
            target, rate = self.coefficient(z)
            marker((1,), (1,), (tag, count, self.clock, np.float64(frac),
                                  ts, starts, steps, fractions, np.int32(SLOTS)))
            capture(((n_state + 255) // 256,), (256,),
                    (tag, count, z, target, rate, xs, aa, rr,
                     np.int32(n_state), np.int32(SLOTS)))
            self.check(self.grid, (256,),
                       (z, target, rate, np.int32(self.n), self.flag))
            return z, target, rate

        graph_class.coeff = traced
        old_build = adapter.build

        def build(drive, light):
            old_build(drive, light)
            from effective_oracle import EffectiveOracle
            context["oracle"] = EffectiveOracle(adapter)

        adapter.build = build
        physical_step = session.events.step

        def step(brain, ns, drive, light):
            epoch = context["steps"]
            context["steps"] += 1
            tag.set(np.asarray([epoch], dtype=np.int64))
            cp.cuda.get_current_stream().synchronize()
            result = physical_step(brain, ns, drive, light)
            if epoch != 1:
                return result
            need(ns == 125000 and context["tested"] == 0, "Wrong accepted block")
            context["tested"] = 1
            oracle = context["oracle"]
            oracle.stream.synchronize()
            m = int(count.get()[0])
            need(3 <= m <= SLOTS, "Query count outside frozen capacity")
            chosen = (2, m - 1)
            report["current_capture_queries"] = m
            report["chosen_query_indices"] = chosen
            report["current_block_events"] = len(session.events.active.times)
            report["current_block_event_version_equal_to_prior"] = "NOT_CHECKED"
            # Event equality is descriptive, not a gate: the earlier cross-run
            # comparison failed, which is why this probe uses in-run buffers.
            weights = b.cuda["weights"]
            need(bool(cp.array_equal(b.cuda["indptr"], cp.asarray(ptr))),
                 "Current CSR row layout changed")
            with oracle.stream:
                pos = cp.asarray(positions)
                old = weights[pos].copy()
            oracle.stream.synchronize()
            need(bool(cp.array_equal(old, cp.asarray(weights_capture[positions]))),
                 "Current selected base weights changed")
            nonzero = int(cp.count_nonzero(old).get())
            need(nonzero > 0, "Trivial lesion")
            report["selected_nonzero_weights"] = nonzero
            baseline = []
            baseline_rows = []
            archive = {
                "query_indices": np.asarray(chosen, dtype=np.int64),
                "consumed_state": np.stack([cp.asnumpy(xs[j], stream=oracle.stream) for j in chosen]),
                "consumed_target": np.stack([cp.asnumpy(aa[j], stream=oracle.stream) for j in chosen]),
                "consumed_rate": np.stack([cp.asnumpy(rr[j], stream=oracle.stream) for j in chosen]),
                "weights_before": cp.asnumpy(old, stream=oracle.stream),
                "selected_positions": positions,
            }
            baseline_z, baseline_target, baseline_rate = [], [], []
            for j in chosen:
                with oracle.stream:
                    x = xs[j].copy()
                t = float(ts[j].get(stream=oracle.stream))
                z, target, rate = oracle.query(x, t)
                with oracle.stream:
                    z_saved, target_saved, rate_saved = z.copy(), target.copy(), rate.copy()
                oracle.stream.synchronize()
                row = {"query": j, "time_s": t,
                       "projected_state_exact": bool(cp.array_equal(z_saved, xs[j])),
                       "target_exact": bool(cp.array_equal(target_saved, aa[j])),
                       "rate_exact": bool(cp.array_equal(rate_saved, rr[j]))}
                baseline_rows.append(row)
                baseline.append((target_saved, rate_saved))
                baseline_z.append(cp.asnumpy(z_saved, stream=oracle.stream))
                baseline_target.append(cp.asnumpy(target_saved, stream=oracle.stream))
                baseline_rate.append(cp.asnumpy(rate_saved, stream=oracle.stream))
            archive["query_time_s"] = np.asarray([x["time_s"] for x in baseline_rows], dtype=np.float64)
            archive["baseline_z"] = np.stack(baseline_z)
            archive["baseline_target"] = np.stack(baseline_target)
            archive["baseline_rate"] = np.stack(baseline_rate)
            report["baseline_rows"] = baseline_rows
            report["baseline_projected_state_exact"] = all(
                x["projected_state_exact"] for x in baseline_rows)
            report["baseline_target_rate_exact"] = all(
                x["target_exact"] and x["rate_exact"] for x in baseline_rows)
            need(report["baseline_projected_state_exact"]
                 and report["baseline_target_rate_exact"],
                 "Independent oracle differs from consumed in-run coefficient")
            lesion_rows = []
            restore_rows = []
            try:
                with oracle.stream:
                    weights[pos] = 0.
                    zero_weights = weights[pos].copy()
                oracle.stream.synchronize()
                archive["weights_lesion"] = cp.asnumpy(zero_weights, stream=oracle.stream)
                need(int(cp.count_nonzero(weights[pos]).get()) == 0,
                     "Selected weights not zero")
                lesion_target, lesion_rate = [], []
                for i, j in enumerate(chosen):
                    with oracle.stream:
                        x = xs[j].copy()
                    t = float(ts[j].get(stream=oracle.stream))
                    _, target, rate = oracle.query(x, t)
                    with oracle.stream:
                        target_saved, rate_saved = target.copy(), rate.copy()
                    oracle.stream.synchronize()
                    base_target, base_rate = baseline[i]
                    lesion_target.append(cp.asnumpy(target_saved, stream=oracle.stream))
                    lesion_rate.append(cp.asnumpy(rate_saved, stream=oracle.stream))
                    changed_t = cp.flatnonzero(target_saved != base_target)
                    changed_r = cp.flatnonzero(rate_saved != base_rate)
                    lesion_rows.append({
                        "query": j,
                        "target_exact": bool(cp.array_equal(target_saved, base_target)),
                        "rate_exact": bool(cp.array_equal(rate_saved, base_rate)),
                        "target_changed_coordinates": int(len(changed_t)),
                        "rate_changed_coordinates": int(len(changed_r)),
                        "target_first_changed": cp.asnumpy(changed_t[:10]).tolist(),
                        "rate_first_changed": cp.asnumpy(changed_r[:10]).tolist(),
                        "target_max_abs": float(cp.max(cp.abs(target_saved - base_target))),
                        "rate_max_abs": float(cp.max(cp.abs(rate_saved - base_rate))),
                    })
                archive["lesion_target"] = np.stack(lesion_target)
                archive["lesion_rate"] = np.stack(lesion_rate)
            finally:
                with oracle.stream:
                    weights[pos] = old
                    restored_weights = weights[pos].copy()
                oracle.stream.synchronize()
                archive["weights_restored"] = cp.asnumpy(restored_weights, stream=oracle.stream)
                report["selected_weights_restored_exact"] = bool(
                    cp.array_equal(weights[pos], old))
                restored_target, restored_rate = [], []
                for i, j in enumerate(chosen):
                    with oracle.stream:
                        x = xs[j].copy()
                    t = float(ts[j].get(stream=oracle.stream))
                    _, target, rate = oracle.query(x, t)
                    with oracle.stream:
                        target_saved, rate_saved = target.copy(), rate.copy()
                    oracle.stream.synchronize()
                    base_target, base_rate = baseline[i]
                    restored_target.append(cp.asnumpy(target_saved, stream=oracle.stream))
                    restored_rate.append(cp.asnumpy(rate_saved, stream=oracle.stream))
                    restore_rows.append({
                        "query": j,
                        "target_exact": bool(cp.array_equal(target_saved, base_target)),
                        "rate_exact": bool(cp.array_equal(rate_saved, base_rate)),
                    })
                archive["restored_target"] = np.stack(restored_target)
                archive["restored_rate"] = np.stack(restored_rate)
            report["lesion_rows"] = lesion_rows
            report["restore_rows"] = restore_rows
            report["post_restore_target_rate_exact"] = all(
                x["target_exact"] and x["rate_exact"] for x in restore_rows)
            report["lesion_full_target_rate_exact"] = all(
                x["target_exact"] and x["rate_exact"] for x in lesion_rows)
            report["maximum_abs_error"] = max(
                max(x["target_max_abs"], x["rate_max_abs"]) for x in lesion_rows)
            report["nonfinite_count"] = sum(
                not np.isfinite(x["target_max_abs"]) or not np.isfinite(x["rate_max_abs"])
                for x in lesion_rows)
            report["oracle_queries"] = oracle.calls
            arrays_path = out / "ABA_ARRAYS.npz"
            np.savez_compressed(arrays_path, **archive)
            report["aba_arrays_sha256"] = sha(arrays_path)
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
        if "graph_class" in context:
            context["graph_class"].coeff = context["old_coeff"]
    report["runner_exit_code"] = code
    report["error"] = error
    report["runtime_installed_count"] = context["installed"]
    report["steps_seen"] = context["steps"]
    report["test_executions"] = context["tested"]
    report["wall_s"] = time.monotonic() - start
    report["maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    free_after, _ = cp.cuda.runtime.memGetInfo()
    report["gpu_allocation_delta_upper_bytes"] = int(max(0, free_before - free_after))
    report["disk_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) if out.exists() else 0
    budget = plan["budget"]
    budget_ok = (report["wall_s"] <= budget["wall_seconds_max"]
                 and report["maxrss_kib"] <= budget["ram_gib_max"] * 1024**2
                 and report["gpu_allocation_delta_upper_bytes"]
                 <= budget["incremental_vram_gib_max"] * 1024**3
                 and report["disk_bytes"] <= budget["disk_bytes_max"])
    gates_ok = all(report.get(k) == v for k, v in plan["gates_frozen"].items())
    report["budget_ok"] = bool(budget_ok)
    report["frozen_gates_ok"] = bool(gates_ok)
    report["status"] = ("PASS_LIMITED" if code == 0 and error is None
                        and context["tested"] == 1 and budget_ok and gates_ok
                        else "INCOMPLETE_OR_FALSIFIED")
    report["scope"] = "One 1-ms real organism run; stream-ordered copies and full A/B/A arrays for two coefficient queries. No pruned-kernel timing or long dynamics."
    need(out.exists(), "Runner did not create result directory")
    (out / "LESION_RESULT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(TimeoutError("120s budget")))
    signal.alarm(120)
    result = run(args.out)
    print(json.dumps({k: result[k] for k in ("status", "runner_exit_code",
                                             "current_capture_queries",
                                             "current_block_events",
                                             "wall_s", "budget_ok",
                                             "frozen_gates_ok", "lesion_rows")
                      if k in result}, ensure_ascii=False, allow_nan=False))
