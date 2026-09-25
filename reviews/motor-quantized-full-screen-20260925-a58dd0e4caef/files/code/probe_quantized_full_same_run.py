"""Screen a persistent release cache against all consumed CNS coefficients.

The cache alters oracle input only; the organism still executes the reference.
Its output never controls state, events, body or admission of any stage.
"""
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
MULTIRATE = ROOT / "motor_nuevo/multirate_real_20260924_01"
CAPTURE = MULTIRATE / "capture_01"
SLOTS = 128


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def run(out):
    started = time.monotonic()
    out = out.resolve()
    need(__debug__ and not out.exists(), "Normal Python and unique output required")
    plan_path = HERE / "QUANTIZED_FULL_PLAN_27.json"
    plan = json.loads(plan_path.read_text())
    need(plan["schema"] == "quantized_full_same_run_plan_v1", "Plan schema")
    need(sha(Path(__file__)) == plan["script_sha256"], "Script changed")
    actual = {name: sha(ROOT / name) for name in plan["frozen_inputs_sha256"]}
    need(actual == plan["frozen_inputs_sha256"], "Frozen input changed")
    mapping = json.loads((CAPTURE / "EFFECTIVE_ARRAY_PATHS.json").read_text())
    with np.load(CAPTURE / "effective_gpu_arrays.npz", allow_pickle=False) as z:
        ptr = z[mapping["cuda/indptr"]]
        indices = z[mapping["cuda/indices"]]
        photo_rows = z[mapping["cuda/pi"]]
    n = len(ptr) - 1
    need(n == 166700 and ptr[-1] == len(indices) == 25582938, "Graph layout")
    outdegree = np.bincount(indices, minlength=n).astype(np.int64)
    transmission_start = n + 2 * len(photo_rows)
    del indices, ptr
    import cupy as cp
    sys.path[:0] = [str(EPOCH), str(HERE), str(MULTIRATE)]
    import run_set
    from runtime_session import RuntimeSession
    from capture_effective_block import TRACE

    report = {"schema": "quantized_full_same_run_result_v1",
              "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
              "inputs_sha256": actual, "status": "STARTED",
              "threshold": plan["absolute_release_threshold"],
              "simulation_ms_requested": 1,
              "base_edges": int(outdegree.sum())}
    context = {"installed": 0, "steps": 0, "tested": 0, "active_epoch": -1}
    free_before, _ = cp.cuda.runtime.memGetInfo()
    old_init = RuntimeSession.__init__

    def installed(session, *args, **kwargs):
        old_init(session, *args, **kwargs)
        context["installed"] += 1
        adapter = session.adapter
        b = adapter.brain
        n_state = len(b.state)
        need(n_state == 359373 and b.transmission_start == transmission_start,
             "Full state layout")
        import graph_core
        graph_class = graph_core.NativeGraph
        old_coeff = graph_class.coeff
        context["graph_class"] = graph_class
        context["old_coeff"] = old_coeff
        module = cp.RawModule(code=TRACE, options=("--std=c++11", "--fmad=false"))
        marker, capture = module.get_function("mark"), module.get_function("capture")
        tag = cp.asarray([-1], dtype=cp.int64)
        count = cp.zeros(1, dtype=cp.uint64)
        ts, starts, steps, fractions = [cp.full(SLOTS, np.nan, dtype=cp.float64)
                                        for _ in range(4)]
        xs, aa, rr = [cp.empty((SLOTS, n_state), dtype=cp.float64) for _ in range(3)]

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
            oracle = EffectiveOracle(adapter)
            context["oracle"] = oracle
            g = adapter.core
            old_advance = g.advance

            def advance(*args, **kwargs):
                if context["active_epoch"] != 1:
                    return old_advance(*args, **kwargs)
                need(context["tested"] == 0 and args and args[0] == 125000,
                     "Wrong or repeated accepted block")
                context["tested"] = 1
                result = old_advance(*args, **kwargs)
                # We are still inside g.advance: the owner has not yet copied
                # g.x back into brain.state or advanced brain.time_ns.
                oracle.stream.synchronize()
                m = int(count.get()[0])
                need(m == plan["expected_queries"] and int(tag.get()[0]) == 1,
                     "Captured query cohort changed")
                event_rows = np.asarray(session.events.rows, dtype=np.int64)
                need(len(event_rows) == 4062 and np.all((event_rows >= 0) & (event_rows < n)),
                     "Event source cohort")
                continuous = np.ones(n, dtype=bool)
                continuous[event_rows] = False
                threshold = float(plan["absolute_release_threshold"])
                rtol = float(b.parameters["rtol"])
                atol = float(b.parameters["atol"])
                need(rtol == float(plan["rtol"]) and atol == float(plan["atol"]),
                     "Integrator error contract changed")
                cache = None
                archive = {"event_rows": event_rows, "outdegree": outdegree,
                           "threshold": np.asarray(threshold)}
                record = {name: [] for name in
                          ("query_s", "trial_step_s", "updated_edges", "updated_sources",
                           "consumed_z", "consumed_target", "consumed_rate",
                           "baseline_z", "baseline_target", "baseline_rate",
                           "candidate_z", "candidate_target", "candidate_rate",
                           "release_cache_delta")}
                per_query = []
                for j in range(m):
                    with oracle.stream:
                        input_original = xs[j].copy()
                    t = float(ts[j].get(stream=oracle.stream))
                    h = float(steps[j].get(stream=oracle.stream))
                    need(np.isfinite(t) and np.isfinite(h) and h > 0., "Query clock")
                    z_base, target_base, rate_base = oracle.query(input_original, t)
                    with oracle.stream:
                        z_base, target_base, rate_base = (
                            z_base.copy(), target_base.copy(), rate_base.copy())
                    oracle.stream.synchronize()
                    consumed = [cp.asnumpy(v[j], stream=oracle.stream)
                                for v in (xs, aa, rr)]
                    baseline = [cp.asnumpy(v, stream=oracle.stream)
                                for v in (z_base, target_base, rate_base)]
                    need(all(np.array_equal(a, b) for a, b in zip(consumed, baseline)),
                         f"Independent baseline differs at query {j}")
                    z, a, r = consumed
                    release = z[transmission_start : transmission_start + n]
                    if cache is None:
                        cache = release.copy()
                        active = np.zeros(n, dtype=bool)
                    else:
                        active = continuous & (np.abs(release - cache) > threshold)
                        cache[active] = release[active]
                        cache[event_rows] = release[event_rows]
                    patched = z.copy()
                    patched[transmission_start : transmission_start + n] = cache
                    with oracle.stream:
                        input_quantized = cp.asarray(patched)
                    zq, aq, rq = oracle.query(input_quantized, t)
                    with oracle.stream:
                        zq, aq, rq = zq.copy(), aq.copy(), rq.copy()
                    oracle.stream.synchronize()
                    qz, qa, qr = [cp.asnumpy(v, stream=oracle.stream) for v in (zq, aq, rq)]
                    # Source dynamics are not quantized by the proposed cache:
                    # only downstream reads use cached release. Restore their
                    # target/rate entries for this necessary-only screen.
                    qa[transmission_start : transmission_start + n] = a[transmission_start : transmission_start + n]
                    qr[transmission_start : transmission_start + n] = r[transmission_start : transmission_start + n]
                    delta_a, delta_r = qa - a, qr - r
                    f_base = r * (a - z)
                    f_candidate = qr * (qa - z)
                    denominator = 3. * (atol + rtol * np.maximum(np.abs(z), np.abs(z + h * f_base)))
                    need(np.isfinite(denominator).all() and np.all(denominator > 0.), "Error scale")
                    proxy = h * np.abs(f_candidate - f_base) / denominator
                    need(np.isfinite(proxy).all(), "Nonfinite coefficient effect")
                    edges = int(outdegree[active].sum(dtype=np.int64))
                    stats = {"query": j, "time_hex": t.hex(), "trial_step_s": h,
                             "updated_sources": int(np.count_nonzero(active)),
                             "updated_edges": edges,
                             "target_max_abs": float(np.max(np.abs(delta_a))),
                             "rate_max_abs": float(np.max(np.abs(delta_r))),
                             "local_rhs_proxy_max": float(np.max(proxy)),
                             "local_rhs_proxy_max_index": int(np.argmax(proxy)),
                             "raw_quantized_state_max_abs": float(np.max(np.abs(qz - z)))}
                    per_query.append(stats)
                    for name, value in (("query_s", t), ("trial_step_s", h),
                                        ("updated_edges", edges), ("updated_sources", stats["updated_sources"]),
                                        ("consumed_z", z), ("consumed_target", a),
                                        ("consumed_rate", r), ("baseline_z", baseline[0]),
                                        ("baseline_target", baseline[1]),
                                        ("baseline_rate", baseline[2]),
                                        ("candidate_z", qz), ("candidate_target", qa),
                                        ("candidate_rate", qr),
                                        ("release_cache_delta", cache - release)):
                        record[name].append(value)
                report["current_capture_queries"] = m
                report["event_sources_held_exact"] = len(event_rows)
                report["baseline_all_queries_exact"] = True
                report["per_query"] = per_query
                report["maximum_local_rhs_proxy"] = max(row["local_rhs_proxy_max"] for row in per_query)
                report["maximum_target_abs"] = max(row["target_max_abs"] for row in per_query)
                report["maximum_rate_abs"] = max(row["rate_max_abs"] for row in per_query)
                report["median_updated_edges"] = float(np.median([row["updated_edges"] for row in per_query[1:]]))
                report["necessary_proxy_screen"] = (
                    "BELOW_ONE_NOT_VALIDATED" if report["maximum_local_rhs_proxy"] <= 1.
                    else "EXCEEDS_ONE_DISCARD_THIS_THRESHOLD")
                archive.update({name: np.asarray(values) for name, values in record.items()})
                raw_path = out / "FULL_COEFFICIENT_ARRAYS.npz"
                np.savez_compressed(raw_path, **archive)
                report["full_coefficient_arrays_sha256"] = sha(raw_path)
                return result

            g.advance = advance
            context["restore_advance"] = lambda: setattr(g, "advance", old_advance)

        adapter.build = build
        physical_step = session.events.step

        def step(brain, ns, drive, light):
            epoch = context["steps"]
            context["steps"] += 1
            context["active_epoch"] = epoch
            tag.set(np.asarray([epoch], dtype=np.int64))
            cp.cuda.get_current_stream().synchronize()
            return physical_step(brain, ns, drive, light)

        session.events.step = step

    RuntimeSession.__init__ = installed
    old_argv = sys.argv
    sys.argv = ["run_set.py", "--out", str(out), "--odor", "sham", "--engine", "causal_cuda",
                "--ms", "1", "--observe", "off", "--cuda-profile", "off",
                "--profile", "off", "--kc-capture", "off"]
    code, error = None, None
    try:
        code = run_set.main()
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc),
                 "traceback": traceback.format_exc()}
    finally:
        sys.argv = old_argv
        RuntimeSession.__init__ = old_init
        if "graph_class" in context:
            context["graph_class"].coeff = context["old_coeff"]
        if "restore_advance" in context:
            context["restore_advance"]()
    report["runner_exit_code"] = code
    report["error"] = error
    report["runtime_installed_count"] = context["installed"]
    report["steps_seen"] = context["steps"]
    report["test_executions"] = context["tested"]
    report["wall_s"] = time.monotonic() - started
    report["maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    free_after, _ = cp.cuda.runtime.memGetInfo()
    report["gpu_allocation_delta_upper_bytes"] = int(max(0, free_before - free_after))
    report["disk_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) if out.exists() else 0
    budget = plan["budget"]
    report["budget_ok"] = (report["wall_s"] <= budget["wall_seconds_max"]
                           and report["maxrss_kib"] <= budget["ram_gib_max"] * 1024**2
                           and report["gpu_allocation_delta_upper_bytes"] <= budget["incremental_vram_gib_max"] * 1024**3
                           and report["disk_bytes"] <= budget["disk_bytes_max"])
    report["status"] = ("COMPLETE_NECESSARY_SCREEN_ONLY" if code == 0 and error is None
                        and context["tested"] == 1 and report["budget_ok"]
                        and report.get("baseline_all_queries_exact")
                        else "INCOMPLETE_OR_FALSIFIED")
    report["scope"] = "One 1-ms real organism run; quantized input examined only by a separate full coefficient oracle at all captured queries. Organism trajectory remains reference. No speed or state-error validation."
    need(out.exists(), "Runner did not create output")
    (out / "QUANTIZED_FULL_RESULT.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(TimeoutError("240s budget")))
    signal.alarm(240)
    result = run(args.out)
    print(json.dumps({k: result.get(k) for k in
                      ("status", "runner_exit_code", "current_capture_queries",
                       "maximum_local_rhs_proxy", "median_updated_edges",
                       "wall_s", "budget_ok", "error")}, allow_nan=False))
