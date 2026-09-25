"""Recompute the full-coefficient release-cache screen from archived arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def run(result_dir, plan_path, out):
    report = json.loads((result_dir / "QUANTIZED_FULL_RESULT.json").read_text())
    plan = json.loads(plan_path.read_text())
    path = result_dir / "FULL_COEFFICIENT_ARRAYS.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    need(digest == report["full_coefficient_arrays_sha256"], "Archive SHA")
    with np.load(path, allow_pickle=False) as z:
        a = {name: z[name] for name in z.files}
    n = 359373
    for name in ("consumed_z", "consumed_target", "consumed_rate",
                 "baseline_z", "baseline_target", "baseline_rate",
                 "candidate_z", "candidate_target", "candidate_rate"):
        need(a[name].shape == (60, n) and a[name].dtype == np.float64
             and np.isfinite(a[name]).all(), "Array layout: " + name)
    need(a["release_cache_delta"].shape == (60, 166700), "Cache layout")
    need(a["outdegree"].shape == (166700,) and int(a["outdegree"].sum()) == 25582938, "Degree")
    event = a["event_rows"]
    need(event.ndim == 1 and len(event) == 4062 and np.all((event >= 0) & (event < 166700)), "Event rows")
    need(np.array_equal(a["consumed_z"], a["baseline_z"])
         and np.array_equal(a["consumed_target"], a["baseline_target"])
         and np.array_equal(a["consumed_rate"], a["baseline_rate"]), "Parent oracle differs")
    need(np.count_nonzero(a["release_cache_delta"][:, event]) == 0, "Event sources altered")
    ts = int(plan["transmission_start"])
    need(ts == 177758, "Transmission layout")
    need(np.array_equal(a["candidate_target"][:, ts:ts + 166700],
                        a["consumed_target"][:, ts:ts + 166700])
         and np.array_equal(a["candidate_rate"][:, ts:ts + 166700],
                            a["consumed_rate"][:, ts:ts + 166700]),
         "Source dynamics were altered")
    mask = np.ones(166700, dtype=bool)
    mask[event] = False
    cache = None
    per_query = report["per_query"]
    need(len(per_query) == 60, "Result query count")
    maximum_proxy = 0.
    max_target = max_rate = 0.
    updated = []
    threshold = float(plan["absolute_release_threshold"])
    rtol, atol = float(plan["rtol"]), float(plan["atol"])
    for j in range(60):
        z = a["consumed_z"][j]
        s = z[ts:ts + 166700]
        if cache is None:
            cache = s.copy()
            active = np.zeros(166700, dtype=bool)
        else:
            active = mask & (np.abs(s - cache) > threshold)
            cache[active] = s[active]
            cache[event] = s[event]
        need(np.array_equal(cache - s, a["release_cache_delta"][j]), "Cached release")
        edges = int(a["outdegree"][active].sum(dtype=np.int64))
        row = per_query[j]
        need(row["query"] == j and row["updated_sources"] == int(np.count_nonzero(active))
             and row["updated_edges"] == edges and int(a["updated_edges"][j]) == edges,
             "Update accounting")
        expected_z = z.copy()
        expected_z[ts:ts + 166700] = cache
        need(np.array_equal(expected_z, a["candidate_z"][j]), "Quantized projection")
        target = a["consumed_target"][j]
        rate = a["consumed_rate"][j]
        target_q = a["candidate_target"][j]
        rate_q = a["candidate_rate"][j]
        h = float(a["trial_step_s"][j])
        f_base = rate * (target - z)
        f_q = rate_q * (target_q - z)
        denom = 3. * (atol + rtol * np.maximum(np.abs(z), np.abs(z + h * f_base)))
        need(np.all(denom > 0.) and np.isfinite(denom).all(), "Norm scale")
        proxy = h * np.abs(f_q - f_base) / denom
        observed = float(np.max(proxy))
        need(abs(observed - row["local_rhs_proxy_max"]) <= plan["verification_max_abs"],
             "Proxy differs")
        td = float(np.max(np.abs(target_q - target)))
        rd = float(np.max(np.abs(rate_q - rate)))
        need(abs(td - row["target_max_abs"]) <= plan["verification_max_abs"]
             and abs(rd - row["rate_max_abs"]) <= plan["verification_max_abs"], "Coefficient extrema")
        maximum_proxy = max(maximum_proxy, observed)
        max_target = max(max_target, td)
        max_rate = max(max_rate, rd)
        if j: updated.append(edges)
    need(abs(maximum_proxy - report["maximum_local_rhs_proxy"]) <= plan["verification_max_abs"], "Global proxy")
    need(max_target == report["maximum_target_abs"] and max_rate == report["maximum_rate_abs"], "Global extrema")
    need(float(np.median(updated)) == report["median_updated_edges"], "Median edge count")
    need(report["status"] == "COMPLETE_NECESSARY_SCREEN_ONLY" and report["runner_exit_code"] == 0
         and report["budget_ok"] and report["baseline_all_queries_exact"], "Runner result")
    result = {"status": "PASS_FULL_ARRAY_RECOMPUTE_NECESSARY_ONLY",
              "archive_sha256": digest, "queries": 60,
              "maximum_local_rhs_proxy": maximum_proxy,
              "median_updated_edges": float(np.median(updated)),
              "maximum_target_abs": max_target, "maximum_rate_abs": max_rate,
              "scope": "All consumed coefficient queries in one real 125-us block. Does not certify integrated state error or runtime speed."}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.result, args.plan, args.out), indent=2))
