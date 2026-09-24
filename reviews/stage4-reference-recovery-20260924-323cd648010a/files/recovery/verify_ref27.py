"""Composite raw verification of exposed Campaign26 native arms and new strict references.

Campaign26 remains blocked. This prospective recovery can confirm only the
mirrored-source-to-command numerical result, never Stage4 navigation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "campanas/etapa4_mirrored_source_20260924_26"


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load verifier " + str(path))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


prior = module("stage4_mirror_prior_v26", OLD / "verify_mirror.py")
current = module("stage4_mirror_recovery_v27", HERE / "verify_mirror.py")


def require(value, message: str) -> None:
    if not value:
        raise ValueError(message)


def compute() -> dict:
    plan = current.PLAN
    errors = []
    missing = []
    sources = {}
    runs = {}
    prepared = {}
    # All source/contract checks run before any scientific decision.
    try:
        sources["prior"] = prior.source_check(OLD)
        sources["recovery"] = current.source_check(HERE)
        require(plan["thresholds"] == prior.PLAN["thresholds"], "Scientific thresholds changed")
        require(plan["fixed_inputs"] == prior.PLAN["fixed_inputs"], "Fixed source inputs changed")
        require(plan["preflight"] == prior.PLAN["preflight"], "Preparation guard changed")
        require(plan["observation"] == prior.PLAN["observation"], "Consumer observation changed")
        links = plan["recovery_links"]
        for name, path in (
            ("prior_plan_sha256", OLD / "PLAN.json"),
            ("prior_close_sha256", OLD / "CLOSE_01.json"),
            ("prior_source_lock_sha256", OLD / "SOURCE_LOCK.json"),
            ("old_native_plus_trace_sha256", OLD / "native_plus_01/traces.npz"),
            ("old_native_minus_trace_sha256", OLD / "native_minus_01/traces.npz"),
            ("old_incomplete_reference_result_sha256", OLD / "reference_plus_01/RESULT.json"),
            ("timing_receipt_sha256", ROOT / "investigacion/roadmap_causal_20260924_01/TIMING_RECEIPT_01.json"),
        ):
            require(links[name] == current.sha(path), "Old evidence changed: " + name)
        close = prior.read_json(OLD / "CLOSE_01.json")
        require(close["classification"] == "BLOQUEADO" and close["stage4_admission"] is False,
                "Prior campaign was reclassified")
        require(close["reference_plus"]["completed_trial_ms"] == 391, "Prior prefix changed")
        require(plan["budget"]["native_runs_max"] == 0 and plan["budget"]["reference_runs_max"] == 2,
                "Recovery exposure budget changed")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        errors.append({"scope": "sources", "error": str(exc)})

    anatomy = None
    try:
        anatomy = prior.read_json(OLD / "reference/ORN_INDEX_MAP.json")
        require(prior.sha(anatomy["source_path"]) == anatomy["source_sha256"], "Anatomical map changed")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        errors.append({"scope": "anatomy", "error": str(exc)})
    expected_orn = None if anatomy is None else anatomy["ORN_DM1"]

    for arm in ("plus", "minus"):
        folder = OLD / ("native_" + arm + "_01")
        try:
            run = prior.inspect_arm(folder, expected_orn)
            require(run["engine"] == "causal_cuda" and run["arm"] == arm and run["metrics"]["complete"],
                    "Exposed native arm incomplete")
            runs["causal_cuda", arm] = run
        except (ValueError, OSError, KeyError, TypeError, IndexError, FileNotFoundError) as exc:
            errors.append({"scope": str(folder), "error": str(exc)})
    if all(("causal_cuda", arm) in runs for arm in ("plus", "minus")):
        try:
            prepared["causal_cuda"] = prior.check_prepared_receipt(
                OLD / "PREPARED_COMPARE_causal_cuda.json",
                OLD / "native_plus_01/prepared_state", OLD / "native_minus_01/prepared_state",
                "causal_cuda")
        except (ValueError, OSError, KeyError, TypeError, FileNotFoundError) as exc:
            errors.append({"scope": "native prepared pair", "error": str(exc)})

    contracts = sorted(HERE.glob("*/RUN_CONTRACT.json"))
    if len(contracts) > 2:
        errors.append({"scope": "budget", "error": "More than two new run contracts"})
    exposure = []
    for path in contracts:
        try:
            contract = current.read_json(path)
            require(contract["engine"] == "reference_cuda" and contract["field"] in ("plus", "minus"),
                    "Only strict plus/minus references allowed")
            require(contract["wall_limit_s"] == plan["budget"]["wall_each_s_max"], "New wall contract")
            require(contract["recovery_from_campaign26"] is True, "Recovery identity")
            exposure.append(contract["field"])
            run = current.inspect_arm(path.parent, expected_orn)
            require(run["engine"] == "reference_cuda" and run["arm"] == contract["field"], "Run identity")
            runs["reference_cuda", run["arm"]] = run
        except (ValueError, OSError, KeyError, TypeError, IndexError, FileNotFoundError) as exc:
            errors.append({"scope": str(path.parent), "error": str(exc)})
    if len(exposure) != len(set(exposure)):
        errors.append({"scope": "budget", "error": "Repeated arm; preserve exposure"})
    for path in HERE.iterdir():
        if path.is_dir() and path not in {p.parent for p in contracts} and any((path / name).exists() for name in ("RESULT.json", "PROGRESS.jsonl", "traces.npz")):
            errors.append({"scope": str(path), "error": "Unbound organism attempt"})

    observed_wall = 0.0
    for path in contracts:
        result_path = path.parent / "RESULT.json"
        if result_path.exists():
            try:
                wall = float(current.read_json(result_path)["wall_total_s"])
                require(math.isfinite(wall) and 0 <= wall <= plan["budget"]["wall_each_s_max"], "Run wall")
                observed_wall += wall
            except (ValueError, OSError, KeyError, TypeError) as exc:
                errors.append({"scope": str(result_path), "error": str(exc)})
    if observed_wall > plan["budget"]["aggregate_wall_s_max"]:
        errors.append({"scope": "budget", "error": "Aggregate new reference wall exceeded"})

    if all(("reference_cuda", arm) in runs for arm in ("plus", "minus")):
        try:
            prepared["reference_cuda"] = current.check_prepared_receipt(
                HERE / "PREPARED_COMPARE_reference_cuda.json",
                HERE / "reference_plus_01/prepared_state", HERE / "reference_minus_01/prepared_state",
                "reference_cuda")
        except FileNotFoundError as exc:
            missing.append({"scope": "reference prepared pair", "missing": str(exc.filename)})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append({"scope": "reference prepared pair", "error": str(exc)})
    for arm in ("plus", "minus"):
        if ("reference_cuda", arm) not in runs and arm not in exposure:
            missing.append({"scope": "reference " + arm, "missing": "Not launched"})

    if errors:
        decision = {"rival": "C", "classification": "BLOQUEADO", "reason": "Invalid or incomplete new reference/identity evidence"}
    elif missing or "reference_cuda" not in prepared:
        decision = {"rival": None, "classification": "INCOMPLETO", "reason": "Both complete strict references and exact preparation comparison are required"}
    else:
        decision = prior.decide(runs)
    summarized = {}
    for key, run in runs.items():
        summarized["/".join(key)] = {
            "folder": run["folder"], "raw_hashes": run["raw_hashes"],
            "resources": run["resources"], "complete": run["metrics"]["complete"],
            "signed_command_deg": run["metrics"]["signed_command_deg"],
        }
    return {
        "schema": "stage4_reference_recovery_composite_v1",
        "sources": sources, "decision": decision, "errors": errors, "missing": missing,
        "new_reference_wall_s": observed_wall, "prepared": prepared, "runs": summarized,
        "stage4_navigation_admitted": False, "stage5_admitted": False,
        "limitation": "The old campaign remains blocked. At most this confirms numerical parity of the source-to-command effect; no online-versus-replay navigation test or animal equivalence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    result = compute()
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.write:
        with args.write.open("x") as stream:
            stream.write(output)
    print(output)
    return 0 if not result["errors"] and not result["missing"] and result["decision"]["classification"] in (
        "CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA", "DESCARTADO_EN_ESTE_CONTRATO") else 2


if __name__ == "__main__":
    raise SystemExit(main())
