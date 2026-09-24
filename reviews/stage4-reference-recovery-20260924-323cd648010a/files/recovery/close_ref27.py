"""Mechanical postclose receipt for the prospective strict-reference recovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "etapa4_mirrored_source_20260924_26"


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    plan = read(HERE / "PLAN.json")
    raw1 = HERE / "RAW_VERIFIED_01.json"
    raw2 = HERE / "RAW_VERIFIED_02.json"
    need(raw1.read_bytes() == raw2.read_bytes(), "Normal/-O raw receipts differ")
    raw = read(raw1)
    decision = raw["decision"]
    need(raw["errors"] == [] and raw["missing"] == [], "Composite verification incomplete")
    need(decision["rival"] == "A" and decision["classification"] == "CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA",
         "Numerical source-to-command result not confirmed")
    need(raw["stage4_navigation_admitted"] is False and raw["stage5_admitted"] is False,
         "Unsupported stage admission")
    need(raw["prepared"]["reference_cuda"]["recomputed_section_equality"] is True,
         "Prepared reference pair not exact")
    need(all(decision["native_checks"].values()) and all(decision["reference_checks"].values()),
         "Native/reference pair checks failed")
    limits = plan["thresholds"]
    for arm in ("plus", "minus"):
        p = decision["parity"][arm]
        need(p["yaw_sup_deg"] <= limits["native_reference_yaw_sup_deg_max"] and
             p["command_L1_deg"] <= limits["native_reference_command_L1_deg_max"],
             arm + " numerical parity")
    need(raw["new_reference_wall_s"] <= plan["budget"]["aggregate_wall_s_max"], "Aggregate wall")
    runs = {}
    for arm in ("plus", "minus"):
        folder = HERE / ("reference_" + arm + "_01")
        result = read(folder / "RESULT.json")
        contract = read(folder / "RUN_CONTRACT.json")
        need(result["status"] == "COMPLETE" and result["completed_preparation_ms"] == 40 and
             result["completed_trial_ms"] == 400 and result["error"] is None and result["cleanup_errors"] == [],
             arm + " not fully complete")
        need(contract["field"] == arm and contract["engine"] == "reference_cuda" and
             contract["plan_sha256"] == sha(HERE / "PLAN.json") and
             contract["source_identity"]["source_lock_sha256"] == sha(HERE / "SOURCE_LOCK.json") and
             contract["wall_limit_s"] == plan["budget"]["wall_each_s_max"], arm + " contract mismatch")
        need(result["wall_total_s"] <= plan["budget"]["wall_each_s_max"], arm + " wall cap")
        runs[arm] = {"result_sha256": sha(folder / "RESULT.json"),
                     "trace_sha256": sha(folder / "traces.npz"),
                     "completed_trial_ms": result["completed_trial_ms"],
                     "wall_s": result["wall_total_s"]}
    need(abs(sum(run["wall_s"] for run in runs.values()) - raw["new_reference_wall_s"]) < 1e-9,
         "Composite wall differs")
    old_close = read(OLD / "CLOSE_01.json")
    need(old_close["classification"] == "BLOQUEADO" and old_close["stage4_admission"] is False,
         "Old closure changed")
    receipt = {
        "schema": "stage4_reference_recovery_postclose_v1",
        "classification": "CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA",
        "scope": "Mirrored finite-source-to-command numerical parity only",
        "stage4_status": "ABIERTO",
        "stage5_status": "ABIERTO",
        "stage4_admission": False,
        "stage5_admission": False,
        "campaign26_classification_unchanged": old_close["classification"],
        "native_pair": decision["native_pair"],
        "reference_pair": decision["reference_pair"],
        "parity": decision["parity"],
        "prepared_reference_exact": True,
        "runs": runs,
        "new_reference_wall_s": raw["new_reference_wall_s"],
        "aggregate_limit_s": plan["budget"]["aggregate_wall_s_max"],
        "source_hashes": {name: sha(HERE / name) for name in
                          ("PLAN.json", "SOURCE_LOCK.json", "verify_ref27.py",
                           "RAW_VERIFIED_01.json", "RAW_VERIFIED_02.json",
                           "PREPARED_COMPARE_reference_cuda.json")},
        "limitations": [
            "No feedback online-versus-yoked perturbation or source-distance causal control.",
            "One prepared organism per side; no compatible live-fly calibration or biological equivalence.",
            "Azimuthal roller-assisted body and effective DNb05 reader, not natural six-leg gait.",
            "Scientific effect is tiny in body position and the late plus command reverses sign.",
            "ChatGPT and Jev did not audit these new complete strict references at this close.",
        ],
        "next_required": "Prospective feedback intervention with matched online and replay input, distance/heading and raw numerical checks before Stage4 admission.",
    }
    with (HERE / "CLOSE_01.json").open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: receipt[k] for k in ("classification", "stage4_status", "stage5_status", "new_reference_wall_s")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
