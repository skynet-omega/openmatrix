"""Post-acquisition correction of the frozen analyzer's literal-zero verdict.

Reads its verified summaries; does not import the organism or change acquisition.
The original analyzer and RESULTADOS.json remain preserved for provenance.
"""
from pathlib import Path
import hashlib
import json
import math
import time

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def classify(cells):
    require(bool(cells), "No observations to classify")
    negative = positive = False
    literal_zero = True
    for cell in cells:
        lo, hi = cell["final_target"]["min"], cell["final_target"]["max"]
        require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi,
                "Invalid final-target extrema")
        negative |= lo < 0
        positive |= hi > 0
        literal_zero &= lo == 0 and hi == 0
    if negative:
        name = "NEGATIVE_FINAL_TARGET_REQUIRES_LOCALIZATION"
    elif positive:
        name = "POSITIVE_FINAL_TARGET_REQUIRES_LOCALIZATION"
    else:
        require(literal_zero, "Unclassified target")
        name = "ZERO_FINAL_TARGET_IN_RECORDED_COMMITTED_EVOLUTION"
    return dict(classification=name, literal_zero=literal_zero,
                has_negative=negative, has_positive=positive)


def summarize(result):
    require(result["status"] == "COMPLETE", "Acquisition not complete")
    require(set(result["arms"]) == {"sham", "odor"}, "Incomplete comparison")
    cells = []
    for arm in result["arms"].values():
        require(arm["completed_ms"] == 3000, "Incomplete arm")
        require(set(arm["phases"]) == {"baseline", "stimulus"}, "Missing phase")
        for phase in arm["phases"].values():
            require([c["id"] for c in phase["cells"]] == [10045, 10056],
                    "Wrong cell identities")
            cells.extend(phase["cells"])
    verdict = classify(cells)
    phases = [p for a in result["arms"].values() for p in a["phases"].values()]
    # This is a sign-based consequence of the existing rectified monotone law.
    # It is not a calibration or an independent validation of that law.
    verdict["generic_suppression_supported_in_recorded_evaluations"] = (
        verdict["literal_zero"]
        and all(p["base_final_target_mismatches"] == 0 for p in phases)
        and all(c["gain"]["min"] > 0 and c["margin"]["max"] <= 0
                and c["final_rate"]["min"] > 0 for c in cells)
    )
    paired = []
    for phase in ("baseline", "stimulus"):
        s = result["arms"]["sham"]["phases"][phase]
        o = result["arms"]["odor"]["phases"][phase]
        for left, right in zip(s["cells"], o["cells"]):
            paired.append(dict(
                phase=phase, id=left["id"],
                odor_minus_sham_RK_weighted_means={
                    key: right[key]["time_weighted_mean"] - left[key]["time_weighted_mean"]
                    for key in ("net", "positive_aux", "negative_aux", "margin", "final_target")
                },
                sham_margin_max=left["margin"]["max"],
                odor_margin_max=right["margin"]["max"],
            ))
    return dict(status="COMPLETE", **verdict, paired=paired,
        stage4_admission=False, stage5_admission=False,
        scope="Successful retained k1-k3 evaluations in47 with observations reproducing45. "
              "RK-weighted summaries are not independent biological samples. "
              "Not a retrospective recovery of unrecorded45 operands or a physiological validation.",
        supersedes="Only the target classification in frozen RESULTADOS.json; original retained.")


def main():
    started = time.process_time()
    source = HERE / "RESULTADOS.json"
    data = source.read_bytes()
    result = summarize(json.loads(data))
    result["original_summary_sha256"] = hashlib.sha256(data).hexdigest()
    result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["CPU_s"] = time.process_time() - started
    (HERE / "DICTAMEN.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
