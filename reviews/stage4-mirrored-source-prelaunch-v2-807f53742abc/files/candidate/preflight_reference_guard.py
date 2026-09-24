"""Read-only prepared-pose tolerance check on previously exposed Stage3 traces."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIELDS = json.loads((HERE / "CAMPOS.json").read_text())
OLD_COMMON = json.loads((ROOT / "campanas/etapa4_next_20260924_25/geometry_preflight_01/RESULT.json").read_text())["calibration"]["old_initial_common"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main() -> None:
    native = ROOT / "campanas/etapa3_pn629_intervention_20260923_15"
    reference = ROOT / "campanas/etapa3_funcional_20260923_16"
    paths = [native / f"full_{arm}_01/traces.npz" for arm in ("odor_left", "odor_right", "sham", "uniform")]
    paths += [reference / f"reference_{arm}_01/traces.npz" for arm in ("odor_left", "odor_right", "sham")]
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            antennae = z["antenas_mm"][39].copy()
            if z["fase"][39] != "preparacion" or int(z["paso"][39]) != 40:
                raise ValueError("Not the frozen 40-ms prepared pose: " + str(path))
        c = {name: np.exp(-np.sum((antennae[:, :2] - spec["source_mm"]) ** 2, axis=1) /
                          (2 * spec["sigma_mm"] ** 2)) for name, spec in FIELDS.items()}
        plus_lr = float(c["plus"][0] - c["plus"][1])
        minus_lr = float(c["minus"][0] - c["minus"][1])
        rows.append({"path": str(path), "trace_sha256": sha(path),
                     "common_pair_abs": abs(float(c["plus"].mean() - c["minus"].mean())),
                     "common_prior_abs": abs(float(c["plus"].mean() - OLD_COMMON)),
                     "contrast_antisymmetry_abs": abs(plus_lr + minus_lr),
                     "plus_signed_LR": plus_lr, "minus_signed_LR": minus_lr})
    result = {"schema": "stage4_preexisting_prepared_pose_guard_v1",
              "scope": "Already exposed Stage3 prepared poses; no new organism or future source outcomes",
              "rows": rows,
              "max_common_pair_abs": max(r["common_pair_abs"] for r in rows),
              "max_common_prior_abs": max(r["common_prior_abs"] for r in rows),
              "max_contrast_antisymmetry_abs": max(r["contrast_antisymmetry_abs"] for r in rows),
              "code_sha256": sha(Path(__file__)), "source_fields_sha256": sha(HERE / "CAMPOS.json"),
              "organism_executed": False}
    out = HERE / "REFERENCE_POSE_GUARD_01.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("max_common_pair_abs", "max_common_prior_abs",
                                              "max_contrast_antisymmetry_abs")}))


if __name__ == "__main__":
    main()
