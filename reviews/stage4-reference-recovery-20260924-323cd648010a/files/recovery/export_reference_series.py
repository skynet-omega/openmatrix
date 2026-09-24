"""Descriptive per-millisecond CSV for external review of complete references."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "REFERENCE_SCALAR_SERIES_01.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message: str):
    if not ok:
        raise ValueError(message)


def main():
    need(not OUT.exists(), "Descriptive export already exists")
    rows = []
    hashes = {}
    for arm in ("plus", "minus"):
        source = HERE / ("reference_" + arm + "_01/traces.npz")
        hashes[arm] = sha(source)
        with np.load(source, allow_pickle=False) as z:
            n = len(z["fase"])
            need(n == 440 and all(len(z[k]) == n for k in z.files), "Trace length")
            need(z["fase"][:40].tolist() == ["preparacion"] * 40 and
                 z["fase"][40:].tolist() == ["ensayo"] * 400, "Trace phase")
            need(z["paso"][40:].tolist() == list(range(1, 401)), "Trial steps")
            for i in range(40, n):
                rows.append({
                    "profile": "reference_cuda", "arm": arm, "trial_ms": int(z["paso"][i]),
                    "CNS_time_ns": int(z["CNS_time_ns"][i]),
                    "field_L": float(z["concentracion_campo"][i, 0]),
                    "field_R": float(z["concentracion_campo"][i, 1]),
                    "sensor_used_L": float(z["sensores_usados"][i, 0]),
                    "sensor_used_R": float(z["sensores_usados"][i, 1]),
                    "DN_q_usada_L": float(z["DN_q_usada"][i, 2]),
                    "DN_q_usada_R": float(z["DN_q_usada"][i, 3]),
                    "DN_baseline_L": float(z["DN_baseline"][i, 2]),
                    "DN_baseline_R": float(z["DN_baseline"][i, 3]),
                    "command_deg_s": float(np.rad2deg(z["command_yaw_rate_rad_s"][i])),
                    "yaw_deg": float(z["yaw_delta_deg"][i]),
                })
    fields = list(rows[0])
    with OUT.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "schema": "stage4_reference_scalar_series_descriptive_v1",
        "classification": "DESCRIPTIVE_ONLY",
        "rows": len(rows), "rows_by_arm": {"plus": 400, "minus": 400},
        "source_trace_sha256": hashes,
        "csv_sha256": sha(OUT),
        "scope": "Selected scalar fields extracted from full reference traces; not the raw verifier or a navigation claim.",
    }
    with (HERE / "REFERENCE_SCALAR_SERIES_MANIFEST_01.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
