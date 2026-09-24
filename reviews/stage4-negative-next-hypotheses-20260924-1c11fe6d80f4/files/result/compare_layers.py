"""Read-only descriptive S+/S- layer contrasts; never changes the frozen gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PLUS = HERE.parent / "etapa4_gaussian_gate_repair_20260923_22" / "native_plus_01"
MINUS = HERE / "native_minus_02"
FIELDS = ("concentracion_campo", "sensores_usados", "ORN_q_L", "ORN_q_R",
          "PN_q_legacy", "DN_q_actual", "DN_q_usada", "command_yaw_rate_rad_s",
          "yaw_delta_deg", "position_mm")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape or a.shape[0] != 440 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid or unmatched S+/S- arrays")
    d = a - b
    if not np.array_equal(d[:40], np.zeros_like(d[:40])):
        raise ValueError("Prepared samples differ")
    return {"shape": list(d.shape), "max_abs_trial": float(np.max(np.abs(d[40:]))),
            "mean_abs_300_400ms": float(np.mean(np.abs(d[339:440]))),
            "last_difference": np.asarray(d[-1]).tolist()}


def main() -> None:
    result = {"schema": "stage4_gaussian_layers_descriptive_v1",
              "scope": "Descriptive differences after exact preparation; no causal attribution to one layer or navigation claim.",
              "trial_late_rows": [339, 439], "fields": {}, "flow": {}, "sources": {}}
    for arm, path in (("plus", PLUS), ("minus", MINUS)):
        result["sources"][arm] = {"traces_sha256": sha(path / "traces.npz"),
                                  "flow_sha256": sha(path / "flow/FLOW.npz")}
    with np.load(PLUS / "traces.npz", allow_pickle=False) as a, np.load(MINUS / "traces.npz", allow_pickle=False) as b:
        for name in FIELDS:
            result["fields"][name] = describe(a[name], b[name])
        for name in ("PN_q_legacy", "DN_q_actual", "DN_q_usada"):
            result["fields"][name]["late_mean_abs_by_column"] = np.mean(
                np.abs(a[name][339:440] - b[name][339:440]), axis=0).tolist()
    with np.load(PLUS / "flow/FLOW.npz", allow_pickle=False) as a, np.load(MINUS / "flow/FLOW.npz", allow_pickle=False) as b:
        if not np.array_equal(a["ids"], b["ids"]) or not np.array_equal(a["time_ns"], b["time_ns"]):
            raise ValueError("Native flow witnesses do not share identity or clock")
        ids = [int(x) for x in a["ids"]]
        for name in ("raw_signed", "target"):
            result["flow"][name] = {"all": describe(a[name], b[name]),
                "late_mean_abs_by_id": dict(zip(ids, np.mean(np.abs(a[name][339:440] - b[name][339:440]), axis=0).tolist()))}
    out = HERE / "LAYER_DIAGNOSTIC_01.json"
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(out), "late_concentration": result["fields"]["concentracion_campo"]["mean_abs_300_400ms"],
        "late_PN_q": result["fields"]["PN_q_legacy"]["late_mean_abs_by_column"],
        "late_DN_q": result["fields"]["DN_q_actual"]["late_mean_abs_by_column"]}))


if __name__ == "__main__":
    main()
