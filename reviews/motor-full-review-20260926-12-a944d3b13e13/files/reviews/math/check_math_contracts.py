"""Finite CPU arithmetic review; no simulator, CuPy, CUDA or neural stepping.

This reproduces the expressions present in frozen engine11 and scans saved
event calendars. It does not claim to reproduce CUDA execution or to know its
unrecorded adaptive mesh. Output is confined to this review directory.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ENGINE = ROOT / "motor_nuevo/event_memory_rk3_20260925_11/engine"
EVENTS = ROOT / "motor_nuevo/event_memory_rk3_20260925_11/candidate_1000ms_01/EVENTS.json"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def side_case(start, stop):
    h = stop - start
    endpoint = start + h
    left = math.nextafter(endpoint, start)
    canonical_left = math.nextafter(stop, start)
    return dict(start_s=start, stop_s=stop, h_s=h,
                reconstructed_endpoint_s=endpoint,
                reconstructed_left_s=left,
                endpoint_delta_ulp=(endpoint-stop)/math.ulp(stop),
                legacy_left_excludes_event=left < stop,
                legacy_right_includes_event=endpoint >= stop,
                canonical_left_excludes_event=canonical_left < stop,
                canonical_right_includes_event=stop >= stop,
                hex={"start": start.hex(), "stop": stop.hex(),
                     "h": h.hex(), "end": endpoint.hex(), "left": left.hex()})


def main():
    runtime = (ENGINE / "graph_runtime.py").read_text()
    controller = (ENGINE / "resident_controller.cu").read_text()
    ports_path = ROOT / "campanas/etapa3_pn629_intervention_20260923_15/event_ports.py"
    ports = ports_path.read_text()
    for text, needle in (
        (runtime, "nextafter(c[0]+c[1],c[0])"),
        (controller, "const double available = stop - c->used_s;"),
        (ports, "double t=clock[0]+fraction*clock[1];"),
        (ports, "if(et[p]<=t)"),
    ):
        need(needle in text, "Frozen arithmetic changed: " + needle)

    blocks = json.loads(EVENTS.read_text())
    need(len(blocks) == 16000, "Unexpected saved calendar")
    summary = {}
    examples = {}
    for index, block in enumerate(blocks):
        kind = "predictor" if index % 2 == 0 else "committed"
        duration = 62500 if kind == "predictor" else 125000
        need(block["duration_ns"] == duration, "Unsupported block contract")
        row = summary.setdefault(kind, dict(adjacent_pairs=0, rounded_above=0,
                                           rounded_below=0))
        times = sorted(set(e["time_s"] for e in block["events"]))
        need(all(math.isfinite(t) and 0 <= t <= duration*1e-9 for t in times),
             "Invalid source event")
        for start, stop in zip([0.] + times, times):
            row["adjacent_pairs"] += 1
            endpoint = start + (stop-start)
            if endpoint == stop:
                continue
            direction = "above" if endpoint > stop else "below"
            row["rounded_" + direction] += 1
            key = kind + "_" + direction
            if key not in examples:
                examples[key] = dict(block=block["block"],
                                     start_elapsed_ns=block["start_elapsed_ns"],
                                     **side_case(start, stop))

    up = examples["committed_above"]
    down = examples["committed_below"]
    need(not up["legacy_left_excludes_event"], "Missing upward counterexample")
    need(not down["legacy_right_includes_event"], "Missing downward counterexample")
    need(up["canonical_left_excludes_event"] and down["canonical_right_includes_event"],
         "Canonical endpoint repair failed")

    # Manufactured legal projected state: q jumps 0 -> 1 at stop; a free
    # component satisfies x'=rate*(q-x), x(start)=0. Before stop x is exactly 0.
    # RK k1,k2,k3 and high x are 0. Wrong-sided k4=rate creates spurious error.
    def spurious_rejection(case, rate):
        h = case["h_s"]
        legacy_k4 = rate if case["reconstructed_left_s"] >= case["stop_s"] else 0.
        error = abs(-h * legacy_k4 / 8.) / 1e-7
        smaller_ns = math.floor(h * 1e9 * .5)
        return dict(model="x'=rate*(q-x), x=0 before the q:0->1 event",
                    rate_per_s=rate, exact_pre_event_free_state=0.,
                    rk_high_free_state=0., all_final_coordinates_in_unit_interval=True,
                    legacy_k4=legacy_k4, normalized_legacy_error=error,
                    canonical_k4=0., normalized_canonical_error=0.,
                    rejected=error > 1., smaller_ns=smaller_ns,
                    minimum_ns=100, accuracy_limit=(error > 1. and smaller_ns < 100))

    false_rejection = spurious_rejection(up, 1.)
    need(false_rejection["rejected"], "Saved-calendar false rejection no longer holds")
    # Power-of-two scaling keeps exactly the same floating-point side error.
    tiny = side_case(up["start_s"] / 128., up["stop_s"] / 128.)
    false_limit = spurious_rejection(tiny, 1000.)
    need(false_limit["accuracy_limit"], "Minimum-step counterexample failed")

    # Exact rational analysis of the pair at z=lambda*h=-1. This is an inherent
    # estimator limitation, not evidence of a new bug in the RK coefficients.
    z = F(-1)
    y = F(1)
    k1 = z*y
    k2 = z*(y+F(1,2)*k1)
    k3 = z*(y+F(3,4)*k2)
    high = y+F(2,9)*k1+F(1,3)*k2+F(4,9)*k3
    k4 = z*high
    embedded = F(-5,72)*k1+F(1,12)*k2+F(1,9)*k3-F(1,8)*k4
    need(high == F(1,3) and embedded == 0, "RK algebra differs")
    estimator_limit = dict(equation="y'=-10000*y, y(0)=1, h=100 microseconds",
                           high_exact_rational=str(high), embedded_exact_rational=str(embedded),
                           true_value=math.exp(-1.), absolute_error=abs(float(high)-math.exp(-1.)),
                           scale=1e-7+1e-5,
                           normalized_true_error=abs(float(high)-math.exp(-1.))/(1e-7+1e-5),
                           scope="Inherent embedded-estimator counterexample; no claim this mode was encountered by organism11")

    sources = [ENGINE/"graph_runtime.py", ENGINE/"resident_controller.cu", ports_path, EVENTS]
    output = dict(status="COUNTEREXAMPLE_CONFIRMED_CPU", gpu_executed=False,
                  neural_simulations_executed=0,
                  scope="Binary64 arithmetic and saved-calendar susceptibility, not a reconstruction of unrecorded CUDA trials",
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                  saved_calendar_summary=summary, first_examples=examples,
                  manufactured_false_rejection=false_rejection,
                  manufactured_accuracy_limit=dict(clock=tiny, **false_limit),
                  embedded_estimator_limit=estimator_limit)
    path = HERE / "MATH_COUNTEREXAMPLES.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"status": output["status"], "saved_calendar_summary": summary,
                      "false_rejection_error": false_rejection["normalized_legacy_error"],
                      "false_accuracy_limit": false_limit["accuracy_limit"],
                      "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
