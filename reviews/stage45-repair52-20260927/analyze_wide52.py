"""Descriptive population summaries fixed in the prospective52 contract.

No selection of another neuron, reader, polarity, gain, or analysis window.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from analyze_pilot import compute, need
from verify_qualification52 import arrays, check_units

H = Path(__file__).resolve().parent


def group_summary(delta, mask, threshold):
    x = delta[:, mask]
    above = np.abs(x) > threshold
    first = np.where(above.any(axis=0), np.argmax(above, axis=0) + 1, -1)
    window = x[50:90]
    signs = np.where(np.abs(window) > threshold, np.sign(window), 0)
    transitions = []
    for values in signs.T:
        v = values[values != 0]
        transitions.append(int(np.count_nonzero(v[1:] != v[:-1])))
    return dict(cells=int(mask.sum()),
                positive_per_ms=(x > threshold).sum(axis=1).tolist(),
                negative_per_ms=(x < -threshold).sum(axis=1).tolist(),
                above_threshold_per_ms=above.sum(axis=1).tolist(),
                signed_mean_per_ms=x.mean(axis=1).tolist(),
                rms_per_ms=np.sqrt(np.mean(x*x, axis=1)).tolist(),
                median_abs_per_ms=np.median(np.abs(x), axis=1).tolist(),
                first_sample_ms_by_cell=first.tolist(),
                sign_transitions_51_90_by_cell=transitions,
                cells_crossing_threshold_any_time=int(above.any(axis=0).sum()),
                cells_above_at_90ms=int(above[-1].sum()),
                fixed_window_RMS=float(np.sqrt(np.mean(window*window))),
                fixed_window_mean_signed=float(window.mean()))


def analyze(root=H):
    import verify_qualification52 as qualification
    qualification.H = root
    plan = json.loads((root / 'PILOT_PLAN.json').read_text())
    result, legacy = compute(root)
    panel = arrays(root / 'PANEL.npz')
    masks = dict(zip(panel['group_names'], panel['group_masks']))
    groups = ['JO_CE', 'AMMC_WED', 'all_DN']
    threshold = plan['observation']['descriptive_threshold_q']
    need(threshold == 1e-6, 'Descriptive threshold frozen')
    wide, artifact_hashes = {}, {}
    for name in plan['arms']:
        a, t = legacy[name]['a'], legacy[name]['t']
        check_units(a, t, plan['prefix_ms'])
        w = arrays(root / name / 'wide_observation.npz')
        need(np.array_equal(w['ids'], panel['ids']) and np.array_equal(w['rows'], panel['canonical_rows']), 'Wide fixed anatomical panel')
        need(w['q'].shape == (90, 2757), 'Population temporal shape')
        need(w['ORN_q'].shape == (90, 694), 'ORN temporal shape')
        need(np.array_equal(w['ORN_ids'], panel['ORN_ids']) and np.array_equal(w['ORN_rows'], panel['ORN_rows']), 'ORN panel identity')
        need(np.isfinite(w['q']).all() and np.isfinite(w['ORN_q']).all(), 'Finite wide traces')
        need(np.array_equal(w['time_ns'], t['CNS_time_ns']), 'Population clock')
        need(np.array_equal(t['CNS_time_ns'], t['PN_time_ns']) and np.array_equal(t['CNS_time_ns'], t['body_time_ns']), 'Committed clocks differ')
        rows16 = np.searchsorted(w['ids'], a['ids'])
        need(np.array_equal(w['ids'][rows16], a['ids']), 'Legacy identities')
        need(np.array_equal(w['q'][:, rows16], a['q']), 'Old and new panel disagree')
        need(np.array_equal(w['q'][-1], a['final_q'][w['rows']]), 'Population endpoint')
        need(np.array_equal(w['ORN_q'][-1], a['final_q'][w['ORN_rows']]), 'ORN endpoint')
        wide[name] = w
        artifact_hashes[name] = hashlib.sha256((root / name / 'wide_observation.npz').read_bytes()).hexdigest()
    # Equal prefix is required within each unchanged law, never across G/I laws.
    for names in [[n for n in wide if n.startswith('air')], ['G_odor0', 'G_odor1'], ['I_odor0', 'I_odor1']]:
        for name in names[1:]:
            need(np.array_equal(wide[name]['q'][:10], wide[names[0]]['q'][:10]), 'Common prefix within law')
    contrasts = {}
    for odor in [0, 1]:
        l, r = f'airL_odor{odor}', f'airR_odor{odor}'
        delta = wide[l]['q'] - wide[r]['q']
        jo_l = legacy[l]['a']['JO_drive'].sum(axis=1)
        jo_r = legacy[r]['a']['JO_drive'].sum(axis=1)
        yaw_half = (legacy[l]['yaw'] - legacy[r]['yaw']) / 2
        window = yaw_half[50:90]
        nonzero = np.sign(window[window != 0])
        contrasts[str(odor)] = dict(
            populations={g: group_summary(delta, masks[g], threshold) for g in groups},
            JO_consumed_sum_L_per_ms=jo_l.tolist(), JO_consumed_sum_R_per_ms=jo_r.tolist(),
            JO_mean_L=float(jo_l[50:90].mean()), JO_mean_R=float(jo_r[50:90].mean()),
            JO_R_over_L_minus1=float(jo_r[50:90].mean()/jo_l[50:90].mean()-1),
            raw_yaw_half_per_ms=yaw_half.tolist(),
            raw_yaw_half_signs_51_90=dict(positive=int((window > 0).sum()), negative=int((window < 0).sum()),
                                          zero=int((window == 0).sum()), changes=int((nonzero[1:] != nonzero[:-1]).sum())))
    correction = {}
    old = arrays(root / 'parent51_projection.npz')
    for name in plan['arms']:
        a, t = legacy[name]['a'], legacy[name]['t']
        item = dict(max_abs_q16=float(np.max(np.abs(a['q'] - old[name+'__q']))),
                    max_abs_final_q_all=float(np.max(np.abs(a['final_q'] - old[name+'__final_q']))),
                    max_abs_JO_drive=float(np.max(np.abs(a['JO_drive'] - old[name+'__JO_drive']))),
                    max_abs_yaw_deg_s=float(np.max(np.abs(t['neural_yaw_unapplied_rad_s'] - old[name+'__neural_yaw_unapplied_rad_s']))*180/np.pi),
                    max_abs_position_mm=float(np.max(np.abs(t['position_mm'] - old[name+'__position_mm']))))
        need(np.array_equal(a['initial_q'], old[name+'__initial_q']), 'Same source preparation as51')
        delta = a['final_q'][panel['canonical_rows']] - old[name+'__final_q'][panel['canonical_rows']]
        item['endpoint_changed_above1e_6'] = {g: int((np.abs(delta[masks[g]]) > threshold).sum()) for g in groups}
        correction[name] = item
    return dict(schema='repair52_population_analysis_v1', groups=groups,
                anatomical_counts={g:int(masks[g].sum()) for g in groups},
                threshold=threshold, threshold_scope='Descriptive, not noise-calibrated or physiological.',
                temporal_scope='Committed1ms samples; not resolved intra-ms latencies, no mediation proof.',
                contrasts=contrasts, corrected_minus_51=correction,
                source_hashes=artifact_hashes,
                input_quantity_matching_next=bool(any(result['air'][str(i)]['both_material'] and abs(contrasts[str(i)]['JO_R_over_L_minus1']) > .01 for i in [0,1])),
                next_scope='Decision aid: retained old material gate plus descriptive1% quantity imbalance; no new behavioral admission.',
                stage4_pass=False, stage5_pass=False), result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=H)
    a = p.parse_args()
    start = time.process_time()
    r, legacy = analyze(a.root)
    r['CPU_s'] = time.process_time()-start
    (a.root / 'POPULATIONS.json').write_text(json.dumps(r, indent=2, allow_nan=False)+'\n')
    (a.root / 'PILOT_RESULTS.json').write_text(json.dumps(legacy, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(CPU_s=r['CPU_s'], next_quantity_control=r['input_quantity_matching_next'], correction=r['corrected_minus_51'])))
