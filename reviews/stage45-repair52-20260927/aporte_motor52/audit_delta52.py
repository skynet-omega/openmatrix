"""CPU-only independent52/51 and population check. No neural simulation.

Reads scientific arrays first; compares author summaries only after recomputing.
No new cell selection, reader, temporal window or biological gate is introduced.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import json
import resource
import time
from pathlib import Path
import numpy as np
import verify52 as v

H = Path(__file__).resolve().parent.parent
OUT = H / 'aporte_motor52/DELTA_Y_POBLACIONES.json'


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (15, 17))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    v.need(not OUT.exists(), 'Preserve previous independent result')
    start = time.process_time()
    plan = v.js(H / 'PILOT_PLAN.json')
    panel = v.arrays(H / 'PANEL.npz')
    masks = dict(zip(panel['group_names'], panel['group_masks']))
    groups = ['JO_CE', 'AMMC_WED', 'all_DN']
    counts = {k: int(masks[k].sum()) for k in groups}
    v.need(counts == {'JO_CE': 335, 'AMMC_WED': 1108, 'all_DN': 1314}, 'Group counts')
    threshold = plan['observation']['descriptive_threshold_q']
    v.need(threshold == 1e-6, 'Frozen descriptive threshold')
    window = slice(plan['analysis_window_ms'][0] - 1, plan['analysis_window_ms'][1])
    old = v.arrays(H / 'parent51_projection.npz')
    projection = v.js(H / 'PROVENANCE.json')
    new, wide, traces, changes, old_metrics, new_metrics, inputs = {}, {}, {}, {}, {}, {}, {}
    for name in plan['arms']:
        d = H / name
        a, t, w = (v.arrays(d / f) for f in ['neural_and_inputs.npz', 'traces.npz', 'wide_observation.npz'])
        new[name], traces[name], wide[name] = a, t, w
        # Verify the portable51 projection against original raw51 archives.
        for filename in ['neural_and_inputs.npz', 'traces.npz']:
            src = H.parent / 'etapa45_alternativas_20260927_51' / name / filename
            source_key = str(src.relative_to(H.parents[1]))
            v.need(v.sha(src) == projection['parent_arrays_sources'][source_key], 'Original51 source hash')
            previous = v.arrays(src)
            for k in previous:
                key = name + '__' + k
                if key in old:
                    v.same(previous[k], old[key], 'Saved51 projection/' + key)
        v.same(a['initial_q'], old[name+'__initial_q'], 'Same initial release51/52')
        for clock in ['PN_time_ns', 'body_time_ns']:
            v.same(t[clock], t['CNS_time_ns'], 'Committed clocks')
        qdiff = a['q'] - old[name+'__q']
        endpoint = a['final_q'] - old[name+'__final_q']
        changes[name] = dict(
            max_abs_q16=float(np.abs(qdiff).max()),
            max_abs_final_q_all=float(np.abs(endpoint).max()),
            max_abs_JO_drive=float(np.abs(a['JO_drive']-old[name+'__JO_drive']).max()),
            max_abs_yaw_deg_s=float(np.abs(t['neural_yaw_unapplied_rad_s']-old[name+'__neural_yaw_unapplied_rad_s']).max()) * 180 / np.pi,
            max_abs_position_mm=float(np.abs(t['position_mm']-old[name+'__position_mm']).max()),
            endpoint_changed_above1e_6={g:int((np.abs(endpoint[panel['canonical_rows'][masks[g]]])>threshold).sum()) for g in groups})
        l, r = [int(np.flatnonzero(a['ids'] == cell)[0]) for cell in [10118, 10065]]
        for target, q, yaw in [(new_metrics, a['q'], t['neural_yaw_unapplied_rad_s']),
                               (old_metrics, old[name+'__q'], old[name+'__neural_yaw_unapplied_rad_s'])]:
            target[name] = dict(q=float((q[:, l]-q[:, r])[window].mean()),
                                yaw=float(np.rad2deg(yaw)[window].mean()))
        inputs[name] = {f:v.sha(d/f) for f in ['neural_and_inputs.npz','traces.npz','wide_observation.npz']}
    for names in [[n for n in new if n.startswith('air')], ['G_odor0','G_odor1'], ['I_odor0','I_odor1']]:
        for name in names[1:]:
            v.same(wide[name]['q'][:10], wide[names[0]]['q'][:10], 'Common prefix within law')
    effects = {}
    for label, m in [('51',old_metrics),('52',new_metrics)]:
        effects[label] = {}
        for kind in ['q','yaw']:
            g = m['G_odor1'][kind]-m['G_odor0'][kind]
            i = m['I_odor1'][kind]-m['I_odor0'][kind]
            effects[label][kind] = dict(G_odor_effect=g, I_odor_effect=i,
                                        G_minus_I_odor_effect=g-i)
    q_limit = plan['criteria']['DNb05_directional_q_min']
    yaw_limit = plan['criteria']['raw_yaw_deg_s_min']
    g_material = abs(effects['52']['q']['G_minus_I_odor_effect']) >= q_limit
    y_material = abs(effects['52']['yaw']['G_minus_I_odor_effect']) >= yaw_limit
    temporal = {}
    for odor in ['0','1']:
        left, right = 'airL_odor'+odor, 'airR_odor'+odor
        delta = wide[left]['q']-wide[right]['q']
        populations = {}
        for group in groups:
            x = delta[:, masks[group]]
            crossed = np.abs(x)>threshold
            latencies = np.full(x.shape[1],-1,dtype=np.int64)
            hit_rows,hit_cols = np.nonzero(crossed)
            for row,col in zip(hit_rows,hit_cols):
                if latencies[col] == -1:
                    latencies[col] = row+1
            nonzero_latency = latencies[latencies>0]
            populations[group] = dict(cells=int(x.shape[1]),
                positive_per_ms=(x>threshold).sum(axis=1).tolist(),
                negative_per_ms=(x < -threshold).sum(axis=1).tolist(),
                above_threshold_per_ms=crossed.sum(axis=1).tolist(),
                first_sample_ms_by_cell=latencies.tolist(),
                cells_crossing_threshold_any_time=int(crossed.any(axis=0).sum()),
                cells_above_at_90ms=int(crossed[-1].sum()),
                first_crossing_ms=int(nonzero_latency.min()) if len(nonzero_latency) else None)
        yaw = .5*(np.rad2deg(traces[left]['neural_yaw_unapplied_rad_s'])-np.rad2deg(traces[right]['neural_yaw_unapplied_rad_s']))
        signs = np.sign(yaw[window][yaw[window]!=0])
        total_l = new[left]['JO_drive'].sum(axis=1)
        total_r = new[right]['JO_drive'].sum(axis=1)
        temporal[odor] = dict(populations=populations,
            JO_mean_L=float(total_l[window].mean()), JO_mean_R=float(total_r[window].mean()),
            JO_R_over_L_minus1=float(total_r[window].mean()/total_l[window].mean()-1),
            raw_yaw_half_signs_51_90=dict(positive=int((yaw[window]>0).sum()),
                negative=int((yaw[window]<0).sum()),zero=int((yaw[window]==0).sum()),
                changes=int(np.count_nonzero(np.diff(signs)))))
    # Compare author summaries after independently deriving these quantities.
    author = v.js(H / 'POPULATIONS.json')
    v.need(author['anatomical_counts']==counts,'Author population counts')
    for name, got in changes.items():
        v.need(got == author['corrected_minus_51'][name], 'Author correction summary differs: '+name)
    for odor, got in temporal.items():
        for k in ['JO_mean_L','JO_mean_R','JO_R_over_L_minus1','raw_yaw_half_signs_51_90']:
            v.need(got[k]==author['contrasts'][odor][k], 'Author temporal summary differs: '+k)
        for group, value in got['populations'].items():
            for k in value:
                if k != 'first_crossing_ms':
                    v.need(value[k]==author['contrasts'][odor]['populations'][group][k], 'Author population trajectory differs: '+k)
    result = dict(status='PASS_INDEPENDENT_RECALCULATION', anatomical_counts=counts,
        parent51_projection_matches_original_arrays=True, corrections=changes,
        G_I_odor_effects=effects, G_I_meets_q_minimum=bool(g_material),
        G_I_meets_yaw_minimum=bool(y_material), G_I_both_material=bool(g_material and y_material),
        G_I_yaw_fraction_of_minimum=float(abs(effects['52']['yaw']['G_minus_I_odor_effect'])/yaw_limit),
        temporal_air=temporal, source_hashes=inputs,
        stage4_admitted=False,stage5_admitted=False,
        scope='Repair changes trajectories, not demonstrated orientation. Descriptive threshold is not physiological or statistical. No simulation.',
        CPU_s=time.process_time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_sha256=v.sha(__file__))
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','anatomical_counts','G_I_odor_effects','G_I_both_material','G_I_yaw_fraction_of_minimum','CPU_s','peak_RSS_bytes']}))
    for odor, item in temporal.items():
        print(odor, 'yaw signs', item['raw_yaw_half_signs_51_90'], 'JO imbalance',item['JO_R_over_L_minus1'],
              'groups',{k:(g['cells_crossing_threshold_any_time'],g['cells_above_at_90ms'],g['first_crossing_ms']) for k,g in item['populations'].items()})


if __name__=='__main__':
    main()
