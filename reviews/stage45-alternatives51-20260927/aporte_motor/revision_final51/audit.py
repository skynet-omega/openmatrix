"""Independent CPU recomputation from raw51 arrays, never the main analyzer.

Not blind: reviewer knows the design and previous negative experiments.
No model imports, CNS integration, GPU use or edits outside this directory.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import hashlib
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

resource.setrlimit(resource.RLIMIT_CPU, (20, 22))
H = Path(__file__).resolve().parent
ROOT = H.parents[1]
start = time.process_time()

def need(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_npz(path):
    with np.load(path, allow_pickle=False) as a:
        return {key: a[key] for key in a.files}

plan = json.loads((ROOT/'PILOT_PLAN.json').read_text())
queue = json.loads((ROOT/'QUEUE_RESULT.json').read_text())
need(queue.get('status') != 'RUNNING', 'Wait until the queue has stopped')
anatomy = load_npz(ROOT/'aporte_motor/JO_anatomy_arrays.npz')
window = slice(50, 90)  # Prospectively fixed samples51..90, inclusive.
limits = {'q': 1.6e-5, 'yaw_deg_s': 0.02}
need(plan['analysis_window_ms'] == [51, 90], 'Window contract')
need(plan['criteria']['DNb05_directional_q_min'] == limits['q'] and
     plan['criteria']['raw_yaw_deg_s_min'] == limits['yaw_deg_s'], 'Threshold contract')
raw = {}
arms = {}
missing = {}
corrected_arrays = {}
hashes = {'PILOT_PLAN.json': sha(ROOT/'PILOT_PLAN.json'),
          'QUEUE_RESULT.json': sha(ROOT/'QUEUE_RESULT.json')}

def reconstruct_air(k, rows, velocity_factor):
    rotations = k[:, 6:].reshape(-1, 3, 3)
    relative = np.einsum('tji,tj->ti', rotations, k[:, :3]-velocity_factor*k[:, 3:6])
    projections = np.column_stack([relative[:, 0]+relative[:, 1],
                                   relative[:, 0]-relative[:, 1]])/np.sqrt(2.)
    side_index = (anatomy['source_side'] == 'R').astype(np.int64)
    sign = np.where(np.char.startswith(anatomy['source_type'], 'JO-C'), 1., -1.)
    speed = np.maximum(projections[:, side_index]*sign, 0.)
    result = 80.*speed/(speed+100.)
    result[:10] = 0.
    return result

for name in plan['arms']:
    folder = ROOT/name
    report_path = folder/'RESULT.json'
    if not report_path.exists():
        missing[name] = 'No final result'
        continue
    report = json.loads(report_path.read_text())
    if report['status'] != 'COMPLETE' or report['committed_ms'] != 90:
        missing[name] = {'status': report['status'], 'committed_ms': report['committed_ms']}
        continue
    paths = [folder/'neural_and_inputs.npz', folder/'traces.npz']
    if not all(p.exists() for p in paths):
        missing[name] = 'Raw arrays missing'
        continue
    n, t = map(load_npz, paths)
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = sha(path)
    need(n['q'].shape == (90, len(n['ids'])), name+' neural shape')
    need(np.array_equal(t['paso'], np.arange(3001, 3091)), name+' exact sample order')
    for clock in ['CNS_time_ns', 'PN_time_ns', 'body_time_ns']:
        need(np.array_equal(np.diff(t[clock]), np.full(89, 1000000)), name+' '+clock)
    need(np.array_equal(t['CNS_time_ns'], t['body_time_ns']) and
         np.array_equal(t['PN_time_ns'], t['body_time_ns']), name+' owner clocks')
    need(np.array_equal(n['JO_ids'], anatomy['source_ids']) and
         np.array_equal(n['JO_rows'], anatomy['source_rows']), name+' JO identity')
    need(np.array_equal(n['q'][:, :4], t['DN_q_actual']), name+' native DN observation')
    for src in (n, t):
        for key, value in src.items():
            if value.dtype.kind in 'fc':
                need(np.isfinite(value).all(), name+' nonfinite '+key)
    il = np.flatnonzero(n['ids'] == 10118); ir = np.flatnonzero(n['ids'] == 10065)
    need(len(il) == len(ir) == 1, name+' DNb05 identities')
    dq = n['q'][:, il[0]]-n['q'][:, ir[0]]
    yaw = t['neural_yaw_unapplied_rad_s']*180./np.pi
    total_drive = n['JO_drive'].sum(axis=1)
    old = reconstruct_air(n['air_kinematics'], n['JO_rows'], 1.)
    corrected = reconstruct_air(n['air_kinematics'], n['JO_rows'], 10.)
    corrected_arrays[name] = corrected
    group_totals = {}
    for side in ['L', 'R']:
        for family in ['JO-C', 'JO-E']:
            mask = (anatomy['source_side'] == side) & np.char.startswith(anatomy['source_type'], family)
            group_totals[family+'/'+side] = float(n['JO_drive'][window][:, mask].sum(axis=1).mean())
    arms[name] = {
        'mean_q': float(dq[window].mean()), 'mean_yaw_deg_s': float(yaw[window].mean()),
        'max_abs_applied_yaw_deg_s': float(np.max(np.abs(t['command_yaw_rate_rad_s']))*180./np.pi),
        'max_abs_forward_command_mm_s': float(np.max(np.abs(t['command_forward_mm_s']))),
        'max_abs_unclipped_forward_mm_s': float(np.max(np.abs(t['forward_unclipped_mm_s']))),
        'displacement_mm_between_samples1_90': (t['position_mm'][-1]-t['position_mm'][0]).tolist(),
        'max_body_speed_mm_s': float(np.max(np.linalg.norm(t['qvel'][:, :3]*10., axis=1))),
        'JO_mean_total_input_window': float(total_drive[window].mean()),
        'JO_integrated_input_11_90_model_units_s': float(total_drive[10:90].sum()*.001),
        'JO_mean_active_neurons_window': float(np.count_nonzero(n['JO_drive'][window], axis=1).mean()),
        'JO_group_mean_totals': group_totals,
        'position_mm_equals_10_qpos_exact': bool(np.array_equal(t['position_mm'], 10.*t['qpos'][:, :3])),
        'reconstructed_as_run_JO_max_abs_error': float(np.max(np.abs(old-n['JO_drive']))),
        'velocity_units_corrected_JO_max_abs_change': float(np.max(np.abs(corrected-n['JO_drive']))),
        'velocity_units_corrected_JO_max_total_change': float(np.max(np.abs(corrected.sum(axis=1)-total_drive))),
        'velocity_units_corrected_JO_mean_total_window': float(corrected[window].sum(axis=1).mean()),
    }
    raw[name] = (n, t)

def contrast(values):
    return {metric: {'value': float(value), 'threshold': limits[metric],
                     'threshold_fraction': float(abs(value)/limits[metric]),
                     'passes_magnitude': bool(abs(value) >= limits[metric])}
            for metric, value in values.items()}

air = {}
for odor in [0, 1]:
    left, right = f'airL_odor{odor}', f'airR_odor{odor}'
    if left in arms and right in arms:
        air[str(odor)] = contrast({k: (arms[left]['mean_'+k]-arms[right]['mean_'+k])/2.
                                  for k in limits})
        air[str(odor)]['both_magnitude_gates'] = all(air[str(odor)][k]['passes_magnitude'] for k in limits)
        totals = [arms[a]['JO_mean_total_input_window'] for a in [left, right]]
        air[str(odor)]['input_total_comparison'] = dict(left=totals[0], right=totals[1],
            right_minus_left=totals[1]-totals[0], right_to_left_ratio=totals[1]/totals[0],
            exact_equal=bool(np.array_equal(raw[left][0]['JO_drive'].sum(axis=1), raw[right][0]['JO_drive'].sum(axis=1))))

interaction = None
if len(air) == 2:
    interaction = contrast({k: air['1'][k]['value']-air['0'][k]['value'] for k in limits})
    # The plan reports this interaction, but does not register a separate promotion threshold for it.
    for item in interaction.values():
        item['comparison_only_not_separate_registered_gate'] = True
conductance = None
odor_effects = {}
for name in ['G', 'I', 'air0']:
    if all(name+suffix in arms for suffix in ['_odor0', '_odor1']):
        odor_effects[name] = {k: arms[name+'_odor1']['mean_'+k]-arms[name+'_odor0']['mean_'+k] for k in limits}
if 'G' in odor_effects and 'I' in odor_effects:
    conductance = contrast({k: odor_effects['G'][k]-odor_effects['I'][k] for k in limits})
    conductance['both_magnitude_gates'] = all(conductance[k]['passes_magnitude'] for k in limits)
    conductance['vs_parent_odor_effect'] = {
        mode: {k: odor_effects[mode][k]-odor_effects['air0'][k] for k in limits}
        for mode in ['G', 'I'] if 'air0' in odor_effects}
reference = raw.get('air0_odor0')
body_equality = {}
if reference is not None:
    for name, (_, trace) in raw.items():
        body_equality[name] = {k: bool(np.array_equal(trace[k], reference[1][k])) for k in
                              ['qpos','qvel','position_mm','upright','contact_active','sensores_usados']}
geometry = {}
for name, (n, trace) in raw.items():
    k = n['air_kinematics']; rotations = k[:, 6:].reshape(-1, 3, 3)
    w, x, y, z = trace['qpos'][:-1, 3:7].T
    expected = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]).transpose(2,0,1)
    geometry[name] = dict(
        native_velocity_matches_previous_qvel_exact=bool(np.array_equal(k[1:,3:6], trace['qvel'][:-1,:3])),
        rotation_matches_previous_quaternion_max_error=float(np.max(np.abs(rotations[1:]-expected))),
        rotation_orthonormality_max_error=float(np.max(np.abs(np.einsum('tji,tjk->tik',rotations,rotations)-np.eye(3)))),
        world_field_constant_exact=bool(np.array_equal(k[:,:3],np.broadcast_to(k[0,:3],(90,3)))))
    need(geometry[name]['native_velocity_matches_previous_qvel_exact'], name+' air velocity sampling')
    need(geometry[name]['rotation_matches_previous_quaternion_max_error'] < 1e-10, name+' air rotation sampling')
    need(geometry[name]['world_field_constant_exact'], name+' changed world field')
executed_sources = json.loads((ROOT/'air0_odor0/EXECUTED_SOURCES.json').read_text())
unit_sources = {}
for source, digest in executed_sources.items():
    if source.endswith(('/matrix_olfactory_diagnostic.py','/flybody_cns_body.py')):
        need(sha(Path(source)) == digest, 'Executed unit source changed')
        unit_sources[source] = digest
np.savez_compressed(H/'ENTRADAS_CORREGIDAS_NO_SIMULADAS.npz',JO_ids=anatomy['source_ids'],**corrected_arrays)
out = dict(status='COMPLETE' if len(arms) == 10 else 'INCOMPLETE',
    queue_status=queue.get('status'), completed_arms=len(arms), missing=missing,
    independence='Raw NPZ computation without analyze_pilot.py or its results; not blind to design/history',
    window_samples_inclusive=[51,90], neural_quantity='q(bodyId10118)-q(bodyId10065)',
    yaw_quantity='recorded neural_yaw_unapplied_rad_s converted to degrees/s',
    air_formula='(mean left-field arm minus mean right-field arm)/2, separately by odor',
    conductance_formula='(G_odor1-G_odor0)-(I_odor1-I_odor0)',
    arms=arms, air=air, air_odor_interaction=interaction, odor_effects=odor_effects,
    conductance=conductance, body_exact_equal_to_air0_odor0=body_equality,
    geometric_equivalence=geometry, executed_unit_source_hashes=unit_sources,
    units_issue='AirOwner passed native qvel as mm/s; trace position_mm=10*qpos. Corrected air arrays here are offline counterfactual inputs, NOT corrected CNS outcomes.',
    applied_motion='Measured separately; raw yaw is not applied yaw',
    stage4=False, stage5=False, new_CNS_steps=0, new_GPU_calls=0,
    completed_UTC=datetime.now(timezone.utc).isoformat(), input_hashes=hashes,
    CPU_s=time.process_time()-start, process_CPU_total_s=time.process_time(),
    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(H/'RESULTADO.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:out[k] for k in ['status','completed_arms','air','air_odor_interaction','conductance','CPU_s','process_CPU_total_s']},indent=2))
