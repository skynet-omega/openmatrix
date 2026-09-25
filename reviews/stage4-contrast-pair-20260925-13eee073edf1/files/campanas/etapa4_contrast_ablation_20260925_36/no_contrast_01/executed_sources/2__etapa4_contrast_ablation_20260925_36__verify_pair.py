"""Rebuild replay integrity and its restricted causal contrast from raw arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = HERE.parent / 'etapa4_long_trajectory_20260925_35'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_trace(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def compare_preparation(left, right):
    # New experiment/runner profile intentionally differs. Every scientific
    # section, scalar and array is still compared, including RNG and operator.
    sys.path.insert(0, str(ROOT / 'motor_nuevo/gaussian_prepared_semantics_20260923'))
    from compare_prepared import validate_manifest, metadata, compare_arrays, REQUIRED
    left, right = Path(left), Path(right)
    li, ri = validate_manifest(left), validate_manifest(right)
    need(li['inventory'] == ri['inventory'], 'Prepared state inventory differs')
    expected = {s + ext for s in REQUIRED for ext in ('.json', '.npz')} | {'boundary.json'}
    need(expected <= set(li['inventory']), 'Missing prepared scientific section')
    differences, arrays = [], 0
    for name in li['inventory']:
        if name.endswith('.npz'):
            need(Path(name).stem + '.json' in li['inventory'], 'Unbound state array')
            continue
        if not name.endswith('.json'):
            need(sha(left / name) == sha(right / name), 'Unknown prepared file differs')
            continue
        a, b = metadata(left / name), metadata(right / name)
        arrname = Path(name).stem + '.npz'
        rows = compare_arrays(left / arrname, right / arrname, a['arrays'], b['arrays']) if arrname in li['inventory'] else []
        arrays += len(rows)
        if a['semantic_sha256'] != b['semantic_sha256'] or any(not r['exact'] for r in rows):
            differences.append(name)
    return dict(scientific_state_exact=not differences, different_sections=differences,
                arrays_compared=arrays, left_manifest_sha256=li['manifest_sha256'],
                right_manifest_sha256=ri['manifest_sha256'],
                scope='All prepared scientific state; new experiment profile differs intentionally; no resume claim')


def bearing(q, source):
    w, x, y, z = q[:, 3:7].T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    direction = np.arctan2(source[1] - 10*q[:, 1], source[0] - 10*q[:, 0])
    delta = direction - yaw
    return np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta)))


def verify_arrays(z, donor, mode, source, sigma):
    need(mode in ('identity', 'no_contrast'), 'Unknown mode')
    need(set(z) == set(donor) | {'concentracion_fisica'}, 'Missing or unexpected trace columns')
    for k, v in z.items():
        need(v.shape[0] == 1040, 'Wrong row count: ' + k)
        if v.dtype.kind != 'U':
            need(v.dtype.kind in 'bfiu' and np.isfinite(v).all(), 'Invalid numeric trace: ' + k)
        if k in donor:
            need(v.shape == donor[k].shape and v.dtype == donor[k].dtype, 'Shape/dtype differs: ' + k)
            need(np.array_equal(v[:40], donor[k][:40]), 'Preparation trace differs: ' + k)
    need(z['fase'].tolist() == ['preparacion']*40 + ['ensayo']*1000, 'Phase order')
    need(z['paso'].tolist() == list(range(1,41)) + list(range(1,1001)), 'Step order')
    for k in ('CNS_time_ns', 'PN_time_ns', 'body_time_ns'):
        need(np.array_equal(z[k], donor['CNS_time_ns']), 'Clock differs: ' + k)
        need(np.all(np.diff(z[k]) == 1_000_000), 'Clock increment')
    need(z['sensores_usados'].shape == z['sensores_pendientes'].shape == (1040,3), 'Sensor shape')
    need(z['antenas_mm'].shape == (1040,2,3) and z['concentracion_fisica'].shape == (1040,2), 'Antenna/physical field shape')
    tape = np.vstack((donor['sensores_usados'][40:], donor['sensores_pendientes'][-1:])).copy()
    need(np.array_equal(donor['sensores_usados'][41:], donor['sensores_pendientes'][40:-1]), 'Donor lag')
    if mode == 'no_contrast':
        common = (tape[:,0] + tape[:,1]) / 2
        tape[:,0], tape[:,1] = common, common
    need(np.array_equal(z['sensores_usados'][40:], tape[:-1]), 'Consumed tape differs')
    need(np.array_equal(z['sensores_pendientes'][40:], tape[1:]), 'Pending tape differs')
    need(np.array_equal(z['concentracion_campo'][40:], tape[1:,:2]), 'Delivered boundary differs')
    need(np.array_equal(z['sensores_usados'][41:], z['sensores_pendientes'][40:-1]), 'One-step lag broken')
    need(np.array_equal(z['DN_q_usada'][40:], z['DN_q_actual'][39:-1]), 'Neural reader lag')
    need(np.array_equal(z['DN_baseline'], donor['DN_baseline']), 'Baseline changed')
    q, b = z['DN_q_usada'][40:], z['DN_baseline'][40:]
    cmd = np.tanh(250 * ((q[:,2]-b[:,2]) - (q[:,3]-b[:,3]))) * np.deg2rad(5)
    need(np.max(abs(cmd-z['command_yaw_rate_rad_s'][40:])) <= 1e-12, 'Reader equation changed')
    need(np.array_equal(z['command_forward_mm_s'], donor['command_forward_mm_s']), 'Forward command changed')
    need(np.max(abs(np.sum(z['qpos'][:,3:7]**2,axis=1)-1)) <= 1e-8, 'Quaternion norm')
    need(np.array_equal(z['position_mm'], 10*z['qpos'][:,:3]), 'Physical position differs from qpos')
    expected = np.exp(-np.sum((z['antenas_mm'][40:,:,:2]-source)**2,axis=2)/(2*sigma*sigma))
    need(np.max(abs(expected-z['concentracion_fisica'][40:])) <= 1e-12, 'Physical Gaussian differs')
    identity = all(np.array_equal(z[k], donor[k]) for k in donor)
    if mode == 'identity':
        need(identity, 'Identity replay did not reproduce every donor trace column')
    beta = bearing(z['qpos'], source)
    err = abs(beta)
    distance = np.linalg.norm(z['position_mm'][:,:2]-source,axis=1)
    cmddeg = np.rad2deg(z['command_yaw_rate_rad_s'][40:])
    upright = float(z['upright'][40:].min())
    contacts = int(z['contact_active'][40:].sum(axis=1).min())
    need(upright >= .95 and contacts >= 4, 'Lost physical support')
    return dict(identity_exact=identity, final_bearing_error_deg=float(err[-1]),
                initial_bearing_error_deg=float(err[39]),
                bearing_error_integral_deg_s=float(np.sum((err[39:-1]+err[40:])*.0005)),
                final_distance_mm=float(distance[-1]), initial_distance_mm=float(distance[39]),
                command_integral_deg=float(cmddeg.sum()*.001),
                absolute_command_integral_deg=float(abs(cmddeg).sum()*.001),
                upright_min=upright, contacts_min=contacts)


def verify_run(path, mode):
    path = Path(path)
    plan = json.loads((HERE/'PLAN.json').read_text())
    donorpath = PRIOR/'native_minus_02/traces.npz'
    need(sha(donorpath) == plan['donor_trace_sha256'], 'Donor changed')
    result = json.loads((path/'RESULT.json').read_text())
    need(result['status'] == 'COMPLETE' and result['cleanup_errors'] == [], 'Incomplete run')
    need(result['mode'] == mode and result['engine'] == 'causal_cuda', 'Wrong mode/engine')
    contract = json.loads((path/'RUN_CONTRACT.json').read_text())
    need(contract['plan_sha256'] == sha(HERE/'PLAN.json'), 'Wrong frozen plan')
    need(result['completed_preparation_ms'] == 40 and result['completed_trial_ms'] == 1000, 'Incomplete duration')
    frozen = json.loads((path/'FROZEN.json').read_text())
    for name, digest in frozen.items():
        need(sha(path/'executed_sources'/name) == digest, 'Executed source changed')
    need(any(name.endswith('__run_replay.py') and digest == sha(HERE/'run_replay.py')
             for name,digest in frozen.items()), 'Wrong runner')
    prepared = compare_preparation(path/'prepared_state', PRIOR/'native_minus_02/prepared_state')
    need(prepared['scientific_state_exact'], 'Prepared scientific state differs')
    spec = json.loads((PRIOR/'CAMPOS.json').read_text())['minus']
    m = verify_arrays(load_trace(path/'traces.npz'), load_trace(donorpath), mode,
                      np.asarray(spec['source_mm']), spec['sigma_mm'])
    return dict(m, mode=mode, trace_sha256=sha(path/'traces.npz'),
                prepared_state=prepared, plan_sha256=sha(HERE/'PLAN.json'),
                verifier_sha256=sha(Path(__file__)), stage4_admission=False, stage5_admission=False)


def pair(left, right):
    a, b = verify_run(left,'identity'), verify_run(right,'no_contrast')
    delta = b['final_bearing_error_deg'] - a['final_bearing_error_deg']
    material = json.loads((HERE/'PLAN.json').read_text())['diagnostic']['effect_size_screen_deg']
    return dict(schema='stage4_contrast_diagnostic_v1', control=a, intervention=b,
                delta_error_noD_minus_D_deg=delta,
                delta_error_integral_deg_s=b['bearing_error_integral_deg_s']-a['bearing_error_integral_deg_s'],
                delta_distance_noD_minus_D_mm=b['final_distance_mm']-a['final_distance_mm'],
                effect_screen=('D_HELPS_IN_THIS_DISCRETIZED_MODEL' if delta>=material else
                               'D_WORSENS_IN_THIS_DISCRETIZED_MODEL' if delta<=-material else
                               'BELOW_MATERIAL_DIAGNOSTIC_SCREEN'),
                classification='PROMETEDOR_NO_CONFIRMADO',
                numerical_limit='No independent refinement reference for last 600 ms or ablation; no certified numerical resolution',
                stage4_admission=False, stage5_admission=False,
                scope='Causal contribution of fixed donor contrast in this model; replay disables spatial feedback')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--no-contrast',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    r=pair(args.identity,args.no_contrast) if args.no_contrast else verify_run(args.identity,'identity')
    with args.out.open('x') as f:
        json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(r,ensure_ascii=False,allow_nan=False))
