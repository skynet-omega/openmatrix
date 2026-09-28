"""Independent CPU-only checks of saved campaign52 data. Never runs the CNS.

Panel test uses saved51 releases, not a new biological experiment. Qualification
checks all common saved arrays and reported owner hashes; it cannot reconstruct
unsaved full states from hashes. Science mode recomputes transfer, not navigation.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import runpy
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent
NODE_IDS = HERE / 'donors/node_ids.npy'
NODE_IDS_SHA256 = 'ab90597b7b0ce07cbc22cb39a65b70bea2ac73cc7fd225951bd9d73f2fc8dd3f'
OWNER_KEYS = {'session', 'prosthesis', 'published', 'stored_operator',
              'effective_operator', 'input_owner', 'interval_owner',
              'motor_owner', 'boundary', 'event_audit', 'air_owner'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k].copy() for k in z.files}
    for k, v in out.items():
        need(not v.dtype.hasobject, 'Object array: ' + k)
        if v.dtype.kind in 'fc':
            need(np.isfinite(v).all(), 'Nonfinite array: ' + k)
    return out


def same(a, b, label):
    need(a.dtype == b.dtype and a.shape == b.shape
         and a.tobytes() == b.tobytes(), 'Exact typed array mismatch: ' + label)


def owners(data, mode):
    expected = OWNER_KEYS | ({'candidate_owner'} if mode != 'parent' else set())
    need(set(data) == expected, 'Scientific-owner set differs')
    need(all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)
             for v in data.values()), 'Invalid owner digest')


def rotation(quat):
    # MuJoCo quaternion order w,x,y,z; unit quaternion from saved native pose.
    w, x, y, z = quat
    return np.array([[w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z]])


def physical_input(a, t, anatomy, prefix, air_sign):
    k = a['air_kinematics']
    n = len(t['CNS_time_ns'])
    need(k.shape == (n, 15), 'Air record dimensions')
    previous_v = np.vstack([a['initial_native_velocity'], t['qvel'][:-1, :3]])
    same(k[:, 3:6], 10. * previous_v, 'cm/s to mm/s, preceding body boundary')
    same(t['position_mm'], 10. * t['qpos'][:, :3], 'Body position mm')
    preceding_q = np.vstack([a['initial_qpos'][3:7], t['qpos'][:-1, 3:7]])
    rot = np.array([rotation(q) for q in preceding_q])
    # Independent quaternion arithmetic can reorder floating point operations.
    need(np.allclose(rot, k[:, 6:].reshape(-1, 3, 3), rtol=0, atol=1e-12),
         'Air rotation belongs to preceding body boundary')
    field = rotation(a['initial_qpos'][3:7]) @ np.array([0., 100. * air_sign, 0.])
    need(np.allclose(k[:, :3], field, rtol=0, atol=1e-11), 'Fixed world air field')
    same(a['JO_ids'], anatomy['source_ids'], 'JO IDs')
    same(a['JO_rows'], anatomy['source_rows'], 'JO rows')
    axes = np.array([[1., 1., 0.], [1., -1., 0.]]) / np.sqrt(2.)
    side = (anatomy['source_side'] == 'R').astype(np.int64)
    sign = np.where(np.char.startswith(anatomy['source_type'], 'JO-C'), 1., -1.)
    expected = np.zeros_like(a['JO_drive'])
    for j in range(prefix, n):
        relative = k[j, 6:].reshape(3, 3).T @ (k[j, :3] - k[j, 3:6])
        speed = np.maximum((axes @ relative)[side] * sign, 0.)
        expected[j] = 80. * (speed / (speed + 100.))
    same(a['JO_drive'], expected, 'Independent JO physical-input reconstruction')
    need(np.all(t['command_yaw_rate_rad_s'] == 0), 'Unexpected applied yaw')


def panel_values(w, a, t, panel, nodes):
    n = len(t['CNS_time_ns'])
    same(w['ids'], panel['ids'], 'Wide identity')
    same(w['rows'], panel['canonical_rows'], 'Wide canonical rows')
    same(nodes[w['rows']], w['ids'], 'Wide canonical identity')
    need(len(np.unique(w['ids'])) == 2757, 'Unique wide identities')
    need(w['q'].shape == (n, 2757), 'Wide shape')
    same(w['time_ns'], t['CNS_time_ns'], 'Wide committed time')
    same(w['q'][-1], a['final_q'][w['rows']], 'Wide final release')
    selected = np.searchsorted(w['ids'], a['ids'])
    same(w['ids'][selected], a['ids'], 'Legacy IDs within wide panel')
    same(w['q'][:, selected], a['q'], 'Legacy time course')
    same(w['ORN_ids'], a['ORN_ids'], 'ORN identities')
    rows = np.searchsorted(nodes, w['ORN_ids'])
    same(nodes[rows], w['ORN_ids'], 'ORN canonical identity')
    same(w['ORN_rows'], rows, 'ORN saved canonical rows')
    need(w['ORN_q'].shape == (n, 694), 'ORN dimensions')
    same(w['ORN_q'][-1], a['final_q'][rows], 'ORN final release')


def panel_test(root, nodes):
    source = root.parent / 'etapa45_alternativas_20260927_51/air0_odor0/neural_and_inputs.npz'
    a = arrays(source)
    module = runpy.run_path(str(root / 'observer52.py'))
    cls = module['Panel']
    frozen = a['final_q'].copy()
    frozen.setflags(write=False)
    before = frozen.tobytes()
    wide = cls(nodes, True, a['ORN_ids'])
    narrow = cls(nodes, False, a['ORN_ids'])
    wide.record(frozen, 17)
    narrow.record(frozen, 17)
    need(frozen.tobytes() == before, 'Observer changed source')
    need(not np.shares_memory(wide.q[0], frozen)
         and not np.shares_memory(wide.orn_q[0], frozen), 'Observer aliases source')
    need(not narrow.q and not narrow.orn_q and not narrow.time_ns, 'OFF observer recorded')
    wide.q[0].fill(0)
    wide.orn_q[0].fill(0)
    need(frozen.tobytes() == before, 'Diagnostic-buffer mutation changed input')
    mutable = a['final_q'].copy()
    wide.record(mutable, 18)
    held = wide.q[1].copy()
    held_orn = wide.orn_q[1].copy()
    mutable.fill(0)
    same(wide.q[1], held, 'Retained wide sample')
    same(wide.orn_q[1], held_orn, 'Retained ORN sample')
    need(not any(k.split('.')[0] in {'cupy', 'mujoco', 'resume49'} for k in sys.modules),
         'Forbidden CNS/GPU/body import')
    # Negative controls test the verifier, not the organism.
    rejected = []
    for label, call in [
        ('missing_all_owners', lambda: owners({}, 'parent')),
        ('same_values_wrong_dtype', lambda: same(np.zeros(2, np.float32),
                                                np.zeros(2, np.float64), 'dtype')),
        ('wrong_anatomical_row', lambda: cls(nodes[::-1], True, a['ORN_ids'])),
    ]:
        try:
            call()
        except (ValueError, IndexError):
            rejected.append(label)
        else:
            raise ValueError('Corruption not detected: ' + label)
    return dict(status='PASS_CPU_ONLY', recorded51_source_sha256=sha(source),
                observer_source_sha256=sha(root / 'observer52.py'),
                source_unchanged=True, arrays_do_not_alias=True,
                corruption_controls=rejected, CNS_ms=0,
                scope='Panel copying only; live ON/OFF qualification still required')


def qualification(root, nodes):
    plan = js(root / 'PILOT_PLAN.json')
    freeze = js(root / 'PILOT_FREEZE.json')
    need(sha(root / 'PILOT_PLAN.json') == freeze['plan_sha256'], 'Plan changed')
    panel = arrays(root / 'PANEL.npz')
    anatomy = arrays(root / 'donors/JO_anatomy_arrays.npz')
    counts = {}
    for name, spec in plan['qualifications'].items():
        d = root / ('qual_' + name)
        result = js(d / 'RESULT.json')
        need(result['status'] == 'COMPLETE' and result['initial_exact'], 'Incomplete qualification')
        need(result['attempted_ms'] == result['committed_ms'] == spec['duration_ms'], 'Duration')
        a, t = arrays(d / 'neural_and_inputs.npz'), arrays(d / 'traces.npz')
        physical_input(a, t, anatomy, spec['prefix_ms'], spec['air'])
        expected_clock = 47486000000 + np.arange(1, spec['duration_ms'] + 1) * 1000000
        same(t['CNS_time_ns'], expected_clock, 'Qualification clock')
        if spec['wide']:
            panel_values(arrays(d / 'wide_observation.npz'), a, t, panel, nodes)
        else:
            need(not (d / 'wide_observation.npz').exists(), 'Wide output in OFF arm')
        owners(js(d / 'OWNERS_FINAL.json'), spec['mode'])
        counts[name] = spec['duration_ms']
    ref, observed = arrays(root / 'identity49_reference.npz'), arrays(root / 'qual_identity/traces.npz')
    need(ref.keys() == observed.keys(), 'Identity trace fields')
    for k in ref:
        same(ref[k], observed[k], 'Identity49/' + k)
    for mode in ['parent', 'conductance', 'current']:
        l, r = [root / ('qual_' + mode + '_' + suffix) for suffix in ['narrow', 'wide']]
        need(js(l / 'OWNERS_FINAL.json') == js(r / 'OWNERS_FINAL.json'), 'ON/OFF owner hashes')
        need(js(l / 'EVENTS.json') == js(r / 'EVENTS.json'), 'ON/OFF event audit sequence')
        for filename in ['traces.npz', 'neural_and_inputs.npz', 'dng100_observed.npz']:
            a, b = arrays(l / filename), arrays(r / filename)
            need(a.keys() == b.keys(), 'ON/OFF schema')
            for k in a:
                same(a[k], b[k], mode + '/' + filename + '/' + k)
    need(sum(counts.values()) == 16, 'Qualification CNS budget')
    return dict(status='PASS_SAVED_QUALIFICATION', CNS_ms_by_arm=counts,
                exact_ON_OFF_common_arrays=True, reported_owner_hashes_equal=True,
                event_audit_sequences_equal=True,
                limits='Owner hashes compared, unsaved full owner states not reconstructed; inherited DNg observer common')


def science(root, nodes):
    qual = qualification(root, nodes)
    plan = js(root / 'PILOT_PLAN.json')
    panel = arrays(root / 'PANEL.npz')
    anatomy = arrays(root / 'donors/JO_anatomy_arrays.npz')
    metrics, wide = {}, {}
    start, end = plan['analysis_window_ms']
    window = slice(start - 1, end)
    for name, spec in plan['arms'].items():
        d = root / name
        r = js(d / 'RESULT.json')
        need(r['status'] == 'COMPLETE' and r['initial_exact'], 'Incomplete scientific arm')
        need(r['attempted_ms'] == r['committed_ms'] == plan['duration_ms'], 'Arm duration')
        a, t, w = (arrays(d / f) for f in ['neural_and_inputs.npz', 'traces.npz', 'wide_observation.npz'])
        physical_input(a, t, anatomy, plan['prefix_ms'], spec['air'])
        panel_values(w, a, t, panel, nodes)
        indices = [int(np.flatnonzero(a['ids'] == v)[0]) for v in [10118, 10065]]
        delta = a['q'][:, indices[0]] - a['q'][:, indices[1]]
        yaw = np.rad2deg(t['neural_yaw_unapplied_rad_s'])
        metrics[name] = dict(mean_DNb05_L_minus_R=float(delta[window].mean()),
                             mean_unapplied_yaw_deg_s=float(yaw[window].mean()),
                             mean_total_JO_drive=float(a['JO_drive'][window].sum(axis=1).mean()),
                             max_forward_command_mm_s=float(t['command_forward_mm_s'].max()),
                             observed_displacement_1_to_90_mm=float(np.linalg.norm(t['position_mm'][-1]-t['position_mm'][0])),
                             applied_yaw_is_zero=True)
        wide[name] = w['q']
    contrasts = {}
    for field in ['mean_DNb05_L_minus_R', 'mean_unapplied_yaw_deg_s']:
        contrasts[field] = {
            'air_no_odor': .5 * (metrics['airL_odor0'][field] - metrics['airR_odor0'][field]),
            'air_with_odor': .5 * (metrics['airL_odor1'][field] - metrics['airR_odor1'][field]),
            'G_minus_I_odor_interaction': metrics['G_odor1'][field] - metrics['G_odor0'][field]
                                        - metrics['I_odor1'][field] + metrics['I_odor0'][field]}
    group_effects = {}
    threshold = plan['observation']['descriptive_threshold_q']
    for suffix in ['0', '1']:
        delta = wide['airL_odor' + suffix] - wide['airR_odor' + suffix]
        group_effects['odor' + suffix] = {}
        for name, mask in zip(panel['group_names'], panel['group_masks']):
            material = np.abs(delta[:, mask]) > threshold
            group_effects['odor' + suffix][str(name)] = dict(
                population=int(mask.sum()), final_count=int(material[-1].sum()),
                ever_count=int(material.any(axis=0).sum()),
                max_abs_q_difference=float(np.abs(delta[:, mask]).max()))
    return dict(status='RECOMPUTED_SAVED_SCIENCE', qualification=qual, arms=metrics,
                contrasts=contrasts, descriptive_group_effects=group_effects,
                stage4_passed=False, stage5_passed=False,
                limits='Unequal natural JO doses remain. No applied yaw or physical wind recovery in this design.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['panel-test', 'qualification', 'science'])
    ap.add_argument('--root', type=Path, default=DEFAULT)
    ap.add_argument('--node-ids', type=Path, default=NODE_IDS,
                    help='Canonical IDs; content must match the reviewed male_v10 mapping')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (90, 92))
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    need(args.out.parent.resolve() == HERE and not args.out.exists(), 'Fresh output in aporte_motor52 required')
    start = time.process_time()
    need(sha(args.node_ids) == NODE_IDS_SHA256, 'Canonical node mapping changed')
    nodes = np.load(args.node_ids, allow_pickle=False)
    action = {'panel-test': panel_test, 'qualification': qualification, 'science': science}[args.mode]
    out = action(args.root, nodes)
    out.update(CPU_s=time.process_time()-start,
               peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               analysis_only_no_CNS=True, verifier_sha256=sha(__file__),
               node_ids_sha256=NODE_IDS_SHA256)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
