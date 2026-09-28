"""Prospective live-pair qualification, recomputable from saved outputs."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

H = Path(__file__).resolve().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def check_units(a, t, prefix):
    import sys
    sys.path.insert(0, str(H / 'donors'))
    from air_interface import AirInterface
    api = AirInterface(H / 'donors/JO_anatomy_arrays.npz')
    k = a['air_kinematics']
    need(np.array_equal(k[0, 3:6], 10 * a['initial_native_velocity']), 'Initial cm/s to mm/s')
    need(np.array_equal(k[1:, 3:6], 10 * t['qvel'][:-1, :3]), 'Consumed physical velocity and latency')
    need(np.array_equal(t['position_mm'], 10 * t['qpos'][:, :3]), 'Native geometry scale')
    expected = np.zeros_like(a['JO_drive'])
    for j in range(prefix, len(k)):
        expected[j] = api.encode(air_velocity_world_mm_s=k[j, :3],
                                body_velocity_world_mm_s=k[j, 3:6],
                                body_to_world=k[j, 6:].reshape(3, 3))
    need(np.array_equal(expected, a['JO_drive']), 'JO replay from corrected physical inputs')
    need(np.all(t['command_yaw_rate_rad_s'] == 0), 'No applied yaw')


def equations(d, mode, initial):
    z = arrays(d / 'candidate_owner.npz')
    a = arrays(d / 'dng100_observed.npz')
    rows = a['rows']
    r = a['records'][:, :128].reshape(-1, 4, 2, 16)
    need(np.array_equal(z['q0'], initial), 'Candidate restored internal state')
    S, g0, q0 = (z[key][rows] for key in ['S', 'gE0', 'q0'])
    X = r[:, :, :, 2] + np.maximum(r[:, :, :, 4], 0)
    Y = -r[:, :, :, 3] + r[:, :, :, 5] + np.maximum(-r[:, :, :, 4], 0)
    need(np.array_equal(S, X[0, 0] + Y[0, 0]), 'Candidate scales from first RHS')
    need(np.array_equal(g0, X[0, 0] / S), 'Candidate initial fraction')
    E, I = X / S, Y / S
    EL = 2 * q0 - g0
    total = 1 + E + I
    if mode == 'conductance':
        target = np.clip((EL + E) / total, 0, 1)
        rate = total / (2 * r[:, :, :, 9])
    else:
        target = np.clip((EL + E * (1 - q0) - I * q0 + q0) / 2, 0, 1)
        rate = 1 / r[:, :, :, 9]
    need(np.array_equal(target, r[:, :, :, 7]), 'CPU/GPU target exact')
    need(np.array_equal(rate, r[:, :, :, 8]), 'CPU/GPU rate exact')
    need(np.array_equal(target, r[:, :, :, 11]), 'Final target override')
    need(np.array_equal(rate, r[:, :, :, 12]), 'Final rate override')
    return int(np.prod(r.shape[:3]))


def verify(root=H):
    global H
    H = Path(root)
    plan = json.loads((H / 'PILOT_PLAN.json').read_text())
    freeze = json.loads((H / 'PILOT_FREEZE.json').read_text())
    need(sha(H / 'PILOT_PLAN.json') == freeze['plan_sha256'], 'Contract changed')
    panel = arrays(H / 'PANEL.npz')
    initial = np.load(H / 'native_initial_q.npy', allow_pickle=False)
    report, used = {}, 0
    for name, spec in plan['qualifications'].items():
        d = H / ('qual_' + name)
        result = json.loads((d / 'RESULT.json').read_text())
        need(result['status'] == 'COMPLETE' and result['initial_exact'], 'Restoration or execution failed')
        n = spec['duration_ms']
        need(result['attempted_ms'] == result['committed_ms'] == n, 'Qualification duration')
        need(result['setting'] == {k: spec[k] for k in ['mode', 'air', 'odor']}, 'Qualification setting')
        used += result['attempted_ms']
        a, t = arrays(d / 'neural_and_inputs.npz'), arrays(d / 'traces.npz')
        check_units(a, t, spec['prefix_ms'])
        need(np.array_equal(t['CNS_time_ns'], 47486000000 + np.arange(1, n + 1) * 1000000), 'Qualification clock')
        need(all(np.isfinite(v).all() for v in a.values()), 'Nonfinite qualification array')
        if spec['wide']:
            w = arrays(d / 'wide_observation.npz')
            need(np.array_equal(w['ids'], panel['ids']), 'Wide identities')
            need(np.array_equal(w['rows'], panel['canonical_rows']), 'Wide rows')
            need(w['q'].shape == (n, 2757) and w['ORN_q'].shape == (n, 694), 'Wide dimensions')
            need(np.isfinite(w['q']).all() and np.isfinite(w['ORN_q']).all(), 'Nonfinite wide observation')
            need(np.array_equal(w['ORN_ids'], a['ORN_ids']), 'ORN identity')
            need(np.array_equal(w['ORN_rows'], panel['ORN_rows']), 'ORN canonical rows')
            need(np.array_equal(w['ORN_q'][-1], a['final_q'][w['ORN_rows']]), 'ORN endpoint')
            need(np.array_equal(w['time_ns'], t['CNS_time_ns']), 'Wide timestamp')
            need(np.array_equal(w['q'][-1], a['final_q'][w['rows']]), 'Wide endpoint')
            legacy_rows = np.searchsorted(w['ids'], a['ids'])
            need(np.array_equal(w['ids'][legacy_rows], a['ids']), 'Legacy identities present')
            need(np.array_equal(w['q'][:, legacy_rows], a['q']), 'Legacy trace copied exactly')
        else:
            need(not (d / 'wide_observation.npz').exists(), 'Narrow control recorded wide panel')
        rhs = equations(d, spec['mode'], initial) if spec['mode'] != 'parent' else 0
        report[name] = dict(corrected_units=True, restored_exact=True, equation_cells=rhs)
    ref, observed = arrays(H / 'identity49_reference.npz'), arrays(H / 'qual_identity/traces.npz')
    need(ref.keys() == observed.keys(), 'Identity schema')
    need(all(np.array_equal(ref[k], observed[k]) for k in ref), 'Identity33fields versus49')
    pairs = {}
    for mode in ['parent', 'conductance', 'current']:
        left, right = (H / ('qual_' + mode + suffix) for suffix in ['_narrow', '_wide'])
        l = json.loads((left / 'OWNERS_FINAL.json').read_text())
        r = json.loads((right / 'OWNERS_FINAL.json').read_text())
        required = {'session', 'prosthesis', 'published', 'effective_operator', 'stored_operator',
                    'input_owner', 'interval_owner', 'motor_owner', 'boundary', 'event_audit', 'air_owner'}
        if mode != 'parent':
            required.add('candidate_owner')
        need(set(l) == set(r) == required, 'Missing scientific owner')
        need(all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in list(l.values()) + list(r.values())), 'Malformed owner hash')
        need(l == r, 'Wide observer changed scientific owner: ' + mode)
        fields = 0
        for filename in ['traces.npz', 'neural_and_inputs.npz', 'dng100_observed.npz']:
            a, b = arrays(left / filename), arrays(right / filename)
            need(a.keys() == b.keys(), 'Pair schema mismatch')
            need(all(np.array_equal(a[k], b[k]) for k in a), 'Pair arrays differ: ' + mode + '/' + filename)
            fields += len(a)
        need(json.loads((left / 'EVENTS.json').read_text()) == json.loads((right / 'EVENTS.json').read_text()), 'Event sequence differs')
        pairs[mode] = dict(owners=list(l), owner_hashes_exact=True, array_fields_exact=fields, event_sequence_exact=True)
    g, i = (arrays(H / ('qual_' + mode + '_wide/candidate_owner.npz')) for mode in ['conductance', 'current'])
    need(all(np.array_equal(g[k], i[k]) for k in ['q0', 'S', 'gE0']), 'Matched G/I adoption')
    need(used == 16, 'Qualification exposure budget')
    return dict(status='PASS',attempted_ms=used, checks=report, pairs=pairs,
                identity33_exact=True, matched_G_I_adoption=True,
                scope='Live ON/OFF pairs for new wide observer with common inherited DNg observer. No52 checkpoint resume claim.')


if __name__ == '__main__':
    out = verify()
    (H / 'QUALIFICATION.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))
