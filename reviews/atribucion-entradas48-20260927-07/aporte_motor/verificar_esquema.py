"""CPU-only schema/membership review; does not load W/state or evaluate a neuron."""
import hashlib
import json
import mmap
import re
import resource
import time
import zipfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/daroch/AXIOMA_ASTRA')
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
CAMP = ROOT / 'campanas/etapa45_composicion_20260927_48'
BRAIN = OLD / 'work/stage234_settling_extension_20260915/settled_700ms/core_carrier/brain'
APL = OLD / 'data/apl_expanded_routes_20260910/routes.npz'
IDS = np.array([10045, 10056, 10118, 10065, 523769, 10360], dtype=np.int64)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def exact_rows(ids, wanted):
    rows = np.searchsorted(ids, wanted)
    need(np.all(rows < len(ids)), 'ID outside graph')
    need(np.array_equal(ids[rows], wanted), 'Wrong ID-to-row mapping')
    return rows


def top_fields(path, names):
    # The serialized history is ~250 MB. Decode only bounded top-level values.
    result = {}
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        for name in names:
            match = re.search(rb'(?m)^  "' + name.encode() + rb'":\s*', m)
            need(match is not None, 'Missing field ' + name)
            result[name], _ = json.JSONDecoder().raw_decode(
                m[match.end():match.end() + 1000000].decode())
    return result


def header(path, key):
    with zipfile.ZipFile(path) as z, z.open(key + '.npy') as f:
        version = np.lib.format.read_magic(f)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, order, dtype = reader(f)
    return {'shape': list(shape), 'fortran': order, 'dtype': str(dtype)}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    start = time.monotonic()
    cpu = time.process_time()
    lock = json.loads((CAMP / 'SOURCES.json').read_text())
    provenance = {}
    sources = [BRAIN / 'state.npz', BRAIN / 'weights_post_pre.npz', APL,
               OLD / 'src/hybrid_visual_brain.py', OLD / 'src/gpu_coefficient_layout.py',
               OLD / 'src/anatomical_rate_brain.py',
               OLD / 'work/motor14_20260922/motor_runtime.py',
               OLD / 'config/matrix_diagnostico_olfativo_v1.json',
               ROOT / 'motor_nuevo/full_pipeline_review_20260925_12/restore_prepared.py',
               CAMP / 'checkpoint48.py']
    for p in sources:
        actual = sha(p)
        need(lock.get(str(p)) == actual, 'Source differs from campaign lock: ' + str(p))
        provenance[str(p)] = actual
    with np.load(BRAIN / 'state.npz', allow_pickle=False) as a:
        ids = a['node_ids']
        rows = exact_rows(ids, IDS)
        caps = a['r_max'][rows]
    # A wrong canonical ID must fail instead of silently selecting its neighbor.
    try:
        exact_rows(ids, np.array([-1], dtype=np.int64))
    except ValueError:
        rejected_wrong_id = True
    else:
        raise AssertionError('Invalid ID accepted')
    with np.load(APL, allow_pickle=False) as a:
        post, outgoing = a['post_rows'], a['apl_pre']
        apl_counts = [int(np.sum((post == r) & outgoing)) for r in rows]
    states = {}
    selectors = ('rows', 'target_rows', 'target_ids', 'post_rows', 'post_ids', 'post_indices')
    for arm in ('sham', 'dm1', 'profile', 'permuted'):
        folder = CAMP / arm / 'final_state'
        saved = top_fields(folder / 'session.json', ('hybrid', 'time_ns'))
        h = saved['hybrid']
        manifest = json.loads((folder / 'MANIFEST.json').read_text())
        published = json.loads((folder / 'published.json').read_text())
        need(saved['time_ns'] == h['time_ns'] == manifest['time_ns'] == published['time_ns'],
             'Checkpoint owner clocks differ')
        membership = {}
        with np.load(folder / 'session.npz', allow_pickle=False) as a:
            nphoto = len(a[h['photo_ids']['__array__']])
            visual = a[h['visual_ids']['__array__']]
            for name, meta in h.items():
                if not name.endswith('_manifest'):
                    continue
                for key in selectors:
                    item = meta.get(key)
                    if isinstance(item, dict) and '__array__' in item:
                        values = a[item['__array__']]
                        wanted = IDS if key.endswith('ids') else rows
                        membership[name + '.' + key] = {
                            'count': int(values.size),
                            'matched_ids': IDS[np.isin(wanted, values)].tolist()}
            terminal = h['pn_online_manifest']['orn_peripheral_terminal']['source_ids']
            terminal_ids = a[terminal['__array__']]
        membership['ORN_terminal'] = {'matched_ids': IDS[np.isin(IDS, terminal_ids)].tolist()}
        membership['visual'] = {'matched_ids': IDS[np.isin(IDS, visual)].tolist()}
        membership['CXHP8_external_drive'] = {
            'matched_ids': IDS[np.isin(IDS, h['cxhp8_manifest']['ids'])].tolist()}
        need(not any(x['matched_ids'] for x in membership.values()), 'Special target found')
        need(not h['pn_online_manifest']['general_outputs']['enabled'], 'PN general changed')
        state_header = header(folder / 'session.npz', h['state']['__array__'])
        begin = len(ids) + 2 * nphoto
        need(state_header['shape'][0] >= begin + len(ids), 'Missing transmission state')
        states[arm] = dict(time_ns=h['time_ns'], neuron_count=len(ids), photo_count=nphoto,
                           generic_transmission_slice=[begin, begin + len(ids)],
                           hybrid_state=state_header, membership=membership,
                           PN_general_enabled=False, visual_output_connected=h['visual_output_connected'],
                           restart_tested=manifest['restart_tested'])
    result = dict(status='SCHEMA_AND_MEMBERSHIP_VERIFIED',
                  ids=IDS.tolist(), rows=rows.tolist(), caps=caps.tolist(),
                  direct_APL_incoming_edges=apl_counts, states=states,
                  CSR_indices=header(BRAIN / 'weights_post_pre.npz', 'indices'),
                  rejected_wrong_id=rejected_wrong_id, sources_verified=provenance,
                  wall_s=time.monotonic() - start, CPU_s=time.process_time() - cpu,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  new_neural_steps=0, new_body_steps=0, GPU_calls=0,
                  scope='No W or full neural-state array loaded; no net attribution, no last-RHS reconstruction.')
    need(result['CPU_s'] < 60 and result['peak_RSS_bytes'] < 4 * 1024**3, 'Budget exceeded')
    (OUT / 'VERIFICACION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'rows', 'direct_APL_incoming_edges',
                                          'CPU_s', 'wall_s', 'peak_RSS_bytes')}))


if __name__ == '__main__':
    main()
