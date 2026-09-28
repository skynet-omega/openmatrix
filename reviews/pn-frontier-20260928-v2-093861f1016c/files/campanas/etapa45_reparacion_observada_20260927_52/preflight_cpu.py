"""CPU regressions of units, read-only selection and anatomy; zero CNS."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
import numpy as np
from pilot52_owners import AirOwner
from observer52 import Panel
from donors.air_interface import AirInterface

H = Path(__file__).resolve().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    start = time.process_time()
    api = AirInterface(H / 'donors/JO_anatomy_arrays.npz')
    owner = object.__new__(AirOwner)
    cases = []
    for v in [np.array([2., -3., .5]), np.array([-7., 1., 0.]), np.zeros(3)]:
        owner.run = SimpleNamespace(obj=SimpleNamespace(body=SimpleNamespace(data=SimpleNamespace(qvel=v.copy()))))
        velocity = owner.velocity_mm_s()
        need(np.array_equal(velocity, 10 * v), 'Exactly one unit conversion')
        actual = api.encode(air_velocity_world_mm_s=10*v, body_velocity_world_mm_s=velocity, body_to_world=np.eye(3))
        need(np.array_equal(actual, np.zeros(335)), 'Comoving air and body')
        cases.append(v.tolist())
    ids = np.load('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/node_ids.npy', allow_pickle=False)
    p = Panel(ids, True, api.ids)  # CPU fixture uses legal identities, not an olfactory claim.
    full = np.linspace(0, 1, len(ids))
    before = hashlib.sha256(full.tobytes()).hexdigest()
    p.record(full, 123000000)
    need(before == hashlib.sha256(full.tobytes()).hexdigest(), 'Observer changed input')
    need(np.array_equal(p.q[0], full[p.rows]), 'Observer projection mismatch')
    p.q[0][0] = -1
    need(before == hashlib.sha256(full.tobytes()).hexdigest(), 'Output aliases input')
    with np.load(H / 'PANEL.npz', allow_pickle=False) as z:
        groups = dict(zip(z['group_names'], z['group_masks']))
        counts = {k: int(groups[k].sum()) for k in ['JO_CE', 'all_DN', 'AMMC_WED']}
        need(counts == dict(JO_CE=335, all_DN=1314, AMMC_WED=1108), 'Anatomical counts')
        need(np.all(sum(groups[k].astype(int) for k in counts) == 1), 'Union overlap/coverage')
    out = dict(status='PASS_CPU', unit_cases=cases, panel_read_only_and_nonalias=True,
               anatomical_counts=counts, CPU_s=time.process_time()-start,
               scope='Software fixture only. Live-pair qualification remains required.', CNS_ms=0)
    (H / 'PREFLIGHT_CPU.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
