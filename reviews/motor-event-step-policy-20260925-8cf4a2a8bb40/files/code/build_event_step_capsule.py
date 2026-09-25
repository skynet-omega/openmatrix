"""Export only scientific arrays needed to review the 20-ms paired result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from compare_event_step_organism_long import read_state

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    arrays = {}
    provenance = {}
    for label, name in [('baseline','event_step_baseline_long_01'),
                        ('candidate','event_step_candidate_long_01')]:
        folder = HERE/name
        state = read_state(folder/'final_state/session')
        arrays[label+'_state'] = np.asarray(state['hybrid']['state'], dtype=np.float64)
        arrays[label+'_pending_excitation'] = np.asarray(state['pending_excitation'],dtype=np.float64)
        arrays[label+'_pending_sensors'] = np.asarray(state['pending_sensors'],dtype=np.float64)
        arrays[label+'_muscles_activation'] = np.asarray(state['muscles']['activation'],dtype=np.float64)
        arrays[label+'_plasticity_factors'] = np.asarray(state['plasticity']['factors'],dtype=np.float64)
        arrays[label+'_rtol'] = np.asarray(state['hybrid']['parameters']['rtol'],dtype=np.float64)
        arrays[label+'_atol'] = np.asarray(state['hybrid']['parameters']['atol'],dtype=np.float64)
        provenance[label] = {rel:sha(folder/rel) for rel in
            ('final_state/session.json','final_state/session.npz','RESULT.json',
             'EVENT_STEP_RECEIPT.json','EVENT_AUDIT.json','traces.npz')}
    target = HERE/'EVENT_STEP_LONG_CAPSULE.npz'
    np.savez_compressed(target,**arrays)
    manifest = {'schema':'event_step_capsule_manifest_v1',
                'capsule_sha256':sha(target),
                'capsule_bytes':target.stat().st_size,
                'source_hashes':provenance,
                'scope':'Scientific state arrays only; complete session checkpoints remain local.'}
    (HERE/'EVENT_STEP_CAPSULE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'capsule_sha256':manifest['capsule_sha256'],
                      'capsule_bytes':manifest['capsule_bytes']}))


if __name__ == '__main__':
    main()
