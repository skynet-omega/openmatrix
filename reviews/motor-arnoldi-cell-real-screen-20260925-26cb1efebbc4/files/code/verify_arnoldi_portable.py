"""Cold, arithmetic-only verifier for the public two-column real-RHS capsule."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

H = 125e-6


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def phi(mat, fraction):
    a = np.zeros((3, 3))
    a[:2, :2] = mat
    a[0, 2] = 1.0
    return expm(fraction * a)[:2, 2]


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    need(manifest['format_version'] == 2, 'manifest format')
    for entry in manifest['files']:
        path = root / entry['path']
        need(path.is_file() and path.stat().st_size == entry['bytes'] and
             sha(path) == entry['sha256'], 'file integrity: ' + entry['path'])
    source = json.loads((root / 'receipts/ARNOLDI_STAGE1_SOURCE_74.json').read_text())
    need(source['output_sha256'] == sha(root / 'data/ARNOLDI_STAGE1_SOURCE_74.npz'),
         'stage extraction hash')
    receipt = json.loads((root / 'receipts/ARNOLDI_REAL_RESULT.json').read_text())
    need(receipt['status'] == 'COMPLETE_DIAGNOSTIC_ONLY' and
         receipt['probe']['full_queries'] == 6 and
         receipt['probe']['arrays_sha256'] == sha(root / 'data/ARNOLDI_REAL_ARRAYS.npz'),
         'real result receipt')
    with np.load(root / 'data/ARNOLDI_REAL_ARRAYS.npz', allow_pickle=False) as a, \
         np.load(root / 'data/JVP_REAL_ARRAYS.npz', allow_pickle=False) as previous, \
         np.load(root / 'data/ARNOLDI_STAGE1_SOURCE_74.npz', allow_pickle=False) as source_arrays:
        z = a['z']; D = a['D']; mask = a['mask']; slow = a['slow']
        points = a['points']; full = a['full_f']; Q = a['q']; K = a['K']
        need(z.shape == D.shape == mask.shape == slow.shape == (359373,) and
             points.shape == full.shape == (6, 359373) and Q.shape == (3, 359373) and
             K.shape == (3, 2) and mask.dtype == np.bool_ and
             np.isfinite(z).all() and np.isfinite(full).all() and np.isfinite(K).all(),
             'array layout/finitude')
        for label, first, second in [('state', z, source_arrays['z']),
                                     ('slow', slow, source_arrays['slow_f']),
                                     ('mask', mask, source_arrays['mask']),
                                     ('full', full[0], source_arrays['full_f']),
                                     ('previous state', z, previous['z']),
                                     ('previous full', full[0], previous['full_f'][0]),
                                     ('previous plus', full[1], previous['full_f'][1]),
                                     ('previous minus', full[2], previous['full_f'][2]),
                                     ('repeat', full[0], full[5])]:
            need(np.array_equal(first, second), label + ' mismatch')
        need(np.array_equal(D, 1e-7 + 1e-5 * np.abs(z)) and
             float(a['t'][0]) == float(source_arrays['time'][0]), 'scale/time')
        need(np.all(Q[:, mask] == 0) and np.all(full[:, mask] == 0), 'prescribed states')
        beta = float(np.linalg.norm(previous['v'] / D))
        need(np.allclose(Q[0], previous['v'] / D / beta, rtol=1e-12, atol=1e-12),
             'first basis vector')
        h = np.zeros_like(K)
        even = []
        for j in range(2):
            amp = 1.0 / np.max(np.abs(Q[j]))
            step = amp * D * Q[j]
            step[mask] = 0.0
            need(np.array_equal(points[2*j+1], z+step) and
                 np.array_equal(points[2*j+2], z-step) and
                 np.all((z+step>=0)&(z+step<=1)&(z-step>=0)&(z-step<=1)),
                 'input/dynamic domain')
            plus, minus = full[2*j+1], full[2*j+2]
            even.append(float(np.max((H*np.abs((plus+minus)/2-full[0])/D)[~mask])))
            w = H*(plus-minus)/(2*amp*D)
            residual = w.copy()
            for _ in range(2):
                for i in range(j+1):
                    value = float(Q[i] @ residual)
                    h[i,j] += value
                    residual -= value*Q[i]
            h[j+1,j] = np.linalg.norm(residual)
            need(np.allclose(Q[j+1], residual/h[j+1,j], rtol=1e-11, atol=1e-11),
                 'next basis vector')
        need(np.allclose(K,h,rtol=1e-11,atol=1e-11) and
             np.allclose(even,a['even'],rtol=1e-12,atol=1e-12), 'K/even values')
        forcing=H*slow/D
        need(np.allclose(forcing,np.linalg.norm(forcing)*Q[0],rtol=1e-12,atol=1e-11),
             'forcing direction')
        fractions=np.array([.25,.5,.75,1.])
        need(np.array_equal(a['sample_fractions'],fractions), 'fractions')
        samples=np.array([np.linalg.norm(forcing)*np.max(np.abs(K[2,1]*Q[2]))*
                          abs(phi(K[:2,:2],t)[1]) for t in fractions])
        need(np.allclose(samples,a['sampled_residual'],rtol=1e-11,atol=1e-11) and
             np.allclose(samples,receipt['probe']['sampled_residuals'],rtol=1e-11,atol=1e-11),
             'sampled residual')
        need(bool(max(even)<=.1)==receipt['probe']['even_gate_pass'] and
             bool(max(samples)<=.1)==receipt['probe']['sampled_gate_pass'], 'gates')
        corrupted=samples.copy();corrupted[0]+=.5
        need(not np.allclose(corrupted,a['sampled_residual']), 'corruption accepted')
    return {'schema':'arnoldi_portable_verify_v1','status':'PASS_ARITHMETIC_ONLY',
            'even_remainder_max':float(max(even)),
            'sampled_residual_max':float(max(samples)),
            'corruption_rejected':True,
            'limitation':'No whole-organism replay, owner timing, continuous defect or integrated step.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    need(not args.out.exists(),'output exists')
    result=verify(args.root)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
