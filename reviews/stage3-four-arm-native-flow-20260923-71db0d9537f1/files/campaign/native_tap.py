"""FP64 sideband inside the two actual CUDA rate-coefficient consumers.

The store never feeds the model. It records the net immediately before tanh,
after temporary weight/release substitutions already installed by the owner.
The graph holds one stable buffer and overwrites it at every coefficient stage;
read it only after an accepted organism step.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TARGET_IDS = (10176, 10208, 10360, 523769)
TARGET_TOL = 1e-9
RATE_TOL = 1e-12
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true')


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError('Unexpected native kernel source; no diagnostic patch')
    return source.replace(before, after)


def instrumented_sources() -> tuple[str, str, dict]:
    import gpu_visual_brain as general
    import prosthetic_olfactory_brain as pn
    a = general.CUDA_SOURCE
    a = replace_once(a, 'double* target, double* rate) {',
                     'double* target, double* rate, double* debug_raw) {')
    a = replace_once(a,
        'target[row]=fmax(0.,tanh(gain[row]*(a+drive[row]-theta[row])));',
        'debug_raw[row]=a; target[row]=fmax(0.,tanh(gain[row]*(a+drive[row]-theta[row])));')
    b = pn.ORN_PN_CUDA_SOURCE
    b = replace_once(b, 'double slow_area, double normalization, double* target) {',
                     'double slow_area, double normalization, double* target, double* debug_raw) {')
    b = replace_once(b,
        'if(lane==0) target[row]=fmax(0.,tanh(gain[row]*(current+drive[row]-theta[row])));',
        'if(lane==0) { debug_raw[row]=current; target[row]=fmax(0.,tanh(gain[row]*(current+drive[row]-theta[row]))); }')
    digest = {k:hashlib.sha256(v.encode()).hexdigest() for k,v in
              [('general_original',general.CUDA_SOURCE),('general_tap',a),
               ('pn_original',pn.ORN_PN_CUDA_SOURCE),('pn_tap',b)]}
    return a,b,digest


class KernelWithTap:
    def __init__(self, kernel, output):
        self.kernel, self.output = kernel, output

    def __call__(self, grid, block, args, **kwargs):
        return self.kernel(grid, block, (*args, self.output), **kwargs)


class NativeFlowTap:
    def __init__(self, hybrid, metadata, output: Path):
        import cupy as cp
        self.h = hybrid
        self.out = output
        self.out.mkdir(parents=True, exist_ok=False)
        self.raw = cp.full(hybrid.brain.n_neurons, np.nan, dtype=cp.float64)
        table = metadata.set_index('bodyId').reindex(hybrid.brain.node_ids)
        types = table['type'].fillna('UNANNOTATED').astype(str).to_numpy()
        ids = np.asarray(hybrid.brain.node_ids)
        positions = [np.flatnonzero(ids == identity) for identity in TARGET_IDS]
        if any(len(p) != 1 for p in positions):
            raise ValueError('Expected four unique bilateral PN/DNa02 identities')
        self.rows = np.asarray([int(p[0]) for p in positions],dtype=np.int32)
        if [types[i] for i in self.rows] != ['DM1_lPN','DM1_lPN','DNa02','DNa02']:
            raise ValueError('Focal types changed')
        for name in ('kcgamma_output_manifest','regional_manifest','pnkc_receptor_manifest'):
            manifest = getattr(hybrid,name)
            if manifest.get('enabled') and np.intersect1d(
                    self.rows,np.asarray(manifest['target_rows'],dtype=np.int64)).size:
                raise ValueError('Another output owner replaces a focal row: '+name)
        pn_rows = set(map(int, np.asarray(hybrid._pn_rows)))
        self.metadata = {}
        for row in self.rows:
            if hybrid.visual_mask[row]:
                raise ValueError('Focal row unexpectedly visual')
            name = types[row]
            if (row in pn_rows) != (name == 'DM1_lPN'):
                raise ValueError('PN kernel ownership differs from focal metadata')
            self.metadata[str(int(hybrid.brain.node_ids[row]))] = {
                'row':int(row), 'type':name,
                'side':str(table.iloc[row]['somaSide']),
                'instance':str(table.iloc[row]['instance']),
                'operator':'orn_pn_synaptic' if row in pn_rows else 'ordinary_rate',
            }
        a,b,self.kernel_digests = instrumented_sources()
        self.general = cp.RawKernel(a,'coefficient',options=OPTIONS)
        self.pn = cp.RawKernel(b,'orn_pn_synaptic',options=OPTIONS)
        self.phase = self.ms = None
        self.last = None
        self.build_calls = 0
        self.records = []
        (self.out/'METADATA.json').write_text(json.dumps({
            'schema':'native_fp64_raw_rate_input_v1',
            'scope':'Last accepted fine-stage coefficient sampled after step; signed net in rate-model units, not membrane current pA.',
            'rows':self.metadata,'kernel_sha256':self.kernel_digests,
            'source_note':'Both kernels have one additional FP64 store before tanh. No read of the sideband in biological equations.'
        },ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        self.log = (self.out/'FLOW.jsonl').open('x',encoding='utf-8',buffering=1)

    def install(self):
        h = self.h
        old_general,old_pn,old_coeff = h.kernel,h._orn_pn_kernel,h.coefficients_gpu
        had_own_coeff = 'coefficients_gpu' in h.__dict__
        old_own_coeff = h.__dict__.get('coefficients_gpu')
        h.kernel = KernelWithTap(self.general,self.raw)
        h._orn_pn_kernel = KernelWithTap(self.pn,self.raw)
        def coefficients(state,drive,light):
            target,rate=old_coeff(state,drive,light)
            self.last=(target,rate,drive)
            self.build_calls+=1
            return target,rate
        h.coefficients_gpu=coefficients
        def undo():
            h.kernel=old_general
            h._orn_pn_kernel=old_pn
            if had_own_coeff:
                h.coefficients_gpu=old_own_coeff
            else:
                h.__dict__.pop('coefficients_gpu',None)
        return undo

    def set_interval(self, phase: str, ms: int):
        self.phase,self.ms=phase,ms

    def sample_last(self, adapter):
        import cupy as cp
        if self.last is None or self.build_calls != 18 or self.phase is None or adapter.core is None:
            raise RuntimeError(f'No captured last coefficient stage; callbacks={self.build_calls}')
        h=self.h
        target,rate,drive=self.last
        # Each array is retained by the CUDA graph and overwritten on every
        # replay. The native controller has returned only after acceptance.
        raw=cp.asnumpy(self.raw[self.rows])
        got=cp.asnumpy(target[self.rows])
        got_rate=cp.asnumpy(rate[self.rows])
        dg=cp.asnumpy(drive[self.rows])
        expected=np.maximum(0.,np.tanh(h.rate_gain[self.rows]*(raw+dg-h.rate_theta[self.rows])))
        expected_rate=1./h.tau[self.rows]
        if not all(np.isfinite(a).all() for a in (raw,got,got_rate,dg)):
            raise ValueError('Nonfinite native sideband or coefficient')
        err=float(np.max(np.abs(expected-got)))
        rate_err=float(np.max(np.abs(expected_rate-got_rate)))
        if err>TARGET_TOL or rate_err>RATE_TOL:
            raise ValueError(f'Native raw input does not reproduce consumed target: {err}, rate: {rate_err}')
        record={'phase':self.phase,'ms':self.ms,'time_ns':int(h.time_ns),
                'graph_build_coefficient_calls':self.build_calls,
                'target_max_error':err,'rate_max_error':rate_err,
                'rows':{str(int(h.brain.node_ids[row])):{
                    'raw_signed':float(raw[j]),'target':float(got[j]),'rate':float(got_rate[j]),
                    'drive':float(dg[j]),'theta':float(h.rate_theta[row])}
                    for j,row in enumerate(self.rows)}}
        self.log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        self.records.append(record)

    def save(self):
        if not self.records:return
        rows=[str(int(self.h.brain.node_ids[i])) for i in self.rows]
        np.savez_compressed(self.out/'FLOW.npz',
            time_ns=np.asarray([r['time_ns'] for r in self.records],dtype=np.int64),
            phase=np.asarray([r['phase'] for r in self.records]),
            ids=np.asarray([int(i) for i in rows],dtype=np.int64),
            raw_signed=np.asarray([[r['rows'][i]['raw_signed'] for i in rows] for r in self.records],dtype=np.float64),
            target=np.asarray([[r['rows'][i]['target'] for i in rows] for r in self.records],dtype=np.float64))
        (self.out/'RESULT.json').write_text(json.dumps({
            'samples':len(self.records),
            'max_target_error':max(r['target_max_error'] for r in self.records),
            'max_rate_error':max(r['rate_max_error'] for r in self.records),
            'scope':'Direct native signed net only, no attribution by edge group, no stage-3 verdict.'
        },ensure_ascii=False,indent=2,allow_nan=False)+'\n')

    def close(self):
        self.log.close()
