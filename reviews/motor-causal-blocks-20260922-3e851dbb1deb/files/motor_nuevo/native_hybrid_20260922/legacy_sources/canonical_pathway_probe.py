"""Read and reversibly disconnect existing pathways in the complete CNS.

Diagnostic contexts cannot be checkpointed as active organisms. They do not
rewrite anatomical weights, infer conductances or fabricate missing neurons.
"""
from pathlib import Path
from contextlib import contextmanager
import json
import numpy as np
from session_io import sha256

ROOT=Path(__file__).resolve().parents[1]

class CanonicalPathways:
    def __init__(self,hybrid):
        self.h=hybrid;p=ROOT/'data/canonical_pathways_20260910'
        self.manifest=json.loads((p/'manifest.json').read_text())
        for f,digest in self.manifest['files'].items():
            if sha256(ROOT/f)!=digest:raise ValueError('Changed pathway provenance: '+f)
        with np.load(p/'routes.npz',allow_pickle=False) as z:self.arrays={k:z[k].copy() for k in z.files}
        b=hybrid.brain;self.groups={};self.active=()
        if b.n_neurons!=166700 or b.W.nnz!=25582938:raise ValueError('Requires canonical CNS')
        for name in self.manifest['groups']:
            a={k:self.arrays[name+'__'+k] for k in ('positions','pre','post','pre_ids','post_ids','contacts')}
            e=a['positions'];pre=a['pre'];post=a['post']
            if (not np.array_equal(b.W.indices[e],pre) or np.any((e<b.W.indptr[post])|(e>=b.W.indptr[post+1]))
                    or not np.array_equal(b.node_ids[pre],a['pre_ids']) or not np.array_equal(b.node_ids[post],a['post_ids'])):
                raise ValueError('Canonical pathway orientation or neuron IDs disagree')
            a['weights']=hybrid.weights64[e].copy()
            if name.endswith(('KCgd','KCabp')):
                c=hybrid._dynamic_cache;j=np.searchsorted(c['rows'],post)
                if not np.array_equal(c['rows'][j],post):raise ValueError('Missing electrical KC target')
                local=c['local_ptr'][j]+e-b.W.indptr[post]
                if np.any(c['mode'][local]) or np.any(c['pn_slot'][local]>=0) or np.any(c['apl_edge_slot'][local]>=0):
                    raise ValueError('Visual probe would overlap a specialized afferent interface')
                a['local']=j
            self.groups[name]=a

    def visual_conductances(self,name):
        h=self.h;a=self.groups[name];pre=a['pre'];w=a['weights']
        f=h.state[h.transmission_start+pre]*h.caps[pre]
        if not h.visual_output_connected:f=np.where(h.visual_mask[pre],0.,f)
        group=h.kc_apl_dynamic_manifest['group'][a['local']]
        values=np.abs(w)*f*h._dynamic_scales[group,(w<0).astype(int)]
        ge=np.zeros(len(h._dynamic_cache['rows']));gi=ge.copy()
        np.add.at(ge,a['local'][w>=0],values[w>=0]);np.add.at(gi,a['local'][w<0],values[w<0])
        return ge,gi

    def observe(self):
        h=self.h;out={};q=h.release()
        for name,a in self.groups.items():
            pre=np.unique(a['pre']);post=np.unique(a['post'])
            rec=dict(source_rows=pre,target_rows=post,source_q=q[pre],target_q=q[post],
                     disconnected=name in self.active)
            if name.endswith(('KCgd','KCabp')):
                ge,gi=self.visual_conductances(name);loc=np.searchsorted(h._dynamic_cache['rows'],post)
                voltage=h.kc_apl_dynamic_state['voltage_mV'][np.searchsorted(h._kc_rows,post)]
                rec.update(ge_nS=ge[loc],gi_nS=gi[loc],voltage_mV=voltage,
                    net_current_pA=ge[loc]*(0.-voltage)+gi[loc]*(-68.-voltage),
                    scale_status='Inherited uncalibrated subtype conductance; readout is pre-disconnection contribution')
            out[name]=rec
        return out

    @contextmanager
    def disconnect(self,*names):
        """Hold a declared route cut through live steps and always restore caches.

        Visual cuts remove only their contributions from the KC conductance
        equations. Motor cuts zero device coefficients read by CNS kernels.
        All changes end on exiting this context, including on exceptions.
        """
        if self.active or len(set(names))!=len(names) or not set(names)<=set(self.groups):
            raise ValueError('Invalid or nested pathway cut')
        import cupy as cp
        h=self.h;original=h.conductances
        if 'conductances' in h.__dict__ or 'state_dict' in h.__dict__:raise ValueError('Pre-existing runtime override')
        visual=[n for n in names if n.endswith(('KCgd','KCabp'))]
        motor=[n for n in names if n not in visual]
        if names:
            all_positions=np.concatenate([self.groups[n]['positions'] for n in names])
            if len(np.unique(all_positions))!=len(all_positions):raise ValueError('Overlapping route aliases would double-remove a current')
        positions=np.unique(np.concatenate([self.groups[n]['positions'] for n in motor])) if motor else np.array([],dtype=np.int64)
        if np.intersect1d(positions,h._apl_positions).size:raise ValueError('Overlapping APL cache')
        gpu_positions=cp.asarray(positions);saved=h.cuda['weights'][gpu_positions].copy()
        self.active=tuple(names)
        def reject_checkpoint():raise RuntimeError('A diagnostic route cut cannot be saved as an active CNS checkpoint')
        def conductances():
            ge,gi,age,agi=original()
            for n in visual:
                a,b=self.visual_conductances(n);ge-=a;gi-=b
            if ge.min() < -1e-12 or gi.min() < -1e-12:raise ValueError('Route current exceeds total current')
            return np.maximum(ge,0.),np.maximum(gi,0.),age,agi
        try:
            h.state_dict=reject_checkpoint
            h.cuda['weights'][gpu_positions]=0.
            if visual:h.conductances=conductances
            yield self
        finally:
            if visual:del h.conductances
            del h.state_dict
            h.cuda['weights'][gpu_positions]=saved
            self.active=()
