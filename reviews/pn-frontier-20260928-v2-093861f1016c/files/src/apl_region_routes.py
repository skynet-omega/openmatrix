"""Conservative anatomical routing for the two existing canonical APL cells.

This is a routing interface, not a compartmental electrical model. The final
region is an unresolved anatomical remainder. No coupling, conductance scale,
voltage dynamics or independent regional physiology is inferred here.
"""
from pathlib import Path
import json
import numpy as np
from session_io import sha256


class AplRegionRoutes:
    @classmethod
    def load(cls, folder, *, node_ids, indptr, indices):
        folder=Path(folder);m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if m.get('schema')!='matrix_apl_expanded_routes_v1' or sha256(folder/'routes.npz')!=m['selection_sha256']:
            raise ValueError('Unsupported or changed APL anatomical routes')
        with np.load(folder/'routes.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        if not a['roi_known'].all() or not np.isfinite(a['region_fractions']).all():
            raise ValueError('Unknown pair localization requires explicit resolution')
        if len(node_ids)!=m['canonical_neurons'] or len(indices)!=m['canonical_pairs']:
            raise ValueError('Canonical graph size changed')
        p=a['csr_positions'];pre=indices[p];post=np.searchsorted(indptr,p,side='right')-1
        for key,actual in [('pre_rows',pre),('post_rows',post),('pre_ids',node_ids[pre]),
                           ('post_ids',node_ids[post]),('apl_ids',node_ids[a['apl_rows']])]:
            if not np.array_equal(a[key],actual):raise ValueError('Changed anatomical mapping: '+key)
        # Require the complete set of incoming and outgoing canonical APL pairs.
        all_positions=np.unique(np.concatenate([np.flatnonzero(np.isin(indices,a['apl_rows'])),
            *[np.arange(indptr[r],indptr[r+1],dtype=np.int64) for r in a['apl_rows']]]))
        if not np.array_equal(p,all_positions):raise ValueError('Incomplete APL route coverage')
        if (not np.array_equal(a['roi_counts'].sum(axis=1),a['total_contacts'])
                or np.any(a['roi_counts']<0) or np.any(a['total_contacts']<=0)
                or not np.array_equal(a['region_fractions'],a['roi_counts']/a['total_contacts'][:,None])):
            raise ValueError('Nonconservative anatomical partition')
        obj=cls();obj.arrays=a;obj.manifest=m
        obj.incoming=np.flatnonzero(a['apl_post']);obj.outgoing=np.flatnonzero(a['apl_pre'])
        obj.incoming_apl=np.searchsorted(a['apl_rows'],a['post_rows'][obj.incoming])
        obj.outgoing_apl=np.searchsorted(a['apl_rows'],a['pre_rows'][obj.outgoing])
        for array in a.values():array.flags.writeable=False
        return obj

    @property
    def shape(self):
        return len(self.arrays['apl_ids']),len(self.arrays['region_names'])

    def incoming_by_region(self, pair_signal):
        """Distribute one additive value per selected pair, retaining its units.

        The caller supplies signed current, conductance or another explicitly
        defined additive signal. This function does not convert contact counts
        or neuronal firing rates into electrical quantities.
        """
        x=np.asarray(pair_signal,dtype=np.float64)
        if x.shape!=self.arrays['total_contacts'].shape or not np.isfinite(x).all():
            raise ValueError('Requires one finite value per canonical selected pair')
        result=np.zeros(self.shape,dtype=np.float64)
        np.add.at(result,self.incoming_apl,
                  self.arrays['region_fractions'][self.incoming]*x[self.incoming,None])
        return result

    def outgoing_release(self, local_release):
        """Return outgoing CSR positions and weighted release in [0,1].

        No release values are invented for the unresolved remainder: the caller
        must supply and justify all regional values, including that remainder.
        Uniform release per APL reproduces its scalar release on every edge.
        """
        q=np.asarray(local_release,dtype=np.float64)
        if q.shape!=self.shape or not np.isfinite(q).all() or np.any((q<0)|(q>1)):
            raise ValueError('Requires an explicit bounded release for every region, including remainder')
        result=np.sum(q[self.outgoing_apl]*self.arrays['region_fractions'][self.outgoing],axis=1)
        return self.arrays['csr_positions'][self.outgoing].copy(),result
