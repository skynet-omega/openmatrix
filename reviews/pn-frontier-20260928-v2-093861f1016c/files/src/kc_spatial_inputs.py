"""Conserve canonical PN/APL conductances while declaring spatial assumptions."""
import numpy as np


class CanonicalKcSpatialInputs:
    domain_names=('soma_proxy','dendrites','SIZ_proxy','axon')

    def __init__(self,brain):
        self.brain=brain;m=brain.kc_electrical_scales_manifest
        self.rows=m['rows'][m['group']==2];self.dynamic_index=np.flatnonzero(m['group']==2)
        self.kc_index=np.searchsorted(brain._kc_rows,self.rows)
        a=brain.routes.arrays;positions=m['apl_positions'];ix=np.searchsorted(a['csr_positions'],positions)
        np.testing.assert_array_equal(a['csr_positions'][ix],positions)
        self.post=np.searchsorted(self.rows,a['post_rows'][ix]);self.apl=np.searchsorted(a['apl_rows'],a['pre_rows'][ix])
        self.fraction=a['region_fractions'][ix].copy();self.gbar=m['apl_gbar_nS'].copy()
        self.region_names=a['region_names'].tolist();self.region_domain=np.zeros(len(self.region_names),dtype=np.int64)
        for j,name in enumerate(self.region_names):
            if name.startswith('CA('):self.region_domain[j]=1
            elif name.startswith('PED('):self.region_domain[j]=2
            elif name!='outside_selected_MB':self.region_domain[j]=3
        self.roi_counts=a['roi_counts'][ix].sum(axis=0)

    def split(self,components,*,pn_enabled=True,apl_enabled=True):
        ge,gi,age,agi,pn,apl=components;j=self.dynamic_index
        positive=np.zeros((len(j),4));negative=np.zeros_like(positive)
        positive[:,0]=ge[j]-pn[j];positive[:,1]=pn[j] if pn_enabled else 0.
        negative[:,0]=gi[j]-apl[j]
        release=self.brain.kc_apl_dynamic_state['apl_transmission']
        by_region=self.gbar[:,None]*self.fraction*release[self.apl]
        regional=np.zeros_like(negative)
        for r,domain in enumerate(self.region_domain):np.add.at(regional[:,domain],self.post,by_region[:,r])
        np.testing.assert_allclose(regional.sum(axis=1),apl[j],atol=1e-13,rtol=1e-12)
        if apl_enabled:negative+=regional
        if pn_enabled:np.testing.assert_allclose(positive.sum(axis=1),ge[j],atol=1e-13,rtol=1e-12)
        if apl_enabled:np.testing.assert_allclose(negative.sum(axis=1),gi[j],atol=1e-13,rtol=1e-12)
        if np.any(positive < -1e-12) or np.any(negative < -1e-12):raise ValueError('Lost/duplicated input conductance')
        # Roundoff subtraction only; exact total is checked above.
        return np.maximum(positive,0.),np.maximum(negative,0.),regional

    def manifest(self):
        return dict(rows=self.rows.copy(),ids=self.brain.brain.node_ids[self.rows].copy(),
            PN_pairs=len(self.brain.kc_electrical_scales_manifest['pn_positions']),APL_pairs=len(self.gbar),
            domains=list(self.domain_names),regions=self.region_names,region_to_domain=self.region_domain.copy(),APL_contacts_by_region=self.roi_counts.copy(),
            PN_location='Area-uniform dendritic conductance, original selected canonical pairs and filters; individual claws unresolved.',
            APL_location='CA to dendrites; PED to SIZ proxy; named MB lobes to entire axon; outside_selected_MB to soma proxy. Region fractions and regional release are multiplied before summing, preserving their correlation.',
            other_inputs='All non-PN/non-APL canonical inputs remain in the soma proxy with their inherited nS scale.',
            unresolved='ROI is not an electrotonic coordinate. PED/SIZ and uniform intra-region distributions are explicit hypotheses, not measured local synapse placements.',
            E_exc_mV=0.,E_APL_mV=-68.,absolute_pair_strength_identified=False,native_chloride_identified=False)
