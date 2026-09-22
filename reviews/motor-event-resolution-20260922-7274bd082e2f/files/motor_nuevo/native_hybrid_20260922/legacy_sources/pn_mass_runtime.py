"""Versioned CNS mass factory, derived from the qualified finite replay factory.

Loading reads only reduced operators and small physical maps, never assembles
the omitted fine volume. A guarded seed supplies all nonzero source histories.
"""
from pathlib import Path
import sys,json,copy
import numpy as np
from scipy.sparse import load_npz
R=Path(__file__).resolve().parents[1];W=R/'work/pn_mass_runtime_20260913'
from pn_mass_backend import FullMassBackend
from pn_mass_source_bridge import MassPnSourceBridge
from pn_calcium_port import CalciumPort
from pn_calcium_release import CalciumReleaseSites
from session_io import sha256


def load_npz_dict(path):
 with np.load(path) as z:return {k:z[k].item() if z[k].ndim==0 else z[k].copy() for k in z.files}


def make():
 base=R/'evidence/pn_internal_charge_20260912/model_rtol1e10';meta=json.loads((base/'model.json').read_text());nv=meta['volume_nodes'];ne=meta['exterior_unknowns']
 assert json.loads((W/'membrane_audit.json').read_text())['physical_membrane_independently_reconstructed']
 backend=FullMassBackend(load_npz(base/'G_nS.npz'),load_npz(base/'M_nF.npz'),np.load(W/'physical_membrane_C_nF.npy'),
  provenance='Galerkin H/Z matrices unchanged; '+meta['basis_identity'])
 data=load_npz_dict(R/'work/pn_calcium_coupling_20260913/preparation.npz')
 prior=json.loads((R/'work/pn_calcium_coupling_20260913/protocol.json').read_text());provenance=json.dumps(prior['calcium_prior'],sort_keys=True)
 keys=['rest_uM','volume_um3','buffer_capacity','clearance_s','Kd_uM','cooperativity','maximum_hazard_per_s','recovery_s','rise_s','decay_s']
 chem=CalciumReleaseSites(data['site_ids'],{k:data[k] for k in keys},provenance=provenance)
 nodes=data['nodes']-nv;sites=data['site_nodes']-nv
 assert np.all(nodes>=0) and np.all(nodes<ne) and np.all(sites>=0) and np.all(sites<ne)
 port=CalciumPort(nodes,data['gbar_nS'],data['reversal_mV'],np.full((len(nodes),1),-40.),np.full((len(nodes),1),7.5),
  np.full((len(nodes),1),.05),[1],sites,data['site_fractions'],chem,provenance=provenance,enabled=True)
 channel=load_npz_dict(R/'evidence/pn_fine_ionic_20260912/channel_patch.npz');geo=load_npz_dict(R/meta['parent_model']/'geometry.npz');keep=geo['kept_original_nodes']
 active=np.searchsorted(keep,channel['original_nodes']);np.testing.assert_array_equal(keep[active],channel['original_nodes'])
 card=R/'evidence/pn_fine_ionic_20260912/model_card.json';c=json.loads(card.read_text())
 pn=MassPnSourceBridge(backend,node_identity=meta['physical_node_order_identity'],active_nodes=active,gbar_nS=channel['gbar_nS'],
  reversal_mV=channel['reversal_mV'],leak_reversal_mV=c['leak_reversal_mV'],channel_provenance=sha256(card)+':focal:candidate',
  calcium_port=port,preparation=prior['preparation']+'; candidate conditioned internal-charge projection, no biological parameter change')
 return pn,meta


def build_source(contract,manifest):
 """Rebuild a source from a guarded, nonzero adoption seed and current state.

 Source origin is explicitly rebased to the mass adoption checkpoint. All
 physical clocks, charges and inherited tails retain their absolute values.
 Earlier origins remain in the contract for provenance; no history is reset.
 """
 from session_io import read_state
 from pn_spatial_orn import SpatialOrnAllocation
 from pn_online_orn_source import OnlineOrnPnSource
 from pn_cholinergic_online_source import CholinergicOnlineSource
 from pn_electrical_output_source import ElectricalOutputSource
 from pn_general_output_source import GeneralOutputSource
 if contract['schema']!='PN_mass_CNS_recipe_v1' or contract['exterior_coordinates']!=178818 or contract['internal_coordinates']!=20:
  raise ValueError('Explicit generalized coordinate recipe required')
 for path,digest in contract['assets_sha256'].items():
  if sha256(R/path)!=digest:raise ValueError('Mass runtime asset changed: '+path)
 seed=read_state(R/contract['seed']);pn,meta=make();pn.load_state_dict(seed['pn'])
 if pn.identity!=contract['pn_identity'] or pn.time_ns!=contract['PN_origin_ns']:
  raise ValueError('Mass seed identity or clock changed')
 om=load_npz_dict(R/'evidence/pn_terminal_tail_20260912/orn_input/mapping.npz');N=len(pn.voltage);nv=38757888
 allocation=SpatialOrnAllocation(om['source_ids'],om['contact_pre_ids'],om['contact_joint_nodes']-nv,om['pair_shares'],full_size=N)
 mapping=load_npz_dict(R/'work/pn_orn_response_20260913/site_KC_map.npz');prep=manifest['preparation']
 base=OnlineOrnPnSource(pn,allocation,prep['caps_hz'],mapping,cns_time_ns=contract['CNS_origin_ns'],
  preparation=prep['preparation']+'; explicit mass coordinate migration',gain_provenance=prep['gain_provenance'])
 spec=copy.deepcopy(manifest['cholinergic']['spec']);spec['mapping']['contact_nodes']-=nv;spec['mapping']['full_size']=N
 ach=CholinergicOnlineSource.adopt(base,**spec,connected=manifest['cholinergic']['connected'],origin_ns=manifest['cholinergic']['adoption_time_ns'])
 electrical=ElectricalOutputSource.adopt(ach,manifest['electrical_outputs']['spec'])
 general=GeneralOutputSource.adopt(electrical,manifest['general_outputs']['spec'])
 saved=copy.deepcopy(seed['source']);saved['identity']=general.identity
 saved['base']['identity']=electrical.identity
 saved['base']['base']['identity']=ach.identity
 saved['base']['base']['base']['identity']=base.identity
 saved['base']['base']['base']['pn']=pn.state_dict()
 general.load_state_dict(saved)
 if general.time_ns!=contract['CNS_origin_ns']:
  raise ValueError('Mass source history clock mismatch')
 return general
