"""Build the existing qualified PN10208 reference for a runtime source.

The construction is transferred without changing geometry or coefficients
from the archived fine-PN bench. Morphology remains an external immutable
asset; loaded PN state identities reject incompatible operators.
"""
from pathlib import Path
import sys,json,time,gc,argparse,resource
import numpy as np
from scipy.sparse import csr_matrix,diags,load_npz
from scipy.sparse.linalg import splu
from scipy.ndimage import label,generate_binary_structure
R=Path(__file__).resolve().parents[1]
from neck_distributed_conductor import physical_terms
from neck_selected_conductor import selected_voxel_topology
from neck_connected_domain import connected_label_domain
from neck_surface_ports import CoupledVolumeCollar
from neck_gpu_multigrid import GPUJointPNMultigrid
from pn_fine_ionic import FinePNIonicSession
from neck_joint_csr import join_volume_exterior_csr
from neck_joint_multigrid import JointPNMultigrid
from neck_multiport import laplacian
from neck_dynamic_handoff import withdraw_source_block
from session_io import sha256

def read_mask(bid,lo,hi):
 lo=np.asarray(lo,dtype=np.int64);hi=np.asarray(hi,dtype=np.int64)
 mask=np.zeros(hi-lo,bool);known=np.zeros_like(mask)
 paths=[R/f'data/dm1_neck_context_20260911/{bid}_labels.npz']+sorted((R/'data/dm1_exterior_volume_20260912').glob(f'{bid}_axis*.npz'))
 for path in paths:
  with np.load(path) as z:seg=z['segmentation'];origin=z['origin_8nm']
  a=np.maximum(lo,origin);b=np.minimum(hi,origin+np.array(seg.shape))
  if np.any(a>=b):continue
  out=tuple(slice(int(x),int(y)) for x,y in zip(a-lo,b-lo));src=tuple(slice(int(x),int(y)) for x,y in zip(a-origin,b-origin));part=seg[src]==bid
  assert np.array_equal(mask[out][known[out]],part[known[out]]),path
  mask[out]=part;known[out]=True
 if not known.all():raise ValueError(f'{np.count_nonzero(~known)} voxels unknown; acquire labels, do not pad or seal')
 return mask


def assemble(output):
    args=argparse.Namespace(body=10208,margin=64,upper_y_shift=0,inputs=[],result_subdir='adjoint',rtol=1e-9,warm_base_orn=False)
    assert args.upper_y_shift==0 or (args.body==10176 and args.upper_y_shift==31), 'Only anatomically selected Y extensions'
    suffix=f'_y{args.upper_y_shift}' if args.upper_y_shift else ''
    bid=args.body;E=R/f'evidence/dm1_exterior_joint_20260912/{bid}_m{args.margin}{suffix}';D=Path(output);D.mkdir(exist_ok=False);start=time.monotonic()
    assert bid==10208 and args.margin==64 and args.rtol==1e-9
    def dump(name,x):(D/name).write_text(json.dumps(x,indent=2)+'\n')
    def progress(**x):print(json.dumps(dict(seconds_total=time.monotonic()-start,rss_peak_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**x)),flush=True)
    def npz(p):
     with np.load(p) as z:return {k:z[k] for k in z.files}
    protocol=json.loads((E/'protocol.json').read_text());lo=np.array(protocol['lower_8nm']);hi=np.array(protocol['upper_8nm']);shape=hi-lo
    unobservable=np.array(protocol.get('unobservable_old_source_ports',[]),int)
    if len(unobservable) and 'old_differential' in args.inputs:raise ValueError('Historical differential current uses unobservable anchors; no snap or substituted current admitted')
    halo=read_mask(bid,lo-1,hi+1);prior=json.loads((R/'evidence/dm1_exterior_volume_20260912/10208_m17/protocol.json').read_text());oldlo=np.array(prior['lower_8nm']);oldhi=np.array(prior['upper_8nm'])
    seed=np.zeros_like(halo);sl=tuple(slice(int(a),int(b)) for a,b in zip(oldlo-lo+1,oldhi-lo+1));seed[sl]=halo[sl]
    pockets=npz(R/'evidence/dm1_complex_coupling_20260912/10208_closed_pockets/pockets.npz')['xyz_8nm'];seed[tuple((pockets-lo+1).T)]=True
    selection,selection_check=connected_label_domain(halo,seed);assert selection_check==protocol['component_selection'];del seed
    mask=selection[1:-1,1:-1,1:-1];own=np.zeros(shape,np.int8)
    own[tuple(slice(int(a),int(b)) for a,b in zip(oldlo-lo,oldhi-lo))]=1;own[tuple((pockets-lo).T)]=1
    t=selected_voxel_topology(halo[1:-1,1:-1,1:-1],mask,own,halo_mask=halo)
    terms=physical_terms(t,voxel_um=.008,Ri_ohm_cm=266.1,Rm_ohm_cm2=20800,Cm_uF_cm2=.79)
    progress(stage='topology',volume=t['voxel_count'],edges=len(t['edges']),open_faces=t['outer_faces'])
    maps=npz(E/'surface_map.npz');key=maps['flat_voxel']*6+maps['axis']*2+(maps['sign']>0);order=np.argsort(key);actual=t['flat_voxel'][t['outer_nodes']]*6+t['outer_axis']*2+(t['outer_sign']>0);np.testing.assert_array_equal(np.sort(key),np.sort(actual));rows=order[np.searchsorted(key[order],actual)];P=maps['voltage_map'][rows];P2=maps['expanded_voltage_map'][rows];del maps,key,actual,order,rows,own
    old=npz(R/f'data/dm1_native_cable_20260911/cables/{bid}_cable.npz');st=npz(E/'source_block.npz');collar=npz(E/'collar.npz');collar['G_nS']=csr_matrix(collar['G_nS'])
    block=dict(nodes=st['original_nodes'],internal_nodes=st['internal_original_nodes'],port_nodes=st['original_port_nodes'],port_indices=st['port_indices'],removed_edges=st['removed_source_edges'],G_nS=csr_matrix(st['G_nS']),C_nF=st['C_nF'])
    Gsource=laplacian(len(old['C_nF']),old['edges'],old['axial_nS'])+diags(old['leak_nS']);outside=withdraw_source_block(Gsource,old['C_nF'],block)
    patch=CoupledVolumeCollar(t,terms,P2,collar);joint=join_volume_exterior_csr(patch,outside['G_nS'],outside['C_nF'],outside['outside_ports']);nv=patch.n;N=len(joint['C_nF']);keep=outside['kept_original_nodes']
    previous_removed=npz(R/'evidence/dm1_exterior_volume_20260912/10208_m17/source_block.npz')['removed_source_edges'];assert np.isin(previous_removed,st['removed_source_edges']).all()
    xyz=np.empty((nv,3),np.int32)
    xyz[:,0]=t['flat_voxel']//(shape[1]*shape[2]);xyz[:,1]=(t['flat_voxel']//shape[2])%shape[1];xyz[:,2]=t['flat_voxel']%shape[2];comp,ncomp=label(mask,generate_binary_structure(3,1));components=comp.ravel()[t['flat_voxel']];del comp,mask,halo,selection
    protected=np.unique(np.r_[old['soma_node'],old['native_site_nodes'],old['ORN_contact_nodes']]);pi=outside['original_to_outside'][protected];assert (pi>=0).all();protected_joint=nv+pi
    oldports=np.array(protocol['old_source_ports']);internalized=np.array(protocol.get('supported_internalized_old_source_ports',protocol['internalized_old_source_ports']));B=load_npz(E/'internalized_old_ports.npz');assert B.shape==(len(internalized),nv)
    oldport_outside=outside['original_to_outside'][oldports];internal=np.isin(oldports,internalized);assert np.array_equal(oldports[internal],internalized);exterior_ports=oldport_outside>=0;missing=np.isin(oldports,unobservable);assert np.all(internal.astype(int)+exterior_ports.astype(int)+missing.astype(int)==1)
    np.savez_compressed(D/'node_maps.npz',kept_original_nodes=keep,protected_original_nodes=protected,protected_joint_nodes=protected_joint,old_source_ports=oldports,internalized_old_source_ports=internalized,unobservable_old_source_ports=unobservable,collar_to_joint=joint['collar_to_joint'])
    accounting=dict(original_C_nF=float(old['C_nF'].sum()),removed_C_nF=float(block['C_nF'].sum()),volume_C_nF=float(terms['C_nF'].sum()),collar_C_nF=float(collar['C_nF'].sum()),joint_C_nF=float(joint['C_nF'].sum()),volume_membrane_um2=float(terms['wall_area_um2'].sum()),old_boundary_faces_now_volume=protocol['old_open_faces_now_true_volume_connections'],all_partition_interfaces=t['interfaces'],previous_304317_open_faces_internalized=int(t['interfaces']),previous_closed_protrusions_preserved=True)
    np.testing.assert_allclose(accounting['joint_C_nF'],accounting['original_C_nF']-accounting['removed_C_nF']+accounting['volume_C_nF']+accounting['collar_C_nF'],rtol=1e-13)
    card=json.loads((R/'evidence/pn_fine_ionic_20260912/model_card.json').read_text())
    run_protocol=dict(model_card=card,geometry_protocol_sha256=sha256(E/'protocol.json'),assembly_parent_sha256=sha256(R/'work/pn_performance_20260912/run_gpu.py'),source_sha256=sha256(Path(__file__)),accounting=accounting,integrator='First-order Rush-Larsen at old local voltage, backward Euler full fine membrane',precision='FP64',rtol=1e-9,atol_pA=2e-12,maxiter=220,CNS_integrated=False,FlyBody_paused=True)
    dump('protocol.json',run_protocol);progress(stage='assembled',volume=nv,joint=N,nnz=joint['G_nS'].nnz)
    available_kib=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
    progress(stage='hierarchy_memory_check',available_GiB=available_kib/1024**2)
    assert available_kib>10*1024**2, 'Insufficient RAM before Galerkin hierarchy; stop, do not displace other tasks'
    mg=JointPNMultigrid(joint['G_nS'],joint['C_nF'],xyz,components,smoothing_steps=1,relaxation='symmetric_gauss_seidel',progress=lambda x:progress(stage='hierarchy',**x));del joint['G_nS'],xyz,components,P;gc.collect()
    dump('hierarchy.json',[dict(level=k,volume_nodes=l.volume_nodes,exterior_nodes=l.G.shape[0]-l.volume_nodes,nnz=l.G.nnz) for k,l in enumerate(mg.levels)])
    
    import hashlib
    import cupy as cp
    from compensated_csr import CompensatedCSR
    ordering=hashlib.sha256(sha256(E/'protocol.json').encode())
    for array in [t['flat_voxel'],keep,joint['collar_to_joint']]:ordering.update(str((array.dtype.str,array.shape)).encode());ordering.update(memoryview(array).cast('B'))
    channel=npz(R/'evidence/pn_fine_ionic_20260912/channel_patch.npz');aidx=outside['original_to_outside'][channel['original_nodes']];assert (aidx>=0).all(), 'No implicit channel placement inside volume'
    np.testing.assert_allclose(mg.levels[0].C[nv+aidx],old['C_nF'][channel['original_nodes']],rtol=1e-13)
    def make_session(arm='candidate'):
     gbar=channel['gbar_nS'].copy()
     if arm=='no_NaT':gbar[:,0]=0.
     elif arm!='candidate':raise ValueError('Unknown experimental arm')
     return FinePNIonicSession(backend,node_identity=ordering.hexdigest(),active_nodes=nv+aidx,gbar_nS=gbar,reversal_mV=channel['reversal_mV'],leak_reversal_mV=card['leak_reversal_mV'],channel_provenance=sha256(R/'evidence/pn_fine_ionic_20260912/model_card.json')+':focal:'+arm)
    t0=time.monotonic();backend=GPUJointPNMultigrid(mg,action_mode='voltage_differences');session=make_session();setup_seconds=time.monotonic()-t0
    dump('gpu_memory.json',dict(**backend.memory_plan,setup_seconds=setup_seconds));progress(stage='GPU_ready',setup_seconds=setup_seconds,**backend.memory_plan)
    
    return dict(backend=backend,make_session=make_session,session=session,original=old,outside=outside,nv=nv,N=N,channel=channel,ordering=ordering.hexdigest(),accounting=accounting,progress=progress)


def online_source(spec,*,context=None,assembly_output=None):
    """Construct from a declared immutable preparation, then return its owner.

    A reused context is for sequential assays only. Every source retains its
    own voltage/gates/calcium/tail state; no shared neuronal state is created.
    """
    import tempfile
    from pn_calcium_coupled import CalciumCoupledFinePN
    from pn_calcium_port import CalciumPort
    from pn_calcium_release import CalciumReleaseSites
    from pn_spatial_orn import SpatialOrnAllocation
    from pn_online_orn_source import OnlineOrnPnSource
    from session_io import read_state
    parent=R/spec['parent_PN_state']
    for suffix,h in spec['parent_PN_sha256'].items():
        if sha256(parent.with_suffix(suffix))!=h:raise ValueError('Changed fine-PN preparation')
    if context is None:
        if assembly_output is None:
            with tempfile.TemporaryDirectory(prefix='axioma-PN-assembly-') as temp:context=assemble(Path(temp)/'assembly')
        else:context=assemble(assembly_output)
    old=context['session'];backend=context['backend']
    def read(p):
        with np.load(p,allow_pickle=False) as z:return {k:z[k].item() if z[k].ndim==0 else z[k].copy() for k in z.files}
    data=read(R/'work/pn_calcium_coupling_20260913/preparation.npz')
    prior=json.loads((R/'work/pn_calcium_coupling_20260913/protocol.json').read_text())
    keys=['rest_uM','volume_um3','buffer_capacity','clearance_s','Kd_uM','cooperativity','maximum_hazard_per_s','recovery_s','rise_s','decay_s']
    provenance=json.dumps(prior['calcium_prior'],sort_keys=True)
    chem=CalciumReleaseSites(data['site_ids'],{k:data[k] for k in keys},provenance=provenance)
    port=CalciumPort(data['nodes'],data['gbar_nS'],data['reversal_mV'],np.full((len(data['nodes']),1),-40.),
        np.full((len(data['nodes']),1),7.5),np.full((len(data['nodes']),1),.05),[1],data['site_nodes'],data['site_fractions'],chem,provenance=provenance,enabled=True)
    pn=CalciumCoupledFinePN(backend,node_identity=context['ordering'],active_nodes=old.active_nodes,gbar_nS=old.gbar_nS,
        reversal_mV=old.reversal_mV,leak_reversal_mV=old.leak_reversal_mV,channel_provenance=old.channel_provenance,
        calcium_port=port,preparation=prior['preparation'])
    pn.load_state_dict(read_state(parent)['pn'])
    om=read(R/'evidence/pn_terminal_tail_20260912/orn_input/mapping.npz')
    allocation=SpatialOrnAllocation(om['source_ids'],om['contact_pre_ids'],om['contact_joint_nodes'],om['pair_shares'],full_size=context['N'])
    mapping=read(R/'work/pn_orn_response_20260913/site_KC_map.npz')
    source=OnlineOrnPnSource(pn,allocation,spec['caps_hz'],mapping,cns_time_ns=spec['cns_origin_ns'],preparation=spec['preparation'],gain_provenance=spec['gain_provenance'])
    return source,context
