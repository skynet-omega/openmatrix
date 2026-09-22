"""Regional KC-gamma axonal hypothesis on the unchanged parent CNS graph.

Only known gL fractions KCg->KCg leave the fast somatodendritic current; their
contact counts feed local axonal ACh modulation. Known gL output fractions to
PAM08/APL/MBON05 use a separate filtered axonal release. Calyx, other, unknown
and unmapped fractions retain their parent transmission. This assignment is a
declared uncalibrated hypothesis, not proof of muscarinic-only transmission.

Every solver stage evaluates soma q, x, b and s_ax from the same state. The
complete bypass preserves parent coefficients and error control exactly while
freezing appended states. eta=0 retains the regional routing and axonal filter.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import numba
import numpy as np

from kcgamma_axon_candidate import AxonParameters
from prosthetic_olfactory_brain import (
    ProstheticOlfactoryBrain, GpuProstheticOlfactoryBrain, ORN_PN_KEYS)
from synaptic_visual_brain import _hash_array

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT/'config/kcgamma_regional_candidate_v1.json'
POLICY = 'kcgamma_regional_axonal_hypothesis_v1'
REGIONAL_KEYS = ORN_PN_KEYS | {'regional_manifest'}
OUTPUT_TYPES = ('PAM08', 'APL', 'MBON05')


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def _record_hash(record):
    def encode(x):
        if isinstance(x,np.ndarray):return {'array_sha256':_hash_array(x)}
        if isinstance(x,dict):return {k:encode(v) for k,v in x.items() if k!='record_sha256'}
        if isinstance(x,(tuple,list)):return [encode(v) for v in x]
        if isinstance(x,np.generic):return x.item()
        return x
    return hashlib.sha256(json.dumps(encode(record),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _path(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p


def _map(value, expected_hash=None):
    if isinstance(value,(str,Path)):
        p=_path(value);digest=_sha(p)
        if expected_hash is not None and digest!=expected_hash:
            raise ValueError('Regional routing artifact differs from its receipt')
        with np.load(p,allow_pickle=False) as f:result={k:f[k].copy() for k in f.files}
        return result,dict(path=str(p.resolve()),sha256=digest)
    if not isinstance(value,dict):raise ValueError('Regional routing must be a path or explicit array mapping')
    result=copy.deepcopy(value)
    return result,dict(in_memory_array_sha256=_record_hash(result))


def _integer(value,name):
    a=np.asarray(value)
    if (a.ndim!=1 or a.dtype.kind not in 'iu' or
            (a.dtype.kind=='u' and np.any(a>np.iinfo(np.int64).max))):
        raise ValueError('Invalid regional integer vector: '+name)
    return a.astype(np.int64,copy=True)


def _selection(reference, raw, target_type):
    b=reference.brain;types=reference.measured_t4_manifest['canonical_node_types']
    names=('pre_id','post_id','csr_position','total_contacts','gamma_contacts','calyx_contacts','other_contacts')
    a={k:_integer(raw[k],k) for k in names};n=len(a['pre_id'])
    if any(len(v)!=n for v in a.values()):raise ValueError('Regional edge vectors differ in size')
    known=np.asarray(raw.get('roi_known',np.ones(n,dtype=bool)))
    if known.dtype!=np.dtype(bool) or known.shape!=(n,):raise ValueError('Invalid regional coverage mask')
    pre=np.searchsorted(b.node_ids,a['pre_id']);post=np.searchsorted(b.node_ids,a['post_id']);pos=a['csr_position']
    if (np.any(pre>=b.n_neurons) or np.any(post>=b.n_neurons) or
            not np.array_equal(b.node_ids[pre],a['pre_id']) or not np.array_equal(b.node_ids[post],a['post_id']) or
            np.any((pos<0)|(pos>=b.W.nnz)) or len(np.unique(pos))!=n):
        raise ValueError('Unknown or duplicate regional neural/edge identities')
    if (not np.array_equal(b.W.indices[pos],pre) or np.any(pos<b.W.indptr[post]) or np.any(pos>=b.W.indptr[post+1])):
        raise ValueError('Regional CSR position is not post <- pre')
    if (not np.char.startswith(types[pre],'KCg').all() or
            not (np.char.startswith(types[post],'KCg').all() if target_type=='KCgamma' else np.all(types[post]==target_type)) or
            np.any(reference.visual_mask[pre]) or np.any(reference.visual_mask[post])):
        raise ValueError('Regional mapping falls outside the declared rate-mode cell types')
    if (np.any(a['total_contacts']<=0) or
            any(np.any(a[k][known]<0) for k in names[-3:]) or
            not np.array_equal((a['gamma_contacts']+a['calyx_contacts']+a['other_contacts'])[known],a['total_contacts'][known])):
        raise ValueError('Regional contacts do not form a conserved partition')
    if any(np.any(a[k][~known]!=-1) for k in names[-3:]):
        raise ValueError('Unknown regions must remain explicitly unassigned')
    # Anatomical emitter identity and the actual inherited efficacy are checked
    # separately. The adapter never manufactures a negative synaptic sign.
    if not hasattr(b,'nt_labels') or not np.all(np.char.lower(np.char.strip(b.nt_labels[pre]))=='acetylcholine'):
        raise ValueError('KC regional ACh routing requires canonical cholinergic emitters')
    if np.any(reference.weights64[pos]<0.) or not np.isfinite(reference.weights64[pos]).all():
        raise ValueError('Regional routing requires finite nonnegative inherited KC weights')
    blocked=np.r_[reference.pvlp_adaptation_manifest['target_rows'],reference.orn_pn_synaptic_manifest['post_indices']]
    if np.intersect1d(post,blocked).size:raise ValueError('Regional targets overlap existing local replacements')
    fraction=np.where(known,a['gamma_contacts']/a['total_contacts'],0.)
    for name in ('gamma_fraction','axonal_fraction'):
        if name in raw and not np.array_equal(np.asarray(raw[name])[known],fraction[known]):
            raise ValueError('Regional fraction disagrees with contact counts')
    order=np.argsort(pos)
    a.update(pre_index=pre.astype(np.int64),post_index=post.astype(np.int64),roi_known=known.copy(),gamma_fraction=fraction)
    return {k:v[order].copy() for k,v in a.items()}


@numba.njit(fastmath=False,cache=True)
def _regional_currents(rows,local_ptr,edge_mode,fraction,source_slot,indptr,indices,weights,
                       old_s,axonal_s,caps,visual,gain,theta,drive,connected,target):
    for j in range(len(rows)):
        row=rows[j];current=0.
        for edge in range(indptr[row],indptr[row+1]):
            col=indices[edge]
            if connected or not visual[col]:
                k=local_ptr[j]+edge-indptr[row];mode=edge_mode[k]
                release=old_s[col]
                if mode==1:release*=1.-fraction[k]
                elif mode==2:release+=fraction[k]*(axonal_s[source_slot[k]]-release)
                current+=weights[edge]*(release*caps[col])
        target[row]=max(0.,math.tanh(gain[row]*(current+drive[row]-theta[row])))


@numba.njit(fastmath=False,cache=True)
def _axonal_input(rows,ptr,pre_slot,contacts,caps,release):
    result=np.empty(len(rows),dtype=np.float64)
    for j in range(len(rows)):
        u=0.
        for edge in range(ptr[j],ptr[j+1]):
            pre=pre_slot[edge]
            u+=contacts[edge]*(release[pre]*caps[rows[pre]])
        result[j]=u
    return result


@numba.njit(fastmath=False,cache=True)
def _axonal_targets(rows,ptr,pre_slot,contacts,caps,soma,x,b,sax,K,eta,x50,slope,dependent):
    k=len(rows);target=np.empty(3*k,dtype=np.float64)
    inputs=_axonal_input(rows,ptr,pre_slot,contacts,caps,sax)
    for j in range(k):
        u=inputs[j]
        occupied=0. if u==0. else 1./(1.+K/u)
        gate=1.
        if dependent:
            z=(x[j]-x50)/slope
            if z>=0.:
                e=math.exp(-z);gate=e/(1.+e)
            else:
                e=math.exp(z);gate=1./(1.+e)
        target[j]=soma[rows[j]]/(1.+eta*b[j])
        target[k+j]=occupied*gate
        target[2*k+j]=x[j]
    return target


REGIONAL_CUDA_SOURCE=r'''
extern "C" __global__ void regional_current(
 int k,const long long* rows,const long long* local_ptr,const signed char* mode,
 const double* fraction,const int* slot,const long long* ptr,const int* idx,
 const double* w,const double* s,const double* sax,const double* caps,const bool* visual,
 const double* gain,const double* theta,const double* drive,bool connected,double* target){
 int j=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(j>=k)return;
 long long row=rows[j];double current=0.;
 for(long long edge=ptr[row]+lane;edge<ptr[row+1];edge+=32){
  int col=idx[edge];if(connected||!visual[col]){
   long long a=local_ptr[j]+edge-ptr[row];double release=s[col];
   if(mode[a]==1)release*=1.-fraction[a];
   else if(mode[a]==2)release+=fraction[a]*(sax[slot[a]]-release);
   current+=w[edge]*(release*caps[col]);}}
 for(int d=16;d>0;d/=2)current+=__shfl_down_sync(0xffffffff,current,d);
 if(lane==0)target[row]=fmax(0.,tanh(gain[row]*(current+drive[row]-theta[row])));
}
extern "C" __global__ void regional_axon(
 int k,const long long* rows,const long long* ptr,const int* pre,const double* contacts,
 const double* caps,const double* soma,const double* x,const double* b,const double* sax,
 double K,double eta,double x50,double slope,bool dependent,double* target){
 int j=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(j>=k)return;
 double u=0.;for(long long e=ptr[j]+lane;e<ptr[j+1];e+=32){int p=pre[e];u+=contacts[e]*(sax[p]*caps[rows[p]]);}
 for(int d=16;d>0;d/=2)u+=__shfl_down_sync(0xffffffff,u,d);
 if(lane==0){double occupied=u==0.?0.:1./(1.+K/u),gate=1.;
  if(dependent){double z=(x[j]-x50)/slope,e;if(z>=0.){e=exp(-z);gate=e/(1.+e);}else{e=exp(z);gate=1./(1.+e);}}
  target[j]=soma[rows[j]]/(1.+eta*b[j]);target[k+j]=occupied*gate;target[2*k+j]=x[j];}
}
'''


class KcGammaRegionalBrain(ProstheticOlfactoryBrain):
    SCHEMA='matrix_kcgamma_regional_brain_fp64_v1'
    REGIONAL_PARENT_CLASS=ProstheticOlfactoryBrain

    @classmethod
    def adopt(cls,reference,config=DEFAULT_CONFIG,*,input_routing=None,output_routings=None,
              enabled=True,eta=None,activity_dependent=True):
        if reference.SCHEMA!=cls.REGIONAL_PARENT_CLASS.SCHEMA or not isinstance(reference,cls.REGIONAL_PARENT_CLASS):
            raise ValueError('Regional adoption requires the exact prosthetic parent backend')
        if type(enabled) is not bool or type(activity_dependent) is not bool:raise ValueError('Explicit regional flags required')
        if isinstance(config,(str,Path)):
            p=_path(config);plan=json.loads(p.read_text());config_source=dict(path=str(p.resolve()),sha256=_sha(p))
        else:
            plan=copy.deepcopy(config);config_source=dict(in_memory_sha256=_record_hash(plan))
        if plan.get('schema')!='matrix_kcgamma_regional_candidate_plan_v1':raise ValueError('Unknown regional configuration schema')
        parameters=dict(plan['parameters'])
        if eta is not None:parameters['eta']=eta
        parameters=AxonParameters(**parameters).as_dict()
        expected_input=None;sources={}
        if input_routing is None:
            input_routing=plan['input_routing']
            receipt_path=_path(plan['input_routing_receipt']);receipt=json.loads(receipt_path.read_text())
            expected_input=receipt['artifact']['sha256'];sources['input_receipt']=dict(path=str(receipt_path),sha256=_sha(receipt_path))
        raw,sources['input']=_map(input_routing,expected_input)
        incoming=_selection(reference,raw,'KCgamma')
        expected_outputs={}
        if output_routings is None:
            receipt_path=_path(plan['output_routing_receipt']);receipt=json.loads(receipt_path.read_text())
            output_routings={name:receipt['groups'][name]['artifact']['path'] for name in OUTPUT_TYPES}
            expected_outputs={name:receipt['groups'][name]['artifact']['sha256'] for name in OUTPUT_TYPES}
            sources['output_receipt']=dict(path=str(receipt_path),sha256=_sha(receipt_path))
        if set(output_routings)!=set(OUTPUT_TYPES):raise ValueError('Exactly PAM08/APL/MBON05 output maps required')
        outgoing={}
        for name in OUTPUT_TYPES:
            raw,sources[name]=_map(output_routings[name],expected_outputs.get(name))
            outgoing[name]=_selection(reference,raw,name)
        rows=np.flatnonzero(np.char.startswith(reference.measured_t4_manifest['canonical_node_types'],'KCg')).astype(np.int64)
        if not len(rows) or np.any(reference.visual_mask[rows]):raise ValueError('Canonical rate-mode KC-gamma cells required')
        saved=reference.state_dict();prefix=saved['state'];q=reference.release();s=reference.transmission_release()
        initial=np.r_[q[rows],np.zeros(len(rows)),s[rows]]
        record=dict(policy=POLICY,source_schema=reference.SCHEMA,enabled=enabled,activity_dependent=activity_dependent,
            parameters=parameters,eta=parameters['eta'],config=plan,config_source=config_source,sources=sources,
            target_rows=rows,target_ids=reference.brain.node_ids[rows].copy(),input_selection=incoming,output_selections=outgoing,
            adoption_time_ns=int(reference.time_ns),parent_state_size=len(prefix),
            initial_parent_state_sha256=_hash_array(prefix),initial_new_state_sha256=_hash_array(initial),
            node_ids_sha256=_hash_array(reference.brain.node_ids),canonical_types_sha256=_hash_array(reference.measured_t4_manifest['canonical_node_types']),
            anatomical_indptr_sha256=_hash_array(reference.brain.W.indptr),anatomical_indices_sha256=_hash_array(reference.brain.W.indices),
            adoption_archived_weights_sha256=_hash_array(reference.brain.W.data),adoption_effective_weights_sha256=_hash_array(reference.weights64),
            axonal_synaptic_tau_s=float(reference.parameters['synaptic_tau_s']),
            initialization='x=q_soma; b=0; s_ax=old_s, same initialization for every eta. Earlier axonal/receptor history is unknown.',
            equations='u=sum(gL_contacts*s_ax_pre*rmax_pre); target_x=q_soma/(1+eta*b); target_b=u/(K+u)*h(x); target_s_ax=x',
            bypass='All inherited currents and solver norm restored exactly; appended states frozen with rate zero.',
            region_scope='Known KCg->KCg gL leaves fast current; known KCg->PAM08/APL/MBON05 gL uses s_ax; all other fractions unchanged.',
            parameter_status='Explicit uncalibrated compartment/kinetic hypothesis; no observed behavioral fitting.',
            biological_validation=False)
        record['record_sha256']=_record_hash(record)
        saved.update(schema=cls.SCHEMA,regional_manifest=record,state=np.r_[prefix,initial])
        obj=cls.from_state(reference.brain,saved)
        if not np.array_equal(obj.weights64,reference.weights64):
            raise ValueError('Parent effective weights include a nonpersistent external override')
        return obj

    @property
    def regional_parent_state_size(self):return int(self.regional_manifest['parent_state_size'])
    @property
    def regional_rows(self):return self.regional_manifest['target_rows'].copy()
    @property
    def regional_ids(self):return self.regional_manifest['target_ids'].copy()
    @property
    def axonal_state(self):return self.state[self.regional_parent_state_size:].reshape(3,-1).copy()
    @property
    def orn_synaptic_state(self):return self.state[self.parent_state_size:self.regional_parent_state_size].reshape(4,-1).copy()

    def regional_ach_input(self,state=None):
        """Pure CPU contact-Hz observation; bypass observes inherited release.

        In bypass this is a diagnostic projection only, never an applied drive.
        The adapter's frozen x/s_ax states are not its bypass emission.
        """
        y=self.state if state is None else np.asarray(state)
        if y.shape!=self.state.shape or y.dtype!=np.float64:
            raise ValueError('Regional observation requires the complete FP64 state')
        c=self._regional_cache;rows=c['kc_rows']
        release=(y[self.regional_parent_state_size:].reshape(3,-1)[2]
                 if self.regional_manifest['enabled'] else y[self.transmission_start+rows])
        if not np.isfinite(release).all() or np.any((release<0.)|(release>1.)):
            raise ValueError('Invalid regional release observation')
        return _axonal_input(rows,c['axon_ptr'],c['axon_pre'],c['axon_contacts'],self.caps,release)

    def _norm_size(self):
        if self.regional_manifest['enabled']:return len(self.state)
        if self.orn_synaptic_enabled:return self.regional_parent_state_size
        return self.parent_state_size if self.adaptation_active else self.inherited_state_size

    def _build_regional_cache(self):
        m=self.regional_manifest;b=self.brain;kc=m['target_rows']
        selections=[m['input_selection']]+[m['output_selections'][name] for name in OUTPUT_TYPES]
        active=[(a,a['roi_known']&(a['gamma_contacts']>0)) for a in selections]
        rows=np.unique(np.concatenate([a['post_index'][keep] for a,keep in active])).astype(np.int64)
        ptr=np.r_[0,np.cumsum(b.W.indptr[rows+1]-b.W.indptr[rows])].astype(np.int64)
        mode=np.zeros(ptr[-1],dtype=np.int8);fraction=np.zeros(ptr[-1]);slot=np.full(ptr[-1],-1,dtype=np.int32)
        for j,(a,keep) in enumerate(active):
            post=a['post_index'][keep];positions=a['csr_position'][keep]
            local=ptr[np.searchsorted(rows,post)]+positions-b.W.indptr[post]
            mode[local]=1 if j==0 else 2;fraction[local]=a['gamma_fraction'][keep]
            slot[local]=np.searchsorted(kc,a['pre_index'][keep]).astype(np.int32)
        a,keep=active[0];post_local=np.searchsorted(kc,a['post_index'][keep])
        axon_ptr=np.r_[0,np.cumsum(np.bincount(post_local,minlength=len(kc)))].astype(np.int64)
        self._regional_cache=dict(rows=rows,local_ptr=ptr,mode=mode,fraction=fraction,slot=slot,
            kc_rows=kc.copy(),axon_ptr=axon_ptr,axon_pre=np.searchsorted(kc,a['pre_index'][keep]).astype(np.int32),
            axon_contacts=a['gamma_contacts'][keep].astype(np.float64))
        p=m['parameters'];k=len(kc)
        self._regional_rates=np.r_[np.full(k,1./p['tau_x_s']),np.full(k,1./p['tau_b_s']),np.full(k,1./m['axonal_synaptic_tau_s'])]

    def _coefficients(self,state,drive,light):
        size=self.regional_parent_state_size;m=self.regional_manifest
        target,rate=ProstheticOlfactoryBrain._coefficients(self,state[:size],drive,light)
        if not m['enabled']:
            return np.r_[target,state[size:]],np.r_[rate,np.zeros(len(state)-size)]
        c=self._regional_cache;p=m['parameters'];x,b,sax=state[size:].reshape(3,-1)
        _regional_currents(c['rows'],c['local_ptr'],c['mode'],c['fraction'],c['slot'],
            self.brain.W.indptr,self.brain.W.indices,self.weights64,
            state[self.transmission_start:self.inherited_state_size],sax,self.caps,self.visual_mask,
            self.rate_gain,self.rate_theta,drive,self.visual_output_connected,target)
        new=_axonal_targets(c['kc_rows'],c['axon_ptr'],c['axon_pre'],c['axon_contacts'],self.caps,
            state,x,b,sax,p['K_ach_contact_hz'],p['eta'],p['x50'],p['slope'],m['activity_dependent'])
        return np.r_[target,new],np.r_[rate,self._regional_rates]

    def state_dict(self):
        saved=ProstheticOlfactoryBrain.state_dict(self)
        saved['regional_manifest']=copy.deepcopy(self.regional_manifest)
        return saved

    @classmethod
    def from_state(cls,brain,saved):
        if not isinstance(saved,dict) or set(saved)!=REGIONAL_KEYS or saved.get('schema')!=cls.SCHEMA:
            raise ValueError('Incomplete regional state or wrong backend schema')
        m=saved['regional_manifest'];state=saved['state']
        if (not isinstance(m,dict) or m.get('policy')!=POLICY or m.get('source_schema')!=cls.REGIONAL_PARENT_CLASS.SCHEMA or
                type(m.get('enabled')) is not bool or type(m.get('activity_dependent')) is not bool or
                m.get('record_sha256')!=_record_hash(m) or m.get('biological_validation') is not False):
            raise ValueError('Changed regional mechanism, parameters or provenance')
        AxonParameters(**m['parameters'])
        if m.get('eta')!=m['parameters']['eta']:
            raise ValueError('Regional effective eta differs from its equations')
        types=saved['measured_t4_manifest']['canonical_node_types']
        rows=np.flatnonzero(np.char.startswith(types,'KCg')).astype(np.int64)
        size=2*brain.n_neurons+2*len(saved['photo_ids'])+len(saved['pvlp_adaptation_manifest']['target_rows'])+4*len(saved['orn_pn_synaptic_manifest']['source_orn_indices'])
        if (not np.array_equal(m['target_rows'],rows) or not np.array_equal(m['target_ids'],brain.node_ids[rows]) or
                m['parent_state_size']!=size or not isinstance(state,np.ndarray) or state.dtype!=np.float64 or
                state.shape!=(size+3*len(rows),) or not np.isfinite(state).all() or np.any((state<0.)|(state>1.)) or
                type(m.get('adoption_time_ns')) is not int or not 0<=m['adoption_time_ns']<=saved['time_ns'] or
                m['axonal_synaptic_tau_s']!=saved['parameters']['synaptic_tau_s']):
            raise ValueError('Invalid regional state layout, identities or clock')
        parent={k:copy.deepcopy(saved[k]) for k in ORN_PN_KEYS}
        parent.update(schema=cls.REGIONAL_PARENT_CLASS.SCHEMA,state=state[:size].copy())
        base=cls.REGIONAL_PARENT_CLASS.from_state(brain,parent)
        for name,array in [('node_ids_sha256',brain.node_ids),('canonical_types_sha256',types),
                ('anatomical_indptr_sha256',brain.W.indptr),('anatomical_indices_sha256',brain.W.indices)]:
            if m[name]!=_hash_array(array):raise ValueError('Regional anatomical identity changed: '+name)
        for target,a in [('KCgamma',m['input_selection'])]+list(m['output_selections'].items()):
            if target not in ('KCgamma',)+OUTPUT_TYPES:raise ValueError('Unknown regional output class')
            actual=_selection(base,a,target)
            if set(actual)!=set(a) or any(not np.array_equal(v,a[k]) for k,v in actual.items()):
                raise ValueError('Regional stored selection differs from canonical mapping')
        if set(m['output_selections'])!=set(OUTPUT_TYPES):raise ValueError('Missing regional output map')
        if saved['time_ns']==m['adoption_time_ns']:
            initial=np.r_[base.release()[rows],np.zeros(len(rows)),base.transmission_release()[rows]]
            if (not np.array_equal(state[size:],initial) or m['initial_parent_state_sha256']!=_hash_array(state[:size]) or
                    m['initial_new_state_sha256']!=_hash_array(initial) or m['adoption_archived_weights_sha256']!=_hash_array(brain.W.data) or
                    m['adoption_effective_weights_sha256']!=_hash_array(base.weights64)):
                raise ValueError('Regional adoption changed inherited state or declared initialization')
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.state=state.copy();obj.regional_manifest=copy.deepcopy(m);obj._build_regional_cache()
        return obj


class GpuKcGammaRegionalBrain(KcGammaRegionalBrain,GpuProstheticOlfactoryBrain):
    SCHEMA='matrix_kcgamma_regional_brain_fp64_cuda_v1'
    REGIONAL_PARENT_CLASS=GpuProstheticOlfactoryBrain

    def _build_regional_cache(self):
        super()._build_regional_cache()
        import cupy as cp
        self._regional_cuda={k:cp.asarray(v) for k,v in self._regional_cache.items()}
        self._regional_cuda['rates']=cp.asarray(self._regional_rates)
        options=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true')
        self._regional_current_kernel=cp.RawKernel(REGIONAL_CUDA_SOURCE,'regional_current',options=options)
        self._regional_axon_kernel=cp.RawKernel(REGIONAL_CUDA_SOURCE,'regional_axon',options=options)

    def coefficients_gpu(self,state,drive,light):
        import cupy as cp
        size=self.regional_parent_state_size;m=self.regional_manifest
        target,rate=GpuProstheticOlfactoryBrain.coefficients_gpu(self,state[:size],drive,light)
        if not m['enabled']:
            return cp.concatenate((target,state[size:])),cp.concatenate((rate,cp.zeros(len(state)-size)))
        r=self._regional_cuda;c=self.cuda;p=m['parameters'];k=len(self.regional_rows)
        x,b,sax=state[size:].reshape(3,-1);n=len(self._regional_cache['rows'])
        if n:
            args=(np.int32(n),r['rows'],r['local_ptr'],r['mode'],r['fraction'],r['slot'],c['indptr'],c['indices'],
                c['weights'],state[self.transmission_start:self.inherited_state_size],sax,c['caps'],c['visual'],c['gain'],c['theta'],drive,
                np.bool_(self.visual_output_connected),target)
            self._regional_current_kernel(((n*32+255)//256,),(256,),args)
        new=cp.empty(3*k,dtype=cp.float64)
        args=(np.int32(k),r['kc_rows'],r['axon_ptr'],r['axon_pre'],r['axon_contacts'],c['caps'],state,x,b,sax,
            np.float64(p['K_ach_contact_hz']),np.float64(p['eta']),np.float64(p['x50']),np.float64(p['slope']),
            np.bool_(m['activity_dependent']),new)
        self._regional_axon_kernel(((k*32+255)//256,),(256,),args)
        return cp.concatenate((target,new)),cp.concatenate((rate,r['rates']))

    @staticmethod
    def backend_identity():
        info=GpuProstheticOlfactoryBrain.backend_identity()
        info['kcgamma_regional']='Known gL compartment fractions; x/b/s_ax stage-consistent FP64 fixed warp reductions; complete bypass freezes extras and restores parent norm.'
        return info
