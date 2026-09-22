"""Small reproducible falsifiers. Explicit exceptions remain active under python -O."""
from pathlib import Path
import sys,copy,json,hashlib,time,shutil
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from model import Model,require,Unsupported
from runtime import Engine
from development import base,hh

HERE=Path(__file__).resolve().parent

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def rejects(fn):
    try:fn()
    except (ValueError,KeyError):return True
    raise RuntimeError('expected rejection did not occur')

def check_models(out,algorithm='A'):
    records=[]
    for name,spec,stop,method in [('recurrent',base(32),.1,'DOP853'),('hh',hh(),.02,'Radau')]:
        m=Model(spec);ts=np.linspace(0,stop,21)
        ref=solve_ivp(m.rhs,(0,stop),m.initial,method=method,rtol=1e-11,atol=1e-13*m.scale,t_eval=ts)
        finer=solve_ivp(m.rhs,(0,stop),m.initial,method=method,rtol=1e-12,atol=1e-14*m.scale,t_eval=ts)
        require(ref.success and finer.success,'reference failed')
        refinement=float(np.max(np.abs(ref.y-finer.y)/(m.scale[:,None]+np.abs(finer.y))))
        require(refinement<=1e-7,'reference not converged')
        for mode,limit in [('precise',1e-5),('fast',.01)]:
            e=Engine(spec,profile=mode,algorithm=algorithm)
            # Different states and times, including the removable HH singularity.
            inputs=[m.initial,m.initial+.0001]
            if name=='hh':
                singular=m.initial.copy();singular[m.slices[('axon','v')]]=-40;inputs.append(singular)
            rhs_error=0.
            for y in inputs:
                cpu=m.rhs(.0123,y);gpu=e.gpu.rhs_read(.0123,y)
                rhs_error=max(rhs_error,float(np.max(np.abs(cpu-gpu)/(1+np.abs(cpu)))))
            require(rhs_error<=1e-11,'CPU/GPU compiled RHS differs')
            if name=='hh':
                with e.gpu.stream:
                    e.gpu.rhs(e.gpu.x,0,e.gpu.k[0],True);diagonal=e.gpu.diag.get(stream=e.gpu.stream)
                numeric=[]
                for i in range(m.n):
                    y=m.initial.copy();eps=1e-6*m.scale[i];y[i]+=eps;a=m.rhs(0,y)[i];y[i]-=2*eps;b=m.rhs(0,y)[i];numeric.append((a-b)/(2*eps))
                require(np.max(np.abs(np.asarray(numeric)-diagonal)/(1+np.abs(diagonal)))<1e-7,'generated diagonal derivative')
            ys=[e.read()];start=time.perf_counter()
            for t in ts[1:]:e.advance(float(t));ys.append(e.read())
            wall=time.perf_counter()-start;y=np.asarray(ys).T
            err=float(np.max(np.abs(y-finer.y)/(m.scale[:,None]+np.abs(finer.y))))
            np.savez_compressed(out/f'{name}_{algorithm}_{mode}.npz',times=ts,candidate=y,reference=finer.y,reference_coarse=ref.y,scale=m.scale)
            records.append({'name':name,'mode':mode,'algorithm':algorithm,'global_scaled_error':err,'limit':limit,'reference_refinement':refinement,'rhs_scaled_error':rhs_error,'wall_s':wall,'steps':e.gpu.accepted,'rejected':e.gpu.rejected})
            records[-1]['eligible_trajectory'] = err<=limit  # Failure is reported, never rescued by changing tolerance.
    return records

def interventions(out):
    spec=base(12);e=Engine(spec);e.advance(.004)
    before=e.read();h=e.next_h;ident=e.model.identity
    bad=copy.deepcopy(spec);bad['connections'][0]['delay']=.01
    rejects(lambda:e.replace(bad))
    require(np.array_equal(before,e.read()) and e.next_h==h and e.model.identity==ident,'failed edit modified live simulation')
    # Commit parameter change, clamp, add a population, remove a population and edges.
    changed=copy.deepcopy(spec);changed['populations'][0]['parameters']['drive']['value']=.3
    changed['clamps']=[{'population':'cells','state':'v','cells':[2,5],'value':-.2}]
    extra=copy.deepcopy(changed['populations'][0]);extra['id']='added';extra['count']=3;changed['populations'].append(extra)
    e.replace(changed)
    require(e.scan('cells','v',[2,5])['values']==[-.2,-.2],'clamp not initialized')
    require(np.array_equal(e.read()[12:24],before[12:24]),'retained states not preserved')
    e.advance(.008)
    require(e.scan('cells','v',[2,5])['values']==[-.2,-.2],'clamp not conserved')
    removed=copy.deepcopy(changed);removed['populations']=removed['populations'][:1];removed['connections']=[]
    old=e.read();e.replace(removed);require(np.array_equal(e.read(),old[:24]),'deletion state migration')
    # A->B->A descriptor invalidation independent of preserved dynamical state.
    e.replace(spec);reference=Model(spec);cpu=reference.rhs(e.t,e.read());gpu=e.gpu.rhs_read(e.t,e.read())
    require(np.allclose(cpu,gpu,rtol=1e-12,atol=1e-12),'stale graph after topology replacement')
    e.checkpoint(out/'checkpoint');restarted=Engine.restore(out/'checkpoint')
    e.advance(.02);restarted.advance(.02)
    require(np.array_equal(e.read(),restarted.read()) and e.t==restarted.t and e.next_h==restarted.next_h,'checkpoint replay not exact')
    scan=e.scan('cells','v',[0,3]);write(out/'scan.json',scan)
    # Corrupt actual checkpoint metadata and actual state bytes, not a stored pass flag.
    shutil.copytree(out/'checkpoint',out/'corrupt_metadata');p=out/'corrupt_metadata'/'checkpoint.json'
    meta=json.loads(p.read_text());meta['time']+=.01;write(p,meta)
    rejects(lambda:Engine.restore(out/'corrupt_metadata'))
    shutil.copytree(out/'checkpoint',out/'corrupt_state');p=out/'corrupt_state'/'state.npy'
    x=np.load(p);x[0]+=.1;np.save(p,x);rejects(lambda:Engine.restore(out/'corrupt_state'))
    return {'failed_edit_rollback':True,'parameter_add_delete_clamp':True,'ABA_recompiled_RHS':True,'checkpoint_bit_exact':True,'metadata_and_state_corruption_detected':True,'transactions':e.edits}

def domains(out):
    bad=base();bad['capabilities']=['stochastic'];rejects(lambda:Model(bad))
    bad=base();bad['populations'][0]['states']['v']['rhs']='tau+v';rejects(lambda:Model(bad))
    bad=base();bad['populations'][0]['states']['v']['rhs']="__import__('os')";rejects(lambda:Model(bad))
    # General non-diagonal invertible mass: independently compare to expm.
    spec={'version':1,'populations':[{'id':'general','count':2,'states':{'x':{'unit':'1','scale':1,'initial':[1.,-.5],'rhs':'-rate*x'}},'parameters':{'rate':{'unit':'1/s','value':3}}}],
          'mass':{'row':[0,0,1,1],'col':[0,1,0,1],'values':[2.,.2,.1,1.]}}
    e=Engine(spec,backend='cpu');e.cpu_method='Radau';e.advance(.1)
    exact=expm(-.3*np.linalg.inv(np.array([[2.,.2],[.1,1.]])))@np.array([1.,-.5])
    err=float(np.max(np.abs(e.read()-exact)));require(err<1e-7,'general mass CPU reference')
    rejects(lambda:Engine(spec,backend='gpu'))
    spec['mass']['values']=[1.,1.,1.,1.];rejects(lambda:Engine(spec,backend='cpu'))
    # Generic cycles beyond old PN core bound. Compare only RHS; small trajectories elsewhere.
    cyc=base(96,95);e=Engine(cyc);y=e.read();err_cycle=float(np.max(np.abs(e.gpu.rhs_read(0,y)-e.model.rhs(0,y))))
    require(err_cycle<1e-11,'general cyclic projection RHS')
    # Nonfinite candidate trial must not commit any state.
    bad=base(4);bad['populations'][0]['parameters']['drive']['value']=1e308
    e=Engine(bad);before=e.read();rejects(lambda:e.gpu.attempt(0,1))
    require(np.array_equal(before,e.read()),'failed numerical step committed')
    return {'unsupported_noise_delay_units_code_rejected':True,'singular_mass_rejected':True,'CPU_general_mass_error':err,'GPU_general_mass_rejected':True,'clique96_RHS_error':err_cycle,'nonfinite_trial_rollback':True}

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result={'models':check_models(out,'A')+check_models(out,'C'),'interventions':interventions(out),'domains':domains(out)}
    result['total_wall_s']=time.perf_counter()-start;write(out/'RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
