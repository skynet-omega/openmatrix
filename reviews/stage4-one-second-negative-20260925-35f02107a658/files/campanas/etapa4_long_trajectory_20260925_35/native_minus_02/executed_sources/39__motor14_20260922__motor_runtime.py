"""Explicit loader; module name avoids the organism's historical runtime module."""
from pathlib import Path
import importlib.util
import sys
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

def load(out):
    import cupy as cp
    import pandas as pd
    import mujoco as mj
    import matrix_olfactory_diagnostic as d
    for rel in ('work/stage4_antennal_contact_adapter_20260915','work/stage3_static_lateral_field_20260916'):
        sys.path.insert(0,str(ROOT/rel))
    from antennal_runtime import AntennalContactRuntime
    from static_field import StaticLateralField
    from pn_cns_ports import PnCnsPorts
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    plan=dict(d.leer_plan(ROOT));plan.update(escala_ipsilateral=1.,escala_contralateral=1.)
    obj=AntennalContactRuntime.load(ROOT/plan['checkpoint'])
    try:
        table=pd.read_parquet(ROOT/'data/male_v10/nodes.parquet')
        controller=d.extraer_controlador(ROOT/plan['parent_script'],cp,np,mj)
        d.preparar_candidata(obj,table,plan,cp,np,controller,out)
        field=obj.core.world.boundary
        d.instalar_campo(obj,StaticLateralField,field,'sham',0.)
        return obj,d,plan,StaticLateralField,field,PnCnsPorts(obj.core.hybrid,10208)
    except BaseException:
        obj.close();raise

def dump(path,value):
    with Path(path).open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

if __name__=='__main__':
    import cupy as cp
    import resource,time
    start=time.perf_counter();obj,*_=load(sys.argv[1]);h=obj.core.hybrid;s=h._online_source
    try:
        info={'parameters':h.parameters,'coupling_ns':h.pn_online_manifest['coupling_ns'],
              'kc_spatial_schedule':{k:h.kc_spatial_manifest[k] for k in ('inner_step_ns','coupling_step_ns')},
              'hybrid_shape':list(h.state.shape),'hybrid_bytes':h.state.nbytes,
              'pn_voltage_shape':list(s.pn.voltage.shape),'pn_active_nodes':len(s.pn.active_nodes),
              'ca_nodes':len(s.pn.calcium_port.nodes),'neural_statistics':h.statistics,
              'next_step_ns':h.next_step_ns,'time_ns':h.time_ns,'gpu_pool_bytes':cp.get_default_memory_pool().total_bytes(),
              'rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'load_wall_s':time.perf_counter()-start,
              'neural_steps_executed':0}
        dump(Path(sys.argv[1])/'INSPECTION.json',info);print(json.dumps(info,indent=2))
    finally:obj.close()
