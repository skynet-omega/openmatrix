"""Six-row actual-consumer capture. No solver, state or motor policy changes."""
from pathlib import Path
import inspect
import numpy as np
import cupy as cp
import fp32_operator
import real_model
from graph_runtime import GraphRK23

HERE=Path(__file__).resolve().parent
IDS=np.array([10045,10056,10118,10065,523769,10360],np.int64)
ROWS=np.array([36,46,104,52,131957,332],np.int32)
FIELDS=('state','net','positive_aux','negative_aux','drive','theta','gain',
        'base_target','base_rate','tau','margin','final_target','final_rate',
        'derivative','evaluation_time_s','stage_fraction')
OPERANDS=('weight_consumed','transmission_consumed','cap_consumed','included')
CAPACITY=256
WIDTH=404
OPTIONS=('--std=c++17','--fmad=false','--prec-div=true','--prec-sqrt=true')
def need(x,msg):
    if not x:raise ValueError(msg)
class Observer:
    def __init__(self,brain):
        self.brain=brain
        self.rows=np.searchsorted(brain.brain.node_ids,IDS).astype(np.int32)
        need(np.array_equal(self.rows,ROWS) and np.array_equal(brain.brain.node_ids[self.rows],IDS),'Observer identity')
        need(not brain.visual_mask[self.rows].any(),'Only generic-rate destinations')
        ptr,idx=brain.brain.W.indptr,brain.brain.W.indices
        self.ptr=np.r_[0,np.cumsum([ptr[r+1]-ptr[r] for r in self.rows])].astype(np.int64)
        self.positions=np.concatenate([np.arange(ptr[r],ptr[r+1],dtype=np.int64) for r in self.rows])
        self.pre_rows=idx[self.positions].astype(np.int32);self.pre_ids=brain.brain.node_ids[self.pre_rows].copy()
        self.edges=len(self.positions);self.size=self.edges*4
        need(self.edges==7227,'Six-row anatomy differs from07')
        self.rows_gpu=cp.asarray(self.rows);self.ptr_gpu=cp.asarray(self.ptr)
        self.csr=cp.full((6,10),np.nan,dtype=cp.float64)
        self.scratch=cp.empty((4,6,16),dtype=cp.float64)
        self.operands=cp.empty((self.edges,4),dtype=cp.float32)
        self.stage_operands=cp.empty((4,self.edges,4),dtype=cp.float32)
        self.operand_records=cp.empty((CAPACITY,4,self.edges,4),dtype=cp.float32)
        self.count=cp.zeros(1,dtype=cp.uint64);self.records=cp.empty((CAPACITY,WIDTH),dtype=cp.float64)
        self.module=cp.RawModule(code=(HERE/'capture.cu').read_text(),options=OPTIONS)
        for name in ('capture_rhs','copy_stage','copy_trial_operands','capture_trial'):setattr(self,name,self.module.get_function(name))
        self.active=None;self.current_ms=0;self.epochs=[];self.arrays=[];self.edge_arrays=[]
        self.total_epochs=self.total_trials=self.total_accepted=self.total_rejected=0
    def bind_context(self):
        from snapshot_execution_brain import GpuSnapshotExecutionBrain
        cns=inspect.getclosurevars(GpuSnapshotExecutionBrain.advance).nonlocals['cns']
        self.active=inspect.getclosurevars(cns).nonlocals['active']
        need(isinstance(self.active,dict),'Missing predictor/committed context')
    def accept_epoch(self,core,ns,counts,data,edges,final):
        need(self.active is not None and type(self.active.get('accepted')) is bool,'Unbound epoch')
        need(data.shape==(sum(counts[:2]),WIDTH),'Incomplete trial record')
        need(edges.shape==(len(data),4,self.edges,4),'Incomplete operand record')
        need(np.isfinite(data).all() and np.isfinite(edges).all(),'Nonfinite observer data')
        need(np.array_equal(data[:,403],np.arange(len(data))),'Trial sequence')
        accepted=data[:,402].astype(bool)
        need(accepted.sum()==counts[0] and (~accepted).sum()==counts[1],'Scheduler counts')
        need(np.all(data[:,388:390]==0) and np.array_equal(accepted,data[:,387]<=1),'Acceptance mismatch')
        last=0.;state=data[0,390:396].copy()
        for row,ok in zip(data,accepted):
            need(row[384]==last and np.array_equal(row[390:396],state),'Commit/reject sequence')
            if ok:last=row[386];state=row[396:402].copy()
        need(last==ns*1e-9 and np.array_equal(state,final),'Final state mismatch')
        item=dict(ms=self.current_ms,start_ns=int(self.brain.time_ns),duration_ns=int(ns),
                  committed=self.active['accepted'],trials=len(data),accepted=int(counts[0]),rejected=int(counts[1]),epoch=self.total_epochs)
        need(self.active['start']==item['start_ns'] and self.active['end']==item['start_ns']+ns,'Epoch clock')
        self.epochs.append(item);self.arrays.append(data);self.edge_arrays.append(edges)
        self.total_epochs+=1;self.total_trials+=len(data);self.total_accepted+=counts[0];self.total_rejected+=counts[1]
    def flush(self,path):
        need(self.epochs,'Nothing to flush')
        path=Path(path);need(not path.exists(),'Preserve observed file')
        np.savez_compressed(path,records=np.concatenate(self.arrays),operands=np.concatenate(self.edge_arrays),
            offsets=np.r_[0,np.cumsum([len(a) for a in self.arrays])],
            **{k:np.asarray([e[k] for e in self.epochs]) for k in self.epochs[0]},
            fields=np.asarray(FIELDS),operand_fields=np.asarray(OPERANDS),rows=self.rows,ids=IDS,
            ptr=self.ptr,positions=self.positions,pre_rows=self.pre_rows,pre_ids=self.pre_ids)
        self.epochs.clear();self.arrays.clear();self.edge_arrays.clear()
    def report(self):
        return dict(ids=IDS.tolist(),edges=self.edges,epochs=self.total_epochs,trials=self.total_trials,
            accepted=self.total_accepted,rejected=self.total_rejected,capacity=CAPACITY,
            observed_operands=list(OPERANDS),solver_or_model_changed=False)
def install(brain,stimulus=None):
    observer=Observer(brain);previous_fast=fp32_operator.FastCSR;previous_real=real_model.RealCNS
    class ObservedFast(previous_fast):
        def __init__(self,b,**kw):
            need(b is brain,'Wrong coefficient owner');super().__init__(b,**kw)
            kernel=cp.RawKernel((HERE/'coefficient_capture.cu').read_text(),'coefficient_fast',options=OPTIONS)
            def diagnostic_kernel(grid,block,args):
                kernel(grid,block,(*args,observer.rows_gpu,observer.ptr_gpu,observer.csr,observer.operands))
            self.kernel=diagnostic_kernel
    class ObservedCNS(GraphRK23):
        def __init__(self,initial,coefficient,*,rtol,atol,norm_size,project=None,freeze=None,native_library=None,state_bounds=(0.,1.)):
            self.observer=observer;self.stage_id=0
            def rhs(y,clock,fraction):
                target,rate=coefficient(y)
                if stimulus is not None:stimulus.audit(target)
                value=rate*(target-y);stage=np.int32(self.stage_id)
                observer.capture_rhs((1,),(6,),(observer.rows_gpu,y,target,rate,value,clock,np.float64(fraction),stage,observer.csr,observer.scratch))
                observer.copy_stage(((observer.size+255)//256,),(256,),
                    (observer.operands,observer.stage_operands,np.int32(observer.size),stage))
                self.stage_id+=1
                return value
            super().__init__(initial,rhs,rtol=rtol,atol=atol,norm_size=norm_size,project=project,freeze=freeze,state_bounds=state_bounds)
        def trial(self):
            self.stage_id=0;super().trial();need(self.stage_id==4,'Four RHS stages required')
            size=observer.size*4
            observer.copy_trial_operands(((size+255)//256,),(256,),
                (observer.stage_operands,observer.count,observer.operand_records,np.int32(size),np.int32(CAPACITY)))
            observer.capture_trial((1,),(256,),
                (observer.rows_gpu,observer.scratch,self.clock,self.status,self.x,self.fine,observer.count,observer.records,np.int32(CAPACITY)))
        def advance(self,ns,*args,**kw):
            with self.stream:observer.count.fill(0)
            result=super().advance(ns,*args,**kw)
            count=int(observer.count.get(stream=self.stream)[0]);need(0<count<=CAPACITY,'Operand capture overflow; stop without changing solver')
            data=observer.records[:count].get(stream=self.stream)
            edges=observer.operand_records[:count].get(stream=self.stream)
            with self.stream:final=self.x[observer.rows_gpu].get(stream=self.stream)
            observer.accept_epoch(self,ns,result[1],data,edges,final)
            return result
    fp32_operator.FastCSR=ObservedFast;real_model.RealCNS=ObservedCNS
    def undo():fp32_operator.FastCSR=previous_fast;real_model.RealCNS=previous_real
    return observer,undo
