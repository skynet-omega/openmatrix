"""Two-row diagnostic sidecar; no controller/model parameter changes."""
from pathlib import Path
import inspect
import numpy as np
import cupy as cp
import fp32_operator
import real_model
from graph_runtime import GraphRK23

HERE=Path(__file__).resolve().parent
FIELDS=('state','net','positive_aux','negative_aux','drive','theta','gain',
        'base_target','base_rate','tau','margin','final_target','final_rate',
        'derivative','evaluation_time_s','stage_fraction')
WIDTH=140
CAPACITY=10000

def need(ok,message):
    if not ok:raise ValueError(message)

class Observer:
    def __init__(self,brain):
        self.brain=brain
        self.rows=np.searchsorted(brain.brain.node_ids,[10045,10056]).astype(np.int32)
        need(np.array_equal(self.rows,[36,46]) and
             np.array_equal(brain.brain.node_ids[self.rows],[10045,10056]),'DNg100 identity')
        need(not np.any(brain.visual_mask[self.rows]),'DNg100 is not a generic rate row')
        self.rows_gpu=cp.asarray(self.rows)
        self.csr=cp.full((2,10),np.nan,dtype=cp.float64)
        self.scratch=cp.empty((4,2,16),dtype=cp.float64)
        self.count=cp.zeros(1,dtype=cp.uint64)
        self.records=cp.empty((CAPACITY,WIDTH),dtype=cp.float64)
        self.module=cp.RawModule(code=(HERE/'observe.cu').read_text(),
                                 options=('--std=c++17','--fmad=false'))
        self.capture_rhs=self.module.get_function('capture_rhs')
        self.capture_trial=self.module.get_function('capture_trial')
        self.current_ms=0;self.active=None;self.epochs=[];self.arrays=[]
        self.total_epochs=self.total_trials=self.total_accepted=self.total_rejected=0

    def bind_context(self):
        from snapshot_execution_brain import GpuSnapshotExecutionBrain
        advance=GpuSnapshotExecutionBrain.advance
        cns=inspect.getclosurevars(advance).nonlocals['cns']
        self.active=inspect.getclosurevars(cns).nonlocals['active']
        need(isinstance(self.active,dict),'Midpoint context unavailable')

    def accept_epoch(self,core,ns,counts,data,final):
        need(self.active is not None and type(self.active.get('accepted')) is bool,
             'Missing real predictor/committed context')
        need(data.shape==(sum(counts[:2]),WIDTH) and np.isfinite(data).all(),
             'Incomplete/nonfinite observed trials')
        need(np.array_equal(data[:,139],np.arange(len(data))),'Trial sequence')
        accepted=data[:,138].astype(bool)
        need(accepted.sum()==counts[0] and (~accepted).sum()==counts[1],
             'Observed decisions differ from scheduler')
        need(np.all(data[:,132:134]==0),'Scheduler flags require explicit investigation')
        need(np.array_equal(accepted,data[:,131]<=1),'Acceptance predicate changed')
        last=0.;state=data[0,134:136].copy()
        for row,ok in zip(data,accepted):
            need(row[128]==last,'Trial time continuity')
            need(np.array_equal(row[134:136],state),'Commit/reject state continuity')
            if ok:last=row[130];state=row[136:138].copy()
        need(last==ns*1e-9 and np.array_equal(state,final),'Final committed observation')
        item=dict(ms=self.current_ms,start_ns=int(self.brain.time_ns),duration_ns=int(ns),
                  committed=self.active['accepted'],trials=len(data),accepted=int(counts[0]),
                  rejected=int(counts[1]),epoch=self.total_epochs)
        need(self.active['start']==item['start_ns'] and
             self.active['end']==item['start_ns']+ns,'Context clock mismatch')
        self.epochs.append(item);self.arrays.append(data)
        self.total_epochs+=1;self.total_trials+=len(data)
        self.total_accepted+=counts[0];self.total_rejected+=counts[1]

    def flush(self,path):
        need(self.epochs and len(self.epochs)==len(self.arrays),'Empty epoch block')
        data=np.concatenate(self.arrays)
        offsets=np.r_[0,np.cumsum([len(a) for a in self.arrays])]
        np.savez_compressed(path,records=data,offsets=offsets,
            **{key:np.asarray([e[key] for e in self.epochs]) for key in self.epochs[0]},
            fields=np.asarray(FIELDS),rows=self.rows,ids=np.array([10045,10056],np.int64))
        self.epochs.clear();self.arrays.clear()

    def report(self):
        return dict(rows=self.rows.tolist(),epochs=self.total_epochs,trials=self.total_trials,
                    accepted=self.total_accepted,rejected=self.total_rejected,
                    scheduler_changed=False,capacity_trials_per_epoch=CAPACITY,
                    recorded_fields=list(FIELDS),acceptance_checked_against_counts_and_committed_state=True)

def install(brain,stimulus=None):
    observer=Observer(brain)
    previous_fast=fp32_operator.FastCSR
    previous_real=real_model.RealCNS
    class ObservedFast(previous_fast):
        def __init__(self,b,**kw):
            need(b is brain,'Unexpected observed operator')
            super().__init__(b,**kw)
            kernel=cp.RawKernel((HERE/'coefficient_observed.cu').read_text(),'coefficient_fast',
                options=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true'))
            def diagnostic_kernel(grid,block,args):
                kernel(grid,block,(*args,observer.rows_gpu,observer.csr))
            self.kernel=diagnostic_kernel
    class ObservedCNS(GraphRK23):
        def __init__(self,initial,coefficient,*,rtol,atol,norm_size,project=None,
                     freeze=None,native_library=None,state_bounds=(0.,1.)):
            self.observer=observer;self.stage_id=0
            def rhs(y,clock,fraction):
                target,rate=coefficient(y)
                if stimulus is not None:stimulus.audit(target)
                value=rate*(target-y)
                observer.capture_rhs((1,),(2,),(observer.rows_gpu,y,target,rate,value,
                    clock,np.float64(fraction),np.int32(self.stage_id),observer.csr,observer.scratch))
                self.stage_id+=1
                return value
            super().__init__(initial,rhs,rtol=rtol,atol=atol,norm_size=norm_size,
                             project=project,freeze=freeze,state_bounds=state_bounds)

        def trial(self):
            self.stage_id=0
            super().trial()
            need(self.stage_id==4,'Expected four RHS calls per trial')
            observer.capture_trial((1,),(256,),(observer.rows_gpu,observer.scratch,
                self.clock,self.status,self.x,self.fine,observer.count,observer.records,np.int32(CAPACITY)))

        def advance(self,ns,*args,**kw):
            with self.stream:observer.count.fill(0)
            result=super().advance(ns,*args,**kw)
            count=int(observer.count.get(stream=self.stream)[0])
            need(0<count<=CAPACITY,'Observed trial buffer overflow')
            data=observer.records[:count].get(stream=self.stream)
            with self.stream:final=self.x[observer.rows_gpu].get(stream=self.stream)
            observer.accept_epoch(self,ns,result[1],data,final)
            return result

    fp32_operator.FastCSR=ObservedFast
    real_model.RealCNS=ObservedCNS
    def undo():
        fp32_operator.FastCSR=previous_fast
        real_model.RealCNS=previous_real
    return observer,undo
