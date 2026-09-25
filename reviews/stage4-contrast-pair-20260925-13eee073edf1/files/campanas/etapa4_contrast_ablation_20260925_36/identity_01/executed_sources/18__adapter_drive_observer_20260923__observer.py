"""Bounded observer, installed before the adapter's first graph build.

No CuPy import. Real device reads use the existing array/stream APIs, outside
capture only. Python coefficient observations describe graph construction;
CUDA replays do not re-enter these Python hooks.
"""
from pathlib import Path
from types import MethodType
import hashlib
import inspect
import json
import numpy as np

HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'PLAN.json').read_text())
ADAPTER_SHA='355666d4f0510ec5d2cb86843c76c65d2035e617271aa21b1d9dba364a18ded6'


def need(ok,message):
    if not ok:raise ValueError(message)


def pointer(array):
    return int(array.data.ptr)


def host_copy(array):
    value=np.asarray(array)
    need(value.dtype.kind=='f' and value.ndim==1,'Expected one-dimensional floating boundary')
    need(value.nbytes<=PLAN['capture_limits']['array_bytes_max'],'Boundary exceeds observation byte budget')
    need(np.isfinite(value).all(),'Nonfinite observed boundary')
    return np.array(value,copy=True,order='C')


def device_copy(array,stream):
    need(int(array.nbytes)<=PLAN['capture_limits']['array_bytes_max'],'Device boundary exceeds observation byte budget')
    stream.synchronize()
    return host_copy(array.get(stream=stream))


def equal(a,b,label):
    need(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b),label)


def descriptor(value,indices):
    need(all(0<=i<len(value) for i in indices),'Selected observation index outside buffer')
    return {'shape':list(value.shape),'dtype':str(value.dtype),'bytes':int(value.nbytes),
            'sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest(),
            'indices':list(indices),'values':value[list(indices)].tolist()}


class DriveObserver:
    def __init__(self,adapter,*,selected_epochs,drive_indices=(),pn_indices=(),read_device=device_copy):
        limits=PLAN['capture_limits']
        selected=tuple(selected_epochs)
        need(selected and len(set(selected))==len(selected)<=limits['selected_epochs_max'],'Explicit bounded unique epochs required')
        need(all(isinstance(x,str) and x for x in selected),'Epoch labels must be nonempty strings')
        need(adapter.core is None,'Install before the first graph build; do not retrofit a captured graph')
        source=Path(inspect.getsourcefile(type(adapter)))
        need(hashlib.sha256(source.read_bytes()).hexdigest()==ADAPTER_SHA,'Unsupported adapter source')
        need(getattr(adapter.events.step,'__self__',None) is adapter and getattr(adapter.events.step,'__func__',None) is type(adapter).step,
             'Unexpected event step owner')
        self.adapter=adapter;self.read_device=read_device;self.selected=selected
        self.drive_indices=tuple(map(int,drive_indices));self.pn_indices=tuple(map(int,pn_indices))
        for indices in (self.drive_indices,self.pn_indices):
            need(len(indices)==len(set(indices))<=limits['selected_indices_max'] and all(i>=0 for i in indices),'Invalid/oversized selected indices')
        self.records=[];self.epochs=[];self.failed=False;self.failure=None;self.closed=False;self.current=None;self.active_step=None
        self.build_calls=0;self.pn_capture_reads=0;self.bindings=None;self.patches=[];self.completed_labels=set()
        self.source={'adapter_sha256':ADAPTER_SHA,'observer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.original_step=adapter.events.step;self.original_build=adapter.build;self.original_host=adapter.host_read
        def entry(owner,b,ns,drive,light):
            try:return self._entry(b,ns,drive,light)
            except BaseException as exc:
                self.failed=True;self.failure=type(exc).__name__+': '+str(exc);raise
        def build(owner,drive,light):return self._build(drive,light)
        def host_read():
            value=self.original_host()
            state=self.active_step
            if state is not None and state['capture']:
                state['host_reads']+=1
                need(state['host_reads']<=limits['actual_pn_host_reads_per_step_max'],'PN host-read capture budget exceeded')
                state['pn_refresh']=host_copy(value)
            return value
        self._patch(adapter.events,'step',MethodType(entry,adapter))
        self._patch(adapter,'build',MethodType(build,adapter))
        self._patch(adapter,'host_read',host_read)

    def _patch(self,obj,name,value):
        present=name in obj.__dict__;old=obj.__dict__.get(name)
        setattr(obj,name,value);self.patches.append((obj,name,present,old,value))

    def begin_epoch(self,label):
        need(not self.closed and not self.failed and self.current is None,'Observer cannot start an epoch')
        need(isinstance(label,str) and label and label not in self.completed_labels,'Repeated/invalid epoch label')
        self.current={'label':label,'calls':0,'captured':0,'selected':label in self.selected}

    def end_epoch(self):
        need(self.current is not None and self.active_step is None and not self.failed,'Incomplete/failed epoch')
        row=self.current
        if row['selected']:
            need(row['captured']==len(PLAN['capture_limits']['capture_ordinals_per_epoch']),'Selected epoch missed a required observation ordinal')
        self.epochs.append(dict(row));self.completed_labels.add(row['label']);self.current=None

    def _build(self,drive,light):
        b=self.adapter.brain;source=b._online_source
        need(self.bindings is None,'Unexpected graph rebuild during observation')
        present='coefficients_gpu' in b.__dict__;old_attr=b.__dict__.get('coefficients_gpu');original=b.coefficients_gpu
        def coefficient(owner,z,actual_drive,actual_light):
            self.build_calls+=1
            need(self.build_calls<=PLAN['capture_limits']['coefficient_build_calls_max'],'Coefficient build observation budget exceeded')
            a=self.adapter
            need(actual_drive is a.drive and actual_light is a.light,'Coefficient did not receive adapter buffers')
            before=(pointer(a.drive),pointer(a.pn))
            prior=source.general_transmission
            def actual_reader():
                value=prior()
                need(value is a.pn and pointer(value)==before[1],'Coefficient PN reader returned another buffer')
                self.pn_capture_reads+=1
                need(self.pn_capture_reads<=PLAN['capture_limits']['coefficient_build_calls_max'],'PN capture-read budget exceeded')
                return value
            source.general_transmission=actual_reader
            try:
                result=original(z,actual_drive,actual_light)
                need(before==(pointer(a.drive),pointer(a.pn)),'Coefficient construction rebound a boundary buffer')
                return result
            finally:source.general_transmission=prior
        b.coefficients_gpu=MethodType(coefficient,b)
        try:result=self.original_build(drive,light)
        finally:
            if present:b.coefficients_gpu=old_attr
            else:del b.coefficients_gpu
        a=self.adapter;core=a.core
        closure=inspect.getclosurevars(core.coefficient).nonlocals
        need(closure.get('self') is a and closure.get('b') is b,'Unexpected captured coefficient owner')
        need(self.build_calls>0,'No coefficient construction observed')
        self.bindings={'drive_object':a.drive,'pn_object':a.pn,'drive_pointer':pointer(a.drive),'pn_pointer':pointer(a.pn)}
        original_advance=core.advance
        def advance(owner,*args,**kwargs):return self._advance(original_advance,*args,**kwargs)
        self._patch(core,'advance',MethodType(advance,core))
        return result

    def _entry(self,b,ns,drive,light):
        need(not self.closed and not self.failed and self.current is not None and self.active_step is None,'Unlabelled, reentrant or failed observer')
        need(b is self.adapter.brain,'Different brain passed to observed adapter')
        epoch=self.current;ordinal=epoch['calls'];epoch['calls']+=1
        need(epoch['calls']<=PLAN['capture_limits']['step_calls_per_epoch_max'],'Per-epoch call budget exceeded')
        capture=epoch['selected'] and ordinal in PLAN['capture_limits']['capture_ordinals_per_epoch']
        state={'capture':capture,'host_reads':0,'graph_calls':0}
        if capture:
            need(len(self.records)<PLAN['capture_limits']['records_total_max'],'Total capture budget exceeded')
            state.update(argument=host_copy(np.asarray(drive,dtype=float)),raw_argument=np.asarray(drive).copy(),
                         row={'epoch':epoch['label'],'ordinal':ordinal,'start_time_ns':int(b.time_ns),'duration_ns':int(ns),
                              'argument_dtype_before_validation':str(np.asarray(drive).dtype),
                              'comparison_conversion':'Inherited _validated_inputs converts drive to float64'})
            state['row']['drive_argument']=descriptor(state['argument'],self.drive_indices)
        self.active_step=state
        try:
            result=self.original_step(b,ns,drive,light)
            if capture:
                need(state['graph_calls']==1,'Captured adapter call did not execute exactly one graph advance')
                equal(np.asarray(drive),state['raw_argument'],'Original drive argument mutated during adapter step')
                state['row'].update(host_pn_reads=state['host_reads'],argument_unchanged=True,status='OBSERVED_EXACT_BOUNDARIES')
                self.records.append(state['row']);epoch['captured']+=1
            return result
        except BaseException as exc:
            self.failed=True
            if capture:
                state['row'].update(status='FAILED_RETAINED',error=type(exc).__name__+': '+str(exc));self.records.append(state['row'])
            raise
        finally:self.active_step=None

    def _advance(self,original,*args,**kwargs):
        state=self.active_step
        need(state is not None,'Graph advanced outside an observed adapter call')
        if not state['capture']:return original(*args,**kwargs)
        state['graph_calls']+=1
        need(state['graph_calls']==1 and 'pn_refresh' in state,'Missing/duplicate graph boundary or PN refresh')
        a=self.adapter;g=a.core;binding=self.bindings
        need(a.drive is binding['drive_object'] and a.pn is binding['pn_object'],'Captured boundary object rebound')
        need((pointer(a.drive),pointer(a.pn))==(binding['drive_pointer'],binding['pn_pointer']),'Captured device address changed')
        before_drive=self.read_device(a.drive,g.stream);before_pn=self.read_device(a.pn,g.stream)
        equal(before_drive,state['argument'],'Device drive differs from actual step argument')
        equal(before_pn,state['pn_refresh'],'Device PN differs from actual host refresh')
        state['row'].update(drive_before=descriptor(before_drive,self.drive_indices),pn_refresh=descriptor(state['pn_refresh'],self.pn_indices),
                            pn_before=descriptor(before_pn,self.pn_indices),captured_device_pointers=[pointer(a.drive),pointer(a.pn)])
        result=original(*args,**kwargs)
        after_drive=self.read_device(a.drive,g.stream);after_pn=self.read_device(a.pn,g.stream)
        need(a.drive is binding['drive_object'] and a.pn is binding['pn_object'],'Graph rebound captured boundary objects')
        need((pointer(a.drive),pointer(a.pn))==(binding['drive_pointer'],binding['pn_pointer']),'Graph changed captured device addresses')
        equal(after_drive,before_drive,'Graph changed held drive buffer')
        equal(after_pn,before_pn,'Graph changed held PN buffer')
        state['row'].update(drive_after=descriptor(after_drive,self.drive_indices),pn_after=descriptor(after_pn,self.pn_indices),
                            buffers_unchanged_during_advance=True)
        return result

    def report(self):
        return {'schema':'bounded_adapter_drive_observation_v1','failed':self.failed,'failure':self.failure,'closed':self.closed,'source':self.source,
                'selected_epochs':list(self.selected),'epochs':list(self.epochs),'open_epoch':self.current,
                'coefficient_build_calls_observed':self.build_calls,'actual_pn_reader_calls_during_build':self.pn_capture_reads,
                'pn_capture_binding_observed':self.pn_capture_reads>0,'capture_limits':PLAN['capture_limits'],'records':list(self.records),
                'scope':'Actual adapter arguments and held buffers at selected advances, plus Python binding observations during graph construction',
                'not_demonstrated':['Every CUDA kernel read or every numerical stage','Causal efficacy of a PN pathway or ORN physiological calibration',
                                    'All epochs/steps outside the explicitly selected captures','GPU noninterference or overhead before a real paired test',
                                    'Whole-organism equivalence, Stage3/Stage4 admission or navigation']}

    def save(self,path):
        with Path(path).open('x') as f:f.write(json.dumps(self.report(),indent=2,allow_nan=False)+'\n')

    def close(self):
        need(self.active_step is None,'Cannot detach during an adapter call')
        if self.closed:return
        for obj,name,present,old,value in reversed(self.patches):
            need(obj.__dict__.get(name) is value,'Observer hook changed before teardown: '+name)
            if present:setattr(obj,name,old)
            else:delattr(obj,name)
        self.closed=True
