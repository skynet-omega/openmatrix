"""Versioned head-attached odor boundary, with one committed input interval."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BOUNDARY=ROOT/'work/stage3_bilateral_entry_20260915/antennal_boundary.py'
spec=importlib.util.spec_from_file_location('stage4_existing_antennal_boundary',BOUNDARY)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class AntennalWorld:
    # Runtime references stay out of __dict__, which the inherited serializer copies.
    __slots__=('body','boundary','__dict__')
    SCHEMA='matrix_head_attached_antennal_world_v1'
    KEYS={'schema','time_ns','source_mm','sigma_mm','events','next_event','installed_ns',
          'control_ns','committed_sensors','boundary_replacements','boundary_sha256','prior_world'}

    @classmethod
    def adopt(cls,previous,body,pending,control_ns):
        old=copy.deepcopy(previous.__dict__)
        expected={'time_ns','source_mm','sigma_mm','contact_radius_mm','antenna_forward_mm',
                  'antenna_half_separation_mm','events','next_event'}
        if set(old)!=expected or type(previous).__name__!='OdorPatchWorld':
            raise ValueError('Expected the explicit prior OdorPatchWorld; boundary already replaced or unknown')
        obj=cls.__new__(cls)
        obj.__dict__.update(schema=cls.SCHEMA,time_ns=old['time_ns'],source_mm=np.asarray(old['source_mm'],float).copy(),
            sigma_mm=float(old['sigma_mm']),events=copy.deepcopy(old['events']),next_event=old['next_event'],
            installed_ns=old['time_ns'],control_ns=control_ns,committed_sensors=np.asarray(pending,float).copy(),
            boundary_replacements=1,boundary_sha256=digest(BOUNDARY),prior_world=old)
        obj.body=None;obj.boundary=None;obj._validate();obj.bind(body)
        return obj

    def _validate(self):
        if set(self.__dict__)!=self.KEYS or self.schema!=self.SCHEMA:
            raise ValueError('Wrong/incomplete antennal world schema')
        if any(type(getattr(self,k)) is not int for k in ('time_ns','installed_ns','control_ns','next_event','boundary_replacements')):
            raise ValueError('Antennal clocks/counts must be integers')
        if self.control_ns!=1000000 or not 0<=self.installed_ns<=self.time_ns or (self.time_ns-self.installed_ns)%self.control_ns:
            raise ValueError('Invalid antennal interval clock')
        if self.boundary_replacements!=1 or self.boundary_sha256!=digest(BOUNDARY):
            raise ValueError('Changed antennal source or duplicate boundary replacement')
        if (not isinstance(self.source_mm,np.ndarray) or self.source_mm.shape!=(2,) or not np.isfinite(self.source_mm).all()
                or not np.isfinite(self.sigma_mm) or self.sigma_mm<=0):
            raise ValueError('Invalid odor field geometry')
        if (not isinstance(self.committed_sensors,np.ndarray) or self.committed_sensors.shape!=(3,)
                or not np.isfinite(self.committed_sensors).all() or np.any((self.committed_sensors<0)|(self.committed_sensors>1))):
            raise ValueError('Invalid committed sensory interval')
        if not isinstance(self.events,list) or not 0<=self.next_event<=len(self.events):
            raise ValueError('Invalid odor event cursor')
        last=-1
        for k,event in enumerate(self.events):
            if set(event)!={'time_ns','source_mm'} or type(event['time_ns']) is not int or event['time_ns']<last:
                raise ValueError('Invalid odor event order')
            p=np.asarray(event['source_mm'],float)
            if p.shape!=(2,) or not np.isfinite(p).all():raise ValueError('Invalid source event')
            if (k<self.next_event)!=(event['time_ns']<=self.time_ns):raise ValueError('Event cursor/time mismatch')
            last=event['time_ns']
        if self.prior_world['time_ns']!=self.installed_ns:raise ValueError('Prior world clock changed')

    def bind(self,body):
        self.body=body;self.boundary=module.AntennalBoundary(body.model)

    def advance_to(self,time_ns):
        self._validate()
        if type(time_ns) is not int or time_ns!=self.time_ns+self.control_ns:
            raise ValueError('World advances exactly one CNS interval at a time')
        self.time_ns=time_ns
        while self.next_event<len(self.events) and self.events[self.next_event]['time_ns']<=time_ns:
            self.source_mm=np.asarray(self.events[self.next_event]['source_mm'],float).copy()
            self.next_event+=1
        self._validate()

    def sample_geometry(self):
        if self.body is None:raise ValueError('Antennal world is not bound to the restored body')
        return self.boundary.sample(self.body.data,self.source_mm,self.sigma_mm)

    def sense(self,observation):
        self._validate()
        if self.body is None:raise ValueError('Unbound antennal world')
        if self.body.steps*round(self.body.dt*1e9)!=self.time_ns:
            raise ValueError('Antennal sampling requires the same body/world clock')
        if self.time_ns==self.installed_ns:
            return self.committed_sensors.copy()
        # Existing sample supplies concentrations; multiplying by odor_drive is
        # still exclusively the unchanged CNS input boundary's responsibility.
        value=self.sample_geometry()['concentration']
        return np.array([value[0],value[1],0.],dtype=np.float64)

    def state_dict(self):
        self._validate();return copy.deepcopy(self.__dict__)

    @classmethod
    def from_state(cls,state,body=None):
        obj=cls.__new__(cls);obj.__dict__.update(copy.deepcopy(state))
        obj.body=None;obj.boundary=None;obj._validate()
        if body is not None:obj.bind(body)
        return obj
