"""Provisional serial muscle geometry driven only by canonical release.

Extends the continuing208 preparation without adding or reweighting neurons.
Serial homology remains an explicitly replaceable model hypothesis.
"""
from pathlib import Path
import copy
import numpy as np
from snapshot_execution_session import SnapshotExecutionSession,SOURCES as PARENT_SOURCES
from serial_ctr_body import SerialCTrBody,PRIOR_SHA256
from serial_ctr_storage import load_serial_ctr_session
from flybody_cns_body import CNSFlyBodyMuscles
from flybody_cns_sensors import FlyBodyEye,FlyBodyProprioception
from pn_graph_solver_session import PnGraphSolverSession
from coxal_body import PRIOR_SHA256 as COXAL_PRIOR_SHA256
from trochanter_body import PRIOR_SHA256 as TROCHANTER_PRIOR_SHA256
from bilateral_front_body import PRIOR_SHA256 as RIGHT_FRONT_PRIOR_SHA256
from rf_tarsal_body import PRIOR_SHA256 as TARSAL_PRIOR_SHA256
from kc_session_storage import save_session
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
SOURCES=tuple(PARENT_SOURCES)+('active_tendon_bank.py','serial_ctr_body.py','serial_ctr_storage.py','serial_ctr_session.py')
ENTRYPOINT='serial_ctr_session.py'


class SerialCTrSession(SnapshotExecutionSession):
    SCHEMA='matrix_serial_ctr_session_v1'

    @classmethod
    def from_checkpoint(cls,path,*,connected=True):
        if type(connected) is not bool:raise ValueError('Explicit serial connection required')
        parent=SnapshotExecutionSession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','body'};before=_fingerprints(parent,excluded)
            obj.body=SerialCTrBody.from_parent(parent.body);obj._index_serial()
            old=set(n for roles in obj.motor_ids.values() for ids in roles.values() for n in ids)
            for groups in [obj.body.coxal_ids,obj.body.tr_ids,obj.body.rf_ids,obj.body.tarsal_ids]:old.update(n for group in groups for n in group)
            new={n for ids in obj.body.serial_ids for n in ids}
            if old&new:raise ValueError('Duplicate existing motor contribution')
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate=cls.SCHEMA,serial_ctr_connected=connected,serial_ctr_prior_sha256=PRIOR_SHA256,
                serial_ctr_origin_ns=obj.time_ns,runtime_source_contract=dict(entrypoint=ENTRYPOINT,static_local_imports=require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)))
            obj.body.pending_serial=obj.serial_command()
            obj.muscles.body=obj.body;obj.proprioception.body=obj.body;obj.eyes.body=obj.body
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Serial adoption changed inherited history')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation='Add24serialCTr hypothesis paths from42concordant canonicalMN; no previous CTr active contribution in these legs',
                parent_checkpoint=str(Path(path).resolve()),parent_manifest_sha256=sha256(Path(path)/'manifest.json'),time_ns=obj.time_ns,
                preserved_state_checks=checks,physical_integration_state_preserved=True,new_activation_zero=True,
                old_motor_ids_disjoint=True,new_canonical_neurons=0,new_neural_parameters=False,new_passive_force=False,
                measured_posterior_geometry=False,biological_force_calibration=False)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES}
            obj._validate();obj._validate_pending();obj._validate_afferent_pending();obj._validate_cxhp8_pending();obj._validate_serial_pending()
            return obj
        except BaseException:obj.close();raise

    def _index_serial(self):
        self.serial_indices=[np.searchsorted(self.brain.node_ids,ids) for ids in self.body.serial_ids]
        for ix,ids in zip(self.serial_indices,self.body.serial_ids):
            if np.any(ix>=len(self.brain.node_ids)) or not np.array_equal(self.brain.node_ids[ix],ids):raise ValueError('Missing serial canonicalMN')

    def serial_command(self):
        if not self.config['serial_ctr_connected']:return np.zeros(24)
        q=self.hybrid.release();return np.array([q[ix].mean() for ix in self.serial_indices])

    def _validate_serial_pending(self):
        if not np.array_equal(self.body.pending_serial,self.serial_command()):raise ValueError('Serial input differs from canonical release')

    def _validate(self):
        PnGraphSolverSession._validate(self)
        if (type(self.body) is not SerialCTrBody or type(self.muscles) is not CNSFlyBodyMuscles or type(self.eyes) is not FlyBodyEye or type(self.proprioception) is not FlyBodyProprioception
            or self.output_connected or self.pending_cyborg_command!=0. or self.config['flybody_clock_origin_ns']!=self.body.origin_ns):raise ValueError('Invalid serial body family')
        for key,digest,origin,required in [('coxal',COXAL_PRIOR_SHA256,self.body.coxal_origin_ns,True),('trochanter',TROCHANTER_PRIOR_SHA256,self.body.tr_origin_ns,True),('right_front',RIGHT_FRONT_PRIOR_SHA256,self.body.rf_origin_ns,False),('rf_tarsal',TARSAL_PRIOR_SHA256,self.body.tarsal_origin_ns,False)]:
            connected=self.config[key+'_input_connected']
            if type(connected) is not bool or (required and not connected) or self.config[key+'_prior_sha256']!=digest or self.config[key+'_origin_ns']!=origin:raise ValueError('Inherited motor policy changed')
        p=self.config['motor_input_intervention']
        if p['removed_id'] is not None or any(p[k] is not None for k in ['leg','role','position','denominator']) or p['policy']!='zero_one_future_release_keep_pool_denominator_v1' or p['inherited_pending_interval_preserved'] is not True:raise ValueError('Serial family requires reference tibial inputs')
        self.body.validate_coxa()
        if self.body.rf_time_ns!=self.time_ns or self.body.tarsal_time_ns!=self.time_ns or self.body.serial_bank.time_ns!=self.time_ns:raise ValueError('Serial session clocks differ')
        if type(self.config['serial_ctr_connected']) is not bool or self.config['serial_ctr_prior_sha256']!=PRIOR_SHA256 or self.config['serial_ctr_origin_ns']!=self.body.serial_origin_ns:raise ValueError('Serial policy/clock differs')

    def step(self):
        self._validate_serial_pending();used=self.body.pending_serial.copy()
        row=super().step();self.body.pending_serial=self.serial_command();self._validate_serial_pending()
        row.update(serial_ctr_command_used=used.tolist(),serial_ctr_command_pending=self.body.pending_serial.tolist(),
                   serial_ctr_activation=self.body.serial_bank.activation.tolist(),serial_ctr_force_N=self.body.last_serial_force_N.tolist(),
                   serial_ctr_generalized_SI=self.body.last_serial_generalized_SI.tolist(),serial_ctr_connected=self.config['serial_ctr_connected'])
        return row

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=self.SCHEMA,runtime_source_files=len(SOURCES),physical_body_schema=SerialCTrBody.SCHEMA,
            serial_ctr_connected=self.config['serial_ctr_connected'],serial_ctr_prior_sha256=PRIOR_SHA256,
            serial_ctr_paths=24,serial_ctr_canonical_MN=42,serial_ctr_measured_geometry=False,serial_ctr_force_calibrated=False)
        return out

    def save(self,path):
        self._validate();self._validate_serial_pending();self._validate_cxhp8_pending()
        if require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('Serial source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
        return load_serial_ctr_session(path,cls,cls.SCHEMA,SOURCES,cls.BRAIN)
