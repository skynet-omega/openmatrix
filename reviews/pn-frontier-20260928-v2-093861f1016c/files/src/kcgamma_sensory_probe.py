"""Explicit local current ablation and timed sensory probe on a continuing CNS.

The anatomical graph and archived synaptic weights remain authoritative and
unchanged. Only the operational KCgamma gamma-lobe current contribution is
masked. This experiment is not a reconstructed neuronal component.
"""
from pathlib import Path
import copy,json,shutil,tempfile
import numba
import numpy as np
from pvlp_adaptation_session import (
    PvlpAdaptationSession, SOURCES as PARENT_SOURCES,
    CYBORG_SCALARS, AnatomicalRateBrain, AnatomicalProprioception,
    CandidateGammaPlasticity, ContractileTibia, CyborgEye, FixedCyborgEye,
    GpuGradedDescendingBrain, GpuVisualBrain, OdorPatchWorld,
    RefinedContactBody, ROOT, SCALARS, guard_visual_session,
    sha256, read_state, write_state, CyborgLoomingWorld,
    PvlpAdaptationBrain, GpuPvlpAdaptationBrain)
from cyborg_centered_servo import CyborgCenteredServo

from pvlp_servo_session import PvlpServoSession, SOURCES as SERVO_SOURCES

SCHEMA = 'matrix_kcgamma_sensory_probe_session_v1'
SOURCES = tuple(SERVO_SOURCES) + ('kcgamma_sensory_probe.py',)


def effective_mask_weights(base_weights, fraction, enabled):
    if type(enabled) is not bool:
        raise ValueError('Mask enabled must be boolean')
    base, fraction = np.asarray(base_weights, dtype=float), np.asarray(fraction, dtype=float)
    if (base.ndim != 1 or fraction.shape != base.shape or not np.isfinite(base).all()
            or not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1))):
        raise ValueError('Invalid current mask arrays')
    return base * (1. - fraction) if enabled else base.copy()


def pulse_amplitude(time_ns, start_ns, onset_ms, pulse_ms, amplitude):
    for v in (time_ns, start_ns, onset_ms, pulse_ms):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0:
            raise ValueError('Pulse times require nonnegative integer coordinates')
    if isinstance(amplitude, (bool, np.bool_)) or not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError('Invalid pulse amplitude')
    elapsed = int(time_ns) - int(start_ns)
    return float(amplitude) if onset_ms*1_000_000 <= elapsed < (onset_ms+pulse_ms)*1_000_000 else 0.


def validate_roi(brain, roi, kc_ids):
    names = ('pre_id', 'post_id', 'csr_position', 'total_contacts', 'gamma_contacts')
    out = {}
    for name in names:
        v = np.asarray(roi[name])
        if v.ndim != 1 or v.dtype.kind not in 'iu' or not len(v):
            raise ValueError('ROI identifiers and counts must be integer vectors')
        if v.dtype.kind == 'u' and np.any(v > np.iinfo(np.int64).max):
            raise ValueError('ROI integer overflow')
        out[name] = v.astype(np.int64).copy()
    size = len(out['csr_position'])
    if any(len(v) != size for v in out.values()):
        raise ValueError('ROI arrays differ in size')
    p, total, gamma = out['csr_position'], out['total_contacts'], out['gamma_contacts']
    if (len(np.unique(p)) != size or np.any((p < 0) | (p >= brain.W.nnz))
            or np.any(total <= 0) or np.any(gamma < 0) or np.any(gamma > total)):
        raise ValueError('Invalid ROI positions/count partition')
    pre = brain.W.indices[p]
    post = np.searchsorted(brain.W.indptr, p, side='right') - 1
    if (not np.array_equal(brain.node_ids[pre], out['pre_id'])
            or not np.array_equal(brain.node_ids[post], out['post_id'])
            or not np.isin(out['pre_id'], kc_ids).all() or not np.isin(out['post_id'], kc_ids).all()):
        raise ValueError('ROI pre/post identities do not match KCgamma CSR edges')
    fraction = gamma.astype(float) / total
    if 'gamma_fraction' in roi and not np.array_equal(np.asarray(roi['gamma_fraction']), fraction):
        raise ValueError('ROI fraction disagrees with contact counts')
    if 'saved_weight' in roi and not np.array_equal(brain.W.data[p], roi['saved_weight']):
        raise ValueError('ROI archived weights differ from the candidate')
    out['gamma_fraction'] = fraction
    out['saved_weight'] = brain.W.data[p].copy()
    return out


class KcGammaSensoryProbeSession(PvlpServoSession):
    @classmethod
    def from_checkpoint(cls, path, *, roi_path, mask_enabled, condition,
                        onset_ms=200, pulse_ms=100, amplitude=80., tight=False):
        parent = PvlpServoSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            roi = validate_roi(obj.brain, dict(np.load(roi_path, allow_pickle=False)), obj.ports['KC_gamma'])
            if np.intersect1d(roi['csr_position'], obj.plasticity.positions).size:
                raise ValueError('Diagnostic mask overlaps the plasticity implementation')
            obj.probe = dict(roi=roi, roi_sha256=sha256(roi_path), mask_enabled=mask_enabled,
                condition=condition, start_ns=obj.time_ns, onset_ms=onset_ms,
                pulse_ms=pulse_ms, amplitude=amplitude, parent_checkpoint=str(Path(path).resolve()),
                parent_manifest_sha256=sha256(Path(path)/'manifest.json'))
            pulse_amplitude(obj.time_ns, obj.time_ns, onset_ms, pulse_ms, amplitude)
            obj.plasticity.enabled = False
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate='KCGAMMA_LOCAL_CURRENT_DIAGNOSTIC_v1',
                              sensory_probe=True, plasticity_updates_enabled=False)
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Freeze weight updates and optionally mask KCgamma gamma-lobe current; timed ORN_DM1 probe',
                time_ns=obj.time_ns, condition=condition, mask_enabled=mask_enabled,
                graph_and_archived_weights_preserved=True, eligibility_continues=True,
                biological_validation=False, new_neural_component=False)
            if tight:
                for key in ('rtol','atol'):
                    obj.hybrid.parameters[key] *= .1
            obj.source_identity = {name:sha256(ROOT/'src'/name) for name in SOURCES}
            obj._apply_probe_mask()
            obj._validate()
            obj._validate_pending()
            return obj
        except BaseException:
            obj.close()
            raise

    def _apply_probe_mask(self):
        if self.plasticity.enabled:
            raise ValueError('All probe arms freeze weight updates')
        p = self.probe
        roi = validate_roi(self.brain, p['roi'], self.ports['KC_gamma'])
        pos = roi['csr_position']
        if np.intersect1d(pos, self.plasticity.positions).size:
            raise ValueError('Current mask overlaps plasticity')
        values = effective_mask_weights(self.brain.W.data[pos], roi['gamma_fraction'], p['mask_enabled'])
        self.hybrid.weights64[pos] = values
        if hasattr(self.hybrid, 'cuda'):
            import cupy as cp
            self.hybrid.cuda['weights'][cp.asarray(pos)] = cp.asarray(values)

    def additional_drive(self):
        p = self.probe
        return pulse_amplitude(self.time_ns, p['start_ns'], p['onset_ms'], p['pulse_ms'], p['amplitude'])

    def _validate(self):
        super()._validate()
        if self.plasticity.enabled is not False:
            raise ValueError('Probe weight updates must remain frozen')
        if type(self.probe['mask_enabled']) is not bool or self.probe['start_ns'] > self.time_ns:
            raise ValueError('Invalid probe intervention or clock')
        self.additional_drive()

    def step(self):
        # This is the inherited live step, with an independent rotor advanced
        # using the previous pending command before the next retinal sample.
        self._validate()
        light = self.pending_light.copy()
        drive = (self.pending_proprioception["drive"].astype(float) if self.proprioception_enabled
                 else np.zeros(self.brain.n_neurons))
        for k, side in enumerate(("L", "R")):
            drive[self.port_indices[f"ORN_DM1_{side}"]] += self.config["odor_drive"]*self.pending_sensors[k]
        extra = self.additional_drive()
        for side in ('L', 'R'):
            drive[self.port_indices[f'ORN_DM1_{side}']] += extra
        excitation = self.pending_excitation.copy()
        command = self.pending_cyborg_command
        try:
            self.hybrid.advance(self.CONTROL_NS, drive, light)
            self.plasticity.step(self.CONTROL_NS*1e-9)
            self.hybrid.sync_plastic_weights(self.plasticity)
            next_excitation = self.motor_excitation()
            next_command = self.cyborg_command()
            physical_dt_ns = round(self.body.dt*1e9)
            for _ in range(self.CONTROL_NS//physical_dt_ns):
                torque = self.muscles.advance(excitation, physical_dt_ns)
                self.body.advance(torque, 1)
            self.rotor.advance(command, self.CONTROL_NS)
            self.time_ns += self.CONTROL_NS
            self.ticks += 1
            self.world.advance_to(self.time_ns)
            observation = self.body.observe()
            self.pending_sensors = self.world.sense(observation)
            self.pending_proprioception = self.proprioception.sample()
            self.pending_light = self.eyes.sample(self.light_world, self.time_ns)
            self.pending_excitation = next_excitation
            self.pending_cyborg_command = next_command
            release = self.hybrid.release()
            frozen = self.eyes.sample(self.light_world, self.time_ns, pose=self.initial_camera)
            rotation, centers = self.eyes.pose()
            self.used_light.append(light)
            row = dict(time_s=self.time_ns*1e-9, position_mm=observation["position_mm"].tolist(),
                qpos=observation["qpos"].tolist(), qvel=observation["qvel"].tolist(),
                contact_count=observation["contact_count"], upright_cos=observation["upright_cos"],
                mn_excitation=next_excitation.tolist(), muscle_activation=self.muscles.activation.tolist(),
                muscle_force_native=self.muscles.last_force.tolist(), torque_native=torque.tolist(),
                retinal_used_mean=float(light.mean()), retinal_pending_mean=float(self.pending_light.mean()),
                retinal_cyborg_motion_rms=float(np.sqrt(np.mean((self.pending_light-frozen)**2))),
                optical_mount_rotation=rotation.tolist(), eye_centers_mm={k: v.tolist() for k,v in centers.items()},
                visual_release={name: dict(count=len(idx), mean=float(release[idx].mean()),
                    maximum=float(release[idx].max())) for name,idx in self.visual_groups.items() if len(idx)},
                active_release_above_point01=int(np.count_nonzero(release > .01)),
                nonvisual_saturated_above_point95=int(np.count_nonzero(release[self.hybrid.ri] > .95)),
                visual_saturated_above_point95=int(np.count_nonzero(release[self.hybrid.vi] > .95)),
                plastic_factor_min=float(self.plasticity.factors.min()),
                numeric_accepted=self.hybrid.statistics["accepted"], numeric_rejected=self.hybrid.statistics["rejected"],
                cyborg=dict(command_used=command, command_pending=next_command, output_connected=self.output_connected,
                    angle_rad=self.rotor.angle_rad, omega_rad_s=self.rotor.omega_rad_s,
                    torque_nm=self.rotor.last_torque_nm, engineering_prosthesis=True))
            row['sensory_probe_current_used'] = extra
            self.history.append(row)
            self._validate()
            return row
        except BaseException:
            self.failed = True
            raise

    def state_dict(self):
        result = super().state_dict()
        result["schema"] = SCHEMA
        result["probe"] = copy.deepcopy(self.probe)
        return result

    def save(self, path):
        self._validate()
        self._validate_pending()
        self.light_world._validate()
        if self.source_identity != {name: sha256(ROOT/"src"/name) for name in SOURCES}:
            raise ValueError("Runtime source changed after cyborg session creation")
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(f"Preserving checkpoint {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="."+path.name+"-", dir=path.parent))
        try:
            self.brain.save_checkpoint(staging/"brain")
            write_state(staging/"session", self.state_dict())
            (staging/"source").mkdir()
            for name in SOURCES:
                shutil.copyfile(ROOT/"src"/name, staging/"source"/name)
            manifest = dict(schema=SCHEMA, time_ns=self.time_ns, neuron_count=self.brain.n_neurons,
                stored_edges=self.brain.W.nnz, mapped_photoreceptors=len(self.eyes.ids),
                mode=self.mode, output_connected=self.output_connected, engineering_prosthesis=True,
                camera_fixed=self.config["looming_camera_fixed"], physical_looming_stimulus=True,
                pvlp_adaptation_enabled=self.hybrid.pvlp_adaptation_manifest["enabled"],
                prosthesis_kind="centered_position_servo",
                diagnostic_current_mask=self.probe["mask_enabled"], plasticity_updates_enabled=False,
                retinal_port_enabled=self.config["retinal_port_enabled"], biological_retina_reconstructed=False,
                biological_validation=False, animal_ability_demonstrated=False,
                files={str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()})
            (staging/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
            staging.rename(path)
            return path
        except BaseException:
            shutil.rmtree(staging)
            raise

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = json.loads((path/"manifest.json").read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported retinal-prosthesis checkpoint")
        actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
        if actual != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Incomplete cyborg checkpoint")
        for name,digest in manifest["files"].items():
            if sha256(path/name) != digest:
                raise ValueError(f"Checkpoint integrity failed: {name}")
        state = read_state(path/"session")
        special = {"schema", "world", "body", "hybrid", "eyes", "light_world", "muscles",
                   "plasticity", "proprioception", "used_light", "rotor", "probe"}
        if set(state) != SCALARS | CYBORG_SCALARS | special or state["schema"] != SCHEMA:
            raise ValueError("Unsupported or incomplete cyborg state")
        if state["source_identity"] != {name: sha256(ROOT/"src"/name) for name in SOURCES}:
            raise ValueError("Checkpoint requires its archived source versions")
        obj = cls()
        obj.brain = AnatomicalRateBrain.load_checkpoint(path/"brain")
        for key in SCALARS | CYBORG_SCALARS:
            setattr(obj, key, state[key])
        if obj.config["numba_version"] != numba.__version__:
            raise ValueError("Checkpoint requires its recorded numerical runtime")
        if obj.config["numerical_backend"] == "cuda_fp64" and obj.config["backend_identity"] != GpuVisualBrain.backend_identity():
            raise ValueError("Recorded GPU backend differs")
        obj._index_ports()
        obj._index_motors()
        obj.world = OdorPatchWorld()
        if set(obj.world.__dict__) != set(state["world"]):
            raise ValueError("Incomplete inherited odor world")
        obj.world.__dict__.update(state["world"])
        obj.plasticity = CandidateGammaPlasticity.from_state(obj.brain, state["plasticity"])
        kinds = {kind.SCHEMA: kind for kind in (PvlpAdaptationBrain, GpuPvlpAdaptationBrain)}
        obj.hybrid = kinds[state["hybrid"]["schema"]].from_state(obj.brain, state["hybrid"])
        if isinstance(obj.hybrid, GpuGradedDescendingBrain) and obj.config["electrical_backend_identity"] != obj.hybrid.backend_identity():
            raise ValueError("Recorded local electrical GPU backend differs")
        obj.hybrid.sync_plastic_weights(obj.plasticity)
        obj.probe = copy.deepcopy(state["probe"])
        obj._apply_probe_mask()
        guard_visual_session(obj)
        obj.body = RefinedContactBody.from_state(state["body"])
        try:
            obj.proprioception = AnatomicalProprioception.from_state(obj.brain, obj.body, state["proprioception"])
            obj.rotor = CyborgCenteredServo.from_state(state["rotor"])
            obj.eyes = CyborgEye.from_state(obj.brain, obj.body, state["eyes"], obj.rotor)
            if obj.config["looming_camera_fixed"]:
                obj.eyes = FixedCyborgEye.adopt_fixed(obj.eyes, obj.rotor, obj.initial_camera)
            obj.light_world = CyborgLoomingWorld.from_state(state["light_world"])
            obj.muscles = ContractileTibia.from_state(obj.body, state["muscles"])
            obj.used_light = list(state["used_light"])
            obj.failed = False
            obj._validate()
            obj._validate_pending()
            if manifest["time_ns"] != obj.time_ns or manifest["output_connected"] != obj.output_connected:
                raise ValueError("Manifest clock or output connection differs")
            return obj
        except BaseException:
            obj.body.close()
            raise
