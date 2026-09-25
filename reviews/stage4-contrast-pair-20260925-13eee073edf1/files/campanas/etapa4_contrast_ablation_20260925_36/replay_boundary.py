"""Clocked sensory replay; physical geometry remains observable separately.

Intervenes on the evaluator's input port only. No neuron, decoder or integrator
is changed. Terminal pending sample is recorded but never consumed in this run.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'etapa4_long_trajectory_20260925_35'
sys.path.insert(0, str(PRIOR))
from gaussiano_400 import DT, FixedGaussian, clock, need, prepared_pose_guard, save


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class Tape:
    def __init__(self, path, expected_sha, mode):
        need(mode in ('identity', 'no_contrast'), 'Unknown replay mode')
        need(digest(path) == expected_sha, 'Donor identity differs')
        with np.load(path, allow_pickle=False) as z:
            need(z['fase'].tolist() == ['preparacion'] * 40 + ['ensayo'] * 1000,
                 'Donor phase or length differs')
            need(z['paso'].tolist() == list(range(1, 41)) + list(range(1, 1001)), 'Donor step order')
            t = z['CNS_time_ns']
            need(np.issubdtype(t.dtype, np.integer) and t.shape == (1040,), 'Donor clock dtype/shape')
            need(np.all(np.diff(t) == DT), 'Donor clock spacing')
            need(all(np.array_equal(t, z[k]) for k in ('PN_time_ns', 'body_time_ns')), 'Donor clocks differ')
            u, p = z['sensores_usados'], z['sensores_pendientes']
            need(u.shape == p.shape == (1040, 3), 'Donor sensory shape')
            need(np.isfinite(u).all() and np.isfinite(p).all(), 'Nonfinite tape')
            need(np.all(u[:40] == 0), 'Nonzero preparation input')
            need(np.array_equal(u[41:], p[40:-1]), 'Donor committed one-step lag')
            self.origin = int(t[39])
            self.samples = np.vstack((u[40:], p[-1:])).astype(np.float64)
        need(np.all((self.samples >= 0) & (self.samples <= 1)), 'Tape outside sensory domain')
        need(np.all(self.samples[:, 2] == 0), 'Boundary cannot replay nonzero third channel')
        if mode == 'no_contrast':
            common = self.samples[:, :2].mean(axis=1)
            self.samples[:, :2] = common[:, None]
        self.samples.flags.writeable = False
        self.mode, self.donor_sha = mode, expected_sha

    def at(self, time_ns):
        dt = int(time_ns) - self.origin
        need(0 <= dt <= 1000 * DT and dt % DT == 0, 'Replay time outside committed tape')
        return self.samples[dt // DT].copy()


class ReplayBoundary:
    def __init__(self, physical, tape):
        self.physical, self.tape = physical, tape
        need(physical.origin_ns == tape.origin, 'Replay/preparation origin mismatch')

    def sample(self, data, source_mm, sigma_mm):
        r = self.physical.sample(data, source_mm, sigma_mm)
        delivered = self.tape.at(self.physical.world.time_ns)
        return dict(r, concentration=delivered[:2], canonical_ORN_drive=80 * delivered[:2])

    def metadata(self):
        return dict(schema='clocked_olfactory_replay_v1', mode=self.tape.mode,
                    donor_trace_sha256=self.tape.donor_sha, origin_ns=self.tape.origin,
                    consumed_samples=1000, terminal_pending_samples=1,
                    physical_field=self.physical.metadata(),
                    concentration_campo_semantics='Delivered tape, not the field at the current body pose',
                    physical_concentration_column='concentracion_fisica',
                    spatial_feedback_enabled=False, resumable=False)


class Intervals:
    def __init__(self, obj, boundary, prepared, pose_guard):
        self.field, self.origin = boundary, clock(obj)
        self.n, self.armed, self.invalid = 0, False, False
        self.q = np.array(prepared['DN_q_actual'], copy=True)
        self.baseline = np.array(prepared['DN_baseline'], copy=True)
        self.prepared_pose_guard = pose_guard
        self.records = []

    def before(self, obj):
        need(not self.invalid and not self.armed and self.n < 1000, 'Invalid/reentrant replay auditor')
        need(obj.core.world.boundary is self.field, 'Replay boundary replaced')
        need(clock(obj) == self.origin + self.n * DT, 'Replay start clock')
        self.used = self.field.tape.at(clock(obj))
        need(np.array_equal(obj.core.pending_sensors, self.used), 'Wrong committed replay sample')
        self.armed = True
        return self.used.copy()

    def after(self, obj, row):
        try:
            need(self.armed and not self.invalid, 'Replay interval not armed')
            end = self.origin + (self.n + 1) * DT
            need(clock(obj) == end, 'Replay end clock')
            need(row['fase'] == 'ensayo' and row['paso'] == self.n + 1, 'Replay row order')
            need(all(row[k] == end for k in ('CNS_time_ns', 'PN_time_ns', 'body_time_ns')), 'Trace clock')
            pending = self.field.tape.at(end)
            need(np.array_equal(row['sensores_usados'], self.used), 'Wrong consumed tape')
            need(np.array_equal(row['sensores_pendientes'], pending) and
                 np.array_equal(obj.core.pending_sensors, pending), 'Wrong pending tape')
            need(np.array_equal(row['concentracion_campo'], pending[:2]), 'Delivered tape trace differs')
            physical = self.field.physical.sample(obj.body.data, None, None)
            need(np.array_equal(row['antenas_mm'], physical['antennae_mm']), 'Live antenna geometry differs')
            need(np.array_equal(row['concentracion_fisica'], physical['concentration']), 'Physical field trace differs')
            need(np.array_equal(row['DN_q_usada'], self.q), 'Neural command lag differs')
            need(np.array_equal(row['DN_baseline'], self.baseline), 'Baseline differs')
            dq = self.q - self.baseline
            expected = np.tanh(250 * (dq[2] - dq[3])) * np.deg2rad(5.)
            need(expected == row['command_yaw_rate_rad_s'], 'Reader equation differs')
            self.records.append(dict(sample_used_ns=end-DT, sample_pending_ns=end,
                                     used=self.used.tolist(), pending=pending.tolist()))
            self.q = np.array(row['DN_q_actual'], copy=True)
            self.n += 1
            self.armed = False
        except BaseException:
            self.invalid = True
            raise

    def export(self, path):
        save(path, dict(field=self.field.metadata(), prepared_pose_guard=self.prepared_pose_guard,
                        intervals=self.records, complete=self.n == 1000 and not self.invalid and not self.armed,
                        stage4_admission=False, stage5_admission=False))


def install(obj, base, arm, spec, prepared_row, *, geometry_path, mode, donor, donor_sha):
    tape = Tape(donor, donor_sha, mode)
    c, w = obj.core, obj.core.world
    need(prepared_row['fase'] == 'preparacion' and prepared_row['paso'] == 40, 'Preparation row')
    need(clock(obj) == tape.origin and prepared_row['CNS_time_ns'] == tape.origin, 'Preparation clock')
    need(np.all(c.pending_sensors == 0), 'Preparation not odor-free')
    need(type(base).__name__ == 'AntennalBoundary', 'Wrong physical Gaussian base')
    ids = [obj.body.model.geom('antenna_' + side + '_collision').id for side in ('left', 'right')]
    need(base.model is obj.body.model and np.array_equal(base.geoms, ids), 'Antenna owner/order')
    need(w.next_event == len(w.events), 'Future source changes present')
    guard = prepared_pose_guard(obj, base, prepared_row, geometry_path)
    b = ReplayBoundary(FixedGaussian(base, w, arm, spec), tape)
    old, pending = w.boundary, c.pending_sensors.copy()
    try:
        w.boundary = b
        c.pending_sensors = w.sense(obj.body.observe())
        need(np.array_equal(c.pending_sensors, tape.at(clock(obj))), 'Initial tape commitment')
        obj._validate_adapter()
        return Intervals(obj, b, prepared_row, guard)
    except BaseException:
        w.boundary, c.pending_sensors = old, pending
        raise
