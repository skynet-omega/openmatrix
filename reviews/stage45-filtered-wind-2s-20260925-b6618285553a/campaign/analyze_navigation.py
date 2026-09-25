"""Rebuild the two-second trajectory and wind screen from raw saved traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def yaw_from_qpos(q):
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    return np.rad2deg(np.arctan2(2 * (w*z+x*y), 1-2*(y*y+z*z)))


def wrap(degrees):
    return (degrees + 180) % 360 - 180


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    record = json.loads((run/'RESULT.json').read_text())
    need(record['status'] == 'COMPLETE' and record['completed_trial_ms'] == 2000
         and record['completed_preparation_ms'] == 40, 'Full organism run incomplete')
    audit = json.loads((run/'MOTOR_WIND_AUDIT.json').read_text())
    need(audit['trial_steps'] == 2000 and audit['body_calls'] == 80000
         and audit['wind_substeps'] == audit['expected_wind_substeps'] == 800,
         'Physical grid or wind pulse incomplete')
    with np.load(run/'traces.npz', allow_pickle=False) as data:
        phase = data['fase']
        prep = np.flatnonzero(phase == 'preparacion')
        trial = np.flatnonzero(phase == 'ensayo')
        need(len(prep) == 40 and len(trial) == 2000
             and np.array_equal(data['paso'][prep], np.arange(1, 41))
             and np.array_equal(data['paso'][trial], np.arange(1, 2001)),
             'Clock or phase mismatch')
        for key in ('CNS_time_ns', 'PN_time_ns', 'body_time_ns'):
            need(np.array_equal(data[key][trial], data['CNS_time_ns'][trial]),
                 'CNS/PN/body clock mismatch: '+key)
        need(np.all(data['sensores_usados'][prep] == 0), 'Preparation consumed odor')
        q = data['qpos'][trial]
        position = data['position_mm'][trial]
        need(q.shape[0] == position.shape[0] == 2000 and np.isfinite(q).all()
             and np.isfinite(position).all(), 'Body trace nonfinite')
        need(np.max(np.abs(position-data['qpos'][trial, :3]*10)) <= 1e-12,
             'Position is not body qpos')
        raw = data['neural_command_raw_rad_s'][trial]
        applied = data['motor_filter_applied_rad_s'][trial]
        actual = data['command_yaw_rate_rad_s'][trial]
        wind = data['wind_torque_native'][trial]
        need(np.array_equal(applied, actual), 'Filtered command differs from physical controller')
        need(np.max(abs(actual)) <= np.deg2rad(5.)+1e-15,
             'Applied command exceeded contact-device envelope')
        need(np.array_equal(np.flatnonzero(wind != 0), np.arange(1000, 1020))
             and np.all(wind[1000:1020] == audit['wind_torque_native']),
             'Wind timing/dose differs from contract')
        need(np.isfinite(raw).all() and np.isfinite(applied).all(), 'Motor data nonfinite')
        source = np.asarray(json.loads((run/'GAUSSIAN_SPEC.json').read_text())[record['field']]['source_mm'], float)
        yaw = yaw_from_qpos(q)
        bearing = np.rad2deg(np.arctan2(source[1]-position[:, 1], source[0]-position[:, 0]))
        error = wrap(bearing-yaw)
        distance = np.linalg.norm(position[:, :2]-source, axis=1)
        prep_yaw = yaw_from_qpos(data['qpos'][prep][-1:])[0]
        prep_pos = data['position_mm'][prep][-1, :2]
        prep_error = float(wrap(np.rad2deg(np.arctan2(source[1]-prep_pos[1],
                                                   source[0]-prep_pos[0]))-prep_yaw))
        checkpoints = {}
        for ms in (1, 400, 1000, 1020, 1500, 2000):
            i = ms-1
            checkpoints[str(ms)] = dict(position_mm=position[i].tolist(), yaw_deg=float(yaw[i]),
                                        source_bearing_deg=float(bearing[i]),
                                        signed_bearing_error_deg=float(error[i]),
                                        absolute_bearing_error_deg=float(abs(error[i])),
                                        source_distance_mm=float(distance[i]),
                                        command_deg_s=float(np.rad2deg(actual[i])),
                                        upright=float(data['upright'][trial[i]]),
                                        active_contacts=int(np.count_nonzero(data['contact_active'][trial[i]])))
        result = dict(schema='stage45_two_second_trace_rebuild_v1',status='COMPLETE_SCREEN',
                      stage4_admission=False,stage5_admission=False,
                      run_result_sha256=digest(run/'RESULT.json'),trace_sha256=digest(run/'traces.npz'),
                      motor_wind_audit_sha256=digest(run/'MOTOR_WIND_AUDIT.json'),
                      analysis_source_sha256=digest(__file__),field=record['field'],
                      source_mm=source.tolist(),prepared_bearing_error_deg=prep_error,
                      checkpoints=checkpoints,
                      raw_integral_deg=float(np.rad2deg(np.sum(raw)*0.001)),
                      raw_absolute_integral_deg=float(np.rad2deg(np.sum(abs(raw))*0.001)),
                      applied_integral_deg=float(np.rad2deg(np.sum(applied)*0.001)),
                      applied_absolute_integral_deg=float(np.rad2deg(np.sum(abs(applied))*0.001)),
                      raw_sign_changes=int(np.count_nonzero(np.diff(np.sign(raw)))),
                      applied_sign_changes=int(np.count_nonzero(np.diff(np.sign(applied)))),
                      minimum_contacts=int(np.min(np.count_nonzero(data['contact_active'][trial], axis=1))),
                      minimum_upright=float(np.min(data['upright'][trial])),
                      bearing_abs_change_prepared_to_1000_deg=float(abs(error[999])-abs(prep_error)),
                      bearing_abs_change_after_wind_to_final_deg=float(abs(error[-1])-abs(error[1019])),
                      distance_change_prepared_to_final_mm=float(distance[-1]-np.linalg.norm(prep_pos-source)),
                      interpretation='One filtered/thresholded motor arm with wind; no matched no-wind or unfiltered control. A favorable trace alone cannot establish Stage4 navigation or Stage5 feedback.')
        (run/'ANALYSIS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,
                                               allow_nan=False)+'\n')
        t = np.arange(1, 2001)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        axes[0, 0].plot(position[:, 0], position[:, 1], lw=1.3, label='MuJoCo')
        axes[0, 0].scatter(source[0], source[1], marker='*', s=150, label='fuente fija')
        for ms, color in ((1, 'tab:green'), (1000, 'tab:orange'), (1020, 'tab:red'), (2000, 'tab:blue')):
            axes[0, 0].scatter(*position[ms-1, :2], color=color, s=35)
            axes[0, 0].annotate(f'{ms} ms', position[ms-1, :2])
        axes[0, 0].set(xlabel='x (mm)', ylabel='y (mm)', title='Trayectoria corporal')
        axes[0, 0].axis('equal'); axes[0, 0].legend()
        axes[0, 1].plot(t, np.abs(error), label='|error angular a fuente|')
        axes[0, 1].axvspan(1000, 1020, color='tab:red', alpha=.2, label='viento')
        axes[0, 1].set(xlabel='tiempo de ensayo (ms)', ylabel='grados', title='Orientación')
        axes[0, 1].legend()
        axes[1, 0].plot(t, np.rad2deg(raw), alpha=.6, label='lector neural crudo')
        axes[1, 0].plot(t, np.rad2deg(applied), lw=1, label='mando aplicado')
        axes[1, 0].axvspan(1000, 1020, color='tab:red', alpha=.2)
        axes[1, 0].set(xlabel='tiempo de ensayo (ms)', ylabel='°/s', title='Interfaz motora')
        axes[1, 0].legend()
        axes[1, 1].plot(t, distance)
        axes[1, 1].axvspan(1000, 1020, color='tab:red', alpha=.2)
        axes[1, 1].set(xlabel='tiempo de ensayo (ms)', ylabel='mm', title='Distancia a la fuente')
        fig.savefig(run/'TRAJECTORY.png', dpi=160)
        plt.close(fig)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
