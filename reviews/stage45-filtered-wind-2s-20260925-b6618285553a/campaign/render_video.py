"""Render recorded MuJoCo poses beside the measured source and motor trace."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import mujoco as mj
import numpy as np
from PIL import Image

OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
sys.path.insert(0, str(OLD/'src'))
from session_io import read_state
from rh_tarsal_body import RHTarsalBody


def need(value, message):
    if not value:
        raise ValueError(message)


def yaw(q):
    w, x, y, z = q[3:7]
    return float(np.rad2deg(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--prepared-state', type=Path,
                        help='Prepared snapshot of the original segment when the trace was continued')
    args = parser.parse_args()
    run = args.run.resolve()
    result = json.loads((run/'RESULT.json').read_text())
    need(result['status'] == 'COMPLETE' and result['completed_trial_ms'] == 2000,
         'Video requires a complete two-second organism trace')
    need((run/'ANALYSIS.json').is_file(), 'Analyze the trace first')
    with np.load(run/'traces.npz', allow_pickle=False) as z:
        trial = np.flatnonzero(z['fase'] == 'ensayo')
        need(len(trial) == 2000, 'Wrong video clock')
        qpos = z['qpos'][trial].copy()
        qvel = z['qvel'][trial].copy()
        pos = z['position_mm'][trial, :2].copy()
        command = z['command_yaw_rate_rad_s'][trial].copy()
    source = np.asarray(json.loads((run/'GAUSSIAN_SPEC.json').read_text())[result['field']]['source_mm'], float)
    prepared = args.prepared_state.resolve() if args.prepared_state else run/'prepared_state'
    state = read_state(prepared/'session')
    body = RHTarsalBody.from_state(state['body'])
    del state
    need(qpos.shape == (2000, body.model.nq) and qvel.shape == (2000, body.model.nv),
         'Trace/model shape mismatch')
    frame_indices = np.rint(np.linspace(0, 1999, 120)).astype(int)
    mins = np.minimum(pos.min(axis=0), source)-0.2
    maxs = np.maximum(pos.max(axis=0), source)+0.2
    centre = (mins+maxs)/2
    span = float(max(maxs-mins))
    mins = centre-span/2
    maxs = centre+span/2
    def pixel(xy):
        p = np.asarray(xy)
        u = int(45+(p[0]-mins[0])/(maxs[0]-mins[0])*310)
        v = int(365-(p[1]-mins[1])/(maxs[1]-mins[1])*310)
        return u, v
    target = run/'MUJOCO_TRAJECTORY.mp4'
    need(not target.exists(), 'Preserve prior render: '+str(target))
    ffmpeg = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
                               '-f','rawvideo','-pix_fmt','rgb24','-s','1040x480',
                               '-r','60','-i','-','-an','-c:v','libx264',
                               '-preset','veryfast','-crf','22','-pix_fmt','yuv420p',
                               str(target)], stdin=subprocess.PIPE)
    renderer = None
    try:
        renderer = mj.Renderer(body.model, height=480, width=640)
        for n, index in enumerate(frame_indices):
            body.data.qpos[:] = qpos[index]
            body.data.qvel[:] = qvel[index]
            mj.mj_forward(body.model, body.data)
            renderer.update_scene(body.data, camera='track1')
            left = np.asarray(renderer.render(), dtype=np.uint8)
            panel = np.full((480, 400, 3), (18, 27, 39), np.uint8)
            cv2.rectangle(panel, (40, 50), (360, 370), (55, 72, 90), 1)
            path = np.asarray([pixel(x) for x in pos[:index+1]], np.int32)
            if len(path) >= 2:
                cv2.polylines(panel, [path], False, (80, 180, 245), 2, cv2.LINE_AA)
            source_px = pixel(source)
            cv2.drawMarker(panel, source_px, (75, 75, 255), cv2.MARKER_STAR, 20, 2)
            fly_px = pixel(pos[index])
            cv2.circle(panel, fly_px, 6, (100, 245, 120), -1)
            heading = np.deg2rad(yaw(qpos[index]))
            tip = (int(fly_px[0]+18*np.cos(heading)), int(fly_px[1]-18*np.sin(heading)))
            cv2.arrowedLine(panel, fly_px, tip, (100, 245, 120), 2, tipLength=.35)
            delta = source-pos[index]
            bearing = float(np.rad2deg(np.arctan2(delta[1], delta[0])))
            error = float((bearing-yaw(qpos[index])+180)%360-180)
            ms = int(index+1)
            lines = [f'{ms} / 2000 ms', f'Error a fuente: {error:+.2f} deg',
                     f'Mando: {np.rad2deg(command[index]):+.2f} deg/s',
                     'VIENTO FISICO' if 1001 <= ms <= 1020 else 'Fuente fija; lazo activo']
            for k, line in enumerate(lines):
                cv2.putText(panel, line, (24, 400+k*19), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(panel, 'Cuerpo MuJoCo con rodillos motores', (24, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1, cv2.LINE_AA)
            frame = np.concatenate((left, panel[:, :, ::-1]), axis=1)
            need(frame.shape == (480, 1040, 3), 'Frame shape')
            if n == 60:
                Image.fromarray(frame).save(run/'VIDEO_POSTER.png')
            ffmpeg.stdin.write(frame.tobytes())
        ffmpeg.stdin.close()
        code = ffmpeg.wait(timeout=120)
        need(code == 0 and target.is_file() and target.stat().st_size > 0,
             'FFmpeg did not produce a video')
        print(json.dumps({'video':str(target),'frames':len(frame_indices),
                          'fps':60,'bytes':target.stat().st_size,
                          'scope':'Recorded MuJoCo qpos/qvel only; plot overlay does not alter simulation'}))
    finally:
        if renderer is not None:
            renderer.close()
        body.close()
        if ffmpeg.poll() is None:
            ffmpeg.kill()
            ffmpeg.wait()


if __name__ == '__main__':
    main()
