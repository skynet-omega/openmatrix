"""Prospective, one-shot readout screen on exposed historical neural records."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from motor_port import CalibratedPort, require


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(row):
    p = Path(row['path'])
    require(sha(p) == row['sha256'], 'Input changed: '+str(p))
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k].copy() for k in ('DN_q_usada','DN_q_actual','DN_baseline',
            'command_yaw_rate_rad_s','command_forward_mm_s','sensores_usados','CNS_time_ns')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    root = Path(__file__).resolve().parent
    plan = json.loads((root/'PLAN.json').read_text())
    require(not args.out.exists(), 'Unique output directory required')
    args.out.mkdir(parents=True)
    prep = plan['calibration']['preparation_rows']
    sham = load(plan['inputs']['sham'])
    require(np.array_equal(sham['sensores_usados'][prep:], np.zeros_like(sham['sensores_usados'][prep:])), 'Calibration is not neutral')
    ids = plan['readout']['neuron_ids']
    train = sham['DN_q_usada'][prep:prep+200, 2:4]
    port = CalibratedPort.from_neutral_record(train, ids, plan['inputs']['sham']['sha256'])
    # Persist calibration before opening outcome records; no iterative fitting.
    (args.out/'CALIBRATION.json').write_text(json.dumps(asdict(port), indent=2)+'\n')
    records, traces = {}, {}
    for name, row in plan['inputs'].items():
        z = sham if name == 'sham' else load(row)
        q = z['DN_q_usada'][prep:]; b = z['DN_baseline'][prep:]
        require(np.isfinite(q).all() and np.isfinite(b).all(), 'Non-finite state')
        require(np.all(np.diff(z['CNS_time_ns']) == 1000000), 'Wrong sample clock')
        old = 5*np.tanh(250*((q[:,2]-b[:,2])-(q[:,3]-b[:,3])))
        require(np.max(abs(np.deg2rad(old)-z['command_yaw_rate_rad_s'][prep:])) <= 1e-12, 'Parent readout mismatch')
        new = port.command(q[:,2:4], ids)
        # Independent scalar recomputation. This validates software only.
        scalar = np.array([5*math.tanh(((float(v[2])-float(v[3]))-port.offset_q)/port.scale_q) for v in q])
        require(np.max(abs(new-scalar)) <= 1e-12, 'Scalar/vector readout mismatch')
        # Diagnostic only: resetting a single prepared sample is not a candidate.
        prepared = z['DN_q_actual'][prep-1]
        reset = 5*np.tanh(250*((q[:,2]-prepared[2])-(q[:,3]-prepared[3])))
        traces[name+'_parent'] = old
        traces[name+'_candidate'] = new
        traces[name+'_reset_diagnostic'] = reset
        eval_slice = slice(200, 400) if name == 'sham' else slice(None)
        def metrics(a):
            a = a[eval_slice]
            return dict(net_command_deg=float(.001*math.fsum(a)),
                        absolute_command_deg=float(.001*math.fsum(abs(a))),
                        positive_fraction=float(np.mean(a>0)),
                        near_ceiling_fraction=float(np.mean(abs(a)>4.5)))
        records[name] = dict(parent=metrics(old), candidate=metrics(new),
            reset_diagnostic=metrics(reset), samples=len(new[eval_slice]),
            neural_forward_channels_vary=bool(np.any(np.ptp(q[:,:2],axis=0))),
            forward_command_range_mm_s=[float(np.min(z['command_forward_mm_s'][prep:])),
                                         float(np.max(z['command_forward_mm_s'][prep:]))])
        require(time.perf_counter()-start < plan['budget']['cpu_wall_s'], 'CPU budget exceeded')
    net = lambda name: records[name]['candidate']['net_command_deg']
    gates = dict(
        heldout_rest_nonregression=abs(net('sham')) <= abs(records['sham']['parent']['net_command_deg'])+plan['screen']['rest_net_nonregression_margin_deg'],
        static_left_direction=net('odor_left')>0,
        static_right_direction=net('odor_right')<0,
        spatial_plus_direction=net('spatial_plus')>0,
        spatial_minus_direction=net('spatial_minus')<0,
        static_bilateral_separation=net('odor_left')-net('odor_right')>=plan['screen']['minimum_pair_directional_separation_deg'],
        spatial_bilateral_separation=net('spatial_plus')-net('spatial_minus')>=plan['screen']['minimum_pair_directional_separation_deg'])
    # Cheap semantic failures: identity reversal, invalid state, absent scale.
    failures = []
    for label, action in (
        ('swapped_identity', lambda: port.command(train, ids[::-1])),
        ('nonfinite_input', lambda: port.command(np.array([[np.nan,.5]]), ids)),
        ('missing_variation', lambda: CalibratedPort.from_neutral_record(np.full((200,2),.5),ids,plan['inputs']['sham']['sha256']))):
        try: action()
        except ValueError: failures.append(label)
    require(len(failures)==3, 'Semantic corruption escaped')
    np.savez_compressed(args.out/'commands.npz', **traces)
    out = dict(classification='PROMETEDOR_NO_CONFIRMADO' if all(gates.values()) else 'DESCARTADO',
        scope='One normalization hypothesis, offline exposed records. Commands are not simulated body trajectories or biological measurements.',
        plan_sha256=sha(root/'PLAN.json'),source_sha256={p.name:sha(p) for p in (root/'screen.py',root/'motor_port.py')},
        records=records,gates=gates,corruptions_rejected=failures,parameter_searches=0,
        new_organism_runs=0,stage4_admitted=False,stage5_admitted=False)
    (args.out/'RESULT.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    rows='\n'.join(f"| {name} | {r['parent']['net_command_deg']:+.6f} | {r['candidate']['net_command_deg']:+.6f} | {r['reset_diagnostic']['net_command_deg']:+.6f} |" for name,r in records.items())
    report=f'''# Puerto motor: criba de una hipótesis de normalización\n\n**{out['classification']}** en el alcance de este filtro. Parámetros calculados sólo con los primeros 200 ms del sham; ninguna búsqueda de ganancia ni aprendizaje con etiquetas olfativas. El sham evaluado usa los 200 ms siguientes. Las otras vidas ya estaban expuestas: no son confirmación ciega.\n\n| Registro | Lector actual, integral ° | Puerto calibrado, integral ° | Reset puntual diagnóstico, integral ° |\n|---|---:|---:|---:|\n{rows}\n\nCriterios: `{json.dumps(gates,ensure_ascii=False)}`.\n\nSon mandos recalculados con estados neuronales congelados; no movimientos corporales predichos ni resultados en lazo cerrado. Alterar el lector cambiaría la propiocepción de una nueva vida. El reset de baseline se muestra como sensibilidad y no se promociona. La normalización de q no lo convierte en frecuencia de disparo o calcio.\n\nEtapas 4 y 5 abiertas. Cero organismos nuevos, cero cambios del motor vigente, sin retocar parámetros tras observar el resultado.\n'''
    (args.out/'REPORT.md').write_text(report)
    runtime=dict(wall_s=time.perf_counter()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    require(runtime['peak_rss_gib'] < plan['budget']['max_rss_gib'], 'Memory budget exceeded')
    (args.out/'RUNTIME.json').write_text(json.dumps(runtime,indent=2)+'\n')
    print(json.dumps(dict(classification=out['classification'],gates=gates,runtime=runtime)))


if __name__ == '__main__':
    main()
