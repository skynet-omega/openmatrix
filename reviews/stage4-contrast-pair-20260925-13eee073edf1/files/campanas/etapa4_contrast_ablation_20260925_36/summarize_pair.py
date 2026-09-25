"""Recompute the displayed measurements from the two raw sensory-replay traces.

Run after verify_pair.py and the external verifier. This produces a report and
figure, not an additional admission test or a new numerical reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from chatgpt_verificar_cintas_cd import cargar, metricas


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angular_error(z, source):
    w, x, y, zq = z['qpos'][:, 3:7].T
    yaw = np.arctan2(2 * (w*zq + x*y), 1 - 2 * (y*y + zq*zq))
    delta = source - 10*z['qpos'][:, :2]
    angle = np.arctan2(delta[:, 1], delta[:, 0]) - yaw
    return np.abs(np.rad2deg(np.arctan2(np.sin(angle), np.cos(angle))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    verified = json.loads((root/'PAIR_VERIFIED_01.json').read_text())
    external = json.loads((root/'EXTERNAL_VERIFIED_01.json').read_text())
    plan = json.loads((root/'PLAN.json').read_text())
    paths = [root/name/'traces.npz' for name in ('identity_01', 'no_contrast_01')]
    for path, key in zip(paths, ('control', 'intervention')):
        require(sha(path) == verified[key]['trace_sha256'], 'Trace changed after verification')
    require(external['status'] == 'COMPARACION_DESCRIPTIVA_NO_ADMISION', 'External verifier failed')
    require(sha(root/'PLAN.json') == verified['control']['plan_sha256'], 'Plan changed')
    field = json.loads((root.parent/'etapa4_long_trajectory_20260925_35/CAMPOS.json').read_text())
    source = np.asarray(field['minus']['source_mm'], dtype=np.float64)
    traces = [cargar(path) for path in paths]
    metrics = [metricas(z, source) for z in traces]
    errors = [angular_error(z, source) for z in traces]
    delta = float(errors[1][-1] - errors[0][-1])
    require(abs(delta - verified['delta_error_noD_minus_D_deg']) < 1e-12, 'Primary metric differs')
    require(abs(delta - external['efecto_D_bearing_deg']) < 1e-12, 'Independent metric differs')
    concentration_mean = [np.mean(z['sensores_usados'][40:, :2], axis=1) for z in traces]
    require(np.array_equal(*concentration_mean), 'Mean input concentration changed')
    # These two means deliberately ignore pair ordering. They do not identify a
    # bilateral comparator, and q is not assumed to be a measured firing rate.
    pn_mean = [np.mean(z['PN_q_legacy'][40:], axis=1) for z in traces]
    dn_mean = [np.mean(z['DN_q_actual'][40:, 2:4], axis=1) for z in traces]
    runtime = [json.loads((path.parent/'RESULT.json').read_text()) for path in paths]
    require(all(x['status'] == 'COMPLETE' for x in runtime), 'Incomplete arm')
    threshold = plan['diagnostic']['effect_size_screen_deg']
    effect = ('D_HELPS_IN_THIS_DISCRETIZED_MODEL' if delta >= threshold else
              'D_WORSENS_IN_THIS_DISCRETIZED_MODEL' if delta <= -threshold else
              'BELOW_MATERIAL_DIAGNOSTIC_SCREEN')
    require(effect == verified['effect_screen'], 'Effect classification differs')
    measures = dict(
        control=metrics[0], no_contrast=metrics[1],
        delta_error_noD_minus_D_deg=delta, effect_screen=effect,
        mean_input_concentration_exact=True,
        max_abs_change_mean_PN_q=float(np.max(np.abs(pn_mean[1]-pn_mean[0]))),
        max_abs_change_mean_DNb05_q=float(np.max(np.abs(dn_mean[1]-dn_mean[0]))),
        wall_seconds=[x['wall_total_s'] for x in runtime],
        total_wall_seconds=sum(x['wall_total_s'] for x in runtime),
        numerical_resolution_1s_confirmed=False,
        stage4_admitted=False, stage5_admitted=False,
        report_source_sha256=sha(Path(__file__)),
        traces_sha256={path.parent.name: sha(path) for path in paths})
    require(not args.out.exists(), 'Output already exists')
    args.out.mkdir(parents=True)
    (args.out/'MEASURES.json').write_text(json.dumps(measures, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout='constrained')
    t = np.arange(1001) / 1000
    for z, e, label, color in zip(traces, errors,
            ('L/R originales', 'Antenas igualadas'), ('#2563eb', '#d97706')):
        axes[0].plot(t, e[39:], label=label, color=color, lw=1.8)
        axes[1].plot(t, z['yaw_delta_deg'][39:]-z['yaw_delta_deg'][39], label=label, color=color, lw=1.8)
    axes[0].set_ylabel('Error absoluto hacia la fuente (°)')
    axes[0].legend(frameon=False)
    axes[1].set_ylabel('Giro desde la preparación (°)')
    axes[2].plot(t, errors[1][39:]-errors[0][39:], color='#0f766e', lw=1.8)
    axes[2].axhline(0, color='gray', lw=.8)
    axes[2].set_ylabel('Error igualadas − error L/R (°)')
    for ax in axes:
        ax.set_xlabel('Tiempo desde el estímulo (s)')
        ax.grid(alpha=.2)
    fig.suptitle('Campaña 36 · Entradas grabadas · Diagnóstico sin admisión de etapas', fontsize=13)
    fig.savefig(args.out/'COMPARISON.png', dpi=170)
    plt.close(fig)

    rows = [
        ('Error inicial hacia la fuente (°)', *[m['bearing_deg'][0] for m in metrics]),
        ('Error a 400 ms (°)', *[m['bearing_deg'][1] for m in metrics]),
        ('Error a 1.000 ms (°)', *[m['bearing_deg'][2] for m in metrics]),
        ('Avance hacia la fuente (mm)', *[m['progreso_mm'] for m in metrics]),
        ('Mando angular neto integrado (°)', *[m['command_integral_deg'] for m in metrics]),
        ('Módulo integrado del mando (°)', *[m['command_L1_deg'] for m in metrics]),
    ]
    table = '\n'.join(f'| {name} | {a:.9f} | {b:.9f} |' for name, a, b in rows)
    text = f'''# Resultado descriptivo de la pareja de 1 s

Criba mecánica: **{effect}**. Diferencia de error final sinD−D: **{delta:+.9f}°**. Positivo indica que conservar la asimetría de esta cinta ayuda. La criba de {threshold}° y la resolución propuesta siguen provisionales; no hay referencia refinada de ambos brazos a 1 s.

| Observable | L/R originales | Antenas igualadas |
|---|---:|---:|
{table}

El mínimo histórico de mejora de rumbo de 0,5° desde preparación y desde 400 ms se conserva: control={metrics[0]['mejora_historica_0p5']}, igualadas={metrics[1]['mejora_historica_0p5']}. El avance tónico no demuestra navegación.

La concentración media de entrada se conserva exactamente. La media neural puede cambiar: diferencia máxima en la media del par PN={measures['max_abs_change_mean_PN_q']:.9g}; del par DNb05={measures['max_abs_change_mean_DNb05_q']:.9g}, ambas en unidades del estado q. Esto no identifica por sí solo una conexión de signo incorrecto ni convierte q en una medida fisiológica.

Tiempo de ejecución de los organismos: {measures['wall_seconds'][0]:.3f} s y {measures['wall_seconds'][1]:.3f} s; total {measures['total_wall_seconds']:.3f} s. Dos vidas distintas de 40+1000 ms; no una trayectoria continua de 2 s. La contención de GPU documentada impide atribuir su diferencia de coste a la intervención sensorial.

Ambos brazos usan entradas grabadas. **Etapas 4 y 5 abiertas**: esta prueba identifica el efecto total de igualar entradas en el modelo discretizado, no utilidad de feedback espacial ni equivalencia biológica. No se modificaron cerebro, lector, cuerpo ni criterios para obtener este resultado.

![Comparación de trazas](COMPARISON.png)
'''
    (args.out/'REPORT.md').write_text(text)
    print(json.dumps(measures, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
