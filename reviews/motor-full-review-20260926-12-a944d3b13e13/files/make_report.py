"""Present verified saved evidence; does not execute or tune the organism."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    pair = json.loads((HERE/'PAIR2000.json').read_text())
    runtime = json.loads((HERE/'RUNTIME_PAIR2000.json').read_text())
    if pair['status'] not in ('PASS', 'FAIL') or runtime['status'] != 'PASS_RUNTIME_CONTRACT':
        raise ValueError('No valid complete pair and runtime contract to present')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms = [('stable','Estable','#34465c'),('reviewed','Revisado','#dc8a21')]
    traces = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for engine, label, color in arms:
        with np.load(HERE/(engine+'_2000ms_01')/'traces.npz', allow_pickle=False) as data:
            z = {k:data[k] for k in data.files}
        traces[engine] = z
        time_s = z['paso']/1000.
        style = '-' if engine == 'stable' else '--'
        axes[0,0].plot(z['position_mm'][:,0], z['position_mm'][:,1], style,
                       color=color, label=label, lw=1.8)
        axes[0,1].plot(time_s, z['yaw_delta_deg'], style, color=color, label=label)
        axes[1,0].plot(time_s, z['command_yaw_rate_rad_s']*180/np.pi, style,
                       color=color, label=label)
    axes[0,0].set(xlabel='x (mm)', ylabel='y (mm)', title='Trayectoria del cuerpo')
    axes[0,0].axis('equal')
    axes[0,0].legend()
    axes[0,1].set(xlabel='Tiempo simulado (s)', ylabel='Cambio de yaw (grados)',
                   title='Respuesta con viento común corregido')
    axes[1,0].set(xlabel='Tiempo simulado (s)', ylabel='Mando de yaw (grados/s)',
                   title='Mando motor aplicado')
    for ax in (axes[0,1], axes[1,0]):
        ax.axvspan(1., 1.02, alpha=.15, color='#3a8ab5', label='Viento')
    timing = pair['timing']
    seconds = [timing['stable_process_s'], timing['reviewed_process_s']]
    axes[1,1].bar(['Estable','Revisado'], np.asarray(seconds)/60,
                  color=[a[2] for a in arms], width=.55)
    axes[1,1].set(ylabel='Minutos de proceso por 2 s simulados', title='Coste integral medido')
    for i, value in enumerate(seconds):
        axes[1,1].text(i, value/60, f'{value/60:.2f}', ha='center', va='bottom')
    for ax in axes.ravel():
        ax.grid(alpha=.18)
    differing_ticks = np.flatnonzero(traces['stable']['command_yaw_rate_rad_s'] !=
                                    traces['reviewed']['command_yaw_rate_rad_s']) + 1
    for tick in differing_ticks:
        axes[1,0].axvline(tick/1000., color='#aa3030', lw=.8, alpha=.8)
    axes[1,0].text(.03, .04, f'Mando diferente en {len(differing_ticks)}/2000 pasos',
                   transform=axes[1,0].transAxes, color='#aa3030')
    fig.suptitle('Motor neuronal: dos vidas continuas de 2 segundos\n'
                 f"Contrato funcional: {pair['status']}; comparación del mismo modelo")
    fig.savefig(HERE/'COMPARACION_2S.png', dpi=180)
    plt.close(fig)
    fields = pair['fields']
    exact = [key for key, value in fields.items() if value['exact']]
    events = pair['events']['by_context']
    admitted = pair['status'] == 'PASS'
    failed_gates = [key for key, value in pair['gates'].items() if not value]
    command_ticks = (np.flatnonzero(traces['stable']['command_yaw_rate_rad_s'] !=
                                   traces['reviewed']['command_yaw_rate_rad_s']) + 1).tolist()
    minutes_per_second = timing['reviewed_process_s']/120
    targets = {str(bound): minutes_per_second <= bound for bound in (12, 20, 25)}
    target_description = ('cumple el objetivo de hasta 20 min/s' if targets['20'] else
                          'cumple la holgura de 25 min/s, pero no el objetivo de 20 min/s' if targets['25'] else
                          'queda por encima del objetivo y de la holgura de 25 min/s')
    metrics = dict(status=('CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA' if admitted else
                           'PROMETEDOR_NO_CONFIRMADO'), pair_status=pair['status'],
                   promoted=admitted, failed_gates=failed_gates,
                   different_yaw_command_ticks=command_ticks,
                   scope='One prepared state, one Gaussian condition, corrected physical wind, continuous2s',
                   functional_status=pair['functional_status'], runtime_status=runtime['status'],
                   trace_fields_exact=len(exact), trace_fields_total=len(fields),
                   body_qpos_exact=fields['qpos']['exact'], body_qvel_exact=fields['qvel']['exact'],
                   yaw_commands_exact=fields['command_yaw_rate_rad_s']['exact'],
                   forward_commands_exact=fields['command_forward_mm_s']['exact'],
                   contacts_exact=fields['contact_active']['exact'],
                   candidate_minutes_per_simulated_second=minutes_per_second,
                   performance_targets_met=targets,
                   timing=timing, committed_events=events['committed'], predictor_events=events['predictor'],
                   final_CNS=pair['neural']['2000']['cns'],
                   gates=pair['gates'], performance_targets_minutes_per_simulated_second=[12,20,25])
    (HERE/'METRICAS.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    text = f'''# Resultado de la revisión integral y comparación de 2 segundos

**{metrics['status']}**, para la condición indicada. Resultado del verificador
funcional: **{pair['status']}**; identidad de ejecución: **{runtime['status']}**.
Fallos del contrato: {failed_gates}. No se afirma equivalencia neurobiológica
general ni mejora de navegación por cambiar el integrador. La corrida negativa
se conserva: no se modificaron criterios ni fuentes para convertirla en PASS.

| Medición | Estable | Motor revisado |
|---|---:|---:|
| Vida simulada | 2.000 ms continuos | 2.000 ms continuos |
| Avance del organismo (s) | {timing['stable_advance_s']:.3f} | {timing['reviewed_advance_s']:.3f} |
| Proceso completo (min) | {seconds[0]/60:.3f} | {seconds[1]/60:.3f} |
| Minutos de proceso por segundo simulado | {seconds[0]/120:.3f} | {seconds[1]/120:.3f} |

Aceleración integral medida: **{timing['process_speedup']:.3f}×**;
aceleración del avance: **{timing['advance_speedup']:.3f}×**. Es una pareja local,
con orden estable→revisado, no un intervalo estadístico sobre múltiples máquinas.

Meta solicitada: 12–20 minutos por segundo simulado, con holgura hasta 25.
El coste integral de esta pareja es **{minutes_per_second:.3f} min/s**:
{target_description}. La fidelidad funcional y esta meta de rendimiento se
informan por separado; no se cambian los criterios para convertir una en otra.

{len(exact)}/{len(fields)} campos de la traza son exactos. Igualdad exacta de
configuración corporal (`qpos`): {fields['qpos']['exact']}; velocidad: {fields['qvel']['exact']};
mando yaw: {fields['command_yaw_rate_rad_s']['exact']}; mando forward:
{fields['command_forward_mm_s']['exact']}; contactos: {fields['contact_active']['exact']}.
Los criterios completos y todas las discrepancias están en `PAIR2000.json`.

Pasos con mando de yaw diferente: **{command_ticks}**. Error máximo de posición:
{fields['position_mm']['max_abs']:.8g} mm; error máximo de yaw:
{fields['yaw_delta_deg']['max_abs']:.8g} grados. Las diferencias de posición y yaw cumplen
sus límites, pero no sustituyen la exigencia congelada de mando idéntico en
cada paso. No se promueve automáticamente el motor por ser más rápido.

Eventos comprometidos: {events['committed']['stable_count']} estable y
{events['committed']['reviewed_count']} revisado; bloques con identidades diferentes:
{events['committed']['identity_different_blocks']}. Los predictores descartados
se informan por separado; los payloads y tiempos no se declaran exactos cuando
no lo son. El informe conserva las exclusiones de emparejamiento.

![Comparación de trayectoria, mando y coste](COMPARACION_2S.png)

Se corrigieron tres mecanismos concretos: dependencia inicial de streams PN,
extremo canónico de eventos RK y mapa del torque con la pose actual. Se revisaron
código, matemáticas, datos y hardware antes del ensayo. Las pruebas CPU/GPU
dirigidas, el diagnóstico real de20ms y la pareja100ms precedieron esta vida.
La referencia histórica con reinicio frío se conservó como antecedente; se
ejecutó una nueva referencia continua con la misma corrección física.

La revisión y oportunidades justificadas están en `REVISION_INTEGRAL.md` y los
informes de `reviews/`. Se conserva el modelo y su acoplamiento: control local de
error y coincidencia funcional no demuestran convergencia global del sistema.
PN629 general y plasticidad siguen apagados según la preparación; el cuerpo usa
su prótesis de contacto. Este ensayo no demuestra vuelo.

Las fuentes, parámetros y criterios permanecieron congelados desde antes de
los100ms. El complemento externo `check_runtime.py` cierra una omisión del
verificador original —comprobar el ejecutor realmente instalado— sin cambiar
umbrales ni repetir vidas. Los snapshots finales guardan propietarios científicos
y memoria motora con `restart_tested=false`: todavía no certifican un reinicio
genérico del organismo entre procesos.
'''
    (HERE/'RESULTADOS.md').write_text(text)
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()
