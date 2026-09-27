"""Finite post-processing dependency of47; no new runs, retries or scheduler.

Wait for this already launched queue, then produce the corrected diagnosis,
figure and report. Fail closed if acquisition, coverage or provenance fails.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import hashlib
import json
import math
import resource
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def wait_for_queue():
    plan = read(HERE/'PLAN.json')
    deadline = read(HERE/'LAUNCH.json')['started_unix_s'] + plan['aggregate_wall_s_max']
    while True:
        queue = read(HERE/'QUEUE.json')
        if queue['status'] in ('COMPLETE', 'INCOMPLETE'):
            require(queue['status'] == 'COMPLETE', 'Queue did not complete successfully')
            return queue, plan
        require(time.time() < deadline, 'Original aggregate wall budget exhausted')
        time.sleep(10)


def curves():
    import numpy as np
    selected = [1, 2, 3, 10, 11]  # net, positive, negative, margin, final target
    table = {}
    hashes = {}
    for arm in ('sham', 'odor'):
        arm_result = read(HERE/arm/'RESULT.json')
        require(arm_result['status'] == 'COMPLETE' and arm_result['parent_observations_exact'],
                'Failed acquisition or changed parent observations')
        means, lows, highs, times = [], [], [], []
        blocks = sorted((HERE/arm/'blocks').glob('*ms'))
        require(len(blocks) == 30, 'Incomplete block set')
        for block in blocks:
            manifest = read(block/'MANIFEST.json')
            path = block/'dng100_observed.npz'
            digest = sha(path)
            require(digest == manifest['hashes'][path.name], 'Changed observation block')
            hashes[str(path)] = digest
            with np.load(path) as z:
                r = z['records']
                epochs = np.repeat(np.arange(len(z['trials'])), z['trials'])
                use = z['committed'][epochs] & r[:, 138].astype(bool)
                f = r[use, :128].reshape(-1, 4, 2, 16)[:, :3, :, :]
                h = r[use, 129]
                weights = h[:, None] * np.asarray([2/9, 1/3, 4/9])[None, :]
                values = f[..., selected]
                require(np.isfinite(values).all() and h.sum() > 0, 'Invalid retained samples')
                means.append((values*weights[:, :, None, None]).sum(axis=(0, 1))/h.sum())
                lows.append(values.min(axis=(0, 1)))
                highs.append(values.max(axis=(0, 1)))
                times.append((manifest['first_ms']-1+manifest['last_ms'])/2000)
        table[arm] = dict(mean=np.stack(means), low=np.stack(lows), high=np.stack(highs),
                          seconds=np.asarray(times))
    require(np.array_equal(table['sham']['seconds'], table['odor']['seconds']), 'Mismatched time bins')
    np.savez_compressed(HERE/'CURVAS.npz', **{
        arm+'_'+key: value for arm, fields in table.items() for key, value in fields.items()})
    return table, hashes


def figure(table, verdict):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout='constrained')
    for col, side in enumerate(('izquierda', 'derecha')):
        top, bottom = axes[:, col]
        for arm, color, label in [('sham', '#6b7280', 'Control'), ('odor', '#15803d', 'Olor')]:
            d = table[arm]; x = d['seconds']
            top.plot(x, d['mean'][:, col, 3], color=color, label=label)
            top.fill_between(x, d['low'][:, col, 3], d['high'][:, col, 3], color=color, alpha=.14)
        top.axhline(0, color='#dc2626', linewidth=1, linestyle='--', label='Umbral del modelo')
        top.set_title('DNg100 '+side); top.set_ylabel('Entrada − umbral\n(unidades internas)')
        delta = table['odor']['mean'][:, col, :] - table['sham']['mean'][:, col, :]
        for index, color, label in [(1, '#16a34a', 'Términos positivos'),
                                     (2, '#dc2626', 'Términos negativos'),
                                     (0, '#1d4ed8', 'Suma neta consumida')]:
            bottom.plot(x, delta[:, index], color=color, label=label)
        bottom.axhline(0, color='#94a3b8', linewidth=.7)
        bottom.set_ylabel('Cambio olor − control\n(unidades internas)')
        bottom.set_xlabel('Tiempo simulado (s)')
        for ax in (top, bottom):
            ax.axvspan(1, 3, color='#16a34a', alpha=.06)
            ax.grid(alpha=.15); ax.legend(fontsize=8)
    headline = ('Objetivo final cero en todas las evaluaciones retenidas'
                if verdict['literal_zero'] else 'Objetivos no nulos: revisar la localización')
    fig.suptitle('Entrada efectiva de DNg100 · '+headline, fontsize=13)
    fig.supxlabel('Medias RK por bloques de100 ms; bandas: mínimos/máximos, no incertidumbre estadística.\n'
                  'Una pareja determinista; entradas del modelo, no corrientes medidas en la mosca.', fontsize=9)
    fig.savefig(HERE/'ENTRADAS.png', dpi=160)
    plt.close(fig)


def report(verdict, original, queue):
    rows = []
    for p in verdict['paired']:
        if p['phase'] != 'stimulus':
            continue
        d = p['odor_minus_sham_RK_weighted_means']
        rows.append(f"| {p['id']} | {p['sham_margin_max']:.3f} | {p['odor_margin_max']:.3f} | "
                    f"{d['positive_aux']:.3f} | {d['negative_aux']:.3f} | {d['net']:.3f} |")
    if verdict['generic_suppression_supported_in_recorded_evaluations']:
        conclusion = ('El objetivo de ambas DNg100 permaneció exactamente en cero en las evaluaciones '
            'retenidas que construyen la solución, tanto con olor como en control. La entrada quedó '
            'por debajo del umbral, con ganancia positiva y sin sustitución del objetivo de base. '
            'Esto localiza la ausencia de reclutamiento en el balance y la ley del modelo bajo este '
            'contexto. No aporta evidencia de una respuesta positiva perdida por el lector.')
        decision = ('Conservar el motor numérico y cerrar este diagnóstico. No bajar umbrales, retirar '
            'inhibición ni prolongar esta misma vida para buscar movimiento. El siguiente cambio '
            'necesita una correspondencia independiente entre estímulo, contexto y respuesta neural, '
            'o una ley de circuito restringida por datos; las alternativas y descartes quedan en '
            '[DECISION.md](DECISION.md). No se lanza automáticamente otra simulación.')
    else:
        conclusion = ('El registro requiere localizar una discrepancia o un objetivo no nulo antes de '
            'atribuir el fallo al balance de entradas. La clasificación detallada está en DICTAMEN.json. '
            'No se declara coherencia fisiológica ni éxito funcional.')
        decision = ('Conservar los registros y localizar las evaluaciones señaladas. No modificar '
            'ganancias, duración o criterios para convertir el resultado en éxito.')
    content = f'''# Diagnóstico funcional47 — resultado de la adquisición

{conclusion}

Dos condiciones completas de3s, cada una restaurada desde el mismo preparado. Las trazas,
publicaciones y eventos guardados coinciden exactamente con los primeros3s de45.
Se observaron operandos consumidos en47; no se recuperaron retrospectivamente estados internos
de45 que no se habían guardado. Etapas4/5 continúan abiertas.

| DNg100 | Mayor margen control | Mayor margen olor | Δ positivos | Δ negativos | Δ neto |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Estímulo consumido en1001–3000ms. Los cambios son medias ponderadas por las etapasRK que
construyen la solución, en unidades internas. No son pA, ensayos independientes ni una
descomposición causal. Predictores descartados, intentos rechazados y k4 están separados.

![Entradas reales del modelo](ENTRADAS.png)

{decision}

Tiempo total de la cola: {queue['elapsed_s']/60:.1f}min. CPU de la cola:
{queue['used_cpu_s']:.1f}s. Sin reintentos, cambios de parámetros ni plasticidad.
Presupuesto original6300s/brazo,12600s agregados. El coste de postproceso está en CIERRE.json.

ChatGPT revisó el código sin ejecutar el organismo y detectó un defecto de clasificación:
ausencia de objetivos positivos no equivale a cero literal. El analizador original y
RESULTADOS.json se conservan; [DICTAMEN.json](DICTAMEN.json) corrige únicamente ese punto
usando los extremos registrados. Jev priorizó separar predictores y evolución retenida;
su clasificación no constituye validación científica.

Reproducción de los resúmenes desde los registros: `python analyze.py` y `python diagnosis.py`.
`finish.py` añade figura e informe cuando la cola completa ya existe; no integra neuronas.
[Contexto marcha/parada de45](CONTEXT45.csv), [revisión de código](CHATGPT_CODIGO.md),
[decisión fundada](DECISION.md).
'''
    (HERE/'RESULTADOS.md').write_text(content)


def main():
    started = time.time()
    save(HERE/'CIERRE.json', dict(status='WAITING_FOR_EXISTING_QUEUE', pid=os.getpid(), started_unix_s=started,
        new_scientific_runs=0, source_sha256=sha(Path(__file__))))
    queue, plan = wait_for_queue()
    used = queue['analysis']['final_wait4_cpu_s'] + read(HERE/'CONTEXT45.json')['CPU_s']
    # Reserve10CPU seconds for short exploratory reads made before this finalizer.
    remaining = int(math.floor(plan['analysis_CPU_s_max'] - used - 10))
    require(remaining > 0, 'Analysis budget exhausted')
    resource.setrlimit(resource.RLIMIT_CPU, (remaining, remaining))
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    cpu_start = time.process_time()
    lock = read(HERE/'SOURCES.json')
    require(all(sha(Path(path)) == digest for path, digest in lock.items()), 'Frozen source changed')
    import diagnosis
    diagnosis.main()
    verdict = read(HERE/'DICTAMEN.json')
    original = read(HERE/'RESULTADOS.json')
    table, hashes = curves()
    figure(table, verdict)
    report(verdict, original, queue)
    outputs = ['DICTAMEN.json', 'RESULTADOS.md', 'ENTRADAS.png', 'CURVAS.npz']
    receipt = dict(status='COMPLETE', source_sha256=sha(Path(__file__)),
        started_unix_s=started, completed_unix_s=time.time(),
        postprocess_CPU_s=time.process_time()-cpu_start,
        prior_analysis_CPU_s=used, exploratory_CPU_reserve_s=10,
        analysis_CPU_limit_s=plan['analysis_CPU_s_max'], new_scientific_runs=0,
        observed_block_hashes=hashes, outputs={name:sha(HERE/name) for name in outputs})
    save(HERE/'CIERRE.json', receipt)
    state = ROOT/'ESTADO_ACTUAL.md'
    text = state.read_text()
    marker = '**26-09 — campaña47 en ejecución:'
    if marker in text:
        start = text.index(marker); end = text.index('\n\n', start)
        replacement = ('**26-09 — campaña47 completa; etapas4/5 abiertas.** '
            '[Resultado y figura](campanas/etapa45_dng100_observado_20260926_47/RESULTADOS.md). '
            'Dos brazos de3s reprodujeron las observaciones45; instrumentación y fuentes conservadas. '
            'Clasificación corregida: '+verdict['classification']+'. '
            'Sin nueva simulación automática. [Cierre](campanas/etapa45_dng100_observado_20260926_47/CIERRE.json).')
        state.write_text(text[:start]+replacement+text[end:])
    print(json.dumps({k:v for k,v in receipt.items() if k != 'observed_block_hashes'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(HERE/'CIERRE.json', dict(status='INCOMPLETE', new_scientific_runs=0,
            error=repr(exc), traceback=traceback.format_exc()))
        raise
