"""Ensayo cerrado sobre MATRIX: preparación heredada frente a aire limpio.

No reentrena ni promueve un organismo. Conserva la candidata histórica en ambos
brazos, cambia solo el campo durante la preparación y genera evidencia propia.
Las dependencias GPU se importan exclusivamente dentro del proceso de cada rama.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

PLAN_REL = 'config/matrix_diagnostico_olfativo_v1.json'
MODULO_REL = 'src/matrix_olfactory_diagnostic.py'


def exigir(condicion: bool, mensaje: str) -> None:
    if not condicion:
        raise ValueError(mensaje)


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with Path(ruta).open('rb') as entrada:
        for bloque in iter(lambda: entrada.read(1024 * 1024), b''):
            h.update(bloque)
    return h.hexdigest()


def escribir_json(ruta: Path, valor) -> None:
    temporal = ruta.with_name(ruta.name + '.tmp')
    temporal.write_text(json.dumps(valor, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporal.replace(ruta)


def leer_plan(raiz: Path) -> dict:
    plan = json.loads((raiz / PLAN_REL).read_text(encoding='utf-8'))
    exigir(plan['schema'] == 'matrix_diagnostico_preparacion_v1', 'Esquema desconocido')
    exigir(plan['preparaciones'] == ['heredada', 'aire_limpio'], 'Preparaciones distintas de la entrega')
    exigir(plan['condiciones'] == ['sham', 'uniform', 'odor_left', 'odor_right'], 'Condiciones distintas de la entrega')
    exigir(plan['preparacion_ms'] == 40 and plan['ensayo_ms'] == 100, 'Ventanas distintas de la entrega')
    return plan


def verificar_archivos(raiz: Path, plan: dict) -> dict:
    """Valida identidad; no interpreta que un hash valide fisiología."""
    errores, verificados = [], []
    pendientes = dict(plan['fuentes_requeridas_sha256'])
    checkpoint = raiz / plan['checkpoint']
    pendientes[str(Path(plan['checkpoint']) / 'manifest.json')] = plan['checkpoint_manifest_sha256']
    # Tres capas reales del checkpoint, sin ejecutar sus cargadores.
    for relativo in ('manifest.json', 'core_carrier/manifest.json', 'core_carrier/brain/manifest.json'):
        manifiesto = checkpoint / relativo
        if not manifiesto.is_file():
            errores.append({'ruta': str(manifiesto.relative_to(raiz)), 'error': 'FALTANTE'})
            continue
        datos = json.loads(manifiesto.read_text(encoding='utf-8'))
        for nombre, digest in datos.get('files', {}).items():
            ruta = (manifiesto.parent / nombre).resolve()
            exigir(ruta.is_relative_to(raiz), 'Ruta fuera del proyecto en manifiesto')
            pendientes[str(ruta.relative_to(raiz))] = digest['sha256'] if isinstance(digest, dict) else digest
    for nombre, esperado in sorted(pendientes.items()):
        ruta = raiz / nombre
        if not ruta.is_file():
            errores.append({'ruta': nombre, 'error': 'FALTANTE'})
        else:
            actual = sha256(ruta)
            if actual != esperado:
                errores.append({'ruta': nombre, 'error': 'HASH_DISTINTO', 'esperado': esperado, 'actual': actual})
            else:
                verificados.append(nombre)
    return dict(estado='VERIFICADO' if not errores else 'BLOQUEADO', errores=errores,
                archivos_verificados=len(verificados), rutas_verificadas=verificados,
                alcance='Identidad de archivos, no disponibilidad de GPU ni validación biológica.')


def indices_exactos(ids, solicitados):
    import numpy as np
    ids = np.asarray(ids)
    exigir(ids.ndim == 1 and np.all(ids[1:] > ids[:-1]), 'IDs sin orden estricto')
    ix = np.searchsorted(ids, solicitados)
    exigir(np.all(ix < len(ids)), 'ID solicitado fuera del grafo')
    np.testing.assert_array_equal(ids[ix], solicitados)
    return ix


def extraer_controlador(ruta: Path, cupy, numpy, mujoco):
    """Reutiliza solo la función original; NO ejecuta el script histórico.

    Sus imports y rutas absolutas no se evalúan. Se exige el SHA de todo el
    archivo antes de llegar aquí. El AST conserva literalmente el cuerpo.
    """
    arbol = ast.parse(ruta.read_text(encoding='utf-8'), filename=str(ruta))
    funciones = [n for n in arbol.body if isinstance(n, ast.FunctionDef) and n.name == 'bind_macro_advance']
    exigir(len(funciones) == 1, 'Controlador histórico ausente o ambiguo')
    modulo = ast.Module(body=funciones, type_ignores=[])
    espacio = {'np': numpy, 'cp': cupy, 'mj': mujoco}
    exec(compile(modulo, str(ruta), 'exec'), espacio)
    return espacio['bind_macro_advance']


def preparar_candidata(objeto, tabla, plan, cp, np, controlador, salida: Path) -> dict:
    """Mismas modificaciones de Gemini, con IDs y orden anatómico comprobados.

    No se presentan el escalamiento de W o la igualdad tau/theta como leyes
    biológicas. Ambas preparaciones reciben exactamente estas operaciones.
    """
    sesion, h = objeto.core, objeto.core.hybrid
    ids = h.brain.node_ids
    exigir(len(ids) == plan['neuronas'] and len(sesion.source_identity) == plan['fuentes_cns'], 'CNS distinto del esperado')
    # El script original usa las filas de la tabla como filas del grafo.
    columna = next((k for k in ('bodyId', 'node_id', 'nodeId', 'root_id', 'id') if k in tabla.columns), None)
    if columna is not None:
        np.testing.assert_array_equal(tabla[columna].to_numpy(), ids)
    else:
        np.testing.assert_array_equal(tabla.index.to_numpy(), ids)
    pn = indices_exactos(ids, plan['ids_pn']); dn = indices_exactos(ids, plan['ids_dn'])
    np.testing.assert_array_equal(pn, plan['indices_pn_legacy'])
    np.testing.assert_array_equal(dn, plan['indices_dn_legacy'])
    recibo = {'alcance': 'Intervenciones heredadas de Gemini, NO validación fisiológica', 'pares': []}
    for nombre, ix in (('PN', pn), ('DNb05', dn)):
        tau_antes = h.tau[ix].copy(); theta_antes = h.rate_theta[ix].copy()
        tau, theta = float(np.mean(tau_antes)), float(np.mean(theta_antes))
        h.tau[ix] = tau; h.cuda['tau'][cp.asarray(ix)] = tau
        h.rate_theta[ix] = theta; h.cuda['theta'][cp.asarray(ix)] = theta
        recibo['pares'].append(dict(nombre=nombre, ids=ids[ix].tolist(), indices=ix.tolist(),
            tau_antes=tau_antes.tolist(), theta_antes=theta_antes.tolist(), tau_despues=tau, theta_despues=theta))
    lado = tabla['somaSide'].fillna('unclear').to_numpy()
    punteros = cp.asnumpy(h.cuda['indptr'])
    pesos, columnas = h.cuda['weights'], h.cuda['indices']
    cambios = {}
    for grupo, ipsi, contra in (('ORN_DM1_L', 'L', 'R'), ('ORN_DM1_R', 'R', 'L')):
        entradas = cp.flatnonzero(cp.isin(columnas, cp.asarray(sesion.port_indices[grupo])))
        entradas_cpu = cp.asnumpy(entradas)
        filas = np.searchsorted(punteros, entradas_cpu, side='right') - 1
        antes = cp.asnumpy(pesos[entradas])
        pesos[entradas[cp.asarray(lado[filas] == ipsi)]] *= plan['escala_ipsilateral']
        pesos[entradas[cp.asarray(lado[filas] == contra)]] *= plan['escala_contralateral']
        cambios[grupo + '_indices_W'] = entradas_cpu
        cambios[grupo + '_antes'] = antes
        cambios[grupo + '_despues'] = cp.asnumpy(pesos[entradas])
    np.savez_compressed(salida / 'intervenciones_W.npz', **cambios)
    controlador(objeto, gain=plan['decoder_gain'])
    q = h.release()
    objeto.dn_ids[2:] = np.asarray(plan['ids_dn'])
    objeto.dn_ix[2:] = dn
    objeto.dn_baseline[2:] = q[dn].copy(); objeto.last_dn[2:] = q[dn].copy()
    objeto.set_command(mode='neural', speed=0., steering=0.)
    recibo['W_sha256'] = sha256(salida / 'intervenciones_W.npz')
    recibo['dn_ids_lector'] = objeto.dn_ids.tolist()
    recibo['baseline_inicial'] = objeto.dn_baseline.tolist()
    escribir_json(salida / 'INTERVENCIONES.json', recibo)
    return recibo


def instalar_campo(objeto, clase_campo, base, condicion: str, onset_ms: float):
    import numpy as np
    s = objeto.core; mundo = s.world
    # Nunca alterar el primer intervalo comprometido del adaptador histórico.
    exigir(mundo.time_ns > mundo.installed_ns, 'No sobrescribir el intervalo sensorial comprometido')
    geometria = mundo.sample_geometry()['antennae_mm'].copy()
    campo = clase_campo(base, mundo, condicion, geometria, onset_ms=onset_ms)
    mundo.boundary = campo
    s.pending_sensors = mundo.sense(objeto.body.observe())
    if condicion == 'sham':
        np.testing.assert_array_equal(s.pending_sensors, np.zeros(3))
    return campo


def yaw_grados(qpos):
    import numpy as np
    w, x, y, z = np.asarray(qpos)[3:7]
    return float(np.rad2deg(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))))


def diferencia_angular(actual, inicial):
    return float((actual - inicial + 180.) % 360. - 180.)


def captura(objeto, puertos, fase, paso, sensores_usados, yaw0, cp):
    import numpy as np
    s, h, b = objeto.core, objeto.core.hybrid, objeto.body
    fuente = h._online_source
    exigir(h.time_ns == fuente.time_ns == s.time_ns == s.world.time_ns == b.steps * round(b.dt*1e9), 'Relojes divergentes')
    q = h.release(); frame = puertos.observe(); geometria = s.world.sample_geometry()
    def arreglo(v):
        return cp.asnumpy(v) if isinstance(v, cp.ndarray) else np.asarray(v).copy()
    pn_ix = np.array([189, 159])  # Identidad ya comprobada una vez en preparar_candidata.
    c = objeto.controller
    exigir(c is not None, 'Falta el controlador protésico comprometido')
    dato = dict(fase=fase, paso=paso, CNS_time_ns=int(s.time_ns), PN_time_ns=int(fuente.time_ns),
        body_time_ns=int(b.steps*round(b.dt*1e9)), sensores_usados=arreglo(sensores_usados),
        sensores_pendientes=arreglo(s.pending_sensors), antenas_mm=arreglo(geometria['antennae_mm']),
        concentracion_campo=arreglo(geometria['concentration']),
        ORN_q_L=arreglo(q[s.port_indices['ORN_DM1_L']]), ORN_q_R=arreglo(q[s.port_indices['ORN_DM1_R']]),
        ORN_filters=arreglo(frame['ORN_filters']), PN_q_legacy=arreglo(q[pn_ix]),
        PN_general_transmission=arreglo(fuente.general_transmission()),
        PN_gamma_nS=arreglo(fuente.output_nS()), PN_additional_nS=arreglo(fuente.additional_output_nS()),
        DN_q_actual=arreglo(q[objeto.dn_ix]), DN_q_usada=arreglo(objeto.last_dn),
        DN_baseline=arreglo(objeto.dn_baseline),
        command_forward_mm_s=float(c.forward_mm_s), command_yaw_rate_rad_s=float(c.yaw_rate_rad_s),
        qpos=arreglo(b.data.qpos), qvel=arreglo(b.data.qvel),
        position_mm=arreglo(b.data.qpos[:3])*10.,
        yaw_delta_deg=diferencia_angular(yaw_grados(b.data.qpos), yaw0),
        upright=float(1 - 2*(b.data.qpos[4]**2 + b.data.qpos[5]**2)),
        contact_active=arreglo(c.last_contact_active), normal_force_N=arreglo(c.last_normal_N),
        contact_force_N=arreglo(c.last_force_N), generalized_force_native=arreglo(objeto.last_force),
        energy_motor_J=float(c.energy_motor_J))
    for k, v in dato.items():
        if k != 'fase':
            exigir(np.isfinite(v).all(), 'Observación no finita: ' + k)
    return dato


def resumen_rama(datos, preparacion, condicion, plan):
    import numpy as np
    fase = np.asarray(datos['fase']); prueba = fase == 'ensayo'; prep = fase == 'preparacion'
    for clave in ('sensores_usados', 'yaw_delta_deg', 'position_mm', 'upright'):
        exigir(np.isfinite(datos[clave]).all(), 'Traza no finita: ' + clave)
    exigir(np.count_nonzero(prueba) == plan['ensayo_ms'], 'Ensayo incompleto')
    exigir(np.count_nonzero(prep) == plan['preparacion_ms'], 'Preparación incompleta')
    sensores = np.asarray(datos['sensores_usados'])
    limpio = bool(np.all(sensores[prep] == 0))
    if preparacion == 'aire_limpio':
        exigir(limpio, 'Aire limpio no cumplido en las entradas utilizadas')
    giro = np.asarray(datos['yaw_delta_deg'])[prueba]
    pos = np.asarray(datos['position_mm'])[prueba]
    usado = sensores[prueba, :2]
    return dict(preparacion=preparacion, condicion=condicion, estado='COMPLETO',
        muestras_ensayo=int(prueba.sum()), final_yaw_deg=float(giro[-1]),
        preparacion_realmente_sin_olor=limpio,
        max_estimulo_preparacion=float(np.max(np.abs(sensores[prep]))),
        primer_paso_con_olor=(int(np.flatnonzero(np.any(usado != 0, axis=1))[0])+1 if np.any(usado != 0) else None),
        recorrido_entre_muestras_mm=float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum()),
        upright_min=float(np.min(np.asarray(datos['upright'])[prueba])),
        contact_samples_with_support=int(np.count_nonzero(np.any(np.asarray(datos['contact_active'])[prueba], axis=1))),
        navegacion_a_fuente_evaluada=False, ensayo='Campo lateral fijo, sin blanco localizado',
        alcance='Una rama determinista. No prueba aprendizaje, equilibrio ni fidelidad biológica.')


def ejecutar_rama(raiz: Path, salida: Path, preparacion: str, condicion: str, *, reescalado_orn=True, variante_03=None) -> int:
    import numpy as np
    inicio = time.monotonic(); objeto = None; filas = []
    salida.mkdir(parents=True, exist_ok=False)
    plan = (plan_efectivo_03(leer_plan(raiz), variante_03) if variante_03 is not None
            else plan_efectivo_02(leer_plan(raiz), reescalado_orn))
    if not reescalado_orn or variante_03 is not None:
        escribir_json(salida / 'PLAN_RAMA.json', plan)
    exigir(preparacion in plan['preparaciones'] and condicion in plan['condiciones'], 'Rama fuera del plan')
    # El coordinador ya verificó las identidades. El cargador original repite sus
    # comprobaciones de checkpoint, fuentes y entorno, sin bypass alguno.
    try:
        import cupy as cp
        import mujoco as mj
        import pandas as pd
        sys.path.insert(0, str(raiz / 'src'))
        for ruta in ('work/stage4_antennal_contact_adapter_20260915', 'work/stage3_static_lateral_field_20260916'):
            sys.path.insert(0, str(raiz / ruta))
        from antennal_runtime import AntennalContactRuntime
        from static_field import StaticLateralField
        from pn_cns_ports import PnCnsPorts
        exigir(not sys.flags.optimize, 'No utilizar Python -O')
        objeto = AntennalContactRuntime.load(raiz / plan['checkpoint'])
        h, s = objeto.core.hybrid, objeto.core
        exigir(s.CONTROL_NS == 1000000, 'El ensayo exige intervalos CNS de 1 ms')
        tabla = pd.read_parquet(raiz / 'data/male_v10/nodes.parquet')
        controlador = extraer_controlador(raiz / plan['parent_script'], cp, np, mj)
        preparar_candidata(objeto, tabla, plan, cp, np, controlador, salida)
        puertos = PnCnsPorts(h, 10208)
        fuente = h._online_source
        esquema = dict(
            ORN_q_L_ids=h.brain.node_ids[s.port_indices['ORN_DM1_L']].tolist(),
            ORN_q_R_ids=h.brain.node_ids[s.port_indices['ORN_DM1_R']].tolist(),
            ORN_filters_ids=puertos.orn_ids.tolist(), PN_q_legacy_ids=[10208, 10176],
            DN_ids=objeto.dn_ids.tolist(),
            PN_general_target_ids=fuente.general_output.targets.tolist(),
            PN_gamma_target_ids=fuente.route.target_ids.tolist(),
            PN_additional_target_ids=fuente.extra_output.route.target_ids.tolist(),
            contacto_geoms=objeto.controller.geom_names,
            unidades=dict(q='proxy normalizado del modelo', PN_general_transmission='adimensional',
                PN_gamma_nS='nS', PN_additional_nS='nS', ORN_filters='estados heredados; no tasa medida',
                qpos='coordenadas nativas MuJoCo, traslación en cm', qvel='velocidades nativas MuJoCo',
                position_mm='mm', antenas_mm='mm', yaw_delta_deg='grados',
                generalized_force_native='unidades nativas del modelo', normal_force_N='N',
                energy_motor_J='J', sensores='concentración adimensional'),
            muestreo='Estado posterior al paso; DN_q_usada y comando pertenecen al intervalo aplicado.',
            orden_contactos='T1_left,T1_right,T2_left,T2_right,T3_left,T3_right',
            PN_scope='Salidas especializadas únicamente para PN10208; PN10176 conserva la ruta legacy.')
        escribir_json(salida/'ESQUEMA_TRAZA.json', esquema)
        base = s.world.boundary
        # El id de condición no se suministra al CNS ni al controlador.
        # Se utiliza exclusivamente para construir el estímulo al final de prep.
        if preparacion == 'aire_limpio':
            instalar_campo(objeto, StaticLateralField, base, 'sham', 0.)
        referencia_yaw = yaw_grados(objeto.body.data.qpos)
        inicio_ns = s.time_ns
        estado_inicial = {'time_ns': int(inicio_ns), 'qpos': objeto.body.data.qpos.tolist(),
                          'sensores_pendientes': s.pending_sensors.tolist()}
        escribir_json(salida / 'INICIAL.json', estado_inicial)
        log = (salida / 'progreso.jsonl').open('x', encoding='utf-8', buffering=1)
        try:
            for fase, duracion in (('preparacion', plan['preparacion_ms']), ('ensayo', plan['ensayo_ms'])):
                if fase == 'ensayo':
                    # Igual que Gemini: baseline del último intervalo leído,
                    # no se lo denomina equilibrio fisiológico.
                    objeto.dn_baseline[2:] = objeto.last_dn[2:].copy()
                    campo = instalar_campo(objeto, StaticLateralField, base, condicion, plan['onset_ms'])
                    referencia_yaw = yaw_grados(objeto.body.data.qpos)
                    escribir_json(salida / 'CAMPO.json', campo.metadata())
                for ms in range(1, duracion+1):
                    utilizados = s.pending_sensors.copy()
                    objeto.step()
                    dato = captura(objeto, puertos, fase, ms, utilizados, referencia_yaw, cp)
                    filas.append(dato)
                    # Diario pequeño, persistente incluso si el proceso es cortado.
                    log.write(json.dumps(dict(fase=fase, ms=ms, time_ns=dato['CNS_time_ns'],
                        sensores_usados=utilizados.tolist(), yaw_delta_deg=dato['yaw_delta_deg'],
                        wall_s=time.monotonic()-inicio), allow_nan=False) + '\n')
                    if ms % 10 == 0:
                        print(json.dumps(dict(preparacion=preparacion, condicion=condicion, fase=fase, ms=ms)), flush=True)
            esperado = inicio_ns + (plan['preparacion_ms'] + plan['ensayo_ms'])*1000000
            exigir(s.time_ns == esperado, 'Duración efectiva distinta de la acordada')
            datos = {k: np.asarray([f[k] for f in filas]) for k in filas[0]}
            np.savez_compressed(salida / 'traza.npz', **datos)
            resultado = resumen_rama(datos, preparacion, condicion, plan)
            resultado.update(wall_s=time.monotonic()-inicio, trace_sha256=sha256(salida / 'traza.npz'),
                             plan_sha256=sha256(raiz / PLAN_REL))
            if not reescalado_orn:
                resultado['reescalado_ORN_aplicado'] = False
                resultado['plan_efectivo_sha256'] = sha256(salida / 'PLAN_RAMA.json')
            if variante_03 is not None:
                resultado['variante_03'] = variante_03
                resultado['plan_efectivo_sha256'] = sha256(salida / 'PLAN_RAMA.json')
                resultado['factores_ORN'] = [plan['escala_ipsilateral'], plan['escala_contralateral']]
            escribir_json(salida / 'RESULTADO.json', resultado)
            return 0
        finally:
            log.close()
    except BaseException as error:
        if filas:
            np.savez_compressed(salida / 'traza_parcial.npz', **{k: np.asarray([f[k] for f in filas]) for k in filas[0]})
        escribir_json(salida / 'ERROR.json', dict(tipo=type(error).__name__, mensaje=str(error),
            traceback=traceback.format_exc(), wall_s=time.monotonic()-inicio))
        print(traceback.format_exc(), file=sys.stderr)
        return 2
    finally:
        if objeto is not None:
            objeto.close()


def criterio_angular(valores: dict, plan: dict) -> dict:
    necesarios = {'odor_left', 'odor_right', 'sham', 'uniform'}
    exigir(set(valores) == necesarios, 'No mezclar ni omitir condiciones')
    izquierda, derecha, sham = (float(valores[k]) for k in ('odor_left', 'odor_right', 'sham'))
    return dict(izquierda=izquierda >= plan['umbral_giro_grados'],
                derecha=derecha <= -plan['umbral_giro_grados'],
                sham=abs(sham) < plan['umbral_sham_abs_grados'])


def comprobar_prefijos(salida: Path, plan: dict) -> dict:
    import numpy as np
    comparaciones = []
    for preparacion in plan['preparaciones']:
        referencia = None
        for condicion in plan['condiciones']:
            ruta = salida / preparacion / condicion / 'traza.npz'
            with np.load(ruta, allow_pickle=False) as z:
                mask = z['fase'] == 'preparacion'
                actual = {k: z[k][mask].copy() for k in z.files}
            if referencia is None:
                referencia = actual
            else:
                for k, valor in actual.items():
                    np.testing.assert_array_equal(valor, referencia[k], err_msg=f'Prefijo distinto {preparacion}/{condicion}/{k}')
                comparaciones.append(f'{preparacion}/{condicion}')
    return dict(estado='PREFIJOS_REGISTRADOS_IGUALES', comparaciones=comparaciones,
                alcance='Paridad de las variables registradas, no inventario completo del estado interno.')


def verificar_resumenes(salida: Path, plan: dict) -> dict:
    import numpy as np
    recibos = []
    for prep in plan['preparaciones']:
        for condicion in plan['condiciones']:
            carpeta = salida/prep/condicion
            ruta = carpeta/'traza.npz'
            guardado = json.loads((carpeta/'RESULTADO.json').read_text(encoding='utf-8'))
            exigir(guardado['trace_sha256'] == sha256(ruta), 'Hash de traza incompatible')
            with np.load(ruta, allow_pickle=False) as z:
                recalculado = resumen_rama(z, prep, condicion, plan)
            for clave, valor in recalculado.items():
                exigir(guardado[clave] == valor, 'Resumen incompatible con traza: '+clave)
            recibos.append(f'{prep}/{condicion}')
    return dict(estado='RESUMENES_RECALCULADOS', ramas=recibos)


def generar_informe(salida: Path, plan: dict, estado: str, detalle: str = '') -> dict:
    resultados = []
    for preparacion in plan['preparaciones']:
        for condicion in plan['condiciones']:
            ruta = salida / preparacion / condicion / 'RESULTADO.json'
            if ruta.is_file():
                resultados.append(json.loads(ruta.read_text(encoding='utf-8')))
    evaluacion = dict(estado=estado, detalle=detalle, ramas_completas=len(resultados),
        ramas_planificadas=8, navegacion_demostrada=False, aprendizaje_demostrado=False,
        biologia_validada=False, promocion_checkpoint=False, resultados=resultados)
    lineas = ['# MATRIX — diagnóstico automático de preparación', '', f'**Estado de ejecución: {estado}.**', detalle, '',
        'Se compara una única modificación: eliminar olor externo durante los 40 ms de preparación.',
        'Las modificaciones históricas de Gemini se conservan en ambos grupos; no se validan como leyes biológicas.',
        'Cada condición dura 100 ms tras 40 ms de preparación. Una rama por condición, sin réplicas independientes.', '',
        '| Preparación | Condición | Giro final (°) | Preparación sin olor |', '|---|---|---:|---|']
    for r in resultados:
        lineas.append(f"| {r['preparacion']} | {r['condicion']} | {r['final_yaw_deg']:+.8f} | {r['preparacion_realmente_sin_olor']} |")
    if estado == 'COMPLETO_DIAGNOSTICO' and len(resultados) == 8:
        grupos = {}
        for preparacion in plan['preparaciones']:
            valores = {r['condicion']:r['final_yaw_deg'] for r in resultados if r['preparacion'] == preparacion}
            pruebas = criterio_angular(valores, plan)
            grupos[preparacion] = dict(pruebas=pruebas, criterio_angular_conjunto=all(pruebas.values()), valores=valores,
                efecto_izquierdo_vs_sham=valores['odor_left']-valores['sham'],
                efecto_derecho_vs_sham=valores['odor_right']-valores['sham'])
        evaluacion['comparacion'] = grupos
        antes = abs(grupos['heredada']['valores']['sham']); despues = abs(grupos['aire_limpio']['valores']['sham'])
        evaluacion['reduccion_abs_sham_porcentaje'] = 100*(antes-despues)/antes if antes else None
        lineas += ['', '## Comparación a igual duración', '',
            'Criterio angular histórico: izquierda ≥ +0,020°, derecha ≤ −0,020°, |sham| < 0,020°. Se exigen conjuntamente.',
            'Uniforme es un control descriptivo; no se le inventa un umbral después de observarlo.']
        for nombre, valor in grupos.items():
            lineas.append(f"\n**{nombre}:** criterio angular conjunto = {valor['criterio_angular_conjunto']}; "
                          f"L−sham = {valor['efecto_izquierdo_vs_sham']:+.8f}°; R−sham = {valor['efecto_derecho_vs_sham']:+.8f}°.")
        lineas += ['', 'Reducción porcentual de |sham|, a 100 ms en ambos grupos: ' + str(evaluacion['reduccion_abs_sham_porcentaje']) + '.']
    lineas += ['', '## Límites que no cambian con una puntuación favorable', '',
        'El estímulo es un semiplano estacionario ON/OFF anclado al mundo, no una fuente localizada. No hay métrica de llegada.',
        'No demuestra aprendizaje, seis patas caminando, equilibrio neuronal, reconstrucción biológica ni superioridad del conectoma.',
        'La preparación es activa con una prótesis de contactos/rodillos; aire limpio significa entradas olfativas externas nulas, no ORN silenciadas.',
        'PN_q_legacy no sustituye la liberación local especializada. La traza conserva por separado las salidas efectivas de PN10208.',
        'Las muestras temporales no son réplicas independientes. No se calculan p-valores ni se promociona un checkpoint.',
        'No se guarda un supuesto checkpoint experimental recargable: el campo lateral no está serializado por el cargador original.',
        '', '## Archivos de evidencia', '',
        '`PREFLIGHT.json`, `PLAN_EJECUTADO.json`, `FUENTES_EJECUCION.json`, `PARIDAD.json` cuando corresponda, logs y trazas por rama.',
        'Un fallo detiene la campaña. No se reajusta ni se reintenta automáticamente. Los archivos parciales se conservan.']
    escribir_json(salida / 'RESULTADOS.json', evaluacion)
    (salida / 'INFORME_AUTOMATICO.md').write_text('\n'.join(lineas)+'\n', encoding='utf-8')
    return evaluacion


def empaquetar_resultados(salida: Path, limite=400_000_000) -> list[Path]:
    """ZIPs independientes bajo 450 MB, sin añadir datos originales pesados."""
    archivos = sorted(p for p in salida.rglob('*') if p.is_file() and not p.name.startswith('RESULTADOS_PARA_CHATGPT_'))
    grupos, grupo, tamano = [], [], 0
    for p in archivos:
        exigir(p.stat().st_size < limite, 'Archivo individual demasiado grande para la entrega: ' + str(p))
        if grupo and tamano + p.stat().st_size > limite:
            grupos.append(grupo); grupo=[]; tamano=0
        grupo.append(p); tamano += p.stat().st_size
    if grupo:
        grupos.append(grupo)
    paquetes = []
    for i, grupo in enumerate(grupos, 1):
        ruta = salida / f'RESULTADOS_PARA_CHATGPT_{i:02d}.zip'
        with zipfile.ZipFile(ruta, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for p in grupo:
                z.write(p, str(p.relative_to(salida)))
        exigir(ruta.stat().st_size < 450_000_000, 'Paquete excede el límite')
        paquetes.append(ruta)
    return paquetes


def campana(raiz: Path, salida: Path, solo_verificar=False) -> int:
    exigir(not salida.exists(), 'La salida ya existe: no sobrescribir ni repetir una campaña cerrada')
    salida.mkdir(parents=True)
    plan = leer_plan(raiz); codigo = 2; estado = 'BLOQUEADO'; detalle = ''
    escribir_json(salida/'PLAN_EJECUTADO.json', plan)
    fuentes = {p:sha256(raiz/p) for p in (PLAN_REL, MODULO_REL, 'laboratorio.py')}
    escribir_json(salida/'FUENTES_EJECUCION.json', dict(python=sys.version, executable=sys.executable,
                    fuentes_sha256=fuentes, entorno={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','PYTHONUTF8','PYTHONHASHSEED')}))
    try:
        chequeo = verificar_archivos(raiz, plan); escribir_json(salida/'PREFLIGHT.json', chequeo)
        exigir(not chequeo['errores'], 'Hay archivos ausentes o con hash distinto. Consultar PREFLIGHT.json.')
        if solo_verificar:
            estado='ARCHIVOS_VERIFICADOS_SIN_SIMULAR'; codigo=0
        else:
            for prep in plan['preparaciones']:
                for condicion in plan['condiciones']:
                    rama = salida / prep / condicion
                    log = salida / f'{prep}__{condicion}.log'
                    comando = [sys.executable, '-X', 'utf8', str(raiz/'laboratorio.py'), 'diagnostico-olfativo',
                               '_rama', '--salida', str(rama), '--preparacion', prep, '--condicion', condicion]
                    print(f'Ejecutando {prep}/{condicion}; trazas en {rama}', flush=True)
                    with log.open('x', encoding='utf-8') as archivo:
                        proceso = subprocess.run(comando, cwd=raiz, stdout=archivo, stderr=subprocess.STDOUT,
                                                 timeout=plan['timeout_rama_s'], check=False)
                    exigir(proceso.returncode == 0, f'Rama bloqueada {prep}/{condicion}, código {proceso.returncode}. No se reintenta.')
            escribir_json(salida/'RECALCULO.json', verificar_resumenes(salida, plan))
            escribir_json(salida/'PARIDAD.json', comprobar_prefijos(salida, plan))
            # Detecta cambios en disco durante la ejecución; no borra el recibo inicial.
            final = verificar_archivos(raiz, plan); escribir_json(salida/'PREFLIGHT_FINAL.json', final)
            exigir(not final['errores'], 'Las fuentes/checkpoint cambiaron durante la campaña')
            exigir(all(sha256(raiz/p)==h for p,h in fuentes.items()), 'Plan o ejecutor modificado durante la campaña')
            estado='COMPLETO_DIAGNOSTICO'; codigo=0
    except BaseException as error:
        detalle=f'{type(error).__name__}: {error}'
        escribir_json(salida/'ERROR_CAMPANA.json', dict(mensaje=detalle, traceback=traceback.format_exc()))
    generar_informe(salida, plan, estado, detalle)
    manifiesto={str(p.relative_to(salida)):sha256(p) for p in salida.rglob('*') if p.is_file()}
    escribir_json(salida/'MANIFEST_RESULTADOS.json', manifiesto)
    for paquete in empaquetar_resultados(salida):
        print(f'ADJUNTAR: {paquete}', flush=True)
    return codigo


def main(argumentos=None, *, raiz=None) -> int:
    raiz = Path(raiz or Path(__file__).resolve().parents[1]).resolve()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('accion', choices=('ejecutar','verificar','_rama','_rama_sin_reescalado','ablar-reescalado','verificar-ablacion','factorial-ORN','verificar-factorial','_rama_factorial_03'))
    parser.add_argument('--salida', required=True, type=Path)
    parser.add_argument('--preparacion', choices=('heredada','aire_limpio'))
    parser.add_argument('--condicion', choices=('sham','uniform','odor_left','odor_right'))
    parser.add_argument('--variante', choices=tuple(VARIANTES_03))
    args=parser.parse_args(argumentos)
    if args.accion in ('factorial-ORN','verificar-factorial'):
        return campana_factorial_03(raiz,args.salida.resolve(),args.accion=='verificar-factorial')
    if args.accion=='_rama_factorial_03':
        exigir(args.variante is not None and args.preparacion=='aire_limpio' and args.condicion, 'Rama factorial incompleta')
        return ejecutar_rama(raiz,args.salida.resolve(),args.preparacion,args.condicion,variante_03=args.variante)
    if args.accion in ('ablar-reescalado','verificar-ablacion'):
        return campana_ablacion_02(raiz,args.salida.resolve(),args.accion=='verificar-ablacion')
    if args.accion in ('_rama','_rama_sin_reescalado'):
        exigir(args.preparacion and args.condicion, 'Rama incompleta')
        return ejecutar_rama(raiz,args.salida.resolve(),args.preparacion,args.condicion,
                             reescalado_orn=args.accion=='_rama')
    return campana(raiz,args.salida.resolve(),args.accion=='verificar')

# Segunda campaña: una operación se retira; las ecuaciones heredadas no cambian.
# No se enseña al CNS el ID del experimento, de la condición ni la fuente.
PROTOCOLO_02 = {
    'id': 'ablacion_reescalado_ORN_02',
    'referencia': 'runs/chatgpt_preparacion_olfativa_01',
    'manifest_referencia_sha256': '6ae0682dcc414d149af82cf9e3741132dc9436d31cf89652b59d519e6f8fce51',
    'plan_referencia_sha256': '108ce7233a3498e3885f3f205677a6308789b6327dc5587cd26d3d38ed02f00a',
    'preparacion': 'aire_limpio',
    'replica_tecnica': 'sham con 1.4/0.1; igualdad exacta de todos los arrays registrados',
    'candidata': 'sin reescalado ipsi/contra ORN, factores 1.0/1.0',
    'condiciones': ['sham','uniform','odor_left','odor_right'],
    'preparacion_ms': 40, 'ensayo_ms': 100,
    'invariantes': ['checkpoint','tau/theta homologadas','decoder_gain=250','baseline al final de preparación',
                   'prótesis','avance tónico','onset_ms=5','fuentes CNS'],
    'criterios': 'Se conservan los tres umbrales históricos; uniforme es descriptivo.',
    'replicas_independientes': 0,
    'no_inferir': ['navegación','aprendizaje','fidelidad biológica','ausencia de todos los supuestos añadidos'],
    'no_promover_checkpoint': True,
    'limite': 'Retirar esta operación en W no elimina las rutas especializadas fuera de W; no es una lesión completa ORN→PN.'
}


def plan_efectivo_02(plan, reescalado_orn=True):
    exigir(type(reescalado_orn) is bool, 'Selector de reescalado debe ser booleano')
    if reescalado_orn:
        return plan
    nuevo = dict(plan)
    nuevo['escala_ipsilateral'] = 1.0
    nuevo['escala_contralateral'] = 1.0
    return nuevo


def verificar_referencia_02(raiz):
    referencia = raiz / PROTOCOLO_02['referencia']
    exigir(sha256(referencia/'MANIFEST_RESULTADOS.json') == PROTOCOLO_02['manifest_referencia_sha256'],
           'Referencia distinta del ZIP auditado; no comparar automáticamente')
    manifest = json.loads((referencia/'MANIFEST_RESULTADOS.json').read_text())
    for nombre, esperado in manifest.items():
        p=(referencia/nombre).resolve()
        exigir(p.is_relative_to(referencia.resolve()), 'Ruta fuera de referencia')
        exigir(p.is_file() and sha256(p)==esperado, 'Referencia ausente o modificada: '+nombre)
    exigir(sha256(referencia/'PLAN_EJECUTADO.json')==PROTOCOLO_02['plan_referencia_sha256'], 'Plan de referencia distinto')
    exigir(sha256(raiz/PLAN_REL)==PROTOCOLO_02['plan_referencia_sha256'], 'Plan instalado diferente')
    return referencia, dict(archivos_verificados=len(manifest), directorio=str(referencia),
                           manifest_sha256=PROTOCOLO_02['manifest_referencia_sha256'])


def comparar_trazas_02(actual, esperada):
    import numpy as np
    with np.load(actual,allow_pickle=False) as a, np.load(esperada,allow_pickle=False) as b:
        exigir(set(a.files)==set(b.files), 'Campos de réplica técnica diferentes')
        diferencias=[k for k in a.files if a[k].dtype!=b[k].dtype or not np.array_equal(a[k],b[k])]
        exigir(not diferencias, 'Réplica técnica no idéntica: '+', '.join(diferencias))
        return dict(estado='ARRAYS_REGISTRADOS_IDENTICOS', arrays=len(a.files),
                    alcance='No se presupone igualdad de estados internos no registrados; no es una semilla independiente.')


def verificar_rama_02(carpeta, plan, sin_reescalado):
    import numpy as np
    recibo=json.loads((carpeta/'RESULTADO.json').read_text())
    with np.load(carpeta/'traza.npz',allow_pickle=False) as a:
        datos={k:a[k] for k in a.files}
    esperado=resumen_rama(datos,'aire_limpio',recibo['condicion'],plan)
    for k,v in esperado.items():
        exigir(recibo[k]==v,'Resumen de rama distinto: '+k)
    exigir(sha256(carpeta/'traza.npz')==recibo['trace_sha256'],'Hash de traza distinto')
    exigir(recibo['plan_sha256']==PROTOCOLO_02['plan_referencia_sha256'],'Plan base diferente')
    if sin_reescalado:
        exigir(recibo.get('reescalado_ORN_aplicado') is False, 'Falta recibo de ablación')
        exigir(sha256(carpeta/'PLAN_RAMA.json')==recibo['plan_efectivo_sha256'],'Plan efectivo alterado')
        efectivo=json.loads((carpeta/'PLAN_RAMA.json').read_text())
        exigir(efectivo==plan_efectivo_02(plan,False),'Cambios extra fuera de la ablación')
        with np.load(carpeta/'intervenciones_W.npz',allow_pickle=False) as z:
            for lado in ('L','R'):
                exigir(np.array_equal(z[f'ORN_DM1_{lado}_antes'],z[f'ORN_DM1_{lado}_despues']),
                       'La candidata todavía reescala W')
    return recibo


def resumen_ablacion_02(referencia, salida, plan, estado, detalle=''):
    resultado=dict(estado=estado,detalle=detalle,replicas_independientes=0,
        promocion_checkpoint=False,navegacion_demostrada=False,biologia_validada=False,
        comparacion={}, resultados_candidata=[])
    for nombre, base in [('con_reescalado_referencia',referencia/'aire_limpio'),
                         ('sin_reescalado_ORN',salida/'sin_reescalado_ORN')]:
        filas=[]
        for c in plan['condiciones']:
            p=base/c/'RESULTADO.json'
            if p.is_file():
                r=json.loads(p.read_text());filas.append(r)
        if nombre=='sin_reescalado_ORN':resultado['resultados_candidata']=filas
        if len(filas)==4:
            y={f['condicion']:f['final_yaw_deg'] for f in filas}
            pruebas=criterio_angular(y,plan)
            resultado['comparacion'][nombre]=dict(valores=y,criterios=pruebas,
                criterio_angular_conjunto=all(pruebas.values()),
                izquierdo_menos_sham=y['odor_left']-y['sham'],
                derecho_menos_sham=y['odor_right']-y['sham'],
                izquierda_menos_derecha=y['odor_left']-y['odor_right'],
                uniforme_menos_sham=y['uniform']-y['sham'])
    linea=['# MATRIX — ablación del reescalado ORN', '',f'**Estado de ejecución: {estado}.**','',detalle,'',
        'Una réplica técnica sham y cuatro condiciones sin reescalado. Se reutilizan las cuatro referencias auditadas.',
        'Solo se retira la multiplicación ORN ipsilateral 1,4 / contralateral 0,1. Se conservan los otros ajustes.', '',
        '| Variante | Condición | Giro final (°) |','|---|---|---:|']
    for nombre, datos in resultado['comparacion'].items():
        for c,v in datos['valores'].items():linea.append(f'| {nombre} | {c} | {v:+.8f} |')
        linea.append(f"\n{nombre}: criterio histórico conjunto = {datos['criterio_angular_conjunto']}; L−R = {datos['izquierda_menos_derecha']:+.8f}°.")
    linea += ['', '## Interpretación permitida', '',
        'COMPLETO_ABLACION significa ejecución íntegra, no hipótesis confirmada. Un resultado favorable sin reescalado permitiría estudiar retirar esa operación; no valida todo el modelo. Un negativo mide dependencia de ese ajuste en este protocolo, no una ley biológica.',
        'No se reajusta la ganancia, no se cambian umbrales y no se guarda ni promueve un checkpoint nuevo.',
        'Semiplano estacionario, no fuente localizada. Sin aprendizaje ni réplicas independientes.',
        'Esta prueba no elimina la igualdad tau/theta ni el lector motor, y no desconecta la PN especializada.',
        'Las referencias permanecen intactas. Si falla la réplica técnica se bloquea la comparación y se conserva el error.', '']
    escribir_json(salida/'RESULTADOS.json',resultado)
    (salida/'INFORME_AUTOMATICO.md').write_text('\n'.join(linea),encoding='utf-8')
    return resultado


def campana_ablacion_02(raiz, salida, solo_verificar=False):
    exigir(not salida.exists(),'La campaña 02 ya existe. No sobrescribir ni repetir automáticamente.')
    salida.mkdir(parents=True)
    plan=leer_plan(raiz);referencia=raiz/PROTOCOLO_02['referencia']
    codigo=2;estado='BLOQUEADO';detalle=''
    escribir_json(salida/'PROTOCOLO_02.json',PROTOCOLO_02)
    escribir_json(salida/'PLAN_BASE.json',plan)
    escribir_json(salida/'PLAN_SIN_REESCALADO.json',plan_efectivo_02(plan,False))
    fuentes={p:sha256(raiz/p) for p in (PLAN_REL,MODULO_REL,'laboratorio.py')}
    escribir_json(salida/'FUENTES_EJECUCION.json',dict(python=sys.version,executable=sys.executable,
        fuentes_sha256=fuentes,entorno={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','PYTHONUTF8','PYTHONHASHSEED')}))
    try:
        check=verificar_archivos(raiz,plan);escribir_json(salida/'PREFLIGHT.json',check)
        exigir(not check['errores'],'Fuentes o checkpoint distintos; consultar PREFLIGHT.json')
        referencia,checkref=verificar_referencia_02(raiz)
        escribir_json(salida/'REFERENCIA_VERIFICADA.json',checkref)
        if solo_verificar:
            estado='ARCHIVOS_VERIFICADOS_SIN_SIMULAR';codigo=0
        else:
            ramas=[('replica_tecnica','sham','_rama')]+[
                ('sin_reescalado_ORN',c,'_rama_sin_reescalado') for c in plan['condiciones']]
            for grupo,condicion,accion in ramas:
                destino=salida/grupo/condicion
                comando=[sys.executable,'-X','utf8',str(raiz/'laboratorio.py'),'diagnostico-olfativo',accion,
                         '--salida',str(destino),'--preparacion','aire_limpio','--condicion',condicion]
                print(f'Ejecutando {grupo}/{condicion}; sin ajuste ni reintento automático',flush=True)
                with (salida/f'{grupo}__{condicion}.log').open('x',encoding='utf-8') as log:
                    r=subprocess.run(comando,cwd=raiz,stdout=log,stderr=subprocess.STDOUT,
                                     timeout=plan['timeout_rama_s'],check=False)
                exigir(r.returncode==0,f'Rama bloqueada {grupo}/{condicion}; código {r.returncode}')
                verificar_rama_02(destino,plan,grupo=='sin_reescalado_ORN')
                if grupo=='replica_tecnica':
                    paridad=comparar_trazas_02(destino/'traza.npz',referencia/'aire_limpio/sham/traza.npz')
                    escribir_json(salida/'PARIDAD_REFERENCIA.json',paridad)
            # El mismo prefijo de preparación debe reproducirse en las cuatro nuevas condiciones.
            import numpy as np
            with np.load(salida/'sin_reescalado_ORN/sham/traza.npz',allow_pickle=False) as a:
                for cond in plan['condiciones'][1:]:
                    with np.load(salida/f'sin_reescalado_ORN/{cond}/traza.npz',allow_pickle=False) as b:
                        exigir(set(a.files)==set(b.files),'Campos diferentes')
                        for k in a.files:
                            exigir(np.array_equal(a[k][:40],b[k][:40]),f'Preparación divergente: {cond}/{k}')
            escribir_json(salida/'PARIDAD_PREPARACIONES.json',dict(estado='PREFIJOS_REGISTRADOS_IDENTICOS',comparaciones=3))
            final=verificar_archivos(raiz,plan);escribir_json(salida/'PREFLIGHT_FINAL.json',final)
            exigir(not final['errores'],'El proyecto cambió durante la campaña')
            verificar_referencia_02(raiz)
            exigir(all(sha256(raiz/p)==h for p,h in fuentes.items()),'Ejecutor cambiado durante la campaña')
            estado='COMPLETO_ABLACION';codigo=0
    except BaseException as e:
        detalle=f'{type(e).__name__}: {e}'
        escribir_json(salida/'ERROR_CAMPANA.json',dict(mensaje=detalle,traceback=traceback.format_exc()))
    # No depender de que un error deje intacta la referencia al redactar el diagnóstico.
    try:
        resumen_ablacion_02(referencia,salida,plan,estado,detalle)
    except Exception as e:
        escribir_json(salida/'ERROR_INFORME.json',dict(tipo=type(e).__name__,mensaje=str(e)))
        (salida/'INFORME_AUTOMATICO.md').write_text('# MATRIX — ejecución bloqueada\n\n'+detalle+'\n'+str(e)+'\n')
        codigo=2
    manifest={str(p.relative_to(salida)):sha256(p) for p in salida.rglob('*') if p.is_file()}
    escribir_json(salida/'MANIFEST_RESULTADOS.json',manifest)
    for p in empaquetar_resultados(salida):
        print('ADJUNTAR: '+str(p),flush=True)
    return codigo

# Campaña 03: factorial de dos operaciones YA existentes. No es una búsqueda de gains.
VARIANTES_03 = {
    'sin_reescalado_referencia': (1.0, 1.0),
    'solo_amplificacion_ipsi': (1.4, 1.0),
    'solo_atenuacion_contra': (1.0, 0.1),
}
PROTOCOLO_03 = {
    'id': 'factorial_operaciones_ORN_03',
    'referencia_01': PROTOCOLO_02['referencia'],
    'referencia_02': 'runs/chatgpt_ablacion_reescalado_02',
    'manifest_02_sha256': 'a733c168b32a9d6f6351831ed676847ea12bc911b995857b4b43f1e7d2d64499',
    'replica_tecnica': {'variante': 'sin_reescalado_referencia', 'condicion': 'odor_right',
                         'comparacion': '31 arrays idénticos a sin_reescalado_ORN/odor_right de 02'},
    'variantes_nuevas': ['solo_amplificacion_ipsi', 'solo_atenuacion_contra'],
    'factores': VARIANTES_03,
    'condiciones': ['sham', 'uniform', 'odor_left', 'odor_right'],
    'preparacion_ms': 40, 'ensayo_ms': 100, 'onset_ms': 5,
    'metrica_primaria': 'giro_L - giro_R a 100 ms',
    'metricas_adicionales': ['giro menos sham', 'contraste DN antes de baseline/gain',
                             'separación ORN por lado', 'uniforme', 'criterio angular histórico conjunto'],
    'invariantes': PROTOCOLO_02['invariantes'],
    'hipotesis': 'La dependencia observada procede de amplificar, atenuar o de su interacción; se separan los dos factores.',
    'presupuesto': {'replicas_tecnicas': 1, 'ramas_nuevas': 8, 'referencias_reutilizadas': 8},
    'replicas_independientes': 0, 'promover_checkpoint': False,
    'limites': ['Factorial de intervenciones de ingeniería, no inferencia de fisiología.',
               'Cada factor actúa durante preparación y ensayo; baseline y realimentación pueden cambiar.',
               'No se cortan las rutas especializadas de PN fuera de W.',
               'No se exploran parámetros nuevos ni se elige una combinación ganadora.',
               'El semiplano no es una fuente localizada; no demuestra navegación.'],
}


def plan_efectivo_03(plan, variante):
    exigir(isinstance(variante, str) and variante in VARIANTES_03, 'Variante factorial desconocida')
    nuevo = dict(plan)
    nuevo['escala_ipsilateral'], nuevo['escala_contralateral'] = VARIANTES_03[variante]
    return nuevo


def verificar_referencias_03(raiz):
    primera, check1 = verificar_referencia_02(raiz)
    segunda = raiz / PROTOCOLO_03['referencia_02']
    exigir(sha256(segunda/'MANIFEST_RESULTADOS.json') == PROTOCOLO_03['manifest_02_sha256'],
           'Paquete 02 distinto del auditado')
    manifest = json.loads((segunda/'MANIFEST_RESULTADOS.json').read_text(encoding='utf-8'))
    for nombre, esperado in manifest.items():
        p = (segunda/nombre).resolve()
        exigir(p.is_relative_to(segunda.resolve()), 'Ruta de referencia 02 insegura')
        exigir(p.is_file() and sha256(p) == esperado, 'Referencia 02 alterada: '+nombre)
    exigir(sha256(segunda/'PLAN_BASE.json') == PROTOCOLO_02['plan_referencia_sha256'], 'Plan base 02 distinto')
    estado = json.loads((segunda/'RESULTADOS.json').read_text(encoding='utf-8'))
    exigir(estado['estado'] == 'COMPLETO_ABLACION', 'Referencia 02 incompleta')
    plan = leer_plan(raiz)
    for condicion in plan['condiciones']:
        verificar_rama_02(primera/'aire_limpio'/condicion, plan, False)
        verificar_rama_02(segunda/'sin_reescalado_ORN'/condicion, plan, True)
    comparar_trazas_02(segunda/'replica_tecnica/sham/traza.npz', primera/'aire_limpio/sham/traza.npz')
    return {'ambas_operaciones': primera/'aire_limpio', 'ninguna_operacion': segunda/'sin_reescalado_ORN'}, {
        'referencia_01': check1, 'referencia_02': {'archivos_verificados': len(manifest),
        'manifest_sha256': PROTOCOLO_03['manifest_02_sha256']}}


def verificar_rama_03(carpeta, plan, variante, referencia_W):
    import numpy as np
    recibo = verificar_rama_02(carpeta, plan, False)
    exigir(recibo.get('variante_03') == variante, 'Variante del recibo incorrecta')
    efectivo = plan_efectivo_03(plan, variante)
    exigir(json.loads((carpeta/'PLAN_RAMA.json').read_text()) == efectivo, 'Plan efectivo fuera del factorial')
    exigir(recibo['plan_efectivo_sha256'] == sha256(carpeta/'PLAN_RAMA.json'), 'Hash de plan efectivo distinto')
    exigir(recibo['factores_ORN'] == list(VARIANTES_03[variante]), 'Factores del recibo distintos')
    intervencion = json.loads((carpeta/'INTERVENCIONES.json').read_text())
    exigir(intervencion['W_sha256'] == sha256(carpeta/'intervenciones_W.npz'), 'Hash del recibo W distinto')
    # Las categorías se reconstruyen a partir de la intervención archivada 1,4/0,1.
    # No se crean nuevas conexiones ni se reasignan lateralidades en esta campaña.
    with np.load(referencia_W, allow_pickle=False) as ref, np.load(carpeta/'intervenciones_W.npz', allow_pickle=False) as actual:
        exigir(set(ref.files) == set(actual.files), 'Esquema W diferente')
        for lado in ('L','R'):
            clave = 'ORN_DM1_'+lado
            antes, despues = ref[clave+'_antes'], ref[clave+'_despues']
            exigir(np.all(antes != 0), 'Clasificación de factores ambigua por pesos cero')
            iguales = despues == antes
            ipsi = despues == antes*1.4
            contra = despues == antes*.1
            exigir(np.all(iguales.astype(int)+ipsi.astype(int)+contra.astype(int) == 1), 'Clasificación de factores ambigua')
            for sufijo in ('indices_W','antes'):
                exigir(np.array_equal(actual[clave+'_'+sufijo], ref[clave+'_'+sufijo]), 'Identidades o valores W de origen cambiados')
            esperado = antes.copy()
            esperado[ipsi] *= efectivo['escala_ipsilateral']
            esperado[contra] *= efectivo['escala_contralateral']
            exigir(np.array_equal(actual[clave+'_despues'], esperado), 'No se aplicaron exactamente los factores declarados')
    with np.load(carpeta/'traza.npz',allow_pickle=False) as z:
        exigir(len(z.files) == 31, 'Cambió el número de arrays')
        for k in z.files:
            exigir(z[k].shape[0] == 140, 'Traza incompleta: '+k)
            if k != 'fase': exigir(np.isfinite(z[k]).all(), 'Valor no finito: '+k)
        exigir(z['fase'].tolist() == ['preparacion']*40+['ensayo']*100, 'Fases incorrectas')
        exigir(z['paso'].tolist() == list(range(1,41))+list(range(1,101)), 'Índices de paso incorrectos')
        for k in ('PN_time_ns','body_time_ns'):
            exigir(np.array_equal(z[k], z['CNS_time_ns']), 'Relojes diferentes')
        inicial = json.loads((carpeta/'INICIAL.json').read_text())
        esperado = int(inicial['time_ns']) + np.arange(1,141,dtype=np.int64)*1_000_000
        exigir(np.array_equal(z['CNS_time_ns'],esperado), 'Reloj/discretización diferente')
        exigir(np.array_equal(z['DN_q_usada'][1:],z['DN_q_actual'][:-1]), 'Desfase DN incorrecto')
        dq = z['DN_q_usada'] - z['DN_baseline']
        mando = np.tanh(250*(dq[:,2]-dq[:,3]))*np.deg2rad(5)
        exigir(np.array_equal(mando,z['command_yaw_rate_rad_s']), 'Lector diferente')
        exigir(np.all(z['command_forward_mm_s']==.2), 'Avance diferente')
        esperado = np.zeros((100,3));c = recibo['condicion']
        if c!='sham': esperado[5:,:2] = {'uniform':[1,1], 'odor_left':[1,0], 'odor_right':[0,1]}[c]
        exigir(np.array_equal(z['sensores_usados'][40:],esperado), 'Estímulo recibido diferente')
        # La referencia ya muestra que no se cruza el semiplano en 100 ms.
        # Si cambia el patrón, se bloquea esta comparación en vez de ocultarlo.
        q = z['qpos'];w,x,y,zz=(q[:,i] for i in (3,4,5,6))
        yaw=np.rad2deg(np.arctan2(2*(w*zz+x*y),1-2*(y*y+zz*zz)))
        delta=(yaw[40:]-yaw[39]+180)%360-180
        exigir(np.max(np.abs(delta-z['yaw_delta_deg'][40:]))<=1e-11,'Yaw inconsistente con el cuerpo')
    return recibo


def metricas_grupo_03(carpeta, plan):
    import numpy as np
    trazas, y, mandos = {}, {}, {}
    for c in plan['condiciones']:
        with np.load(carpeta/c/'traza.npz',allow_pickle=False) as z:
            a={k:z[k] for k in ('DN_q_usada','DN_baseline','ORN_q_L','ORN_q_R','command_yaw_rate_rad_s','yaw_delta_deg')}
        trazas[c]=a; y[c]=float(a['yaw_delta_deg'][-1])
        mandos[c]=float(np.rad2deg(a['command_yaw_rate_rad_s'][40:].sum()*.001))
    L,R = trazas['odor_left'],trazas['odor_right']
    dnL=L['DN_q_usada'][40:,2]-L['DN_q_usada'][40:,3]
    dnR=R['DN_q_usada'][40:,2]-R['DN_q_usada'][40:,3]
    pruebas=criterio_angular(y,plan)
    return dict(yaw_deg=y,L_menos_R=y['odor_left']-y['odor_right'],
        L_menos_sham=y['odor_left']-y['sham'],R_menos_sham=y['odor_right']-y['sham'],
        uniforme_menos_sham=y['uniform']-y['sham'],criterios_historicos=pruebas,
        criterio_angular_conjunto=all(pruebas.values()),
        DN_LR_media=float((dnL-dnR).mean()),DN_LR_final=float((dnL-dnR)[-1]),
        ORN_L_RMS_LR=float(np.sqrt(np.mean((L['ORN_q_L'][40:]-R['ORN_q_L'][40:])**2))),
        ORN_R_RMS_LR=float(np.sqrt(np.mean((L['ORN_q_R'][40:]-R['ORN_q_R'][40:])**2))),
        baseline_DNb05=trazas['sham']['DN_baseline'][40,2:].tolist(),
        integrales_mando_deg=mandos)


def contrastes_factoriales_03(y00,y10,y01,y11):
    return dict(amplificar_sin_atenuar=y10-y00,amplificar_con_atenuacion=y11-y01,
                atenuar_sin_amplificar=y01-y00,atenuar_con_amplificacion=y11-y10,
                interaccion=y11-y10-y01+y00)


def informe_factorial_03(salida, plan, referencias, estado, detalle=''):
    r=dict(estado=estado,detalle=detalle,promocion_checkpoint=False,navegacion_demostrada=False,
           biologia_validada=False,replicas_independientes=0,grupos={},contrastes={})
    bases=dict(referencias or {})
    for grupo in PROTOCOLO_03['variantes_nuevas']: bases[grupo]=salida/grupo
    for grupo, carpeta in bases.items():
        if all((carpeta/c/'RESULTADO.json').is_file() and (carpeta/c/'traza.npz').is_file() for c in plan['condiciones']):
            r['grupos'][grupo]=metricas_grupo_03(carpeta,plan)
    if estado=='COMPLETO_FACTORIAL' and len(r['grupos'])==4:
        g=r['grupos']
        for k in ('L_menos_R','L_menos_sham','R_menos_sham','uniforme_menos_sham','DN_LR_media'):
            r['contrastes'][k]=contrastes_factoriales_03(g['ninguna_operacion'][k],g['solo_amplificacion_ipsi'][k],g['solo_atenuacion_contra'][k],g['ambas_operaciones'][k])
    lineas=['# MATRIX — separación factorial de operaciones ORN','',f'**Estado de ejecución: {estado}.**','',detalle,'',
        'Se separan dos operaciones ya existentes: amplificación ipsilateral 1,4 y atenuación contralateral 0,1. No se buscan parámetros nuevos. Los factores se clasifican según la regla histórica somaSide; esto no valida su mecanismo biológico.',
        'Una réplica técnica de olor derecho sin reescalado y ocho ramas nuevas. Referencias 01/02 autenticadas, no réplicas independientes. Preparación 40 ms, ensayo 100 ms; mismo lector y cuerpo.',
        '', '| Grupo | Sin olor (°) | Uniforme (°) | Izquierda (°) | Derecha (°) | L−R (°) | Criterio histórico conjunto |',
        '|---|---:|---:|---:|---:|---:|---|']
    for grupo,m in r['grupos'].items():
        y=m['yaw_deg'];lineas.append(f"| {grupo} | {y['sham']:+.8f} | {y['uniform']:+.8f} | {y['odor_left']:+.8f} | {y['odor_right']:+.8f} | {m['L_menos_R']:+.8f} | {m['criterio_angular_conjunto']} |")
    if r['contrastes']:
        lineas+=['','## Contrastes del factorial','',
            'A = amplificación ipsilateral. B = atenuación contralateral. 00 sin ambas; 11 con ambas. La interacción es Y11−Y10−Y01+Y00. Son contrastes de estas historias deterministas, no estimaciones poblacionales.',
            '', '| Métrica | A sin B | A con B | B sin A | B con A | Interacción |','|---|---:|---:|---:|---:|---:|']
        for k,v in r['contrastes'].items():lineas.append('| '+k+' | '+' | '.join(f'{x:+.10g}' for x in v.values())+' |')
    lineas+=['','## Límites y decisión','',
        'No se selecciona una variante ganadora ni se promueve un checkpoint. Un efecto de atenuar/amplificar no valida un mecanismo fisiológico. Se conserva aire limpio, pero el resto de las simplificaciones sigue presente.',
        'El resultado mide la intervención durante preparación y ensayo, incluidos cambios de baseline y realimentación. DN_LR_media se calcula antes del lector, sin baseline ni gain. No es una prueba de mediación por una única ruta.',
        'Los umbrales angulares son los históricos, sin reajuste. Semiplano estacionario, avance tónico, sin aprendizaje ni fuente localizada. La PN especializada no queda eliminada por intervenir W.',
        'Si la campaña queda bloqueada, las filas completas son diagnósticas y no certifican el factorial. Los parciales no se sustituyen por ceros ni se reintentan.', '']
    escribir_json(salida/'RESULTADOS.json',r)
    (salida/'INFORME_AUTOMATICO.md').write_text('\n'.join(lineas),encoding='utf-8')
    return r


def campana_factorial_03(raiz,salida,solo_verificar=False):
    exigir(not salida.exists(),'La campaña 03 ya existe. No borrar, repetir ni sobrescribir.')
    salida.mkdir(parents=True)
    codigo,estado,detalle=2,'BLOQUEADO','';referencias={};plan=None;ejecuciones=[]
    try:
        plan=leer_plan(raiz)
        escribir_json(salida/'PROTOCOLO_03.json',PROTOCOLO_03)
        escribir_json(salida/'PLAN_BASE.json',plan)
        fuentes={p:sha256(raiz/p) for p in (PLAN_REL,MODULO_REL,'laboratorio.py')}
        escribir_json(salida/'FUENTES_EJECUCION.json',dict(python=sys.version,executable=sys.executable,
            fuentes_sha256=fuentes,entorno={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','PYTHONUTF8','PYTHONHASHSEED')}))
        check=verificar_archivos(raiz,plan);escribir_json(salida/'PREFLIGHT.json',check)
        exigir(not check['errores'],'Fuentes/checkpoint distintos; consultar PREFLIGHT.json')
        referencias,checkref=verificar_referencias_03(raiz)
        escribir_json(salida/'REFERENCIAS_VERIFICADAS.json',checkref)
        if solo_verificar:
            estado,codigo='ARCHIVOS_VERIFICADOS_SIN_SIMULAR',0
        else:
            ramas=[('replica_tecnica','odor_right','sin_reescalado_referencia')]
            ramas += [(g,c,g) for g in PROTOCOLO_03['variantes_nuevas'] for c in plan['condiciones']]
            ref_W=referencias['ambas_operaciones']/'sham/intervenciones_W.npz'
            for grupo,c,variante in ramas:
                destino=salida/grupo/c
                comando=[sys.executable,'-X','utf8',str(raiz/'laboratorio.py'),'diagnostico-olfativo','_rama_factorial_03',
                    '--salida',str(destino),'--preparacion','aire_limpio','--condicion',c,'--variante',variante]
                print(f'Ejecutando {grupo}/{c}; factores fijos, sin ajuste ni reintento',flush=True)
                ejecuciones.append(dict(grupo=grupo,condicion=c,variante=variante,comando=comando,estado='INICIADA'))
                escribir_json(salida/'EJECUCIONES.json',ejecuciones)
                with (salida/f'{grupo}__{c}.log').open('x',encoding='utf-8') as log:
                    sub=subprocess.run(comando,cwd=raiz,stdout=log,stderr=subprocess.STDOUT,timeout=plan['timeout_rama_s'],check=False)
                ejecuciones[-1].update(returncode=sub.returncode,estado='PROCESO_TERMINADO')
                escribir_json(salida/'EJECUCIONES.json',ejecuciones)
                exigir(sub.returncode==0,f'Rama bloqueada {grupo}/{c}; código {sub.returncode}')
                verificar_rama_03(destino,plan,variante,ref_W)
                if grupo=='replica_tecnica':
                    paridad=comparar_trazas_02(destino/'traza.npz',referencias['ninguna_operacion']/'odor_right/traza.npz')
                    escribir_json(salida/'PARIDAD_REFERENCIA.json',paridad)
                ejecuciones[-1]['estado']='VERIFICADA';escribir_json(salida/'EJECUCIONES.json',ejecuciones)
            import numpy as np
            for g in PROTOCOLO_03['variantes_nuevas']:
                with np.load(salida/g/'sham/traza.npz',allow_pickle=False) as a:
                    for c in plan['condiciones'][1:]:
                        with np.load(salida/g/c/'traza.npz',allow_pickle=False) as b:
                            exigir(set(a.files)==set(b.files),'Campos de preparación distintos')
                            for k in a.files:exigir(np.array_equal(a[k][:40],b[k][:40]),f'Preparación divergente: {g}/{c}/{k}')
            escribir_json(salida/'PARIDAD_PREPARACIONES.json',dict(comparaciones=6,estado='PREFIJOS_REGISTRADOS_IDENTICOS'))
            check=verificar_archivos(raiz,plan);escribir_json(salida/'PREFLIGHT_FINAL.json',check)
            exigir(not check['errores'],'Proyecto alterado durante ejecución')
            verificar_referencias_03(raiz)
            exigir(all(sha256(raiz/p)==h for p,h in fuentes.items()),'Ejecutor alterado durante campaña')
            estado,codigo='COMPLETO_FACTORIAL',0
    except BaseException as e:
        detalle=f'{type(e).__name__}: {e}'
        escribir_json(salida/'ERROR_CAMPANA.json',dict(mensaje=detalle,traceback=traceback.format_exc()))
    try:
        if plan is None:raise ValueError('Plan no cargado; consultar ERROR_CAMPANA.json')
        informe_factorial_03(salida,plan,referencias,estado,detalle)
    except Exception as e:
        escribir_json(salida/'ERROR_INFORME.json',dict(tipo=type(e).__name__,mensaje=str(e)))
        (salida/'INFORME_AUTOMATICO.md').write_text('# MATRIX — ejecución bloqueada\n\n'+detalle+'\n'+str(e)+'\n')
        codigo=2
    manifest={str(p.relative_to(salida)):sha256(p) for p in salida.rglob('*') if p.is_file()}
    escribir_json(salida/'MANIFEST_RESULTADOS.json',manifest)
    for p in empaquetar_resultados(salida):print('ADJUNTAR: '+str(p),flush=True)
    return codigo


# Cuarta entrega: medición de coste; NO variante conductual ni optimización.
# Las funciones y protocolos 01/02/03 de arriba se conservan byte a byte.
_MAIN_HISTORICO_03 = main
PROTOCOLO_04 = {
    'id': 'perfil_prefijo_04', 'referencia': 'runs/chatgpt_factorial_orn_03',
    'manifest_referencia_sha256': '97db7403647af9b597988532502a64f7233e08bf3ab84dfca86bcf92e9dc7ba1',
    'rama_referencia': 'replica_tecnica/odor_right',
    'preparacion_ms': 40, 'ensayo_observado_ms': 10, 'muestras': 50,
    'condicion': 'odor_right', 'factores_ORN': [1.0, 1.0],
    'max_wall_subproceso_s': 1800, 'telemetria_intervalo_s': 5,
    'nuevas_hipotesis_biologicas': 0, 'procesos_simulacion': 1,
    'comparacion': 'Igualdad exacta de los 31 arrays, primer prefijo de 50 muestras.',
    'no_hacer': ['cambiar dt o tolerancias', 'desactivar validaciones', 'paralelizar organismos',
                'ajustar pesos', 'promover checkpoint', 'llamar mejora al ensayo acortado'],
    'alcance': 'Perfil de Python y tiempos de pared por bloque; no tiempo de kernels CUDA.',
}

def comando_lectura_04(comando):
    """No usa shell, no instala nada y no modifica la GPU."""
    try:
        r = subprocess.run(comando, capture_output=True, text=True, timeout=4)
        return {'comando': comando, 'codigo': r.returncode, 'stdout': r.stdout[-20000:],
                'stderr': r.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'comando': comando, 'no_disponible': type(e).__name__}

def memoria_04():
    p = Path('/proc/meminfo')
    if not p.is_file():
        return {'disponible': False}
    datos = {}
    for linea in p.read_text().splitlines():
        campos = linea.split()
        if campos and campos[0].rstrip(':') in ('MemTotal','MemAvailable','SwapTotal','SwapFree'):
            datos[campos[0].rstrip(':')+'_kB'] = int(campos[1])
    return datos

def inventario_04():
    import platform, importlib.metadata
    versiones = {}
    for nombre in ('numpy','scipy','pandas','cupy','cupy-cuda12x','cupy-cuda11x',
                   'mujoco','dm-control','pyarrow'):
        try: versiones[nombre] = importlib.metadata.version(nombre)
        except importlib.metadata.PackageNotFoundError: versiones[nombre] = None
    return {'python': sys.version, 'ejecutable': sys.executable, 'plataforma': platform.platform(),
            'cpu_logicas': os.cpu_count(),
            'afinidad_cpu': sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'memoria': memoria_04(), 'versiones': versiones,
            'hilos_entorno': {k: os.environ.get(k) for k in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')},
            'gpu': comando_lectura_04(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits']),
            'limite': 'Inventario, no benchmark ni garantía de recursos libres.'}

class Telemetria04:
    """Una lectura externa cada 5 s; ningún proceso adicional calcula una mosca."""
    def __init__(self, carpeta):
        import threading
        self.carpeta = Path(carpeta); self.fin = threading.Event(); self.hilo = None
    def iniciar(self):
        import threading
        self.hilo = threading.Thread(target=self._bucle, name='telemetria_matrix04', daemon=True)
        self.hilo.start()
    def _bucle(self):
        import resource
        inicio = time.monotonic()
        with (self.carpeta/'RECURSOS.jsonl').open('x',encoding='utf-8',buffering=1) as f:
            while not self.fin.is_set():
                uso = resource.getrusage(resource.RUSAGE_SELF)
                fila = {'wall_s':time.monotonic()-inicio, 'cpu_user_s':uso.ru_utime,
                    'cpu_system_s':uso.ru_stime, 'rss_max_kB_linux':uso.ru_maxrss,
                    'memoria':memoria_04(),
                    'gpu':comando_lectura_04(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw',
                                            '--format=csv,noheader,nounits'])}
                f.write(json.dumps(fila,ensure_ascii=False,allow_nan=False)+'\n')
                self.fin.wait(PROTOCOLO_04['telemetria_intervalo_s'])
    def cerrar(self):
        self.fin.set()
        if self.hilo: self.hilo.join(timeout=6)

def estadisticas_perfil_04(perfil, raiz):
    import pstats
    filas=[]
    for (archivo,linea,funcion),(primitivas,total,propio,acumulado,llamadores) in pstats.Stats(perfil).stats.items():
        try: nombre=str(Path(archivo).resolve().relative_to(raiz))
        except (ValueError,OSError): nombre=archivo
        filas.append({'archivo':nombre,'linea':linea,'funcion':funcion,
                      'llamadas_primitivas':primitivas,'llamadas_totales':total,
                      'self_s':propio,'cumulative_s':acumulado})
    return sorted(filas,key=lambda x:x['self_s'],reverse=True)

def comparar_prefijo_04(actual, referencia, muestras=50):
    import numpy as np
    exigir(set(actual)==set(referencia) and len(actual)==31, 'Esquema de arrays no idéntico')
    for k in sorted(actual):
        exigir(len(actual[k])==muestras and len(referencia[k])>=muestras, 'Prefijo incompleto: '+k)
        exigir(actual[k].dtype==referencia[k].dtype, 'dtype distinto: '+k)
        np.testing.assert_array_equal(actual[k],referencia[k][:muestras],err_msg='Prefijo distinto: '+k)
    return {'arrays_identicos':len(actual),'muestras':muestras,
            'alcance':'Solo el prefijo registrado, no todo el estado oculto ni el ensayo de 100 ms.'}

def referencia_04(raiz):
    carpeta=(raiz/PROTOCOLO_04['referencia']).resolve()
    exigir(carpeta.is_relative_to(raiz), 'Referencia fuera del laboratorio')
    exigir(sha256(carpeta/'MANIFEST_RESULTADOS.json')==PROTOCOLO_04['manifest_referencia_sha256'],
           'Referencia 03 distinta de la recibida')
    manifest=json.loads((carpeta/'MANIFEST_RESULTADOS.json').read_text())
    for nombre, esperado in manifest.items():
        p=(carpeta/nombre).resolve()
        exigir(p.is_relative_to(carpeta) and p.is_file() and sha256(p)==esperado, 'Referencia alterada: '+nombre)
    estado=json.loads((carpeta/'RESULTADOS.json').read_text())
    exigir(estado['estado']=='COMPLETO_FACTORIAL','La campaña 03 no está completa')
    return carpeta/PROTOCOLO_04['rama_referencia'], {'archivos':len(manifest),
        'manifest_sha256':PROTOCOLO_04['manifest_referencia_sha256']}

def ejecutar_perfil_04(raiz, salida, *, al_cargar=None):
    """Repite un prefijo exacto con cProfile. No reduce el paso del integrador."""
    import cProfile, numpy as np
    salida.mkdir(parents=True,exist_ok=False)
    inicio=time.monotonic();objeto=None;politica_local=None;filas=[];bloques=[];perfiles={}
    monitor=Telemetria04(salida);monitor.iniciar()
    try:
        escribir_json(salida/'INVENTARIO.json',inventario_04())
        import cupy as cp, mujoco as mj, pandas as pd
        plan=plan_efectivo_02(leer_plan(raiz),False)
        exigir(plan['preparacion_ms']==40 and plan['ensayo_ms']==100,'Plan histórico incompatible')
        escribir_json(salida/'PLAN_HEREDADO.json',plan)
        escribir_json(salida/'PROTOCOLO_PERFIL.json',PROTOCOLO_04)
        sys.path.insert(0,str(raiz/'src'))
        for ruta in ('work/stage4_antennal_contact_adapter_20260915','work/stage3_static_lateral_field_20260916'):
            sys.path.insert(0,str(raiz/ruta))
        from antennal_runtime import AntennalContactRuntime
        from static_field import StaticLateralField
        from pn_cns_ports import PnCnsPorts
        exigir(not sys.flags.optimize,'No usar -O ni omitir validaciones')
        t=time.monotonic()
        objeto=AntennalContactRuntime.load(raiz/plan['checkpoint'])
        bloques.append({'bloque':'cargar_checkpoint','wall_s':time.monotonic()-t})
        # 05B: el cargador valida primero la política original. La adopción es posterior.
        if al_cargar is not None:
            t=time.monotonic()
            politica_local=al_cargar(objeto)
            bloques.append({'bloque':'adopcion_politica_local','wall_s':time.monotonic()-t})
        h,s=objeto.core.hybrid,objeto.core
        exigir(s.CONTROL_NS==1000000,'Intervalo de control distinto de 1 ms')
        t=time.monotonic()
        tabla=pd.read_parquet(raiz/'data/male_v10/nodes.parquet')
        controlador=extraer_controlador(raiz/plan['parent_script'],cp,np,mj)
        preparar_candidata(objeto,tabla,plan,cp,np,controlador,salida)
        puertos=PnCnsPorts(h,10208);base=s.world.boundary
        instalar_campo(objeto,StaticLateralField,base,'sham',0.)
        bloques.append({'bloque':'preparacion_instrumental','wall_s':time.monotonic()-t})
        escribir_json(salida/'CLASES_Y_LECTOR.json',{
          'CNS_MRO':[c.__module__+'.'+c.__name__ for c in type(h).__mro__],
          'session_MRO':[c.__module__+'.'+c.__name__ for c in type(s).__mro__],
          'runtime_MRO':[c.__module__+'.'+c.__name__ for c in type(objeto).__mro__],
          'DN_ids_lector':objeto.dn_ids.tolist(),
          'PN_ids_legacy':[10208,10176], 'PN_especializada':10208,
          'lectura_fuentes':'No se reasignan salidas a vuelo, cabeza o patas.'})
        # Inventario de las filas del lector, manteniendo el ID exacto usado.
        columna=next((k for k in ('bodyId','node_id','nodeId','root_id','id') if k in tabla.columns),None)
        indices=indices_exactos(h.brain.node_ids,objeto.dn_ids)
        vista=tabla.iloc[indices].copy()
        vista.insert(0,'_matrix_node_id',h.brain.node_ids[indices])
        vista.to_csv(salida/'NEURONAS_DEL_LECTOR.csv',index=False)
        referencia_yaw=yaw_grados(objeto.body.data.qpos);inicio_ns=s.time_ns
        with (salida/'progreso.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for fase,duracion in (('preparacion',40),('ensayo',10)):
                if fase=='ensayo':
                    objeto.dn_baseline[2:]=objeto.last_dn[2:].copy()
                    campo=instalar_campo(objeto,StaticLateralField,base,'odor_right',plan['onset_ms'])
                    referencia_yaw=yaw_grados(objeto.body.data.qpos)
                    escribir_json(salida/'CAMPO.json',campo.metadata())
                perfiles[fase+'_step']=cProfile.Profile()
                perfiles[fase+'_capture']=cProfile.Profile()
                for ms in range(1,duracion+1):
                    utilizados=s.pending_sensors.copy()
                    reloj=time.monotonic()
                    perfiles[fase+'_step'].runcall(objeto.step)
                    dt_step=time.monotonic()-reloj
                    reloj=time.monotonic()
                    dato=perfiles[fase+'_capture'].runcall(
                        captura,objeto,puertos,fase,ms,utilizados,referencia_yaw,cp)
                    dt_capture=time.monotonic()-reloj
                    filas.append(dato)
                    fila={'fase':fase,'ms':ms,'CNS_time_ns':int(s.time_ns),
                          'step_host_wall_s':dt_step,'capture_host_wall_s':dt_capture,
                          'wall_s':time.monotonic()-inicio}
                    bloques.append(fila);log.write(json.dumps(fila,allow_nan=False)+'\n')
                    if ms%5==0: print(json.dumps(fila),flush=True)
        exigir(s.time_ns==inicio_ns+50*1000000,'Duración no igual a 50 ms')
        datos={k:np.asarray([f[k] for f in filas]) for k in filas[0]}
        np.savez_compressed(salida/'traza_prefijo.npz',**datos)
        escribir_json(salida/'RESULTADO_PERFIL.json',{'estado':'PREFIJO_EJECUTADO',
             'muestras':len(filas),'wall_s':time.monotonic()-inicio,
             'trace_sha256':sha256(salida/'traza_prefijo.npz'),
             'navegacion':False,'mejora_velocidad_demostrada':False})
        return 0
    except BaseException as e:
        if filas:
            np.savez_compressed(salida/'traza_parcial.npz',**{k:np.asarray([f[k] for f in filas]) for k in filas[0]})
        escribir_json(salida/'ERROR.json',{'tipo':type(e).__name__,'mensaje':str(e),
             'traceback':traceback.format_exc(),'wall_s':time.monotonic()-inicio})
        print(traceback.format_exc(),file=sys.stderr)
        return 2
    finally:
        monitor.cerrar()
        escribir_json(salida/'BLOQUES.json',bloques)
        for nombre,perfil in perfiles.items():
            escribir_json(salida/('PERFIL_'+nombre+'.json'),estadisticas_perfil_04(perfil,raiz))
        try:
            if politica_local is not None: politica_local.restaurar()
        finally:
            if objeto is not None: objeto.close()

def informe_perfil_04(salida, estado, paridad=None):
    lineas=['# MATRIX — perfil instrumentado 04','',f'**Estado: {estado}.**','',
      'Una repetición del prefijo: 40 ms de preparación limpia + 10 ms de olor derecho; factores ORN 1/1.',
      'No cambia dt, tolerancias, lector ni cuerpo. No se evalúa navegación ni se promueve un checkpoint.','']
    if paridad: lineas += [f"Paridad exacta: {paridad['arrays_identicos']} arrays, {paridad['muestras']} muestras.",'']
    p=salida/'perfil/BLOQUES.json'
    if p.is_file():
        bloques=json.loads(p.read_text())
        for fase in ('preparacion','ensayo'):
            filas=[r for r in bloques if r.get('fase')==fase]
            if filas:
                lineas += [f"## {fase}",
                    f"Avance (host, incluye esperas): {sum(r['step_host_wall_s'] for r in filas):.6f} s.",
                    f"Observación (host, incluye transferencias/esperas): {sum(r['capture_host_wall_s'] for r in filas):.6f} s.",'']
        for perfil in sorted((salida/'perfil').glob('PERFIL_*.json')):
            filas=json.loads(perfil.read_text())
            lineas += ['## '+perfil.stem,'',
                '| Función y ubicación | Llamadas | Tiempo propio (s) | Acumulado (s) |',
                '|---|---:|---:|---:|']
            for r in filas[:15]:
                nombre=(r['funcion']+' '+r['archivo']+':'+str(r['linea'])).replace('|','/')
                lineas.append(f"| {nombre} | {r['llamadas_totales']} | {r['self_s']:.6f} | {r['cumulative_s']:.6f} |")
            lineas.append('')
    lineas += ['## Límites',
      'cProfile añade sobrecoste. Las mediciones son de Python y espera, no duración exclusiva de kernels CUDA.',
      'No sumar tiempos acumulados anidados. Ni CPU ocupada ni GPU ocupada por sí solas demuestran el cuello de botella.',
      'La telemetría GPU incluye otros procesos del equipo; CPU/RSS corresponde al proceso de perfil.',
      'La paridad cubre las observaciones del prefijo, no todos los estados ocultos ni 100 ms completos.',
      'La versión histórica ya contiene snapshots y buffers reutilizables: no se proclama descubrir esa optimización.',
      'No cambiar ni omitir hashes/validaciones hasta medir el coste y diseñar una preservación equivalente.',
      'No se lanza una segunda mosca en paralelo. No se cambia la configuración de hilos del entorno.',
      'Un perfil fallido conserva los parciales y no autoriza comparaciones de rendimiento.','']
    (salida/'INFORME_AUTOMATICO.md').write_text('\n'.join(lineas),encoding='utf-8')

def campana_perfil_04(raiz,salida):
    import numpy as np
    salida.mkdir(parents=True,exist_ok=False)
    estado='BLOQUEADO';codigo=2;paridad=None
    try:
        identidad_inicial={n:sha256(raiz/n) for n in (PLAN_REL,MODULO_REL,'laboratorio.py')}
        plan=leer_plan(raiz)
        preflight=verificar_archivos(raiz,plan);escribir_json(salida/'PREFLIGHT.json',preflight)
        exigir(preflight['estado']=='VERIFICADO','Archivos base incompatibles')
        referencia,recibo=referencia_04(raiz)
        escribir_json(salida/'REFERENCIA.json',recibo)
        escribir_json(salida/'PROTOCOLO_04.json',PROTOCOLO_04)
        escribir_json(salida/'FUENTES_EJECUCION.json',identidad_inicial)
        with (salida/'perfil.log').open('x',encoding='utf-8') as log:
            print('Perfil 04: una sola historia de 50 ms, sin cambiar integradores ni parámetros.',flush=True)
            r=subprocess.run([sys.executable,'-X','utf8',str(raiz/'laboratorio.py'),
               'diagnostico-olfativo','_perfil_04','--salida',str(salida/'perfil')],
               cwd=raiz,stdout=log,stderr=subprocess.STDOUT,
               timeout=PROTOCOLO_04['max_wall_subproceso_s'])
        exigir(r.returncode==0,'El perfil falló. Consultar perfil.log y perfil/ERROR.json; no reintentar.')
        with np.load(salida/'perfil/traza_prefijo.npz',allow_pickle=False) as a:
            actual={k:a[k] for k in a.files}
        with np.load(referencia/'traza.npz',allow_pickle=False) as a:
            ref={k:a[k] for k in a.files}
        paridad=comparar_prefijo_04(actual,ref)
        escribir_json(salida/'PARIDAD_PREFIJO.json',paridad)
        final=verificar_archivos(raiz,plan);escribir_json(salida/'PREFLIGHT_FINAL.json',final)
        exigir(final['estado']=='VERIFICADO','Las fuentes base cambiaron durante el perfil')
        for n,h in identidad_inicial.items(): exigir(sha256(raiz/n)==h,'Fuente modificada durante ejecución: '+n)
        # La referencia se comprueba de nuevo; no se modifica su contenido.
        referencia_04(raiz)
        estado='COMPLETO_PERFIL_PREFIJO';codigo=0
    except BaseException as e:
        escribir_json(salida/'ERROR_CAMPANA.json',{'tipo':type(e).__name__,'mensaje':str(e),
            'traceback':traceback.format_exc()})
    finally:
        informe_perfil_04(salida,estado,paridad)
        escribir_json(salida/'RESULTADOS.json',{'estado':estado,'codigo':codigo,'paridad':paridad,
            'optimizado':False,'promocion_checkpoint':False,'aprendizaje':False})
        manifest={str(p.relative_to(salida)):sha256(p) for p in salida.rglob('*') if p.is_file()
                  and p.name!='MANIFEST_RESULTADOS.json' and not p.name.startswith('RESULTADOS_PARA_CHATGPT_')}
        escribir_json(salida/'MANIFEST_RESULTADOS.json',manifest)
        empaquetar_resultados(salida)
    return codigo

def main(argumentos=None, *, raiz=None):
    args=list(sys.argv[1:] if argumentos is None else argumentos)
    if not args or args[0] not in ('perfil-04','_perfil_04'):
        return _MAIN_HISTORICO_03(argumentos,raiz=raiz)
    parser=argparse.ArgumentParser(description='Perfil sin optimización ni cambio de modelo')
    parser.add_argument('accion',choices=('perfil-04','_perfil_04'))
    parser.add_argument('--salida',required=True,type=Path)
    a=parser.parse_args(args)
    raiz=Path(raiz or Path(__file__).resolve().parents[1]).resolve()
    salida=a.salida.resolve()
    exigir(salida.is_relative_to(raiz/'runs') and salida!=raiz/'runs','Salida fuera de runs/')
    return (campana_perfil_04(raiz,salida) if a.accion=='perfil-04'
            else ejecutar_perfil_04(raiz,salida))


# Entrega 05: reutiliza el perfil 04 y recopila evidencia de predictores EXISTENTES.
# No crea otro generador; no cambia las ecuaciones ni la integración.
_MAIN_HISTORICO_04 = main
PROTOCOLO_05 = {
    'id': 'hilos_y_predictores_05',
    'referencia': 'runs/chatgpt_perfil_04',
    'manifest_referencia_sha256': 'aa4f8f29c7b458727ae59538a2218784c660d7862d9be01169bff656a5f941dc',
    'orden': [14, 1],
    'preparacion_ms': 40, 'ensayo_ms': 10, 'arrays': 31, 'muestras': 50,
    'modificacion': 'GpuProjectionParallelBrain.THREADS, solo dentro de cada proceso desechable',
    'sin_cambios': ['pesos','dt','tolerancias','lector','cuerpo','precision','validadores','familias'],
    'max_wall_subproceso_s': 1800,
    'min_reduccion_desarrollo': 0.20,
    'nota': 'Comparacion secuencial instrumentada por cProfile; no benchmark libre de instrumentacion, ni replicas biologicas.',
    'limite_copia_evidencia_bytes': 400000000,
    'limite_archivo_evidencia_bytes': 30000000,
    'sources': {
        'src/projection_parallel_brain.py': '796eab6a25c705608f03b28d77fb25804caa8f9a7dd41232599150445fbbb51a',
        'src/kc_projection_parallel.py': 'c87f87b52d5d160dc99fa5ae78460f427a05ae5f6deb772aef204bc6988dad3c'
    }
}
EVIDENCIA_FAMILIAS_05 = (
    'src/physiological_generator',
    'src/matrix_workbench/README.md',
    'config/predictor_program_v2.json',
    'config/conditional_family_reference_v1.json',
    'config/workbench_components.json',
    'reports/CICLO_MEJORA_PROGRESIVA.md',
    'work/physiological_generator_v1_20260915',
    'work/pulse_context_transfer_20260915',
    'work/family_input_domain_20260915',
    'work/feco_motor_predictor_20260915',
    'work/predictor_observer_refactor_20260916',
    'work/adaptive_error_radar_20260916',
    'work/live_forecast_transfer_20260916',
    'work/stage3_live_field_pair_20260916',
    'work/stage06_release_uncertainty_20260917',
    'work/aotu_current_reference_20260916',
    'work/stage06_tool_registry_followup_20260917',
    'work/predictor_utility_review_20260916',
    'work/stage3_native_reader_balance_20260918',
    'work/stage3_native_population_audit_20260918',
)
FECO_ARCHIVOS_05 = (
    'hook_flexion_01_magnet.parquet',
    'hook_flexion_01_treadmill_platform.parquet',
    'claw_magnet_Mamiya2018.parquet',
    'claw_treadmill.parquet',
    'hook_flexion_03_bdn2.parquet',
)

def recoger_predictores_05(raiz, salida):
    """Solo lee fuentes/resultados existentes. Ausente NO equivale a inexistente."""
    import shutil
    raiz=Path(raiz).resolve(); salida=Path(salida)
    salida.mkdir(parents=True,exist_ok=False)
    registros=[]; raices=[]; vistos=set(); total=0
    extensiones={'.md','.json','.py','.csv','.npz','.txt','.yaml','.yml'}
    for relativo in EVIDENCIA_FAMILIAS_05:
        origen=raiz/relativo
        if not origen.exists():
            raices.append({'ruta':relativo,'estado':'NO_ENCONTRADO_LOCALMENTE'})
            continue
        if origen.is_symlink() or not origen.resolve().is_relative_to(raiz):
            raices.append({'ruta':relativo,'estado':'OMITIDO_ENLACE'})
            continue
        raices.append({'ruta':relativo,'estado':'PRESENTE'})
        if origen.is_dir():
            candidatos=[]
            for directorio, dirs, nombres in os.walk(origen,followlinks=False):
                dirs[:]=sorted(d for d in dirs if d not in ('__pycache__','.git') and not (Path(directorio)/d).is_symlink())
                candidatos.extend(Path(directorio)/n for n in sorted(nombres))
        else: candidatos=[origen]
        for p in candidatos:
            if p.is_symlink() or p.suffix.lower() not in extensiones or not p.is_file():continue
            nombre=str(p.relative_to(raiz))
            if nombre in vistos:continue
            vistos.add(nombre);tam=p.stat().st_size
            r={'ruta':nombre,'bytes':tam}
            if tam>PROTOCOLO_05['limite_archivo_evidencia_bytes']:
                r['estado']='OMITIDO_TAMANO_INDIVIDUAL'
            elif total+tam>PROTOCOLO_05['limite_copia_evidencia_bytes']:
                r['estado']='OMITIDO_LIMITE_TOTAL'
            else:
                antes=sha256(p);destino=salida/'archivos'/nombre
                destino.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(p,destino)
                exigir(sha256(destino)==antes and sha256(p)==antes,'Archivo cambió al recopilar: '+nombre)
                total+=tam;r.update(estado='COPIADO',sha256=antes)
            registros.append(r)
    # No volver a descargar por 403: localizar posibles copias previas, SIN copiarlas.
    datasets=[]
    for rel in ('data','evidence'):
        carpeta=raiz/rel
        if not carpeta.is_dir():continue
        for directorio,dirs,nombres in os.walk(carpeta,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in ('__pycache__','.git') and not (Path(directorio)/d).is_symlink())
            for nombre in sorted(set(nombres).intersection(FECO_ARCHIVOS_05)):
                p=Path(directorio)/nombre
                if not p.is_symlink() and p.is_file():
                    datasets.append({'ruta':str(p.relative_to(raiz)),'bytes':p.stat().st_size,
                        'estado':'LOCALIZADO_NO_LEIDO_NI_VALIDADO'})
    resultado={'alcance':'Recogida de evidencia existente, no ajuste, validacion o nueva ejecucion',
        'raices':raices,'archivos':registros,'bytes_copiados':total,
        'feco_local':datasets,'nueva_descarga':False,'datos_feco_reanalizados':False}
    escribir_json(salida/'INVENTARIO.json',resultado)
    return resultado

def referencia_05(raiz):
    carpeta=(raiz/PROTOCOLO_05['referencia']).resolve()
    exigir(carpeta.is_relative_to(raiz),'Referencia fuera del laboratorio')
    exigir(sha256(carpeta/'MANIFEST_RESULTADOS.json')==PROTOCOLO_05['manifest_referencia_sha256'],
           'Referencia 04 distinta')
    manifest=json.loads((carpeta/'MANIFEST_RESULTADOS.json').read_text())
    for n,digest in manifest.items():
        p=(carpeta/n).resolve()
        exigir(p.is_relative_to(carpeta) and p.is_file() and sha256(p)==digest,'Referencia alterada: '+n)
    exigir(json.loads((carpeta/'RESULTADOS.json').read_text())['estado']=='COMPLETO_PERFIL_PREFIJO',
           'El perfil 04 no esta completo')
    return carpeta/'perfil/traza_prefijo.npz'

def evaluar_hilos_05(baseline, candidato):
    """Mismos bloques instrumentados; no compara contra una referencia sin perfil."""
    resultado={}
    for fase in ('preparacion','ensayo'):
        a=[r for r in baseline if r.get('fase')==fase]
        b=[r for r in candidato if r.get('fase')==fase]
        esperado=40 if fase=='preparacion' else 10
        exigir(len(a)==len(b)==esperado,'Bloques incompletos: '+fase)
        exigir([r['ms'] for r in a]==[r['ms'] for r in b]==list(range(1,esperado+1)),
               'Orden de bloques distinto')
        for filas in (a,b):
            exigir(all(isinstance(r['step_host_wall_s'],(float,int)) and
                       0<r['step_host_wall_s']<PROTOCOLO_05['max_wall_subproceso_s'] for r in filas),
                   'Tiempo no positivo o no finito')
        ta=sum(r['step_host_wall_s'] for r in a);tb=sum(r['step_host_wall_s'] for r in b)
        resultado[fase]={'h14_step_s':ta,'h1_step_s':tb,'factor_ta_sobre_tb':ta/tb,
            'reduccion_fraccion':1-tb/ta}
    ta=sum(r['h14_step_s'] for r in resultado.values());tb=sum(r['h1_step_s'] for r in resultado.values())
    resultado['total']={'h14_step_s':ta,'h1_step_s':tb,'factor_ta_sobre_tb':ta/tb,'reduccion_fraccion':1-tb/ta,
        'supera_umbral_desarrollo_20pct':(1-tb/ta)>=PROTOCOLO_05['min_reduccion_desarrollo']}
    resultado['alcance']='Una repeticion tecnica por brazo; tiempos con cProfile; sin p-valores ni promocion automatica.'
    return resultado

def ejecutar_hilos_05(raiz,salida,hilos):
    """Reutiliza el trabajador 04, cambiando solo la politica local de hilos."""
    import importlib
    exigir(hilos in PROTOCOLO_05['orden'],'Numero de hilos no registrado')
    global PROTOCOLO_04
    previo_protocolo=PROTOCOLO_04
    PROTOCOLO_04={**PROTOCOLO_04,'id':'hilos_05_'+str(hilos),
        'hilos_proyeccion_KC':hilos,
        'politica':'Mismas ecuaciones y kernel. THREADS ajustado solo en proceso desechable; ninguna escritura del core.',
        'perfil_04_reutilizado':True}
    cls=None; anterior=None
    try:
        import numba
        exigir(numba.config.NUMBA_NUM_THREADS>=14,'Numba no permite la referencia de 14 hilos; no cambiar el entorno automaticamente')
        sys.path.insert(0,str(raiz/'src'))
        modulo=importlib.import_module('projection_parallel_brain')
        cls=modulo.GpuProjectionParallelBrain;anterior=cls.THREADS
        exigir(anterior==14,'THREADS original no es 14')
        cls.THREADS=hilos
        codigo=ejecutar_perfil_04(raiz,salida)
        if Path(salida).is_dir():
            escribir_json(Path(salida)/'HILOS_EFECTIVOS.json',{
                'atributo_original':anterior,'atributo_durante_prefijo':cls.THREADS,
                'cambio_permanente':False,'numba':numba.__version__,
                'limite_pool_numba':numba.config.NUMBA_NUM_THREADS,
                'threading_layer':numba.threading_layer() if codigo==0 else None,
                'nota':'backend_identity heredado conserva su texto 14; este recibo declara la politica real del proceso. No guardar checkpoint.'})
        return codigo
    except BaseException as e:
        Path(salida).mkdir(parents=True,exist_ok=True)
        escribir_json(Path(salida)/'ERROR.json',{'tipo':type(e).__name__,'mensaje':str(e),'traceback':traceback.format_exc()})
        return 2
    finally:
        if cls is not None and anterior is not None:cls.THREADS=anterior
        PROTOCOLO_04=previo_protocolo

def campana_hilos_05(raiz,salida,solo_evidencia=False):
    import numpy as np
    salida.mkdir(parents=True,exist_ok=False)
    estado='BLOQUEADO';codigo=2;comparacion=None;paridades={}
    try:
        # Preserve the collector even if runtime prerequisites later fail.
        recoger_predictores_05(raiz,salida/'predictores_existentes')
        escribir_json(salida/'PROTOCOLO_05.json',PROTOCOLO_05)
        if solo_evidencia:
            estado='RECOPILACION_SIN_SIMULACION';codigo=0
        else:
            identidad={n:sha256(raiz/n) for n in (PLAN_REL,MODULO_REL,'laboratorio.py')}
            for n,digest in PROTOCOLO_05['sources'].items():
                exigir(sha256(raiz/n)==digest,'Fuente numerica distinta: '+n)
            pre=verificar_archivos(raiz,leer_plan(raiz));escribir_json(salida/'PREFLIGHT.json',pre)
            exigir(pre['estado']=='VERIFICADO','Archivos base incompatibles')
            ref=referencia_05(raiz)
            escribir_json(salida/'FUENTES_EJECUCION.json',identidad)
            with np.load(ref,allow_pickle=False) as z: referencia={k:z[k] for k in z.files}
            for hilos in PROTOCOLO_05['orden']:
                nombre='hilos_'+str(hilos)
                with (salida/(nombre+'.log')).open('x',encoding='utf-8') as log:
                    print('Ejecutando '+nombre+': prefijo 40+10 ms; sin cambio fisiologico.',flush=True)
                    r=subprocess.run([sys.executable,'-X','utf8',str(raiz/'laboratorio.py'),
                        'diagnostico-olfativo','_hilos_05','--hilos',str(hilos),
                        '--salida',str(salida/nombre)],cwd=raiz,stdout=log,stderr=subprocess.STDOUT,
                        timeout=PROTOCOLO_05['max_wall_subproceso_s'])
                exigir(r.returncode==0,'Trabajador fallido: '+nombre+'; no repetir')
                politica=json.loads((salida/nombre/'HILOS_EFECTIVOS.json').read_text())
                exigir(politica['atributo_durante_prefijo']==hilos,'Politica de hilos distinta')
                with np.load(salida/nombre/'traza_prefijo.npz',allow_pickle=False) as z:
                    actual={k:z[k] for k in z.files}
                paridades[nombre]=comparar_prefijo_04(actual,referencia)
                escribir_json(salida/'PARIDAD.json',paridades)
            comparacion=evaluar_hilos_05(
                json.loads((salida/'hilos_14/BLOQUES.json').read_text()),
                json.loads((salida/'hilos_1/BLOQUES.json').read_text()))
            final=verificar_archivos(raiz,leer_plan(raiz));escribir_json(salida/'PREFLIGHT_FINAL.json',final)
            exigir(final['estado']=='VERIFICADO','Cambio de fuentes base durante el ensayo')
            for n,digest in identidad.items():exigir(sha256(raiz/n)==digest,'Fuente editada: '+n)
            referencia_05(raiz)
            escribir_json(salida/'COMPARACION.json',comparacion)
            estado='COMPLETO_COMPARACION_HILOS';codigo=0
    except BaseException as e:
        escribir_json(salida/'ERROR_CAMPANA.json',{'tipo':type(e).__name__,'mensaje':str(e),'traceback':traceback.format_exc()})
    finally:
        lineas=['# MATRIX — predictores existentes y politica de hilos 05','',f'**Estado: {estado}.**','',
            'No se crea un predictor nuevo. Ver predictores_existentes/INVENTARIO.json para evidencia disponible y ausente.',
            'No se cambia ninguna ecuacion, dt, tolerancia, peso, receptor o actuacion. No se guardan checkpoints.',
            'El ZIP FeCO previo solo contenia un HTTP 403; esta entrega no reintenta descargas.','']
        if comparacion:
            lineas+=['## Comparacion instrumentada','',
                '| Fase | 14 hilos (s) | 1 hilo (s) | Factor |','|---|---:|---:|---:|']
            for fase in ('preparacion','ensayo','total'):
                a=comparacion[fase]
                lineas.append(f"| {fase} | {a['h14_step_s']:.6f} | {a['h1_step_s']:.6f} | {a['factor_ta_sobre_tb']:.4f} |")
            lineas+=['','Umbral de desarrollo >=20% reduccion: '+str(comparacion['total']['supera_umbral_desarrollo_20pct'])]
        lineas+=['','## Limites',
            'Paridad exacta de 31 arrays y 50 muestras, no todo el estado oculto ni la simulacion de 100 ms.',
            'Dos prefijos instrumentados en orden fijo: comparacion descriptiva, no benchmark sin cProfile o replicacion estadistica.',
            'No sumar tiempos acumulados anidados. Telemetria GPU incluye otros procesos.',
            'Un resultado veloz no valida fisiologia. No se cambia la referencia permanente, incluso si pasa.',
            'El nucleo y los predictores permanecen intactos. Ausencia en el paquete no demuestra inexistencia en el laboratorio.','']
        (salida/'INFORME_AUTOMATICO.md').write_text('\n'.join(lineas),encoding='utf-8')
        escribir_json(salida/'RESULTADOS.json',{'estado':estado,'codigo':codigo,'paridades':paridades,
            'promocion_checkpoint':False,'cambio_permanente_hilos':False,'solo_evidencia':solo_evidencia})
        manifest={str(p.relative_to(salida)):sha256(p) for p in salida.rglob('*') if p.is_file()
            and p.name!='MANIFEST_RESULTADOS.json' and not p.name.startswith('RESULTADOS_PARA_CHATGPT_')}
        escribir_json(salida/'MANIFEST_RESULTADOS.json',manifest)
        for p in empaquetar_resultados(salida):print('ADJUNTAR: '+str(p),flush=True)
    return codigo

def main(argumentos=None,*,raiz=None):
    args=list(sys.argv[1:] if argumentos is None else argumentos)
    if not args or args[0] not in ('hilos-05','_hilos_05','recoger-predictores-05'):
        return _MAIN_HISTORICO_04(argumentos,raiz=raiz)
    parser=argparse.ArgumentParser(description='Comparacion de hilos y evidencia de predictores existentes')
    parser.add_argument('accion',choices=('hilos-05','_hilos_05','recoger-predictores-05'))
    parser.add_argument('--salida',type=Path,required=True)
    parser.add_argument('--hilos',type=int,choices=(14,1))
    a=parser.parse_args(args);raiz=Path(raiz or Path(__file__).resolve().parents[1]).resolve()
    salida=a.salida.resolve()
    exigir(salida.is_relative_to(raiz/'runs') and salida!=raiz/'runs','Salida fuera de runs/')
    if a.accion=='_hilos_05':
        exigir(a.hilos in (14,1),'Falta numero de hilos')
        return ejecutar_hilos_05(raiz,salida,a.hilos)
    exigir(a.hilos is None,'Los hilos se fijan por protocolo, no por CLI')
    return campana_hilos_05(raiz,salida,solo_evidencia=a.accion=='recoger-predictores-05')


# Corrección 05B: carga intacta, adopción explícita y coherente de política EN MEMORIA.
_MAIN_HISTORICO_05 = main
PROTOCOLO_05B = {
    'id':'correccion_hilos_05b',
    'referencia_05':'runs/chatgpt_hilos_predictores_05',
    'manifest_05_sha256':'02400153d60bc5687a9e6b0f3b544a9ac64ec7930119487794a8d853b09bd88a',
    'preparacion_ms':40,'ensayo_ms':10,'arrays':31,'muestras':50,
    'hilos_candidato':1,'hilos_referencia':14,
    'max_wall_subproceso_s':1800,
    'repite_referencia_14':False,'reintento_automatico':False,
    'sources':{
        'src/projection_parallel_brain.py':'796eab6a25c705608f03b28d77fb25804caa8f9a7dd41232599150445fbbb51a',
        'src/kc_projection_parallel.py':'c87f87b52d5d160dc99fa5ae78460f427a05ae5f6deb772aef204bc6988dad3c',
        'src/projection_parallel_session.py':'1aa93a6a84ba041ded44faea4a54c16dddc6651682376e01c47a55f60dd975fb'
    },
    'alcance':'Reparación del arnés; no cambio fisiológico. Referencia 14 archivada, comparación exploratoria instrumentada.',
}

def huella_estado_05b(valor):
    """Hash tipado de estados serializables, sin pickle y sin convertir arrays en listas."""
    import numpy as np
    h=hashlib.sha256()
    def campo(datos):
        h.update(str(len(datos)).encode()+b':'+datos)
    def recorrer(x):
        if isinstance(x,np.ndarray):
            exigir(not x.dtype.hasobject,'Estado con array object no admitido')
            campo(b'ndarray');campo(x.dtype.str.encode());campo(str(x.shape).encode())
            campo(np.ascontiguousarray(x).tobytes())
        elif isinstance(x,np.generic):
            recorrer(np.asarray(x))
        elif isinstance(x,dict):
            campo(b'dict')
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))):
                recorrer(k);recorrer(x[k])
        elif isinstance(x,(tuple,list)):
            campo(type(x).__name__.encode());campo(str(len(x)).encode())
            for v in x:recorrer(v)
        elif x is None:campo(b'none')
        elif isinstance(x,bool):campo(b'true' if x else b'false')
        elif isinstance(x,int):campo(b'int');campo(str(x).encode())
        elif isinstance(x,float):
            import struct
            campo(b'float');campo(struct.pack('!d',x))
        elif isinstance(x,str):campo(b'str');campo(x.encode())
        elif isinstance(x,bytes):campo(b'bytes');campo(x)
        elif isinstance(x,Path):campo(b'path');campo(str(x).encode())
        else:raise TypeError('Estado no serializable para huella: '+type(x).__name__)
    recorrer(valor)
    return h.hexdigest()

def validar_sesion_05b(sesion):
    """Ejecuta los validadores originales, sin interceptar ni ignorar sus errores."""
    ejecutados=[]
    for nombre in ('_validate','_validate_pending','_validate_afferent_pending','_validate_coxal_pending'):
        getattr(sesion,nombre)()
        ejecutados.append(nombre)
    return ejecutados

class PoliticaLocal05B:
    """Adopta THREADS en la instancia y su configuración, después de cargar el checkpoint.

    La clase, el kernel y los archivos originales no cambian. El getter de identidad
    llama al getter original y actualiza solamente su texto de hilos; conserva el
    resto de la identidad. No se eluden comparaciones de identidad.
    """
    def __init__(self,objeto,salida,hilos=1):
        import copy
        self.objeto=objeto;self.sesion=objeto.core;self.h=self.sesion.hybrid
        self.salida=Path(salida);self.hilos=hilos;self.activa=False
        self.config_original=self.sesion.config
        self.backend_original=self.h.backend_identity
        self.recibo={'estado':'PREPARANDO','hilos_clase':type(self.h).THREADS,
            'hilos_checkpoint':self.config_original.get('projection_threads'),
            'hilos_solicitados':hilos,'cambio_archivos_core':False,'checkpoint_guardado':False,
            'validadores_omitidos':False}
        exigir(type(hilos) is int and hilos in (1,14),'Política no registrada')
        exigir(type(self.h).THREADS==14 and self.h.THREADS==14,'Clase/instancia original distinta')
        exigir('THREADS' not in self.h.__dict__ and 'backend_identity' not in self.h.__dict__,
               'Ya existe una política sobreescrita en esta instancia')
        exigir(self.config_original.get('projection_threads')==14,'Checkpoint no declara 14 hilos')
        self.recibo['validadores_antes']=validar_sesion_05b(self.sesion)
        identidad=self.backend_original()
        exigir(identidad==self.config_original['electrical_backend_identity'],
               'Identidad original incompatible')
        texto=identidad.get('projection_assembly','')
        exigir(texto.count('in14 CPU threads;')==1,'Texto de identidad no reconocido')
        self.recibo['identidad_original']=copy.deepcopy(identidad)
        antes=self._huella()
        def identidad_efectiva():
            actual=self.backend_original()
            exigir(actual.get('projection_assembly')==texto,'Identidad original cambió en ejecución')
            actual['projection_assembly']=texto.replace('in14 CPU threads;',f'in{hilos} CPU threads;')
            return actual
        try:
            nueva=copy.deepcopy(self.config_original)
            nueva['projection_threads']=hilos
            self.h.THREADS=hilos
            self.h.backend_identity=identidad_efectiva
            nueva['electrical_backend_identity']=self.h.backend_identity()
            self.sesion.config=nueva
            self.activa=True
            self.recibo['identidad_efectiva']=copy.deepcopy(nueva['electrical_backend_identity'])
            self.recibo['validadores_despues']=validar_sesion_05b(self.sesion)
            despues=self._huella()
            exigir(antes==despues,'La adopción alteró estados serializados del modelo')
            self.recibo.update(estado='ADOPTADO_EN_MEMORIA',estado_antes_sha256=antes,
                              estado_despues_sha256=despues,hilos_instancia=self.h.THREADS,
                              hilos_config=self.sesion.config['projection_threads'],
                              hilos_clase_despues=type(self.h).THREADS,
                              alcance_huella='state_dict neural + relojes/sensores y qpos/qvel/act/ctrl, no todos los buffers ocultos')
            escribir_json(self.salida/'POLITICA_LOCAL_05B.json',self.recibo)
        except BaseException:
            self._restablecer()
            self.recibo.update(estado='ADOPCION_RECHAZADA',error=traceback.format_exc())
            escribir_json(self.salida/'POLITICA_LOCAL_05B.json',self.recibo)
            raise

    def _huella(self):
        return huella_estado_05b({
            'neural':self.h.state_dict(),
            'CNS_time_ns':int(self.sesion.time_ns),
            'pending_sensors':self.sesion.pending_sensors,
            'qpos':self.objeto.body.data.qpos,
            'qvel':self.objeto.body.data.qvel,
            'act':self.objeto.body.data.act,
            'ctrl':self.objeto.body.data.ctrl,
        })

    def _restablecer(self):
        self.sesion.config=self.config_original
        self.h.__dict__.pop('THREADS',None)
        self.h.__dict__.pop('backend_identity',None)
        self.activa=False

    def restaurar(self):
        if not self.activa:return
        antes=self._huella()
        self._restablecer()
        self.recibo['validadores_restaurados']=validar_sesion_05b(self.sesion)
        despues=self._huella()
        exigir(antes==despues,'La restauración alteró estados serializados')
        self.recibo.update(estado='RESTAURADO',restauracion_estado_igual=True,
                          hilos_restaurados=self.h.THREADS,config_hilos_restaurados=self.sesion.config['projection_threads'])
        escribir_json(self.salida/'POLITICA_LOCAL_05B.json',self.recibo)

def referencias_05b(raiz):
    """Autentica la campaña bloqueada y rescata exclusivamente su rama completa."""
    import numpy as np
    raiz=Path(raiz).resolve()
    carpeta=(raiz/PROTOCOLO_05B['referencia_05']).resolve()
    exigir(carpeta.is_relative_to(raiz/'runs'),'Referencia fuera de runs')
    exigir(sha256(carpeta/'MANIFEST_RESULTADOS.json')==PROTOCOLO_05B['manifest_05_sha256'],
           'La campaña 05 no coincide con los archivos auditados')
    manifest=json.loads((carpeta/'MANIFEST_RESULTADOS.json').read_text())
    for n,digest in manifest.items():
        p=(carpeta/n).resolve()
        exigir(p.is_relative_to(carpeta) and p.is_file() and sha256(p)==digest,
               'Evidencia 05 alterada: '+n)
    exigir(json.loads((carpeta/'RESULTADOS.json').read_text())['estado']=='BLOQUEADO','Estado 05 inesperado')
    exigir(json.loads((carpeta/'hilos_1/ERROR.json').read_text())['mensaje']=='Parallel execution policy differs',
           'Bloqueo distinto del que corrige esta entrega')
    exigir(json.loads((carpeta/'hilos_1/BLOQUES.json').read_text())==[],
           'La rama fallida contiene avance no previsto')
    exigir(json.loads((carpeta/'hilos_14/RESULTADO_PERFIL.json').read_text())['estado']=='PREFIJO_EJECUTADO',
           'Referencia hilos14 no completada')
    ref04=referencia_05(raiz)
    with np.load(ref04,allow_pickle=False) as z:original={k:z[k] for k in z.files}
    with np.load(carpeta/'hilos_14/traza_prefijo.npz',allow_pickle=False) as z:ref={k:z[k] for k in z.files}
    paridad=comparar_prefijo_04(ref,original)
    exigir(paridad['arrays_identicos']==31 and paridad['muestras']==50,'Referencia incompleta')
    return carpeta,ref,paridad

def ejecutar_hilos_05b(raiz,salida):
    """Un prefijo nuevo; la configuración 14 se mantiene hasta terminar la carga."""
    global PROTOCOLO_04
    original=PROTOCOLO_04
    PROTOCOLO_04={**original,'id':'hilos_05b_1','hilos_proyeccion_KC':1,
                  'politica':'Cargar14, adoptar instancia/config/identidad1, validar, restaurar. No guardar.'}
    try:
        import numba
        exigir(not sys.flags.optimize,'No ejecutar con -O')
        exigir(numba.config.NUMBA_NUM_THREADS>=14,'Pool Numba menor que14; no modificar entorno')
        return ejecutar_perfil_04(raiz,Path(salida),
            al_cargar=lambda obj:PoliticaLocal05B(obj,Path(salida),1))
    except BaseException as e:
        Path(salida).mkdir(parents=True,exist_ok=True)
        escribir_json(Path(salida)/'ERROR.json',{'tipo':type(e).__name__,'mensaje':str(e),'traceback':traceback.format_exc()})
        return 2
    finally:PROTOCOLO_04=original

def campana_hilos_05b(raiz,salida):
    """Rescata 14 hilos; ejecuta sólo el brazo1 bajo un protocolo corregido."""
    import numpy as np
    raiz=Path(raiz).resolve();salida=Path(salida)
    salida.mkdir(parents=True,exist_ok=False)
    estado='BLOQUEADO';codigo=2;comparacion=None;paridad=None
    try:
        escribir_json(salida/'PROTOCOLO_05B.json',PROTOCOLO_05B)
        identidad={n:sha256(raiz/n) for n in (PLAN_REL,MODULO_REL,'laboratorio.py')}
        for n,digest in PROTOCOLO_05B['sources'].items():
            exigir(sha256(raiz/n)==digest,'Fuente numérica distinta: '+n)
        plan=leer_plan(raiz)
        pre=verificar_archivos(raiz,plan);escribir_json(salida/'PREFLIGHT.json',pre)
        exigir(pre['estado']=='VERIFICADO','Fuente base o checkpoint incompatibles')
        carpeta,referencia,paridad_ref=referencias_05b(raiz)
        escribir_json(salida/'REFERENCIA_REUTILIZADA.json',{'ruta':str(carpeta),
            'manifest_sha256':PROTOCOLO_05B['manifest_05_sha256'],
            'paridad_14_con_04':paridad_ref,'nuevas_simulaciones_14':0})
        escribir_json(salida/'FUENTES_EJECUCION.json',identidad)
        with (salida/'hilos_1.log').open('x',encoding='utf-8') as log:
            print('05B: referencia14 autenticada; ejecutar sólo candidato1. Sin cambiar fisiología.',flush=True)
            r=subprocess.run([sys.executable,'-X','utf8',str(raiz/'laboratorio.py'),
                'diagnostico-olfativo','_hilos_05b','--salida',str(salida/'hilos_1')],
                cwd=raiz,stdout=log,stderr=subprocess.STDOUT,
                timeout=PROTOCOLO_05B['max_wall_subproceso_s'])
        exigir(r.returncode==0,'Falló el candidato corregido; conservar ERROR y no reintentar')
        politica=json.loads((salida/'hilos_1/POLITICA_LOCAL_05B.json').read_text())
        exigir(politica['estado']=='RESTAURADO' and politica['hilos_instancia']==1 and
               politica['hilos_config']==1 and politica['hilos_clase_despues']==14 and
               politica['hilos_restaurados']==14 and politica['config_hilos_restaurados']==14,
               'Política no coherente o no restaurada')
        exigir(politica['estado_antes_sha256']==politica['estado_despues_sha256'],
               'La adopción modificó el estado')
        with np.load(salida/'hilos_1/traza_prefijo.npz',allow_pickle=False) as z:
            actual={k:z[k] for k in z.files}
        paridad=comparar_prefijo_04(actual,referencia)
        escribir_json(salida/'PARIDAD.json',paridad)
        comparacion=evaluar_hilos_05(json.loads((carpeta/'hilos_14/BLOQUES.json').read_text()),
                                    json.loads((salida/'hilos_1/BLOQUES.json').read_text()))
        comparacion['alcance']='Exploratorio: h14 archivado y h1 nuevo con cProfile; fecha distinta y metadato de backend coherente. No benchmark causal de velocidad ni promoción.'
        final=verificar_archivos(raiz,plan);escribir_json(salida/'PREFLIGHT_FINAL.json',final)
        exigir(final['estado']=='VERIFICADO','Fuentes/checkpoint cambiaron durante la prueba')
        for n,digest in identidad.items():exigir(sha256(raiz/n)==digest,'Archivo modificado: '+n)
        referencias_05b(raiz)
        escribir_json(salida/'COMPARACION.json',comparacion)
        estado='COMPLETO_COMPARACION_EXPLORATORIA_05B';codigo=0
    except BaseException as e:
        escribir_json(salida/'ERROR_CAMPANA.json',{'tipo':type(e).__name__,'mensaje':str(e),'traceback':traceback.format_exc()})
    finally:
        lineas=['# MATRIX — corrección del ejecutor05B','',f'**Estado: {estado}.**','',
            'Se conserva la campaña05 bloqueada. Se reutiliza su rama14 verificada y sólo se simula la candidata1.',
            'Carga original a14 hilos; adopción en instancia/configuración/identidad, validadores activos y restauración.',
            'No se modifican ecuaciones, dt, tolerancias, pesos, receptores, cuerpo ni checkpoints.','']
        error_path=salida/'ERROR_CAMPANA.json'
        if error_path.exists():
            e=json.loads(error_path.read_text())
            lineas+=['## Error concreto',f"{e['tipo']}: {e['mensaje']}",'']
            child=salida/'hilos_1/ERROR.json'
            if child.exists():
                e=json.loads(child.read_text());lineas += [f"Trabajador: {e['tipo']}: {e['mensaje']}",'']
        if paridad:lineas += [f"Paridad: {paridad['arrays_identicos']} arrays, {paridad['muestras']} muestras.",'']
        if comparacion is not None:
            lineas += ['| Fase | 14 archivado (s) | 1 nuevo (s) | Cociente |','|---|---:|---:|---:|']
            for fase in ('preparacion','ensayo','total'):
                d=comparacion[fase]
                lineas += [f"| {fase} | {d['h14_step_s']:.6f} | {d['h1_step_s']:.6f} | {d['factor_ta_sobre_tb']:.4f} |"]
        lineas+=['','## Límites',
            'Comparación exploratoria instrumentada, referencia de otra fecha. No aceleración general confirmada.',
            'Paridad de observaciones de50ms y huella durante adopción, no todos los estados ocultos ni100ms.',
            'No se eliminan validadores. Getter de identidad conserva llamadas originales y corrige su metadato de hilos.',
            'No se crea predictor, no se reanalizan datos FeCO ni se promueve un modelo.','']
        (salida/'INFORME_AUTOMATICO.md').write_text('\n'.join(lineas),encoding='utf-8')
        escribir_json(salida/'RESULTADOS.json',{'estado':estado,'codigo':codigo,'paridad':paridad,
            'promocion_checkpoint':False,'cambio_permanente_hilos':False})
        manifest={str(p.relative_to(salida)):sha256(p) for p in salida.rglob('*') if p.is_file()
            and p.name!='MANIFEST_RESULTADOS.json' and not p.name.startswith('RESULTADOS_PARA_CHATGPT_')}
        escribir_json(salida/'MANIFEST_RESULTADOS.json',manifest)
        for p in empaquetar_resultados(salida):print('ADJUNTAR: '+str(p),flush=True)
    return codigo

def main(argumentos=None,*,raiz=None):
    args=list(sys.argv[1:] if argumentos is None else argumentos)
    if not args or args[0] not in ('hilos-05b','_hilos_05b'):
        return _MAIN_HISTORICO_05(argumentos,raiz=raiz)
    parser=argparse.ArgumentParser(description='Corrección local de la adopción de hilos; sin cambios fisiológicos')
    parser.add_argument('accion',choices=('hilos-05b','_hilos_05b'))
    parser.add_argument('--salida',type=Path,required=True)
    a=parser.parse_args(args)
    raiz=Path(raiz or Path(__file__).resolve().parents[1]).resolve();salida=a.salida.resolve()
    exigir(salida.is_relative_to(raiz/'runs') and salida!=raiz/'runs','Salida fuera de runs/')
    return (campana_hilos_05b(raiz,salida) if a.accion=='hilos-05b' else ejecutar_hilos_05b(raiz,salida))
