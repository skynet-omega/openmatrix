"""Generate concise report tables from completed, recomputed comparisons."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent

def read(name):return json.loads((HERE/name).read_text())
def num(x,n=3):return f'{x:.{n}f}'

def main():
    pair=read('PAIR100.json');full=read('PARENT1000.json');stable=read('STABLE1000.json')
    cr=read('control_100ms_01/RESULT.json');short=read('candidate_100ms_01/RESULT.json');run=read('candidate_1000ms_01/RESULT.json')
    integrity=read('INTEGRITY1000.json')
    passed=all(full['gates'].values()) and all(stable['engineering_screen'].values()) and integrity['status']=='INTEGRITY_CONFIRMED'
    wall=run['wall_total_s']/60;advance=run['advance_total_s']/60
    headline='Candidata conservada: comparación funcional de 1 s aprobada.' if passed else 'Candidata no admitida: discrepancia funcional en 1 s.'
    cns=run['runtime']['CNS'];events=stable['events'];own=full['events']
    command=stable['motor_commands'];neural=stable['neural_snapshots']['1000']
    exact=sum(v['exact'] for v in stable['fields'].values())
    stats=[('CNS residente',run['cns_resident_s']),('Región celular',run['runtime']['cell']['native_wall_s']),('PN',run['runtime']['PN']['wall_s'])]
    other=run['advance_total_s']-sum(v for _,v in stats);stats.append(('Diferencia sin atribuir',other))
    lines=[
      '# Motor RK3 con memoria de propuesta: resultado real', '',f'**{headline}**', '',
      f'La vida completa consumió **{num(advance,2)} minutos de avance y {num(wall,2)} minutos totales**.',
      f'Margen de 25 minutos totales: **{"cumplido" if wall<=25 else "no alcanzado"}**; diferencia {num(abs(wall-25),2)} minutos {"por debajo" if wall<=25 else "por encima"}.',
      f'Metas de 20 y 12 minutos: {"20 alcanzada" if wall<=20 else "20 pendiente"}; {"12 alcanzada" if wall<=12 else "12 pendiente"}.',
      'Una condición con olor y cuerpo, sin viento. No equivale a validación biológica general, navegación ni sustitución automática del motor estable.', '',
      '## Cambio concreto', '',
      'El controlador distingue un paso pequeño impuesto por un evento de uno exigido por el error. Tras un corte interior aceptado con error<0,1, la propuesta anterior actúa como suelo de la propuesta normal siguiente. Nunca se aplica al final de época ni ante rechazo. El siguiente intento vuelve a comprobar error y dominio.',
      'Sólo cambió esa regla en `engine/resident_controller.cu`; `ENGINE.diff` muestra el porte. RK3, proyección izquierda/derecha, pesos persistentes FP32, estados FP64, PN, membranas, cuerpo y tolerancias permanecieron iguales. El motor07 y el estable no se editaron.', '',
      '## Pareja controlada de 100 ms', '',
      '| Medida | Motor07 | Candidata11 |','|---|---:|---:|',
      f'| Avance integral, s | {num(cr["advance_total_s"])} | {num(short["advance_total_s"])} |',
      f'| Proceso completo, s | {num(cr["wall_total_s"])} | {num(short["wall_total_s"])} |',
      f'| Evaluaciones CNS | {cr["runtime"]["CNS"]["rhs_evaluations"]} | {short["runtime"]["CNS"]["rhs_evaluations"]} |',
      f'| Tiempo CNS, s | {num(cr["cns_resident_s"])} | {num(short["cns_resident_s"])} |','',
      f'Reducción observada: **{num(100*pair["timing"]["advance_fraction_saved"],2)}% de avance y {num(100*pair["timing"]["process_fraction_saved"],2)}% de proceso**. Pasó la puerta fijada de10% de ahorro de avance, proceso sin regresión y compatibilidad funcional; por eso se ejecutó una sola confirmación de1s.',
      'El control reprodujo exactamente35/35campos del primer100ms del motor07 guardado previamente. Ambas ejecuciones usaron BLAS1, misma preparación y registro. Una pareja y orden fijo: no estima variabilidad ni elimina todo efecto de sistema/cachés. Los muestreos no encontraron otra simulación; durante el control apareció un verificador remoto de archivos. No se acredita exclusividad completa de CPU/GPU.', '',
      '## Comparación funcional durante el segundo', '',
      f'- Criba frente a07: {full["status"]}; frente al estable: {stable["status"]}.',
      f'- Integridad externa: {integrity["status"]}; filas completas, relojes CNS/PN/cuerpo a1ms, árboles finales presentes y finitos, presupuesto respetado. Estados iniciales de las tres corridas nuevas coinciden directamente.',
      f'- Mandos de giro diferentes: {command["different_ticks"]}/1000; mando de avance exacto: {stable["engineering_screen"]["same_forward_commands"]}.',
      f'- Pose/cuerpo exacto: {stable["fields"]["qpos"]["exact"]}; velocidades exactas: {stable["fields"]["qvel"]["exact"]}; contactos exactos: {stable["fields"]["contact_active"]["exact"]}.',
      f'- Campos de traza exactos: {exact}/35. Máxima diferencia CNS a1s: {neural["cns"]["max_abs"]:.6g}; voltaje celular: {neural["cell_delta"]["max_abs"]:.6g}mV.',
      f'- Eventos comprometidos estable/candidata: {events["committed"]["stable"]}/{events["committed"]["optimized"]}; bloques con distinta identidad: {events["committed"]["blocks_with_different_identity"]}.',
      f'- Predictores descartados estable/candidata: {events["predictor"]["stable"]}/{events["predictor"]["optimized"]}; bloques diferentes: {events["predictor"]["blocks_with_different_identity"]}.',
      f'- Frente a07, mismos conteos totales por identidad comprometida: {own["committed"]["same_total_identity_counts"]}.', '',
      'Las diferencias neuronales y de eventos se conservan en los JSON; mismas órdenes no significan estados internos idénticos. La posición de un evento en un bloque vecino tampoco mide por sí sola su retraso físico. El criterio de mando incluye yaw y forward; esa aclaración se registró antes de ejecutar la candidata.', '',
      'La auditoría externa `INTEGRITY1000.json` informa también diferencias de payload y cuántos eventos se excluyen del emparejamiento temporal. No añade umbrales de aceptación. La corrida07 histórica no guardó PN/publicación al inicio: se cotejaron su estado neuronal inicial y procedencia, y los árboles iniciales de la candidata contra el control nuevo. No se atribuye al histórico una observación inexistente.', '',
      '## Coste y recursos', '',
      '| Región en1s | Segundos |','|---|---:|',
      *[f'| {label} | {num(value)} |' for label,value in stats], '',
      f'CNS: {cns["accepted"]} pasos aceptados, {cns["rejected"]} rechazados y {cns["rhs_evaluations"]} evaluaciones. La diferencia de tiempos no se atribuye automáticamente a Python: los temporizadores cubren regiones distintas y algunos incluyen esperas.',
      f'Presupuesto: pareja {num(cr["wall_total_s"]+short["wall_total_s"])}s/900s; confirmación {num(run["wall_total_s"])}s/2400s. Exposición:1200ms nuevos en tres procesos, más fixtures sintéticos. La carga/archivo se informa dentro del proceso, fuera del avance.',
      'La referencia estable y el segundo previo07 tienen tiempos históricos con condiciones distintas; sus cocientes no son una pareja de rendimiento controlada.', '',
      '## Revisión y reproducción', '',
      'ChatGPT Motor revisó diff/controlador y posteriormente código de evaluación y cifras. Sus hallazgos sobre registros incompletos motivaron la comprobación externa sin tocar el motor. Jev priorizó la medición integral. Sus revisiones no son ejecuciones independientes; véanse `CHATGPT.md`, `CHATGPT_DATOS.md` y `jev_01/response.json`.',
      'La evidencia se recalcula mediante `python -O verify_saved.py --long`. El paquete de reproducción incluye código del núcleo, comparadores y datos necesarios para estas métricas. Reejecutar el organismo requiere el proyecto y checkpoint originales instalados: el paquete no se presenta como un simulador autónomo.', '',
      ('Se conserva la candidata para el alcance observado.' if passed else 'No se admite la candidata para uso funcional hasta resolver la discrepancia observada.')+' La residencia conjunta de estados/puertos sigue como posible mejora futura, con coste por medir; no se mezclaron otras optimizaciones en esta ronda.', ''
    ]
    (HERE/'RESULTADOS.md').write_text('\n'.join(lines))
    print(headline,advance,wall)

if __name__=='__main__':main()
