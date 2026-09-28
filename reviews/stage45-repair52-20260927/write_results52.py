"""Generate numerical report from reconstructed metrics, not handwritten values."""
from pathlib import Path
import json
import numpy as np

H=Path(__file__).resolve().parent


def main():
    result=json.loads((H/'PILOT_RESULTS.json').read_text())
    population=json.loads((H/'POPULATIONS.json').read_text())
    queue=json.loads((H/'QUEUE_RESULT.json').read_text())
    qualification=json.loads((H/'QUALIFICATION.json').read_text())
    body=json.loads((H/'BODY_SUPPLEMENT.json').read_text())
    plan=json.loads((H/'PILOT_PLAN.json').read_text())
    fp32=json.loads((H/'FP32_BOUNDARY.json').read_text())
    if queue['status']!='COMPLETE' or qualification['status']!='PASS':
        raise ValueError('Cannot write completed-campaign report')
    air_material=all(result['air'][str(o)]['both_material'] for o in [0,1])
    verdict=dict(classification='PROMETEDOR_NO_CONFIRMADO' if air_material or result['conductance']['promising_screen'] else 'DESCARTADO',
                 repair_classification='CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA',
                 air_original_joint_gate_both_odors=air_material,
                 conductance_specific_gate=result['conductance']['promising_screen'],
                 stage4_pass=False,stage5_pass=False,
                 physical_air_units_reconstructed=True,wide_observer_qualified=True,
                 attempted_neural_ms=queue['attempted_ms'],
                 next_quantity_control=population['input_quantity_matching_next'])
    (H/'VERDICT.json').write_text(json.dumps(verdict,indent=2)+'\n')
    table=['| Condición | ½(L−R) DNb05, q | ½(L−R) giro calculado, °/s | Diferencia total JO R/L, fronteraFP64 |',
           '|---|---:|---:|---:|']
    for odor in [0,1]:
        a=result['air'][str(odor)];p=population['contrasts'][str(odor)]
        table.append(f"| {'Con olor' if odor else 'Sin olor'} | {a['half_L_minus_R_DNb_q']:.9g} | {a['half_L_minus_R_yaw_deg_s']:.9g} | {100*p['JO_R_over_L_minus1']:.6f}% |")
    groups=['| Población | Células observadas | Diferentes en algún instante, sin olor | Diferentes al final, sin olor |',
            '|---|---:|---:|---:|']
    names={'JO_CE':'JO-C/E','AMMC_WED':'AMMC/WED','all_DN':'Descendentes'}
    for group in population['groups']:
        a=population['contrasts']['0']['populations'][group]
        groups.append(f"| {names[group]} | {a['cells']} | {a['cells_crossing_threshold_any_time']} | {a['cells_above_at_90ms']} |")
    corrections=['| Brazo | Máxima diferencia q en las16células heredadas frente a51 | Máxima diferencia JO |', '|---|---:|---:|']
    for name,a in population['corrected_minus_51'].items():
        corrections.append(f"| {name} | {a['max_abs_q16']:.9g} | {a['max_abs_JO_drive']:.9g} |")
    g=result['conductance']
    motion=['| Brazo | Mando medio de avance (mm/s) | Desplazamiento físico entre muestras1y90 (mm) |','|---|---:|---:|']
    for name in ['air0_odor0','air0_odor1','G_odor0','G_odor1','I_odor0','I_odor1']:
        displacement=float(np.linalg.norm(body['arms'][name]['recorded_step1_to90_displacement_mm']))
        motion.append(f"| {name} | {result['arms'][name]['mean_forward']:.9g} | {displacement:.9g} |")
    text=f'''# Campaña52 — reparación cualificada y comparación terminada

**Las etapas4/5 siguen abiertas.** Se completaron diez condiciones de90ms y16ms de cualificación desde la misma preparación48. La conversión de velocidad corporal cm/s→mm/s quedó corregida y comprobada en la simulación. El registro pasó de16salidas a una población anatómica de2757células, más694ORN identificadas por separado. Ningún resultado de esta campaña aplica el giro calculado al cuerpo.

Clasificación de la pista de transferencia: **{verdict['classification']}**. Reparación e instrumentación: **{verdict['repair_classification']}**, dentro de los pares y tiempos comprobados. No hay equivalencia fisiológica ni aprendizaje general demostrado.

## Resultado con los criterios conservados

Ventana51–90ms; mínimos originales1,6e−5 en q y0,02°/s en giro calculado. CampoL/R son condiciones opuestas aplicadas al conjuntoJO de ambas antenas, no mediciones separadas de una sola antena.

{chr(10).join(table)}

Los dos contrastes de aire cumplen el umbral conjunto: **{air_material}**. Eso conserva una pista de transferencia en este modelo. No demuestra que el circuito codifique correctamente el rumbo: la cantidad totalJO sigue confundida con su configuración, y el signo temporal se informa completo en `POPULATIONS.json`.

La interacción específica(Gconolor−Gsinolor)−(Iconolor−Isinolor) es **{g['G_minus_I_odor_effect_q']:.9g}q** y **{g['G_minus_I_odor_effect_yaw_deg_s']:.9g}°/s**. Supera ambos criterios: **{g['both_material']}**; promoción tras control de saturación: **{g['promising_screen']}**. No confundir excitabilidad basal G/I con un rescate olfativo selectivo.

{chr(10).join(motion)}

El mando tiene unidades de velocidad, pero no es el desplazamiento observado. La tabla conserva ambos. El desplazamiento indicado abarca89ms entre las muestras1y90; no se presenta como marcha estable. El giro aplicado fue cero por diseño, con comprobación desde las trazas. No se aplicó perturbación mecánica para la etapa5.

## Lo que permite ver el registro ampliado

{chr(10).join(groups)}

Conteos del contrasteL−R con|Δq|>1e−6. Es un umbral descriptivo; no está calibrado contra ruido fisiológico ni identifica neuronas de orientación. Las curvas guardan tiempos cada1ms, magnitud y signo, sin seleccionar otro lector o ajustar ganancia/polaridad. Los puertosPN y los registrosRHS de dosDNg100 conservan variables y relojes distintos. No se impone una cadenaJO→PN olfativa.

## Qué cambió al reparar las unidades

{chr(10).join(corrections)}

La reparación cambió más de1e−6 la salida final de **{population['corrected_minus_51']['G_odor0']['endpoint_changed_above1e_6']['all_DN']}descendentes en G sin olor** y **{population['corrected_minus_51']['I_odor0']['endpoint_changed_above1e_6']['all_DN']} en I sin olor**. No era legítimo acotar el cambio cerebral a partir de una diferencia pequeña de entrada. La repetición demuestra ahora que esos cambios no modificaron la conclusión del contraste olfativo específico.

**Precisión del registro:** `JO_drive` y su testigo corresponden a la suma aditivaFP64, antes de la conversión de `FastCSR` aFP32. La reconstrucciónCPU de esa conversión explica la invariancia de los camposL/R: cambian valoresFP64 hasta5,03e−8, pero ningún valorFP32 lateral cambia. La velocidad corporal era diminuta, no exactamente cero. Campo0 y G/I sí cambian enFP32. Tras convertir componentes y sumarlos enFP64, la diferencia de total lateral sigue siendo **{100*fp32['postcast_R_over_L_minus1']['0']:.9f}%**. Esto es una reconstrucción del componenteJO bajo el contrato nativo de entradaJO basal cero, no una nueva capturaJO dentro de todos losRHS.

Se compara con los arrays51 expuestos, no con cifras copiadas del informe. Sólo la conversión física y la observación ampliada cambian en52; las leyes, preparación y criterios quedan conservados. Los originales51 permanecen intactos. Una diferencia pequeña en estas condiciones no habría permitido omitir la comprobación física, y no acota otros regímenes corporales.

## Cualificación, presupuesto y alcance

Identidad de33campos frente a49; tres pares vivos con/sin observador ampliado;11/12propietarios científicos,58campos comunes y secuenciaevent_audit exactos. El observadorDNg heredado es común: esta prueba no demuestra su neutralidad universal. Motor C++/CUDA hizo verificación independiente desde copia limpia y reconstrucciónJO con otra implementación. ChatGPT ASTRA_V2 y ChatGPT_Motor_V2 aportaron críticas conceptuales; no se les atribuye ejecución de arrays ni modoPRO verificado.

Ejecución de cola: **{queue['attempted_ms']}msCNS**, **{queue['CPU_s']:.3f}sCPU** y **{queue['queue_wall_s']:.3f}s de pared**. Límites previos:916msCNS/5000sCPU/4200spared/24GiBRAM/14GiBVRAM/8GiB. Análisis y empaquetado tienen recibos adicionales; el tiempo de cola no representa toda la sesión. No hubo reintentos ni ampliación por observar un resultado negativo. Una preparación; diez condiciones no son diez organismos independientes ni una cohorte ciega.

## Decisión y alternativas

El control de cantidad tiene justificación según la puerta declarada: **{population['input_quantity_matching_next']}**. El siguiente contraste propuesto esL1 totalJO emparejado conservando soporte/proporciones, en otra campaña de cuatro brazos90ms con presupuesto propio. Sólo elimina la diferencia de total; no controla simultáneamenteL2, número activo y anatomía. No se ha ejecutado en52.

La cribaCPU de entradas51 mostró que una permutación dentro de(tipo,rootSide) movía320identidades pero dejaba160vectores exactamente iguales. Esa variante es no-op y se descarta como prueba informativa. IgualarL2 y reducir soporte para un multiconjunto común siguen siendo intervenciones distintas, con otros factores de confusión; un negativo tras retirar identidades no se atribuye automáticamente a cantidad.

La normalización idealL1 de esa criba tenía un residualFP64 de1,82e−12; después de convertir componentes aFP32, la diferencia máxima de sumasL/R es **{fp32['L1_screen51_after_cast']['max_pair_sum_difference']:.9g}** unidades. No afirmar igualdad exacta del operador a partir del primer residual. La siguiente campaña deberá fijar y verificar prospectivamente la frontera y el criterio numérico posterior a esa conversión, sin elegirlos para cambiar un veredicto neural.

Se mantienen las alternativas de transmisión/estado y contexto propioceptivo con sus controles propios. A orientación con el cuerpo actual; B CNS→VNC acotado; C músculos y seis patas posteriores. No adoptar un diccionario o predictor externo como controlador ni exigirlo como nueva puerta de etapa.

Hito de esta ronda: reparación y observación cualificadas, repetición finita completa y comparación reproducible. La etapa4 requiere una prueba conductual online contra controles emparejados; la5 requiere recuperación ante perturbación reservada. Esta campaña no las reemplaza.

![Comparación](COMPARACION.png)
'''
    (H/'RESULTADOS.md').write_text(text)
    print(json.dumps(verdict))


if __name__=='__main__':main()
