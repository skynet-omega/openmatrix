"""Build numerical report and scientific figure from verified outputs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
subprocess.run([sys.executable,'-O',str(H/'verificar_criba.py')],check=True,capture_output=True)
r=json.loads((H/'CRIBA.json').read_text());e=json.loads((H/'aporte_motor/ENDPOINTS.json').read_text())
con=e['contrasts']['airL_minus_airR_no_odor'];changed=con['all_DN']['above_1e_6'];missing=con['DN_without_temporal_q']['above_1e_6']
names=['persistence','state16','PCA4']+['dictionary_'+str(s) for s in [11,29,47,71]]
labels=['Persistencia','Estado + Ridge','PCA4 + Ridge','Diccionario 11','Diccionario 29','Diccionario 47','Diccionario 71']
arms=r['conditions']['excluded_from_fit']; rows=[]
fig,axs=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
for i,(ax,arm) in enumerate(zip(axs,arms)):
    vals=[r['metrics'][n][arm]['DNb05_difference_rmse']*1e6 for n in names]
    ax.barh(labels,vals,color=['#84919b','#156082','#66a5ad']+['#bb7740']*4)
    ax.set_title(('Aire izquierdo + olor','Aire derecho + olor')[i]);ax.set_xlabel('Error de predicción bilateral DN (q × 10⁻⁶)')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
axs[0].invert_yaxis();fig.suptitle('Pronóstico condicional a 10 ms · condiciones excluidas del ajuste',fontsize=12)
fig.text(.5,.01,'Criba retrospectiva, una preparación. Diccionarios con avisos de convergencia. No es prueba de navegación.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(H/'COMPARACION.png',dpi=170);plt.close(fig)
for n,label in zip(names,labels):
    v=[r['metrics'][n][arm]['DNb05_difference_rmse']*1e6 for arm in arms]
    rows.append(f'| {label} | {v[0]:.3f} | {v[1]:.3f} |')
warning_counts={n:len(w) for n,w in r['warnings'].items()}
parameters={'representation':{'state16':0,'PCA4':4*16,**{'dictionary_'+str(s):32*16 for s in [11,29,47,71]}},
 'ridge_parameters_including_intercept':{n:16*(r['input_columns_retained']+(16 if n=='state16' else 4 if n=='PCA4' else 32)+1) for n in names[1:]},
 'warning_distinct_messages':warning_counts,'meaning':'Representation parameters exclude common centering/scaling; no claim of exact matched capacity.'}
(H/'SUPLEMENTO_CRIBA.json').write_text(json.dumps(parameters,indent=2)+'\n')
text=f'''# Qué falta para avanzar en las etapas 4/5

**Prioridad: mejorar la observación causal y cerrar la interfaz física. No está demostrado que falte un SAE, una ley celular concreta o más datos en general.** La investigación consultó a ChatGPT ASTRA_V2, ChatGPT_Motor_V2 y Motor C++/CUDA; cada uno propuso tres alternativas antes del contraste. La terna propia también quedó registrada. No se crearon subagentes ni se atribuye modo PRO verificado.

## Dos comprobaciones ejecutadas

**1. Cobertura insuficiente del registro temporal.** De {e['groups']['all_DN']} descendentes anotadas, sólo {e['groups']['DN_with_temporal_q']} tenían q registrada cada ms en51. En el contraste aire izquierdo−derecho sin olor, al final de90ms, {changed} presentan |Δq|>10⁻⁶; **{missing} de ellas no tienen trayectoria temporal guardada**. El umbral es descriptivo, no una tolerancia numérica ni un criterio de navegación. El efecto todavía mezcla configuración/dosis y el error de unidades. No identifica neuronas de giro; sí demuestra que el panel seleccionado deja muchas respuestas sin describir. [Datos y verificador](aporte_motor/INFORME.md).

**2. Dictionary Learning frente a métodos simples.** Se ajustó sobre cuatro condiciones parentales de51, excluyendo las dos combinaciones aire+olor. Predice q10ms después a partir del estado observado y las entradas presentes; no es una trayectoria libre desde el checkpoint. Los datos ya estaban expuestos y conservan el error de unidades51. Hay una sola preparación, no420 animales independientes. No se mezclaron G/I ni campañas de leyes distintas.

Error bilateral DNb05 en unidades q×10⁻⁶; menor es mejor:

| Instrumento externo | Izquierda + olor | Derecha + olor |
|---|---:|---:|
{chr(10).join(rows)}

![Comparación](COMPARACION.png)

El diccionario no cumple el criterio previo de mejorar al mejor control en ambas condiciones y en las cuatro inicializaciones sin regresión global. El cálculo consumió **{r['CPU_s']:.3f}sCPU**; la reejecución en el mismo entorno bajo `-O` coincidió exactamente. Las cuatro variantes de diccionario emitieron avisos de convergencia de la optimización: el negativo pertenece a **esta configuración y presupuesto**, no demuestra que un SAE bien entrenado sea inútil. No se reajustó para cambiar la decisión.

La reconstrucción de q mejora frente a PCA4, pero esa mejora no se traduce en mejor pronóstico bilateral en ambas condiciones. Los tamaños tampoco son idénticos: PCA usa64 coeficientes de base y el diccionario512, con cuatro coeficientes activos por muestra. Todo ello queda registrado. **No adoptar hoy este diccionario; mantener Ridge como referencia externa, aún sin validez para escoger intervenciones causales.** El clasificador mecánico `DESCARTADO_EN_ESTA_CRIBA` no descarta la familia de representaciones dispersas.

## Alternativas contrastadas y decisión

| Opción | Qué permitiría descubrir | Estado / siguiente discriminador |
|---|---|---|
| Escáner causal por puertos y poblaciones | Dónde se modifica o pierde una perturbación realmente consumida | Prioridad inmediata: reparar unidades, cualificar el observador y repetir los brazos afectados. Registrar todas las DN y fronteras sensoriales, sin escoger por ranking. Después una intervención finita valida la ruta propuesta. |
| Predictor dinámico y calibración fisiológica | Si el fallo depende de memoria/estado, excitabilidad o transferencia | Reutilizar ARX/DMDc, controles físicos y Jaxley/SBI locales cuando haya datos compatibles. Exigir condición e historia excluidas del ajuste; no convertir el predictor en controlador. |
| Diccionario/SAE y otras representaciones poblacionales | Si una combinación distribuida aporta una descripción más útil que células o PCA | Criba pequeña ejecutada, sin ventaja robusta. Reconsiderar sólo con una pregunta, cobertura y validación causal que lo justifiquen; no entrenar uno grande por analogía con Claude. |

Hay datos reales disponibles, pero sus dominios importan. Suver2019 ya está local; no se descargaron sus9,48GB otra vez. Jaxley y SBI ya tienen pruebas técnicas locales, no identificación biológica completa. Se recuperó únicamente el README de Kathman2026 (9.511bytes), que describe imagen y conducta sincronizadas: permite seleccionar después un subconjunto para estudiar persistencia tras retirar olor. No se adquirió su archivo completo ni se integró su política de navegación programada.

## El adjunto de Anthropic

Es parcialmente verdadero y exagera algunas conclusiones. Dictionary Learning, J-space (julio2026) y el estudio de171conceptos emocionales (abril2026) son reales. Los171conceptos se eligieron previamente; no son171emociones humanas descubiertas. Los rasgos de un diccionario no quedan garantizados como independientes y las neuronas individuales no son inútiles. [Verificación y fuentes primarias](FUENTES_Y_ADJUNTO.md).

## Cómo continúa el proyecto

La decisión operativa está en [PLAN_CONTINUACION.md](PLAN_CONTINUACION.md): reparar y medir antes de cambiar otra ley global; mantener las hipótesis de interfaz, transmisión/estado y propiocepción, con máximo dos prototipos completos por ronda. Se corrigió el asesoramiento que trataba JO→PN como una cadena obligatoria. Aire y olor tienen ramas distintas cuya convergencia hay que comprobar.

**Etapa4:** falta demostrar orientación útil dependiente de información online frente a controles pertinentes. **Etapa5:** falta recuperación tras una perturbación física reservada. Un escáner, una neurona activa o una predicción acertada no sustituyen esas pruebas. Es razonable continuar investigando; todavía no hay evidencia que garantice superar ambas etapas con el preparado actual.

## Reproducción y límites

Entorno del ajuste: Python3.10, NumPy{r['numpy']}, scikit-learn{r['sklearn']}. Verificación corta: `python -O verificar_criba.py`; recalcula métricas y decisiones desde arrays y detecta seis corrupciones. `python -O aporte_motor/verify_projection.py` reproduce los conteos y cuatro corrupciones. Reproducción del ajuste: `python -O criba_instrumentos.py --verify` con las versiones indicadas. No carga fuentes externas al paquete ni ejecuta CNS. Las comprobaciones de hashes deben hacerse antes de ejecutar el verificador de Motor, que escribe su recibo local.

Presupuesto de esta investigación:120sCPU instrumental,0CNS/GPU,2GiBRAM,100MiB nuevos y25min de revisión activa; asesor local con tope25sCPU. El ajuste principal consumió1,10s y su repetición1,12s; Motor informó1,088sCPU en sus cribas/exportación/verificador. Estas cifras no incluyen todas las lecturas auxiliares ni el razonamiento remoto. Se conservaron los avisos de optimización y los resultados negativos. Motivo de cierre: hito de comparación de herramientas, cobertura y plan concreto completo. No se ha ejecutado la reparación CNS51 ni se inició una simulación de etapas4/5 en esta consulta.
'''
(H/'RESULTADOS.md').write_text(text)
print(json.dumps({'report':str(H/'RESULTADOS.md'),'figure':str(H/'COMPARACION.png'),'warning_counts':warning_counts,'endpoint_DN_changed':changed,'without_time_series':missing}))
