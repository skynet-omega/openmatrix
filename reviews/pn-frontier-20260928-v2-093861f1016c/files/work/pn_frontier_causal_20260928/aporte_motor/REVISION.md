# Revisión de la intervención PN común/lateral

**Dictamen:** la descomposición descriptiva es correcta, pero la intervención p0 ± d no es admisible con los datos de 57. Un contraste causal válido puede comparar p0, m=(pL+pR)/2, pL y pR como entradas terminales explícitas sobre un mismo receptor, conservando además el control natural intacto. Identifica efectos de esa frontera parcial; no reproduce la historia natural ni demuestra orientación biológica.

Se leyó el PLAN del principal antes de revisar código y arrays. Esta revisión conoce los resultados expuestos de 54–57; no es una confirmación ciega. No se modificaron fuentes compartidas ni se ejecutaron CNS/GPU.

## 1. Incompatibilidad algebraica comprobada

En las 686 coordenadas PN consumidas por la frontera genérica:

- c=(pL+pR)/2−p0.
- d=(pL−pR)/2.
- m=p0+c=(pL+pR)/2.

Las entradas originales están en [0,1]. Para que **ambos** vectores p0+d y p0−d estén en ese dominio es necesario y suficiente, coordenada por coordenada, que |d| ≤ min(p0,1−p0). Los datos incumplen esa condición.

Comprobación sobre natural_{none,L,R}/PN_consumed.npz de 57, convirtiendo las muestras FP32 a FP64 para hacer el álgebra sin cambiar los originales:

| Muestras comparadas por índice de ms | Coordenadas ms×PN fuera del dominio en al menos un signo | PN afectadas | Mínimo de p0+d | Mínimo de p0−d |
|---|---:|---|---:|---:|
| first | 573 / 263.424 | 72405, 101669 | −0,000200825511 | −0,000333292817 |
| last | 575 / 263.424 | 72405, 101669 | −0,000200825511 | −0,000333545438 |

Esa comparación por índice no es una alineación de todos los RHS. Por ello comprobé también un contraejemplo que **no necesita emparejar RHS**. En el ms local 349, PN **101669**, los extremos retenidos son:

- p0 ∈ [0,0].
- pL ∈ [0,0019785852637141943; 0,001979854889214039].
- pR ∈ [0,0013233119389042258; 0,0013240680564194918].

Incluso el mayor p0−d posible dentro de esas envolventes es:

p0_hi + (pR_hi−pL_lo)/2 = **−0,00032725860364735126**.

Ninguna selección de valores de esos intervalos repara el dominio. Las envolventes certifican incompatibilidad en 545 coordenadas ms×PN; otras 30 no quedan certificadas como válidas por ese test conservador. Las 262.849 restantes sí satisfacen las cotas de ambos signos.

Promediar tampoco salva la propuesta: las plantillas promedio de last en 257–384 ms dan mínimos −0,000106110376 y −0,000319110733; en 51–89 ms también falla una coordenada. No corresponde recortar, suprimir esas PN ni buscar otra ganancia para obtener el signo deseado: cambiaría el contraste.

En cambio, m, pL y pR permanecen en [0,1] por construcción convexa. Los promedios aritméticos de muestras también pueden usarse como entradas constantes válidas, pero deben nombrarse **plantillas fijas derivadas de muestras**, no medias temporales de todos los RHS.

## 2. Los resúmenes no permiten un replay natural exacto

Cada brazo conserva 384×686 valores de primera/última muestra, mínimo, máximo y conteo. pn_probe55.py lee el operando FP32 justo antes del kernel CSR. Los conteos totales son 165.392 / 167.612 / 166.268 llamadas para none/L/R; por ms abarcan 316–588, 308–620 y 312–580. Incluyen calentamientos iniciales y evaluaciones de prueba.

Esos resúmenes no determinan el orden, tiempo ni valor de las muestras intermedias. No existe una correspondencia natural «RHS número k» entre soluciones adaptativas con distintos conteos. Los extremos sirven para cotas de dominio, no para reconstruir trayectorias. PN_q comprometida es otro observable y no sustituye automáticamente la transmisión filtrada consumida.

Interpolar first/last o mantener una muestra por ms serían intervenciones nuevas. Repetir los tres brazos completos de 384 ms para capturar otra cinta exigiría **1.152 ms CNS**, superior a la reserva de 600 ms de esta ronda, antes de probar una intervención. No lo recomiendo como requisito de este discriminador.

## 3. Frontera, propietarios y control nulo

La cadena inspeccionada es gpu_coefficient_layout._GpuSynapticVisualBrain → FastCSR.__call__ → coefficient_fast. La entrada PN procede de los estados de transmisión; FastCSR copia el operando a su **buffer FP32 propio** antes de la suma. El escritor de 56 modifica las 686 posiciones de ese buffer antes del observador de consumo y del kernel. Comprobé IDs y filas únicos e iguales en los tres brazos respecto de reference/TERMINALS.npz.

Ése es el punto para reutilizar el instrumento. No reemplazar el estado del integrador ni modificar pesos, caps o lector. El espejo de pesos y sus propietarios conservan su orden original. La copia del operando debe preceder a cada intervención, también dentro del CUDA Graph; los modos deben ser visibles en dispositivo.

general_outputs=False deshabilita una sustitución especializada concreta; **no significa que las 686 PN carezcan de salida genérica**. PN fina, filtros ORN→PN, salidas gamma/adicionales y consumidores especializados siguen usando sus propias rutas/estados. El escritor CSR no las interviene directamente. Pueden cambiar indirectamente por recurrencia: «sin intervención directa» no significa «señales congeladas».

Algunos consumidores especializados recalculan objetivos tras el CSR. El efecto se atribuye a la salida genérica intervenida con sus consumidores efectivos, no a toda la neurona PN ni a todas sus conexiones. Un efecto persistente fuera de esa frontera tampoco prueba por sí solo un origen exclusivamente posterior a PN.

El no-op debe usar el mismo instrumento en modo desactivado y en modo identidad que escribe exactamente el valor entrante. No sustituirlo por un replay first/last ni por recomponer m±d: una identidad algebraica puede cambiar bits por redondeo. Comparar con el padre estado comprometido, eventos/relojes, operandos, cuerpo y mandos; conservar los valores ajenos a las 686 posiciones. Los saltos deben coincidir con fronteras temporales declaradas, sin depender del orden de llamadas de prueba.

## 4. Operación mínima y efecto identificable

Elegir **una sola** regla de extracción antes del resultado nuevo: una muestra retenida o un promedio de una ventana ya expuesta. Mantener las 686 coordenadas, el mismo receptor inicial, contexto sensorial, duración, inicio y lector.

Comparar entradas constantes p0, m, pL y pR, más el padre natural sin sustitución. Este último controla el efecto de fijar la frontera y retirar su variación/feedback. Un p0 fijado no es «natural intacto».

Para un observable descendente Y, informar:

1. **Común sobre esa base:** Y(m)−Y(p0).
2. **Contraste entre fuentes alrededor de m:** [Y(pL)−Y(pR)]/2.
3. **Curvatura/interacción:** [Y(pL)+Y(pR)]/2−Y(m).

El tercer término importa: el CNS tiene rectificación, saturación, estado y recurrencia. Por identidad, [Y(pL)+Y(pR)]/2−Y(p0) suma el primer y tercer término; no es automáticamente el efecto común aislado. El efecto de p0±d sobre otra base no queda identificado por estos cuatro brazos.

m iguala el vector medio de L/R, **no la cantidad de cada brazo individual**. En la plantilla last tardía, las sumas PN sin ponderar son 454,271512900 / 458,205906733 / 458,213381335 para p0/L/R. La suma de d es −0,00373730116, no cero. Caps y pesos convierten esas entradas de forma dependiente del consumidor. Este contraste no identifica «dirección pura a dosis idéntica». Si ésa es la pregunta exigida, hace falta un contraste de cantidad justificado; no presentarlo como ya controlado.

Como dimensión compatible con la reserva, cinco brazos de 89 ms más padre/no-op de 2 ms cada uno suman **449 ms y siete procesos**. No prueba que ese horizonte baste ni congela el contrato del principal. Si no discrimina, conservar la indeterminación sin alargar buscando el resultado. No he implementado ni cualificado aquí el nuevo escritor CUDA; se reutilizaría el patrón de 56 bajo un contrato nuevo.

## 5. Tres alternativas sustanciales y falsadores

| Alternativa | Operación y decisión | Falsador o límite |
|---|---|---|
| **A. Frontera genérica — prioridad** | Plantillas válidas p0/m/L/R, receptor común y natural intacto; separar común, contraste de fuente y curvatura con un instrumento reutilizado. | Un no-op distinto invalida el instrumento. Con consumo correcto y entradas distintas, ausencia del efecto descendente material/signo prefijado debilita esta explicación en esa preparación; no demuestra ausencia de función PN ni admite orientación. |
| **B. Transferencia con referencia fisiológica** | Un tramo y observable compatibles; contrastar cinética actual y alternativa justificada. | Sin identidad/estímulo/observable emparejados no es identificable. Si la alternativa no mejora una predicción pertinente fuera de su ajuste, conservar la actual. El fallo de dominio de A no autoriza cambiar una ley. |
| **C. Contexto transmisor e iniciación** | Vía efectiva y respaldada hacia DNg100, entrada/estado controlados, observación de mediación antes de intervenirla. | Una vía sin transmisión representada no permite probar mediación por activación. Un contexto identificado que no cambia operandos/mando debilita ese contexto; no justifica bajar umbrales ni equiparar dopamina con excitación. |

La incompatibilidad p0±d ya cambia una decisión y cierra el preflight de esa operación. La próxima ronda debe producir un efecto causal interpretable o un negativo acotado. Si dos rondas no reducen incertidumbre, advertir estancamiento y cambiar mecanismo/preparación; no añadir otra variante de interpolación, ganancias o clipping al mismo contraste.

## Fuentes y reproducción

- [PLAN del principal](/home/daroch/AXIOMA_FLYWIRE/matrix/work/pn_frontier_causal_20260928/PLAN.md).
- [Observador PN](/home/daroch/AXIOMA_FLYWIRE/matrix/campanas/etapa45_transferencia_causal_20260928_55/pn_probe55.py).
- [Escritor de 56](/home/daroch/AXIOMA_FLYWIRE/matrix/campanas/etapa45_transferencia_temporal_20260928_56/hold_probe56.py).
- [Copia FP32](/home/daroch/AXIOMA_FLYWIRE/matrix/motor_nuevo/full_pipeline_review_20260925_12/engine/fp32_operator.py).
- [Ensamblador y consumidores](/home/daroch/AXIOMA_FLYWIRE/matrix/src/gpu_coefficient_layout.py).
- [Verificador histórico de cobertura](/home/daroch/AXIOMA_FLYWIRE/matrix/campanas/etapa45_senal_natural_20260928_57/verify57.py).

Los cálculos propios sólo leen PN_consumed.npz, PN_ids/PN_rows de neural_and_inputs.npz y reference/TERMINALS.npz de 57. Para reproducir: construir d con first/last, contar p0±d fuera de [0,1] y usar d_lo=(L_lo−R_hi)/2, d_hi=(L_hi−R_lo)/2 para las cotas. El contraejemplo robusto usa índice de ms 348, PN 101669, sin emparejar RHS individuales. No se importaron módulos GPU ni se repitió el verificador completo de 57.

**Coste medido:** 0,1050874 + 0,0942278 = **0,1993152 s CPU**, más lecturas breves de texto no instrumentadas; presupuesto asesor 60 s. Cero CNS/GPU y sin descargas. La única escritura es este informe en el destino asignado; un primer intento de guardar el Markdown fue rechazado por formato antes de escribir. Etapas 4/5 abiertas; no-op nuevo y contraste causal aún no ejecutados.

