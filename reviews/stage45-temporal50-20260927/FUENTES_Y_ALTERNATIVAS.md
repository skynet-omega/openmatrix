# Investigación paralela50, previa al resultado conjunto

Se conservan A/interacción sensorial temporal, B/transferencia o estado neuronal y C/discrepancia concreta de implementación. La prueba neuronal50 sólo interviene A; no ajusta las otras para lograr el criterio. Los ocho efectos no se consultaron para redactar esta nota.

## Restricciones verificadas en fuentes locales

La ley genérica del CNS procede de `anatomical_rate_brain.py` y se declara una candidata no calibrada. Sus parámetros tau/gain/theta/rmax se muestrean de priors positivos; el código no los presenta como fisiología medida por tipo. La escala sináptica inicial0,03, los signos por consenso de transmisor y la actividad exterior cero son hipótesis explícitas. El constructor base omite normalización morfológica, pero el estado ACTUAL sí conserva una intervención posterior por volumen: gain/=size/median y theta*=size/median, adoptada a520ms. Se verificó el recibo dentro del checkpoint48; no confundir la descripción del constructor con el preparado continuado. Ese escalado tampoco mide capacitancia ni resistencia por tipo. DNg100 tiene theta663,2585/559,1861 internos en el operador49, no el prior7,5 original. La evolución posterior también cambia integrador y añade modelos especializados; el comentario histórico sobreEuler no describe el motor actual. Esta procedencia ya existía y no es un bug nuevo descubierto50.

`pn_general_output_brain.py`, líneas78–79, conserva la implementación heredada cuando `general_outputs.enabled=false`. En `pn_electrical_output_brain.py` se distingue reemplazo especializado de466 salidas de PN10208 frente a629 consumidores generales que conservan transmisión heredada. Por tanto no procede activar el booleano como reparación de una supuesta desconexión de todaPN. Son1095 consumidores de una neurona, no1095 neuronasPN. El alcance concreto de49/50 debe cotejarse con su manifiesto y fuentes ejecutadas.

La pareja registrada es PN10208/10176 (DM1_lPN, izquierda/derecha) y no toda la población; MotorC++ comprobó identidades. La sustitución de objetivos ORN en48/50 deja las actividades recurrentes evolucionar pero sustituye explícitamente la modulación entrante de esos objetivos; no es un transductor químico calibrado.

## Comparador publicado

Shiu2024, DOI10.1038/s41586-024-07763-9, PDF ya local `/home/daroch/AXIOMA_FLYWIRE/papers/shiu_nature_2024_lif_model.pdf`, SHA256abf14d5bad39b7dcdd5c3383a81b55590992a2fe03ab3a226a2b873dae04abf6, páginas11–12. ModeloLIF con parámetros compartidos tomados de trabajos previos y un peso sináptico global seleccionado usando la respuestaMN9 a neuronas gustativas. El artículo declara límites de tasas absolutas, basal0 y omisión de receptores/neuromodulación. No ofrece una conversión q→mV ni una calibración de nuestras DNb05/DNg100. Su éxito en alimentación/acicalamiento no admite nuestras etapas4/5.

Fuente de código primaria consultada: https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py . Nature yPMC no entregaron texto en la consulta; se usó el PDF local original y el repositorio del autor, sin atravesar sus controles de acceso. No se adoptó un modelo alternativo ni se descargó otro conectoma.

## Decisiones que esta revisión evita

A: esperar el factorial congelado; no elegir otro retardo/dosis después. B: una restricción biológica debe identificar variables y condiciones; modificar un parámetro provisional puede ser ingeniería legítima, pero no se disfraza de reparación ni equivalencia biológica. C: el booleanoPN no constituye evidencia de desconexión; sólo una contradicción reproducible del consumidor autoriza corregirlo. No se incorporan patas, VNC ni un lector entrenado externo.

Se pidieron a ChatGPT ASTRA_V2 y ChatGPT_Motor_V2 fuentes primarias y una única acción discriminante posterior; respuestas pendientes al crear esta nota. No se atribuye ejecución de archivos ni modoPRO verificado.

## Comprobación algebraica sin otra simulación

En las diez ventanas capturadas49, net+drive de ambas DNg100 permanece negativo aun antes de restar theta: máximos −668,66491699/−675,64398193 internos. Con esas entradas congeladas, cualquier theta no negativo, gain positivo y cap positivo conserva objetivo rectificado0; tau positivo sólo cambia la relajación. Por eso ajustar sólo esos parámetros de las dos DN no proporciona un rescate directo de esos balances. Esto NO descarta cambios de la red recurrente, otros estados o una ley diferente y no constituye una intervención dinámica. Resultado/procedencia en SUBTHRESHOLD49_CERTIFICATE.json.

## Cobertura directa PN→DN comprobada sin pasos neuronales

La propuesta externa de comprobar cobertura se ejecutó con presupuesto15sCPU/0CNS. Se usó la claseALPN del metadato original (686células), el conteo anatómico pre→post y las filas efectivas48 verificadas en07/49. DNb05izquierda recibe94ALPN/2006sinapsis y derecha95ALPN/2450sinapsis. El100% de esas sinapsis tiene arista presente y coeficiente no nulo, con eficacia por sinapsis±0,03 dentro del redondeoFP32 y signo consistente con la hipótesis de transmisor del modelo. Esto no mide actividad, fuerza fisiológica ni caminos indirectos. DNg100L yDNa02R reciben cada una1sinapsis directaALPN; sus homólogas ninguna. Denominadorcero se informa null, nunca cobertura0 o1.

Resultado: la desconexión general de estas entradas directas no explica el negativo observado. Ley de eficacia, otras entradas y estado recurrente siguen abiertos; topología presente no demuestra transferencia funcional suficiente. ArchivosPN_COVERAGE.json/csv; coste1,035sCPU. No ajuste de pesos/ganancias, no simulación nueva.

ChatGPT ASTRA_V2 propuso replayfísico condicional; se descartó escoger un brazo supuesto 'seleccionado' por50, porque el contrato sólo define un contraste de ocho brazos. La respuesta corporal a un replay tampoco prueba que un mando sea correcto para orientar respecto del olor. Cheong96084 ya estaba catalogado, no es recurso local nuevo. Las nuevas sugerencias Ispizua/Li son pistas externas no verificadas aquí como fisiología DN y no se integran. ChatGPT_Motor_V2 coincide con la ausencia de calibración directa suficiente; los PDFs/código y datos locales sostienen únicamente las afirmaciones acotadas anteriores. NoPRO verificado ni ejecuciónNPZ porChatGPT.
