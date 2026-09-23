# Etapa 3: cruce preliminar con datos reales y diagnóstico del motor

**Resultado: etapa 3 aún abierta.** Esta ronda no ejecutó otro organismo ni ajustó sus parámetros. Recalculó desde trazas conservadas el punto donde la orientación bilateral deja de cumplirse, descargó un solo registro fisiológico publicado y midió si los estados del conectoma sustentan la premisa de un 95 % espacialmente inactivo. [Plan prospectivo](PLAN.json), [código reproducible](analyze_preliminary.py) y [resultado completo](RESULT.json).

## Qué mostró la mosca simulada completa

La preparación tiene 166.700 neuronas, cuerpo MuJoCo, un único estado inicial y cuatro condiciones. Ventana común 161–311 ms tras un ON a 11 ms; sin nuevas corridas. El lado de los 74 ORN_DM1 se obtuvo de `rootSide` de MCNS v1.0 y el de DNa02 de `instance`, sin inferirlo de la conducta.

| Condición | ORN q izquierda − derecha | DNa02 q izquierda − derecha | Entrada nativa neta DNa02 L/R | Giro ° |
|---|---:|---:|---:|---:|
| Olor izquierdo | +0,4718 | +0,1766 | +691 / −1396 | +0,3794 |
| Olor derecho | −0,5498 | +0,1938 | +848 / −1457 | +0,4022 |
| Uniforme | −0,00006 | +0,1886 | +820 / −1446 | +0,3915 |
| Sin nuevo olor | +0,00046 | +0,1737 | +670 / −1438 | +0,3829 |

El contraste sensorial correcto llega a los ORN, pero **no aparece con el signo bilateral requerido en la entrada ni en el estado de DNa02**. El sesgo previo existe sin olor y el olor derecho incrementa la dominancia izquierda. El mando aplicado y el cuerpo siguen ese signo. Esto acota el primer fallo observado al trayecto **entre ORN y la entrada DNa02**; las capturas actuales no localizan una sola sinapsis culpable ni prueban que el cuerpo responda correctamente a un mando espejo. La hipótesis de sólo una inversión del signo motor perdió prioridad, no quedó falsada causalmente. La intervención PFG ensayada sigue descartada como solución de esta preparación.

## Qué permite y qué no permite el registro de mosca viva

Se descargó **un solo archivo** de [Rayshubskiy et al., eLife 102230](https://elifesciences.org/articles/102230), [Harvard Dataverse DOI:10.7910/DVN/0NCLP1](https://doi.org/10.7910/DVN/0NCLP1), ID 11634643. SHA256 `75a48fab5a9809189e3439719ab9c65163e37e226398a766f5720d91f6cedb67`; 63.045.930 bytes. Contiene 1.900 s de registro continuo DNa02/velocidad de bola, a 4.000/100 Hz, con **165 pulsos de 500 ms**. El [código de los autores](https://github.com/SashaRayshubskiy/eLife_102230_analysis_code) agrupa ensayos LeftOdor/RightOdor/BothOdor por nombres de archivos separados; este archivo Dataverse presenta un canal binario de estímulo 0/5, yaw y fisiología, **sin esa etiqueta por pulso**. Tampoco hay nombres LeftOdor/RightOdor/BothOdor en los 380 archivos enumerados por el manifiesto del dataset. No asignamos lateralidad a partir del giro observado, pues sería circular. La ausencia de etiqueta afecta a este archivo y al manifiesto consultado; no afirma que nunca pueda recuperarse desde otra fuente o consultando autores.

El artículo estudió hembras jóvenes sujetas sobre una bola con activación optogenética Orco-LexA/CsChrimson de cada antena, observó giro hacia el lado estimulado y DNa02 ipsilateral; la activación DNa02 precedía el giro en promedio unos 150 ms. La preparación computacional usa un conectoma **macho**, estímulo químico/modelado a ORN_DM1, cuerpo FlyBody y una ley q que no es la tasa de espigas medida. El artículo además excluyó 2/6 moscas de la cohorte de olor ficticio por respuestas asimétricas, con criterios de inclusión no predeterminados. Su signo cualitativo es una referencia pertinente; las magnitudes y latencias no se pueden trasplantar ni usar como umbral de 335 ms sin calibrar estas diferencias. El `reference_behavior_source` del protocolo local era `null`: la campaña anterior no tenía un enlace conductual crudo emparejado. [Artículo y métodos](https://elifesciences.org/articles/102230).

## Qué revela la propuesta espacial de Gemini

En las diez capturas del organismo, **73,5–75,6 %** de las neuronas tienen q>0,01; al contar aristas salientes reales, **67,4–68,5 %** parten de esas neuronas en tres instantes. Es una medición del estado q y de la conectividad, **no** la fracción que dispara ni una cota sobre qué estados requieren integración fina. Por eso no prueba que el 95 % esté en reposo ni que un active set acelere este modelo. Tampoco descarta saltos analíticos para subdominios cuya influencia futura se acote correctamente. Antes de implementarlos se debe instrumentar entrada recibida, jacobiano/acoplamiento y errores de proyección en el organismo, no escoger 2–3 % por analogía.

Gemini acierta en que la velocidad segura actual basta para un piloto **acotado**: 50 ms tomaron ~183,6 s de avance, por lo que 200–300 ms serían del orden de minutos a decenas de minutos si el coste escala parecido. Eso no obliga a cerrar el motor ni justifica repetir sin mecanismo nuevo los 335 ms negativos. El evento tardío que engañó al estimador ya se reprodujo en CUDA; los cortes siguen activos. La meta de 1 s simulado/minuto real permanece operativa, no condición biológica.

## Decisión A/B/C y siguiente intervención acotada

- **Etapa 3 A, pérdida o inversión de lateralidad aguas arriba de DNa02:** mapear, en las trazas ya conservadas, contrastes firmados ORN→PN→CX/PFL3→entradas DNa02 por tipo y lado, con sham y uniforme. La suma de los 2.208 aferentes ya se reconstruyó en la ampliación de abajo; falta localizar el primer grupo causal mediante intervención fijada de antemano. Falsador: una señal derecha correcta y dominante en el puerto DNa02.
- **Etapa 3 B, historia/basal:** comparar una segunda preparación simétrica fijada antes de mirar la salida, manteniendo estímulo y parámetros; alternar historias equivalentes de inicio. Falsador: mismo sesgo aun cuando el estado basal y la historia se igualan. No volver a ajustar ganancias ORN después del giro.
- **Etapa 3 C, lector/cuerpo:** aplicar mandos espejo desde un mismo estado corporal y medir yaw, apoyo y latencia. Falsador: respuesta espejo correcta y suficientemente grande; entonces priorizar A/B. Una intervención diagnóstica no se convierte en controlador operante.

Motor: **A** implícito acoplado con masa/JVP; **B** multirritmo espacial con decaimiento analítico y cortes/eventos cubiertos; **C** compilación de un IR común a CPU/GPU con modelo heterogéneo. El siguiente prototipo debe medirse contra el organismo seguro, incluir un modelo no olfativo y reproducir el falsador de evento tardío; máximo dos prototipos completos y presupuesto previo. El muestreo q de esta ronda baja la prioridad de B hasta demostrar localidad real. No se elige un solver universal por fixture pequeño ni se promete un salto de velocidad sin perfil acoplado.

ChatGPT recibió la fuente pública anterior y un encargo de código/matemáticas ejecutables. Su primera respuesta produjo [código CPU reproducido](chatgpt_temporal_cpu.py); el segundo encargo, con trazas y aferentes reales, está entregado y su respuesta aún pendiente. Jev recibió una consulta de triage y devolvió HTTP 403, sin dictamen; [recibo](jev_01/receipt.json). Esta ausencia no cuenta como conformidad. Sin subagentes Codex.

Reproducción desde la raíz de AXIOMA_ASTRA, con el archivo raw en este directorio y las trazas históricas intactas:

```bash
OPENBLAS_NUM_THREADS=1 /home/daroch/miniconda3/envs/GPU/bin/python -B campanas/etapa3_datos_reales_motor_20260922_01/analyze_preliminary.py
```

El script revisa estructuras, identidades, relojes, finitud y hashes de entradas, y reconstruye el JSON. No reproduce la simulación de 335 ms ni el análisis estadístico completo del artículo.

Para la revisión externa se prepararon copias de las cuatro trazas de 1 ms, los cuatro recibos de entrada nativa a DNa02, una captura de q de 166.700 neuronas, [metadatos mínimos de lados](side_metadata.json) y [grados salientes](outdegree.npy). Estos dos últimos derivan de la anatomía local inmutable; sustituyen los grandes archivos completos en la lectura externa y reprodujeron exactamente las cifras numéricas del modelo. El archivo de mosca viva se obtiene directamente del [Dataverse, ID 11634643](https://dataverse.harvard.edu/api/access/datafile/11634643); no se redistribuye. Se puede ejecutar el paquete externo desde su directorio así:

```bash
python -B analyze_preliminary.py --source traces --nodes-side side_metadata.json --outdegree outdegree.npy --skip-live --out simulation_readback.json
```

Para añadir el archivo vivo descargado, sustituir `--skip-live` por `--live-raw /ruta/al/archivo.mat`. Los resultados de un solo individuo no generan una distribución poblacional.

## Ampliación exploratoria: balance de aferentes

Tras el primer readback se registró una [ampliación acotada](AMENDMENT_AFTER_READBACK.json), sin nuevas corridas ni cambios de modelo. [El cálculo](rank_afferents.py) reconstruye las 2.208 transmisiones que entran a cada DNa02 a partir de `transmission × cap × signed_weight`; máximo error frente al `net` nativo: 1,87×10⁻¹¹. [Resultado íntegro](AFFERENT_RANKING.json).

La diferencia nativa L−R de sham es +2.109,36 unidades. Respecto de sham, el olor izquierdo cambia ese balance −20,75; el derecho **+197,22**; uniforme +158,59. Las mayores contribuciones absolutas al incremento anómalo del brazo derecho provienen de grupos anotados `CB0431|L` (+65,58), `IB068|R` (+32,46) y `PS018_a|L` (+30,22). Los grupos tienen cancelaciones importantes: son términos de una suma observada, no una prueba de función ni autorización para suprimirlos. El siguiente experimento mecanístico deberá intervenir uno o varios grupos ya fijados **y** conservar sham/uniform, no escoger la manipulación que fuerce el yaw esperado.

El [subconjunto de metadatos de aferentes](afferent_metadata.json) conserva los 2.208 ID, tipos y lados. Sustituye el Parquet anatómico completo en una revisión externa. Se comprobó igualdad exacta del JSON local y el recalculado con este subconjunto:

```bash
python -B rank_afferents.py --source traces --nodes afferent_metadata.json --out afferent_readback.json
```

## Cortes a 100/200/300 ms: control directo de la propuesta externa

La [ampliación temporal prospectiva](AMENDMENT_TIME_WINDOWS.json) leyó la corrida completa existente de 335 ms, sin iniciar otra. [Código](read_time_windows.py), [todos los valores y hashes](TIME_WINDOWS.json). Giro desde ON a 11 ms:

| Condición | 100 ms | 200 ms | 300 ms |
|---|---:|---:|---:|
| Olor izquierdo | −0,213185° | −0,217633° | +0,162518° |
| Olor derecho | −0,213185° | −0,217633° | +0,185270° |
| Uniforme | −0,213186° | −0,217633° | +0,174588° |
| Sham | −0,213185° | −0,217632° | +0,165952° |

A 200 ms el olor derecho parece girar hacia su lado si se mira solo su signo, pero su diferencia de giro frente a sham es −0,0000008°. A 300 ms los cuatro giran a la izquierda y el derecho supera a sham en +0,019319°; el estado DNa02 L−R y su entrada nativa también son positivos en todos. Un solo giro a 200 ms sería un falso criterio de éxito. Estos son cortes de la misma corrida PFG ya descartada, no un ensayo independiente ni evidencia de que otro mecanismo no pueda funcionar.

La [medición espacial fijada antes de calcular](AMENDMENT_SPATIAL_PROXY.json) usa exactamente 0→1 ms de una captura de los 166.700 estados y 25.582.938 aristas con ID alineados. [Código](measure_spatial_proxy.py), [resultado](SPATIAL_PROXY.json). Un umbral |Δq|>10⁻⁴ incluye 12,11 % de neuronas y 17,67 % de aristas salientes; el 3 % con mayor |Δq| explica 49,40 % de la suma `outdegree×|Δq|`, mientras el 10 % explica 99,27 %. Esto no mide disparos ni entrada postsináptica y no certifica un umbral seguro; sí impide adoptar por supuesto el 2–3 % activo del comentario de Gemini. Quedan por instrumentar pesos, químicos, retardos, jacobiano y error real de una omisión.

## Código matemático externo probado, sin promoción al motor

ChatGPT entregó dos prototipos CPU en [chatgpt_temporal_cpu.py](chatgpt_temporal_cpu.py), conservado con SHA256 `c8c3e177424d7d7e16c15e8dd0e67bf34c3f68c55f68e4b9c59dc9795b60bd79`. Se ejecutó sin cambios de fuente: [resultado bruto](chatgpt_cpu_run_01/RESULTADO.json). A usa Radau con cortes y masa; B usa propagación afín de coeficientes constantes con cota de truncación. En el evento tardío, A tuvo error 1,91×10⁻¹¹ y B 2,05×10⁻¹⁰ frente a `expm`, menor que su cota 2,14×10⁻¹⁰. En cuatro estados recurrentes durante 1 s, ambos también pasaron su referencia CPU. B rechaza el sistema no lineal; su cota excluye redondeo y error de resolver masa. Son referencias de ecuaciones pequeñas, **no** una implementación CUDA del organismo, una certificación universal de eventos ni un benchmark de cerebro completo. La segunda tarea de ChatGPT exige código verificable sobre los arrays públicos reales.

Las dos lecturas nuevas y el código CPU de ChatGPT se repitieron desde un ZIP extraído a otro directorio, combinando las trazas del snapshot principal mediante [dependencia fijada](TIME_SPATIAL_SOURCE_DEPENDENCY.json). Coincidieron exactamente los resultados numéricos, comprobaciones y arrays; se excluyeron rutas y cronómetros de esa comparación. [Recibo de verificación](TIME_SPATIAL_PORTABLE_VERIFICATION.json). Se mantiene aparte el dato fisiológico original en Dataverse.
