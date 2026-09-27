# Qué falta observar para resolver4/5

**Priorizar observación causal por puertos y cobertura descendente antes de cambiar otra ley global o ampliar un diccionario.** La campaña51 contiene respuesta distribuida que sus16trayectorias no describen. Esto no demuestra que haya una orden correcta esperando ser leída: la dosis aérea y el error de unidades siguen limitando su interpretación.

Terna inicial fijada en `TERNA_INICIAL.md` antes de recibir las propuestas nuevas de Matrix/ChatGPT. Conozco y he analizado el historial; no es una revisión ciega. Los cálculos de esta entrega son de arrays/metadata, no una nueva ejecución CNS ni reproducción de experimentos biológicos.

## Evidencia nueva ejecutada

- Las10condiciones tienen `q[90,16]` y estados inicial/final de166.700células. Son900filas temporales de **una preparación**, con una sola q inicial distinta. Los seis brazos parentales aportan540filas; sólo6de16columnas tienen rango mayor que1e−9, umbral descriptivo. Dos componentes de q cruda centrada explican99,979998% de su varianza, dominada por relés de gran amplitud; eso no mide información biológica ni anticipa el resultado de un predictor normalizado.
- Hay1.314descendentes anotadas, de las cuales sólo12tienen trayectoria q registrada. Al comparar aireL−aireR sin olor al final de90ms,532DN cambian más de1e−6; **529deesas532carecen de trayectoria temporal**. JO-C/E:235/335; AMMC/WED:580/1108. Estos umbrales no son tolerancias ni criterios de admisión. No se hizo un ranking para elegir otro lector.
- El efecto al final existe fuera de las16células observadas. No conocemos su latencia, persistencia, relación con el mando ni mediadores a partir de ese único corte. No llamar a esas532células «neuronas de rumbo».
- `PROYECCION.npz` conserva80.301bytes:2.757IDs, cinco finales, tipos y máscaras para recalcular los cuatro contrastes. `verify_projection.py` pasó bajo `-O`, sin leer MATRIX ni51, y detectó NaN, IDduplicado, máscara modificada y conteo alterado. Esto reproduce estadísticas del modelo ejecutado, no su simulación ni una corrección del error de unidades.

## Cobertura real: lo que ya tenemos y lo que no

| Capa | Datos guardados en51 | Límite para diagnosticar |
|---|---|---|
| Entrada olfativa |694valores nominales por ms;74q ORN heredadas y filtros |NominalHz no prueba respuesta de todas las694ORN ni es medida fisiológica; comprobar el operador realmente consumido|
| Aire |335entradasJO por ms; campo, velocidad y rotación |Drive no es qJO. Hay qJO inicial/final, pero falta trayectoria completa de esas335células|
| Relés |4células entre AMMC012/013 yWED203 dentro del panel |No representa todos los relés; JO→AMMC/WED yORN→ALPN son ramas distintas cuya convergencia se investiga|
| PN |629salidas normalizadas,100gamma y366adicionales en nS por ms |Son puertos de una fuente especializada, no629PN observadas. `PN_q_legacy` no sustituye sus emisiones actuales|
| DN |12células q por ms; E/I, target/rate y estado por RHS en dosDNg100 |Falta dinámica de1.302DN y contabilidad comparable en las candidatas de giro/relés|
| Cuerpo |Mando crudo/aplicado, avance, pose, velocidades, contactos |Distinguir señal neural, acción efectivamente aplicada y respuesta física|
| Todo el cerebro |q inicial/final, estados completos de cierre |No hay trayectoria cerebral completa cada ms; el checkpoint no contiene automáticamente todos sus pasos intermedios|

Las16células son seis paresDN (DNg100,DNb05,DNa01,DNa02,DNb06,DNg13) y cuatro relés. IDs/tipos comprobados contra `male_v10/nodes.parquet`, en `NEURONAS_OBSERVADAS.csv`.

## Herramientas existentes que reutilizar

**Escáner:** `/home/daroch/AXIOMA_FLYWIRE/matrix/scripts/scan_stimulus_reach.py` ya alineaIDs, calcula contrastes por célula/grupo y separa puertosPN. Su propio código no certifica causalidad por igualdad inicial de q. Espera matriz tiempo×células, tiempos desde0 y semillas bajo esquemaORN; requiere adaptar explícitamente entradaJO y observadores actuales. Con sólo los endpoints51 puede describir el cambio final; no debe inventar picos/latencias entre muestras. El histórico demostraba alcance en otro motor/estado, no valida los números actuales.

**Predictor:** `src/physiological_generator/live_forecast.py` es un pronóstico numérico desechable que cambiaba el acoplamiento15,625→125µs conservando las ecuaciones/contexto. Tuvo un éxito prospectivo acotado, no una ley fisiológica aprendida. Su función exige el tipo de runtime antiguo y origen15.625ns; el motor actual ya usa125.000ns. **Su aceleración histórica5,286× no vuelve a estar disponible cambiando el mismo ajuste.** Reutilizar contratos/observación, no ejecutar el donante como sustituto automático. La mezcla mecanismo+MLP ORN tuvo negativos de transferencia conservados.

**Observadores de puertos:** `campanas/etapa45_composicion_20260927_48/observer.py` y adaptadores51 ya guardan operadorBASE, overrides finales y continuidad. Extender por selección anatómica previa puede distinguir entrada consumida equivocada, una transformación de estado inesperada y pérdida en el lector. Un qfinal solo no distingue esas tres causas.

**Leyes actuales:** el BASE genérico usa objetivo `max(0,tanh(gain*(net+drive−theta)))` y relajación1/τ; el visual y las fuentesPN/celulares especializadas no usan todos esa ley. La varianteG51 fue una hipótesis normalizada de ingeniería y no obtuvo ventaja específica frente aI. Los modelosPN físicos existentes proporcionan unidades y modelos comparables en su dominio; no convierten automáticamente pesosEM o q de cualquierDN en conductancias/voltajes reales.

## Datos biológicos: disponibilidad comprobada y límites

| Recurso | Original o evidencia comprobada | Uso legítimo / falta |
|---|---|---|
| Suver2019, viento |ZIP local9.477.291.327bytes con `DATA_SETS_SuverEtAl2019.7z`, listado externo confirmado; recibo previo cotejó MD5oficial |Siguiente acceso selectivo: esquema de estímulo, deflexión antenal y respuestaJO/APN/WPN por animal. No se abrió7z ni se recalculó su hash completo aquí. No descargar otra copia|
| KadakiaORN |Original y extracción local dePID/válvula/espigas; revisión23sep leída |Distinguir entrega y transducción. ab3A/acetato de etilo y ab2/2butanona no calibranDM1 oJO. El candidato temporal previo no justificó integración|
| PN/ModelDB118662 |Código y manifiestosDM1 pasivos verificados; parámetros publicados deORN→PN con errata conservada |Modelo de referencia compuesto y específico, no registro sináptico individual deMaleCNS ni escala PN→DN|
| DNa02 |MAT original71.381.032bytes presente; voltajes bilaterales y bola según adquisición/código primario |Un animal expuesto; FIR previo falló criterio. No equivale aDNb05 ni identifica q→Hz|
| DNb05 |`extractKernels.m` original inspeccionado: fluorescencia, conducta y kernels |Código disponible; grabaciones no verificadas en esta revisión. No sustituirlas con el código o con datosDNa02|
| MamiyaFeCO |ZIP/READMEoriginal y directorio interno leídos: DF/F por mosca frente a ángulo tibial |Permite comparar forma/signo angular, condicionado a observador de calcio; no identifica unidades80drive ni fuerza|
| ChenFeCO |ZIPoriginal/listado interno confirmados; recepción anterior documenta108MAT |MapeoLexA→tipo canónico pendiente; DF/F no es corriente|
| Fujiwara |Documentación y código ya comprobados en51:2D/fases de tres patas izquierdas |No identifica seis ángulos3D. La prueba geométrica anterior conserva ese bloqueo específico|
| Prattcinta |CSVlineal3D de1.553.195.603bytes presente, disponibilidad previa documentada |Alternativa cinemática aFujiwara; xyz en unidades arbitrarias requiere conversión/registro de articulaciones. No se analizaron sus respuestas aquí|

Fuentes primarias: [Suver](https://doi.org/10.5061/dryad.k06kh8f), [Kadakia](https://doi.org/10.5061/dryad.1ns1rn8xd), [PN](https://modeldb.science/118662), [DNa02](https://doi.org/10.7910/DVN/0NCLP1), [códigoDNb05](https://zenodo.org/records/12775493), [Mamiya](https://doi.org/10.5061/dryad.dbrv15f6q), [Chen](https://doi.org/10.5061/dryad.rfj6q57bm), [Fujiwara](https://zenodo.org/records/6365304), [Pratt](https://doi.org/10.5061/dryad.mpg4f4r73). Esta tabla distingue disponibilidad de archivos, contenido documentado y validación; no afirma reanalizar cada grabación.

## Tres alternativas, discriminadores y costes

| Alternativa | Información y control | Qué la refutaría | Coste acotado / decisión |
|---|---|---|---|
| A:observación causal por puertos |Unidades corregidas, entradaJO realmente consumida, controles de intensidad, estado común y toda poblaciónDN. Separar configuración lateral, cantidad total y realimentación corporal |La supuesta dirección se explica por intensidad/offset, o desaparece al verificar el puerto; tampoco sobrevive un control de replay pertinente |Primera cribaCPU ya ejecutada. En futura corrida,2757q por ms son22,06MB/s por brazo sin compresión; no es una medición de costeGPU. Prioridad alta; presupuesto de integración deberá fijarse antes|
| B:identificación fisiológica de dinámica |Registro estímulo→respuesta y operador de observación comparable. Modelo instantáneo frente a dinámica temporal mínima; condiciones/animales completos separados |La dinámica no supera al estático, parámetros no identificables, o falla transferencia entre condiciones. Sin equivalencia celular/unidades, abstenerse de trasplantar |Siguiente criba propuesta≤30sCPU sobre un subconjunto≤10MiB;0CNS. SiSuver no permite acceso selectivo, detener esa criba y registrar el recurso necesario; no extraer9GB por defecto|
| C:validación causal del lector |Actividad y conducta emparejadas, signos/relojes/estado conocidos. Comparar lector estático y temporal fuera del organismo, frente a media/sin señal, usando episodios reservados |No supera al control simple o el signo no transfiere; falta el observable biológico apropiado; ajuste retrospectivo de navegación no cuenta |Primero≤30sCPU de esquema/ajuste pequeño,0CNS. ReutilizarDNa02 sólo para su dominio y buscar un subconjuntoDNb05comparable antes de afirmar calibración del lector actual|

Los topes de30sCPU de B/C son propuestas **para tareas posteriores**, no ejecución autorizada en esta entrega de25sCPU. No se prescriben nuevos pasos ni se fusionan A/B/C en una arquitectura.

Dictionary Learning aprende una representación dispersa que reconstruye los datos; su objetivo no identifica por sí mismo ecuaciones o causalidad ([algoritmo primario](https://www.jmlr.org/papers/v11/mairal10a.html), [API oficial](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html)). **Mi inferencia:** aquí puede servir como compresor/diagnóstico condicionado al pasado, pero no llenar las1.302trayectoriasDN ausentes ni demostrar un mecanismo. Comparar con persistencia/PCA/Ridge, evaluar por condición completa y comunicar por separado la salidaDN; un error agregado dominado porAMMC podría engañar. Tras fijar mi terna, Matrix comunicó un benchmark retrospectivo propio; no lo ejecuté ni incorporé sus números como resultado de esta criba.

## Panel prospectivo sin seleccionar por respuesta

En la siguiente corrida autorizada, elegir por anotación las1.314DN,335JO-C/E y1.108AMMC/WED, conservando IDs y tipos. Registrar q cada ms y los puertosPN con unidades separadas; observar la ramaORN→ALPN además deJO→AMMC/WED, sin imponer una cadenaJO→PN. Conservar señal motora cruda, aplicada, pose, velocidad física y los relojes efectivos de consumo. E/I, objetivo y tasa de las células previamente definidas permiten interpretar la transferencia; añadir sólo esas columnas útiles, sin volcar todos los RHS del cerebro.

Para4falta finalmente orientación online distinguible de replay, dosis y movimiento basal con la misma arquitectura; para5una perturbación física reservada y recuperación contra su control. El escáner, el predictor y el diccionario son herramientas externas para decidir qué corregir; ninguno sustituye esa evidencia conductual.

## Reproducción y cierre

Copiar sólo `PROYECCION.npz`, `PROYECCION.json`, `ENDPOINTS.json` y `verify_projection.py` a una carpeta y ejecutar `python -B -O verify_projection.py`. RequiereNumPy; noMATRIX,CUDA ni checkpoint. `VERIFICACION.json` conserva el resultado local y cuatro corrupciones detectadas.

Cribas y exportación/verificación medidas: menos de2sCPU de proceso en conjunto y pico159,3MB; recibos individuales con alcance preciso. Las lecturas auxiliares no están incluidas en esa suma instrumental. CeroCNS/GPU, sin entrenar otro predictor, sin cambiar51/raíz/histórico y sin extracción masiva. Salidas nuevas muy inferiores a10MiB; hashes y total exacto en `ENTREGA.json`. Hito de observabilidad/inventario y tres propuestas completo; no se abre otra exploración en este cierre.
