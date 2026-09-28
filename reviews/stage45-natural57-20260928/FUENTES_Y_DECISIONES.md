# Fuentes que acotan la siguiente decisión

Consulta focal del28-09-2026; históricos sólo lectura. No repetir arqueología general, descargar grandes datasets ni cambiar ley/lector por una analogía.

## Evidencia propia reutilizada

- 54: geometría espacial y lector/cuerpo con reparación de unidades, seis brazos89ms. La nueva cualificación exige identidad de los tres prefijos padre completos. Motor verificó que el panel2757 contiene cero de las686ALPN; se retienen2PN legacy y salidasfinas, por lo que esto no significa ausencia de toda observaciónPN.
- 55/56: fronteraPN686 realmente consumida y control de observación; la muestra fija prolongada influye en giro. Se reutiliza sólo el observador, no el escritor/clamp. Datos y hash de procedencia en reference/PROVENANCE.json.
- 56/B: puenteDM1 ya contiene recursos/filtrado rápido y lento; no añadir otro mecanismo de depresión por duplicación. El banco verificó su cálculo, no su equivalencia fisiológica.
- Índice histórico matrix/catalogo/README.md → work/stage3_matched_sources_20260917/family_function_leads.md: identidadMaleCNS DNb05L10118/R10065, limitaciones de transferencia desde la referencia externa Shiu/P9 y de la inferencia conductual desde un componente común. No convertir sus capturas de espigas en medidas del CNS continuo actual.

## Evidencia primaria y disponibilidad

1. [Yang et al., Cell2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12778575/), doi10.1016/j.cell.2024.08.033. La publicación declara datos a solicitud al contacto responsable y código público enZenodo. Esto se comprobó en el fragmento indexado de la sección Data Availability; la apertura directaPMC devolvió una comprobación de navegador. No se descargaron trazas ni se contactó a autores. La figura2 del [preprint2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614758/) distingue los panelesA/B, que excluyen periodos sin marcha, deC/D, que los incluyen. La intervención detallada sobre gestos de patas estudiaDNa02/DNg13; no proporciona por transferencia una ganancia causalDNb05. **Decisión:** conservar lectorDNb05 como instrumento ya fijado, sin validación biológica absoluta ni obligación universal de iniciar avance antes de estudiar giro.

2. Datos locales `matrix/evidence/steering_physiology_20260909/steering_source`, doi10.7910/DVN/0NCLP1: archivo y análisis deDNa01/Rayshubskiy, noDNb05. El código local `calibrate_dna01_steering.py` ajustó una prótesis a un registro con una separación temporal interna; la pendiente no es una medición de conectividadVNC ni una regla intercambiable entre clases. **Decisión:** conservar como donante y referencia de observación; no copiar su ganancia al lector actual. Sólo leer fuentes, no ejecutar el ajuste de nuevo.

## Comparación de propuestas

Codex registróA/B/C antes de los mensajes nuevos; los tres asesores declararon exposición al historial y al resumen recibido. ChatGPT ASTRA_V2 y ChatGPT_Motor_V2 aportaron revisión conceptual, sin examinar los arrays57 ni validar sus archivos. Motor examinó cobertura/geometría y fuentes de reloj; su revisión numérica57 se pedirá tras completar la cola. ModoPRO no verificado, Jev no utilizado, ninguna clave consultada o enviada.

Se incorporó observaciónORN genérica junto conPN, actividadORN comprometida,74filtrosDM1 y salidasespecializadas. No llamar a la frontera genérica la entrada fina. Contrastes fuenteL/R son diferencias entre condiciones, no una prueba independiente de código direccional correcto. La reflexión conserva campo, pero no iguala la suma por las poblaciones323L/371R; diferencia nominal inicial aproximada0.77%. Las métricas de plantilla temprana son descriptivas de la misma vida, nunca un lector entrenado que actúe en el organismo.

Los resúmenesfirst/last/min/max/count deCSR incluyen pruebas predictoras/rechazadas y8llamados de calentamiento sólo al comenzar. La comparación de ventanas51–89/257–384 no incluye ese calentamiento; las sumas por1ms son resúmenes discretos, no dosis física exacta. Las llamadas y los registrosDNg se concilian explícitamente en el verificador reparado.

## Alternativa C: DopaMeander y una limitación concreta del operador

Consulta focal adicional, con presupuesto previo en `RESEARCH_CONTEXT_PLAN.json`: 30 s CPU dentro de preparación, 30 MiB de adquisición, cero CNS/GPU. [Liessem, Dahlhoff y colaboradores, 2026](https://doi.org/10.1016/j.cub.2026.06.045) estudia DopaMeander y el contexto de la dirección de marcha. No aporta por sí solo una ley receptorial de dopamina para nuestro modelo. Se conservaron tablas y código de los autores, **sin ejecutar su código**: [repositorio fijado al commit](https://github.com/MertErginkaya/DopaMeander/tree/28580fade2d226fbfa85d7244240305bd0c462e6), más [metadatos del dataset](https://doi.org/10.6084/m9.figshare.29625239). Adquisición inicial: 87.821 bytes. El archivo bruto de 13.152.124.249 bytes no se descargó.

El R del autor selecciona explícitamente `AVLP476` para ambas DopaMeander. Ese alias permite localizar dos correspondencias de tipo en MaleCNS: 10898/AVLP476_R y 529811/AVLP476_L. No son los mismos root IDs entre sexos ni una demostración de equivalencia funcional. Su etiqueta dopamine es una predicción del catálogo, no una medición de receptores.

La tabla de entradas del autor contiene DNg100 → DopaMeander con 33 sinapsis. **No invertir esa dirección para proponer que DopaMeander activa DNg100.** Las tablas de salida no incluyen DNg100 ni DNb05; ello no excluye vías indirectas o extrasinápticas.

Comprobación local de `natural_none/final_state`, con topología canónica verificada y los valores actuales del operador CPU/CUDA: las 2.239 aristas salientes de 10898 y las 2.117 de 529811 tienen peso efectivo exactamente cero. Ambas tienen `nt_sign=0`; sí conservan entradas efectivas no nulas. Se extrajeron todas esas aristas en `research_context/DM_OPERATOR_EDGES.npz`; procedencia, parámetros actuales, hashes y coste están en `DM_OPERATOR.json`. Esto prueba una limitación del **CSR genérico efectivo**, no de toda ruta especial posible ni del cerebro biológico. No se deduce únicamente del constructor histórico.

Los estados q inicial/final del control son subnormales, aproximadamente 2,67e−322 y 1,83e−322; no se redondean a una afirmación de cero exacto, espigas o concentración de dopamina. No se registró aquí su historia completa. Aun activando esas variables, sus aristas genéricas actuales seguirían sin transmitir. La ausencia de una acción dopaminérgica por esta ruta es una abstracción incompleta para esa pregunta, **no un bug numérico demostrado ni una causa demostrada del fracaso de navegación**.

Decisión: C deja de ser una propuesta inespecífica de «hambre». Tiene una identidad celular y una barrera comprobable, pero requiere una ley/observación independiente que distinga acción lenta, cotransmisión rápida y contexto motor. No asignar excitación uniforme a dopamine, invertir conexiones ni ajustar receptores para lograr PASS. Mantener intacta la campaña 57. ChatGPT_Motor_V2 aportó crítica conceptual; Motor confirmó los archivos locales por separado en `aporte_motor_final/DM_REVISION.json`.

Motor precisó otra frontera: `pn_online_manifest/general_outputs/enabled=false` deshabilita la sustitución de ciertas salidas por el modelo PN fino; no deshabilita las 686 salidas genéricas ni todas las rutas finas. Gamma/additional se analizaron por separado. Esa bandera no es un bug demostrado ni justifica llamarlas una única señal.
