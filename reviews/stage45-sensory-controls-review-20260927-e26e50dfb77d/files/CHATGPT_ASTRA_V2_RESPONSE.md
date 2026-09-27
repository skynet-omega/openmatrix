## Dictamen

**Sí, es defendible una criba provisional de composición con cantidad incremental emparejada por antena y basal común**, siempre que se formule como **contraste funcional de representación**, no como simulación fisiológicamente calibrada de 1-hexanol. El control decisivo debe separar tres cosas: cantidad total, expansión del número de ORN estimuladas y patrón glomerular.

**Exposición previa:** esta no es una revisión ciega. Antes de formular lo siguiente leí `RESULTADOS.md`, que ya contiene el A/B/C del paquete, además de los hechos de tu mensaje. Por eso no atribuyo independencia ciega a mis alternativas; deliberadamente las formulo como **contrastes experimentales distintos**, no como variantes de «estímulo/ley/implementación».

### Acceso real

Examiné en el commit `4bf1e94e20831ebb9a258ad1c114ca21d0112396`: `files/RESULTADOS.md`, `compare.py`, `CONTRACT.json`, `PERFIL_1_HEXANOL.csv` y las seis fuentes de frontera: `olfactory_synapse_candidate.py`, `orn_peripheral_terminal.py`, `orn_peripheral_terminal_brain.py`, `orn_pn_synaptic_brain.py`, `pn_cns_orn_stages.py` y `protocol45.py`. [Commit fijo examinado](https://github.com/skynet-omega/openmatrix/tree/4bf1e94e20831ebb9a258ad1c114ca21d0112396/reviews/sensory-transfer-post47-20260927-757428eb8a22?utm_source=chatgpt.com)

**No ejecuté `compare.py`, no cargué `OBSERVATIONS.npz`, no abrí `evidence.zip` y no ejecuté una vida neuronal.** Los cocientes 7.366/7.449, 2.528/2.532 y 2.588%/2.503% los considero observaciones del paquete, no reproducción independiente. Sí recalculé directamente desde CSV+CONTRACT: 772 ORN = 323 L + 371 R + 78 unknown; suma completa incremental = **32054.075**; L conocida = **13525.405**, R = **15512.056**.

## Mis A/B/C

| Alternativa | Contraste | Rechazo |
|---|---|---|
| **A — composición vs expansión** | Perfil multiglomerular vs población expandida sin estructura, igualando incremento por antena | Si el patrón estructurado no produce una diferencia post-ORN respecto al control expandido |
| **B — código lateral** | Misma cantidad global y composición, pero contraste L/R antisimétrico y luego espejo L↔R | Si invertir L/R no invierte/cambia coherentemente el observable neural de giro predeclarado |
| **C — feedback causal** | Avance asistido fijo idéntico; sensado online vs replay **sensorial**, nunca replay de órdenes motoras | Si online no supera al replay o una perturbación lateral reservada no afecta según la predicción |

**Escogería A ahora.** No tocaría todavía la ley neuronal ni intentaría «hacer aparecer» DNg100.

### A: contraste ejecutable exacto

La referencia cuantitativa más limpia es el incremento nominal de 45, no la suma completa del nuevo perfil.

Para DM1 a `u=0.5`:

- L: \(35\times83.667\times0.5=\mathbf{1464.1725}\)
- R: \(39\times83.667\times0.5=\mathbf{1631.5065}\)
- total: **3095.679**

Por tanto, para el perfil multicanal conocido:

\[
\alpha_L=1464.1725/13525.405=\mathbf{0.1082535}
\]

\[
\alpha_R=1631.5065/15512.056=\mathbf{0.1051767}
\]

Para cada ORN conocida de receptor \(r\) y lado \(s\):

\[
r_i=b_r+\alpha_s\Delta_r
\]

durante ON; el mismo \(b_r\) debe estar presente en **todos** los brazos.

La igualdad se impone **sólo sobre la suma del incremento nominal de tasa exógena antes de las no linealidades terminales y del puente ORN→PN**. No iguala ni garantiza `q`, conductancia, corriente PN, carga metabólica, concentración química física, salida DN ni conducta.

Usaría tres brazos con exactamente el mismo estado inicial y los 78 `unknown` manteniendo su actividad heredada idéntica:

1. **P-profile:** fórmula anterior.
2. **E-expanded:** las mismas 323L/371R ORN conocidas reciben incremento, pero plano: +4.53304 por ORN L y +4.39759 por ORN R. Misma población activa, mismo basal, mismo total L/R, misma duración.
3. **D-DM1:** mismos basales en toda la población conocida, pero el incremento sólo cae en DM1: +41.8335 por célula DM1.

Así, **P vs E** pregunta si importa el patrón receptor/glomerular; **E vs D** pregunta si importa únicamente expandir población. P vs D por sí solo no sería interpretable porque cambia ambas cosas.

## Obstáculo previo que sí cambia el diseño

La frontera examinada confirma que `orn_peripheral_terminal.py` es específica de las 74 `ORN_DM1`: `9+83.667u`, umbral 40 Hz y modulación 0.3. Esos parámetros están declarados como provisionales. Los otros tipos no tienen en este paquete un contrato verificado **Hz experimental → entrada terminal genérica**.

Por tanto, **no debe clonarse esa ley a los otros 14 tipos**. Antes de ejecutar A sólo falta definir y auditar una interfaz exógena común que introduzca \(b_r+\Delta_r\) sin cambiar ecuaciones intrínsecas/sinápticas. Eso es mucho menor que calibrar 166700 neuronas, pero es necesario para que A no sea un scaffold oculto.

Los otros faltantes no bloquean una criba provisional: los 78 sin lado pueden quedar de fondo; dinámica temporal por receptor, variabilidad entre animales y vehículo puro son necesarios para una afirmación fisiológica fuerte, no para este contraste funcional. El vehículo sin descontar impide además llamar al vector una respuesta «pura» a 1-hexanol.

### Valencia

No usaría **atracción**, **inicio de marcha** ni DNg100 como expectativa biológica de A. De Bruyne 2001 caracteriza respuestas ORN, no demuestra valencia conductual. :chatgpt-content-reference{index="1"} Además, la literatura conductual no da una valencia transferible sin especificar protocolo: un ensayo de locomoción a \(10^{-2}\) reportó respuestas de evitación/neutrales para ese conjunto, mientras un Y-maze posterior encontró atracción a otras concentraciones. :chatgpt-content-reference{index="2"} Esto basta para impedir usar «debería atraer/iniciar» como gate.

## Secuencia de gates

**A pasa** sólo si P y E, después de estar matemáticamente emparejados, siguen siendo distinguibles en PN y en al menos un observable post-PN predeclarado; una diferencia únicamente en ORN no basta.

Después haría **B**: pequeño contraste L/R con suma global conservada y ensayo espejo. Debe aparecer una respuesta lateral neural con signo coherente; no se introduce dirección motora externa.

Sólo entonces **C**: avance asistido fijo y común, yaw exclusivamente neuronal, online contra replay de **sensores**, no de órdenes neurales. Esto es orientación funcional con locomoción asistida, **no marcha autónoma**. Como perturbación reservada, un intercambio L↔R preservando amplitudes debe invertir o destruir específicamente el componente direccional.

La **factibilidad técnica de A es defendible** con cobertura parcial. La probabilidad de superar etapas 4/5 sigue **no identificada**: Post47 no contiene datos que permitan asignarle una probabilidad, y más canales no implican recuperación de DNg100.
