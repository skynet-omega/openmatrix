**Priorizaría A como cambio algorítmico y C como comparación de ejecución; mantendría B en el sondeo lineal hasta medir su coste completo.** La formulación de A es correcta, pero la implementación publicada todavía no ejecuta el Newton reducido anunciado.

**Leí completos** el índice, `PLAN.md`, `STRUCTURE.json`, `CONTRACT.json`, `condensation.py`, `portable.py`, `linear_screen.py`, `BUG_01.md`, `PORTABLE_REPLAY.json`, `base_refinement/RESULT.json` y los archivos `vendor/pn_mass_coupled_step.py`, `vendor/pn_graph_elimination_backend.py` y `vendor/pn_mass_backend.py`. Consulté además la cabecera y `stage_channels` de `vendor/pn_coupled_ionic.py`. **No ejecuté código ni reproduje la captura binaria.**

## 1. A: ecuación correcta, con una condición indispensable

El cambio de variable es correcto:

\[
A_0\delta+I(v_b+\delta)=b,\qquad
A_0=G+sM,\quad b=current-Gv_b.
\]

No falta un término \(sMv_b\): desaparece al sustituir \(\delta=v-v_b\). Esto coincide algebraicamente con el residuo del integrador original. 

Separando coordenadas eliminadas \(P\) y retenidas \(R\), **si \(I_P=0\)**:

\[
\delta_P=A_{PP}^{-1}(b_P-A_{PR}\delta_R),
\]

\[
S\delta_R+I_R(v_b+\delta)=
b_R-A_{RP}A_{PP}^{-1}b_P.
\]

Es eliminación algebraica **de cada etapa**, conservando su masa; no aproximación cuasiestática.

### Carencia concreta: se ignoran correcciones fuera del núcleo

`Condensation.solve()` utiliza únicamente `diag[self.core]`. Si existe una corrección diagonal en una coordenada eliminada, devuelve la solución de **otro sistema**, sin advertirlo. `linear_screen.py` protege los soportes de canales, calcio y sinapsis de `case_00`, pero la clase no impone esa precondición. Debe comprobarse en cada soporte nuevo, junto con identidad de matrices y `shift`. **No afirmo que la captura actual viole esa condición.**  

### La primitiva actual no implementa todavía el ahorro principal de A

Cada llamada a `solve()` vuelve a reducir el RHS completo y reconstruye todas las coordenadas. Eso prueba una factorización reutilizable, **no “condensar una vez por etapa y ejecutar Newton exclusivamente en el núcleo”**. El coste de esas operaciones seguirá repitiéndose mientras se conecte como sustituto directo del solver lineal. 

## 2. Residuo y backtracking: la sutileza que no debe perderse

El `guess` original puede incumplir inicialmente las ecuaciones pasivas. Reconstruirlo desde el núcleo lo proyecta sobre otra trayectoria de iterados. **La raíz buscada sigue siendo la misma, pero el Newton y su backtracking ya no son idénticos.**

Para una corrección Newton completa amortiguada por \(\alpha\), el residuo pasivo satisface algebraicamente:

\[
r_P^{nuevo}=(1-\alpha)r_P^{anterior}.
\]

Reconstruir imponiendo \(r_P=0\) después de cada movimiento reducido no reproduce esa relación cuando \(\alpha<1\). Por tanto, no identificar automáticamente norma reducida y norma original.

El umbral debe seguir calculándose con el residuo original: primera etapa en \(v^n\), segunda en \(v_1\), incluso con predictor lineal. La aceptación final debe usar `mass_action` y `action` originales, no solamente el Schur ensamblado: `action` utiliza diferencias de voltaje y suma compensada. También deben conservarse el criterio de gates y la recuperación ante pivotes/residuos lineales fallidos.   

## 3. Mayor salto dentro de A: condensar también sinapsis lineales

**Los nodos exclusivamente sinápticos no tienen por qué permanecer como incógnitas no lineales.** Sus conductancias están prefijadas durante cada etapa; el código calcula \(g_s(v+E_{leak}-E_s)\). Pueden incorporarse exactamente al operador de esa etapa. 

Con \(\bar E_s=E_s-E_{leak}\):

\[
A_s=G+sM+D_s,
\]

\[
A_s\delta+I_{\mathrm{Na/K,Ca}}(v_b+\delta)
=current-Gv_b-D_s(v_b-\bar E_s).
\]

Así, el soporte no lineal podría limitarse a la unión Na/K–Ca —**como máximo 7.791 nodos según los recuentos del plan**, antes de considerar solapamientos—, más las uniones necesarias. Las salidas en nodos eliminados se reconstruyen; no desaparecen. 

**El coste contrario es importante:** los factores numéricos dependerían también de \(D_s\), que cambia entre etapas. Se conserva el plan simbólico, pero no necesariamente la factorización numérica. Esta variante solo merece conservarse si reducir Newton compensa esa refactorización; no la presento como mejora demostrada.

## 4. Un falsador específico por ruta

| Ruta | Falsador |
|---|---|
| **A** | La solución reconstruida satisface el reducido, pero incumple el residuo completo original o cualquiera de los límites de estado del contrato. No aceptar un residuo reducido como sustituto. |
| **B** | La eliminación GPU, incluyendo planificación, sincronización, reconstrucción y transferencias de su frontera real, no alcanza el ahorro declarado. Los 451 niveles describen ese orden; no refutan otro. Cuidado: dos pivotes no adyacentes pueden escribir sobre un mismo vecino. |
| **C** | Una segunda etapa rechazada deja modificados gates, calcio, cargas o reloj. Compilar no autoriza avanzar historia durante evaluaciones Newton: el original confirma los estados al completar ambas etapas. |

Los límites numéricos y la exigencia material de **2×** permanecen como condiciones comunes; un benchmark del solve aislado no acredita ese factor para el paso PN completo.   

**La referencia de un segundo no amplía esos contratos:** `RESULT.json` evalúa el bloque base en 106 muestras. Su refinamiento queda aproximadamente un **1,42 % por debajo del límite**, no constituye una cota certificada del error verdadero ni valida PN u organismo completo.
