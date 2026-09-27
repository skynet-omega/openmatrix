"""Generate the numerical review from recalculated input controls."""
import json
from pathlib import Path
import PREFLIGHT
import PERMUTATION_CONTROL

HERE = Path(__file__).resolve().parent


def main():
    p = PREFLIGHT.calculate(HERE)
    perm = PERMUTATION_CONTROL.calculate()
    advisor = HERE / 'CHATGPT_ASTRA_V2_RESPONSE.md'
    advisory = ('La respuesta de ChatGPT ASTRA_V2 se conserva en [el original](CHATGPT_ASTRA_V2_RESPONSE.md). '
                'La comparación y las decisiones adoptadas están en [CONTRASTE.md](CONTRASTE.md).'
                if advisor.exists() else
                'La consulta a ChatGPT ASTRA_V2 está enviada y todavía no se recibió una respuesta. '
                'No se atribuye a ChatGPT la propuesta propia siguiente.')
    rows = []
    for side in ('L', 'R'):
        v, w = p['sides'][side], perm['sides'][side]
        rows.append(f"| {side} | {v['known_cells']} | {v['target_weighted_increment']:.4f} | "
                    f"{v['empirical_scale']:.9f} | {w['changed_assignments']} |")
    text = f'''# Revisión y siguiente contraste de etapas 4/5 — 27-09-2026

**El análisis previo es reproducible y acota una hipótesis sensorial; no demuestra mejora de navegación.**
La propuesta de entrada queda PROMETEDOR_NO_CONFIRMADO, sin integración en el cerebro.
Las etapas 4/5 siguen abiertas. No se ha medido una probabilidad de superarlas con este modelo.

## Qué se comprobó y qué cambia

Se ejecutó otra vez `compare.py --verify` del paquete anterior: coinciden sus resultados completos.
La ventana2501–3000ms conserva terminal ORN7,366/7,449×, puente2,528/2,532× y salidas localesPN10208
aproximadamente+2,5%. Son observables distintos; el cociente entre ellos no identifica un fallo fisiológico.
El CSV conserva15canales de una sola columna experimental, sin convertir45 retrospectivamente en1-hexanol.

La suma de incrementos nominales del perfil completo es **{p['full_profile_weighted_increment']:.3f}**, frente a
**{p['selective45_weighted_increment']:.3f}** enDM1 al nivel45: **{p['unnormalized_increment_ratio']:.6f}×**.
Una comparación sin emparejar esa cantidad no aislaría composición.
De{p['mapped_neurons']}ORN mapeadas por tipo, **{p['unknown_side_neurons']}** tienen `rootSide=unknown`.
Se comprobó además que esas78carecen de `somaSide` y `somaLocation`; `entryNerve=AN` no da lado.
[Selección anatómica y hash del original](ANATOMIA_SELECCIONADA.json).
Esto limita una entrada espacial por antena; no impide una exposición uniforme ni obliga a borrar esas células.

La selectividadDM1 no es por sí sola un error. [Gaudry2013](https://pubmed.ncbi.nlm.nih.gov/23263180/)
estudia lateralización bajo estimulación olfativa; el antecedente local ya contrastó una entrada breve distinta
y su negativo se conserva. [Tao2023](https://www.nature.com/articles/s41467-023-42613-8)
muestra dependencia de combinación e historia de las ORN. Ninguna de esas observaciones demuestra que
añadir14canales a esta preparación inicie marcha o que1-hexanol deba atraerla.

## Control construido, todavía fuera del organismo

Se construyeron entradas nominales por receptor y antena con aritmética racional exacta:
DM1selectivo, perfil multiglomerular escalado, reparto uniforme y un sham con basal común.
La preparación anterior45 no sirve automáticamente de sham para un nuevo basal multiglomerular.
El patrón completo no se concentra enDM1a una dosis fuera de su ancla: se reduce el patrón distribuido
hasta la suma incremental de referencia por antena. Esto no identifica una dilución química.

| Antena | ORN con lado conocido | Suma incremental emparejada | Escala del perfil | Asignaciones cambiadas por el control permutado |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Para distinguir la asignación del patrón de la mera dispersión/heterogeneidad, el control preferido
es una **permutación por célula dentro de cada antena**, fijada antes de observar respuestas nuevas.
Conserva exactamente la suma y el histograma de incrementos, las mismas694células elegibles,
el basal de cada célula y la forma temporal aplicada. Cambia qué célula recibe cada incremento.
Es un control sintético; no se presenta como otro olor biológico. Una sola permutación es exploratoria.
El reparto uniforme queda calculado como referencia de diseño, sin obligar a ejecutar otro brazo.

Las78ORN sin lado conservan sus entradas de fondo y conexiones, sin un nuevo estímulo periférico
directo en esta versión parcial. Su actividad recurrente puede cambiar libremente durante la evolución;
no se congela un estado neural para igualar los brazos.

**La igualdad se exige antes de la modulación terminal**, en la suma de tasa nominal adicional por antena.
No iguala corrientes sinápticas, filtros, saturación, recurrencia ni actividad dePN.
Las cantidades realmente consumidas deben registrarse al integrar la interfaz.
El puerto especial actual está restringido aDM1; sus constantes0,3/40Hz no se copian a otros tipos
por pertenecer al mismo perfil. Un cambio de interfaz o ley tendría que declararse y ser común
a los brazos; la tabla preparada no constituye por sí sola ese adaptador.

## Propuesta propia y decisión

Se registraron tres alternativas antes de recibir esta nueva revisión externa en [PLAN.json](PLAN.json):
A/composición y contexto sensorial, B/ley de transferencia contradicha por datos independientes,
C/contradicción concreta de implementación/interfaz. El padre y el sham son controles, no hipótesis.

Priorizar una criba funcional deA, sin exigir identificar todas las166.700neuronas. El primer diseño
compara cuatro condiciones con preparado y basal comunes: sham, DM1selectivo emparejado,
perfil distribuido emparejado y su permutación. El objeto del contraste es la representación sensorial
y su efecto descendente, no una calibración fisiológica ni la admisión4/5.
Si el perfil y su control no se distinguen materialmente en las salidas preregistradas, no se afirma
ventaja de la composición ni se elige otro umbral, dosis o permutación para obtenerla.
Si ambos mejoran respecto deDM1, el dato favorece distribución/reclutamiento sobre especificidad del perfil.
Si existe una contradicción de implementación, se conserva y repara la operación concreta.
Una diferenciaPN sin una consecuencia pertinente para el mando no basta para declarar progreso motor.

Antes de integrar: cerrar la traducción tasa periférica→puerto terminal y su propiedad de estado,
congelar fuente/lector/ventana, observables descendentes, materialidad, basales y presupuesto propio.
No hay contrato neuronal48ni una nueva vida lanzada por esta revisión.

## Factibilidad y prueba de las etapas

Hay infraestructura para hacer estos contrastes y el cuerpo actual ya responde a órdenes de giro
en controles físicos. Esto apoya la factibilidad experimental; no prueba que el núcleo actual genere
la política necesaria.47observóDNg100nulo y46no alcanzó materialidad lateral. Con esos datos no es
honesto prometer superación, un porcentaje de éxito o un plazo.

Etapa4 requiere una ventaja de orientación/aproximación debida al feedback online frente a replay
o perturbación sensorial equivalente, conservando fuente y soporte. Primero debe haber una orden
direccional útil y verificable, con lectura y efectos separados. Si se estudia orientación con avance
asistido fijo, éste ha de ser idéntico y declarado en los controles; no se denomina marcha autónoma.
La iniciación espontánea porDNg100no es una puerta universal para investigar ese alcance funcional.

Etapa5 requiere después recuperación ante una perturbación reservada durante segundos, mejor que
sus controles emparejados. Se conservan negativos40/41/42. No hace falta integrar seis patas para
formular esa prueba, ni una mejor entrada sensorial la supera automáticamente.

## Asesoría, reproducción y límites

{advisory}
El usuario redirigió expresamente la consulta a este chat mediante su enlace compartido.
Las consultas anteriores devolvieron errores. El enlace identifica un modeloThinking; no se atribuye
PROverificado ni se pretende cambiar el selector mediante texto.
No se usaron subagentesCodex. La clasificaciónJev del análisis previo se conserva como antecedente,
sin pedirle que decida corrección científica ni factibilidad.

Este paquete sólo reproduce el diseño aritmético y sus controles:

```bash
python3 PREFLIGHT.py --verify
python3 PERMUTATION_CONTROL.py --verify
python3 -O VERIFY.py
```

Se rechazan corrupción de fuente y un cambio de población incluso después de recalcular su hash.
Fuentes de entrada, anatomía seleccionada y resultados están incluidos; no se requiereNumPy/GPU/CNS.
El paquete no reproduce las vidas45/47ni su paridad. El análisis sensorial anterior tiene su propio
[paquete público](https://github.com/skynet-omega/openmatrix/tree/4bf1e94e20831ebb9a258ad1c114ca21d0112396/reviews/sensory-transfer-post47-20260927-757428eb8a22).
Resultado de esta revisión: controles preparados y una siguiente decisión delimitada, con cero pasos nuevos.
'''
    (HERE / 'RESULTADOS.md').write_text(text)


if __name__ == '__main__':
    main()
