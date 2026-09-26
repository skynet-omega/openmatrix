"""Generate review measurements from the reconstructed CPU result."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
r = json.loads((HERE/'run_01/RESULTADOS.json').read_text())
p = r['pair2s_relay_boundary'][0]
d = r['exploratory12s']
j = json.loads((HERE/'jev_01/response.json').read_text())['answers']
text = f'''# Revisión cruzada de motor y etapas 4/5

Se reconstruyó exactamente DN→lectura→filtro→relé sobre los datos guardados de las dos vidas de 2 s y la exploración de 12 s. Cero pasos nuevos del cerebro o del cuerpo; no se modificó campaña45 ni su cola activa. El alcance confirmado localmente es este análisis CPU y los contraejemplos del verificador. No se admite una etapa ni se confirma una arquitectura.

En intervalo **{p['interval_ms']}**, el filtrado estable es **{p['stable_filter_rad_s']:.16g} rad/s** y el revisado **{p['reviewed_filter_rad_s']:.16g} rad/s**. El umbral congelado es **{p['threshold_rad_s']} rad/s**. Una diferencia de **{p['filter_difference_rad_s']:.16g} rad/s** determina mando **{p['stable_applied_deg_s']:g} frente a {p['reviewed_applied_deg_s']:g}°/s**. El FAIL histórico permanece: identificar esta frontera no justifica cambiar sus tolerancias ni demuestra que el integrador cause el mal rumbo.

En los {d['ticks']} intervalos de la vida exploratoria, DNg100 L/R permanece en {d['DNg100_release_initial']}, con rango de cambio {d['DNg100_delta_range']}; son valores subnormales, no prueba de inactividad biológica. Objetivo de avance único: {d['forward_target_unique_mm_s']} mm/s. Entre 6 y 12 s, yaw crudo medio **{d['raw_yaw_mean_6_12_deg_s']:.9f}°/s**, aplicado único **{d['applied_yaw_unique_6_12_deg_s']}°/s**. Hay {d['applied_nonzero_sign_disagrees_raw_count']} intervalos con mando no nulo y signo opuesto al crudo; la memoria temporal de 200 ms puede producirlo y esto por sí solo no constituye un bug. La magnitud del mando y la dinámica del relé requieren separarse del aporte neural al interpretar orientación.

El analizador congelado de45 acepta tres contraejemplos sintéticos:

1. Su criterio físico da True con desplazamiento y velocidad distintos del sham, aunque DNg100 y los mandos de ambos brazos sean idénticos y nulos. El criterio físico, por sí solo, no acredita propulsión por esa ruta. No se afirma que este caso sea realizable en la mecánica del runner.
2. Informa primer mando diferencial en1001 aun cuando no corresponde a DN_q_usada. El runner sí vigila el contrato en vivo; el analizador aislado no lo reconstruye. Bajo el lag declarado, primera DN diferencial publicada1001 sólo puede influir en mando1002.
3. Acepta PLAN modificado después de adquirir datos y cambia el veredicto. El suplemento rechaza el hash de criterio alterado; no propone otro umbral.

Estos son fallos de independencia/procedencia de la reconstrucción y una limitación de atribución del criterio físico, **no evidencia de que los datos reales de45 estén corruptos**. La prueba sintética y el código congelado están incluidos. El suplemento lector comprueba baseline fijo, lag de DN, decodificación, agenda, finitos y primer intervalo permitido; no modela ni verifica por sí solo fuerzas de la prótesis.

Tres explicaciones activas, con discriminadores:

| Alternativa | Dato que decide | Resultado que la refuta en ese alcance |
|---|---|---|
| A: falta de reclutamiento sensorial/descendente | Olor consumido→ORN/PN→DNg100/DNb05 frente al sham, por fase | DNg100 se modula y su efecto llega al lector |
| B: pérdida/amplificación en lector y relé | DN usada→crudo→filtrado→aplicado, sin recalibrar | Magnitud/signo útil conservados y sin efecto de frontera pertinente |
| C: apoyo/freno/feedback físico | Mando→fuerza/contactos→velocidad/desplazamiento | Respuesta física explicada por el mando sin conflicto de fuerzas |

Solicitud concreta a «Motor C++/CUDA»: conservar45 y terminarla dentro de su presupuesto; añadir después un informe CPU separado que contraste los hashes de PLAN/fuentes registrados, reconstruya la cadena del lector desde datos de base y reporte las diferencias ORN/PN/DNg100/DNb05/crudo/aplicado/cuerpo por fase. Extraer identidad/propietario y entradas/objetivo de DNg100 de estados guardados si el canal no se recluta; no cambiar sus parámetros para obtener respuesta. El pulso bilateral uniforme de45 estudia iniciación/modulación, no dirección o distancia. Para orientación sigue pendiente el contraste diferencial congelado de43, con presupuesto nuevo prospectivo medido en el motor escogido, preservando el bloqueo de44 y el FAIL de12. No repetir41/42 como pruebas nuevas ni integrar patas/plasticidad ahora.

Jev realizó una petición de clasificación: primero **{j['first_handoff']['choice']}** (confianza declarada {j['first_handoff']['confidence']}); siguiente discriminador **{j['next_discriminator']['choice']}** ({j['next_discriminator']['confidence']}). Es prioridad sugerida, no verificación científica. Nuestra reconstrucción CPU respalda el diagnóstico del relé; la debilidad del analizador se apoya en pruebas, no en la clasificación de Jev.

ChatGPT Matrix recibió una revisión paralela del análisis y la conexión con etapas4/5. Consultar CHATGPT_REVIEW.md/CONTRASTE.md para respuesta y límites; no se atribuye ejecución de arrays a una lectura del texto. El selector PRO/máximo razonamiento no se comprobó desde esta herramienta. No se usaron subagentes Codex.

Verificación original: {r['CPU_s']:.6f}s CPU, {r['wall_s']:.6f}s pared, RSS {r['peak_RSS_bytes']/1024**2:.3f} MiB. Las cifras son del analizador CPU, no del motor neuronal. Fuentes/arrays seleccionados en INPUTS.json; no se incluyen estados completos del cerebro ni se reivindica reproducción de la vida neuronal.
'''
(HERE/'RESULTADOS.md').write_text(text)
print('RESULTADOS.md generated from CPU result and advisory receipt')
