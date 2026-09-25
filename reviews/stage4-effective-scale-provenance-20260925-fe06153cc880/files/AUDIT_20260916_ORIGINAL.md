# DNa02: normalización vigente y umbral efectivo

**La normalización de4307 células de `motor_size_brain.py` fue retirada y no está activa en CNS223. Los umbrales actuales ya vienen escalados por una intervención volumétrica global anterior.** Lectura de archivos, sin cargar/avanzar CNS ni modificar parámetros; 16 de septiembre de2026. Todas las rutas parten de `/home/daroch/AXIOMA_FLYWIRE/matrix/`.

## Parámetros realmente conservados

Fuente directa: `work/stage234_settling_extension_20260915/settled_700ms/core_carrier/brain/{state.npz,manifest.json}`. El `session.json` no incluye `motor_size_brain.py` entre sus fuentes ni `motor_size_manifest`. El hash conjunto actual de `gain/theta`, `b7000cdb790f7c77116efdbe9d179f2f917f14a6a2c6d4a40bfd1caf4182aac5`, coincide exactamente con el recibo posterior a la intervención global a0,52 s.

| Variable | DNa02 L523769 | DNa02 R10360 |
|---|---:|---:|
| Volumen, voxeles fuente | 15.678.046.977 | 15.625.110.542 |
| Factor global s=volumen/189.014.792,5 | 82,946138 | 82,666073 |
| `brain.gain` vigente | 0,01116182562 | 0,01355309598 |
| `brain.theta` vigente | 547,997253418 | 641,866027832 |
| `r_max` / `caps`, Hz del modelo | 199,536041260 | 200,266586304 |
| `rate_gain=brain.gain/caps` | 5,5938894794e−5 | 6,7675273402e−5 |

`src/anatomical_morphometry.py:75–96` aplica una sola vez `gain←gain/s`, `theta←theta*s` a166.678 volúmenes observados; preserva22 ausentes. La tabla es `data/anatomical_morphometry/v0.1/canonical_morphometry.parquet`, MaleCNSv1.0, generaciónGCS1780895325252595. Los valores originales aproximados, invirtiendo el escalado, eran gain0,925830/1,120381 y theta6,606664/7,764564: sorteos del prior compartido, no medidas individuales (`src/anatomical_rate_brain.py:120–124`).

## Qué significa entrada positiva con objetivo cero

`src/hybrid_visual_brain.py:69–71` divide gain por `caps` para representar actividad normalizada. El kernel calcula:

`q_obj=max(0,tanh(rate_gain*(sum_j W_ij*caps_j*q_j + drive_i − theta_i)))`

(`src/gpu_visual_brain.py:23–40`). `caps*q` recupera Hz del modelo; **caps no es capacitancia**. La suma ponderada, drive y theta están en unidades arbitrarias del modelo, no pA/mV (`src/anatomical_rate_brain.py:11–12`). Un saldo preumbral+186 sigue dando margen−361,997/−455,866 y objetivo0. No hay contradicción algebraica. El margen FeCO observado incorpora su entrada real en cada evaluación; no debe confundirse con esa cifra ilustrativa.

## Priors publicados, retiro y límite semántico

Ambas DNa02 **sí pertenecen** a las4307 filas del candidato retirado. Sus factores adicionales habrían sido15,699973/15,646963; no deben aplicarse. `src/motor_size_brain.py:63–65` habría escalado nuevamente arrays ya escalados. El retiro está documentado en `config/motor_size_withdrawal_v1.json` y `reports/RETIRO_DOBLE_NORMALIZACION_20260913.md:3–11`.

Fuente primaria local: `data/model_assets_20260910/pugliese_faee4b068698/source/src/utils/sim_utils.py:73–84` normaliza por mediana y modifica ambos parámetros. `src/simulation/vnc_sim.py:1721–1725`, bajo esa misma raíz, usa `size` si existe y sólo alternativamente `surf_area_um2`. **No convierte voxeles a superficie ni aplica exponente2/3.** La tabla frontal4310 tiene mediana998.603.428,5, 5,2832veces la global. Los dos volúmenes DNa02 coinciden exactamente entre tablas; el nombre `vncRoisOnly` no demuestra que esos volúmenes estén recortados al VNC.

La transferencia del denominador frontal al CNS completo está declarada **hipótesis no validada** (`src/anatomical_morphometry.py:3–10,115–119`). Identificar volumen con superficie, resistencia o excitabilidad sería un error semántico; no se encontró esa conversión oculta ni doble escalado vigente. El constructor dice “sin normalización”, pero el checkpoint registra la intervención posterior: manda ese recibo. Esta auditoría justifica distinguir ley/normalización de la referencia externa; **no identifica un umbral correcto ni autoriza reajustarlo para obtener giro**.
