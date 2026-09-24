# Contraste de fuentes espejo — cierre del 24-09-2026

**Resultado: BLOQUEADO para admisión de etapa 4.** Los dos brazos nativos completaron 40 ms sin olor + 400 ms con fuentes espaciales opuestas, desde estados preparados semánticamente exactos. El efecto lateral es material en la entrada, el mando DNb05 y el giro corporal, pero la primera referencia numérica se detuvo por el tiempo máximo preregistrado en 391/400 ms. La segunda referencia no se lanzó, porque no podía reparar la ausencia de la primera. Etapas 4 y 5 siguen abiertas. [Veredicto mecánico](CLOSE_01.json), [plan previo](PLAN.json), [fuentes congeladas](SOURCE_LOCK.json).

| Medida del par nativo | Observación | Umbral previo |
| --- | ---: | ---: |
| Contraste L−R tardío, fuente izquierda/derecha | +0,385420 / −0,385562 | >+0,25 / <−0,25 |
| Diferencia media absoluta de campo, 300–400 ms | 0,385491 | >0,25 |
| Diferencia L1 del mando angular integrado | 0,057747° | >0,01° |
| Diferencia firmada del mando integrado | 0,057747° | >0,01° |
| Diferencia final de giro corporal | 0,057321° | >0,005° |

Estos valores se reconstruyeron desde [trazas y testigos crudos](RAW_VERIFIED_01.json), no de banderas del runner. El [comparador](PREPARED_COMPARE_causal_cuda.json) encontró estados preparados exactos. La [figura](PAIR_DIAGNOSTIC.png) muestra las señales muestreadas, sin sustituir el veredicto. La [serie CSV](SCALAR_SERIES_01.csv) conserva cada milisegundo escalar para revisión externa; [manifiesto](SCALAR_SERIES_MANIFEST_01.json).

La referencia izquierda preservó un prefijo íntegro de 391 ms y alcanzó 2041,512 s de pared dentro del máximo 2050 s; se detuvo antes de aceptar el paso 392 por la guarda operativa. En ese prefijo, el giro nativo/referencia difiere como máximo 3,42×10⁻⁷° y el mando integrado L1 5,41×10⁻⁷° ([recibo](PREFIX_PARITY_01.json)). **Esto no acota los 9 ms faltantes ni confirma el contrato de 400 ms.** La [relectura cruda final](RAW_VERIFIED_02.json) marca bloqueo por ejecución incompleta; la salida 2 de su CLI es la esperada. Un arranque anterior con Python del sistema carecía de CuPy, terminó a 0 ms y quedó conservado en `preflight_failures` con [recibo](PREFLIGHT_FAILURE_01.json).

Ambos cuerpos se acercaron a sus respectivas fuentes unos 0,075 mm durante el ensayo, pero la diferencia de acercamiento entre brazos fue apenas 0,000649 mm ([cálculo](TRAJECTORY_DESCRIPTIVE_01.json)). Ese avance común no demuestra navegación ni utilidad del feedback. La media de olor se emparejó **sólo en la pose inicial**; las dos historias posteriores son geometrías diferentes. El cuerpo usa un efector azimutal asistido por rodillos, y sólo se evaluó una preparación por lado. No hay equivalencia animal ni película temporal del cerebro completo.

ChatGPT revisó código y geometría **antes** de las corridas y reprodujo un defecto de longitud de trazas capaz de alterar la decisión; se corrigió con 11 pruebas normales y `-O` aprobadas antes del primer milisegundo ([revisión](CHATGPT_REVIEW_01.md)). No ejecutó CUDA/MuJoCo ni revisó el cierre. Jev hizo una sola clasificación acotada de tareas en la campaña de diseño previa; no avala la hipótesis biológica.

Relectura local, desde la raíz del proyecto y con el entorno GPU existente:

```bash
/home/daroch/miniconda3/envs/GPU/bin/python -m unittest campanas/etapa4_mirrored_source_20260924_26/test_verify_mirror.py -q
/home/daroch/miniconda3/envs/GPU/bin/python campanas/etapa4_mirrored_source_20260924_26/verify_mirror.py --campaign campanas/etapa4_mirrored_source_20260924_26
```

El segundo comando comprueba también el intento incompleto y por ello devuelve 2 y `BLOQUEADO`. Los checkpoints, assets y dependencias transitorias del organismo permanecen en el árbol local; el paquete público compacto contiene sólo datos seleccionados y **no permite reejecutar el organismo desde una extracción limpia**. No se alteraron ecuaciones, fuente, horizonte, lector ni umbrales tras observar resultados. El presupuesto se agotó en esta campaña. Para la siguiente decisión, [tres rutas de etapa 5](../etapa4_next_20260924_25/STAGE5_ABC_DRAFT.md) y [limitaciones de la propuesta de publicación](../etapa4_next_20260924_25/GEMINI_ETAPA5_REVISION_01.md) quedan como opciones, no como etapa superada.
