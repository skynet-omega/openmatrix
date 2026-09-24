# Referencias estrictas de fuentes especulares — cierre del 24-09-2026

**El contraste fuente lateral → mando y giro queda confirmado localmente, pendiente de auditoría externa. La navegación de etapa 4 y la etapa 5 siguen abiertas.** La campaña anterior terminó bloqueada al llegar a 391/400 ms por su límite de tiempo; ese veredicto no se cambió. Esta campaña registró **antes** de ejecutar un nuevo presupuesto máximo de 2300 s por referencia y 4600 s agregados. Las fuentes, ecuaciones, intervención PN629-off, preparación de 40 ms, ensayo de 400 ms y umbrales científicos son los mismos.

Las referencias izquierda y derecha terminaron 40+400 ms, sin error ni fallo de guardado, en 2049,212 s y 2084,231 s. Los estados preparados se compararon por estructura y contenido semántico exactos. El [verificador compuesto](RAW_VERIFIED_01.json) leyó los dos brazos nativos expuestos de la [campaña26](../etapa4_mirrored_source_20260924_26/CLOSE_01.json) y las dos referencias nuevas desde archivos crudos; `python -O` produjo un [recibo](RAW_VERIFIED_02.json) byte por byte idéntico. Dos corrupciones deliberadas del umbral y del hash de una traza fueron rechazadas antes de ejecutar el organismo ([recibo](CORRUPTION_PREFLIGHT_01.json)).

| Medida | Nativo | Referencia estricta |
| --- | ---: | ---: |
| Diferencia integrada de mando entre fuentes | 0,057746767° | 0,057746766° |
| Diferencia final de giro corporal | 0,057320516° | 0,057320515° |
| Error máximo de giro nativo–referencia izquierda/derecha | — | 3,42×10⁻⁷° / 3,38×10⁻⁷° |
| Diferencia integrada de mando nativo–referencia izquierda/derecha | — | 5,58×10⁻⁷° / 5,48×10⁻⁷° |

Los cuatro controles prospectivos de entrada y salida pasaron en ambos perfiles. El [cierre mecánico](CLOSE_01.json) clasifica sólo el efecto numérico fuente→mando como `CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA`. El límite de error permitido era 0,002° por lado; no fue relajado. Los recibos de [lanzamiento](LAUNCH_PLUS_01.json) y [lanzamiento opuesto](LAUNCH_MINUS_01.json) fijan contrato, fuentes y comprobación secuencial. El consumo nuevo total fue 4133,443/4600 s.

Para revisión externa, la [serie escalar completa](REFERENCE_SCALAR_SERIES_01.csv) enumera los 800 ms de ensayo de las dos referencias, extraídos directamente de sus NPZ ([hashes](REFERENCE_SCALAR_SERIES_MANIFEST_01.json)). El CSV es descriptivo; el verificador opera sobre los archivos crudos.

**Esto todavía no demuestra aproximación dirigida por feedback.** El avance corporal fue de unos 0,075 mm por brazo con mando de avance constante 0,2 mm/s; la diferencia de acercamiento a las fuentes fue sólo 0,000649 mm en las corridas nativas. En los últimos 100 ms, incluso el mando del brazo con fuente izquierda integró −0,002624°. No hubo perturbación online contra reproducción sensorial ni comparación compatible con moscas vivas; el cuerpo sigue usando un efector azimutal asistido por rodillos. Una animación sería descriptiva. La próxima decisión requiere un contrato nuevo de navegación causal y un motor suficientemente rápido para el horizonte necesario; no hay corrida de etapa 5.

Revisión local desde la raíz, con el entorno GPU del proyecto (el verificador sólo lee archivos y no carga el organismo):

```bash
/home/daroch/miniconda3/envs/GPU/bin/python campanas/etapa4_reference_budget_20260924_27/verify_ref27.py
/home/daroch/miniconda3/envs/GPU/bin/python -O campanas/etapa4_reference_budget_20260924_27/verify_ref27.py
```

Los comandos reconstruyen la decisión, pero no reproducen las corridas de ~35 minutos ni sirven como auditoría externa. [Plan y tres alternativas](PLAN.json), [fuentes congeladas](SOURCE_LOCK.json), [comparación de preparaciones](PREPARED_COMPARE_reference_cuda.json) y [hoja de ruta causal](../../HOJA_DE_RUTA_CAUSAL.md).
