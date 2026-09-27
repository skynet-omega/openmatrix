# Entradas efectivas de campaña48 — diagnóstico07

**Etapas4/5 abiertas.** [Resultado calculado](RESULTADOS.md), [decisión y siguiente captura](DECISION.md), [revisión del motor](aporte_motor/REVISION.md).

La extracción contiene 7227 aristas entrantes a DNg100, DNb05 y DNa02 en cuatro estados finales. Reproduce exactamente los ocho saldos finales DNg100 registrados por el motor. El balance permanece negativo; no se identifica una intervención causal ni se modifica el cerebro. Las referencias científicas y negativos aplicables se conservan en el [plan previo](https://github.com/skynet-omega/openmatrix/tree/71fa2df3d8087334ca0e3df3ab9accf6701bff3e/reviews/plan-post48-20260927-06).

## Reproducir desde una extracción limpia

Se necesita Python3 y NumPy. No requiere el árbol de trabajo, CUDA, checkpoints completos ni acceso a servicios externos.

```bash
cd ATRIBUCION_ENTRADAS48_20260927
# Modo corto: verifica hashes, reconstruye cifras y veredictos desde NPZ.
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 verify_capsule.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -O verify_capsule.py
# Modo completo de este diagnóstico: recalcula tablas y caja de rangos.
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 analyze_endpoint.py
# Corrupciones reales: carpeta nueva en cada ejecución.
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -O verify_capsule.py --corruption-dir comprobacion_nueva
```

No es una reproducción de las cuatro simulaciones48 ni una continuación del CNS. `extract_endpoint.py` documenta cómo volver a extraer desde los originales cuyos hashes constan en EXTRACTION.json; ese modo necesita los archivos grandes originales. `aporte_motor/verificar_esquema.py` también necesita fuentes locales históricas: su resultado y código se incluyen con alcance separado.

La cápsula contiene todos los operandos, código y resultados necesarios para el **diagnóstico de entradas**. No contiene credenciales, datasets completos de artículos ni el estado de 166700 neuronas. Los códigos de referencia muestran el consumidor original, sin importarlo ni ejecutarlo. Las revisiones ChatGPT son conceptuales, sin ejecución de arrays ni modo PRO verificado.

Clasificación del cálculo: **CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA**. No es una admisión biológica, de arquitectura ni de etapas4/5. La ronda se detiene por hito completo con cero pasos neuronales/corporales nuevos; [errores y correcciones](CORRECCIONES.md) preservados.
