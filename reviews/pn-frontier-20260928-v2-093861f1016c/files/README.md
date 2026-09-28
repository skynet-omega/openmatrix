# Frontera PN: reproducción CPU de evidencia causal parcial

Se reproduce la lectura de los ocho brazos adquiridos (558 ms CNS originales). El paquete incluye arrays completos de esos brazos, referencias que consume el lector, contratos congelados, código de lectura, pruebas, informe y revisión independiente. Incluye fuentes de adquisición para inspección, pero no todos los datos/checkpoints y dependencias que exigiría volver a simular el CNS.

**Alcance: efecto parcial en PN→CSR; mando de avance cero; etapas 4/5 abiertas.** No es una confirmación con organismos nuevos ni una validación fisiológica.

[Informe y figura](work/pn_frontier_causal_20260928/README.md) · [Decisión](work/pn_frontier_causal_20260928/DECISION.md) · [Revisión independiente](work/pn_frontier_causal_20260928/aporte_motor/REVISION_FINAL.md).

Versión 2: corrige la omisión de hilos BLAS en la CLI. Conserva los mismos datos y tolerancias. Reproducir con el entorno registrado (Python3.10, NumPy1.26.4); la CLI fija OPENBLAS/OMP a un hilo antes de cargar NumPy.

En una extracción nueva:

```bash
python -I -O src/pn_frontier_readback.py INPUTS.json reproduced
```

El directorio de salida debe ser nuevo. El lector comprueba identidad del contrato y referencias, fase del lector descendente, consumo PN, soporte/dominio, control de cantidad y criterios; reconstruye las cifras desde los arrays declarados. No ejecuta el cerebro. Para las cuatro pruebas pequeñas de corrupción, con pytest disponible:

```bash
PYTHONPATH=src python -O -m pytest -q tests/test_pn_frontier_readback.py
```

Antes de ejecutar, validar todos los archivos contra MANIFEST.json (SHA256 y tamaño). El publicador y verify_remote.py incluidos documentan el formato; la verificación de descarga no ejecuta código. Los recibos históricos contienen rutas canónicas locales como procedencia, no como dependencia de esta lectura: INPUTS.json declara todas sus rutas relativas al paquete.

El código de adquisición conserva su contexto histórico. La presencia de una fuente o checkpoint mencionado en un manifiesto no equivale a una reproducción CNS completa. Las revisiones ChatGPT fueron conceptuales, sin acceso nuevo a estos arrays; Motor verificó arrays localmente. No hay modo PRO verificado. El lote Jev no discriminó las opciones.
