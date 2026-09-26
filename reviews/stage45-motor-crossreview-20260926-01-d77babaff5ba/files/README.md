# Revisión cruzada CPU — 26 de septiembre de 2026

Entrada: RESULTADOS.md. Objetivo: aportar hallazgos verificables y una solicitud útil a «Motor C++/CUDA» para etapas4/5. La revisión usa datos guardados, sin ejecutar ni alterar el cerebro, el cuerpo o la campaña45 activa.

El paquete contiene arrays seleccionados de las vidas estable/revisada2s y exploratoria12s, fuentes congeladas del lector/analizador, contrato de45, hashes/procedencia, reconstrucción exacta, contraejemplos sintéticos y asesorías externas. Es autocontenido para esta revisión CPU. No contiene el estado y dependencias completos para reproducir el organismo.

Python3.10 y NumPy1.26.4 medidos; no requiere CUDA, MuJoCo ni acceso al árbol original. Desde una extracción limpia:

```bash
cd REVISION_CRUZADA_MOTOR_ETAPA45_20260926_01
python -B check_review.py --out verificacion_nueva
```

Modo completo de esta revisión, con reconstrucción de las tres trazas y tres contraejemplos, conservando comprobaciones con optimización:

```bash
python -O -B check_review.py --out verificacion_completa_nueva
```

Ambos modos revisan el mismo alcance finito; no hay modo que simule el CNS. Los destinos deben ser nuevos. RESULTADOS.json permite comparar resultados semánticos excluyendo tiempo/RSS. PLAN.json registra presupuesto previo y exposición; INPUTS.json identifica las fuentes originales y los campos seleccionados, cuyos hashes se comprueban antes de analizar.
