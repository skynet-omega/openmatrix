# Cierre verificable de la campaña 52

**Comparación terminada; etapas 4/5 abiertas.** La corrección física y el registro ampliado quedaron cualificados. La señal del aire conserva el contraste previo, pero no prueba orientación; las modificaciones G/I no rescataron selectivamente la respuesta al olor. [Resultados](RESULTADOS.md), [gráfica](COMPARACION.png), [decisión A/B/C](DECISION_FINAL.md).

Se completaron diez condiciones de 90 ms y 16 ms de cualificación: 916 ms CNS, 2909,568 s CPU y 2601,541 s de pared de cola. Se observan 1314 DN entre 2757 células anatómicas, más 694 ORN separadas. No se aplicó giro ni perturbación mecánica. La población que cambia no equivale a una ruta causal identificada. No se ejecutó el control posterior de cuatro brazos; requiere un contrato nuevo que compruebe el total JO después de FP32.

## Entrega compacta pública

Archivo reconstruido: `ETAPA45_REPARACION_20260927_52.zip`, **128152934 bytes**, SHA256 `f93f3020a13045a2f766e251e32a1b44e3811f9bbdd15791c57782972738d216`. Se publica en dos partes binarias por el límite de tamaño de GitHub. Las partes son fragmentos del mismo ZIP, no dos ZIP independientes.

```bash
cat ETAPA45_REPARACION_20260927_52.zip.part01 ETAPA45_REPARACION_20260927_52.zip.part02 > ETAPA45_REPARACION_20260927_52.zip
sha256sum ETAPA45_REPARACION_20260927_52.zip
python -m zipfile -e ETAPA45_REPARACION_20260927_52.zip extraccion52_nueva
cd extraccion52_nueva/ETAPA45_REPARACION_20260927_52
python -B -O verify_complete52.py
python -B -O aporte_motor52/verify52.py science --out "$PWD/aporte_motor52/RECOMPUTACION_NUEVA.json"
```

Descargado desde el commit `e4853f731325761f4bccc4b3f7858980701bba28`, idéntico por hash, y recalculado en extracción limpia por el verificador canónico y el independiente de Motor C++/CUDA. Se rechazaron diez corrupciones deliberadas bajo `-O`. Los recibos están en [REMOTE_DELIVERY.json](REMOTE_DELIVERY.json) y `cierre/RECOMPUTE_REMOTO.json`. Las copias de extracción se retiraron después de comprobar identidad; el ZIP original y todos los datos científicos se conservan.

## Cápsula completa local

`ETAPA45_REPARACION_20260927_52_COMPLETO.zip`: **1727101237 bytes**, SHA256 `748b5156cae27eff658181be2f94b29d2b1068a6a4419deb266a820817895872`. Incluye 1343 rutas con los diez estados finales, siete registros de cualificación, preparación48 y dependencias locales explícitas. Verificados 39288 bloques y 234 archivos segmentados reconstruidos byte por byte, incluidos 211 NPZ. El programa de extracción se ejercitó además materializando el informe en otra carpeta; no se duplicaron todos los gigabytes descomprimidos.

La cápsula completa se entrega localmente en la carpeta de intercambio autorizada; no está subida a OpenMatrix. Allí se publican el compacto reproducible y los recibos/fuentes del ensamblado. [Modo de extracción completo](REPRODUCIR.md). **No se cualificó reanudación GPU portable52 ni se repitió el CNS desde el ZIP.**

## Presupuesto y continuidad

Los dos ensamblados interrumpidos por reservas de espacio se conservan como fallos de entrega, no como fallos científicos. Sólo se retiraron copias de transporte verificadas; [procedencia](COMPRESION_Y_ENTREGA.md). Contabilidad conservadora de CPU: 4847.848/5000 s, incluyendo los topes completos de los empaquetados fallidos y reservas auxiliares/publicación. Espacio contabilizado con reserva de cierre: 8144291808/8589934592 bytes. [CPU](CPU_CLOSURE.json), [almacenamiento](STORAGE_CLOSURE.json).

ChatGPT ASTRA_V2 y ChatGPT_Motor_V2 revisaron la interpretación; Motor C++/CUDA recalculó datos. No nuevos subagentes Codex, PRO no verificado. Clasificación: reparación/instrumentación **CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA**; pista sensorial **PROMETEDOR_NO_CONFIRMADO**. Motivo de parada: hito acotado terminado y exposición916ms consumida. A: configuración frente a cantidad; B: transmisión/estado con control no-op; C: contexto propioceptivo con marginales emparejadas. Orientación con cuerpo actual sigue prioritaria; integración completa de seis patas permanece posterior.
