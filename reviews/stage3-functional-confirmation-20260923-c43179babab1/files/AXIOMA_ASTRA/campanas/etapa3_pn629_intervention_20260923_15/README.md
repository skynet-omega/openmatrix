# Intervención causal PN629

Esta campaña compara una modificación del adaptador del PN izquierdo contra el padre conservado. Desactiva `general_outputs.enabled` antes de la preparación de40ms y conserva las466 salidas dinámicas, conectividad anatómica, parámetros celulares y lector motor. **No es una homologación completa de ambos PN ni una calibración biológica.**

El [plan congelado](PLAN.json) distingue A, contribución suficiente de las629 salidas al sesgo; B, convergencia posterior/lector; C, historia neural basal. La intervención y sus controles usan el organismo completo de166.700 neuronas y cuerpo. Los resultados numéricos vigentes se generan en [RESULTS.md](RESULTS.md); [CLOSE.json](CLOSE.json) reconstruye la decisión exploratoria. Esta campaña no puede admitir etapa3 por contrato.

Dos controles de1ms verifican la neutralidad del observador. Las cuatro condiciones comparten fuente y preparación científica dentro del candidato. La preparación del padre puede diferir porque la intervención precede a la preparación: esa diferencia forma parte de su efecto y se registra. El [factorial corporal](body_factorial_01/RESULT.json) cruza estado físico preparado y comandos grabados para separar sus contribuciones; no sustituye retirar el lector en el organismo con propiocepción activa.

El código Python organiza cargas, informes y archivos; no se propone como reemplazo de los kernels del bucle neuronal. Los coeficientes y su observador se ejecutan en CUDA. La medición separada del controlador C++ está en `motor_nuevo/native_stream_cost_20260923`.

## Reconstrucción compacta

Desde una extracción del paquete de revisión, en su raíz:

```bash
python3 -B campaign/analyze_long.py
python3 -B campaign/verify_readback.py
python3 -O -B campaign/test_readback_corruption.py
python3 -B campaign/render_results.py
python3 -B campaign/plot_results.py
```

Requiere Python3, NumPy y Matplotlib para la figura. Reconstruye observables desde NPZ, flujos firmados y decisiones; las pruebas deliberadamente alteran copias de banderas, cifras y criterio y deben rechazarlas incluso con `python -O`. La comparación local de estados científicos preparados usa los checkpoints completos. Esos activos grandes y los datos originales del organismo no forman parte del paquete compacto: **la extracción no reproduce una vida completa ni prueba por sí sola los estados ocultos**.

## Reejecución completa en el entorno local existente

Desde `/home/daroch/AXIOMA_ASTRA`, usar directorios nuevos y el entorno GPU ya instalado. Los presupuestos de esta campaña no autorizan repeticiones automáticas después de cerrarla:

```bash
/home/daroch/miniconda3/envs/GPU/bin/python -B \
  campanas/etapa3_pn629_intervention_20260923_15/run_set.py \
  --out /ruta/nueva/run --odor sham --engine causal_cuda --ms 400 --observe on
```

El runner registra fuentes ejecutadas, contrato, evento físico aceptado/predictor, manifiesto de intervención, sensores, flujo firmado y snapshots. Los hashes no convierten sus supuestos biológicos en hechos medidos. No extrapolar un giro pequeño a navegación hasta una fuente, aprendizaje o cerebro completo validado.

## Revisión externa

Se enviaron código y dos brazos terminados a ChatGPT para proponer la siguiente confirmación **antes** de observar nuevas corridas largas de referencia. Su consulta está en [CHATGPT_REQUEST.md](CHATGPT_REQUEST.md) y tiene presupuesto separado [REVIEW_PLAN.json](REVIEW_PLAN.json). Conservar el fallo anterior de estado KC; evaluar cualquier contrato funcional nuevo de forma prospectiva. Jev clasificó tareas en la ronda14 y no emitió una certificación biológica.
