# Campaña41 — diagnóstico de orientación con el cuerpo actual

[Informe calculado](REPORT.md), [trazas verificadas y resultados](VERIFIED.json), [figura](REPORT.png), [contrato previo](PLAN.json), [fallo de identidad y reparación](REPAIR_EXECUTION.json).

Se ejecutaron cuatro reproducciones físicas: una identidad continua fallida, una identidad que reproduce exactamente el reinicio histórico, giro nulo tras el viento y avance nulo tras el viento. El control sin viento quedó pendiente para conservar el presupuesto de cuatro ejecuciones; no se amplió tras el fallo. Cero neuronas simuladas y cero ajustes de parámetros. La clasificación de la ronda es **PROMETEDOR_NO_CONFIRMADO**, con etapas4/5 abiertas.

ChatGPT aportó tres explicaciones propias y código de descomposición ejecutado aquí sobre la traza real. Declaró que no había podido acceder al paquete original. Jev priorizó el diagnóstico físico y el riesgo de atribuir feedback neural al replay. Sus opiniones no deciden el resultado. No se usaron subagentes Codex.

La cápsula contiene el modelo MuJoCo compilado, estado de integración, estado y observación de contacto del controlador, referencia exacta de soporte, fuentes Python transitivas del replay, datos originales y resultados. La clase de contacto y su controlador de soporte son copias literales; el cambio de carga de la referencia HDF5 a sus arrays exactos queda en `PROTOTYPE_DIFF.txt`. Se conserva también la fuente original. El MJB requiere MuJoCo3.2.7; reconstruir el XML histórico o el organismo neuronal entero queda fuera de esta cápsula.

Reproducción desde una extracción nueva del ZIP, estando en su raíz. Python3.10.18, NumPy1.26.4 y MuJoCo3.2.7; matplotlib sólo para regenerar la figura. Instalar esas dependencias en un entorno elegido si no existen. El exportador `prepare_inputs.py` es procedencia, no el punto de entrada portátil.

```bash
# El modelo compilado se comprime por separado dentro del ZIP.
tar -xzf model.tar.gz
export OPENBLAS_NUM_THREADS=1

# Modo corto: reconstruye tres resultados desde trazas y rechaza corrupciones.
python -B verify_clock.py --run run_02 --out RECOMPUTED.json --corruption-tests
python -O -B verify_clock.py --run run_02 --out RECOMPUTED_O.json --corruption-tests
cmp RECOMPUTED.json RECOMPUTED_O.json
python -B probe_force.py --out FORCE_RECOMPUTED.json
python -B run_external_audit.py --out CHATGPT_RECOMPUTED.json

# Identidad completa: 2s corporales, con el reinicio histórico.
python -B run_replay.py --out identity_new --identity-only

# Modo completo: tres condiciones físicas útiles, sin cargar el cerebro.
python -B run_replay.py --out replay_new
python -B verify_clock.py --run replay_new --out NEW_RESULT.json --corruption-tests
```

Cada salida debe tener un nombre nuevo. El runner rechaza `python -O` por las comprobaciones históricas de su controlador; el verificador usa excepciones explícitas y funciona también con `-O`. Se conserva el verificador inicial que interpretó erróneamente el origen temporal como40ms. `verify_clock.py` usa el entero exacto del checkpoint44.486s: [reparación documentada](VERIFIER_REPAIR.json), sin tolerancias modificadas.

El primer torque del viento no se aplica en el reinicio frío original, aunque se cuenta la llamada. Este replay conserva esa peculiaridad para reproducir la evidencia. Una futura corrección debe inicializar y probar los datos cinemáticos usados por fuerzas nuevas después de restaurar, sin alterar estado comprometido ni reetiquetar campaña40.

Retirar el avance o giro cambia la física con las órdenes neuronales congeladas: no es el contrafactual del organismo con feedback. No convierte detenerse en navegación ni suma los efectos como causas independientes. Los pasos futuros se deciden con A/orientación e interfaz conjunta, B/CNS→VNC→MN acotado y C/músculos y seis patas posterior.

Extracción nueva comprobada: análisis de las tres condiciones byte idéntico en Python normal y `-O`, tres corrupciones rechazadas, y repetición física completa de identidad2s con diferencias0;27,5s aproximadamente. No se repitieron los dos contrafactuales dentro de la extracción ni el cerebro. [Recibo](CLEAN_VERIFIED.json). CPU del primer intento fallido no registrada; las tres ejecuciones útiles sí registran CPU, pared y memoria.
