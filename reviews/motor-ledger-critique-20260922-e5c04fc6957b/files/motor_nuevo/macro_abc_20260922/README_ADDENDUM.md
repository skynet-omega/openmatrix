# Addendum: crítica de Gemini y reparación del registro ADD/SET

La publicación77611a3 y sus corridas permanecen inmutables. Este addendum incorpora la observación posterior del usuario y un defecto encontrado por ChatGPT al leer el código exacto. No añade simulaciones corporales ni una aprobación del motor.

ChatGPT observó que Ledger descartaba SET cuando jump==0; su prueba CPU fue reproducida localmente usando la clase publicada. Las pruebas anteriores entraban directamente al puerto, por lo que omitían este enlace. Se conserva `reset_adapter_before_review.py` y el contraejemplo. No hay evidencia de que este caso alterara las corridas publicadas; tampoco se les atribuye retrospectivamente la nueva versión.

Corrección: `reset_ledger.py` conserva SET por su operación, valida antes de publicar y mantiene el comportamiento ADD positivo heredado. `Ledger.at` y el puertoGPU interpretan ahora la misma semántica; no se introduce saturación del solver. El adaptador es para el modelo histórico normalizado; el puerto matemático admite otros dominios. No confundir esa distinción con un motor biológico universal.

Prueba local: Ledger real→puertoCUDA, SET de salto cero, simultaneidad no conmutativa, consultas retrospectivas, errorCPU/GPU, rechazo de publicación inválida sin cambio parcial y caminoADD idéntico. El snapshot completo20ms anterior corresponde a la versión anterior; no se repitió el organismo por este contraejemplo aislado.

La crítica de Gemini se traduce en prioridadCNS y separación de velocidad/biología. Las afirmaciones98%prescindible y fallo15ms=rigidez no se adoptan como hechos. `analyze_activity.py` reconstruye fracciones descriptivas desde tres snapshots reales incluidos. No son una política dispersa ni demuestran qué estados pueden saltarse.

Desde una extracción limpia, con NumPy/SciPy/Numba/CuPy y CUDA disponibles:

```bash
PY=/home/daroch/miniconda3/envs/GPU/bin/python
$PY -B -O motor_nuevo/macro_abc_20260922/check_ledger_bridge.py
$PY -B -O motor_nuevo/macro_abc_20260922/analyze_activity.py
```

Se verificaron ambos comandos desde una extracción nueva. La meta sigue siendo un motor general eficiente; A implícito global, B trayectorias multirritmo corregidas, C programa de operadores compilado, másD de orden alto. Esta reparación no es el prototipo de velocidad pendiente.

[Campaña completa anterior y fuentes originales](https://github.com/skynet-omega/openmatrix/tree/77611a3ab5e2fa8fa9d504f104743d7401260015/reviews/motor-general-events-macro-20260922-53d84a1c3eaf).
