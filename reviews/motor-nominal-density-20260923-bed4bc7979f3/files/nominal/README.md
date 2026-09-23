# Recuperar propuestas tras cortes por eventos

ChatGPT entregó un generador C++ aislado. Su original se ejecutó sin cambios:14 controles con un doble CPU de CUDA aprobaron y las dos bibliotecas compilaron contra la instalación CUDA real. Candidata SHA256 `116bdae3bd7a4374cb3aca09af45629052c618d0cee88ba6baa034f73ce235e5`. No se ejecutó esta variante enGPU ni en el organismo. Estado: **PROMETEDOR_NO_CONFIRMADO**.

Cuando un evento obliga a acortar un paso aceptado y su error resulta menor que0,1, la candidata recupera la propuesta nominal anterior; cada nuevo paso sigue pasando por su propio estimador, controles de dominio y fronteras. Tras rechazo conserva la reducción del padre. No extrapola el error de una cola para aceptar el intervalo siguiente.

En el doble CPU la cola fácil reduce6→3 intentos. Otro caso provoca tres rechazos nuevos y26→20 intentos: el rechazo sigue operativo. Son ejemplos de control, no aceleraciones de la mosca. La revisión consumió menos de2/180s, cero cargas del organismo y cero llamadas externas adicionales. El motor de la confirmación16 no cambió.

Rivales: **A**, propuesta nominal independiente del corte; **B**, integrador exponencial embebido con menos evaluaciones del RHS completo; **C**, integración espacial con influencia recurrente acotada. B debe conservar la dependencia de la tasa con el estado; C no puede equiparar ausencia de espigas con ausencia de transmisión gradual. Los negativos deETDRK4/MOTOR14 se conservan; no son una arquitectura general validada.

Reproducción corta, desde esta carpeta en una extracción nueva:

```bash
python3 -S generar_nominal_original.py generated_01/graph_control_parent.cpp nueva_generacion --selftest
```

El comando crea una carpeta nueva, compila controlesCPU y rechaza un padre de hash distinto. `BUILD.json` registra los comandos de compilaciónCUDA realmente ejecutados; compilación no es ejecuciónGPU. La campaña propuesta para comparar padre/candidata en una carga real aún no está iniciada ni debe interferir con las referencias vigentes.
