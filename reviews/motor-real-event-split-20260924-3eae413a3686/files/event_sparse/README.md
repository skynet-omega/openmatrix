# Operador disperso de puertos — discriminador del 24-09-2026

**PROMETEDOR_NO_CONFIRMADO, sólo como donante de operador.** La contribución de
4.062 fuentes impulsivas al CSR base pasó la tolerancia congelada en las 60 marcas
reales de un bloque de 125 µs. No hay motor, integración MRI/QSS2, calibración
biológica ni admisión de etapa. `CLOSE.json` es el cierre mecánico vigente;
`gpu_01/RESULT.json` y sus hashes se conservan intactos.

La matriz completa contiene 25.582.938 aristas. La partición selecciona 885.587
aristas hacia 6.342 receptores, conservando el orden dentro de cada fila. El
complemento de fuentes continuas se representa explícitamente al preparar la
partición y se registra con hash, pero **no se evalúa** en este discriminador.
El control recorre el CSR completo con liberación cero fuera de los puertos; no
es la corriente total del cerebro.

Cada consulta reconstruye `q'=-q/tau_q`, `s'=(q-s)/tau_s` en CUDA desde el inicio
de época y sus eventos SET/ADD ordenados. Se mantienen los 4.060 valores distintos
de `tau_q`, el límite `tau_q=tau_s`, la inclusión `evento<=consulta` y el orden de
eventos simultáneos. No se agrupan constantes ni se crea una matriz densa por
grupo. Las consultas son puras y admiten los retrocesos especulativos registrados.
CUDA reduce corrientes por receptor, aplica caps a las fuentes de los receptores
no visuales y conserva las conductancias visuales positivas/negativas separadas.
Los buffers son privados y residentes; ninguna consulta transfiere arrays al host.

`cuda/weights` es una **instantánea al retornar la evaluación capturada**. No se
afirma que contenga todos los pesos temporales consumidos por PN, APL u otras
sobrescrituras especializadas. Esas reglas necesitan una extensión con versiones
propias antes de poder usar el donante dentro de un motor. Un token W incorrecto
rechaza la consulta, y añadir un evento después de preparar la época la invalida.
El cliente debe reconstruir y repetir las consultas afectadas por un ingreso
tardío. No se actualiza silenciosamente el pasado.

Resultados recalculados desde los arrays:

- Error máximo de corriente: `1.4551915228366852e-11`; error normalizado máximo:
  `4.43054670505357e-6`, frente a puerta `<=1` con `atol=rtol=1e-10`.
- El verificador CPU independiente suma por receptor los datos compactos para
  las 60 consultas: error normalizado máximo frente a las corrientes CUDA
  `2.630344497092738e-5`. `q/s` concuerdan con CPU y con la traza a `5.55e-17`.
- Trabajo contado de aristas: `28.8881x` en régimen y `19.4996x` al cargar una
  pasada global de partición a estas 60 consultas. Este conteo no incluye como
  ahorro la lectura/auditoría del control; la ejecución de la sonda hace ambos
  brazos completos.
- La partición tardó `236.614 ms`. Los eventos CUDA registraron `65.819 ms` para
  el CSR entero, `3.558 ms` para el reducido y `0.978 ms` de proyección. Su suma
  `70.355 ms` excede los `65.160 ms` de pared del bucle. **Relojes inconsistentes;
  ningún cociente temporal se admite como aceleración.** Aun tomando esos tiempos
  nominalmente, cargar la partición inicial da sólo `0.274x` en estas 60 consultas.
- Una sonda, `4.47 s` de pared total; pico RAM `1.407 GiB`, VRAM adicional
  observada `0.613 GiB`; aproximadamente `12.1 MB` de fuentes/datos del operador
  antes del empaquetado. Presupuesto original: 120 s, 3 GiB RAM, 2 GiB VRAM,
  100 MiB disco, cero interacciones del organismo. No se repitió la medición.

Se ejecutaron cuatro fixtures CPU, normales y con `python -O`: partición/caps/
visual±/constantes iguales y heterogéneas; evento tardío dentro del trial y
SET/SET/ADD simultáneos con rollback; ingreso tardío que invalida; versión W o
eventos incorrecta. El mismo fixture numérico se contrastó también en cuatro
consultas CUDA, fuera del cronometraje real. Los errores se comprueban con
excepciones explícitas, nunca `assert`.

La cápsula numérica es autocontenida: `gpu_01/capsule.npz` contiene el CSR
rectangular 6.342×4.062, mapas de fuentes/receptores, caps y máscaras separadas,
estado inicial, taus, eventos y 60 tiempos; `gpu_01/currents.npz` conserva ambas
corrientes FP64 completas en todos los receptores activos, `q/s` CUDA y todos los
intervalos de cronometraje. Los receptores omitidos son cero exacto, comprobado y
enlazado por hash del tensor de tamaño original. La matriz CSR completa y la
sesión neuronal no están incluidas; su procedencia queda en `FROZEN.json`.

Verificación corta desde una extracción limpia, sólo Python 3 y NumPy:

```bash
cd event_sparse_capsule
python3 -B verify_capsule.py gpu_01
python3 -B -O verify_capsule.py gpu_01
python3 -B test_cpu.py --out cpu_repeat_01
```

Esto recalcula todas las corrientes compactas en CPU y comprueba hashes, versiones
y puertas. No reejecuta el CSR completo en GPU ni el organismo. El paquete se
probó desde `extracted_01/`, sin importar fuentes del árbol de trabajo.

La sonda completa queda en `run_probe.py`; requiere CuPy/CUDA y los archivos
originales de `capture_01`. Su invocación original fue:

```bash
/home/daroch/AXIOMA_FLYWIRE/matrix/.venv/bin/python -B run_probe.py \
  --capture ../capture_01 --out gpu_repeat_01
```

Ese modo consume otra sonda y **no se ejecutó** después del cierre. El histórico
se usó sólo para su intérprete de Python; no se modificaron archivos fuera de
este directorio. Próximo discriminador: incorporar el donante con invalidación
del operador a un integrador acoplado y medir error/coste de todos sus componentes.
Las alternativas rivales A/B/C y sus falsadores están en `CONTRACT.json`.
