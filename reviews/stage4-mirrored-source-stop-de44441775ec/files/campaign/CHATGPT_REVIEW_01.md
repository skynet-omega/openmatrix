**Corregiría una validación del verificador antes de congelar la ejecución. No cambiaría las fuentes, los umbrales ni las ecuaciones.** Reproduje un caso en que una muestra de mando adicional, sin tiempo correspondiente, cambia la decisión de B a A. Es un fallo del control de integridad de las trazas, **no evidencia de que el runner vaya a generar ese error**.

## 1. Defecto reproducido: longitud temporal no comprobada

En `candidate/verify_mirror.py::check_trace()` —líneas 92–152—, `n` se obtiene de `fase`, pero no se exige que todas las señales tengan esa longitud. El lector puede contener 441 filas mientras fases, sensores y relojes contienen 440. La fórmula del lector y su desfase siguen siendo coherentes entre esos arrays largos; después, `command[40:]` integra **401 muestras** y devuelve `steps=400`. 

Ejecuté las funciones `check_trace → pair_metrics → decide` con trazas sintéticas de interfaz, la geometría publicada y los valores del plan:

| Caso CPU | Muestras de ensayo declaradas | Muestras de mando integradas | Contraste de mando | Decisión |
|---|---:|---:|---:|---|
| Longitudes correctas | 400 | 400 | 0,009975° | B |
| Una fila adicional en DN y comandos | 400 | **401** | **0,010175°** | **A** |

Las referencias del fixture eran copias equivalentes; **no ejecuté `inspect_arm()` ni `compute()` completos**. El caso demuestra el defecto en la cadena de cálculo/decisión, no una reproducción integral con todos los recibos.

El runner construye normalmente cada array desde la misma lista `rows`, por lo que no he encontrado allí el origen de una longitud desigual. La validación sigue siendo necesaria para detectar archivos mal formados o cambios de exportación. 

### Corrección mínima

Añadir inmediatamente después de calcular `n` en `check_trace()`:

```python
    for key, value in t.items():
        value = np.asarray(value)
        need(
            value.ndim >= 1 and value.shape[0] == n,
            'Trace time length mismatch: ' + key
        )

    for key in ('DN_q_actual', 'DN_q_usada', 'DN_baseline'):
        need(np.asarray(t[key]).shape == (n, 4),
             'Reader layout mismatch: ' + key)

    for key in ('command_yaw_rate_rad_s',
                'command_forward_mm_s', 'yaw_delta_deg'):
        need(np.asarray(t[key]).shape == (n,),
             'Scalar time-series layout mismatch: ' + key)
```

Probé este guard: conserva los dos casos correctamente dimensionados y rechaza ambos brazos con la fila extra. Añadiría la regresión a `test_verify_mirror.py`; sus ocho pruebas actuales se concentran en geometría y decisiones, no ejercitan este defecto de la lectura de trazas. Actualizar el cierre de fuentes **antes de ejecutar**, preservando el commit original y sin cambiar criterios científicos. 

## 2. Runner y geometría: no encontré filtración de resultado futuro

El runner guarda el preparado **antes** de instalar la fuente. Evalúa ambas gaussianas sobre esa pose para comprobar media y contraste; esas comprobaciones no ajustan la fuente ni emplean yaw futuro. Después instala únicamente el campo seleccionado. El helper mantiene fuente y anchura fijas y conserva el intervalo sensorial comprometido.  

Reconstruí en CPU las coordenadas de ambas fuentes: coinciden exactamente con `CAMPOS.json`. Obtuve:

- **S+:** \(c_L=0,7766285476,\ c_R=0,3987343915\).
- **S−:** \(c_L=0,3987343915,\ c_R=0,7766285476\).
- Media inicial ≈0,5876814695 y contraste ±0,3778941561.  

La media se iguala **solo al inicio**. La trayectoria posterior puede cambiar tanto la componente común como la diferencial. El resultado permite contrastar **estas dos geometrías**, no identificar aisladamente un circuito de resta bilateral.

## 3. La inferencia permitida por A/B/C

**A no exige giros absolutos opuestos.** También comprobé que `decide()` puede devolver A cuando ambos mandos y ambos giros son positivos, siempre que S+ supere materialmente a S−. Esto **es compatible con “modulación según el lado de la fuente”**, pero no con afirmar “cada rama gira hacia su fuente”. No añadiría ahora otra condición: conservaría el alcance escrito. 

Hay una precisión para **B**: `output_ok()` exige mando **y** separación corporal. Por ello B puede significar “mando material, respuesta corporal insuficiente”, no solamente “el circuito no diferencia los olores”. Recomiendo registrar por separado `command_ok` y `yaw_ok`, manteniendo la decisión conjunta y los umbrales. No atribuir a PN/CNS un fallo exclusivamente corporal.

Los márgenes numéricos son coherentes para esta comparación:

\[
|\delta(\Theta_+-\Theta_-)|\le0,002+0,002=0,004^\circ.
\]

La misma desigualdad se aplica al contraste integrado de mando usando los límites L1 por brazo. Los mínimos de **0,01° de mando y 0,005° corporal** superan esa reserva, aunque el segundo deja solo **0,001°**. Son reservas frente a la otra discretización, **no cotas de la solución exacta**. El código exige efecto en ambos perfiles y concordancia por brazo.  

La salida del programa ya distingue incompleto/bloqueado. Aun así, `exit=0` incluye tanto B descartado como prometedor sin confirmar: la automatización debe consultar `decision`, no usar solo el retorno.

## 4. Navegación, animación y publicación

El control posterior necesario sigue siendo **online frente a replay con una perturbación espacial predeclarada**, desde estados emparejados. Ambos brazos deben compartir la perturbación y conservar propiocepción/cuerpo; solo el replay recibe la cinta del donante no perturbado. Evaluar corrección de rumbo y distancia a la fuente, con incertidumbre XY y angular. Reproducir la cinta sobre su misma trayectoria sin perturbación sería un control técnico, no prueba de utilidad del feedback.

Un video puede representar las **posturas guardadas**, siempre que estén disponibles el modelo corporal y sus assets exactos. Interpolar fotogramas no reproduce subpasos físicos; fuerzas, contactos, actividad muscular o estados neuronales no guardados no pueden inventarse. La animación no identifica causalidad ni prioridad histórica. El paquete no aporta una búsqueda que fundamente una primicia mundial. 

## Alcance de mi trabajo

**Leí completos por tramos:** `REVIEW_REQUEST.md`, `PLAN.json`, `run_mirror.py`, `verify_mirror.py`, `test_verify_mirror.py`, `CAMPOS.json`, índice y `ARCHIVE.json`; además `PREPARED_GEOMETRY.json`. Leí parcialmente el preflight —líneas 1–95— y el helper gaussiano —25–169—.

**Ejecuté CPU:** funciones copiadas de lectura/decisión con sus constantes declaradas, el contraejemplo de longitud, el guard propuesto y la reconstrucción geométrica. Son fixtures de software, **no trayectorias físicas**. No ejecuté la suite original completa, `compute()`, CUDA, MuJoCo ni el organismo. La descarga del ZIP falló por DNS; no verifiqué sus 15 miembros.

:chatgpt-content-reference{index="11"}[Reproductor CPU, entradas sintéticas y resultados conservados](sandbox:/mnt/data/AXIOMA_MIRROR_1F9D_CPU_REVIEW.zip)

**Dictamen pre-ejecución:** reparar esa comprobación de dimensiones y conservar el diseño experimental. En lo leído no encontré otro bloqueo material del runner. Un A futuro respaldaría modulación lateral de este modelo, no navegación ni equivalencia biológica.
