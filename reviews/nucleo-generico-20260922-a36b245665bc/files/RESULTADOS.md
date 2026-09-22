# Resultado del primer núcleo genérico

**PROMETEDOR_NO_CONFIRMADO.** Ya existe un compilador de modelos declarativos, dos perfiles de precisión y ejecución GPU con estado residente. Se añadieron mecanismos de siete estados y doce especies mediante descripciones del modelo. Esto es un avance de ingeniería; no acredita todavía un motor general completo ni supera etapa3.

La prueba grande integra un segundo continuo de una carga sintética heterogénea con **726,900 estados y 23,296,700 conexiones**. Contiene tres poblaciones continuamente acopladas. No contiene la biofísica completa del conectoma, espigas con reinicio, retardos, ruido ni cuerpo.

| Ruta/perfil | Avance y escáner (s) | Preparación (s) | Error normalizado medido | Criterio en esta carga |
|---|---:|---:|---:|---|
| A/fast | 0.658 | 1.992 | 9.446e-05 | Cumple |
| A/precise | 3.243 | 1.788 | 5.646e-08 | Cumple |
| C/fast | 0.925 | 1.917 | 2.070e-03 | Cumple |
| C/precise | 22.107 | 1.840 | 3.870e-06 | Cumple |

Referencia CPU DOP853 con dos tolerancias: cambio máximo observado 2.936e-11. Se comparan4096sondas en21instantes y los726900estados al final. No es una cota continua ni una prueba de exactitud de todos los estados intermedios. Error=|candidata−referencia|/(escala física declarada+|referencia|); no equivale al porcentaje relativo de un voltaje cerca de cero. Los límites prospectivos son1e−2(rápido) y1e−5(preciso).

Una ejecución por combinación: tiempos exploratorios, no una cohorte confirmatoria. El arranque se muestra por separado; el tiempo completo del proceso, incluida compresión, está en los JSON. El poolGPU no es la VRAM total del proceso. Las cantidades de copias registradas son payloads explícitos de la API, no una medición del tráfico real PCIe. No se calcula una aceleración contra el motor antiguo usando esta carga diferente.

## Rivalidad A/B/C y negativo conservado

A es acoplamiento global: en esta ronda, RK4 adaptativo residente y referencia CPU. B propone multirrate con corrección del acoplamiento, pero no se implementó. C ensaya punto medio exponencial con separación diagonal y derivadas generadas. No es Krylov ni un integrador exponencial global completo. Se implementaron dos candidatos, sin mezclar sus resultados.

En Hodgkin–Huxley:

| Ruta/perfil | Error medido | Criterio |
|---|---:|---|
| A/precise | 7.477e-07 | Cumple |
| A/fast | 1.891e-03 | Cumple |
| C/precise | 1.373e-04 | Falla |
| C/fast | 6.620e-02 | Falla |

**C se descarta como candidato general en ambos perfiles de esta ronda**, aunque pueda pasar la carga suave grande. No se ajustaron sus tolerancias para rescatarlo. Los estimadores locales no garantizan error global en un sistema excitable. El negativo no descarta otros métodos exponenciales.

## Qué funciona y qué falta

Funcionan: descripciones con unidades, estados y ecuaciones; puertos graduados; conectividad dispersa con ciclos; generaciónCPU/GPU; perfiles FP64 fast/precise; escáner de estados confirmados; cambios de parámetros, clamp constante, altas/bajas y conexiones mediante reemplazo transaccional; conservación de identidades; checkpoints y reanudación exacta en el entorno probado. Los cambios inválidos no modificaron el estado vivo.

Las extensiones de siete estados y doce especies pasaron en cuatro condiciones de cableado con ambas rutas/perfiles, sin modificar model.py, runtime.py ni autodiff.py para añadirlos. Después se reparó una conversión genérica de tipos y se repitieron todos los brazos GPU afectados; CORE_REPAIR.json conserva ambas versiones. Cambiar etiquetas dejó la trayectoria idéntica; reordenar poblaciones conservó la trayectoria dentro del límite declarado. Esta es una extensión formulada por el autor después de congelar el código, no una reserva ciega independiente.

Faltan: GPU para masa no diagonal/DAE, integración implícita dispersa de rigidez heterogénea, eventos y retardos, ruido reproducible, plasticidad estructural interna, acoplamiento corporal y contraste de un cerebro real completo. CPU admite masa constante no diagonal sólo hasta4096estados. No se anuncia soporte de otra especie completa por haber ejecutado una descripción HH.

## Autocrítica y siguiente decisión

El error anterior fue identificar la optimización de un sector con la construcción del motor. Aquí la prueba de progreso cambia: nuevos mecanismos entran como descripciones y se conserva un núcleo común. Tampoco conviene elegir una arquitectura por la velocidad de una carga suave: el fallo de C en HH obliga a conservar el contraste rígido.

La siguiente ampliación concreta es un backend implícito general con masa constante por época, residuo completo y Jacobiano/JVP de todas las conexiones. ChatGPT propone reutilizar ARKStep y Ginkgo; la documentación primaria confirma el método y la interfaz. Las reducciones del precondicionador no deben retirar acoplamientos del residuo. B sigue como rival si aparece una separación temporal útil; C requiere una hipótesis distinta del candidato descartado.

[ARKODE: métodos](https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html) · [Ginkgo–SUNDIALS](https://sundials.readthedocs.io/en/latest/sunlinsol/SUNLinSol_links.html). No se integraron esas bibliotecas todavía.

ChatGPT aportó dos revisiones documentales, sin ejecutar el núcleo nuevo. Su estado de revisión de fuentes publicadas se registra en el recibo de entrega. Jev realizó una llamada de clasificación:1003tokens de entrada,214de salida; no aprueba conclusiones. No hubo subagentesCodex.

## Verificación y reproducción

El verificador reconstruye errores y veredictos desde los arrays y los límites congelados. Detecta corrupción de criterio, bandera, estado y tiempo; checkpoints detectan alteraciones de estado y metadatos. Las comprobaciones funcionan con Python−O. El fallo inicial de compilación CUDA y el problema de tipo entero en el reloj se corrigieron antes de congelar; sus logs y el ensayo C fallido están preservados. Después apareció un fallo real de escala int64 interpretada como double enGPU; se corrigió aFP64 explícito y se repitieron todas las variantesGPU afectadas, sin cambiar los límites numéricos. Las referenciasCPU se reutilizaron con hashes.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -B -O verify_package.py
python -B -O verify.py verificacion_nueva.json
python -B -O extension_test.py extension_nueva
python -B -O review_falsifiers.py falsadores_nuevos
python -B -O cli.py --model examples/mixed.json --mode fast --until 0.1 --edits examples/schedule.json --scan cells:v --out demo_nueva
# Reproducción completa de la campaña, seis corridas de hasta300s:
python -B large_campaign.py campana_nueva
```

No repetir automáticamente las seis corridas sólo para entregar el paquete. La comprobación de la extracción limpia consta en DELIVERY.json local. El motor histórico y su organismo no fueron modificados. Motivo de parada: primera prueba de arquitectura completa a este alcance, dos candidatos ejecutados y negativo conservado; faltan ampliaciones antes de promover el motor o retomar etapa3.
