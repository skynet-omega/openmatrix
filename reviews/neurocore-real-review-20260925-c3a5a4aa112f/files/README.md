# Neurocore: integración en el organismo real

Una implementación de RK3(2) FP64 con controlador C++/CUDA residente y RHS
capturado como grafo CUDA. `graph_runtime.py` y `resident_controller.cu` no
importan anatomía, conducta, PN, KC ni nombres de neuronas. El modelo entrega
las derivadas, tolerancias, dominio y, opcionalmente, una proyección exacta
dependiente del tiempo. Las decisiones adaptativas se toman en la GPU.

`real_model.py` conecta ese contrato con las ecuaciones efectivas existentes.
`run_real.py` ejecuta el organismo completo con cuerpo, entradas, masas,
conectividad y estados originales. Sustituye el avance CNS; PN y membranas
espaciales conservan sus propios solvers. Es una integración parcial real,
no una reescritura completa del organismo ni una calibración biológica nueva.

## Uso en este equipo

Desde esta carpeta, con el entorno GPU instalado:

```bash
/home/daroch/miniconda3/envs/GPU/bin/python run_real.py --mode candidate --ms 100 --field odor_left --out resultado_nuevo --wall-limit 680
/home/daroch/miniconda3/envs/GPU/bin/python run_real.py --mode reference --ms 100 --field odor_left --out referencia_nueva --wall-limit 680
/home/daroch/miniconda3/envs/GPU/bin/python compare_real.py referencia_nueva resultado_nuevo --out comparacion_nueva.json
```

Cada destino debe ser nuevo. El adaptador se instala sólo en el proceso de la
corrida; las fuentes del motor estable permanecen intactas. El preparado usa
el campo lateral estático existente: entrada binaria dependiente de la posición
de las antenas, activada al comienzo de la prueba. No es la campaña gaussiana,
una prueba de vuelo, ni una perturbación de viento. Todos los mecanismos y
prótesis biológicos heredados pertenecen al modelo de comparación.

## Núcleo y compilación

```bash
/home/daroch/miniconda3/envs/GPU/bin/python build.py --out libresident_nuevo.so
/home/daroch/miniconda3/envs/GPU/bin/python check_runtime.py --library ./libresident_nuevo.so
/home/daroch/miniconda3/envs/GPU/bin/python -O check_runtime.py --library ./libresident_nuevo.so
```

`build.py` admite rutas de nvcc, compilador C++, cabeceras, biblioteca CUDA y
arquitectura. La compilación comprobada en este equipo usa CUDA 12.1, GCC 12
y sm_89. GCC 13 con ese nvcc falló en las cabeceras del sistema; no se modificó
el sistema ni se eludió el error. Se conserva la fuente exacta del controlador
genérico del 24 de septiembre, donada por `native_clean_20260924_01`.
La primera extracción detectó además nvcc 12.0 en PATH junto a bibliotecas 12.1.
El selector se corrigió para usar el conjunto 12.1 coherente del equipo;
`build_initial.py` y `BUILD_DISCOVERY_FAILURE.txt` conservan ese fallo de entrega.

El RHS genérico tiene firma `rhs(y, clock, fraction)` y devuelve un vector
CuPy FP64. `clock` contiene el comienzo y tamaño del intento **relativos a la
época**; el modelo incorpora su origen absoluto si lo necesita. `project` debe
ser puro, sin publicar efectos de etapas rechazadas. Las entradas y estructuras
capturadas mantienen dirección y forma; se actualizan antes de cada época.
El núcleo acepta RHS heterogéneos mediante esa interfaz, pero RK explícito no
garantiza eficiencia para cualquier rigidez. Las masas espaciales conservadas
siguen en sus propietarios actuales; no se reemplazaron por la identidad.

Cada intento evalúa cuatro derivadas; el método de referencia evalúa seis.
El error embebido y el dominio deciden si el estado se publica. Los eventos
obligatorios acortan el paso, su extremo se evalúa por la izquierda para la
derivada y la proyección confirmada queda por la derecha. Un error revierte
todo el estado de la época, incluso pasos ya aceptados dentro de ella.

El scheduler conserva un máximo de 10.000 intentos por época. El argumento
`budget` del adaptador es de compatibilidad: no implementa un plazo de pared
dentro de un kernel en curso. El runner acota la campaña externamente y guarda
un prefijo si alcanza el límite. Los fallos del núcleo son excepciones; no
dependen de `assert`. La preparación histórica exige Python normal; `-O` se
comprueba para el núcleo y el comparador independiente.

## Evidencia y límites

`PLAN.md` contiene presupuesto y umbrales previos. `PAIR20_FULL.json` compara
las trayectorias completas de 20 ms, todos los eventos registrados y el estado
PN final. Las carpetas de cada brazo retienen trayectoria, sucesos, estado final,
parámetros, pesos identificados por hash y tiempos. Los registros de eventos
incluyen predictores; no son un conteo de espigas físicas únicas.

La GPU estuvo compartida con otra sesión durante parte del trabajo. Los tiempos
observados se informan como tales: una pareja no demuestra una aceleración
estable. Las dos parejas de 100 ms FALLARON sus límites de comparación,
también después de refinar tolerancias. Consultar `RESULTADOS.md`; esta candidata
experimental no queda activada en el motor estable.

El refinamiento se reproduce añadiendo `--precision-factor 0.1 --compact-state`
a ambos brazos. El guardado compacto conserva trayectoria completa y estado PN
final, pero no un checkpoint reiniciable del organismo. `PRECISION.md` conserva
la decisión previa a esas dos corridas; las cuatro primeras usaron exactamente
la fuente guardada como `run_real_initial.py`.

Las verificaciones `check_runtime.py` son complementarias y sintéticas: fuerzan
siete rechazos, un salto de entrada y un fallo después de progreso parcial para
comprobar restauración exacta. La evidencia de utilidad procede de las corridas
del organismo completo, no de estos controles de implementación.
