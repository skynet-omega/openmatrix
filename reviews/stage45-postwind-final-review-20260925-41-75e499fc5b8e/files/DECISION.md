# Decisión tras los controles físicos41

La prioridad A del usuario se conserva. El mando entregado después del viento empeora el error frente a giro nulo, y el avance tónico añade demanda de orientación. Ninguna de esas observaciones localiza el defecto en una neurona o convierte una detención en una solución.

El diagnóstico adicional del filtro es posterior a observar las trazas; no modifica el contrato ni propone parámetros. Reproduce exactamente lector, recurrencia EMA y relé. Durante980ms posteriores al viento:

| Señal | Integral neta (°) | ms positivos | ms negativos | ms cero |
|---|---:|---:|---:|---:|
| raw | +0.014936810 | 517 | 463 | 0 |
| EMA | +0.028397106 | 705 | 275 | 0 |
| applied | +2.190000000 | 507 | 69 | 404 |

La diferencia de concentración L−R sigue negativa durante todo ese tramo; no se perdió el lado de la fuente en el campo muestreado. Esto no demuestra que esa información llegue intacta al lector ni que la actividad DN deba ser una función estática de ella. La integral cruda positiva ya existe; EMA/relé la amplifican. No es un bug aritmético del filtro, ni prueba de que bastaría invertir su signo.

Rivales para la siguiente decisión dentro de A, sin nuevas ejecuciones autorizadas por este archivo:

- A1: el estado/representación dentro del CNS deja de expresar una dirección útil. Información permitida: señales sensoriales consumidas y poblaciones anatómicas prefijadas. Discriminador: intervenciones sensoriales izquierda/derecha emparejadas con intensidad/historia conservadas, lectura prefijada. Falsador: señal direccional útil llega de forma sostenida a la salida neural.
- A2: una señal neural útil pierde su relación temporal/direccional en la interfaz EMA/relé. Información: actividad DN y calibración de interfaz independiente de esta trayectoria. Discriminador: comparar señal liberada y lector calibrado sobre perturbaciones ajenas a navegar a esta fuente. Falsador: ya falta dirección útil en la señal prelector o una interfaz validada no compra control.
- A3: avance y orientación necesitan coordinación contextual. Información: propio estado/sensores disponibles al organismo, sin bearing suministrado. Discriminador: primero establecer giro ante estímulos conocidos con avance controlado y luego exigir conservación al combinar avance/giro. Falsador: controlar avance no recupera la relación neural/motora o el supuesto aporte sólo consiste en detenerse.

El padre no cuenta como hipótesis. No fusionar las tres por analogía ni lanzar dos organismos largos para completar una tabla. B/CNS→VNC→MN y C/seis patas mantienen el lugar indicado por el usuario. Una siguiente ronda necesita su propio presupuesto y elección causal; la ronda41 se detiene con cuatro intentos consumidos. El brazo sin viento y la contribución específica de estados/receptores no se dan por resueltos.
