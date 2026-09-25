# Campaña 37 — cerrar ajustes de ganancia y revisar la interfaz motora

Etapas 4 y 5 abiertas. Ningún motor, peso sináptico, lector ni resultado histórico fue reemplazado. Se terminó una criba de una candidata y dos comprobaciones sobre datos existentes; cero organismos nuevos. La candidata quedó **DESCARTADA**. Las cotas son álgebra comprobada sobre señales fijas, no admisión científica del organismo.

## Resultados que cambian la decisión

1. **Normalizar el lector por actividad de reposo no transportó bien.** El único prototipo fijó offset y escala con los primeros 200 ms de un sham. En los siguientes 200 ms el mando neto residual pasó de +0,009566° a +0,095724°: falló el margen prospectivo de no regresión de 0,02°. Mejorar el contraste de los ensayos con olor no rescata ese fallo. No se ajustaron parámetros después. Ver [tabla calculada](screen_01/REPORT.md), [plan previo](PLAN.json) y [código](motor_port.py).
2. **Se evaluó una familia entera, sin barrer ganancias.** Sobre la señal DNb05 consumida durante el segundo de campaña 36, todo lector estático impar, monótono creciente y acotado a ±5°/s, con el baseline original, tiene integral de mando entre **−0,545° y 0°**. En antenas igualadas el intervalo es [0°, +1,675°]. El intervalo para la diferencia entre brazos con un mismo lector es [0°, +2,135°]. Un lector con otro offset, memoria o nuevas señales queda fuera de esta cota. Tampoco es una cota de yaw corporal o de una vida regenerada en lazo cerrado. No se eligió ningún umbral extremizador.
3. **Hay una carencia de control conjunto.** En los registros completos, los dos canales de avance DNg100 no varían y el mando de avance permanece en 0,2 mm/s por su término tónico. En un panel anatómico prefijado de 22 células, 17 tienen tasa publicada cero en ambos estados finales; incluye DNa02, DNa03, DNa11, DNg13, DNg100, DNp09 y LAL013. Son estados inicial/final, no una observación continua del silencio. DNa01 derecha y dos DNb02 derechas responden al contraste al final, además de las dos DNb05. Su actividad no autoriza sustituir el lector ni elegir células por éxito conductual. [Datos](population_01/POPULATIONS.csv), [selección anterior a la lectura](POPULATION_PLAN.json).

La tercera comprobación se motivó por circuitos descritos en [Braun et al., Nature 2024](https://www.nature.com/articles/s41586-024-07523-9), el preprint [A central steering circuit in Drosophila](https://www.biorxiv.org/content/10.1101/2024.06.27.601106v1.full) y la correspondencia [DNg100/BDN2 del atlas](https://www.virtualflybrain.org/term/dng100-fbbt_20007473/). No se trasladan sus tasas o efectos experimentales al modelo por analogía. Los valores publicados aquí son **Hz del modelo**, no mediciones en moscas vivas.

## Decisión estructural y alternativas

Se cierra esta normalización de reposo y la búsqueda de ganancias sobre esta vida. No se repite un segundo esperando que la misma dinámica navegue mejor. La etapa 4 necesita actividad de orientación sostenida y una relación útil entre giro y avance; luego la etapa 5 requiere separar feedback online de reproducción, con una perturbación alcanzable por el cuerpo.

- **A — interfaz motora identificada:** caracterizar la transferencia de actividad a movimiento con pulsos independientes de la tarea y un estado de reposo observable. Una eventual ley con memoria necesita justificación independiente; no basta cambiar el offset tras ver esta trayectoria. Falsador: deriva neutral o pérdida de transporte a lados/duraciones no empleados para identificarla.
- **B — reclutamiento y leyes de la población:** comprobar escalas, excitabilidad y entradas efectivas de poblaciones motoras documentadas, manteniendo su anatomía. La ausencia de activación puede ser legítima para este estímulo o deberse al modelo; los endpoints no lo deciden. Falsador: una supuesta corrección de unidades debe derivarse sin usar la dirección objetivo y conservar controles; no se bajan umbrales hasta obtener un giro.
- **C — avance y giro neurales coordinados:** identificar una interfaz conjunta mediante poblaciones descendentes/VNC, en lugar de atribuir navegación cerebral al avance tónico. No introducir bearing, posición de la fuente ni un planificador en el controlador. Falsador: retirar la señal neural de avance debe retirar su contribución, y su integración debe ayudar en geometrías no usadas para construirla sin perder el giro bilateral.

Estas propuestas de Codex ya están expuestas a las conversaciones anteriores; no se presentan como ideas ciegas o nuevas en la literatura. La siguiente decisión externa debe elegir un discriminador que produzca una implementación o descarte estructural, con horizonte acotado, no otra cadena de ajustes de ganancia. Como máximo dos prototipos nuevos por ronda. No se lanzó aún esa nueva campaña.

## Contraste externo y autocrítica

ChatGPT propuso la cota y entregó la función central original. Su [código preservado](chatgpt_bounds_original.py) se ejecutó **localmente** sobre los datos reales y coincidió con la implementación independiente de duraciones enteras; discrepancia máxima 1,34×10⁻¹⁵° por suma flotante. No se trató esa diferencia como bloqueo. ChatGPT no ejecutó estos NPZ en su entorno. [Recibo](CHATGPT_CODE_CHECK.json), [revisión recibida](CHATGPT_REVIEW.md).

Jev priorizó calcular la cota (confianza 0,86); fue clasificación consultiva, no verificación matemática ni admisión biológica. Un especialista Codex revisó el pipeline y detectó que el baseline se toma antes de los 40 ms de preparación. Eso forma parte del contrato ejecutado; no es un bug demostrado. El reset puntual empeora otros controles y no se promovió. [Revisión estructural](STRUCTURAL_REVIEW.md).

Correcciones expresas de interpretación: conservar concentración media y observar cambio de media neural **no demuestra no linealidad ni adaptación**; también puede producirlo un sistema lineal asimétrico. Una componente común idéntica se cancela en el lector diferencial inmediato. Además, el mando total oscila, pero el efecto pareado del contraste en DNb05 fue negativo en 999/1000 muestras y cero en la restante: no se demostró inversión periódica de la señal de olor. La revisión debe distinguir estos fenómenos antes de proponer una reparación.

Autocrítica: el giro breve validó una respuesta funcional limitada; no garantizaba orientación sostenida. El lector con escala arbitraria y el avance tónico son supuestos centrales del modelo, y merecen mayor prioridad que otra mejora decimal o cosmética del integrador. Un simulador preciso puede ejecutar con fidelidad un modelo insuficiente.

## Reproducción portátil y presupuesto

El subconjunto contiene siete arrays exactos de cada una de ocho trazas, las cuatro tasas publicadas completas y fuentes de las pruebas. `DATA.json` conserva hashes de los archivos originales y de cada array. No contiene sesiones completas para reejecutar el organismo, ni permite volver a demostrar sus 588 arrays preparados. El mapa anatómico seleccionado conserva la procedencia del parquet original; el verificador portátil comprueba filas y tasas, no vuelve a descargar el atlas.

Desde una extracción nueva, con Python y NumPy:

```bash
python3 verify_capsule.py --root . --out VERIFICADO.json
python3 -O verify_capsule.py --root . --out VERIFICADO_OPT.json
cmp VERIFICADO.json VERIFICADO_OPT.json
```

Los programas originales `screen.py`, `reader_bounds.py` y `population_screen.py` preservan rutas locales de procedencia; el comando portátil anterior no las usa. Reconstruye el rechazo de la normalización, las cotas y los endpoints desde arrays. La ejecución completa del organismo requiere las campañas de origen y sus dependencias; no se promete con este paquete.

Presupuestos previos: criba 180 s CPU/4 GiB, cota 30 s/1 GiB, poblaciones 60 s/2 GiB. Cada cálculo real terminó en menos de un segundo; la captura documental y revisión externa no están incluidas en esos tiempos de cómputo. GPU 0; 0 búsquedas de parámetros; 1 prototipo offline descartado. No hay corrida larga pendiente de esta campaña. Parada por discriminador completo; etapas 4/5 siguen abiertas.
