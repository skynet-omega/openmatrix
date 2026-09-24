# Contrato de vistas y cruce con puertos — 24-09-2026

**Decisión:** conservar la corrección dispersa como primitiva numérica; no reutilizar deltas con el guard de tres tokens ni promoverla al operador completo. Etapas 4/5 siguen abiertas. El resultado publicado antes se recalculó en cada consulta, por lo que el defecto nuevo no cambia sus cifras; sí invalida la interpretación de que su guard era suficiente para una caché.

ChatGPT señaló un caso reproducible y entregó [código CPU de diagnóstico](chatgpt_original/overlay_vistas.py). Su SHA256 declarado coincidió con el texto extraído: `a27a45cdaa4597115435320f8ea0d8b958332176992843ee5553f52d128f0181`. Aquí pasaron el selftest normal y bajo `python -O`, así como la [cápsula real](capsule_01/capsule.npz) en ambos modos: error máximo `2,3646862e-11` frente al delta esperado, bajo el límite previo `1e-8`. Esto prueba sólo la primitiva de corriente por arista; el módulo externo no ejecutó CUDA ni compuso el operador entero.

El [falsador independiente](guard_collision_01/RESULT.json) construyó una arista con peso 2→3, liberación anterior 0,25 o 0,5 y posterior 1. El guard antiguo aceptó ambos estados con los mismos tokens de peso posterior, fuente posterior e historia; los deltas correctos fueron 2,5 y 2,0. Reutilizar el primero en el segundo erraría 0,5. La identidad de ambas vistas rechaza esa aplicación. Para componer eventos, el propietario debe recibir la vista ya proyectada por ellos: `C0 + (CE−C0) + (CO−CE)`. El contraejemplo sintético devuelve 3,0; usar el delta desde C0 después de sumar eventos devuelve 3,5.

La [sonda del grafo real](overlap_01/RESULT.json) encontró **cero** aristas compartidas y cero fuentes compartidas entre los 885.587 contactos de puertos fechados y las 140.489 aristas del overlay PN/APL/PNKC de este fixture. Su composición por tres canales coincidió con el cálculo directo a `2,27e-13` o menos; precisamente por la intersección nula, este bloque **no puede validar** el término cruzado evento-propietario para esos parches. La proyección q/s de dos consultas coincidió con el estado guardado a `7,53e-37` o menos. Los factores APL/PNKC siguen siendo sintéticos.

El cruce importante aparece **en el receptor**, donde opera el orden de los propietarios: [4.786 de 6.342](port_owner_01/RESULT.json) filas que reciben puertos tienen algún toque especial potencial en `target`, y 1.682 más de uno. Las 4.064 filas del propietario KC/APL dinámico reciben 647.588 aristas de puerto; las capas regional y PNKC reciben 340.313 y 288.298, respectivamente. Son incidencias estructurales, sin resolver banderas de activación ni valores escritos. Las capas visuales T4/retina no intersectan estas filas de puertos en este bloque, aunque siguen siendo necesarias para un motor general.

Con el estado corregido —incluido el negativo espacial previo de 94,17–99,57% de alcance topológico a cuatro saltos—, [Jev](jev_02/response.json) priorizó la captura de `target/rate` efectivos del propietario KC/APL (0,87 de probabilidad, confianza 0,81) y el orden de escritura como riesgo (0,68, confianza 0,52). Su primera consulta, que omitía aquel negativo, había priorizado partición espacial; **ninguno de los dos dictámenes ejecutó código o decide la arquitectura**.

La siguiente prueba de A debe capturar un bloque candidato A→B→A y comparar el `target/rate` que consume la dinámica con el oráculo completo, empezando por las 4.064 filas KC/APL. Debe registrar orden `set/add/transform`, escrituras parciales, tiempo, candidato, peso, evento y vistas antes/después por consumidor, además de las aristas y la pared total. Si no mantiene el error de estado y el presupuesto de seis barridos equivalentes, A cae en ese contrato. B, partición con residual recurrente, y C, runtime nativo transaccional exacto, permanecen alternativas. Una identidad de subgrafo o una votación externa no acredita 5 segundos en 10 minutos ni navegación.

Reproducción portátil de la porción que sí contiene la cápsula:

```bash
OPENBLAS_NUM_THREADS=1 python3 -B chatgpt_original/overlay_vistas.py --selftest
OPENBLAS_NUM_THREADS=1 python3 -O -B chatgpt_original/overlay_vistas.py --selftest
OPENBLAS_NUM_THREADS=1 python3 -B chatgpt_original/overlay_vistas.py --capsule capsule_01
OPENBLAS_NUM_THREADS=1 python3 -O -B chatgpt_original/overlay_vistas.py --capsule capsule_01
```

Las otras sondas requieren la captura local grande, identificada por hashes en sus planes. No se incluyen esos 91+123 MB en el paquete de revisión.
