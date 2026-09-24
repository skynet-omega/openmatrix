# Propietarios del operador: coste y corrección dispersa — 24-09-2026

**Resultado acotado:** una representación algebraica genérica de parches de pesos y sustituciones de liberación pasó en CPU y CUDA sobre el CSR real de 166.700 neuronas y 25.582.938 aristas. No es aún el `coefficient(z)` completo ni un motor acelerado; no cambia las etapas 4/5.

La [primera criba](run_01/RESULT.json), con [umbral previo](PLAN.json), halló 26.044 filas receptoras afectadas por propietarios especiales. Una relectura completa de esas filas recorre 3.901.860 aristas (15,25% del CSR); cabe una vez en el margen nominal de la propuesta multirritmo de seis barridos, pero dos relecturas ya lo excederían. Esto sólo descarta la repetición ingenua del respaldo por fila dentro de ese presupuesto, no las reglas especializadas ni otras arquitecturas.

La alternativa probada representa los cambios de las 6.474 posiciones de peso PN/APL y las sustituciones de 301 fuentes como una **corrección por arista** sobre una corriente base. El conjunto unión tiene 140.489 aristas (0,549% del CSR) hacia 14.016 receptores; los parches de pesos están contenidos en esas aristas. La corriente ordinaria se corrige con signo y los canales visuales excitador/inhibidor por separado. El algoritmo usa, en cada arista alterada, `w_nuevo·s_nueva − w_base·s_base` antes de acumular en su receptor, preservando los términos cruzados cuando cambian peso y fuente a la vez.

| Comprobación | Resultado | Alcance |
| --- | ---: | --- |
| [CPU, dos estados capturados](run_02/RESULT.json) | Error máximo 1,019×10⁻¹⁰; tres versiones obsoletas rechazadas | Comparación con CSR completo bajo factores estructurales del fixture |
| [CUDA, A–B–A](run_04/RESULT.json) | Error máximo 2,365×10⁻¹¹ frente a CPU; A restaurado bit a bit; 12 MiB VRAM adicional observada | Sólo corriente de la corrección, sin medir velocidad integral |
| [Cápsula portátil](capsule_01/RESULT.json) | 5,55 MB; reconstrucción independiente y dos corrupciones detectadas, también con `python -O` | Datos reducidos para revisión externa; no contiene checkpoint ni todo el operador |

Se usaron **factores PN generales reales** del bloque capturado. Los factores APL y la sustitución PNKC de estas pruebas son **sintéticos y declarados**: comprueban la identidad del algoritmo, no la identidad del operador biológico real. Faltan los propietarios que sobrescriben `target/rate` (PN, APL, KC, retina, T4 y otros), la coexistencia exacta con los 885.587 contactos de puertos fechados, invalidación de pesos/eventos durante un bloque, y paridad de estado/espigas/cuerpo en una trayectoria candidata. Un programa que sólo sume este delta al CSR base sería incompleto.

La primera compilación GPU [falló antes del primer kernel](run_03/FAILURE.json) porque NVRTC no resolvió una cabecera glibc. Se conserva el [código inicial](owner_overlay_v1.cu). La [reparación prospectiva](REPAIR_PLAN_04.json) quitó sólo dos inclusiones no necesarias; el kernel y las puertas matemáticas no cambiaron. El tiempo de 2,60 s del recibo CUDA incluye preparación/compilación y no es tiempo por milisegundo del organismo.

El siguiente falsador de la ruta A es reconstruir `target/rate` **efectivamente consumidos** en estados candidatos, con dueños activos y puertos fechados, usando este delta donde corresponda y callbacks versionados para sobrescrituras. Si exige relecturas completas adicionales que rompan la cuenta de aristas, la ruta A cae en este contrato. B (partición/compresión con residual) y C (runtime nativo exacto) permanecen rivales. Ninguno recibió un resultado de velocidad integral aquí.

Verificación portátil desde una extracción limpia con NumPy, sin el proyecto original:

```bash
python3 -B verify_capsule.py --capsule capsule_01 --out VERIFY_NUEVO.json
python3 -O -B verify_capsule.py --capsule capsule_01 --out VERIFY_NUEVO_O.json
```

Los [planes](PLAN_02.json) y [recibos](run_04/RESULT.json) conservan hashes de entradas y fuentes. La cápsula trae sólo el subgrafo pertinente y salida esperada; ejecutar su verificador no reejecuta la mosca ni valida una etapa.
