# Criba del RHS congelado, 25-09-2026

Sobre las 60 consultas reales de un bloque CNS de 125 µs se reconstruyó `F = rate × (target − z)` y se comparó cada consulta con dos reglas de orden cero: conservar `F` de la primera consulta del bloque, o tomar la primera consulta observada de cada intervalo entre eventos. El indicador es `h·|F_i−F_ancla|/[3(atol+rtol·max(|z_i|,|z_ancla|))]`, máximo sobre coordenadas. No es el error de un integrador nuevo ni una cota de trayectoria.

| Regla | Consultas con indicador >1 | Mediana | Máximo |
|---|---:|---:|---:|
| Una sola ancla para todo el bloque | 42/60 | 3,357 | 11,916 |
| Un ancla por cada intervalo entre eventos | 13/60 | 0,173 | 5,375 |

La congelación ingenua del RHS no tiene aquí una perturbación local uniformemente pequeña. Incluso la segunda regla necesitaría ocho evaluaciones globales para ocho intervalos, antes de auditoría, y no cumple el objetivo previo de ≤6 por bloque. Esto **no descarta** multirritmo con corrección recurrente, un proveedor de eventos disperso, Krylov o métodos implícitos. Tampoco prueba inestabilidad: las diferencias se midieron sobre estados que el padre ya consultó, sin avanzar una candidata.

El plan44 fijó los datos, fórmula y presupuesto antes de ejecutar. El cálculo tomó2,038s/120s, sin nuevos organismos. Un verificador independiente recompuso las dos series desde el archivo completo local de1.030.765.558bytes, pasó normal y `python -O`, y rechazó una alteración de recuento. El archivo completo no se duplica en este paquete; su SHA-256 está en el plan. La cápsula de liberación de la misma corrida y el operador completo parcial están publicados previamente en OpenMatrix; no permiten recomputar este indicador sobre todas las coordenadas.
