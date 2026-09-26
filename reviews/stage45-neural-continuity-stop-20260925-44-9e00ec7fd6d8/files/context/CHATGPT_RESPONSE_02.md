**44 queda cerrada con reproducción del sham lograda, incumplimiento del presupuesto de CPU y efecto de la pareja no estimado.** Reviso únicamente tu resumen: no he leído NPZ ni verificado el selector PRO.

### 1. Alcance del resultado y clasificación

El sham aporta evidencia fuerte de **continuidad reproducible durante esos120ms**, desde esa restauración y utilizando el mismo wrapper: reproduce los35campos observados, incluido el intercambio sensorial y la aplicación física. Eso valida un requisito del experimento, **no la respuesta a las cintas**, porque `common` y `virtual` no se ejecutaron. Tampoco demuestra identidad de todos los estados internos en cada instante, más allá de lo efectivamente comparado.

La anchura Unicode distinta de `fase`, con texto idéntico, debe conservarse como **diferencia de representación**, sin llamarla divergencia dinámica ni afirmar identidad binaria total.

El exceso, calculado con tus cifras, fue **169,303591s de CPU, un5,64345%**. La clasificación apropiada sería:

**SHAM REPRODUCIDO · BLOQUEO POR PRESUPUESTO CPU EXCEDIDO · OMISIÓN DEL GUARD · CONTRASTE NO EJECUTADO.**

No es fallo de aislamiento demostrado, resultado neural nulo ni fracaso de orientación. La detención posterior evitó más gasto, pero no vuelve conforme el sham ya ejecutado. Conservar runner, recibos y corrección externa por separado mantiene esa distinción.

### 2. Tres rutas futuras, con funciones diferentes

**A — Acotar paralelismo en el motor original.**  
Es una hipótesis de ahorro, no un ahorro establecido. Fijaría previamente una configuración de hilos y una calibración breve con presupuesto propio cerrado, midiendo **CPU total y pared**, junto con fidelidad. Las variables deben establecerse antes de inicializar las bibliotecas y comprobarse dentro del proceso real; OpenBLAS distingue configuraciones pthreads y OpenMP. :chatgpt-content-reference{index="0"}

No asumir que pasar de14hilos a1 divide la CPU por14. Sólo admitiría una continuación futura si la configuración conserva la fidelidad exigida y resulta compatible con el coste previsto. El contraste seguiría siendo2°/100ms; la calibración no autoriza recortarlo ni repetir configuraciones hasta obtener un efecto neural.

**B — Guard externo de CPU agregada, sin cambiar el organismo.**  
Es la corrección directa de la omisión, **no una optimización**. Debe contabilizar desde el inicio todos los procesos e hilos del ámbito presupuestado, sin reiniciar el saldo entre sham y brazos. En Linux, `cgroup v2/cpu.stat` permite contabilizar CPU del grupo y sus descendientes; `cpu.max` limita consumo por período, no establece por sí solo un presupuesto acumulado. :chatgpt-content-reference{index="1"}

Lo validaría con cargas sintéticas multihilo y con hijos, sin CNS. Fijaría margen de corte para sondeo, terminación y cualquier guardado final: esperar a observar3000s no evita sobrepasarlos. **Este guard puede impedir completar el sham; no demuestra que la pareja quepa en presupuesto.**

**C — Motor11 como estudio separado de fidelidad y coste.**  
Exigiría su propia comparación preregistrada de restauración, relojes, entradas, ruido, estados neurales y mando; no bastaría parecido del yaw final. La validación y su coste pertenecerían a ese estudio. Sólo después podría plantearse el mismo contraste2°/100ms como experimento nuevo. No sería completar44 ni trasladar automáticamente el sham exacto del motor original.

### 3. Qué falta para atribuir CPU a los hilos

El sondeo posterior identifica una configuración en **otro proceso**, no el responsable del consumo histórico. Además, el número configurado en OpenBLAS es un límite de hilos, no una medición de trabajo realizado. :chatgpt-content-reference{index="2"}

Falta **contabilidad por hilo y por fase dentro del proceso del organismo**, acompañada de perfiles de ejecución que identifiquen las bibliotecas responsables. Un perfil con pilas de llamadas permite localizar dónde se ejecuta el trabajo, en vez de atribuirlo al mero tamaño del grupo de hilos. :chatgpt-content-reference{index="3"} Para demostrar que limitar hilos reduce el coste, haría falta además una comparación futura emparejada y fiel.

Tu cociente CPU/pared es aproximadamente **6,787**: si ambos contadores cubren el mismo ámbito, indica ese consumo agregado medio equivalente, **no identifica OpenBLAS,14hilos activos ni sobreparalelización**.

**Prioridad: corregir y verificar B sin CNS; A queda como hipótesis de ahorro y C como investigación independiente. Ninguna reabre44.**
