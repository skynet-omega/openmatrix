# Solución densa por warp — entrega sin ejecutar

Se leyó completo, en sólo lectura, el donante
`/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor13_20260922/kc_fused_warp.py`
(SHA256 `b68af327632e836112563f72d0f8b8605353a9c7b273f43a84da95db23e197bc`).
Se extrajo únicamente su eliminación, sustitución y residual, líneas 35–54;
no se importó ni ejecutó el módulo ni su dependencia `kc_fused_step`.

## Interfaz y contrato

`neurocore::dense_warp_solve<N>(original_row, rhs, residual_tolerance)` devuelve
`DenseWarpResult {x, error, ok}`. El parámetro de tolerancia es obligatorio:
usar `1e-12` conserva el criterio del donante. `original_row` es un array
FP64 de N elementos privado del lane; se copia a espacio de trabajo y queda
intacto, igual que `rhs`. No hay escrituras globales ni compromiso de estado.
En fallo todos los participantes reciben `ok=false`, `x=NaN`, `error=+inf`.

Un sistema ocupa los lanes físicos `0..N-1` de un warp, `1<=N<=32`.
Todos esos lanes deben llamar a la función y a sus colectivas juntos; no puede
faltar ninguno. La máscara se deriva de N, siempre con ancho de shuffle 32.
Los lanes superiores pueden abstenerse o llamar: retornan sin participar.
La tolerancia debe ser idéntica en todos los participantes; se rechazan valores
distintos, negativos o no finitos. No se admiten máscaras arbitrarias.
Cada warp adicional puede resolver otro sistema independiente.

Compilar la unidad que incluye el header con `--fmad=false --prec-div=true`,
sin `--use_fast_math` ni reasociación aritmética. La función exige pivotes
estrictamente positivos y finitos; no pivota, regulariza ni aplica un fallback.
Recibe A completa: por ejemplo `A=(2/dt)*M+K`, con M **densa no identidad**;
ensamblar A y el RHS completo, incluyendo términos fuera de la diagonal de M,
corresponde al modelo/integrador llamante.

Se conservan los bucles aritméticos del donante y su residual FP64
`||A*x-rhs||inf / max(||A||inf*||x||inf + ||rhs||inf, 1e-300)`.
La reducción conserva el árbol `16,8,4,2,1`, pero usa una fuente activa propia
cuando no existe vecino; ese valor no se incorpora. Así no lee lanes inactivos.
El contrato de participación sigue las
[reglas CUDA de colectivas warp](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-sync-intrinsic-constraints).
Se añaden rechazos uniformes para entradas/soluciones/residuales no finitos y
overflow del denominador; este último no puede convertirse en un PASS con error
cero. Son guardas de fallo, sin cambiar la operación de una solución válida.

## Tres controles propuestos — NO ejecutados

1. **Generalidad y masa densa:** CPU construye M simétrica estrictamente
   dominante con términos fuera de la diagonal no nulos, K definida positiva
   y `A=(2/dt)*M+K`, con solución manufacturada y RHS completo. GPU prueba
   N=1, 3, 17 y 32, varios warps/sistemas y padding superior envenenado para
   N<32. CPU recalcula el residual desde los A/RHS originales y el x devuelto;
   debe ser <=1e-12, con `error/ok` idénticos entre participantes y entradas
   intactas. Para N=17, contrastar además x/error con la secuencia del donante
   aislada usando los mismos A/RHS y flags; igualdad bit a bit se propone,
   no se afirma demostrada. Ejecutar también con Compute Sanitizer.
2. **Rechazo uniforme:** una fila/pivote singular, diagonal negativa, NaN/Inf
   en una fila distinta de lane 0 o en su RHS; y tolerancia inválida/desigual.
   Esperar `false/NaN/+inf` en todos los lanes participantes, sin modificar
   entradas ni dejar que el llamante publique una solución parcial. Incluir
   A finita cuya suma de valores absolutos de fila desborde, para impedir
   normalización engañosa por infinito.
3. **Residual original y umbral efectivo:** una matriz densa bien condicionada
   con elementos no representables exactamente y pivotes positivos, N=5.
   CPU reproduce el orden explícito de eliminación y calcula A*x-rhs original;
   escoger antes del control un caso cuyo residual FP64 sea positivo y <1e-12.
   GPU debe aceptarlo a 1e-12 y rechazarlo al pasar la mitad del error positivo
   observado, sin cambiar A/RHS. Recalcular el residual a partir del sistema
   triangular sería un oráculo incorrecto para este control.

No se compiló, no se ejecutaron pruebas CPU/GPU ni se midió rendimiento.
Riesgos pendientes: una máscara incompleta viola el contrato y puede invalidar
las colectivas; el residual es error hacia atrás, no una cota de error de la
solución si A está mal condicionada; N=32 puede elevar registros/derrames.
Los nuevos rechazos conservadores ante overflow son deliberados. No se ha
integrado ni modificado el motor estable.
