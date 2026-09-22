Message af09a96e-7f8d-4ffa-afb6-683fd2198a21
Documentary response; not an execution or a review of new code.

**Recomiendo un puente C++ pequeño con SUNDIALS 7.9.0, `N_Vector CUDA` y dos solvers SPGMR separados: uno para Newton y otro para la masa.** RHS, JVP, producto de masa y precondicionadores pueden utilizar vuestros kernels generados sobre memoria GPU. **No hace falta introducir Ginkgo para este primer enlace.**

Hay una novedad relevante: **SUNDIALS 7.9.0 incorpora soporte oficial CUDA/CuPy en `sundials4py`**. Sin embargo, la interfaz Python todavía no expone la configuración de las políticas de ejecución CUDA necesaria para controlar vuestro stream no predeterminado. Esto favorece ampliar ese enlace o crear un wrapper acotado, en lugar de usar `ctypes` directamente. :chatgpt-content-reference{index="0"} 

**Esta respuesta es investigación documental:** leí documentación, cabeceras y ejemplos oficiales; no compilé, ejecuté ni revisé las reparaciones nuevas de vuestro núcleo.

## 1. Ejemplo oficial que sí sirve de punto de partida

El ejemplo verificado en **v7.9.0** es:

`examples/python/arkode/ark_heat1D_cupy.py`

Utiliza exactamente estas piezas:

- `ARKStepCreate(None, f, …)` para integración completamente implícita.
- `N_VMake_Cuda` para envolver almacenamiento host y CuPy.
- `N_VGetCupyArray` para acceder a los vectores de los callbacks.
- `ARKodeSetLinearSolver` y `ARKodeSetJacTimes`.
- Escritura **in situ** del RHS y del producto Jacobiano–vector. 

**Sus límites importan:** utiliza masa identidad, PCG para su problema particular y `Device().synchronize()` al terminar los callbacks. No es una demostración de masa general, clamps o interconexión eficiente sobre un stream compartido.

Para la configuración del stream, el ejemplo C++/CUDA:

`examples/cvode/cuda/cvAdvDiff_kry_cuda.cu`

muestra `SUNCudaThreadDirectExecPolicy`, `SUNCudaBlockReduceExecPolicy` y `N_VSetKernelExecPolicy_Cuda`, aplicadas **antes de inicializar el integrador**. Esa parte pertenece a `N_Vector` y es reutilizable con ARKStep, aunque el ejemplo integre con CVODE. 

**No encontré un ejemplo oficial único que reúna todas vuestras condiciones.** El puente se obtiene combinando esas piezas y la interfaz de masa documentada.

## 2. Contrato matemático: qué entrega cada callback

Durante una época:

\[
M\dot y=f(t,y),\qquad M=\text{constante}.
\]

Para la ejecución completamente implícita, ARKStep construye sistemas de Newton con operador:

\[
W=M-\gamma J_f,
\]

donde **el `gamma` recibido en los callbacks ya contiene el factor temporal de la etapa**. No debe multiplicarse otra vez por \(h\). La interfaz distingue explícitamente Jacobiano y masa. :chatgpt-content-reference{index="4"}

Tres reglas evitan errores de integración:

**RHS devuelve \(f(t,y)\), no \(M^{-1}f(t,y)\).**

**JTimes devuelve \(J_fv\), no \(Wv\) ni \(M^{-1}J_fv\).**

**MassTimes devuelve \(Mv\), no la solución de un sistema con \(M\).**

Por tanto, el evaluador destinado a este backend debe exponer el RHS **anterior a la división o resolución de masa**. Aplicar \(M^{-1}\) en vuestro evaluador y registrar además la masa en ARKStep la incorporaría dos veces.

### APIs verificadas en las cabeceras de v7.9.0

| Operación | Registro C actual | Trabajo del callback |
|---|---|---|
| RHS implícito | `ARKStepCreate(nullptr, fi, t0, y, ctx)` | Evaluar el RHS acoplado. |
| Datos de la época | `ARKodeSetUserData(mem, datos)` | Mantener kernels, matrices, buffers y stream. |
| Solver de Newton | `ARKodeSetLinearSolver(mem, LS_W, nullptr)` | Configurar resolución matricial implícita sin ensamblar necesariamente \(W\). |
| JVP | `ARKodeSetJacTimes(mem, jtsetup, jtimes)` | Evaluar \(J_fv\), incluidas todas las conexiones. |
| Solver de masa | `ARKodeSetMassLinearSolver(mem, LS_M, nullptr, SUNFALSE)` | Resolver sistemas con masa constante. |
| Producto de masa | `ARKodeSetMassTimes(mem, mtsetup, mtimes, datos_masa)` | Multiplicar por el CSR de masa. |
| Precondicionador de Newton | `ARKodeSetPreconditioner(mem, psetup, psolve)` | Preparar/aplicar una aproximación de \(W^{-1}\). |
| Precondicionador de masa | `ARKodeSetMassPreconditioner(mem, mpsetup, mpsolve)` | Preparar/aplicar una aproximación de \(M^{-1}\). |

Los nombres compartidos actuales son `ARKodeSet…`; el constructor permanece `ARKStepCreate`. Las firmas están en `include/arkode/arkode_ls.h`. 

**`LS_W` y `LS_M` deben ser objetos diferentes cuando ambos son matrix-free.** Para comenzar usaría `SUNLinSol_SPGMR` en ambos. PCG para la masa solo corresponde cuando su simetría y definitud positiva estén establecidas; no debe deducirse de que la matriz sea constante. Además, `ARKodeSetUserData` debe preceder la configuración de los solvers cuando los precondicionadores lo necesiten. :chatgpt-content-reference{index="6"}

Las tres firmas fundamentales son:

```cpp
int fi(sunrealtype t, N_Vector y, N_Vector fy, void* datos);

int jtimes(N_Vector v, N_Vector Jv, sunrealtype t,
           N_Vector y, N_Vector fy, void* datos, N_Vector temporal);

int mtimes(N_Vector v, N_Vector Mv, sunrealtype t,
           void* datos_masa);
```

Cada callback debe usar **el tiempo y los vectores recibidos**, no el último estado confirmado del inspector. Las evaluaciones de Newton pueden utilizar estados de prueba que nunca serán aceptados.

### Precondicionamiento sin alterar ecuaciones

`psetup` recibe `jok`, `jcurPtr` y `gamma`; `psolve` recibe además RHS, destino, tolerancia `delta` y lado de aplicación `lr`. Sus firmas completas están en la misma cabecera. 

Mi propuesta inicial es un precondicionador diagonal o por bloques matemáticos pequeños, con dos cachés distintas:

\[
P_W\approx M-\gamma J_f,
\qquad
P_M\approx M.
\]

**Reutilizar información de \(J_f\) no implica reutilizar sin cambios \(P_W\)**: si cambia `gamma`, también cambia ese operador. El residuo y JVP completos conservan todas las conexiones que el precondicionador simplifique.

La masa constante permite reutilizar preparación de \(P_M\) durante la época. No elimina las resoluciones de masa que solicite el método.

## 3. Por qué prefiero el wrapper C++ a `ctypes`

Hay un motivo técnico concreto: en la cabecera v7.9.0, **`N_VGetDeviceArrayPointer_Cuda` es `static inline`**. No debe suponerse que exista como símbolo exportado que `ctypes` pueda localizar. Las políticas CUDA también son objetos C++, no estructuras cuyo diseño convenga reconstruir manualmente en Python. 

El wrapper debería poseer únicamente:

**Contexto SUNDIALS, vectores, integrador, ambos solvers, callbacks, stream y referencias a los propietarios de memoria.** La descripción del modelo y la generación de ecuaciones permanecen fuera.

Una primera implementación razonable es que el callback C++ entre en una función Python que lance los `RawKernel` ya generados. Eso mantiene vuestro compilador y permite comprobar el enlace. **No elimina el coste de cruzar Python/GIL en cada llamada**; si ese coste resulta relevante, posteriormente se puede lanzar el código generado desde C++ sin cambiar la interfaz matemática.

Las excepciones de Python/CUDA deben quedar registradas y convertirse en el retorno admitido por cada callback. No deben atravesar sin control la frontera C ni convertirse siempre en “éxito”.

### Stream único

Configuraría sobre el vector inicial, antes de crear los clones del integrador:

```cpp
SUNCudaThreadDirectExecPolicy politica_vector(256, stream);
SUNCudaBlockReduceExecPolicy politica_reduccion(256, 0, stream);
```

Después se aplica `N_VSetKernelExecPolicy_Cuda`. La forma de construcción anterior aparece en el ejemplo oficial. 

Los callbacks CuPy deben lanzar sus operaciones sobre ese mismo handle mediante `cupy.cuda.ExternalStream`, o utilizar directamente el objeto CuPy propietario del stream. `ExternalStream` no convierte un stream ajeno en propiedad de CuPy: su vida útil debe seguir controlada por el propietario. :chatgpt-content-reference{index="10"}

**No basta eliminar los `synchronize()` del ejemplo Python.** Antes hay que establecer ese orden común entre operaciones SUNDIALS, kernels generados y operaciones dispersas. Para comunicar errores de dispositivo al callback puede seguir siendo necesaria una espera del stream y una lectura escalar; eso debe medirse separadamente de transferir estados completos.

## 4. Memoria y vida útil: los riesgos concretos

**Los vectores de etapa no son siempre vuestro vector inicial.** En cada callback se obtiene el puntero del `N_Vector` recibido. Conservar una única vista de `y` para todas las llamadas produciría lecturas del buffer equivocado.

**La salida pertenece al vector que entrega SUNDIALS.** El kernel debe escribir allí; asignar una nueva variable Python a otro array no actualiza esa salida.

**Una vista no posee necesariamente la asignación.** `UnownedMemory` permite envolver un puntero externo, pero exige mantener vivo al propietario y declarar correctamente tamaño, dtype y disposición. El argumento `owner` conserva una referencia; no hace segura una dirección cuya biblioteca ya ha liberado o reutilizado el almacenamiento. :chatgpt-content-reference{index="11"}

Por eso propongo que las vistas de vectores internos no escapen de los callbacks, que la sesión retenga matrices/kernels/stream y que la destrucción espere el trabajo pendiente antes de liberar objetos. Si se usa `N_VMake_Cuda` con arrays CuPy externos, esos arrays también deben permanecer vivos durante toda la utilización del vector.

**Un snapshot del modelo no debe guardar punteros CUDA ni asumir que todos los temporales internos son estado físico.** El reinicio crea un integrador nuevo con la identidad numérica y el estado declarados; la reproducción exacta del historial adaptativo completo exige un contrato adicional.

## 5. Qué permanece en CPU

| Componente | Ubicación en el puente propuesto |
|---|---|
| Estados, evaluaciones RHS/JVP, CSR de masa y aplicaciones del precondicionador | GPU, mediante los callbacks. |
| Operaciones grandes de vectores Krylov | GPU mediante `N_Vector CUDA`. |
| Control adaptativo, decisiones de Newton, rechazos y coordinación de callbacks | CPU. |
| Operaciones escalares de GMRES y factorización QR del Hessenberg pequeño | CPU en SPGMR nativo. |
| Resolución completa con masa | No está obligada a ser CPU: SPGMR utiliza los productos y vectores anteriores. |

La separación general está documentada por SUNDIALS, y el código de SPGMR muestra tanto las operaciones `N_Vector` como el control y la QR del Hessenberg en host. **“Estado residente” no significa “todo el algoritmo de integración capturado en un grafo CUDA”.** :chatgpt-content-reference{index="12"} 

No cambiaría ahora de solver solo para eliminar esos escalares: primero mediría cuánto cuestan respecto de RHS, JVP y precondicionamiento.

## 6. Masa constante y clamps por época

Para clamps constantes, prefiero que el puente integre únicamente las coordenadas libres:

\[
M_{ff}\dot y_f=f_f(t,y_f,c).
\]

El compilador reconstruye el estado completo con \(y_c=c\) al evaluar ecuaciones y puertos. En JTimes, eleva el vector de prueba con **\(v_c=0\)** y devuelve únicamente las componentes libres.

Esto mantiene la semántica del clamp sin resolver primero el sistema completo y anular derivadas después. También evita perder innecesariamente simetría al sustituir filas de una masa originalmente simétrica.

**`SUNFALSE` significa constante dentro de la época, no permiso para modificar el CSR mientras el integrador conserva sus cachés.** Ante una mutación, recomendaría detenerse exactamente en el tiempo declarado, preparar el nuevo sistema libre y construir una instancia nueva del puente. `ARKodeSetStopTime` permite impedir que el integrador sobrepase esa frontera; pedir solamente una salida interpolada en ese tiempo no expresa lo mismo. :chatgpt-content-reference{index="14"}

Para el primer puente, reconstruir por época es más sencillo de verificar que implementar inmediatamente todos los caminos de redimensionamiento y reutilización.

## 7. Compilación y tolerancias: mínimo necesario

Opciones CMake verificadas para **v7.9.0**:

```cmake
CMAKE_BUILD_TYPE=Release
SUNDIALS_ENABLE_ARKODE=ON
SUNDIALS_ENABLE_CUDA=ON
SUNDIALS_PRECISION=double
SUNDIALS_INDEX_SIZE=64
BUILD_SHARED_LIBS=ON
```

Debe fijarse además `CMAKE_CUDA_ARCHITECTURES` según la GPU local. MPI, Ginkgo, KLU y LAPACK no son necesarios para el puente matrix-free descrito. Los componentes principales son ARKODE, NVECTOR CUDA y SPGMR; Newton puede configurarse explícitamente con el módulo correspondiente. Los nombres actuales `SUNDIALS_ENABLE_*` sustituyen opciones antiguas. **No he compilado esta configuración.** :chatgpt-content-reference{index="15"}

Mantendría dos controles distintos:

**`ARKodeSVtolerances` para los estados** y **`ARKodeResVtolerance` para las escalas del residuo cuando \(M\ne I\)**. No son intercambiables: estas últimas ponderaciones utilizan \(My\). Las tolerancias internas lineales y de masa también deben quedar registradas. :chatgpt-content-reference{index="16"}

Hay una diferencia de versión que merece quedar en el contrato: **7.9.0 corrigió un factor adicional de 0,1 aplicado a `ARKodeSetEpsLin`**. No deben copiarse ajustes de versiones anteriores suponiendo que tienen exactamente el mismo efecto. :chatgpt-content-reference{index="17"}

Asimismo, ARKODE no utiliza necesariamente la misma norma local que vuestro RK actual. Conservar las mismas cifras `rtol/atol` no implica reproducir sus decisiones de aceptación; la comparación debe conservar el criterio externo de error.

## 8. Alternativa acotada: CuPy GMRES

**Es una alternativa útil para comprobar el integrador implícito sin completar el enlace C++, pero no es un sustituto ya completo de ARKStep.**

El esquema sería un SDIRK con Newton y operador:

\[
R(Y)=M(Y-y_b)-h\,a_{ii}f(t_i,Y),
\]

\[
Wv=Mv-h\,a_{ii}J_fv.
\]

Ese \(Wv\) se proporciona mediante `cupyx.scipy.sparse.linalg.LinearOperator`. **El argumento `M=` de `cupyx.scipy.sparse.linalg.gmres` representa un precondicionador, no vuestra matriz de masa.** La firma revisada en CuPy 14.2.0 utiliza `rtol`, `atol`, `restart` y `maxiter`. 

La implementación revisada también:

- Mantiene los vectores grandes en GPU.
- Copia el Hessenberg a CPU en cada reinicio.
- Resuelve allí mínimos cuadrados con `numpy.linalg.lstsq`.
- Controla las iteraciones desde Python. 

Por tanto, **tampoco es un GMRES completamente residente sin sincronización host**. Habría que implementar y verificar el controlador temporal, Newton, rechazo, tolerancias anidadas y confirmación del estado. El `info=0` de GMRES no sustituye comprobar el residuo no lineal completo.

### Decisión de integración

**Primera opción:** ARKStep + NVECTOR CUDA + SPGMR, mediante wrapper C++ acotado y un solo stream. Utilizar el ejemplo oficial CuPy para comprobar la interoperabilidad, sin heredar sus sincronizaciones globales como diseño definitivo.

**Alternativa:** SDIRK–Newton con CuPy GMRES sobre los mismos RHS/JVP/MassTimes, si la integración del wrapper resulta desproporcionada. Eso sería el segundo prototipo, no una tercera arquitectura añadida.

Antes de incorporarlo al núcleo, la prueba pequeña decisiva debe detectar **masa aplicada dos veces, JVP sin conexiones, clamp incorrecto, reutilización del puntero de otra etapa y trabajo lanzado en un stream distinto**. Resolver esos contratos permite evaluar el coste del puente; todavía no demuestra rendimiento de un cerebro completo.
