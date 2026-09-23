# Revisión externa aplicada a las trazas reales

**Etapa 3 abierta.** ChatGPT propuso un programa de contraste por grupo y dos falsadores de retención espacial. [Fuente original](chatgpt_spatial_original.py) SHA256 `f6340e021c81ce24923127d023e3636cc7be4bb4f9fffdb0ebd5a2aff2eaf337`, igual al hash que informó ChatGPT. ChatGPT lo ejecutó sólo con un fixture sintético: no pudo descargar los NPZ públicos desde su entorno. Aquí se ejecutó en los NPZ reales de los cuatro brazos, sin nuevas simulaciones ni parámetros biológicos cambiados.

El [selftest original](chatgpt_spatial_selftest.json) pasó, pero su primera ejecución real produjo [FALLO.json](chatgpt_spatial_original_real_01/FALLO.json): exigía `sensors_used.shape==(335,2)`, mientras las trazas poseen `(335,3)`. [Corrección mínima](CHATGPT_SPATIAL_BUGFIX.json): validación a tres canales y tercer canal cero en el fixture; todas las ecuaciones, umbrales y contrastes permanecieron iguales. El [selftest corregido](chatgpt_spatial_fixed_selftest.json) pasó. [Fuente corregida](chatgpt_spatial_real_fixed.py), [resultado real](chatgpt_spatial_real_fixed_01/RESULTADO.json) y [tablas íntegras](chatgpt_spatial_real_fixed_01/GRUPOS.csv). El núcleo parcial ejecutado antes de recuperar el resto de la respuesta coincidió byte a byte en CSV, NPZ y hashes con la fuente completa corregida; se excluyó sólo `wall_s` del JSON.

El paquete de esta ampliación se extrajo en otro directorio y se combinó únicamente con la extracción verificada del paquete principal. El selftest, los CSV, todos los campos científicos del JSON y los arrays NPZ coincidieron exactamente; se excluyeron rutas absolutas y tiempo de pared. [Recibo portable](CHATGPT_SPATIAL_PORTABLE_VERIFICATION.json).

| Interfaz, media 161–311 ms | Contraste antisimétrico (izquierda−derecha)/2 |
|---|---:|
| ORN q L−R | +0,510775 |
| Entrada nativa DNa02 L−R | **−108,982240** |
| Target DNa02 L−R | −0,012005 |
| Estado DNa02 q L−R | −0,008597 |
| Mando aplicado | −0,015138 |
| Yaw absoluto, contraste entre brazos | −0,006034° |

El signo direccional se invierte **antes de la entrada DNa02** y no se recupera aguas abajo. El sham tiene un balance basal DNa02 L−R de +2.109,36 unidades; el componente común izquierda/derecha frente a sham suma +88,23. La entrada derecha−uniforme aún difiere +38,62. El protocolo heredado confirma que el primer intervalo de preparación consumió olor antes del blanco; la preparación no era neutral. Hay señal direccional invertida y sesgo de historia a la vez. Los contrastes son de una única preparación y una ventana ya examinada, no una cohorte independiente.

El ranking previo por `derecha−sham` destacaba CB0431|L, IB068|R y PS018_a|L. Sus contribuciones `derecha−uniforme` son **−2,91, −8,47 y −2,78**, respectivamente: gran parte de sus incrementos frente a sham también aparece con olor uniforme. El ranking antisimétrico sitúa CB0431|L (−33,20), SAD085|R (−15,42) y PLP228|L (−11,59) como contribuciones negativas principales. Los tres suman −60,21 de −108,98, pero hay 1.487 grupos y cancelación: no se identifica una sinapsis culpable ni conviene convertir el mayor término en manipulación de rescate. [Tabla completa](chatgpt_spatial_real_fixed_01/GRUPOS.csv), [interfaces](chatgpt_spatial_real_fixed_01/INTERFACES.csv).

Las dos reglas retrospectivas que mantienen constante una transmisión durante 1 ms fallan incluso en la entrada directa DNa02: `s≤0,01` tiene 29.504 decisiones fuente×ventana fallidas de 802.022 (3,68 %), con error máximo de transmisión 0,0105; `historia |Δs|≤10⁻⁴` tiene 3.532/954.315 (0,37 %), error máximo 0,00504. A 25 ms suben a 11,57 % y 3,14 %. [Todos los brazos y horizontes](chatgpt_spatial_real_fixed_01/HOLD_DIAGNOSTICO.csv). La cota reportada sólo cubre la suma ponderada en muestras guardadas, sin tiempos intermedios, recurrencia alterada ni resto del conectoma. Refuta estas políticas de **congelamiento**, no la integración espacial con solución analítica pasiva y cota global.

Decisión: priorizar A, una perturbación causal del **contraste antisimétrico** ORN→DNa02 con controles sham y uniforme, y B, una preparación inicial neutral emparejada. C, probar mandos espejo y cuerpo desde el mismo estado, sigue siendo control necesario antes de admitir orientación. El siguiente diseño debe congelar intervención, horizonte y presupuesto; no seleccionar un grupo sólo por esta tabla. El motor seguro permite diagnósticos acotados, pero el motor general y la comparación con moscas vivas siguen sin validación completa.
