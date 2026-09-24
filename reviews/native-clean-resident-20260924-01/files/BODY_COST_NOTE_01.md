# Coste observado del cuerpo en el perfil del organismo

Lectura sin nueva corrida del [perfil bruto salvado](../round_20260924_fast5s_02/noncns_profile_01/PROFILE.json) y su [verificación](../round_20260924_fast5s_02/NONCNS_SALVAGED_VERIFY_01.json). El segundo milisegundo sham del organismo tardó 3,082603 s bajo `cProfile`. La preparación efectiva usa la función `macro_body_advance` extraída del script de Gemini; no es el método base de `ContactRuntime` sin intervención.

| Función perfilada | Llamadas en 1 ms | Tiempo propio | Tiempo acumulado |
| --- | ---: | ---: | ---: |
| `macro_body_advance` | 40 | 0,001024 s | **0,041059 s** |
| `controller.torque` de prótesis de contacto | 40 | 0,002557 s | 0,017212 s |
| `mujoco.mj_step` | 40 | **0,005243 s** | 0,005243 s |

Los acumulados son **anidados**: `torque` y `mj_step` están dentro de los 0,041059 s y no se suman. Si el coste total de `macro_body_advance` desapareciera por arte de magia manteniendo igual todo lo demás, el máximo en **esa muestra perfilada** sería `3,082603/(3,082603−0,041059)=1,0135×`. MuJoCo C ya hace su física en ~0,00524 s de ese ms; mover sólo la llamada de Python a C++ no justifica una ganancia grande. `cProfile` perturba los tiempos, una muestra de 1 ms no es una estimación sostenida y el 0,041059 incluye control sintético, validaciones y MuJoCo. La razón de ingeniería para portar el cuerpo sigue siendo quitar fronteras y asegurar rollback dentro del coordinador, **no** porque `mj_step` aislado sea el cuello dominante observado.

La mayor deuda de velocidad sigue en recurrencia CNS y otros propietarios/fronteras fuera del cuerpo. El perfil corto previo separó 2,118685 s/ms propios de `NativeGraph.advance` y 1,033325 s/ms en el resto; este cuadro no descompone íntegramente ese resto ni permite sumar ahorros de forma independiente.
