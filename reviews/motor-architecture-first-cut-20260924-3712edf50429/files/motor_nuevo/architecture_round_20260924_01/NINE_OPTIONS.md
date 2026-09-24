# Nueve propuestas, cinco familias técnicas

Este inventario precede a cualquier promoción. G son ideas del texto de Gemini aportado por el usuario; C son propuestas Codex formuladas sin leer ese texto; T son las tres respuestas de ChatGPT en el turno `3cb364ec-b04d-4460-961d-7cc1bf7bb828`, con exposición declarada a la historia previa pero sin leer el nuevo texto Gemini. Son nueve **propuestas de procedencia**, no nueve arquitecturas independientes.

| ID | Operación causal o numérica | Familia, discriminador más barato |
| --- | --- | --- |
| G-A | Controlador/agenda persistente en GPU para no volver al host en cada ensayo. | **Compilar época/residencia**; pared total + estados/eventos en una época real. CUDA Graph ya existe y no basta por sí solo. |
| G-B | QSS2/3 con trayectorias cuantizadas y actualización sólo ante cambios. | **Multirritmo certificado**; motivo recurrente con evento tardío y defecto temporal. No hay cota global demostrada. |
| G-C | Resolver fugas por exponencial en un salto de 1 ms. | **Exponencial escalar**; falla si ignora entradas/eventos internos. El CNS actual ya usa exponencial por subetapa. |
| C-A | Mover agenda, aceptación y estado al dispositivo conservando ecuaciones. | **Compilar época/residencia**, parcialmente G-A; comparar cerebro+cuerpo y memoria. |
| C-B | Bloques espaciales según influencia y defecto temporal, con frontera recurrente. | **Multirritmo certificado**, parcialmente G-B; el negativo topológico previo obliga a medir el conjunto activo real. |
| C-C | Reutilizar propuesta nominal tras corte por evento, sin saltar la nueva prueba de error. | **Control de paso**; máximo ideal condicional 1,47× de pared total en la sonda 1 ms, insuficiente como ruta única a la meta. |
| T-A | Programa de época con propietarios, puertos, buffers y operaciones fusionadas sin cambiar método. Primera pieza: fusión de la relajación exponencial. | **Compilar época/residencia**, se solapa con G-A/C-A pero identifica además coste de operadores. Código CPU entregado, CUDA y organismo aún no medidos. |
| T-B | Método implícito acoplado `M-γJ`, JVP y precondicionador por bloques matemáticos. | **Integración implícita**; medir coste/residuo/eventos en un segmento real. Cero rechazos observados no sugieren por sí solos rigidez. |
| T-C | Multirritmo con waveforms, defecto y cota recurrente antes de publicar. | **Multirritmo certificado**, se solapa con G-B/C-B; probar el evento tardío y el coste de corregir fronteras. |

La evidencia medida exige ~52,53× para 1 segundo simulado/60 segundos reales frente al prefijo del organismo. El 67,22% atribuido a `NativeGraph.advance` incluye espera GPU; eliminar **toda** esa región hipotéticamente sólo daría 3,05× si lo demás permaneciera. El coste fuera de ella ya es 1,033 s por ms, 17,2× el presupuesto objetivo. Los 2.279 ms de intervalos CUDA frente a 2.181 ms de pared de llamadas son incompatibles como atribución exclusiva; no derivar un porcentaje de transferencia PCIe. La máquina observada es RTX 4070 Ti **SUPER de 16 GiB**, no la 4070 Ti de 12 GiB mencionada por Gemini.

**Selección provisional para medir:** T-A/G-A/C-A como una sola arquitectura de compilación de época y residencia, pero la pieza de fusión de ChatGPT es sólo una sonda de coste, no el motor nuevo. El prototipo completo necesita incluir propietarios PN/KC y cuerpo, o no puede explicar la pared. Mantener T-B y T-C como rivales reales con sus falsadores; G-C y C-C son controles/ideas limitadas, no un atajo a segundos por minuto. El modo rápido requeriría contrato de error funcional nuevo, y el modo preciso conservaría estados, eventos, masa PN y fronteras causales.

Jev recibirá una sola consulta de **clasificación y priorización asesoras** sobre estas opciones; sus probabilidades no promueven ninguna hipótesis. La selección final se hará con tiempo de pared y error desde datos crudos. Los criterios prospectivos están en `NUMERICS_GATE.md`.
