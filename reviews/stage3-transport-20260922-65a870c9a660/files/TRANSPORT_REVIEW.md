# Transporte del organismo a ejecución nativa

Campaña local del 22-09-2026. La etapa 3 sigue abierta. Estos son resultados de
transporte de 5 ms, no una certificación del segundo completo ni de fidelidad
biológica. El piloto largo está separado y sus resultados se publicarán cuando
termine. No hay eliminación de sistemas para conseguir velocidad.

## Resultado

El controlador C++ reproduce el operador completo CNS, con 166.700 neuronas,
359.373 variables globales, membranas PN/KC, eventos, retardos y cuerpo MuJoCo.
Conserva la partición de intercambio de 125 us, ya ensayada en MOTOR14, y usa
puertos de eventos analíticos en cada etapa del método. Las membranas PN de
178.838 coordenadas retienen su masa no diagonal y química de calcio.

En sham, la ejecución CNS nativa coincide exactamente en todas las hojas del
estado, trazas y cuerpo con MOTOR14 a igual partición. La ejecución PN residente
añade diferencias máximas de 4,97e-14 mV, sin cambiar CNS ni cuerpo. En izquierda
frente a referencia fina corregida: PN 0,000334089 mV; compuertas 6,39713e-5;
estado global 5,45801e-5; yaw 1,48646e-10 grados. Conteos de eventos, relojes,
metadatos y banderas coinciden. Los límites no se modificaron para aprobar.

`verify_transport.py` reconstruye la comparación desde arrays y comprueba además
la función futura de las historias retardadas: distintas segmentaciones no son
estados idénticos ni un error automáticamente. Compara cobertura, continuidad,
relojes y la unión de puntos de interpolación. El mayor error de historia es
2,34488e-5; no otorga una cota de error para un segundo.

## Corrección importante de alcance y velocidad

Los 35,6 s por segundo del motor genérico anterior corresponden a una carga
sintética. El organismo completo cuesta aproximadamente 2,65 s por milisegundo
al comienzo de este ensayo. No se ha demostrado aún segundos en minutos para
esta preparación. El perfil real muestra ahora predominio de las membranas KC.

Se probó ensamblado de matrices con cuBLAS y soluciones por lotes: conserva la
trayectoria corta, pero es más lento; queda descartado como mejora de velocidad.
Se incluyen sus fuentes y fallos de integración. La llamada nativa usa las
[interfaces documentadas de cuBLAS](https://docs.nvidia.com/cuda/archive/12.2.2/cublas/index.html),
porque la versión instalada de CuPy bloquea su envoltorio BLAS durante captura.
No se altera el criterio de residuo para usar esta alternativa.

## Revisión solicitada

Revisar `organism_adapter.py`, `graph_core.py`, `graph_control.cpp`,
`event_ports.py`, `pn_execution.py` y `verify_transport.py`. Buscar capturas con
datos obsoletos, modificaciones del dueño de estados, pérdidas de eventos,
rollback incompleto y comparaciones que acepten diferencias estructurales.
Los estados y las condiciones son reales; no afirmar reproducción sin ejecución.

La rama conserva las intervenciones de ingeniería del controlador y la igualdad
de tau/theta PN-DNb05 heredadas, sin reescalado ORN. No son leyes fisiológicas.
Las hipótesis rivales para etapa3 son pérdida de contraste aferente, estado basal
y lectura motora/cuerpo. No se ajustará la fisiología para cambiar el veredicto.

Este paquete corto sirve para inspección y verificación de arrays. La ejecución
completa todavía necesita el checkpoint y los recursos del organismo local,
identificados en sus recibos; no se anuncia como distribución independiente de
todo AXIOMA_FLYWIRE.
