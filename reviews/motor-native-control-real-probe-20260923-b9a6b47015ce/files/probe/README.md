# Tiempo dentro del controlador C++ sobre el organismo

La sonda propuesta por ChatGPT separa intervalos del stream CUDA y llamadas del host. El generador original está en `generar_sonda_original.py`; sus archivos generados y compilación real contra CUDA12.6 están en `generated/` y `BUILD.json`. No se modificó la biblioteca de producción.

El [plan](PLAN.json) compara padre, instrumentación sólo de host/decisiones y marcadores CUDA, con1ms de organismo por condición. Primero debe comprobar estados científicos y eventos idénticos. Compara también cada decisión adaptativa entre ambos modos instrumentados. No promueve una optimización: ayuda a elegir A, controlador residente; B, integración espacial con error acotado; C, integración implícita acoplada. Un fallo de neutralidad invalida atribuir costes a la versión de producción.

`run_after_campaign.py` espera el cierre del PN629 antes de cargar otro organismo. El máximo es tres cargas,720s de runner y ningún reintento automático. `RESULT.json` se escribe sólo después de terminar esas cargas y verificar neutralidad. La ejecución acotada quedó encolada desde el entorno local; este README no inicia nada.

El intervalo de grafo **no es tiempo exclusivo de kernels**: incluye dependencias y planificación dentro del grafo. Host y stream se solapan; no sumar sus columnas como componentes independientes. Un único milisegundo inicial tampoco estima toda una vida de400ms ni el futuro sistema visual. Se registra la perturbación temporal de instrumentar.

ChatGPT leyó fuentes públicas y entregó código. Declaró pruebas con un mock de CPU, no con nuestra GPU. La ejecución local real y su evidencia deben mantenerse separadas de esa declaración. La fuente padre está fijada porSHA256 en el generador, y los hashes de bibliotecas y del arnés se guardan por carga.
