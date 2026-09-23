# Compatibilidad del prototipo B, antes de ejecutar

Usar el código recibido de ChatGPT con SHA256 a43e828372091f7895992bf638e29a2d4ef75927d4bc875e2ec62103aa8ec287 sin cambios. Una corrida CPU con sus límites60s/2GiB. Reconstruir errores desde arrays y distinguir métricas locales de las externas.

Dentro del mismo prototipoB, añadir un adaptador CPU puro del IR dimensional existente general_v2 a Bloque: suministra F sin resolver masa, M completa, salida y entrada. No modificar scheduler, modelo IR ni fuentes externas. Alcance declarado: un puerto escalar de entrada/salida, una población de una entidad por bloque, sin clamps; masa constante no singular dentro de cada bloque. No soporte implícito de masa entre bloques.

Prueba prospectiva adicional <=30s CPU: dos bloques de ecuaciones distintas, voltajes y concentraciones con unidades explícitas, masa no diagonal en ambos, realimentación cerrada y eventos. Comparar trayectorias iteradas y global DOP853 contra Radau con límite normalizado1e-4, sin modificar criterios tras ver resultados. Control de unidades inválidas debe rechazarse antes de ejecutar. Verificar que callbacks conservan port_bias, estado inicial e identidad de los modelos. No es un tercer prototipo ni prueba de aceleración; es un puente de compatibilidad y un falsador de doble aplicación de M^-1.
