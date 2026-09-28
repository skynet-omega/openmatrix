# Datos y alternativas que pueden cambiar la siguiente decisión

Estas propuestas complementan los dos instrumentos ejecutados en 56. No son una tercera candidata integrada ni una nueva declaración de admisión. Se distingue acceso a un índice, acceso a series numéricas y validación de un mecanismo.

| Decisión | Evidencia localizada | Disponibilidad comprobada aquí | Qué falta antes de usarla |
|---|---|---|---|
| B: cinética continua ORN→PN | Nagel 2015, modelo continuo y curvas de figuras 7/8; parámetros y texto ya locales | Banco matemático ejecutado; no se recuperó nueva serie fisiológica numérica | Emparejar entrada temporal, preparado e inhibición; decidir observador PN antes de ajustar parámetros. |
| B: modulación presináptica | Raccuglia 2016, contraste de bloqueo GABA en terminales olfativas señalado por ASTRA_V2 | El asesor localizó figuras, no archivo experimental numérico. La apertura directa de eNeuro desde esta sesión devolvió 403 | No transferir DM2 a DM1 ni equiparar voltaje ArcLight a q. Una digitalización tendría incertidumbre y procedencia propias. |
| B: correspondencia de observable | Bhandawat 2007, tasas ORN/PN por glomérulo y olor | Artículo/suplemento; nueva serie original no recuperada | q no está calibrado como tasa. Comparar formas exige explicitar la función de observación. |
| C: estado de red descendente | Braun 2024 y código oficial `NeLy-EPFL/dn_networks` | Índice GitHub recuperado: 144293 bytes, commit `b9e66203ab825187e5d9bd651a63e46c2d5f0ab0`; no aparecieron archivos summary/CSV en ese árbol | Dataverse INYAYV devolvió 403. No se verificó ni descargó una mosca `*_novideo.tar.gz`; tamaño, cobertura celular y condición siguen sin comprobar. |
| C: respuesta celular DNg100/BDN2 | Sapkal 2024 y correspondencia MaleCNS 10045/10056 | Activación/silenciamiento respaldan función; identidad coincide con las células observadas | No hay aquí una curva entrada controlada→membrana/espigas de BDN2 con la que identificar theta/gain/tau. |

Referencias: [Nagel](https://doi.org/10.1038/nn.3895), [Raccuglia](https://doi.org/10.1523/ENEURO.0080-16.2016), [Bhandawat](https://doi.org/10.1038/nn1976), [Braun](https://www.nature.com/articles/s41586-024-07523-9), [código Braun](https://github.com/NeLy-EPFL/dn_networks), [dataset de imagen](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/INYAYV), [Sapkal](https://www.nature.com/articles/s41586-024-07854-7).

## Discriminador corregido con los asesores

ChatGPT_Motor_V2 prioriza contexto de red DN, seguido de estado intrínseco y propiocepción. La explicación de contexto debe poder cambiar la entrada efectiva a DNg100. Fijar durante toda la vida esa entrada completa y todos los estados locales haría la salida determinista idéntica por construcción. Se corrigió conjuntamente: fijar estado inicial DNg100 y entrada ORN; intervenir sólo un contexto DN predefinido por evidencia independiente; medir la entrada consumida, el margen y el target resultantes. Una cinta igualada es un control secundario de mediación. No elegir el conjunto después de observar cuál produce movimiento. Esta prueba aún no se ejecutó.

ChatGPT ASTRA_V2 propone restricciones temporales para cinética, inhibición presináptica y correspondencia de observable. Se corrigió que normalizar amplitud no elimina no linealidades de observación ni identifica un mecanismo único. Sus respuestas finales distinguen figura digitalizable, datos originales no localizados y disponibilidad por petición no verificada. No convertir la dirección del autor correspondiente en una política de disponibilidad de datos.

El criterio propio es conservar una preparación común y separar un cambio de respuesta funcional de equivalencia biológica. La revisión no demuestra que falten todos los datos ni exige construir un nuevo escáner. La información local permite pruebas causales del modelo; la equivalencia de sus parámetros con biología requiere los contrastes específicos anteriores. No se descargó ninguna colección masiva, no se accedió a credenciales y el error 403 se conservó en `research/ACQUISITION.json`.
