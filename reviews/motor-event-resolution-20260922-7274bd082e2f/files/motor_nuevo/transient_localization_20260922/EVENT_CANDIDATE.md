# Candidata A: resolución temporal explícita cerca de eventos

Limitación observada: las membranas cumplen estimadores de voltaje/compuertas, pero el detector heredado fecha máximos muestreados. La primera diferencia aparece en eventos con inputs y estado inicial idénticos; el filtro exacto reproduce la separación. Igual conteo final no conserva conteos intermedios.

Antes de programar/medir: mantener la dinámica y detector declarados, añadiendo al estimador de cada bloque un límite temporal de1562ns cuando el voltaje de observación somático o axonal, en estado comprometido o estados del ensayo, alcance el umbral existente de-40mV. La información es local y disponible en el ensayo; no se usa el resultado conductual, tipo celular ni datos de referencia. El límite no se aplica a bloques lejos de un posible evento. Un ensayo que lo exceda se rechaza antes de comprometerlo y se subdivide con el controlador existente.

Es una política numérica distinta, no un bugfix ni nueva fisiología. Se comparará con el padre independiente y la referencia global, más los techos uniformes6250/1562ns ya declarados, primero sobre inputs reales compartidos. No cambia tolerancias del estado ni parámetros de liberación. El número1562 es el techo de refinamiento previamente fijado, no elegido después de buscar un PASS.

Antecedente más cercano: integradores adaptativos con tratamiento explícito de eventos, por ejemplo la búsqueda de raíces de CVODE/SUNDIALS. Esta guardia NO implementa búsqueda de raíces ni garantiza que no exista un pico oculto entre todas las muestras. El detector sigue siendo muestreado; no atribuirle equivalencia a un detector continuo.

Predicción falsable: se reducen las diferencias temporales y de filtros frente a la referencia refinada conservando el conteo total, sin imponer el techo uniforme a células subumbrales. Control: mismos1557estados/inputs reales y filtro independiente, con padre/global/refinamientos. Coste de todos los bloques incluido. Si no mejora o excede el contrato, no promover ni elegir otro umbral para rescatar.

Candidata B reservada: separar propuesta nominal CNS y recorte por frontera, manteniendo fronteras y rechazo por error. Solo evaluar después de aclarar historia de eventos; no corregir el transitorio cambiando el integrador de un filtro que ya es exacto.
