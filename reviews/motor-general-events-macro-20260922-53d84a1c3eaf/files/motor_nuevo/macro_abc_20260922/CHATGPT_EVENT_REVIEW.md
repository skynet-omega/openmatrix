# Revisión documental externa

Conversación 6ab06db7-9908-83e9-a515-58c9e6e18a1a, respuesta 9a7d48a0-60d1-4f0c-9b59-5fe73a59e442. Sin ejecución ni inspección de las fuentes nuevas. No equivale a auditoría independiente.

**Probaría A como reparación del contrato de eventos, no como reforma de rendimiento.** El recálculo que comunicas indica que los operandos almacenados ya describen un resultado superior a uno incluso con mayor precisión: el problema no es exclusivamente `exp` en GPU. La transformación del estado posterior en una diferencia aditiva perdió una propiedad de la transición original. Eso no justifica aflojar el dominio ni la tolerancia.

### La condición para que SET sea una reparación fiel

**`SET(event_value)` debe representar el estado inmediatamente posterior al evento, en su marca temporal declarada.** No un valor al final del intervalo ni una corrección aplicada después de descubrir una infracción.

Si el modelo original produjo exactamente \(q^+=1\), transportar ese resultado conserva la transición; no introduce un clamp nuevo. Pero una transición condicional, como una saturación dependiente de \(q^-\), **no equivale universalmente a `SET(1)`**. El evento y su valor deben regenerarse cuando cambie la trayectoria de la fuente, incluidos predictores o iteraciones recurrentes. No deben reutilizarse desde una historia anterior para imponer su respuesta.

### A, B y C pertenecen a niveles diferentes

**A — ADD/SET:** representación mínima de dos operaciones explícitas. Es suficiente para comprobar esta reparación, no para declarar cobertura de cualquier transición.

**B — mapa del modelo:** expresa la semántica general \(x^+=R_e(t,x^-,p)\). ADD y SET son casos particulares. Permite transiciones dependientes del estado sin convertirlas en valores constantes obtenidos de otra ejecución.

**C — evaluador propiedad de la fuente:** resuelve quién calcula la historia y quién puede consultarla. Puede utilizar A o B; no es necesariamente una tercera representación rival. Sus consultas de etapas rechazadas no deben confirmar eventos ni modificar retrospectivamente otras consultas.

Por tanto, **no construiría tres implementaciones completas para este defecto**. Probaría A con una interfaz compatible con B; mantendría C como decisión explícita de propiedad.

### Tres falsadores pertinentes

1. **Saturación y evento no saturado:** conservar respectivamente el valor posterior físico y la transición aditiva, también al dividir el intervalo.
2. **ADD y SET simultáneos:** respetar el orden declarado —no conmutan— y obtener el mismo resultado tras rechazo/reintento, sin doble aplicación.
3. **Dominios heterogéneos:** admitir estados negativos o superiores a uno cuando su contrato lo permita, y rechazar infracciones únicamente del dominio correspondiente. No heredar una comprobación global \([0,1]\).

**Esta reparación cerraría una pérdida concreta de semántica; no explicaría ni resolvería el coste global del CNS.** Los candidatos macro permanecen separados. No he examinado las fuentes nuevas ni reproducido el diagnóstico: esta valoración se basa en los datos que comunicas.

