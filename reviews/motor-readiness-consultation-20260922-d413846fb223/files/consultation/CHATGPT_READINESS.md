Turno c479fb22-15c0-4c1d-9a61-3e210f7031cc. Respuesta directa a la consulta de admisión; revisión documental.

**1) ¿Listos para una campaña interpretativa de etapa3? NO.**  
No estoy conforme con la suficiencia científica demostrada del motor. Los acuerdos cerebrales a 1/5 ms son parciales; la referencia refinada falla durante el milisegundo 15 y el contraste antiguo a 20 ms excede \(10^{-4}\), sin constituir una referencia fiable. No existe un segundo completo medido. Mi revisión fue documental: **no ejecuté arrays ni validé independientemente esos resultados**.

**2) Diagnósticos que sí autorizaría.**  
Reconstrucción y replay de puertos; fixtures de umbral, dominio y rollback; reproducción instrumentada del fallo; convergencia en ventanas acotadas y medición del coste por componente. También contrastes numéricos B/C con presupuesto cerrado. Son pruebas del motor, **no evidencia conductual interpretable**. No reabriría automáticamente las seis cargas agotadas.

**3) Evidencia mínima para pasar, sin auditoría infinita.**  
Fijaría una única ronda y estos requisitos científicos:

- **Explicar o aislar reproduciblemente el fallo real de dominio**, capturando índice, valor y operación responsable; justificar cualquier corrección sin clamp. La violación inducida solo comprueba el rechazo/restauración.
- **Completar una trayectoria acoplada del horizonte que se interpretará**, incluyendo cerebro, puertos, eventos y cuerpo, frente a una referencia refinada válida con evidencia de convergencia. Exigir error continuo ≤\(10^{-4}\), con la normalización acordada, y criterios temporales/discretos fijados previamente. Si se interpretará un segundo, 20 ms no bastan.

**Velocidad:** alcanzar 1 s simulado/60 s reales es una meta operativa, no condición de fidelidad. Sí debe poder completarse la campaña dentro de su presupuesto. B/C no son requisitos obligatorios. Jev puede clasificar requisitos, no certificar su cumplimiento.
