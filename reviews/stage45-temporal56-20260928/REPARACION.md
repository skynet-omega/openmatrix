# Reparación 01 — metadatos antes de cualquier paso CNS

El primer trabajador restauró el estado48 y falló al escribir la descripción de la intervención: el escritor heredado usa la codificación por defecto y la descripción contenía `→`. El entorno de ese proceso usabaASCII al escribir. También falló al guardar el error por contener el mismo carácter. Se preservan traceback, archivos `.tmp`, estado inicial y cola originalSTOPPED; no se completó ni intentó un pasoCNS según la ubicación del fallo y el prefijo del recibo parcial.

La reparación modifica sólo el escritor de los nuevos metadatos: UTF-8 explícito, el mismo JSON, rechazo de no finitos, fsync y reemplazo atómico. El runner y supervisor congelados se importan mediante wrappers hacia un directorio nuevo. No se modifica una ecuación, propietario físico, criterio, ventana, valor fuente ni módulo histórico. Los estados iniciales no se reconstruyen a mano.

La prueba mínima reproduce el fallo bajo localeC, comprueba el recorridoUnicode y verifica que un no finito no sustituye el archivo anterior, en Python normal y `-O`. No es una cualificación viva; ambas cualificaciones originales siguen pendientes y se repiten antes de los ocho brazos.

Coste conservador del intento sin recibo completo: reservar170sCPU y2ms de la partida prevista, además de62.129142s reales de cola. Son cargos conservadores, no170sCPU medidos ni2ms realmente integrados. El recibo parcial muestra0intentos/0comprometidos. El techo de ronda se mantiene1200ms/6000sCPU/5000scola; A reparada usa como máximo1198ms/5030sCPU/4637.870858scola. Los1028ms científicos planificados permanecen iguales y los criterios originales se conservan exactamente.
