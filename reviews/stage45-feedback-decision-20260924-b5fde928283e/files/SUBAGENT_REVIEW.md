# Revisión independiente recibida

Especialista `/root/stage45_independent_decision`, encargo explícito del usuario. Trabajo de sólo lectura; cero GPU, cambios de archivos y subdelegación. Este documento sintetiza su respuesta; no es una transcripción literal.

Reconstruyó NPZ y PROGRESS de las campañas 26/27. Confirmó signos finales opuestos +0,04491418° y −0,01240634°; corrigió que 0,05732052° es la diferencia entre brazos. Identificó avance exactamente 0,2 mm/s, inversión tardía del mando izquierdo y aumento descriptivo de error de rumbo 17,34885→18,55769° y 18,37685→19,72289°. Codex recompuso esos resultados por separado en RAW_FEASIBILITY_01/02.json. No sostiene fracaso de una trayectoria larga aún no corrida.

El mando ±5°/s permite como máximo 2,5° integrados tras una perturbación a1,5s en un ensayo total de2s, o7,5° en uno de3s. No es un límite de rotación pasiva. El dueño corporal rechaza `xfrc_applied` no declarado y sobrescribe `qfrc_applied`; el puerto de perturbación física necesita implementación explícita. El contrato gaussiano dice `resumable_gaussian=False`; no asumir ramificación validada por tener checkpoint serializado.

Su control yoked conserva cuerpo, fuerza de perturbación y autoridad motora. Sólo cambia antenas actuales por cinta olfativa donante; desactivar el timón confunde capacidad motora con feedback. El cuerpo actual usa una prótesis de contacto/rodillos y el vídeo sólo puede representar lo simulado.

Tres rutas independientes: A, ensayo largo físico con donante/online/replay y perturbación viable; B, discriminador corto por cambio de fuente, online/cinta previa/sham; C, motor primero con MRI/QSS/Krylov y coste integral. Recomienda B antes de A y continuar C de forma acotada. B revela adaptación sensorial; no basta para declarar autocorrección corporal ni navegación. Estimó ~5,6h de integración para tres brazos nativos de2s antes de preparación/cierre; el cálculo local posterior incluye ese coste y da5,80–5,83h.

Fuentes comprobadas: `campanas/etapa4_mirrored_source_20260924_26/native_{plus,minus}_01/{traces.npz,PROGRESS.jsonl}`; equivalentes estrictas27; `campanas/etapa4_reference_budget_20260924_27/run_mirror.py:261`; histórico de sólo lectura `/home/daroch/AXIOMA_FLYWIRE/matrix/gemini_work/test_self_calibrating_embodied.py:52` y controlador `work/stage2_contact_prosthesis_20260915/controller.py:110`.
