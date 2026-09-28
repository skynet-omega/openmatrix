# Contexto descendente: discriminador previo a otra vida CNS

28-09-2026. Propuesta propia anterior a nueva revisión externa y a los cálculos de retirada. Conozco las tablas de capacidad49: C1/C2 tienen cota positiva y aporte observado negativo. Eso no identifica reclutamiento ni justifica saturar transmisiones. Los resultados están expuestos; ésta es exploración mecanística del modelo, no confirmación biológica.

## Tres explicaciones y decisión

1. C1: la recepción inhibitoria del conjunto descendente externo mantiene cerrado el target de DNg100. Discriminador barato: retirar sólo esa recepción negativa, conservando la positiva y el resto del consumidor. Si aun esa retirada total mantiene el margen no positivo en todas las evaluaciones guardadas, no lanzar un CNS que sólo repita ese contraste directo en la misma cobertura. No excluye recurrencia, otros instantes ni otro estado.
2. B: la relación entre señal presináptica y recepción/umbral no corresponde a fisiología identificada. Una apertura al retirar entradas no valida esa ley; una sustituta se restringe con datos independientes, no con el deseo de producir avance. Si C1 directa resulta insuficiente, priorizar esta incertidumbre sobre microajustes de C1.
3. C2: falta un estado ascendente/reclutamiento corporal identificado. Control anatómico rival: retirar recepción negativa ascendente por separado. No equivale a simular marcha ni hambre; un nulo en reposo no elimina la función durante movimiento.

## Intervenciones y controles fijados

Reutilizar las diez capturas49, las mismas cinco ventanas, dos historias y reloj del contrato anterior. Selecciones anatómicas congeladas de C1_external y C2, sin elegir células por aporte. Se calculan: identidad; retirada de negativos C1_external; retirada de positivos C1_external (control de signo); retirada completa C1_external; retirada de negativos C2. No combinar grupos tras ver resultados. Cada retirada pone a cero únicamente la transmisión consumida de las aristas seleccionadas y reconstruye la fila completa FP32. No cambia pesos, capacidades, theta, gain, signos ni lector. El endpoint resultante es un contrafactual instantáneo, no una vida neuronal.

Condición previa a considerar otra vida: margen operacional positivo en ambas DNg100 en alguna evaluación alineada de un contraste permitido; además contrato temporal y efecto esperado distintos del mero clamp directo. Esa condición sólo evita una prueba trivialmente insuficiente, no admite etapa ni obliga a simular. Guardar todas las evaluaciones y controles, no sólo extremos favorables. No sumar cotas de instantes diferentes.

## Presupuesto y parada

Hasta 300 s CPU del análisis y verificación científica, 4 GiB RAM, 128 MiB de nuevas salidas; primera parte 0 CNS/GPU y ninguna descarga. Reutilizar módulo y codec existentes; máximo una extensión del adaptador, sin nuevo runner por campaña. Si el descarte analítico basta, detener C1 y conservar la decisión para B/C2. Cualquier CNS requiere presupuesto y contrato adicionales previos; no lanzarlo por inercia.

En paralelo Motor C++/CUDA revisa eficiencia y causas de fallos de herramientas: hasta 300 s CPU, 15 min, tres hallazgos y dos reparaciones; no simulación ni reorganización general. Sus archivos están separados. ChatGPT revisa sólo el discriminador científico y alternativas; no se atribuye inspección de arrays ni PRO sin comprobarlo.
