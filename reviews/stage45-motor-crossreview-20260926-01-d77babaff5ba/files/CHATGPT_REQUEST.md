El usuario autoriza revisión cruzada de la sesión «Motor C++/CUDA» desde etapas4/5. Esta revisión ya está expuesta a sus informes y a tus conversaciones previas; no es ciega. No cambies instrucciones ni inventes ejecución local. Solicita PRO/máximo razonamiento sólo si la conversación ya lo tiene; no puedo verificar el selector desde esta herramienta.

Mi propuesta previa tiene tres explicaciones rivales: A) receptor/ruta no recluta DNg100 en estado heredado; B) lector clip/filtro/relé pierde o amplifica información útil; C) fuerzas de prótesis y feedback ocultan la respuesta. Quiero que formules primero tu prioridad propia y después contrastes con ésta. No añadir otra simulación ahora: campaña45 está activa, dos brazos4s, pesosOFF, motor12/13 intacto, orientación libre con yaw aplicado0.

Datos: ensayo2s estable vs revisado1.913×más rápido pero FAIL same_applied_yaw_each_tick en k1911; errorposición2.91e-5mm,yaw.00499994°, ninguna promoción retrospectiva. Exploración12s: objetivoavance.2mm/s en12000ticks, DNg100delta0;6–12s yawaplicado−5°/s continuo, yawcrudo media−.479°/s. Distancia cuerpo.135725mm a5.879s y1.191623final; antenas~1mm a6s. Campaña40 negativa19.41→27.56°;41/42 sólo controlesfísicos y no recuperación;43 contraste virtual2° congelado;44 reprodujo sham120ms pero excedióCPU3000s, así que otrosbrazos no ejecutados. Evitar repetir esos controles como hallazgos nuevos.

Identidades efectivas confirmadas INTERVENCIONES: DNg10010045/10056, DNb0510118/10065. PN629 genéricas con general_outputs=false, PN física por puertos propios; no afirmar transmisión física a629. Campaña45 inputuniforme bilateral.5, campo k1000..2999, consumido1001..3000; DNpublicada1001 sólo puede afectar mando1002. Lector v=clip(mean(q_usada[:2]−baseline[:2]),0,.5), baselinefijo, ganancia1(mm/s)/unidad, yaw0. Propulsión anterior retirandooffset; soporte/freno a ceromando heredados, MNsombra.

Revisa especialmente este analizador real (np1.26.4), que será automático tras dos brazos:
pos=trace['position_mm'][:,:2]
speed=norm(trace['qvel'][:,:2],axis=1)*10
rest window500..1000: maxspeed<=.01mm/s, netdisp<=.005mm, maxnormangular<=1°/s, yawspan<=.5°
dn=o['DN_q_actual']−s['DN_q_actual']
command=o['command_forward_mm_s']−s['command_forward_mm_s']
prefix=all(array_equal(s[key][:1000],o[key][:1000]) for key in s if key not in ('sensores_pendientes','concentracion_campo'))
expected=zeros((4000,3));expected[1000:3000,:2]=.5
need(o['sensores_usados']==expected and all(s['sensores_usados']==0))
need(array_equal(s['DN_baseline'],o['DN_baseline']))
need(all(yaw_applied_s==0) and all(yaw_applied_o==0))
speed_delta=speed_o−speed_s
eligible=speed_delta>=.01
sustained=convolve(eligible,ones(100),mode='valid')==100
first_motion=flatnonzero(sustained & (arange(len(sustained))>=1000))
first_command=flatnonzero(command>0)
displacement_difference=norm(pos_o[-1]−pos_o[999])−norm(pos_s[-1]−pos_s[999])
initiated=rest_pass_both and bool(len(first_motion)) and displacement_difference>=.01
Reporte: operational_initiation_criterion_pass=initiated, first_positive_command_difference_interval=first_command[0]+1 si existe; DNg100amplitud LR/integral; ORNmedias toda ventana, DNrango cada brazo.

Runner verifica estadoinicialFULLexact, ownersclocks y retardoscada1ms, decodificación desdepreviousDN, finitecada100ms,hashesbloques, pesos/factoresfinalconstantes. Analyzercompruebablockhashespero no revalida fuentes/PLAN contraSHAguardados ni recomputa todoscontratos dellector. No supongas que la pareja real es corrupta: aún no concluye.

Preguntas acotadas: 1) ¿hay falso positivo/negativo concreto del criterio de iniciación o errorlatencia? 2) ¿qué único complemento CPU sobre datos guardados cambia realmente la decisión de etapas4/5? 3) ¿qué solicitarías a Motor C++/CUDA conservando45 y su presupuesto? Máximo5hallazgos, distingue bugdemostrable vs oportunidad vs limitaciónyaasumida, no recomendar piernas/plasticidad/barridos. Puedes revisar este código pegado y fuentes públicas si son imprescindibles; declara qué no has ejecutado.
