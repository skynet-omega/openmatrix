"""Render the completed initiation pair from saved observations, on CPU only.

One command produces two arm videos, a narrated comparison and a scientific
figure. No call to mj_step or a neuronal integrator is permitted here.
"""
import os
for k,v in {'CUDA_VISIBLE_DEVICES':'','MUJOCO_GL':'osmesa','PYOPENGL_PLATFORM':'osmesa',
            'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','LP_NUM_THREADS':'1','HF_HUB_OFFLINE':'1'}.items():
    os.environ[k]=v
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
import cv2
import mujoco as mj
from OpenGL import GL

ROOT=Path('/home/daroch/AXIOMA_ASTRA')
BASE=ROOT/'motor_nuevo/animation_v7_12s_20260926'
TRIAL=ROOT/'campanas/iniciacion_olfativa_20260926_45'
RESEARCH=ROOT/'investigacion/avance_etapas45_20260926_02'
sys.path.insert(0,str(BASE))
from video_data import field_state,need,sha,PREPARED,CACHE,OLD
from render_v7 import text,rounded,font,BG,PANEL,WHITE,MUTED,CYAN,GOLD,PINK,GREEN
from rh_tarsal_body import RHTarsalBody
sys.path.insert(0,str(ROOT/'investigacion/iniciacion_sensorimotora_20260926_01'))
from supplement45 import load_arm,signed_motion

W,H,FPS,DURATION=1920,1080,24,40
COLORS={'OL':(65,125,255),'CB':(185,90,255),'PN':(25,225,235),
        'KC':(255,70,75),'CX':(255,215,50),'DN':(40,230,110)}
NAMES={'OL':'Lóbulos ópticos','CB':'Otras neuronas centrales','PN':'Neuronas de proyección',
       'KC':'Kenyon · cuerpos fungiformes','CX':'Complejo central','DN':'Neuronas descendentes'}

def write(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

class Pair:
    def __init__(self):
        self.plan=json.loads((TRIAL/'PLAN.json').read_text())
        self.inputs={str(TRIAL/'PLAN.json'):sha(TRIAL/'PLAN.json'),str(TRIAL/'SOURCES.json'):sha(TRIAL/'SOURCES.json')}
        self.arms={}
        for name in ('sham','odor'):
            result,t,initial,inputs=load_arm(name,self.inputs[str(TRIAL/'PLAN.json')],self.inputs[str(TRIAL/'SOURCES.json')])
            self.inputs.update(inputs)
            outputs=[initial['published_output']];times=[int(initial['time_ns'])]
            for b in sorted((TRIAL/name/'blocks').glob('*ms')):
                with np.load(b/'published.npz',allow_pickle=False) as z:
                    outputs.append(z['output'].copy());times.append(int(z['time_ns']))
            need(np.array_equal(times,int(initial['time_ns'])+np.arange(41)*100_000_000),'Saved neural clock differs')
            speed,step=signed_motion(t,initial)
            self.arms[name]=dict(trace=t,initial=initial,outputs=np.stack(outputs),speed=speed,result=result)
        s,o=self.arms['sham'],self.arms['odor']
        need(all(np.array_equal(s['initial'][k],o['initial'][k]) for k in s['initial']),'Different initial pair')
        self.body_identical=all(np.array_equal(s['trace'][k],o['trace'][k]) for k in ('qpos','qvel','contact_force_N'))
        need(self.body_identical,'Expected actual identical mechanics; revise narration for a different pair')
        need(all(np.count_nonzero(a['trace']['command_forward_mm_s'])==0 for a in self.arms.values()),'Narrated zero motor command differs')
        self.ids=np.load(OLD/'data/male_v10/node_ids.npy')
        with np.load(BASE/'SCANNER_DATA.npz',allow_pickle=False) as z:
            self.visible=z['ids'].copy();self.coords=z['coords'].copy()
            self.edges=np.column_stack([z['pre'],z['post']])
        self.rows=np.searchsorted(self.ids,self.visible)
        need(np.array_equal(self.ids[self.rows],self.visible),'Scanner ID mapping differs')
        with np.load(CACHE,allow_pickle=False) as z:
            need(np.array_equal(z['coords'],self.coords),'Scanner coordinates differ')
            self.types=z['types'].copy()
        for p in (BASE/'SCANNER_DATA.npz',BASE/'SCANNER_RECEIPT.json',CACHE,PREPARED/'session.json',PREPARED/'session.npz'):
            self.inputs[str(p)]=sha(p)
        self.body_state=field_state(PREPARED/'session','body')
        self.times=np.arange(1,4001)/1000

    def metrics(self):
        s,o=[self.arms[n]['trace'] for n in ('sham','odor')]
        phases={}
        for name,sl in [('baseline',slice(0,1000)),('stimulus',slice(1000,3000)),('poststimulus',slice(3000,4000))]:
            phases[name]={
                'ORN_mean_difference_LR':[float((o[k][sl].mean(1)-s[k][sl].mean(1)).mean()) for k in ('ORN_q_L','ORN_q_R')],
                'DN_mean_difference':np.mean(o['DN_q_actual'][sl]-s['DN_q_actual'][sl],axis=0).tolist(),
                'PN_legacy_mean_difference':np.mean(o['PN_q_legacy'][sl]-s['PN_q_legacy'][sl],axis=0).tolist(),
                'forward_command_max_difference':float(np.max(abs(o['command_forward_mm_s'][sl]-s['command_forward_mm_s'][sl])))}
        return dict(phases=phases,body_identical=self.body_identical,total_new_simulated_ms=0,
                    stage4='OPEN',stage5='OPEN',claim='Sensory propagation without DNg100-driven initiation in this prepared model/context')

def plot(pair,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,2,figsize=(13,9),sharex=True,layout='constrained')
    for name,color in [('sham','#708398'),('odor','#16a04b')]:
        a=pair.arms[name];t=a['trace'];label='Sin olor' if name=='sham' else 'Olor bilateral'
        series=[(t['sensores_usados'][:,:2].mean(1),'Entrada olfativa (u. modelo)'),
                ((t['ORN_q_L'].mean(1)+t['ORN_q_R'].mean(1))/2,'ORN: liberación media q'),
                (t['PN_q_legacy'].mean(1),'PN: dos salidas legacy q'),
                ((t['DN_q_actual'][:,2:]-t['DN_baseline'][:,2:]).mean(1),'DNb05: cambio medio q'),
                (t['command_forward_mm_s'],'Avance aplicado (mm/s)'),
                (a['speed'],'Velocidad longitudinal (mm/s)')]
        for axis,(y,title) in zip(ax.flat,series):
            axis.plot(pair.times,y,color=color,label=label,lw=1.5,alpha=.9)
            axis.axvspan(1,3,color='#16a04b',alpha=.07)
            axis.set_title(title,fontsize=11);axis.grid(alpha=.2)
    for axis in ax[-1]:axis.set_xlabel('Tiempo simulado (s)')
    ax[0,0].legend();ax[2,0].set_ylim(-.01,.01)
    fig.suptitle('Campaña 45 · respuesta sensorial, sin iniciación del avance\nCambio DNg100 L/R y avance crudo = 0 en ambos brazos; física idéntica',fontsize=14)
    fig.savefig(path,dpi=160);plt.close(fig)

class Render:
    def __init__(self,pair):
        self.p=pair;cv2.setNumThreads(1)
        self.body=RHTarsalBody.from_state(pair.body_state)
        self.renderer=mj.Renderer(self.body.model,height=355,width=576)
        self.gl=GL.glGetString(GL.GL_RENDERER).decode()
        need('llvmpipe' in self.gl.lower(),'CPU-only replay expected')
        self.camera=mj.MjvCamera();self.camera.type=mj.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:]=pair.arms['sham']['initial']['qpos'][:3]
        self.camera.distance=.78;self.camera.azimuth=45.;self.camera.elevation=-25.
        self.body_key=None;self.body_image=None;self.anatomy={}
        points=np.column_stack([pair.coords[:,0]+.22*pair.coords[:,2],pair.coords[:,1]-.14*pair.coords[:,2]])
        low,high=points.min(0),points.max(0);scale=min(544/(high[0]-low[0]),286/(high[1]-low[1]))
        self.points=np.column_stack([290+(points[:,0]-(low[0]+high[0])/2)*scale,
                                    156+(points[:,1]-(low[1]+high[1])/2)*scale]).astype(int)
        for name in ('sham','odor','difference'):
            for sample in range(41):
                if name=='difference':
                    delta=pair.arms['odor']['outputs'][sample,pair.rows].astype(float)-pair.arms['sham']['outputs'][sample,pair.rows].astype(float)
                    scale=5.
                else:
                    output=pair.arms[name]['outputs']
                    delta=output[sample,pair.rows].astype(float)-output[0,pair.rows].astype(float);scale=5.
                canvas=np.zeros((640,1160,3),np.uint8)
                for a,b in pair.edges:
                    cv2.line(canvas,tuple(self.points[a]*2),tuple(self.points[b]*2),(21,28,36),1,cv2.LINE_AA)
                strength=np.sqrt(np.clip(abs(delta)/scale,0,1))
                for point,tp,d in zip(self.points,pair.types,strength):
                    color=tuple(np.asarray(COLORS[tp],float)*(.30+.70*d))
                    cv2.circle(canvas,tuple(point*2),2,color,-1,cv2.LINE_AA)
                self.anatomy[name,sample]=Image.fromarray(cv2.resize(canvas,(580,320),interpolation=cv2.INTER_AREA))

    def body_frame(self,ms):
        if self.body_key==ms:return self.body_image
        a=self.p.arms['sham'];t=a['trace'];i=max(0,ms-1)
        for k in ('qpos','qvel'):getattr(self.body.data,k)[:]=a['initial'][k] if ms==0 else t[k][i]
        self.body.data.time=(int(a['initial']['time_ns'])+ms*1_000_000)/1e9
        mj.mj_forward(self.body.model,self.body.data)
        self.renderer.update_scene(self.body.data,camera=self.camera)
        self.renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION]=False
        self.body_image=Image.fromarray(self.renderer.render());self.body_key=ms
        return self.body_image

    def curve(self,img,x,y,title,series,limits,ms,fmt):
        d=ImageDraw.Draw(img);w=550;h=66
        text(img,(x,y-27),title,16,MUTED)
        d.rectangle((x+w/4,y,x+w*3/4,y+h),fill=(18,38,33))
        for f in (0,.5,1):d.line((x,y+h*f,x+w,y+h*f),fill=(41,54,68))
        text(img,(x+w-75,y+2),fmt.format(limits[1]),11,MUTED)
        text(img,(x+w-75,y+h-16),fmt.format(limits[0]),11,MUTED)
        for values,color in series:
            if ms:
                v=np.asarray(values)[:ms];ix=np.arange(1,ms+1)
                pts=np.column_stack([x+ix*w/4000,y+h*(1-np.clip((v-limits[0])/(limits[1]-limits[0]),0,1))]).astype(int)
                if len(pts)>1:d.line([tuple(p) for p in pts],fill=color,width=2)
        d.line((x+ms*w/4000,y,x+ms*w/4000,y+h),fill=WHITE,width=1)

    def brain(self,img,x,name,ms):
        sample=ms//100
        img.paste(self.anatomy[name,sample],(x+14,208))
        text(img,(x+20,534),'20.342 somas visibles / 166.700 neuronas',16,MUTED)
        text(img,(x+20,563),f'Estado guardado {sample/10:.1f} s · edad {ms-sample*100} ms',16,CYAN)
        text(img,(x+20,591),'887 conexiones: anatomía; no flujo inferido',14,MUTED)
        scale='5'
        desc='Brillo: |olor − control|, escala 0…5 u.' if name=='difference' else 'Brillo: |actual − inicial|, escala 0…5 u.'
        text(img,(x+20,620),desc,15,WHITE)
        text(img,(x+460,645),'|Δ| medio',12,MUTED)
        for j,tp in enumerate(COLORS):
            yy=669+j*28;rows=self.p.rows[self.p.types==tp]
            if name=='difference':
                d=self.p.arms['odor']['outputs'][sample,rows].astype(float)-self.p.arms['sham']['outputs'][sample,rows].astype(float)
            else:
                a=self.p.arms[name]['outputs'];d=a[sample,rows].astype(float)-a[0,rows].astype(float)
            text(img,(x+20,yy),tp,16,COLORS[tp],True)
            text(img,(x+65,yy+1),NAMES[tp],13,COLORS[tp])
            text(img,(x+478,yy+1),f'{np.mean(abs(d)):.4g}',14,WHITE)
        text(img,(x+20,845),'Base anatómica 30%; brillo no significa energía ni Hz.',12,MUTED)

    def frame(self,name,ms):
        comparison=name=='comparacion';arm='odor' if comparison else name
        a=self.p.arms[arm];t=a['trace'];i=max(0,ms-1)
        img=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(img)
        text(img,(28,27),'45',30,GOLD,True)
        title={'sham':'SIN OLOR · CONTROL','odor':'OLOR BILATERAL · ENSAYO','comparacion':'¿EL OLOR INICIA EL MOVIMIENTO? · COMPARACIÓN'}[name]
        text(img,(105,28),title,27,WHITE,True)
        text(img,(105,70),'Cuatro segundos registrados · mismo estado inicial · plasticidad deshabilitada',18,MUTED)
        rounded(img,(1500,25,1892,88),fill=(20,36,48))
        text(img,(1520,43),f'TIEMPO SIMULADO {ms/1000:5.3f} / 4 s',18,CYAN,True)
        for x,heading in [(28,'01 · CUERPO'),(656,'02 · EFECTO DEL OLOR' if comparison else '02 · SCANNER NEURONAL'),(1284,'03 · RESPUESTA REGISTRADA')]:
            rounded(img,(x,113,x+608,884));text(img,(x+18,129),heading,23,WHITE,True)
        text(img,(48,169),'Cámara fija · sin impulso ni avance constante',16,MUTED)
        img.paste(self.body_frame(ms),(44,206))
        cue=bool(ms>1000 and ms<=3000 and arm=='odor')
        badge='OLOR BILATERAL ACTIVO' if cue else ('CONTROL: SIN ESTÍMULO' if arm=='sham' else 'ANTES DEL OLOR' if ms<=1000 else 'OLOR RETIRADO')
        rounded(img,(44,195,620,248),fill=(16,39,30) if cue else (19,30,43))
        text(img,(59,209),badge,20,GREEN if cue else MUTED,True)
        text(img,(53,567),'DM1 virtual · sin molécula o concentración calibrada',15,GREEN)
        text(img,(53,597),'Estímulo uniforme: no hay fuente puntual que perseguir.',13,MUTED)
        text(img,(53,626),'Reposo verificado en 0,5–1 s; asentamiento inicial heredado.',13,MUTED)
        contacts=t['contact_active'][i]
        for j,label in enumerate(('T1-L','T2-L','T3-L','T1-R','T2-R','T3-R')):
            x=51+j*94;rounded(img,(x,665,x+82,697),fill=(20,40,35) if contacts[j] and ms else (25,33,45),radius=8)
            text(img,(x+11,672),label,15,GREEN if contacts[j] and ms else MUTED)
        text(img,(53,727),'AVANCE APLICADO',16,MUTED,True)
        text(img,(53,756),f'{t["command_forward_mm_s"][i]:.3f} mm/s',32,WHITE,True)
        text(img,(53,812),'Apoyo/freno activos · giro y viento: 0 · física idéntica',14,MUTED)
        text(img,(53,845),'Se reproducen poses guardadas; no se integra otra vida.',12,MUTED)
        text(img,(676,171),'Color = sector · brillo = magnitud del cambio',16,MUTED)
        self.brain(img,656,'difference' if comparison else arm,ms)
        if comparison:
            curves=[]
            for n,c in [('sham',(147,161,181)),('odor',GREEN)]:
                tr=self.p.arms[n]['trace']
                curves.append((tr,c))
            subtitle='Gris: sin olor · verde: olor · escala común'
        else:
            curves=[(t,GREEN)]
            subtitle='Banda: ventana comparada; aquí sin olor' if arm=='sham' else 'Bandas verdes: estímulo entre 1 y 3 s'
        text(img,(1306,172),subtitle,16,MUTED)
        defs=[(240,'Entrada utilizada L/R · unidades del modelo',lambda tr:tr['sensores_usados'][:,:2].mean(1),(0,.55),'{:.2f}'),
              (377,'Antenas ORN · liberación q, media L/R',lambda tr:(tr['ORN_q_L'].mean(1)+tr['ORN_q_R'].mean(1))/2,(0,.3),'{:.2f}'),
              (514,'PN · dos salidas legacy q, media',lambda tr:tr['PN_q_legacy'].mean(1),(.45,.65),'{:.2f}'),
              (651,'DNb05 · cambio desde basal, media L/R',lambda tr:(tr['DN_q_actual'][:,2:]-tr['DN_baseline'][:,2:]).mean(1),(-.0008,.0002),'{:.4f}'),
              (788,'DNg100 · cambio desde basal, media L/R',lambda tr:(tr['DN_q_actual'][:,:2]-tr['DN_baseline'][:,:2]).mean(1),(-.001,.001),'{:.3f}')]
        for y,title,fn,limits,fmt in defs:self.curve(img,1310,y,title,[(fn(tr),c) for tr,c in curves],limits,ms,fmt)
        text(img,(30,907),'VIDA SIMULADA',16,MUTED,True)
        x0,x1=230,1868;y=943
        d.line((x0,y,x1,y),fill=(49,65,79),width=5)
        d.line((x0+(x1-x0)/4,y,x0+3*(x1-x0)/4,y),fill=(50,154,89),width=7)
        for s in range(5):
            x=x0+(x1-x0)*s/4;d.line((x,y-8,x,y+8),fill=MUTED)
            text(img,(x-9,y+16),f'{s}s',16,MUTED)
        x=x0+(x1-x0)*ms/4000;d.ellipse((x-6,y-6,x+6,y+6),fill=WHITE)
        caption='VENTANA COMPARADA: 1–3 s · este control no recibe olor' if name=='sham' else 'OLOR: 1–3 s · entrada consumida con el desfase registrado de 1 ms'
        text(img,(638,909),caption,15,GREEN)
        rounded(img,(28,1000,1892,1064))
        message='Resultado completo: respuesta olfativa presente; DNg100 y avance sin respuesta. Etapas 4/5 abiertas.'
        text(img,(49,1010),message,20,WHITE,True)
        text(img,(49,1041),'Reproducción 6× más lenta · estados neuronales cada 100 ms · curvas cada 1 ms · sin inferir aprendizaje',14,MUTED)
        return img

    def close(self):self.renderer.close()

NARRATION={
 'comparacion':'Comparamos dos vidas de cuatro segundos, desde el mismo estado inicial. La línea gris es el control y la verde, el ensayo con olor. Entre uno y tres segundos llega un estímulo bilateral uniforme. El olor es virtual: no representa una molécula concreta. Cambian las antenas, las neuronas de proyección y la salida descendente de giro. Pero la señal de avance permanece en cero, y el cuerpo se queda en reposo. El brillo del cerebro compara olor contra control; no mide energía. Hay respuesta sensorial, pero todavía no iniciación del movimiento, navegación ni recuperación ante viento.',
 'sham':'Este es el control de cuatro segundos, sin estímulo de olor. Parte del mismo cerebro y cuerpo que el ensayo con olor. La actividad neuronal puede cambiar por su propia dinámica, aunque no se añada el estímulo. El brillo muestra cambio respecto al inicio, y los colores identifican sectores. No se impone avance constante. La señal de avance permanece en cero, con apoyo y frenado de la prótesis. Tras el asentamiento inicial, la mosca queda en reposo. Este control permite separar la respuesta al olor de la evolución espontánea del modelo.',
 'odor':'Este ensayo dura cuatro segundos. Entre uno y tres segundos se entrega olor bilateral uniforme. Es una entrada virtual al canal DM uno, sin molécula específica ni concentración física calibrada. Las curvas muestran respuesta en antenas, neuronas de proyección y neuronas descendentes de giro. Sin embargo, las neuronas elegidas para el avance no aumentan su salida. El mando de avance sigue en cero. Tras el asentamiento inicial, el cuerpo queda en reposo. Los colores muestran sectores del cerebro; el brillo es cambio desde el inicio. Detectar el olor aún no equivale a iniciar la marcha.'}

def voice(cache):
    import soundfile as sf
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pipeline=None;result={}
    for name,words in NARRATION.items():
        key=hashlib.sha256((words+'ef_dora:1.10:24000').encode()).hexdigest()
        p=cache/(key+'.wav')
        if not p.exists():
            if pipeline is None:
                from kokoro import KPipeline
                pipeline=KPipeline(lang_code='e',repo_id='hexgrad/Kokoro-82M',device='cpu')
            parts=[np.asarray(r.audio,np.float32) for r in pipeline(words,voice='ef_dora',speed=1.10)]
            need(bool(parts),'No voice output');sf.write(p,np.concatenate(parts),24000)
        clip,rate=sf.read(p,dtype='float32');need(rate==24000,'Voice sample rate')
        need(len(clip)/rate<39,'Narration exceeds its slot')
        full=np.zeros(DURATION*rate,np.float32);full[rate//2:rate//2+len(clip)]=clip
        out=cache/(name+'.wav');sf.write(out,full,rate);result[name]=out
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('/mnt/f/Videos/Flaywire/ACTUAL_ETAPA45'))
    ap.add_argument('--preview',action='store_true')
    args=ap.parse_args();started=time.monotonic();cpu=time.process_time()
    cache=Path(__file__).parent/'.cache/paired45';cache.mkdir(parents=True,exist_ok=True)
    p=Pair();r=Render(p)
    try:
        if args.preview:
            for n in ('sham','odor','comparacion'):r.frame(n,2500).save(cache/(n+'.png'))
            plot(p,cache/'datos.png');write(RESEARCH/'METRICAS45.json',p.metrics())
            print(json.dumps(dict(status='PREVIEW',path=str(cache),neural_steps=0,physical_steps=0)));return
        need(not args.output.exists(),'Existing completed delivery is preserved; use a new owned output for a revised contract')
        work=args.output.with_name(args.output.name+'.building');need(not work.exists(),'Inspect existing partial build')
        work.mkdir(parents=True);audio=voice(cache);encoders={}
        try:
            for n in NARRATION:
                encoders[n]=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24',
                    '-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast',
                    '-crf','19','-pix_fmt','yuv420p',str(work/(n+'_silent.mp4'))],stdin=subprocess.PIPE)
            last=-1;frames={}
            for i in range(DURATION*FPS):
                ms=int(np.clip(round((i/FPS-3)/6*1000),0,4000))
                if ms!=last:
                    frames={n:r.frame(n,ms) for n in encoders};last=ms
                for n,enc in encoders.items():enc.stdin.write(frames[n].tobytes())
                if i==round(18*FPS):frames['comparacion'].save(work/'poster.png')
                if i%120==0:print(json.dumps(dict(frame=i,frames=DURATION*FPS,simulated_ms=ms,elapsed_s=time.monotonic()-started)),flush=True)
                need(time.monotonic()-started<1200,'20 minute render budget')
            for enc in encoders.values():enc.stdin.close();need(enc.wait(timeout=60)==0,'Encoder failed')
        finally:
            for enc in encoders.values():
                if enc.poll() is None:enc.kill();enc.wait()
        from generate import verify_media
        media={}
        for n in encoders:
            out=work/(n+'.mp4')
            subprocess.run(['ffmpeg','-v','error','-nostdin','-n','-i',str(work/(n+'_silent.mp4')),'-i',str(audio[n]),
                '-c:v','copy','-c:a','aac','-b:a','160k','-movflags','+faststart',str(out)],check=True,timeout=60)
            media[n]=verify_media(out,DURATION*FPS,DURATION);(work/(n+'_silent.mp4')).unlink()
        plot(p,work/'datos.png');write(work/'metricas.json',p.metrics());write(work/'narracion.json',NARRATION)
        write(RESEARCH/'METRICAS45.json',p.metrics())
        need(all(sha(k)==v for k,v in p.inputs.items()),'Source evidence changed during rendering')
        (work/'README.md').write_text('''# Campaña 45: comparación de iniciación olfativa

`comparacion.mp4`: comparación narrada; gris = control, verde = olor. El cerebro representa |olor − control| (escala fija 0…5 unidades de salida).
`sham.mp4` y `odor.mp4`: cada vida por separado; cerebro = |actual − inicial| (escala fija 0…5). La misma escala está escrita en cada vídeo. Los valores de la tabla son medias absolutas de cambio entre los somas visibles de cada sector. No se interpretan como trabajo, energía o tasas fisiológicas calibradas.

4 s simulados, 40 s por vídeo: 3 s de pausa inicial, reproducción 6× más lenta hasta 27 s, imagen final retenida. Curvas observadas cada 1 ms; cerebro guardado cada 100 ms, último estado retenido sin interpolación. Colores fuertes identifican sectores; 30% de brillo base representa anatomía, no actividad.

Entrada virtual bilateral uniforme por ORN DM1, entre 1 y 3 s. Sin molécula ni concentración física calibrada; no hay fuente puntual o gradiente que perseguir. El primer intervalo que consume el olor termina en 1001 ms. El indicador respeta ese desfase. Viento, plasticidad y mando de giro deshabilitados. El avance sólo lee DNg100; apoyo/frenado de la prótesis siguen presentes. MN registrados son observadores, no accionan directamente las patas en este protocolo.

Resultado: señal olfativa presente; DNg100, avance crudo y aplicado sin respuesta; qpos/qvel y fuerzas idénticos entre brazos. Esto no supera etapas4/5 ni permite concluir que una mosca real no iniciaría movimiento.

Reproducción: `/home/daroch/miniconda3/envs/GPU/bin/python /home/daroch/AXIOMA_ASTRA/instrumentos/neurovideo/paired45.py`. El comando preserva entregas existentes; `--preview` verifica una imagen y figura sin codificar vídeos. Cero pasos neuronales/físicos nuevos; CPU OSMesa. `manifest.json` identifica datos y medios verificados por decodificación completa.
''',encoding='utf-8')
        write(work/'manifest.json',dict(status='COMPLETE',source_hashes=p.inputs,source_code_sha256=sha(__file__),
            media=media,body_identical=p.body_identical,renderer=r.gl,neural_steps=0,physical_steps=0,
            wall_s=time.monotonic()-started,cpu_s=time.process_time()-cpu,
            delivery_hashes={f.name:sha(f) for f in work.iterdir() if f.is_file()}))
        work.replace(args.output);print(json.dumps(dict(status='COMPLETE',output=str(args.output),wall_s=time.monotonic()-started)),flush=True)
    finally:r.close()

if __name__=='__main__':main()
