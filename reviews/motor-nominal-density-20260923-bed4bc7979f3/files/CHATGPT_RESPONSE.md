**Elegiría A: recuperar la propuesta anterior tras un corte impuesto, sin aceptar ningún paso nuevo sin evaluarlo.** Es la intervención mejor sustentada por el registro actual y no requiere cambiar ecuaciones ni su estimador. Entrego un generador aislado, compilado y probado aquí **con un doble CPU de la API CUDA**, no con CUDA ni con el organismo.

## 1. La inconsistencia de tiempos sigue abierta

En la sonda, `graph_interval` usa un par de eventos distinto por propuesta; el reloj host comienza antes del avance y termina después de su sincronización final. **No encontré una suma de regiones anidadas que explique el 104,49 %.** Tampoco basta decir que otro stream intercaló trabajo: eso puede ampliar el intervalo medido, pero no explica por sí solo superar un intervalo host que lo contiene, si ambos relojes son comparables.  

NVIDIA advierte que los eventos pueden medir trabajo intercalado y no solamente el kernel. Esa advertencia **no autoriza normalizar 2279,313 ms a 2181,308 ms**, ni atribuir la diferencia a WSL. :chatgpt-content-reference{index="2"}

**Discriminador finito:** en una única época de la próxima medición prevista, añadir un evento ancla antes del primer trabajo y otro después del último commit; medir además el intervalo host exterior con `steady_clock` y `CLOCK_MONOTONIC_RAW`. Consultar los marcadores existentes respecto del mismo ancla:

\[
\sum_i D_{\mathrm{graph},i}\le D_{\mathrm{span\ CUDA}}
\le D_{\mathrm{host\ exterior}}
\]

salvo resolución instrumental. Si falla la primera relación, investigar marcadores/orden/agregación. Si solo falla la segunda y los dos relojes host coinciden, queda localizada una inconsistencia entre dominios temporales/API. **No adjudica todavía su causa.** No necesita otra trayectoria larga ni cambiar el presupuesto.

## 2. A/B/C: qué trabajo eliminaría cada una

| Alternativa | Operación y límite |
|---|---|
| **A. Propuesta independiente del corte** | Evitar la recuperación geométrica artificial después de colas pequeñas. Mantiene los seis barridos por intento y todas las fronteras. En el registro fijo, el techo combinatorio es **191/100 = 1,91× en número de intentos**, no en tiempo total. |
| **B. Exponencial embebido** | Sustituir completo+dos medios por un par de orden distinto que reutilice etapas. Puede reducir barridos por intento, pero necesita otro estimador, análisis de eventos y convergencia: no se obtiene borrando evaluaciones del actual. |
| **C. Multirritmo espacial con influencia controlada** | Evitar actualizar globalmente componentes cuya entrada temporal puede representarse con error acotado; corregir la recurrencia antes de confirmar. Pierde si las correcciones cuestan lo ahorrado o la cota no cubre receptores/realimentación. |

El techo de A supone **la misma agenda física** y un intento aceptado mínimo por segmento. No extrapola los 191 intentos a la vida completa ni usa errores antiguos para predecir aceptación de pasos nuevos. 

Para B, el contrato leído es \(f(t,y)=b(t,y)\odot[a(t,y)-y]\), y **\(b\) no es constante en general**: por ejemplo, la tasa visual depende de `photo`. Una separación válida sería \(L_n=-\mathrm{diag}(b_n)\), \(N_n(t,y)=f(t,y)-L_ny\), reevaluando **todo** \(f\). Un candidato ETD2 podría usar:

\[
Y=e^{hL_n}y+h\varphi_1(hL_n)N_n(t,y),
\]
\[
y_2=Y+h\varphi_2(hL_n)[N_n(t+h,Y)-N_n(t,y)].
\]

Esto conserva algebraicamente el RHS variable; **no garantiza estabilidad, dominio ni cobertura de eventos**. El par \(Y/y_2\) tampoco usa el mismo error de orden tres del padre. Por eso no implemento B a escondidas dentro de A.   :chatgpt-content-reference{index="6"}

## 3. Variante elegida: recuperación conservadora

Sean \(H\) la propuesta nominal, \(d\) la distancia a la frontera y \(h=\min(H,d)\). **Solo después de aceptar y comprobar dominio**:

\[
H_{\mathrm{siguiente}}=
\begin{cases}
H,& h<H\ \text{y}\ e<0,1,\\
H_{\mathrm{padre}}(h,e),&\text{en otro caso}.
\end{cases}
\]

No duplico \(H\) por observar error pequeño en una cola: esa cola no probó la precisión de un paso completo \(H\). Tras rechazo se conserva exactamente la reducción del padre; **no se resucita una propuesta anterior al rechazo**. El umbral 0,1 ya existe en el controlador. 

Es una **hipótesis de planificación**, no una cota de error global. Puede aumentar rechazos tras eventos, cambiar trayectorias y afectar al detector. Conserva pruebas, límites, rollback externo y presupuestos; no conserva necesariamente las decisiones, porque propone pasos diferentes.

## 4. Código completo: `generar_nominal.py`

Genera padre y candidata en una carpeta nueva. `--selftest` compila ambos con una API CUDA simulada, usando errores nuevos dependientes de `h`; **no reproduce rendimiento GPU**.

```python
"""Generador aislado A: recuperar propuesta tras cola forzada con e<0.1.
No modifica ecuaciones, eventos, norma, dominio, presupuesto ni rollback Python.
--selftest compila con doble CPU de CUDA; NO valida CUDA ni organismo.
"""
import argparse, hashlib, json, subprocess, sys, time, traceback
from pathlib import Path
SHA="5aa0935103ad2e29d7bf05683ad3e291949f04c7abcd6537f945cc90daa62642"
OLD_H="double requested=std::min(*next,maxstep)*1e-9,h=std::min(available,requested);"
NEW_H="""long nominal=std::min(*next,maxstep);
   double requested=nominal*1e-9,h=std::min(available,requested);
   const bool forced_cut=h<requested;"""
OLD_N="*next=std::min(maxstep,std::max(minstep,(long)std::floor(h*1e9*(e<.1?2:1))));"
NEW_N="""if(forced_cut && e<.1) {
     // Recover a proposal, never accept an untested step or enlarge from a tail.
     *next=std::max(minstep,nominal);
    } else {
     *next=std::min(maxstep,std::max(minstep,(long)std::floor(h*1e9*(e<.1?2:1))));
    }"""
PLAN={"variant":"A_nominal_easy_cut_v1","restore_if":"h<requested AND accepted AND e<0.1",
      "rejection":"parent policy, no memory of pre-rejection proposal",
      "event_boundaries":"all retained, including sub-minimum accepted tails",
      "historical_gate":"UNCHANGED","organism_speedup":None,
      "selftest_scope":"synchronous CPU CUDA double; fresh h-dependent test errors"}
def require(ok,msg):
    if not ok: raise ValueError(msg)
def save(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
def generate(p,out):
    raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==SHA,"Wrong parent SHA256")
    text=raw.decode("utf-8")
    require(text.count(OLD_H)==text.count(OLD_N)==1,"Unexpected source anchors")
    new=text.replace(OLD_H,NEW_H).replace(OLD_N,NEW_N)
    # engine_advance (no event contract) and the original rejection remain intact.
    (out/"graph_control_parent.cpp").write_bytes(raw)
    (out/"graph_control_nominal.cpp").write_text(new,encoding="utf-8")
    result=dict(parent_sha256=SHA,candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),
                plan=PLAN,source_lines_changed_only="proposal and next after accepted cut")
    save(out/"MANIFEST.json",result);return result

STUB=r"""
#pragma once
// Synchronous test double. No device timing, capture or concurrent execution.
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <cmath>
using cudaStream_t=void*;using cudaGraphExec_t=void*;using cudaError_t=int;
const int cudaSuccess=0;
enum cudaMemcpyKind {cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice};
struct Mock {double*c,*s,*x,*f;int mode;std::vector<std::array<double,4>> log;};
inline const char*cudaGetErrorString(int){return "mock error";}
inline int cudaMallocHost(void**p,size_t n){*p=calloc(1,n);return *p?0:1;}
inline int cudaFreeHost(void*p){free(p);return 0;}
inline int cudaMemcpyAsync(void*d,const void*s,size_t n,cudaMemcpyKind,cudaStream_t){memcpy(d,s,n);return 0;}
inline int cudaStreamSynchronize(cudaStream_t){return 0;}
inline int cudaGraphLaunch(void*p,cudaStream_t){
 auto*g=(Mock*)p;double t=g->c[0],h=g->c[1],e=.05,flag=0,domain=0;
 if(g->mode==1)e=.5;
 if(g->mode==2)e=h>300e-9?2.:.5;
 if(g->mode==3 && t>=80e-9)e=h>300e-9?2.:.5;
 if(g->mode==4)domain=1;
 if(g->mode==5)e=NAN;
 if(g->mode==6)flag=1;
 if(g->mode==7)e=2.;
 if(g->mode==8)e=.1;
 if(g->mode==9 && t>=80e-9)domain=1;
 g->s[0]=e;g->s[1]=flag;g->s[2]=domain;
 g->f[0]=t+h; // Probe function only, NOT a neuronal equation.
 g->log.push_back({t,h,e,domain});return 0;
}
"""
DRIVER=r"""
#include CONTROL_FILE
#include <cstdio>
#include <cstring>
static_assert(sizeof(long)==8,"Linux64 ABI required");
int main(){
 const char*names[]={"no_cut","easy_tail","rejection","new_rejection","domain",
 "nonfinite","flag","short_reject","fractional","bad_events","bad_budget",
 "threshold_equal","late_domain","ordinary_no_events"};
 puts("[");
 for(int i=0;i<14;i++){
  double clock[2]={},status[3]={},x[1]={0},fine[1]={0};
  int mode=i==0?1:i==2?2:i==3?3:i==4?4:i==5?5:i==6?6:
           i==7?7:i==11?8:i==12?9:0;
  Mock g{clock,status,x,fine,mode,{}};
  void*r=engine_create(&g,(void*)1,clock,status,x,fine,1);if(!r)return 3;
  long n=i==0?500:2000,c[3]={};double e=0;
  double simple[]={80e-9}, reject[]={800e-9};
  double fractional[]={0,80e-9,80e-9,1.000001e-6,3.99995e-6};
  double bad[]={2e-6,1e-6};
  const double*ev=i==0?nullptr:i==2?reject:i==8?fractional:i==9?bad:simple;
  long count=i==0?0:i==8?5:i==9?2:1;double budget=i==10?0:10.;
  int code=i==13?engine_advance(r,4000,&n,100,4000,budget,c,&e):
                 engine_advance_events(r,4000,&n,100,4000,budget,ev,count,c,&e);
  bool cuts=true;
  for(auto row:g.log)for(long k=0;k<count;k++){
   if(i==13)break;
   if(row[0]<ev[k]&&row[0]+row[1]>ev[k]+1e-20)cuts=false;
  }
  double partial=x[0];if(code)x[0]=0.; // Emulates graph_core's outer rollback.
  printf("%s{\"case\":\"%s\",\"return\":%d,\"next\":%ld,\"accepted\":%ld,\"rejected\":%ld,"
         "\"state_hex\":\"%a\",\"partial_hex\":\"%a\",\"cuts_ok\":%s,\"error\":\"%s\",\"trace\":[",
     i?",":"",names[i],code,n,c[0],c[1],x[0],partial,cuts?"true":"false",engine_error());
  for(size_t k=0;k<g.log.size();k++){
   auto v=g.log[k];printf("%s[\"%a\",\"%a\",\"%a\",\"%a\"]",k?",":"",v[0],v[1],v[2],v[3]);
  }
  puts("]}");engine_destroy(r);
 }
 puts("]");return 0;
}
"""
def selftest(out):
    d=out/"cpu_test";d.mkdir()
    (d/"cuda_runtime.h").write_text(STUB,encoding="utf-8")
    (d/"driver.cpp").write_text(DRIVER,encoding="utf-8")
    results={}
    for name,filename in (("parent","graph_control_parent.cpp"),("candidate","graph_control_nominal.cpp")):
        exe=d/name
        cmd=["g++","-std=c++17","-O3","-ffp-contract=off","-I",str(d),
             '-DCONTROL_FILE="'+str((out/filename).resolve())+'"',str(d/"driver.cpp"),"-o",str(exe)]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
        (d/(name+"_build.txt")).write_text(r.stdout+r.stderr,encoding="utf-8")
        require(r.returncode==0,"Compilation failed: "+name)
        r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
        (d/(name+".json")).write_text(r.stdout,encoding="utf-8")
        (d/(name+"_stderr.txt")).write_text(r.stderr,encoding="utf-8")
        require(r.returncode==0,"Test failed: "+name)
        results[name]={v["case"]:v for v in json.loads(r.stdout)}
    p,c=results["parent"],results["candidate"]
    for k in p:
        require(p[k]["return"]==c[k]["return"] and p[k]["error"]==c[k]["error"],"Failure policy changed: "+k)
        require(c[k]["cuts_ok"],"Crossed boundary: "+k)
        require(p[k]["state_hex"]==c[k]["state_hex"],"Mock final/rollback state differs: "+k)
        if c[k]["return"]!=0:require(float.fromhex(c[k]["state_hex"])==0,"Outer rollback")
    for k in ("no_cut","rejection","domain","nonfinite","flag","short_reject",
              "bad_events","bad_budget","threshold_equal","ordinary_no_events"):
        require(p[k]==c[k],"Unexpected change in control: "+k)
    require(c["easy_tail"]["accepted"]<p["easy_tail"]["accepted"],"No recovery on easy cut")
    require(c["new_rejection"]["rejected"]>0,"New large steps bypassed error")
    require(float.fromhex(c["new_rejection"]["trace"][1][1])==2000*1e-9,"Nominal not restored")
    require(c["short_reject"]["return"]==-1 and c["short_reject"]["accepted"]==0,"Rejected short tail")
    result={"cases":len(p),"CPU_control_double_pass":True,
            "easy_tail_trials":[len(p["easy_tail"]["trace"]),len(c["easy_tail"]["trace"])],
            "new_rejection_trials":[len(p["new_rejection"]["trace"]),len(c["new_rejection"]["trace"])],
            "candidate_new_rejections":c["new_rejection"]["rejected"],
            "CUDA_executed":False,"organism_executed":False,
            "scope":"Synthetic h-dependent errors; not reuse of old organism errors; no speedup measurement"}
    save(d/"RESULTADO.json",result);return result
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("source",type=Path);a.add_argument("out",type=Path)
    a.add_argument("--selftest",action="store_true");x=a.parse_args()
    x.out.mkdir(parents=True,exist_ok=False);save(x.out/"PLAN.json",PLAN)
    wall=time.perf_counter();result={}
    try:
        result["generation"]=generate(x.source,x.out)
        if x.selftest:result["tests"]=selftest(x.out)
        result["status"]="COMPLETE"
    except Exception:
        result.update(status="FAILED_RETAINED",error=traceback.format_exc())
    result["wall_s"]=time.perf_counter()-wall
    result["generator_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(x.out/"RESULTADO.json",result)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0 if result["status"]=="COMPLETE" else 1
if __name__=="__main__":sys.exit(main())
```

### Generación y compilación real

```bash
python -S generar_nominal.py \
  /ruta/probe/generated/graph_control_parent.cpp \
  candidato_nominal_01 --selftest

# Usar la misma instalación CUDA y flags para ambos.
CUDA_RT="$HOME/miniconda3/envs/GPU/lib/python3.10/site-packages/nvidia/cuda_runtime"
for variante in parent nominal; do
  g++ -std=c++17 -O3 -ffp-contract=off -fPIC -shared \
    "candidato_nominal_01/graph_control_${variante}.cpp" \
    -I "$CUDA_RT/include" -L "$CUDA_RT/lib" \
    -Wl,-rpath,"$CUDA_RT/lib" -l:libcudart.so.12 \
    -o "candidato_nominal_01/libgraph_control_${variante}.so"
done
```

Seleccionar la biblioteca mediante `native_library` **en un arnés aislado y una sesión nueva**, antes de construir el grafo. No sobrescribir la biblioteca operativa ni activar la variante a mitad de una trayectoria. La interfaz C y el rollback Python permanecen iguales. Los flags corresponden al `BUILD.json` publicado. 

## 5. Resultado propio y falsador del organismo

**14 controles CPU pasaron.** En el fixture de cola fácil hubo **6→3 propuestas**. En otro, recuperar la propuesta provocó **tres rechazos nuevos**, todos procesados: **26→20 propuestas totales**. Son verificaciones de control, no aceleraciones medidas del organismo.

La primera ejecución falló en una aserción del arnés: esperaba el literal `2000e-9`, distinto en un bit de la operación del controlador `2000*1e-9`. Corregí la expresión esperada, **no una tolerancia**. El C++ candidato permaneció idéntico. Fallo y resultados se conservan. La última generación, compilación y prueba tomó **1,155 s de pared**.

**Campaña propuesta después de las referencias:** un par padre/candidata de 1 ms desde el mismo estado, máximo 240 s por brazo, sin concurrencia GPU. Contar propuestas/rechazos y medir pared nativa y total. Conservar la variante solo si satisface el comparador numérico predeclarado y consigue una reducción material prefijada —propongo **20 % de pared nativa**, acompañada por menos evaluaciones—. Si falla precisión, eventos, dominio o presupuesto, detener; no buscar otro umbral de recuperación.

La equivalencia de decisiones **no** es el objetivo de A. Sí lo son la conservación de reglas, la calidad de la trayectoria y el ahorro real. El gate KC antiguo y el contrato funcional de etapa 3 siguen separados.

**Lectura efectiva:** README/ARCHIVE, `CONTROL_READBACK.json`, `read_control.py`, `RESULT.json`, `TIMING_LIMIT.json`, `BUILD.json`, ambos C++ generados, `graph_core.py`, `EVENT_CUT_POLICY.md`; líneas 1–140 de `gpu_coefficient_layout.py`. Leí también el índice de la prerregistración funcional, no sus nuevas referencias.

**No descargué el ZIP nuevo ni ejecuté sus 191 intentos.** Recuperé el padre de nuestro artefacto anterior y comprobé SHA256 y blob Git contra la fuente actual: ambos coinciden. No hay `nvcc` disponible; **no compilé contra CUDA ni ejecuté GPU/organismo**.

SHA256 de la candidata C++:  
`116bdae3bd7a4374cb3aca09af45629052c618d0cee88ba6baa034f73ce235e5`

:chatgpt-content-reference{index="10"}[Paquete con generador, fuentes, pruebas y fallo conservado](sandbox:/mnt/data/AXIOMA_PLAN16_NOMINAL_CPU.zip)

**A es una mejora potencial acotada, no el salto completo a 1 s/60 s.** Su ventaja es que elimina trabajo identificable sin confundir un cambio de planificador con una ley neuronal nueva.
