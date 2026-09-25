"""Criba CPU de zona recurrente; NO integra ni certifica una trayectoria.
El score procede del defecto MRI completo, no del error final de una neurona.
Los jobs deben describir el trabajo REAL requerido para coeficientes parciales:
si un propietario solo ejecuta un kernel completo, se carga completo.
"""
import argparse, hashlib, json, time, zipfile
from collections import deque
from pathlib import Path
import numpy as np

def exigir(ok, texto):
    if not ok:
        raise ValueError(texto)

def entero(a, nombre):
    a=np.asarray(a)
    exigir(a.shape==() and a.dtype.kind in "iu", "Escalar entero: "+nombre)
    return int(a)

def csr(d, prefijo, n, limite):
    p=d[prefijo+"_ptr"];j=d[prefijo+"_ids"]
    exigir(p.dtype==np.int64 and p.shape==(n+1,), "ptr: "+prefijo)
    exigir(j.dtype in (np.dtype("int32"),np.dtype("int64")) and j.ndim==1,
           "ids: "+prefijo)
    exigir(p[0]==0 and p[-1]==len(j) and np.all((p>=0)&(p<=len(j))) and np.all(np.diff(p)>=0),
           "Offsets: "+prefijo)
    exigir(np.all((j>=0)&(j<limite)), "Indices: "+prefijo)
    return p,j

def cierre(semilla, p, j):
    visto=set(map(int,semilla));cola=deque(sorted(visto))
    while cola:
        i=cola.popleft()
        for v in j[p[i]:p[i+1]]:
            v=int(v)
            if v not in visto:
                visto.add(v);cola.append(v)
    return np.array(sorted(visto),dtype=np.int64)

def planificar(d):
    score=d["score"];prescrito=d["prescribed"];lectores=d["event_readers"];n=len(score)
    exigir(score.dtype==np.float64 and score.shape==(n,) and n>0
           and np.isfinite(score).all() and np.all(score>=0),"Score")
    exigir(prescrito.dtype==np.bool_ and prescrito.shape==(n,), "Prescribed")
    exigir(lectores.dtype==np.bool_ and lectores.shape==(n,), "Lectores finales de eventos")
    aristas=d["job_edges"];jlocal=d["job_local_elements"];jbytes=d["job_copy_bytes"]
    exigir(aristas.dtype==jlocal.dtype==jbytes.dtype==np.int64
           and aristas.ndim==1 and aristas.shape==jlocal.shape==jbytes.shape
           and np.all(aristas>=0) and np.all(jlocal>=0) and np.all(jbytes>=0),
           "Coste completo de jobs")
    p,j=csr(d,"closure",n,n)
    hp,hj=csr(d,"halo",n,n)
    jp,jj=csr(d,"jobs",n,len(aristas))
    E=entero(d["base_edges"],"base_edges")
    full=entero(d["full_edges"],"full_edges")
    setup=entero(d["setup_edges"],"setup_edges")
    cap=entero(d["fast_cap"],"fast_cap")
    exigir(E>0 and full>=E and setup>=0 and 0<cap<=10000, "Presupuesto")
    # Una seleccion inicial y UNA expansion offline; no replays repetidos.
    elegidos=cierre(np.flatnonzero(((score>1)|lectores)&~prescrito),p,j)
    exigir(not np.any(prescrito[elegidos]), "El cierre incluye estados prescritos")
    planes=[]
    for ronda in (0,1):
        trabajos=set()
        for i in elegidos:
            trabajos.update(map(int,jj[jp[i]:jp[i+1]]))
        ids=sorted(trabajos)
        coste=sum(int(aristas[k]) for k in ids)
        total=setup+5*full+cap*coste
        planes.append({
            "ronda":ronda,"filas":elegidos.tolist(),"jobs":ids,
            "aristas_por_fast":coste,"fast_cap":cap,
            "aristas_totales_con_setup_y_cinco_full":total,
            "barridos_equivalentes":total/E,
            "gate_coste_le_6":total<=6*E,
            "elementos_locales_fast_max":cap*sum(int(jlocal[k]) for k in ids),
            "bytes_copias_fast_max":cap*sum(int(jbytes[k]) for k in ids)
        })
        vecinos=set(map(int,elegidos))
        for i in elegidos:
            vecinos.update(map(int,hj[hp[i]:hp[i+1]]))
        elegidos=cierre(vecinos,p,j)
        exigir(not np.any(prescrito[elegidos]),"El halo/cierre incluye estados prescritos")
    return {
        "status":"OFFLINE_ZONE_COST_ONLY","planes":planes,
        "seleccion":"lectores de eventos o score>1; cierre masa/propietario; un halo",
        "certifica_error":False,"organismo_ejecutado":False,"CUDA_ejecutado":False,
        "limites":"El mapa de jobs y el score son entradas auditables, no inferencias por anatomia. "
                  "El gate no cuenta como velocidad; costos de full/setup externos deben medirse."
    }

def ejemplo():
    return {
      "score":np.array([0.,2.,0.,0.,0.,0.]),
      "prescribed":np.array([0,0,0,0,0,1],bool),
      "event_readers":np.zeros(6,bool),
      "closure_ptr":np.array([0,0,1,2,2,2,2],np.int64),
      "closure_ids":np.array([2,1],np.int64),
      "halo_ptr":np.array([0,0,1,2,2,2,2],np.int64),
      "halo_ids":np.array([0,3],np.int64),
      "jobs_ptr":np.array([0,1,2,3,4,5,5],np.int64),
      "jobs_ids":np.array([0,1,1,2,3],np.int64),
      "job_edges":np.array([30,60,90,10000],np.int64),
      "job_local_elements":np.array([1,2,1,1],np.int64),
      "job_copy_bytes":np.zeros(4,np.int64),
      "base_edges":np.array(10000,np.int64),
      "full_edges":np.array(10000,np.int64),
      "setup_edges":np.array(1000,np.int64),
      "fast_cap":np.array(100,np.int64)
    }

def selftest():
    d=ejemplo();a=planificar(d)
    exigir(a["planes"][0]["filas"]==[1,2],"Cierre de masa/propietario")
    exigir(a["planes"][0]["jobs"]==[1],"Job compartido duplicado")
    exigir(a["planes"][0]["gate_coste_le_6"] and
           not a["planes"][1]["gate_coste_le_6"],"Expansion no refutada")
    d["full_edges"]=np.array(11000,np.int64)
    exigir(not planificar(d)["planes"][0]["gate_coste_le_6"],"Costo owner omitido")
    d=ejemplo();d["event_readers"][4]=True
    exigir(not planificar(d)["planes"][0]["gate_coste_le_6"], "Lector de evento omitido")
    d=ejemplo();d["closure_ids"][0]=99
    try:planificar(d)
    except ValueError:pass
    else:raise ValueError("Indice corrupto aceptado")
    E=25582938;eventos=885587;nf=229
    return {"CPU_fixture":"PASS","frozen_fast_record_cost":
          {"event_only_fast_equivalents":nf*eventos/E,
           "five_full_plus_events":5+nf*eventos/E,
           "max_edges_per_fast_with_five_full_no_setup":E//nf},
          "synthetic_zone":a,"CUDA_executed":False,"organism_executed":False}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest",action="store_true")
    g.add_argument("--input",type=Path)
    p.add_argument("--sha256");p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();exigir(not a.out.exists(),"Salida existente")
    inicio=time.perf_counter()
    if a.selftest:r=selftest()
    else:
        b=a.input.read_bytes();h=hashlib.sha256(b).hexdigest()
        exigir(h==a.sha256,"Hash de entrada")
        with zipfile.ZipFile(a.input) as z:
            exigir(sum(v.file_size for v in z.infolist())<1024**3,"NPZ >1GiB")
        with np.load(a.input,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        r=planificar(d);r["input_sha256"]=h
    r["wall_s"]=time.perf_counter()-inicio
    r["code_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print(json.dumps(r,indent=2,allow_nan=False))
