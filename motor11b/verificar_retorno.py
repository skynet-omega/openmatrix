"""Verify an extracted return bundle without MATRIX, GPU, or simulator imports."""
from pathlib import Path
import argparse,hashlib,json,sys

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def require(ok,message):
    if not ok:raise ValueError(message)

def main():
    p=argparse.ArgumentParser();p.add_argument('root',nargs='?',default=Path(__file__).resolve().parent)
    root=Path(p.parse_args().root).resolve()
    m=json.loads((root/'MANIFIESTO_RETORNO.json').read_text())
    for rel,h in m.items():
        f=(root/rel).resolve()
        require(f.is_relative_to(root) and f.is_file(), 'Missing or unsafe path: '+rel)
        require(sha(f)==h,'Changed payload: '+rel)
    package=root/'propuesta_recibida'
    sys.path.insert(0,str(package))
    from comparacion_numerica import comparar_ramas, auditar_tiempo
    r=json.loads((root/'ejecucion/RESULTADOS_11B.json').read_text())
    prior=root/'referencia11'
    comparisons={
        'cruzada_A1_B1':(prior/'A1',prior/'B1'),
        'repetibilidad_B':(prior/'B1',root/'ejecucion/B2'),
        'repetibilidad_A':(prior/'A1',root/'ejecucion/A2'),
        'cruzada_A2_B2':(root/'ejecucion/A2',root/'ejecucion/B2'),
        'sonda_vs_A1':(prior/'A1',root/'ejecucion/SONDA'),
    }
    computed={}
    for key,(a,b) in comparisons.items():
        if key in r:
            result=comparar_ramas(a,b)
            require(result==r[key],'Numerical audit differs: '+key)
            computed[key]=result['igualdad_exacta']
    post=json.loads((root/'POSTPROCESO_LOCAL.json').read_text(encoding='utf-8'))
    sonda=comparar_ramas(prior/'A1',root/'ejecucion/SONDA')
    require(sonda==post['sonda_vs_A1'],'Recovered SONDA comparison differs')
    computed['sonda_recovered_from_arrays']=sonda['igualdad_exacta']
    if 'benchmark' in r:
        times={k:auditar_tiempo((prior if k in ('A1','B1') else root/'ejecucion')/k)
               for k in ('A1','B1','B2','A2')}
        require(times==r['benchmark']['brazos'],'Timing reconstruction differs')
        a=(times['A1']['pared_medida_s']+times['A2']['pared_medida_s'])/2
        b=(times['B1']['pared_medida_s']+times['B2']['pared_medida_s'])/2
        require(1-b/a==r['benchmark']['reduccion_mediana'],'Reduction differs')
    require(r['modelo_promovido'] is False and r['adecuacion_numerica_aprobada'] is False,
            'Unjustified acceptance or promotion')
    print(json.dumps({'manifest_files':len(m),'state':r['estado'],
                      'comparisons_recomputed':computed,'new_simulations':0},indent=2))

if __name__=='__main__':main()
