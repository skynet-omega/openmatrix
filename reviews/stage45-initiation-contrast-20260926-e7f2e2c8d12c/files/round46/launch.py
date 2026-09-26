"""New finite execution budget for the unchanged scientific contrast44.

The stopped round44 remains stopped. These two branches belong to round46.
No model, input, scientific threshold or historical artifact is modified.
"""
from pathlib import Path
import hashlib,json,os,resource,signal,subprocess,sys,time
import numpy as np
import psutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'campanas/etapa45_neural_contrast_20260925_44'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,why):
    if not ok:raise RuntimeError(why)
def write(p,x):
    t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)

def main():
    plan=json.loads((HERE/'PLAN.json').read_text());plan_hash=sha(HERE/'PLAN.json')
    need(not (HERE/'QUEUE.json').exists(),'This round has already started; no retry')
    for p,digest in plan['fixed_inputs'].items():need(sha(p)==digest,'Changed input '+p)
    sys.path.insert(0,str(OLD));from run_branch import lock
    lock()
    from verify_sham import calculate
    sham=calculate(OLD/'sham_01');need(sham['classification']=='SHAM_EXACT','Historical sham invalid')
    report=dict(status='RUNNING',plan_sha256=plan_hash,arms=[],neural_ms_max=240,stage4_admission=False,stage5_admission=False)
    start=time.monotonic();cpu_total=0.
    write(HERE/'QUEUE.json',report)
    for arm in ('common','virtual'):
        out=HERE/arm;need(not out.exists(),'Branch exists; no overwrite/retry')
        used={};reason=None;initial_cpu=resource.getrusage(resource.RUSAGE_CHILDREN)
        with (HERE/(arm+'.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,'-B',str(OLD/'run_branch.py'),'--arm',arm,'--out',str(out)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            process=psutil.Process(child.pid);branch=time.monotonic()
            while child.poll() is None:
                try:members=[process]+process.children(recursive=True)
                except psutil.NoSuchProcess:members=[]
                rss=0
                for member in members:
                    try:
                        c=member.cpu_times();used[member.pid]=max(used.get(member.pid,0),c.user+c.system)
                        rss+=member.memory_info().rss
                    except psutil.NoSuchProcess:pass
                cpu=sum(used.values());wall=time.monotonic()-branch
                # Reserve90wall/300CPU for the unchanged runner's failure save.
                if wall>=810 or time.monotonic()-start>=1710:reason='WALL_RESERVE'
                elif cpu>=3700 or cpu_total+cpu>=7700:reason='CPU_RESERVE'
                elif rss>=18*1024**3:reason='RAM'
                if reason:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=75)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(.2)
            child.wait()
        final_cpu=resource.getrusage(resource.RUSAGE_CHILDREN)
        charged=max(sum(used.values()),final_cpu.ru_utime+final_cpu.ru_stime-initial_cpu.ru_utime-initial_cpu.ru_stime)
        cpu_total+=charged
        result=json.loads((out/'RESULT.json').read_text()) if (out/'RESULT.json').exists() else {}
        record=dict(arm=arm,returncode=child.returncode,guard_reason=reason,cpu_s=charged,wall_s=time.monotonic()-branch,
                    completed_ms=result.get('completed_ms'),source_result_sha256=sha(out/'RESULT.json') if result else None)
        report['arms'].append(record);report.update(wall_s=time.monotonic()-start,cpu_s=cpu_total)
        good=reason is None and child.returncode==0 and result.get('status')=='COMPLETE' and result.get('completed_ms')==120
        if good:
            with np.load(out/'traces.npz') as a,np.load(OLD/'sham_01/traces.npz') as s:
                good=set(a.files)==set(s.files) and all(np.array_equal(a[k][:19 if k in ('sensores_pendientes','concentracion_campo') else 20],s[k][:19 if k in ('sensores_pendientes','concentracion_campo') else 20]) for k in a.files)
            record['prefix_matches_historical_sham']=bool(good)
        good=good and cpu_total<=8000 and report['wall_s']<=1800 and record['wall_s']<=900
        if not good:
            report['status']='STOP';write(HERE/'QUEUE.json',report);raise RuntimeError('Round46 stopped; no remaining branch or automatic retry')
        write(HERE/'QUEUE.json',report)
    from verify_pair import calculate as pair
    comparison=pair(HERE/'common',HERE/'virtual')
    comparison.update(execution_round=46,new_execution_plan_sha256=plan_hash,historical_stop44_preserved=True)
    write(HERE/'RESULTADOS.json',comparison)
    need(sha(HERE/'PLAN.json')==plan_hash,'Plan changed during execution')
    for p,digest in plan['fixed_inputs'].items():need(sha(p)==digest,'Input changed during execution')
    report.update(status='COMPLETE',wall_s=time.monotonic()-start,cpu_s=cpu_total,classification=comparison['classification'])
    write(HERE/'QUEUE.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
