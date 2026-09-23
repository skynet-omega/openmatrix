from pathlib import Path
import json,sys,getpass,time,urllib.request,datetime
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
sys.path.insert(0,str(ROOT/'instrumentos/jev_workflow'))
from jev_workflow import MODEL,ENDPOINT,NoRedirect,encode,digest,probability,require
payload=json.loads((D/'prepared_request.json').read_text());questions=payload['questions'];choices=next(iter(questions.values()))['criteria']
require(len(encode(payload))<=20000,'Request budget')

receipt={'model':MODEL,'endpoint':ENDPOINT,'request_sha256':digest(payload),'maximum_requests':1,'api_requests':0,'automatic_retries':0,'maximum_wait_seconds':30,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'advisory_only':True,'automatic_scientific_admission':False}
(D/'request.json').write_bytes(encode(payload));(D/'PLAN.json').write_text(json.dumps({'scope':'Consultation only; no simulator run, no new numerical prototype.','Jev_requests_max':1,'ChatGPT_messages_max':1,'network_timeout_s':30,'secret_storage':False},indent=2)+'\n')
(D/'receipt.json').write_bytes(encode(receipt));key=getpass.getpass('TypeSafe API key (hidden): ');start=time.perf_counter()
try:
 require(bool(key) and not any(c.isspace() for c in key),'Key format')
 receipt.update(api_requests=1,state='REQUEST_STARTED');(D/'receipt.json').write_bytes(encode(receipt))
 request=urllib.request.Request(ENDPOINT,data=encode(payload),method='POST',headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
 with opener.open(request,timeout=30) as response:raw=response.read(100001)
 require(len(raw)<=100000 and key.encode() not in raw,'Response rejected')
 result=json.loads(raw);require(result.get('model')==MODEL,'Model mismatch');require(set(result['answers'])==set(questions),'Answer layout')
 for answer in result['answers'].values():
  require(answer.get('type')=='choice' and answer.get('choice') in choices,'Answer type')
  probs=answer.get('probabilities',{});require(set(probs)==set(choices) and all(probability(v) for v in probs.values()),'Probabilities')
  require(abs(sum(probs.values())-1)<=1e-5 and probs[answer['choice']]>=max(probs.values())-1e-7,'Distribution')
  require(probability(answer.get('confidence')),'Confidence')
 usage=result.get('usage',{});require(all(type(usage.get(k))is int and usage[k]>=0 for k in ('input_tokens','output_tokens')),'Usage')
 (D/'response.json').write_bytes(encode(result));receipt.update(state='COMPLETE_ADVISORY',usage=usage,response_sha256=digest(result))
 print(json.dumps({'answers':result['answers'],'usage':usage,'scientific_acceptance':False},indent=2))
except BaseException as exc:
 receipt.update(state='FAILED_OR_UNCERTAIN',error_type=type(exc).__name__)
 print(json.dumps({'state':receipt['state'],'error_type':type(exc).__name__}))
 raise
finally:
 key=None;receipt['elapsed_s']=time.perf_counter()-start;(D/'receipt.json').write_bytes(encode(receipt))
