"""One bounded evidence-classification request, never scientific admission."""
from pathlib import Path
import json,sys,getpass,time,urllib.request,datetime
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
sys.path.insert(0,str(ROOT/'instrumentos/jev_workflow'))
from jev_workflow import MODEL,ENDPOINT,NoRedirect,encode,digest,probability,require
facts={
 'task':'Assess documented evidence for readiness to run interpretable stage3 experiments of a complete coupled organism.',
 'measured_organism':{'simulated_ms':20,'advance_wall_s':73.10815714695491,'independent_one_second_run':False},
 'desired_speed':{'approximate_seconds_wall_per_simulated_second':60,'explicitly_not_a_biological_acceptance_proof':True},
 'refined_comparison':{'brain_snapshot_ms':[1,5],'maximum_normalized_errors':[2.183440538261472e-6,1.7416722058280487e-5],'unchanged_limit':1e-4,'exact_discrete_fields_at_snapshots':True,'reference_completed_ms':14,'failed_during_step':15,'failure':'accepted event state outside domain','offending_index_and_magnitude':'not captured in failed run; unknown','refined20ms_endpoint_available':False},
 'other_evidence':{'old_reference20ms_difference':1.3664493971381544e-4,'old_reference_is_not_ground_truth':True,'source_event_time_cause_located_locally':True,'independent_recorded_filter_replay_passed':True,'seven_clean_extraction_checks_passed':True,'full_organism_clean_reproduction_or_body_resume_tested':False},
 'review':{'ChatGPT':'documentary review does not approve engine; recommends reducing global recalculations with recurrent temporal inputs, compared with a higher-order full-operator backend','no_external_reviewer_executed_arrays':True},
 'roles':'Jev classifies supplied textual facts. It cannot inspect these raw arrays in this request or certify numerical/biological equivalence. Output never launches an experiment or changes criteria.'}
choices={'supported':'The supplied facts provide the evidence explicitly required for the proposition.','not_supported':'The supplied facts explicitly fail at least one required condition for the proposition.','insufficient_information':'Neither support nor contradiction of a required condition is established by the supplied facts.'}
questions={
 'ready_interpretive_stage3':{'type':'choice','instructions':'Classify whether readiness for interpreting biological stage3 outcomes is supported. Required: a usable numerical reference over the planned trajectory and no unresolved numerical failure that prevents interpreting that trajectory. Unit tests or reviewer agreement alone do not supply this evidence. Do not require that every future biological experiment already pass.','criteria':choices},
 'speed_target_met':{'type':'choice','instructions':'Classify whether approximately one simulated second per60wallseconds has been demonstrated with the complete coupled organism. Use actual measurements; do not infer performance from GPU specifications.','criteria':choices},
 'bounded_diagnostics_supported':{'type':'choice','instructions':'Classify whether the supplied evidence supports running bounded engineering diagnostics with retained failures to localize the domain violation or compare numerical backends, clearly excluded from scientific stage3 success claims. It does not require full engine admission first.','criteria':choices},
 'domain_failure_is_roundoff_only':{'type':'choice','instructions':'Classify whether the evidence establishes that the domain failure is only harmless floating-point roundoff. A missing offending coordinate and magnitude do not establish that conclusion.','criteria':choices}}
payload={'model':MODEL,'state':facts,'questions':questions}
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
