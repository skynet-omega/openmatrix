"""Independent arithmetic audit of a recorded failed event, without replay."""
from __future__ import annotations

from decimal import Decimal, localcontext
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def d(x):return Decimal.from_float(float(x))


def main():
    p=HERE/'full_sham_capture_01/DOMAIN_CAPTURE.json'
    x=json.loads(p.read_text())
    port,owner=x['port'],x['owner']
    if x['row']!=28296 or x['port_q_row']!=28296:
        raise ValueError('Captured coordinate does not belong to expected KC port')
    same=(port['q0']==owner['q0'] and port['s0']==owner['s0'] and
          port['tau_q_s']==owner['tau_q_s'] and port['count']==len(owner['times_s']) and
          port['times_s']==owner['times_s'] and port['jumps']==owner['jumps'])
    t=d(x['query_time_s']);tau=d(port['tau_q_s'])
    with localcontext() as ctx:
        ctx.prec=80
        q=d(port['q0'])*(-t/tau).exp()
        event_values=[]
        for te,j in zip(port['times_s'],port['jumps']):
            if te<=x['query_time_s']:
                q+=d(j)*(-(t-d(te))/tau).exp()
                at_event=d(port['q0'])*(-d(te)/tau).exp()
                for tj,jj in zip(port['times_s'],port['jumps']):
                    if tj<=te:at_event+=d(jj)*(-(d(te)-d(tj))/tau).exp()
                event_values.append(str(at_event))
        delta=q-Decimal(1)
    attempted=x['attempted_fine_q']
    if not same:classification='PORT_OWNER_TRANSFER_DIFFERENCE'
    elif delta>0:classification='ENCODED_EVENTS_ABOVE_DOMAIN_IN_HIGH_PRECISION'
    elif attempted>1:classification='FP64_EVALUATION_ABOVE_DOMAIN'
    else:classification='FAILURE_NOT_REPRODUCED_BY_PORT_TOTAL'
    report={'schema':'failed_event_arithmetic_audit_v1',
            'source':str(p),'owner_port_inputs_exact':same,
            'high_precision_encoded_q':str(q),'high_precision_minus_one':str(delta),
            'event_post_q_from_encoded_values':event_values,
            'port_cpu_formula':port['projected_q_cpu_formula'],
            'owner_numpy_at_q':owner['at_q_cpu'],
            'attempted_fine_q':attempted,
            'classification':classification,
            'scope':'80-digit Decimal evaluation of stored binary64 inputs. Does not reveal the exact physiological continuous state or validate a correction.'}
    (HERE/'ARITHMETIC_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
