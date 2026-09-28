import math
import json
import hashlib
import numpy as np
import pytest
from pn_frontier_readback import command_check, writer_check, frozen_contract


def test_reader_uses_previous_committed_state_not_same_endpoint():
    q=np.array([[0,0,.001,.002],[0,0,.002,.003]])
    used=np.array([[0,0,.004,.001],q[0]])
    b=np.zeros_like(q)
    y=np.tanh(250*(used[:,2]-used[:,3]))*math.radians(5)
    t={'DN_q_actual':q,'DN_q_usada':used,'DN_baseline':b,
       'command_yaw_rate_rad_s':y,'command_forward_mm_s':np.zeros(2)}
    command_check(t)
    t['DN_q_usada']=q
    with pytest.raises(ValueError,match='phase'):command_check(t)


def test_writer_corruption_inside_extrema_is_detected():
    held=np.ones((2,686),np.float32)*.25
    before={k:held.copy() for k in ('first','last','lo','hi')}
    before['counts']=np.ones((2,686),np.uint64)*8
    after={k:v.copy() for k,v in before.items()}
    writer_check(held,before,after,'self')
    after['hi'][1,137]=.3
    with pytest.raises(ValueError,match='consumed'):writer_check(held,before,after,'self')


def test_writer_counts_are_not_a_substitute_for_value_identity():
    held=np.zeros((2,686),np.float32)
    before={k:held.copy() for k in ('first','last','lo','hi')}
    before['counts']=np.ones((2,686),np.uint64)*8
    after={k:v.copy() for k,v in before.items()}
    for k in ('first','last','lo','hi'):after[k][0,10]=.001
    with pytest.raises(ValueError,match='identity'):writer_check(held,before,after,'live')


def test_changed_criterion_is_detected(tmp_path):
    c=tmp_path/'contract.json';f=tmp_path/'freeze.json'
    c.write_text(json.dumps({'threshold':.002}))
    f.write_text(json.dumps({'contract_sha256':hashlib.sha256(c.read_bytes()).hexdigest()}))
    frozen_contract(c,f)
    c.write_text(json.dumps({'threshold':.2}))
    with pytest.raises(ValueError,match='criteria'):frozen_contract(c,f)
