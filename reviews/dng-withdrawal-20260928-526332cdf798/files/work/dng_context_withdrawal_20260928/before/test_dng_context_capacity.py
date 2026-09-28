from pathlib import Path
import itertools
import json
import hashlib
import numpy as np
import pytest
from dng_context_capacity import group_bound, phases, freeze_check, load_module


WARP = load_module(Path(__file__).parents[1]/'campanas/etapa45_operands_20260927_49/verify_operands.py', '_test_dng_verifier').warp


def test_bound_silences_negative_inputs_and_respects_exclusions():
    # The huge excluded edge must never contribute to the extremum.
    o = np.array([[2,.25,2,1],[-3,.5,1,1],[1e8,.5,2,0],[4,.25,1,1]], np.float32)
    mask = np.array([True,True,True,False])
    r = group_bound(o, mask, 0, 3, WARP)
    assert r['consumer_margin32'] == -2.5
    assert r['upper_margin32'] == 2
    assert r['observed_group_product_sum64'] == -.5
    assert r['upper_group_real64'] == 4
    for q in itertools.product([0,.5,1], repeat=3):
        trial = o.copy(); trial[:3,1] = q
        current = group_bound(trial, mask, 0, 3, WARP)
        assert current['consumer_margin32'] <= r['upper_margin32']


def test_original_edge_lanes_survive_bound_and_empty_group_is_identity():
    # Cancellation makes reassociation visibly wrong; preserve all 65 positions.
    o = np.zeros((65,4), np.float32);o[:,1:4] = 1
    o[[0,1,32,33,64],0] = [2**25,2,1,4,-2**25]
    mask = np.zeros(65, bool)
    r = group_bound(o, mask, 0, 0, WARP)
    assert r['upper_net32'] == WARP(o[:,0]) == r['consumer_net32']
    assert r['reduction_residual64'] != 0
    mask[1] = True;o[1,1] = 0
    upper = o.copy();upper[1,1] = 1
    r = group_bound(o, mask, 0, 0, WARP)
    assert r['upper_net32'] == WARP(upper[:,0]*(upper[:,1]*upper[:,2]))


@pytest.mark.parametrize('column,value,match',[(1,1.01,'domain'),(1,-.01,'domain'),(2,-1,'cap'),(3,.5,'inclusion'),(0,np.nan,'nonfinite')])
def test_invalid_consumed_operands_are_not_clipped(column, value, match):
    o = np.ones((2,4), np.float32);o[0,column] = value
    with pytest.raises(ValueError, match=match):
        group_bound(o,np.ones(2,bool),0,0,WARP)


def test_anchor_uses_accepted_committed_stage_zero_not_rejected_or_stage_three():
    records = np.zeros((4,404)); records[:,402] = [1,0,1,1]
    records[3,384] = .1
    z = dict(records=records, offsets=np.array([0,1,4]),committed=np.array([False,True]),start_ns=np.array([10,10]))
    accepted, anchor, clocks = phases(z)
    assert accepted.tolist() == [2,3] and anchor.tolist() == [2] and clocks.tolist() == [10]
    records[2,14] = .2
    with pytest.raises(ValueError,match='local zero'):phases(z)


def test_criterion_tamper_is_rejected_before_reading_data(tmp_path):
    c = tmp_path/'c.json';f = tmp_path/'freeze.json'
    c.write_text('{}');f.write_text(json.dumps({'contract_sha256':hashlib.sha256(c.read_bytes()).hexdigest()}))
    c.write_text('{"criterion":"altered"}')
    with pytest.raises(ValueError,match='frozen contract'):freeze_check({'contract':c,'freeze':f})
