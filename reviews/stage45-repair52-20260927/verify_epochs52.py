"""Reconstruct accepted DNg trajectories and clocks from recorded trials."""
import numpy as np
from verify_qualification52 import need


def verify(a, legacy_q, initial_dn):
    n = len(legacy_q)
    epochs = 16 * n
    need(a['committed'].dtype == np.bool_, 'Commit flag type')
    need(np.array_equal(a['ids'], [10045, 10056]), 'RHS cell identity')
    need(np.array_equal(a['epoch'], np.arange(epochs)), 'Epoch order')
    need(np.array_equal(a['ms'], np.repeat(np.arange(3001, 3001+n), 16)), 'Input context order')
    need(np.array_equal(a['committed'], np.tile([False, True], 8*n)), 'Predictor/committed contexts')
    need(np.array_equal(a['duration_ns'], np.tile([62500, 125000], 8*n)), 'Epoch duration')
    need(np.array_equal(a['start_ns'], 47486000000 + np.repeat(np.arange(8*n)*125000, 2)), 'Epoch start clock')
    need(len(a['offsets']) == epochs+1 and a['offsets'][0] == 0 and a['offsets'][-1] == len(a['records']), 'Record offsets')
    need(np.array_equal(np.diff(a['offsets']), a['trials']), 'Trials per epoch')
    need(np.isfinite(a['records']).all(), 'Nonfinite trials')
    state = initial_dn.copy()
    counted = accepted = rejected = 0
    for e in range(epochs):
        rows = a['records'][a['offsets'][e]:a['offsets'][e+1]]
        need(len(rows) > 0, 'Empty epoch')
        need(np.array_equal(rows[:,139], np.arange(len(rows))), 'Trial index')
        need(np.all(rows[:,132:134] == 0), 'Nonfinite or bound scheduler flags')
        need(np.isin(rows[:,138], [0.,1.]).all(), 'Nonbinary acceptance')
        flags = rows[:,138].astype(bool)
        need(np.array_equal(flags, rows[:,131] <= 1), 'Acceptance predicate')
        need(int(flags.sum()) == a['accepted'][e] and int((~flags).sum()) == a['rejected'][e], 'Acceptance counts')
        need(np.array_equal(rows[0,134:136], state), 'Epoch restored/continued state')
        local = state.copy()
        clock = 0.
        for r, ok in zip(rows, flags):
            need(r[128] == clock and np.array_equal(r[134:136], local), 'Trial state or clock continuity')
            if ok:
                clock = r[130]
                local = r[136:138].copy()
        need(clock == int(a['duration_ns'][e])*1e-9, 'Epoch endpoint time')
        if a['committed'][e]:
            state = local
        if (e+1) % 16 == 0:
            need(np.array_equal(state, legacy_q[(e+1)//16-1,:2]), 'Committed DNg output differs from trajectory')
        counted += len(rows)
        accepted += int(flags.sum())
        rejected += int((~flags).sum())
    return dict(epochs=epochs, trials=counted, accepted=accepted, rejected=rejected,
                committed_contexts=8*n, temporal_reconstruction_exact=True)
