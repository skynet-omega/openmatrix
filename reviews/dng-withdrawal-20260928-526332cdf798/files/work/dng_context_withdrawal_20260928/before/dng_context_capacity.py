"""Direct input accounting and fixed-consumer box bounds, without CNS imports.

The campaign49 codec/verifier are explicit, hashed dependencies, not implicit
imports. Anatomy selects groups before effects. A box bound is not an attainable
neural state; FP64 accounting is not the consumer's FP32 reduction.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import time
import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_check(inputs):
    receipt = json.loads(Path(inputs['freeze']).read_text())
    need(sha(inputs['contract']) == receipt['contract_sha256'], 'frozen contract changed')
    c = json.loads(Path(inputs['contract']).read_text())
    need(c['schema'] == 'matrix_dng_context_capacity_v1', 'contract schema')
    need(set(inputs) == set(c['bindings']) | {'contract', 'freeze', 'source'}, 'undeclared or missing inputs')
    for key, expected in c['bindings'].items():
        need(sha(inputs[key]) == expected, 'frozen input changed: ' + key)
    need(sha(inputs['source']) == sha(__file__), 'adapter identity')
    need(c['transmission_domain'] == [0., 1.], 'unsupported transmission box')
    return c


def phases(z):
    """All accepted/committed RHS and the unambiguous stage-0 alignment anchors."""
    r = z['records']
    accepted = []
    anchors = []
    clocks = []
    for epoch, (begin, end) in enumerate(zip(z['offsets'][:-1], z['offsets'][1:])):
        if not z['committed'][epoch]:
            continue
        indices = np.flatnonzero(r[begin:end, 402] == 1) + begin
        need(len(indices) > 0, 'empty committed epoch')
        first = int(indices[0])
        need(r[first, 384] == 0 and r[first, 14] == 0, 'anchor must be first accepted RHS at local zero')
        accepted.extend(indices.tolist())
        anchors.append(first)
        clocks.append(int(z['start_ns'][epoch]))
    need(len(set(clocks)) == len(clocks) and clocks == sorted(clocks), 'ambiguous absolute clock')
    return np.asarray(accepted), np.asarray(anchors), np.asarray(clocks, np.int64)


def group_bound(operands, mask, drive, theta, warp):
    """Preserve original edge positions, products and warp order for upper net."""
    o = np.asarray(operands)
    need(o.dtype == np.float32 and o.shape[-1] == 4, 'FP32 operand schema')
    need(mask.dtype == np.bool_ and mask.shape == (o.shape[-2],), 'group mask shape')
    need(np.isfinite(o).all(), 'nonfinite operands')
    w, q, cap, included = (o[..., i] for i in range(4))
    need(np.all(cap >= 0), 'negative cap')
    need(np.all((included == 0) | (included == 1)), 'nonbinary inclusion')
    need(np.all((q >= 0) & (q <= 1)), 'consumed transmission outside frozen domain')
    terms = np.where(included != 0, w * (q * cap), np.float32(0))
    upper_q = q.copy()
    upper_q = np.where(mask & (w > 0), np.float32(1), upper_q)
    upper_q = np.where(mask & (w < 0), np.float32(0), upper_q)
    upper_terms = np.where(included != 0, w * (upper_q * cap), np.float32(0))
    net = warp(terms)
    upper_net = warp(upper_terms)
    margin = np.float32(net + np.asarray(drive, np.float32)) - np.asarray(theta, np.float32)
    upper_margin = np.float32(upper_net + np.asarray(drive, np.float32)) - np.asarray(theta, np.float32)
    need(np.isfinite(upper_margin).all() and np.all(upper_margin >= margin), 'nonfinite/nonmonotone upper bound')
    group = np.where(mask, terms, np.float32(0)).sum(axis=-1, dtype=np.float64)
    outside = np.where(~mask, terms, np.float32(0)).sum(axis=-1, dtype=np.float64)
    positive = np.where(mask & (terms > 0), terms, np.float32(0)).sum(axis=-1, dtype=np.float64)
    negative = np.where(mask & (terms < 0), terms, np.float32(0)).sum(axis=-1, dtype=np.float64)
    upper_group64 = np.where(mask & (included != 0), np.maximum(w.astype(float)*cap, 0), 0).sum(axis=-1)
    return dict(observed_group_product_sum64=group, observed_positive_product_sum64=positive,
                observed_negative_product_sum64=negative, outside_product_sum64=outside,
                consumer_net32=net, consumer_margin32=margin,
                reduction_residual64=net.astype(float)-group-outside,
                upper_group_real64=upper_group64, upper_net32=upper_net,
                upper_margin32=upper_margin, available_increment32=upper_net.astype(float)-net)


def anatomy(inputs, c, selection):
    with Path(inputs['anatomy']).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    need(len(rows) == c['extraction']['rows'], 'anatomy count')
    ids = np.array([int(r['bodyId']) for r in rows], np.int64)
    indices = np.array([int(r['node_index']) for r in rows], np.int64)
    need(np.array_equal(indices, np.arange(len(rows))) and len(np.unique(ids)) == len(ids), 'anatomy ordering/identity')
    need(np.array_equal(ids[selection['pre_rows']], selection['pre_ids']), 'presynaptic identity/row disagreement')
    need(np.array_equal(ids[selection['target_rows']], selection['target_ids']), 'target identity/row disagreement')
    frozen = json.loads(Path(inputs['selectors']).read_text())
    need(set(frozen) == set(c['groups']), 'selector groups')
    masks = {}
    for name, rule in c['groups'].items():
        chosen = [int(r['bodyId']) for r in rows if r[rule['column']] in rule['values']
                  and int(r['bodyId']) not in rule['exclude_ids']]
        need(chosen == frozen[name], 'anatomical selection changed: ' + name)
        masks[name] = np.isin(selection['pre_ids'], chosen)
    primary = [masks[k] for k in c['primary_groups']]
    need(np.all(sum(primary) <= 1), 'primary groups overlap')
    return rows, masks, {k:len(v) for k,v in frozen.items()}


def ranges(values):
    return {'min': float(np.min(values)), 'max': float(np.max(values))}


def save_csv(path, rows):
    need(len(rows) > 0, 'empty table')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(inputs, parameters, output):
    start = time.process_time()
    need(not parameters, 'parameters belong in frozen contract')
    c = freeze_check(inputs)
    codec = load_module(inputs['codec_source'], '_dng_operand_codec')
    verifier = load_module(inputs['verifier_source'], '_dng_operand_verifier')
    need((verifier.HERE/'frozen_selection.npz').resolve() == Path(inputs['selection']).resolve(), 'implicit verifier dependency not declared')
    with np.load(inputs['selection'], allow_pickle=False) as z:
        selection = {k:z[k] for k in z.files}
    rows, masks, group_sizes = anatomy(inputs, c, selection)
    target_indices = [int(np.flatnonzero(selection['target_ids'] == target)[0]) for target in c['targets']]
    for cell in target_indices:
        ids = selection['pre_ids'][selection['ptr'][cell]:selection['ptr'][cell+1]]
        need(len(np.unique(ids)) == len(ids), 'duplicate presynaptic variable within row')
    summaries, aligned, fidelity = [], [], []
    static_reference = None
    for history in c['histories']:
        for window in c['windows_ms']:
            key = f'{history}_{window:03}'
            z, meta = codec.decode(inputs[key])
            verification = verifier.verify(z)
            verification.pop('margin_ranges')  # the six totals are old evidence, not new discoveries
            accept, anchors, clocks = phases(z)
            need(len(clocks) == 8 and np.all(z['ms'] == 3000 + window), 'window coverage')
            o = z['operands']
            rhs = z['records'][:, :384].reshape(len(o), 4, 6, 16)
            static = o[0, 0, :, :][:, [0, 2, 3]]
            need(np.array_equal(o[..., [0, 2, 3]], np.broadcast_to(static, o[..., [0, 2, 3]].shape)), 'dynamic weight/cap/mask')
            if static_reference is None:
                static_reference = static.copy()
            else:
                need(np.array_equal(static_reference, static), 'weight/cap/mask differ between captures')
            domain = o[accept, :, :, 1]
            need(np.isfinite(domain).all() and np.all((domain >= 0) & (domain <= 1)), 'accepted transmission outside domain')
            fidelity.append(dict(capture=key, **verification, accepted_committed_RHS=len(accept)*4,
                                 aligned_count=len(anchors), consumed_domain=ranges(domain),
                                 decoded_original_sha256=meta['original_npz_sha256']))
            for target, cell in zip(c['targets'], target_indices):
                a, b = selection['ptr'][cell:cell+2]
                incoming = [rows[int(row)] for row in selection['pre_rows'][a:b]]
                sample = o[accept, :, a:b]
                drive, theta = rhs[accept, :, cell, 4], rhs[accept, :, cell, 5]
                need(np.all(rhs[accept, :, cell, 6] > 0), 'destination gain must be positive')
                for group, full_mask in masks.items():
                    mask = full_mask[a:b]
                    result = group_bound(sample, mask, drive, theta, verifier.warp)
                    need(np.array_equal(result['consumer_margin32'], rhs[accept, :, cell, 10]), 'margin identity')
                    weights, caps, included = static[:, 0][a:b], static[:, 1][a:b], static[:, 2][a:b] != 0
                    effective = mask & included & (weights != 0) & (caps > 0)
                    summary = dict(history=history, window_ms=window, target=target, group=group,
                                   global_annotated_neurons=group_sizes[group], incoming_edges=int(mask.sum()),
                                   included_edges=int((mask & included).sum()), effective_edges=int(effective.sum()),
                                   positive_edges=int((effective & (weights > 0)).sum()),
                                   negative_edges=int((effective & (weights < 0)).sum()),
                                   incoming_missing_class=sum(not r['class'] for r in incoming),
                                   incoming_missing_superclass=sum(not r['superclass'] for r in incoming),
                                   group_missing_NT=sum(not r['nt_consensus_nt'] for r,m in zip(incoming,mask) if m),
                                   accepted_committed_RHS=len(accept)*4,
                                   direct_box_excluded_all_samples=bool(np.all(result['upper_margin32'] <= 0)))
                    for name, value in result.items():
                        summary.update({name+'_min':float(value.min()), name+'_max':float(value.max())})
                    summaries.append(summary)
                    # Anchors are indices in the original adaptive-trial array, never paired by trial number.
                    positions = np.searchsorted(accept, anchors)
                    need(np.array_equal(accept[positions], anchors), 'anchor selection not accepted')
                    for point, clock in zip(positions, clocks):
                        row = dict(history=history, window_ms=window, target=target, group=group, absolute_ns=int(clock))
                        row.update({name:float(value[point, 0]) for name, value in result.items()})
                        aligned.append(row)
            del z, o, domain, sample
    lookup = {(r['history'], r['target'], r['group'], r['absolute_ns']):r for r in aligned}
    need(len(lookup) == len(aligned), 'duplicate aligned key')
    differences = []
    for row in aligned:
        if row['history'] != c['histories'][0]:
            continue
        other = lookup.get((c['histories'][1], row['target'], row['group'], row['absolute_ns']))
        need(other is not None and other['window_ms'] == row['window_ms'], 'unmatched absolute clocks')
        diff = {k:row[k] for k in ('window_ms', 'target', 'group', 'absolute_ns')}
        for k in ('observed_group_product_sum64', 'consumer_margin32', 'upper_margin32'):
            diff['profile_minus_sham_'+k] = other[k] - row[k]
        differences.append(diff)
    overall = []
    for group in masks:
        for target in c['targets']:
            ss = [r for r in summaries if r['group'] == group and r['target'] == target]
            ds = [r for r in differences if r['group'] == group and r['target'] == target]
            need(len(ss) == 10 and len(ds) == 40, 'incomplete histories/windows/aligned coverage')
            overall.append(dict(group=group, target=target, effective_edges=ss[0]['effective_edges'],
                                positive_edges=ss[0]['positive_edges'], negative_edges=ss[0]['negative_edges'],
                                observed_group_min=min(s['observed_group_product_sum64_min'] for s in ss),
                                observed_group_max=max(s['observed_group_product_sum64_max'] for s in ss),
                                upper_group_real64=ss[0]['upper_group_real64_min'],
                                maximum_upper_margin32=max(s['upper_margin32_max'] for s in ss),
                                minimum_upper_margin32=min(s['upper_margin32_min'] for s in ss),
                                direct_box_excluded_all_samples=all(s['direct_box_excluded_all_samples'] for s in ss),
                                aligned_delta_group_min=min(d['profile_minus_sham_observed_group_product_sum64'] for d in ds),
                                aligned_delta_group_max=max(d['profile_minus_sham_observed_group_product_sum64'] for d in ds)))
    assessment = dict(schema='matrix_dng_context_capacity_result_v1', raw_verified=True,
                      groups=overall, fidelity=fidelity, new_CNS_ms=0, GPU_calls=0,
                      all_primary_direct_groups_excluded=all(r['direct_box_excluded_all_samples'] for r in overall if r['group'] in c['primary_groups']),
                      group_row_missing_class_counts={str(t):next(s['incoming_missing_class'] for s in summaries if s['target']==t) for t in c['targets']},
                      original_margin_range=[min(s['consumer_margin32_min'] for s in summaries), max(s['consumer_margin32_max'] for s in summaries)],
                      stage4_admission=False, stage5_admission=False, scope=c['scope'],
                      units=c['units'], bound_scope=c['bound_scope'],
                      interpretation='Per-row box capacity with fixed outside at observed evaluations only. Not attainable population activity, stimulation, causality, time average or independent animals.')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out/'windows.csv', summaries)
    save_csv(out/'aligned.csv', aligned)
    save_csv(out/'differences.csv', differences)
    save_csv(out/'groups.csv', overall)
    (out/'assessment.json').write_text(json.dumps(assessment, indent=2, allow_nan=False)+'\n')
    lines = ['# Contexto directo hacia DNg100', '', c['scope'], '',
             'Unidades: entrada nativa del modelo. Intervalos observados, no incertidumbre estadística.', '',
             '| Grupo | DNg100 | Aristas efectivas | Aporte observado | Máximo margen de la cota | Excluido directo en todas las muestras |',
             '|---|---:|---:|---:|---:|---|']
    for r in overall:
        lines.append(f"| {r['group']} | {r['target']} | {r['effective_edges']} | {r['observed_group_min']:.7g} a {r['observed_group_max']:.7g} | {r['maximum_upper_margin32']:.7g} | {r['direct_box_excluded_all_samples']} |")
    lines += ['', 'La cota cambia sólo transmisiones del grupo, maximizando entradas positivas y silenciando negativas. Conserva resto, máscaras, capacidades, pesos, productos y orden FP32. Una cota positiva no demuestra reclutamiento realizable ni locomoción.',
              '', 'Se verifican las diez capturas y las seis filas originales; la pregunta nueva corresponde a las dos DNg100. Comparación sham/profile: 40 instantes absolutos por grupo/destino, RHS0 del primer intento aceptado de cada época comprometida. Las envolventes incluyen todos los RHS aceptados comprometidos.',
              '', 'Los campos de clase ausentes se informan en assessment.json y windows.csv; no se reinterpretan como una clase conocida distinta de MBON. Etapas4/5 abiertas.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    elapsed = time.process_time()-start
    need(elapsed <= c['budget']['CPU_seconds'], 'analysis CPU budget exceeded')
    (out/'execution.json').write_text(json.dumps(dict(CPU_s=elapsed, new_CNS_ms=0, GPU_calls=0), indent=2)+'\n')
    return {'metrics':{'raw_verified':1, 'new_CNS_ms':0,
                       'all_primary_direct_groups_excluded':int(assessment['all_primary_direct_groups_excluded'])},
            'assessment':assessment}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CPU reconstruction; never starts CNS/GPU.')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    need(manifest['schema'] == 'matrix_dng_context_inputs_v1', 'manifest schema')
    base = args.manifest.resolve().parent
    inputs = {k:(base/v).resolve() for k,v in manifest['inputs'].items()}
    need(all(p.is_relative_to(base) for p in inputs.values()), 'inputs outside extracted package')
    need(not args.output.exists(), 'preserve previous output')
    result = evaluate(inputs, {}, args.output)
    print(json.dumps(result['metrics'], allow_nan=False))
