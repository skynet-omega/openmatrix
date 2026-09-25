"""Join the verified 0-1000 and 1001-2000 ms trace segments without edits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
PREFIX = HERE/'navigation_minus_filtered_wind_03'
FINISH = HERE/'navigation_minus_filtered_wind_05_continue'
OUT = HERE/'merged_full_01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False)+'\n', encoding='utf-8')


def main():
    a = json.loads((PREFIX/'RESULT.json').read_text(encoding='utf-8'))
    b = json.loads((FINISH/'RESULT.json').read_text(encoding='utf-8'))
    r = json.loads((FINISH/'RESTORATION.json').read_text(encoding='utf-8'))
    need(a['completed_preparation_ms'] == 40 and a['completed_trial_ms'] == 1000
         and a['error']['message'] == 'Unregistered snapshot name',
         'Prefix was not the exact storage-only failure')
    need(b['status'] == 'COMPLETE' and b['completed_continuation_ms'] == 1000
         and r['initial']['exact'], 'Cold continuation or exact restore not complete')
    need(not OUT.exists(), 'Preserve any existing merged result')
    with np.load(PREFIX/'traces.npz', allow_pickle=False) as x, \
         np.load(FINISH/'traces.npz', allow_pickle=False) as y:
        need(set(x.files) == set(y.files), 'Segment fields differ')
        need(len(x['fase']) == 1040 and len(y['fase']) == 1000,
             'Segment lengths differ')
        need(np.array_equal(x['paso'][-1000:], np.arange(1,1001))
             and np.array_equal(y['paso'], np.arange(1001,2001))
             and np.array_equal(y['CNS_time_ns'][:1], x['CNS_time_ns'][-1:]+1_000_000),
             'Seam clock/ordinal differs')
        combined = {key:np.concatenate((x[key],y[key]), axis=0) for key in x.files}
    need(np.all(combined['fase'][:40] == 'preparacion')
         and np.all(combined['fase'][40:] == 'ensayo'), 'Merged phases differ')
    restore_filter = json.loads((FINISH/'FILTER_RESTORE.json').read_text(encoding='utf-8'))
    need(restore_filter['source_trace_sha256'] == sha(PREFIX/'traces.npz')
         and restore_filter['filtered_rad_s'] == float(combined['motor_filter_state_rad_s'][1039]),
         'Filter was not restored from the exact boundary')
    OUT.mkdir()
    np.savez_compressed(OUT/'traces.npz', **combined)
    shutil.copy2(PREFIX/'GAUSSIAN_SPEC.json', OUT/'GAUSSIAN_SPEC.json')
    shutil.copy2(FINISH/'MOTOR_WIND_AUDIT.json', OUT/'MOTOR_WIND_AUDIT.json')
    receipt = {'schema':'stage45_exact_cold_merged_trace_v1',
               'prefix_result_sha256':sha(PREFIX/'RESULT.json'),
               'prefix_trace_sha256':sha(PREFIX/'traces.npz'),
               'prefix_final_snapshot_manifest_sha256':sha(PREFIX/'final_state/MANIFEST.json'),
               'finish_result_sha256':sha(FINISH/'RESULT.json'),
               'finish_trace_sha256':sha(FINISH/'traces.npz'),
               'finish_restoration_sha256':sha(FINISH/'RESTORATION.json'),
               'exact_restored_initial_state':True,
               'filter_restored_exactly':True,
               'phase_rows':{'preparation':40,'trial':2000},
               'joined_trace_sha256':sha(OUT/'traces.npz'),
               'scope':'The prefix failed only at an unregistered snapshot after 1000 accepted ms. The cold restored continuation completed the next 1000 ms. No biological step was skipped or repeated.'}
    write(OUT/'MERGE_RECEIPT.json', receipt)
    write(OUT/'RESULT.json', {'status':'COMPLETE','field':'minus',
                              'completed_preparation_ms':40,'completed_trial_ms':2000,
                              'stage4_pass':None,'stage5_pass':None,
                              'origin':'exactly merged prefix plus cold continuation',
                              'merge_receipt_sha256':sha(OUT/'MERGE_RECEIPT.json')})
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
