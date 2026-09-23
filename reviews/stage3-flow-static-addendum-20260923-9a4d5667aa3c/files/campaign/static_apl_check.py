"""Check the observed PN afferents against the temporary APL operator path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MIDPOINT = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor13_20260922/block_midpoint.py')
LAYOUT = ROOT/'motor_nuevo/native_hybrid_20260922/legacy_sources/gpu_coefficient_layout.py'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    metadata = json.loads((HERE/'smoke_on_02/flow/FLOW_METADATA.json').read_text())
    found = {}
    for body_id in (10176, 10208, 10360, 523769):
        row = metadata['rows'][str(body_id)]
        ids = [int(i) for i, typ in zip(row['pre_ids'], row['pre_types']) if typ == 'APL']
        found[str(body_id)] = ids
    if found != {'10176':[10540], '10208':[10977], '10360':[], '523769':[]}:
        raise ValueError('Unexpected focal APL afferents')
    midpoint = MIDPOINT.read_text()
    layout = LAYOUT.read_text()
    observer = (HERE/'flow_observer.py').read_text()
    tokens = {
        'midpoint_weights':'self.cuda[\'weights\'][positions]=self._apl_base_weights*mid[\'apl_release_gpu\']',
        'layout_view':'view[self.transmission_start + self._apl_gpu_rows] = 1.0',
    }
    if tokens['midpoint_weights'] not in midpoint or tokens['layout_view'] not in layout:
        raise ValueError('Temporary APL operator source changed')
    if '_apl_gpu_positions' in observer or '_apl_release_gpu' in observer:
        raise ValueError('Observer may now handle APL; re-audit before using this finding')
    report = {
        'schema':'stage3_apl_static_omission_v1',
        'APL_afferents_by_post_ID':found,
        'source_sha256':{'block_midpoint.py':sha(MIDPOINT),'gpu_coefficient_layout.py':sha(LAYOUT),'flow_observer.py':sha(HERE/'flow_observer.py')},
        'finding':'The observer reads post-step weights and unmodified stage transmission, while the accepted coefficient path temporarily scales APL weights by local release and sets APL transmission to 1. Both PNs have one APL afferent. This is a concrete omitted operator, not a demonstrated explanation of the entire 0.0059274696 discrepancy.',
        'falsifier':'Capture the APL term at the same PN kernel evaluation and compare its signed difference with the observed residual; also validate all remaining input terms.',
    }
    (HERE/'STATIC_APL_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
