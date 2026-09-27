"""Recover study-specific receptor means from a pinned DoOR release, not merged scores."""
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request

HERE = Path(__file__).resolve().parent
BASE = 'https://raw.githubusercontent.com/ropensci/DoOR.data/v2.0.1/'
LIMIT = 8 * 1024**2

def need(ok, message):
    if not ok:
        raise ValueError(message)

def rows(path):
    with path.open(newline='') as f:
        values = list(csv.reader(f, delimiter=';'))
    header = ['csv_row_label'] + values[0]
    need(all(len(v) == len(header) for v in values[1:]), 'Shifted CSV columns: ' + str(path))
    return [dict(zip(header, v)) for v in values[1:]]

def main():
    start = time.monotonic()
    tree = json.loads((HERE / 'door_tree.json').read_text())
    selected = [v for v in tree['tree'] if v['path'].startswith('data/') and
                v['path'].endswith('.csv') and (Path(v['path']).name.startswith(('Or', 'ab')) or
                Path(v['path']).name in ('door_mappings.csv', 'door_dataset_info.csv'))]
    need(sum(v['size'] for v in selected) < LIMIT, 'Dataset budget')
    folder = HERE / 'primary_door'
    folder.mkdir(exist_ok=True)
    def fetch(entry):
        target = folder / Path(entry['path']).name
        if target.exists():
            raw = target.read_bytes()
            fetched = 0
        else:
            with urllib.request.urlopen(BASE + entry['path'], timeout=25) as response:
                raw = response.read(entry['size'] + 1)
            fetched = len(raw)
        need(len(raw) == entry['size'], 'File size')
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        need(blob == entry['sha'], 'Pinned Git blob differs')
        if not target.exists():
            target.write_bytes(raw)
        return dict(path=str(target.relative_to(HERE)), url=BASE + entry['path'],
                    bytes=len(raw), fetched_bytes=fetched, git_blob=blob,
                    sha256=hashlib.sha256(raw).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch, selected))
    profile = []
    mappings = rows(folder / 'door_mappings.csv')
    for source in records:
        p = HERE / source['path']
        if not p.name.startswith(('Or', 'ab')):
            continue
        data = rows(p)
        matches = {key: [x for x in data if x['Name'] == key] for key in ('sfr', '1-hexanol')}
        if not all(len(v) == 1 for v in matches.values()):
            continue
        baseline = matches['sfr'][0].get('Bruyne.2001.WT', 'NA')
        response = matches['1-hexanol'][0].get('Bruyne.2001.WT', 'NA')
        if baseline == 'NA' or response == 'NA':
            continue
        mapping = [m for m in mappings if m['receptor'] == p.stem or m['OSN'] == p.stem]
        glomeruli = sorted(set(m['glomerulus'] for m in mapping if m['adult'] == 'TRUE'))
        profile.append(dict(receptor=p.stem, study='Bruyne.2001.WT', odor='1-hexanol',
                            baseline_Hz=float(baseline), increment_Hz=float(response),
                            reconstructed_total_Hz=float(baseline)+float(response),
                            glomeruli=glomeruli, source=source['path']))
    result = dict(status='STUDY_SPECIFIC_MEANS_EXTRACTED', release='v2.0.1',
                  tree_sha=tree['sha'], records=records, profile=profile,
                  new_bytes=sum(r['fetched_bytes'] for r in records), wall_seconds=time.monotonic()-start,
                  scope='Curated study means, not raw trials, complete odor physiology, or cross-validated predictions.')
    (HERE / 'DOOR_EXTRACTION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(bytes=result['new_bytes'], profile=profile), indent=2))

if __name__ == '__main__':
    main()
