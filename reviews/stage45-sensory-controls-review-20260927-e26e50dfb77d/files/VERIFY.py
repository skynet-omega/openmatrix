"""Verify this portable input-design artifact and two deliberate corruptions."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import PREFLIGHT
import PERMUTATION_CONTROL

HERE = Path(__file__).resolve().parent


def main():
    current = PREFLIGHT.calculate(HERE)
    saved = json.loads((HERE / 'PREFLIGHT_RESULT.json').read_text())
    current.pop('CPU_s'); saved.pop('CPU_s')
    PREFLIGHT.need(current == saved, 'Different preflight arithmetic')
    PREFLIGHT.need(PERMUTATION_CONTROL.calculate() ==
        json.loads((HERE / 'PERMUTATION_RESULT.json').read_text()), 'Different control')
    with tempfile.TemporaryDirectory(prefix='corruption-', dir=HERE) as td:
        td = Path(td)
        for name in ('INPUT_HASHES.json', 'PERFIL_1_HEXANOL.csv', 'CONTRACT.json'):
            shutil.copy2(HERE / name, td / name)
        p = td / 'PERFIL_1_HEXANOL.csv'
        p.write_bytes(p.read_bytes() + b'\n')
        try:
            PREFLIGHT.calculate(td)
        except ValueError as e:
            PREFLIGHT.need(str(e).startswith('Changed input:'), 'Unexpected rejection')
        else:
            raise ValueError('Changed source accepted')
        shutil.copy2(HERE / 'PERFIL_1_HEXANOL.csv', p)
        p = td / 'CONTRACT.json'
        bad = json.loads(p.read_text())
        bad['anatomy']['Or42b']['sides']['L'] += 1
        p.write_text(json.dumps(bad))
        hashes = json.loads((td / 'INPUT_HASHES.json').read_text())
        hashes['CONTRACT.json'] = hashlib.sha256(p.read_bytes()).hexdigest()
        (td / 'INPUT_HASHES.json').write_text(json.dumps(hashes))
        try:
            PREFLIGHT.calculate(td)
        except ValueError as e:
            PREFLIGHT.need(str(e) == 'Population count mismatch', 'Unexpected semantic rejection')
        else:
            raise ValueError('Impossible rehashed population accepted')
    print(json.dumps(dict(recalculated=True, source_corruption_rejected=True,
        rehashed_population_corruption_rejected=True, control_histograms_exact=True,
        neural_steps=0)))


if __name__ == '__main__':
    main()
