"""Record imported local modules and their adjacent native source dependencies."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
EXTENSIONS = {'.py', '.cu', '.cuh', '.cpp', '.hpp', '.h', '.so'}
_executed = set()


def _audit(event, args):
    # Dynamic spec loaders do not always register modules in sys.modules.
    # The exec audit event still exposes their real code filename.
    if event == 'exec':
        name = getattr(args[0], 'co_filename', None)
        if name and name.startswith('/'):
            path = Path(name)
            if path.is_relative_to(ROOT) or path.is_relative_to(OLD):
                _executed.add(path)


sys.addaudithook(_audit)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def imported():
    paths = {p for p in _executed if p.is_file() and p.suffix in EXTENSIONS}
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if not path:
            continue
        path = Path(path).resolve()
        if path.is_relative_to(ROOT) or path.is_relative_to(OLD):
            if path.is_file() and path.suffix in EXTENSIONS:
                paths.add(path)
    # Dynamic compilation reads neighboring native sources without importing
    # a Python module. Include those, and the PN trees referenced by the overlay.
    folders = {p.parent for p in paths} | {HERE/'engine', HERE/'pn',
               ROOT/'motor_nuevo/resident_pn_20260922', ROOT/'motor_nuevo/pn_abc_20260922'}
    for folder in folders:
        paths.update(p for p in folder.iterdir() if p.is_file() and p.suffix in EXTENSIONS - {'.py'})
    return {str(p): sha(p) for p in sorted(paths)}


def verify(mapping):
    changed = [p for p, digest in mapping.items() if not Path(p).is_file() or sha(p) != digest]
    if changed:
        raise RuntimeError('Source identity changed: ' + str(changed[:5]))


def check_loaded(lock):
    current = imported()
    missing = sorted(set(current)-set(lock))
    changed = [p for p in current.keys() & lock.keys() if current[p] != lock[p]]
    if missing or changed:
        raise RuntimeError('Unfrozen local execution source: ' + str((missing+changed)[:8]))
    return current
