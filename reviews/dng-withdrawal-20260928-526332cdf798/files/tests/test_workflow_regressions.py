"""Focused CPU regressions for read queries and evidence-writer lock ownership."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest
import lab_evidence as lab

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('efficiency_code_catalog', ROOT/'scripts/code_catalog.py')
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


def available_lock(root):
    with (root/'catalogo/lab_write.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def evidence_fixture(root):
    (root/'config').mkdir()
    (root/'campanas/one').mkdir(parents=True)
    (root/'campanas/one/README.md').write_text('Resultado observado, sin admisión.')
    (root/'config/lab_evidence_v1.json').write_text(json.dumps({
        'roots':[{'path':'campanas','category':'campaign'}], 'current':{}, 'cell_recordings':{}}))


@pytest.mark.parametrize('operation', ['search', 'stats', 'duplicates'])
def test_code_queries_work_while_another_writer_has_uncommitted_work(tmp_path, monkeypatch, operation):
    (tmp_path/'src').mkdir()
    (tmp_path/'src/a.py').write_text('def traceable_science():\n    return 1\n')
    db=tmp_path/'catalogo/code.sqlite'
    catalog.build(tmp_path, db)
    original_connect=sqlite3.connect
    writer=original_connect(db)
    writer.execute('BEGIN IMMEDIATE')
    writer.execute("UPDATE meta SET value=value WHERE key='built_utc'")
    def fast_timeout(*args, **kwargs):
        kwargs['timeout']=.01
        return original_connect(*args, **kwargs)
    monkeypatch.setattr(catalog.sqlite3, 'connect', fast_timeout)
    try:
        if operation=='search':
            assert catalog.search(tmp_path, db, 'traceable_science')['returned']>0
        else:
            getattr(catalog, operation)(tmp_path, db)
    finally:
        writer.rollback();writer.close()


def test_code_queries_do_not_rewrite_database(tmp_path):
    (tmp_path/'src').mkdir()
    (tmp_path/'src/a.py').write_text('known_value=1\n')
    db=tmp_path/'catalogo/code.sqlite';catalog.build(tmp_path, db)
    before=hashlib.sha256(db.read_bytes()).hexdigest()
    catalog.search(tmp_path, db, 'known_value');catalog.stats(tmp_path, db);catalog.duplicates(tmp_path, db)
    assert hashlib.sha256(db.read_bytes()).hexdigest()==before
    reader=catalog.connect(db,tmp_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("INSERT OR REPLACE INTO meta VALUES ('unexpected','write')")
    finally:reader.close()


def test_lab_schema_error_releases_lease_while_traceback_is_retained(tmp_path):
    evidence_fixture(tmp_path)
    (tmp_path/'catalogo').mkdir()
    db=sqlite3.connect(tmp_path/'catalogo/catalogo.sqlite')
    db.execute('CREATE TABLE lab_records(incompatible INTEGER)');db.commit();db.close()
    with pytest.raises(sqlite3.OperationalError) as caught:
        lab.connect(tmp_path, write=True)
    assert caught.value.__traceback__ is not None
    assert available_lock(tmp_path), 'schema failure leaked the evidence writer lease'


def test_lab_invalid_selection_does_not_hold_writer_lease(tmp_path):
    evidence_fixture(tmp_path);lab.refresh(tmp_path)
    with pytest.raises(ValueError) as caught:
        lab.refresh(tmp_path, only='../outside')
    assert caught.value.__traceback__ is not None
    assert available_lock(tmp_path), 'selection validation leaked the evidence writer lease'
    assert lab.refresh(tmp_path, only='campanas/one')['reused']==1
