#!/usr/bin/env python3
"""Incremental, non-executing Python code inventory for MATRIX.

Usage: code_catalog.py build|stats|search QUERY|duplicates [--root ROOT] [--db DB]
Only .py/.pyi in declared project source/archive areas and root files are covered. Symlinks (including internal
ones), environment/cache/dependency directories and non-Python files are excluded.
AST imports/calls are syntactic evidence, not runtime dispatch or a call graph.
MANIFEST.json presence marks a sealed archive, not validation or verified hashes.
Exact/AST duplicate candidates never imply interchangeable scientific behavior.
The AST digest ignores positions/comments but retains names, literals/docstrings.
Unchanged files reuse hashes when size, mtime, ctime, inode and device agree;
build --rehash forces content verification. No indexed module is imported.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import tokenize

SCHEMA = 1
AREAS = ("src", "scripts", "work", "tests", "reports", "evidence", "models", "storage/source_history",
         "campanas", "motor_nuevo", "investigacion", "instrumentos")
EXCLUDED = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "envs", "environments",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
    "node_modules", "site-packages", "dist-packages", "vendor", "vendors",
    "third_party", "third-party", "dependencies", ".tox", ".nox",
})
MAX_BYTES = 4 * 1024 * 1024


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return (base + "." if base else "") + node.attr
    return ""


class IndexAST(ast.NodeVisitor):
    """Record lexical definitions/imports/calls without evaluating anything."""

    def __init__(self, tree):
        self.entries = []
        self.scope = []
        self.aliases = {}
        # Module imports may occur after a function's definition. Alias resolution
        # is deliberately syntactic; conditional rebinding cannot be certified.
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._aliases(node)

    def _aliases(self, node):
        if isinstance(node, ast.Import):
            for item in node.names:
                self.aliases[item.asname or item.name.split(".")[0]] = (
                    item.name if item.asname else item.name.split(".")[0])
        else:
            prefix = "." * node.level + (node.module or "")
            for item in node.names:
                if item.name != "*":
                    target = prefix + ("." if prefix and not prefix.endswith(".") else "") + item.name
                    self.aliases[item.asname or item.name] = target

    def add(self, node, kind, name, signature="", docstring="", resolved=""):
        qualname = ".".join(self.scope + [name]) if kind in ("class", "function", "async_function") else ".".join(self.scope)
        row = dict(kind=kind, name=name, qualname=qualname,
                   line=node.lineno, end_line=getattr(node, "end_lineno", node.lineno),
                   signature=signature, docstring=docstring, resolved=resolved)
        row["search_text"] = " ".join(str(row[k]) for k in
            ("kind", "name", "qualname", "signature", "docstring", "resolved"))
        self.entries.append(row)

    def definition(self, node, kind):
        if isinstance(node, ast.ClassDef):
            signature = node.name + "(" + ", ".join(ast.unparse(x) for x in node.bases) + ")"
        else:
            signature = node.name + "(" + ast.unparse(node.args) + ")"
            if node.returns is not None:
                signature += " -> " + ast.unparse(node.returns)
        self.add(node, kind, node.name, signature, ast.get_docstring(node, clean=False) or "")
        old_aliases = self.aliases
        self.aliases = dict(old_aliases)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()
        self.aliases = old_aliases

    def visit_ClassDef(self, node):
        self.definition(node, "class")

    def visit_FunctionDef(self, node):
        self.definition(node, "function")

    def visit_AsyncFunctionDef(self, node):
        self.definition(node, "async_function")

    def visit_Import(self, node):
        self._aliases(node)
        for item in node.names:
            self.add(node, "import", item.asname or item.name, resolved=item.name)

    def visit_ImportFrom(self, node):
        self._aliases(node)
        for item in node.names:
            name = item.asname or item.name
            self.add(node, "import", name, resolved=self.aliases.get(name, (node.module or "") + ".*"))

    def visit_Call(self, node):
        name = dotted(node.func)
        if name:
            head, _, tail = name.partition(".")
            resolved = self.aliases.get(head, head) + ("." + tail if tail else "")
            self.add(node, "call", name, resolved=resolved)
        self.generic_visit(node)


def connect(db, root, create=False, use_fts=True):
    db = Path(db)
    if not create and not db.is_file():
        raise ValueError("Index absent; run build first: " + str(db))
    if not create:
        # Queries must read the last committed index while a build is writing.
        # Reinitializing schema/meta here took a writer lock for every search.
        con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA query_only=ON")
            meta = dict(con.execute("SELECT key,value FROM meta"))
            if meta.get("schema") != str(SCHEMA) or meta.get("root") != str(root):
                raise ValueError("Index schema/root mismatch; choose a separate DB or rebuild a new one")
            return con
        except BaseException:
            con.close()
            raise
    if create:
        db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript("""
      CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS files(
        id INTEGER PRIMARY KEY,path TEXT UNIQUE NOT NULL,stat_key TEXT,
        size INTEGER,sha256 TEXT,ast_sha256 TEXT,source TEXT NOT NULL,
        semantic TEXT NOT NULL,role TEXT,base_role TEXT,seal TEXT,error TEXT);
      CREATE TABLE IF NOT EXISTS entries(
        id INTEGER PRIMARY KEY,file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
        kind TEXT,name TEXT,qualname TEXT,line INTEGER,end_line INTEGER,
        signature TEXT,docstring TEXT,resolved TEXT,search_text TEXT);
      CREATE INDEX IF NOT EXISTS entries_file ON entries(file_id);
      CREATE INDEX IF NOT EXISTS files_sha ON files(sha256);
      CREATE INDEX IF NOT EXISTS files_ast ON files(ast_sha256);
    """)
    meta = dict(con.execute("SELECT key,value FROM meta"))
    if meta and (meta.get("schema") != str(SCHEMA) or meta.get("root") != str(root)):
        con.close()
        raise ValueError("Index schema/root mismatch; choose a separate DB or rebuild a new one")
    for key, value in (("schema", str(SCHEMA)), ("root", str(root))):
        con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
    if "fts" not in meta:
        enabled = False
        if use_fts:
            try:
                con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS file_fts USING fts5(path,source,semantic)")
                enabled = True
            except sqlite3.OperationalError:
                pass
        con.execute("INSERT OR REPLACE INTO meta VALUES ('fts',?)", (str(int(enabled)),))
    con.commit()
    return con


def walk_sources(root):
    """Return paths and explicit coverage. Never enter or read symlinks."""
    paths, seals = [], set()
    coverage = dict(roots=list(AREAS), root_files=True, extensions=[".py", ".pyi"], missing_roots=[],
                    excluded_directory_names=sorted(EXCLUDED), excluded_directories=[],
                    skipped_symlinks=[], non_python_files=0, walk_errors=[])
    def error(exc):
        coverage["walk_errors"].append(str(exc))
    try:
        for path in sorted(root.iterdir()):
            if path.suffix not in ('.py','.pyi'):
                continue
            if path.is_symlink():
                coverage['skipped_symlinks'].append(path.name)
            elif path.is_file():
                paths.append(path)
    except OSError as exc:
        error(exc)
    for area in AREAS:
        start = root / area
        if start.is_symlink():
            coverage["skipped_symlinks"].append(area)
            continue
        if not start.is_dir():
            coverage["missing_roots"].append(area)
            continue
        for directory, dirs, names in os.walk(start, followlinks=False, onerror=error):
            folder = Path(directory)
            keep = []
            for name in sorted(dirs):
                child = folder / name
                rel = child.relative_to(root).as_posix()
                if child.is_symlink():
                    coverage["skipped_symlinks"].append(rel)
                elif name in EXCLUDED:
                    coverage["excluded_directories"].append(rel)
                else:
                    keep.append(name)
            dirs[:] = keep
            marker = folder / "MANIFEST.json"
            if "MANIFEST.json" in names and not marker.is_symlink():
                seals.add(folder)
            for name in sorted(names):
                path = folder / name
                if path.is_symlink():
                    coverage["skipped_symlinks"].append(path.relative_to(root).as_posix())
                elif path.suffix in (".py", ".pyi"):
                    paths.append(path)
                else:
                    coverage["non_python_files"] += 1
    return paths, seals, coverage


def build(root, db, rehash=False, use_fts=True):
    root = Path(root).resolve()
    con = connect(db, root, create=True, use_fts=use_fts)
    paths, seals, coverage = walk_sources(root)
    old = {r["path"]: dict(r) for r in con.execute("SELECT id,path,stat_key,role,seal,error FROM files")}
    seen, counts = set(), dict(parsed=0, reused=0, failed=0, removed=0)
    fts = con.execute("SELECT value FROM meta WHERE key='fts'").fetchone()[0] == "1"
    with con:
        for path in paths:
            rel = path.relative_to(root).as_posix()
            seen.add(rel)
            area=rel.split('/')[0]
            base_role = ('active_source' if area in ('src','scripts') or '/' not in rel else
                         'test' if area=='tests' else
                         'historical_source' if area in ('evidence','models','storage','reports') else 'experiment')
            parent_seal = next((p / "MANIFEST.json" for p in path.parents if p in seals), None)
            seal = parent_seal.relative_to(root).as_posix() if parent_seal else ""
            role = "sealed_archive" if seal else base_role
            stat_key, size = "", 0
            source, semantic, content_hash, ast_hash, error, entries = "", "", None, None, "", []
            try:
                st = path.stat(follow_symlinks=False)
                if not stat.S_ISREG(st.st_mode):
                    raise ValueError("Not a regular file; not read")
                stat_key = dumps([st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev])
                size = st.st_size
                previous = old.get(rel)
                if previous and previous["stat_key"] == stat_key and not rehash and not previous["error"]:
                    con.execute("UPDATE files SET role=?,base_role=?,seal=? WHERE id=?", (role, base_role, seal, previous["id"]))
                    counts["reused"] += 1
                    continue
                if size > MAX_BYTES:
                    raise ValueError("File exceeds 4 MiB safety limit; not read")
                # O_NOFOLLOW also prevents a raced replacement by a symlink.
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as stream:
                    raw = stream.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise ValueError("File grew beyond 4 MiB safety limit; not parsed")
                content_hash = sha(raw)
                encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
                source = raw.decode(encoding)
                tree = ast.parse(source, filename=rel)
                indexer = IndexAST(tree)
                indexer.visit(tree)
                entries = indexer.entries
                semantic = "\n".join(e["search_text"] for e in entries)
                ast_hash = sha(ast.dump(tree, include_attributes=False).encode())
                counts["parsed"] += 1
            except (OSError, SyntaxError, UnicodeError, ValueError, RecursionError) as exc:
                error = type(exc).__name__ + ": " + str(exc)
                counts["failed"] += 1
            if rel in old:
                file_id = old[rel]["id"]
                con.execute("DELETE FROM entries WHERE file_id=?", (file_id,))
                con.execute("DELETE FROM files WHERE id=?", (file_id,))
                if fts:
                    con.execute("DELETE FROM file_fts WHERE rowid=?", (file_id,))
            else:
                file_id = None
            # Explicit names protect the schema from positional drift.
            cur = con.execute("""INSERT INTO files(id,path,stat_key,size,sha256,ast_sha256,source,semantic,role,base_role,seal,error)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", (file_id, rel, stat_key, size, content_hash, ast_hash,
                source, semantic, role, base_role, seal, error))
            file_id = cur.lastrowid
            con.executemany("""INSERT INTO entries(file_id,kind,name,qualname,line,end_line,signature,docstring,resolved,search_text)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", [(file_id, *(e[k] for k in
                ("kind", "name", "qualname", "line", "end_line", "signature", "docstring", "resolved", "search_text"))) for e in entries])
            if fts:
                con.execute("INSERT INTO file_fts(rowid,path,source,semantic) VALUES (?,?,?,?)", (file_id, rel, source, semantic))
        # A traversal error must not turn inaccessible sources into deletions.
        if not coverage["walk_errors"]:
            for rel in old.keys() - seen:
                file_id = old[rel]["id"]
                con.execute("DELETE FROM files WHERE id=?", (file_id,))
                if fts:
                    con.execute("DELETE FROM file_fts WHERE rowid=?", (file_id,))
                counts["removed"] += 1
        coverage.update(discovered_python_files=len(paths), deletion_pruning_suppressed=bool(coverage["walk_errors"]))
        for key, value in (("coverage", dumps(coverage)), ("last_build", dumps(counts)),
                           ("built_utc", dt.datetime.now(dt.timezone.utc).isoformat())):
            con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
    con.close()
    return stats(root, db)


def stats(root, db):
    con = connect(db, Path(root).resolve())
    meta = dict(con.execute("SELECT key,value FROM meta"))
    result = dict(schema=SCHEMA, root=meta["root"], db=str(Path(db).resolve()),
                  built_utc=meta.get("built_utc"), fts5=meta["fts"] == "1",
                  files=con.execute("SELECT count(*) FROM files").fetchone()[0],
                  entries=con.execute("SELECT count(*) FROM entries").fetchone()[0],
                  roles=dict(con.execute("SELECT role,count(*) FROM files GROUP BY role")),
                  errors=[dict(r) for r in con.execute("SELECT path,error FROM files WHERE error!='' ORDER BY path")],
                  coverage=json.loads(meta.get("coverage", "{}")),
                  last_build=json.loads(meta.get("last_build", "{}")),
                  limits=["Python only; no notebooks, archives, non-Python or external library roots.",
                          "MANIFEST.json presence, not integrity verification or scientific validation.",
                          "Lexical AST and text search; no runtime dispatch, inferred types or dynamic imports.",
                          "active_source describes location, not proof of current invocation."])
    con.close()
    return result


def search(root, db, query, limit=20, kind=None):
    terms = query.lower().split()
    if not terms:
        raise ValueError("Search query must not be empty")
    con = connect(db, Path(root).resolve())
    # Literal substring AND semantics also find qDNa02, driving_force and paths.
    # instr avoids treating % or _ in user queries as SQL wildcards. FTS remains
    # available to consumers, but substring fallback is always complete here.
    expr = "lower(path || char(10) || source || char(10) || semantic)"
    where = " AND ".join("instr(" + expr + ",?)>0" for _ in terms)
    files = con.execute("SELECT * FROM files WHERE " + where + " ORDER BY path", terms).fetchall()
    found = []
    for file in files:
        entries = [dict(e) for e in con.execute("SELECT * FROM entries WHERE file_id=? ORDER BY line,id", (file["id"],))]
        hits = {}
        for e in entries:
            if kind and e["kind"] != kind:
                continue
            text = e["search_text"].lower()
            if all(t in text for t in terms):
                e["match_basis"] = "ast"
                hits.setdefault(e["line"], e)
            elif not all(t in file["path"].lower() for t in terms) and all(t in file["path"].lower() + " " + text for t in terms):
                e["match_basis"] = "ast_and_path"
                hits.setdefault(e["line"], e)
        if not kind or kind == "text":
            for number, line in enumerate(file["source"].splitlines(), 1):
                direct = all(t in line.lower() for t in terms)
                mixed = not all(t in file["path"].lower() for t in terms) and all(t in (file["path"] + " " + line).lower() for t in terms)
                if direct or mixed:
                    hits.setdefault(number, dict(kind="text", name="", qualname="", line=number,
                        end_line=number, signature="", docstring="", resolved="",
                        match_basis=("comment_line" if line.lstrip().startswith("#") else "source_line") if direct else "source_line_and_path"))
            if not hits and all(t in file["path"].lower() for t in terms):
                hits[1] = dict(kind="path", name="", qualname="", line=1, end_line=1,
                    signature="", docstring="", resolved="", match_basis="path_only")
        lines = file["source"].splitlines()
        for e in hits.values():
            item = {key: e[key] for key in ("kind", "name", "qualname", "line", "end_line", "signature", "resolved", "match_basis")}
            item.update(path=file["path"], role=file["role"], base_role=file["base_role"],
                        seal_marker=file["seal"] or None, validation="not_inferred",
                        sha256=file["sha256"], ast_sha256=file["ast_sha256"],
                        parse_error=file["error"] or None,
                        docstring=e["docstring"][:500],
                        snippet=lines[e["line"] - 1].strip()[:400] if e["line"] <= len(lines) else "")
            item["_score"] = (2 if e["match_basis"] == "path_only" else 1 if e["match_basis"] == "comment_line" else 0,
                0 if e["name"].lower() == query.lower() else 1,
                0 if e["kind"] in ("function", "class", "async_function") else 1,
                0 if file["base_role"] == "active_source" else 1, file["path"], e["line"])
            found.append(item)
    con.close()
    found.sort(key=lambda e: e.pop("_score"))
    return dict(query=query, matching_files=len(files), matching_locations=len(found),
                returned=min(limit, len(found)), truncated=len(found)>limit,
                results=found[:limit], semantics="Literal case-insensitive terms AND; import aliases are lexical, not runtime certified.")


def duplicates(root, db, mode="both", limit=20):
    con = connect(db, Path(root).resolve())
    result = {}
    for name, column in (("exact", "sha256"), ("ast", "ast_sha256")):
        if mode not in ("both", name):
            continue
        groups = con.execute(f"SELECT {column} digest,count(*) n FROM files WHERE {column} IS NOT NULL GROUP BY {column} HAVING count(*)>1 ORDER BY n DESC,digest").fetchall()
        result[name] = dict(total_groups=len(groups), truncated=len(groups)>limit, groups=[])
        for group in groups[:limit]:
            members = [dict(r) for r in con.execute(f"SELECT path,role,seal,sha256,ast_sha256 FROM files WHERE {column}=? ORDER BY path", (group["digest"],))]
            result[name]["groups"].append(dict(digest=group["digest"], count=group["n"], files=members))
    con.close()
    return dict(candidates=result, interpretation="No deletion. Exact bytes or same normalized AST are not scientific equivalence, runtime equivalence or validation.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=argparse.SUPPRESS)
    parser.add_argument("--db", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "stats", "search", "duplicates"):
        command = sub.add_parser(name)
        command.add_argument("--root", default=argparse.SUPPRESS)
        command.add_argument("--db", default=argparse.SUPPRESS)
        if name == "build":
            command.add_argument("--rehash", action="store_true")
            command.add_argument("--no-fts", action="store_true", help="For a new DB, force substring-only fallback")
        if name == "search":
            command.add_argument("query", nargs="+")
            command.add_argument("--kind", choices=["function", "async_function", "class", "call", "import", "text"])
        if name in ("search", "duplicates"):
            command.add_argument("--limit", type=int, default=20)
        if name == "duplicates":
            command.add_argument("--mode", choices=["both", "exact", "ast"], default="both")
    args = parser.parse_args(argv)
    root = Path(getattr(args, "root", Path(__file__).resolve().parents[1])).resolve()
    db = Path(getattr(args, "db", root / "catalogo/code.sqlite")).resolve()
    try:
        if getattr(args, "limit", 1) < 1:
            raise ValueError("--limit must be positive")
        if args.command == "build":
            result = build(root, db, args.rehash, not args.no_fts)
        elif args.command == "stats":
            result = stats(root, db)
        elif args.command == "search":
            result = search(root, db, " ".join(args.query), args.limit, args.kind)
        else:
            result = duplicates(root, db, args.mode, args.limit)
        print(dumps(result))
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(dumps(dict(error=type(exc).__name__ + ": " + str(exc))))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
