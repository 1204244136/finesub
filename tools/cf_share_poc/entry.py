"""PoC (assessment §5): does the *existing* KnowledgeStore run unchanged on a
Durable Object's SQLite?

Three unknowns, one request:

1. the node store's DDL + migration ladder applies through
   ``ctx.storage.sql.exec()`` (synchronous, so no async rewrite);
2. ``store.begin()`` — which issues ``BEGIN IMMEDIATE`` / ``COMMIT``, both
   forbidden inside ``sql.exec()`` — can be carried by ``transactionSync()``
   with the pair neutralized in the connection adapter;
3. an exception inside that transaction actually rolls the revision back.

Only the connection object is new. ``knode/`` is a byte-for-byte copy of
``src/finesub/llm/knowledge/node/{model,schema,store}.py``.
"""

import json
import sys
import types


def _sqlite3_availability() -> str:
    """The store modules import sqlite3 at module scope (for ``sqlite3.Row``
    annotations and ``sqlite3.OperationalError`` in the migration ladder). If
    Pyodide does not ship it here, a stub keeps those names resolvable — which
    is itself a finding for the port, not a workaround to keep."""

    try:
        import sqlite3  # noqa: F401

        return "real"
    except ImportError:
        pass

    module = types.ModuleType("sqlite3")

    class Row(dict):
        pass

    class Error(Exception):
        pass

    class OperationalError(Error):
        pass

    class IntegrityError(Error):
        pass

    class Connection:
        pass

    class Cursor:
        pass

    module.Row = Row
    module.Error = Error
    module.OperationalError = OperationalError
    module.IntegrityError = IntegrityError
    module.DatabaseError = Error
    module.Connection = Connection
    module.Cursor = Cursor
    module.connect = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no file-backed sqlite3 in a Worker; use the DO adapter")
    )
    sys.modules["sqlite3"] = module
    return "stub"


SQLITE3 = _sqlite3_availability()

from knode.schema import init_schema  # noqa: E402
from knode.store import KnowledgeStore  # noqa: E402
from workers import DurableObject, Response, WorkerEntrypoint  # noqa: E402


def _as_list(value):
    """Pyodide already hands back a Python list here; keep the JsProxy path for
    runtime versions that do not."""

    return value.to_py() if hasattr(value, "to_py") else list(value)


def _as_dict(row):
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_py"):
        return dict(row.to_py())
    return dict(row)


class _Cursor:
    """The slice of sqlite3.Cursor the store actually touches."""

    def __init__(self, rows, lastrowid=None):
        self._rows = rows
        self.lastrowid = lastrowid

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class DoConnection:
    """sqlite3.Connection's surface, backed by a Durable Object's SQL storage.

    Three differences the port has to absorb:

    - ``BEGIN``/``COMMIT``/``ROLLBACK`` cannot go through ``sql.exec()``.
      Atomicity comes from ``transactionSync()`` wrapped around the caller, so
      the pair is neutralized here; a ROLLBACK is counted and swallowed so the
      original exception — the thing transactionSync actually rolls back on —
      keeps propagating.
    - the cursor has no ``lastrowid``; an INSERT pays one extra query.
    - rows come back as JS objects, so ``row["col"]`` needs a dict.
    """

    def __init__(self, sql):
        self._sql = sql
        self.begins = 0
        self.commits = 0
        self.rollbacks = 0
        self.statements = 0

    def execute(self, statement, params=()):
        head = statement.strip().split(None, 1)[0].upper() if statement.strip() else ""
        if head == "BEGIN":
            self.begins += 1
            return _Cursor([])
        if head == "COMMIT":
            self.commits += 1
            return _Cursor([])
        if head == "ROLLBACK":
            self.rollbacks += 1
            return _Cursor([])
        self.statements += 1
        cursor = self._sql.exec(statement, *tuple(params))
        rows = [_as_dict(row) for row in _as_list(cursor.toArray())]
        lastrowid = None
        if head == "INSERT":
            one = _as_dict(self._sql.exec("SELECT last_insert_rowid() AS id").one())
            lastrowid = int(one["id"])
        return _Cursor(rows, lastrowid)

    def close(self):
        pass


_SUBJECT = {"surface": "原神", "intro": "", "category": "common", "section_order": []}


def _store_on(sql) -> KnowledgeStore:
    """The real KnowledgeStore, with only its connection swapped out."""

    store = KnowledgeStore.__new__(KnowledgeStore)
    store.path = None
    store.conn = DoConnection(sql)
    return store


def run_poc(ctx) -> dict:
    storage = ctx.storage
    store = _store_on(storage.sql)
    conn = store.conn
    report = {"sqlite3": SQLITE3, "steps": []}

    # 1. schema
    init_schema(conn)
    tables = sorted(
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    )
    report["steps"].append(
        {
            "step": "init_schema",
            "tables": len(tables),
            "has_node_versions": "node_versions" in tables,
            "schema_version": (conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone() or {}).get("value"),
        }
    )

    # 2. one committed revision through store.begin()
    def write():
        with store.begin("import", note="poc write") as txn:
            txn.create_node("n-poc", "subject", _SUBJECT)
        return store.current_rev()

    committed_rev = storage.transactionSync(write)
    node = store.node("n-poc", committed_rev)
    report["steps"].append(
        {
            "step": "transactionSync commit",
            "rev": committed_rev,
            "node_read_back": None if node is None else node.payload.get("surface"),
            "begin_commit_pairs": [conn.begins, conn.commits],
        }
    )

    # 3. rollback: the store raises inside the transaction, nothing survives
    def failing_write():
        with store.begin("import", note="poc rollback") as txn:
            txn.create_node("n-doomed", "subject", _SUBJECT)
            raise RuntimeError("deliberate")

    rolled_back = "no exception"
    try:
        storage.transactionSync(failing_write)
    except Exception as exc:  # noqa: BLE001 - the point is that it escapes
        rolled_back = f"{type(exc).__name__}: {exc}"
    survivor = store.node("n-doomed", store.current_rev())
    report["steps"].append(
        {
            "step": "transactionSync rollback",
            "raised": rolled_back,
            "doomed_node_present": survivor is not None,
            "rev_after": store.current_rev(),
            "rollbacks_seen_by_adapter": conn.rollbacks,
        }
    )
    report["statements"] = conn.statements
    return report


class StorePoc(DurableObject):
    async def ping(self) -> str:
        return json.dumps({"alive": True, "sqlite3": SQLITE3, "has_ctx": hasattr(self, "ctx")})

    async def poc(self) -> str:
        """An RPC method rather than fetch(): the report crosses the boundary
        as a plain string, so a failure inside shows its own traceback instead
        of a generic DO 500."""

        try:
            body = run_poc(self.ctx)
        except Exception as exc:  # noqa: BLE001 - a PoC reports its own failure
            import traceback

            body = {
                "failed": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-14:],
                "sqlite3": SQLITE3,
            }
        return json.dumps(body, ensure_ascii=False, indent=2)


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        stub = self.env.STORE.get(self.env.STORE.idFromName("poc"))
        if "ping" in request.url:
            body = await stub.ping()
        else:
            body = await stub.poc()
        return Response(body, headers={"Content-Type": "application/json; charset=utf-8"})
