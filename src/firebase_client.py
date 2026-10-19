"""Firestore access layer with an automatic local-JSON fallback.

``get_database()`` returns a live Cloud Firestore repository when
``serviceAccountKey.json`` exists in the project root, otherwise a
``LocalMockFirestore`` that persists to ``local_mock_db.json``. Both expose the
exact same method signatures, and both log every operation to ``QUERY_LOG`` so
the Developer tab can show the queries and JSON payloads.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
SERVICE_KEY_FILE = "serviceAccountKey.json"
MOCK_DB_FILE = "local_mock_db.json"

INCOME_CYCLES = "income_cycles"
TRANSACTIONS = "transactions"
DEBTS = "debts"
SAVINGS_ACCOUNTS = "savings_accounts"
COLLECTIONS = (INCOME_CYCLES, TRANSACTIONS, DEBTS, SAVINGS_ACCOUNTS)

# A filter is (field, operator, value); a write op is (action, collection, doc_id, data).
_OPERATORS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "in": lambda a, b: a in b,
}
_ACTIONS = ("set", "update", "delete")


def _jsonable(obj: Any) -> Any:
    return None if obj is None else json.loads(json.dumps(obj, default=str))


def _matches(doc: dict, filters: list[tuple]) -> bool:
    for field, op, value in filters:
        if field not in doc:
            return False
        try:
            if not _OPERATORS[op](doc[field], value):
                return False
        except TypeError:
            return False
    return True


def _query_repr(collection: str, filters: list[tuple]) -> str:
    text = f"db.collection({collection!r})"
    for field, op, value in filters:
        text += f".where({field!r}, {op!r}, {value!r})"
    return text + ".stream()"


class QueryLogger:
    """Thread-safe ring buffer of every Firestore operation performed."""

    def __init__(self, max_entries: int = 300) -> None:
        self._entries: list[dict] = []
        self._max = max_entries
        self._lock = threading.Lock()

    def log(self, backend: str, operation: str, collection: str, query: str,
            result_count: Optional[int] = None, payload: Any = None) -> None:
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "backend": backend,
            "operation": operation,
            "collection": collection,
            "query": query,
            "result_count": result_count,
            "payload": _jsonable(payload),
        }
        with self._lock:
            self._entries.append(entry)
            del self._entries[:-self._max]

    def recent(self, limit: Optional[int] = None) -> list[dict]:
        with self._lock:
            items = list(reversed(self._entries))
        return items[:limit] if limit else items

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


QUERY_LOG = QueryLogger()


class BaseRepository:
    """Public, logged API shared by the mock and live backends."""

    backend_name = "base"
    is_live = False
    supports_reset = False

    def __init__(self, logger: QueryLogger = QUERY_LOG) -> None:
        self.logger = logger

    # -- backend hooks -------------------------------------------------
    def _raw_get(self, collection: str, doc_id: str) -> Optional[dict]:
        raise NotImplementedError

    def _raw_query(self, collection: str, filters: list[tuple]) -> list[dict]:
        raise NotImplementedError

    def _raw_commit(self, ops: list[tuple]) -> None:
        """Apply all ops atomically (all succeed or none are applied)."""
        raise NotImplementedError

    def _raw_snapshot(self) -> dict[str, list[dict]]:
        raise NotImplementedError

    # -- public API ----------------------------------------------------
    def get(self, collection: str, doc_id: str) -> Optional[dict]:
        doc = self._raw_get(collection, doc_id)
        self.logger.log(self.backend_name, "READ", collection,
                        f"db.collection({collection!r}).document({doc_id!r}).get()",
                        1 if doc else 0, doc)
        return doc

    def query(self, collection: str, filters: Optional[Iterable[tuple]] = None) -> list[dict]:
        flt = [tuple(f) for f in (filters or [])]
        for _, op, _ in flt:
            if op not in _OPERATORS:
                raise ValueError(f"Unsupported operator {op!r}.")
        docs = self._raw_query(collection, flt)
        self.logger.log(self.backend_name, "QUERY", collection,
                        _query_repr(collection, flt), len(docs), docs)
        return docs

    def set(self, collection: str, doc_id: str, data: dict) -> None:
        self._write([("set", collection, doc_id, data)])

    def update(self, collection: str, doc_id: str, fields: dict) -> None:
        self._write([("update", collection, doc_id, fields)])

    def delete(self, collection: str, doc_id: str) -> None:
        self._write([("delete", collection, doc_id, None)])

    def batch_write(self, ops: Iterable[tuple]) -> None:
        """Atomic multi-document write (Firestore ``WriteBatch`` semantics)."""
        self._write(list(ops))

    def snapshot(self) -> dict[str, list[dict]]:
        """Unlogged read of every collection (used by the Developer tab)."""
        return self._raw_snapshot()

    # -- internals -----------------------------------------------------
    def _write(self, ops: list[tuple]) -> None:
        if not ops:
            return
        clean: list[tuple] = []
        for action, collection, doc_id, data in ops:
            if action not in _ACTIONS:
                raise ValueError(f"Unsupported write action {action!r}.")
            if action == "set":
                data = {**data, "id": doc_id}
            clean.append((action, collection, doc_id, copy.deepcopy(data)))
        self._raw_commit(clean)

        if len(clean) == 1:
            action, collection, doc_id, data = clean[0]
            ref = f"db.collection({collection!r}).document({doc_id!r})"
            call = f"{ref}.delete()" if action == "delete" else f"{ref}.{action}(<payload>)"
            self.logger.log(self.backend_name, f"WRITE ({action})", collection, call, 1, data)
        else:
            lines = ["batch = db.batch()"]
            for action, collection, doc_id, _ in clean:
                ref = f"db.collection({collection!r}).document({doc_id!r})"
                lines.append(f"batch.{action}({ref}{'' if action == 'delete' else ', <payload>'})")
            lines.append("batch.commit()  # atomic")
            payload = [{"op": a, "path": f"{c}/{i}", "data": d} for a, c, i, d in clean]
            self.logger.log(self.backend_name, f"BATCH ({len(clean)} writes)",
                            ", ".join(sorted({c for _, c, _, _ in clean})),
                            "\n".join(lines), len(clean), payload)


class LocalMockFirestore(BaseRepository):
    """Drop-in Firestore stand-in persisted to a local JSON file."""

    backend_name = "LocalMockFirestore"
    supports_reset = True

    def __init__(self, path: Optional[os.PathLike] = None, logger: QueryLogger = QUERY_LOG) -> None:
        super().__init__(logger)
        self.path = Path(path) if path else ROOT_DIR / MOCK_DB_FILE
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, dict[str, dict]]:
        data: dict[str, Any] = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{self.path} is not valid JSON ({exc}). Fix or delete it to continue."
                ) from exc
        return {name: dict(data.get(name, {})) for name in COLLECTIONS}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _raw_get(self, collection: str, doc_id: str) -> Optional[dict]:
        with self._lock:
            doc = self._data.get(collection, {}).get(doc_id)
            return copy.deepcopy(doc) if doc is not None else None

    def _raw_query(self, collection: str, filters: list[tuple]) -> list[dict]:
        with self._lock:
            return [copy.deepcopy(d) for d in self._data.get(collection, {}).values()
                    if _matches(d, filters)]

    def _raw_commit(self, ops: list[tuple]) -> None:
        with self._lock:
            staged = copy.deepcopy(self._data)  # validate everything before touching real state
            for action, collection, doc_id, data in ops:
                docs = staged.setdefault(collection, {})
                if action == "set":
                    docs[doc_id] = data
                elif action == "update":
                    if doc_id not in docs:
                        raise KeyError(f"No document to update: {collection}/{doc_id}")
                    docs[doc_id].update(data)
                else:
                    docs.pop(doc_id, None)
            self._data = staged
            self._save()

    def _raw_snapshot(self) -> dict[str, list[dict]]:
        with self._lock:
            return {c: copy.deepcopy(list(self._data.get(c, {}).values())) for c in COLLECTIONS}

    def reset(self) -> None:
        with self._lock:
            self._data = {name: {} for name in COLLECTIONS}
            self._save()
        self.logger.log(self.backend_name, "RESET", "*", "local_mock_db.json cleared", None, None)


class LiveFirestore(BaseRepository):
    """Cloud Firestore via ``firebase-admin`` (requires serviceAccountKey.json)."""

    backend_name = "Cloud Firestore"
    is_live = True

    def __init__(self, key_path: os.PathLike, logger: QueryLogger = QUERY_LOG) -> None:
        super().__init__(logger)
        import firebase_admin
        from firebase_admin import credentials, firestore

        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(credentials.Certificate(str(key_path)))
        self._db = firestore.client()
        # Fail fast on bad credentials / no network so we can fall back to the mock.
        list(self._db.collection(INCOME_CYCLES).limit(1).stream(timeout=10))

    def _raw_get(self, collection: str, doc_id: str) -> Optional[dict]:
        snap = self._db.collection(collection).document(doc_id).get()
        return {**snap.to_dict(), "id": snap.id} if snap.exists else None

    def _raw_query(self, collection: str, filters: list[tuple]) -> list[dict]:
        ref = self._db.collection(collection)
        for field, op, value in filters:
            try:
                from google.cloud.firestore_v1.base_query import FieldFilter
                ref = ref.where(filter=FieldFilter(field, op, value))
            except ImportError:  # older google-cloud-firestore
                ref = ref.where(field, op, value)
        return [{**d.to_dict(), "id": d.id} for d in ref.stream()]

    def _raw_commit(self, ops: list[tuple]) -> None:
        batch = self._db.batch()
        for action, collection, doc_id, data in ops:
            ref = self._db.collection(collection).document(doc_id)
            if action == "set":
                batch.set(ref, data)
            elif action == "update":
                batch.update(ref, data)
            else:
                batch.delete(ref)
        batch.commit()

    def _raw_snapshot(self) -> dict[str, list[dict]]:
        return {c: [{**d.to_dict(), "id": d.id} for d in self._db.collection(c).stream()]
                for c in COLLECTIONS}


def get_database(root: Optional[os.PathLike] = None,
                 logger: QueryLogger = QUERY_LOG) -> tuple[BaseRepository, str]:
    """Return ``(repository, status_message)``.

    Uses live Firestore if ``serviceAccountKey.json`` is in ``root`` (default:
    the project root) and connects; otherwise falls back to the local mock.
    """
    base = Path(root) if root else ROOT_DIR
    key = base / SERVICE_KEY_FILE
    if key.exists():
        try:
            return LiveFirestore(key, logger), "Connected to live Cloud Firestore."
        except Exception as exc:  # noqa: BLE001 - any failure should fall back gracefully
            note = f"{SERVICE_KEY_FILE} found but Firestore connection failed ({exc}). Using local mock."
    else:
        note = f"No {SERVICE_KEY_FILE} in the project root. Using the local JSON mock database."
    return LocalMockFirestore(base / MOCK_DB_FILE, logger), note
