"""Persistent, optional multi-view teaching profiles over immutable references."""

from __future__ import annotations

import hashlib
import time
import uuid

from .errors import UserError


class ReferenceLibrary:
    def __init__(self, similarity):
        self.similarity = similarity
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS reference_groups(id TEXT PRIMARY KEY,name TEXT,updated REAL);
                CREATE TABLE IF NOT EXISTS reference_members(target TEXT REFERENCES reference_groups(id) ON DELETE CASCADE,reference TEXT REFERENCES image_references(id),enabled INTEGER NOT NULL DEFAULT 1,position INTEGER,digest TEXT,PRIMARY KEY(target,reference));
                CREATE TABLE IF NOT EXISTS similarity_meta(key TEXT PRIMARY KEY,value TEXT);
            """)
            group_columns = {r[1] for r in db.execute("PRAGMA table_info(reference_groups)")}
            for column, kind in [
                ("description", "TEXT NOT NULL DEFAULT ''"),
                ("excluded_description", "TEXT NOT NULL DEFAULT ''"),
                ("archived", "INTEGER NOT NULL DEFAULT 0"),
            ]:
                if column not in group_columns:
                    db.execute(f"ALTER TABLE reference_groups ADD COLUMN {column} {kind}")
            columns = {r[1] for r in db.execute("PRAGMA table_info(reference_members)")}
            for column, kind in [
                ("label", "INTEGER NOT NULL DEFAULT 1"),
                ("partition", 'TEXT NOT NULL DEFAULT "train"'),
            ]:
                if column not in columns:
                    db.execute(f"ALTER TABLE reference_members ADD COLUMN {column} {kind}")
            if not db.execute("SELECT 1 FROM similarity_meta WHERE key='groups-v1'").fetchone():
                # Existing unrelated screenshots must never become one training target.
                for row in db.execute("SELECT id,name,created FROM image_references").fetchall():
                    target = "legacy-" + row["id"]
                    db.execute(
                        "INSERT INTO reference_groups(id,name,updated) VALUES(?,?,?)",
                        (target, row["name"], row["created"]),
                    )
                    db.execute(
                        "INSERT INTO reference_members(target,reference,enabled,position,digest) VALUES(?,?,1,0,NULL)",
                        (target, row["id"]),
                    )
                db.execute("INSERT INTO similarity_meta VALUES('groups-v1','1')")

    def connect(self):
        return self.similarity.workshop.connect()

    def groups(self):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT g.*,COUNT(m.reference) AS total,COALESCE(SUM(CASE WHEN m.label=1 AND m.partition='train' THEN m.enabled ELSE 0 END),0) AS enabled FROM reference_groups g LEFT JOIN reference_members m ON m.target=g.id WHERE g.archived=0 GROUP BY g.id ORDER BY g.updated DESC,g.id"
                )
            ]

    def active(self):
        with self.connect() as db:
            row = db.execute("SELECT value FROM similarity_meta WHERE key='active-group'").fetchone()
        if not row:
            return None
        with self.connect() as db:
            active = db.execute(
                "SELECT id FROM reference_groups WHERE id=? AND archived=0", (row["value"],)
            ).fetchone()
        return active["id"] if active else None

    def select(self, id):
        if self.group(id)["archived"]:
            raise UserError("این هدف حذف شده است؛ ابتدا آن را بازگردان.")
        with self.connect() as db:
            db.execute(
                "INSERT INTO similarity_meta VALUES('active-group',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (id,),
            )
        return {"active": id}

    def group(self, id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM reference_groups WHERE id=?", (id,)).fetchone()
            if not row:
                raise UserError("هدف تصویری پیدا نشد.")
            members = db.execute(
                "SELECT r.*,m.enabled,m.label,m.partition FROM reference_members m JOIN image_references r ON r.id=m.reference WHERE m.target=? ORDER BY m.position,r.id",
                (id,),
            ).fetchall()
        return {
            **dict(row),
            "references": [
                {**self.similarity._reference(dict(r)), "enabled": bool(r["enabled"])} for r in members
            ],
        }

    def create(self, name):
        name = name.strip()[:100]
        if not name:
            raise UserError("نام هدف را بنویس.")
        id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO reference_groups(id,name,updated) VALUES(?,?,?)", (id, name, time.time()))
        self.select(id)
        return self.group(id)

    def rename(self, id, name):
        self.group(id)
        name = name.strip()[:100]
        if not name:
            raise UserError("نام هدف را بنویس.")
        with self.connect() as db:
            db.execute("UPDATE reference_groups SET name=?,updated=? WHERE id=?", (name, time.time(), id))
        return self.group(id)

    def describe(self, id, description=None, excluded_description=None):
        self.group(id)
        for value in (description, excluded_description):
            if value is not None and len(value) > 8000:
                raise UserError("توضیح هدف بیش از ظرفیت ورودی است؛ آن را کوتاه‌تر کن.")
        with self.connect() as db:
            if description is not None:
                db.execute(
                    "UPDATE reference_groups SET description=?,updated=? WHERE id=?",
                    (description.strip(), time.time(), id),
                )
            if excluded_description is not None:
                db.execute(
                    "UPDATE reference_groups SET excluded_description=?,updated=? WHERE id=?",
                    (excluded_description.strip(), time.time(), id),
                )
        return self.group(id)

    def archive(self, id, archived=True):
        self.group(id)
        with self.connect() as db:
            db.execute(
                "UPDATE reference_groups SET archived=?,updated=? WHERE id=?",
                (int(archived), time.time(), id),
            )
            if archived:
                db.execute("DELETE FROM similarity_meta WHERE key='active-group' AND value=?", (id,))
        return dict(id=id, deleted=archived, restored=not archived)

    def add(self, id, reference, label=1, partition="train"):
        if label not in (0, 1) or partition not in ("train", "test"):
            raise UserError("نوع نمونه معتبر نیست.")
        if self.group(id)["archived"]:
            raise UserError("این هدف حذف شده است؛ یک هدف فعال را انتخاب کن.")
        self.similarity.reference(reference)
        digest = hashlib.sha256((self.similarity.folder / f"{reference}.png").read_bytes()).hexdigest()
        with self.connect() as db:
            for row in db.execute(
                "SELECT reference FROM reference_members WHERE target=? AND digest IS NULL", (id,)
            ).fetchall():
                path = self.similarity.folder / f"{row['reference']}.png"
                if path.is_file():
                    db.execute(
                        "UPDATE reference_members SET digest=? WHERE target=? AND reference=?",
                        (hashlib.sha256(path.read_bytes()).hexdigest(), id, row["reference"]),
                    )
            duplicate = db.execute(
                "SELECT reference FROM reference_members WHERE target=? AND digest=?", (id, digest)
            ).fetchone()
            if duplicate:
                reference = duplicate["reference"]
            position = db.execute(
                "SELECT COALESCE(MAX(position),-1)+1 FROM reference_members WHERE target=?", (id,)
            ).fetchone()[0]
            db.execute(
                "INSERT INTO reference_members(target,reference,enabled,position,digest,label,partition) VALUES(?,?,1,?,?,?,?) ON CONFLICT(target,reference) DO UPDATE SET enabled=1,label=excluded.label,partition=excluded.partition",
                (id, reference, position, digest, label, partition),
            )
            db.execute("UPDATE reference_groups SET updated=? WHERE id=?", (time.time(), id))
        return {**self.similarity.reference(reference), "group": id}

    def change(self, id, reference, enabled=None, remove=False, label=None, partition=None):
        self.group(id)
        with self.connect() as db:
            if remove:
                changed = db.execute(
                    "DELETE FROM reference_members WHERE target=? AND reference=?", (id, reference)
                ).rowcount
            else:
                fields = []
                values = []
                if enabled is not None:
                    fields.append("enabled=?")
                    values.append(int(enabled))
                if label is not None:
                    if label not in (0, 1):
                        raise UserError("برچسب نمونه معتبر نیست.")
                    fields.append("label=?")
                    values.append(label)
                if partition is not None:
                    if partition not in ("train", "test"):
                        raise UserError("کاربرد نمونه معتبر نیست.")
                    fields.append("partition=?")
                    values.append(partition)
                if not fields:
                    raise UserError("تغییری برای نمونه مشخص نشده است.")
                changed = db.execute(
                    "UPDATE reference_members SET " + ",".join(fields) + " WHERE target=? AND reference=?",
                    (*values, id, reference),
                ).rowcount
            if not changed:
                raise UserError("نمونه در این هدف پیدا نشد.")
            db.execute("UPDATE reference_groups SET updated=? WHERE id=?", (time.time(), id))
        return self.group(id)

    def attach_upload(self, reference, group=None, label=1, partition="train"):
        group = group or self.create(reference["name"])["id"]
        return self.add(group, reference["id"], label, partition)

    def replace(self, id, old, new):
        self.group(id)
        self.similarity.reference(new)
        digest = hashlib.sha256((self.similarity.folder / f"{new}.png").read_bytes()).hexdigest()
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM reference_members WHERE target=? AND reference=?", (id, old)
            ).fetchone()
            if not row:
                raise UserError("نمونه در هدف پیدا نشد.")
            db.execute("DELETE FROM reference_members WHERE target=? AND reference=?", (id, old))
            db.execute("DELETE FROM reference_members WHERE target=? AND reference=?", (id, new))
            db.execute(
                "INSERT INTO reference_members(target,reference,enabled,position,digest,label,partition) VALUES(?,?,?,?,?,?,?)",
                (id, new, row["enabled"], row["position"], digest, row["label"], row["partition"]),
            )
            db.execute(
                "UPDATE reference_feedback SET reference=? WHERE target=? AND reference=?", (new, id, old)
            )
            db.execute("UPDATE reference_groups SET updated=? WHERE id=?", (time.time(), id))
