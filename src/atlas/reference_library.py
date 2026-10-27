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
            if not db.execute("SELECT 1 FROM similarity_meta WHERE key='groups-v1'").fetchone():
                # Existing unrelated screenshots must never become one training target.
                for row in db.execute("SELECT id,name,created FROM image_references").fetchall():
                    target = "legacy-" + row["id"]
                    db.execute(
                        "INSERT INTO reference_groups VALUES(?,?,?)", (target, row["name"], row["created"])
                    )
                    db.execute("INSERT INTO reference_members VALUES(?,?,1,0,NULL)", (target, row["id"]))
                db.execute("INSERT INTO similarity_meta VALUES('groups-v1','1')")

    def connect(self):
        return self.similarity.workshop.connect()

    def groups(self):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT g.*,COUNT(m.reference) AS total,COALESCE(SUM(m.enabled),0) AS enabled FROM reference_groups g LEFT JOIN reference_members m ON m.target=g.id GROUP BY g.id ORDER BY g.updated DESC,g.id"
                )
            ]

    def active(self):
        with self.connect() as db:
            row = db.execute("SELECT value FROM similarity_meta WHERE key='active-group'").fetchone()
        return row["value"] if row else None

    def select(self, id):
        self.group(id)
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
                "SELECT r.*,m.enabled FROM reference_members m JOIN image_references r ON r.id=m.reference WHERE m.target=? ORDER BY m.position,r.id",
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
            db.execute("INSERT INTO reference_groups VALUES(?,?,?)", (id, name, time.time()))
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

    def add(self, id, reference):
        self.group(id)
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
                "INSERT INTO reference_members VALUES(?,?,1,?,?) ON CONFLICT(target,reference) DO UPDATE SET enabled=1",
                (id, reference, position, digest),
            )
            db.execute("UPDATE reference_groups SET updated=? WHERE id=?", (time.time(), id))
        return {**self.similarity.reference(reference), "group": id}

    def change(self, id, reference, enabled=None, remove=False):
        self.group(id)
        with self.connect() as db:
            if remove:
                changed = db.execute(
                    "DELETE FROM reference_members WHERE target=? AND reference=?", (id, reference)
                ).rowcount
            else:
                changed = db.execute(
                    "UPDATE reference_members SET enabled=? WHERE target=? AND reference=?",
                    (int(enabled), id, reference),
                ).rowcount
            if not changed:
                raise UserError("نمونه در این هدف پیدا نشد.")
            db.execute("UPDATE reference_groups SET updated=? WHERE id=?", (time.time(), id))
        return self.group(id)

    def attach_upload(self, reference, group=None):
        group = group or self.create(reference["name"])["id"]
        return self.add(group, reference["id"])
