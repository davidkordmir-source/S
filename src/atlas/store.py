from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np

from .config import INDEX_VERSION
from .errors import UserError


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.images = root / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.database = root / "atlas.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS buildings (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, lat REAL NOT NULL,
                lon REAL NOT NULL, source TEXT NOT NULL, image TEXT NOT NULL,
                geometry TEXT, vector BLOB, dimension INTEGER, index_version TEXT
            )""")

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def public(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "title": row["title"],
            "lat": row["lat"],
            "lon": row["lon"],
            "source": row["source"],
            "image_url": f"/api/images/{row['id']}",
            "geometry": json.loads(row["geometry"]) if row["geometry"] else None,
        }

    def put(
        self,
        *,
        id: str,
        title: str,
        lat: float,
        lon: float,
        source: str,
        image: Path,
        geometry: dict | None = None,
    ) -> None:
        # Replacing imagery always invalidates embeddings; no stale vectors.
        with self.connect() as db:
            db.execute(
                """INSERT INTO buildings(id,title,lat,lon,source,image,geometry)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,lat=excluded.lat,lon=excluded.lon,
                source=excluded.source,image=excluded.image,geometry=excluded.geometry,
                vector=NULL,dimension=NULL,index_version=NULL""",
                (id, title, lat, lon, source, str(image), json.dumps(geometry) if geometry else None),
            )

    def counts(self) -> dict:
        with self.connect() as db:
            total = db.execute("SELECT count(*) FROM buildings").fetchone()[0]
            indexed = db.execute(
                "SELECT count(*) FROM buildings WHERE index_version=? AND vector IS NOT NULL",
                (INDEX_VERSION,),
            ).fetchone()[0]
        return {"total": total, "indexed": indexed}

    def list(self, limit: int = 200) -> list[dict]:
        with self.connect() as db:
            return [
                self.public(row)
                for row in db.execute("SELECT * FROM buildings ORDER BY rowid DESC LIMIT ?", (limit,))
            ]

    def pending(self) -> list[dict]:
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM buildings WHERE vector IS NULL OR index_version!=?", (INDEX_VERSION,)
                )
            ]

    def image_path(self, id: str) -> Path | None:
        with self.connect() as db:
            row = db.execute("SELECT image FROM buildings WHERE id=?", (id,)).fetchone()
        if not row:
            return None
        path = Path(row[0]).resolve()
        return path if path.is_relative_to(self.images.resolve()) and path.is_file() else None

    def embed(self, id: str, vector: np.ndarray) -> None:
        vector = normalize(vector)
        with self.connect() as db:
            db.execute(
                "UPDATE buildings SET vector=?,dimension=?,index_version=? WHERE id=?",
                (vector.astype("<f4").tobytes(), len(vector), INDEX_VERSION, id),
            )

    def search(self, vector: np.ndarray, limit: int = 12, area: dict | None = None) -> list[dict]:
        query = normalize(vector)
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM buildings WHERE index_version=? AND dimension=? AND vector IS NOT NULL",
                (INDEX_VERSION, len(query)),
            ).fetchall()
        if area:
            rows = [row for row in rows if within_radius(row["lat"], row["lon"], area)]
        if not rows:
            raise UserError("هنوز تصویری برای جست‌وجو نمایه نشده است. ابتدا کتابخانه را آماده کنید.")
        matrix = np.stack([np.frombuffer(row["vector"], dtype="<f4") for row in rows])
        scores = matrix @ query
        order = np.argsort(-scores, kind="stable")[:limit]
        return [{**self.public(rows[i]), "score": round(float(scores[i]), 5)} for i in order]


def within_radius(lat: float, lon: float, area: dict) -> bool:
    first, second = math.radians(lat), math.radians(area["lat"])
    delta_lat = first - second
    delta_lon = math.radians(lon - area["lon"])
    distance = (
        2
        * 6_371_000
        * math.asin(
            min(
                1,
                math.sqrt(
                    math.sin(delta_lat / 2) ** 2
                    + math.cos(first) * math.cos(second) * math.sin(delta_lon / 2) ** 2
                ),
            )
        )
    )
    return distance <= area["radius"]


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or len(vector) != 512 or not np.isfinite(vector).all():
        raise ValueError("Expected a finite 512-dimensional CLIP embedding")
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError("Cannot normalize an empty embedding")
    return vector / norm
