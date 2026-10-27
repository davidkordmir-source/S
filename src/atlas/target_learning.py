"""User feedback, hard negatives, local aerial-feature adaptation and honest evaluation."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from .config import AERIAL_REVISION, DETAIL_REVISION, INDEX_VERSION
from .errors import UserError
from .workshop import box_tuple, intersection_over_union

LEARNING_VERSION = "aerial-linear-v1:" + DETAIL_REVISION + ":" + INDEX_VERSION + ":" + AERIAL_REVISION


class TargetLearning:
    def __init__(self, similarity):
        self.similarity = similarity
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS target_models(target TEXT PRIMARY KEY,revision TEXT,model TEXT,report TEXT,created REAL);
                CREATE TABLE IF NOT EXISTS reference_feedback(run TEXT,idx INTEGER,target TEXT,reference TEXT,label INTEGER,scene TEXT,bbox TEXT,created REAL,PRIMARY KEY(run,idx));
            """)

    def connect(self):
        return self.similarity.workshop.connect()

    def revision(self, target):
        group = self.similarity.library.group(target)
        values = [(r["id"], r["enabled"], r["label"], r["partition"]) for r in group["references"]]
        return hashlib.sha256(json.dumps([LEARNING_VERSION, values], sort_keys=True).encode()).hexdigest()

    def status(self, target):
        group = self.similarity.library.group(target)
        revision = self.revision(target)
        train = [r for r in group["references"] if r["enabled"] and r["partition"] == "train"]
        with self.connect() as db:
            row = db.execute("SELECT * FROM target_models WHERE target=?", (target,)).fetchone()
        return dict(
            revision=revision,
            positive=sum(r["label"] == 1 for r in train),
            negative=sum(r["label"] == 0 for r in train),
            test=sum(r["enabled"] and r["partition"] == "test" for r in group["references"]),
            ready=bool(row and row["revision"] == revision),
            model=json.loads(row["model"]) if row and row["revision"] == revision else None,
            report=json.loads(row["report"]) if row else None,
            stale=bool(row and row["revision"] != revision),
        )

    @staticmethod
    def vectors(features):
        # Same normalized feature spaces in training and prediction, no image-colour channels.
        return np.concatenate([features["pooled"], features["aerial"]], axis=-1).astype(np.float32) / 2**0.5

    def train(self, job, target):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        from sklearn.model_selection import StratifiedGroupKFold

        from .reference_matcher import ReferenceMatcher

        state = self.status(target)
        if not state["positive"] or not state["negative"]:
            raise UserError(
                "برای آموزش اختصاصی دست‌کم یک نمونهٔ هدف و یک نمونهٔ غیرهدف لازم است؛ جست‌وجوی معمولی همچنان با یک عکس کار می‌کند."
            )
        members = [r for r in self.similarity.library.group(target)["references"] if r["enabled"]]
        training = [r for r in members if r["partition"] == "train"]
        held_out = [r for r in members if r["partition"] == "test"]
        excluded = set()
        for test_reference in held_out:
            for train_reference in training:
                same_source = (test_reference.get("source_reference") or test_reference["id"]) == (
                    train_reference.get("source_reference") or train_reference["id"]
                )
                overlapping = False
                if (
                    test_reference.get("scene")
                    and test_reference.get("scene") == train_reference.get("scene")
                    and test_reference.get("bbox")
                    and train_reference.get("bbox")
                ):
                    info = self.similarity.workshop.scene(test_reference["scene"])
                    overlapping = (
                        intersection_over_union(
                            box_tuple(test_reference["bbox"], info["width"], info["height"]),
                            box_tuple(train_reference["bbox"], info["width"], info["height"]),
                        )
                        > 0.1
                    )
                if same_source or overlapping:
                    excluded.add(test_reference["id"])
        members = [r for r in members if r["id"] not in excluded]
        helper = ReferenceMatcher(
            self.similarity,
            dict(
                references=[],
                negative_references=[dict(id=r["id"]) for r in members if r["label"] == 0],
                rotate=True,
                broad=False,
                intelligent=False,
                group=target,
            ),
            job,
        )
        data = []
        for i, member in enumerate(members):
            job.update(i / len(members) * 65, f"آموزش اختصاصی تصاویر هوایی: نمونهٔ {i + 1} از {len(members)}")
            features = helper.features(member["id"])
            data.append((member, self.vectors(features)))
        train = [d for d in data if d[0]["partition"] == "train"]
        test = [d for d in data if d[0]["partition"] == "test"]

        def fit(items):
            x = np.concatenate([v for _, v in items])
            y = np.concatenate([np.full(len(v), r["label"]) for r, v in items])
            return LogisticRegression(
                C=4, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=43
            ).fit(x, y)

        model = fit(train)
        job.check()
        report = dict(
            training_references=len(train),
            positive=state["positive"],
            negative=state["negative"],
            evaluation_references=len(test),
            excluded_overlapping_test_references=len(excluded),
            evaluation_kind="independent" if test else "unavailable",
            metrics=None,
            calibrated_probability=False,
            evaluation_scope="User-provided examples only; not an independent satellite benchmark",
            evaluation_note="برای ارزیابی به نمونه‌های جداگانهٔ هدف و غیرهدف نیاز است.",
        )
        labels = []
        predictions = []
        groups = [r.get("scene") or r.get("source_reference") or r["id"] for r, _ in train]
        group_counts = [
            len({groups[i] for i, (r, _) in enumerate(train) if r["label"] == label}) for label in (0, 1)
        ]
        if test:
            for reference, vectors in test:
                labels.append(reference["label"])
                predictions.append(float(model.predict_proba(vectors)[:, 1].mean()))
        elif not held_out and min(group_counts) >= 3:
            # All rotations of a reference stay in one fold; never test a sibling augmentation.
            ys = np.array([r["label"] for r, _ in train])
            folds = StratifiedGroupKFold(n_splits=min(5, *group_counts), shuffle=True, random_state=43)
            probabilities = np.zeros(len(train))
            for a, b in folds.split(np.zeros(len(train)), ys, groups):
                job.check()
                if len(set(ys[a])) < 2:
                    probabilities = None
                    break
                fold = fit([train[i] for i in a])
                for i in b:
                    probabilities[i] = fold.predict_proba(train[i][1])[:, 1].mean()
            if probabilities is not None:
                labels = ys.tolist()
                predictions = probabilities.tolist()
                report.update(evaluation_references=len(train), evaluation_kind="grouped_cross_validation")
        if len(set(labels)) == 2:
            hard = np.asarray(predictions) >= 0.5
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, hard, average="binary", zero_division=0
            )
            tn, fp, fn, tp = confusion_matrix(labels, hard, labels=[0, 1]).ravel()
            report["metrics"] = dict(
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                true_positive=int(tp),
                false_positive=int(fp),
                false_negative=int(fn),
                true_negative=int(tn),
            )
            report["evaluation_note"] = (
                "ارزیابی فقط روی نمونه‌های همین هدف انجام شده است؛ معیار دقت عمومی نرم‌افزار نیست."
            )
        if excluded:
            report["evaluation_note"] += (
                f" {len(excluded)} نمونهٔ آزمون به‌دلیل هم‌پوشانی با آموزش از ارزیابی کنار گذاشته شد."
            )
        weights = dict(
            version=LEARNING_VERSION, weights=model.coef_[0].tolist(), bias=float(model.intercept_[0])
        )
        job.check()
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO target_models VALUES(?,?,?,?,?)",
                (target, state["revision"], json.dumps(weights), json.dumps(report), time.time()),
            )
        return dict(
            teaching=target,
            message="آموزش اختصاصی آماده شد؛ نتیجهٔ ارزیابی در بخش آموزش نمایش داده می‌شود.",
            report=report,
        )

    @staticmethod
    def predict(model, pooled, aerial):
        vector = np.concatenate([pooled, aerial]) / 2**0.5
        weights = np.asarray(model["weights"], np.float32)
        if weights.shape != (896,) or not np.isfinite(weights).all():
            raise UserError("فایل آموزش اختصاصی معتبر نیست؛ دوباره آموزش بده.")
        value = float(vector @ weights + model["bias"])
        return float(1 / (1 + np.exp(-np.clip(value, -30, 30))))

    def feedback(self, job, run, index, label):
        result = self.similarity.workshop.run(run)
        if result["settings"].get("method") != "similar" or not result["settings"].get("group"):
            raise UserError("این نتیجه به یک هدف تصویری وصل نیست.")
        target = result["settings"]["group"]
        self.similarity.library.group(target)
        detection = self.similarity.one(run, index)
        scene = result["scene"]
        info = self.similarity.workshop.scene(scene)
        box = box_tuple(detection["bbox"], info["width"], info["height"])
        with self.connect() as db:
            previous = db.execute(
                "SELECT * FROM reference_feedback WHERE run=? AND idx=?", (run, index)
            ).fetchone()
        if previous:
            reference = previous["reference"]
            self.similarity.library.add(target, reference, label)
        else:
            with self.similarity.workshop.reader(scene) as reader:
                picture = reader.read(box, 1024, exact=True)
            ref = self.similarity.add_reference(
                picture,
                ("تأیید" if label else "غیرهدف") + f" · نتیجه {index + 1}",
                scene,
                detection["bbox"],
                (box[2] - box[0], box[3] - box[1]),
            )
            reference = self.similarity.library.add(target, ref["id"], label)["id"]
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO reference_feedback VALUES(?,?,?,?,?,?,?,?)",
                (run, index, target, reference, label, scene, json.dumps(detection["bbox"]), time.time()),
            )
        state = self.status(target)
        if state["positive"] and state["negative"]:
            return self.train(job, target) | dict(
                feedback_run=run,
                feedback_index=index,
                message="بازخورد ذخیره و آموزش اختصاصی به‌روز شد.",
                feedback_label=label,
            )
        return dict(
            teaching=target,
            feedback_run=run,
            feedback_index=index,
            feedback_label=label,
            message="بازخورد ذخیره شد و در جست‌وجوی بعدی استفاده می‌شود.",
        )

    def exclusions(self, target, scene):
        with self.connect() as db:
            rows = db.execute(
                'SELECT f.bbox FROM reference_feedback f JOIN reference_members m ON m.target=f.target AND m.reference=f.reference WHERE f.target=? AND f.scene=? AND m.label=0 AND m.enabled=1 AND m.partition="train"',
                (target, scene),
            ).fetchall()
        return [json.loads(r["bbox"]) for r in rows]
