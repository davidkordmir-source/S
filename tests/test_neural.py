"""Lifecycle/cache/math regression tests. Actual weights are checked in native CI."""

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import wait_job
from fastapi.testclient import TestClient
from neural_support import AssistantDouble, NeuralDouble
from offline_support import scene_bytes
from scipy.special import expit

from atlas.errors import UserError
from atlas.jobs import Job
from atlas.object_detector import ObjectDetector
from atlas.offline_assistant import OfflineAssistant
from atlas.offline_server import OfflineServices, create_app
from atlas.visual_features import unit
from atlas.workshop import Workshop


def test_model_head_keeps_absolute_scores_and_selects_real_reference():
    scene = dict(
        classes=np.eye(3, 512, dtype=np.float32),
        shift=np.array([-0.3, -0.2, -0.1]),
        scale=np.array([3.0, 5.0, 8.0]),
        objectness=np.array([-2.0, 0.0, 2.0]),
    )
    queries = np.eye(2, 512, dtype=np.float32)[::-1]
    result = ObjectDetector.classify(scene, images=queries)
    expected = expit((scene["classes"] @ queries.T + scene["shift"][:, None]) * scene["scale"][:, None])
    np.testing.assert_allclose(result["scores"], expected.max(axis=1))
    assert list(result["image_indexes"][:2]) == [1, 0]
    assert result["scores"].max() < 1
    assert result["scores"][2] < 0.5
    other = ObjectDetector.classify({k: v[:1] for k, v in scene.items()}, images=queries)
    assert result["scores"][0] == other["scores"][0]  # no tile max normalization


def test_cached_head_matches_transformers_learned_head():
    import torch
    from transformers import Owlv2Config
    from transformers.models.owlv2.modeling_owlv2 import Owlv2ClassPredictionHead

    torch.manual_seed(30)
    config = Owlv2Config()
    head = Owlv2ClassPredictionHead(config).eval()
    values = torch.randn(1, 11, config.vision_config.hidden_size)
    queries = torch.randn(1, 3, config.text_config.hidden_size)
    with torch.no_grad():
        logits, classes = head(values, queries, None)
        shift = head.logit_shift(values)
        scale = head.elu(head.logit_scale(values)) + 1
    cached = dict(
        classes=unit(classes[0].numpy()),
        shift=shift[0, :, 0].numpy(),
        scale=scale[0, :, 0].numpy(),
        objectness=np.zeros(11),
    )
    actual = ObjectDetector.classify(cached, images=queries[0].numpy())["scores"]
    np.testing.assert_allclose(actual, torch.sigmoid(logits[0]).max(dim=-1).values.numpy(), atol=2e-6)


def test_text_only_search_reuses_scene_on_new_instruction_and_threshold(tmp_path):
    engine = NeuralDouble()
    w = Workshop(tmp_path, engine)
    scene = w.import_scene(scene_bytes(), "scene.png", Job("import"))["scene"]["id"]
    options = dict(
        match_mode="instances", scene=scene, instruction="دلخواه بدون فهرست دسته‌ها", quality="fast"
    )
    result = w.similarity.search(Job("scan"), **options)
    assert result["detections"] and result["settings"]["match_mode"] == "instances"
    assert engine.objects.vision_calls == 1
    assert len(engine.assistant.last_pictures) == 1  # real scene context also in text-only mode
    w.similarity.search(Job("scan"), **options, threshold=0.8)
    assert engine.objects.vision_calls == 1 and engine.assistant.calls == 1
    next_result = w.similarity.search(Job("scan"), **(options | dict(instruction="یک درخواست متفاوت")))
    assert engine.objects.vision_calls == 1 and next_result["settings"]["neural_cache"]["hits"] == 1
    cache = next((w.similarity.feature_store.folder).glob("object-*.npz"))
    cache.write_bytes(b"corrupt")
    w.similarity.search(Job("scan"), **(options | dict(instruction="درخواست سوم")))
    assert engine.objects.vision_calls == 2


def test_profile_description_clear_archive_and_restore_preserve_samples(tmp_path):
    w = Workshop(tmp_path, NeuralDouble())
    lib = w.similarity.library
    ref = w.similarity.add_file(scene_bytes(), "sample.png")
    group = lib.create("هدف آزمایشی")["id"]
    lib.add(group, ref["id"])
    lib.describe(group, "نوشتهٔ اشتباه", "نامطلوب")
    assert lib.describe(group, "", "")["description"] == ""
    lib.archive(group)
    assert not lib.groups() and lib.active() is None
    with pytest.raises(UserError):
        lib.select(group)
    with pytest.raises(UserError):
        lib.add(group, ref["id"])
    lib.archive(group, False)
    lib.select(group)
    assert lib.groups()[0]["id"] == group and lib.group(group)["references"][0]["id"] == ref["id"]
    concept = w.new_concept("اشتباه")["id"]
    w.rename_concept(concept, "اصلاح شد")
    w.archive_concept(concept)
    assert w.concepts() == []
    w.archive_concept(concept, False)
    assert w.concepts()[0]["name"] == "اصلاح شد"


def test_freeform_assistant_and_target_crud_api(tmp_path):
    services = OfflineServices(tmp_path, NeuralDouble())
    app = create_app(services=services, token="test-session")
    with TestClient(app) as client:
        client.get("/?token=test-session")
        group = client.post("/api/workspace/reference-groups", json={"name": "هدف"}).json()["id"]
        reply = wait_job(
            client,
            client.post(
                "/api/workspace/assistant", json={"group": group, "message": "درخواست دلخواه"}
            ).json(),
        )
        assert reply["status"] == "done", reply
        assert reply["result"]["instruction"] == "درخواست دلخواه"
        rows = client.get("/api/workspace/assistant/messages", params={"group": group}).json()["messages"]
        assert len(rows) == 2 and rows[1]["model"] == "explicit wiring double"
        assert (
            client.patch(
                "/api/workspace/assistant/messages/" + rows[0]["id"], json={"message": "اصلاح شد"}
            ).status_code
            == 200
        )
        assert client.delete("/api/workspace/assistant/messages/" + rows[1]["id"]).status_code == 200
        assert client.delete("/api/workspace/assistant/messages", params={"group": group}).status_code == 200
        assert (
            client.get("/api/workspace/assistant/messages", params={"group": group}).json()["messages"] == []
        )
        assert client.delete("/api/workspace/reference-groups/" + group).status_code == 200
        assert client.get("/api/workspace/reference-groups").json()["groups"] == []
        assert client.post("/api/workspace/reference-groups/" + group + "/restore").status_code == 200


def test_prepared_plan_bound_to_actual_inputs_and_model_revision(tmp_path):
    engine = SimpleNamespace(compute=SimpleNamespace(cpu_threads=lambda: 1))
    assistant = OfflineAssistant(tmp_path, engine, tmp_path / "models")
    message = "هدف آزاد"
    key = "a" * 64
    plan = AssistantDouble().understand(Job("test"), message)
    plan = {k: v for k, v in plan.items() if k not in ("model", "cached", "prepared_key")}
    context = dict(message=message, images=[], revision=assistant.revision())
    (assistant.cache / f"{key}.json").write_text(json.dumps(dict(context=context, plan=plan)))
    assert assistant.understand(Job("test"), message, prepared=key)["cached"]
    with pytest.raises(UserError, match="فایل‌های مدل"):
        assistant.understand(Job("test"), "ورودی متفاوت", prepared=key)


def installer_module():
    spec = importlib.util.spec_from_file_location(
        "offline_installer", Path(__file__).resolve().parents[1] / "scripts/offline-installer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multipart_installer_seek_integrity_and_atomic_extraction(tmp_path):
    module = installer_module()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("AtlasScout/AtlasScout.exe", b"not executable test data" * 100)
    data = stream.getvalue()
    parts = []
    for i, start in enumerate(range(0, len(data), 97)):
        chunk = data[start : start + 97]
        name = f"part{i}"
        (tmp_path / name).write_bytes(chunk)
        parts.append(dict(name=name, size=len(chunk), sha256=hashlib.sha256(chunk).hexdigest()))
    with module.JoinedReader([tmp_path / p["name"] for p in parts]) as source:
        source.seek(-30, 2)
        assert source.read() == data[-30:]
        source.seek(95)
        assert source.read(108) == data[95:203]
    manifest = dict(parts=parts, archive_bytes=len(data), archive_sha256=hashlib.sha256(data).hexdigest())
    destination = tmp_path / "installed"
    assert module.install(manifest, tmp_path, destination).read_bytes() == b"not executable test data" * 100
    with pytest.raises(ValueError, match="NEW destination"):
        module.install(manifest, tmp_path, destination)
    (tmp_path / parts[-1]["name"]).write_bytes(b"x" * parts[-1]["size"])
    with pytest.raises(ValueError, match="Checksum"):
        module.install(manifest, tmp_path, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_old_library_schema_migrates_without_losing_targets_or_images(tmp_path):
    engine = NeuralDouble()
    workshop = Workshop(tmp_path, engine)
    library = workshop.similarity.library
    group = library.create("هدف قدیمی")["id"]
    reference = workshop.similarity.add_file(scene_bytes(), "old.png")
    library.add(group, reference["id"])
    concept = workshop.new_concept("آموزش قدیمی")["id"]
    with workshop.connect() as db:
        for column in ("description", "excluded_description", "archived"):
            db.execute(f"ALTER TABLE reference_groups DROP COLUMN {column}")
        db.execute("ALTER TABLE concepts DROP COLUMN archived")
    restored = Workshop(tmp_path, engine)
    assert restored.concepts()[0]["id"] == concept
    assert restored.similarity.library.group(group)["description"] == ""
    assert restored.similarity.library.group(group)["references"][0]["id"] == reference["id"]
    restored.similarity.library.archive(group)
    restored.similarity.library.archive(group, False)
    assert (restored.similarity.folder / f"{reference['id']}.png").is_file()


def test_image_only_neural_search_does_not_invent_or_generate_a_language_plan(tmp_path):
    engine = NeuralDouble()
    workshop = Workshop(tmp_path, engine)
    scene = workshop.import_scene(scene_bytes(), "scene.png", Job("import"))["scene"]["id"]
    ref = workshop.similarity.add_file(scene_bytes(), "reference.png")
    result = workshop.similarity.search(
        Job("search"), match_mode="instances", scene=scene, reference=ref["id"], quality="fast"
    )
    assert result["detections"] and result["settings"]["native_image_guidance"]
    assert engine.assistant.calls == 0
    assert "assistant" not in result and "assistant" not in result["settings"]
