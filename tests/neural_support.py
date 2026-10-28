"""Explicit deterministic doubles for neural API/UI wiring; never shipped."""

import numpy as np
from offline_support import OfflineTestEngine

from atlas.object_detector import ObjectDetector


class AssistantDouble:
    def __init__(self):
        self.calls = 0

    def status(self):
        return dict(available=True, loaded=False, model="explicit wiring double")

    def revision(self):
        return "test-assistant"

    def understand(self, job, message, pictures=(), **kwargs):
        self.calls += 1
        self.last_pictures = list(pictures)
        return dict(
            intent="search",
            answer_fa="پاسخ آزمایشی رابط؛ مدل واقعی نیست.",
            target_description=message,
            detector_phrases=[message or "target"],
            excluded_phrases=[],
            required_attributes=[],
            use_colour=False,
            reference_regions=[],
            model="explicit wiring double",
            cached=False,
            prepared_key="a" * 64,
        )

    def close(self):
        pass


class DetectorDouble(ObjectDetector):
    def __init__(self):
        self.vision_calls = 0

    def status(self):
        return dict(available=True, loaded=False, model="explicit wiring double")

    def revision(self):
        return "test-detector"

    def scene(self, picture, resolution=768, colour=False):
        self.vision_calls += 1
        return dict(
            classes=np.eye(3, 512, dtype=np.float32),
            boxes=np.array([[0.2, 0.2, 0.2, 0.2], [0.7, 0.2, 0.2, 0.2], [0.5, 0.6, 0.2, 0.2]], np.float32),
            shift=np.full(3, -0.2, np.float32),
            scale=np.ones(3, np.float32) * 3,
            objectness=np.ones(3, np.float32),
        )

    def text_queries(self, phrases):
        return np.eye(len(phrases), 512, dtype=np.float32)

    def image_query(self, picture, resolution=768, colour=False):
        return np.eye(1, 512, dtype=np.float32)


class NeuralDouble(OfflineTestEngine):
    def __init__(self):
        super().__init__()
        self.assistant, self.objects = AssistantDouble(), DetectorDouble()

    def status(self):
        return super().status() | dict(
            assistant=self.assistant.status(), object_detector=self.objects.status()
        )
