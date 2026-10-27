"""Bounded local feature retrieval; no network or whole-scene feature arrays."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageOps

PATCH = 14
BLOCK = 448


def shape_picture(picture):
    """Discard hue/saturation for every proposal and learned feature input.

    Local brightness contrast remains useful for geometry/texture; it cannot
    turn a colour histogram into evidence of an object match.
    """
    return ImageOps.autocontrast(picture.convert("L"), cutoff=0).convert("RGB")


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def model_picture(picture, maximum=224):
    scale = maximum / max(picture.size)
    size = tuple(max(PATCH, round(d * scale / PATCH) * PATCH) for d in picture.size)
    return picture.convert("RGB").resize(size, Image.Resampling.BICUBIC)


def descriptor(features):
    grid = features["grid"]
    spatial = []
    for y in range(3):
        for x in range(3):
            # Tiny elongated references may have fewer than three patch rows.
            y0, x0 = (
                min(grid.shape[0] - 1, y * grid.shape[0] // 3),
                min(grid.shape[1] - 1, x * grid.shape[1] // 3),
            )
            y1, x1 = max(y0 + 1, (y + 1) * grid.shape[0] // 3), max(x0 + 1, (x + 1) * grid.shape[1] // 3)
            spatial.append(grid[y0:y1, x0:x1].mean(axis=(0, 1)))
    return dict(cls=features["cls"], pooled=unit(grid.mean(axis=(0, 1))), spatial=unit(np.array(spatial)))


def compare(candidate, references):
    """Score rotation alternatives using object and 3×3 spatial descriptors."""
    scores = [
        0.3 * float(candidate["cls"] @ ref["cls"])
        + 0.1 * float(candidate["pooled"] @ ref["pooled"])
        + 0.6 * float((candidate["spatial"] * ref["spatial"]).sum(axis=-1).mean())
        for ref in references
    ]
    best = int(np.argmax(scores))
    return float(np.clip(scores[best], 0, 1)), best


def peaks(values, gate, radius):
    values = np.nan_to_num(values, nan=-1, posinf=-1, neginf=-1)
    if not values.size:
        return []
    kernel = max(3, round(radius)) | 1
    maxima = cv2.dilate(values, np.ones((kernel, kernel), np.uint8))
    mask = ((values >= gate) & (values >= maxima - 1e-7)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = []
    for label in range(1, count):
        x, y, w, h = stats[label, :4]
        region = values[y : y + h, x : x + w]
        py, px = np.unravel_index(
            np.argmax(np.where(labels[y : y + h, x : x + w] == label, region, -2)), region.shape
        )
        result.append((x + int(px), y + int(py), float(region[py, px])))
    return result


def positions(length, step):
    return sorted({*range(0, max(1, length - BLOCK + 1), step), max(0, length - BLOCK)})


def dense_proposals(picture, engine, references, width, height, scales, rotate, check, sizes=None):
    """DINO proposal path independent of RGB/template gates.

    Features are bounded to one 448×448 block. Overlap covers complete target
    windows; ownership and final scene suppression remove duplicate proposals.
    The target's long side is 112 px (8 patches) at the base scale.
    """
    ratio = 112 / max(width, height)
    full_width, full_height = (max(PATCH, round(n * ratio)) for n in picture.size)
    largest = math.ceil(112 * max(scales) * (1.415 if rotate else 1))
    step = max(PATCH, (BLOCK - largest) // PATCH * PATCH)
    reference_vectors = np.array([r["pooled"] for r in references], np.float32)
    reference_spatial = np.array([r["spatial"] for r in references], np.float32)
    geometries = set()
    for vw, vh in sizes or [(width, height)]:
        for scale in scales:
            tw, th = vw * ratio * scale, vh * ratio * scale
            for angle in (0, 45, 90) if rotate else (0,):
                if angle == 90:
                    tw1, th1 = th, tw
                elif angle == 45:
                    tw1 = th1 = (tw + th) / 2**0.5
                else:
                    tw1, th1 = tw, th
                geometries.add((max(1, round(tw1 / PATCH)), max(1, round(th1 / PATCH))))
    for top in positions(full_height, step):
        for left in positions(full_width, step):
            check()
            block_width, block_height = min(BLOCK, full_width - left), min(BLOCK, full_height - top)
            bw, bh = (max(PATCH, (d // PATCH) * PATCH) for d in (block_width, block_height))
            prepared = picture.transform(
                (bw, bh),
                Image.Transform.EXTENT,
                (left / ratio, top / ratio, (left + block_width) / ratio, (top + block_height) / ratio),
                Image.Resampling.BICUBIC,
            )
            grid = engine.details([prepared])[0]["grid"]
            if not np.any(grid):
                continue
            integral = np.pad(grid.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)))
            for tw, th in sorted(geometries):
                if tw > grid.shape[1] or th > grid.shape[0]:
                    continue
                pooled = unit(
                    integral[th:, tw:] - integral[:-th, tw:] - integral[th:, :-tw] + integral[:-th, :-tw]
                )
                # Avoid launching a large BLAS thread team for these small
                # matrices; CPU inference already has its own bounded threads.
                scores = 0.35 * np.einsum("hwc,rc->hwr", pooled, reference_vectors, optimize=False)
                oh, ow = pooled.shape[:2]
                for cy in range(3):
                    for cx in range(3):
                        y0, x0 = min(th - 1, cy * th // 3), min(tw - 1, cx * tw // 3)
                        y1, x1 = max(y0 + 1, (cy + 1) * th // 3), max(x0 + 1, (cx + 1) * tw // 3)
                        cell = unit(
                            integral[y1 : y1 + oh, x1 : x1 + ow]
                            - integral[y0 : y0 + oh, x1 : x1 + ow]
                            - integral[y1 : y1 + oh, x0 : x0 + ow]
                            + integral[y0 : y0 + oh, x0 : x0 + ow]
                        )
                        scores += (0.65 / 9) * np.einsum(
                            "hwc,rc->hwr", cell, reference_spatial[:, cy * 3 + cx], optimize=False
                        )
                values = scores.max(axis=-1)
                for x, y, score in peaks(values, 0.65, min(tw, th) / 2):
                    sx, sy = block_width / bw / ratio, block_height / bh / ratio
                    x0, y0 = left / ratio + x * PATCH * sx, top / ratio + y * PATCH * sy
                    yield (
                        (round(x0), round(y0), round(x0 + tw * PATCH * sx), round(y0 + th * PATCH * sy)),
                        score,
                    )


def structural(picture):
    """Exposure/color tolerant gradient magnitude, retaining spatial geometry."""
    gray = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.6)
    dx, dy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return cv2.magnitude(dx, dy)


def appearance_scores(picture, sample, angles):
    image = np.asarray(picture.convert("RGB"))
    edges = structural(image)
    best = (0.0, 0.0, 0)
    for angle in angles:
        rotated = sample.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True).resize(
            picture.size, Image.Resampling.BICUBIC
        )
        template = np.asarray(rotated)
        constant = template.astype(np.float32).std(axis=(0, 1)).max() < 3
        rgb = float(
            cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED if constant else cv2.TM_CCOEFF_NORMED)[
                0, 0
            ]
        )
        if constant:
            rgb = 1 - rgb
        template_edges = structural(template)
        edge = (
            float(cv2.matchTemplate(edges, template_edges, cv2.TM_CCOEFF_NORMED)[0, 0])
            if template_edges.std() > 1e-3
            else 0.0
        )
        if not np.isfinite(rgb):
            rgb = 0.0
        if not np.isfinite(edge):
            edge = 0.0
        if max(rgb, edge) > max(best[:2]):
            best = (max(0.0, rgb), max(0.0, edge), angle)
    return best
