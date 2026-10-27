"""Local geometric verification and continuous pose proposals, independent of colour."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

from .visual_features import shape_picture


def local_features(picture, mask=None):
    image = np.asarray(shape_picture(picture).convert("L"))
    points, descriptors = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.025).detectAndCompute(
        image, mask
    )
    return np.array([p.pt for p in points], np.float32).reshape(-1, 2), descriptors


def matches(reference, target):
    if reference is None or target is None or min(len(reference), len(target)) < 4:
        return []
    forward = cv2.BFMatcher(cv2.NORM_L2).knnMatch(reference, target, k=2)
    reverse = cv2.BFMatcher(cv2.NORM_L2).match(target, reference)
    return [
        a
        for pair in forward
        if len(pair) == 2
        for a, b in [pair]
        if a.distance < 0.75 * b.distance and reverse[a.trainIdx].trainIdx == a.queryIdx
    ]


def fit_pose(reference_points, target_points, pairs, size):
    if len(pairs) < 6:
        return None
    source = np.array([reference_points[p.queryIdx] for p in pairs], np.float32)
    target = np.array([target_points[p.trainIdx] for p in pairs], np.float32)
    matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3, maxIters=2000, confidence=0.995)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return None
    valid = inliers.ravel().astype(bool)
    count = int(valid.sum())
    fraction = count / len(pairs)
    if count < 6 or fraction < 0.35:
        return None
    coverage = cv2.contourArea(cv2.convexHull(source[valid])) / max(1, size[0] * size[1])
    if coverage < 0.08:
        return None
    corners = np.array([[[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]], np.float32)
    quad = cv2.perspectiveTransform(corners, matrix)[0]
    if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
        return None
    area = abs(cv2.contourArea(quad))
    scale = (area / max(1, size[0] * size[1])) ** 0.5
    if not 0.2 <= scale <= 5:
        return None
    sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if min(sides) < 4 or max(sides) / max(1, min(sides)) > 15:
        return None
    angle = math.degrees(math.atan2(float(quad[1, 1] - quad[0, 1]), float(quad[1, 0] - quad[0, 0]))) % 360
    return dict(
        matrix=matrix,
        quad=quad,
        inliers=count,
        fraction=fraction,
        coverage=coverage,
        scale=scale,
        rotation=angle,
        score=float(np.clip(0.6 * fraction + 0.4 * min(1, coverage * 3), 0, 1)),
        valid=valid,
    )


def verify(reference, candidate, reference_mask=None):
    rp, rd = local_features(reference, reference_mask)
    tp, td = local_features(candidate)
    return verify_features(rp, rd, reference.size, tp, td, candidate.size)


def verify_features(rp, rd, reference_size, tp, td, candidate_size):
    pairs = matches(rd, td)
    pose = fit_pose(rp, tp, pairs, reference_size)
    if pose is not None:
        quad = pose["quad"]
        w, h = candidate_size
        if (
            quad[:, 0].min() < -w * 0.3
            or quad[:, 1].min() < -h * 0.3
            or quad[:, 0].max() > w * 1.3
            or quad[:, 1].max() > h * 1.3
        ):
            pose = None
    if pose is None:
        return dict(
            status="insufficient" if min(len(rp), len(tp)) < 6 else "unverified",
            score=None,
            inliers=0,
            rotation=None,
            scale=None,
            quad=None,
        )
    return {k: pose[k] for k in ("score", "inliers", "rotation", "scale", "quad")} | dict(status="verified")


def pose_proposals(reference, reference_mask, tile, check, maximum_scale=5):
    rp, rd = local_features(reference, reference_mask)
    tp, td = local_features(tile)
    remaining = matches(rd, td)
    # One set of reference descriptors can identify multiple disjoint instances.
    while len(remaining) >= 6:
        check()
        pose = fit_pose(rp, tp, remaining, reference.size)
        if pose is None:
            break
        if pose["scale"] <= maximum_scale:
            yield pose
        inlier_indices = {p.trainIdx for p, keep in zip(remaining, pose["valid"], strict=True) if keep}
        remaining = [p for p in remaining if p.trainIdx not in inlier_indices]


def rectify(reference, candidate, quad):
    destination = np.array(
        [[0, 0], [reference.width, 0], [reference.width, reference.height], [0, reference.height]], np.float32
    )
    matrix = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), destination)
    return Image.fromarray(
        cv2.warpPerspective(
            np.asarray(candidate.convert("RGB")), matrix, reference.size, borderMode=cv2.BORDER_REFLECT
        )
    )
