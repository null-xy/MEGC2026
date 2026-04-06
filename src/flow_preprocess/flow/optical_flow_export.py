"""Canonical optical-flow export reused from the reference project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from flow_preprocess.utils import ensure_directory


FLOW_CHANNEL_DESCRIPTIONS = {
    "tv_l1": {
        "0": "horizontal flow u",
        "1": "vertical flow v",
        "2": "optical strain",
    },
    "farneback": {
        "0": "horizontal flow u",
        "1": "vertical flow v",
        "2": "flow magnitude",
    },
}


def compute_optical_strain(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute the optical strain channel reused from the reference project."""
    u_x = u - pd.DataFrame(u).shift(-1, axis="columns")
    v_y = v - pd.DataFrame(v).shift(-1, axis="index")
    u_y = u - pd.DataFrame(u).shift(-1, axis="index")
    v_x = v - pd.DataFrame(v).shift(-1, axis="columns")
    return np.array(
        np.sqrt(u_x**2 + v_y**2 + 0.5 * (u_y + v_x) ** 2).ffill(axis="columns").ffill(axis="index")
    )


def _prepare_image(image: np.ndarray, image_size: int | None = None) -> np.ndarray:
    prepared = image
    if image_size is not None:
        prepared = cv2.resize(prepared, (int(image_size), int(image_size)))
    return prepared


def _prepare_grayscale(image: np.ndarray, image_size: int | None = None) -> np.ndarray:
    prepared = _prepare_image(image, image_size=image_size)
    if prepared.ndim == 3:
        return cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
    return prepared


def describe_flow_tensor_channels(method: str) -> dict[str, str]:
    """Return the exported tensor channel meanings for one flow backend."""
    try:
        return FLOW_CHANNEL_DESCRIPTIONS[method]
    except KeyError as error:
        raise ValueError(f"Unsupported flow method: {method}") from error


def compute_tv_l1_flow(img1: np.ndarray, img2: np.ndarray, image_size: int | None = None) -> np.ndarray:
    """Compute the reference-project TV-L1 flow tensor [u, v, strain]."""
    if not hasattr(cv2, "optflow") or not hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        raise RuntimeError("cv2.optflow.DualTVL1OpticalFlow_create is unavailable in this interpreter.")

    image1 = _prepare_grayscale(img1, image_size=image_size)
    image2 = _prepare_grayscale(img2, image_size=image_size)
    optical_flow_create = cv2.optflow.DualTVL1OpticalFlow_create()
    optical_flow = optical_flow_create.calc(image1, image2, None)
    u = optical_flow[..., 0]
    v = optical_flow[..., 1]
    optical_strain = compute_optical_strain(u, v)
    pair_feature = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.float32)
    pair_feature[:, :, 0] = u
    pair_feature[:, :, 1] = v
    pair_feature[:, :, 2] = optical_strain
    return pair_feature


def compute_farneback_flow(img1: np.ndarray, img2: np.ndarray, image_size: int | None = None) -> np.ndarray:
    """Compute a Farneback dense flow tensor [u, v, magnitude]."""
    image1 = _prepare_grayscale(img1, image_size=image_size)
    image2 = _prepare_grayscale(img2, image_size=image_size)
    optical_flow = cv2.calcOpticalFlowFarneback(
        image1,
        image2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    u = optical_flow[..., 0]
    v = optical_flow[..., 1]
    magnitude = np.sqrt(u**2 + v**2)
    pair_feature = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.float32)
    pair_feature[:, :, 0] = u
    pair_feature[:, :, 1] = v
    pair_feature[:, :, 2] = magnitude
    return pair_feature


def compute_flow(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    method: str = "tv_l1",
    image_size: int | None = None,
) -> np.ndarray:
    """Compute a backend-selected optical-flow tensor."""
    if method == "tv_l1":
        return compute_tv_l1_flow(img1, img2, image_size=image_size)
    if method == "farneback":
        return compute_farneback_flow(img1, img2, image_size=image_size)
    raise ValueError(f"Unsupported flow method: {method}")


def compute_optical_flow(img1: np.ndarray, img2: np.ndarray, image_size: int = 128) -> np.ndarray:
    """Backwards-compatible TV-L1 wrapper used by the current scripts."""
    return compute_tv_l1_flow(img1, img2, image_size=image_size)


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Create a directly viewable RGB image from normalized u, v, and magnitude."""
    u = flow[..., 0]
    v = flow[..., 1]
    magnitude = np.sqrt(u**2 + v**2)
    stacked = np.stack([u, v, magnitude], axis=-1)
    output = np.zeros_like(stacked, dtype=np.uint8)
    for channel in range(3):
        channel_data = stacked[..., channel]
        normalized = cv2.normalize(channel_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        output[..., channel] = normalized.astype(np.uint8)
    return output


def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    """Create a standard HSV-direction optical-flow visualization and return it as RGB."""
    u = flow[..., 0]
    v = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def save_flow_bundle(
    flow: np.ndarray,
    npy_path: str | Path,
    rgb_path: str | Path,
    hsv_path: str | Path,
) -> dict[str, str]:
    """Save the raw flow tensor and both visualization images."""
    npy_path = Path(npy_path)
    rgb_path = Path(rgb_path)
    hsv_path = Path(hsv_path)
    ensure_directory(npy_path.parent)
    ensure_directory(rgb_path.parent)
    ensure_directory(hsv_path.parent)

    np.save(npy_path, flow)
    rgb_image = flow_to_rgb(flow)
    hsv_image = flow_to_hsv(flow)
    if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write RGB flow visualization: {rgb_path}")
    if not cv2.imwrite(str(hsv_path), cv2.cvtColor(hsv_image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write HSV flow visualization: {hsv_path}")
    return {
        "npy_path": str(npy_path),
        "rgb_path": str(rgb_path),
        "hsv_path": str(hsv_path),
    }


def build_flow_export_metadata(
    *,
    method: str,
    resize_before_flow: bool,
    resized_shape_used: list[int],
    original_aligned_shape: dict[str, list[int]],
    saved_files: dict[str, Any],
    crop_status: dict[str, Any],
    align_status: dict[str, Any],
    pair_failures: dict[str, str],
) -> dict[str, Any]:
    """Build the standard metadata fields shared by trainset and testset exports."""
    return {
        "flow_method": method,
        "flow_tensor_channels": describe_flow_tensor_channels(method),
        "resize_before_flow": resize_before_flow,
        "resized_shape_used": resized_shape_used,
        "original_aligned_shape": original_aligned_shape,
        "saved_files": saved_files,
        "crop_status": crop_status,
        "align_status": align_status,
        "pair_failures": pair_failures,
    }
