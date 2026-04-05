from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MAME_DIR = Path(os.getenv("MAME_DIR", str(ROOT_DIR)))
DEFAULT_MEGC_JSONL = Path(
    os.getenv(
        "MEGC_JSONL",
        str(ROOT_DIR.parent / "MEGC" / "data" / "task1" / "me_vqa_samm_casme2_smic_v2.jsonl"),
    )
)
DEFAULT_SAMPLE_ANSWER_DIR = Path(
    os.getenv(
        "SAMPLE_ANSWER_DIR",
        str(ROOT_DIR.parent / "MEGC" / "data" / "sampleAnswer" / "1"),
    )
)


ABLATION_CHOICES = (
    "full",
    "no_specialists",
    "no_critic",
    "no_prior",
    "no_geometry_calibration",
    "no_present_gate",
)


@dataclass(frozen=True)
class AblationConfig:
    name: str = "full"
    disable_specialists: bool = False
    disable_critic: bool = False
    disable_prior: bool = False
    disable_geometry_calibration: bool = False
    disable_present_gate: bool = False

    @classmethod
    def from_name(cls, value: str | None) -> "AblationConfig":
        name = str(value or "full").strip().lower().replace("-", "_")
        aliases = {
            "": "full",
            "none": "full",
            "baseline": "full",
            "wo_specialists": "no_specialists",
            "wospecialists": "no_specialists",
            "wo_critic": "no_critic",
            "wocritic": "no_critic",
            "wo_prior": "no_prior",
            "woprior": "no_prior",
            "wo_geometry_calibration": "no_geometry_calibration",
            "wo_geometry": "no_geometry_calibration",
            "wogeometry": "no_geometry_calibration",
            "wo_present_gate": "no_present_gate",
            "wopresentgate": "no_present_gate",
        }
        normalized = aliases.get(name, name)
        if normalized == "full":
            return cls()
        if normalized == "no_specialists":
            return cls(name=normalized, disable_specialists=True)
        if normalized == "no_critic":
            return cls(name=normalized, disable_critic=True)
        if normalized == "no_prior":
            return cls(name=normalized, disable_prior=True)
        if normalized == "no_geometry_calibration":
            return cls(name=normalized, disable_geometry_calibration=True)
        if normalized == "no_present_gate":
            return cls(name=normalized, disable_present_gate=True)
        raise ValueError(f"Unsupported ablation mode: {value}")


_ACTIVE_ABLATION = AblationConfig()


def get_active_ablation() -> AblationConfig:
    return _ACTIVE_ABLATION


@contextmanager
def ablation_context(config: AblationConfig | None):
    global _ACTIVE_ABLATION

    previous = _ACTIVE_ABLATION
    _ACTIVE_ABLATION = config or AblationConfig()
    try:
        yield _ACTIVE_ABLATION
    finally:
        _ACTIVE_ABLATION = previous


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--mame-dir", default=str(DEFAULT_MAME_DIR))
    parser.add_argument("--megc-jsonl", default=str(DEFAULT_MEGC_JSONL))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vllm_directory", default="", help="Local vision model directory for vLLM/Qwen.")
    parser.add_argument(
        "--llm_directory",
        default="",
        help="Local reasoning LLM directory. If omitted, falls back to --vllm_directory.",
    )
    parser.add_argument("--gpu-mem", type=float, default=0.90, dest="gpu_mem")
    parser.add_argument(
        "--reasoning-gpu-mem",
        type=float,
        default=0.90,
        dest="reasoning_gpu_mem",
    )
    parser.add_argument(
        "--reasoning-device",
        default="",
        dest="reasoning_device",
        help="CUDA device id(s) for the reasoning model, e.g. 0 or 0,1.",
    )
    parser.add_argument(
        "--reasoning-tensor-parallel-size",
        type=int,
        default=1,
        dest="reasoning_tensor_parallel_size",
        help="Tensor parallel size for the reasoning model. Use 2 with --reasoning-device 0,1 to shard across two GPUs.",
    )
    parser.add_argument(
        "--vision-device",
        default="",
        dest="vision_device",
        help="CUDA device id for the vision model, e.g. 1.",
    )
    parser.add_argument("--memory", default="au_emotion_memory.json")
    parser.add_argument("--scope", default="")
    parser.add_argument("--no-vision", action="store_true", dest="no_vision")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--flow-format", default="npy", choices=["auto", "rgb", "hsv", "npy"])
    parser.add_argument(
        "--ablation",
        default="full",
        choices=ABLATION_CHOICES,
        help="Ablation mode for controlled experiments.",
    )
    return parser
