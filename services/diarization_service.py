"""说话人分割服务 — Pyannote speaker-diarization-3.0

离线环境变量（HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 等）已由 config.py 统一设置，
此处无需重复声明。
"""

import os
import logging
import threading
from unittest.mock import patch

import torch
from pyannote.audio import Pipeline
from config import TORCH_DEVICE

logger = logging.getLogger("diarization_service")
MODEL_NAME = "pyannote/speaker-diarization-3.0"

# PyTorch 2.6+ 兼容补丁：torch.load 默认 weights_only=True 会让 pyannote 加载失败
_original_torch_load = torch.load
_original_load_state_dict = torch.nn.Module.load_state_dict


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


def _patched_load_state_dict(self, state_dict, strict=True, assign=False):
    return _original_load_state_dict(self, state_dict, strict=strict, assign=True)


class DiarizationService:
    def __init__(self):
        self._pipeline = None
        self._lock = threading.Lock()

    def ensure_loaded(self):
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            token = os.getenv("HF_TOKEN") or None  # 离线缓存可用时不需要 token
            logger.info("加载 %s ... (device=%s, token=%s)",
                        MODEL_NAME, TORCH_DEVICE, "yes" if token else "no")
            with patch("torch.load", _patched_torch_load), \
                 patch("torch.nn.Module.load_state_dict", _patched_load_state_dict):
                pipe = Pipeline.from_pretrained(MODEL_NAME, use_auth_token=token)
            if pipe is None:
                raise RuntimeError(
                    f"Pyannote 加载失败：{MODEL_NAME}。"
                    "请确认 MODEL_DIR 下已存在该模型缓存，或设置 HF_TOKEN 后联网首拉一次。"
                )
            self._pipeline = pipe.to(torch.device(TORCH_DEVICE))
            logger.info("说话人分割模型加载完成")

    # 兼容旧调用
    _ensure_loaded = ensure_loaded

    def diarize(self, audio_path: str) -> list[dict]:
        self.ensure_loaded()
        diarization = self._pipeline(audio_path)
        return [
            {
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            }
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]


diarization_service = DiarizationService()