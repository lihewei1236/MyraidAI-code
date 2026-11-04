"""ASR 服务 — Whisper large-v3-turbo

提供两个核心能力：
1. transcribe(audio_path, language)            —— 整段音频直接转写
2. transcribe_with_speakers(audio_path, lang)  —— 先做说话人分割，再逐段转写
                                                   返回 [{speaker, start, end, text}, ...]

说话人分割依赖 diarization_service；这层只负责"编排"，不负责文件上传/转码
（那是 HTTP 层的事，由 routers/_audio_helpers 处理）。
"""

from config import TORCH_DEVICE, get_model_path, get_torch_dtype
import logging
import threading
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from services.diarization_service import diarization_service

logger = logging.getLogger("asr_service")

MODEL_NAME = "openai/whisper-large-v3-turbo"

# Whisper 单次推理最长 30 秒，留 1 秒保险
_MAX_SEG_DURATION = 29.0

# Whisper 防幻觉：温度回退 + 压缩比/静音/置信度阈值（OpenAI 官方推荐配置）
ANTI_HALLUCINATION_KWARGS = {
    "temperature": [0.0, 0.2, 0.4],
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}


class ASRService:
    def __init__(self):
        self._pipeline = None
        self._lock = threading.Lock()

    def ensure_loaded(self):
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            dtype = get_torch_dtype()
            model_path = get_model_path(MODEL_NAME)
            logger.info("加载 %s ... (device=%s, dtype=%s)", MODEL_NAME, TORCH_DEVICE, dtype)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
            ).to(TORCH_DEVICE)
            processor = AutoProcessor.from_pretrained(model_path)
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=TORCH_DEVICE,
            )

    # 兼容旧调用
    _ensure_loaded = ensure_loaded

    # ────────── 音频加载 ──────────

    @staticmethod
    def _load_mono_16k(audio_path: str, frame_offset: int = 0, num_frames: int = -1):
        """加载音频（可指定区间），转单声道并重采样到 16kHz"""
        waveform, sr = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=num_frames)
        audio = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            sr = 16000
        return audio, sr

    def _run(self, audio, sr: int, language: str) -> str:
        """统一调 pipeline 并返回文本"""
        result = self._pipeline(
            {"array": audio.numpy(), "sampling_rate": sr},
            generate_kwargs={"language": language, **ANTI_HALLUCINATION_KWARGS},
            return_timestamps=True,  # 长音频分块推理必需
        )
        return result.get("text", "").strip()

    # ────────── 公开接口 ──────────

    def transcribe(self, audio_path: str, language: str = "zh") -> dict:
        """转写整段音频"""
        self._ensure_loaded()
        audio, sr = self._load_mono_16k(audio_path)
        return {"text": self._run(audio, sr, language)}

    def transcribe_segment(self, audio_path: str, start: float, end: float,
                           language: str = "zh") -> str:
        """转写指定时间区间的音频片段（用于带说话人的逐段转写）"""
        self._ensure_loaded()
        info = torchaudio.info(audio_path)
        frame_offset = int(start * info.sample_rate)
        num_frames = int((end - start) * info.sample_rate)
        audio, sr = self._load_mono_16k(audio_path, frame_offset, num_frames)
        return self._run(audio, sr, language)

    def transcribe_with_speakers(
        self,
        audio_path: str,
        language: str = "zh",
    ) -> list[dict]:
        """说话人分割 → 合并/拆分 → 逐段 ASR。

        返回：[{speaker, start, end, text}, ...]
        """
        self._ensure_loaded()
        raw = diarization_service.diarize(audio_path)
        segments = self._normalize_segments(raw)
        logger.info("分割完成，共 %d 段，开始逐段转写...", len(segments))

        transcripts = []
        for i, seg in enumerate(segments, 1):
            text = self.transcribe_segment(audio_path, seg["start"], seg["end"], language)
            logger.info(
                "[%d/%d] %s (%.2fs-%.2fs): %s",
                i, len(segments), seg["speaker"], seg["start"], seg["end"], text[:30],
            )
            transcripts.append({**seg, "text": text})

        logger.info("全部转写完成")
        return transcripts

    # ────────── 内部：段处理 ──────────

    @staticmethod
    def _normalize_segments(segments: list[dict], gap: float = 0.5) -> list[dict]:
        """合并相邻同说话人片段（间隔 <gap 秒），并把超过 29 秒的段切开"""
        if not segments:
            return []
        # 合并
        merged = [dict(segments[0])]
        for seg in segments[1:]:
            prev = merged[-1]
            if (
                seg["speaker"] == prev["speaker"]
                and seg["start"] - prev["end"] < gap
                and seg["end"] - prev["start"] <= _MAX_SEG_DURATION
            ):
                prev["end"] = seg["end"]
            else:
                merged.append(dict(seg))
        # 拆分
        result = []
        for seg in merged:
            pos = seg["start"]
            while pos < seg["end"] - 0.1:
                chunk_end = min(pos + _MAX_SEG_DURATION, seg["end"])
                result.append({**seg, "start": pos, "end": chunk_end})
                pos = chunk_end
        return result


asr_service = ASRService()
