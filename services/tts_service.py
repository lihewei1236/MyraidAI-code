import os
import uuid
import re
import logging
import numpy
import soundfile
from kokoro import KPipeline, KModel
import threading

from config import TORCH_DEVICE, get_model_path

KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
KOKORO_WEIGHTS_FILENAME = "kokoro-v1_0.pth"
VOICE_ID = "zm_yunyang"
SPEED = 0.70
SAMPLE_RATE = 24000
SILENCE_DURATION = 0.35
TRIM_MS = 580
TRIM_SAMPLES = int(SAMPLE_RATE * TRIM_MS / 1000)

logger = logging.getLogger("tts_service")


def split_text_by_sentence(text: str) -> list[str]:
    sentences = re.split(r'(?<=[。！？!?…\n])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    result = []
    for s in sentences:
        if len(s) > 80:
            sub = re.split(r'(?<=[，,；;])', s)
            result.extend(part.strip() for part in sub if part.strip())
        else:
            result.append(s)

    final = []
    for s in result:
        while len(s) > 80:
            cut = 80
            for i in range(80, 60, -1):
                if s[i - 1] in '，,。！？!?…、；;：:':
                    cut = i
                    break
            final.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            final.append(s)

    return final


class TTSService:
    def __init__(self):
        self._pipeline = None
        self._lock = threading.Lock()

    def ensure_loaded(self):
        if self._pipeline is None:
            logger.info("加载 Kokoro-82M 模型...")
            # 解析本地 snapshot 路径，避免 KModel 内部走 hf_hub_download
            snapshot_dir = get_model_path(KOKORO_REPO_ID)
            config_path = os.path.join(snapshot_dir, "config.json")
            weights_path = os.path.join(snapshot_dir, KOKORO_WEIGHTS_FILENAME)
            if not (os.path.isfile(config_path) and os.path.isfile(weights_path)):
                raise RuntimeError(
                    f"Kokoro 本地模型缺失：请确认 {snapshot_dir} 下存在 "
                    f"config.json 和 {KOKORO_WEIGHTS_FILENAME}"
                )
            kmodel = KModel(
                repo_id=KOKORO_REPO_ID,
                config=config_path,
                model=weights_path,
            ).to(TORCH_DEVICE).eval()
            # 注：voices/*.pt 仍由 KPipeline 内部调用 hf_hub_download 读取，
            # 因 config.py 已设 HF_HUB_OFFLINE=1 + HF_HUB_CACHE=MODEL_DIR，
            # 它会直接命中本地缓存，不会联网。
            self._pipeline = KPipeline(
                lang_code='z',
                repo_id=KOKORO_REPO_ID,
                model=kmodel,
            )
            logger.info("Kokoro-82M 加载完成")

    # 兼容旧调用
    _ensure_loaded = ensure_loaded

    def text_to_speech(
            self,
            text: str,
            filename: str | None = None,
            voice: str = VOICE_ID,
            output_dir: str = "outputs",
    ) -> str:
        self.ensure_loaded()
        with self._lock:
            if filename is None:
                filename = f"tts_{uuid.uuid4().hex}.wav"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)

            sentences = split_text_by_sentence(text)
            all_audio = []
            silence = numpy.zeros(int(SAMPLE_RATE * SILENCE_DURATION), dtype=numpy.float32)

            for i, sentence in enumerate(sentences, 1):
                logger.info("  [%d/%d] (%d字) %s", i, len(sentences), len(sentence), sentence[:50])
                try:
                    audio_parts = []
                    for _, _, audio in self._pipeline(sentence, voice=voice, speed=SPEED):
                        audio_parts.append(audio)
                    if audio_parts:
                        audio = numpy.concatenate(audio_parts)
                        if len(audio) > TRIM_SAMPLES * 2:
                            audio = audio[TRIM_SAMPLES:]
                        all_audio.append(audio)
                        if i < len(sentences):
                            all_audio.append(silence)
                except Exception as e:
                    logger.error("单句合成失败：%s... 错误：%s", sentence[:30], e)

            if not all_audio:
                raise RuntimeError("所有句子都合成失败，请检查模型状态")

            full_audio = numpy.concatenate(all_audio)
            soundfile.write(output_path, full_audio, SAMPLE_RATE)
            logger.info("合成完成：%s（时长=%.1fs）", output_path, len(full_audio) / SAMPLE_RATE)
            return output_path

tts_service = TTSService()