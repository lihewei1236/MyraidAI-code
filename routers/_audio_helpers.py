"""音频路由共享 helper

HTTP 层的纯工具：上传文件 → 落盘 → 必要时 ffmpeg 转 16kHz 单声道 wav。
asr / summary 两个路由都会用到。

业务层（asr_service / diarization_service / summary_service）不应该依赖这里
—— 它们只接收已经准备好的本地音频路径。
"""
import os
import uuid
import asyncio
import logging
import subprocess

from fastapi import UploadFile, HTTPException

from config import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    SOUNDFILE_NATIVE_EXTENSIONS,
    OUTPUT_DIR,
)

logger = logging.getLogger("router.audio_helpers")


def _check_ffmpeg() -> bool:
    """启动时检测 ffmpeg 是否可调用"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False, timeout=5)
        return True
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


_FFMPEG_AVAILABLE = _check_ffmpeg()


async def save_and_convert(file: UploadFile, prefix: str) -> str:
    """保存上传文件；非 wav/flac/ogg 的音频以及所有视频，统一转为 16kHz 单声道 wav。

    返回最终落盘的 wav 路径。
    """
    ext = os.path.splitext(file.filename)[1].lower() or ".wav"
    if ext not in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式：{ext}")

    src = os.path.join(OUTPUT_DIR, f"{prefix}_{uuid.uuid4().hex[:8]}{ext}")
    with open(src, "wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            f.write(chunk)
    if os.path.getsize(src) == 0:
        raise HTTPException(400, "上传文件为空")

    if ext in SOUNDFILE_NATIVE_EXTENSIONS:
        return src

    if not _FFMPEG_AVAILABLE:
        os.remove(src)
        raise HTTPException(
            501,
            f"处理 {ext} 格式需要安装 ffmpeg，请执行: "
            f"winget install ffmpeg 或 apt install ffmpeg"
        )

    wav = src.rsplit(".", 1)[0] + ".wav"
    cmd = ["ffmpeg", "-i", src, "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", "-y", wav]
    r = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
    os.remove(src)
    if r.returncode != 0:
        raise HTTPException(500, f"ffmpeg 音频转换失败：{r.stderr[:200]}")
    return wav


def cleanup(path: str):
    if path and os.path.exists(path):
        os.remove(path)
