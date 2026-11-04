import os
import uuid
import asyncio
import logging

from fastapi import UploadFile, File, APIRouter, HTTPException
from config import OUTPUT_DIR, AUDIO_EXTENSIONS
from services.diarization_service import diarization_service

logger = logging.getLogger("router.diarization")
router = APIRouter(prefix="/diarization", tags=["说话人分割"])


def _cleanup(path: str):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            logger.warning("临时文件清理失败: %s", path)


@router.post("/analyze")
async def analyze_speakers(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower() or ".wav"
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式：{ext}")

    save_path = os.path.join(OUTPUT_DIR, f"diarize_{uuid.uuid4().hex[:8]}{ext}")
    try:
        with open(save_path, "wb") as f:
            while chunk := await file.read(8 * 1024 * 1024):
                f.write(chunk)
        if os.path.getsize(save_path) == 0:
            raise HTTPException(400, "上传文件为空")

        segments = await asyncio.to_thread(diarization_service.diarize, save_path)
        return {
            "code": 200,
            "message": "说话人分割成功",
            "data": {"segments": segments},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("说话人分割失败 %s", e)
        raise HTTPException(500, "说话人分割服务内部错误，请稍后重试")
    finally:
        _cleanup(save_path)