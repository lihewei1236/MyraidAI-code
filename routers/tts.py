import os
import time
import uuid
import asyncio
import logging
from config import MAX_TTS_LENGTH, OUTPUT_DIR
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from fastapi.responses import FileResponse
from services.tts_service import tts_service



logger = logging.getLogger("router.tts")
router = APIRouter(prefix="/tts", tags=["文字转语音"])






class TTSRequest(BaseModel):
    text: str
    voice: str = "zm_yunyang"

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('文本内容不能为空')
        if len(v) > MAX_TTS_LENGTH:
            raise ValueError(f'文本过长（{len(v)}字），单次最多 {MAX_TTS_LENGTH} 字，请精简后重试')
        return v


@router.post("/generate")
async def generate_speech(request: TTSRequest):
    try:
        start_time = time.time()
        filename = f"tts_{uuid.uuid4().hex}.wav"
        await asyncio.to_thread(
            tts_service.text_to_speech,
            request.text,
            filename,
            request.voice,
            OUTPUT_DIR,
        )
        return {
            "code": 200,
            "message": "语音生成成功",
            "data": {
                "file_url": f"/tts/download/{filename}",
                "text_length": len(request.text),
                "processing_time_seconds": round(time.time() - start_time, 2),
            }
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("语音生成失败 %s", e)
        raise HTTPException(500, "TTS服务内部错误，请稍后重试")


@router.get("/download/{filename}")
async def download_file(filename: str):
    real_path = os.path.realpath(os.path.join(OUTPUT_DIR, filename))
    if not real_path.startswith(os.path.realpath(OUTPUT_DIR)):
        raise HTTPException(403, "禁止访问")

    if not os.path.exists(real_path):
        raise HTTPException(404, "文件不存在或已过期")

    return FileResponse(
        path=real_path,
        media_type="audio/wav",
        filename=filename
    )