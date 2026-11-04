"""OCR 路由 — 图片文字提取"""
import os
import uuid
import time
import asyncio
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from config import OUTPUT_DIR, IMAGE_EXTENSIONS
from services.ocr_service import ocr_service

logger = logging.getLogger("router.ocr")
router = APIRouter(prefix="/ocr", tags=["OCR 图片文字提取"])


@router.post("/extract")
async def extract_text(
    file: UploadFile = File(...),
    prompt: str = Form(None),
):
    ext = os.path.splitext(file.filename)[1].lower() or ".png"
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(400, f"不支持的图片格式：{ext}，仅支持 {', '.join(IMAGE_EXTENSIONS)}")

    tmp_path = os.path.join(OUTPUT_DIR, f"ocr_{uuid.uuid4().hex}{ext}")
    try:
        with open(tmp_path, "wb") as f:
            while chunk := await file.read(8 * 1024 * 1024):
                f.write(chunk)
        if os.path.getsize(tmp_path) == 0:
            raise HTTPException(400, "上传文件为空")

        start = time.time()
        text = await asyncio.to_thread(ocr_service.extract_text, tmp_path, prompt)
        return {
            "code": 200,
            "message": "文字提取成功",
            "data": {
                "text": text,
                "processing_time_seconds": round(time.time() - start, 2),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("OCR 失败 %s", e)
        raise HTTPException(500, "OCR 服务内部错误，请稍后重试")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)