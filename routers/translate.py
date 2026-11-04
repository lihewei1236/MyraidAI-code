"""翻译路由 — 文本翻译（中英互译）

接口：
- POST /translate/text   一次性返回完整译文
"""
import time
import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from config import MAX_TRANSLATE_LENGTH
from services.translate_service import translate_service

logger = logging.getLogger("router.translate")
router = APIRouter(prefix="/translate", tags=["文本翻译"])


# ────────── 请求模型 ──────────

class TranslateRequest(BaseModel):
    text: str
    source_lang: Literal["zh", "en"] = "zh"   # 仅支持 zh / en
    target_lang: Literal["zh", "en"] = "en"   # 仅支持 zh / en

    @field_validator("text")
    @classmethod
    def _v_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("翻译内容不能为空")
        if len(v) > MAX_TRANSLATE_LENGTH:
            raise ValueError(
                f"翻译内容过长，最多 {MAX_TRANSLATE_LENGTH} 字（当前 {len(v)} 字）"
            )
        return v


# ────────── 接口 ──────────

@router.post("/text")
async def translate_text(req: TranslateRequest):
    try:
        start = time.time()
        translated = await asyncio.to_thread(
            translate_service.translate,
            req.text,
            req.source_lang,
            req.target_lang,
        )
        return {
            "code": 200,
            "message": "翻译成功",
            "data": {
                "original": req.text,
                "translated": translated,
                "source_lang": req.source_lang,
                "target_lang": req.target_lang,
                "processing_time_seconds": round(time.time() - start, 2),
            },
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("翻译失败 %s", e)
        raise HTTPException(500, "翻译服务内部错误，请稍后重试")
