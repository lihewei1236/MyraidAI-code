"""ASR 路由 — 语音转文字

接口：
- POST /asr/transcribe                 直接整段转写，返回纯文本
- POST /asr/transcribe_with_speakers   说话人分割 + 逐段转写，返回带说话人标签的逐字稿

业务编排（diarize + 逐段 ASR）放在 services.asr_service.transcribe_with_speakers，
这里只负责 HTTP 入参 / 文件落盘 / 错误处理。
"""
import asyncio
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.asr_service import asr_service
from routers._audio_helpers import save_and_convert, cleanup

logger = logging.getLogger("router.asr")
router = APIRouter(prefix="/asr", tags=["ASR 语音转文字"])


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("zh")):
    """整段直接转写"""
    wav = await save_and_convert(file, "asr")
    try:
        result = await asyncio.to_thread(asr_service.transcribe, wav, language)
        return {
            "code": 200,
            "message": "转写成功",
            "data": {"text": result.get("text", "")},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("转写失败 %s", e)
        raise HTTPException(500, "ASR 转写服务内部错误，请稍后重试")
    finally:
        cleanup(wav)


@router.post("/transcribe_with_speakers")
async def transcribe_with_speakers(
    file: UploadFile = File(...),
    language: str = Form("zh"),
):
    """说话人分割 + 逐段转写，返回带说话人标签的逐字稿"""
    wav = await save_and_convert(file, "asr_diar")
    try:
        transcripts = await asyncio.to_thread(
            asr_service.transcribe_with_speakers, wav, language
        )
        return {
            "code": 200,
            "message": "转写成功",
            "data": {"transcripts": transcripts},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("带说话人转写失败 %s", e)
        raise HTTPException(500, "ASR 转写服务内部错误，请稍后重试")
    finally:
        cleanup(wav)
