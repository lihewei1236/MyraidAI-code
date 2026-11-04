"""会议总结路由

只有一个接口：
- POST /meeting/summarize   上传一段会议音频 → 说话人分割 + 逐段 ASR → 结构化总结

业务逻辑就是一句话：
    transcripts = asr_service.transcribe_with_speakers(wav)
    summary     = summary_service.summarize(transcripts)
"""
import time
import asyncio
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from services.summary_service import summary_service
from services.asr_service import asr_service
from routers._audio_helpers import save_and_convert, cleanup

logger = logging.getLogger("router.meeting")
router = APIRouter(prefix="/meeting", tags=["会议总结"])


@router.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    language: str = Form("zh"),
):
    """一站式：上传会议音频 → 说话人分割 → 逐段 ASR → 结构化总结。

    返回数据里同时带 transcripts 和 summary，方便客户端展示完整链路。
    """
    wav = await save_and_convert(file, "meeting")
    try:
        # 1) 说话人分割 + 逐段转写
        t0 = time.time()
        transcripts = await asyncio.to_thread(
            asr_service.transcribe_with_speakers, wav, language
        )
        t_asr = time.time() - t0
        if not transcripts:
            raise HTTPException(400, "音频中未识别到任何有效语音")

        # 2) 结构化总结
        t1 = time.time()
        summary = await asyncio.to_thread(summary_service.summarize, transcripts)
        t_sum = time.time() - t1

        return {
            "code": 200,
            "message": "会议总结生成成功",
            "data": {
                "transcripts": transcripts,
                "summary": summary,
                "processing_time_seconds": {
                    "asr": round(t_asr, 2),
                    "summary": round(t_sum, 2),
                    "total": round(t_asr + t_sum, 2),
                },
            },
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("音频会议总结失败 %s", e)
        raise HTTPException(500, "会议总结服务内部错误，请稍后重试")
    finally:
        cleanup(wav)
