"""Chat 路由 — 普通对话

接口：
- POST /chat/generate   非流式，一次性返回完整回复
- POST /chat/stream     流式 (SSE)，逐 token 推送
- POST /chat/cancel     取消某个 session_id 正在进行的生成
"""
import json
import uuid
import queue
import asyncio
import logging
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from services.chat_service import chat_service

logger = logging.getLogger("router.chat")
router = APIRouter(prefix="/chat", tags=["对话"])


# ────────── 请求模型 ──────────

class ChatHistoryItem(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatHistoryItem] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    max_new_tokens: int = 2048
    # 客户端可以自己传 session_id；若不传，服务端会自动生成并返回
    session_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def _v_msg(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message 不能为空")
        return v


class CancelRequest(BaseModel):
    session_id: Optional[str] = None  # 不传则取消所有正在进行的


# ────────── 非流式 ──────────

@router.post("/generate")
async def chat_generate(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    try:
        history = [h.model_dump() for h in req.history]
        text = await asyncio.to_thread(
            chat_service.chat,
            req.message,
            history,
            req.system_prompt,
            req.max_new_tokens,
            session_id,
        )
        return {
            "code": 200,
            "message": "OK",
            "data": {
                "session_id": session_id,
                "reply": text,
            },
        }
    except Exception as e:
        logger.exception("chat 生成失败 %s", e)
        raise HTTPException(500, "Chat 服务内部错误，请稍后重试")


# ────────── 流式 (SSE) ──────────

def _sse_format(event: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    history = [h.model_dump() for h in req.history]

    # 用 threading.Queue + 后台线程跑同步生成器，主协程消费
    q: queue.Queue = queue.Queue()
    _SENTINEL = object()

    def produce():
        try:
            for token in chat_service.chat_stream(
                req.message,
                history,
                req.system_prompt,
                req.max_new_tokens,
                session_id,
            ):
                q.put(("token", token))
        except Exception as e:
            logger.exception("流式生成异常 %s", e)
            q.put(("error", str(e)))
        finally:
            q.put((_SENTINEL, None))

    threading.Thread(target=produce, daemon=True).start()

    async def event_gen():
        # 先把 session_id 推给客户端，前端拿来挂"停止"按钮
        yield _sse_format("session", {"session_id": session_id})
        while True:
            kind, payload = await asyncio.to_thread(q.get)
            if kind is _SENTINEL:
                yield _sse_format("done", "[DONE]")
                break
            if kind == "error":
                yield _sse_format("error", {"error": payload})
                break
            yield _sse_format("token", {"token": payload})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关掉 nginx 缓冲
        },
    )


# ────────── 取消 ──────────

@router.post("/cancel")
async def chat_cancel(req: CancelRequest):
    ok = chat_service.cancel(req.session_id)
    return {
        "code": 200 if ok else 404,
        "message": "已发送停止信号" if ok else "未找到对应会话",
        "data": {"session_id": req.session_id},
    }