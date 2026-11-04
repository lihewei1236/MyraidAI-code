"""Chat 服务 — 普通对话业务层

职责：
- 构造 chat 用的 messages（system_prompt + history + 当前 message）
- 委托 qwen_service 完成生成
- 暴露 chat / chat_stream / cancel 三个能力

调用方（路由）只关心业务字段，不需要懂 qwen 的 messages 结构。
"""
import logging
from typing import Optional, Generator

from prompts import CHAT_DEFAULT
from services.qwen_service import qwen_service

logger = logging.getLogger("chat_service")


class ChatService:
    @staticmethod
    def _build_messages(
        message: str,
        history: Optional[list[dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """构造 Qwen 多模态 chat 所需的 messages 结构。

        history 形如 [{"role": "user"/"assistant", "content": "..."}]
        """
        messages = []
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt or CHAT_DEFAULT}],
        })
        if history:
            for h in history:
                role = h.get("role")
                content = h.get("content", "")
                if role not in ("user", "assistant") or not content:
                    continue
                messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": message}],
        })
        return messages

    def chat(
        self,
        message: str,
        history: Optional[list[dict]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 2048,
        session_id: Optional[str] = None,
    ) -> str:
        """非流式：一次性返回完整回复"""
        messages = self._build_messages(message, history, system_prompt)
        return qwen_service.generate(
            messages=messages,
            max_new_tokens=max_new_tokens,
            session_id=session_id,
        )

    def chat_stream(
        self,
        message: str,
        history: Optional[list[dict]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 2048,
        session_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """流式：按 token 块 yield 文本"""
        messages = self._build_messages(message, history, system_prompt)
        yield from qwen_service.generate_stream(
            messages=messages,
            max_new_tokens=max_new_tokens,
            session_id=session_id,
        )

    @staticmethod
    def cancel(session_id: Optional[str] = None) -> bool:
        """取消指定 session_id 的生成（None 表示取消全部）"""
        return qwen_service.cancel(session_id)


chat_service = ChatService()