"""Qwen3-VL-2B 单例服务 — 纯底层模型服务

这一层只负责：
1. 模型加载（懒加载 + 线程安全）
2. 通用的 generate / generate_stream（接收已经构造好的 messages）
3. 取消机制（cancel by session_id）

所有 prompt / system_prompt / 业务逻辑 都由调用方（chat_service / ocr_service /
summary_service）负责构造 messages 后传进来。

公开 API：
- ensure_loaded()                                    # 显式预加载
- generate(messages, images=None, ...)               # 非流式（文本 / 图文统一入口）
- generate_stream(messages, images=None, ...)        # 流式（文本 / 图文统一入口）
- cancel(session_id=None)                            # 取消指定会话；None 表示取消全部
"""
import re
import logging
import threading
from typing import Optional, Generator

from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from config import TORCH_DEVICE, get_model_path, get_torch_dtype

logger = logging.getLogger("qwen_service")

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# 视觉 token 数控制：512~2048 个 28×28 的视觉 patch
MIN_PIXELS = 512 * 28 * 28
MAX_PIXELS = 2048 * 28 * 28
# 流式输出时清理 chat template 残留的 role 前缀
_ROLE_PREFIX_RE = re.compile(r"^user\n.*?\nassistant\n", re.DOTALL)


class _CancelCriteria(StoppingCriteria):
    """通过 threading.Event 来打断 model.generate()"""

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.stop_event.is_set()


class QwenService:
    def __init__(self):
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._sessions: dict[str, threading.Event] = {}
        self._sessions_lock = threading.Lock()

    # ────────── 加载 ──────────

    def ensure_loaded(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            logger.info("加载 %s (device=%s)", MODEL_NAME, TORCH_DEVICE)
            model_path = get_model_path(MODEL_NAME)
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=get_torch_dtype(),
                attn_implementation="sdpa",
            ).to(TORCH_DEVICE)
            self._model.eval()
            self._processor = AutoProcessor.from_pretrained(
                model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
            )
            logger.info("Qwen3-VL-2B 加载完成")

    # ────────── 会话 / 取消 ──────────

    def _register(self, session_id: Optional[str]) -> threading.Event:
        """注册一个会话用于支持取消；session_id 为 None 时返回一个一次性 Event"""
        event = threading.Event()
        if session_id:
            with self._sessions_lock:
                self._sessions[session_id] = event
        return event

    def _unregister(self, session_id: Optional[str]):
        if not session_id:
            return
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    def cancel(self, session_id: Optional[str] = None) -> bool:
        """取消指定 session_id 的生成；session_id 为 None 时取消全部。
        返回 True 表示找到了至少一个目标会话。
        """
        with self._sessions_lock:
            if session_id:
                event = self._sessions.get(session_id)
                if event:
                    event.set()
                    logger.info("停止信号已发送 session=%s", session_id)
                    return True
                logger.warning("未找到会话 session=%s", session_id)
                return False
            for event in self._sessions.values():
                event.set()
            logger.info("停止信号已发送 (全部 %d 个会话)", len(self._sessions))
            return bool(self._sessions)

    # ────────── 内部工具 ──────────

    @staticmethod
    def _clean_prefix(text: str) -> str:
        return _ROLE_PREFIX_RE.sub("", text, count=1).strip()

    def _prepare_inputs(self, messages: list[dict], images: Optional[list[Image.Image]]):
        """messages → 模型 inputs"""
        text_input = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        kwargs = dict(text=[text_input], padding=True, return_tensors="pt")
        if images:
            kwargs["images"] = images
        return self._processor(**kwargs).to(self._model.device)

    # ────────── 非流式 ──────────

    def generate(
        self,
        messages: list[dict],
        images: Optional[list[Image.Image]] = None,
        max_new_tokens: int = 2048,
        session_id: Optional[str] = None,
    ) -> str:
        """非流式生成（文本 / 图文共用）。"""
        self.ensure_loaded()
        stop_event = self._register(session_id)

        try:
            with self._gen_lock:
                inputs = self._prepare_inputs(messages, images)
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.05,
                    stopping_criteria=StoppingCriteriaList([_CancelCriteria(stop_event)]),
                )
                generated = output_ids[0][len(inputs.input_ids[0]):]
                result = self._processor.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                return self._clean_prefix(result)
        finally:
            self._unregister(session_id)

    # ────────── 流式 ──────────

    def generate_stream(
        self,
        messages: list[dict],
        images: Optional[list[Image.Image]] = None,
        max_new_tokens: int = 2048,
        session_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """流式生成（文本 / 图文共用，按 token 块返回）。"""
        self.ensure_loaded()
        stop_event = self._register(session_id)

        try:
            with self._gen_lock:
                inputs = self._prepare_inputs(messages, images)

                streamer = TextIteratorStreamer(
                    self._processor.tokenizer,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.05,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([_CancelCriteria(stop_event)]),
                )

                gen_thread = threading.Thread(
                    target=self._model.generate, kwargs=gen_kwargs, daemon=True
                )
                gen_thread.start()

                buffer, cleaned = "", False
                try:
                    for token in streamer:
                        if stop_event.is_set():
                            break
                        if cleaned:
                            yield token
                            continue
                        # 清掉 chat template 残留的 "user\n...\nassistant\n" 前缀
                        buffer += token
                        m = _ROLE_PREFIX_RE.match(buffer)
                        if m:
                            cleaned = True
                            rest = buffer[m.end():]
                            if rest:
                                yield rest
                        elif len(buffer) > 100:
                            cleaned = True
                            yield buffer
                            buffer = ""
                    if not cleaned and buffer:
                        flushed = self._clean_prefix(buffer)
                        if flushed:
                            yield flushed
                finally:
                    gen_thread.join()
        finally:
            self._unregister(session_id)


# 全局单例
qwen_service = QwenService()