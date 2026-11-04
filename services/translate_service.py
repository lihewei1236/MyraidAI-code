"""翻译服务 — 中英互译，复用 qwen_service 的 Qwen3-VL 模型

职责：
- 校验源语言 / 目标语言（仅支持 zh ↔ en）
- 拼装 system + user 两条 messages
- 委托 qwen_service 完成生成
- 对模型输出做兜底清理（去前缀 / 去成对引号）

调用方（路由）只关心业务字段，不需要懂 qwen 的 messages 结构。
"""
import logging
from typing import Optional

from prompts import TRANSLATE_SYSTEM, LANG_NAME_MAP, build_translate_user_prompt
from services.qwen_service import qwen_service

logger = logging.getLogger("translate_service")

# 模型偶尔会自作主张加上的前缀
_OUTPUT_PREFIXES = (
    "译文：", "译文:", "翻译：", "翻译:",
    "Translation:", "Translated:",
)
# 模型偶尔会把整段译文包在成对引号里
_QUOTE_PAIRS = [
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
]


def _build_messages(text: str, src_name: str, tgt_name: str) -> list[dict]:
    user_prompt = build_translate_user_prompt(text, src_name, tgt_name)
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": TRANSLATE_SYSTEM}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        },
    ]


def _post_clean(raw: str) -> str:
    """兜底清理模型输出：去掉多余前缀 / 成对引号 / 首尾空白"""
    text = raw.strip()
    for prefix in _OUTPUT_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    for left, right in _QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
            break
    return text


class TranslateService:
    @staticmethod
    def translate(
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en",
        max_new_tokens: int = 2048,
        session_id: Optional[str] = None,
    ) -> str:
        """中英互译入口。

        - source_lang / target_lang：仅支持 "zh" 或 "en"
        - 返回纯译文字符串
        """
        src_name = LANG_NAME_MAP.get(source_lang)
        tgt_name = LANG_NAME_MAP.get(target_lang)
        supported = list(LANG_NAME_MAP.keys())

        if src_name is None:
            raise ValueError(f"不支持的源语言：{source_lang}，仅支持 {supported}")
        if tgt_name is None:
            raise ValueError(f"不支持的目标语言：{target_lang}，仅支持 {supported}")
        if source_lang == target_lang:
            raise ValueError("源语言和目标语言不能相同")

        logger.info(
            "开始翻译：%s → %s，文本长度 %d 字", source_lang, target_lang, len(text)
        )

        raw = qwen_service.generate(
            messages=_build_messages(text, src_name, tgt_name),
            max_new_tokens=max_new_tokens,
            session_id=session_id,
        )
        result = _post_clean(raw)
        logger.info("翻译完成（前 100 字）：%s", result[:100])
        return result


translate_service = TranslateService()
