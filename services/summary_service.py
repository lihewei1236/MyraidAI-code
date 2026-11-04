"""会议总结服务 — 拼装 transcript + prompt，调 qwen，解析 JSON"""
import re
import json
import logging
from typing import Any

from prompts import build_meeting_summary_prompt
from services.qwen_service import qwen_service

logger = logging.getLogger("summary_service")

# JSON 抽取：去掉 markdown 代码块包裹
_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _format_transcript(segments: list[dict]) -> str:
    """把 [{speaker, text, start?, end?}, ...] 拼成可读的逐字稿文本"""
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    """尽最大努力从模型输出中抽出 JSON。"""
    # 优先匹配 markdown 代码块里的内容
    m = _CODEBLOCK_RE.search(raw)
    candidate = m.group(1) if m else raw

    # 找到第一个 { 到最后一个 }
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        logger.warning("JSON 解析失败：%s；原始输出前 200 字：%s", e, raw[:200])
        return {"raw": raw, "parse_error": str(e)}


def _build_messages(transcript_text: str, speakers: list[str]) -> list[dict]:
    system_prompt = build_meeting_summary_prompt(speakers)
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": transcript_text}],
        },
    ]


class SummaryService:
    @staticmethod
    def _collect_speakers(segments: list[dict]) -> list[str]:
        seen = []
        for seg in segments:
            sp = seg.get("speaker")
            if sp and sp not in seen:
                seen.append(sp)
        return seen

    def summarize(
        self,
        segments: list[dict],
        max_new_tokens: int = 2048,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """根据已带说话人标注的 transcript 段，生成结构化会议总结。"""
        if not segments:
            raise ValueError("segments 不能为空")

        speakers = self._collect_speakers(segments)
        transcript_text = _format_transcript(segments)
        if not transcript_text:
            raise ValueError("所有 transcript 段文本均为空，无法生成总结")

        logger.info("生成会议总结：说话人=%s，文本长度=%d",
                    speakers, len(transcript_text))

        raw = qwen_service.generate(
            messages=_build_messages(transcript_text, speakers),
            max_new_tokens=max_new_tokens,
            session_id=session_id,
        )
        return _extract_json(raw)


summary_service = SummaryService()