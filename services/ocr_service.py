"""OCR 服务 — 图片预处理 + Qwen 推理 + 输出清理

预处理逻辑非常关键：Qwen-VL-2B 对小图/极端长宽比图会退化为输出坐标。
- 小图先放大到 ≥1800px
- 极端长宽比做白色 padding
- 输出检测到全是坐标则返回错误提示
"""
import re
import logging
from typing import Optional

from PIL import Image, ImageOps

from prompts import OCR_DEFAULT
from services.qwen_service import qwen_service

logger = logging.getLogger("ocr_service")

# ── 预处理阈值 ──
MIN_WIDTH_FOR_OCR = 1800        # 小于此宽度的图直接放大
MAX_WIDTH_AFTER_UPSCALE = 2400  # 放大上限
ASPECT_RATIO_THRESHOLD = 3.0    # 长宽比超过 3:1 触发 padding
TARGET_ASPECT = 2.0

# ── 坐标识别 ──
_COORD_DOUBLE = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)\s*\(\s*\d+\s*,\s*\d+\s*\)")
_COORD_SINGLE = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)")
_PREFIX_PATTERNS = re.compile(
    r"^(图片中的文字内容[:：]?|提取的文字[:：]?|文字内容[:：]?|图片中的文字[:：]?)\s*"
)


# ────────── 预处理 ──────────

def _upscale_if_small(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w >= MIN_WIDTH_FOR_OCR:
        return image
    scale = min(MIN_WIDTH_FOR_OCR / w, MAX_WIDTH_AFTER_UPSCALE / w)
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info("放大图片 %.2fx：%dx%d → %dx%d", scale, w, h, new_w, new_h)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _pad_extreme_aspect(image: Image.Image) -> Image.Image:
    w, h = image.size
    aspect = w / h
    if aspect > ASPECT_RATIO_THRESHOLD:
        new_h = int(w / TARGET_ASPECT)
        pad = new_h - h
        logger.info("图片过宽 (%.2f:1)，padding 至 %dx%d", aspect, w, new_h)
        return ImageOps.expand(image, (0, pad // 2, 0, pad - pad // 2), fill="white")
    if aspect < 1 / ASPECT_RATIO_THRESHOLD:
        new_w = int(h / TARGET_ASPECT)
        pad = new_w - w
        logger.info("图片过高 (1:%.2f)，padding 至 %dx%d", 1 / aspect, new_w, h)
        return ImageOps.expand(image, (pad // 2, 0, pad - pad // 2, 0), fill="white")
    return image


# ────────── 后处理 ──────────

def _is_mostly_coordinates(raw: str) -> bool:
    stripped = _COORD_DOUBLE.sub("", raw)
    stripped = _COORD_SINGLE.sub("", stripped)
    stripped = re.sub(r"[\s,,.。、；;:：\n\r]+", "", stripped)
    return len(stripped) < 5 and len(raw) > 20


def _clean_output(raw: str) -> str:
    text = _COORD_DOUBLE.sub("", raw)
    text = _COORD_SINGLE.sub("", text)
    text = _PREFIX_PATTERNS.sub("", text)
    # 去掉清坐标后剩下的纯标点空行
    lines = [l for l in text.split("\n") if re.sub(r"[\s,,.。、；;:：]+", "", l)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ────────── 内部：构造 messages ──────────

def _build_messages(prompt: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]


# ────────── 主入口 ──────────

class OcrService:
    @staticmethod
    def extract_text(image_path: str, prompt: Optional[str] = None) -> str:
        prompt = prompt or OCR_DEFAULT

        image = Image.open(image_path).convert("RGB")
        logger.info("原始图片尺寸：%dx%d", image.width, image.height)
        image = _upscale_if_small(image)
        image = _pad_extreme_aspect(image)
        logger.info("最终送入模型尺寸：%dx%d", image.width, image.height)

        raw = qwen_service.generate(
            messages=_build_messages(prompt),
            images=[image],
        )
        logger.info("模型 raw 输出（前 300 字）：%s", raw[:300])

        if _is_mostly_coordinates(raw):
            logger.warning("模型仍输出坐标，可能图片质量过低")
            return "[OCR 识别失败：模型输出了坐标信息。请尝试提高图片分辨率或换更清晰的图片]"

        result = _clean_output(raw) or raw.strip() or "[OCR 未识别到文字]"
        logger.info("OCR 完成，提取文字 %d 字", len(result))
        return result


ocr_service = OcrService()