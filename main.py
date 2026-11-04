import config  # noqa: F401  保证环境变量先于任何 transformers / pyannote 导入
import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from routers import tts, asr, diarization, chat, ocr, summary, translate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("main")


# ==================== 启动预热（受 config.EAGER_LOAD 控制） ====================
async def _preload_all_models():
    """串行加载四个模型；任何一个失败都让进程启动失败（fail-fast）。

    注意：必须串行加载，且 Pyannote 放最后。
    因为 diarization_service.ensure_loaded() 内部用 unittest.mock.patch
    临时改了全局的 torch.load 和 torch.nn.Module.load_state_dict，
    如果并行加载，会污染其他模型（Whisper / Qwen / Kokoro）的加载路径，
    导致出现 'Cannot copy out of meta tensor' 等奇怪错误。
    """
    from services.qwen_service import qwen_service
    from services.asr_service import asr_service
    from services.diarization_service import diarization_service
    from services.tts_service import tts_service

    # 顺序很重要：Pyannote 放最后，避免它的全局补丁污染其他模型
    targets = [
        ("Qwen3-VL", qwen_service.ensure_loaded),
        ("Whisper", asr_service.ensure_loaded),
        ("Kokoro-TTS", tts_service.ensure_loaded),
        ("Pyannote", diarization_service.ensure_loaded),
    ]

    for name, fn in targets:
        t0 = time.perf_counter()
        logger.info("⏳ 预加载 %s ...", name)
        try:
            await asyncio.to_thread(fn)
        except Exception:
            logger.exception("✗ %s 加载失败", name)
            raise
        logger.info("✓ %s 加载完成 (%.1fs)", name, time.perf_counter() - t0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PROFILE=%s  DEVICE=%s  DTYPE=%s  MODEL_DIR=%s",
                config.PROFILE, config.TORCH_DEVICE, config.TORCH_DTYPE_STR, config.MODEL_DIR)
    if config.EAGER_LOAD:
        t0 = time.perf_counter()
        await _preload_all_models()
        logger.info("🚀 所有模型预热完成，总耗时 %.1fs", time.perf_counter() - t0)
    else:
        logger.info("懒加载模式：模型将在首次调用时加载")
    yield
    logger.info("应用退出")


# ==================== 创建 FastAPI 应用 ====================
app = FastAPI(
    title="万象 MyriadAI",
    description="多模态 AI 智能平台 — 听、说、看、思、译",
    version="1.0.0",
    lifespan=lifespan,
)

# ==================== CORS 跨域配置 ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 全局异常处理 ====================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    messages = []
    for error in exc.errors():
        field = error.get("loc", [])[-1] if error.get("loc") else ""
        msg = error.get("msg", "")
        messages.append(f"{field}: {msg}" if field else msg)
    return JSONResponse(
        status_code=422,
        content={"code": 422, "message": "; ".join(messages)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.status_code, "message": exc.detail},
        )
    logging.exception("未捕获的异常 %s", exc)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "服务内部错误，请稍后重试"},
    )


app.include_router(tts.router)
app.include_router(asr.router)
app.include_router(diarization.router)

app.include_router(chat.router)
app.include_router(ocr.router)
app.include_router(summary.router)
app.include_router(translate.router)


# ==================== 健康检查 ====================
@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"status": "ok", "profile": config.PROFILE}


# ==================== 静态前端页面 ====================
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/static/myriad-ai.html")