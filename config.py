"""全局配置 —— 一切环境差异都收敛到这里 + .env 文件

加载顺序：
1. .env                  （通用，可放共享配置）
2. .env.<PROFILE>        （按 profile 覆盖，例如 .env.server）

部署切换：在服务器上只需 `export PROFILE=server`（或写在 .env 里），
无需修改任何代码。
"""
import os
from dotenv import load_dotenv

# ────────── 1. 加载 .env ──────────
# 先加载根 .env，拿到 PROFILE
load_dotenv()
PROFILE = os.getenv("PROFILE", "local").lower()
# 再加载 profile 对应的 .env.<profile>，允许覆盖前面的值
load_dotenv(f".env.{PROFILE}", override=True)


# ────────── 2. 工具：env -> bool ──────────
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ────────── 3. 派生开关（按 profile 自动给默认值，env 可单独覆盖） ──────────
IS_SERVER = PROFILE == "server"

# 启动时是否一次性加载所有模型（服务器默认开，本地默认懒加载）
EAGER_LOAD = _env_bool("EAGER_LOAD", IS_SERVER)
# 加载后是否常驻显存（服务器常驻；本地若加 LRU 卸载逻辑，可关掉）
KEEP_MODELS_RESIDENT = _env_bool("KEEP_MODELS_RESIDENT", IS_SERVER)

# 设备 & 精度
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cuda" if IS_SERVER else "cpu")
TORCH_DTYPE_STR = os.getenv("TORCH_DTYPE", "float16" if IS_SERVER else "float32")


# ────────── 4. 模型目录（**全部离线读取**，不联网） ──────────
# 服务器把 HF 缓存目录上传到这里即可，结构为：
#   <MODEL_DIR>/models--<org>--<name>/snapshots/<hash>/...
MODEL_DIR = os.getenv(
    "MODEL_DIR",
    r"C:\Users\sysf\Desktop\AI\MyraidAI-model" if not IS_SERVER else "/home/ubuntu/MyraidAI-model",
)

# 强制离线 —— 必须在任何 transformers / huggingface_hub / pyannote 导入之前设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_CACHE"] = MODEL_DIR
# 兼容旧变量名（部分库还在读 TRANSFORMERS_CACHE / HF_HOME）
os.environ.setdefault("TRANSFORMERS_CACHE", MODEL_DIR)
os.environ.setdefault("HF_HOME", MODEL_DIR)


# ────────── 5. 业务常量 ──────────
MAX_TTS_LENGTH = int(os.getenv("MAX_TTS_LENGTH", "3000"))
MAX_TRANSLATE_LENGTH = int(os.getenv("MAX_TRANSLATE_LENGTH", "1500"))

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".ts"}
SOUNDFILE_NATIVE_EXTENSIONS = {".wav", ".flac", ".ogg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ────────── 6. 模型相关 helper ──────────

def get_torch_dtype():
    """把 TORCH_DTYPE_STR 翻译成 torch 真实 dtype。"""
    import torch
    s = TORCH_DTYPE_STR.lower()
    if s == "auto":
        if TORCH_DEVICE == "cuda" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if TORCH_DEVICE == "mps":
            return torch.float16
        return torch.float32
    return {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }.get(s, torch.float32)


def get_model_path(model_name: str) -> str:
    """把 HuggingFace 模型名解析成 MODEL_DIR 里的实际 snapshot 路径。

    HF 缓存目录的标准结构：
        <MODEL_DIR>/models--<org>--<name>/snapshots/<commit-hash>/
    """
    if not MODEL_DIR or not os.path.isdir(MODEL_DIR):
        return model_name

    folder = "models--" + model_name.replace("/", "--")
    snapshot_root = os.path.join(MODEL_DIR, folder, "snapshots")
    if not os.path.isdir(snapshot_root):
        return model_name

    snapshots = [
        d for d in os.listdir(snapshot_root)
        if os.path.isdir(os.path.join(snapshot_root, d))
    ]
    if not snapshots:
        return model_name

    snapshots.sort(
        key=lambda d: os.path.getmtime(os.path.join(snapshot_root, d)),
        reverse=True,
    )
    return os.path.join(snapshot_root, snapshots[0])
