import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Model profiles — add new GGUF models here
# ---------------------------------------------------------------------------
MODEL_PROFILES: Dict[str, Dict] = {
    "qwen3-vl-8b": {
        "repo_id": "mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF",
        "model_file": "Qwen3-VL-8B-Instruct-abliterated-v2.0.Q5_K_S.gguf",
        "mmproj_file": "Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-f16.gguf",
        "chat_handler": "qwen3vl",
        "n_ctx": 2048,
        "n_batch": 32,
    },
    "ui-tars-1.5-7b": {
        "repo_id": "Mungert/UI-TARS-1.5-7B-GGUF",
        "model_file": "UI-TARS-1.5-7B-q5_k_m.gguf",
        "mmproj_file": "UI-TARS-1.5-7B-f16.mmproj",
        "chat_handler": "qwen25vl",
        "n_ctx": 32768,
        "n_batch": 512,
    },
    "ui-tars-1.5-7b-q8": {
        "repo_id": "mradermacher/UI-TARS-1.5-7B-GGUF",
        "model_file": "UI-TARS-1.5-7B.Q8_0.gguf",
        "mmproj_file": "UI-TARS-1.5-7B.mmproj-Q8_0.gguf",
        "chat_handler": "qwen25vl",
        "n_ctx": 32768,
        "n_batch": 512,
    },
}

# Active model — change this to switch models (or set env var CUAOS_MODEL)
DEFAULT_MODEL = os.environ.get("CUAOS_MODEL", "qwen3-vl-8b")

# HuggingFace cache — store all models in L:\models\huggingface
HF_CACHE_DIR = os.environ.get("HF_HOME", r"L:\models\huggingface")
os.environ["HF_HOME"] = HF_CACHE_DIR

# ---------------------------------------------------------------------------
# Windows: ensure CUDA DLLs from PyTorch are on the DLL search path
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    _torch_lib = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".venv", "Lib", "site-packages", "torch", "lib",
    )
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")


def _active_profile() -> Dict[str, str]:
    return MODEL_PROFILES[DEFAULT_MODEL]


@dataclass
class CFG:
    # ----------------------
    # LLM Model config
    # ----------------------
    MODEL_NAME: str = DEFAULT_MODEL
    GGUF_REPO_ID: str = _active_profile()["repo_id"]
    GGUF_MODEL_FILENAME: str = _active_profile()["model_file"]
    GGUF_MMPROJ_FILENAME: str = _active_profile()["mmproj_file"]
    CHAT_HANDLER: str = _active_profile()["chat_handler"]

    # ----------------------
    # Planning LLM (API-based)
    # ----------------------
    PLANNER_PROVIDER: str = "openrouter"       # openrouter | openai | local
    PLANNER_API_KEY: str = ""                  # set via UI or env var
    PLANNER_MODEL: str = ""
    PLANNER_MAX_TOKENS: int = 1024
    # Llama runtime
    N_CTX: int = 2048
    N_THREADS: int = 12
    N_GPU_LAYERS: int = -1
    N_BATCH: int = 32

    FORCE_REASONING: bool = False

    # Repeat guard
    STOP_ON_REPEAT: bool = True
    MAX_NUDGES: int = 3             # nudge model N times before hard-stopping

    # Tolerance for detecting repeated clicks on the same point
    REPEAT_XY_EPS: float = 0.01     # normalized 0..1

    # Open VM screen as a separate window
    OPEN_VNC_VIEWER: bool = True
    IMAGE_MIN_TOKENS: int = 1024
    IMAGE_MAX_TOKENS: int = 4096

    # ----------------------
    # Sandbox Docker config
    # ----------------------
    SANDBOX_IMAGE: str = "docker.io/trycua/cua-xfce:latest"
    SANDBOX_NAME: str = "cua_xfce_agent"

    # (Host side) VNC & API ports. 
    # container 5901: VNC
    # container 6901: noVNC (mapped externally)
    # container 8000: Computer Server API
    VNC_PORT: int = 5901
    NOVNC_PORT: int = 6901
    API_PORT: int = 8001

    # Docker run settings
    DOCKER_SHM_SIZE: str = "512m"
    VNC_RESOLUTION: str = "1920x1080"
    VNC_COL_DEPTH: int = 24

    # If True: launch VNC viewer automatically
    OPEN_VNC_VIEWER: bool = True

    # ----------------------
    # Agent loop timing
    # ----------------------
    WAIT_BEFORE_SCREENSHOT_SEC: float = 2.2
    PAUSE_AFTER_CLICK_SEC: float = 0.25

    SCREENSHOT_PATH: str = "./img/screen.png"
    MAX_DIM: int = 1280

    PREVIEW_PATH_TEMPLATE: str = "./img/click_preview_step_{i}.png"

    MIN_MARGIN: float = 0.005
    CONFIDENCE_MIN: float = 0.15

    WAIT_CHANGE_TIMEOUT: float = 3.0
    WAIT_CHANGE_INTERVAL: float = 0.25
    CHANGE_THRESHOLD: float = 0.01

    MAX_STEPS: int = 100
    MODEL_RETRY: int = 2
    API_READY_TIMEOUT: int = 120  # seconds

    # Sandbox screen size cache (seconds)
    SCREEN_CACHE_TTL: float = 0.5

    # Anti-loop
    REPEAT_CLICK_DISTANCE_PX: int = 10

    ALLOWED_PRESS_KEYS: Tuple[str, ...] = (
        "enter", "tab", "esc", "backspace", "delete",
        "up", "down", "left", "right",
        "home", "end", "pageup", "pagedown",
        "space"
    )

    ALLOWED_HOTKEYS: Tuple[Tuple[str, ...], ...] = (
        ("ctrl", "l"),
        ("ctrl", "t"),
        ("ctrl", "w"),
        ("alt", "tab"),
    )


# Instantiate config
cfg = CFG()

# ----------------------
# Regex / MIME tables
# ----------------------

# vision.image_to_data_uri needs this
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

# Used in llm_client to parse JSON from model text
JSON_RE = re.compile(r"\{.*\}", re.S)
