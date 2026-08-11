import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

MODEL_NAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

PROJECT_MODELS = Path(__file__).resolve().parents[2] / "models"
CACHE_DIR = Path.home() / ".cache" / "sign-language-to-text"

class ModelUnavailable(RuntimeError):
    pass

def candidates() -> List[Path]:
    env = os.environ.get("SLT_HAND_MODEL")
    paths = [Path(env)] if env else []
    return paths + [PROJECT_MODELS / MODEL_NAME, CACHE_DIR / MODEL_NAME]

def find() -> Optional[Path]:
    for path in candidates():
        if path.is_file():
            return path
    return None

def download(target: Optional[Path] = None, timeout: float = 60.0) -> Path:
    target = target or (CACHE_DIR / MODEL_NAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                shutil.copyfileobj(response, tmp)
                tmp_path = Path(tmp.name)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ModelUnavailable(
            f"cannot download {MODEL_NAME}: {exc}\n"
            f"Download {MODEL_URL} manually and put it in {target}"
        ) from exc
    tmp_path.replace(target)
    return target

def hand_landmarker_model() -> Path:
    found = find()
    if found is not None:
        return found
    if os.environ.get("SLT_NO_DOWNLOAD"):
        raise ModelUnavailable(
            f"{MODEL_NAME} not found and downloading is disabled by SLT_NO_DOWNLOAD.\n"
            f"Looked in: {', '.join(str(p) for p in candidates())}"
        )
    print(f"[sign-language-to-text] downloading {MODEL_NAME} -> {CACHE_DIR}")
    return download()

if __name__ == "__main__":
    print(hand_landmarker_model())