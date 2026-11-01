"""Resolve historical project paths without depending on the original workstation."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(os.environ.get("MEDFORGE_WORKSPACE", REPO_ROOT / "workspace")).expanduser()
DATA_ROOT = Path(os.environ.get("MEDFORGE_DATA_ROOT", REPO_ROOT / "data")).expanduser()
MODEL_ROOT = Path(os.environ.get("MEDFORGE_MODEL_ROOT", WORKSPACE_ROOT / "models")).expanduser()


def release_path(relative):
    """Map the original project-relative suffix to a configurable local path."""
    for prefix, root in (
        ("AAAI/MedForge++", DATA_ROOT),
        ("CIRCE", REPO_ROOT),
        ("Prompts", REPO_ROOT / "prompts"),
        ("models", MODEL_ROOT),
    ):
        if relative == prefix or relative.startswith(prefix + "/"):
            suffix = relative[len(prefix):].lstrip("/")
            result = str(root / suffix) if suffix else str(root)
            return result + ("/" if relative.endswith("/") else "")
    result = str(WORKSPACE_ROOT / relative)
    return result + ("/" if relative.endswith("/") else "")
