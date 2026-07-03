"""
load_env.py — Load API keys from .env file into environment.

Add this import to the top of pipeline.py:
    from load_env import load_env; load_env()

Or run standalone to verify keys are set:
    py -3.11 load_env.py
"""
import os
from pathlib import Path


def load_env(path: str = ".env") -> dict:
    """
    Load key=value pairs from .env file into os.environ.
    Skips blank lines and comments (lines starting with #).
    Does NOT override keys already set in the environment.

    Returns dict of keys that were loaded.
    """
    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded = {}
    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip()
            # Strip inline comments
            if " #" in value:
                value = value[:value.index(" #")].strip()
            # Strip quotes
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            if len(value) >= 2 and value[0] == value[-1] == "'":
                value = value[1:-1]
            # Don't override existing env vars
            if key and value and key not in os.environ:
                os.environ[key] = value
                loaded[key] = value

    return loaded


if __name__ == "__main__":
    loaded = load_env()
    print(f"Loaded {len(loaded)} keys from .env\n")

    # Verify each key
    keys = [
        ("OPENROUTER_API_KEY", "sk-or-"),
        ("GEMINI_API_KEY",     "AIzaSy"),
        ("GROQ_API_KEY",       "gsk_"),
        ("CEREBRAS_API_KEY",   "csk-"),
    ]
    for env_var, prefix in keys:
        val = os.environ.get(env_var, "")
        if not val:
            status = "NOT SET"
        elif val.endswith("-your-key-here") or "your-key" in val:
            status = "PLACEHOLDER (not a real key)"
        elif val.startswith(prefix):
            status = f"OK  ({val[:12]}...)"
        else:
            status = f"SET but unexpected format ({val[:8]}...)"
        print(f"  {env_var:<25} {status}")
