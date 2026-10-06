"""
run_utils.py — Seeding, crash-safe state files and sleep prevention

Shared by run_fl.py, run_queue.py and run_centralised.py so every long job
survives a power cut (load shedding) and resumes where it stopped.

Crash-safe writes:
    The new file is written to "<name>.tmp", flushed to disk, and only then
    renamed over the old one, so a cut mid-write never leaves a half-written
    file. The previous copy is kept as "<name>.prev" and used as a fallback.
"""

import os
import sys
import json
import random

import numpy as np
import torch


def _xpu_ready():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


# ── Seeding ────────────────────────────────────────────────────────────────────

def set_seed(seed):
    """Seed Python, NumPy and PyTorch (CPU + CUDA/XPU) for a repeatable run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _xpu_ready():
        torch.xpu.manual_seed_all(seed)


def get_rng_state():
    """Snapshot every random generator so a resumed run continues the same stream."""
    state = {
        "python": random.getstate(),
        "numpy" : np.random.get_state(),
        "torch" : torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if _xpu_ready():
        state["xpu"] = torch.xpu.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "xpu" in state and _xpu_ready():
        torch.xpu.set_rng_state_all(state["xpu"])


# ── Crash-safe files ───────────────────────────────────────────────────────────

def atomic_torch_save(obj, path, keep_previous=False):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    if keep_previous and os.path.exists(path):
        os.replace(path, path + ".prev")
    os.replace(tmp, path)


def atomic_write_text(text, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(obj, path):
    atomic_write_text(json.dumps(obj, indent=2), path)


def load_state_with_fallback(path):
    """
    Load a state file saved with atomic_torch_save(keep_previous=True).

    Returns (state, path_used), or (None, None) if no readable copy exists.
    Only our own files are loaded here, so weights_only=False is safe.
    """
    for candidate in (path, path + ".prev"):
        if not os.path.exists(candidate):
            continue
        try:
            return torch.load(candidate, map_location="cpu", weights_only=False), candidate
        except Exception as e:
            print(f"[Resume] Could not read {candidate} ({e}) — trying the older copy")
    return None, None


def remove_state_files(path):
    for p in (path, path + ".prev", path + ".tmp"):
        if os.path.exists(p):
            os.remove(p)


# ── Console output ─────────────────────────────────────────────────────────────

def use_utf8_output():
    """
    Print emoji/arrows safely. When output is piped (e.g. Git Bash windows,
    `| tee`), Python on Windows falls back to cp1252 and crashes on '✅'.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# ── Windows sleep ──────────────────────────────────────────────────────────────

def keep_awake():
    """Stop Windows from sleeping while this process runs (no-op on Linux/macOS)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS      = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
