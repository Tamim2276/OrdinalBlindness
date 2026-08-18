"""
model.py — EfficientNet-B3 wrapper + device selection

Handles:
  - EfficientNet-B3 with 5-class DR grading head
  - Checkpoint save / load helpers
  - Device selection: CUDA > XPU (Intel Arc) > CPU
"""

import torch
import timm
import warnings


def get_device():
    """
    Select the best available accelerator.

    Priority: CUDA (NVIDIA/AMD) > XPU (Intel Arc) > CPU

    This allows the same codebase to run on:
      - Any NVIDIA GPU (CUDA)
      - Intel Arc B580 (XPU)
      - Any CPU fallback
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        print(f"[Device] Using CUDA: {name}")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        name   = torch.xpu.get_device_name(0)
        print(f"[Device] Using XPU: {name}")
    else:
        device = torch.device("cpu")
        print("[Device] No GPU found — falling back to CPU (~3-10x slower).")
        print("         Training will be very slow. A GPU is strongly recommended.")
    return device


def get_autocast_context(device):
    """
    Returns the correct autocast context manager for the active device.

    CUDA supports float16 and bfloat16. XPU supports bfloat16.
    CPU autocast is a no-op but safe to call.

    Usage:
        with get_autocast_context(device):
            outputs = model(images)
    """
    device_type = device.type  # "cuda", "xpu", or "cpu"

    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    elif device_type == "xpu":
        return torch.autocast(device_type="xpu", dtype=torch.bfloat16)
    else:
        # CPU: autocast with bfloat16 is valid but usually not beneficial
        # Return a no-op context to keep calling code clean
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)


def get_model(num_classes=5, pretrained=True):
    """
    Create EfficientNet-B3 with a custom classification head.

    Args:
        num_classes : Number of output classes (5 for DR grades 0-4)
        pretrained  : Load ImageNet weights. Should be True for all real runs.
                      False only for architecture smoke tests.

    Returns:
        torch.nn.Module — EfficientNet-B3 with replaced classifier head
    """
    if not pretrained:
        warnings.warn(
            "get_model(pretrained=False) — model weights are random. "
            "Only use this for architecture smoke tests, never for real training.",
            UserWarning
        )

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def save_checkpoint(model, path, extra=None):
    """
    Save model weights and optional metadata to a .pth file.

    Args:
        model : The nn.Module to save
        path  : File path to save to (e.g. "results/best_centralised.pth")
        extra : Optional dict of extra info to store alongside weights
                e.g. {"epoch": 12, "best_qwk": 0.847}
    """
    payload = {"model_state_dict": model.state_dict()}
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, num_classes=5, device=None, pretrained=False):
    """
    Load a saved checkpoint back into a model.

    Args:
        path        : Path to the saved .pth file
        num_classes : Must match the saved model's head (default 5)
        device      : Device to load onto. Uses get_device() if not specified.
        pretrained  : Passed to get_model() — False since we're loading saved weights

    Returns:
        model       : nn.Module with loaded weights, set to eval() mode
        meta        : Dict of any extra metadata saved alongside weights
    """
    if device is None:
        device = get_device()

    payload = torch.load(path, map_location=device, weights_only=True)

    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {k: v for k, v in payload.items() if k != "model_state_dict"}

    print(f"[Checkpoint] Loaded from '{path}'")
    if meta:
        print(f"[Checkpoint] Metadata: {meta}")

    return model, meta


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    device = get_device()

    model  = get_model(num_classes=5, pretrained=True).to(device)

    total, trainable = count_parameters(model)
    print(f"[Model] Total params    : {total:,}")
    print(f"[Model] Trainable params: {trainable:,}")

    x   = torch.randn(2, 3, 224, 224, device=device)
    with get_autocast_context(device):
        out = model(x)

    assert out.shape == (2, 5), f"Expected output (2, 5), got {out.shape}"
    print(f"[Smoke Test] Input {x.shape} → Output {out.shape} ✅")
    print(f"\n✅ model.py fully verified on {device}")