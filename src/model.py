"""
model.py — EfficientNet-B3 wrapper + device selection

Handles:
  - EfficientNet-B3 with 5-class DR grading head
  - Checkpoint save / load helpers
"""

import torch
import timm
import warnings


def get_device():
    """
    Select the best available device.
    Priority: XPU (Intel Arc) > CPU

    Note: CUDA is intentionally not listed — this project runs on
    Intel Arc B580 which uses torch.xpu, not torch.cuda.
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"[Device] Using XPU: {torch.xpu.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] XPU not available — falling back to CPU (~3x slower)")
    return device


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
    # FIX 4: Warn loudly if pretrained=False to prevent silent bad runs
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

    Usage:
        save_checkpoint(model, "results/best.pth", {"epoch": 10, "qwk": 0.85})
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
                      (e.g. {"epoch": 12, "best_qwk": 0.847})
                      Empty dict if nothing extra was saved.

    Usage:
        model, meta = load_checkpoint("results/best_centralised.pth")
        print(f"Loaded checkpoint from epoch {meta.get('epoch', '?')}")
    """
    if device is None:
        device = get_device()

    # FIX 2: Load checkpoint with weights_only=True for security
    # (prevents arbitrary code execution from malicious .pth files)
    payload = torch.load(path, map_location=device, weights_only=True)

    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    # Return everything except the state dict as metadata
    meta = {k: v for k, v in payload.items() if k != "model_state_dict"}

    print(f"[Checkpoint] Loaded from '{path}'")
    if meta:
        print(f"[Checkpoint] Metadata: {meta}")

    return model, meta


def count_parameters(model):
    """
    Count total and trainable parameters.
    Useful for the paper's model description section.

    Returns:
        total     : Total parameter count
        trainable : Parameters with requires_grad=True
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    device = get_device()

    #Build model
    model  = get_model(num_classes=5, pretrained=True).to(device)

    # FIX 3: Print parameter count to confirm head is correctly replaced
    total, trainable = count_parameters(model)
    print(f"[Model] Total params    : {total:,}")
    print(f"[Model] Trainable params: {trainable:,}")

    #Forward pass smoke test
    x   = torch.randn(2, 3, 300, 300, device=device)
    out = model(x)

    assert out.shape == (2, 5), f"Expected output (2, 5), got {out.shape}"
    print(f"[Smoke Test] Input {x.shape} → Output {out.shape} ✅")

    # Checkpoint round-trip test
    # Save -> reload -> verify outputs match
    save_checkpoint(model, "/tmp/test_checkpoint.pth", {"epoch": 0, "best_qwk": -1.0})

    loaded_model, meta = load_checkpoint(
        "/tmp/test_checkpoint.pth",
        num_classes=5,
        device=device
    )

    with torch.no_grad():
        out_loaded = loaded_model(x)

    assert torch.allclose(out, out_loaded, atol=1e-5), \
        "Checkpoint round-trip failed — outputs differ after reload!"

    print(f"[Checkpoint] Round-trip test passed ✅ | Meta: {meta}")
    print(f"\n✅ model.py fully verified on {device}")