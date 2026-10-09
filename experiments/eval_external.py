"""
eval_external.py — External validation on APTOS 2019 (Reviewer #2: independent validation data)

Every finished run's best-validation model (results/runs/<run>/best.pth, the model the paper
reports) and the centralised model are scored on the 3,662 labelled APTOS 2019 training
images. Nothing is trained or tuned on APTOS.

Data: the public 224x224 copy of the APTOS 2019 training set on Hugging Face
(bumbledeep/aptos, grades 0-4, same counts as the Kaggle release). Download once with
    python experiments/eval_external.py --download

Output: results/external_aptos.json (per-run QWK, accuracy, confusion matrix)
"""

import os
import io
import sys
import json
import argparse

os.environ.setdefault("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "16384")

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model      import get_device, load_checkpoint
from src.evaluation import predict, summarise, make_eval_loader
from src.run_utils  import atomic_write_json, use_utf8_output

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
APTOS_DIR  = os.path.join(BASE_DIR, 'data', 'APTOS')
PARQUET    = os.path.join(APTOS_DIR, 'bumbledeep', 'data', 'train-00000-of-00001.parquet')
CACHE_NPY  = os.path.join(APTOS_DIR, 'aptos_224.npy')
LABELS     = os.path.join(APTOS_DIR, 'aptos_labels.json')
RUNS_DIR   = os.path.join(BASE_DIR, 'results', 'runs')
OUT_JSON   = os.path.join(BASE_DIR, 'results', 'external_aptos.json')


def download():
    from huggingface_hub import hf_hub_download
    hf_hub_download('bumbledeep/aptos', 'data/train-00000-of-00001.parquet',
                    repo_type='dataset', local_dir=os.path.join(APTOS_DIR, 'bumbledeep'))


def build_cache():
    """Decode the parquet once into a uint8 array (N, 224, 224, 3) plus labels."""
    if os.path.exists(CACHE_NPY) and os.path.exists(LABELS):
        return
    import pyarrow.parquet as pq
    from PIL import Image
    table  = pq.read_table(PARQUET)
    labels = [int(v) for v in table.column('label_code').to_pylist()]
    images = table.column('image').to_pylist()
    arr = np.zeros((len(images), 224, 224, 3), dtype=np.uint8)
    for i, rec in enumerate(images):
        im = Image.open(io.BytesIO(rec['bytes'])).convert('RGB')
        if im.size != (224, 224):
            im = im.resize((224, 224), Image.BILINEAR)
        arr[i] = np.asarray(im)
    np.save(CACHE_NPY, arr)
    atomic_write_json({"labels": labels, "names": [rec.get('path') for rec in images]}, LABELS)


class APTOSDataset(Dataset):
    """Same evaluation transform as the DDR validation/test splits."""

    def __init__(self):
        self.images = np.load(CACHE_NPY, mmap_mode='r')
        with open(LABELS) as f:
            self.labels = json.load(f)["labels"]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(np.array(self.images[idx]))
        return image, torch.tensor(self.labels[idx], dtype=torch.long)


def finished_runs():
    for name in sorted(os.listdir(RUNS_DIR)):
        run_dir = os.path.join(RUNS_DIR, name)
        cfg_path = os.path.join(run_dir, 'config.json')
        if not (os.path.exists(os.path.join(run_dir, 'done.json')) and os.path.exists(cfg_path)):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        if name != f"{cfg['method']}_{cfg['partition']}_s{cfg['seed']}":
            continue   # tagged test run
        yield name, cfg, os.path.join(run_dir, 'best.pth')


def main():
    use_utf8_output()
    parser = argparse.ArgumentParser(description="External validation on APTOS 2019")
    parser.add_argument("--download", action="store_true", help="Download the APTOS parquet first")
    args = parser.parse_args()

    if args.download or not os.path.exists(PARQUET):
        download()
    build_cache()

    device = get_device()
    loader = make_eval_loader(APTOSDataset(), device)
    print(f"APTOS 2019: {len(loader.dataset)} images")

    results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            results = json.load(f)

    jobs = [("centralised", {"method": "centralised", "partition": "-", "seed": 42},
             os.path.join(BASE_DIR, 'results', 'best_centralised.pth'))] + list(finished_runs())
    for name, cfg, ckpt in jobs:
        if name in results:
            continue   # resumable after a power cut
        model, _ = load_checkpoint(ckpt, num_classes=5, device=device)
        labels, preds, _ = predict(model, loader, device)
        m = summarise(labels, preds)
        results[name] = {"method": cfg["method"], "partition": cfg["partition"],
                         "seed": cfg["seed"], **m}
        atomic_write_json(results, OUT_JSON)
        print(f"  {name:36s} QWK {m['qwk']:.4f} | acc {m['acc']:.4f}")
        del model

    print(f"\nSaved {len(results)} results to {OUT_JSON}")


if __name__ == "__main__":
    main()
