import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms


class DDRDataset(Dataset):
    """
    DDR Dataset loader for Diabetic Retinopathy grading.
    Skips ungradable images (Grade 5).
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split

        # 1. Define transforms based on split
        if transform is not None:
            self.transform = transform
        else:
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

        # 2. Read the text file mapping images to labels
        txt_path = os.path.join(root_dir, f"{split}.txt")
        self.samples = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_name, label = parts
                    if int(label) != 5:   # Skip ungradable images
                        self.samples.append((img_name, int(label)))

        # 3. Define the image folder path
        self.img_dir = os.path.join(root_dir, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # FIX: explicit existence check with clear error message
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class APTOSDataset(Dataset):
    """
    APTOS 2019 Blindness Detection dataset loader.
    CSV format: id_code, diagnosis
    """
    def __init__(self, root_dir, csv_file, img_dir, split="test", transform=None):
        self.root_dir = root_dir

        # FIX 1: Split-aware transforms (matches DDRDataset pattern)
        if transform is not None:
            self.transform = transform
        else:
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

        # FIX 2: Read CSV and validate expected columns exist
        csv_path = os.path.join(root_dir, csv_file)
        self.df = pd.read_csv(csv_path)

        expected_cols = {'id_code', 'diagnosis'}
        if not expected_cols.issubset(self.df.columns):
            raise ValueError(
                f"CSV must contain columns {expected_cols}. "
                f"Found: {set(self.df.columns)}"
            )

        # Reset index to guarantee iloc[idx] == row idx
        self.df = self.df.reset_index(drop=True)

        # 3. Define the image directory
        self.img_dir = os.path.join(root_dir, img_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # FIX 3: Use named columns via iloc — robust to column reordering
        row = self.df.iloc[idx]
        img_id = str(row['id_code'])
        label = int(row['diagnosis'])

        # FIX 4: Extension fallback with clear FileNotFoundError
        if not img_id.endswith(('.png', '.jpg')):
            img_path = os.path.join(self.img_dir, f"{img_id}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        else:
            img_path = os.path.join(self.img_dir, img_id)

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image not found for id '{img_id}' in {self.img_dir}"
            )

        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)