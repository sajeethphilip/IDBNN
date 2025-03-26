import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# -------------------- Enhanced Model Architecture --------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.leaky_relu(out, 0.1)

class DeepFeatureExtractor(nn.Module):
    def __init__(self, input_dims: Tuple, feature_dim: int = 128, dropout_prob: float = 0.3):
        super().__init__()
        self.input_dims = input_dims

        self.encoder = nn.Sequential(
            ResidualBlock(3, 32),
            nn.Dropout2d(dropout_prob),
            ResidualBlock(32, 64, stride=2),
            nn.Dropout2d(dropout_prob),
            ResidualBlock(64, 128, stride=2),
            nn.Dropout2d(dropout_prob),
            ResidualBlock(128, 256),
            nn.Dropout2d(dropout_prob),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim)
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.projector(features)

class PrototypeClusterer(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.temperature = nn.Parameter(torch.tensor(0.1))  # Lower initial temperature
        self.scale = nn.Parameter(torch.tensor(10.0))  # Learnable scale factor

    def forward(self, features):
        # Cosine similarity with learnable scaling
        norm_features = F.normalize(features, dim=1)
        norm_prototypes = F.normalize(self.prototypes, dim=1)
        cos_sim = torch.matmul(norm_features, norm_prototypes.T)

        # Scaled and temperature-adjusted probabilities
        logits = self.scale * cos_sim / self.temperature.clamp(min=1e-6)
        return F.softmax(logits, dim=1)

# -------------------- Data Loading --------------------
class GalaxyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or self.get_default_transform()
        self.classes, self.label_encoder, self.samples = self.load_data()

    def get_default_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        if not class_dirs:
            raise ValueError(f"No class directories found in {self.root_dir}")

        label_encoder = LabelEncoder().fit([d.name for d in class_dirs])
        samples = []

        for class_dir in class_dirs:
            class_name = class_dir.name
            label = label_encoder.transform([class_name])[0]
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    samples.append((img_path, label))

        return [d.name for d in class_dirs], label_encoder, samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, str(img_path)

# -------------------- Fixed Config Manager --------------------
class ConfigManager:
    @staticmethod
    def generate_conf_file(output_path: Path, feature_dim: int, dataset_name: str):
        """Generate config file with string paths"""
        conf = {
            "file_path": str(output_path/f"{dataset_name}.csv"),
            "column_names": [f"feature_{i}" for i in range(feature_dim)] + [
                "target", "file_path", "assigned_class", "assigned_class_label", "confidence"
            ],
            "separator": ",",
            "has_header": True,
            "target_column": "target",
            "modelType": "PrototypeCluster",
            "feature_group_size": 2,
            "max_combinations": 10000,
            "bin_sizes": [128],
        }

        with open(output_path/f"{dataset_name}.conf", "w") as f:
            json.dump(conf, f, indent=4, default=str)  # Handle Path objects

    @staticmethod
    def generate_json_config(output_path: Path, config: Dict, dataset_name: str):
        """Generate JSON config with string paths"""
        base_config = {
            "dataset": {
                "name": dataset_name,
                "type": "image_folder",
                "num_classes": config["n_classes"],
                "input_size": [224, 224],
            },
            "model": {
                "feature_dims": config["feature_dim"],
                "learning_rate": config["lr"],
                "dropout_prob": config["dropout_prob"],
            },
            "training": {
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "best_loss": config.get("best_loss", float('inf')),
            },
            "paths": {
                "model_dir": str(output_path/"models"),
                "features_file": str(output_path/f"{dataset_name}.csv"),
            }
        }

        with open(output_path/f"{dataset_name}.json", "w") as f:
            json.dump(base_config, f, indent=4, default=str)

# -------------------- Enhanced Training Pipeline --------------------
class GalaxyClassifier:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = None
        self.model = None
        self.clusterer = None
        self.optimizer = None
        self.scheduler = None
        self.best_loss = float('inf')

    def _default_config(self):
        return {
            "feature_dim": 128,
            "lr": 0.001,
            "prototype_lr": 0.01,
            "dropout_prob": 0.3,
            "epochs": 50,
            "batch_size": 32,
            "patience": 5,
            "min_delta": 1e-4,
            "warmup_epochs": 5,
            "save_dir": "data",
            "label_smoothing": 0.1,
        }

    def train(self, data_path: str):
        """Enhanced training with warmup and label smoothing"""
        try:
            # Initialize data and model
            dataset = GalaxyDataset(data_path)
            self.label_encoder = dataset.label_encoder
            self.config["n_classes"] = len(dataset.classes)

            output_dir = Path(self.config["save_dir"]) / Path(data_path).name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize model
            self.model = DeepFeatureExtractor(
                (3, 224, 224),
                self.config["feature_dim"],
                self.config["dropout_prob"]
            ).to(self.device)

            self.clusterer = PrototypeClusterer(
                self.config["feature_dim"],
                self.config["n_classes"]
            ).to(self.device)

            # Optimizer and schedulers
            self.optimizer = torch.optim.AdamW([
                {'params': self.model.parameters()},
                {'params': self.clusterer.parameters(), 'lr': self.config["prototype_lr"]}
            ], lr=self.config["lr"], weight_decay=1e-4)

            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=self.config["warmup_epochs"]
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["epochs"] - self.config["warmup_epochs"]
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                [warmup, cosine],
                milestones=[self.config["warmup_epochs"]]
            )

            # Data loader
            loader = DataLoader(
                dataset,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            # Training loop
            for epoch in range(self.config["epochs"]):
                self.model.train()
                self.clusterer.train()
                epoch_loss = 0.0

                for images, labels, _ in tqdm(loader, desc=f"Epoch {epoch+1}"):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    # Label smoothing
                    targets = torch.full_like(
                        torch.zeros(images.size(0), self.config["n_classes"]),
                        self.config["label_smoothing"]/(self.config["n_classes"]-1)
                    ).to(self.device)
                    targets.scatter_(1, labels.unsqueeze(1), 1-self.config["label_smoothing"])

                    # Forward pass
                    features = self.model(images)
                    probs = self.clusterer(features)
                    loss = F.kl_div(torch.log(probs), targets, reduction='batchmean')

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                # Update learning rate
                self.scheduler.step()

                # Check for improvement
                avg_loss = epoch_loss / len(loader)
                if avg_loss < self.best_loss - self.config["min_delta"]:
                    self.best_loss = avg_loss
                    self._save_model(output_dir)
                    patience = 0
                else:
                    patience += 1

                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} (Best={self.best_loss:.4f}) LR={self.optimizer.param_groups[0]['lr']:.2e}")

                if patience >= self.config["patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Save final results
            self._save_features(output_dir, dataset)
            ConfigManager.generate_conf_file(output_dir, self.config["feature_dim"], Path(data_path).name)
            ConfigManager.generate_json_config(output_dir, self.config, Path(data_path).name)

        except Exception as e:
            print(f"Error: {str(e)}")
            raise

    def _save_model(self, output_dir: Path):
        """Save model with proper path handling"""
        model_dir = output_dir / "models"
        model_dir.mkdir(exist_ok=True)

        torch.save({
            'model_state': self.model.state_dict(),
            'clusterer_state': self.clusterer.state_dict(),
            'config': self.config,
            'classes': self.label_encoder.classes_.tolist()
        }, model_dir / "model.pth")

        with open(model_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

    def _save_features(self, output_dir: Path, dataset: GalaxyDataset):
        """Save features with proper path handling"""
        loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4
        )

        self.model.eval()
        self.clusterer.eval()

        features, labels, paths, assignments, confidences = [], [], [], [], []

        with torch.no_grad():
            for images, lbls, pths in loader:
                feats = self.model(images.to(self.device))
                probs = self.clusterer(feats)
                confs, preds = torch.max(probs, 1)

                features.append(feats.cpu())
                labels.append(lbls)
                paths.extend(pths)
                assignments.append(preds.cpu())
                confidences.append(confs.cpu())

        # Create DataFrame
        df = pd.DataFrame({
            **{f"feature_{i}": torch.cat(features)[:,i].numpy() for i in range(self.config["feature_dim"])},
            "target": torch.cat(labels).numpy(),
            "file_path": paths,
            "assigned_class": torch.cat(assignments).numpy(),
            "assigned_class_label": self.label_encoder.inverse_transform(torch.cat(assignments).numpy()),
            "confidence": torch.cat(confidences).numpy(),
            "class_name": self.label_encoder.inverse_transform(torch.cat(labels).numpy())
        })

        df.to_csv(output_dir / f"{Path(dataset.root_dir).name}.csv", index=False)

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    classifier = GalaxyClassifier()
    classifier.train("path/to/galaxies")
