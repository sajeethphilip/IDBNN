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
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class EMA:
    """Exponential Moving Average for prototype stabilization"""
    def __init__(self, decay=0.99):
        self.decay = decay
        self.shadow = {}

    def __call__(self, name, x):
        if name not in self.shadow:
            self.shadow[name] = x.clone()
        else:
            self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * x
        return self.shadow[name]

class EnhancedPrototypeClusterer(nn.Module):
    """Upgraded prototype clustering with:
    - Learnable prototype temperature
    - Prototype diversity regularization
    - EMA prototype stabilization
    - Adaptive prototype allocation
    """
    def __init__(self, feature_dim: int, num_classes: int,
                 initial_temp: float = 0.1,
                 reg_weight: float = 0.1,
                 ema_decay: float = 0.99):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.ema_decay = ema_decay

        # Prototypes with Xavier initialization
        self.prototypes = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.xavier_normal_(self.prototypes)

        # Learnable temperature with softplus activation
        self.temperature = nn.Parameter(torch.tensor(initial_temp))
        self.temp_min = 0.01

        # EMA stabilization
        self.register_buffer('ema_prototypes', torch.zeros_like(self.prototypes))
        self.register_buffer('ema_counts', torch.zeros(num_classes))
        self.ema = EMA(decay=ema_decay)

    def forward(self, features: Tensor, labels: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns:
        - assignments: cluster assignments
        - probs: soft assignment probabilities
        - reg_loss: regularization loss term
        """
        # Normalize features and prototypes
        norm_features = F.normalize(features, dim=1)
        norm_prototypes = F.normalize(self.prototypes, dim=1)

        # Calculate similarity
        sim = torch.matmul(norm_features, norm_prototypes.T)

        # Temperature scaling with minimum threshold
        temp = F.softplus(self.temperature) + self.temp_min
        probs = F.softmax(sim / temp, dim=1)

        # Update EMA statistics if labels are provided
        if labels is not None and self.training:
            self._update_ema(norm_features.detach(), labels)

        # Calculate regularization losses
        reg_loss = self._calculate_regularization(norm_prototypes)

        return torch.argmax(probs, dim=1), probs, reg_loss

    def _update_ema(self, features: Tensor, labels: Tensor):
        """Update EMA prototypes using ground truth labels"""
        with torch.no_grad():
            unique_labels = torch.unique(labels)
            for cls_idx in unique_labels:
                mask = labels == cls_idx
                cls_features = features[mask]

                if cls_features.size(0) > 0:
                    self.ema_prototypes[cls_idx] = self.ema(
                        f'proto_{cls_idx}',
                        cls_features.mean(dim=0)
                    )
                    self.ema_counts[cls_idx] += 1

    def _calculate_regularization(self, prototypes: Tensor) -> Tensor:
        """Calculate prototype regularization terms"""
        # Diversity loss (encourage orthogonal prototypes)
        proto_sim = prototypes @ prototypes.T
        eye = torch.eye(self.num_classes, device=prototypes.device)
        div_loss = F.mse_loss(proto_sim, eye)

        # Uniformity loss (prevent degenerate solutions)
        avg_sim = proto_sim.mean()
        uniform_loss = (avg_sim - 0.5)**2

        return self.reg_weight * (div_loss + uniform_loss)

    def adapt_prototypes(self, features: Tensor):
        """Adapt prototypes based on current feature distribution"""
        with torch.no_grad():
            # Use EMA prototypes if available
            if self.ema_counts.sum() > 0:
                valid = self.ema_counts > 0
                self.prototypes.data[valid] = self.ema_prototypes[valid]

                # Use k-means for unused prototypes
                if not valid.all():
                    unused = ~valid
                    k = unused.sum().item()
                    if k > 0 and len(features) >= k:
                        km = KMeans(n_clusters=k)
                        km.fit(features.cpu().numpy())
                        self.prototypes.data[unused] = (
                            torch.from_numpy(km.cluster_centers_)
                            .to(features.device)
                        )
            else:
                # Fallback to k-means initialization
                km = KMeans(n_clusters=self.num_classes)
                km.fit(features.cpu().numpy())
                self.prototypes.data.copy_(
                    torch.from_numpy(km.cluster_centers_)
                    .to(features.device)
                )

class DeepClusteringPipeline:
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = None
        self.feature_extractor = None
        self.clusterer = None
        self.optimizer = None
        self.scheduler = None
        self.best_loss = float('inf')
        self.patience_counter = 0

    def _default_config(self):
        return {
            "feature_dim": 128,
            "lr": 0.001,
            "proto_lr": 0.01,
            "warmup_epochs": 10,
            "temp_init": 0.1,
            "reg_weight": 0.1,
            "ema_decay": 0.99,
            "dropout_prob": 0.3,
            "epochs": 50,
            "batch_size": 32,
            "patience": 5,
            "min_delta": 0.001,
            "save_dir": "results"
        }

    def initialize_from_data(self, data_path):
        data_path = Path(data_path)
        self.dataset_name = data_path.name

        try:
            dataset = ImageDataset(data_path)
            self.label_encoder = dataset.label_encoder
            self.config["num_classes"] = len(dataset.classes)

            # Initialize models
            self.feature_extractor = DeepFeatureExtractor(
                input_dims=(3, 224, 224),
                feature_dim=self.config["feature_dim"],
                dropout_prob=self.config["dropout_prob"]
            ).to(self.device)

            self.clusterer = EnhancedPrototypeClusterer(
                feature_dim=self.config["feature_dim"],
                num_classes=self.config["num_classes"],
                initial_temp=self.config["temp_init"],
                reg_weight=self.config["reg_weight"],
                ema_decay=self.config["ema_decay"]
            ).to(self.device)

            # Optimizer with separate learning rates
            self.optimizer = torch.optim.AdamW([
                {'params': self.feature_extractor.parameters()},
                {'params': self.clusterer.parameters(), 'lr': self.config["proto_lr"]}
            ], lr=self.config["lr"], weight_decay=1e-4)

            # Cosine annealing scheduler
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["epochs"],
                eta_min=1e-6
            )

        except Exception as e:
            raise ValueError(f"Initialization failed: {str(e)}")

    def train(self, data_path):
        try:
            self.initialize_from_data(data_path)
            output_dir = Path(self.config["save_dir"]) / self.dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Data loading with augmentation
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            dataset = ImageDataset(data_path, transform)
            loader = DataLoader(dataset, batch_size=self.config["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True)

            # Training loop
            best_epoch = 0
            for epoch in range(self.config["epochs"]):
                train_loss = self._train_epoch(loader, epoch)

                # Update learning rate
                self.scheduler.step()

                # Initialize prototypes after warmup
                if epoch == self.config["warmup_epochs"] - 1:
                    with torch.no_grad():
                        features = []
                        for images, _, _ in loader:
                            features.append(self.feature_extractor(images.to(self.device)))
                        all_features = torch.cat(features)
                        self.clusterer.adapt_prototypes(all_features)
                        print("\nInitialized prototypes from feature distribution")

                # Early stopping check
                if train_loss < self.best_loss - self.config["min_delta"]:
                    self.best_loss = train_loss
                    best_epoch = epoch
                    self._save_model(output_dir, best=True)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                print(f"Epoch {epoch+1}/{self.config['epochs']} - Loss: {train_loss:.4f} (Best: {self.best_loss:.4f})")

                if self.patience_counter >= self.config["patience"]:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            # Final evaluation
            self._load_best_model(output_dir)
            self._save_features(output_dir, dataset)
            ConfigManager.generate_conf_file(output_dir, self.config["feature_dim"], self.dataset_name)

            print(f"\nTraining complete. Best model saved from epoch {best_epoch+1}")
            print(f"Results saved to: {output_dir}")

        except Exception as e:
            print(f"Training error: {str(e)}", file=sys.stderr)
            raise

    def _train_epoch(self, loader, epoch):
        self.feature_extractor.train()
        self.clusterer.train()
        total_loss = 0

        for batch_idx, (images, labels, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            features = self.feature_extractor(images)
            assignments, probs, reg_loss = self.clusterer(features, labels)

            # Loss calculation
            cls_loss = F.cross_entropy(probs, labels)
            loss = cls_loss + reg_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Periodic prototype adaptation
            if batch_idx % 100 == 0 and epoch >= self.config["warmup_epochs"]:
                self.clusterer.adapt_prototypes(features.detach())

        return total_loss / len(loader)

    def _save_features(self, output_dir, dataset):
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False)

        self.feature_extractor.eval()
        self.clusterer.eval()

        all_features = []
        all_true_labels = []
        all_pred_labels = []
        all_paths = []
        all_confidences = []

        with torch.no_grad():
            for images, labels, paths in tqdm(loader, desc="Extracting features"):
                features = self.feature_extractor(images.to(self.device))
                preds, probs, _ = self.clusterer(features)
                confidences = probs.gather(1, preds.unsqueeze(1)).squeeze()

                all_features.append(features.cpu())
                all_true_labels.append(labels)
                all_pred_labels.append(preds.cpu())
                all_paths.extend(paths)
                all_confidences.append(confidences.cpu())

        # Save results
        features = torch.cat(all_features).numpy()
        true_labels = torch.cat(all_true_labels).numpy()
        pred_labels = torch.cat(all_pred_labels).numpy()
        confidences = torch.cat(all_confidences).numpy()

        df = pd.DataFrame({
            **{f"feature_{i}": features[:,i] for i in range(features.shape[1])},
            "true_label": true_labels,
            "predicted_label": pred_labels,
            "confidence": confidences,
            "file_path": all_paths,
            "true_class": self.label_encoder.inverse_transform(true_labels),
            "predicted_class": self.label_encoder.inverse_transform(pred_labels)
        })

        df.to_csv(output_dir / f"{self.dataset_name}_features.csv", index=False)

        # Generate confusion matrix
        self._generate_confusion_matrix(true_labels, pred_labels, output_dir)

    def _save_model(self, output_dir: Path, best: bool = False):
        """Save model and label encoder"""
        model_dir = output_dir/"models"
        model_dir.mkdir(exist_ok=True)

        suffix = "_best" if best else ""
        torch.save({
            'feature_extractor_state': self.feature_extractor.state_dict(),
            'clusterer_state': self.clusterer.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'classes': self.label_encoder.classes_.tolist(),
            'loss': self.best_loss
        }, model_dir/f"model{suffix}.pth")

        with open(model_dir/f"label_encoder{suffix}.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

    def _load_best_model(self, output_dir: Path):
        """Load the best saved model"""
        model_path = output_dir/"models"/"model_best.pth"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state'])
            self.clusterer.load_state_dict(checkpoint['clusterer_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.best_loss = checkpoint['loss']
            print("Loaded best model weights")

    def _generate_confusion_matrix(self, true_labels, pred_labels, output_dir):
        """Generate and save confusion matrix"""
        cm = confusion_matrix(true_labels, pred_labels)
        accuracy = np.trace(cm) / np.sum(cm)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()

        print(f"\nValidation Accuracy: {accuracy:.2%}")

class DeepFeatureExtractor(nn.Module):
    """Enhanced 7-layer CNN feature extractor with:
    - Adaptive architecture for 1D/2D inputs
    - Skip connections for better gradient flow
    - Configurable dropout
    - Projection head for feature refinement
    """
    def __init__(self, input_dims: Tuple, feature_dim: int = 128, dropout_prob: float = 0.3):
        super().__init__()
        self.input_dims = input_dims
        self.feature_dim = feature_dim
        self.dropout_prob = dropout_prob

        if len(input_dims) == 1:  # 1D data (e.g., time series)
            self.encoder = nn.Sequential(
                # Block 1
                nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout_prob),

                # Block 2 with skip connection
                ResidualBlock1D(32, 64, stride=1),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_prob),

                # Block 3
                ResidualBlock1D(64, 128, stride=1),
                nn.Dropout(dropout_prob),

                # Block 4
                ResidualBlock1D(128, 256, stride=1),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
        else:  # 2D image data
            self.encoder = nn.Sequential(
                # Block 1
                nn.Conv2d(input_dims[0], 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(dropout_prob),

                # Block 2 with skip connection
                ResidualBlock2D(32, 64, stride=1),
                nn.MaxPool2d(2),
                nn.Dropout(dropout_prob),

                # Block 3
                ResidualBlock2D(64, 128, stride=1),
                nn.Dropout(dropout_prob),

                # Block 4
                ResidualBlock2D(128, 256, stride=1),
                nn.MaxPool2d(2),
                nn.Dropout(dropout_prob),

                # Block 5
                ResidualBlock2D(256, 512, stride=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )

        # Projection head with LayerNorm
        self.projector = nn.Sequential(
            nn.Linear(512 if len(input_dims) > 1 else 256, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(feature_dim, feature_dim)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 1D input case
        if x.ndim == 2 and len(self.input_dims) == 1:
            x = x.unsqueeze(1)  # Add channel dimension

        features = self.encoder(x)
        return self.projector(features)


class ResidualBlock1D(nn.Module):
    """Residual block for 1D data"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class ResidualBlock2D(nn.Module):
    """Residual block for 2D data"""
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
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

# -------------------- Data Loading (unchanged from original) --------------------
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        if not self.class_dirs:
            raise ValueError(f"No class directories found in {root_dir}")

        self.classes = sorted([d.name for d in self.class_dirs])
        self.label_encoder = LabelEncoder().fit(self.classes)
        self.samples = []

        for class_dir in self.class_dirs:
            class_name = class_dir.name
            label = self.label_encoder.transform([class_name])[0]
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, str(img_path)

# -------------------- Configuration Management (unchanged) --------------------
class ConfigManager:
    @staticmethod
    def generate_conf_file(output_path: Path, feature_dim: int, dataset_name: str):
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
            "active_learning": {
                "tolerance": 1.0,
                "cardinality_threshold_percentile": 95,
                "strong_margin_threshold": 0.3,
                "marginal_margin_threshold": 0.1,
                "min_divergence": 0.1
            },
            "training_params": {
                "trials": 100,
                "epochs": 1000,
                "learning_rate": 0.001,
                "batch_size": 128,
                "test_fraction": 0.2,
                "random_seed": 42,
                "minimum_training_accuracy": 0.95,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "n_bins_per_dim": 21,
                "enable_adaptive": True,
                "invert_DBNN": True,
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.001,
                "Save_training_epochs": True,
                "training_save_path": str(output_path/"training_data"),
                "enable_vectorized": False,
                "vectorization_warning_acknowledged": False,
                "compute_device": "auto",
                "use_interactive_kbd": False
            },
            "execution_flags": {
                "train": True,
                "train_only": False,
                "predict": True,
                "fresh_start": False,
                "use_previous_model": True,
                "gen_samples": False
            }
        }
        with open(output_path/f"{dataset_name}.conf", "w") as f:
            json.dump(conf, f, indent=4)

    @staticmethod
    def generate_json_config(output_path: Path, config: Dict, dataset_name: str):
        config_copy = deepcopy(config)

        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_paths(x) for x in obj]
            return obj

        config_copy = convert_paths(config_copy)

        base_config = {
            "dataset": {
                "name": dataset_name,
                "type": "image_folder",
                "in_channels": 3,
                "num_classes": config_copy["n_classes"],
                "input_size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "train_dir": str(output_path),
                "test_dir": None
            },
            "model": {
                "encoder_type": "deep_cnn",
                "feature_dims": config_copy["feature_dim"],
                "learning_rate": config_copy["lr"],
                "prototype_learning_rate": config_copy["prototype_lr"],
                "temperature": 0.5,
                "dropout_prob": config_copy.get("dropout_prob", 0.3),
                "enhancement_modules": {
                    "prototype_clustering": {
                        "enabled": True,
                        "weights": {
                            "cluster_loss": 1.0,
                            "uniformity": 0.1
                        }
                    }
                }
            },
            "training": config_copy,
            "output": {
                "features_file": str(output_path/f"{dataset_name}.csv"),
                "model_dir": str(output_path/"models"),
                "visualization_dir": str(output_path/"visualizations")
            }
        }
        with open(output_path/f"{dataset_name}.json", "w") as f:
            json.dump(base_config, f, indent=4)

# -------------------- Command Line Interface (unchanged) --------------------
class DeepClusteringApp:
    def __init__(self):
        self.parser = self._create_parser()
        self.pipeline = None

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Deep Clustering Pipeline",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--path", help="Path to data directory")
        parser.add_argument("--mode", choices=["train","predict","auto"], default="auto",
                          help="Operation mode")
        parser.add_argument("--output", help="Custom output directory")
        parser.add_argument("--force", action="store_true", help="Overwrite existing files")
        return parser

    def _interactive_setup(self) -> Dict:
        print("\n==== Deep Clustering Setup ====")
        params = {}

        while True:
            path = input("Path to data directory: ").strip()
            if Path(path).exists():
                params["path"] = path
                break
            print("Error: Path does not exist")

        params["name"] = Path(path).name
        params["output"] = f"data/{params['name']}"

        model_path = Path(params["output"])/"models"/"model.pth"
        params["mode"] = "predict" if model_path.exists() else "train"

        if model_path.exists():
            choice = input(f"Found existing model at {model_path}. Train anyway? [y/N]: ").lower()
            if choice == 'y':
                params["mode"] = "train"
            else:
                print("Using existing model for prediction")

        return params

    def run(self):
        args = self.parser.parse_args()

        if not any(vars(args).values()):  # Interactive mode
            params = self._interactive_setup()
            data_path = params["path"]
            mode = params["mode"]
            output_dir = params["output"]
            force = True
        else:  # Command-line mode
            data_path = args.path
            mode = args.mode
            output_dir = args.output or f"data/{Path(data_path).name}"
            force = args.force

        config = {
            "save_dir": Path(output_dir).parent if output_dir.endswith(Path(data_path).name) else output_dir,
            "feature_dim": 128,
            "lr": 0.001,
            "proto_lr": 0.01,
            "warmup_epochs": 10,
            "temp_init": 0.1,
            "reg_weight": 0.1,
            "ema_decay": 0.99,
            "dropout_prob": 0.3,
            "epochs": 50,
            "batch_size": 32,
            "patience": 5,
            "min_delta": 0.001
        }

        self.pipeline = DeepClusteringPipeline(config)

        try:
            if mode in ("train", "auto"):
                print(f"\nStarting training on {data_path}...")
                self.pipeline.train(data_path)
            else:
                print("\nPrediction mode not yet implemented")
        except Exception as e:
            print(f"\nError: {str(e)}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    DeepClusteringApp().run()
