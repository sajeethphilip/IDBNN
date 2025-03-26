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
from torch.optim.lr_scheduler import ReduceLROnPlateau
# First, add these imports at the top of your file
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor
from sklearn.cluster import KMeans
import numpy as np

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

class StabilizedDeepClustering(nn.Module):
    def __init__(self, input_dims, num_classes, feature_dim=128, warmup_epochs=10):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Feature extractor with projection head
        self.feature_extractor = DeepFeatureExtractor(input_dims, feature_dim)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Prototype clusterer with EMA stabilization
        self.clusterer = StabilizedPrototypeClusterer(feature_dim, num_classes)

        # Classification head for warmup phase
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Feature memory bank
        self.register_buffer('feature_bank', torch.zeros(num_classes, feature_dim))
        self.register_buffer('feature_counts', torch.zeros(num_classes))
        self.ema = EMA(decay=0.99)

    def forward(self, x):
        features = F.normalize(self.projector(self.feature_extractor(x)), dim=1)

        if self.training and self.current_epoch < self.warmup_epochs:
            return self.classifier(features), None
        else:
            assignments, probs, proto_reg = self.clusterer(features)
            return None, (assignments, probs, proto_reg)

    def update_feature_bank(self, features, labels):
        """Update feature direction memory bank"""
        with torch.no_grad():
            for cls_idx in torch.unique(labels):
                mask = (labels == cls_idx)
                if mask.any():
                    cls_features = features[mask]
                    self.feature_bank[cls_idx] = self.ema(
                        f'proto_{cls_idx}',
                        cls_features.mean(dim=0)
                    )
                    self.feature_counts[cls_idx] = mask.sum().item()

    def initialize_prototypes(self):
        """Initialize prototypes using feature bank statistics"""
        with torch.no_grad():
            valid_classes = self.feature_counts > 0
            if valid_classes.sum() == len(self.clusterer.prototypes):
                # Direct initialization if all classes represented
                self.clusterer.prototypes.data.copy_(
                    F.normalize(self.feature_bank[valid_classes], dim=1))
            else:
                # K-means fallback
                kmeans = KMeans(n_clusters=len(self.clusterer.prototypes))
                kmeans.fit(self.feature_bank.cpu().numpy())
                self.clusterer.prototypes.data.copy_(
                    torch.from_numpy(kmeans.cluster_centers_).to(self.feature_bank.device))

class StabilizedPrototypeClusterer(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.xavier_normal_(self.prototypes)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.reg_weight = 0.1
        self.epsilon = 1e-6

    def forward(self, features):
        # Normalized similarity
        norm_features = F.normalize(features, dim=1)
        norm_prototypes = F.normalize(self.prototypes, dim=1)
        sim = torch.matmul(norm_features, norm_prototypes.T)

        # Temperature-scaled probabilities
        temp = self.temperature.clamp(min=self.epsilon)
        probs = F.softmax(sim / temp, dim=1)

        # Prototype regularization
        proto_reg = self.prototype_regularization(norm_prototypes)

        return torch.argmax(probs, dim=1), probs, proto_reg

    def prototype_regularization(self, prototypes):
        """Diversity and uniformity regularization"""
        # Orthogonality constraint
        proto_sim = prototypes @ prototypes.T
        eye = torch.eye(len(prototypes), device=prototypes.device)
        ortho_loss = F.mse_loss(proto_sim, eye)

        # Uniform distribution constraint
        avg_sim = proto_sim.mean()
        uniform_loss = (avg_sim - 0.5)**2

        return self.reg_weight * (ortho_loss + uniform_loss)

class DeepClusteringPipeline:
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_loss = float('inf')

    def _default_config(self):
        return {
            "feature_dim": 128,
            "lr": 0.001,
            "prototype_lr": 0.01,
            "warmup_epochs": 10,
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
            self.config["n_classes"] = len(dataset.classes)

            # Initialize stabilized model
            self.model = StabilizedDeepClustering(
                input_dims=(3, 224, 224),
                num_classes=self.config["n_classes"],
                feature_dim=self.config["feature_dim"],
                warmup_epochs=self.config["warmup_epochs"]
            ).to(self.device)

            # Optimizer with separate learning rates
            self.optimizer = torch.optim.AdamW([
                {'params': self.model.feature_extractor.parameters()},
                {'params': self.model.projector.parameters()},
                {'params': self.model.classifier.parameters(), 'lr': self.config["lr"]},
                {'params': self.model.clusterer.parameters(), 'lr': self.config["prototype_lr"]}
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

            # Data loading with strong augmentation
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
            for epoch in range(self.config["epochs"]):
                self.model.current_epoch = epoch
                train_loss = self._train_epoch(loader, epoch)

                # Update learning rate
                self.scheduler.step()

                # Initialize prototypes after warmup
                if epoch == self.config["warmup_epochs"] - 1:
                    self.model.initialize_prototypes()
                    print("\nInitialized prototypes from feature bank")

                # Early stopping check
                if train_loss < self.best_loss - self.config["min_delta"]:
                    self.best_loss = train_loss
                    self._save_model(output_dir, best=True)

                if epoch - self.best_epoch > self.config["patience"]:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            # Final evaluation
            self._load_best_model(output_dir)
            self._save_features(output_dir, dataset)
            ConfigManager.generate_conf_file(output_dir, self.config["feature_dim"], self.dataset_name)

            print(f"\nTraining complete. Results saved to {output_dir}")

        except Exception as e:
            print(f"Training error: {str(e)}", file=sys.stderr)
            raise

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, (images, labels, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits, cluster_output = self.model(images)

            # Loss calculation
            if epoch < self.config["warmup_epochs"]:
                # Supervised warmup
                loss = F.cross_entropy(logits, labels)
            else:
                # Clustering phase
                assignments, probs, proto_reg = cluster_output
                cls_loss = F.cross_entropy(probs, labels)
                loss = cls_loss + proto_reg

                # Update feature bank
                with torch.no_grad():
                    features = self.model.feature_extractor(images)
                    self.model.update_feature_bank(features, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Logging
            if batch_idx % 50 == 0:
                print(f"\nBatch {batch_idx}: Loss={loss.item():.4f}")
                if epoch >= self.config["warmup_epochs"]:
                    print(f"Class distribution: {torch.softmax(probs,1).mean(0).detach().cpu().numpy()}")

        return total_loss / len(loader)

    def _save_features(self, output_dir, dataset):
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False)

        self.model.eval()
        all_features, all_labels, all_preds = [], [], []

        with torch.no_grad():
            for images, labels, paths in tqdm(loader, desc="Extracting features"):
                features = self.model.feature_extractor(images.to(self.device))
                _, probs, _ = self.model.clusterer(features)
                preds = torch.argmax(probs, dim=1)

                all_features.append(features.cpu())
                all_labels.append(labels)
                all_preds.append(preds.cpu())

        # Save results
        features = torch.cat(all_features).numpy()
        labels = torch.cat(all_labels).numpy()
        preds = torch.cat(all_preds).numpy()

        df = pd.DataFrame({
            **{f"feature_{i}": features[:,i] for i in range(features.shape[1])},
            "true_label": labels,
            "predicted_label": preds,
            "true_class": self.label_encoder.inverse_transform(labels),
            "predicted_class": self.label_encoder.inverse_transform(preds)
        })

        df.to_csv(output_dir / f"{self.dataset_name}_features.csv", index=False)

        # Generate confusion matrix
        self._generate_confusion_matrix(labels, preds, output_dir)

    def _save_model(self, output_dir: Path, best: bool = False):
        """Save model and label encoder"""
        model_dir = output_dir/"models"
        model_dir.mkdir(exist_ok=True)

        suffix = "_best" if best else ""
        torch.save({
            'model_state': self.model.state_dict(),
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
            self.model.load_state_dict(checkpoint['model_state'])
            self.clusterer.load_state_dict(checkpoint['clusterer_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.best_loss = checkpoint['loss']
            print("Loaded best model weights")


    def _generate_confusion_matrix(self, true_labels, predicted_labels, output_dir):
        """Generate and save a confusion matrix"""
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        accuracy = np.trace(cm) / np.sum(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Save the figure
        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        # Print colored confusion matrix to console
        print("\nConfusion Matrix:")
        print(f"Accuracy: {accuracy:.2%}\n")

        # Create a colored text version for console
        class_names = self.label_encoder.classes_
        max_len = max(len(name) for name in class_names)
        header = " " * (max_len + 2) + " ".join([f"{name:^{max_len}}" for name in class_names])
        print(header)

        for i, true_name in enumerate(class_names):
            row = f"{true_name:<{max_len}} "
            for j in range(len(class_names)):
                count = cm[i, j]
                color_code = 32 if i == j else 31  # Green for correct, red for incorrect
                row += f"\033[{color_code}m{count:^{max_len}}\033[0m "
            print(row)

# -------------------- Data Loading --------------------
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

        # Find all class directories
        self.class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        if not self.class_dirs:
            raise ValueError(f"No class directories found in {root_dir}")

        # Create label encoder
        self.classes = sorted([d.name for d in self.class_dirs])
        self.label_encoder = LabelEncoder().fit(self.classes)

        # Get all image paths with their labels
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

# -------------------- Configuration Management --------------------
# In the ConfigManager class, modify both methods to convert Path objects to strings:

class ConfigManager:
    @staticmethod
    def generate_conf_file(output_path: Path, feature_dim: int, dataset_name: str):
        conf = {
            "file_path": str(output_path/f"{dataset_name}.csv"),  # Convert to string
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
                "training_save_path": str(output_path/"training_data"),  # Convert to string
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
        # Create a deep copy of the config to avoid modifying the original
        config_copy = deepcopy(config)

        # Convert any Path objects in the config to strings
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
# -------------------- Enhanced Training Pipeline --------------------

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

        # Adaptive allocation
        self.register_buffer('usage_stats', torch.zeros(num_classes))

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

        # Track prototype usage
        self.usage_stats += probs.sum(dim=0).detach()

        return torch.argmax(probs, dim=1), probs, reg_loss

    def _update_ema(self, features: Tensor, labels: Tensor):
        """Update EMA prototypes using ground truth labels"""
        with torch.no_grad():
            unique_labels = torch.unique(labels)
            for cls_idx in unique_labels:
                mask = labels == cls_idx
                cls_features = features[mask]

                if cls_features.size(0) > 0:
                    cls_mean = cls_features.mean(dim=0)

                    # Update EMA
                    if self.ema_counts[cls_idx] == 0:
                        self.ema_prototypes[cls_idx] = cls_mean
                    else:
                        self.ema_prototypes[cls_idx] = (
                            self.ema_decay * self.ema_prototypes[cls_idx] +
                            (1 - self.ema_decay) * cls_mean
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

        # Usage loss (encourage balanced prototype usage)
        usage = self.usage_stats / (self.usage_stats.sum() + 1e-6)
        usage_loss = (usage - 1/self.num_classes).pow(2).sum()

        return self.reg_weight * (div_loss + uniform_loss + usage_loss)

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
        self.model = None
        self.clusterer = None
        self.optimizer = None

    def _default_config(self):
        return {
            "feature_dim": 256,
            "num_classes": 10,
            "warmup_epochs": 10,
            "lr": 0.001,
            "proto_lr": 0.01,
            "temp_init": 0.1,
            "reg_weight": 0.1,
            "ema_decay": 0.99,
            "epochs": 50,
            "batch_size": 256
        }

    def initialize_model(self):
        """Initialize the complete model architecture"""
        self.feature_extractor = DeepFeatureExtractor(
            input_dims=(3, 224, 224),
            feature_dim=self.config["feature_dim"]
        ).to(self.device)

        self.clusterer = EnhancedPrototypeClusterer(
            feature_dim=self.config["feature_dim"],
            num_classes=self.config["num_classes"],
            initial_temp=self.config["temp_init"],
            reg_weight=self.config["reg_weight"],
            ema_decay=self.config["ema_decay"]
        ).to(self.device)

        # Separate optimizers for different components
        self.optimizer = torch.optim.AdamW([
            {'params': self.feature_extractor.parameters()},
            {'params': self.clusterer.parameters(), 'lr': self.config["proto_lr"]}
        ], lr=self.config["lr"])

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["epochs"],
            eta_min=1e-6
        )

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, (images, labels, _) in enumerate(loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            features = self.feature_extractor(images)
            assignments, probs, reg_loss = self.clusterer(features, labels)

            # Loss calculation
            cls_loss = F.cross_entropy(probs, labels)
            loss = cls_loss + reg_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Periodic prototype adaptation
            if batch_idx % 100 == 0 and epoch >= self.config["warmup_epochs"]:
                self.clusterer.adapt_prototypes(features.detach())

        # Update learning rate
        self.scheduler.step()

        return total_loss / len(loader)

    def predict(self, loader):
        self.model.eval()
        all_features, all_assignments = [], []

        with torch.no_grad():
            for images, _, _ in loader:
                features = self.feature_extractor(images.to(self.device))
                assignments, _, _ = self.clusterer(features)

                all_features.append(features.cpu())
                all_assignments.append(assignments.cpu())

        return (
            torch.cat(all_features).numpy(),
            torch.cat(all_assignments).numpy()
        )


# -------------------- Command Line Interface --------------------
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
        """Interactive configuration"""
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

        # Check for existing model
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
        """Main application entry point"""
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

        # Initialize pipeline with enhanced configuration
        config = {
            "save_dir": Path(output_dir).parent if output_dir.endswith(Path(data_path).name) else output_dir,
            "feature_dim": 128,
            "lr": 0.001,
            "prototype_lr": 0.01,
            "dropout_prob": 0.3,
            "epochs": 50,
            "batch_size": 32,
            "patience": 5,
            "min_delta": 0.001
        }
        self.pipeline = DeepClusteringPipeline(config)

        # Run pipeline
        try:
            if mode in ("train", "auto"):
                print(f"\nStarting training on {data_path}...")
                self.pipeline.train(data_path)
            else:
                print("\nPrediction mode not yet implemented")
        except Exception as e:
            print(f"\nError: {str(e)}", file=sys.stderr)
            sys.exit(1)
# -------------------- Enhanced Model Architecture --------------------
class DeepFeatureExtractor(nn.Module):
    """7-layer CNN with dropout for better feature extraction"""
    def __init__(self, input_dims: Tuple, feature_dim: int = 128, dropout_prob: float = 0.3):
        super().__init__()
        self.input_dims = input_dims

        if len(input_dims) == 1:  # 1D data
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_prob),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(256, feature_dim))
        else:  # Image data
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dims[0], 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(dropout_prob),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(dropout_prob),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, feature_dim))

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2 and len(self.input_dims) == 1:
            x = x.unsqueeze(1)
        features = self.encoder(x)
        return self.projector(features)


if __name__ == "__main__":
    DeepClusteringApp().run()
