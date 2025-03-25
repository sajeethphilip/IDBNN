import json
import os
import argparse
import shutil
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import warnings

class SelfAttention(nn.Module):
    """Self-attention module for feature maps"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()

        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return self.gamma * out + x

class FeatureExtractorCNN(nn.Module):
    """7-layer CNN feature extractor with self-attention"""
    def __init__(self, in_channels: int = 3, feature_dims: int = 128, dropout_prob: float = 0.5):
        super().__init__()
        self.dropout_prob = dropout_prob

        # Layer 1-3
        self.conv1 = self._conv_block(in_channels, 32)
        self.conv2 = self._conv_block(32, 64)
        self.conv3 = self._conv_block(64, 128)
        self.attention1 = SelfAttention(128)

        # Layer 4-6
        self.conv4 = self._conv_block(128, 256)
        self.conv5 = self._conv_block(256, 512)
        self.conv6 = self._conv_block(512, 512)
        self.attention2 = SelfAttention(512)

        # Layer 7
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Output
        self.fc = nn.Linear(512, feature_dims)
        self.batch_norm = nn.BatchNorm1d(feature_dims)

    def _conv_block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_prob)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention1(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.attention2(x)

        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if x.size(0) > 1:
            x = self.batch_norm(x)

        return x

class KLDivergenceClusterer:
    """Handles KL-divergence based clustering"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ae_config = config["model"]["CNN_config"]["enhancements"]
        self.temperature = ae_config["clustering_temperature"]
        self.min_confidence = ae_config["min_cluster_confidence"]
        self.num_classes = config["dataset"]["num_classes"]

    def compute_kl_divergence(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between features and target distributions"""
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=self.num_classes).float()

        # Normalize features to probability distribution
        features_dist = F.softmax(features / self.temperature, dim=1)
        targets_dist = F.softmax(targets / self.temperature, dim=1)

        # Ensure targets match feature dimensions
        if features.size(1) != targets.size(1):
            # If feature dim > num_classes, pad targets with zeros
            if features.size(1) > targets.size(1):
                padding = torch.zeros(targets.size(0), features.size(1) - targets.size(1)).to(targets.device)
                targets = torch.cat([targets, padding], dim=1)
            # If feature dim < num_classes, truncate targets
            else:
                targets = targets[:, :features.size(1)]

        # Normalize targets
        targets_dist = F.softmax(targets / self.temperature, dim=1)

        # Compute KL divergence
        return F.kl_div(
            features_dist.log(),
            targets_dist,
            reduction='batchmean',
            log_target=False
        )


    def cluster_features(self, features: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cluster features using KL divergence and target information"""
        # Compute cluster probabilities
        similarities = torch.matmul(features, features.T)
        cluster_probs = F.softmax(similarities / self.temperature, dim=1)

        # Filter by confidence
        max_probs = cluster_probs.max(dim=1)[0]
        confident_mask = max_probs > self.min_confidence
        confident_features = features[confident_mask]
        confident_targets = targets[confident_mask]

        return confident_features, confident_targets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Dataset that preserves image paths"""
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)

class FeatureExtractorPipeline:
    """Complete feature extraction pipeline"""
    def __init__(self, source_dir: str, datafolder: str, merge_train_test: bool = False, interactive: bool = False):
        """
        Args:
            source_dir: Path to source directory containing the original data
            datafolder: Path to destination directory (under data/)
            merge_train_test: Whether to merge train/test sets
            interactive: Whether to prompt for user input
        """
        self.source_dir = os.path.abspath(source_dir)
        self.datafolder = os.path.abspath(datafolder)
        self.dataset_name = os.path.basename(datafolder)
        self.interactive = interactive
        self.merge_train_test = merge_train_test
        self.class_to_idx = None
        self.num_classes = None

        # Setup paths
        self.config_path = os.path.join(datafolder, f"{self.dataset_name}.json")
        self.conf_path = os.path.join(datafolder, f"{self.dataset_name}.conf")
        self.csv_path = os.path.join(datafolder, f"{self.dataset_name}.csv")
        self.model_dir = os.path.join(datafolder, "models")
        self.train_dir = os.path.join(datafolder, "train")
        self.test_dir = os.path.join(datafolder, "test")

        # Initialize
        self.config = self._initialize_config()
        self.conf = self._initialize_conf()
        self.device = self._setup_device()
        self.model = self._initialize_model()
        self.optimizer = self._configure_optimizer()
        self.scheduler = self._configure_scheduler()
        self.clusterer = KLDivergenceClusterer(self.config)
        self.transform = self._configure_transforms()

        # Prepare data structure
        self._prepare_data_structure()

    def _initialize_config(self) -> Dict[str, Any]:
        """Initialize config file in data/<datafolder>/"""
        config_path = os.path.join(self.datafolder, f"{self.dataset_name}.json")

        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_path}")
        else:
            config = self._create_default_config()
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Created default configuration at {config_path}")

        return config

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            "dataset": {
                "name": self.dataset_name,
                "type": "image_folder",
                "in_channels": 3,
                "num_classes": 10,
                "input_size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize_images": True,
                "train_dir": os.path.join(self.datafolder, "train"),
                "test_dir": os.path.join(self.datafolder, "train", "test") if not self.merge_train_test else None,
            },
            "model": {
                "encoder_type": "cnn",
                "enable_adaptive": True,
                "feature_dims": 128,
                "learning_rate": 0.001,
                "dropout_prob": 0.5,
                "optimizer": {
                    "type": "Adam",
                    "weight_decay": 0.0001,
                    "momentum": 0.9,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-08
                },
                "scheduler": {
                    "type": "ReduceLROnPlateau",
                    "factor": 0.1,
                    "patience": 10,
                    "min_lr": 1e-06,
                    "verbose": True
                },
                "CNN_config": {
                    "reconstruction_weight": 1.0,
                    "feature_weight": 0.1,
                    "convergence_threshold": 0.001,
                    "min_epochs": 10,
                    "patience": 5,
                    "enhancements": {
                        "enabled": True,
                        "use_kl_divergence": True,
                        "use_class_encoding": False,
                        "kl_divergence_weight": 0.5,
                        "classification_weight": 0.5,
                        "clustering_temperature": 1.0,
                        "min_cluster_confidence": 0.7
                    }
                }
            },
            "training": {
                "batch_size": 128,
                "epochs": 20,
                "num_workers": 4,
                "validation_split": 0.2,
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                }
            },
            "augmentation": {
                "enabled": True,
                "random_crop": {
                    "enabled": True,
                    "padding": 4
                },
                "random_rotation": {
                    "enabled": True,
                    "degrees": 10
                },
                "horizontal_flip": {
                    "enabled": True,
                    "probability": 0.5
                },
                "color_jitter": {
                    "enabled": True,
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.1
                },
                "normalize": {
                    "enabled": True,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            },
            "execution_flags": {
                "mode": "train_and_predict",
                "use_gpu": True,
                "mixed_precision": True,
                "debug_mode": False,
                "use_previous_model": True,
                "fresh_start": False
            },
            "output": {
            "features_file": os.path.join(self.datafolder, f"{self.dataset_name}.csv"),
            "model_dir": self.model_dir,
                "visualization_dir": os.path.join(self.datafolder, "visualizations")
            }
        }

    def _initialize_conf(self) -> Dict[str, Any]:
        """Initialize .conf file in data/<datafolder>/"""
        conf_path = os.path.join(self.datafolder, f"{self.dataset_name}.conf")

        if os.path.exists(conf_path):
            with open(conf_path) as f:
                conf = json.load(f)
            print(f"Loaded conf from {conf_path}")
        else:
            conf = self._create_default_conf()
            with open(conf_path, 'w') as f:
                json.dump(conf, f, indent=4)
            print(f"Created default conf at {conf_path}")

        return conf

    def _create_default_conf(self) -> Dict[str, Any]:
        """Create default .conf file"""
        feature_columns = [f"feature_{i}" for i in range(128)] + ["target"]

        return {
            "file_path": self.csv_path,
            "column_names": feature_columns,
            "separator": ",",
            "has_header": True,
            "target_column": "target",
            "modelType": "Histogram",
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
                "training_save_path": "training_data",
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


    def _handle_interactive_setup(self) -> None:
        """Handle interactive setup"""
        print(f"\nSetting up data structure for {self.dataset_name}")

        # Ask about train/test merge
        if not os.path.exists(self.test_dir):
            self.merge_train_test = True
        else:
            response = input("Do you want to merge train and test sets? (y/n): ").lower()
            self.merge_train_test = response == 'y'

        if self.merge_train_test:
            self._merge_train_test_sets()

        # Verify structure
        if not os.listdir(self.train_dir):
            print(f"Warning: No images found in {self.train_dir}")
            print("Please copy your images into class subfolders in this directory.")

    def _find_data_folders(self, start_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Recursively search for train/test folders or image subfolders"""
        train_path, test_path = None, None
        image_subfolders = []

        for root, dirs, files in os.walk(start_path):
            # Check if current directory is train or test
            if os.path.basename(root) == "train":
                train_path = root
            elif os.path.basename(root) == "test":
                test_path = root

            # Check for image subfolders (potential classes)
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                if root != start_path:  # Don't count root directory
                    image_subfolders.append(root)

        return train_path, test_path, image_subfolders

    def _prepare_data_structure(self) -> None:
        """Ensure proper training directory structure exists with class folders"""
        # First verify the source directory contains valid data
        if not self._has_valid_source_structure():
            raise FileNotFoundError(
                f"No valid training data found in {self.source_dir}\n"
                "Expected either:\n"
                "1. Class folders directly in this directory\n"
                "2. 'train/' and optionally 'test/' subdirectories"
            )

        # Handle existing training directory
        if os.path.exists(self.train_dir):
            if self.interactive:
                print(f"Warning: Training directory already exists at {self.train_dir}")
                response = input("Do you want to: [R]eplace, [M]erge, or [A]bort? ").lower()
                if response == 'a':
                    raise FileExistsError("Aborted by user")
                elif response == 'r':
                    print(f"Removing existing directory {self.train_dir}")
                    shutil.rmtree(self.train_dir)
                    os.makedirs(self.train_dir)
                elif response == 'm':
                    print("Merging with existing data")
                else:
                    print("Invalid choice, defaulting to merge")
            else:
                # Non-interactive mode - default to merge
                print(f"Training directory exists, merging new data into {self.train_dir}")

        # Now ensure we have class folders in the training directory
        self._count_actual_classes()
        self._ensure_class_folders_exist()

    def _count_actual_classes(self) -> None:
        """Count actual class folders with images in training directory"""
        class_folders = []
        for f in os.listdir(self.train_dir):
            folder_path = os.path.join(self.train_dir, f)
            if os.path.isdir(folder_path):
                # Check if folder contains images
                if any(fname.lower().endswith(('.png', '.jpg', '.jpeg'))
                   for fname in os.listdir(folder_path)):
                    class_folders.append(f)

        if not class_folders:
            raise FileNotFoundError(
                f"No valid class folders found in {self.train_dir}\n"
                "Please ensure the directory contains subfolders with images"
            )

        # Create class mapping and count
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(class_folders))}
        self.num_classes = len(self.class_to_idx)

        print(f"Found {self.num_classes} classes: {self.class_to_idx}")

    def _has_valid_source_structure(self) -> bool:
        """Check if source directory has valid structure"""
        # Check for direct class folders
        if any(os.path.isdir(os.path.join(self.source_dir, f)) and
               any(fname.lower().endswith(('.png', '.jpg', '.jpeg'))
                   for fname in os.listdir(os.path.join(self.source_dir, f)))
               for f in os.listdir(self.source_dir)):
            return True

        # Check for train/test structure
        train_path = os.path.join(self.source_dir, "train")
        if os.path.exists(train_path) and any(os.path.isdir(os.path.join(train_path, f))
                                             for f in os.listdir(train_path)):
            return True

        return False

    def _ensure_class_folders_exist(self) -> None:
        """Copy class folders from source to training directory if needed"""
        # Check if training directory already has class folders
        if any(os.path.isdir(os.path.join(self.train_dir, f))
               for f in os.listdir(self.train_dir)):
            print("Training directory already contains class folders")
            return

        print("Copying class folders to training directory...")

        # Case 1: Source has direct class folders
        class_folders = [f for f in os.listdir(self.source_dir)
                        if os.path.isdir(os.path.join(self.source_dir, f)) and
                        any(fname.lower().endswith(('.png', '.jpg', '.jpeg'))
                            for fname in os.listdir(os.path.join(self.source_dir, f)))]

        if class_folders:
            print(f"Found {len(class_folders)} class folders in source directory")
            for class_name in class_folders:
                src = os.path.join(self.source_dir, class_name)
                dst = os.path.join(self.train_dir, class_name)
                print(f"Copying {class_name}...")
                shutil.copytree(src, dst)
            return

        # Case 2: Source has train/test structure
        train_path = os.path.join(self.source_dir, "train")
        if os.path.exists(train_path):
            print("Copying from train/ directory...")
            for class_name in os.listdir(train_path):
                src = os.path.join(train_path, class_name)
                if os.path.isdir(src):
                    dst = os.path.join(self.train_dir, class_name)
                    print(f"Copying {class_name}...")
                    shutil.copytree(src, dst)
            return

        raise FileNotFoundError("No class folders found in source directory")

    def _merge_train_test_sets(self) -> None:
        """Merge train and test sets into train folder"""
        if os.path.exists(self.test_dir):
            print("Merging train and test sets...")

            # Copy all test subfolders to train
            for class_name in os.listdir(self.test_dir):
                test_class_dir = os.path.join(self.test_dir, class_name)
                train_class_dir = os.path.join(self.train_dir, class_name)

                if os.path.isdir(test_class_dir):
                    os.makedirs(train_class_dir, exist_ok=True)
                    for item in os.listdir(test_class_dir):
                        src = os.path.join(test_class_dir, item)
                        dst = os.path.join(train_class_dir, item)

                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)

            print(f"Merged test set into {self.train_dir}")
        else:
            print(f"No test directory found at {self.test_dir}")

    def _setup_device(self) -> torch.device:
        use_gpu = self.config["execution_flags"]["use_gpu"] and torch.cuda.is_available()
        return torch.device("cuda" if use_gpu else "cpu")

    def _initialize_model(self) -> nn.Module:
        model = FeatureExtractorCNN(
            in_channels=self.config["dataset"]["in_channels"],
            feature_dims=self.config["model"]["feature_dims"],
            dropout_prob=self.config["model"].get("dropout_prob", 0.5)
        )
        return model.to(self.device)

    def _configure_optimizer(self) -> optim.Optimizer:
        opt_config = self.config["model"]["optimizer"]
        params = {
            'params': self.model.parameters(),
            'lr': self.config["model"]["learning_rate"],
            'weight_decay': opt_config["weight_decay"]
        }

        if opt_config["type"] == "Adam":
            return optim.Adam(**params, betas=(opt_config["beta1"], opt_config["beta2"]), eps=opt_config["epsilon"])
        elif opt_config["type"] == "SGD":
            return optim.SGD(**params, momentum=opt_config["momentum"])
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")

    def _configure_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        sched_config = self.config["model"]["scheduler"]
        if sched_config["type"] == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=sched_config["factor"],
                patience=sched_config["patience"],
                min_lr=sched_config["min_lr"],
                verbose=sched_config["verbose"]
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_config['type']}")

    def _configure_transforms(self) -> transforms.Compose:
        aug_config = self.config["augmentation"]
        transform_list = []

        if aug_config["enabled"]:
            if aug_config["random_crop"]["enabled"]:
                transform_list.append(
                    transforms.RandomCrop(
                        self.config["dataset"]["input_size"],
                        padding=aug_config["random_crop"]["padding"]
                    )
                )
            else:
                transform_list.append(
                    transforms.Resize(self.config["dataset"]["input_size"])
                )

            if aug_config["random_rotation"]["enabled"]:
                transform_list.append(transforms.RandomRotation(aug_config["random_rotation"]["degrees"]))

            if aug_config["horizontal_flip"]["enabled"]:
                transform_list.append(transforms.RandomHorizontalFlip(aug_config["horizontal_flip"]["probability"]))

            if aug_config["color_jitter"]["enabled"]:
                transform_list.append(transforms.ColorJitter(
                    brightness=aug_config["color_jitter"]["brightness"],
                    contrast=aug_config["color_jitter"]["contrast"],
                    saturation=aug_config["color_jitter"]["saturation"],
                    hue=aug_config["color_jitter"]["hue"]
                ))

        transform_list.append(transforms.ToTensor())

        if aug_config["normalize"]["enabled"]:
            transform_list.append(transforms.Normalize(
                mean=self.config["dataset"]["mean"],
                std=self.config["dataset"]["std"]
            ))

        return transforms.Compose(transform_list)

    def train(self) -> Dict[str, int]:
        """Train the feature extractor model and return label encoding dictionary

        Returns:
            Dictionary mapping class names to numeric labels
        """
        # Create dataset and establish label encoding
        self.class_to_idx = train_dataset.class_to_idx
        self._save_label_encoding()
        # Verify we have classes
        if self.class_to_idx is None or self.num_classes is None:
            raise RuntimeError("Classes not properly initialized")
        # Create dataset using actual class folders
        train_dataset = ImageFolderWithPaths(
            self.train_dir,
            transform=self.transform
        )
        # Update config with actual class count
        self.config["dataset"]["num_classes"] = self.num_classes
        self.clusterer = KLDivergenceClusterer(self.config)

        # Split into training and validation
        val_size = int(len(train_dataset) * self.config["training"]["validation_split"])
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True
        )

        # Training setup
        num_epochs = self.config["training"]["epochs"]
        best_val_loss = float('inf')
        patience = self.config["training"]["early_stopping"]["patience"]
        patience_counter = 0

        # Mixed precision training if enabled
        scaler = torch.cuda.amp.GradScaler(enabled=self.config["execution_flags"]["mixed_precision"])

        print(f"\nTraining on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        print(f"Classes: {self.class_to_idx}")

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, (data, target, _) in enumerate(progress_bar):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # Mixed precision forward pass
                with torch.cuda.amp.autocast(enabled=self.config["execution_flags"]["mixed_precision"]):
                    features = self.model(data)
                    kl_loss = self.clusterer.compute_kl_divergence(features, target)

                    # Total loss with weighting
                    loss = kl_loss * self.config["model"]["CNN_config"]["enhancements"]["kl_divergence_weight"]

                # Backward pass with scaling for mixed precision
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                train_loss += loss.item()
                progress_bar.set_postfix({"train_loss": f"{train_loss/(batch_idx+1):.4f}"})

            # Validation
            val_loss = self._validate(val_loader)
            avg_train_loss = train_loss / len(train_loader)

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Print epoch summary
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping and model checkpointing
            if val_loss < best_val_loss - self.config["training"]["early_stopping"]["min_delta"]:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model()
                print("  Saved best model")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

        # Load best model weights
        self._load_model()
        print("Training completed. Loaded best model weights.")

        return self.class_to_idx

    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                features = self.model(data)
                kl_loss = self.clusterer.compute_kl_divergence(features, target)
                val_loss += kl_loss.item()

        return val_loss / len(val_loader)

    def _save_label_encoding(self) -> None:
        """Save label encoding to file"""
        encoding_path = os.path.join(self.model_dir, "label_encoding.pkl")
        with open(encoding_path, 'wb') as f:
            pickle.dump(self.class_to_idx, f)

    def _load_label_encoding(self) -> Optional[Dict[str, int]]:
        """Load label encoding from file"""
        encoding_path = os.path.join(self.model_dir, "label_encoding.pkl")
        if os.path.exists(encoding_path):
            with open(encoding_path, 'rb') as f:
                return pickle.load(f)
        return None

def predict(self, image_dir: str, output_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Predict features for images in a directory and save to CSV

    Args:
        image_dir: Directory containing images (can have subfolders)
        output_csv: Optional custom output path. Defaults to data/<datafolder>/<datafolder>.csv

    Returns:
        DataFrame with columns: image_path, target_name, target, feature_0...feature_N
    """
    # Set default output path if not specified
    if output_csv is None:
        output_csv = os.path.join(self.datafolder, f"{self.dataset_name}.csv")

    # Load label encoding if available
    self.class_to_idx = self._load_label_encoding()

    # Create dataset and loader
    dataset = datasets.ImageFolder(image_dir, transform=self.transform)
    loader = DataLoader(
        dataset,
        batch_size=self.config["training"]["batch_size"],
        shuffle=False,
        num_workers=self.config["training"]["num_workers"]
    )

    # Get all image paths (including subdirectories)
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    # Validate we found images
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    # Extract features
    self.model.eval()
    features_list = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting features"):
            images = images.to(self.device)
            features = self.model(images).cpu().numpy()
            features_list.append(features)

    # Create DataFrame
    features_array = np.concatenate(features_list, axis=0)
    feature_cols = [f"feature_{i}" for i in range(features_array.shape[1])]

    df = pd.DataFrame(features_array, columns=feature_cols)
    df["image_path"] = image_paths

    # Add target information based on folder structure
    df["target_name"] = df["image_path"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )

    # Set numeric targets if we have label encoding
    if self.class_to_idx:
        df["target"] = df["target_name"].map(self.class_to_idx).fillna(-1).astype(int)
    else:
        df["target"] = -1

    # Ensure consistent column order
    df = df[["image_path", "target_name", "target"] + feature_cols]

    # Save results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nPrediction results saved to: {output_csv}")
    print(f"Total images processed: {len(df)}")
    if self.class_to_idx:
        known = sum(df["target"] != -1)
        print(f"Images with known classes: {known} ({known/len(df):.1%})")

    return df

    def _save_model(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "feature_extractor.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, model_path)

    def _load_model(self) -> None:
        model_path = os.path.join(self.model_dir, "feature_extractor.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def extract_features(self) -> None:
        """Extract features and save to CSV"""
        dataset = ImageFolderWithPaths(self.train_dir, transform=self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"]
        )

        self.model.eval()
        features_list = []
        paths_list = []
        labels_list = []
        class_names_list = []

        with torch.no_grad():
            for data, target, paths in tqdm(loader, desc="Extracting features"):
                data = data.to(self.device)
                features = self.model(data).cpu().numpy()

                features_list.append(features)
                paths_list.extend(paths)
                labels_list.extend(target.numpy())
                class_names_list.extend([os.path.basename(os.path.dirname(p)) for p in paths])

        # Create DataFrame
        features_array = np.concatenate(features_list, axis=0)
        feature_cols = [f"feature_{i}" for i in range(features_array.shape[1])]

        df = pd.DataFrame(features_array, columns=feature_cols)
        df["image_path"] = paths_list
        df["target"] = labels_list
        df["target_name"] = class_names_list

        # Save to CSV
        df.to_csv(self.csv_path, index=False)
        print(f"Features saved to {self.csv_path}")

    def run(self) -> None:
        """Run the complete pipeline"""
        mode = self.config["execution_flags"]["mode"]

        if mode in ["train", "train_and_predict"]:
            if self.config["execution_flags"]["fresh_start"] or not self.config["execution_flags"]["use_previous_model"]:
                print("Starting fresh training...")
                self.train()
            else:
                try:
                    print("Loading existing model...")
                    self._load_model()
                except FileNotFoundError:
                    print("No existing model found, starting fresh training")
                    self.train()

        if mode in ["predict", "train_and_predict"]:
            self.extract_features()

def main():
    print("Welcome to Convolution DBNN")
    parser = argparse.ArgumentParser(description="Feature Extraction Pipeline")
    parser.add_argument("--source", type=str, required=True,
                      help="Path to source directory containing the data")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Name for the dataset (will create data/<dataset>/)")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "train_and_predict"],
                       default="train_and_predict", help="Execution mode")
    parser.add_argument("--predict_dir", type=str,
                       help="Directory to predict on (required for predict mode)")
    parser.add_argument("--predict_output", type=str,
                       help="Optional custom path for output CSV. Default: data/<datafolder>/<datafolder>.csv")
    parser.add_argument("--merge_train_test", action="store_true",
                       help="Force merge of train and test sets without prompt")
    parser.add_argument("--force", action="store_true",
                          help="Automatically overwrite existing files without prompt")
    args = parser.parse_args()

    # Set up destination path
    dest_dir = os.path.join("data", args.dataset)
    #os.makedirs(dest_dir, exist_ok=True)

    print(f"\nSetting up data structure in {dest_dir}")
    print(f"Source data from: {args.source}")

    # Initialize pipeline
    pipeline = FeatureExtractorPipeline(
        source_dir=args.source,
        datafolder=dest_dir,
        merge_train_test=args.merge_train_test,
        interactive=not args.force
    )

    if args.mode in ["train", "train_and_predict"]:
        print("\nStarting training...")
        class_mapping = pipeline.train()
        print("\nTraining completed successfully!")
        print("Class mapping:", class_mapping)
    # Handle prediction
    if args.predict_dir:
        # Set default output path if not specified
        if args.predict_output is None:
            args.predict_output = os.path.join(datafolder, f"{base_datafolder}.csv")

        print(f"\nStarting prediction on images in: {args.predict_dir}")
        print(f"Results will be saved to: {args.predict_output}")

        predictions = pipeline.predict(args.predict_dir, args.predict_output)

        # Print sample of results
        print("\nSample predictions:")
        print(predictions[["image_path", "target_name", "target"]].head())

if __name__ == "__main__":
    main()
