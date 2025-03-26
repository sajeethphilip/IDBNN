import json
import sys
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

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {
    # Standard formats
    'bmp', 'gif', 'jpeg', 'jpg', 'png', 'tif', 'tiff', 'webp',

    # Astronomy-specific formats
    'fits', 'fit', 'fts',

    # Medical imaging formats
    'dcm', 'dicom', 'nii', 'nii.gz',

    # Satellite/Remote sensing formats
    'hdf', 'hdf5', 'h5', 'nc', 'img', 'dat',

    # Agriculture/Geospatial formats
    'jp2', 'j2k', 'sid', 'ecw', 'las', 'laz'
}

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that handles multiple image formats and preserves paths"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Find all supported files recursively
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        # Build class mapping
        class_names = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        if not class_names:
            class_names = ['']  # Handle case with no subdirectories
            self.samples = self._find_images(root)
        else:
            for idx, class_name in enumerate(class_names):
                self.class_to_idx[class_name] = idx
                class_dir = os.path.join(root, class_name)
                self.samples.extend([(p, idx) for p in self._find_images(class_dir)])

        self.classes = list(self.class_to_idx.keys())

    def _find_images(self, directory):
        """Recursively find all supported image files"""
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1][1:].lower()
                if ext in SUPPORTED_IMAGE_FORMATS:
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __getitem__(self, index):
        path, target = self.samples[index]

        # Handle different file formats
        ext = os.path.splitext(path)[1][1:].lower()

        if ext in {'fits', 'fit', 'fts'}:
            # Astronomy FITS files
            with fits.open(path) as hdul:
                image = hdul[0].data
                if image.ndim == 2:
                    image = np.stack([image]*3, axis=-1)  # Convert to 3-channel
                image = Image.fromarray(image)
        elif ext in {'dcm', 'dicom'}:
            # Medical DICOM files
            ds = pydicom.dcmread(path)
            image = ds.pixel_array
            if image.ndim == 2:
                image = np.stack([image]*3, axis=-1)
            image = Image.fromarray(image)
        elif ext in {'nii', 'nii.gz'}:
            # NIfTI medical images (middle slice)
            img = nib.load(path)
            data = img.get_fdata()
            slice_idx = data.shape[-1] // 2
            image = data[:, :, slice_idx]
            image = Image.fromarray(image).convert('RGB')
        else:
            # Standard image formats
            image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, target, path

class FeatureExtractorPipeline:
    """Complete feature extraction pipeline"""
    def __init__(self, source_dir: str, dataname: str, merge_train_test: bool = False, interactive: bool = False):
        """
        Args:
            source_dir: Path to source directory containing the original data
            output_dir: Path to destination directory (under data/)
            merge_train_test: Whether to merge train/test sets
            interactive: Whether to prompt for user input
        """
        """Initialize with dataname instead of datafolder"""
        self.dataname = dataname
        self.source_dir = os.path.abspath(source_dir)
        self.interactive = interactive

        # Create output directory structure
        self.output_dir = os.path.join("data", self.dataname)  # data/<dataname>/
        self.model_dir = os.path.join(self.output_dir, "models")
        self.train_dir = os.path.join(self.output_dir, "train")

        # Config paths
        self.config_path = os.path.join(self.output_dir, f"{self.dataname}.json")
        self.conf_path = os.path.join(self.output_dir, f"{self.dataname}.conf")

        # Initialize components
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
        config_path = self.config_path

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
                "name": self.dataname,
                "type": "image_folder",
                "in_channels": 3,
                "num_classes": 10,
                "input_size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize_images": True,
                "train_dir": os.path.join(self.output_dir, "train"),
                "test_dir": os.path.join(self.output_dir, "train", "test") if not self.merge_train_test else None,
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
            "features_file": os.path.join(self.output_dir, f"{self.dataname}.csv"),
            "model_dir": self.model_dir,
                "visualization_dir": os.path.join(self.output_dir, "visualizations")
            }
        }

    def _initialize_conf(self) -> Dict[str, Any]:
        """Initialize .conf file in data/<datafolder>/"""
        conf_path = self.conf_path

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
        print(f"\nSetting up data structure for {self.dataname}")

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

    def _prepare_data_structure(self, mode: str = 'train') -> None:
        """Prepare data structure for training or prediction"""
        # Create necessary directories
        os.makedirs(self.model_dir, exist_ok=True)

        # Handle train directory based on mode
        if mode == 'train':
            # Clear existing training data if needed
            if os.path.exists(self.train_dir):
                if self.interactive:
                    response = input(f"Training directory exists at {self.train_dir}. Overwrite? [y/n] ").lower()
                    if response != 'y':
                        raise FileExistsError("Training directory exists and user chose not to overwrite")
                shutil.rmtree(self.train_dir)
            os.makedirs(self.train_dir)

            # Copy data from source to training directory
            self._copy_data_from_source()

        elif mode == 'predict':
            # For prediction, we don't need a train directory
            if os.path.exists(self.train_dir):
                shutil.rmtree(self.train_dir)

        # Verify we have either training data or a trained model
        if mode == 'train' and not os.listdir(self.train_dir):
            raise FileNotFoundError(f"No valid training data found in {self.train_dir}")
        elif mode == 'predict' and not os.path.exists(os.path.join(self.model_dir, "feature_extractor.pth")):
            raise FileNotFoundError("No trained model found for prediction")

    def _copy_data_from_source(self) -> None:
        """Copy only supported files from source to training directory"""
        # First check for supported image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.fits'}
        found_files = False

        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    found_files = True
                    break
            if found_files:
                break

        if not found_files:
            raise ValueError(f"No supported image files found in {self.source_dir}. Supported formats: {image_extensions}")

        print(f"Copying supported files from {self.source_dir} to {self.train_dir}")

        # Create class folders and copy files
        class_folders = [d for d in os.listdir(self.source_dir)
                        if os.path.isdir(os.path.join(self.source_dir, d))]

        if not class_folders:
            # Handle case with no subdirectories (all files in root)
            os.makedirs(os.path.join(self.train_dir, "unclassified"), exist_ok=True)
            self._copy_supported_files(self.source_dir, os.path.join(self.train_dir, "unclassified"))
        else:
            for class_name in class_folders:
                src_dir = os.path.join(self.source_dir, class_name)
                dst_dir = os.path.join(self.train_dir, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                self._copy_supported_files(src_dir, dst_dir)

    def _copy_supported_files(self, src_dir: str, dst_dir: str) -> None:
        """Copy only supported files from source to destination"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.fits'}

        for file in os.listdir(src_dir):
            src_path = os.path.join(src_dir, file)
            if os.path.isfile(src_path) and os.path.splitext(file)[1].lower() in image_extensions:
                shutil.copy2(src_path, dst_dir)

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
                "Please ensure the source directory contains:\n"
                "1. Subfolders with images\n"
                "2. Or a 'train/' subdirectory with class folders"
            )

        # Create class mapping and count
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(class_folders))}
        self.num_classes = len(self.class_to_idx)

        print(f"Found {self.num_classes} classes: {self.class_to_idx}")
        print(f"Total training samples: {sum(len(os.listdir(os.path.join(self.train_dir, cls))) for cls in class_folders)}")

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
        """Initialize model, loading previous best if available"""
        model = FeatureExtractorCNN(
            in_channels=self.config["dataset"]["in_channels"],
            feature_dims=self.config["model"]["feature_dims"],
            dropout_prob=self.config["model"].get("dropout_prob", 0.5)
        ).to(self.device)

        # Check for existing best model
        model_path = os.path.join(self.model_dir, "feature_extractor.pth")
        if os.path.exists(model_path) and not self.config["execution_flags"]["fresh_start"]:
            print("Loading previously trained model...")
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # If class mapping exists in checkpoint, use it
            if 'class_to_idx' in checkpoint:
                self.class_to_idx = checkpoint['class_to_idx']
                self.num_classes = len(self.class_to_idx)
                print(f"Loaded class mapping: {self.class_to_idx}")
        else:
            print("Initializing new model")

        return model

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
        """Train until early stopping condition is met"""
        train_dataset = ImageFolderWithPaths(self.train_dir, transform=self.transform)
        self.class_to_idx = train_dataset.class_to_idx
        self.num_classes = len(self.class_to_idx)

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
        best_val_loss = float('inf')
        patience = self.config["training"]["early_stopping"]["patience"]
        patience_counter = 0
        epoch = 0

        print(f"\nTraining on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        print(f"Classes: {self.class_to_idx}")
        print(f"Training until validation doesn't improve for {patience} epochs")

        while True:  # Continue until early stopping
            epoch += 1
            self.model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for batch_idx, (data, target, _) in enumerate(progress_bar):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                features = self.model(data)
                kl_loss = self.clusterer.compute_kl_divergence(features, target)

                # Total loss
                loss = kl_loss * self.config["model"]["CNN_config"]["enhancements"]["kl_divergence_weight"]

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({"train_loss": f"{train_loss/(batch_idx+1):.4f}"})

            # Validation
            val_loss = self._validate(val_loader)
            avg_train_loss = train_loss / len(train_loader)

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Print epoch summary
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.2e}")

            # Check for improvement
            if val_loss < best_val_loss - self.config["training"]["early_stopping"]["min_delta"]:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_model()
                print("  Saved best model")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

        # Load best model weights before returning
        self._load_best_model()
        # After training completes, extract and save features
        train_csv_path = os.path.join(self.output_dir, f"{self.dataname}_train_features.csv")
        self._extract_and_save_features(self.train_dir, train_csv_path)

        # If validation exists, save those features too
        if hasattr(self, 'val_dir') and os.path.exists(self.val_dir):
            val_csv_path = os.path.join(self.output_dir, f"{self.dataname}_val_features.csv")
            self._extract_and_save_features(self.val_dir, val_csv_path)
        return self.class_to_idx

    def _load_best_model(self) -> None:
        """Load the best saved model weights"""
        model_path = os.path.join(self.model_dir, "feature_extractor.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best model weights")
        else:
            print("Warning: No saved model found to load")

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

    def predict(self, input_dir: str, output_csv: Optional[str] = None) -> pd.DataFrame:
        """Predict features from input directory"""
        # Set output path
        if output_csv is None:
            output_csv = os.path.join(self.output_dir, f"{self.dataname}.csv")

        # Verify input directory exists
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Create temporary prediction directory structure
        temp_pred_dir = os.path.join(self.output_dir, "temp_pred")
        if os.path.exists(temp_pred_dir):
            shutil.rmtree(temp_pred_dir)
        os.makedirs(temp_pred_dir)

        try:
            # Copy input files to temporary directory (maintaining structure)
            if os.path.isdir(input_dir):
                self._copy_supported_files(input_dir, temp_pred_dir)
            else:
                # Handle single file input
                ext = os.path.splitext(input_dir)[1].lower()
                if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.fits'}:
                    shutil.copy2(input_dir, temp_pred_dir)
                else:
                    raise ValueError(f"Unsupported file format: {input_dir}")

            # Extract features
            return self._extract_and_save_features(temp_pred_dir, output_csv)
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_pred_dir)

    def _extract_and_save_features(self, input_dir: str, output_csv: str) -> pd.DataFrame:
        """Unified feature extraction and CSV saving"""
        # Determine data type
        if any(f.endswith('.npy') for f in os.listdir(input_dir)):
            df = self._extract_timeseries_features(input_dir)
        else:
            df = self._extract_image_features(input_dir)

        # Save to CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
        return df

    def _extract_timeseries_features(self, timeseries_dir: str) -> pd.DataFrame:
        """Extract features from time-series data"""
        results = {
            'file_path': [],
            'window_start': [],
            'window_end': [],
            'features': []
        }

        for ts_file in tqdm(sorted(glob.glob(os.path.join(timeseries_dir, '*.npy'))),
                          desc="Processing time-series"):
            ts_data = np.load(ts_file)
            dataset = TimeSeriesDataset(ts_data, self.window_size)
            loader = DataLoader(dataset, batch_size=self.config["training"]["batch_size"], shuffle=False)

            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    features = self.model(batch.to(self.device)).cpu().numpy()

                    for i in range(batch.size(0)):
                        idx = len(results['features'])
                        results['file_path'].append(ts_file)
                        results['window_start'].append(idx * self.window_size)
                        results['window_end'].append((idx + 1) * self.window_size - 1)
                        results['features'].append(features[i])

        return self._create_feature_dataframe(results)

    def _extract_image_features(self, image_dir: str) -> pd.DataFrame:
        """Extract features from image dataset"""
        dataset = ImageFolderWithPaths(image_dir, transform=self.transform)
        loader = DataLoader(dataset, batch_size=self.config["training"]["batch_size"], shuffle=False)

        results = {
            'file_path': [],
            'subfolder': [],
            'target_name': [],
            'target': [],
            'format': [],
            'features': []
        }

        self.model.eval()
        with torch.no_grad():
            for images, targets, paths in tqdm(loader, desc="Extracting image features"):
                features = self.model(images.to(self.device)).cpu().numpy()

                for i in range(len(paths)):
                    path = paths[i]
                    results['file_path'].append(path)
                    results['subfolder'].append(os.path.basename(os.path.dirname(path)))
                    results['target_name'].append(dataset.classes[targets[i]])
                    results['target'].append(targets[i].item())
                    results['format'].append(os.path.splitext(path)[1][1:].lower())
                    results['features'].append(features[i])

        return self._create_feature_dataframe(results)

    def _create_feature_dataframe(self, results: Dict) -> pd.DataFrame:
        """Convert extraction results to DataFrame"""
        feature_cols = [f'feature_{i}' for i in range(len(results['features'][0]))]
        df = pd.DataFrame(np.vstack(results['features']), columns=feature_cols)

        # Add metadata columns
        for col in [k for k in results.keys() if k != 'features']:
            df.insert(0, col, results[col])

        # Make paths relative to datafolder
        df['file_path'] = df['file_path'].apply(
            lambda x: os.path.relpath(x, self.output_dir) if x.startswith(self.output_dir) else x)

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

    def extract_features(self, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Extract features from both images and time-series data, preserving all metadata

        Args:
            output_csv: Optional output path (default: data/<datafolder>/<datafolder>.csv)

        Returns:
            DataFrame with complete tracking information including:
            - For images: file_path, subfolder, target, format, timestamp (if available)
            - For time-series: window_start, window_end, channel_stats, original_file
        """
        # Set default output path
        if output_csv is None:
            output_csv = os.path.join(self.output_dir, f"{self.dataname}.csv")

        # Initialize results storage
        results = {
            'type': [],               # 'image' or 'timeseries'
            'file_path': [],          # Original file path
            'subfolder': [],          # For images
            'target_name': [],        # Class name
            'target': [],             # Numeric label
            'format': [],             # File format
            'timestamp': [],          # For time-series
            'window_start': [],       # For time-series
            'window_end': [],         # For time-series
            'features': []            # Extracted features
        }

        # Process image data if available
        if hasattr(self, 'train_dir') and os.path.exists(self.train_dir):
            image_dataset = ImageFolderWithPaths(self.train_dir, transform=self.transform)
            image_loader = DataLoader(
                image_dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                num_workers=self.config["training"]["num_workers"]
            )

            self.model.eval()
            with torch.no_grad():
                for images, targets, paths in tqdm(image_loader, desc="Processing images"):
                    features = self.model(images.to(self.device)).cpu().numpy()

                    for i in range(len(paths)):
                        path = paths[i]
                        results['type'].append('image')
                        results['file_path'].append(path)
                        results['subfolder'].append(os.path.basename(os.path.dirname(path)))
                        results['target_name'].append(image_dataset.classes[targets[i]])
                        results['target'].append(targets[i].item())
                        results['format'].append(os.path.splitext(path)[1][1:].lower())
                        results['timestamp'].append(os.path.getmtime(path))
                        results['window_start'].append(None)
                        results['window_end'].append(None)
                        results['features'].append(features[i])

        # Process time-series data if available
        if hasattr(self, 'timeseries_dir') and os.path.exists(self.timeseries_dir):
            for ts_file in tqdm(sorted(glob.glob(os.path.join(self.timeseries_dir, '*.npy'))),
                              desc="Processing time-series"):
                # Load time-series data (channels x timepoints)
                ts_data = np.load(ts_file)

                # Create sliding window dataset
                ts_dataset = TimeSeriesDataset(ts_data, window_size=256)
                ts_loader = DataLoader(
                    ts_dataset,
                    batch_size=self.config["training"]["batch_size"],
                    shuffle=False
                )

                # Extract features for each window
                self.model.eval()
                with torch.no_grad():
                    for batch in ts_loader:
                        batch_features = self.model(batch.to(self.device)).cpu().numpy()

                        for i in range(batch.size(0)):
                            idx = len(results['features'])
                            window_idx = ts_dataset.indices[idx] if hasattr(ts_dataset, 'indices') else idx

                            results['type'].append('timeseries')
                            results['file_path'].append(ts_file)
                            results['subfolder'].append(os.path.basename(os.path.dirname(ts_file)))
                            results['target_name'].append(os.path.splitext(os.path.basename(ts_file))[0])
                            results['target'].append(-1)  # Can be set during post-processing
                            results['format'].append('npy')
                            results['timestamp'].append(os.path.getmtime(ts_file))
                            results['window_start'].append(window_idx)
                            results['window_end'].append(window_idx + 255)
                            results['features'].append(batch_features[i])

        # Convert to DataFrame
        if not results['features']:
            raise ValueError("No features extracted - check input data paths")

        feature_cols = [f'feature_{i}' for i in range(len(results['features'][0]))]
        df = pd.DataFrame(
            np.vstack(results['features']),
            columns=feature_cols
        )

        # Add metadata columns
        for col in ['type', 'file_path', 'subfolder', 'target_name', 'target',
                   'format', 'timestamp', 'window_start', 'window_end']:
            df.insert(0, col, results[col])

        # Make paths relative to datafolder when possible
        try:
            df['file_path'] = df['file_path'].apply(
                lambda x: os.path.relpath(x, start=self.output_dir))
        except ValueError:
            pass  # Keep absolute paths if they're not under datafolder

        # Save results
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)

        print(f"\nFeature extraction complete. Results saved to {output_csv}")
        print(f"Total samples processed: {len(df)}")
        print(f"Breakdown by type:\n{df['type'].value_counts()}")
        print(f"Columns: {df.columns.tolist()}")

        return df

def main():
    parser = argparse.ArgumentParser(description="Feature Extraction Pipeline")
    parser.add_argument("--dataname", type=str, required=True,
                      help="Name for the dataset (will create data/<dataname>/ directory)")
    parser.add_argument("--mode", choices=["train", "predict", "full"],
                      default="full", help="Execution mode")
    parser.add_argument("--input", type=str, required=True,
                      help="Input directory containing data")
    parser.add_argument("--output_csv", type=str, default=None,
                      help="Optional custom CSV output path")
    parser.add_argument("--force", action="store_true",
                      help="Force overwrite without prompts")

    args = parser.parse_args()

    # Initialize pipeline with new dataname parameter
    pipeline = FeatureExtractorPipeline(
        dataname=args.dataname,  # Changed from datafolder
        source_dir=args.input,
        interactive=not args.force
    )

    try:
        if args.mode in ["train", "full"]:
            print("\n===== Training Mode =====")
            pipeline._prepare_data_structure(mode='train')
            class_mapping = pipeline.train()
            print(f"\nTraining complete. Class mapping: {class_mapping}")

        if args.mode in ["predict", "full"]:
            print("\n===== Prediction Mode =====")
            features_df = pipeline.predict(args.input, args.output_csv)
            print(f"\nPrediction complete. Features saved to {args.output_csv or pipeline.output_dir}")
            print(f"Total samples processed: {len(features_df)}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
