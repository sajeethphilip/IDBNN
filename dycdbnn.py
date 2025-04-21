# ----------------------------
# Dynamic CNN Integrated Version
# Maintains original CSV/config outputs
# ----------------------------
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import numpy as np
import pandas as pd
import os
import json
import csv
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional
import torch.nn.functional as F
import logging
from collections import defaultdict
from pathlib import Path
import shutil
import zipfile
import tarfile
import gzip
import bz2
import lzma
from datetime import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#------------------------
# Helper Functions
#-------------------------

class DatasetProcessor:
    SUPPORTED_FORMATS = {
        'zip': zipfile.ZipFile,
        'tar': tarfile.TarFile,
        'tar.gz': tarfile.TarFile,
        'tgz': tarfile.TarFile,
        'gz': gzip.GzipFile,
        'bz2': bz2.BZ2File,
        'xz': lzma.LZMAFile
    }

    SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    def __init__(self, datafile: str = "MNIST", datatype: str = "torchvision",
                 output_dir: str = "data", config: Optional[Dict] = None):
        self.datafile = datafile
        self.datatype = datatype.lower()
        self.output_dir = output_dir
        self.config = config if config is not None else {}  # Initialize config

        if self.datatype == 'torchvision':
            self.dataset_name = self.datafile.lower()
        else:
            self.dataset_name = Path(self.datafile).stem.lower()

        self.dataset_dir = os.path.join("data", self.dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.config_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.json")
        self.conf_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.conf")
        self.dbnn_conf_path = os.path.join(self.dataset_dir, "adaptive_dbnn.conf")

    def _extract_archive(self, archive_path: str) -> str:
        """Extract compressed archive to temporary directory"""
        extract_dir = os.path.join(self.dataset_dir, 'temp_extract')
        os.makedirs(extract_dir, exist_ok=True)

        file_ext = Path(archive_path).suffix.lower()
        if file_ext.startswith('.'):
            file_ext = file_ext[1:]

        if file_ext == 'zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif file_ext in ['tar', 'tgz'] or archive_path.endswith('tar.gz'):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        elif file_ext == 'gz':
            output_path = os.path.join(extract_dir, Path(archive_path).stem)
            with gzip.open(archive_path, 'rb') as gz_file:
                with open(output_path, 'wb') as out_file:
                    shutil.copyfileobj(gz_file, out_file)
        elif file_ext == 'bz2':
            output_path = os.path.join(extract_dir, Path(archive_path).stem)
            with bz2.open(archive_path, 'rb') as bz2_file:
                with open(output_path, 'wb') as out_file:
                    shutil.copyfileobj(bz2_file, out_file)
        elif file_ext == 'xz':
            output_path = os.path.join(extract_dir, Path(archive_path).stem)
            with lzma.open(archive_path, 'rb') as xz_file:
                with open(output_path, 'wb') as out_file:
                    shutil.copyfileobj(xz_file, out_file)
        else:
            raise ValueError(f"Unsupported archive format: {file_ext}")

        return extract_dir

    def _process_data_path(self, data_path: str) -> str:
        """Process input data path, handling compressed files if necessary"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

        file_ext = Path(data_path).suffix.lower()
        if file_ext.startswith('.'):
            file_ext = file_ext[1:]

        # Check if it's a compressed file
        if file_ext in self.SUPPORTED_FORMATS or data_path.endswith('tar.gz'):
            logger.info(f"Extracting compressed file: {data_path}")
            extract_dir = self._extract_archive(data_path)

            # Find the main data directory
            contents = os.listdir(extract_dir)
            if len(contents) == 1 and os.path.isdir(os.path.join(extract_dir, contents[0])):
                return os.path.join(extract_dir, contents[0])
            return extract_dir

        return data_path

    def process(self) -> Tuple[str, Optional[str]]:
        """Process dataset and return paths to train and test directories"""
        if self.datatype == 'torchvision':
            # Only process as torchvision if explicitly specified
            return self._process_torchvision()
        else:
            # Process as local path
            processed_path = self._process_data_path(self.datafile)
            return self._process_custom(processed_path)

    def _handle_existing_directory(self, path: str):
        """Handle existing directory by either removing it or merging its contents."""
        if os.path.exists(path):
            response = input(f"The directory '{path}' already exists. Do you want to (R)emove it or (M)erge its contents? [R/M]: ").lower()
            if response == 'r':
                shutil.rmtree(path)
                os.makedirs(path)
            elif response == 'm':
                # Merge contents (no action needed, as shutil.copytree will handle it with dirs_exist_ok=True)
                pass
            else:
                raise ValueError("Invalid choice. Please choose 'R' to remove or 'M' to merge.")

    def _process_custom(self, data_path: str) -> Tuple[str, Optional[str]]:
        """Process custom dataset structure"""
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        # Access enable_adaptive from training_params
        try:
             enable_adaptive = self.config['model'].get('enable_adaptive', True)
        except:
            enable_adaptive = True
            print(f"Enable Adaptive mode is set {enable_adaptive} in process custom")
        # Check if dataset already has train/test structure
        if os.path.isdir(os.path.join(data_path, "train")) and \
           os.path.isdir(os.path.join(data_path, "test")):
            # Check if adaptive_fit_predict is active
            if enable_adaptive:
                # Handle existing train directory
                self._handle_existing_directory(train_dir)
                # Merge train and test folders into a single train folder

                # Copy train data
                shutil.copytree(os.path.join(data_path, "train"), train_dir, dirs_exist_ok=True)

                return train_dir, test           # return train folder populated with both train and test data and the test folder for consistency.

            else:
                # Normal processing with separate train and test folders
                if os.path.exists(train_dir):
                    shutil.rmtree(train_dir)
                if os.path.exists(test_dir):
                    shutil.rmtree(test_dir)

                shutil.copytree(os.path.join(data_path, "train"), train_dir)
                shutil.copytree(os.path.join(data_path, "test"), test_dir)
                return train_dir, test_dir
        # Handle single directory with class subdirectories
        if not os.path.isdir(data_path):
            raise ValueError(f"Invalid dataset path: {data_path}")

        class_dirs = [d for d in os.listdir(data_path)
                     if os.path.isdir(os.path.join(data_path, d))]

        if not class_dirs:
            raise ValueError(f"No class directories found in {data_path}")

        # Ask user about train/test split
        response = input("Create train/test split? (y/n): ").lower()
        if response == 'y':
            test_size = float(input("Enter test size (0-1, default: 0.2): ") or "0.2")
            return self._create_train_test_split(data_path, test_size)
        else:
            # Use all data for training
            os.makedirs(train_dir, exist_ok=True)
            for class_dir in class_dirs:
                src = os.path.join(data_path, class_dir)
                dst = os.path.join(train_dir, class_dir)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            return train_dir, None

    def cleanup(self):
        """Clean up temporary files"""
        temp_dir = os.path.join(self.dataset_dir, 'temp_extract')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
#------------------------
    def get_transforms(self, config: Dict, is_train: bool = True) -> transforms.Compose:
        """Get transforms based on configuration"""
        transform_list = []

        # Handle resolution and channel conversion first
        target_size = tuple(config['dataset']['input_size'])
        target_channels = config['dataset']['in_channels']

        # Resolution adjustment
        transform_list.append(transforms.Resize(target_size))

        # Channel conversion
        if target_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        # Training augmentations
        if is_train and config.get('augmentation', {}).get('enabled', True):
            aug_config = config['augmentation']
            if aug_config.get('random_crop', {}).get('enabled', False):
                transform_list.append(transforms.RandomCrop(target_size, padding=4))
            if aug_config.get('horizontal_flip', {}).get('enabled', False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug_config.get('color_jitter', {}).get('enabled', False):
                transform_list.append(transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ))

        # Final transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=config['dataset']['mean'],
                               std=config['dataset']['std'])
        ])

        return transforms.Compose(transform_list)


    def _generate_main_config(self, train_dir: str) -> Dict:
        """Generate main configuration with all necessary parameters"""
        input_size, in_channels = self._detect_image_properties(train_dir)
        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        num_classes = len(class_dirs)

        mean = [0.5] if in_channels == 1 else [0.485, 0.456, 0.406]
        std = [0.5] if in_channels == 1 else [0.229, 0.224, 0.225]
        feature_dims = min(128, np.prod(input_size) // 4)

        return {
            "dataset": {
                "name": self.dataset_name,
                "type": self.datatype,
                "in_channels": in_channels,
                "num_classes": num_classes,
                "input_size": list(input_size),
                "mean": mean,
                "std": std,
                "resize_images": False,
                "train_dir": train_dir,
                "test_dir": os.path.join(os.path.dirname(train_dir), 'test')
            },
             "model": {
                "encoder_type": "autoenc",
                'enable_adaptive': True,  # Default value
                "feature_dims": feature_dims,
                "learning_rate": 0.001,
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
                "autoencoder_config": {
                    "reconstruction_weight": 1.0,
                    "feature_weight": 0.1,
                    "convergence_threshold": 0.0001,
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
                },
                "loss_functions": {
                    "structural": {
                        "enabled": True,
                        "weight": 1.0,
                        "params": {
                            "edge_weight": 1.0,
                            "smoothness_weight": 0.5
                        }
                    },
                    "color_enhancement": {
                        "enabled": True,
                        "weight": 0.8,
                        "params": {
                            "channel_weight": 0.5,
                            "contrast_weight": 0.3
                        }
                    },
                    "morphology": {
                        "enabled": True,
                        "weight": 0.6,
                        "params": {
                            "shape_weight": 0.7,
                            "symmetry_weight": 0.3
                        }
                    },
                    "detail_preserving": {
                        "enabled": True,
                        "weight": 0.8,
                        "params": {
                            "detail_weight": 1.0,
                            "texture_weight": 0.8,
                            "frequency_weight": 0.6
                        }
                    },
                    "astronomical_structure": {
                        "enabled": True,
                        "weight": 1.0,
                        "components": {
                            "edge_preservation": True,
                            "peak_preservation": True,
                            "detail_preservation": True
                        }
                    },
                    "medical_structure": {
                        "enabled": True,
                        "weight": 1.0,
                        "components": {
                            "boundary_preservation": True,
                            "tissue_contrast": True,
                            "local_structure": True
                        }
                    },
                    "agricultural_pattern": {
                        "enabled": True,
                        "weight": 1.0,
                        "components": {
                            "texture_preservation": True,
                            "damage_pattern": True,
                            "color_consistency": True
                        }
                    }
                },
                "enhancement_modules": {
                    "astronomical": {
                        "enabled": True,
                        "components": {
                            "structure_preservation": True,
                            "detail_preservation": True,
                            "star_detection": True,
                            "galaxy_features": True,
                            "kl_divergence": True
                        },
                        "weights": {
                            "detail_weight": 1.0,
                            "structure_weight": 0.8,
                            "edge_weight": 0.7
                        }
                    },
                    "medical": {
                        "enabled": True,
                        "components": {
                            "tissue_boundary": True,
                            "lesion_detection": True,
                            "contrast_enhancement": True,
                            "subtle_feature_preservation": True
                        },
                        "weights": {
                            "boundary_weight": 1.0,
                            "lesion_weight": 0.8,
                            "contrast_weight": 0.6
                        }
                    },
                    "agricultural": {
                        "enabled": True,
                        "components": {
                            "texture_analysis": True,
                            "damage_detection": True,
                            "color_anomaly": True,
                            "pattern_enhancement": True,
                            "morphological_features": True
                        },
                        "weights": {
                            "texture_weight": 1.0,
                            "damage_weight": 0.8,
                            "pattern_weight": 0.7
                        }
                    }
                }
            },
            "training": {
                "batch_size": 128,
                "epochs": 200,
                "num_workers": min(4, os.cpu_count() or 1),
                "checkpoint_dir": os.path.join(self.dataset_dir, "checkpoints"),
                "validation_split": 0.2,
                "invert_DBNN": True,
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.1,
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                }
            },
            "augmentation": {
                "enabled": True,
                "random_crop": {"enabled": True, "padding": 4},
                "random_rotation": {"enabled": True, "degrees": 10},
                "horizontal_flip": {"enabled": True, "probability": 0.5},
                "vertical_flip": {"enabled": False},
                "color_jitter": {
                    "enabled": True,
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.1
                },
                "normalize": {
                    "enabled": True,
                    "mean": mean,
                    "std": std
                }
            },
            "execution_flags": {
                "mode": "train_and_predict",
                "use_gpu": torch.cuda.is_available(),
                "mixed_precision": True,
                "distributed_training": False,
                "debug_mode": False,
                "use_previous_model": True,
                "fresh_start": False
            },
            "output": {
                "features_file": os.path.join(self.dataset_dir, f"{self.dataset_name}.csv"),
                "model_dir": os.path.join(self.dataset_dir, "models"),
                "visualization_dir": os.path.join(self.dataset_dir, "visualizations")
            }
        }

    def _generate_dataset_conf(self, feature_dims: int) -> Dict:
        """Generate dataset-specific configuration"""
        return {
            "file_path": os.path.join(self.dataset_dir, f"{self.dataset_name}.csv"),
            "column_names": [f"feature_{i}" for i in range(feature_dims)] + ["target"],
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
                "batch_size":128,
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
                "use_interactive_kbd": False,
                "class_preference": True
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

    def _generate_dbnn_config(self, main_config: Dict) -> Dict:
        """Generate DBNN-specific configuration"""
        return {
            "training_params": {
                "trials": main_config['training']['epochs'],
                "epochs": main_config['training']['epochs'],
                "learning_rate": main_config['model']['learning_rate'],
                "batch_size":128,
                "test_fraction": 0.2,
                "random_seed": 42,
                "minimum_training_accuracy": 0.95,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "n_bins_per_dim": 128,
                "enable_adaptive": True,
                "invert_DBNN": main_config['training'].get('invert_DBNN', False),
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.1,
                "Save_training_epochs": False,
                "training_save_path": os.path.join(self.dataset_dir, "training_data"),
                "modelType": "Histogram",
                "compute_device": "auto",
                "class_preference": True
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

    def generate_default_config(self, train_dir: str) -> Dict:
        """Generate and manage all configuration files"""
        os.makedirs(self.dataset_dir, exist_ok=True)
        logger.info(f"Starting configuration generation for dataset: {self.dataset_name}")

        # 1. Generate and handle main configuration (json)
        logger.info("Generating main configuration...")
        config = self._generate_main_config(train_dir)
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    existing_config = json.load(f)
                    logger.info(f"Found existing main config, merging...")
                    config = self._merge_configs(existing_config, config)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {self.config_path}, using default template")

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Main configuration saved: {self.config_path}")

        # 2. Generate and handle dataset.conf using _generate_dataset_conf
        logger.info("Generating dataset configuration...")
        dataset_conf = self._generate_dataset_conf(config['model']['feature_dims'])
        if os.path.exists(self.conf_path):
            try:
                with open(self.conf_path, 'r') as f:
                    existing_dataset_conf = json.load(f)
                    logger.info(f"Found existing dataset config, merging...")
                    dataset_conf = self._merge_configs(existing_dataset_conf, dataset_conf)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {self.conf_path}, using default template")

        with open(self.conf_path, 'w') as f:
            json.dump(dataset_conf, f, indent=4)
        logger.info(f"Dataset configuration saved: {self.conf_path}")

        # 3. Generate and handle adaptive_dbnn.conf using _generate_dbnn_config
        logger.info("Generating DBNN configuration...")
        dbnn_config = self._generate_dbnn_config(config)
        if os.path.exists(self.dbnn_conf_path):
            try:
                with open(self.dbnn_conf_path, 'r') as f:
                    existing_dbnn_config = json.load(f)
                    logger.info(f"Found existing DBNN config, merging...")
                    dbnn_config = self._merge_configs(existing_dbnn_config, dbnn_config)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {self.dbnn_conf_path}, using default template")

        with open(self.dbnn_conf_path, 'w') as f:
            json.dump(dbnn_config, f, indent=4)
        logger.info(f"DBNN configuration saved: {self.dbnn_conf_path}")

        # Return the main config for further use
        return config

    def _merge_configs(self, existing: Dict, default: Dict) -> Dict:
        """Recursively merge configs, preserving existing values"""
        result = existing.copy()
        for key, value in default.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
        return result

    def _ensure_required_configs(self, config: Dict) -> Dict:
        """Ensure all required configurations exist"""
        if 'loss_functions' not in config['model']:
            config['model']['loss_functions'] = {}

        if 'autoencoder' not in config['model']['loss_functions']:
            config['model']['loss_functions']['autoencoder'] = {
                'enabled': True,
                'type': 'AutoencoderLoss',
                'weight': 1.0,
                'params': {
                    'reconstruction_weight': 1.0,
                    'feature_weight': 0.1
                }
            }

        return config


    def _detect_image_properties(self, folder_path: str) -> Tuple[Tuple[int, int], int]:
        """Detect actual image properties but use config values if specified"""
        # Load existing config if available
        config_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'dataset' in config:
                    dataset_config = config['dataset']
                    if all(key in dataset_config for key in ['input_size', 'in_channels']):
                        logger.info("Using image properties from config file")
                        return (tuple(dataset_config['input_size']),
                                dataset_config['in_channels'])

        # Fall back to detection from files
        size_counts = defaultdict(int)
        channel_counts = defaultdict(int)
        samples_checked = 0

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS):
                    try:
                        with Image.open(os.path.join(root, file)) as img:
                            tensor = transforms.ToTensor()(img)
                            height, width = tensor.shape[1], tensor.shape[2]
                            channels = tensor.shape[0]

                            size_counts[(width, height)] += 1
                            channel_counts[channels] += 1
                            samples_checked += 1

                            if samples_checked >= 50:
                                break
                    except Exception as e:
                        logger.warning(f"Could not process image {file}: {str(e)}")
                        continue

            if samples_checked >= 50:
                break

        if not size_counts:
            raise ValueError(f"No valid images found in {folder_path}")

        input_size = max(size_counts.items(), key=lambda x: x[1])[0]
        in_channels = max(channel_counts.items(), key=lambda x: x[1])[0]

        return input_size, in_channels


    def _process_torchvision(self) -> Tuple[str, str]:
        """Process torchvision dataset"""
        dataset_name = self.datafile.upper()
        if not hasattr(datasets, dataset_name):
            raise ValueError(f"Torchvision dataset {dataset_name} not found")

        # Setup paths in dataset-specific directory
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Download and process datasets
        transform = transforms.ToTensor()

        train_dataset = getattr(datasets, dataset_name)(
            root=self.output_dir,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = getattr(datasets, dataset_name)(
            root=self.output_dir,
            train=False,
            download=True,
            transform=transform
        )

        # Save images with class directories
        def save_dataset_images(dataset, output_dir, split_name):
            logger.info(f"Processing {split_name} split...")

            class_to_idx = getattr(dataset, 'class_to_idx', None)
            if class_to_idx:
                idx_to_class = {v: k for k, v in class_to_idx.items()}

            with tqdm(total=len(dataset), desc=f"Saving {split_name} images") as pbar:
                for idx, (img, label) in enumerate(dataset):
                    class_name = idx_to_class[label] if class_to_idx else str(label)
                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)

                    if isinstance(img, torch.Tensor):
                        img = transforms.ToPILImage()(img)

                    img_path = os.path.join(class_dir, f"{idx}.png")
                    img.save(img_path)
                    pbar.update(1)

        save_dataset_images(train_dataset, train_dir, "training")
        save_dataset_images(test_dataset, test_dir, "test")

        return train_dir, test_dir


    def _create_train_test_split(self, source_dir: str, test_size: float) -> Tuple[str, str]:
        """Create train/test split from source directory"""
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for class_name in tqdm(os.listdir(source_dir), desc="Processing classes"):
            class_path = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            # Create class directories
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Get all image files
            image_files = [f for f in os.listdir(class_path)
                         if f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)]

            # Random split
            random.shuffle(image_files)
            split_idx = int((1 - test_size) * len(image_files))
            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]

            # Copy files
            for fname in train_files:
                shutil.copy2(
                    os.path.join(class_path, fname),
                    os.path.join(train_class_dir, fname)
                )

            for fname in test_files:
                shutil.copy2(
                    os.path.join(class_path, fname),
                    os.path.join(test_class_dir, fname)
                )

        return train_dir, test_dir


# ----------------------------
# 1. Dynamic CNN Model
# ----------------------------

class DynamicCNN(nn.Module):
    """Core feature extraction model with adaptive depth"""
    def __init__(self, input_channels=3, feature_dim=128, min_size=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.min_size = min_size
        self.input_channels = input_channels

        # Initial conv block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Dynamically created conv blocks
        self.adaptive_blocks = nn.ModuleList()
        self._build_adaptive_blocks(32)

        # Calculate flattened dimension after all conv blocks
        self.flatten_dim = self._get_flatten_dim()

        # Final projection
        self.fc = nn.Linear(self.flatten_dim, feature_dim)

    def _build_adaptive_blocks(self, in_channels):
        """Dynamically adds conv blocks based on input size"""
        dummy = torch.zeros(1, self.input_channels, 64, 64)
        x = self.initial_conv(dummy)
        spatial_dim = x.shape[2]

        while spatial_dim > self.min_size:
            out_channels = in_channels * 2
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.adaptive_blocks.append(block)
            spatial_dim = spatial_dim // 2
            in_channels = out_channels

    def _get_flatten_dim(self):
        """Calculates flattened dimension for FC layer"""
        dummy = torch.zeros(1, self.input_channels, 64, 64)
        x = self.initial_conv(dummy)
        for block in self.adaptive_blocks:
            x = block(x)
        return torch.flatten(x, 1).shape[1]

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.adaptive_blocks:
            x = block(x)
        x = torch.flatten(x, 1)
        return F.normalize(self.fc(x), p=2, dim=1)

# ----------------------------
# 2. Orthogonal Loss Functions
# ----------------------------

class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss"""
    def __init__(self, margin=0.5, scale=30):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, z, labels):
        # Cosine similarity
        cos_theta = torch.matmul(z, z.T)

        # Apply angular margin
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        theta = torch.acos(torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7))
        cos_target_theta = torch.cos(theta + self.margin * mask)
        logits = self.scale * torch.where(mask == 1, cos_target_theta, cos_theta)

        return F.cross_entropy(logits, labels)

class OrthogonalLoss(nn.Module):
    """Combined feature orthogonality and classification loss"""
    def __init__(self, ortho_lambda=0.1, margin=0.5, scale=30):
        super().__init__()
        self.ortho_lambda = ortho_lambda
        self.arcface = ArcFaceLoss(margin=margin, scale=scale)

    def forward(self, z, labels):
        # Classification loss
        arc_loss = self.arcface(z, labels)

        # Feature orthogonality penalty
        z_centered = z - z.mean(dim=0)
        cov = torch.matmul(z_centered.T, z_centered) / (z.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        ortho_loss = torch.norm(off_diag, p='fro')

        return arc_loss + self.ortho_lambda * ortho_loss

# ----------------------------
# 3. Integrated Feature Extractor
# ----------------------------

class DynamicFeatureExtractor(nn.Module):
    """Main model class replacing autoencoder, maintaining original interface"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dims = feature_dims
        self.config = config
        self.device = torch.device('cuda' if config['execution_flags']['use_gpu'] and torch.cuda.is_available() else 'cpu')

        # Core CNN
        self.cnn = DynamicCNN(
            input_channels=input_shape[0],
            feature_dim=feature_dims,
            min_size=8
        )

        # Loss function
        self.loss_fn = OrthogonalLoss(
            ortho_lambda=config['model'].get('ortho_lambda', 0.1),
            margin=config['model'].get('margin', 0.5),
            scale=config['model'].get('scale', 30)
        )

        # Checkpoint setup
        self.checkpoint_dir = config['training']['checkpoint_dir']
        self.dataset_name = config['dataset']['name']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Maintains original output format for compatibility"""
        features = self.cnn(x)
        return {
            'embedding': features,
            'features': features  # Alias
        }

    def extract_features(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Identical interface to original autoencoder's feature extraction"""
        self.eval()
        all_features = []
        all_labels = []
        all_filenames = []
        all_indices = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc="Extracting features")):
                inputs = inputs.to(self.device)
                features = self.cnn(inputs)

                all_features.append(features.cpu())
                all_labels.append(labels.cpu())

                # Handle metadata if available
                if hasattr(loader.dataset, 'get_additional_info'):
                    for i in range(len(inputs)):
                        idx, fname = loader.dataset.get_additional_info(batch_idx * loader.batch_size + i)
                        all_indices.append(idx)
                        all_filenames.append(fname)

        return {
            'embeddings': torch.cat(all_features),
            'labels': torch.cat(all_labels),
            'indices': all_indices if all_indices else [],
            'filenames': all_filenames if all_filenames else []
        }

    def save_features(self, features: Dict[str, torch.Tensor], output_path: str) -> None:
        """Maintains original CSV output format exactly"""
        feature_array = features['embeddings'].numpy()
        labels = features['labels'].numpy()

        # Build DataFrame matching original structure
        data = {
            'original_filename': [os.path.basename(f) for f in features.get('filenames', [])],
            'filepath': features.get('filenames', []),
            'label_type': ['true' if 'labels' in features else 'predicted'] * len(feature_array),
            'target': labels,
            'cluster_assignment': ['NA'] * len(feature_array),
            'cluster_confidence': ['NA'] * len(feature_array)
        }

        # Add feature columns
        for i in range(self.feature_dims):
            data[f'feature_{i}'] = feature_array[:, i]

        # Add indices if available
        if features.get('indices'):
            data['index'] = features['indices']

        pd.DataFrame(data).to_csv(output_path, index=False)

# ----------------------------
# 4. Model Factory
# ----------------------------

class ModelFactory:
    """Creates appropriate model based on config"""

    @staticmethod
    def create_model(config: Dict) -> nn.Module:
        input_shape = (
            config['dataset']['in_channels'],
            config['dataset']['input_size'][0],
            config['dataset']['input_size'][1]
        )
        feature_dims = config['model']['feature_dims']

        return DynamicFeatureExtractor(input_shape, feature_dims, config)

# ----------------------------
# 5. Training Manager
# ----------------------------

class TrainingManager:
    """Handles model training with original checkpointing"""

    def __init__(self, config: Dict):
        self.config = config
        self.checkpoint_dir = config['training']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, model: nn.Module, train_loader: DataLoader) -> Dict:
        """Single-phase training with original logging"""
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['model']['learning_rate']
        )

        history = {'loss': []}
        best_loss = float('inf')

        for epoch in range(self.config['training']['epochs']):
            model.train()
            epoch_loss = 0.0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)

                optimizer.zero_grad()
                features = model.cnn(inputs)
                loss = model.loss_fn(features, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            history['loss'].append(avg_loss)

            # Original checkpointing logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss,
                    'config': self.config
                }, os.path.join(self.checkpoint_dir, 'best_model.pth'))

        return history

# ----------------------------
# 6. Prediction Manager
# ----------------------------

class PredictionManager:
    """Maintains original prediction interface and CSV output"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if config['execution_flags']['use_gpu'] and torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        """Loads trained model with original checkpoint handling"""
        model = ModelFactory.create_model(self.config)
        model.to(self.device)

        checkpoint_path = os.path.join(
            self.config['training']['checkpoint_dir'],
            'best_model.pth'
        )

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {checkpoint_path}")

        return model

    def predict_images(self, data_path: str, output_csv: str = None) -> None:
        """Maintains original CSV output format"""
        features = self._extract_features(data_path)

        # Default output path if not specified
        if output_csv is None:
            output_csv = os.path.join(
                'data',
                self.config['dataset']['name'],
                f"{self.config['dataset']['name']}.csv"
            )

        # Save in original format
        self.model.save_features(features, output_csv)
        logger.info(f"Predictions saved to {output_csv}")

    def _extract_features(self, data_path: str) -> Dict[str, torch.Tensor]:
        """Handles image loading and feature extraction"""
        transform = transforms.Compose([
            transforms.Resize(self.config['dataset']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['dataset']['mean'],
                std=self.config['dataset']['std']
            )
        ])

        # Create temporary dataset
        dataset = CustomImageDataset(data_path, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )

        return self.model.extract_features(loader)

# ----------------------------
# 7. Dataset Handling (Unchanged)
# ----------------------------

class CustomImageDataset(Dataset):
    """Original dataset class maintained for compatibility"""
    def __init__(self, data_dir: str, transform=None, config: Dict = None):
        self.data_dir = data_dir
        self.transform = transform
        self.config = config or {}

        # Original implementation
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_additional_info(self, idx):
        """Maintains original metadata interface"""
        path, _ = self.samples[idx]
        return idx, os.path.basename(path)

# ----------------------------
# 8. Configuration Management
# ----------------------------

class ConfigManager:
    """Generates config files matching original format"""

    @staticmethod
    def generate_config(dataset_name: str, input_size: Tuple[int, int], num_classes: int) -> Dict:
        """Produces config identical to original but for CNN"""
        return {
            "dataset": {
                "name": dataset_name,
                "in_channels": 3,
                "num_classes": num_classes,
                "input_size": list(input_size),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "train_dir": f"data/{dataset_name}/train",
                "test_dir": f"data/{dataset_name}/test"
            },
            "model": {
                "feature_dims": 128,
                "learning_rate": 0.001,
                "ortho_lambda": 0.1,
                "margin": 0.5,
                "scale": 30
            },
            "training": {
                "batch_size": 32,
                "epochs": 100,
                "checkpoint_dir": f"data/{dataset_name}/checkpoints",
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                }
            },
            "execution_flags": {
                "use_gpu": torch.cuda.is_available(),
                "mixed_precision": True
            }
        }

# ----------------------------
# 9. Main Workflow
# ----------------------------

def run_full_workflow(data_path: str, output_dir: str = "output"):
    """Complete workflow matching original functionality"""

    # 1. Dataset processing
    processor = DatasetProcessor(data_path)
    train_dir, test_dir = processor.process()

    # 2. Config generation
    input_size, in_channels = processor._detect_image_properties(train_dir)
    num_classes = len(os.listdir(train_dir))
    config = ConfigManager.generate_config(
        processor.dataset_name,
        input_size,
        num_classes
    )

    # 3. Training
    train_dataset = CustomImageDataset(train_dir, transform=processor.get_transforms(config))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    model = ModelFactory.create_model(config)
    trainer = TrainingManager(config)
    history = trainer.train(model, train_loader)

    # 4. Prediction/Feature Extraction
    predictor = PredictionManager(config)
    predictor.predict_images(test_dir if test_dir else train_dir)

    return history

def main():
    parser = argparse.ArgumentParser(description="Dynamic CNN Feature Extractor")
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory or file')
    parser.add_argument('--output', type=str, default=None, help='Output directory path')
    parser.add_argument('--config', type=str, default=None, help='Custom config file path')
    parser.add_argument('--feature-dims', type=int, default=128, help='Dimension of output features')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                       help='Operation mode: "train" or "predict"')
    parser.add_argument('--datatype', type=str, default='local', choices=['torchvision', 'local'],
                       help='Dataset type: "torchvision" for built-in datasets or "local" for custom data')

    args = parser.parse_args()

    # Initialize with either custom or default config
    config = initialize_config(args)

    # Setup output directory structure
    output_dir = setup_output_directory(args, config)

    # Dataset processing - explicitly set datatype to local
    processor = DatasetProcessor(args.data, datatype='local', config=config)
    train_dir, test_dir = processor.process()

    # Update config with dataset-specific parameters
    update_config_with_dataset_info(config, processor, train_dir)

    # Initialize model
    model = ModelFactory.create_model(config)
    logger.info(f"Created {model.__class__.__name__} with {sum(p.numel() for p in model.parameters()):,} parameters")

    if args.mode == 'train':
        # Data loading for training
        train_loader, test_loader = create_data_loaders(config, train_dir, test_dir)

        # Training
        trainer = TrainingManager(config)
        history = trainer.train(model, train_loader)

        # Save the trained model
        torch.save(model.state_dict(), os.path.join(config['output']['model_dir'], 'final_model.pth'))
        logger.info("Training complete! Model saved.")

    elif args.mode == 'predict':
        # Load the trained model
        model_path = os.path.join(config['output']['model_dir'], 'final_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=model.device))
            logger.info(f"Loaded trained model from {model_path}")
        else:
            raise FileNotFoundError(f"No trained model found at {model_path}. Please train first or check path.")

    # Feature extraction and saving (done for both modes)
    train_loader, test_loader = create_data_loaders(config, train_dir, test_dir)
    extract_and_save_features(model, train_loader, test_loader, config, output_dir)

    # Generate configuration files
    generate_config_files(config, processor, output_dir)

    logger.info("Processing complete!")

def initialize_config(args):
    """Initialize configuration from file or create default"""
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = {
            "dataset": {
                "name": os.path.basename(args.data) if os.path.isdir(args.data) else Path(args.data).stem,
                "in_channels": 3,
                "input_size": [128, 128],  # Will be updated during processing
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "model": {
                "feature_dims": args.feature_dims,
                "learning_rate": 0.001,
                "ortho_lambda": 0.1,
                "margin": 0.5,
                "scale": 30
            },
            "training": {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "checkpoint_dir": "checkpoints",
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                }
            },
            "execution_flags": {
                "use_gpu": torch.cuda.is_available(),
                "mixed_precision": True
            }
        }
    return config

def setup_output_directory(args, config):
    """Create output directory structure"""
    output_dir = args.output if args.output else os.path.join("data", config['dataset']['name'])
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    config['training']['checkpoint_dir'] = os.path.join(output_dir, "checkpoints")
    config['output'] = {
        "features_file": os.path.join(output_dir, f"{config['dataset']['name']}.csv"),
        "model_dir": os.path.join(output_dir, "models"),
        "visualization_dir": os.path.join(output_dir, "visualizations")
    }

    for path in [config['training']['checkpoint_dir'],
                config['output']['model_dir'],
                config['output']['visualization_dir']]:
        os.makedirs(path, exist_ok=True)

    return output_dir

def update_config_with_dataset_info(config, processor, train_dir):
    """Update config with actual dataset parameters"""
    input_size, in_channels = processor._detect_image_properties(train_dir)
    config['dataset']['input_size'] = list(input_size)
    config['dataset']['in_channels'] = in_channels

    # Detect number of classes
    if os.path.isdir(train_dir):
        config['dataset']['num_classes'] = len([d for d in os.listdir(train_dir)
                                            if os.path.isdir(os.path.join(train_dir, d))])
    else:
        config['dataset']['num_classes'] = 1  # For single file input

def create_data_loaders(config, train_dir, test_dir):
    """Create data loaders with proper transforms"""
    transform = transforms.Compose([
        transforms.Resize(config['dataset']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['dataset']['mean'],
            std=config['dataset']['std']
        )
    ])

    train_dataset = CustomImageDataset(train_dir, transform=transform, config=config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=min(4, os.cpu_count())
    )

    test_loader = None
    if test_dir:
        test_dataset = CustomImageDataset(test_dir, transform=transform, config=config)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=min(4, os.cpu_count())
        )

    return train_loader, test_loader

def extract_and_save_features(model, train_loader, test_loader, config, output_dir):
    """Handle feature extraction and CSV generation"""
    # Extract features
    logger.info("Extracting training features...")
    train_features = model.extract_features(train_loader)

    test_features = None
    if test_loader:
        logger.info("Extracting test features...")
        test_features = model.extract_features(test_loader)

    # Save features
    features_path = config['output']['features_file']
    logger.info(f"Saving features to {features_path}")

    if test_features:
        # Combine train and test features if both exist
        combined_features = {
            'embeddings': torch.cat([train_features['embeddings'], test_features['embeddings']]),
            'labels': torch.cat([train_features['labels'], test_features['labels']]),
            'filenames': train_features['filenames'] + test_features['filenames']
        }
        model.save_features(combined_features, features_path)
    else:
        model.save_features(train_features, features_path)

def generate_config_files(config, processor, output_dir):
    """Generate all configuration files"""
    # Main config
    config_path = os.path.join(output_dir, f"{config['dataset']['name']}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Dataset config
    dataset_conf = {
        "file_path": config['output']['features_file'],
        "column_names": [f"feature_{i}" for i in range(config['model']['feature_dims'])] + ["target"],
        "separator": ",",
        "has_header": True,
        "target_column": "target",
        "modelType": "Histogram",
        "feature_group_size": 2,
        "max_combinations": 10000,
        "bin_sizes": [128],
        "training_params": {
            "trials": 100,
            "epochs": 1000,
            "learning_rate": 0.001,
            "batch_size": config['training']['batch_size'],
            "enable_adaptive": True,
            "invert_DBNN": config['training'].get('invert_DBNN', False)
        }
    }

    conf_path = os.path.join(output_dir, f"{config['dataset']['name']}.conf")
    with open(conf_path, 'w') as f:
        json.dump(dataset_conf, f, indent=4)

    logger.info(f"Configuration files saved to {output_dir}")

if __name__ == "__main__":
    main()

