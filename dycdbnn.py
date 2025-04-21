# ----------------------------
# Dynamic CNN Integrated Version
# Maintains original CSV/config outputs
# ----------------------------

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

# ----------------------------
# 1. Dynamic CNN Model
# ----------------------------

class DynamicCNN(nn.Module):
    """Core feature extraction model with adaptive depth"""
    def __init__(self, input_channels=3, feature_dim=128, min_size=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.min_size = min_size

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

        # Final projection
        self.fc = nn.Linear(self._get_flatten_dim(input_channels), feature_dim)

    def _build_adaptive_blocks(self, in_channels):
        """Dynamically adds conv blocks based on input size"""
        dummy = torch.zeros(1, in_channels, 64, 64)
        spatial_dim = self.initial_conv(dummy).shape[2]

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

    def _get_flatten_dim(self, input_channels):
        """Calculates flattened dimension for FC layer"""
        dummy = torch.zeros(1, input_channels, 64, 64)
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
    args = parser.parse_args()

    # Initialize with either custom or default config
    config = initialize_config(args)

    # Setup output directory structure
    output_dir = setup_output_directory(args, config)

    # Dataset processing
    processor = DatasetProcessor(args.data, config=config)
    train_dir, test_dir = processor.process()

    # Update config with dataset-specific parameters
    update_config_with_dataset_info(config, processor, train_dir)

    # Initialize model
    model = ModelFactory.create_model(config)
    logger.info(f"Created {model.__class__.__name__} with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Data loading
    train_loader, test_loader = create_data_loaders(config, train_dir, test_dir)

    # Training
    trainer = TrainingManager(config)
    history = trainer.train(model, train_loader)

    # Feature extraction and saving
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

