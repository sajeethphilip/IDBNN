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

class PrototypeClusterer(nn.Module):
    """Enhanced prototype clustering with temperature scaling"""
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.temperature = nn.Parameter(torch.tensor(0.5))  # Start with lower temperature
        self.epsilon = 1e-6  # Small constant for numerical stability

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalized prototype similarity
        norm_features = F.normalize(features, dim=1)
        norm_prototypes = F.normalize(self.prototypes, dim=1)
        sim = torch.matmul(norm_features, norm_prototypes.T)

        # Temperature-scaled probabilities
        probs = F.softmax(sim / self.temperature.clamp(min=self.epsilon), dim=1)
        return torch.argmax(probs, dim=1), probs

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
        with open(output_path/f"{dataset_name}.conf", "w") as f:
            json.dump(conf, f, indent=4)

    @staticmethod
    def generate_json_config(output_path: Path, config: Dict, dataset_name: str):
        # Convert Path objects to strings
        base_config = {
            "dataset": {
                "name": dataset_name,
                "type": "image_folder",
                "in_channels": 3,
                "num_classes": config["n_classes"],
                "input_size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "train_dir": str(output_path),  # Convert to string
                "test_dir": None
            },
            "model": {
                "encoder_type": "deep_cnn",
                "feature_dims": config["feature_dim"],
                "learning_rate": config["lr"],
                "prototype_learning_rate": config["prototype_lr"],
                "temperature": 0.5,
                "dropout_prob": config.get("dropout_prob", 0.3),
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
            "training": config,
            "output": {
                "features_file": str(output_path/f"{dataset_name}.csv"),  # Convert to string
                "model_dir": str(output_path/"models"),  # Convert to string
                "visualization_dir": str(output_path/"visualizations")  # Convert to string
            }
        }
        with open(output_path/f"{dataset_name}.json", "w") as f:
            json.dump(base_config, f, indent=4)
# -------------------- Enhanced Training Pipeline --------------------
class DeepClusteringPipeline:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = None
        self.model = None
        self.clusterer = None
        self.optimizer = None
        self.scheduler = None
        self.dataset_name = "dataset"
        self.best_loss = float('inf')
        self.patience_counter = 0

    def _default_config(self) -> Dict:
        return {
            "feature_dim": 128,
            "lr": 0.001,
            "prototype_lr": 0.01,
            "dropout_prob": 0.3,
            "epochs": 50,
            "batch_size": 32,
            "patience": 5,
            "min_delta": 0.001,
            "save_dir": "data"
        }

    def initialize_from_data(self, data_path: str):
        """Initialize from directory structure"""
        data_path = Path(data_path)
        self.dataset_name = data_path.name

        # Create dataset to detect classes
        try:
            dataset = ImageDataset(data_path)
            self.label_encoder = dataset.label_encoder
            self.config["n_classes"] = len(dataset.classes)
            self.config["data_type"] = "image"
        except Exception as e:
            raise ValueError(f"Failed to initialize from data: {str(e)}")

        # Initialize model with deeper architecture
        input_dims = (3, 224, 224)
        self.model = DeepFeatureExtractor(
            input_dims,
            self.config["feature_dim"],
            dropout_prob=self.config["dropout_prob"]
        ).to(self.device)

        self.clusterer = PrototypeClusterer(
            self.config["feature_dim"],
            self.config["n_classes"]
        ).to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.clusterer.parameters(), 'lr': self.config["prototype_lr"]}
        ], lr=self.config["lr"])

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )

    def train(self, data_path: str):
        """Enhanced training with early stopping and model checkpointing"""
        try:
            self.initialize_from_data(data_path)
            output_dir = Path(self.config["save_dir"])/self.dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nTraining Configuration:")
            print(f"- Dataset: {self.dataset_name}")
            print(f"- Classes: {list(self.label_encoder.classes_)}")
            print(f"- Number of classes: {self.config['n_classes']}")
            print(f"- Feature dimension: {self.config['feature_dim']}")
            print(f"- Output directory: {output_dir}")

            # Create data loader
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            dataset = ImageDataset(data_path, transform=transform)
            loader = DataLoader(
                dataset,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            # Training loop with early stopping
            print("\nTraining model...")
            for epoch in range(self.config["epochs"]):
                self.model.train()
                self.clusterer.train()
                epoch_loss = 0.0

                for batch_idx, (images, labels, _) in enumerate(tqdm(loader,
                    desc=f"Epoch {epoch+1}/{self.config['epochs']}")):

                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    # Forward pass
                    features = self.model(images)
                    assignments, probs = self.clusterer(features)

                    # Loss calculation
                    loss = F.cross_entropy(probs, labels)

                    # Backward pass
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                # Calculate average epoch loss
                avg_loss = epoch_loss / len(loader)
                self.scheduler.step(avg_loss)

                # Early stopping check
                if avg_loss < self.best_loss - self.config["min_delta"]:
                    self.best_loss = avg_loss
                    self.patience_counter = 0
                    # Save best model
                    self._save_model(output_dir, best=True)
                else:
                    self.patience_counter += 1

                print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} (Best: {self.best_loss:.4f})")
                print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

                if self.patience_counter >= self.config["patience"]:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            # Load best model before saving features
            self._load_best_model(output_dir)

            # Save final artifacts
            self._save_features(output_dir, dataset)
            ConfigManager.generate_conf_file(output_dir, self.config["feature_dim"], self.dataset_name)
            ConfigManager.generate_json_config(output_dir, self.config, self.dataset_name)

            print(f"\nTraining complete. Results saved to:")
            print(f"- Features CSV: {output_dir}/{self.dataset_name}.csv")
            print(f"- Model: {output_dir}/models/model.pth")
            print(f"- Label encoder: {output_dir}/models/label_encoder.pkl")
            print(f"- Config files: {output_dir}/{self.dataset_name}.{{conf,json}}")

        except Exception as e:
            print(f"\nError during training: {str(e)}", file=sys.stderr)
            raise

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

    def _save_features(self, output_dir: Path, dataset: ImageDataset):
        """Extract and save features to CSV with cluster assignments"""
        loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4
        )

        self.model.eval()
        self.clusterer.eval()

        all_features = []
        all_labels = []
        all_paths = []
        all_assignments = []
        all_confidences = []

        with torch.no_grad():
            for images, labels, paths in tqdm(loader, desc="Extracting features"):
                features = self.model(images.to(self.device))
                assignments, probs = self.clusterer(features)
                confidences = probs.gather(1, assignments.unsqueeze(1)).squeeze()

                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
                all_paths.extend(paths)
                all_assignments.append(assignments.cpu().numpy())
                all_confidences.append(confidences.cpu().numpy())

        # Concatenate all batches
        features = np.concatenate(all_features)
        labels = np.concatenate(all_labels)
        assignments = np.concatenate(all_assignments)
        confidences = np.concatenate(all_confidences)

        # Create DataFrame with all information
        feature_cols = {f"feature_{i}": features[:,i] for i in range(features.shape[1])}
        df = pd.DataFrame({
            **feature_cols,
            "target": labels,
            "file_path": all_paths,
            "assigned_class": assignments,
            "assigned_class_label": self.label_encoder.inverse_transform(assignments),
            "confidence": confidences,
            "class_name": self.label_encoder.inverse_transform(labels)
        })

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir/f"{self.dataset_name}.csv", index=False)

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

if __name__ == "__main__":
    DeepClusteringApp().run()
