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

# -------------------- Model Architecture --------------------
class UniversalFeatureExtractor(nn.Module):
    """Handles both 1D (EEG) and 2D (image) inputs"""
    def __init__(self, input_dims: Tuple, feature_dim: int = 128):
        super().__init__()
        if len(input_dims) == 1:  # 1D data
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, stride=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, feature_dim)
        else:  # Image data
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dims[0], 32, kernel_size=3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, feature_dim))

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2 and len(self.input_dims) == 1:  # Add channel dim for 1D
            x = x.unsqueeze(1)
        features = self.encoder(x)
        return self.projector(features)

class PrototypeClusterer(nn.Module):
    """Learnable prototype-based clustering"""
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = torch.matmul(F.normalize(features, dim=1),
                          F.normalize(self.prototypes, dim=1).T)
        probs = F.softmax(sim / self.temperature.clamp(min=0.1), dim=1)
        return torch.argmax(probs, dim=1), probs

# -------------------- Configuration Management --------------------
class ConfigManager:
    @staticmethod
    def generate_conf_file(output_path: Path, feature_dim: int, dataset_name: str):
        conf = {
            "file_path": str(output_path/f"{dataset_name}.csv"),
            "column_names": [f"feature_{i}" for i in range(feature_dim)] + ["target"],
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
        base_config = {
            "dataset": {
                "name": dataset_name,
                "type": "image_folder" if config["data_type"] == "image" else "1d_signal",
                "in_channels": 3 if config["data_type"] == "image" else 1,
                "num_classes": config["n_classes"],
                "input_size": [224, 224] if config["data_type"] == "image" else [1000],
                "mean": [0.485, 0.456, 0.406] if config["data_type"] == "image" else [0.0],
                "std": [0.229, 0.224, 0.225] if config["data_type"] == "image" else [1.0],
                "train_dir": str(output_path/"train"),
                "test_dir": str(output_path/"test") if (output_path/"test").exists() else None
            },
            "model": {
                "encoder_type": "universal",
                "feature_dims": config["feature_dim"],
                "learning_rate": config["lr"],
                "prototype_learning_rate": config["prototype_lr"],
                "temperature": 1.0,
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
                "features_file": str(output_path/f"{dataset_name}.csv"),
                "model_dir": str(output_dir/"models"),
                "visualization_dir": str(output_dir/"visualizations")
            }
        }
        with open(output_path/f"{dataset_name}.json", "w") as f:
            json.dump(base_config, f, indent=4)

# -------------------- Main Pipeline --------------------
class DeepClusteringPipeline:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = None
        self.model = None
        self.clusterer = None
        self.optimizer = None
        self.dataset_name = "dataset"

    def _default_config(self) -> Dict:
        return {
            "feature_dim": 128,
            "lr": 0.001,
            "prototype_lr": 0.01,
            "epochs": 20,
            "batch_size": 32,
            "save_dir": "data"
        }

    def initialize_from_data(self, data_path: str):
        """Initialize from directory structure"""
        data_path = Path(data_path)
        self.dataset_name = data_path.name

        # Detect classes
        if (data_path/"train").exists():
            class_dirs = list((data_path/"train").iterdir())
        else:
            class_dirs = list(data_path.iterdir())

        if not class_dirs:
            raise ValueError("No class directories found")

        # Initialize label encoder
        classes = sorted([d.name for d in class_dirs if d.is_dir()])
        self.label_encoder = LabelEncoder().fit(classes)
        self.config["n_classes"] = len(classes)

        # Detect data type
        sample_file = next((f for d in class_dirs for f in d.iterdir() if f.is_file()), None)
        if not sample_file:
            raise ValueError("No data files found")
        self.config["data_type"] = "image" if sample_file.suffix.lower() in ('.jpg','.png','.jpeg') else "1d"

        # Initialize model
        input_dims = (3, 224, 224) if self.config["data_type"] == "image" else (1,)
        self.model = UniversalFeatureExtractor(input_dims, self.config["feature_dim"]).to(self.device)
        self.clusterer = PrototypeClusterer(self.config["feature_dim"], self.config["n_classes"]).to(self.device)
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.clusterer.parameters(), 'lr': self.config["prototype_lr"]}
        ], lr=self.config["lr"])

    def train(self, data_path: str):
        """Complete training workflow"""
        self.initialize_from_data(data_path)
        output_dir = Path(self.config["save_dir"])/self.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Dummy training loop (replace with actual implementation)
        print(f"Training with {self.config['n_classes']} classes...")
        for epoch in tqdm(range(self.config["epochs"])):
            pass  # Real training would happen here

        # Save artifacts
        self._save_model(output_dir)
        self._save_features(output_dir)

        # Generate config files
        ConfigManager.generate_conf_file(output_dir, self.config["feature_dim"], self.dataset_name)
        ConfigManager.generate_json_config(output_dir, self.config, self.dataset_name)

        print(f"\nTraining complete. Results saved to:\n{output_dir}")

    def _save_model(self, output_dir: Path):
        """Save model and label encoder"""
        model_dir = output_dir/"models"
        model_dir.mkdir(exist_ok=True)

        torch.save({
            'model_state': self.model.state_dict(),
            'clusterer_state': self.clusterer.state_dict(),
            'config': self.config
        }, model_dir/"model.pth")

        with open(model_dir/"label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

    def _save_features(self, output_dir: Path):
        """Save features to CSV"""
        # Generate dummy features (replace with real features)
        features = np.random.randn(100, self.config["feature_dim"])
        labels = np.random.randint(0, self.config["n_classes"], 100)

        df = pd.DataFrame(
            {f"feature_{i}": features[:,i] for i in range(features.shape[1])},
            columns=[f"feature_{i}" for i in range(features.shape[1])]
        )
        df["target"] = labels
        df["file_path"] = [f"sample_{i}.ext" for i in range(100)]

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
            choice = input(f"Found existing model. Train anyway? [y/N]: ").lower()
            if choice == 'y':
                params["mode"] = "train"

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

        # Initialize pipeline
        config = {
            "save_dir": output_dir,
            "feature_dim": 128,
            "epochs": 20
        }
        self.pipeline = DeepClusteringPipeline(config)

        # Run pipeline
        try:
            if mode == "train":
                print(f"\nStarting training on {data_path}...")
                self.pipeline.train(data_path)
            else:
                print("\nPrediction mode not yet implemented")
        except Exception as e:
            print(f"\nError: {str(e)}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    DeepClusteringApp().run()
