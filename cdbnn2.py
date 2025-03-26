import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
import pickle
from sklearn.preprocessing import LabelEncoder

class ConfigManager:
    """Handles configuration files generation"""
    @staticmethod
    def generate_conf_file(output_path: Path, feature_dim: int, dataset_name: str):
        """Generate the .conf file with feature columns"""
        conf = {
            "file_path": str(output_path/"features.csv"),
            "column_names": [f"feature_{i}" for i in range(feature_dim)] + ["target"],
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

        with open(output_path/f"{dataset_name}.conf", "w") as f:
            json.dump(conf, f, indent=4)

    @staticmethod
    def generate_json_config(output_path: Path, config: Dict, dataset_name: str):
        """Generate comprehensive JSON config"""
        base_config = {
            "dataset": {
                "name": dataset_name,
                "type": "image_folder" if config["data_type"] == "image" else "1d_signal",
                "in_channels": 3 if config["data_type"] == "image" else 1,
                "num_classes": config["n_classes"],
                "input_size": [224, 224] if config["data_type"] == "image" else [config.get("signal_length", 1000)],
                "mean": [0.485, 0.456, 0.406] if config["data_type"] == "image" else [0.0],
                "std": [0.229, 0.224, 0.225] if config["data_type"] == "image" else [1.0],
                "train_dir": str(output_path/"train"),
                "test_dir": str(output_path/"test") if (output_path/"test").exists() else None
            },
            "model": {
                "encoder_type": "prototype_cluster",
                "feature_dims": config["feature_dim"],
                "learning_rate": config["lr"],
                "prototype_learning_rate": config["prototype_lr"],
                "cluster_temperature": 1.0,
                "enhancement_modules": {
                    "prototype_clustering": {
                        "enabled": True,
                        "components": {
                            "feature_projection": True,
                            "cluster_assignment": True,
                            "temperature_scaling": True
                        },
                        "weights": {
                            "cluster_loss_weight": 1.0,
                            "uniformity_weight": 0.1
                        }
                    }
                }
            },
            "training": {
                "batch_size": config["batch_size"],
                "epochs": config["epochs"],
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                }
            },
            "output": {
                "features_file": str(output_path/"features.csv"),
                "model_dir": str(output_path/"models"),
                "visualization_dir": str(output_path/"visualizations")
            }
        }

        with open(output_path/f"{dataset_name}.json", "w") as f:
            json.dump(base_config, f, indent=4)

class EnhancedClusteringPipeline:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = None
        self.dataset_name = "dataset"  # Will be updated from data path

    def _prepare_output_structure(self, output_path: Union[str, Path]):
        """Create necessary output directories"""
        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_path/"models", exist_ok=True)
        os.makedirs(output_path/"visualizations", exist_ok=True)
        return output_path

    def _save_features_csv(self, features: np.ndarray, labels: np.ndarray, paths: List[str], output_path: Path):
        """Save features as CSV with individual feature columns"""
        # Create DataFrame with feature columns
        feature_cols = {f"feature_{i}": features[:, i] for i in range(features.shape[1])}
        df = pd.DataFrame({
            **feature_cols,
            "target": labels,
            "file_path": paths,
            "assigned_class": getattr(self, 'cluster_assignments', [-1]*len(labels)),
            "assigned_class_label": getattr(self, 'cluster_labels', ["unknown"]*len(labels))
        })

        # Save to CSV
        df.to_csv(output_path/"features.csv", index=False)

        # Generate configuration files
        ConfigManager.generate_conf_file(output_path, features.shape[1], self.dataset_name)
        ConfigManager.generate_json_config(output_path, self.config, self.dataset_name)

    def train(self, data_path: str):
        """Train with automatic configuration generation"""
        # Initialize from data structure
        self.initialize_from_data(data_path)

        # Prepare output directory structure
        output_path = self._prepare_output_structure(Path(self.config["save_dir"]))
        self.dataset_name = Path(data_path).name

        # Training logic (simplified)
        features, labels, paths = self._extract_training_features(data_path)

        # Save features and configurations
        self._save_features_csv(features, labels, paths, output_path)

        # Save label encoder
        with open(output_path/"label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

    def predict(self, data_path: str) -> pd.DataFrame:
        """Predict with configuration generation"""
        output_path = self._prepare_output_structure(Path(self.config["save_dir"]))
        self.dataset_name = Path(data_path).name

        # Load label encoder if available
        if (output_path/"label_encoder.pkl").exists():
            with open(output_path/"label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)

        # Extract features
        features, labels, paths = self._extract_features(data_path)

        # Cluster assignments
        if hasattr(self, 'clusterer'):
            assignments = self.clusterer.predict(features)
            self.cluster_assignments = assignments
            if self.label_encoder:
                self.cluster_labels = self.label_encoder.inverse_transform(assignments)

        # Save results
        self._save_features_csv(features, labels, paths, output_path)
        return pd.read_csv(output_path/"features.csv")

    def _extract_training_features(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Dummy feature extraction - replace with actual model"""
        # This would be replaced with actual feature extraction
        n_samples = 100
        features = np.random.randn(n_samples, self.config["feature_dim"])
        labels = np.random.randint(0, self.config["n_classes"], n_samples)
        paths = [f"sample_{i}.{'jpg' if self.config['data_type'] == 'image' else 'npy'}"
                for i in range(n_samples)]
        return features, labels, paths

import argparse
import os
import sys
from pathlib import Path

class DeepClusteringApp:
    def __init__(self):
        self.parser = self._create_parser()
        self.pipeline = None
        self.config = {
            "feature_dim": 128,
            "lr": 0.001,
            "prototype_lr": 0.01,
            "epochs": 20,
            "batch_size": 32
        }

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Deep Clustering Pipeline for Images and 1D Data",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--path", type=str, help="Path to data directory")
        parser.add_argument("--mode", choices=["train", "predict", "auto"], default="auto",
                          help="Operation mode")
        parser.add_argument("--output", type=str, default=None,
                          help="Custom output directory")
        parser.add_argument("--force", action="store_true",
                          help="Overwrite existing files without prompt")
        return parser

    def _get_interactive_input(self) -> Dict:
        """Get parameters through interactive prompts"""
        print("\n==== Deep Clustering Configuration ====")
        params = {}

        # Get data path
        while True:
            path = input("Enter path to data directory: ").strip()
            if Path(path).exists():
                params["path"] = path
                break
            print(f"Error: Path '{path}' does not exist")

        # Determine dataset name
        params["name"] = Path(path).name

        # Check for existing model
        model_path = Path(f"data/{params['name']}/models")
        has_model = model_path.exists() and any(model_path.glob("*.pth"))

        # Determine mode
        if has_model:
            print(f"\nFound existing model in {model_path}")
            mode = input("Choose mode (train/predict) [predict]: ").strip().lower()
            params["mode"] = mode if mode in ("train", "predict") else "predict"
        else:
            print("\nNo existing model found")
            params["mode"] = "train"

        return params

    def _validate_model_exists(self, dataset_name: str) -> bool:
        """Check if trained model exists"""
        model_dir = Path(f"data/{dataset_name}/models")
        required_files = {
            model_dir/"feature_extractor.pth",
            model_dir/"label_encoder.pkl"
        }
        return all(f.exists() for f in required_files)

    def _prepare_output(self, dataset_name: str, force: bool = False) -> Path:
        """Prepare output directory structure"""
        output_dir = Path(f"data/{dataset_name}")

        if output_dir.exists() and not force:
            response = input(f"Output directory {output_dir} exists. Overwrite? [y/N]: ").lower()
            if response != 'y':
                print("Operation cancelled")
                sys.exit(0)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir/"models", exist_ok=True)
        os.makedirs(output_dir/"visualizations", exist_ok=True)
        return output_dir

    def run(self):
        """Main entry point for the application"""
        # Parse command line arguments
        args = self.parser.parse_args()

        # Interactive mode if no arguments provided
        if not any(vars(args).values()):
            params = self._get_interactive_input()
            data_path = params["path"]
            mode = params["mode"]
            dataset_name = params["name"]
            force = False
        else:
            data_path = args.path
            mode = args.mode
            dataset_name = Path(data_path).name
            force = args.force

        # Auto-detect mode if needed
        if mode == "auto":
            mode = "predict" if self._validate_model_exists(dataset_name) else "train"

        # Prepare output directory
        output_dir = self._prepare_output(dataset_name, force)
        self.config["save_dir"] = str(output_dir)

        # Initialize pipeline
        self.pipeline = EnhancedClusteringPipeline(self.config)

        # Run the requested operation
        try:
            if mode == "train":
                print(f"\nStarting training on {data_path}...")
                self.pipeline.train(data_path)
                print(f"\nTraining complete. Results saved to {output_dir}")

                # Generate output files
                self._generate_output_files(dataset_name)

            elif mode == "predict":
                if not self._validate_model_exists(dataset_name):
                    print("\nError: No trained model found. Please train first.")
                    sys.exit(1)

                print(f"\nStarting prediction on {data_path}...")
                results = self.pipeline.predict(data_path)
                print(f"\nPrediction complete. Results saved to {output_dir}")

                # Generate output files
                self._generate_output_files(dataset_name)

        except Exception as e:
            print(f"\nError: {str(e)}", file=sys.stderr)
            sys.exit(1)

    def _generate_output_files(self, dataset_name: str):
        """Generate all required output files"""
        output_dir = Path(f"data/{dataset_name}")

        # Generate CSV with features
        features_path = output_dir/f"{dataset_name}.csv"
        if not features_path.exists():
            # This would actually load and format the features from the pipeline
            features = np.random.randn(100, self.config["feature_dim"])  # Dummy data
            labels = np.random.randint(0, self.config.get("n_classes", 10), 100)
            paths = [f"sample_{i}.ext" for i in range(100)]

            pd.DataFrame({
                **{f"feature_{i}": features[:,i] for i in range(features.shape[1])},
                "target": labels,
                "file_path": paths
            }).to_csv(features_path, index=False)

        # Generate config files if they don't exist
        ConfigManager.generate_conf_file(output_dir, self.config["feature_dim"], dataset_name)
        ConfigManager.generate_json_config(output_dir, self.config, dataset_name)

        print(f"Generated output files:")
        print(f"- {features_path}")
        print(f"- {output_dir/dataset_name}.conf")
        print(f"- {output_dir/dataset_name}.json")

if __name__ == "__main__":
    app = DeepClusteringApp()
    app.run()
