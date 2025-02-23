import torch
import argparse
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Union
from collections import defaultdict
import requests
from io import StringIO
import os,re
import json
from itertools import combinations
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import normaltest
import numpy as np
from itertools import combinations
import torch
import os
import pickle
import configparser
import traceback  # Add to provide debug

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from typing import Dict, List, Union, Optional
from collections import defaultdict
import requests
from io import StringIO
#-----------------------------------Optimised  Adaptive Learning--------------------------------------
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import torch.amp

import logging
import argparse
import sys
from dbnn import DatasetProcessor,  InvertibleDBNN



class GlobalConfig:
    """Enhanced GlobalConfig with proper parameter handling"""
    def __init__(self):
        # Basic parameters
        self.learning_rate = None
        self.epochs = None
        self.test_fraction = None
        self.random_seed = None
        self.fresh_start = None
        self.use_previous_model = None
        self.model_type = None
        self.enable_adaptive = None
        self.cardinality_threshold = None
        self.cardinality_tolerance = None
        self.n_bins_per_dim = None
        self.minimum_training_accuracy = None

        # Inverse DBNN parameters
        self.invert_DBNN = False
        self.reconstruction_weight = 0.5
        self.feedback_strength = 0.3
        self.inverse_learning_rate = 0.1

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'GlobalConfig':
        """Create configuration from dictionary with debug tracking"""
        print("\nDEBUG: Creating GlobalConfig from dictionary")
        # print(f"DEBUG:  Input config: {json.dumps(config_dict, indent=2)}")

        config = cls()
        training_params = config_dict.get('training_params', {})
        execution_flags = config_dict.get('execution_flags', {})

        # Load training parameters with debug
        print("\nDEBUG: Loading training parameters:")
        for param, default in [
            ('learning_rate', 0.1),
            ('epochs', 1000),
            ('test_fraction', 0.2),
            ('random_seed', 42),
            ('model_type', 'Histogram'),
            ('enable_adaptive', True),
            ('cardinality_threshold', 0.9),
            ('cardinality_tolerance', 4),
            ('n_bins_per_dim', 20),
            ('minimum_training_accuracy', 0.95),
            ('invert_DBNN', False),
            ('reconstruction_weight', 0.5),
            ('feedback_strength', 0.3),
            ('inverse_learning_rate', 0.1)
        ]:
            value = training_params.get(param, default)
            setattr(config, param, value)
            # print(f"DEBUG:  {param} = {value}")

        # Load execution flags
        #print("\nDEBUGLoading execution flags:")
        config.fresh_start = execution_flags.get('fresh_start', False)
        config.use_previous_model = execution_flags.get('use_previous_model', True)
        # print(f"DEBUG:  fresh_start = {config.fresh_start}")
        # print(f"DEBUG:  use_previous_model = {config.use_previous_model}")

        ##print("\nDEBUGFinal GlobalConfig state:")
        #print(json.dumps(config.to_dict(), indent=2))

        return config

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'training_params': {
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'test_fraction': self.test_fraction,
                'random_seed': self.random_seed,
                'modelType': self.model_type,
                'enable_adaptive': self.enable_adaptive,
                'cardinality_threshold': self.cardinality_threshold,
                'cardinality_tolerance': self.cardinality_tolerance,
                'n_bins_per_dim': self.n_bins_per_dim,
                'minimum_training_accuracy': self.minimum_training_accuracy,
                'invert_DBNN': self.invert_DBNN,
                'reconstruction_weight': self.reconstruction_weight,
                'feedback_strength': self.feedback_strength,
                'inverse_learning_rate': self.inverse_learning_rate
            },
            'execution_flags': {
                'fresh_start': self.fresh_start,
                'use_previous_model': self.use_previous_model
            }
        }

class BinningHandler:
    """Handles binning, scaling, and outlier detection for histogram-based DBNN"""

    def __init__(self, n_bins_per_dim: int = 20, padding_factor: float = 0.01, device=None):
        self.n_bins = n_bins_per_dim
        self.padding_factor = padding_factor
        self.feature_bounds = {}
        self.bin_edges = {}
        self.outliers = []
        self.categorical_features = {}
        self.categorical_mappings = {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to(self, device):
        """Move handler to specified device"""
        self.device = device
        # Move bin edges to device
        for name in self.bin_edges:
            self.bin_edges[name] = self.bin_edges[name].to(device)
        return self

    def setup_categorical_features(self, categorical_encoders: Dict):
        """Setup categorical feature information from DBNN encoders"""
        self.categorical_mappings = {}
        for column, mapping in categorical_encoders.items():
            # Create reverse mapping
            reverse_mapping = {v: k for k, v in mapping.items()}
            self.categorical_mappings[column] = {
                'forward': mapping,
                'reverse': reverse_mapping
            }
            self.categorical_features[column] = {
                'unique_values': list(mapping.values()),
                'original_labels': list(mapping.keys())
            }


    def fit(self, data: torch.Tensor, feature_names: List[str], categorical_encoders: Dict = None):
        """Fit binning parameters with device handling"""
        # Ensure input tensor is on correct device
        data = data.to(self.device).contiguous()

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]

        if categorical_encoders:
            self.setup_categorical_features(categorical_encoders)

        for i, name in enumerate(feature_names):
            if name in self.categorical_features:
                continue

            feature_data = data[:, i]
            min_val = float(feature_data.min().item())
            max_val = float(feature_data.max().item())

            padding = (max_val - min_val) * self.padding_factor
            min_val -= padding
            max_val += padding

            self.feature_bounds[name] = {
                'min': min_val,
                'max': max_val,
                'original_min': float(feature_data.min().item()),
                'original_max': float(feature_data.max().item())
            }

            # Create bin edges on correct device
            self.bin_edges[name] = torch.linspace(
                min_val, max_val, self.n_bins + 1,
                device=self.device
            ).contiguous()

    def transform(self, data: torch.Tensor, feature_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform data with contiguous tensors and optimized operations"""
        # Ensure input tensor is on correct device and contiguous
        data = data.to(self.device).contiguous()

        # Pre-allocate output tensors
        binned_data = torch.zeros_like(data, device=self.device)
        outlier_mask = torch.zeros(data.shape[0], dtype=torch.bool, device=self.device)

        for i, name in enumerate(feature_names):
            feature_data = data[:, i].contiguous()

            if name in self.categorical_features:
                # Handle categorical features
                binned_data[:, i] = feature_data
                valid_values = set(self.categorical_features[name]['unique_values'])
                feature_outliers = torch.tensor(
                    [v.item() not in valid_values for v in feature_data],
                    dtype=torch.bool,
                    device=self.device
                )
            else:
                # Ensure edges are contiguous
                edges = self.bin_edges[name].contiguous()

                # Check for outliers
                below_min = feature_data < edges[0]
                above_max = feature_data > edges[-1]
                feature_outliers = below_min | above_max

                # Clip values to bin range
                feature_data = torch.clamp(feature_data, edges[0], edges[-1])

                # Use bucketize with contiguous tensors
                bin_indices = torch.bucketize(feature_data, edges).sub_(1)
                bin_indices = bin_indices.clamp_(0, self.n_bins - 1)
                binned_data[:, i] = bin_indices

            # Update outlier mask
            outlier_mask |= feature_outliers

        return binned_data, outlier_mask


    def inverse_transform(self, binned_data: torch.Tensor, feature_names: List[str]) -> Tuple[torch.Tensor, pd.DataFrame]:
        """Inverse transform with optimized DataFrame construction"""
        # Ensure input tensor is on correct device and contiguous
        binned_data = binned_data.to(self.device).contiguous()
        # Initialize tensors for original scale
        original_scale = torch.zeros_like(binned_data, dtype=torch.float32, device=self.device)
        # Prepare data for DataFrame construction
        data_dict = {}
        # Process all features at once
        for i, name in enumerate(feature_names):
            if name in self.categorical_features:
                # Handle categorical features
                numeric_values = binned_data[:, i].cpu().numpy()
                categorical_labels = [
                    self.categorical_mappings[name]['reverse'].get(val, 'UNKNOWN')
                    for val in numeric_values
                ]
                data_dict[name] = categorical_labels
                original_scale[:, i] = binned_data[:, i]
            else:
                # Handle numerical features
                edges = self.bin_edges[name].contiguous()
                bin_indices = binned_data[:, i].long()
                bin_centers = (edges[:-1] + edges[1:]) / 2
                numeric_values = bin_centers[bin_indices]
                data_dict[name] = numeric_values.cpu().numpy()
                original_scale[:, i] = numeric_values
        # Create DataFrame all at once
        results_df = pd.DataFrame(data_dict)
        # Ensure DataFrame is defragmented
        results_df = results_df.copy()
        return original_scale, results_df

    # Modify DBNN class to use BinningHandler
    def _compute_pairwise_likelihood_parallel(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """Compute pairwise likelihood with proper binning and scaling"""
        dataset = dataset.to(self.device)
        labels = labels.to(self.device)

        # Initialize binning handler if not exists
        if not hasattr(self, 'binning_handler'):
            self.binning_handler = BinningHandler(
                n_bins_per_dim=self.n_bins_per_dim,
                padding_factor=0.01
            )
            self.binning_handler.fit(dataset, self.feature_columns)

        # Transform data to bin indices and get outlier mask
        binned_data, outlier_mask = self.binning_handler.transform(dataset, self.feature_columns)

        # Store outlier information
        self.outlier_indices = torch.where(outlier_mask)[0].cpu().numpy()

        # Continue with your existing likelihood computation using binned_data
        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)

        # Rest of your existing likelihood computation code...

        return likelihood_params

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def color_value(current_value, previous_value=None, higher_is_better=True):
        """Color a value based on whether it improved or declined"""
        if previous_value is None:
            return f"{current_value:.4f}"

        if higher_is_better:
            if current_value > previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value < previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"
        else:  # lower is better
            if current_value < previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value > previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"

        return f"{current_value:.4f}"  # No color if equal

    @staticmethod
    def highlight_dataset(name):
        """Highlight dataset name in red"""
        return f"{Colors.RED}{name}{Colors.ENDC}"

    @staticmethod
    def highlight_time(time_value):
        """Color time values based on threshold"""
        if time_value < 10:
            return f"{Colors.GREEN}{time_value:.2f}{Colors.ENDC}"
        elif time_value < 30:
            return f"{Colors.YELLOW}{time_value:.2f}{Colors.ENDC}"
        else:
            return f"{Colors.RED}{time_value:.2f}{Colors.ENDC}"

class DatasetConfig:
    """Enhanced dataset configuration handling with support for column names and URLs"""

    DEFAULT_CONFIG = {
        "file_path": None,
        "column_names": None,
        "target_column": "target",
        "separator": ",",
        "has_header": True,
        "likelihood_config": {
            "feature_group_size": 2,
            "max_combinations": 1000,
            "bin_sizes": [20]
        },
        "active_learning": {
            "tolerance": 1.0,
            "cardinality_threshold_percentile": 95
        },
        "training_params": {
            "Save_training_epochs": False,
            "training_save_path": "training_data"
            # Remove hardcoded values here
        }
    }


    @staticmethod
    def is_url(path: str) -> bool:
        """Check if the given path is a URL"""
        return path.startswith(('http://', 'https://'))

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate that a URL is accessible"""
        try:
            response = requests.head(url)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def validate_columns(config: Dict) -> bool:
        """Validate column configuration"""
        if 'column_names' in config and config['column_names']:
            if not isinstance(config['column_names'], list):
                print("Error: column_names must be a list")
                return False

            # Validate target column is in column names
            if config['target_column'] not in config['column_names']:
                print(f"Error: target_column '{config['target_column']}' not found in column_names")
                return False

        return True

    @staticmethod
    def create_default_config(dataset_name: str) -> Dict:
        """Create a default configuration file with enhanced defaults"""
        config = DatasetConfig.DEFAULT_CONFIG.copy()
        config['file_path'] = f"{dataset_name}.csv"

        # Try to infer column names from CSV if it exists
        if os.path.exists(config['file_path']):
            try:
                with open(config['file_path'], 'r') as f:
                    header = f.readline().strip()

                    # Check if the first line looks like a header
                    first_row = pd.read_csv(config['file_path'], nrows=1)
                    if first_row.iloc[0].astype(str).str.match(r'^-?\d*\.?\d+$').all():
                        # First row looks like data, not a header
                        config['has_header'] = False
                        config['column_names'] = [f'col_{i}' for i in range(len(first_row.columns))]
                        config['target_column'] =[ -1]  # Default to last column
                    else:
                        # First row looks like a header
                        config['has_header'] = True
                        config['column_names'] = header.split(config['separator'])
                        config['target_column'] = config['column_names'][-1]
            except Exception as e:
                print(f"Warning: Could not read header from {config['file_path']}: {str(e)}")
        config[ "separator"]= ","
        config["has_header"]= "true"
        config["target_column"]="target"
        # Add model type configuration
        config['modelType'] = "Histogram"  # Default to Histogram model

        #Add likelihood parameter estimation config
        config[    "likelihood_config"]={
        "feature_group_size": 2,
        "max_combinations": 1000,
        "bin_sizes": [20]
        }

        #Add active Learning Parameters
        config[    "active_learning"]={
        "tolerance": 1.0,
        "cardinality_threshold_percentile": 95,
        "strong_margin_threshold": 0.3,
        "marginal_margin_threshold": 0.1,
        "min_divergence": 0.1
        }
        # Add training parameters
        config['training_params'] = {
            "trials": 100,
             "minimum_training_accuracy": 0.95,
            "cardinality_threshold": 0.9,
            "cardinality_tolerance": 4,
            "learning_rate": 0.1,
            "random_seed": 42,
            "epochs": 1000,
            "test_fraction": 0.2,
            "enable_adaptive": "true",
            "compute_device": "auto",
            "n_bins_per_dim": 20,
            "enable_adaptive": "true",
            "invert_DBNN": "false",
            "reconstruction_weight": 0.5,
            "feedback_strength": 0.3,
            "inverse_learning_rate": 0.1,
            "Save_training_epochs":" false",
            "training_save_path": "training_data"
        }
        # Add config execution flags
        config["execution_flags"]= {
        "train":" true",
        "train_only": "false",
        "predict": "true",
        "fresh_start": "false",
        "use_previous_model": "true"
        }


        # Save the configuration
        config_path = f"data/{dataset_name}/{dataset_name}.conf"
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Created default configuration file: {config_path}")
        except Exception as e:
            print(f"Warning: Could not save configuration file: {str(e)}")

        return config

    def _ensure_complete_config(self, dataset_name: str) -> Dict:
        """Ensure configuration file is complete with all options and default values."""
        # Define default configuration with minimal defaults
        default_config = {
            "file_path": f"data/{dataset_name}/{dataset_name}.csv",
            "separator": ",",
            "has_header": True,
            "target_column": "target",  # Will be updated when reading CSV
            "modelType": "Histogram",

            "likelihood_config": {
                "feature_group_size": 2,
                "max_combinations": 1000,
                "bin_sizes": [20]
            },

            "active_learning": {
                "tolerance": 1.0,
                "cardinality_threshold_percentile": 95,
                "strong_margin_threshold": 0.3,
                "marginal_margin_threshold": 0.1,
                "min_divergence": 0.1
            },

            "training_params": {
                "Save_training_epochs": False,
                "training_save_path": f"training_data/{dataset_name}",
                "enable_vectorized": False,
                "vectorization_warning_acknowledged": False
            },

            "execution_flags": {
                "train": True,
                "train_only": False,
                "predict": True,
                "fresh_start": False,
                "use_previous_model": True
            }
        }

        # Create dataset folder if it doesn't exist
        dataset_folder = os.path.join('data', dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)

        config_path = os.path.join(dataset_folder, f"{dataset_name}.conf")

        # Load and validate existing configuration
        existing_config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = f.read()
                    try:
                        existing_config = json.loads(config_data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON in config file: {str(e)}")
                        print("Creating new configuration with defaults")
                    else:
                        print(f"Loaded existing configuration from {config_path}")
            except Exception as e:
                print(f"Warning: Error loading existing config: {str(e)}")

        # Deep merge with validation
        def deep_merge(default: Dict, existing: Dict) -> Dict:
            result = default.copy()
            for key, value in existing.items():
                if key in result:
                    if isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        # Type validation for known parameters
                        try:
                            if key == "epochs" or key == "trials":
                                if not isinstance(value, int) or value < 1:
                                    print(f"Warning: Invalid {key} value: {value}. Must be positive integer.")
                                    continue
                            elif key == "learning_rate":
                                if not isinstance(value, (int, float)) or value <= 0:
                                    print(f"Warning: Invalid {key} value: {value}. Must be positive number.")
                                    continue
                        except TypeError as e:
                            print(f"Warning: Type error for {key}: {str(e)}")
                            continue
                        result[key] = value
                else:
                    result[key] = value
            return result

        # Merge with existing config taking precedence
        merged_config = deep_merge(default_config, existing_config)

        # Add missing training parameters with validation
        if 'training_params' in merged_config:
            default_training = {
                "trials": 5,
                "epochs": 10,
                "learning_rate": 0.1,
                "test_fraction": 0.2,
                "random_seed": 42,
                "minimum_training_accuracy": 0.95,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "n_bins_per_dim": 20,
                "enable_adaptive": True,
                "invert_DBNN": False,
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.1
            }

            for key, value in default_training.items():
                if key not in merged_config['training_params']:
                    merged_config['training_params'][key] = value

        # Save merged configuration
        try:
            with open(config_path, 'w') as f:
                json.dump(merged_config, f, indent=4)
            print(f"Saved complete configuration to {config_path}")

            # Save documented version with complete parameter documentation
            documented_path = os.path.join(dataset_folder, f"{dataset_name}_documented.conf")
            with open(documented_path, 'w') as f:
                f.write("{\n")
                f.write("    // Basic dataset configuration\n")
                f.write(f'    "file_path": "{merged_config["file_path"]}",  // Path to the dataset file\n')
                f.write(f'    "separator": "{merged_config["separator"]}",  // CSV separator character\n')
                f.write(f'    "has_header": {str(merged_config["has_header"]).lower()},  // Whether CSV has header row\n')
                f.write(f'    "target_column": "{merged_config["target_column"]}",  // Target/label column name\n')
                f.write(f'    "modelType": "{merged_config["modelType"]}",  // Model type (Histogram/Gaussian)\n\n')

                f.write("    // Likelihood computation configuration\n")
                f.write('    "likelihood_config": {\n')
                f.write('        "feature_group_size": 2,  // Number of features to group\n')
                f.write('        "max_combinations": 1000,  // Maximum feature combinations\n')
                f.write('        "bin_sizes": [20]  // Bin sizes for histogram\n')
                f.write('    },\n\n')

                f.write("    // Active learning parameters\n")
                f.write('    "active_learning": {\n')
                f.write('        "tolerance": 1.0,  // Learning tolerance\n')
                f.write('        "cardinality_threshold_percentile": 95,  // Percentile for cardinality threshold\n')
                f.write('        "strong_margin_threshold": 0.3,  // Threshold for strong classification margin\n')
                f.write('        "marginal_margin_threshold": 0.1,  // Threshold for marginal classification\n')
                f.write('        "min_divergence": 0.1  // Minimum divergence threshold\n')
                f.write('    },\n\n')

                f.write("    // Training parameters\n")
                f.write('    "training_params": {\n')
                f.write(f'        "trials": {merged_config["training_params"]["trials"]},  // Maximum training trials\n')
                f.write(f'        "epochs": {merged_config["training_params"]["epochs"]},  // Maximum training epochs\n')
                f.write('        "learning_rate": 0.1,  // Initial learning rate\n')
                f.write('        "test_fraction": 0.2,  // Fraction of data for testing\n')
                f.write('        "random_seed": 42,  // Random seed for reproducibility\n')
                f.write('        "minimum_training_accuracy": 0.95,  // Minimum required training accuracy\n')
                f.write('        "enable_adaptive": true,  // Enable adaptive learning\n')
                f.write('        "enable_vectorized": false,  // Enable vectorized training\n')
                f.write('        "vectorization_warning_acknowledged": false  // Vectorization warning flag\n')
                f.write('    },\n\n')

                f.write("    // Execution flags\n")
                f.write('    "execution_flags": {\n')
                f.write('        "train": true,  // Enable training\n')
                f.write('        "train_only": false,  // Only perform training\n')
                f.write('        "predict": true,  // Enable prediction\n')
                f.write('        "fresh_start": false,  // Start fresh training\n')
                f.write('        "use_previous_model": true  // Use previously trained model\n')
                f.write('    }\n')
                f.write("}\n")

        except Exception as e:
            print(f"Warning: Error saving configuration: {str(e)}")
            return merged_config

        return merged_config

    @staticmethod
    def load_config(dataset_name: str) -> Dict:
        """Load and validate dataset configuration with enhanced error handling."""
        config_path = f"{dataset_name}.conf"
        # print(f"\nDEBUG: Attempting to load config from: {config_path}")

        config_path = os.path.join('data', dataset_name, f"{dataset_name}.conf")
        # print(f"DEBUG:  Trying alternate path: {config_path}")

        try:
            # Create DatasetConfig instance to use instance methods
            config_handler = DatasetConfig()

            # If config doesn't exist, create default
            if not os.path.exists(config_path):
                print(f"Configuration file not found at {config_path}")
                return config_handler._ensure_complete_config(dataset_name)

            # Read and parse configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()

            # Remove comments and parse
            def remove_comments(json_str):
                lines = []
                in_multiline_comment = False
                for line in json_str.split('\n'):
                    if '_comment' in line:
                        continue
                    if '/*' in line and '*/' in line:
                        line = line[:line.find('/*')] + line[line.find('*/') + 2:]
                    elif '/*' in line:
                        in_multiline_comment = True
                        line = line[:line.find('/*')]
                    elif '*/' in line:
                        in_multiline_comment = False
                        line = line[line.find('*/') + 2:]
                    elif in_multiline_comment:
                        continue
                    if '//' in line and not ('http://' in line or 'https://' in line):
                        line = line.split('//')[0]
                    stripped = line.strip()
                    if stripped and not stripped.startswith('_comment'):
                        lines.append(stripped)
                return '\n'.join(lines)

            clean_config = remove_comments(config_text)
            try:
                config = json.loads(clean_config)
            except json.JSONDecodeError:
                print(f"Invalid config, attempting to infer from CSV...")
                csv_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, nrows=0)
                    columns = df.columns.tolist()
                    config = {
                        'file_path': csv_path,
                        'column_names': columns,
                        'target_column': columns[-1],
                        'separator': ',',
                        'has_header': True,
                        'modelType': 'Histogram',
                        'likelihood_config': {
                            'feature_group_size': 2,
                            'max_combinations': 1000,
                            'bin_sizes': [20]
                        },
                        'active_learning': {
                            'tolerance': 1.0,
                            'cardinality_threshold_percentile': 95
                        },
                        'training_params': DatasetConfig.DEFAULT_CONFIG['training_params']
                    }

            # Ensure all required parameters are present
            config = config_handler._ensure_complete_config(dataset_name)

            return config

        except Exception as e:
            print(f"Error loading config: {str(e)}")
            traceback.print_exc()
            return None


    @staticmethod
    def download_dataset(url: str, local_path: str) -> bool:
        """Download dataset from URL to local path with proper error handling"""
        try:
            print(f"Downloading dataset from {url}")
            response = requests.get(url, timeout=30)  # Add timeout
            response.raise_for_status()  # Check for HTTP errors

            # Handle potential text/csv content
            content = response.content.decode('utf-8')

            # Save to local file
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Dataset downloaded successfully to {local_path}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {str(e)}")
            return False
        except UnicodeDecodeError:
            # Handle binary content
            try:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print(f"Dataset downloaded successfully to {local_path}")
                return True
            except Exception as e:
                print(f"Error saving binary content: {str(e)}")
                return False
        except Exception as e:
            print(f"Unexpected error downloading dataset: {str(e)}")
            return False



    @staticmethod
    def get_available_datasets(create_configs: bool = False) -> List[str]:
        """Get list of available dataset configurations with better filename handling"""
        # Get all config and CSV files
        conf_files = {f.split('.')[0] for f in os.listdir()
                     if f.endswith('.conf') and f != 'adaptive_dbnn.conf'}  # Exclude own config
        csv_files = {f.split('.')[0] for f in os.listdir()
                    if f.endswith('.csv')}

        # Filter out derived filenames
        exclude_suffixes = [
            '_last_testing', '_Last_testing',
            '_last_training', '_Last_training',
            '_predictions', '_training_metrics',
            '_training_metrics_metrics'
        ]

        # Filter CSV files that don't have config
        csv_without_conf = csv_files - conf_files
        csv_without_conf = {name for name in csv_without_conf
                           if not any(name.endswith(suffix) for suffix in exclude_suffixes)}

        # Start with datasets that have config files
        datasets = conf_files

        # If requested, ask about creating configs for remaining CSVs
        if create_configs and csv_without_conf:
            print("\nFound CSV files without configuration:")
            for csv_name in sorted(csv_without_conf):
                response = input(f"Create configuration for {csv_name}.csv? (y/n): ")
                if response.lower() == 'y':
                    try:
                        DatasetConfig.create_default_config(csv_name)
                        datasets.add(csv_name)
                    except Exception as e:
                        print(f"Error creating config for {csv_name}: {str(e)}")

        return sorted(list(datasets))


    @staticmethod
    def validate_dataset(dataset_name: str) -> bool:
        """Validate dataset with better name handling"""
        # Check if this is a derived filename
        exclude_suffixes = [
            '_last_testing', '_Last_testing',
            '_last_training', '_Last_training',
            '_predictions', '_training_metrics',
            '_training_metrics_metrics'
        ]

        if any(dataset_name.endswith(suffix) for suffix in exclude_suffixes):
            print(f"Skipping validation for derived dataset: {dataset_name}")
            return False

        config = DatasetConfig.load_config(dataset_name)
        file_path = config['file_path']

        # Handle URL-based datasets
        if DatasetConfig.is_url(file_path):
            if not DatasetConfig.validate_url(file_path):
                print(f"Warning: Dataset URL {file_path} is not accessible")
                return False

            # Download to local cache if needed
            local_path = f"{dataset_name}.csv"
            if not os.path.exists(local_path):
                if not DatasetConfig.download_dataset(file_path, local_path):
                    return False
            file_path = local_path

        if not os.path.exists(file_path):
            print(f"Warning: Dataset file {file_path} not found")
            return False

        return True

#---------------------------------------Feature Filter with a #------------------------------------
def _filter_features_from_config(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Filter DataFrame columns based on commented features in config

    Args:
        df: Input DataFrame
        config: Configuration dictionary containing column names

    Returns:
        DataFrame with filtered columns
    """
    # If no column names in config, return original DataFrame
    if 'column_names' not in config:
        return df

    # Get column names from config
    column_names = config['column_names']

    # Create mapping of position to column name
    col_mapping = {i: name.strip() for i, name in enumerate(column_names)}

    # Identify commented features (starting with #)
    commented_features = {
        i: name.lstrip('#').strip()
        for i, name in col_mapping.items()
        if name.startswith('#')
    }

    # Get current DataFrame columns
    current_cols = df.columns.tolist()

    # Columns to drop (either by name or position)
    cols_to_drop = []

    for pos, name in commented_features.items():
        # Try to drop by name first
        if name in current_cols:
            cols_to_drop.append(name)
        # If name not found, try position
        elif pos < len(current_cols):
            cols_to_drop.append(current_cols[pos])

    # Drop identified columns
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped commented features: {cols_to_drop}")

    return df
#-------------------------------------------------
class ComputationCache:
    """Cache for frequently used computations"""
    def __init__(self, device):
        self.device = device
        self.bin_edges_cache = {}
        self.bin_indices_cache = {}
        self.feature_group_cache = {}
        self.max_cache_size = 1000

    def get_bin_edges(self, group_idx, edges):
        if group_idx not in self.bin_edges_cache:
            self.bin_edges_cache[group_idx] = torch.stack([
                edge.contiguous() for edge in edges
            ]).to(self.device)
        return self.bin_edges_cache[group_idx]

    def get_feature_group(self, features, group_idx, feature_pairs):
        key = (features.shape[0], group_idx)
        if key not in self.feature_group_cache:
            if len(self.feature_group_cache) >= self.max_cache_size:
                self.feature_group_cache.clear()
            self.feature_group_cache[key] = features[:, feature_pairs[group_idx]].contiguous()
        return self.feature_group_cache[key]

class BinWeightUpdater:
    def __init__(self, n_classes, feature_pairs, n_bins_per_dim=5):
        self.n_classes = n_classes
        self.feature_pairs = feature_pairs
        self.n_bins_per_dim = n_bins_per_dim
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        # Initialize histogram_weights as empty dictionary first
        self.histogram_weights = {}

        # Create weights for each class and feature pair
        for class_id in range(n_classes):
            self.histogram_weights[class_id] = {}
            for pair_idx in range(len(feature_pairs)):
                # Initialize with default weight of 0.1
                #print(f"[DEBUG] Creating weights for class {class_id}, pair {pair_idx}")
                self.histogram_weights[class_id][pair_idx] = torch.full(
                    (n_bins_per_dim, n_bins_per_dim),
                    0.1,
                    dtype=torch.float32,
                    device=self.device  # Ensure weights are created on correct device
                ).contiguous()

        # Initialize weights for each class and feature pair
        self.gaussian_weights = {}
        for class_id in range(n_classes):
            self.gaussian_weights[class_id] = {}
            for pair_idx in range(len(feature_pairs)):
                # Initialize with default weight of 0.1
                self.gaussian_weights[class_id][pair_idx] = torch.tensor(0.1,
                    dtype=torch.float32,
                    device=self.device
                ).contiguous()

        # Verify initialization
        print(f"[DEBUG] Weight initialization complete. Structure:")
        print(f"- Number of classes: {len(self.histogram_weights)}")
        for class_id in self.histogram_weights:
            print(f"- Class {class_id}: {len(self.histogram_weights[class_id])} feature pairs")

        # Use a single contiguous tensor for all weights
        self.weights = torch.full(
            (n_classes, len(feature_pairs), n_bins_per_dim, n_bins_per_dim),
            0.1,
            dtype=torch.float32,
            device=self.device  # Ensure weights are created on correct device
        ).contiguous()

        # Pre-allocate update buffers
        self.update_indices = torch.zeros((3, 1000), dtype=torch.long)  # [dim, max_updates]
        self.update_values = torch.zeros(1000, dtype=torch.float32)
        self.update_count = 0

    def _calculate_adaptive_adjustment(self, true_prob: float, pred_prob: float,
                                    base_learning_rate: float = 0.1) -> float:
        """
        Calculate adaptive weight adjustment.

        Args:
            true_prob: Probability of true class
            pred_prob: Probability of predicted (wrong) class
            base_learning_rate: Base learning rate

        Returns:
            float: Adaptive weight adjustment
        """
        # Calculate probability difference
        prob_diff = pred_prob - true_prob

        # Calculate confidence factor
        confidence_factor = pred_prob / (true_prob + 1e-10)

        # Calculate error magnitude
        error_magnitude = abs(prob_diff) / (true_prob + pred_prob)

        # Adaptive learning rate
        adaptive_rate = base_learning_rate * (1.0 + error_magnitude) * confidence_factor

        # Scale adjustment
        adjustment = adaptive_rate * (1.0 - (true_prob / (pred_prob + 1e-10)))

        # Add stability bounds
        adjustment = max(min(adjustment, 2.0), -2.0)

        return float(adjustment)  # Ensure we return a float

    def batch_update_weights(self, class_indices, pair_indices, bin_indices, adjustments):
            """Batch update with compatibility and proper shape handling"""
            n_updates = len(class_indices)

            # Process in batches for memory efficiency
            batch_size = 100  # Adjust based on available memory
            for i in range(0, n_updates, batch_size):
                end_idx = min(i + batch_size, n_updates)

                for idx in range(i, end_idx):
                    class_id = int(class_indices[idx])
                    pair_idx = int(pair_indices[idx])

                    # Handle bin indices properly based on their structure
                    if isinstance(bin_indices[idx], tuple):
                        bin_i, bin_j = bin_indices[idx]
                    else:
                        # If bin_indices is a tensor or array
                        bin_i = bin_indices[idx][0] if len(bin_indices[idx].shape) > 1 else bin_indices[idx]
                        bin_j = bin_indices[idx][1] if len(bin_indices[idx].shape) > 1 else bin_indices[idx]

                    # Ensure indices are properly shaped scalars
                    bin_i = int(bin_i.item() if torch.is_tensor(bin_i) else bin_i)
                    bin_j = int(bin_j.item() if torch.is_tensor(bin_j) else bin_j)

                    adjustment = float(adjustments[idx].item() if torch.is_tensor(adjustments[idx]) else adjustments[idx])

                    # Update weight with proper shape handling
                    self.histogram_weights[class_id][pair_idx][bin_i, bin_j] += adjustment


    def get_histogram_weights(self, class_id: int, pair_idx: int) -> torch.Tensor:
        """Get weights ensuring proper dimensions"""
        class_id = int(class_id)
        pair_idx = int(pair_idx)

        if class_id not in self.histogram_weights:
            raise KeyError(f"Invalid class_id: {class_id}")
        if pair_idx not in self.histogram_weights[class_id]:
            raise KeyError(f"Invalid pair_idx: {pair_idx}")

        weights = self.histogram_weights[class_id][pair_idx]
        if len(weights.shape) != 2:
            raise ValueError(f"Invalid weight shape: {weights.shape}, expected 2D tensor")
        if weights.shape[0] != self.n_bins_per_dim or weights.shape[1] != self.n_bins_per_dim:
            raise ValueError(f"Invalid weight dimensions: {weights.shape}, expected ({self.n_bins_per_dim}, {self.n_bins_per_dim})")

        return weights



    def _ensure_buffers(self, batch_size):
        """Ensure buffers exist and are the right size"""
        if (self.batch_indices_buffer is None or
            self.batch_indices_buffer.shape[0] < batch_size):
            self.batch_indices_buffer = torch.zeros(
                (batch_size, 2),
                dtype=torch.long,
                device=next(iter(next(iter(self.histogram_weights.values())).values())).device
            ).contiguous()

            self.batch_adjustments_buffer = torch.zeros(
                batch_size,
                dtype=torch.float32,
                device=self.batch_indices_buffer.device
            ).contiguous()

    def update_weight(self, class_id: int, pair_idx: int, bin_i: int, bin_j: int, adjustment: float):
        """Single weight update with proper error handling"""
        try:
            # Convert all indices to integers
            class_id = int(class_id)
            pair_idx = int(pair_idx)
            bin_i = int(bin_i)
            bin_j = int(bin_j)

            # Ensure indices are within bounds
            bin_i = min(max(0, bin_i), self.n_bins_per_dim - 1)
            bin_j = min(max(0, bin_j), self.n_bins_per_dim - 1)

            # Update the weight
            self.histogram_weights[class_id][pair_idx][bin_i, bin_j] += adjustment

        except Exception as e:
            print(f"Error updating weight: {str(e)}")
            print(f"class_id: {class_id}, pair_idx: {pair_idx}")
            print(f"bin_i: {bin_i}, bin_j: {bin_j}")
            print(f"adjustment: {adjustment}")
            raise

    def update_histogram_weights(self, failed_case, true_class, pred_class,
                               bin_indices, posteriors, learning_rate):
        """Update weights with proper type checking"""
        try:
            # Ensure proper types
            true_class = int(true_class)
            pred_class = int(pred_class)

            # Get the posterior values needed for adjustment
            true_posterior = float(posteriors[true_class])
            pred_posterior = float(posteriors[pred_class])

            # Calculate simple weight adjustment
            #adjustment = learning_rate * (1.0 - (true_posterior / pred_posterior))

            # Calculate adaptive adjustment
            adjustment = self._calculate_adaptive_adjustment(
                true_prob=true_posterior,
                pred_prob=pred_posterior,
                base_learning_rate=learning_rate
            )

            for pair_idx, (bin_i, bin_j) in bin_indices.items():
                # Ensure integer indices
                pair_idx = int(pair_idx)
                bin_i = int(bin_i)
                bin_j = int(bin_j)

                # Get and update weights
                weights = self.histogram_weights[true_class][pair_idx]
                weights[bin_i, bin_j] += adjustment

        except Exception as e:
            DEBUG.log(f"Error updating histogram weights:")
            DEBUG.log(f"- True class: {true_class}")
            DEBUG.log(f"- Pred class: {pred_class}")
            DEBUG.log(f"- Adjustment: {adjustment}")
            DEBUG.log(f"- Error: {str(e)}")
            raise

    def update_gaussian_weights(self, failed_case, true_class, pred_class,
                               component_responsibilities, posteriors, learning_rate):
        """Update weights for Gaussian components with improved efficiency"""
        DEBUG.log(f" Updating Gaussian weights for class {true_class}")

        try:
            # Convert tensor values to Python types and validate
            true_class = int(true_class) if isinstance(true_class, torch.Tensor) else true_class
            pred_class = int(pred_class) if isinstance(pred_class, torch.Tensor) else pred_class

            if true_class not in self.gaussian_weights:
                raise ValueError(f"Invalid true_class: {true_class}")

            # Get posterior values with type checking
            true_posterior = float(posteriors[true_class]) if isinstance(posteriors, torch.Tensor) else posteriors[true_class]
            pred_posterior = float(posteriors[pred_class]) if isinstance(posteriors, torch.Tensor) else posteriors[pred_class]

            # Calculate adjustment based on posterior ratio with stability check
            adjustment = learning_rate * (1.0 - max(min(true_posterior / pred_posterior, 10), 0.1))
            DEBUG.log(f" Weight adjustment: {adjustment}")

            # Process each feature pair efficiently
            for pair_idx in range(len(self.feature_pairs)):
                # Get and validate responsibility matrix
                resp_matrix = component_responsibilities[pair_idx]
                if not isinstance(resp_matrix, torch.Tensor):
                    resp_matrix = torch.tensor(resp_matrix)
                resp_matrix = resp_matrix.to(self.gaussian_weights[true_class][pair_idx].device)

                # Ensure shapes match
                current_weights = self.gaussian_weights[true_class][pair_idx]
                if resp_matrix.shape != current_weights.shape:
                    DEBUG.log(f" Shape mismatch - resp_matrix: {resp_matrix.shape}, weights: {current_weights.shape}")
                    resp_matrix = resp_matrix[:current_weights.shape[0], :current_weights.shape[1]]

                # Update weights with stability check
                weight_update = resp_matrix * adjustment
                weight_update = torch.clamp(weight_update, -1.0, 1.0)  # Prevent extreme updates
                self.gaussian_weights[true_class][pair_idx] += weight_update

                # Apply non-negativity constraint
                self.gaussian_weights[true_class][pair_idx].clamp_(min=0.0)

        except Exception as e:
            DEBUG.log(f" Error updating Gaussian weights: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            raise

    def get_gaussian_weights(self, class_id, pair_idx):
        """Get Gaussian weights with proper type conversion and validation"""
        try:
            # Convert tensor values to Python integers
            class_id = int(class_id) if isinstance(class_id, torch.Tensor) else class_id
            pair_idx = int(pair_idx) if isinstance(pair_idx, torch.Tensor) else pair_idx

            if class_id not in self.gaussian_weights:
                raise KeyError(f"Invalid class_id: {class_id}")
            if pair_idx not in self.gaussian_weights[class_id]:
                raise KeyError(f"Invalid pair_idx: {pair_idx}")

            weights = self.gaussian_weights[class_id][pair_idx]
            DEBUG.log(f" Retrieved Gaussian weights for class {class_id}, pair {pair_idx}, shape: {weights.shape}")
            return weights

        except Exception as e:
            DEBUG.log(f" Error retrieving Gaussian weights: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            raise

    # Modified posterior computation for Histogram model
    def compute_histogram_posterior(self, features, bin_indices):
        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params['classes'])
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        for group_idx, feature_group in enumerate(self.likelihood_params['feature_pairs']):
            bin_edges = self.likelihood_params['bin_edges'][group_idx]
            bin_probs = self.likelihood_params['bin_probs'][group_idx]

            # Get bin-specific weights
            bin_weights = self.weight_updater.get_histogram_weights(
                class_idx,
                group_idx
            )[bin_indices[group_idx]]

            # Apply bin-specific weights to probabilities
            weighted_probs = bin_probs * bin_weights.unsqueeze(0)

            # Continue with regular posterior computation...
            group_log_likelihoods = torch.log(weighted_probs + epsilon)
            log_likelihoods.add_(group_log_likelihoods)

        return log_likelihoods

    # Modified posterior computation for Gaussian model
    def compute_gaussian_posterior(self, features, component_responsibilities):
        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params['classes'])
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        for group_idx, feature_group in enumerate(self.likelihood_params['feature_pairs']):
            # Get component-specific weights
            component_weights = self.weight_updater.get_gaussian_weights(
                class_idx,
                group_idx
            )

            # Weight the Gaussian components
            weighted_resp = component_responsibilities[group_idx] * component_weights

            # Continue with regular posterior computation...
            group_log_likelihoods = torch.log(weighted_resp.sum() + epsilon)
            log_likelihoods.add_(group_log_likelihoods)

        return log_likelihoods



#------------------------------------------------------Invertable DBNN -------------------------


#----------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Process ML datasets')
    parser.add_argument("file_path", nargs='?', help="Path to dataset file or folder")
    args = parser.parse_args()

    processor = DatasetProcessor()

    if not args.file_path:
        parser.print_help()
        input("\nPress any key to search data folder for datasets (or Ctrl-C to exit)...")
        try:
            dataset_pairs = processor.find_dataset_pairs()
            if dataset_pairs:
                for basename, conf_path, csv_path in dataset_pairs:
                    print(f"\nFound dataset: {basename}")
                    print(f"Config: {conf_path}")
                    print(f"Data: {csv_path}")

                    if input("\nProcess this dataset? (y/n): ").lower() == 'y':
                        processor.process_dataset(csv_path)
            else:
                print("\nNo datasets found in data folder")
        except KeyboardInterrupt:
            print("\nProcessing interrupted")
            sys.exit(0)
    else:
        processor.process_dataset(args.file_path)

if __name__ == "__main__":
    main()
