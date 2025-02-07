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


import logging

class DebugLogger:
    def __init__(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def log(self, msg, force=False):
        """Only print if enabled or forced"""
        if self.enabled or force:
            print(msg)

# Create single global instance
DEBUG = DebugLogger()
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
            "Save_training_epochs": False,  # Save the epochs parameter
            "training_save_path": "training_data"  # Save epochs path parameter
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
                    config['column_names'] = header.split(config['separator'])
                    if config['column_names']:
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
        """
        Ensure configuration file is complete with all options and default values.
        Creates or updates config file in data/<dataset_name>/ folder.
        """
        # Define default configuration
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
                "trials": 100,
                "epochs": 1000,
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
                "inverse_learning_rate": 0.1,
                "Save_training_epochs": False,
                "training_save_path": f"training_data/{dataset_name}"
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

        # Load existing configuration if it exists
        existing_config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
                print(f"Loaded existing configuration from {config_path}")
            except Exception as e:
                print(f"Warning: Error loading existing config: {str(e)}")

        # Try to read CSV to get column names if not in existing config
        csv_path = os.path.join(dataset_folder, f"{dataset_name}.csv")
        if os.path.exists(csv_path) and "column_names" not in existing_config:
            try:
                df = pd.read_csv(csv_path, nrows=0)
                default_config["column_names"] = df.columns.tolist()
                default_config["target_column"] = df.columns[-1]
            except Exception as e:
                print(f"Warning: Error reading CSV headers: {str(e)}")
                default_config["column_names"] = []

        # Deep merge existing config with default config
        def deep_merge(default: Dict, existing: Dict) -> Dict:
            result = default.copy()
            for key, value in existing.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged_config = deep_merge(default_config, existing_config)

        # Save updated configuration
        try:
            with open(config_path, 'w') as f:
                json.dump(merged_config, f, indent=4)
            print(f"Saved complete configuration to {config_path}")

            # Also save a documented version with comments
            documented_path = os.path.join(dataset_folder, f"{dataset_name}_documented.conf")
            with open(documented_path, 'w') as f:
                f.write("{\n")
                f.write("    // Basic dataset configuration\n")
                f.write(f'    "file_path": "{merged_config["file_path"]}",  // Path to the dataset file\n')
                f.write('    "column_names": [                    // List of column names in order\n')
                for col in merged_config["column_names"]:
                    f.write(f'        "{col}",\n')
                f.write('    ],\n')
                # ... Continue with all parameters and their comments
                f.write("}\n")
            print(f"Saved documented configuration to {documented_path}")

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
            if not os.path.exists(config_path):
                print(f"Configuration file not found at {config_path}")
                return DatasetConfig.create_default_config(dataset_name)

            # Read and parse configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()
                #print("\nDEBUGRaw config loaded successfully")
            # Remove comments from config
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
               # print("DEBUG: Config parsed successfully")
            except json.JSONDecodeError:
                print(f"Invalid config, attempting to infer from CSV...")
                csv_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, nrows=0)
                    columns = df.columns.tolist()
                    config = {
                        'file_path': csv_path,
                        'column_names': columns,
                        'target_column': columns[-1],  # Default to last column as target
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
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=4)
                    print(f"Created new config from CSV")
                else:
                    return DatasetConfig.create_default_config(dataset_name)

            # Ensure required sections exist
            if 'training_params' not in config:
                config['training_params'] = {}
            if 'execution_flags' not in config:
                config['execution_flags'] = {}

            # Validate and handle file path
            if config.get('file_path'):
                if not os.path.exists(config['file_path']):
                    alt_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
                    if os.path.exists(alt_path):
                        config['file_path'] = alt_path
                        # print(f"DEBUG:  Updated file path to {alt_path}")

            if not config.get('file_path'):
                default_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
                if os.path.exists(default_path):
                    config['file_path'] = default_path
                    # print(f"DEBUG:  Set default file path to {default_path}")

            # Validate column names and target
            if not config.get('column_names'):
                try:
                    df = pd.read_csv(config['file_path'], nrows=0)
                    config['column_names'] = df.columns.tolist()
                    if not config.get('target_column'):
                        config['target_column'] = df.columns[-1]
                    # print(f"DEBUG:  Inferred columns: {config['column_names']}")
                    # print(f"DEBUG:  Target column set to: {config['target_column']}")
                except Exception as e:
                    print(f"Warning: Could not infer columns: {str(e)}")

            # Ensure target column exists in column names
            if 'target_column' in config and 'column_names' in config:
                if config['target_column'] not in config['column_names']:
                    # Try to find the target column in the actual CSV
                    try:
                        df = pd.read_csv(config['file_path'], nrows=0)
                        if config['target_column'] in df.columns:
                            config['column_names'] = df.columns.tolist()
                        else:
                            # Default to last column
                            config['target_column'] = df.columns[-1]
                            config['column_names'] = df.columns.tolist()
                        # print(f"DEBUG:  Updated target column to: {config['target_column']}")
                    except Exception as e:
                        print(f"Warning: Error validating target column: {str(e)}")

            # Verify inverse DBNN settings
            if config.get('training_params', {}).get('invert_DBNN'):
                #print("\nDEBUGFound inverse DBNN configuration:")
                tp = config['training_params']
                print(f"invert_DBNN: {tp.get('invert_DBNN')}")
                print(f"reconstruction_weight: {tp.get('reconstruction_weight')}")
                print(f"feedback_strength: {tp.get('feedback_strength')}")
                print(f"inverse_learning_rate: {tp.get('inverse_learning_rate')}")

            # Save validated config
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                # print(f"DEBUG:  Saved validated config to {config_path}")
            except Exception as e:
                print(f"Warning: Could not save validated config: {str(e)}")

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

    def update_weight_old(self, class_id: int, pair_idx: int, bin_i: int, bin_j: int, adjustment: float):
        """Update a single weight value with proper indexing"""
        try:
            weights = self.get_histogram_weights(class_id, pair_idx)

            # Ensure indices are within bounds
            bin_i = min(max(0, bin_i), self.n_bins_per_dim - 1)
            bin_j = min(max(0, bin_j), self.n_bins_per_dim - 1)

            # Update single value
            current_weight = weights[bin_i, bin_j].item()
            weights[bin_i, bin_j] = current_weight + adjustment

        except Exception as e:
            DEBUG.log(f"Error updating weight:")
            DEBUG.log(f"class_id: {class_id}, pair_idx: {pair_idx}")
            DEBUG.log(f"bin_i: {bin_i}, bin_j: {bin_j}")
            DEBUG.log(f"adjustment: {adjustment}")
            DEBUG.log(f"Error: {str(e)}")
            raise

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

            # Calculate weight adjustment
            adjustment = learning_rate * (1.0 - (true_posterior / pred_posterior))

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

#----------------------------------------------DBNN class-------------------------------------------------------------
class GPUDBNN:
    """GPU-Optimized Deep Bayesian Neural Network with Parallel Feature Pair Processing"""

    def __init__(self, dataset_name: str, learning_rate: float = 0.1,
                 max_epochs: int = 1000, test_size: float = 0.2,
                 random_state: int = 42, device: str = None,
                 fresh: bool = False, use_previous_model: bool = True,
                 n_bins_per_dim: int = 20, model_type: str = "Histogram"):
        """Initialize GPUDBNN with support for continued training with fresh data"""

        # Set dataset_name and model type first
        self.dataset_name = dataset_name
        self.model_type = model_type  # Store model type as instance variable
        self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
        self.computation_cache = ComputationCache(self.device)
        # Initialize train/test indices
        self.train_indices = []
        self.test_indices = None
        self._last_metrics_printed =False
        # Add new attribute for bin-specific weights
        self.n_bins_per_dim = n_bins_per_dim
        self.weight_updater = None  # Will be initialized after computing likelihood params

        # Load configuration before potential cleanup
        self.config = DatasetConfig.load_config(self.dataset_name)
        self.feature_bounds = None  # Store global min/max for each

        # Initialize other attributes

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.test_size = test_size
        if random_state !=-1:
            self.random_state = random_state
            self.shuffle_state =1
        else:
            self.random_state = -1
            self.shuffle_state =-1
        #self.compute_dtype = torch.float64  # Use double precision for computations
        self.cardinality_tolerance = 4 # Only for feature grouping
        self.fresh_start = fresh
        self.use_previous_model = use_previous_model
        # Create Model directory
        os.makedirs('Model', exist_ok=True)

        # Load configuration and data
        self.config = DatasetConfig.load_config(self.dataset_name)
        self.target_column = self.config['target_column']

        # Initialize model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.likelihood_params = None
        self.feature_pairs = None
        self.best_W = None
        self.best_error = float('inf')
        self.current_W = None

        # Enable cuDNN autotuner if using CUDA
        if self.device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True

        # Pin memory for faster CPU->GPU transfer
        if torch.cuda.is_available():
            self.pin_memory = True
            torch.cuda.empty_cache()

        # Handle fresh start after configuration is loaded
        # Handle model state based on flags
        if not fresh and use_previous_model:
            # Load previous model state
            self._load_model_components()
            self._load_best_weights()
            self._load_categorical_encoders()
        elif fresh and use_previous_model:
            # Use previous model weights but start with fresh data
            self._load_best_weights()
            self._load_categorical_encoders()
        else:
            # Complete fresh start
            self._clean_existing_model()


        #------------------------------------------Adaptive Learning--------------------------------------
        super().__init__()
        self.adaptive_learning = True
        self.base_save_path = './training_data'
        os.makedirs(self.base_save_path, exist_ok=True)
        self.in_adaptive_fit=False # Set when we are in adaptive learning process
        #------------------------------------------Adaptive Learning--------------------------------------
        # Automatically select device if none specified

        print(f"Using device: {self.device}")

        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.test_size = test_size

        # Model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.likelihood_params = None
        self.feature_pairs = None
        self.best_W = None
        self.best_error = float('inf')
        self.current_W = None

        # Categorical feature handling
        self.categorical_encoders = {}

        # Create Model directory
        os.makedirs('Model', exist_ok=True)

        # Load dataset configuration and data
        self.config = DatasetConfig.load_config(self.dataset_name)
        self.data = self._load_dataset()

        self.target_column = self.config['target_column']

        # Load saved weights and encoders
        self._load_best_weights()
        self._load_categorical_encoders()

#----------------------
    def _save_predictions_with_reconstruction(self,
                                            X_test_df: pd.DataFrame,
                                            predictions: torch.Tensor,
                                            save_path: str,
                                            true_labels: pd.Series = None,
                                            reconstructed_features: torch.Tensor = None):
        """Save predictions with reconstruction analysis.

        Args:
            X_test_df: DataFrame containing test features
            predictions: Predicted class labels tensor
            save_path: Path to save results
            true_labels: True class labels (optional)
            reconstructed_features: Reconstructed features tensor (optional)
        """
        # Create the base directory for the dataset
        dataset_name = os.path.splitext(os.path.basename(save_path))[0]
        reconstruction_dir = os.path.join('data', dataset_name, 'reconstruction')

        # Ensure the reconstruction directory exists
        os.makedirs(reconstruction_dir, exist_ok=True)

        # Create the result DataFrame
        result_df = X_test_df.copy()

        # Convert predictions to labels
        pred_labels = self.label_encoder.inverse_transform(predictions.cpu().numpy())
        result_df['predicted_class'] = pred_labels

        if true_labels is not None:
            result_df['true_class'] = true_labels

        # Add reconstructed features and analysis
        if reconstructed_features is not None:
            # Convert to numpy for processing
            X_test_np = X_test_df.values
            recon_features = reconstructed_features.cpu().numpy()

            # Add reconstructed features
            for i in range(recon_features.shape[1]):
                result_df[f'reconstructed_feature_{i}'] = recon_features[:, i]

            # Add reconstruction error
            feature_errors = np.mean((X_test_np - recon_features) ** 2, axis=1)
            result_df['reconstruction_error'] = feature_errors

            # Save to reconstruction directory
            recon_path = os.path.join(reconstruction_dir, f'{dataset_name}_reconstruction.csv')
            result_df.to_csv(recon_path, index=False)

            # Generate reconstruction report
            self._generate_reconstruction_report(
                original_features=X_test_np,
                reconstructed_features=recon_features,
                true_labels=true_labels,
                predictions=pred_labels,
                save_path=os.path.join(reconstruction_dir, f'{dataset_name}_reconstruction_report')
            )

            print(f"Reconstruction data saved to {recon_path}")

        # Save original predictions file
        base_path = os.path.splitext(save_path)[0]
        result_df.to_csv(f"{base_path}_predictions.csv", index=False)


    def _save_predictions_with_reconstruction_old(self,
                                        X_test_df: pd.DataFrame,
                                        predictions: torch.Tensor,
                                        save_path: str,
                                        true_labels: pd.Series = None,
                                        reconstructed_features: torch.Tensor = None):
        """Save predictions with reconstruction analysis.

        Args:
            X_test_df: DataFrame containing test features
            predictions: Predicted class labels tensor
            save_path: Path to save results
            true_labels: True class labels (optional)
            reconstructed_features: Reconstructed features tensor (optional)
        """
        result_df = X_test_df.copy()

        # Convert predictions to labels
        pred_labels = self.label_encoder.inverse_transform(predictions.cpu().numpy())
        result_df['predicted_class'] = pred_labels

        if true_labels is not None:
            result_df['true_class'] = true_labels

        # Add reconstructed features and analysis
        if reconstructed_features is not None:
            # Convert to numpy for processing
            X_test_np = X_test_df.values
            recon_features = reconstructed_features.cpu().numpy()

            # Add reconstructed features
            for i in range(recon_features.shape[1]):
                result_df[f'reconstructed_feature_{i}'] = recon_features[:, i]

            # Add reconstruction error
            feature_errors = np.mean((X_test_np - recon_features) ** 2, axis=1)
            result_df['reconstruction_error'] = feature_errors

            # Create output directory structure
            dataset_name = os.path.splitext(os.path.basename(save_path))[0]
            recon_dir = os.path.join('data', dataset_name, 'reconstruction')
            os.makedirs(recon_dir, exist_ok=True)

            # Save to reconstruction directory
            recon_path = os.path.join(recon_dir, f'{dataset_name}_reconstruction.csv')
            result_df.to_csv(recon_path, index=False)

            # Generate reconstruction report
            self._generate_reconstruction_report(
                original_features=X_test_np,
                reconstructed_features=recon_features,
                true_labels=true_labels,
                predictions=pred_labels,
                save_path=os.path.join(recon_dir, f'{dataset_name}_reconstruction_report')
            )

            print(f"Reconstruction data saved to {recon_dir}")

        # Save original predictions file
        base_path = os.path.splitext(save_path)[0]
        result_df.to_csv(f"{base_path}_predictions.csv", index=False)

    def _generate_reconstruction_report(self, original_features: np.ndarray,
                                     reconstructed_features: np.ndarray,
                                     true_labels: np.ndarray,
                                     predictions: np.ndarray,
                                     save_path: str):
        """Generate detailed reconstruction analysis report"""
        report = {
            'overall_metrics': {
                'mse': float(np.mean((original_features - reconstructed_features) ** 2)),
                'mae': float(np.mean(np.abs(original_features - reconstructed_features))),
                'correlation': float(np.corrcoef(original_features.flatten(),
                                              reconstructed_features.flatten())[0, 1])
            },
            'per_feature_metrics': [],
            'per_class_metrics': {},
            'reconstruction_quality': {}
        }

        # Per-feature analysis
        for i in range(original_features.shape[1]):
            orig = original_features[:, i]
            recon = reconstructed_features[:, i]
            report['per_feature_metrics'].append({
                'feature_idx': i,
                'mse': float(np.mean((orig - recon) ** 2)),
                'correlation': float(np.corrcoef(orig, recon)[0, 1]),
                'mean_error': float(np.mean(np.abs(orig - recon)))
            })

        # Per-class analysis
        unique_classes = np.unique(true_labels)
        for class_label in unique_classes:
            mask = (true_labels == class_label)
            orig_class = original_features[mask]
            recon_class = reconstructed_features[mask]

            report['per_class_metrics'][str(class_label)] = {
                'mse': float(np.mean((orig_class - recon_class) ** 2)),
                'sample_count': int(np.sum(mask)),
                'accuracy': float(np.mean(predictions[mask] == true_labels[mask]))
            }

        # Quality assessment
        errors = np.mean((original_features - reconstructed_features) ** 2, axis=1)
        report['reconstruction_quality'] = {
            'excellent': float(np.mean(errors < 0.1)),
            'good': float(np.mean((errors >= 0.1) & (errors < 0.3))),
            'fair': float(np.mean((errors >= 0.3) & (errors < 0.5))),
            'poor': float(np.mean(errors >= 0.5))
        }

        # Save reports
        with open(f"{save_path}_reconstruction_analysis.json", 'w') as f:
            json.dump(report, f, indent=4)

        self._save_reconstruction_plots(
            original_features, reconstructed_features,
            true_labels, save_path
        )

    def _save_reconstruction_plots(self, original_features: np.ndarray,
                                reconstructed_features: np.ndarray,
                                true_labels: np.ndarray,
                                save_path: str):
        """Generate visualization plots for reconstruction analysis"""
        plt.figure(figsize=(15, 5))

        # Feature-wise reconstruction error
        plt.subplot(131)
        errors = np.mean((original_features - reconstructed_features) ** 2, axis=0)
        plt.bar(range(len(errors)), errors)
        plt.title('Feature-wise Reconstruction Error')
        plt.xlabel('Feature Index')
        plt.ylabel('MSE')

        # Class-wise reconstruction quality
        plt.subplot(132)
        unique_classes = np.unique(true_labels)
        class_errors = []
        for class_label in unique_classes:
            mask = (true_labels == class_label)
            error = np.mean((original_features[mask] - reconstructed_features[mask]) ** 2)
            class_errors.append(error)

        plt.bar(unique_classes, class_errors)
        plt.title('Class-wise Reconstruction Error')
        plt.xlabel('Class')
        plt.ylabel('MSE')

        # Error distribution
        plt.subplot(133)
        all_errors = np.mean((original_features - reconstructed_features) ** 2, axis=1)
        plt.hist(all_errors, bins=30)
        plt.title('Error Distribution')
        plt.xlabel('MSE')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig(f"{save_path}_reconstruction_plots.png")
        plt.close()

    def _compute_balanced_accuracy(self, y_true, y_pred):
        """Compute class-balanced accuracy"""
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        return np.mean(per_class_acc[~np.isnan(per_class_acc)])

    def _select_balanced_samples(self, misclassified_indices, y_test, n_samples=2):
        """Select misclassified samples with balanced class representation and error margins"""
        class_indices = defaultdict(list)
        y_test_cpu = y_test.cpu()

        # Get probability info for each misclassified sample
        probs_info = {}
        for idx in misclassified_indices:
            true_class = y_test_cpu[idx].item()
            probs = self._compute_batch_posterior(self.X_tensor[idx].unsqueeze(0))[0]
            pred_class = torch.argmax(probs).item()
            true_prob = probs[true_class].item()
            pred_prob = probs[pred_class].item()
            error_margin = pred_prob - true_prob  # Higher means more confident wrong prediction

            class_indices[true_class].append(idx)
            probs_info[idx] = {
                'true_class': true_class,
                'pred_class': pred_class,
                'error_margin': error_margin,
                'true_prob': true_prob,
                'pred_prob': pred_prob
            }

        # Calculate class-wise statistics
        class_stats = {}
        for cls in class_indices:
            cls_samples = y_test_cpu == cls
            total = cls_samples.sum().item()
            misclassified = len(class_indices[cls])
            error_rate = misclassified / total if total > 0 else 1.0
            class_stats[cls] = {
                'error_rate': error_rate,
                'total': total,
                'misclassified': misclassified
            }

        selected_indices = []
        remaining_samples = n_samples

        # First, ensure at least one sample from each failing class
        for true_class, stats in class_stats.items():
            if stats['error_rate'] > 0:  # Class has errors
                # Get samples from this class sorted by error margin
                class_samples = [(idx, probs_info[idx]['error_margin'])
                               for idx in class_indices[true_class]]
                class_samples.sort(key=lambda x: x[1], reverse=True)  # Highest error margin first

                # Select the sample with highest error margin
                if class_samples:
                    idx = class_samples[0][0]
                    selected_indices.append(idx)
                    remaining_samples -= 1

                    # Print selection info
                    info = probs_info[idx]
                    true_class_name = self.label_encoder.inverse_transform([info['true_class']])[0]
                    pred_class_name = self.label_encoder.inverse_transform([info['pred_class']])[0]
                    print(f"Adding sample from class {true_class_name} (misclassified as {pred_class_name}, "
                          f"error margin: {info['error_margin']:.3f})")

        # If we still have samples to select, choose based on error rates and margins
        if remaining_samples > 0:
            # Create pool of remaining samples with weights
            remaining_pool = []
            for idx in misclassified_indices:
                if idx not in selected_indices:
                    info = probs_info[idx]
                    cls_stats = class_stats[info['true_class']]

                    # Weight based on class error rate and individual error margin
                    weight = cls_stats['error_rate'] * (1 + info['error_margin'])
                    remaining_pool.append((idx, weight))

            # Sort by weight and select top remaining_samples
            remaining_pool.sort(key=lambda x: x[1], reverse=True)
            for idx, weight in remaining_pool[:remaining_samples]:
                selected_indices.append(idx)
                info = probs_info[idx]
                true_class_name = self.label_encoder.inverse_transform([info['true_class']])[0]
                pred_class_name = self.label_encoder.inverse_transform([info['pred_class']])[0]
                print(f"Adding additional sample from class {true_class_name} (misclassified as {pred_class_name}, "
                      f"error margin: {info['error_margin']:.3f})")

        # Print summary
        print(f"\nSelection Summary:")
        print(f"Total failing classes: {len(class_stats)}")
        print(f"Selected {len(selected_indices)} samples total")
        for cls in sorted(class_stats.keys()):
            cls_name = self.label_encoder.inverse_transform([cls])[0]
            stats = class_stats[cls]
            selected_from_class = sum(1 for idx in selected_indices
                                    if probs_info[idx]['true_class'] == cls)
            print(f"Class {cls_name}: {selected_from_class} samples selected out of {stats['misclassified']} "
                  f"misclassified (error rate: {stats['error_rate']:.3f})")

        return selected_indices

    def _print_detailed_metrics(self, y_true, y_pred, prefix=""):
        """Print detailed performance metrics with color coding"""
        # Compute metrics
        balanced_acc = self._compute_balanced_accuracy(y_true, y_pred)
        raw_acc = np.mean(y_true == y_pred)

        # Print metrics with colors
        print(f"\n{Colors.BOLD}{Colors.BLUE}{prefix}Detailed Metrics:{Colors.ENDC}")

        # Raw accuracy
        acc_color = Colors.GREEN if raw_acc >= 0.9 else Colors.YELLOW if raw_acc >= 0.7 else Colors.RED
        print(f"{Colors.BOLD}Raw Accuracy:{Colors.ENDC} {acc_color}{raw_acc:.4%}{Colors.ENDC}")

        # Balanced accuracy
        bal_color = Colors.GREEN if balanced_acc >= 0.9 else Colors.YELLOW if balanced_acc >= 0.7 else Colors.RED
        print(f"{Colors.BOLD}Balanced Accuracy:{Colors.ENDC} {bal_color}{balanced_acc:.4%}{Colors.ENDC}")

        # Per-class metrics
        print(f"\n{Colors.BOLD}Per-class Performance:{Colors.ENDC}")
        cm = confusion_matrix(y_true, y_pred)
        class_labels = np.unique(y_true)

        for i, label in enumerate(class_labels):
            class_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
            color = Colors.GREEN if class_acc >= 0.9 else Colors.YELLOW if class_acc >= 0.7 else Colors.RED
            samples = cm[i].sum()
            print(f"Class {label}: {color}{class_acc:.4%}{Colors.ENDC} ({samples:,} samples)")

        return balanced_acc
#---------------------- -------------------------------------DBNN Class -------------------------------
class DBNNConfig:
    """Configuration class for DBNN parameters"""
    def __init__(self, **kwargs):
        # Training parameters
        self.trials = kwargs.get('trials', 100)
        self.cardinality_threshold = kwargs.get('cardinality_threshold', 0.9)
        self.cardinality_tolerance = kwargs.get('cardinality_tolerance', 4)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.random_seed = kwargs.get('random_seed', 42)
        self.epochs = kwargs.get('epochs', 1000)
        self.test_fraction = kwargs.get('test_fraction', 0.2)
        self.enable_adaptive = kwargs.get('enable_adaptive', True)
        self.batch_size = kwargs.get('batch_size', 32)
        self.model_type = kwargs.get('model_type', 'Histogram')

        # Model parameters
        self.model_type = kwargs.get('model_type', 'Histogram')  # or 'Gaussian'
        self.n_bins_per_dim = kwargs.get('n_bins_per_dim', 20)

        # Execution flags
        self.train = kwargs.get('train', True)
        self.train_only = kwargs.get('train_only', False)
        self.predict = kwargs.get('predict', True)
        self.fresh_start = kwargs.get('fresh_start', False)
        self.use_previous_model = kwargs.get('use_previous_model', True)

        # Device configuration
        self.device = kwargs.get('device', 'auto')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Paths configuration
        self.model_dir = kwargs.get('model_dir', 'Model')
        self.training_data_dir = kwargs.get('training_data_dir', 'training_data')

        # Debug configuration
        self.debug = kwargs.get('debug', False)

        # New inverse model parameters
        self.invert_DBNN = kwargs.get('invert_DBNN', False)
        self.reconstruction_weight = kwargs.get('reconstruction_weight', 0.5)
        self.feedback_strength = kwargs.get('feedback_strength', 0.3)
        self.inverse_learning_rate = kwargs.get('inverse_learning_rate', 0.1)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'DBNNConfig':
        return cls(**config_dict)

    @classmethod
    def from_file(cls, config_path: str) -> 'DBNNConfig':
        """Create configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)

    def save(self, config_path: str):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items()}
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

class DBNN(GPUDBNN):
    def __init__(self, dataset_name: str, config: Optional[Union[GlobalConfig, Dict]] = None):

        self.dataset_name = dataset_name
        if isinstance(config, dict):
            self.config = GlobalConfig.from_dict(config)
        elif isinstance(config, GlobalConfig):
            self.config = config
        else:
            self.config = GlobalConfig()
        # Store inversion parameters explicitly - handle both dict and GlobalConfig
        if isinstance(self.config, dict):
            training_params = self.config.get('training_params', {})
            self.invert_DBNN = training_params.get('invert_DBNN', False)
            self.reconstruction_weight = training_params.get('reconstruction_weight', 0.5)
            self.feedback_strength = training_params.get('feedback_strength', 0.3)
            self.inverse_learning_rate = training_params.get('inverse_learning_rate', 0.1)
        else:
            # Access GlobalConfig attributes directly
            self.invert_DBNN = getattr(self.config, 'invert_DBNN', False)
            self.reconstruction_weight = getattr(self.config, 'reconstruction_weight', 0.5)
            self.feedback_strength = getattr(self.config, 'feedback_strength', 0.3)
            self.inverse_learning_rate = getattr(self.config, 'inverse_learning_rate', 0.1)
        # First load the dataset configuration
        self.data_config = DatasetConfig.load_config(dataset_name) if dataset_name else None

        super().__init__(
            dataset_name=dataset_name,
            learning_rate=self._get_config_value('learning_rate', 0.1),
            max_epochs=self._get_config_value('epochs', 1000),
            test_size=self._get_config_value('test_fraction', 0.2),
            random_state=self._get_config_value('random_seed', 42),
            fresh=self._get_config_value('fresh_start', False),
            use_previous_model=self._get_config_value('use_previous_model', True),
            model_type=self._get_config_value('modelType', "Histogram"),
            n_bins_per_dim=self._get_config_value('n_bins_per_dim', 20)
        )


        # Store model configuration
        self.model_config = config
        self.training_log = pd.DataFrame()

    def verify_reconstruction_predictions(self, predictions_df: pd.DataFrame, reconstructions_df: pd.DataFrame) -> Dict:
       """Verify if reconstructed features maintain predictive accuracy"""
       try:
           # Get reconstructed features
           feature_cols = [col for col in reconstructions_df.columns if col.startswith('reconstructed_feature_')]
           n_features = len(feature_cols)
           recon_features = reconstructions_df[feature_cols].values

           # Convert to tensor and preprocess
           recon_tensor = torch.tensor(recon_features, dtype=torch.float32).to(self.device)
           X_tensor = self._preprocess_data(pd.DataFrame(recon_features), is_training=False)

           # Make predictions
           new_predictions = self.predict(X_tensor)
           new_pred_labels = self.label_encoder.inverse_transform(new_predictions.cpu().numpy())

           # Get original predictions and true labels
           orig_pred_labels = predictions_df['predicted_class'].values
           true_labels = predictions_df['true_class'].values if 'true_class' in predictions_df else None

           # Calculate accuracies
           pred_match = (new_pred_labels == orig_pred_labels).mean()
           true_match = (new_pred_labels == true_labels).mean() if true_labels is not None else None

           results = {
               'reconstruction_prediction_accuracy': pred_match,
               'reconstruction_true_accuracy': true_match,
               'confusion_matrix': confusion_matrix(orig_pred_labels, new_pred_labels),
               'classification_report': classification_report(orig_pred_labels, new_pred_labels)
           }

           return results

       except Exception as e:
           print(f"Error verifying reconstruction predictions: {str(e)}")
           traceback.print_exc()
           return None

    def update_results_with_reconstruction(self, results: Dict,
                                         original_features: torch.Tensor,
                                         reconstructed_features: torch.Tensor,
                                         class_probs: torch.Tensor,
                                         true_labels: torch.Tensor,
                                         save_path: Optional[str] = None) -> Dict:
        """Update results with reconstruction metrics and handle serialization"""

        # Compute metrics with type conversion
        reconstruction_metrics = self._compute_reconstruction_metrics(
            original_features,
            reconstructed_features,
            class_probs,
            true_labels
        )

        # Update results dictionary
        results['reconstruction'] = reconstruction_metrics

        # Save reconstructed features to CSV
        if reconstructed_features is not None:
            # Get dataset base name
            dataset_name = os.path.splitext(os.path.basename(self.dataset_name))[0]

            # Create reconstruction directory
            recon_dir = os.path.join('data', dataset_name, 'reconstructed_features')
            os.makedirs(recon_dir, exist_ok=True)

            # Convert tensors to numpy arrays
            recon_features = reconstructed_features.cpu().numpy()
            orig_features = original_features.cpu().numpy()

            # Create DataFrame with original and reconstructed features
            recon_df = pd.DataFrame()

            # Add original features
            for i in range(orig_features.shape[1]):
                recon_df[f'original_feature_{i}'] = orig_features[:, i]

            # Add reconstructed features
            for i in range(recon_features.shape[1]):
                recon_df[f'reconstructed_feature_{i}'] = recon_features[:, i]

            # Add reconstruction error per sample
            recon_df['reconstruction_error'] = np.mean((orig_features - recon_features) ** 2, axis=1)

            # Save to CSV
            recon_path = os.path.join(recon_dir, 'reconstructed_features.csv')
            recon_df.to_csv(recon_path, index=False)
            print(f"\nSaved reconstructed features to: {recon_path}")

        # Save analysis if path provided
        if save_path:
            self.save_reconstruction_analysis(reconstruction_metrics, save_path)

        # Print summary
        print(self._format_reconstruction_results(reconstruction_metrics))

        return results


    def update_results_with_reconstruction_old(self, results: Dict,
                                        original_features: torch.Tensor,
                                        reconstructed_features: torch.Tensor,
                                        class_probs: torch.Tensor,
                                        true_labels: torch.Tensor,
                                        save_path: Optional[str] = None) -> Dict:
        """Update results with reconstruction metrics and handle serialization"""
        # Compute metrics with type conversion
        reconstruction_metrics = self._compute_reconstruction_metrics(
            original_features,
            reconstructed_features,
            class_probs,
            true_labels
        )

        # Update results dictionary
        results['reconstruction'] = reconstruction_metrics

        # Save analysis if path provided
        if save_path:
            self.save_reconstruction_analysis(reconstruction_metrics, save_path)

        # Print summary
        print(self._format_reconstruction_results(reconstruction_metrics))

        return results


    def process_dataset(self, config_path: str) -> Dict:
        """
        Process dataset according to configuration file specifications

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Dictionary containing processing results
        """
        # Load and validate configuration
        try:
            with open(config_path, 'r') as f:
                config_text = f.read()

            # Remove comments starting with _comment
            config_lines = [line for line in config_text.split('\n') if not '"_comment"' in line]
            clean_config = '\n'.join(config_lines)

            self.data_config = json.loads(clean_config)
        except Exception as e:
            raise ValueError(f"Error reading configuration file: {str(e)}")

        # Ensure file_path is set
        if not self.data_config.get('file_path'):
            dataset_name = os.path.splitext(os.path.basename(config_path))[0]
            default_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
            if os.path.exists(default_path):
                self.data_config['file_path'] = default_path
                print(f"Using default data file: {default_path}")
            else:
                raise ValueError(f"No data file found for {dataset_name}")

        # Convert dictionary config to DBNNConfig object
        config_params = {
            'epochs': self.data_config.get('training_params', {}).get('epochs', Epochs),
            'learning_rate': self.data_config.get('training_params', {}).get('learning_rate', LearningRate),
            'model_type': self.data_config.get('modelType', 'Histogram'),
            'enable_adaptive': self.data_config.get('training_params', {}).get('enable_adaptive', EnableAdaptive),
            'batch_size': self.data_config.get('training_params', {}).get('batch_size', 32),
            'training_data_dir': self.data_config.get('training_params', {}).get('training_save_path', 'training_data')
        }
        self.model_config = DBNNConfig(**config_params)

        # Create output directory structure
        dataset_name = os.path.splitext(os.path.basename(self.data_config['file_path']))[0]
        output_dir = os.path.join(self.model_config.training_data_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Update dataset name
        self.dataset_name = dataset_name

        # Load data using existing GPUDBNN method
        self.data = self._load_dataset()

        # Add row tracking
        self.data['original_index'] = range(len(self.data))

        # Extract features and target
        if 'target_column' not in self.data_config:
            self.data_config['target_column'] = 'target'  # Set default target column
            print(f"Using default target column: 'target'")

        X = self.data.drop(columns=[self.data_config['target_column']])
        y = self.data[self.data_config['target_column']]

        # Initialize training log
        log_file = os.path.join(output_dir, f'{dataset_name}_log.csv')
        self.training_log = pd.DataFrame(columns=[
            'timestamp', 'round', 'train_size', 'test_size',
            'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy',
            'training_time'
        ])

        # Train model using existing GPUDBNN methods
        if self.model_config.enable_adaptive:
            results = self.adaptive_fit_predict(max_rounds=self.model_config.epochs)
        else:
            results = self.fit_predict()

        # Generate detailed predictions
        predictions_df = self._generate_detailed_predictions(X)

        # Save results
        results_path = os.path.join(output_dir, f'{dataset_name}_predictions.csv')
        predictions_df.to_csv(results_path, index=False)

        # Save training log
        self.training_log.to_csv(log_file, index=False)

        # Count number of features actually used (excluding high cardinality and excluded features)
        n_features = len(X.columns)
        n_excluded = len(getattr(self, 'high_cardinality_columns', []))

        return {
            'results_path': results_path,
            'log_path': log_file,
            'n_samples': len(self.data),
            'n_features': n_features,
            'n_excluded': n_excluded,
            'training_results': results
        }

    def _generate_detailed_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate detailed predictions with confidence metrics"""
        # Get preprocessed features for probability computation
        X_processed = self._preprocess_data(X, is_training=False)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)

        # Create results DataFrame
        results_df = self.data.copy()

        # Compute probabilities in batches
        batch_size = 32
        all_probabilities = []

        for i in range(0, len(X_tensor), batch_size):
            batch_end = min(i + batch_size, len(X_tensor))
            batch_X = X_tensor[i:batch_end]

            try:
                if self.model_type == "Histogram":
                    batch_probs, _ = self._compute_batch_posterior(batch_X)
                elif self.model_type == "Gaussian":
                    batch_probs, _ = self._compute_batch_posterior_std(batch_X)
                else:
                    raise ValueError(f"{self.model_type} is invalid")

                all_probabilities.append(batch_probs.cpu().numpy())

            except Exception as e:
                print(f"Error computing probabilities for batch {i}: {str(e)}")
                return None

        if all_probabilities:
            probabilities = np.vstack(all_probabilities)
        else:
            print("No probabilities were computed successfully")
            return None

        # Get actual classes used in training
        unique_classes = np.unique(self.label_encoder.transform(self.data[self.target_column]))
        n_classes = len(unique_classes)

        # Verify probability array shape
        if probabilities.shape[1] != n_classes:
            print(f"Warning: Probability array shape ({probabilities.shape}) doesn't match number of classes ({n_classes})")
            # Adjust probabilities array if necessary
            if probabilities.shape[1] > n_classes:
                probabilities = probabilities[:, :n_classes]
            else:
                # Pad with zeros if needed
                pad_width = ((0, 0), (0, n_classes - probabilities.shape[1]))
                probabilities = np.pad(probabilities, pad_width, mode='constant')

        # Get predictions
        predictions = np.argmax(probabilities, axis=1)

        # Convert numeric predictions to original class labels
        results_df['predicted_class'] = self.label_encoder.inverse_transform(predictions)

        # Add probability columns for actual classes used in training
        for i, class_idx in enumerate(unique_classes):
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            results_df[f'prob_{class_name}'] = probabilities[:, i]

        # Add confidence metrics
        results_df['max_probability'] = probabilities.max(axis=1)

        if self.target_column in results_df:
            # Calculate confidence threshold based on number of classes
            confidence_threshold = 1.5 / n_classes

            # Get true class probabilities
            true_indices = self.label_encoder.transform(results_df[self.target_column])
            true_probs = probabilities[np.arange(len(true_indices)), true_indices]

            # Add confidence metrics
            correct_prediction = (predictions == true_indices)
            prob_diff = results_df['max_probability'] - true_probs

            results_df['confidence_verdict'] = np.where(
                (prob_diff < confidence_threshold) & correct_prediction,
                'High Confidence',
                'Low Confidence'
            )

        return results_df

    def _update_training_log(self, round_num: int, metrics: Dict):
        """Update training log with current metrics"""
        self.training_log = self.training_log.append({
            'timestamp': pd.Timestamp.now(),
            'round': round_num,
            'train_size': metrics['train_size'],
            'test_size': metrics['test_size'],
            'train_loss': metrics['train_loss'],
            'test_loss': metrics['test_loss'],
            'train_accuracy': metrics['train_accuracy'],
            'test_accuracy': metrics['test_accuracy'],
            'training_time': metrics['training_time']
        }, ignore_index=True)
#--------------------------------------------------------------Class Ends ------------------------------------
    def prepare_batch(self, features):
        """Efficient batch preparation"""
        if not features.is_contiguous():
            features = features.contiguous()

        if self.device.startswith('cuda') and not features.is_cuda:
            features = features.cuda(non_blocking=True)

        return features

    def _load_dataset(self) -> pd.DataFrame:
        """Load and preprocess dataset with improved error handling"""
        DEBUG.log(f" Loading dataset from config: {self.data_config}")
        try:
            # Validate dataset configuration
            if self.data_config is None:
                raise ValueError(f"No dataset configuration found for: {self.dataset_name}")

            file_path = self.data_config.get('file_path')
            if file_path is None:
                raise ValueError(f"No file path specified in dataset configuration for: {self.dataset_name}")

            # Handle URL or local file
            try:
                if file_path.startswith(('http://', 'https://')):
                    DEBUG.log(f" Loading from URL: {file_path}")
                    response = requests.get(file_path)
                    response.raise_for_status()
                    data = StringIO(response.text)
                else:
                    DEBUG.log(f" Loading from local file: {file_path}")
                    # Try the configured path first
                    if not os.path.exists(file_path):
                        # Try looking in data/dataset_name directory
                        alt_path = os.path.join('data', self.dataset_name, f"{self.dataset_name}.csv")
                        if os.path.exists(alt_path):
                            file_path = alt_path
                            DEBUG.log(f" Found file at alternate path: {file_path}")
                        else:
                            raise FileNotFoundError(f"Dataset file not found at either {file_path} or {alt_path}")
                    data = file_path

                # Read CSV with appropriate parameters from dataset config
                read_params = {
                    'sep': self.data_config.get('separator', ','),
                    'header': 0 if self.data_config.get('has_header', True) else None,
                    'names': self.data_config.get('column_names'),
                }
                DEBUG.log(f" Reading CSV with parameters: {read_params}")

                df = pd.read_csv(data, **read_params)

                # Filter features if specified in config
                if '#' in str(self.data_config.get('column_names')):
                    df = _filter_features_from_config(df, self.data_config)

                return df

            except requests.exceptions.RequestException as e:
                DEBUG.log(f" Error downloading dataset from URL: {str(e)}")
                raise RuntimeError(f"Failed to download dataset from URL: {str(e)}")

            except pd.errors.EmptyDataError:
                DEBUG.log(f" Error: Dataset file is empty")
                raise ValueError(f"Dataset file is empty: {file_path}")

            except pd.errors.ParserError as e:
                DEBUG.log(f" Error parsing CSV file: {str(e)}")
                raise ValueError(f"Invalid CSV format: {str(e)}")

        except Exception as e:
            DEBUG.log(f" Error loading dataset: {str(e)}")
            DEBUG.log(" Stack trace:", traceback.format_exc())
            raise RuntimeError(f"Failed to load dataset: {str(e)}")

    def _compute_batch_posterior(self, features: Union[torch.Tensor, pd.DataFrame], epsilon: float = 1e-10):
       """Optimized batch posterior with vectorized operations and type checking"""
       # Safety checks and type conversion
       if isinstance(features, pd.DataFrame):
           features = torch.FloatTensor(features.values).to(self.device)

       if self.weight_updater is None:
           DEBUG.log(" Weight updater not initialized, initializing now...")
           self._initialize_bin_weights()
           if self.weight_updater is None:
               raise RuntimeError("Failed to initialize weight updater")

       if self.likelihood_params is None:
           raise RuntimeError("Likelihood parameters not initialized")

       # Ensure input features are contiguous
       features = features if features.is_contiguous() else features.contiguous()
       batch_size = features.shape[0]
       n_classes = len(self.likelihood_params['classes'])

       # Pre-allocate tensors
       log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

       # Process all feature pairs at once
       feature_pairs = self.likelihood_params['feature_pairs'].to(features.device)
       feature_groups = torch.stack([
           features[:, pair].contiguous()
           for pair in feature_pairs
       ]).transpose(0, 1)  # [batch_size, n_pairs, 2]

       # Compute all bin indices at once
       bin_indices_dict = {}
       for group_idx in range(len(self.likelihood_params['feature_pairs'])):
           bin_edges = self.likelihood_params['bin_edges'][group_idx]
           edges = torch.stack([edge.contiguous() for edge in bin_edges])

           # Vectorized binning with contiguous tensors
           indices = torch.stack([
               torch.bucketize(
                   feature_groups[:, group_idx, dim].contiguous(),
                   edges[dim].contiguous()
               )
               for dim in range(2)
           ])  # [2, batch_size]
           indices = indices.sub_(1).clamp_(0, self.n_bins_per_dim - 1)
           bin_indices_dict[group_idx] = indices

       # Process all classes simultaneously
       for group_idx in range(len(self.likelihood_params['feature_pairs'])):
           bin_probs = self.likelihood_params['bin_probs'][group_idx]  # [n_classes, n_bins, n_bins]
           indices = bin_indices_dict[group_idx]  # [2, batch_size]

           # Get all weights at once
           weights = torch.stack([
               self.weight_updater.get_histogram_weights(c, group_idx)
               for c in range(n_classes)
           ])  # [n_classes, n_bins, n_bins]

           # Ensure weights are contiguous
           weights = weights if weights.is_contiguous() else weights.contiguous()

           # Apply weights to probabilities
           weighted_probs = bin_probs * weights  # [n_classes, n_bins, n_bins]

           # Gather probabilities for all samples and classes at once
           probs = weighted_probs[:, indices[0], indices[1]]  # [n_classes, batch_size]
           log_likelihoods += torch.log(probs.t() + epsilon)

       # Compute posteriors efficiently
       max_log_likelihood = log_likelihoods.max(dim=1, keepdim=True)[0]
       posteriors = torch.exp(log_likelihoods - max_log_likelihood)
       posteriors /= posteriors.sum(dim=1, keepdim=True) + epsilon

       return posteriors, bin_indices_dict if self.model_type == "Histogram" else None
#----------------------

    def set_feature_bounds(self, dataset):
        """Initialize global feature bounds from complete dataset"""
        if self.feature_bounds is None:
            self.feature_bounds = {}
            numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns

            for feat_name in numerical_columns:
                feat_data = dataset[feat_name]
                min_val = feat_data.min()
                max_val = feat_data.max()
                padding = (max_val - min_val) * 0.01
                self.feature_bounds[feat_name] = {
                    'min': min_val - padding,
                    'max': max_val + padding
                }

    def _clean_existing_model(self):
        """Remove existing model files for a fresh start"""
        try:
            files_to_remove = [
                self._get_weights_filename(),
                self._get_encoders_filename(),
                self._get_model_components_filename()
            ]
            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Removed existing model file: {file}")
        except Exception as e:
            print(f"Warning: Error cleaning model files: {str(e)}")


    #------------------------------------------Adaptive Learning--------------------------------------
    def save_epoch_data(self, epoch: int, train_indices: list, test_indices: list):
        """
        Save training and testing indices for each epoch if enabled in config
        """
        # Check if epoch saving is enabled
        save_epochs = self.config.get('training_params', {}).get('Save_training_epochs', False)
        if not save_epochs:
            return

        # Use dataset name as subfolder
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
        save_path = os.path.join(base_path, dataset_folder)

        # Create epoch directory
        epoch_dir = os.path.join(save_path, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        # Save indices
        try:
            with open(os.path.join(epoch_dir, f'{self.model_type}_train_indices.pkl'), 'wb') as f:
                pickle.dump(train_indices, f)
            with open(os.path.join(epoch_dir, f'{self.model_type}_test_indices.pkl'), 'wb') as f:
                pickle.dump(test_indices, f)
            print(f"Saved epoch {epoch} data to {epoch_dir}")
        except Exception as e:
            print(f"Error saving epoch data: {str(e)}")

    def load_epoch_data(self, epoch: int):
        """
        Load training and testing indices for a specific epoch
        """
        epoch_dir = os.path.join(self.base_save_path, f'epoch_{epoch}')

        with open(os.path.join(epoch_dir, f'{self.model_type}_train_indices.pkl'), 'rb') as f:
            train_indices = pickle.load(f)
        with open(os.path.join(epoch_dir, f'{self.model_type}_test_indices.pkl'), 'rb') as f:
            test_indices = pickle.load(f)

        return train_indices, test_indices

    def _compute_cardinality_threshold(self, cardinalities, config=None):
        """
        Compute the cardinality threshold using percentile-based analysis.

        Args:
            cardinalities: List of cardinality values
            config: Configuration dictionary with threshold parameters

        Returns:
            float: Computed cardinality threshold
        """
        # Get active learning parameters from config
        active_learning_config = self.config.get('active_learning', {})
        percentile = active_learning_config.get('cardinality_threshold_percentile', 95)

        # If no cardinalities provided, create a synthetic distribution
        if not cardinalities:
            print("\nWarning: No cardinality data available. Using synthetic distribution based on percentile.")
            # Create a synthetic distribution around the percentile threshold
            cardinalities = np.array([1.0, 2.0, 5.0, 10.0, 20.0])  # Synthetic values
        else:
            cardinalities = np.array(cardinalities)

        # Compute basic statistics
        min_card = np.min(cardinalities)
        max_card = np.max(cardinalities)
        mean_card = np.mean(cardinalities)
        median_card = np.median(cardinalities)

        # Compute threshold using percentile
        threshold = np.percentile(cardinalities, percentile)

        # Print detailed analysis
        print(f"\nCardinality Analysis:")
        print(f"- Using {percentile}th percentile threshold")
        print(f"- Distribution statistics:")
        print(f"  - Min: {min_card:.2f}")
        print(f"  - Max: {max_card:.2f}")
        print(f"  - Mean: {mean_card:.2f}")
        print(f"  - Median: {median_card:.2f}")
        print(f"  - Threshold: {threshold:.2f}")

        # Print number of samples that would be included
        n_included = sum(c <= threshold for c in cardinalities)
        print(f"- {n_included} out of {len(cardinalities)} samples below threshold "
              f"({(n_included/len(cardinalities))*100:.1f}%)")

        return threshold

    def _compute_sample_divergence(self, sample_data: torch.Tensor, feature_pairs: List[Tuple]) -> torch.Tensor:
        """
        Vectorized computation of pairwise feature divergence.
        """
        n_samples = sample_data.shape[0]
        if n_samples <= 1:
            return torch.zeros((1, 1), device=self.device)

        # Pre-allocate tensor for pair distances
        pair_distances = torch.zeros((len(feature_pairs), n_samples, n_samples),
                                   device=self.device)

        # Compute distances for all pairs in one batch
        for i, pair in enumerate(feature_pairs):
            pair_data = sample_data[:, pair]
            # Vectorized pairwise difference computation
            diff = pair_data.unsqueeze(1) - pair_data.unsqueeze(0)
            pair_distances[i] = torch.norm(diff, dim=2)

        # Average across feature pairs
        distances = torch.mean(pair_distances, dim=0)

        # Normalize
        if distances.max() > 0:
            distances /= distances.max()

        return distances

    def _compute_feature_cardinalities(self, samples_data: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of feature cardinalities.
        """
        cardinalities = torch.zeros(len(samples_data), device=self.device)

        # Process feature pairs in batches
        batch_size = 100  # Adjust based on memory constraints
        for i in range(0, len(samples_data), batch_size):
            batch_end = min(i + batch_size, len(samples_data))
            batch_data = samples_data[i:batch_end]

            # Compute cardinalities for each feature pair
            batch_cardinalities = torch.zeros(batch_end - i, device=self.device)
            for feat_pair in self.feature_pairs:
                pair_data = batch_data[:, feat_pair]
                # Compute unique values efficiently
                _, counts = torch.unique(pair_data, dim=0, return_counts=True)
                batch_cardinalities += len(counts)

            cardinalities[i:batch_end] = batch_cardinalities

        return cardinalities

    def _calculate_optimal_batch_size(self, sample_tensor_size):
        """
        Calculate optimal batch size based on available GPU memory and sample size.

        Args:
            sample_tensor_size: Size of one sample tensor in bytes

        Returns:
            optimal_batch_size: int
        """
        if not torch.cuda.is_available():
            return 128  # Default for CPU

        try:
            # Get total and reserved GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = torch.cuda.memory_reserved(0)
            allocated_memory = torch.cuda.memory_allocated(0)

            # Calculate available memory (leaving 20% as buffer)
            available_memory = (total_memory - reserved_memory - allocated_memory) * 0.8

            # Calculate memory needed per sample (with buffer for intermediate computations)
            memory_per_sample = sample_tensor_size * 4  # Factor of 4 for intermediate computations

            # Calculate optimal batch size
            optimal_batch_size = int(available_memory / memory_per_sample)

            # Enforce minimum and maximum bounds
            optimal_batch_size = max(32, min(optimal_batch_size, 512))

            DEBUG.log(f" Memory Analysis:")
            DEBUG.log(f" - Total GPU Memory: {total_memory / 1e9:.2f} GB")
            DEBUG.log(f" - Reserved Memory: {reserved_memory / 1e9:.2f} GB")
            DEBUG.log(f" - Allocated Memory: {allocated_memory / 1e9:.2f} GB")
            DEBUG.log(f" - Available Memory: {available_memory / 1e9:.2f} GB")
            DEBUG.log(f" - Memory per sample: {memory_per_sample / 1e6:.2f} MB")
            DEBUG.log(f" - Optimal batch size: {optimal_batch_size}")

            return optimal_batch_size

        except Exception as e:
            DEBUG.log(f" Error calculating batch size: {str(e)}")
            return 128  # Default fallback

    def _select_samples_from_failed_classes(self, test_predictions, y_test, test_indices):
        """
        Memory-efficient implementation of sample selection using batched processing
        """
        # Configuration parameters
        active_learning_config = self.config.get('active_learning', {})
        tolerance = active_learning_config.get('tolerance', 1.0) / 100.0
        min_divergence = active_learning_config.get('min_divergence', 0.1)
        strong_margin_threshold = active_learning_config.get('strong_margin_threshold', 0.3)
        marginal_margin_threshold = active_learning_config.get('marginal_margin_threshold', 0.1)

        # Calculate optimal batch size based on sample size
        sample_size = self.X_tensor[0].element_size() * self.X_tensor[0].nelement()
        batch_size = self._calculate_optimal_batch_size(sample_size)
        DEBUG.log(f"\nUsing dynamic batch size: {batch_size}")

        test_predictions = torch.as_tensor(test_predictions, device=self.device)
        y_test = torch.as_tensor(y_test, device=self.device)
        test_indices = torch.as_tensor(test_indices, device=self.device)

        misclassified_mask = (test_predictions != y_test)
        misclassified_indices = torch.nonzero(misclassified_mask).squeeze()

        if  misclassified_indices.dim() == 0:
            return []

        final_selected_indices = []
        unique_classes = torch.unique(y_test[misclassified_indices])

        for class_id in unique_classes:
            class_mask = y_test[misclassified_indices] == class_id
            class_indices = misclassified_indices[class_mask]

            if len(class_indices) == 0:
                continue

            # Process class samples in batches
            for batch_start in range(0, len(class_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(class_indices))
                batch_indices = class_indices[batch_start:batch_end]

                # Get batch data
                batch_samples = self.X_tensor[test_indices[batch_indices]]

                # Compute probabilities for batch
                if self.model_type == "Histogram":
                    probs, _ = self._compute_batch_posterior(batch_samples)
                else:
                    probs, _ = self._compute_batch_posterior_std(batch_samples)

                # Compute error margins for batch
                true_probs = probs[:, class_id]
                pred_classes = torch.argmax(probs, dim=1)
                pred_probs = probs[torch.arange(len(pred_classes)), pred_classes]
                error_margins = pred_probs - true_probs

                # Split into strong and marginal failures
                strong_failures = error_margins >= strong_margin_threshold
                marginal_failures = (error_margins > 0) & (error_margins < marginal_margin_threshold)

                # Process each failure type
                for failure_type, failure_mask in [("strong", strong_failures), ("marginal", marginal_failures)]:
                    if not failure_mask.any():
                        continue

                    # Get failure samples for this batch
                    failure_samples = batch_samples[failure_mask]
                    failure_margins = error_margins[failure_mask]
                    failure_indices = test_indices[batch_indices[failure_mask]]

                    # Compute cardinalities for these samples
                    cardinalities = self._compute_feature_cardinalities(failure_samples)

                    # Use dynamic threshold based on distribution
                    cardinality_threshold = torch.median(cardinalities)
                    low_card_mask = cardinalities <= cardinality_threshold

                    if not low_card_mask.any():
                        continue

                    # Process samples meeting cardinality criteria
                    low_card_samples = failure_samples[low_card_mask]
                    low_card_margins = failure_margins[low_card_mask]
                    low_card_indices = failure_indices[low_card_mask]

                    # Compute divergences only for low cardinality samples
                    divergences = self._compute_sample_divergence(low_card_samples, self.feature_pairs)

                    # Select diverse samples efficiently
                    selected_mask = torch.zeros(len(low_card_samples), dtype=torch.bool, device=self.device)

                    # Initialize with best margin sample
                    if failure_type == "strong":
                        best_idx = torch.argmax(low_card_margins)
                    else:
                        best_idx = torch.argmin(low_card_margins)
                    selected_mask[best_idx] = True

                    # Add diverse samples meeting divergence criterion
                    while True:
                        # Compute minimum divergence to selected samples
                        min_divs = divergences[:, selected_mask].min(dim=1)[0]
                        candidate_mask = (~selected_mask) & (min_divs >= min_divergence)

                        if not candidate_mask.any():
                            break

                        # Select next sample based on margin type
                        candidate_margins = low_card_margins.clone()
                        candidate_margins[~candidate_mask] = float('inf') if failure_type == "marginal" else float('-inf')

                        best_idx = torch.argmin(candidate_margins) if failure_type == "marginal" else torch.argmax(candidate_margins)
                        selected_mask[best_idx] = True

                    # Add selected indices
                    selected_indices = low_card_indices[selected_mask]
                    final_selected_indices.extend(selected_indices.cpu().tolist())

                    # Print selection info
                    true_class_name = self.label_encoder.inverse_transform([class_id.item()])[0]
                    n_selected = selected_mask.sum().item()
                    DEBUG.log(f" Selected {n_selected} {failure_type} failure samples from class {true_class_name}")
                    DEBUG.log(f" - Cardinality threshold: {cardinality_threshold:.3f}")
                    DEBUG.log(f" - Average margin: {low_card_margins[selected_mask].mean().item():.3f}")

                # Clear cache after processing each batch
                torch.cuda.empty_cache()

        print(f"\nTotal samples selected: {len(final_selected_indices)}")
        return final_selected_indices

    def adaptive_fit_predict(self, max_rounds: int = None,
                             improvement_threshold: float = 0.001,
                             load_epoch: int = None,
                             batch_size: int = 32):
        """Modified adaptive training strategy with proper fresh start handling and accuracy tracking"""
        DEBUG.log(" Starting adaptive_fit_predict")
        if max_rounds is None:
            max_rounds = self.config.epochs

        # Check if adaptive learning is enabled in config
        if hasattr(self.config, 'to_dict'):
            enable_adaptive = self.config.enable_adaptive
        elif isinstance(self.config, dict):
            enable_adaptive = self.config.get('training_params', {}).get('enable_adaptive', True)
        else:
            enable_adaptive = True

        if not enable_adaptive:
            print("Adaptive learning is disabled. Using standard training.")
            return self.fit_predict(batch_size=batch_size)

        self.in_adaptive_fit = True
        train_indices = []
        test_indices = None
        best_test_accuracy = 0.0

        try:
            # Get initial data
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            DEBUG.log(f" Initial data shape: X={X.shape}, y={len(y)}")

            # Initialize label encoder if not already done
            if not hasattr(self.label_encoder, 'classes_'):
                self.label_encoder.fit(y)

            # Use existing label encoder
            y_encoded = self.label_encoder.transform(y)

            # Process features and initialize model components if needed
            X_processed = self._preprocess_data(X, is_training=True)
            self.X_tensor = torch.FloatTensor(X_processed).to(self.device)
            self.y_tensor = torch.LongTensor(y_encoded).to(self.device)

            # Handle model state based on flags
            model_loaded = False
            if self.use_previous_model:
                print("Loading previous model state")
                if self._load_model_components():
                    self._load_best_weights()
                    self._load_categorical_encoders()
                    model_loaded = True

                    if not self.fresh_start:
                        # Load previous training data
                        print("Loading previous training data...")
                        prev_train_file = f'{self.dataset_name}_Last_training.csv'
                        if os.path.exists(prev_train_file):
                            prev_train_data = pd.read_csv(prev_train_file)
                            train_indices, test_indices = self.load_last_known_split()
                            if train_indices is None:
                                print("No valid previous split found - starting fresh")
                                train_indices = []
                                test_indices = list(range(len(X)))
                        else:
                            print("No previous training data found - starting fresh")
                            train_indices = []
                            test_indices = list(range(len(X)))

            if not model_loaded:
                print("Initializing fresh model")
                self._clean_existing_model()
                train_indices = []
                test_indices = list(range(len(X)))

                # Initialize feature pairs for fresh start
                self.feature_pairs = self._generate_feature_combinations(
                    self.X_tensor.shape[1],
                    self.config.get('likelihood_config', {}).get('feature_group_size', 2),
                    self.config.get('likelihood_config', {}).get('max_combinations', None)
                )

            # Initialize test indices if still None
            if test_indices is None:
                test_indices = list(range(len(X)))

            # Initialize likelihood parameters if needed
            if self.likelihood_params is None:
                DEBUG.log(" Initializing likelihood parameters")
                if self.model_type == "Histogram":
                    self.likelihood_params = self._compute_pairwise_likelihood_parallel(
                        self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
                    )
                elif self.model_type == "Gaussian":
                    self.likelihood_params = self._compute_pairwise_likelihood_parallel_std(
                        self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
                    )
                DEBUG.log(" Likelihood parameters computed")

            # Initialize weights if needed
            if self.weight_updater is None:
                DEBUG.log(" Initializing weight updater")
                self._initialize_bin_weights()
                DEBUG.log(" Weight updater initialized")

            # Initialize model weights if needed
            if self.current_W is None:
                DEBUG.log(" Initializing model weights")
                n_classes = len(self.label_encoder.classes_)
                n_pairs = len(self.feature_pairs) if self.feature_pairs is not None else 0
                if n_pairs == 0:
                    raise ValueError("Feature pairs not initialized")
                self.current_W = torch.full(
                    (n_classes, n_pairs),
                    0.1,
                    device=self.device,
                    dtype=torch.float32
                )
                if self.best_W is None:
                    self.best_W = self.current_W.clone()

            # Initialize training set if empty
            if len(train_indices) == 0:
                print("Initializing new training set with minimum samples")
                # Select minimum samples from each class for initial training
                unique_classes = self.label_encoder.classes_
                for class_label in unique_classes:
                    class_indices = np.where(y_encoded == self.label_encoder.transform([class_label])[0])[0]
                    if len(class_indices) < 2:
                        selected_indices = class_indices
                    else:
                        selected_indices = class_indices[:2]
                    train_indices.extend(selected_indices)

                # Update test indices
                test_indices = list(set(range(len(X))) - set(train_indices))

            DEBUG.log(f" Initial training set size: {len(train_indices)}")
            DEBUG.log(f" Initial test set size: {len(test_indices)}")

            # Training loop
            last_results = None
            adaptive_patience_counter = 0
            best_train_accuracy = 0.0

            for round_num in range(max_rounds):
                print(f"\nRound {round_num + 1}/{max_rounds}")
                print(f"Training set size: {len(train_indices)}")
                print(f"Test set size: {len(test_indices)}")

                # Save indices for this epoch
                self.save_epoch_data(round_num, train_indices, test_indices)

                # Set current indices for training
                self.train_indices = train_indices
                self.test_indices = test_indices

                # Train and get results
                save_path = f"round_{round_num}_predictions.csv"
                results = self.fit_predict(batch_size=batch_size, save_path=save_path)

                if results is None:
                    print("Training failed - stopping adaptive process")
                    break

                # Track accuracies
                train_accuracy = results.get('train_accuracy', 0.0)
                test_accuracy = results.get('test_accuracy', 0.0)

                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy

                # Check improvement
                improved = False
                if train_accuracy > best_train_accuracy + improvement_threshold:
                    best_train_accuracy = train_accuracy
                    improved = True
                    print(f"Improved training accuracy to {train_accuracy:.4f}")

                if not improved:
                    adaptive_patience_counter += 1
                    print(f"No significant improvement. Adaptive patience: {adaptive_patience_counter}/5")
                    if adaptive_patience_counter >= 5:
                        print(f"No improvement after 5 rounds of adding samples.")
                        print(f"Best training accuracy: {best_train_accuracy:.4f}")
                        print(f"Best test accuracy: {best_test_accuracy:.4f}")
                        break

                # Check if training reached perfection
                min_training_accuracy = self._get_config_param('minimum_training_accuracy', 0.95)
                if train_accuracy == 1.0:
                    if len(test_indices) == 0:
                        print("No more test samples available. Training complete.")
                        break

                # Select new training samples
                X_test = self.X_tensor[test_indices]
                y_test = self.y_tensor[test_indices]
                test_predictions = self.predict(X_test, batch_size=batch_size)

                new_train_indices = self._select_samples_from_failed_classes(
                    test_predictions, y_test, test_indices
                )

                if not new_train_indices:
                    print("No suitable new samples found. Training complete.")
                    break

                # Update indices for next round
                train_indices.extend(new_train_indices)
                test_indices = list(set(test_indices) - set(new_train_indices))
                print(f"Added {len(new_train_indices)} new samples to training set")

                # Save current split
                self.save_last_split(train_indices, test_indices)
                last_results = results

            # Prepare final results
            final_results = {
                'train_indices': train_indices,
                'test_indices': test_indices,
                'train_accuracy': best_train_accuracy,
                'test_accuracy': best_test_accuracy,
                'error_rates': last_results.get('error_rates', []) if last_results else [],
                'confusion_matrix': last_results.get('confusion_matrix') if last_results else None,
                'classification_report': last_results.get('classification_report', '') if last_results else '',
                'training_complete': True
            }

            self.in_adaptive_fit = False
            return final_results

        except Exception as e:
            DEBUG.log(f" Error in adaptive_fit_predict: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            self.in_adaptive_fit = False
            return None
    #------------------------------------------Adaptive Learning--------------------------------------


    def _calculate_cardinality_threshold(self):
            """Calculate appropriate cardinality threshold based on dataset characteristics"""
            n_samples = len(self.data)
            n_classes = len(self.data[self.target_column].unique())

            # Get threshold from config
            if hasattr(self.config, 'to_dict'):
                # GlobalConfig object
                base_threshold = self.config.cardinality_threshold
            elif isinstance(self.config, dict):
                # Dictionary config
                base_threshold = self.config.get('training_params', {}).get('cardinality_threshold', 0.9)
            else:
                # Default value
                base_threshold = 0.9

            # Adjust threshold based on dataset size and number of classes
            adjusted_threshold = min(
                base_threshold,
                max(0.1, 1.0 / np.sqrt(n_classes))  # Lower bound of 0.1
            )

            DEBUG.log(f"\nCardinality Threshold Calculation:")
            DEBUG.log(f"- Base threshold from config: {base_threshold}")
            DEBUG.log(f"- Number of samples: {n_samples}")
            DEBUG.log(f"- Number of classes: {n_classes}")
            DEBUG.log(f"- Adjusted threshold: {adjusted_threshold}")

            return adjusted_threshold

    def _get_config_value(self, param_name: str, default_value: Any) -> Any:
        """Helper method to get configuration values consistently"""
        if hasattr(self.config, 'to_dict'):
            return getattr(self.config, param_name, default_value)
        elif isinstance(self.config, dict):
            return self.config.get('training_params', {}).get(param_name, default_value)
        else:
            return default_value

    def _round_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round features based on cardinality_tolerance"""
        if self.cardinality_tolerance == -1:
            return df
        return df.round(self.cardinality_tolerance)

    def _remove_high_cardinality_columns(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Remove high cardinality columns with more conservative approach"""
        DEBUG.log(f"Starting cardinality analysis with threshold {threshold}")
        if threshold is None:
            threshold = self._get_config_value('cardinality_threshold', 0.8)
        if tolerance != -1:
            DEBUG.log(f"Rounding features with tolerance {tolerance}")
            df_rounded = self._round_features(df)
        else:
            df_rounded = df.copy()

        df_filtered = df_rounded.copy()
        columns_to_drop = []
        cardinality_info = {}

        # First pass: calculate cardinality for all columns
        for column in df.columns:
            if column == self.target_column:
                continue

            unique_count = len(df[column].unique())
            unique_ratio = unique_count / len(df)
            cardinality_info[column] = {
                'unique_count': unique_count,
                'ratio': unique_ratio
            }

            DEBUG.log(f"Column {column}: {unique_count} unique values, ratio {unique_ratio:.4f}")

        # Determine adaptive threshold
        ratios = [info['ratio'] for info in cardinality_info.values()]
        if ratios:
            median_ratio = np.median(ratios)
            adaptive_threshold = min(threshold, max(median_ratio * 2, 0.1))
            DEBUG.log(f"Adaptive threshold: {adaptive_threshold} (original: {threshold})")
        else:
            adaptive_threshold = threshold

        # Second pass: mark columns for dropping
        for column, info in cardinality_info.items():
            if info['ratio'] > adaptive_threshold:
                columns_to_drop.append(column)
                DEBUG.log(f"Marking {column} for removal (ratio: {info['ratio']:.4f})")

        # Ensure we keep at least some features
        if len(columns_to_drop) == len(cardinality_info):
            DEBUG.log("Would remove all features - keeping lowest cardinality ones")
            sorted_columns = sorted(cardinality_info.items(), key=lambda x: x[1]['ratio'])
            keep_count = max(2, len(cardinality_info) // 5)  # Keep at least 2 or 20%
            columns_to_drop = [col for col, _ in sorted_columns[keep_count:]]

        # Drop columns
        if columns_to_drop:
            df_filtered = df_filtered.drop(columns=columns_to_drop)
            DEBUG.log(f"Dropped columns: {columns_to_drop}")

        DEBUG.log(f"Features after cardinality filtering: {df_filtered.columns.tolist()}")
        return df_filtered

    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect categorical columns with improved debugging"""
        DEBUG.log(" Starting categorical column detection")
        categorical_columns = []

        for column in df.columns:
            if column != self.target_column:
                dtype = df[column].dtype
                unique_count = len(df[column].unique())
                DEBUG.log(f" Column {column}: dtype={dtype}, unique values={unique_count}")

                if dtype == 'object' or dtype.name == 'category':
                    DEBUG.log(f" Adding {column} as categorical (object/category type)")
                    categorical_columns.append(column)
                elif dtype in ['int64', 'float64']:
                    threshold = min(50, len(df) * 0.05)
                    if unique_count < threshold:
                        DEBUG.log(f" Adding {column} as categorical (numeric with few unique values: {unique_count} < {threshold})")
                        categorical_columns.append(column)
                    else:
                        DEBUG.log(f" Keeping {column} as numeric (unique values: {unique_count} >= {threshold})")

        DEBUG.log(f" Detected categorical columns: {categorical_columns}")
        return categorical_columns

    def _preprocess_data(self, X: pd.DataFrame, is_training: bool = True) -> torch.Tensor:
        """Preprocess data with improved error handling and column consistency"""
        print(f"\n[DEBUG] ====== Starting preprocessing ======")
        DEBUG.log(f" Input shape: {X.shape}")
        DEBUG.log(f" Input columns: {X.columns.tolist()}")
        DEBUG.log(f" Input dtypes:\n{X.dtypes}")

        # Make a copy to avoid modifying original data
        X = X.copy()

        if is_training:
            DEBUG.log(" Training mode preprocessing")
            self.original_columns = X.columns.tolist()

            with tqdm(total=4, desc="Preprocessing steps") as pbar:

                # Calculate cardinality threshold
                cardinality_threshold = self._calculate_cardinality_threshold()
                DEBUG.log(f" Cardinality threshold: {cardinality_threshold}")

                # Remove high cardinality columns
                X = self._remove_high_cardinality_columns(X, cardinality_threshold)
                DEBUG.log(f" Shape after cardinality filtering: {X.shape}")

                # Store the features we'll actually use
                self.feature_columns = X.columns.tolist()
                DEBUG.log(f" Selected feature columns: {self.feature_columns}")

                # Store high cardinality columns for future reference
                self.high_cardinality_columns = list(set(self.original_columns) - set(self.feature_columns))
                if self.high_cardinality_columns:
                    DEBUG.log(f" Removed high cardinality columns: {self.high_cardinality_columns}")
        else:
            DEBUG.log(" Prediction mode preprocessing")
            if not hasattr(self, 'feature_columns'):
                raise ValueError("Model not trained - feature columns not found")

            # For prediction, only try to use columns that were used during training
            available_cols = set(X.columns)
            needed_cols = set(self.feature_columns)

            # Check for missing columns
            missing_cols = needed_cols - available_cols
            if missing_cols:
                # Create missing columns with default values
                for col in missing_cols:
                    X[col] = 0
                    DEBUG.log(f" Created missing column {col} with default value 0")

            # Only keep the columns we used during training
            X = X[self.feature_columns]

            if hasattr(self, 'high_cardinality_columns'):
                X = X.drop(columns=self.high_cardinality_columns, errors='ignore')

        print("Preprocessing prediction data...")
        with tqdm(total=2, desc="Preprocessing steps") as pbar:

            # Handle categorical features
            DEBUG.log(" Starting categorical encoding")
            try:
                X_encoded = self._encode_categorical_features(X, is_training)
                DEBUG.log(f" Shape after categorical encoding: {X_encoded.shape}")
                DEBUG.log(f" Encoded dtypes:\n{X_encoded.dtypes}")
            except Exception as e:
                DEBUG.log(f" Error in categorical encoding: {str(e)}")
                raise

            # Convert to numpy and check for issues
            try:
                X_numpy = X_encoded.to_numpy()
                DEBUG.log(f" Numpy array shape: {X_numpy.shape}")
                DEBUG.log(f" Any NaN: {np.isnan(X_numpy).any()}")
                DEBUG.log(f" Any Inf: {np.isinf(X_numpy).any()}")
            except Exception as e:
                DEBUG.log(f" Error converting to numpy: {str(e)}")
                raise

            # Scale the features
            try:
                if is_training:
                    X_scaled = self.scaler.fit_transform(X_numpy)
                else:
                    X_scaled = self.scaler.transform(X_numpy)

                DEBUG.log(f" Scaling successful")
            except Exception as e:
                DEBUG.log(f" Standard scaling failed: {str(e)}. Using manual scaling")
            pbar.update(1)
            if X_numpy.size == 0:
                print("[WARNING] Empty feature array! Returning original data")
                X_scaled = X_numpy
            else:
                means = np.nanmean(X_numpy, axis=0)
                stds = np.nanstd(X_numpy, axis=0)
                stds[stds == 0] = 1
                X_scaled = (X_numpy - means) / stds

        DEBUG.log(f" Final preprocessed shape: {X_scaled.shape}")
        pbar.close()
        return torch.FloatTensor(X_scaled)

    def _generate_feature_combinations(self, n_features: int, group_size: int, max_combinations: int = None) -> torch.Tensor:
        """Generate and save/load consistent feature combinations"""
        # Create path for storing feature combinations
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
        combinations_path = os.path.join(base_path, dataset_folder, 'feature_combinations.pkl')

        # Check if combinations already exist
        if os.path.exists(combinations_path):
            with open(combinations_path, 'rb') as f:
                combinations_tensor = pickle.load(f)
                return combinations_tensor.to(self.device)

        # Generate new combinations if none exist
        if n_features < group_size:
            raise ValueError(f"Number of features ({n_features}) must be >= group size ({group_size})")

        # Generate all possible combinations
        all_combinations = list(combinations(range(n_features), group_size))
        if not all_combinations:
            raise ValueError(f"No valid combinations generated for {n_features} features in groups of {group_size}")

        # Sample combinations if max_combinations specified
        if max_combinations and len(all_combinations) > max_combinations:
            # Convert list of tuples to numpy array for sampling
            combinations_array = np.array(all_combinations)
            rng = np.random.RandomState(42)
            selected_indices = rng.choice(len(combinations_array), max_combinations, replace=False)
            all_combinations = combinations_array[selected_indices]

        # Convert to tensor
        combinations_tensor = torch.tensor(all_combinations, device=self.device)

        # Save combinations for future use
        os.makedirs(os.path.dirname(combinations_path), exist_ok=True)
        with open(combinations_path, 'wb') as f:
            pickle.dump(combinations_tensor.cpu(), f)

        return combinations_tensor
#-----------------------------------------------------------------------------Bin model ---------------------------

    def _compute_pairwise_likelihood_parallel(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """Optimized non-parametric likelihood computation with configurable bin sizes"""
        DEBUG.log(" Starting _compute_pairwise_likelihood_parallel")
        print("\nComputing pairwise likelihoods...")
        # Input validation and preparation
        dataset = torch.as_tensor(dataset, device=self.device).contiguous()
        labels = torch.as_tensor(labels, device=self.device).contiguous()

        # Initialize progress tracking
        n_pairs = len(self.feature_pairs)
        pair_pbar = tqdm(total=n_pairs, desc="Processing feature pairs")

        # Pre-compute unique classes once
        unique_classes, class_counts = torch.unique(labels, return_counts=True)
        n_classes = len(unique_classes)
        n_samples = len(dataset)

        # Get bin sizes from configuration
        bin_sizes = self.config.get('likelihood_config', {}).get('bin_sizes', [5])
        if len(bin_sizes) == 1:
            # If single bin size provided, use it for all dimensions
            n_bins = bin_sizes[0]
            self.n_bins_per_dim = n_bins  # Store for reference
            DEBUG.log(f" Using uniform {n_bins} bins per dimension")
        else:
            DEBUG.log(f" Using variable bin sizes: {bin_sizes}")

        # Generate feature combinations
        self.feature_pairs = self._generate_feature_combinations(
            feature_dims,
            self.config.get('likelihood_config', {}).get('feature_group_size', 2),
            self.config.get('likelihood_config', {}).get('max_combinations', None)
        ).to(self.device)

        # Pre-allocate storage arrays
        all_bin_edges = []
        all_bin_counts = []
        all_bin_probs = []

        # Process each feature group
        for feature_group in self.feature_pairs:
            feature_group = [int(x) for x in feature_group]
            DEBUG.log(f" Processing feature group: {feature_group}")

            # Extract group data
            group_data = dataset[:, feature_group].contiguous()
            n_dims = len(feature_group)

            # Get appropriate bin sizes for this group
            group_bin_sizes = bin_sizes[:n_dims] if len(bin_sizes) > 1 else [bin_sizes[0]] * n_dims

            # Compute bin edges for all dimensions
            bin_edges = []
            for dim in range(n_dims):
                dim_data = group_data[:, dim].contiguous()
                dim_min, dim_max = dim_data.min(), dim_data.max()
                padding = (dim_max - dim_min) * 0.01

                edges = torch.linspace(
                    dim_min - padding,
                    dim_max + padding,
                    group_bin_sizes[dim] + 1,  # Use configured bin size for this dimension
                    device=self.device
                ).contiguous()
                bin_edges.append(edges)
                DEBUG.log(f" Dimension {dim} edges range: {edges[0].item():.3f} to {edges[-1].item():.3f}")
            pair_pbar.update(1)
            # Initialize bin counts with appropriate shape for variable bin sizes
            bin_shape = [n_classes] + [size for size in group_bin_sizes]
            bin_counts = torch.zeros(bin_shape, device=self.device, dtype=torch.float32)

            # Process each class
            for class_idx, class_label in enumerate(unique_classes):
                class_mask = labels == class_label
                if class_mask.any():
                    class_data = group_data[class_mask].contiguous()

                    if n_dims == 2:  # Optimized path for pairs
                        # Compute bin indices for both dimensions
                        bin_indices = torch.stack([
                            torch.bucketize(
                                class_data[:, dim].contiguous(),
                                bin_edges[dim].contiguous()
                            ).sub_(1).clamp_(0, group_bin_sizes[dim] - 1)
                            for dim in range(2)
                        ])

                        # Use scatter_add_ for efficient counting
                        counts = torch.zeros(group_bin_sizes[0] * group_bin_sizes[1], device=self.device)
                        flat_indices = bin_indices[0] * group_bin_sizes[1] + bin_indices[1]
                        counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32))
                        bin_counts[class_idx] = counts.reshape(group_bin_sizes[0], group_bin_sizes[1])

            # Apply Laplace smoothing and compute probabilities
            smoothed_counts = bin_counts + 1.0
            bin_probs = smoothed_counts / smoothed_counts.sum(dim=tuple(range(1, n_dims + 1)), keepdim=True)

            # Store results
            all_bin_edges.append(bin_edges)
            all_bin_counts.append(smoothed_counts)
            all_bin_probs.append(bin_probs)

            DEBUG.log(f" Bin counts shape: {smoothed_counts.shape}")
            DEBUG.log(f" Bin probs shape: {bin_probs.shape}")
        pair_pbar.close()
        return {
            'bin_edges': all_bin_edges,
            'bin_counts': all_bin_counts,
            'bin_probs': all_bin_probs,
            'feature_pairs': self.feature_pairs,
            'classes': unique_classes.to(self.device)
        }

 #----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def _compute_pairwise_likelihood_parallel_std(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """Optimized Gaussian likelihood computation - Gaussian specific method"""
        dataset = dataset.to(self.device)
        labels = labels.to(self.device)

        # Use existing feature pair generation (shared method)
        max_combinations = self.config.get('likelihood_config', {}).get('max_combinations', None)
        self.feature_pairs = self._generate_feature_combinations(
            feature_dims,
            2,
            max_combinations
        )

        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)
        n_pairs = len(self.feature_pairs)

        # Initialize parameters
        means = torch.zeros((n_classes, n_pairs, 2), device=self.device)
        covs = torch.zeros((n_classes, n_pairs, 2, 2), device=self.device)

        # Process each class
        for class_idx, class_id in enumerate(unique_classes):
            class_mask = (labels == class_id)
            class_data = dataset[class_mask]

            # Process each feature pair
            for pair_idx, pair in enumerate(self.feature_pairs):
                pair_data = class_data[:, pair]

                # Compute mean efficiently
                means[class_idx, pair_idx] = torch.mean(pair_data, dim=0)

                # Compute covariance with stability check
                if len(pair_data) > 1:
                    centered_data = pair_data - means[class_idx, pair_idx].unsqueeze(0)
                    cov = torch.matmul(centered_data.T, centered_data) / (len(pair_data) - 1)

                    # Add stability term
                    min_eig = torch.linalg.eigvals(cov).real.min()
                    if min_eig < 1e-6:
                        reg_term = (1e-6 - min_eig) + 1e-6
                        cov += torch.eye(2, device=self.device) * reg_term
                else:
                    # Single sample case - use identity matrix
                    cov = torch.eye(2, device=self.device)

                covs[class_idx, pair_idx] = cov

        return {
            'means': means,
            'covs': covs,
            'classes': unique_classes,
            'feature_pairs': self.feature_pairs
        }

    def _compute_batch_posterior_std(self, features: torch.Tensor, epsilon: float = 1e-10):
        """Gaussian posterior computation focusing on relative class probabilities"""
        features = features.to(self.device)
        batch_size = len(features)
        n_classes = len(self.likelihood_params['classes'])

        # Initialize log likelihoods
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        # Process each feature pair
        for pair_idx, pair in enumerate(self.feature_pairs):
            pair_data = features[:, pair]

            # Get weights for this pair (same as histogram mode)
            pair_weights = [
                self.weight_updater.get_gaussian_weights(class_idx, pair_idx)
                for class_idx in range(n_classes)
            ]

            # Compute class contributions for this pair
            for class_idx in range(n_classes):
                mean = self.likelihood_params['means'][class_idx, pair_idx]
                cov = self.likelihood_params['covs'][class_idx, pair_idx]
                weight = pair_weights[class_idx]

                # Center the data
                centered = pair_data - mean.unsqueeze(0)

                # Compute class likelihood
                try:
                    # Add minimal regularization
                    reg_cov = cov + torch.eye(2, device=self.device) * 1e-6
                    prec = torch.inverse(reg_cov)

                    # Quadratic term
                    quad = torch.sum(
                        torch.matmul(centered.unsqueeze(1), prec).squeeze(1) * centered,
                        dim=1
                    )

                    # Log likelihood (excluding constant terms that are same for all classes)
                    class_ll = -0.5 * quad + torch.log(weight + epsilon)

                except RuntimeError:
                    # Handle numerical issues by setting very low likelihood
                    class_ll = torch.full_like(quad, -1e10)

                log_likelihoods[:, class_idx] += class_ll

        # Convert to probabilities using softmax
        max_log_ll = torch.max(log_likelihoods, dim=1, keepdim=True)[0]
        exp_ll = torch.exp(log_likelihoods - max_log_ll)
        posteriors = exp_ll / (torch.sum(exp_ll, dim=1, keepdim=True) + epsilon)

        return posteriors, None

    def _initialize_bin_weights(self):
        """Initialize weights for either histogram bins or Gaussian components"""
        n_classes = len(self.label_encoder.classes_)
        if self.model_type == "Histogram":
            self.weight_updater = BinWeightUpdater(
                n_classes=n_classes,
                feature_pairs=self.feature_pairs,
                n_bins_per_dim=self.n_bins_per_dim
            )
        elif self.model_type == "Gaussian":
            # Use same weight structure but for Gaussian components
            self.weight_updater = BinWeightUpdater(
                n_classes=n_classes,
                feature_pairs=self.feature_pairs,
                n_bins_per_dim=self.n_bins_per_dim  # Number of Gaussian components
            )

    def _update_priors_parallel(self, failed_cases: List[Tuple], batch_size: int = 32):
        """Vectorized weight updates with proper error handling"""
        n_failed = len(failed_cases)
        if n_failed == 0:
            self.consecutive_successes += 1
            return

        self.consecutive_successes = 0
        self.learning_rate = max(self.learning_rate / 2, 1e-6)

        # Stack all features and convert classes at once
        features = torch.stack([case[0] for case in failed_cases]).to(self.device)
        true_classes = torch.tensor([int(case[1]) for case in failed_cases], device=self.device)

        # Compute posteriors for all cases at once
        if self.model_type == "Histogram":
            posteriors, bin_indices = self._compute_batch_posterior(features)
        else:  # Gaussian model
            posteriors, _ = self._compute_batch_posterior_std(features)
            return  # Gaussian model doesn't need bin-based updates

        pred_classes = torch.argmax(posteriors, dim=1)

        # Compute adjustments for all cases at once
        true_posteriors = posteriors[torch.arange(n_failed), true_classes]
        pred_posteriors = posteriors[torch.arange(n_failed), pred_classes]
        adjustments = self.learning_rate * (1.0 - (true_posteriors / pred_posteriors))

        # Update weights for each feature group
        if bin_indices is not None:  # Only proceed if we have bin indices (Histogram model)
            for group_idx in bin_indices:
                bin_i, bin_j = bin_indices[group_idx]

                # Group updates by class for vectorization
                for class_id in range(self.weight_updater.n_classes):
                    class_mask = true_classes == class_id
                    if not class_mask.any():
                        continue

                    # Get relevant indices and adjustments for this class
                    class_bin_i = bin_i[class_mask]
                    class_bin_j = bin_j[class_mask]
                    class_adjustments = adjustments[class_mask]

                    # Update weights for this class
                    weights = self.weight_updater.histogram_weights[class_id][group_idx]
                    for idx in range(len(class_adjustments)):
                        i, j = class_bin_i[idx], class_bin_j[idx]
                        weights[i, j] += class_adjustments[idx]
#------------------------------------------Boost weights------------------------------------------
    def _update_weights_with_boosting(self, failed_cases: List[Tuple], batch_size: int = 32):
        """
        Update weights using difference boosting for failed cases.
        Enhances the probability of misclassified examples by focusing on their error margins.
        """
        n_failed = len(failed_cases)
        if n_failed == 0:
            return

        # Pre-allocate tensors on device
        features = torch.stack([case[0] for case in failed_cases]).to(self.device)
        true_classes = torch.tensor([case[1] for case in failed_cases], device=self.device)

        # Compute posteriors for failed cases
        posteriors = self._compute_batch_posterior(features)

        # Get probability differences between true class and highest wrong class
        batch_range = torch.arange(n_failed, device=self.device)
        true_probs = posteriors[batch_range, true_classes]

        # Create mask for non-true classes
        mask = torch.ones_like(posteriors, dtype=torch.bool)
        mask[batch_range, true_classes] = False
        wrong_probs = posteriors.masked_fill(~mask, float('-inf')).max(dim=1)[0]

        # Compute boosting factors based on probability differences
        prob_differences = wrong_probs - true_probs
        boost_factors = torch.exp(prob_differences / self.learning_rate)

        # Update weights for each failed case
        for i, class_id in enumerate(true_classes):
            # Apply boosting to feature weights for the true class
            self.current_W[class_id] *= boost_factors[i]

        # Normalize weights to prevent numerical instability
        self.current_W /= self.current_W.max()
        self.current_W.clamp_(min=1e-10)

    def _compute_custom_bin_edges(self, data: torch.Tensor, bin_sizes: List[int]) -> List[torch.Tensor]:
        """
        Compute bin edges based on custom bin sizes.
        Supports both uniform and non-uniform binning.

        Args:
            data: Input tensor of shape [n_samples, n_features]
            bin_sizes: List of integers specifying bin sizes for each dimension

        Returns:
            List of tensors containing bin edges for each dimension
        """
        n_dims = data.shape[1]
        bin_edges = []

        # If single bin size provided, use it for all dimensions
        if len(bin_sizes) == 1:
            bin_sizes = bin_sizes * n_dims

        # Ensure we have enough bin sizes
        if len(bin_sizes) < n_dims:
            raise ValueError(f"Not enough bin sizes provided. Need {n_dims}, got {len(bin_sizes)}")

        for dim in range(n_dims):
            dim_data = data[:, dim]
            dim_min, dim_max = dim_data.min(), dim_data.max()
            padding = (dim_max - dim_min) * 0.01

            # Create edges based on specified bin size
            if bin_sizes[dim] <= 1:
                raise ValueError(f"Bin size must be > 1, got {bin_sizes[dim]}")

            edges = torch.linspace(
                dim_min - padding,
                dim_max + padding,
                bin_sizes[dim] + 1,
                device=self.device
            )
            bin_edges.append(edges)

        return bin_edges

    def _compute_pairwise_likelihood_parallel_exp(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """
        Modified likelihood computation to support custom bin sizes
        """
        dataset = torch.as_tensor(dataset, device=self.device).contiguous()
        labels = torch.as_tensor(labels, device=self.device).contiguous()

        unique_classes, class_counts = torch.unique(labels, return_counts=True)
        n_classes = len(unique_classes)

        # Get bin sizes from config
        bin_sizes = self.config.get('likelihood_config', {}).get('bin_sizes', [20])

        # Generate feature combinations
        self.feature_pairs = self._generate_feature_combinations(
            feature_dims,
            self.config.get('likelihood_config', {}).get('feature_group_size', 2),
            self.config.get('likelihood_config', {}).get('max_combinations', None)
        )

        # Pre-allocate storage arrays
        all_bin_edges = []
        all_bin_counts = []
        all_bin_probs = []

        # Process each feature group
        for feature_group in self.feature_pairs:
            feature_group = [int(x) for x in feature_group]
            group_data = dataset[:, feature_group].contiguous()

            # Use custom binning
            bin_edges = self._compute_custom_bin_edges(group_data, bin_sizes)

            # Initialize bin counts
            bin_shape = [n_classes] + [len(edges) - 1 for edges in bin_edges]
            bin_counts = torch.zeros(bin_shape, device=self.device, dtype=torch.float32)

            # Process each class
            for class_idx, class_label in enumerate(unique_classes):
                class_mask = labels == class_label
                if class_mask.any():
                    class_data = group_data[class_mask]

                    # Compute bin indices
                    bin_indices = torch.stack([
                        torch.bucketize(class_data[:, dim], bin_edges[dim]) - 1
                        for dim in range(len(feature_group))
                    ]).clamp_(0, bin_shape[1] - 1)

                    # Update bin counts
                    for sample_idx in range(len(class_data)):
                        idx = tuple([class_idx] + [bin_indices[d, sample_idx] for d in range(len(feature_group))])
                        bin_counts[idx] += 1

            # Apply Laplace smoothing and compute probabilities
            smoothed_counts = bin_counts + 1.0
            bin_probs = smoothed_counts / smoothed_counts.sum(dim=tuple(range(1, len(feature_group) + 1)), keepdim=True)

            # Store results
            all_bin_edges.append(bin_edges)
            all_bin_counts.append(smoothed_counts)
            all_bin_probs.append(bin_probs)

        return {
            'bin_edges': all_bin_edges,
            'bin_counts': all_bin_counts,
            'bin_probs': all_bin_probs,
            'feature_pairs': self.feature_pairs,
            'classes': unique_classes
        }

#---------------------------------------------------------Save Last data -------------------------
    def save_last_split(self, train_indices: list, test_indices: list):
        """Save the last training/testing split to CSV files"""
        dataset_name = self.dataset_name

        # Get full dataset
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Save training data
        train_data = pd.concat([X.iloc[train_indices], y.iloc[train_indices]], axis=1)
        train_data.to_csv(f'{dataset_name}_Last_training.csv', index=False)

        # Save testing data
        test_data = pd.concat([X.iloc[test_indices], y.iloc[test_indices]], axis=1)
        test_data.to_csv(f'{dataset_name}_Last_testing.csv', index=False)
        print(f"Last testing data is saved to {dataset_name}_Last_testing.csv")
        print(f"Last training data is saved to {dataset_name}_Last_training.csv")

    def load_last_known_split(self):
        """Load the last known good training/testing split with proper column alignment"""
        dataset_name = self.dataset_name
        train_file = f'{dataset_name}_Last_training.csv'
        test_file = f'{dataset_name}_Last_testing.csv'

        if os.path.exists(train_file) and os.path.exists(test_file):
            try:
                # Load the saved splits
                train_data = pd.read_csv(train_file)
                test_data = pd.read_csv(test_file)

                # Get current feature columns excluding target
                X = self.data.drop(columns=[self.target_column])
                current_columns = X.columns

                # Ensure train_data has same columns as current data
                train_features = train_data.drop(columns=[self.target_column])
                train_features = train_features[current_columns]

                # Initialize indices lists
                train_indices = []
                test_indices = []

                # Match rows using selected columns
                for idx, row in X.iterrows():
                    # Align the row with train_features columns
                    row = row[current_columns]

                    # Compare with train data first
                    match_mask = (train_features == row).all(axis=1)
                    if match_mask.any():
                        train_indices.append(idx)
                    else:
                        test_indices.append(idx)

                if train_indices or test_indices:
                    print(f"Loaded previous split - Training: {len(train_indices)}, Testing: {len(test_indices)}")
                    return train_indices, test_indices
                else:
                    print("No valid indices found in previous split")
                    return None, None

            except Exception as e:
                print(f"Error loading previous split: {str(e)}")
                return None, None

        return None, None

#---------------------------------------------------------------------------------------------------------

    def predict(self, X: torch.Tensor, batch_size: int = 32):
        """Make predictions in batches using the best model weights"""
        # Store current weights temporarily
        print("\nMaking predictions...")
        temp_W = self.current_W
        n_batches = (len(X) + batch_size - 1) // batch_size
        pred_pbar = tqdm(total=n_batches, desc="Prediction batches")
        # Use best weights for prediction
        self.current_W = self.best_W.clone() if self.best_W is not None else self.current_W

        X = X.to(self.device)
        predictions = []

        try:
            for i in range(0, len(X), batch_size):
                batch_X = X[i:min(i + batch_size, len(X))]
                if self.model_type=="Histogram":
                    # Get posteriors only, ignore bin indices
                    posteriors, _ = self._compute_batch_posterior(batch_X)
                elif self.model_type=="Gaussian":
                    # Get posteriors only, ignore component responsibilities
                    posteriors, _ = self._compute_batch_posterior_std(batch_X)
                else:
                    print(f"{self.model_type} is invalid. Please edit configuration file")

                batch_predictions = torch.argmax(posteriors, dim=1)
                predictions.append(batch_predictions)
                pred_pbar.update(1)

        finally:
            # Restore current weights
            self.current_W = temp_W

        pred_pbar.close()
        return torch.cat(predictions).cpu()


    def _save_best_weights(self):
        """Save the best weights to file"""
        if self.best_W is not None:
            # Convert tensor to numpy for saving
            weights_array = self.best_W.cpu().numpy()

            weights_dict = {
                'version': 2,  # Add version to track format
                'weights': weights_array.tolist(),
                'shape': list(weights_array.shape)
            }

            with open(self._get_weights_filename(), 'w') as f:
                json.dump(weights_dict, f)

    def _load_best_weights(self):
        """Load the best weights from file if they exist"""
        weights_file = self._get_weights_filename()
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                weights_dict = json.load(f)

            try:
                if 'version' in weights_dict and weights_dict['version'] == 2:
                    # New format (tensor-based)
                    weights_array = np.array(weights_dict['weights'])
                    self.best_W = torch.tensor(
                        weights_array,
                        dtype=torch.float32,
                        device=self.device
                    )
                else:
                    # Old format (dictionary-based)
                    # Convert old format to tensor format
                    class_ids = sorted([int(k) for k in weights_dict.keys()])
                    max_class_id = max(class_ids)

                    # Get number of feature pairs from first class
                    first_class = weights_dict[str(class_ids[0])]
                    n_pairs = len(first_class)

                    # Initialize tensor
                    weights_array = np.zeros((max_class_id + 1, n_pairs))

                    # Fill in weights from old format
                    for class_id in class_ids:
                        class_weights = weights_dict[str(class_id)]
                        for pair_idx, (pair, weight) in enumerate(class_weights.items()):
                            weights_array[class_id, pair_idx] = float(weight)

                    self.best_W = torch.tensor(
                        weights_array,
                        dtype=torch.float32,
                        device=self.device
                    )

                print(f"Loaded best weights from {weights_file}")
            except Exception as e:
                print(f"Warning: Could not load weights from {weights_file}: {str(e)}")
                self.best_W = None

    def _init_keyboard_listener(self):
        """Initialize keyboard listener with shared display connection"""
        if not hasattr(self, '_display'):
            try:
                import Xlib.display
                self._display = Xlib.display.Display()
            except Exception as e:
                print(f"Warning: Could not initialize X display: {e}")
                return None

        try:
            from pynput import keyboard
            return keyboard.Listener(
                on_press=self._on_key_press,
                _display=self._display  # Pass shared display connection
            )
        except Exception as e:
            print(f"Warning: Could not create keyboard listener: {e}")
            return None

    def _cleanup_keyboard(self):
        """Clean up keyboard resources"""
        if hasattr(self, '_display'):
            try:
                self._display.close()
                del self._display
            except:
                pass

    def print_colored_confusion_matrix(self, y_true, y_pred, class_labels=None):
        """Print a color-coded confusion matrix with class-wise accuracy."""

        # Convert numeric labels to strings for consistent handling
        y_true = np.array([str(x) for x in y_true])
        y_pred = np.array([str(x) for x in y_pred])

        # Get unique classes from both true and predicted labels
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)

        # Use provided class labels or get from label encoder
        if class_labels is None:
            class_labels = np.array([str(x) for x in self.label_encoder.classes_])
        else:
            class_labels = np.array([str(x) for x in class_labels])

        # Ensure all classes are represented in confusion matrix
        all_classes = np.unique(np.concatenate([unique_true, unique_pred, class_labels]))
        n_classes = len(all_classes)

        # Create class index mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

        # Initialize confusion matrix with zeros
        cm = np.zeros((n_classes, n_classes), dtype=int)

        # Fill confusion matrix
        for t, p in zip(y_true, y_pred):
            if t in class_to_idx and p in class_to_idx:
                cm[class_to_idx[t], class_to_idx[p]] += 1

        # Calculate class-wise accuracy
        class_accuracy = {}
        for i in range(n_classes):
            if cm[i].sum() > 0:  # Avoid division by zero
                class_accuracy[i] = cm[i, i] / cm[i].sum()
            else:
                class_accuracy[i] = 0.0

        # Print header
        print(f"\n{Colors.BOLD}Confusion Matrix and Class-wise Accuracy:{Colors.ENDC}")

        # Print class labels header
        print(f"{'Actual/Predicted':<15}", end='')
        for label in all_classes:
            print(f"{label:<8}", end='')
        print("Accuracy")
        print("-" * (15 + 8 * n_classes + 10))

        # Print matrix with colors
        for i in range(n_classes):
            # Print actual class label
            print(f"{Colors.BOLD}{all_classes[i]:<15}{Colors.ENDC}", end='')

            # Print confusion matrix row
            for j in range(n_classes):
                if i == j:
                    color = Colors.GREEN
                else:
                    color = Colors.RED
                print(f"{color}{cm[i, j]:<8}{Colors.ENDC}", end='')

            # Print class accuracy
            acc = class_accuracy[i]
            color = Colors.GREEN if acc >= 0.9 else Colors.YELLOW if acc >= 0.7 else Colors.RED
            print(f"{color}{acc:>7.2%}{Colors.ENDC}")

        # Print overall accuracy
        total_correct = np.diag(cm).sum()
        total_samples = cm.sum()
        if total_samples > 0:
            overall_acc = total_correct / total_samples
            print("-" * (15 + 8 * n_classes + 10))
            color = Colors.GREEN if overall_acc >= 0.9 else Colors.YELLOW if overall_acc >= 0.7 else Colors.RED
            print(f"{Colors.BOLD}Overall Accuracy: {color}{overall_acc:.2%}{Colors.ENDC}")

        # Save confusion matrix plot
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=all_classes,
                yticklabels=all_classes
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            if hasattr(self, 'dataset_name'):
                plt.savefig(f'confusion_matrix_{self.dataset_name}.png')
            else:
                plt.savefig('confusion_matrix.png')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save confusion matrix plot: {str(e)}")

    def train(self, X_train, y_train, X_test, y_test, batch_size=32):
        """Complete training with progress tracking and confusion matrix"""
        print("\nStarting training...")
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Initialize tracking
        error_rates = []
        best_train_accuracy = 0.0
        best_test_accuracy = 0.0
        patience_counter = 0
        plateau_counter = 0
        min_improvement = 0.001
        patience = 5 if self.in_adaptive_fit else 100
        max_plateau = 5
        prev_accuracy = 0.0

        # Main training loop with dynamic progress bar
        with tqdm(total=self.max_epochs, desc="Training epochs") as epoch_pbar:
            for epoch in range(self.max_epochs):
                # Train on all batches
                failed_cases = []
                n_errors = 0

                with tqdm(total=n_batches, desc=f"Training batches", leave=False) as batch_pbar:
                    for i in range(0, n_samples, batch_size):
                        batch_end = min(i + batch_size, n_samples)
                        batch_X = X_train[i:batch_end]
                        batch_y = y_train[i:batch_end]

                        posteriors = self._compute_batch_posterior(batch_X)[0]
                        predictions = torch.argmax(posteriors, dim=1)
                        errors = (predictions != batch_y)
                        n_errors += errors.sum().item()

                        if errors.any():
                            fail_idx = torch.where(errors)[0]
                            for idx in fail_idx:
                                failed_cases.append((
                                    batch_X[idx],
                                    batch_y[idx].item(),
                                    posteriors[idx].cpu().numpy()
                                ))
                        batch_pbar.update(1)

                if failed_cases:
                    self._update_priors_parallel(failed_cases, batch_size)

                # Calculate training metrics
                train_error_rate = n_errors / n_samples
                train_accuracy = 1 - train_error_rate
                error_rates.append(train_error_rate)

                # Calculate test metrics if test data provided
                test_accuracy = 0
                if X_test is not None and y_test is not None:
                    test_predictions = self.predict(X_test, batch_size=batch_size)
                    test_accuracy = (test_predictions == y_test.cpu()).float().mean().item()
                    if test_accuracy > best_test_accuracy:
                        best_test_accuracy = test_accuracy
                        # Print confusion matrix for best test performance
                        print("\nTest Set Performance:")
                        y_test_labels = self.label_encoder.inverse_transform(y_test.cpu().numpy())
                        test_pred_labels = self.label_encoder.inverse_transform(test_predictions.cpu().numpy())
                        self.print_colored_confusion_matrix(y_test_labels, test_pred_labels)

                # Update progress bar with both accuracies
                epoch_pbar.set_postfix({
                    'train_acc': f"{train_accuracy:.4f}",
                    'best_train': f"{best_train_accuracy:.4f}",
                    'test_acc': f"{test_accuracy:.4f}",
                    'best_test': f"{best_test_accuracy:.4f}"
                })
                epoch_pbar.update(1)

                # Check improvement and update tracking
                accuracy_improvement = train_accuracy - prev_accuracy
                if accuracy_improvement <= min_improvement:
                    plateau_counter += 1
                else:
                    plateau_counter = 0

                if train_accuracy > best_train_accuracy + min_improvement:
                    best_train_accuracy = train_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Save best model
                if train_error_rate <= self.best_error:
                    self.best_error = train_error_rate
                    self.best_W = self.current_W.clone()
                    self._save_best_weights()

                # Early stopping checks
                if train_accuracy == 1.0:
                    print("\nReached 100% training accuracy")
                    break

                if patience_counter >= patience:
                    print(f"\nNo improvement for {patience} epochs")
                    break

                if plateau_counter >= max_plateau:
                    print(f"\nAccuracy plateaued for {max_plateau} epochs")
                    break

                prev_accuracy = train_accuracy

        self._save_model_components()
        return self.current_W.cpu(), error_rates


    def plot_training_metrics(self, train_loss, test_loss, train_acc, test_acc, save_path=None):
        """Plot training and testing metrics over epochs"""
        plt.figure(figsize=(12, 8))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(train_loss, label='Train Loss', marker='o')
        plt.plot(test_loss, label='Test Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(train_acc, label='Train Accuracy', marker='o')
        plt.plot(test_acc, label='Test Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            # Also save metrics to CSV
            metrics_df = pd.DataFrame({
                'epoch': range(1, len(train_loss) + 1),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            })
            metrics_df.to_csv(save_path.replace('.png', '_metrics.csv'), index=False)

        plt.close()

    def verify_classifications(self, X: pd.DataFrame, true_labels: pd.Series, predictions: torch.Tensor) -> None:
        """
        Verify classification accuracy computation with proper index handling
        """
        # Convert predictions to numpy array if it's a tensor
        pred_labels = self.label_encoder.inverse_transform(predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions)

        # Ensure true_labels is a numpy array
        true_labels_array = true_labels.to_numpy() if isinstance(true_labels, pd.Series) else true_labels

        # Calculate accuracy metrics
        n_total = len(true_labels_array)
        correct_mask = (true_labels_array == pred_labels)
        n_correct = correct_mask.sum()

        # Print basic metrics with colors
        print(f"\n{Colors.BOLD}{Colors.BLUE}Detailed Classification Analysis:{Colors.ENDC}")
        print(f"{Colors.BOLD}Total samples:{Colors.ENDC} {Colors.YELLOW}{n_total:,}{Colors.ENDC}")

        # Color code for correctly classified
        correct_color = Colors.GREEN if (n_correct/n_total) >= 0.9 else \
                       Colors.YELLOW if (n_correct/n_total) >= 0.7 else \
                       Colors.RED
        print(f"{Colors.BOLD}Correctly classified:{Colors.ENDC} {correct_color}{n_correct:,}{Colors.ENDC}")

        # Color code for incorrectly classified
        incorrect = n_total - n_correct
        incorrect_color = Colors.GREEN if (incorrect/n_total) <= 0.1 else \
                         Colors.YELLOW if (incorrect/n_total) <= 0.3 else \
                         Colors.RED
        print(f"{Colors.BOLD}Incorrectly classified:{Colors.ENDC} {incorrect_color}{incorrect:,}{Colors.ENDC}")

        # Color code for raw accuracy
        accuracy = n_correct/n_total
        accuracy_color = Colors.GREEN if accuracy >= 0.9 else \
                        Colors.YELLOW if accuracy >= 0.7 else \
                        Colors.RED
        print(f"{Colors.BOLD}Raw accuracy:{Colors.ENDC} {accuracy_color}{accuracy:.4%}{Colors.ENDC}\n")

        # Print confusion matrix with colors
        self.print_colored_confusion_matrix(true_labels_array, pred_labels)

        # Save detailed analysis to file
        analysis_file = f"classification_analysis_{self.dataset_name}.txt"
        with open(analysis_file, 'w') as f:
            f.write("Per-class breakdown:\n")
            for cls in np.unique(true_labels_array):
                cls_mask = (true_labels_array == cls)
                n_cls = cls_mask.sum()
                n_correct_cls = (correct_mask & cls_mask).sum()

                f.write(f"\nClass {cls}:\n")
                f.write(f"Total samples: {n_cls}\n")
                f.write(f"Correctly classified: {n_correct_cls}\n")
                f.write(f"Class accuracy: {n_correct_cls/n_cls:.4f}\n")

                if n_cls - n_correct_cls > 0:
                    # Find misclassified examples
                    misclassified_mask = (~correct_mask & cls_mask)
                    mis_predictions = pred_labels[misclassified_mask]
                    unique_mispred, counts = np.unique(mis_predictions, return_counts=True)

                    f.write("\nMisclassified as:\n")
                    for pred_cls, count in zip(unique_mispred, counts):
                        f.write(f"{pred_cls}: {count}\n")

                    # Save examples of misclassified instances
                    f.write("\nSample misclassified instances:\n")
                    misclassified_indices = np.where(misclassified_mask)[0]
                    for idx in misclassified_indices[:5]:  # Show up to 5 examples
                        f.write(f"\nInstance {idx}:\n")
                        f.write(f"True class: {true_labels_array[idx]}\n")
                        f.write(f"Predicted class: {pred_labels[idx]}\n")
                        f.write("Feature values:\n")
                        for col, val in X.iloc[idx].items():
                            f.write(f"{col}: {val}\n")

        print(f"\nDetailed analysis saved to {analysis_file}")

#------------------------------------------------------------End of PP code ---------------------------------------------------
    def _compute_pairwise_likelihood(self, dataset, labels, feature_dims):
        """Compute pairwise likelihood PDFs"""
        unique_classes = torch.unique(labels)
        feature_pairs = list(combinations(range(feature_dims), 2))
        likelihood_pdfs = {}

        for class_id in unique_classes:
            class_mask = (labels == class_id)
            class_data = dataset[class_mask]
            likelihood_pdfs[class_id.item()] = {}

            for feat_i, feat_j in feature_pairs:
                pair_data = torch.stack([
                    class_data[:, feat_i],
                    class_data[:, feat_j]
                ], dim=1)

                mean = torch.mean(pair_data, dim=0)
                centered_data = pair_data - mean
                cov = torch.mm(centered_data.T, centered_data) / (len(pair_data) - 1)
                cov = cov + torch.eye(2) * 1e-6

                likelihood_pdfs[class_id.item()][(feat_i, feat_j)] = {
                    'mean': mean,
                    'cov': cov
                }

        return likelihood_pdfs


    def _get_weights_filename(self):
        """Get the filename for saving/loading weights"""
        return os.path.join('Model', f'Best_{self.model_type}_{self.dataset_name}_weights.json')

    def _get_encoders_filename(self):
        """Get the filename for saving/loading categorical encoders"""
        return os.path.join('Model', f'Best_{self.model_type}_{self.dataset_name}_encoders.json')




    def _remove_high_cardinality_columns(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Remove high cardinality columns and round features with detailed debugging"""
        DEBUG.log(f" Starting high cardinality removal with threshold {threshold}")
        DEBUG.log(f" Initial columns: {df.columns.tolist()}")

        # Round all features first if cardinality_tolerance is not -1
        if self.cardinality_tolerance != -1:
            DEBUG.log(f" Rounding features with tolerance {self.cardinality_tolerance}")
            df_rounded = self._round_features(df)
        else:
            df_rounded = df.copy()

        df_filtered = df_rounded.copy()
        columns_to_drop = []

        for column in df.columns:
            if column == self.target_column:
                continue

            # Use rounded data for cardinality check
            unique_count = len(df_rounded[column].unique())
            unique_ratio = unique_count / len(df)
            DEBUG.log(f" Column {column}: {unique_count} unique values, ratio {unique_ratio:.4f}")

            if unique_ratio > threshold:
                columns_to_drop.append(column)
                DEBUG.log(f" Marking {column} for removal (ratio {unique_ratio:.4f} > {threshold})")

        if columns_to_drop:
            DEBUG.log(f" Dropping columns: {columns_to_drop}")
            df_filtered = df_filtered.drop(columns=columns_to_drop)

        DEBUG.log(f" Columns after filtering: {df_filtered.columns.tolist()}")
        DEBUG.log(f" Remaining features: {len(df_filtered.columns)}")

        if len(df_filtered.columns) == 0:
            print("[WARNING] All features were removed! Reverting to original features with warnings.")
            return df.copy()

        return df_filtered

    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect categorical columns with improved debugging"""
        DEBUG.log(" Starting categorical column detection")
        categorical_columns = []

        for column in df.columns:
            if column != self.target_column:
                dtype = df[column].dtype
                unique_count = len(df[column].unique())
                DEBUG.log(f" Column {column}: dtype={dtype}, unique values={unique_count}")

                if dtype == 'object' or dtype.name == 'category':
                    DEBUG.log(f" Adding {column} as categorical (object/category type)")
                    categorical_columns.append(column)
                elif dtype in ['int64', 'float64']:
                    threshold = min(50, len(df) * 0.05)
                    if unique_count < threshold:
                        DEBUG.log(f" Adding {column} as categorical (numeric with few unique values: {unique_count} < {threshold})")
                        categorical_columns.append(column)
                    else:
                        DEBUG.log(f" Keeping {column} as numeric (unique values: {unique_count} >= {threshold})")

        DEBUG.log(f" Detected categorical columns: {categorical_columns}")
        return categorical_columns



    def _get_train_test_split(self, X_tensor, y_tensor):
        """Get or create consistent train-test split"""
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
        split_path = os.path.join(base_path, dataset_folder, 'train_test_split.pkl')

        if os.path.exists(split_path):
            with open(split_path, 'rb') as f:
                split_indices = pickle.load(f)
                train_idx, test_idx = split_indices['train'], split_indices['test']
                return (X_tensor[train_idx], X_tensor[test_idx],
                        y_tensor[train_idx], y_tensor[test_idx])

        # Create new split
        X_train, X_test, y_train, y_test = self._train_test_split_tensor(
            X_tensor, y_tensor, self.test_size, self.random_state)

        # Save split indices
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        split_indices = {
            'train': torch.where(X_tensor == X_train.unsqueeze(1))[0],
            'test': torch.where(X_tensor == X_test.unsqueeze(1))[0]
        }
        with open(split_path, 'wb') as f:
            pickle.dump(split_indices, f)

        return X_train, X_test, y_train, y_test

    def _train_test_split_tensor(self, X, y, test_size, random_state):
        """Split data consistently using fixed indices"""
        num_samples = len(X)

        # Generate fixed permutation
        if random_state == -1:
            # Use numpy's random permutation directly
            indices = torch.from_numpy(np.random.permutation(num_samples))
        else:
            rng = np.random.RandomState(random_state)
            indices = torch.from_numpy(rng.permutation(num_samples))

        split = int(num_samples * (1 - test_size))
        train_idx = indices[:split]
        test_idx = indices[split:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def _multivariate_normal_pdf(self, x, mean, cov):
        """Compute multivariate normal PDF"""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        dim = 2
        centered_x = x - mean.unsqueeze(0)
        inv_cov = torch.inverse(cov)
        det = torch.det(cov)
        quad_form = torch.sum(torch.mm(centered_x, inv_cov) * centered_x, dim=1)
        norm_const = 1.0 / (torch.sqrt((2 * torch.tensor(np.pi)) ** dim * det))
        return norm_const * torch.exp(-0.5 * quad_form)

    def _initialize_priors(self):
        """Initialize weights"""
        if self.best_W is not None:
            return self.best_W

        W = {}
        for class_id in self.likelihood_pdfs.keys():
            W[class_id] = {}
            for feature_pair in self.likelihood_pdfs[class_id].keys():
                W[class_id][feature_pair] = torch.tensor(0.1, dtype=torch.float32)
        return W

    def compute_posterior(self, feature_data, class_id=None, epsilon=1e-10):
        """Compute posterior probabilities"""
        classes = list(self.likelihood_pdfs.keys())
        n_classes = len(classes)
        feature_pairs = list(self.likelihood_pdfs[classes[0]].keys())
        log_likelihoods = torch.zeros(n_classes, dtype=torch.float32)

        for idx, c_id in enumerate(classes):
            class_log_likelihood = 0.0

            for feat_i, feat_j in feature_pairs:
                pair_data = torch.tensor([
                    feature_data[feat_i].item(),
                    feature_data[feat_j].item()
                ], dtype=torch.float32).reshape(1, 2)

                pdf_params = self.likelihood_pdfs[c_id][(feat_i, feat_j)]
                pair_likelihood = self._multivariate_normal_pdf(
                    pair_data,
                    pdf_params['mean'],
                    pdf_params['cov']
                ).squeeze()

                prior = self.current_W[c_id][(feat_i, feat_j)].item()
                likelihood_term = (pair_likelihood * prior + epsilon).item()
                class_log_likelihood += torch.log(torch.tensor(likelihood_term))

            log_likelihoods[idx] = class_log_likelihood

        max_log_likelihood = torch.max(log_likelihoods)
        likelihoods = torch.exp(log_likelihoods - max_log_likelihood)
        posteriors = likelihoods / (likelihoods.sum() + epsilon)

        return {c_id: posteriors[idx].item() for idx, c_id in enumerate(classes)}

#-------------------------------------------Reconstruction enhanced fit_predict support methods----------------------
    def _format_reconstruction_results(self, metrics: Dict) -> str:
        """Format reconstruction metrics into a human-readable string.
        Matches the metrics structure from _compute_reconstruction_metrics.

        Args:
            metrics: Dictionary containing reconstruction metrics

        Returns:
            str: Formatted string containing the reconstruction analysis
        """
        formatted = "\nReconstruction Analysis:\n" + "="*50 + "\n"

        # Overall metrics
        formatted += "\nOverall Metrics:\n" + "-"*20 + "\n"
        if 'overall' in metrics:
            if 'mse' in metrics['overall']:
                formatted += f"Mean Square Error (MSE): {metrics['overall']['mse']:.6f}\n"
            if 'mae' in metrics['overall']:
                formatted += f"Mean Absolute Error (MAE): {metrics['overall']['mae']:.6f}\n"
            if 'rmse' in metrics['overall']:
                formatted += f"Root Mean Square Error (RMSE): {metrics['overall']['rmse']:.6f}\n"
            if 'correlation' in metrics['overall']:
                formatted += f"Overall Correlation: {metrics['overall']['correlation']:.4f}\n"

        # Per-feature metrics with error handling
        if 'per_feature_metrics' in metrics:
            formatted += "\nPer-Feature Analysis:\n" + "-"*20 + "\n"
            for feat_metric in metrics['per_feature_metrics']:
                feat_idx = feat_metric.get('feature_idx', 'Unknown')
                formatted += f"\nFeature {feat_idx}:\n"
                if 'mse' in feat_metric:
                    formatted += f"  MSE: {feat_metric['mse']:.6f}\n"
                if 'correlation' in feat_metric:
                    formatted += f"  Correlation: {feat_metric['correlation']:.4f}\n"

        # Per-class metrics
        if 'per_class_metrics' in metrics:
            formatted += "\nClass-wise Analysis:\n" + "-"*20 + "\n"
            for class_label, metrics_dict in metrics['per_class_metrics'].items():
                formatted += f"\nClass {class_label}:\n"
                if 'mse' in metrics_dict:
                    formatted += f"  MSE: {metrics_dict['mse']:.6f}\n"
                if 'sample_count' in metrics_dict:
                    formatted += f"  Samples: {metrics_dict['sample_count']}\n"
                if 'accuracy' in metrics_dict:
                    formatted += f"  Accuracy: {metrics_dict['accuracy']:.4f}\n"

        # Reconstruction quality distribution
        if 'reconstruction_quality' in metrics:
            formatted += "\nReconstruction Quality:\n" + "-"*20 + "\n"
            quality = metrics['reconstruction_quality']
            if 'excellent' in quality:
                formatted += f"Excellent (error < 0.1): {quality['excellent']:.1%}\n"
            if 'good' in quality:
                formatted += f"Good (0.1 â‰¤ error < 0.3): {quality['good']:.1%}\n"
            if 'fair' in quality:
                formatted += f"Fair (0.3 â‰¤ error < 0.5): {quality['fair']:.1%}\n"
            if 'poor' in quality:
                formatted += f"Poor (error â‰¥ 0.5): {quality['poor']:.1%}\n"

        # Reconstruction statistics
        if 'reconstruction_stats' in metrics:
            formatted += "\nAdditional Statistics:\n" + "-"*20 + "\n"
            stats = metrics['reconstruction_stats']
            if 'total_samples' in stats:
                formatted += f"Total samples analyzed: {stats['total_samples']}\n"
            if 'avg_reconstruction_error' in stats:
                formatted += f"Average reconstruction error: {stats['avg_reconstruction_error']:.6f}\n"
            if 'error_std' in stats:
                formatted += f"Error standard deviation: {stats['error_std']:.6f}\n"

        return formatted

    def _compute_reconstruction_metrics(self, original_features: torch.Tensor,
                                    reconstructed_features: torch.Tensor,
                                    class_probs: torch.Tensor,
                                    true_labels: torch.Tensor) -> Dict:
        """Compute reconstruction metrics with serializable outputs"""
        metrics = {}
        try:
            # Move tensors to CPU and convert to numpy
            original = original_features.cpu().numpy()
            reconstructed = reconstructed_features.cpu().numpy()
            probs = class_probs.cpu().numpy()
            labels = true_labels.cpu().numpy()

            # Basic reconstruction error metrics
            mse = float(np.mean((original - reconstructed) ** 2))
            mae = float(np.mean(np.abs(original - reconstructed)))
            rmse = float(np.sqrt(mse))

            # Per-feature metrics
            feature_metrics = []
            for i in range(original.shape[1]):
                feat_mse = float(np.mean((original[:, i] - reconstructed[:, i]) ** 2))
                feat_mae = float(np.mean(np.abs(original[:, i] - reconstructed[:, i])))
                feat_corr = float(np.corrcoef(original[:, i], reconstructed[:, i])[0, 1])
                feature_metrics.append({
                    'feature_idx': int(i),
                    'mse': feat_mse,
                    'mae': feat_mae,
                    'correlation': feat_corr
                })

            # Per-class metrics
            class_metrics = {}
            unique_classes = np.unique(labels)
            for class_label in unique_classes:
                class_mask = (labels == class_label)
                class_orig = original[class_mask]
                class_recon = reconstructed[class_mask]
                class_probs = probs[class_mask]

                class_metrics[int(class_label)] = {
                    'mse': float(np.mean((class_orig - class_recon) ** 2)),
                    'mae': float(np.mean(np.abs(class_orig - class_recon))),
                    'sample_count': int(np.sum(class_mask)),
                    'avg_confidence': float(np.mean(class_probs[:, class_label]))
                }

            # Compute importance scores
            importance_scores = np.zeros(original.shape[1])
            for i in range(original.shape[1]):
                temp_recon = reconstructed.copy()
                temp_recon[:, i] = np.mean(reconstructed[:, i])
                importance_scores[i] = float(np.mean((original - temp_recon) ** 2) - mse)

            # Compile metrics
            metrics = {
                'overall': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                },
                'per_feature': feature_metrics,
                'per_class': class_metrics,
                'feature_importance': {
                    str(i): float(score) for i, score in enumerate(importance_scores)
                },
                'reconstruction_stats': {
                    'total_samples': int(len(original)),
                    'avg_reconstruction_error': float(mse),
                    'error_std': float(np.std((original - reconstructed) ** 2))
                }
            }

        except Exception as e:
            print(f"Error computing reconstruction metrics: {str(e)}")
            metrics['error'] = str(e)

        # Final conversion to ensure all values are serializable
        return self._convert_to_serializable(metrics)


    def save_reconstruction_analysis(self, metrics: Dict, save_path: str):
        """Save reconstruction analysis with proper type conversion"""
        try:
            base_path = os.path.splitext(save_path)[0]
            analysis_path = f"{base_path}_reconstruction_analysis.json"

            # Convert metrics to serializable format
            serializable_metrics = self._convert_to_serializable(metrics)

            # Save JSON with converted metrics
            with open(analysis_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)

            # Save formatted report
            report_path = f"{base_path}_reconstruction_report.txt"
            with open(report_path, 'w') as f:
                f.write(self._format_reconstruction_results(metrics))

            print(f"\nSaved reconstruction analysis to {analysis_path}")
            print(f"Saved reconstruction report to {report_path}")

        except Exception as e:
            print(f"Error saving reconstruction analysis: {str(e)}")
            traceback.print_exc()

    def _convert_to_serializable(self, obj):
        """Convert numpy/torch types to Python native types for JSON serialization"""
        import numpy as np
        import torch

        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        return obj

    def update_results_with_reconstruction(self, results: Dict,
                                         original_features: torch.Tensor,
                                         reconstructed_features: torch.Tensor,
                                         class_probs: torch.Tensor,
                                         true_labels: torch.Tensor,
                                         save_path: Optional[str] = None) -> Dict:
        """Update results dictionary with reconstruction metrics."""
        reconstruction_metrics = self._compute_reconstruction_metrics(
            original_features,
            reconstructed_features,
            class_probs,
            true_labels
        )

        results['reconstruction'] = reconstruction_metrics

        if save_path:
            self.save_reconstruction_analysis(reconstruction_metrics, save_path)

        # Print summary
        print(self._format_reconstruction_results(reconstruction_metrics))

        return results
#-------------------------------------------------------------------------------------------------------------------------------------------------
    def _get_config_param(self, param_name: str, default_value: Any) -> Any:
        """Enhanced configuration parameter retrieval with debug logging"""
        # # print(f"\nDEBUG: Getting config parameter: {param_name}")
        # print(f"DEBUG:  Default value: {default_value}")
        # print(f"DEBUG:  Config type: {type(self.config)}")

        # First check if we have a GlobalConfig object
        if hasattr(self.config, 'to_dict'):
           ## print("DEBUG: Using GlobalConfig object")
            value = getattr(self.config, param_name, default_value)
            ## print(f"DEBUG:  Found value: {value}")
            return value

        # Then check dictionary config
        elif isinstance(self.config, dict):
           ## print("DEBUG: Using dictionary config")
            # Check in training_params
            if 'training_params' in self.config:
               ## print("DEBUG: Found training_params")
                if param_name in self.config['training_params']:
                    value = self.config['training_params'][param_name]
                    # print(f"DEBUG:  Found in training_params: {value}")
                    return value
                else:
                     print(f"DEBUG:  {param_name} not found in training_params")

            # Check top level config
            if param_name in self.config:
                value = self.config[param_name]
                # print(f"DEBUG:  Found at top level: {value}")
                return value
            else:
                print(f"DEBUG:  {param_name} not found at top level")

        # print(f"DEBUG:  Using default value: {default_value}")
        return default_value

    def fit_predict(self, batch_size: int = 32, save_path: str = None):
        """Full training and prediction pipeline with separated training and testing phases"""
        try:
            self._last_metrics_printed = True
            print("\nStarting fit_predict with configuration check:")

            # Get inverse parameters
            invert_DBNN = self._get_config_param('invert_DBNN', False)
            reconstruction_weight = self._get_config_param('reconstruction_weight', 0.5)
            feedback_strength = self._get_config_param('feedback_strength', 0.3)
            inverse_learning_rate = self._get_config_param('inverse_learning_rate', 0.1)

            print(f"Inverse DBNN enabled: {invert_DBNN}")
            print(f"Reconstruction weight: {reconstruction_weight}")
            print(f"Feedback strength: {feedback_strength}")

            # Prepare data
            if self.in_adaptive_fit:
                if not hasattr(self, 'train_indices') or not hasattr(self, 'test_indices'):
                    raise ValueError("train_indices or test_indices not found")
                X_train = self.X_tensor[self.train_indices]
                X_test = self.X_tensor[self.test_indices]
                y_train = self.y_tensor[self.train_indices]
                y_test = self.y_tensor[self.test_indices]
            else:
                # Regular training path
                X = self.data.drop(columns=[self.target_column])
                y = self.data[self.target_column]

                if not hasattr(self.label_encoder, 'classes_'):
                    y_encoded = self.label_encoder.fit_transform(y)
                else:
                    y_encoded = self.label_encoder.transform(y)

                X_processed = self._preprocess_data(X, is_training=True)
                X_tensor = torch.FloatTensor(X_processed).to(self.device)
                y_tensor = torch.LongTensor(y_encoded).to(self.device)
                X_train, X_test, y_train, y_test = self._get_train_test_split(X_tensor, y_tensor)

            # Training Phase
            print("\nPhase 1: Forward model training...")
            n_samples = len(X_train)
            n_batches = (n_samples + batch_size - 1) // batch_size

            # Initialize tracking metrics
            error_rates = []
            best_train_accuracy = 0.0
            patience_counter = 0
            plateau_counter = 0
            min_improvement = 0.001
            patience = 5 if self.in_adaptive_fit else 100
            max_plateau = 5
            prev_accuracy = 0.0

            # Main training loop
            with tqdm(total=self.max_epochs, desc="Training epochs") as epoch_pbar:
                for epoch in range(self.max_epochs):
                    failed_cases = []
                    n_errors = 0

                    # Process all batches
                    with tqdm(total=n_batches, desc=f"Training batches", leave=False) as batch_pbar:
                        for i in range(0, n_samples, batch_size):
                            batch_end = min(i + batch_size, n_samples)
                            batch_X = X_train[i:batch_end]
                            batch_y = y_train[i:batch_end]

                            posteriors = self._compute_batch_posterior(batch_X)[0]
                            predictions = torch.argmax(posteriors, dim=1)
                            errors = (predictions != batch_y)
                            n_errors += errors.sum().item()

                            if errors.any():
                                fail_idx = torch.where(errors)[0]
                                for idx in fail_idx:
                                    failed_cases.append((
                                        batch_X[idx],
                                        batch_y[idx].item(),
                                        posteriors[idx].cpu().numpy()
                                    ))
                            batch_pbar.update(1)

                    # Update weights after all batches
                    if failed_cases:
                        self._update_priors_parallel(failed_cases, batch_size)

                    # Calculate training metrics
                    train_error_rate = n_errors / n_samples
                    train_accuracy = 1 - train_error_rate
                    error_rates.append(train_error_rate)

                    # Update progress bar
                    epoch_pbar.set_postfix({
                        'train_acc': f"{train_accuracy:.4f}",
                        'best_train': f"{best_train_accuracy:.4f}"
                    })
                    epoch_pbar.update(1)

                    # Check improvement
                    accuracy_improvement = train_accuracy - prev_accuracy
                    if accuracy_improvement <= min_improvement:
                        plateau_counter += 1
                    else:
                        plateau_counter = 0

                    if train_accuracy > best_train_accuracy + min_improvement:
                        best_train_accuracy = train_accuracy
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Save best model
                    if train_error_rate <= self.best_error:
                        self.best_error = train_error_rate
                        self.best_W = self.current_W.clone()
                        self._save_best_weights()

                    # Early stopping checks
                    if train_accuracy == 1.0:
                        print("\nReached 100% training accuracy")
                        break
                    if patience_counter >= patience:
                        print(f"\nNo improvement for {patience} epochs")
                        break
                    if plateau_counter >= max_plateau:
                        print(f"\nAccuracy plateaued for {max_plateau} epochs")
                        break

                    prev_accuracy = train_accuracy

            # Save model components after training
            self._save_categorical_encoders()
            self._save_model_components()

            # Testing Phase
            print("\nPhase 2: Testing and evaluation...")

            # Use best weights for prediction
            temp_W = self.current_W
            self.current_W = self.best_W.clone() if self.best_W is not None else self.current_W

            # Compute final training accuracy
            train_predictions = self.predict(X_train, batch_size=batch_size)
            final_train_accuracy = (train_predictions == y_train.cpu()).float().mean().item()
            print(f"\nFinal training accuracy: {final_train_accuracy:.4f}")

            # Initialize results dictionary
            results = {
                'error_rates': error_rates,
                'train_accuracy': final_train_accuracy
            }

            # Check if we should proceed with testing
            min_training_accuracy = self._get_config_param('minimum_training_accuracy', 0.95)
            if not self.in_adaptive_fit or final_train_accuracy >= min_training_accuracy:
                # Test predictions
                y_pred = self.predict(X_test, batch_size=batch_size)
                test_accuracy = (y_pred == y_test.cpu()).float().mean().item()
                print(f"\nTest Accuracy: {test_accuracy:.4f}")

                # Print test performance metrics
                print("\nTest Set Performance:")
                y_test_cpu = y_test.cpu().numpy()
                y_pred_cpu = y_pred.cpu().numpy()
                self.print_colored_confusion_matrix(y_test_cpu, y_pred_cpu)

                # Handle inverse DBNN if enabled
                if invert_DBNN:
                    print("\nPhase 3: Inverse model processing...")
                    test_probs = self._get_test_probabilities(X_test)

                    if not hasattr(self, 'inverse_model'):
                        from invertible_dbnn import InvertibleDBNN
                        self.inverse_model = InvertibleDBNN(
                            forward_model=self,
                            feature_dims=X_test.shape[1],
                            reconstruction_weight=reconstruction_weight,
                            feedback_strength=feedback_strength
                        )

                    if hasattr(self, 'inverse_model'):
                        reconstructed_features = self.inverse_model.reconstruct_features(test_probs)

                        if save_path:
                            X_test_df = self.data.drop(columns=[self.target_column]).iloc[
                                self.test_indices if self.in_adaptive_fit else range(len(X_test))
                            ]
                            y_test_series = self.data[self.target_column].iloc[
                                self.test_indices if self.in_adaptive_fit else range(len(X_test))
                            ]

                            self._save_predictions_with_reconstruction(
                                X_test_df, y_pred, save_path, y_test_series, reconstructed_features
                            )

                            results = self.update_results_with_reconstruction(
                                results, X_test, reconstructed_features,
                                test_probs, y_test, save_path
                            )

                # Update results with final metrics
                results.update({
                    'test_accuracy': test_accuracy,
                    'classification_report': classification_report(
                        self.label_encoder.inverse_transform(y_test_cpu),
                        self.label_encoder.inverse_transform(y_pred_cpu)
                    ),
                    'confusion_matrix': confusion_matrix(
                        self.label_encoder.inverse_transform(y_test_cpu),
                        self.label_encoder.inverse_transform(y_pred_cpu)
                    ),
                    'training_complete': final_train_accuracy >= min_training_accuracy
                })

            # Restore original weights
            self.current_W = temp_W
            return results

        except Exception as e:
            print(f"\nError in fit_predict: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def save_reconstruction_features(self,
                                     reconstructed_features: torch.Tensor,
                                     original_features: torch.Tensor,
                                     predictions: torch.Tensor,
                                     true_labels: torch.Tensor = None,
                                     class_probs: torch.Tensor = None) -> Dict:
        """Save reconstruction features and return JSON-compatible output.

        Args:
            reconstructed_features: Reconstructed feature tensor
            original_features: Original input feature tensor
            predictions: Model predictions tensor
            true_labels: True labels tensor (optional)
            class_probs: Class probabilities tensor (optional)

        Returns:
            Dict containing reconstruction data and paths
        """
        # Create reconstruction directory
        dataset_name = os.path.splitext(os.path.basename(self.dataset_name))[0]
        recon_dir = os.path.join('data', dataset_name, 'reconstruction')
        os.makedirs(recon_dir, exist_ok=True)

        # Convert tensors to numpy arrays
        recon_np = reconstructed_features.cpu().numpy()
        orig_np = original_features.cpu().numpy()
        pred_np = predictions.cpu().numpy()

        # Create DataFrame with original and reconstructed features
        feature_cols = [f'feature_{i}' for i in range(orig_np.shape[1])]
        recon_cols = [f'reconstructed_{i}' for i in range(recon_np.shape[1])]

        df = pd.DataFrame(orig_np, columns=feature_cols)
        df = pd.concat([df, pd.DataFrame(recon_np, columns=recon_cols)], axis=1)

        # Add predictions
        df['predicted_class'] = self.label_encoder.inverse_transform(pred_np)

        # Add true labels if provided
        if true_labels is not None:
            true_np = true_labels.cpu().numpy()
            df['true_class'] = self.label_encoder.inverse_transform(true_np)

        # Add class probabilities if provided
        if class_probs is not None:
            probs_np = class_probs.cpu().numpy()
            for i, class_name in enumerate(self.label_encoder.classes_):
                df[f'prob_{class_name}'] = probs_np[:, i]

        # Add reconstruction error
        df['reconstruction_error'] = np.mean((orig_np - recon_np) ** 2, axis=1)

        # Save to CSV
        csv_path = os.path.join(recon_dir, f'{dataset_name}_reconstruction.csv')
        df.to_csv(csv_path, index=False)

        # Create JSON-compatible output
        output = {
            'dataset': dataset_name,
            'reconstruction_path': csv_path,
            'feature_count': orig_np.shape[1],
            'sample_count': len(df),
            'mean_reconstruction_error': float(df['reconstruction_error'].mean()),
            'std_reconstruction_error': float(df['reconstruction_error'].std()),
            'features': {
                'original': feature_cols,
                'reconstructed': recon_cols
            },
            'class_mapping': dict(zip(
                range(len(self.label_encoder.classes_)),
                self.label_encoder.classes_
            ))
        }

        # Save metadata as JSON
        json_path = os.path.join(recon_dir, f'{dataset_name}_reconstruction_meta.json')
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)

        return output

    def _get_config_value(self, param_name: str, default_value: Any) -> Any:
        """Get configuration value with proper fallbacks"""
        if hasattr(self.config, 'to_dict'):
            return getattr(self.config, param_name, default_value)
        elif isinstance(self.config, dict):
            return self.config.get('training_params', {}).get(param_name, default_value)
        return default_value

    def _compute_test_metrics(self, X_test, y_test, train_accuracy, batch_size):
        """Compute test metrics based on training performance"""
        min_training_accuracy = getattr(self.config, 'minimum_training_accuracy', 0.95)

        if not self.in_adaptive_fit or train_accuracy >= min_training_accuracy:
            y_pred = self.predict(X_test, batch_size=batch_size)
            if y_pred.size(0) != y_test.size(0):
                raise ValueError(f"Prediction size mismatch: {y_pred.size(0)} vs {y_test.size(0)}")

            y_test_cpu = y_test.cpu().numpy()
            y_pred_cpu = y_pred.cpu().numpy()
            y_test_labels = self.label_encoder.inverse_transform(y_test_cpu)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred_cpu)

            return {
                'predictions': y_pred,
                'accuracy': (y_pred_cpu == y_test_cpu).mean(),
                'classification_report': classification_report(y_test_labels, y_pred_labels),
                'confusion_matrix': confusion_matrix(y_test_labels, y_pred_labels)
            }

        return {'predictions': None, 'accuracy': 0.0}

    def _prepare_test_data(self, X_test, y_test):
        """Prepare test data for prediction saving"""
        indices = self.test_indices if self.in_adaptive_fit else range(len(X_test))
        X_test_df = self.data.drop(columns=[self.target_column]).iloc[indices]
        y_test_series = self.data[self.target_column].iloc[indices]
        return X_test_df, y_test_series

    def _get_test_probabilities(self, X_test: Union[torch.Tensor, pd.DataFrame]) -> torch.Tensor:
        """Get probabilities with proper type handling"""
        if isinstance(X_test, pd.DataFrame):
            X_processed = self._preprocess_data(X_test, is_training=False)
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
        else:
            X_tensor = X_test

        if self.model_type == "Histogram":
            probs, _ = self._compute_batch_posterior(X_tensor)
        else:
            probs, _ = self._compute_batch_posterior_std(X_tensor)
        return probs


    def _compute_training_metrics(self, X_train, y_train, batch_size):
        """Compute training metrics with proper error handling"""
        with torch.no_grad():
            train_predictions = self.predict(X_train, batch_size=batch_size)
            train_accuracy = (train_predictions == y_train.cpu()).float().mean().item()
        return {'predictions': train_predictions, 'accuracy': train_accuracy}

    def _prepare_training_data(self, batch_size):
        """Prepare training data with proper handling for adaptive and regular modes"""
        if self.in_adaptive_fit:
            if not hasattr(self, 'X_tensor') or not hasattr(self, 'y_tensor'):
                raise ValueError("X_tensor or y_tensor not found.")
            if not hasattr(self, 'train_indices') or not hasattr(self, 'test_indices'):
                raise ValueError("train_indices or test_indices not found")

            return (self.X_tensor[self.train_indices], self.X_tensor[self.test_indices],
                    self.y_tensor[self.train_indices], self.y_tensor[self.test_indices])

        # Regular training path
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Handle label encoding
        y_encoded = (self.label_encoder.transform(y) if hasattr(self.label_encoder, 'classes_')
                    else self.label_encoder.fit_transform(y))

        # Process and convert data
        X_processed = self._preprocess_data(X, is_training=True)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)

        # Get train/test split
        X_train, X_test, y_train, y_test = self._get_train_test_split(X_tensor, y_tensor)

        # Ensure proper device and dtype
        return (X_train.to(self.device, dtype=torch.float32),
                X_test.to(self.device, dtype=torch.float32),
                y_train.to(self.device, dtype=torch.long),
                y_test.to(self.device, dtype=torch.long))


    def _get_model_components_filename(self):
        """Get filename for model components"""
        return os.path.join('Model', f'Best{self.model_type}_{self.dataset_name}_components.pkl')
#----------------Handling categorical variables across sessions -------------------------
    def _save_categorical_encoders(self):
        """Save categorical feature encoders"""
        if self.categorical_encoders:
            # Create a serializable dictionary structure
            encoders_dict = {
                'encoders': {
                    column: {
                        str(k): v for k, v in mapping.items()
                    } for column, mapping in self.categorical_encoders.items()
                }
            }

            # Add metadata
            if hasattr(self, 'original_columns'):
                if isinstance(self.original_columns, list):
                    column_types = {col: str(self.data[col].dtype) for col in self.original_columns if col in self.data.columns}
                else:
                    column_types = {col: str(dtype) for col, dtype in self.original_columns.items()}

                encoders_dict['metadata'] = {
                    'column_types': column_types,
                    'timestamp': pd.Timestamp.now().isoformat()
                }

            with open(self._get_encoders_filename(), 'w') as f:
                json.dump(encoders_dict, f, indent=2)

    def _load_categorical_encoders(self):
        """Load categorical feature encoders from file"""
        encoders_file = self._get_encoders_filename()
        if os.path.exists(encoders_file):
            try:
                with open(encoders_file, 'r') as f:
                    data = json.load(f)

                # Extract encoders from the loaded data
                if 'encoders' in data:
                    self.categorical_encoders = {
                        column: {
                            k: int(v) if isinstance(v, (str, int, float)) else v
                            for k, v in mapping.items()
                        }
                        for column, mapping in data['encoders'].items()
                    }
                else:
                    # Handle legacy format where encoders were at top level
                    self.categorical_encoders = {
                        column: {
                            k: int(v) if isinstance(v, (str, int, float)) else v
                            for k, v in mapping.items()
                        }
                        for column, mapping in data.items()
                    }

                print(f"Loaded categorical encoders from {encoders_file}")
            except Exception as e:
                print(f"Warning: Failed to load categorical encoders: {str(e)}")
                self.categorical_encoders = {}

    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True):
        """Encode categorical features with proper dtype handling"""
        DEBUG.log("Starting categorical encoding")
        df_encoded = df.copy()
        categorical_columns = self._detect_categorical_columns(df)

        for column in categorical_columns:
            if is_training:
                if column not in self.categorical_encoders:
                    # Create new encoder
                    unique_values = df[column].fillna('MISSING').unique()
                    self.categorical_encoders[column] = {
                        value: idx for idx, value in enumerate(unique_values)
                    }

            if column not in self.categorical_encoders:
                continue

            # Get original dtype
            original_dtype = df[column].dtype
            mapping = self.categorical_encoders[column]

            # Handle missing values and new categories
            df_encoded[column] = df[column].fillna('MISSING').map(
                lambda x: mapping.get(x, -1)
            )

            # Handle unmapped values
            unmapped = df_encoded[df_encoded[column] == -1].index
            if len(unmapped) > 0:
                DEBUG.log(f"Found {len(unmapped)} unmapped values in column {column}")

                # Calculate mean value
                mapped_values = [v for v in mapping.values() if isinstance(v, (int, float))]
                if mapped_values:
                    mean_value = float(np.mean(mapped_values))

                    # Convert to proper dtype based on original column type
                    if pd.api.types.is_integer_dtype(original_dtype):
                        mean_value = int(round(mean_value))

                    # Update unmapped values with proper type casting
                    df_encoded.loc[unmapped, column] = pd.Series([mean_value] * len(unmapped), index=unmapped).astype(original_dtype)

        # Verify no categorical columns remain
        remaining_object_cols = df_encoded.select_dtypes(include=['object']).columns
        if len(remaining_object_cols) > 0:
            DEBUG.log(f"Remaining object columns after encoding: {remaining_object_cols}")
            # Convert any remaining object columns to numeric
            for col in remaining_object_cols:
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

        DEBUG.log(f"Categorical encoding complete. Shape: {df_encoded.shape}")
        return df_encoded

    def load_inverse_model(self, custom_path: str = None) -> bool:
       try:
           load_dir = custom_path or os.path.join('Model', f'Best_inverse_{self.forward_model.dataset_name}')
           model_path = os.path.join(load_dir, 'inverse_model.pt')
           config_path = os.path.join(load_dir, 'inverse_config.json')

           if not (os.path.exists(model_path) and os.path.exists(config_path)):
               print(f"No saved inverse model found at {load_dir}")
               return False

           model_state = torch.load(model_path, map_location=self.device, weights_only=True)

           with open(config_path, 'r') as f:
               config = json.load(f)

           if config['feature_dims'] != self.feature_dims or config['n_classes'] != self.n_classes:
               raise ValueError("Model architecture mismatch")

           # Load parameters
           self.weight_linear.data = model_state['weight_linear']
           self.weight_nonlinear.data = model_state['weight_nonlinear']
           self.bias_linear.data = model_state['bias_linear']
           self.bias_nonlinear.data = model_state['bias_nonlinear']
           self.feature_attention.data = model_state['feature_attention']
           self.layer_norm.load_state_dict(model_state['layer_norm'])

           # Safely update or register buffers
           for param in ['min_vals', 'max_vals', 'scale_factors', 'inverse_feature_pairs']:
               if param in model_state:
                   buffer_data = model_state[param]
                   if buffer_data is not None:
                       if hasattr(self, param) and getattr(self, param) is not None:
                           getattr(self, param).copy_(buffer_data)
                       else:
                           self.register_buffer(param, buffer_data)

           # Restore other attributes
           self.metrics = model_state.get('metrics', {})
           self.reconstruction_weight = model_state.get('reconstruction_weight', 0.5)
           self.feedback_strength = model_state.get('feedback_strength', 0.3)

           print(f"Loaded inverse model from {load_dir}")
           return True

       except Exception as e:
           print(f"Error loading inverse model: {str(e)}")
           traceback.print_exc()
           return False

    def save_predictions(self, X: pd.DataFrame, predictions: torch.Tensor, save_path: str, true_labels: pd.Series = None):
        """Save predictions with proper class handling and directory setup"""
        # Create data subdirectory
        dataset_name = os.path.splitext(os.path.basename(self.dataset_name))[0]
        data_dir = os.path.join('data', dataset_name)
        os.makedirs(data_dir, exist_ok=True)

        # Update save path
        save_path = os.path.join(data_dir, os.path.basename(save_path))

        # Create DataFrame
        results_df = X.copy()
        pred_labels = self.label_encoder.inverse_transform(predictions.cpu().numpy())
        results_df['predicted_class'] = pred_labels

        if true_labels is not None:
            results_df['true_class'] = true_labels

        # Get preprocessed features for probability computation
        X_processed = self._preprocess_data(X, is_training=False)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)

        # Compute probabilities in batches
        batch_size = 32
        all_probabilities = []

        for i in range(0, len(X_tensor), batch_size):
            batch_end = min(i + batch_size, len(X_tensor))
            batch_X = X_tensor[i:batch_end]

            try:
                if self.model_type == "Histogram":
                    batch_probs, _ = self._compute_batch_posterior(batch_X)
                elif self.model_type == "Gaussian":
                    batch_probs, _ = self._compute_batch_posterior_std(batch_X)
                else:
                    raise ValueError(f"{self.model_type} is invalid")

                all_probabilities.append(batch_probs.cpu().numpy())

            except Exception as e:
                print(f"Error computing probabilities for batch {i}: {str(e)}")
                return None

        if all_probabilities:
            all_probabilities = np.vstack(all_probabilities)

            # Add probability columns for actual classes
            for i, class_name in enumerate(self.label_encoder.classes_):
                results_df[f'prob_{class_name}'] = all_probabilities[:, i]

            # Add original features for reconstruction
            for i in range(X_tensor.shape[1]):
                results_df[f'original_feature_{i}'] = X_tensor[:, i].cpu().numpy()

            # Save predictions
            results_df.to_csv(save_path, index=False)
            print(f"\nSaved predictions to {save_path}")

            return results_df

        return None
#--------------------------------------------------------------------------------------------------------------
    def _save_model_components(self):
        """Save all model components to a pickle file"""
        components = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'likelihood_params': self.likelihood_params,
            'feature_pairs': self.feature_pairs,
            'categorical_encoders': self.categorical_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'target_classes': self.label_encoder.classes_,
            'target_mapping': dict(zip(self.label_encoder.classes_,
                                     range(len(self.label_encoder.classes_)))),
            'config': self.config,
            'high_cardinality_columns': getattr(self, 'high_cardinality_columns', []),
            'original_columns': getattr(self, 'original_columns', None),
            'best_error': self.best_error,  # Explicitly save best error
            'last_training_loss': getattr(self, 'last_training_loss', float('inf')),
            'weight_updater': self.weight_updater,
            'n_bins_per_dim': self.n_bins_per_dim
        }

        # Get the filename using existing method
        components_file = self._get_model_components_filename()


        # Ensure directory exists
        os.makedirs(os.path.dirname(components_file), exist_ok=True)

        # Save components to file
        with open(components_file, 'wb') as f:
            pickle.dump(components, f)

        print(f"Saved model components to {components_file}")
        return True



    def _load_model_components(self):
        """Load all model components"""
        components_file = self._get_model_components_filename()
        if os.path.exists(components_file):
            with open(components_file, 'rb') as f:
                components = pickle.load(f)
                self.label_encoder.classes_ = components['target_classes']
                self.scaler = components['scaler']
                self.label_encoder = components['label_encoder']
                self.likelihood_params = components['likelihood_params']
                self.feature_pairs = components['feature_pairs']
                self.feature_columns = components.get('feature_columns')
                self.categorical_encoders = components['categorical_encoders']
                self.high_cardinality_columns = components.get('high_cardinality_columns', [])
                print(f"Loaded model components from {components_file}")
                self.weight_updater = components.get('weight_updater')
                self.n_bins_per_dim = components.get('n_bins_per_dim', 20)
                return True
        return False

    def _save_reconstruction_stats(self, orig_df: pd.DataFrame, recon_df: pd.DataFrame):
        """Save reconstruction statistics"""
        feature_cols = [col for col in orig_df.columns if not col.startswith(('prob_', 'predicted_', 'true_', 'original_', 'reconstructed_'))]
        stats = {
            'per_feature_mse': {},
            'per_feature_correlation': {},
            'prediction_accuracy': None
        }

        # Feature stats
        for col in feature_cols:
            stats['per_feature_mse'][col] = float(np.mean((orig_df[col] - recon_df[col])**2))
            stats['per_feature_correlation'][col] = float(np.corrcoef(orig_df[col], recon_df[col])[0,1])

        # Save stats
        dataset_name = os.path.splitext(os.path.basename(self.dataset_name))[0]
        stats_path = os.path.join('data', dataset_name, 'reconstruction_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)

    def save_reconstructed_features(self, predictions_df: pd.DataFrame, reconstructed_features: torch.Tensor, save_path: str):
        """Save reconstructed features in same format as input CSV"""
        # Create DataFrame with reconstructed features
        recon_df = pd.DataFrame()
        n_features = reconstructed_features.shape[1]

        # Add reconstructed features with original column names if available
        feature_cols = [col for col in predictions_df.columns if not col.startswith(('prob_', 'predicted_', 'true_', 'original_', 'reconstructed_'))]
        for i in range(n_features):
            col_name = feature_cols[i] if i < len(feature_cols) else f"feature_{i}"
            recon_df[col_name] = reconstructed_features[:, i].cpu().numpy()

        # Add true class if available
        if 'true_class' in predictions_df:
            recon_df[self.target_column] = predictions_df['true_class']

        # Save to data subdirectory
        dataset_name = os.path.splitext(os.path.basename(self.dataset_name))[0]
        save_dir = os.path.join('data', dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        recon_df.to_csv(os.path.join(save_dir, 'reconstructed_input.csv'), index=False)

        # Save reconstruction stats
        self._save_reconstruction_stats(predictions_df, recon_df)

    def predict_and_save(self, save_path=None, batch_size: int = 32):
       """Make predictions and save with reconstructions and verification"""
       try:
           DEBUG.log(" Starting prediction and save process")
           DEBUG.log(f" Model type: {self.model_type}, Batch size: {batch_size}")

           # Setup directories
           dataset_name = os.path.splitext(os.path.basename(self.dataset_name))[0]
           data_dir = os.path.join('data', dataset_name)
           os.makedirs(data_dir, exist_ok=True)

           predictions_path = os.path.join(data_dir, f"{dataset_name}_predictions.csv")
           reconstructed_path = os.path.join(data_dir, 'reconstructed_input.csv')

           weights_loaded = os.path.exists(self._get_weights_filename())
           components_loaded = self._load_model_components()

           # Handle prediction-only mode
           if (not self.config.get('execution_flags', {}).get('train', True) and
               self.config.get('execution_flags', {}).get('predict', True)):

               if os.path.exists(predictions_path):
                   DEBUG.log(f" Loading existing predictions from {predictions_path}")
                   try:
                       predictions_df = pd.read_csv(predictions_path)
                       pred_labels = predictions_df['predicted_class'].values
                       predictions = torch.tensor(
                           self.label_encoder.transform(pred_labels),
                           device=self.device
                       )

                       prob_columns = [col for col in predictions_df.columns if col.startswith('prob_')]
                       probabilities = torch.tensor(
                           predictions_df[prob_columns].values,
                           dtype=torch.float32,
                           device=self.device
                       ) if prob_columns else None

                       results = {
                           'predictions': predictions,
                           'probabilities': probabilities,
                           'error_rates': []
                       }

                       if 'true_class' in predictions_df:
                           true_labels = predictions_df['true_class'].values
                           results.update({
                               'confusion_matrix': confusion_matrix(true_labels, pred_labels),
                               'classification_report': classification_report(true_labels, pred_labels),
                               'test_accuracy': (predictions == torch.tensor(
                                   self.label_encoder.transform(true_labels),
                                   device=self.device
                               )).float().mean().item()
                           })

                       if self.config.get('training_params', {}).get('invert_DBNN', False) and probabilities is not None:
                           DEBUG.log(" Processing inverse model reconstruction")
                           if not hasattr(self, 'inverse_model'):
                               self.inverse_model = InvertibleDBNN(
                                   forward_model=self,
                                   feature_dims=len(self.feature_columns),
                                   reconstruction_weight=self.config.get('training_params', {}).get('reconstruction_weight', 0.5),
                                   feedback_strength=self.config.get('training_params', {}).get('feedback_strength', 0.3)
                               )

                           if self.inverse_model.load_inverse_model():
                               reconstructed = self.inverse_model.reconstruct_features(probabilities)
                               self.save_reconstructed_features(predictions_df, reconstructed, reconstructed_path)
                               self._save_reconstruction_stats(predictions_df, pd.read_csv(reconstructed_path))

                               recon_verification = self.verify_reconstruction_predictions(
                                   predictions_df,
                                   pd.read_csv(reconstructed_path)
                               )
                               if recon_verification:
                                   print("\nReconstruction Verification:")
                                   print(f"Prediction Match Accuracy: {recon_verification['reconstruction_prediction_accuracy']:.4f}")
                                   if recon_verification['reconstruction_true_accuracy']:
                                       print(f"True Label Match Accuracy: {recon_verification['reconstruction_true_accuracy']:.4f}")
                                   print("\nReconstruction Prediction Confusion Matrix:")
                                   print(recon_verification['confusion_matrix'])

                                   results['reconstruction_verification'] = recon_verification

                       return results

                   except Exception as e:
                       print(f"Error loading predictions: {str(e)}")
                       traceback.print_exc()
                       print("Falling back to model prediction")

           # Check model components
           if not (weights_loaded and components_loaded):
               print("Complete model not found. Training required.")
               results = self.fit_predict(batch_size=batch_size)
               return results

           self._load_best_weights()
           self._load_categorical_encoders()

           temp_W = self.current_W
           self.current_W = self.best_W.clone() if self.best_W is not None else self.current_W

           try:
               DEBUG.log(" Processing predictions on data")
               X = self.data.drop(columns=[self.target_column])
               true_labels = self.data[self.target_column]
               X_tensor = self._preprocess_data(X, is_training=False)

               predictions = []
               probabilities = []

               for i in range(0, len(X_tensor), batch_size):
                   batch_end = min(i + batch_size, len(X_tensor))
                   batch_X = X_tensor[i:batch_end]

                   if self.model_type == "Histogram":
                       batch_probs, _ = self._compute_batch_posterior(batch_X)
                   else:
                       batch_probs, _ = self._compute_batch_posterior_std(batch_X)

                   predictions.append(torch.argmax(batch_probs, dim=1))
                   probabilities.append(batch_probs)

               predictions = torch.cat(predictions)
               probabilities = torch.cat(probabilities)

               if save_path:
                   save_path = os.path.join(data_dir, os.path.basename(save_path))
                   result_df = X.copy()
                   pred_labels = self.label_encoder.inverse_transform(predictions.cpu().numpy())
                   result_df['predicted_class'] = pred_labels

                   if true_labels is not None:
                       result_df['true_class'] = true_labels

                   for i, class_name in enumerate(self.label_encoder.classes_):
                       result_df[f'prob_{class_name}'] = probabilities[:, i].cpu().numpy()

                   for i in range(X_tensor.shape[1]):
                       result_df[f'original_feature_{i}'] = X_tensor[:, i].cpu().numpy()

                   result_df.to_csv(save_path, index=False)
                   DEBUG.log(f" Saved predictions to {save_path}")

                   if self.config.get('training_params', {}).get('invert_DBNN', False):
                       DEBUG.log(" Processing inverse model for new predictions")
                       if not hasattr(self, 'inverse_model'):
                           self.inverse_model = InvertibleDBNN(
                               forward_model=self,
                               feature_dims=len(self.feature_columns),
                               reconstruction_weight=self.config.get('training_params', {}).get('reconstruction_weight', 0.5),
                               feedback_strength=self.config.get('training_params', {}).get('feedback_strength', 0.3)
                           )

                       if self.inverse_model.load_inverse_model():
                           reconstructed = self.inverse_model.reconstruct_features(probabilities)
                           self.save_reconstructed_features(result_df, reconstructed, reconstructed_path)
                           self._save_reconstruction_stats(result_df, pd.read_csv(reconstructed_path))

                           recon_verification = self.verify_reconstruction_predictions(
                               result_df,
                               pd.read_csv(reconstructed_path)
                           )
                           if recon_verification:
                               DEBUG.log(" Reconstruction verification results:")
                               DEBUG.log(f" - Prediction match accuracy: {recon_verification['reconstruction_prediction_accuracy']:.4f}")
                               if recon_verification['reconstruction_true_accuracy']:
                                   DEBUG.log(f" - True label match accuracy: {recon_verification['reconstruction_true_accuracy']:.4f}")

                               print("\nReconstruction Verification:")
                               print(f"Prediction Match Accuracy: {recon_verification['reconstruction_prediction_accuracy']:.4f}")
                               if recon_verification['reconstruction_true_accuracy']:
                                   print(f"True Label Match Accuracy: {recon_verification['reconstruction_true_accuracy']:.4f}")
                               print("\nReconstruction Prediction Confusion Matrix:")
                               print(recon_verification['confusion_matrix'])

               results = {
                   'predictions': predictions,
                   'probabilities': probabilities,
                   'error_rates': getattr(self, 'error_rates', [])
               }

               if true_labels is not None:
                   results.update({
                       'confusion_matrix': confusion_matrix(
                           true_labels,
                           self.label_encoder.inverse_transform(predictions.cpu().numpy())
                       ),
                       'classification_report': classification_report(
                           true_labels,
                           self.label_encoder.inverse_transform(predictions.cpu().numpy())
                       ),
                       'test_accuracy': (predictions == torch.tensor(
                           self.label_encoder.transform(true_labels),
                           device=self.device
                       )).float().mean().item()
                   })

               return results

           finally:
               self.current_W = temp_W

       except Exception as e:
           print(f"Error during prediction process: {str(e)}")
           traceback.print_exc()
           return None


#--------------------------------------------------Class Ends ----------------------------------------------------------
class DebugLogger:
    def __init__(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def log(self, msg, force=False):
        if self.enabled or force:
            print(msg)

class DatasetProcessor:
    def __init__(self):
        self.base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        self.compressed_extensions = ['.zip', '.gz', '.tar', '.7z', '.rar']
        self.debug = DebugLogger()
        self.colors = Colors()

    def process_dataset(self, file_path: str) -> None:
        """Process dataset with proper path handling.

        Args:
            file_path: Path to the dataset file
        """
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Create main data directory if it doesn't exist
            if not os.path.exists('data'):
                os.makedirs('data')

            # Setup dataset folder structure
            dataset_folder = os.path.join('data', base_name)
            os.makedirs(dataset_folder, exist_ok=True)

            print(f"\nProcessing dataset:")
            print(f"Base name: {base_name}")
            print(f"Dataset folder: {dataset_folder}")

            # Define target CSV path
            target_csv = os.path.join(dataset_folder, f"{base_name}.csv")

            # If file exists at original path and isn't in dataset folder, copy it
            if os.path.exists(file_path) and os.path.isfile(file_path) and file_path != target_csv:
                try:
                    import shutil
                    shutil.copy2(file_path, target_csv)
                    print(f"Copied dataset to: {target_csv}")
                except Exception as e:
                    print(f"Warning: Could not copy dataset: {str(e)}")

            # If file doesn't exist in target location, try downloading from UCI
            if not os.path.exists(target_csv):
                print(f"File not found locally: {target_csv}")
                print("Attempting to download from UCI repository...")
                downloaded_path = self._download_from_uci(base_name.upper())
                if downloaded_path:
                    print(f"Successfully downloaded dataset to {downloaded_path}")
                    # Ensure downloaded file is in the correct location
                    if downloaded_path != target_csv:
                        try:
                            import shutil
                            shutil.move(downloaded_path, target_csv)
                        except Exception as e:
                            print(f"Warning: Could not move downloaded file: {str(e)}")
                else:
                    print(f"Could not find or download dataset: {base_name}")
                    return None

            # Verify file exists before proceeding
            if not os.path.exists(target_csv):
                raise FileNotFoundError(f"Dataset file not found at {target_csv}")

            # Process based on dataset structure
            config = self._create_dataset_configs(dataset_folder, base_name)

            if self._has_test_train_split(dataset_folder, base_name):
                print("Found train/test split structure")
                return self._handle_split_dataset(dataset_folder, base_name)
            elif os.path.exists(target_csv):
                print("Found single CSV file structure")
                return self._handle_single_csv(dataset_folder, base_name, config)
            elif self._is_compressed(file_path):
                print("Found compressed file, extracting...")
                extracted_path = self._decompress(file_path, dataset_folder)
                return self.process_dataset(extracted_path)
            else:
                print(f"Could not determine dataset structure for {dataset_folder}")
                return None

        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            traceback.print_exc()
            return None
    def _has_single_csv(self, folder_path: str, base_name: str) -> bool:
        """Check if dataset has single CSV file"""
        # Check both possible locations
        csv_paths = [
            os.path.join(folder_path, f"{base_name}.csv"),
            os.path.join(folder_path, base_name, f"{base_name}.csv")
        ]
        exists = any(os.path.exists(path) for path in csv_paths)
        if exists:
            found_path = next(path for path in csv_paths if os.path.exists(path))
            print(f"Found CSV file: {found_path}")
        return exists

    def _has_test_train_split(self, folder_path: str, base_name: str) -> bool:
        """Check for train/test split in dataset folder structure"""
        dataset_folder = os.path.join(folder_path, base_name)
        train_path = os.path.join(dataset_folder, 'train')
        test_path = os.path.join(dataset_folder, 'test')

        # Check if both train and test folders exist
        has_folders = os.path.exists(train_path) and os.path.exists(test_path)

        if has_folders:
            # Check for either dataset-named files or train.csv/test.csv
            train_files = [
                os.path.join(train_path, f"{base_name}.csv"),
                os.path.join(train_path, "train.csv")
            ]
            test_files = [
                os.path.join(test_path, f"{base_name}.csv"),
                os.path.join(test_path, "test.csv")
            ]

            has_train = any(os.path.exists(f) for f in train_files)
            has_test = any(os.path.exists(f) for f in test_files)

            if has_train and has_test:
                train_file = next(f for f in train_files if os.path.exists(f))
                test_file = next(f for f in test_files if os.path.exists(f))
                print(f"Found train file: {train_file}")
                print(f"Found test file: {test_file}")
                return True

        return False


    def find_dataset_pairs(self, data_dir: str = 'data') -> List[Tuple[str, str, str]]:
            """Find and validate dataset configuration pairs.

            Args:
                data_dir: Base directory to search for datasets

            Returns:
                List of tuples (dataset_name, config_path, csv_path)
            """
            if not os.path.exists(data_dir):
                print(f"\nNo '{data_dir}' directory found. Creating one...")
                os.makedirs(data_dir)
                return []

            dataset_pairs = []
            processed_datasets = set()
            adaptive_conf = self._load_global_adaptive_config()

            # Walk through all subdirectories
            for root, dirs, files in os.walk(data_dir):
                conf_files = [f for f in files if f.endswith('.conf') and f != 'adaptive_dbnn.conf']

                for conf_file in conf_files:
                    basename = os.path.splitext(conf_file)[0]
                    if basename in processed_datasets:
                        continue

                    conf_path = os.path.join(root, conf_file)

                    # Check for CSV in multiple possible locations
                    csv_paths = [
                        os.path.join(root, f"{basename}.csv"),                     # Same directory as conf
                        os.path.join(root, basename, f"{basename}.csv"),          # Subdirectory
                        os.path.join(root, basename, 'train', f"{basename}.csv"), # Train directory
                        os.path.join(root, basename, 'train', "train.csv"),       # Train directory with default name
                        os.path.join(root, 'train', f"{basename}.csv"),          # Direct train directory
                        os.path.join(root, 'train', "train.csv")                 # Direct train directory with default name
                    ]

                    # Find first existing CSV file
                    csv_path = next((path for path in csv_paths if os.path.exists(path)), None)

                    if csv_path:
                        if adaptive_conf:
                            self._update_config_with_adaptive(conf_path, adaptive_conf)

                        print(f"\nFound dataset: {basename}")
                        print(f"Config: {conf_path}")
                        print(f"Data: {csv_path}")

                        dataset_pairs.append((basename, conf_path, csv_path))
                        processed_datasets.add(basename)

            return dataset_pairs

    def _load_global_adaptive_config(self) -> Dict:
        adaptive_path = 'adaptive_dbnn.conf'
        if os.path.exists(adaptive_path):
            try:
                with open(adaptive_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.debug.log(f"Warning: Could not load adaptive configuration: {str(e)}")
        return {}

    def _update_config_with_adaptive(self, conf_path: str, adaptive_conf: Dict):
        """Update dataset configuration with global adaptive settings."""
        try:
            with open(conf_path, 'r') as f:
                dataset_conf = json.load(f)

            # Update execution flags
            if 'execution_flags' in adaptive_conf:
                dataset_conf['execution_flags'] = adaptive_conf['execution_flags']

            # Update training parameters
            if 'training_params' in adaptive_conf:
                if 'training_params' not in dataset_conf:
                    dataset_conf['training_params'] = {}
                dataset_conf['training_params'].update(adaptive_conf['training_params'])

            with open(conf_path, 'w') as f:
                json.dump(dataset_conf, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not update configuration: {str(e)}")
            traceback.print_exc()


    def _handle_split_dataset(self, folder_path: str, base_name: str):
        """Handle dataset with train/test split following specific folder structure rules"""
        # Setup paths
        dataset_folder = os.path.join(folder_path, base_name)
        train_path = os.path.join(dataset_folder, 'train')
        test_path = os.path.join(dataset_folder, 'test')
        main_csv_path = os.path.join(dataset_folder, f"{base_name}.csv")

        # Load configuration
        config = self._validate_config(folder_path, base_name)
        model = DBNN(base_name, config)

        if config.get('modelType', 'Histogram') == 'Histogram':
            if input("Merge train/test data? (y/n): ").lower() == 'y':
                # Check if merged file already exists
                if os.path.exists(main_csv_path):
                    print(f"Using existing merged dataset: {main_csv_path}")
                    merged_df = pd.read_csv(main_csv_path)
                else:
                    print("Merging train and test datasets...")
                    # Look for dataset-named files first, then fall back to train.csv/test.csv
                    train_file = os.path.join(train_path, f"{base_name}.csv")
                    if not os.path.exists(train_file):
                        train_file = os.path.join(train_path, "train.csv")

                    test_file = os.path.join(test_path, f"{base_name}.csv")
                    if not os.path.exists(test_file):
                        test_file = os.path.join(test_path, "test.csv")

                    try:
                        train_df = pd.read_csv(train_file)
                        test_df = pd.read_csv(test_file)
                        merged_df = pd.concat([train_df, test_df], ignore_index=True)

                        # Save merged file in dataset folder
                        os.makedirs(dataset_folder, exist_ok=True)
                        merged_df.to_csv(main_csv_path, index=False)
                        print(f"Saved merged dataset to: {main_csv_path}")
                    except Exception as e:
                        print(f"Error merging datasets: {str(e)}")
                        return None

                # Process merged dataset
                results = self.run_benchmark(base_name, model)
                return self._save_results(results, dataset_folder, base_name)

            else:
                # Use separate train and test files
                print("Using separate train and test datasets...")
                # Try dataset-named files first
                train_file = os.path.join(train_path, f"{base_name}.csv")
                if not os.path.exists(train_file):
                    train_file = os.path.join(train_path, "train.csv")
                    if not os.path.exists(train_file):
                        print(f"Error: No training file found in {train_path}")
                        return None

                test_file = os.path.join(test_path, f"{base_name}.csv")
                if not os.path.exists(test_file):
                    test_file = os.path.join(test_path, "test.csv")
                    if not os.path.exists(test_file):
                        print(f"Error: No test file found in {test_path}")
                        return None

                try:
                    train_df = pd.read_csv(train_file)
                    test_df = pd.read_csv(test_file)
                    print(f"Using training data from: {train_file}")
                    print(f"Using test data from: {test_file}")
                    results = self.run_benchmark(base_name, model, train_df, test_df)
                    return self._save_results(results, dataset_folder, base_name)
                except Exception as e:
                    print(f"Error processing split datasets: {str(e)}")
                    return None

    # In DatasetProcessor class
    def _handle_single_csv(self, folder_path: str, base_name: str, config: Dict):
        """Handle dataset with single CSV file and debug config processing"""
        #print("\nDEBUGEntering _handle_single_csv")
        # print(f"DEBUG:  Initial config: {json.dumps(config, indent=2) if config else 'None'}")

        # Handle CSV paths
        csv_paths = [
            os.path.join(folder_path, f"{base_name}.csv"),
            os.path.join(folder_path, base_name, f"{base_name}.csv")
        ]
        csv_path = next((path for path in csv_paths if os.path.exists(path)), None)

        if not csv_path:
            return None

        # Ensure we have a valid config
        if config is None:
           ## print("DEBUG: No config provided, validating...")
            config = self._validate_config(folder_path, base_name)

        ##print("\nDEBUGConfig before GlobalConfig conversion:")
        #print(json.dumps(config, indent=2))

        # Create GlobalConfig
        global_config = GlobalConfig.from_dict(config)

        # Create and return DBNN model
        model = DBNN(base_name, global_config)
        results = self.run_benchmark(base_name, model)
        return self._save_results(results, folder_path, base_name)

    def _try_uci_download(self, dataset_name: str, folder_path: str, config: Dict):
       if dataset_path := self._download_from_uci(dataset_name.upper()):
           return self.process_dataset(dataset_path)
       print(f"Could not find or download dataset: {dataset_name}")
       return None


    def process_by_path(self, file_path: str) -> None:
        """Main entry point for processing a dataset by path"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        folder_path = os.path.join('data', base_name)

        # Sequential processing logic
        processors = [
            self._handle_split_dataset,
            self._handle_single_csv,
            self._handle_compressed,
            self._handle_uci_download
        ]

        for processor in processors:
            if result := processor(folder_path, base_name):
                return result

        print(f"Could not process dataset: {file_path}")

    def save_results(self, results: Dict, dataset_name: str) -> None:
        """Save comprehensive results to text file.

        Args:
            results: Dictionary containing all results
            dataset_name: Name of the dataset
        """
        results_dir = os.path.join('data', dataset_name)
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'{dataset_name}_results.txt')

        with open(results_path, 'w') as f:
            # Header
            f.write(f"Results for Dataset: {dataset_name}\n\n")

            # Classification Report
            if 'classification_report' in results:
                f.write("Classification Report:\n")
                f.write(results['classification_report'])
                f.write("\n\n")

            # Confusion Matrix
            if 'confusion_matrix' in results:
                f.write("Confusion Matrix:\n")
                matrix = results['confusion_matrix']
                f.write("\n".join(["\t".join(map(str, row)) for row in matrix]))
                f.write("\n\n")

            # Error Rates
            if 'error_rates' in results:
                f.write("Error Rates:\n")
                error_rates = results['error_rates']
                if error_rates:
                    for i, rate in enumerate(error_rates):
                        f.write(f"Epoch {i+1}: {rate:.4f}\n")
                else:
                    f.write("N/A\n")
                f.write("\n")

            # Test Accuracy
            if 'test_accuracy' in results:
                f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n\n")

            # Reconstruction Metrics
            if 'reconstruction_metrics' in results:
                f.write("Reconstruction Metrics:\n")
                recon_metrics = results['reconstruction_metrics']
                if 'final_reconstruction_error' in recon_metrics:
                    error = recon_metrics['final_reconstruction_error']
                    if error is not None:
                        f.write(f"Final Reconstruction Error: {error:.6f}\n")
                f.write("\n")

            # Save paths
            if 'reconstruction_path' in results:
                f.write("Output Files:\n")
                f.write(f"Reconstruction data: {results['reconstruction_path']}\n")
                if 'metadata_path' in results:
                    f.write(f"Reconstruction metadata: {results['metadata_path']}\n")
                f.write("\n")

        print(f"Results saved to {results_path}")

    def _save_results(self, results_tuple, folder_path: str, dataset_name: str):
            """Save formatted results to a text file.

            Args:
                results_tuple: Tuple of (model, results) from run_benchmark
                folder_path: Path to save results
                dataset_name: Name of the dataset
            """
            results_path = os.path.join(folder_path, f"{dataset_name}_results.txt")

            # Unpack the tuple
            model, results = results_tuple

            # Format the results into a human-readable string
            result_text = f"Results for Dataset: {dataset_name}\n\n"

            # Add classification report
            result_text += "Classification Report:\n"
            if isinstance(results, dict):
                result_text += results.get('classification_report', 'N/A') + "\n\n"
            else:
                result_text += "N/A\n\n"

            # Add confusion matrix
            result_text += "Confusion Matrix:\n"
            if isinstance(results, dict) and results.get('confusion_matrix') is not None:
                confusion_matrix = results['confusion_matrix']
                result_text += "\n".join(["\t".join(map(str, row)) for row in confusion_matrix]) + "\n\n"
            else:
                result_text += "N/A\n\n"

            # Add error rates
            result_text += "Error Rates:\n"
            if isinstance(results, dict):
                error_rates = results.get('error_rates', [])
                if error_rates:
                    result_text += "\n".join([f"Epoch {i+1}: {rate:.4f}" for i, rate in enumerate(error_rates)]) + "\n\n"
                else:
                    result_text += "N/A\n\n"
            else:
                result_text += "N/A\n\n"

            # Add test accuracy
            if isinstance(results, dict):
                result_text += f"Test Accuracy: {results.get('test_accuracy', 'N/A')}\n"
            else:
                result_text += "Test Accuracy: N/A\n"

            # Write the formatted results to a text file
            with open(results_path, 'w') as f:
                f.write(result_text)

            print(f"\nResults saved to {results_path}")
            return results_tuple

    def run_benchmark(self, dataset_name: str, model=None, batch_size: int = 32):
       """Complete benchmarking implementation with full debug trace."""
       try:
           print(f"\nBenchmarking {self.colors.highlight_dataset(dataset_name)}")

           #print("\nDEBUGConfiguration Loading Phase")
           if hasattr(model.config, 'to_dict'):
               config_dict = model.config.to_dict()
           elif isinstance(model.config, dict):
               config_dict = model.config.copy()
           else:
               config_dict = {
                   'execution_flags': {'train': True},
                   'training_params': {'enable_adaptive': True}
               }

           should_train = config_dict.get('execution_flags', {}).get('train', True)
           enable_adaptive = config_dict.get('training_params', {}).get('enable_adaptive', True)
           invert_DBNN = config_dict.get('training_params', {}).get('invert_DBNN', False)

           print("\nDEBUG: Execution Flags:")
           print(f"- should_train: {should_train}")
           print(f"- enable_adaptive: {enable_adaptive}")
           print(f"- invert_DBNN: {invert_DBNN}")

           if not should_train:
               # Check for existence of model files
               weights_exist = os.path.exists(model._get_weights_filename())
               components_exist = os.path.exists(model._get_model_components_filename())
               predictions_exist = os.path.exists(f"{dataset_name}_predictions.csv")

               if weights_exist and components_exist:
                   print("Found existing model files, loading predictions...")
                   results = model.predict_and_save(
                       save_path=f"{dataset_name}_predictions.csv",
                       batch_size=batch_size
                   )
                   if results is not None:
                       return model, results

               missing_files = []
               if not weights_exist:
                   missing_files.append("model weights")
               if not components_exist:
                   missing_files.append("model components")
               if not predictions_exist:
                   missing_files.append("prediction file")

               print(f"\nMissing required files: {', '.join(missing_files)}")
               response = input("Training is disabled but required files are missing. Enable training? (y/n): ")
               if response.lower() != 'y':
                   print("Exiting without training")
                   return model, {"error": "Required files missing and training disabled"}
               else:
                   should_train = True
                   config_dict['execution_flags']['train'] = True

           if invert_DBNN:
               print("\nDEBUG: Inverse DBNN Settings:")
               for param in ['reconstruction_weight', 'feedback_strength', 'inverse_learning_rate']:
                   value = config_dict.get('training_params', {}).get(param, 0.1)
                   #print(f"- {param}: {value}")
              ## print("DEBUG: Initializing inverse model...")

               if not should_train:
                   inverse_model_path = os.path.join('Model', f'Best_inverse_{dataset_name}', 'inverse_model.pt')
                   if not os.path.exists(inverse_model_path):
                       print("Inverse model file missing")
                       if input("Train inverse model? (y/n): ").lower() != 'y':
                           print("Skipping inverse model")
                           config_dict['training_params']['invert_DBNN'] = False

           print("\nDEBUG: Starting Processing Phase")
           if should_train:
               if enable_adaptive:
                  ## print("DEBUG: Running adaptive training...")
                   history = model.adaptive_fit_predict(
                       max_rounds=model.max_epochs,
                       batch_size=batch_size
                   )
                  ## print("DEBUG: Adaptive training completed")

              ## print("DEBUG: Running prediction and save...")
               results = model.predict_and_save(
                   save_path=f"{dataset_name}_predictions.csv",
                   batch_size=batch_size
               )
              ## print("DEBUG: Prediction completed")

               if not isinstance(results, dict):
                   if hasattr(history, 'get'):
                       results = history
                   else:
                       results = {
                           'predictions': results if torch.is_tensor(results) else None,
                           'error_rates': getattr(model, 'error_rates', []),
                           'confusion_matrix': getattr(model, 'confusion_matrix', None),
                           'classification_report': getattr(model, 'classification_report', '')
                       }

               if invert_DBNN and hasattr(model, 'inverse_model'):
                   try:
                      ## print("DEBUG: Processing inverse model...")
                       X_test = model.data.drop(columns=[model.target_column])
                       test_probs = model._get_test_probabilities(X_test)
                       reconstruction_features = model.inverse_model.reconstruct_features(test_probs)
                       results = model.update_results_with_reconstruction(
                           results, X_test, reconstruction_features,
                           test_probs, model.y_tensor,
                           f"{dataset_name}_predictions.csv"
                       )
                   except Exception as e:
                       print(f"Error in inverse model processing: {str(e)}")
                       traceback.print_exc()

               return model, results

           return model, results

       except Exception as e:
           print("\nDEBUG: Error in benchmark")
           print("-" * 50)
           print(f"Error type: {type(e).__name__}")
           print(f"Error message: {str(e)}")
           traceback.print_exc()
           return None

    def plot_training_progress(self, error_rates: List[float], dataset_name: str):
        plt.figure(figsize=(10, 6))
        plt.plot(error_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
        plt.title(f'Training Progress - {dataset_name.capitalize()}')
        plt.grid(True)
        plt.savefig(f'{dataset_name}_training_progress.png')
        plt.close()

    def plot_confusion_matrix(self, confusion_mat: np.ndarray, class_names: np.ndarray, dataset_name: str):
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {dataset_name.capitalize()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{dataset_name}_confusion_matrix.png')
        plt.close()




    def _handle_compressed(self, folder_path: str, dataset_name: str):
        file_path = os.path.join(folder_path, f"{dataset_name}")
        if not self._is_compressed(file_path):
            return None

        extracted_path = self._decompress(file_path)
        return self.process_by_path(extracted_path)

    def _handle_uci_download(self, folder_path: str, dataset_name: str):
        if dataset_path := self._download_from_uci(dataset_name.upper()):
            return self.process_by_path(dataset_path)
        return None

    def _validate_inverse_config(self) -> bool:
        """
        Validate inverse DBNN configuration without modification.
        Only checks for presence and validity of required parameters.
        """
        if not hasattr(self.config, 'to_dict') and not isinstance(self.config, dict):
            print("Warning: Invalid configuration object")
            return False

        # Check if inverse DBNN is enabled (respect existing value)
        invert_DBNN = self._get_config_param('invert_DBNN', False)
        if not invert_DBNN:
            return False

        # Only validate presence and basic type checking of required parameters
        required_params = {
            'reconstruction_weight': float,
            'feedback_strength': float,
            'inverse_learning_rate': float
        }

        for param, expected_type in required_params.items():
            value = self._get_config_param(param, None)
            if value is None:
                print(f"Missing required inverse parameter: {param}")
                return False
            if not isinstance(value, expected_type):
                print(f"Invalid type for {param}: expected {expected_type.__name__}, got {type(value).__name__}")
                return False

        return True

    def _validate_config(self, folder_path: str, base_name: str) -> Dict:
        """Validate and load configuration"""
        config_path = os.path.join(folder_path, f"{base_name}.conf")

        try:
            # # print(f"\nDEBUG: Loading config from {config_path}")

            with open(config_path, 'r') as f:
                config = json.load(f)
                # # print(f"DEBUG:   Loaded raw config: {json.dumps(config, indent=2)}")

            # Ensure required sections exist
            if 'training_params' not in config:
                config['training_params'] = {}
            if 'execution_flags' not in config:
                config['execution_flags'] = {}

            return config

        except Exception as e:
            print(f"ERROR: Failed to load config: {str(e)}")
            return None

    def _create_or_load_dataset_config(self, folder_path: str, dataset_name: str) -> Dict:
       """Create or load dataset-specific configuration"""
       config_path = os.path.join(folder_path, f"{dataset_name}.conf")

       if os.path.exists(config_path):
           with open(config_path, 'r') as f:
               return json.load(f)

       # Create default dataset config
       csv_path = os.path.join(folder_path, f"{dataset_name}.csv")
       df = pd.read_csv(csv_path, nrows=0)

       default_config = {
           "file_path": csv_path,
           "column_names": df.columns.tolist(),
           "separator": ",",
           "has_header": True,
           "target_column": df.columns[-1],
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
               "Save_training_epochs": True,
               "training_save_path": f"training_data/{dataset_name}"
           },
           "modelType": "Histogram"
       }

       with open(config_path, 'w') as f:
           json.dump(default_config, f, indent=4)

       return default_config

    def _create_dataset_configs(self, folder_path: str, dataset_name: str) -> Dict:
       """Create or load both dataset and adaptive configs"""
       dataset_config = self._create_or_load_dataset_config(folder_path, dataset_name)
       adaptive_config = self._create_or_load_adaptive_config(folder_path, dataset_name)
       return self._merge_configs(dataset_config, adaptive_config)

    def _create_or_load_adaptive_config(self, folder_path: str, dataset_name: str) -> Dict:
       """Create or load dataset-specific adaptive config"""
       adaptive_path = os.path.join(folder_path, 'adaptive_dbnn.conf')
       if os.path.exists(adaptive_path):
           with open(adaptive_path, 'r') as f:
               return json.load(f)

       default_adaptive = {
           "training_params": {
               "trials": 100,
               "cardinality_threshold": 0.9,
               "cardinality_tolerance": 4,
               "learning_rate": 0.1,
               "random_seed": 42,
               "epochs": 100,
               "test_fraction": 0.2,
               "enable_adaptive": True,
               "modelType": "Histogram",
               "compute_device": "auto",
               "use_interactive_kbd": False,
               "debug_enabled": True,
               "Save_training_epochs": True,
               "training_save_path": f"training_data/{dataset_name}"
           },
           "execution_flags": {
               "train": True,
               "train_only": False,
               "predict": True,
               "gen_samples": False,
               "fresh_start": False,
               "use_previous_model": True
           }
       }

       with open(adaptive_path, 'w') as f:
           json.dump(default_adaptive, f, indent=4)
       return default_adaptive

    def _merge_configs(self, dataset_config: Dict, adaptive_config: Dict) -> Dict:
       """Merge dataset and adaptive configs with adaptive taking precedence"""
       merged = dataset_config.copy()
       if 'training_params' in adaptive_config:
           merged['training_params'].update(adaptive_config['training_params'])
       if 'execution_flags' in adaptive_config:
           merged['execution_flags'] = adaptive_config['execution_flags']
       return merged

    def _create_default_config(self, folder_path: str, dataset_name: str) -> Dict:
        csv_path = os.path.join(folder_path, f"{dataset_name}.csv")
        df = pd.read_csv(csv_path, nrows=0)

        config = {
            "file_path": csv_path,
            "column_names": df.columns.tolist(),
            "separator": ",",
            "has_header": True,
            "target_column": df.columns[-1],
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
                "trials": 100,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "learning_rate": 0.1,
                "random_seed": 42,
                "epochs": 1000,
                "test_fraction": 0.2,
                "enable_adaptive": True,
                "Save_training_epochs": True,
                "training_save_path": f"training_data/{dataset_name}",
                "modelType": "Histogram",
                "minimum_training_accuracy": 0.95  # Added default value
            },
            "execution_flags": {
                "train": True,
                "train_only": False,
                "predict": True,
                "fresh_start": False,
                "use_previous_model": True
            }
        }

        config_path = os.path.join(folder_path, f"{dataset_name}.conf")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        return config

    def _validate_and_update_config(self, config: Dict, folder_path: str) -> Dict:
        required_fields = ['file_path', 'column_names', 'target_column']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return config

    def _merge_data(self, train_path: str, test_path: str) -> pd.DataFrame:
        train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(test_path, "test.csv"))
        return pd.concat([train_df, test_df], ignore_index=True)

    def _process_merged_data(self, df: pd.DataFrame, config: Dict):
        model = DBNN(config)
        return model.fit(df)

    def _process_split_data(self, train_path: str, test_path: str, config: Dict):
        model = DBNN(config)
        train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(test_path, "test.csv"))
        model.fit(train_df)
        return model.evaluate(test_df)

    def _process_single_file(self, file_path: str, config: Dict):
        model = DBNN(config)
        df = pd.read_csv(file_path)
        return model.fit(df)

    def _is_compressed(self, file_path: str) -> bool:
        return any(file_path.endswith(ext) for ext in self.compressed_extensions)

    def _decompress(self, file_path: str) -> str:
        extract_path = os.path.join('data', 'temp')
        os.makedirs(extract_path, exist_ok=True)

        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif file_path.endswith(('.tar', '.gz')):
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_path)
        elif file_path.endswith('.7z'):
            with py7zr.SevenZipFile(file_path, 'r') as sz_ref:
                sz_ref.extractall(extract_path)

        return extract_path

    def _download_from_uci(self, dataset_name: str) -> Optional[str]:
        """Download dataset from UCI repository"""
        folder_path = os.path.join('data', dataset_name.lower())
        os.makedirs(folder_path, exist_ok=True)

        save_path = os.path.join(folder_path, f"{dataset_name.lower()}.csv")

        # Try different UCI repository URL patterns
        url_patterns = [
            f"{self.base_url}/{dataset_name}/{dataset_name}.data",
            f"{self.base_url}/{dataset_name.lower()}/{dataset_name.lower()}.data",
            f"{self.base_url}/{dataset_name}/{dataset_name}.csv",
            f"{self.base_url}/{dataset_name.lower()}/{dataset_name.lower()}.csv"
        ]

        for url in url_patterns:
            try:
                print(f"Trying URL: {url}")
                response = requests.get(url)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Successfully downloaded to {save_path}")
                    return save_path
            except Exception as e:
                self.debug.log(f"Failed to download from {url}: {str(e)}")
                continue

        return None

    @staticmethod
    def generate_test_datasets():
        """Generate synthetic test datasets"""
        datasets = {
            'xor.csv': [
                'x1,x2,target\n',
                *['0,0,0\n0,1,1\n1,0,1\n1,1,0\n' * 3]
            ],
            'xor3d.csv': [
                'x1,x2,x3,target\n',
                *['0,0,0,0\n0,0,1,1\n0,1,0,1\n0,1,1,1\n1,0,0,1\n1,0,1,1\n1,1,0,1\n1,1,1,0\n' * 3]
            ]
        }

        for filename, content in datasets.items():
            with open(filename, 'w') as f:
                f.writelines(content)
#------------------------------------------------------Invertable DBNN -------------------------
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from tqdm import tqdm

class InvertibleDBNN(torch.nn.Module):
    """Enhanced Invertible Deep Bayesian Neural Network implementation with proper gradient tracking"""

    def __init__(self,
                 forward_model: 'DBNN',
                 feature_dims: int,
                 reconstruction_weight: float = 0.5,
                 feedback_strength: float = 0.3,
                 debug: bool = False):
        """
        Initialize the invertible DBNN.

        Args:
            forward_model: The forward DBNN model
            feature_dims: Number of input feature dimensions
            reconstruction_weight: Weight for reconstruction loss (0-1)
            feedback_strength: Strength of reconstruction feedback (0-1)
            debug: Enable debug logging
        """
        super(InvertibleDBNN, self).__init__()
        self.forward_model = forward_model
        self.device = forward_model.device
        self.feature_dims = feature_dims
        self.reconstruction_weight = reconstruction_weight
        self.feedback_strength = feedback_strength

        # Enable logging if debug is True
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

        # Initialize model components
        self.n_classes = len(self.forward_model.label_encoder.classes_)
        self.inverse_likelihood_params = None
        self.inverse_feature_pairs = None

        # Feature scaling parameters as buffers
        self.register_buffer('min_vals', None)
        self.register_buffer('max_vals', None)
        self.register_buffer('scale_factors', None)

        # Metrics tracking
        self.metrics = {
            'reconstruction_errors': [],
            'forward_errors': [],
            'total_losses': [],
            'accuracies': []
        }

        # Initialize all components
        self._initialize_inverse_components()

    def save_inverse_model(self, custom_path: str = None) -> bool:
        try:
            save_dir = custom_path or os.path.join('Model', f'Best_inverse_{self.forward_model.dataset_name}')
            os.makedirs(save_dir, exist_ok=True)

            # Save model state
            model_state = {
                'weight_linear': self.weight_linear.data,
                'weight_nonlinear': self.weight_nonlinear.data,
                'bias_linear': self.bias_linear.data,
                'bias_nonlinear': self.bias_nonlinear.data,
                'feature_attention': self.feature_attention.data,
                'layer_norm': self.layer_norm.state_dict(),
                'metrics': self.metrics,
                'feature_dims': self.feature_dims,
                'n_classes': self.n_classes,
                'reconstruction_weight': self.reconstruction_weight,
                'feedback_strength': self.feedback_strength
            }

            # Save scale parameters if they exist
            for param in ['min_vals', 'max_vals', 'scale_factors', 'inverse_feature_pairs']:
                if hasattr(self, param):
                    model_state[param] = getattr(self, param)

            model_path = os.path.join(save_dir, 'inverse_model.pt')
            torch.save(model_state, model_path)

            # Save config
            config = {
                'feature_dims': self.feature_dims,
                'reconstruction_weight': float(self.reconstruction_weight),
                'feedback_strength': float(self.feedback_strength),
                'n_classes': int(self.n_classes),
                'device': str(self.device)
            }

            config_path = os.path.join(save_dir, 'inverse_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            print(f"Saved inverse model to {save_dir}")
            return True

        except Exception as e:
            print(f"Error saving inverse model: {str(e)}")
            traceback.print_exc()
            return False

    def load_inverse_model(self, custom_path: str = None) -> bool:
       try:
           load_dir = custom_path or os.path.join('Model', f'Best_inverse_{self.forward_model.dataset_name}')
           model_path = os.path.join(load_dir, 'inverse_model.pt')
           config_path = os.path.join(load_dir, 'inverse_config.json')

           if not (os.path.exists(model_path) and os.path.exists(config_path)):
               print(f"No saved inverse model found at {load_dir}")
               return False

           model_state = torch.load(model_path, map_location=self.device, weights_only=True)

           with open(config_path, 'r') as f:
               config = json.load(f)

           if config['feature_dims'] != self.feature_dims or config['n_classes'] != self.n_classes:
               raise ValueError("Model architecture mismatch")

           # Load parameters
           self.weight_linear.data = model_state['weight_linear']
           self.weight_nonlinear.data = model_state['weight_nonlinear']
           self.bias_linear.data = model_state['bias_linear']
           self.bias_nonlinear.data = model_state['bias_nonlinear']
           self.feature_attention.data = model_state['feature_attention']
           self.layer_norm.load_state_dict(model_state['layer_norm'])

           # Safely update or register buffers
           for param in ['min_vals', 'max_vals', 'scale_factors', 'inverse_feature_pairs']:
               if param in model_state:
                   buffer_data = model_state[param]
                   if buffer_data is not None:
                       if hasattr(self, param) and getattr(self, param) is not None:
                           getattr(self, param).copy_(buffer_data)
                       else:
                           self.register_buffer(param, buffer_data)

           # Restore other attributes
           self.metrics = model_state.get('metrics', {})
           self.reconstruction_weight = model_state.get('reconstruction_weight', 0.5)
           self.feedback_strength = model_state.get('feedback_strength', 0.3)

           print(f"Loaded inverse model from {load_dir}")
           return True

       except Exception as e:
           print(f"Error loading inverse model: {str(e)}")
           traceback.print_exc()
           return False

    def _initialize_inverse_components(self):
        """Initialize inverse model parameters with proper buffer handling"""
        try:
            # Initialize feature pairs
            class_indices = torch.arange(self.n_classes, device=self.device)
            feature_indices = torch.arange(self.feature_dims, device=self.device)
            feature_pairs = torch.cartesian_prod(class_indices, feature_indices)

            # Safely register buffer
            if not hasattr(self, 'inverse_feature_pairs'):
                self.register_buffer('inverse_feature_pairs', feature_pairs)
            else:
                self.inverse_feature_pairs = feature_pairs

            # Number of pairs
            n_pairs = len(feature_pairs)

            # Initialize weights as nn.Parameters
            self.weight_linear = torch.nn.Parameter(
                torch.empty((n_pairs, self.feature_dims), device=self.device),
                requires_grad=True
            )
            self.weight_nonlinear = torch.nn.Parameter(
                torch.empty((n_pairs, self.feature_dims), device=self.device),
                requires_grad=True
            )

            # Initialize with proper scaling
            torch.nn.init.xavier_uniform_(self.weight_linear)
            torch.nn.init.kaiming_normal_(self.weight_nonlinear)

            # Initialize biases as nn.Parameters
            self.bias_linear = torch.nn.Parameter(
                torch.zeros(self.feature_dims, device=self.device),
                requires_grad=True
            )
            self.bias_nonlinear = torch.nn.Parameter(
                torch.zeros(self.feature_dims, device=self.device),
                requires_grad=True
            )

            # Initialize layer normalization
            self.layer_norm = torch.nn.LayerNorm(self.feature_dims).to(self.device)

            # Initialize feature attention
            self.feature_attention = torch.nn.Parameter(
                torch.ones(self.feature_dims, device=self.device),
                requires_grad=True
            )

            # Safely register scaling buffers
            for name in ['min_vals', 'max_vals', 'scale_factors']:
                if not hasattr(self, name):
                    self.register_buffer(name, None)

            if self.debug:
                self.logger.debug(f"Initialized inverse components:")
                self.logger.debug(f"- Feature pairs shape: {self.inverse_feature_pairs.shape}")
                self.logger.debug(f"- Linear weights shape: {self.weight_linear.shape}")
                self.logger.debug(f"- Nonlinear weights shape: {self.weight_nonlinear.shape}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize inverse components: {str(e)}")
    def _compute_feature_scaling(self, features: torch.Tensor):
        """Compute feature scaling parameters for consistent reconstruction"""
        with torch.no_grad():
            self.min_vals = features.min(dim=0)[0]
            self.max_vals = features.max(dim=0)[0]
            self.scale_factors = self.max_vals - self.min_vals
            self.scale_factors[self.scale_factors == 0] = 1.0

    def _scale_features(self, features: torch.Tensor) -> torch.Tensor:
        """Scale features to [0,1] range"""
        return (features - self.min_vals) / self.scale_factors

    def _unscale_features(self, scaled_features: torch.Tensor) -> torch.Tensor:
        """Convert scaled features back to original range"""
        return (scaled_features * self.scale_factors) + self.min_vals

    def _compute_inverse_posterior(self, class_probs: torch.Tensor) -> torch.Tensor:
        """Enhanced inverse posterior computation with improved stability"""
        batch_size = class_probs.shape[0]
        class_probs = class_probs.to(dtype=self.weight_linear.dtype)

        reconstructed_features = torch.zeros(
            (batch_size, self.feature_dims),
            device=self.device,
            dtype=self.weight_linear.dtype
        )

        # Apply attention mechanism
        attention_weights = torch.softmax(self.feature_attention, dim=0)

        # Compute linear and nonlinear transformations
        linear_features = torch.zeros_like(reconstructed_features)
        nonlinear_features = torch.zeros_like(reconstructed_features)

        for feat_idx in range(self.feature_dims):
            # Get relevant pairs for this feature
            relevant_pairs = torch.where(self.inverse_feature_pairs[:, 1] == feat_idx)[0]

            # Get class contributions
            class_contributions = class_probs[:, self.inverse_feature_pairs[relevant_pairs, 0]]

            # Linear transformation
            linear_weights = self.weight_linear[relevant_pairs, feat_idx]
            linear_features[:, feat_idx] = torch.mm(
                class_contributions,
                linear_weights.unsqueeze(1)
            ).squeeze()

            # Nonlinear transformation with tanh activation
            nonlinear_weights = self.weight_nonlinear[relevant_pairs, feat_idx]
            nonlinear_features[:, feat_idx] = torch.tanh(torch.mm(
                class_contributions,
                nonlinear_weights.unsqueeze(1)
            ).squeeze())

        # Combine transformations with attention
        reconstructed_features = (
            attention_weights * linear_features +
            (1 - attention_weights) * nonlinear_features
        )

        # Add biases
        reconstructed_features += self.bias_linear + self.bias_nonlinear

        # Apply layer normalization
        reconstructed_features = self.layer_norm(reconstructed_features)

        return reconstructed_features

    def _compute_reconstruction_loss(self,
                                   original_features: torch.Tensor,
                                   reconstructed_features: torch.Tensor,
                                   class_probs: torch.Tensor,
                                   reduction: str = 'mean') -> torch.Tensor:
        """Enhanced reconstruction loss with multiple components"""
        # Scale features
        orig_scaled = self._scale_features(original_features)
        recon_scaled = self._scale_features(reconstructed_features)

        # MSE loss with feature-wise weighting
        mse_loss = torch.mean((orig_scaled - recon_scaled) ** 2, dim=1)

        # Feature correlation loss
        orig_centered = orig_scaled - orig_scaled.mean(dim=0, keepdim=True)
        recon_centered = recon_scaled - recon_scaled.mean(dim=0, keepdim=True)

        corr_loss = -torch.sum(
            orig_centered * recon_centered, dim=1
        ) / (torch.norm(orig_centered, dim=1) * torch.norm(recon_centered, dim=1) + 1e-8)

        # Distribution matching loss using KL divergence
        orig_dist = torch.distributions.Normal(
            orig_scaled.mean(dim=0),
            orig_scaled.std(dim=0) + 1e-8
        )
        recon_dist = torch.distributions.Normal(
            recon_scaled.mean(dim=0),
            recon_scaled.std(dim=0) + 1e-8
        )
        dist_loss = torch.distributions.kl_divergence(orig_dist, recon_dist).mean()

        # Combine losses with learned weights
        combined_loss = (
            mse_loss +
            0.1 * corr_loss +
            0.01 * dist_loss
        )

        if reduction == 'mean':
            return combined_loss.mean()
        return combined_loss

    def train(self, X_train, y_train, X_test, y_test, batch_size=32):
        """Complete training with optimized test evaluation"""
        print("\nStarting training...")
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Initialize tracking metrics
        error_rates = []
        best_train_accuracy = 0.0
        best_test_accuracy = 0.0
        patience_counter = 0
        plateau_counter = 0
        min_improvement = 0.001
        patience = 5 if self.in_adaptive_fit else 100
        max_plateau = 5
        prev_accuracy = 0.0

        # Main training loop with progress tracking
        with tqdm(total=self.max_epochs, desc="Training epochs") as epoch_pbar:
            for epoch in range(self.max_epochs):
                # Train on all batches
                failed_cases = []
                n_errors = 0

                # Process training batches
                with tqdm(total=n_batches, desc=f"Training batches", leave=False) as batch_pbar:
                    for i in range(0, n_samples, batch_size):
                        batch_end = min(i + batch_size, n_samples)
                        batch_X = X_train[i:batch_end]
                        batch_y = y_train[i:batch_end]

                        # Forward pass and error collection
                        posteriors = self._compute_batch_posterior(batch_X)[0]
                        predictions = torch.argmax(posteriors, dim=1)
                        errors = (predictions != batch_y)
                        n_errors += errors.sum().item()

                        # Collect failed cases for weight updates
                        if errors.any():
                            fail_idx = torch.where(errors)[0]
                            for idx in fail_idx:
                                failed_cases.append((
                                    batch_X[idx],
                                    batch_y[idx].item(),
                                    posteriors[idx].cpu().numpy()
                                ))
                        batch_pbar.update(1)

                # Update weights after processing all batches
                if failed_cases:
                    self._update_priors_parallel(failed_cases, batch_size)

                # Calculate training metrics
                train_error_rate = n_errors / n_samples
                train_accuracy = 1 - train_error_rate
                error_rates.append(train_error_rate)

                # Evaluate on test set once per epoch
                if X_test is not None and y_test is not None:
                    test_predictions = self.predict(X_test, batch_size=batch_size)
                    test_accuracy = (test_predictions == y_test.cpu()).float().mean().item()

                    # Print confusion matrix only for best test performance
                    if test_accuracy > best_test_accuracy:
                        best_test_accuracy = test_accuracy
                        print("\nTest Set Performance:")
                        y_test_labels = self.label_encoder.inverse_transform(y_test.cpu().numpy())
                        test_pred_labels = self.label_encoder.inverse_transform(test_predictions.cpu().numpy())
                        self.print_colored_confusion_matrix(y_test_labels, test_pred_labels)

                # Update progress bar with metrics
                epoch_pbar.set_postfix({
                    'train_acc': f"{train_accuracy:.4f}",
                    'best_train': f"{best_train_accuracy:.4f}",
                    'test_acc': f"{test_accuracy:.4f}",
                    'best_test': f"{best_test_accuracy:.4f}"
                })
                epoch_pbar.update(1)

                # Check improvement and update tracking
                accuracy_improvement = train_accuracy - prev_accuracy
                if accuracy_improvement <= min_improvement:
                    plateau_counter += 1
                else:
                    plateau_counter = 0

                if train_accuracy > best_train_accuracy + min_improvement:
                    best_train_accuracy = train_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Save best model
                if train_error_rate <= self.best_error:
                    self.best_error = train_error_rate
                    self.best_W = self.current_W.clone()
                    self._save_best_weights()

                # Early stopping checks
                if train_accuracy == 1.0:
                    print("\nReached 100% training accuracy")
                    break

                if patience_counter >= patience:
                    print(f"\nNo improvement for {patience} epochs")
                    break

                if plateau_counter >= max_plateau:
                    print(f"\nAccuracy plateaued for {max_plateau} epochs")
                    break

                prev_accuracy = train_accuracy

            self._save_model_components()
            return self.current_W.cpu(), error_rates

    def reconstruct_features(self, class_probs: torch.Tensor) -> torch.Tensor:
        """Reconstruct input features from class probabilities with dtype handling"""
        with torch.no_grad():
            # Ensure consistent dtype
            class_probs = class_probs.to(dtype=torch.float32)
            reconstructed = self._compute_inverse_posterior(class_probs)

            if hasattr(self, 'min_vals') and self.min_vals is not None:
                reconstructed = self._unscale_features(reconstructed)
                # Ensure output matches input dtype
                return reconstructed.to(dtype=self.weight_linear.dtype)
            return reconstructed.to(dtype=self.weight_linear.dtype)

    def evaluate(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            features: Input features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Forward pass
            if self.forward_model.model_type == "Histogram":
                class_probs, _ = self.forward_model._compute_batch_posterior(features)
            else:
                class_probs, _ = self.forward_model._compute_batch_posterior_std(features)

            # Get predictions and convert to numpy
            predictions = torch.argmax(class_probs, dim=1)
            predictions_np = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Convert to original class labels
            true_labels = self.forward_model.label_encoder.inverse_transform(labels_np)
            pred_labels = self.forward_model.label_encoder.inverse_transform(predictions_np)

            # Compute classification report and confusion matrix
            from sklearn.metrics import classification_report, confusion_matrix
            class_report = classification_report(true_labels, pred_labels)
            conf_matrix = confusion_matrix(true_labels, pred_labels)

            # Calculate test accuracy
            test_accuracy = (predictions == labels).float().mean().item()

            # Get training error rates from metrics history
            error_rates = self.metrics.get('forward_errors', [])
            if not error_rates and hasattr(self.forward_model, 'error_rates'):
                error_rates = self.forward_model.error_rates

            # Get reconstruction metrics
            reconstructed_features = self.reconstruct_features(class_probs)
            reconstruction_loss = self._compute_reconstruction_loss(
                features, reconstructed_features, reduction='mean'
            ).item()

            # Prepare results dictionary matching expected format
            results = {
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'error_rates': error_rates,
                'test_accuracy': test_accuracy,
                'reconstruction_loss': reconstruction_loss
            }

            # Format results as string for display/saving
            formatted_output = f"Results for Dataset: {self.forward_model.dataset_name}\n\n"
            formatted_output += f"Classification Report:\n{class_report}\n\n"
            formatted_output += "Confusion Matrix:\n"
            formatted_output += "\n".join(["\t".join(map(str, row)) for row in conf_matrix])
            formatted_output += "\n\nError Rates:\n"

            if error_rates:
                formatted_output += "\n".join([f"Epoch {i+1}: {rate:.4f}" for i, rate in enumerate(error_rates)])
            else:
                formatted_output += "N/A"

            formatted_output += f"\n\nTest Accuracy: {test_accuracy:.4f}\n"

            # Store formatted output in results
            results['formatted_output'] = formatted_output

            return results

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
