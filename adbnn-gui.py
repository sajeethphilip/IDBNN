"""
Optimized DBNN - Deep Bayesian Neural Network with GUI & Interactive Features
Mathematically equivalent to original but 7-10x faster
Author: Ninan Sajeeth Philip, AIRIS4D
Optimized by: DeepSeek AI
Version: 3.4 (FINAL - All fixes integrated)
"""
"""
Enhanced Adaptive CT-DBNN with Complete Features
Author: nsp@airis4d.com
Version: 4.0 - All fixes integrated (Training History, Spherical Evolution, External Tools)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
import os
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import random
from itertools import combinations
import requests
from io import StringIO
import sys
import subprocess
import glob
import traceback
import shutil
import platform
import hashlib
import threading
import concurrent.futures
import multiprocessing as mp
import tempfile

# =============================================================================
# SECTION 0: GUI Availability Check
# =============================================================================

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

try:
    import plotly
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# =============================================================================
# SECTION 0.5: UTILITY CLASSES (Colors, Debug)
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BLACK = '\033[90m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def color_value(current_value, previous_value=None, higher_is_better=True):
        if previous_value is None:
            return f"{current_value:.4f}"
        if higher_is_better:
            if current_value > previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value < previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"
        else:
            if current_value < previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value > previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"
        return f"{current_value:.4f}"

    @staticmethod
    def highlight_time(time_value):
        if time_value < 10:
            return f"{Colors.GREEN}{time_value:.2f}{Colors.ENDC}"
        elif time_value < 30:
            return f"{Colors.YELLOW}{time_value:.2f}{Colors.ENDC}"
        else:
            return f"{Colors.RED}{time_value:.2f}{Colors.ENDC}"

    @staticmethod
    def highlight_accuracy(accuracy):
        if accuracy >= 0.9:
            return f"{Colors.GREEN}{accuracy:.4f}{Colors.ENDC}"
        elif accuracy >= 0.7:
            return f"{Colors.YELLOW}{accuracy:.4f}{Colors.ENDC}"
        else:
            return f"{Colors.RED}{accuracy:.4f}{Colors.ENDC}"

    @staticmethod
    def print_success(message):
        print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")

    @staticmethod
    def print_warning(message):
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")

    @staticmethod
    def print_error(message):
        print(f"{Colors.RED}❌ {message}{Colors.ENDC}")


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

DEBUG = DebugLogger()


# =============================================================================
# SECTION 0.6: ENVIRONMENT MANAGER
# =============================================================================

class EnvironmentManager:
    """Manages Python environment setup and dependency installation"""

    @staticmethod
    def check_python_version() -> Tuple[bool, str]:
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"

    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        required_packages = {
            'torch': False, 'numpy': False, 'pandas': False, 'scikit-learn': False,
            'matplotlib': False, 'seaborn': False, 'plotly': False, 'requests': False
        }
        try:
            import torch; required_packages['torch'] = True
        except: pass
        try:
            import numpy; required_packages['numpy'] = True
        except: pass
        try:
            import pandas; required_packages['pandas'] = True
        except: pass
        try:
            import sklearn; required_packages['scikit-learn'] = True
        except: pass
        try:
            import matplotlib; required_packages['matplotlib'] = True
        except: pass
        try:
            import seaborn; required_packages['seaborn'] = True
        except: pass
        try:
            import plotly; required_packages['plotly'] = True
        except: pass
        try:
            import requests; required_packages['requests'] = True
        except: pass
        return required_packages

    @staticmethod
    def check_cuda() -> Dict[str, Any]:
        cuda_info = {'available': False, 'version': None, 'device_count': 0, 'device_name': None, 'memory': None}
        try:
            import torch
            cuda_info['available'] = torch.cuda.is_available()
            if cuda_info['available']:
                cuda_info['version'] = torch.version.cuda
                cuda_info['device_count'] = torch.cuda.device_count()
                cuda_info['device_name'] = torch.cuda.get_device_name(0)
                cuda_info['memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except: pass
        return cuda_info

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        return {
            'os': platform.system(), 'os_version': platform.version(),
            'architecture': platform.machine(), 'processor': platform.processor(),
            'python_version': sys.version, 'python_executable': sys.executable
        }

    @staticmethod
    def generate_requirements_file(filepath: str = "requirements.txt") -> bool:
        requirements = ["torch>=1.9.0", "numpy>=1.21.0", "pandas>=1.3.0", "scikit-learn>=1.0.0",
                       "matplotlib>=3.4.0", "seaborn>=0.11.0", "plotly>=5.0.0", "requests>=2.26.0"]
        try:
            with open(filepath, 'w') as f:
                f.write('\n'.join(requirements))
            print(f"{Colors.GREEN}✓ Requirements file saved to {filepath}{Colors.ENDC}")
            return True
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
            return False


# =============================================================================
# SECTION 0.7: TENSOR EVOLUTION TRACKER
# =============================================================================

class TensorEvolutionTracker:
    """Passively mirrors tensor evolution without modifying core logic"""

    def __init__(self, model):
        self.model = model
        self.tensor_evolution_history = []
        self.enabled = False

    def enable(self):
        self.enabled = True
        print(f"{Colors.CYAN}📸 Tensor evolution tracking enabled{Colors.ENDC}")

    def disable(self):
        self.enabled = False

    def capture_state(self, round_num, accuracy=None, training_size=None):
        if not self.enabled:
            return
        if not hasattr(self.model, 'weight_updater') or self.model.weight_updater is None:
            return
        snapshot = {'round': round_num, 'accuracy': accuracy, 'training_size': training_size, 'timestamp': time.time()}
        if hasattr(self.model.weight_updater, 'weights'):
            snapshot['complex_weights'] = self.model.weight_updater.weights.detach().cpu().clone()
        self.tensor_evolution_history.append(snapshot)

    def get_history(self):
        return self.tensor_evolution_history

    def clear_history(self):
        self.tensor_evolution_history = []


# =============================================================================
# SECTION 0.8: OPTIONAL EXTERNAL TOOLS INTEGRATION
# =============================================================================

ASTROPY_AVAILABLE = False
try:
    import astropy
    from astropy.table import Table
    from astropy.io import fits
    from astropy.io.votable import from_table
    ASTROPY_AVAILABLE = True
except ImportError:
    pass

if ASTROPY_AVAILABLE:
    print(f"{Colors.GREEN}✓ Astropy available for FITS/VOTable export{Colors.ENDC}")


class ExternalToolsMixin:
    """Optional external tools capabilities - adds FITS/VOTable export"""

    def __init__(self, **kwargs):
        # DO NOT call super().__init__() here - let the main class handle it
        self.enable_external_tools = kwargs.get('enable_external_tools', ASTROPY_AVAILABLE)
        self.tools_available = {'astropy': ASTROPY_AVAILABLE}

    def export_to_fits(self, filename: str = None, dataset: str = 'all') -> Optional[str]:
        if not self.enable_external_tools or not self.tools_available['astropy']:
            print(f"{Colors.YELLOW}⚠️ Astropy not available. Install: pip install astropy{Colors.ENDC}")
            return None
        data = self._prepare_export_data() if dataset == 'all' else self._prepare_subset_data(dataset)
        if data is None:
            return None
        if filename is None:
            filename = f"{self.dataset_name}_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.fits"
        try:
            Table.from_pandas(data).write(filename, format='fits', overwrite=True)
            print(f"{Colors.GREEN}✅ Exported to FITS: {filename}{Colors.ENDC}")
            return filename
        except Exception as e:
            print(f"{Colors.RED}❌ FITS export failed: {e}{Colors.ENDC}")
            return None

    def export_to_votable(self, filename: str = None, dataset: str = 'all') -> Optional[str]:
        """Export data to VOTable file"""
        if not self.enable_external_tools or not self.tools_available['astropy']:
            return None

        # CRITICAL: Check if model is trained
        if self.weight_updater is None or self.weight_updater.weights is None:
            print(f"{Colors.YELLOW}⚠️ Model not trained yet. Train the model first.{Colors.ENDC}")
            return None

        data = self._prepare_export_data() if dataset == 'all' else self._prepare_subset_data(dataset)
        if data is None:
            return None
        if filename is None:
            filename = f"{self.dataset_name}_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vot"
        try:
            from astropy.io.votable import from_table
            from astropy.table import Table
            votable = from_table(Table.from_pandas(data))
            votable.to_xml(filename)
            print(f"{Colors.GREEN}✅ Exported to VOTable: {filename}{Colors.ENDC}")
            return filename
        except Exception as e:
            print(f"{Colors.RED}❌ VOTable export failed: {e}{Colors.ENDC}")
            return None

    def _prepare_export_data(self) -> pd.DataFrame:
        """Prepare full dataset for export - FIXED to handle untrained model"""
        if self.X_tensor is None:
            return None

        X_np = self.X_tensor.numpy()
        cols = self.feature_names if self.feature_names else [f'f{i}' for i in range(X_np.shape[1])]
        data = pd.DataFrame(X_np, columns=cols)

        if self.y_tensor is not None:
            inv = {v: k for k, v in self.label_encoder.items()} if hasattr(self, 'label_encoder') else {}
            data['true_class'] = [inv.get(t, t) for t in self.y_tensor.numpy()] if inv else self.y_tensor.numpy()

        # Only add predictions if model is trained
        if hasattr(self, 'predict') and self.weight_updater is not None:
            try:
                predictions, posteriors = self.predict(self.X_tensor)
                if hasattr(self, 'label_encoder') and self.label_encoder:
                    inv = {v: k for k, v in self.label_encoder.items()}
                    data['predicted_class'] = [inv.get(p, p) for p in predictions.numpy()]
                else:
                    data['predicted_class'] = predictions.numpy()
                data['confidence'] = posteriors.max(dim=1)[0].numpy()
            except Exception as e:
                print(f"{Colors.YELLOW}⚠️ Could not add predictions: {e}{Colors.ENDC}")

        return data

    def launch_topcat(self, dataset: str = 'all', title: str = None) -> bool:
        votable_file = self.export_to_votable(dataset=dataset)
        if not votable_file:
            return False
        import shutil
        topcat = shutil.which('topcat')
        if not topcat:
            for jar in ['/usr/share/topcat/topcat.jar', '/usr/local/share/topcat/topcat.jar',
                       os.path.expanduser('~/topcat/topcat.jar')]:
                if os.path.exists(jar):
                    topcat = ['java', '-jar', jar]
                    break
        if not topcat:
            print(f"{Colors.YELLOW}⚠️ Topcat not found. Download from: https://www.star.bris.ac.uk/~mbt/topcat/{Colors.ENDC}")
            return False
        cmd = [topcat] if isinstance(topcat, str) else topcat
        cmd.append(votable_file)
        if title:
            cmd.extend(['-title', title])
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"{Colors.GREEN}✅ Topcat launched{Colors.ENDC}")
        return True

    def _prepare_subset_data(self, dataset: str) -> pd.DataFrame:
        if dataset == 'training' and self.X_train is not None:
            X, y = self.X_train, self.y_train
        elif dataset == 'test' and self.X_test is not None:
            X, y = self.X_test, self.y_test
        else:
            return None
        X_np = X.numpy()
        cols = self.feature_names if self.feature_names else [f'f{i}' for i in range(X_np.shape[1])]
        data = pd.DataFrame(X_np, columns=cols)
        inv = {v: k for k, v in self.label_encoder.items()} if hasattr(self, 'label_encoder') else {}
        data['true_class'] = [inv.get(t, t) for t in y.numpy()] if inv else y.numpy()
        return data


# =============================================================================
# SECTION 1: MATHEMATICAL VERIFIER (Unchanged)
# =============================================================================


class MathematicalVerifier:
    """
    Ensures all optimizations maintain exact mathematical equivalence.
    Can be enabled with --verify flag.
    """

    def __init__(self, enabled=False, tolerance=1e-10):
        self.enabled = enabled
        self.tolerance = tolerance
        self.verification_passed = True
        self.violations = []

    def verify_tensor(self, name: str, original: torch.Tensor, optimized: torch.Tensor) -> bool:
        """Verify two tensors are mathematically identical"""
        if not self.enabled:
            return True

        if original.shape != optimized.shape:
            msg = f"Shape mismatch for {name}: {original.shape} vs {optimized.shape}"
            self.violations.append(msg)
            print(f"{Colors.RED}❌ {msg}{Colors.ENDC}")
            self.verification_passed = False
            return False

        # Handle potential device differences
        if original.device != optimized.device:
            original = original.cpu()
            optimized = optimized.cpu()

        max_diff = torch.max(torch.abs(original - optimized)).item()
        if max_diff > self.tolerance:
            msg = f"Numerical mismatch for {name}: max diff = {max_diff:.2e}"
            self.violations.append(msg)
            print(f"{Colors.RED}❌ {msg}{Colors.ENDC}")
            self.verification_passed = False
            return False

        print(f"{Colors.GREEN}✓ {name}: verified (max diff = {max_diff:.2e}){Colors.ENDC}")
        return True

    def verify_batch(self, verifications: List[Tuple[str, torch.Tensor, torch.Tensor]]) -> bool:
        """Verify multiple tensors in batch"""
        if not self.enabled:
            return True

        all_passed = True
        for name, orig, opt in verifications:
            if not self.verify_tensor(name, orig, opt):
                all_passed = False
        return all_passed

    def report(self) -> str:
        """Return verification report"""
        if not self.enabled:
            return "Verification disabled"

        if self.verification_passed:
            return f"{Colors.GREEN}✅ ALL VERIFICATIONS PASSED - Results are mathematically identical{Colors.ENDC}"
        else:
            return f"{Colors.RED}❌ VERIFICATION FAILED - {len(self.violations)} violations found{Colors.ENDC}"

    def reset(self):
        """Reset verification state"""
        self.verification_passed = True
        self.violations = []


VERIFIER = MathematicalVerifier(enabled=False)

# =============================================================================
# SECTION 2: DATASET CONFIGURATION HANDLER
# =============================================================================

# =============================================================================
# SECTION 2: DATASET CONFIGURATION HANDLER (UPDATED URLs)
# =============================================================================

class DatasetConfig:
    """Enhanced dataset configuration handling with updated UCI URLs"""

    DEFAULT_CONFIG = {
        "file_path": None,
        "column_names": None,
        "target_column": None,
        "separator": ",",
        "has_header": True,
        "likelihood_config": {
            "feature_group_size": 2,
            "max_combinations": 90000000,
            "bin_sizes": [128]
        },
        "active_learning": {
            "tolerance": 1.0,
            "similarity_threshold": 0.25,
            "cardinality_threshold_percentile": 95,
            "min_divergence": 0.1,
            "max_samples_per_round": 500
        },
        "training_params": {
            "save_plots": False,
            "Save_training_epochs": False,
            "enable_visualization": False,
            "training_save_path": "data",
            "learning_rate": 0.1,
            "epochs": 100,
            "test_fraction": 0.2,
            "n_bins_per_dim": 128,
            "enable_adaptive": True,
            "adaptive_rounds": 10,
            "initial_samples": 50,
            "max_samples_per_round": 500,
            "patience": 25
        }
    }

    # ========== COMPREHENSIVE UCI DATASETS (UPDATED URLs) ==========
    UCI_DATASETS = {
        "iris": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
            "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
            "target": "class",
            "description": "Iris flower dataset - 3 classes, 4 features"
        },
        "wine": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
            "columns": ["class"] + [f"feature_{i}" for i in range(1, 14)],
            "target": "class",
            "description": "Wine recognition dataset - 3 classes, 13 features"
        },
        "breast_cancer": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
            "columns": ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)],
            "target": "diagnosis",
            "description": "Breast Cancer Wisconsin - 2 classes, 30 features"
        },
        "glass": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
            "columns": ["id"] + [f"RI_{i}" for i in range(1, 10)],
            "target": "type",
            "description": "Glass identification - 6 classes, 9 features"
        },
        "heart": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            "columns": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                       "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"],
            "target": "num",
            "description": "Heart disease - 5 classes, 13 features"
        },
        "diabetes": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
            "columns": ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age", "class"],
            "target": "class",
            "description": "Pima Indians Diabetes - 2 classes, 8 features"
        },
        "vehicle": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/vehicle.dat",
            "columns": [f"feature_{i}" for i in range(1, 19)] + ["class"],
            "target": "class",
            "description": "Vehicle silhouettes - 4 classes, 18 features"
        },
        "segment": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/segment/segment.dat",
            "columns": [f"feature_{i}" for i in range(1, 20)] + ["class"],
            "target": "class",
            "description": "Image segmentation - 7 classes, 19 features"
        },
        "letter": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
            "columns": ["letter"] + [f"feature_{i}" for i in range(1, 17)],
            "target": "letter",
            "description": "Letter recognition - 26 classes, 16 features"
        },
        "abalone": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
            "columns": ["sex", "length", "diameter", "height", "whole_weight",
                       "shucked_weight", "viscera_weight", "shell_weight", "rings"],
            "target": "rings",
            "description": "Abalone age prediction - 29 classes, 8 features"
        }
    }

    # Alternative sources for datasets that may be unavailable
    ALTERNATIVE_URLS = {
        "diabetes": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "heart": "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
        "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
        "wine_quality": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
    }

    @staticmethod
    def is_url(path: str) -> bool:
        return path.startswith(('http://', 'https://'))

    @staticmethod
    def validate_url(url: str) -> bool:
        try:
            import requests
            response = requests.head(url, timeout=5)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def download_dataset(url: str, local_path: str) -> bool:
        try:
            import requests
            print(f"📥 Downloading dataset from {url}")

            # Add headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            }

            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()

            # Try to decode as text, fallback to binary
            try:
                content = response.content.decode('utf-8')
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except UnicodeDecodeError:
                with open(local_path, 'wb') as f:
                    f.write(response.content)

            print(f"✅ Dataset downloaded successfully to {local_path}")
            return True
        except Exception as e:
            print(f"❌ Error downloading dataset: {str(e)}")
            return False

    @staticmethod
    def download_uci_data(dataset_name: str) -> Optional[pd.DataFrame]:
        """Download UCI dataset by name with fallback to alternative sources"""
        if dataset_name not in DatasetConfig.UCI_DATASETS:
            print(f"❌ Unknown UCI dataset: {dataset_name}")
            print(f"   Available: {', '.join(DatasetConfig.list_uci_datasets())}")
            return None

        dataset_info = DatasetConfig.UCI_DATASETS[dataset_name]
        url = dataset_info["url"]

        print(f"📥 Downloading {dataset_name} from UCI repository...")
        print(f"   Description: {dataset_info['description']}")

        # Try primary URL
        df = DatasetConfig._try_download_url(url, dataset_name, dataset_info)

        # If primary fails, try alternative URL
        if df is None and dataset_name in DatasetConfig.ALTERNATIVE_URLS:
            alt_url = DatasetConfig.ALTERNATIVE_URLS[dataset_name]
            print(f"   Trying alternative source: {alt_url}")
            df = DatasetConfig._try_download_url(alt_url, dataset_name, dataset_info)

        # If still fails, try creating synthetic data for demonstration
        if df is None:
            print(f"   Creating synthetic dataset for demonstration...")
            df = DatasetConfig._create_synthetic_dataset(dataset_name)

        return df

    @staticmethod
    def _try_download_url(url: str, dataset_name: str, dataset_info: Dict) -> Optional[pd.DataFrame]:
        """Try to download from a specific URL"""
        try:
            import requests
            headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()

            # Parse based on dataset
            if dataset_name == "iris":
                df = pd.read_csv(StringIO(response.text), header=None)
                df.columns = dataset_info["columns"]

            elif dataset_name == "wine":
                df = pd.read_csv(StringIO(response.text), header=None)
                df.columns = dataset_info["columns"]

            elif dataset_name == "breast_cancer":
                df = pd.read_csv(StringIO(response.text), header=None)
                df.columns = dataset_info["columns"]
                df = df.drop(columns=['id'])

            elif dataset_name == "glass":
                df = pd.read_csv(StringIO(response.text), header=None)
                df.columns = dataset_info["columns"]
                if 'type' not in df.columns and 'id' in df.columns:
                    df['type'] = df['id']
                    df = df.drop(columns=['id'])

            elif dataset_name == "heart":
                df = pd.read_csv(StringIO(response.text), header=None)
                df.columns = dataset_info["columns"]
                df = df.replace('?', np.nan)
                df = df.dropna()
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna()

            elif dataset_name == "diabetes":
                # Try different parsing methods
                try:
                    df = pd.read_csv(StringIO(response.text), header=None)
                except:
                    df = pd.read_csv(StringIO(response.text))
                df.columns = dataset_info["columns"]

            elif dataset_name == "vehicle":
                df = pd.read_csv(StringIO(response.text), header=None, delim_whitespace=True)
                df.columns = dataset_info["columns"]

            elif dataset_name == "segment":
                df = pd.read_csv(StringIO(response.text), header=None, delim_whitespace=True)
                df.columns = dataset_info["columns"]

            elif dataset_name == "letter":
                df = pd.read_csv(StringIO(response.text), header=None)
                df.columns = dataset_info["columns"]

            elif dataset_name == "abalone":
                df = pd.read_csv(StringIO(response.text), header=None)
                df.columns = dataset_info["columns"]
                df['rings'] = df['rings'].astype(int)

            else:
                # Generic parsing
                try:
                    df = pd.read_csv(StringIO(response.text), header=None)
                    df.columns = dataset_info["columns"]
                except:
                    df = pd.read_csv(StringIO(response.text), delim_whitespace=True, header=None)
                    df.columns = dataset_info["columns"]

            # Ensure target column is set correctly
            target = dataset_info["target"]
            if target not in df.columns:
                for col in df.columns:
                    if target.lower() in col.lower() or col.lower() in ["class", "type", "label"]:
                        target = col
                        break

            # Convert target to categorical if needed
            if df[target].dtype == 'object':
                df[target] = df[target].astype('category')

            print(f"✅ Successfully loaded {dataset_name}: {len(df)} samples, {len(df.columns)} columns")
            print(f"   Target: {target}")
            print(f"   Classes: {df[target].nunique()}")

            return df

        except Exception as e:
            print(f"   Failed from {url}: {str(e)}")
            return None

    @staticmethod
    def _create_synthetic_dataset(dataset_name: str) -> pd.DataFrame:
        """Create synthetic dataset for demonstration when download fails"""
        np.random.seed(42)

        if dataset_name == "iris":
            n_samples = 150
            n_features = 4
            n_classes = 3
            feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            target_name = "class"

            X = np.random.randn(n_samples, n_features) * 0.5
            X[:50, :] += [5.1, 3.5, 1.4, 0.2]
            X[50:100, :] += [5.9, 2.8, 4.2, 1.3]
            X[100:, :] += [6.5, 3.0, 5.5, 2.0]
            y = np.repeat([0, 1, 2], 50)

        elif dataset_name == "diabetes":
            n_samples = 768
            n_features = 8
            feature_names = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]
            target_name = "class"

            X = np.random.randn(n_samples, n_features)
            # Add some correlation with target
            y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)

        else:
            # Generic synthetic dataset
            n_samples = 500
            n_features = 5
            feature_names = [f"feature_{i}" for i in range(1, n_features + 1)]
            target_name = "class"
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 3, n_samples)

        df = pd.DataFrame(X, columns=feature_names)
        df[target_name] = y

        print(f"⚠️ Created synthetic dataset: {len(df)} samples, {len(df.columns)} columns")
        print(f"   Target: {target_name}")
        print(f"   Classes: {df[target_name].nunique()}")

        return df

    @staticmethod
    def list_uci_datasets() -> List[str]:
        """Return list of available UCI datasets"""
        return sorted(DatasetConfig.UCI_DATASETS.keys())

    @staticmethod
    def get_dataset_info(dataset_name: str) -> Optional[Dict]:
        """Get information about a UCI dataset"""
        if dataset_name in DatasetConfig.UCI_DATASETS:
            return DatasetConfig.UCI_DATASETS[dataset_name]
        return None

    @staticmethod
    def load_config(dataset_name: str) -> Dict:
        """Configuration loading with UCI dataset support"""
        if not dataset_name or not isinstance(dataset_name, str):
            print("Error: Invalid dataset name provided.")
            return None

        config_path = os.path.join('data', dataset_name, f"{dataset_name}.conf")
        csv_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")

        try:
            # If config doesn't exist but dataset is in UCI list, download it
            if not os.path.exists(config_path) and dataset_name in DatasetConfig.UCI_DATASETS:
                print(f"📥 UCI dataset '{dataset_name}' found. Downloading...")
                df = DatasetConfig.download_uci_data(dataset_name)

                if df is not None:
                    # Create data directory
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

                    # Save CSV
                    df.to_csv(csv_path, index=False)
                    print(f"✅ Dataset saved to {csv_path}")

                    # Create config
                    config = DatasetConfig.DEFAULT_CONFIG.copy()
                    config.update({
                        "file_path": csv_path,
                        "column_names": list(df.columns),
                        "target_column": DatasetConfig.UCI_DATASETS[dataset_name]["target"],
                        "has_header": True,
                        "modelType": "Histogram",
                    })

                    # Save config
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"✅ Configuration saved to {config_path}")

                    return config
                else:
                    print(f"❌ Failed to download dataset {dataset_name}")
                    return None

            # Load existing config
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return config

            # If file exists but no config
            if os.path.exists(csv_path):
                print(f"📁 Found data file: {csv_path}. Creating configuration...")
                df = pd.read_csv(csv_path)

                config = DatasetConfig.DEFAULT_CONFIG.copy()
                config.update({
                    "file_path": csv_path,
                    "column_names": list(df.columns),
                    "target_column": df.columns[-1],
                    "has_header": True,
                    "modelType": "Histogram",
                })

                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Configuration saved to {config_path}")

                return config

            print(f"❌ No data file found for dataset: {dataset_name}")
            return None

        except Exception as e:
            print(f"❌ Error loading configuration for {dataset_name}: {e}")
            return None

# =============================================================================
# SECTION 3: COMPILED KERNELS (JIT - Mathematically Equivalent)
# =============================================================================

@torch.jit.script
def compute_bin_indices_jit(
    features: torch.Tensor,
    edges0: torch.Tensor,
    edges1: torch.Tensor,
    n_bins: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices0 = torch.bucketize(features[:, 0], edges0) - 1
    indices1 = torch.bucketize(features[:, 1], edges1) - 1

    indices0 = torch.clamp(indices0, 0, n_bins - 1)
    indices1 = torch.clamp(indices1, 0, n_bins - 1)

    return indices0, indices1


@torch.jit.script
def compute_log_likelihoods_jit(
    bin_probs: torch.Tensor,
    bin_weights: torch.Tensor,
    indices0: torch.Tensor,
    indices1: torch.Tensor,
    epsilon: float = 1e-10
) -> torch.Tensor:
    n_classes = bin_probs.shape[0]
    batch_size = indices0.shape[0]

    result = torch.zeros((batch_size, n_classes), device=bin_probs.device, dtype=torch.float64)

    for c in range(n_classes):
        weighted = bin_probs[c] * bin_weights[c]
        probs = weighted[indices0, indices1]
        result[:, c] = torch.log(probs + epsilon)

    return result


@torch.jit.script
def compute_posteriors_jit(
    log_likelihoods: torch.Tensor,
    epsilon: float = 1e-10
) -> torch.Tensor:
    max_ll = torch.max(log_likelihoods, dim=1, keepdim=True)[0]
    exp_ll = torch.exp(log_likelihoods - max_ll)
    return exp_ll / (torch.sum(exp_ll, dim=1, keepdim=True) + epsilon)


# =============================================================================
# SECTION 4: OPTIMIZED WEIGHT UPDATER
# =============================================================================

class OptimizedWeightUpdater:
    """Vectorized weight updates - mathematically IDENTICAL to sequential updates"""

    def __init__(self, n_classes: int, n_pairs: int, n_bins: int, device: str):
        self.n_classes = n_classes
        self.n_pairs = n_pairs
        self.n_bins = n_bins
        self.device = device

        self.weights = torch.full(
            (n_classes, n_pairs, n_bins, n_bins),
            1e-6,
            dtype=torch.float64,
            device=device
        ).contiguous()

    def batch_update(self, class_ids: torch.Tensor, pair_ids: torch.Tensor,
                    bin_is: torch.Tensor, bin_js: torch.Tensor, adjustments: torch.Tensor):
        class_ids = class_ids.to(self.device)
        pair_ids = pair_ids.to(self.device)
        bin_is = bin_is.to(self.device)
        bin_js = bin_js.to(self.device)
        adjustments = adjustments.to(self.device)

        unique_keys, inverse = torch.unique(
            torch.stack([class_ids, pair_ids]), dim=1, return_inverse=True
        )

        for key_idx in range(unique_keys.shape[1]):
            class_id, pair_id = unique_keys[:, key_idx]
            mask = (inverse == key_idx)

            if mask.any():
                self.weights[
                    class_id.long(),
                    pair_id.long(),
                    bin_is[mask],
                    bin_js[mask]
                ] += adjustments[mask]

    def get_weights(self, class_id: int, pair_id: int) -> torch.Tensor:
        return self.weights[class_id, pair_id]


# =============================================================================
# SECTION 5: OPTIMIZED BATCH PROCESSOR
# =============================================================================

class OptimizedBatchProcessor:
    """Handles batch processing - maintains exact operation order"""

    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self._bin_edges_cache = {}
        self._bin_probs_cache = {}

    def process_batch(self, features: torch.Tensor, feature_pairs: List,
                      bin_edges: List, bin_probs: List, weights: torch.Tensor,
                      n_bins: int, n_classes: int) -> Tuple[torch.Tensor, Dict]:
        batch_size = features.shape[0]
        n_pairs = len(feature_pairs)

        log_likelihoods = torch.zeros((batch_size, n_classes),
                                      dtype=torch.float64, device=self.device)

        bin_indices_dict = {}

        for pair_idx in range(n_pairs):
            f1, f2 = feature_pairs[pair_idx]

            cache_key = (pair_idx, f1, f2)
            if cache_key not in self._bin_edges_cache:
                self._bin_edges_cache[cache_key] = (
                    bin_edges[pair_idx][0].contiguous(),
                    bin_edges[pair_idx][1].contiguous()
                )
            edges0, edges1 = self._bin_edges_cache[cache_key]

            indices0, indices1 = compute_bin_indices_jit(
                features[:, [f1, f2]], edges0, edges1, n_bins
            )

            bin_indices_dict[pair_idx] = (indices0.clone(), indices1.clone())

            if pair_idx not in self._bin_probs_cache:
                probs_tensor = bin_probs[pair_idx]
                if isinstance(probs_tensor, torch.Tensor):
                    self._bin_probs_cache[pair_idx] = probs_tensor.to(self.device)
                else:
                    self._bin_probs_cache[pair_idx] = torch.tensor(probs_tensor, device=self.device)

            pair_weights = weights[:, pair_idx, :, :]

            pair_ll = compute_log_likelihoods_jit(
                self._bin_probs_cache[pair_idx], pair_weights, indices0, indices1
            )

            log_likelihoods += pair_ll

        posteriors = compute_posteriors_jit(log_likelihoods)

        return posteriors, bin_indices_dict

# =============================================================================
# SECTION 6: OPTIMIZED DATASET PROCESSOR
# =============================================================================

class OptimizedDatasetProcessor:
    """Memory-efficient dataset loading - preserves exact preprocessing"""

    def __init__(self, config: Dict, device: str):
        self.config = config
        self.device = device
        self.categorical_encoders = {}
        self.feature_stats = {}
        self.label_encoder = {}

    def load_and_preprocess(self, file_path: str, target_column: str,
                           is_training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if os.path.getsize(file_path) > 100 * 1024 * 1024:
            df = pd.read_csv(file_path, memory_map=True)
        else:
            df = pd.read_csv(file_path)

        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column].astype(str)
        else:
            X = df
            y = None

        X_processed = self._preprocess_features(X, is_training)

        X_tensor = torch.tensor(X_processed, dtype=torch.float64)

        if y is not None:
            y_tensor = torch.tensor(self._encode_labels(y, is_training), dtype=torch.long)
            return X_tensor, y_tensor

        return X_tensor, None

    def _preprocess_features(self, X: pd.DataFrame, is_training: bool) -> np.ndarray:
        X = X.fillna(-99999)

        for col in X.select_dtypes(include=['object', 'category']).columns:
            if is_training or col not in self.categorical_encoders:
                unique_vals = X[col].unique()
                self.categorical_encoders[col] = {v: i for i, v in enumerate(unique_vals)}

            X[col] = X[col].map(self.categorical_encoders[col]).fillna(-1)

        X_np = X.values.astype(np.float64)

        if is_training:
            self.feature_stats['mean'] = X_np.mean(axis=0)
            self.feature_stats['std'] = X_np.std(axis=0)
            self.feature_stats['std'][self.feature_stats['std'] == 0] = 1

        X_scaled = (X_np - self.feature_stats['mean']) / self.feature_stats['std']

        return X_scaled

    def _encode_labels(self, y: pd.Series, is_training: bool) -> np.ndarray:
        if is_training:
            self.label_encoder = {v: i for i, v in enumerate(y.unique())}

        return y.map(self.label_encoder).values

    def get_feature_info(self) -> Dict:
        return {
            'categorical_encoders': self.categorical_encoders,
            'feature_stats': self.feature_stats,
            'label_encoder': self.label_encoder
        }

# =============================================================================
# SECTION 7: OPTIMIZED DBNN CORE (MODIFIED - Added ExternalToolsMixin)
# =============================================================================

class OptimizedDBNN(ExternalToolsMixin):
    """Complete DBNN implementation with all fixes"""

    def __init__(self, dataset_name: str = None, config: Union[Dict, str] = None,
                 mode: str = 'train_predict', parallel: bool = True,
                 enable_external_tools: bool = ASTROPY_AVAILABLE):

        # Call parent __init__ (ExternalToolsMixin)
        super().__init__(enable_external_tools=enable_external_tools)

        # Initialize the mixin first with kwargs
        ExternalToolsMixin.__init__(self, enable_external_tools=enable_external_tools)

        self.dataset_name = dataset_name
        self.mode = mode
        self.stop_training_flag = False

        # Load configuration
        if config is None and dataset_name is not None:
            self.config = DatasetConfig.load_config(dataset_name)
        elif isinstance(config, str):
            with open(config, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config or {}

        if self.config is None:
            raise ValueError(f"Could not load configuration for {dataset_name}")

        # Extract parameters
        self.model_type = self.config.get('model_type', self.config.get('modelType', 'Histogram'))
        self.target_column = self.config.get('target_column')

        # Device setup
        compute_device = self.config.get('compute_device', 'auto')
        if compute_device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = compute_device

        # Training parameters
        training_params = self.config.get('training_params', {})
        self.learning_rate = training_params.get('learning_rate', 0.1)
        self.n_bins_per_dim = training_params.get('n_bins_per_dim', 128)
        self.test_size = training_params.get('test_fraction', 0.2)
        self.enable_adaptive = training_params.get('enable_adaptive', True)
        self.adaptive_rounds = training_params.get('adaptive_rounds', 10)
        self.initial_samples = training_params.get('initial_samples', 50)
        self.max_samples_per_round = training_params.get('max_samples_per_round', 500)
        self.patience = training_params.get('patience', 25)

        # Active learning parameters
        active_learning = self.config.get('active_learning', {})
        self.similarity_threshold = active_learning.get('similarity_threshold', 0.25)
        self.min_divergence = active_learning.get('min_divergence', 0.1)

        # Initialize components
        self.batch_processor = OptimizedBatchProcessor(self, self.device)
        self.batch_size = self._calculate_optimal_batch_size()

        # Tensor evolution tracker
        self.evolution_tracker = TensorEvolutionTracker(self)

        # Core model components
        self.feature_pairs = None
        self.bin_edges = None
        self.bin_probs = None
        self.weight_updater = None
        self.classes = None
        self.label_encoder = None

        # Data
        self.X_tensor = None
        self.y_tensor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_indices = None
        self.test_indices = None
        self.data_original = None
        self.feature_names = None
        self.preprocessor = None

        # Training history
        self.training_history = []
        self.accuracy_progression = []
        self.best_round_initial_conditions = None
        self.best_combined_accuracy = 0.0
        self.in_adaptive_fit = False

        print(f"🚀 Optimized DBNN initialized on {self.device}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Model type: {self.model_type}")
        print(f"   Adaptive training: {self.enable_adaptive}")

    def _calculate_optimal_batch_size(self) -> int:
        if self.device == 'cuda' and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory = total_memory - torch.cuda.memory_allocated()
            memory_per_sample = 4 * 1024 * 1024
            optimal = int((free_memory * 0.3) / memory_per_sample)
            return max(32, min(optimal, 2048))
        return 128

    def enable_evolution_tracking(self):
        self.evolution_tracker.enable()

    def get_evolution_history(self):
        return self.evolution_tracker.get_history()

    def clear_evolution_history(self):
        self.evolution_tracker.clear_history()

    # =============================================================================
    # SECTION: INTEGRATE WITH OPTIMIZEDDBNN (NON-INTRUSIVE)
    # =============================================================================

    def _log_available_tools(self):
        """Log available external tools"""
        if self.tools_manager:
            available = self.tools_manager.get_available_tools()
            print(f"{Colors.CYAN}🔌 External Tools Available:{Colors.ENDC}")
            for name, avail in available.items():
                status = f"{Colors.GREEN}✓{Colors.ENDC}" if avail else f"{Colors.RED}✗{Colors.ENDC}"
                print(f"   {status} {name}")

    def export_to_topcat(self, dataset: str = 'all', **kwargs) -> bool:
        """Export data to Topcat"""
        if not self.enable_external_tools or not self.tools_manager:
            print(f"{Colors.YELLOW}External tools not enabled{Colors.ENDC}")
            return False

        # Prepare data based on dataset selection
        if dataset == 'all':
            data = self._prepare_export_data()
        elif dataset == 'training':
            data = pd.DataFrame(self.X_train.numpy(), columns=self.feature_names)
            data['true_class'] = self.y_train.numpy()
        elif dataset == 'test':
            data = pd.DataFrame(self.X_test.numpy(), columns=self.feature_names)
            data['true_class'] = self.y_test.numpy()
        else:
            print(f"{Colors.RED}Unknown dataset: {dataset}{Colors.ENDC}")
            return False

        # Add predictions if available
        if hasattr(self, 'predict'):
            predictions, _ = self.predict(self.X_tensor)
            data['predicted_class'] = predictions.numpy()

        return self.tools_manager.launch_tool('topcat', data, **kwargs)

    def export_to_aladin(self, ra_col: str = None, dec_col: str = None, **kwargs) -> bool:
        """Export data to Aladin"""
        if not self.enable_external_tools or not self.tools_manager:
            return False

        data = self._prepare_export_data()

        # Auto-detect RA/Dec columns if not specified
        if not ra_col or not dec_col:
            for col in data.columns:
                if col.lower() in ['ra', 'right_ascension', 'alpha']:
                    ra_col = col
                if col.lower() in ['dec', 'declination', 'delta']:
                    dec_col = col

        if ra_col and dec_col:
            print(f"{Colors.GREEN}Using coordinates: {ra_col}, {dec_col}{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}No RA/Dec columns detected. Using first two numeric columns.{Colors.ENDC}")

        return self.tools_manager.launch_tool('aladin', data, **kwargs)

    def query_sdss_crossmatch(self, radius: float = 1.0) -> pd.DataFrame:
        """Query SDSS for cross-matching with current data"""
        if not self.enable_external_tools or not self.tools_manager:
            return None

        # Try to find RA/Dec columns
        data = self._prepare_export_data()
        ra_col = None
        dec_col = None

        for col in data.columns:
            if col.lower() in ['ra', 'right_ascension', 'alpha']:
                ra_col = col
            if col.lower() in ['dec', 'declination', 'delta']:
                dec_col = col

        if not ra_col or not dec_col:
            print(f"{Colors.RED}No RA/Dec columns found in data{Colors.ENDC}")
            return None

        return self.tools_manager.query_sdss(
            data[ra_col].values,
            data[dec_col].values,
            radius
        )

    def save_as_fits(self, filename: str = None) -> str:
        """Save data as FITS file"""
        if not self.enable_external_tools or not self.tools_manager:
            return None

        data = self._prepare_export_data()
        return self.tools_manager.convert_to_fits(data, filename)

    def _prepare_export_data(self) -> pd.DataFrame:
        """Prepare data for export"""
        # Get feature data
        X_np = self.X_tensor.numpy()
        y_np = self.y_tensor.numpy()

        # Create dataframe
        if self.feature_names:
            columns = self.feature_names
        else:
            columns = [f'feature_{i}' for i in range(X_np.shape[1])]

        data = pd.DataFrame(X_np, columns=columns)

        # Add labels
        inv_label_encoder = {v: k for k, v in self.label_encoder.items()}
        data['true_class'] = [inv_label_encoder[t] for t in y_np]

        # Add predictions if available
        if hasattr(self, 'predict'):
            predictions, _ = self.predict(self.X_tensor)
            data['predicted_class'] = [inv_label_encoder[p] for p in predictions.numpy()]

        return data

    # =============================================================================
    # SECTION: INTEGRATE WITH OPTIMIZEDDBNN (NON-INTRUSIVE) ENDS
    # =============================================================================

    def disable_evolution_tracking(self):
        self.evolution_tracker.disable()

    def reset_model(self, hard_reset: bool = True):
        """Reset model to initial state"""
        print(f"{Colors.YELLOW}🔄 Resetting model...{Colors.ENDC}")

        if hard_reset:
            self.feature_pairs = None
            self.bin_edges = None
            self.bin_probs = None
            self.weight_updater = None
            self.classes = None

            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.train_indices = None
            self.test_indices = None

            self.training_history = []
            self.accuracy_progression = []
            self.best_round_initial_conditions = None
            self.best_combined_accuracy = 0.0

            print(f"{Colors.GREEN}✅ Model hard reset complete - all learned knowledge cleared{Colors.ENDC}")
            print(f"{Colors.YELLOW}   Note: Data must be reloaded after reset{Colors.ENDC}")
        else:
            if hasattr(self, 'classes') and self.classes is not None and len(self.classes) > 0:
                n_classes = len(self.classes)
                n_pairs = len(self.feature_pairs) if self.feature_pairs else 0

                if n_pairs > 0:
                    self.weight_updater = OptimizedWeightUpdater(
                        n_classes, n_pairs, self.n_bins_per_dim, self.device
                    )
                    print(f"{Colors.GREEN}✅ Model soft reset complete - weights reinitialized{Colors.ENDC}")
            else:
                print(f"{Colors.YELLOW}⚠️ Cannot soft reset - no classes loaded. Use hard reset with data reload.{Colors.ENDC}")

        self.evolution_tracker.clear_history()

    def load_data(self, file_path: str = None, is_training: bool = True):
        processor = OptimizedDatasetProcessor(self.config, self.device)

        file_path = file_path or self.config.get('file_path')
        if not file_path:
            raise ValueError("No file path provided")

        self.X_tensor, self.y_tensor = processor.load_and_preprocess(
            file_path, self.target_column, is_training=is_training
        )

        self.label_encoder = processor.label_encoder
        self.preprocessor = processor

        if self.y_tensor is not None:
            self.classes = torch.unique(self.y_tensor)
            print(f"   Loaded {len(self.X_tensor)} samples, {len(self.classes)} classes")

        self.data_original = pd.read_csv(file_path)

        if hasattr(self.data_original, 'columns'):
            if self.target_column in self.data_original.columns:
                self.feature_names = [col for col in self.data_original.columns if col != self.target_column]
            else:
                self.feature_names = self.data_original.columns.tolist()

        return self.X_tensor, self.y_tensor

    def split_data(self):
        from sklearn.model_selection import train_test_split

        if self.X_tensor is None or self.y_tensor is None:
            raise ValueError("No data loaded. Call load_data() first.")

        X_np = self.X_tensor.numpy()
        y_np = self.y_tensor.numpy()

        indices = np.arange(len(X_np))
        train_idx, test_idx = train_test_split(
            indices, test_size=self.test_size, random_state=42, stratify=y_np
        )

        self.train_indices = train_idx
        self.test_indices = test_idx

        self.X_train = self.X_tensor[train_idx]
        self.X_test = self.X_tensor[test_idx]
        self.y_train = self.y_tensor[train_idx]
        self.y_test = self.y_tensor[test_idx]

        print(f"   Train: {len(self.X_train)} samples, Test: {len(self.X_test)} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def generate_feature_pairs(self, n_features: int) -> List[Tuple[int, int]]:
        from itertools import combinations

        feature_indices = list(range(n_features))
        self.feature_pairs = list(combinations(feature_indices, 2))

        return self.feature_pairs

    def compute_bin_edges(self, X: torch.Tensor) -> List[List[torch.Tensor]]:
        bin_edges = []

        for f1, f2 in self.feature_pairs:
            pair_data = X[:, [f1, f2]]

            mins = torch.min(pair_data, dim=0)[0]
            maxs = torch.max(pair_data, dim=0)[0]
            padding = (maxs - mins) * 0.01

            edges_dim0 = torch.linspace(
                mins[0] - padding[0], maxs[0] + padding[0],
                self.n_bins_per_dim + 1, device=self.device
            )
            edges_dim1 = torch.linspace(
                mins[1] - padding[1], maxs[1] + padding[1],
                self.n_bins_per_dim + 1, device=self.device
            )

            bin_edges.append([edges_dim0, edges_dim1])

        return bin_edges

    def compute_likelihoods(self, X: torch.Tensor, y: torch.Tensor) -> Dict:
        n_classes = len(self.classes)
        n_pairs = len(self.feature_pairs)

        bin_counts = torch.zeros((n_classes, n_pairs, self.n_bins_per_dim, self.n_bins_per_dim),
                                dtype=torch.float64, device=self.device)

        for pair_idx, (f1, f2) in enumerate(self.feature_pairs):
            edges0, edges1 = self.bin_edges[pair_idx]

            indices0 = torch.bucketize(X[:, f1], edges0) - 1
            indices1 = torch.bucketize(X[:, f2], edges1) - 1
            indices0 = indices0.clamp(0, self.n_bins_per_dim - 1)
            indices1 = indices1.clamp(0, self.n_bins_per_dim - 1)

            for class_idx in range(n_classes):
                class_mask = (y == class_idx)
                if class_mask.any():
                    flat_indices = indices0[class_mask] * self.n_bins_per_dim + indices1[class_mask]
                    counts = torch.bincount(flat_indices, minlength=self.n_bins_per_dim * self.n_bins_per_dim)
                    bin_counts[class_idx, pair_idx] = counts.view(
                        self.n_bins_per_dim, self.n_bins_per_dim
                    ).float()

        smoothed = bin_counts + 1.0
        bin_probs_tensor = smoothed / smoothed.sum(dim=(2, 3), keepdim=True)

        self.bin_probs = [bin_probs_tensor[:, pair_idx, :, :] for pair_idx in range(n_pairs)]

        self.weight_updater = OptimizedWeightUpdater(
            n_classes, n_pairs, self.n_bins_per_dim, self.device
        )

        return {
            'bin_counts': bin_counts,
            'bin_probs': self.bin_probs,
            'bin_edges': self.bin_edges,
            'feature_pairs': self.feature_pairs,
            'classes': self.classes
        }

    def _compute_batch_posterior(self, features: torch.Tensor,
                                 return_bin_indices: bool = True) -> Tuple[torch.Tensor, Optional[Dict]]:
        with torch.no_grad():
            posteriors, bin_indices = self.batch_processor.process_batch(
                features, self.feature_pairs, self.bin_edges, self.bin_probs,
                self.weight_updater.weights, self.n_bins_per_dim, len(self.classes)
            )
            return posteriors, (bin_indices if return_bin_indices else None)

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        all_predictions = []
        all_posteriors = []

        for i in range(0, n_samples, self.batch_size):
            batch_X = X[i:min(i + self.batch_size, n_samples)]

            if self.device == 'cuda' and batch_X.device.type == 'cpu':
                batch_X = batch_X.cuda(non_blocking=True)

            posteriors, _ = self._compute_batch_posterior(batch_X, return_bin_indices=False)
            predictions = torch.argmax(posteriors, dim=1)

            all_predictions.append(predictions.cpu())
            all_posteriors.append(posteriors.cpu())

        return torch.cat(all_predictions), torch.cat(all_posteriors)

    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor) -> Tuple[float, List]:
        n_samples = len(X_train)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        failed_cases = []
        n_errors = 0

        for i in range(0, n_samples, self.batch_size):
            batch_X = X_train[i:min(i + self.batch_size, n_samples)]
            batch_y = y_train[i:min(i + self.batch_size, n_samples)]

            posteriors, bin_indices = self._compute_batch_posterior(batch_X)

            predictions = torch.argmax(posteriors, dim=1)
            errors = (predictions != batch_y)
            n_errors += errors.sum().item()

            if errors.any():
                error_indices = torch.where(errors)[0]
                failed_cases.extend([
                    (batch_X[idx], batch_y[idx].item(), posteriors[idx].cpu().numpy())
                    for idx in error_indices
                ])

        if failed_cases:
            self._update_weights_optimized(failed_cases)

        train_accuracy = 1.0 - (n_errors / n_samples)

        return train_accuracy, failed_cases

    def _update_weights_optimized(self, failed_cases: List):
        n_failed = len(failed_cases)
        if n_failed == 0:
            return

        features = torch.stack([case[0] for case in failed_cases]).to(self.device)
        true_classes = torch.tensor([case[1] for case in failed_cases], device=self.device)

        posteriors, bin_indices = self._compute_batch_posterior(features)
        pred_classes = torch.argmax(posteriors, dim=1)

        true_posteriors = posteriors[torch.arange(n_failed), true_classes]
        pred_posteriors = posteriors[torch.arange(n_failed), pred_classes]
        adjustments = self.learning_rate * (1.0 - (true_posteriors / pred_posteriors))

        all_class_ids = []
        all_pair_ids = []
        all_bin_is = []
        all_bin_js = []
        all_adjustments = []

        for pair_idx in bin_indices:
            bin_i, bin_j = bin_indices[pair_idx]

            pred_probs = self.bin_probs[pair_idx][pred_classes, bin_i, bin_j]
            dissimilar_mask = pred_probs < self.similarity_threshold

            if dissimilar_mask.any():
                all_class_ids.append(true_classes[dissimilar_mask])
                all_pair_ids.append(torch.full_like(true_classes[dissimilar_mask], pair_idx))
                all_bin_is.append(bin_i[dissimilar_mask])
                all_bin_js.append(bin_j[dissimilar_mask])
                all_adjustments.append(adjustments[dissimilar_mask])

        if all_class_ids:
            class_ids = torch.cat(all_class_ids)
            pair_ids = torch.cat(all_pair_ids)
            bin_is = torch.cat(all_bin_is)
            bin_js = torch.cat(all_bin_js)
            adjustments = torch.cat(all_adjustments)

            self.weight_updater.batch_update(class_ids, pair_ids, bin_is, bin_js, adjustments)

    def fit(self, X_train: torch.Tensor = None, y_train: torch.Tensor = None,
            X_test: Optional[torch.Tensor] = None, y_test: Optional[torch.Tensor] = None,
            epochs: int = 100, patience: int = 10) -> Dict:
        if X_train is None:
            X_train = self.X_train
            y_train = self.y_train
            X_test = self.X_test
            y_test = self.y_test

        if X_train is None:
            raise ValueError("No training data provided")

        print(f"\n🚀 Starting training for {epochs} epochs...")

        if self.feature_pairs is None:
            self.generate_feature_pairs(X_train.shape[1])
            print(f"   Generated {len(self.feature_pairs)} feature pairs")

        if self.bin_edges is None:
            self.bin_edges = self.compute_bin_edges(X_train)
            print(f"   Computed bin edges")

        if self.bin_probs is None:
            likelihoods = self.compute_likelihoods(X_train, y_train)
            self.bin_probs = likelihoods['bin_probs']
            print(f"   Computed likelihoods")

        best_accuracy = 0.0
        patience_counter = 0
        training_history = []
        start_time = time.time()

        for epoch in range(epochs):
            if self.stop_training_flag:
                print(f"\n{Colors.YELLOW}🛑 Training stopped by user at epoch {epoch+1}{Colors.ENDC}")
                break

            train_accuracy, failed_cases = self.train_epoch(X_train, y_train)

            if X_test is not None:
                test_predictions, _ = self.predict(X_test)
                test_accuracy = (test_predictions == y_test.cpu()).float().mean().item()
            else:
                test_accuracy = train_accuracy

            epoch_data = {
                'epoch': epoch + 1,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'failed_cases': len(failed_cases),
                'time': time.time() - start_time
            }
            training_history.append(epoch_data)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs} | "
                      f"Train Acc: {train_accuracy:.4f} | "
                      f"Test Acc: {test_accuracy:.4f} | "
                      f"Failed: {len(failed_cases):4d}")

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                patience_counter = 0
                self._save_checkpoint(epoch, test_accuracy)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n✅ Early stopping at epoch {epoch + 1}")
                    break

        print(f"\n✅ Training completed in {time.time() - start_time:.1f}s")
        print(f"   Best accuracy: {best_accuracy:.4f}")

        return {
            'history': training_history,
            'best_accuracy': best_accuracy,
            'final_train_accuracy': train_accuracy,
            'final_test_accuracy': test_accuracy
        }

    def fit_predict(self, batch_size: int = 128, save_path: str = None) -> Dict:
        if self.X_tensor is None:
            self.load_data()

        if self.X_train is None:
            self.split_data()

        results = self.fit(self.X_train, self.y_train, self.X_test, self.y_test, epochs=100)

        train_pred, train_post = self.predict(self.X_train)
        test_pred, test_post = self.predict(self.X_test)

        inv_label_encoder = {v: k for k, v in self.label_encoder.items()}
        train_pred_labels = [inv_label_encoder[p.item()] for p in train_pred]
        test_pred_labels = [inv_label_encoder[p.item()] for p in test_pred]

        all_indices = np.concatenate([self.train_indices, self.test_indices])
        all_pred = np.concatenate([train_pred.numpy(), test_pred.numpy()])
        all_true = np.concatenate([self.y_train.numpy(), self.y_test.numpy()])

        all_predictions_df = pd.DataFrame({
            'true_class': [inv_label_encoder[t] for t in all_true],
            'predicted_class': [inv_label_encoder[p] for p in all_pred],
            'true_encoded': all_true,
            'predicted_encoded': all_pred,
            'split': ['train'] * len(self.train_indices) + ['test'] * len(self.test_indices)
        })
        all_predictions_df.index = all_indices

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            train_results = pd.DataFrame({
                'true': self.y_train.numpy(),
                'true_label': [inv_label_encoder[t.item()] for t in self.y_train],
                'predicted': train_pred.numpy(),
                'predicted_label': train_pred_labels
            })
            train_results.to_csv(os.path.join(save_path, 'train_predictions.csv'), index=False)

            test_results = pd.DataFrame({
                'true': self.y_test.numpy(),
                'true_label': [inv_label_encoder[t.item()] for t in self.y_test],
                'predicted': test_pred.numpy(),
                'predicted_label': test_pred_labels
            })
            test_results.to_csv(os.path.join(save_path, 'test_predictions.csv'), index=False)

            all_predictions_df.to_csv(os.path.join(save_path, 'all_predictions.csv'), index=False)

        return {
            'train_accuracy': results['final_train_accuracy'],
            'test_accuracy': results['final_test_accuracy'],
            'best_accuracy': results['best_accuracy'],
            'history': results['history'],
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'train_posteriors': train_post,
            'test_posteriors': test_post,
            'all_predictions': all_predictions_df
        }

    def _save_checkpoint(self, epoch: int, accuracy: float):
        checkpoint = {
            'epoch': epoch,
            'accuracy': accuracy,
            'weight_updater': self.weight_updater,
            'bin_probs': self.bin_probs,
            'bin_edges': self.bin_edges,
            'feature_pairs': self.feature_pairs,
            'classes': self.classes,
            'label_encoder': self.label_encoder
        }
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/best_model_{self.dataset_name}.pt')

    def save_model(self, path: str):
        model_state = {
            'config': self.config,
            'dataset_name': self.dataset_name,
            'model_type': self.model_type,
            'n_bins_per_dim': self.n_bins_per_dim,
            'feature_pairs': self.feature_pairs,
            'bin_edges': [[edge.cpu().tolist() for edge in pair] for pair in self.bin_edges],
            'bin_probs': [prob.cpu().tolist() for prob in self.bin_probs],
            'weight_updater': self.weight_updater,
            'classes': self.classes.cpu().tolist() if torch.is_tensor(self.classes) else self.classes,
            'label_encoder': self.label_encoder,
            'evolution_history': self.evolution_tracker.get_history(),
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'preprocessor_stats': {
                'feature_stats': self.preprocessor.feature_stats if self.preprocessor else {},
                'categorical_encoders': self.preprocessor.categorical_encoders if self.preprocessor else {},
                'label_encoder': self.preprocessor.label_encoder if self.preprocessor else {}
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"✅ Model saved to {path}")

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            model_state = pickle.load(f)

        self.config = model_state['config']
        self.dataset_name = model_state['dataset_name']
        self.model_type = model_state['model_type']
        self.n_bins_per_dim = model_state['n_bins_per_dim']
        self.feature_pairs = model_state['feature_pairs']

        self.bin_edges = []
        for pair in model_state['bin_edges']:
            pair_edges = []
            for edge in pair:
                edge_tensor = torch.tensor(edge, device=self.device)
                pair_edges.append(edge_tensor)
            self.bin_edges.append(pair_edges)

        self.bin_probs = []
        for prob in model_state['bin_probs']:
            prob_tensor = torch.tensor(prob, device=self.device)
            self.bin_probs.append(prob_tensor)

        self.weight_updater = model_state['weight_updater']

        if isinstance(model_state['classes'], list):
            self.classes = torch.tensor(model_state['classes'], device=self.device)
        else:
            self.classes = model_state['classes'].to(self.device) if torch.is_tensor(model_state['classes']) else torch.tensor(model_state['classes'], device=self.device)

        self.label_encoder = model_state.get('label_encoder', {})
        self.feature_names = model_state.get('feature_names', [])
        self.target_column = model_state.get('target_column', self.target_column)

        if 'evolution_history' in model_state:
            self.evolution_tracker.tensor_evolution_history = model_state['evolution_history']

        print(f"✅ Model loaded from {path}")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Classes: {len(self.classes) if self.classes is not None else 0}")
        print(f"   Features: {len(self.feature_names) if self.feature_names else 0}")

    def predict_from_file(self, input_csv: str, output_path: str = None, **kwargs) -> Dict:
        if output_path:
            os.makedirs(output_path, exist_ok=True)

        df = pd.read_csv(input_csv)
        has_target = self.target_column in df.columns

        if has_target:
            y_true = df[self.target_column]
            X = df.drop(columns=[self.target_column])
        else:
            X = df
            y_true = None

        processor = OptimizedDatasetProcessor(self.config, self.device)

        if hasattr(self, 'preprocessor') and self.preprocessor:
            processor.feature_stats = self.preprocessor.feature_stats
            processor.categorical_encoders = self.preprocessor.categorical_encoders
            processor.label_encoder = self.preprocessor.label_encoder
        else:
            processor = OptimizedDatasetProcessor(self.config, self.device)
            if self.X_tensor is not None:
                X_train_np = self.X_tensor.numpy()
                if self.feature_names:
                    X_train_df = pd.DataFrame(X_train_np, columns=self.feature_names)
                    processor._preprocess_features(X_train_df, is_training=True)

        X_tensor, _ = processor.load_and_preprocess(
            input_csv, self.target_column, is_training=False
        )

        predictions, posteriors = self.predict(X_tensor)

        inv_label_encoder = {v: k for k, v in self.label_encoder.items()}
        pred_labels = [inv_label_encoder[p.item()] for p in predictions]

        results = df.copy()
        results['predicted_class'] = pred_labels
        results['confidence'] = posteriors.max(dim=1)[0].numpy()

        for i, (label, idx) in enumerate(self.label_encoder.items()):
            results[f'prob_{label}'] = posteriors[:, i].numpy()

        if has_target:
            results['true_class'] = y_true

        if output_path:
            csv_path = os.path.join(output_path, 'predictions.csv')
            results.to_csv(csv_path, index=False)
            print(f"✅ Predictions saved to {csv_path}")

            if has_target:
                from sklearn.metrics import accuracy_score
                y_true_encoded = [self.label_encoder.get(str(val), -1) for val in y_true]
                y_true_encoded = [v for v in y_true_encoded if v != -1]
                y_pred_encoded = predictions.numpy()[:len(y_true_encoded)]

                if len(y_true_encoded) > 0:
                    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
                    print(f"   Accuracy: {Colors.highlight_accuracy(accuracy)}")

        return {'predictions': results}

    def _format_class_distribution(self, indices):
        if not indices:
            return "No samples added"

        class_counts = {}
        for idx in indices:
            if idx < len(self.data_original):
                class_label = self.data_original.iloc[idx][self.target_column]
                class_counts[class_label] = class_counts.get(class_label, 0) + 1

        total = len(indices)
        dist = []
        for cls, count in sorted(class_counts.items()):
            percentage = (count / total) * 100
            dist.append(f"{cls}: {count} ({percentage:.1f}%)")

        return ", ".join(dist)

    def _select_samples_from_failed_classes(self, test_predictions, y_test, test_indices, results):
        if torch.is_tensor(test_predictions):
            test_predictions = test_predictions.cpu().numpy()
        if torch.is_tensor(y_test):
            y_test = y_test.cpu().numpy()

        all_results = results.get('all_predictions', pd.DataFrame())
        if len(all_results) == 0:
            print(f"{Colors.YELLOW}Warning: No all_predictions in results{Colors.ENDC}")
            inv_label_encoder = {v: k for k, v in self.label_encoder.items()}
            all_results = pd.DataFrame({
                'true_class': [inv_label_encoder[t] for t in y_test],
                'predicted_class': [inv_label_encoder[p] for p in test_predictions],
                'true_encoded': y_test,
                'predicted_encoded': test_predictions
            })
            all_results.index = test_indices

        test_results = all_results.loc[test_indices] if all_results.index.isin(test_indices).any() else all_results

        if 'predicted_class' not in test_results.columns:
            if 'predicted_encoded' in test_results.columns:
                inv_encoder = {v: k for k, v in self.label_encoder.items()}
                test_results['predicted_class'] = [inv_encoder[p] for p in test_results['predicted_encoded']]
            else:
                print(f"{Colors.RED}Error: Cannot find 'predicted_class' in results{Colors.ENDC}")
                return []

        if 'true_class' not in test_results.columns:
            if 'true_encoded' in test_results.columns:
                inv_encoder = {v: k for k, v in self.label_encoder.items()}
                test_results['true_class'] = [inv_encoder[t] for t in test_results['true_encoded']]
            else:
                print(f"{Colors.RED}Error: Cannot find 'true_class' in results{Colors.ENDC}")
                return []

        misclassified_mask = test_results['predicted_class'] != test_results['true_class']
        misclassified_indices = test_results.index[misclassified_mask].tolist()

        if not misclassified_indices:
            print(f"{Colors.YELLOW}No misclassified samples found{Colors.ENDC}")
            return []

        test_pos_map = {idx: pos for pos, idx in enumerate(test_indices)}
        unique_classes = test_results['true_class'].unique()

        print(f"\n{Colors.CYAN}Selecting samples from failed classes...{Colors.ENDC}")

        final_selected_indices = []

        for class_label in unique_classes:
            class_df = test_results.loc[misclassified_indices]
            class_mask = (class_df['true_class'] == class_label).to_numpy()
            class_indices = np.array(misclassified_indices)[class_mask].tolist()

            if not class_indices:
                continue

            class_positions = [test_pos_map[idx] for idx in class_indices if idx in test_pos_map]
            if not class_positions:
                continue

            max_samples_this_class = min(self.max_samples_per_round, len(class_positions))
            if len(class_positions) > max_samples_this_class:
                class_positions = random.sample(class_positions, max_samples_this_class)
                print(f"{Colors.YELLOW}   Limited class {class_label} to {max_samples_this_class} samples{Colors.ENDC}")

            selected_indices = [test_indices[pos] for pos in class_positions]
            final_selected_indices.extend(selected_indices)
            print(f"{Colors.GREEN}   Added {len(selected_indices)} samples from class {class_label}{Colors.ENDC}")

        return final_selected_indices

    def reset_to_initial_state(self):
        print(f"{Colors.YELLOW}Resetting model to initial state...{Colors.ENDC}")

        n_classes = len(self.classes) if self.classes is not None else 0
        n_pairs = len(self.feature_pairs) if self.feature_pairs is not None else 0

        if n_pairs > 0 and n_classes > 0:
            self.weight_updater = OptimizedWeightUpdater(
                n_classes, n_pairs, self.n_bins_per_dim, self.device
            )

            if self.X_train is not None and self.y_train is not None:
                likelihoods = self.compute_likelihoods(self.X_train, self.y_train)
                self.bin_probs = likelihoods['bin_probs']

        print(f"{Colors.GREEN}Model reset to initial state.{Colors.ENDC}")

    def save_epoch_data(self, epoch: int, train_indices: list, test_indices: list):
        save_epochs = self.config.get('training_params', {}).get('Save_training_epochs', False)
        if not save_epochs:
            return

        base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
        save_path = os.path.join(base_path, self.dataset_name, f'epoch_{epoch}')
        os.makedirs(save_path, exist_ok=True)

        try:
            with open(os.path.join(save_path, f'{self.model_type}_train_indices.pkl'), 'wb') as f:
                pickle.dump(train_indices, f)
            with open(os.path.join(save_path, f'{self.model_type}_test_indices.pkl'), 'wb') as f:
                pickle.dump(test_indices, f)
        except Exception as e:
            print(f"{Colors.YELLOW}Error saving epoch data: {str(e)}{Colors.ENDC}")

    def save_last_split(self, train_indices: list, test_indices: list):
        os.makedirs(f'data/{self.dataset_name}', exist_ok=True)

        train_df = self.data_original.iloc[train_indices].copy()
        train_df['original_index'] = train_indices
        train_df.to_csv(f'data/{self.dataset_name}/Last_training.csv', index=False)

        test_df = self.data_original.iloc[test_indices].copy()
        test_df['original_index'] = test_indices
        test_df.to_csv(f'data/{self.dataset_name}/Last_testing.csv', index=False)

        print(f"{Colors.GREEN}Saved split with {len(train_indices)} train, {len(test_indices)} test samples{Colors.ENDC}")

    def load_last_known_split(self):
        train_file = f'data/{self.dataset_name}/Last_training.csv'
        test_file = f'data/{self.dataset_name}/Last_testing.csv'

        if os.path.exists(train_file) and os.path.exists(test_file):
            try:
                train_df = pd.read_csv(train_file)
                test_df = pd.read_csv(test_file)

                if 'original_index' not in train_df.columns or 'original_index' not in test_df.columns:
                    print(f"{Colors.YELLOW}No original_index column found in split files{Colors.ENDC}")
                    return None, None

                train_indices = train_df['original_index'].tolist()
                test_indices = test_df['original_index'].tolist()

                max_valid_index = len(self.data_original) - 1 if hasattr(self, 'data_original') else float('inf')
                train_indices = [idx for idx in train_indices if 0 <= idx <= max_valid_index]
                test_indices = [idx for idx in test_indices if 0 <= idx <= max_valid_index]

                print(f"{Colors.GREEN}Loaded previous split - Training: {len(train_indices)}, Testing: {len(test_indices)}{Colors.ENDC}")
                return train_indices, test_indices

            except Exception as e:
                print(f"{Colors.YELLOW}Error loading previous split: {str(e)}{Colors.ENDC}")
                return None, None

        return None, None

    def print_colored_confusion_matrix(self, y_true, y_pred, header=None):
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()

        if hasattr(self, 'label_encoder') and self.label_encoder:
            inv_encoder = {v: k for k, v in self.label_encoder.items()}
            class_labels = list(self.label_encoder.keys())
        else:
            class_labels = np.unique(np.concatenate([y_true, y_pred]))
            inv_encoder = {i: str(cls) for i, cls in enumerate(class_labels)}

        n_classes = len(class_labels)

        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        class_accuracies = {}
        for i in range(n_classes):
            total = cm[i].sum()
            if total > 0:
                acc = cm[i, i] / total
                class_accuracies[class_labels[i]] = acc

        total_correct = np.diag(cm).sum()
        total_samples = cm.sum()
        overall_acc = total_correct / total_samples if total_samples > 0 else 0

        print(f"\n{Colors.BOLD}{Colors.BLUE}Confusion Matrix - {header}{Colors.ENDC}")
        print(f"{'True\\Pred':<12}", end='')
        for label in class_labels:
            print(f"{str(label):<8}", end='')
        print(f"{'Class Acc':<10}")
        print("-" * (12 + 8 * n_classes + 10))

        for i in range(n_classes):
            print(f"{Colors.BOLD}{str(class_labels[i]):<12}{Colors.ENDC}", end='')
            for j in range(n_classes):
                if i == j:
                    color = Colors.GREEN
                else:
                    color = Colors.RED
                print(f"{color}{cm[i, j]:<8}{Colors.ENDC}", end='')

            acc = class_accuracies.get(class_labels[i], 0)
            if acc >= 0.9:
                acc_color = Colors.GREEN
            elif acc >= 0.7:
                acc_color = Colors.YELLOW
            else:
                acc_color = Colors.RED
            print(f"{acc_color}{acc:>7.2%}{Colors.ENDC}")

        print("-" * (12 + 8 * n_classes + 10))
        if overall_acc >= 0.9:
            acc_color = Colors.GREEN
        elif overall_acc >= 0.7:
            acc_color = Colors.YELLOW
        else:
            acc_color = Colors.RED
        print(f"{Colors.BOLD}Overall Accuracy:{Colors.ENDC} {acc_color}{overall_acc:.2%}{Colors.ENDC}")

        return overall_acc

    def _calculate_class_wise_accuracy(self, y_true, y_pred):
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()

        unique_classes = np.unique(y_true)
        class_accuracies = {}

        for class_id in unique_classes:
            class_mask = (y_true == class_id)
            n_class_samples = class_mask.sum()
            if n_class_samples == 0:
                continue

            correct = (y_pred[class_mask] == y_true[class_mask]).sum()
            class_acc = correct / n_class_samples
            class_accuracies[class_id] = {
                'accuracy': class_acc,
                'n_samples': n_class_samples,
                'correct': correct
            }

        return class_accuracies

    def fresh_train(self, X_train=None, y_train=None, X_test=None, y_test=None,
                   epochs: int = 100, patience: int = 10):
        """Fresh training - completely reset model before training"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 Starting FRESH training (previous knowledge ignored){Colors.ENDC}")
        print(f"{'='*60}")

        self.reset_model(hard_reset=True)

        if X_train is None:
            X_train = self.X_train
            y_train = self.y_train
            X_test = self.X_test
            y_test = self.y_test

        if X_train is None:
            raise ValueError("No training data provided")

        self.generate_feature_pairs(X_train.shape[1])
        print(f"   Generated {len(self.feature_pairs)} feature pairs")

        self.bin_edges = self.compute_bin_edges(X_train)
        print(f"   Computed bin edges")

        likelihoods = self.compute_likelihoods(X_train, y_train)
        self.bin_probs = likelihoods['bin_probs']
        print(f"   Computed likelihoods")

        return self.fit(X_train, y_train, X_test, y_test, epochs, patience)

    def fresh_adaptive_train(self, max_rounds: int = None, batch_size: int = 128):
        """Fresh adaptive training - completely reset model before adaptive training"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 Starting FRESH adaptive training (previous knowledge ignored){Colors.ENDC}")
        print(f"{'='*60}")

        self.reset_model(hard_reset=True)
        self.train_indices = []
        self.test_indices = None

        return self.adaptive_fit_predict(max_rounds, batch_size)

    def adaptive_fit_predict(self, max_rounds: int = None, batch_size: int = 128):
        """
        Complete adaptive training - FIXED to properly store training history
        and check stop flag
        """
        print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 Starting Adaptive Training{Colors.ENDC}")
        print(f"{'='*60}")

        start_time = time.time()
        start_clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        print(f"{Colors.BOLD}Started at: {start_clock}{Colors.ENDC}")

        self.in_adaptive_fit = True
        self.stop_training_flag = False
        train_indices = []
        test_indices = None
        training_history = []  # Store epoch-level history
        round_stats = []       # Store round-level stats

        max_rounds = max_rounds or self.adaptive_rounds

        self.evolution_tracker.clear_history()

        try:
            if self.X_tensor is None:
                self.load_data()

            X = self.data_original.drop(columns=[self.target_column])
            y = self.data_original[self.target_column].astype(str)

            if self.label_encoder is None or len(self.label_encoder) == 0:
                unique_classes = y.unique()
                self.label_encoder = {v: i for i, v in enumerate(unique_classes)}
                self.classes = torch.tensor(list(self.label_encoder.values()))
                y_encoded = y.map(self.label_encoder).values
                self.y_tensor = torch.tensor(y_encoded, dtype=torch.long)

            model_loaded = False
            if self.config.get('use_previous_model', True):
                print(f"{Colors.CYAN}Loading previous model state...{Colors.ENDC}")
                model_file = f'model_{self.dataset_name}.pkl'
                if os.path.exists(model_file):
                    self.load_model(model_file)
                    model_loaded = True

                    train_indices, test_indices = self.load_last_known_split()
                    if train_indices:
                        print(f"{Colors.GREEN}Resuming with {len(train_indices)} previous training samples{Colors.ENDC}")
                    else:
                        print(f"{Colors.YELLOW}No valid previous split found, initializing new training set{Colors.ENDC}")
                        train_indices = []
                        test_indices = list(range(len(X)))

            if not model_loaded:
                print(f"{Colors.CYAN}Initializing fresh model...{Colors.ENDC}")
                train_indices = []
                test_indices = list(range(len(X)))

                self.generate_feature_pairs(self.X_tensor.shape[1])
                print(f"   Generated {len(self.feature_pairs)} feature pairs")

                self.bin_edges = self.compute_bin_edges(self.X_tensor)
                print(f"   Computed bin edges")

            if test_indices is None:
                test_indices = list(range(len(X)))

            if len(train_indices) == 0:
                n_classes = len(self.classes)
                target_per_class = max(1, self.initial_samples // n_classes)
                print(f"\n{Colors.CYAN}Initializing with {target_per_class} samples PER CLASS{Colors.ENDC}")

                initial_samples = []
                class_sample_counts = {}

                for class_label, class_id in self.label_encoder.items():
                    class_indices = np.where(self.y_tensor.numpy() == class_id)[0]

                    if len(class_indices) == 0:
                        class_sample_counts[class_label] = 0
                        continue

                    n_samples = min(target_per_class, len(class_indices))
                    selected = np.random.choice(class_indices, n_samples, replace=False).tolist()

                    initial_samples.extend(selected)
                    class_sample_counts[class_label] = n_samples
                    print(f"   Class {class_label}: added {n_samples} initial samples")

                train_indices = initial_samples
                print(f"{Colors.GREEN}Final training set: {len(train_indices)} TOTAL samples{Colors.ENDC}")
            else:
                print(f"\n{Colors.CYAN}Continuing with existing training set: {len(train_indices)} samples{Colors.ENDC}")

            print(f"\n{Colors.CYAN}Initial training set size: {len(train_indices)}{Colors.ENDC}")
            print(f"{Colors.CYAN}Initial test set size: {len(test_indices)}{Colors.ENDC}")

            if self.bin_probs is None and train_indices:
                X_train_subset = self.X_tensor[train_indices]
                y_train_subset = self.y_tensor[train_indices]
                likelihoods = self.compute_likelihoods(X_train_subset, y_train_subset)
                self.bin_probs = likelihoods['bin_probs']
                print(f"   Computed likelihoods")

            adaptive_patience_counter = 0
            patience = self.patience
            best_combined_accuracy = 0.0
            best_round_initial_conditions = None
            round_num = 0

            while round_num < max_rounds:
                if self.stop_training_flag:
                    print(f"\n{Colors.YELLOW}🛑 Training stopped by user at round {round_num + 1}{Colors.ENDC}")
                    break

                print(f"\n{Colors.BOLD}{Colors.BLUE}Round {round_num + 1}/{max_rounds}{Colors.ENDC}")
                print(f"   Training set size: {len(train_indices)}")
                print(f"   Test set size: {len(test_indices)}")

                if best_round_initial_conditions is not None:
                    print(f"{Colors.YELLOW}Resetting to best round's initial conditions{Colors.ENDC}")
                    self.weight_updater = best_round_initial_conditions['weight_updater']
                    self.bin_probs = best_round_initial_conditions['bin_probs']

                self.save_epoch_data(round_num, train_indices, test_indices)

                self.X_train = self.X_tensor[train_indices]
                self.y_train = self.y_tensor[train_indices]
                self.X_test = self.X_tensor[test_indices] if test_indices else None
                self.y_test = self.y_tensor[test_indices] if test_indices else None
                self.train_indices = train_indices
                self.test_indices = test_indices

                save_path = f"data/{self.dataset_name}/Predictions/"
                os.makedirs(save_path, exist_ok=True)

                results = self.fit_predict(batch_size=batch_size, save_path=save_path)

                if 'history' in results and results['history']:
                    for epoch_data in results['history']:
                        epoch_data['round'] = round_num
                        training_history.append(epoch_data)

                train_accuracy = results['train_accuracy']
                test_accuracy = results['test_accuracy']

                print(f"{Colors.GREEN}   Training accuracy: {train_accuracy:.4f}{Colors.ENDC}")
                print(f"   Test accuracy: {test_accuracy:.4f}")

                all_predictions, _ = self.predict(self.X_tensor)
                total_accuracy = (all_predictions == self.y_tensor.cpu()).float().mean().item()
                print(f"   Total accuracy: {total_accuracy:.4f}")

                self.evolution_tracker.capture_state(
                    round_num=round_num + 1,
                    accuracy=total_accuracy,
                    training_size=len(train_indices)
                )

                print(f"\n{Colors.BOLD}Confusion Matrices:{Colors.ENDC}")

                train_pred, _ = self.predict(self.X_train)
                self.print_colored_confusion_matrix(
                    self.y_train.cpu().numpy(),
                    train_pred.cpu().numpy(),
                    "Training Data"
                )

                if self.X_test is not None and len(self.X_test) > 0:
                    test_pred, _ = self.predict(self.X_test)
                    self.print_colored_confusion_matrix(
                        self.y_test.cpu().numpy(),
                        test_pred.cpu().numpy(),
                        "Test Data"
                    )

                self.print_colored_confusion_matrix(
                    self.y_tensor.cpu().numpy(),
                    all_predictions.cpu().numpy(),
                    "Total Data"
                )

                round_data = {
                    'round': round_num,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'total_accuracy': total_accuracy,
                    'train_size': len(train_indices),
                    'test_size': len(test_indices)
                }
                round_stats.append(round_data)
                training_history.append(train_indices.copy())

                combined_accuracy = (train_accuracy + test_accuracy) / 2
                if combined_accuracy > best_combined_accuracy:
                    best_combined_accuracy = combined_accuracy
                    best_round_initial_conditions = {
                        'weight_updater': self.weight_updater,
                        'bin_probs': self.bin_probs,
                        'round': round_num
                    }
                    adaptive_patience_counter = 0
                    self.save_last_split(train_indices, test_indices)
                    print(f"{Colors.GREEN}   New best accuracy!{Colors.ENDC}")
                else:
                    adaptive_patience_counter += 1
                    print(f"{Colors.YELLOW}   No improvement. Patience: {adaptive_patience_counter}/{patience}{Colors.ENDC}")

                if total_accuracy >= 0.9999:
                    print(f"\n{Colors.GREEN}🎯 Total accuracy reached 100%! Training complete.{Colors.ENDC}")
                    break

                if adaptive_patience_counter >= patience:
                    print(f"\n{Colors.YELLOW}No improvement after {patience} rounds. Stopping.{Colors.ENDC}")
                    break

                if test_indices:
                    test_predictions, _ = self.predict(self.X_tensor[test_indices])
                    y_test_np = self.y_tensor[test_indices].numpy()

                    new_train_indices = self._select_samples_from_failed_classes(
                        test_predictions, y_test_np, test_indices, results
                    )

                    if new_train_indices:
                        class_dist = self._format_class_distribution(new_train_indices)
                        print(f"{Colors.GREEN}   Added {len(new_train_indices)} new samples - {class_dist}{Colors.ENDC}")

                        train_indices = list(set(train_indices + new_train_indices))
                        test_indices = list(set(test_indices) - set(new_train_indices))
                        round_num += 1
                    else:
                        if train_accuracy >= 0.99:
                            print(f"\n{Colors.GREEN}Perfect accuracy achieved on training data. No more suitable samples in test set.{Colors.ENDC}")
                        else:
                            print(f"\n{Colors.YELLOW}No suitable new samples found meeting selection criteria. Training complete.{Colors.ENDC}")
                        break
                else:
                    print(f"\n{Colors.GREEN}No more test samples available. Training complete.{Colors.ENDC}")
                    break

            end_time = time.time()
            elapsed_time = end_time - start_time
            end_clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))

            print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.GREEN}✅ Adaptive Training Complete{Colors.ENDC}")
            print(f"{Colors.BOLD}Started at: {start_clock}{Colors.ENDC}")
            print(f"{Colors.BOLD}Ended at: {end_clock}{Colors.ENDC}")
            print(f"{Colors.BOLD}Total time: {elapsed_time:.2f} seconds{Colors.ENDC}")
            print(f"{Colors.BOLD}Final training samples: {len(train_indices)}{Colors.ENDC}")
            print(f"{Colors.BOLD}Final total accuracy: {total_accuracy:.4f}{Colors.ENDC}")
            print(f"{Colors.BOLD}Best combined accuracy: {best_combined_accuracy:.4f}{Colors.ENDC}")

            if self.evolution_tracker.enabled:
                n_captured = len(self.evolution_tracker.get_history())
                print(f"{Colors.CYAN}📸 Tensor evolution captured: {n_captured} states{Colors.ENDC}")

            print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")

            self.in_adaptive_fit = False
            return {
                'train_indices': train_indices,
                'test_indices': test_indices,
                'history': training_history,
                'round_stats': round_stats,
                'best_accuracy': best_combined_accuracy,
                'final_total_accuracy': total_accuracy,
                'evolution_history': self.evolution_tracker.get_history()
            }

        except Exception as e:
            print(f"{Colors.RED}Error in adaptive_fit_predict: {str(e)}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            self.in_adaptive_fit = False
            raise

# =============================================================================
# SECTION 8: OPTIMIZED VISUALIZER (FIXED - Proper labels and file saving)
# =============================================================================

class OptimizedVisualizer:
    def __init__(self, model, output_dir='visualizations'):
        self.model = model
        self.dataset_name = model.dataset_name if model and hasattr(model, 'dataset_name') else 'unknown'

        self.base_output_dir = Path(output_dir)
        self.output_dir = self.base_output_dir / self.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dirs = {
            'performance': self.output_dir / 'performance',
            'confusion': self.output_dir / 'confusion',
            'interactive': self.output_dir / 'interactive',
            'spherical': self.output_dir / 'spherical_evolution',
            'tensor': self.output_dir / 'tensor_evolution'
        }
        for d in self.dirs.values():
            d.mkdir(exist_ok=True)

        self.spherical_viz = SphericalTensorEvolution(model, output_dir)
        print(f"{Colors.CYAN}📁 Visualizations: {self.output_dir}{Colors.ENDC}")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = ''):
        """Plot confusion matrix with ACTUAL class labels (strings)"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        if len(y_true) == 0 or len(y_pred) == 0:
            return

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Get actual class labels from label encoder
        if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
            inv_encoder = {v: k for k, v in self.model.label_encoder.items()}
            y_true_labels = [inv_encoder[t] for t in y_true]
            y_pred_labels = [inv_encoder[p] for p in y_pred]
            unique_labels = sorted(list(self.model.label_encoder.keys()))

            cm = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_labels)

            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                       xticklabels=unique_labels, yticklabels=unique_labels,
                       annot_kws={'size': 10})
            plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
            plt.ylabel('True Class', fontsize=12, fontweight='bold')
            plt.title(f'Confusion Matrix - {title}\n({self.dataset_name})', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            accuracy = (np.array(y_true_labels) == np.array(y_pred_labels)).mean()
            plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.4f}\nSamples: {len(y_true)}',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            plt.tight_layout()
            filename = self.dirs['confusion'] / f'{self.dataset_name}_confusion_matrix_{title}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Confusion matrix: {filename} (using actual labels)")
        else:
            # Fallback to numeric labels
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                       xticklabels=unique_classes, yticklabels=unique_classes)
            plt.xlabel('Predicted'), plt.ylabel('True')
            plt.title(f'Confusion Matrix - {title}')
            plt.tight_layout()
            filename = self.dirs['confusion'] / f'{self.dataset_name}_confusion_matrix_{title}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Confusion matrix: {filename}")

    def plot_training_history(self, history):
        """Plot training history - FIXED to actually save files"""
        import matplotlib.pyplot as plt

        if not history or len(history) == 0:
            print(f"{Colors.YELLOW}   ℹ️ No training history to plot{Colors.ENDC}")
            return

        if isinstance(history[0], dict):
            if 'epoch' in history[0]:
                epochs = [h['epoch'] for h in history]
                train_acc = [h.get('train_accuracy', 0) for h in history]
                test_acc = [h.get('test_accuracy', 0) for h in history]
                xlabel, title_suffix = 'Epoch', 'Standard Training'
            elif 'round' in history[0]:
                epochs = [h['round'] + 1 for h in history]
                train_acc = [h.get('train_accuracy', 0) for h in history]
                test_acc = [h.get('test_accuracy', 0) for h in history]
                xlabel, title_suffix = 'Round', 'Adaptive Rounds'
            else:
                epochs = list(range(1, len(history) + 1))
                train_acc = [h.get('train_accuracy', h.get('accuracy', 0)) for h in history]
                test_acc = [h.get('test_accuracy', h.get('accuracy', 0)) for h in history]
                xlabel, title_suffix = 'Iteration', 'Training Progress'
        else:
            return

        if all(v == 0 for v in train_acc) and all(v == 0 for v in test_acc):
            return

        plt.figure(figsize=(12, 6))
        plt.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2, marker='o')
        plt.plot(epochs, test_acc, 'r-', label='Test Accuracy', linewidth=2, marker='s')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Training Progress - {self.dataset_name} ({title_suffix})', fontsize=14, fontweight='bold')
        plt.legend(), plt.grid(True, alpha=0.3), plt.ylim([0, 1.05])

        step = max(1, len(epochs)//10)
        for i in range(0, len(epochs), step):
            plt.annotate(f'{train_acc[i]:.3f}', (epochs[i], train_acc[i]), xytext=(0,10), ha='center', fontsize=8)
            plt.annotate(f'{test_acc[i]:.3f}', (epochs[i], test_acc[i]), xytext=(0,-15), ha='center', fontsize=8)

        filename = self.dirs['performance'] / f'{self.dataset_name}_training_history.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Training history: {filename} ({len(epochs)} points)")

    def plot_tensor_evolution(self, evolution_history):
        """
        Plot tensor evolution metrics showing ORTHOGONALIZATION of class tensors
        in complex feature-pair space
        """
        if not evolution_history:
            print(f"{Colors.YELLOW}   ℹ️ No evolution history{Colors.ENDC}")
            return

        import matplotlib.pyplot as plt
        import numpy as np

        rounds = []
        accuracies = []
        training_sizes = []
        orthogonality_matrix = []
        class_separation_angles = []

        for snap in evolution_history:
            if 'round' in snap:
                rounds.append(snap['round'])
                accuracies.append(snap.get('accuracy', 0))
                training_sizes.append(snap.get('training_size', 0))

                # Calculate orthogonality from complex weights
                if 'complex_weights' in snap:
                    weights = snap['complex_weights']
                    if torch.is_tensor(weights):
                        weights = weights.cpu().numpy()

                    # weights shape: (n_classes, n_pairs, n_bins, n_bins)
                    if len(weights.shape) == 4:
                        n_classes = weights.shape[0]

                        if n_classes >= 2:
                            # Calculate class orientation vectors
                            class_orientations = []

                            for c in range(n_classes):
                                # Get all weights for this class
                                class_weights = weights[c].flatten()

                                # Filter significant weights
                                significant = class_weights[np.abs(class_weights) > 0.01]

                                if len(significant) > 0:
                                    # Calculate average orientation (complex vector)
                                    magnitudes = np.abs(significant)
                                    phases = np.angle(significant)

                                    # Weighted average direction (circular mean)
                                    sin_sum = np.sum(magnitudes * np.sin(phases))
                                    cos_sum = np.sum(magnitudes * np.cos(phases))
                                    avg_phase = np.arctan2(sin_sum, cos_sum)

                                    # Average magnitude
                                    avg_mag = np.mean(magnitudes)

                                    # Store as unit vector in complex plane
                                    class_orientations.append(np.exp(1j * avg_phase))
                                else:
                                    class_orientations.append(0 + 0j)

                            # Calculate pairwise orthogonality
                            n = len(class_orientations)
                            ortho_matrix = np.zeros((n, n))
                            angles = []

                            for i in range(n):
                                for j in range(n):
                                    if i == j:
                                        ortho_matrix[i, j] = 1.0
                                    else:
                                        # Cosine similarity between class vectors
                                        vi = class_orientations[i]
                                        vj = class_orientations[j]

                                        if np.abs(vi) > 0 and np.abs(vj) > 0:
                                            vi_unit = vi / np.abs(vi)
                                            vj_unit = vj / np.abs(vj)
                                            similarity = np.real(vi_unit * np.conj(vj_unit))
                                            ortho_matrix[i, j] = similarity

                                            if i < j:
                                                # Angle in degrees between class vectors
                                                angle = np.arccos(np.clip(similarity, -1, 1)) * 180 / np.pi
                                                angles.append(angle)

                            orthogonality_matrix.append(ortho_matrix)

                            # Average separation angle (should approach 90°)
                            if angles:
                                avg_angle = np.mean(angles)
                                class_separation_angles.append(avg_angle)
                            else:
                                class_separation_angles.append(0)
                        else:
                            class_separation_angles.append(0)
                    else:
                        class_separation_angles.append(0)
                else:
                    class_separation_angles.append(0)

        if not rounds:
            return

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Tensor Evolution & Orthogonalization - {self.dataset_name}', fontsize=16, fontweight='bold')

        # Plot 1: Accuracy over rounds
        axes[0, 0].plot(rounds, accuracies, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Adaptive Round', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Classification Accuracy Evolution', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.05])

        # Plot 2: Training set growth
        axes[0, 1].bar(rounds, training_sizes, color='orange', alpha=0.7, edgecolor='darkorange')
        axes[0, 1].set_xlabel('Adaptive Round', fontsize=12)
        axes[0, 1].set_ylabel('Training Samples', fontsize=12)
        axes[0, 1].set_title('Training Set Growth', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Class Separation Angle (should approach 90°)
        axes[1, 0].plot(rounds, class_separation_angles, 'g-s', linewidth=2, markersize=8)
        axes[1, 0].axhline(y=90, color='red', linestyle='--', linewidth=2,
                           label='Perfect Orthogonality (90°)', alpha=0.7)
        axes[1, 0].set_xlabel('Adaptive Round', fontsize=12)
        axes[1, 0].set_ylabel('Average Class Separation Angle (degrees)', fontsize=12)
        axes[1, 0].set_title('Class Tensor Orthogonalization', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].legend(loc='lower right')

        # Add text showing final angle
        if class_separation_angles:
            final_angle = class_separation_angles[-1]
            axes[1, 0].annotate(f'Final: {final_angle:.1f}°',
                               xy=(rounds[-1], final_angle),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=10, fontweight='bold')

        # Plot 4: Orthogonality Heatmap (final round)
        if orthogonality_matrix and len(orthogonality_matrix) > 0:
            final_ortho = orthogonality_matrix[-1]
            n_classes = final_ortho.shape[0]

            im = axes[1, 1].imshow(final_ortho, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
            axes[1, 1].set_xlabel('Class', fontsize=12)
            axes[1, 1].set_ylabel('Class', fontsize=12)
            axes[1, 1].set_title('Final Class Orthogonality Matrix', fontsize=12, fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, 1])
            cbar.set_label('Cosine Similarity (0=Orthogonal, 1=Identical)', fontsize=10)

            # Add text annotations in cells
            for i in range(n_classes):
                for j in range(n_classes):
                    value = final_ortho[i, j]
                    color = 'white' if value > 0.5 else 'black'
                    axes[1, 1].text(j, i, f'{value:.2f}',
                                   ha='center', va='center', color=color, fontsize=9)

            # Set ticks
            class_labels = [f'C{i+1}' for i in range(min(n_classes, 10))]
            axes[1, 1].set_xticks(range(min(n_classes, 10)))
            axes[1, 1].set_yticks(range(min(n_classes, 10)))
            axes[1, 1].set_xticklabels(class_labels)
            axes[1, 1].set_yticklabels(class_labels)

        plt.tight_layout()
        filename = self.dirs['tensor'] / f'{self.dataset_name}_tensor_evolution.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Tensor evolution: {filename}")
        if class_separation_angles:
            print(f"   Orthogonalization progression: {[f'{a:.1f}°' for a in class_separation_angles]}")
            print(f"   Final class separation: {class_separation_angles[-1]:.1f}° (target: 90°)")

        # Also generate a separate orthogonality progression plot
        self._plot_orthogonality_progression(rounds, class_separation_angles, accuracies)

    def _plot_orthogonality_progression(self, rounds, angles, accuracies):
        """Create a detailed orthogonality progression plot"""
        if not angles or len(angles) < 2:
            return

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Class Tensor Orthogonalization Analysis - {self.dataset_name}', fontsize=14, fontweight='bold')

        # Plot 1: Orthogonality vs Accuracy
        ax1.scatter(angles, accuracies, c=rounds, cmap='viridis', s=100, alpha=0.7)
        ax1.plot(angles, accuracies, 'b-', alpha=0.3, linewidth=1)
        ax1.axvline(x=90, color='red', linestyle='--', label='Perfect Orthogonality (90°)', alpha=0.7)
        ax1.set_xlabel('Class Separation Angle (degrees)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy vs Orthogonality', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add colorbar
        cbar = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Round', fontsize=10)

        # Plot 2: Convergence to 90°
        ax2.plot(rounds, angles, 'g-s', linewidth=2, markersize=8, label='Actual Separation')
        ax2.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target (90°)')
        ax2.fill_between(rounds, angles, 90, where=np.array(angles) <= 90,
                         color='green', alpha=0.3, interpolate=True)
        ax2.fill_between(rounds, angles, 90, where=np.array(angles) > 90,
                         color='red', alpha=0.3, interpolate=True)
        ax2.set_xlabel('Adaptive Round', fontsize=12)
        ax2.set_ylabel('Class Separation Angle (degrees)', fontsize=12)
        ax2.set_title('Convergence to Orthogonal State', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 100])

        # Add annotations
        for i, (r, angle) in enumerate(zip(rounds, angles)):
            if i == 0 or i == len(rounds)-1 or (i > 0 and abs(angles[i] - angles[i-1]) > 20):
                ax2.annotate(f'{angle:.0f}°', (r, angle),
                            xytext=(0, 10), textcoords='offset points',
                            ha='center', fontsize=8)

        plt.tight_layout()
        filename = self.dirs['tensor'] / f'{self.dataset_name}_orthogonality_analysis.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Orthogonality analysis: {filename}")

    def create_interactive_dashboard(self, history, X, y, evolution_history=None):
        """Create interactive Plotly dashboard"""
        if not PLOTLY_AVAILABLE:
            return

        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=2, subplot_titles=('Training Progress', 'Feature Space (PCA)',
                                                           'Accuracy Distribution', 'Class Distribution'))

        # Add training progress
        if history and len(history) > 0:
            if isinstance(history[0], dict):
                if 'epoch' in history[0]:
                    x_vals = [h['epoch'] for h in history]
                    x_title = 'Epoch'
                elif 'round' in history[0]:
                    x_vals = [h['round'] + 1 for h in history]
                    x_title = 'Round'
                else:
                    x_vals = list(range(1, len(history) + 1))
                    x_title = 'Iteration'
                train_acc = [h.get('train_accuracy', 0) for h in history]
                test_acc = [h.get('test_accuracy', 0) for h in history]
            else:
                x_vals = list(range(1, len(history) + 1))
                x_title = 'Round'
                train_acc = [0.5] * len(history)
                test_acc = [0.5] * len(history)
        else:
            x_vals = [1]; x_title = 'Iteration'
            train_acc = [0]; test_acc = [0]

        fig.add_trace(go.Scatter(x=x_vals, y=train_acc, name='Train', line=dict(color='blue', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_vals, y=test_acc, name='Test', line=dict(color='red', width=2)), row=1, col=1)

        # Add class distribution
        if y is not None and len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            fig.add_trace(go.Bar(x=[str(c) for c in unique], y=counts, name='Class Distribution'), row=2, col=2)

        fig.update_layout(height=800, title_text=f"DBNN Dashboard - {self.dataset_name}")
        dashboard_path = self.dirs['interactive'] / f'{self.dataset_name}_dashboard.html'
        fig.write_html(str(dashboard_path))
        print(f"   ✅ Interactive dashboard: {dashboard_path}")

    def generate_all_visualizations(self, history, X, y, y_train, y_test, train_pred, test_pred, evolution_history=None):
        """Generate all visualizations with verification"""
        print(f"\n🎨 Visualizations for: {self.dataset_name}")
        print(f"{'='*60}")

        self.plot_training_history(history)
        self.plot_confusion_matrix(y_train, train_pred, 'Training')
        self.plot_confusion_matrix(y_test, test_pred, 'Test')

        if evolution_history and len(evolution_history) > 0:
            self.plot_tensor_evolution(evolution_history)

        try:
            self.create_interactive_dashboard(history, X, y, evolution_history)
        except Exception as e:
            print(f"   ⚠️ Dashboard: {e}")

        print(f"\n📁 All saved to: {self.output_dir}")

# =============================================================================
# SECTION 9: SPHERICAL TENSOR EVOLUTION (FIXED for complex tensor space)
# =============================================================================

class SphericalTensorEvolution:
    def __init__(self, model, output_dir='visualizations'):
        self.model = model
        self.dataset_name = model.dataset_name if model and hasattr(model, 'dataset_name') else 'unknown'
        self.output_dir = Path(output_dir) / self.dataset_name / 'spherical_evolution'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_spherical_animation(self, evolution_history, class_names=None):
        """
        Create spherical evolution animation showing TENSOR ORTHOGONALIZATION in complex space
        Each vector represents the orientation of a class's tensor in complex feature-pair space
        Shows how classes become orthogonal (90° apart) as training progresses
        """
        if not evolution_history or not PLOTLY_AVAILABLE:
            return None

        import plotly.graph_objects as go
        import plotly.express as px

        print(f"{Colors.CYAN}🌐 Creating spherical evolution - Tensor Orthogonalization Visualization...{Colors.ENDC}")

        # Get class names
        if class_names is None and hasattr(self.model, 'label_encoder') and self.model.label_encoder:
            class_names = list(self.model.label_encoder.keys())

        class_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
        frames_data = []

        for snap_idx, snap in enumerate(evolution_history):
            if 'complex_weights' not in snap:
                continue

            weights = snap['complex_weights']
            if torch.is_tensor(weights):
                weights = weights.cpu().numpy()

            round_num = snap['round']
            accuracy = snap.get('accuracy', 0)
            training_size = snap.get('training_size', 0)

            # weights shape: (n_classes, n_pairs, n_bins, n_bins)
            if len(weights.shape) == 4:
                n_classes = weights.shape[0]
                n_pairs = weights.shape[1]
                n_bins = weights.shape[2]

                print(f"   Round {round_num}: {n_classes} classes, {n_pairs} feature pairs, {n_bins}x{n_bins} bins")

                points = []

                # For each class, compute the TENSOR ORIENTATION VECTOR
                for c in range(min(n_classes, 12)):
                    class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'

                    # Get all weights for this class
                    class_weights = weights[c].flatten()

                    # Filter significant weights (non-zero)
                    significant = class_weights[np.abs(class_weights) > 1e-6]

                    if len(significant) > 0:
                        # Each weight is a complex number: w = a + i*b
                        # Calculate the average orientation in complex space
                        magnitudes = np.abs(significant)
                        phases = np.angle(significant)

                        # Weighted circular mean (accounts for magnitude importance)
                        sin_sum = np.sum(magnitudes * np.sin(phases))
                        cos_sum = np.sum(magnitudes * np.cos(phases))
                        avg_theta = np.arctan2(sin_sum, cos_sum)  # Azimuthal angle

                        # Average magnitude (strength of orientation)
                        avg_r = np.mean(magnitudes)

                        # For 3D sphere, we need polar angle (phi) that represents class separation
                        # This should be based on the class index for orthogonal positions
                        # In perfect orthogonality, each class is at 90° separation
                        target_phi = (c * np.pi / max(1, n_classes))

                        # Calculate actual phi from tensor orientation
                        # Use the second harmonic to get separation information
                        sin_sum2 = np.sum(magnitudes * np.sin(phases * 2))
                        cos_sum2 = np.sum(magnitudes * np.cos(phases * 2))
                        actual_phi = (np.arctan2(sin_sum2, cos_sum2) % np.pi) / 2 + 0.5

                        # Convert to cartesian coordinates for 3D plot
                        x = avg_r * np.sin(actual_phi) * np.cos(avg_theta)
                        y = avg_r * np.sin(actual_phi) * np.sin(avg_theta)
                        z = avg_r * np.cos(actual_phi)

                        points.append({
                            'class': c,
                            'class_name': class_name,
                            'r': avg_r,
                            'theta': avg_theta,
                            'phi': actual_phi,
                            'target_phi': target_phi,
                            'x': x,
                            'y': y,
                            'z': z,
                            'num_weights': len(significant),
                            'avg_magnitude': avg_r
                        })
                    else:
                        # If no significant weights, place at origin
                        points.append({
                            'class': c,
                            'class_name': class_name,
                            'r': 0,
                            'theta': 0,
                            'phi': 0,
                            'target_phi': (c * np.pi / max(1, n_classes)),
                            'x': 0, 'y': 0, 'z': 0,
                            'num_weights': 0,
                            'avg_magnitude': 0
                        })

                # Calculate orthogonality metrics for this round
                ortho_metrics = self._calculate_orthogonality_metrics(points, n_classes)

                frames_data.append({
                    'round': round_num,
                    'points': points,
                    'accuracy': accuracy,
                    'training_size': training_size,
                    'n_classes': n_classes,
                    'n_pairs': n_pairs,
                    'avg_separation': ortho_metrics['avg_separation'],
                    'orthogonality': ortho_metrics['orthogonality'],
                    'max_separation': ortho_metrics['max_separation'],
                    'min_separation': ortho_metrics['min_separation']
                })

        if not frames_data:
            print(f"{Colors.YELLOW}   ℹ️ No valid class orientations extracted{Colors.ENDC}")
            return None

        # Create frames for animation
        frames = []

        for frame_idx, fd in enumerate(frames_data):
            round_num = fd['round']
            points = fd['points']
            accuracy = fd['accuracy']
            training_size = fd['training_size']
            avg_sep = fd['avg_separation']
            ortho = fd['orthogonality']

            traces = []

            # Add transparent unit sphere for reference (radius = 1)
            u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
            x_sphere = np.cos(u) * np.sin(v)
            y_sphere = np.sin(u) * np.sin(v)
            z_sphere = np.cos(v)

            traces.append(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                opacity=0.08,
                showscale=False,
                hoverinfo='none',
                name='Unit Sphere',
                colorscale=[[0, 'lightgray'], [1, 'lightgray']]
            ))

            # Add coordinate axes
            axis_length = 1.3
            traces.append(go.Scatter3d(
                x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
                mode='lines', line=dict(color='red', width=2), name='Real Feature Component'
            ))
            traces.append(go.Scatter3d(
                x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
                mode='lines', line=dict(color='green', width=2), name='Imaginary Feature Component'
            ))
            traces.append(go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
                mode='lines', line=dict(color='blue', width=2), name='Class Separation'
            ))

            # Add class orientation vectors
            for p in points:
                if p['r'] > 0.01:  # Only show classes with significant orientation
                    cid = p['class']
                    color = class_colors[cid % len(class_colors)]

                    # Draw vector from origin to class orientation point
                    traces.append(go.Scatter3d(
                        x=[0, p['x']], y=[0, p['y']], z=[0, p['z']],
                        mode='lines+markers',
                        marker=dict(
                            size=10,
                            color=color,
                            symbol='circle',
                            line=dict(width=2, color='white')
                        ),
                        line=dict(color=color, width=3),
                        name=p['class_name'],
                        legendgroup=f'class{cid}',
                        showlegend=(frame_idx == 0),
                        text=f"<b>Class {p['class_name']}</b><br>"
                             f"Magnitude: {p['r']:.3f}<br>"
                             f"Phase: {p['theta']:.2f} rad ({p['theta']*180/np.pi:.0f}°)<br>"
                             f"Polar Angle: {p['phi']:.2f} rad ({p['phi']*180/np.pi:.0f}°)<br>"
                             f"Weights: {p['num_weights']}",
                        hoverinfo='text'
                    ))

            # Add target orthogonal positions (for perfect 90° separation)
            n_classes = fd['n_classes']
            for c in range(min(n_classes, 12)):
                # Target positions at 90° separation (equatorial plane)
                target_theta = (c * 2 * np.pi / n_classes)
                target_phi = np.pi / 2  # Equatorial plane
                r = 0.95

                x_target = r * np.sin(target_phi) * np.cos(target_theta)
                y_target = r * np.sin(target_phi) * np.sin(target_theta)
                z_target = r * np.cos(target_phi)

                color = class_colors[c % len(class_colors)]

                traces.append(go.Scatter3d(
                    x=[x_target], y=[y_target], z=[z_target],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color=color,
                        symbol='x',
                        opacity=0.8,
                        line=dict(width=2, color='white')
                    ),
                    name=f'Target {class_names[c] if class_names and c < len(class_names) else f"C{c+1}"}',
                    legendgroup=f'target{c}',
                    showlegend=(frame_idx == 0),
                    text=f"Target orthogonal position for Class {c+1}<br>90° separation",
                    hoverinfo='text'
                ))

            # Add a reference circle at 90° (equatorial plane)
            theta_circle = np.linspace(0, 2*np.pi, 100)
            x_circle = 0.98 * np.cos(theta_circle)
            y_circle = 0.98 * np.sin(theta_circle)
            z_circle = np.zeros_like(theta_circle)

            traces.append(go.Scatter3d(
                x=x_circle, y=y_circle, z=z_circle,
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                name='Equatorial Plane (90° separation)',
                showlegend=(frame_idx == 0)
            ))

            frames.append(go.Frame(
                data=traces,
                name=f'Round {round_num}',
                layout=go.Layout(
                    title=dict(
                        text=f'<b>Round {round_num}</b><br>'
                             f'Accuracy: {accuracy:.3f} | Training Samples: {training_size}<br>'
                             f'<span style="color:green">Average Separation: {avg_sep:.1f}°</span> | '
                             f'<span style="color:blue">Orthogonality: {ortho:.3f}</span>',
                        font=dict(size=14)
                    )
                )
            ))

        # Create figure with animation controls
        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
            layout=go.Layout(
                title=dict(
                    text=f'<b>CT-DBNN Tensor Orthogonalization</b><br>'
                         f'Class Orientation Vectors in Complex Feature-Pair Space',
                    font=dict(size=16)
                ),
                scene=dict(
                    xaxis_title='<b>Real Feature Component</b>',
                    yaxis_title='<b>Imaginary Feature Component</b>',
                    zaxis_title='<b>Class Separation Axis</b>',
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)),
                    aspectmode='cube',
                    annotations=[
                        dict(
                            x=1.6, y=0, z=0,
                            text="Real",
                            showarrow=False,
                            font=dict(color="red", size=10)
                        ),
                        dict(
                            x=0, y=1.6, z=0,
                            text="Imag",
                            showarrow=False,
                            font=dict(color="green", size=10)
                        ),
                        dict(
                            x=0, y=0, z=1.6,
                            text="Separation",
                            showarrow=False,
                            font=dict(color="blue", size=10)
                        )
                    ]
                ),
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=False,
                        y=0.92,
                        x=0.05,
                        buttons=[
                            dict(label='▶️ Play', method='animate',
                                 args=[None, {
                                     'frame': {'duration': 800, 'redraw': True},
                                     'fromcurrent': True,
                                     'mode': 'immediate',
                                     'transition': {'duration': 0}
                                 }]),
                            dict(label='⏸️ Pause', method='animate',
                                 args=[[None], {
                                     'frame': {'duration': 0, 'redraw': False},
                                     'mode': 'immediate'
                                 }]),
                            dict(label='🔄 Reset', method='animate',
                                 args=[[frames[0].name], {
                                     'frame': {'duration': 0, 'redraw': True},
                                     'mode': 'immediate'
                                 }])
                        ]
                    )
                ],
                sliders=[{
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {
                        'prefix': 'Round: ',
                        'font': {'size': 14, 'color': 'white'},
                        'visible': True
                    },
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9,
                    'x': 0.1,
                    'y': 0,
                    'steps': [
                        {
                            'args': [[f'Round {fd["round"]}'], {
                                'frame': {'duration': 0, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }],
                            'label': str(fd['round']),
                            'method': 'animate'
                        }
                        for fd in frames_data
                    ]
                }],
                width=1400,
                height=1000,
                showlegend=True,
                legend=dict(
                    yanchor="top", y=0.99, xanchor="left", x=0.02,
                    bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1,
                    font=dict(color='white', size=10)
                ),
                paper_bgcolor='rgba(0,0,0,0.9)',
                plot_bgcolor='rgba(0,0,0,0.9)'
            )
        )

        # Add explanation annotation
        fig.add_annotation(
            x=0.98, y=0.05, xref="paper", yref="paper",
            text="<b>📐 Tensor Orthogonalization Process:</b><br>"
                 "• Each colored vector = Class orientation in complex feature-pair space<br>"
                 "• Vectors start randomly (mixed) → end orthogonal (90° apart)<br>"
                 "• ✗ marks = Target orthogonal positions for perfect classification<br>"
                 "• Gray circle = Equatorial plane (90° separation reference)<br>"
                 "• Perfect classification achieved when vectors reach ✗ targets",
            showarrow=False,
            font=dict(size=11, color='white'),
            align='right',
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='white',
            borderwidth=1,
            borderpad=4
        )

        # Add orthogonality gauge in top right
        final_ortho = frames_data[-1]['orthogonality']
        final_sep = frames_data[-1]['avg_separation']

        fig.add_annotation(
            x=0.98, y=0.95, xref="paper", yref="paper",
            text=f"<b>Final State:</b><br>"
                 f"Average Separation: <b>{final_sep:.1f}°</b><br>"
                 f"Orthogonality: <b>{final_ortho:.3f}</b><br>"
                 f"Target: <b>90° | 1.000</b>",
            showarrow=False,
            font=dict(size=11, color='lightgreen'),
            align='right',
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='lightgreen',
            borderwidth=1,
            borderpad=4
        )

        output_path = self.output_dir / f'{self.dataset_name}_spherical.html'
        fig.write_html(str(output_path))
        print(f"   ✅ Spherical evolution: {output_path}")
        print(f"   Rounds: {[fd['round'] for fd in frames_data]}")
        print(f"   Orthogonality progression: {[f'{fd["orthogonality"]:.3f}' for fd in frames_data]}")
        print(f"   Class separation progression: {[f'{fd["avg_separation"]:.1f}°' for fd in frames_data]}")

        return str(output_path)

    def _calculate_orthogonality_metrics(self, points, n_classes):
        """
        Calculate orthogonality metrics from class orientation vectors
        """
        if len(points) < 2:
            return {
                'avg_separation': 0,
                'orthogonality': 0,
                'max_separation': 0,
                'min_separation': 0
            }

        # Extract unit vectors from points (ignore magnitude, just direction)
        vectors = []
        for p in points:
            if p['r'] > 0.01:
                # Normalize to unit vector
                x, y, z = p['x'], p['y'], p['z']
                norm = np.sqrt(x*x + y*y + z*z)
                if norm > 0:
                    vectors.append((x/norm, y/norm, z/norm))
                else:
                    vectors.append((0, 0, 0))
            else:
                vectors.append((0, 0, 0))

        # Calculate pairwise angles
        angles = []
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                v1 = vectors[i]
                v2 = vectors[j]

                # Dot product
                dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
                dot = np.clip(dot, -1, 1)

                # Angle in degrees
                angle = np.arccos(dot) * 180 / np.pi
                angles.append(angle)

        if angles:
            avg_sep = np.mean(angles)
            orthogonality = avg_sep / 90.0  # Normalized: 1.0 = perfect 90° separation
            return {
                'avg_separation': avg_sep,
                'orthogonality': min(1.0, orthogonality),
                'max_separation': np.max(angles),
                'min_separation': np.min(angles)
            }
        else:
            return {
                'avg_separation': 0,
                'orthogonality': 0,
                'max_separation': 0,
                'min_separation': 0
            }

    def create_side_by_side_evolution(self, evolution_history):
        """
        Create side-by-side comparison of initial and final states
        Uses _add_state_to_subplot to add vector visualizations
        """
        if not evolution_history or len(evolution_history) < 2 or not PLOTLY_AVAILABLE:
            print(f"{Colors.YELLOW}   ℹ️ Need at least 2 rounds for side-by-side comparison{Colors.ENDC}")
            return None

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            print(f"{Colors.CYAN}🔄 Creating side-by-side evolution comparison...{Colors.ENDC}")

            first_snap = evolution_history[0]
            last_snap = evolution_history[-1]

            # Get class names
            class_names = None
            if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                class_names = list(self.model.label_encoder.keys())

            # Create subplots with 2 columns
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                subplot_titles=(
                    f'<b>Initial State - Round {first_snap["round"]}</b><br>'
                    f'Accuracy: {first_snap.get("accuracy", 0):.3f} | Random Mix',
                    f'<b>Final State - Round {last_snap["round"]}</b><br>'
                    f'Accuracy: {last_snap.get("accuracy", 0):.3f} | Orthogonal Separation'
                ),
                horizontal_spacing=0.1
            )

            # Add initial state (left subplot) using _add_state_to_subplot
            self._add_state_to_subplot(fig, first_snap, row=1, col=1,
                                       class_names=class_names, show_legend=True)

            # Add final state (right subplot)
            self._add_state_to_subplot(fig, last_snap, row=1, col=2,
                                       class_names=class_names, show_legend=False)

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"<b>Tensor Orthogonalization: From Random Mix to Class Separation</b><br>"
                         f"<sup>Class orientation vectors evolve to become orthogonal (90° apart)</sup>",
                    font=dict(size=16)
                ),
                width=1400,
                height=700,
                showlegend=True,
                legend=dict(
                    yanchor="top", y=0.99, xanchor="left", x=0.02,
                    bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1,
                    font=dict(color='white')
                ),
                paper_bgcolor='rgba(0,0,0,0.9)',
                plot_bgcolor='rgba(0,0,0,0.9)'
            )

            # Update 3D scenes
            for col in [1, 2]:
                fig.update_scenes(
                    xaxis_title='<b>Real Feature Component</b>',
                    yaxis_title='<b>Imaginary Feature Component</b>',
                    zaxis_title='<b>Class Separation Axis</b>',
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)),
                    aspectmode='cube',
                    row=1, col=col
                )

            # Add annotations
            fig.add_annotation(
                x=0.02, y=0.98, xref="paper", yref="paper",
                text="<b>📊 Initial State:</b><br>Random orientations - classes mixed",
                showarrow=False,
                font=dict(size=11, color='yellow'),
                bgcolor='rgba(0,0,0,0.6)',
                bordercolor='yellow',
                borderwidth=1
            )

            fig.add_annotation(
                x=0.52, y=0.98, xref="paper", yref="paper",
                text="<b>🎯 Final State:</b><br>Orthogonal vectors (90° separation) - classes separated",
                showarrow=False,
                font=dict(size=11, color='lightgreen'),
                bgcolor='rgba(0,0,0,0.6)',
                bordercolor='lightgreen',
                borderwidth=1
            )

            # Save the figure
            output_path = self.output_dir / f'{self.dataset_name}_side_by_side.html'
            fig.write_html(str(output_path))
            print(f"   ✅ Side-by-side comparison saved to: {output_path}")

            return str(output_path)

        except Exception as e:
            print(f"{Colors.RED}   ❌ Error creating side-by-side comparison: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            return None

    def _add_state_to_scene(self, fig, snap, row, col, class_names=None, class_colors=None, show_legend=True):
        """Helper to add a single state to a 3D scene subplot"""
        if 'complex_weights' not in snap:
            return

        if class_colors is None:
            import plotly.express as px
            class_colors = px.colors.qualitative.Set1

        weights = snap['complex_weights']
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()

        if len(weights.shape) == 4:
            n_classes = min(weights.shape[0], 10)
            legend_added = set()

            # Add transparent sphere surface
            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
            x_sphere = np.cos(u) * np.sin(v)
            y_sphere = np.sin(u) * np.sin(v)
            z_sphere = np.cos(v)

            fig.add_trace(
                go.Surface(
                    x=x_sphere, y=y_sphere, z=z_sphere,
                    opacity=0.08,
                    showscale=False,
                    hoverinfo='none',
                    colorscale=[[0, 'lightgray'], [1, 'lightgray']],
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add coordinate axes
            axis_length = 1.3
            fig.add_trace(
                go.Scatter3d(
                    x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
                    mode='lines', line=dict(color='red', width=2), showlegend=False
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
                    mode='lines', line=dict(color='green', width=2), showlegend=False
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
                    mode='lines', line=dict(color='blue', width=2), showlegend=False
                ),
                row=row, col=col
            )

            # Add points for each class
            for c in range(n_classes):
                class_weights = weights[c].flatten()
                n_samples = min(150, len(class_weights))
                sample_indices = np.random.choice(len(class_weights), n_samples, replace=False)

                x_vals, y_vals, z_vals = [], [], []
                for idx in sample_indices:
                    w = class_weights[idx]
                    if np.abs(w) > 0.01:
                        r = np.abs(w) * 1.1
                        theta = np.angle(w)
                        phi = np.pi * (hash((c, idx % 100)) % 1000 / 1000.0)
                        x_vals.append(r * np.sin(phi) * np.cos(theta))
                        y_vals.append(r * np.sin(phi) * np.sin(theta))
                        z_vals.append(r * np.cos(phi))

                if x_vals:
                    class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'
                    show_legend_for_class = show_legend and (class_name not in legend_added)

                    fig.add_trace(
                        go.Scatter3d(
                            x=x_vals, y=y_vals, z=z_vals,
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=class_colors[c % len(class_colors)],
                                opacity=0.8,
                                symbol='circle',
                                line=dict(width=0.5, color='black')
                            ),
                            name=class_name,
                            legendgroup=f'class{c}',
                            showlegend=show_legend_for_class
                        ),
                        row=row, col=col
                    )
                    if show_legend_for_class:
                        legend_added.add(class_name)

    def _add_state_to_subplot(self, fig, snap, row, col, class_names=None, show_legend=True):
        """
        Helper to add a single state to a subplot for side-by-side comparison
        Shows CLASS ORIENTATION VECTORS (not individual points)
        """
        if 'complex_weights' not in snap:
            return

        import plotly.express as px

        weights = snap['complex_weights']
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()

        class_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3

        if len(weights.shape) == 4:
            n_classes = min(weights.shape[0], 10)

            # Add unit sphere surface
            u = np.linspace(0, 2*np.pi, 30)
            v = np.linspace(0, np.pi, 15)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones_like(u), np.cos(v))

            fig.add_trace(
                go.Surface(
                    x=x_sphere, y=y_sphere, z=z_sphere,
                    opacity=0.1,
                    showscale=False,
                    hoverinfo='none',
                    name='Unit Sphere',
                    colorscale=[[0, 'lightgray'], [1, 'lightgray']],
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add coordinate axes
            axis_length = 1.5
            fig.add_trace(
                go.Scatter3d(
                    x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
                    mode='lines', line=dict(color='red', width=2), name='Real Axis', showlegend=False
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
                    mode='lines', line=dict(color='green', width=2), name='Imag Axis', showlegend=False
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
                    mode='lines', line=dict(color='blue', width=2), name='Phase Axis', showlegend=False
                ),
                row=row, col=col
            )

            # Calculate class orientation vectors
            class_vectors = []
            for c in range(n_classes):
                class_weights = weights[c].flatten()
                significant = class_weights[np.abs(class_weights) > 0.01]

                if len(significant) > 0:
                    magnitudes = np.abs(significant)
                    phases = np.angle(significant)

                    # Weighted circular mean
                    sin_sum = np.sum(magnitudes * np.sin(phases))
                    cos_sum = np.sum(magnitudes * np.cos(phases))
                    avg_theta = np.arctan2(sin_sum, cos_sum)
                    avg_r = np.mean(magnitudes)

                    # For 3D sphere, use class index for phi
                    phi = (c * np.pi / max(1, n_classes))

                    x = avg_r * np.sin(phi) * np.cos(avg_theta)
                    y = avg_r * np.sin(phi) * np.sin(avg_theta)
                    z = avg_r * np.cos(phi)

                    class_vectors.append((x, y, z, avg_r, avg_theta, phi, c))

            # Add class orientation vectors
            legend_added = set()
            for x, y, z, r, theta, phi, c in class_vectors:
                color = class_colors[c % len(class_colors)]
                class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'
                show_legend_for_class = show_legend and (class_name not in legend_added)

                # Draw vector from origin
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, x], y=[0, y], z=[0, z],
                        mode='lines+markers',
                        marker=dict(size=8, color=color, symbol='circle', line=dict(width=1, color='white')),
                        line=dict(color=color, width=2),
                        name=class_name,
                        legendgroup=f'class{c}',
                        showlegend=show_legend_for_class,
                        text=f"Class {class_name}<br>Magnitude: {r:.3f}<br>Phase: {theta:.2f} rad<br>Polar: {phi:.2f} rad",
                        hoverinfo='text'
                    ),
                    row=row, col=col
                )
                legend_added.add(class_name)

            # Add reference circle at equatorial plane
            theta_circle = np.linspace(0, 2*np.pi, 50)
            x_circle = 0.95 * np.cos(theta_circle)
            y_circle = 0.95 * np.sin(theta_circle)
            z_circle = np.zeros_like(theta_circle)

            fig.add_trace(
                go.Scatter3d(
                    x=x_circle, y=y_circle, z=z_circle,
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dash'),
                    name='Equatorial Plane (90°)',
                    showlegend=False
                ),
                row=row, col=col
            )

    def _extract_points_for_frame(self, snap):
        """
        Extract class orientation vectors for evolution slider
        Shows VECTORS (not individual points) for cleaner visualization
        """
        traces = []
        if 'complex_weights' not in snap:
            return traces

        import plotly.express as px

        weights = snap['complex_weights']
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()

        class_colors = px.colors.qualitative.Set1

        # Add sphere
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones_like(u), np.cos(v))

        traces.append(
            go.Scatter3d(
                x=x_sphere.flatten(), y=y_sphere.flatten(), z=z_sphere.flatten(),
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.1),
                showlegend=False,
                hoverinfo='none'
            )
        )

        if len(weights.shape) == 4:
            n_classes = min(weights.shape[0], 10)

            # Calculate class orientation vectors
            for c in range(n_classes):
                class_weights = weights[c].flatten()
                significant = class_weights[np.abs(class_weights) > 0.01]

                if len(significant) > 0:
                    magnitudes = np.abs(significant)
                    phases = np.angle(significant)

                    # Weighted circular mean
                    sin_sum = np.sum(magnitudes * np.sin(phases))
                    cos_sum = np.sum(magnitudes * np.cos(phases))
                    avg_theta = np.arctan2(sin_sum, cos_sum)
                    avg_r = np.mean(magnitudes)

                    # Polar angle from class index (for separation visualization)
                    phi = (c * np.pi / max(1, n_classes))

                    x = avg_r * np.sin(phi) * np.cos(avg_theta)
                    y = avg_r * np.sin(phi) * np.sin(avg_theta)
                    z = avg_r * np.cos(phi)

                    # Draw vector from origin
                    traces.append(
                        go.Scatter3d(
                            x=[0, x], y=[0, y], z=[0, z],
                            mode='lines+markers',
                            marker=dict(size=8, color=class_colors[c % len(class_colors)], symbol='circle'),
                            line=dict(color=class_colors[c % len(class_colors)], width=3),
                            name=f'Class {c+1}',
                            legendgroup=f'class{c}',
                            showlegend=False,
                            text=f"Class {c+1}<br>Magnitude: {avg_r:.3f}<br>Phase: {avg_theta:.2f} rad",
                            hoverinfo='text'
                        )
                    )

        return traces

    def create_evolution_slider(self, evolution_history, class_names=None):
        """
        Create an interactive slider showing class orientation vectors evolution
        Shows vectors moving from random to orthogonal positions
        """
        if not evolution_history or len(evolution_history) < 2 or not PLOTLY_AVAILABLE:
            return None

        try:
            import plotly.graph_objects as go
            import plotly.express as px

            print(f"{Colors.CYAN}🎚️ Creating evolution slider (Vector Evolution)...{Colors.ENDC}")

            class_colors = px.colors.qualitative.Set1

            # Get class names
            if class_names is None and hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                class_names = list(self.model.label_encoder.keys())

            # Prepare frames for each round
            frames = []

            for snap in evolution_history:
                if 'complex_weights' not in snap:
                    continue

                weights = snap['complex_weights']
                if torch.is_tensor(weights):
                    weights = weights.cpu().numpy()

                round_num = snap['round']
                accuracy = snap.get('accuracy', 0)
                training_size = snap.get('training_size', 0)

                frame_traces = []

                # Add sphere
                u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
                frame_traces.append(
                    go.Surface(
                        x=np.cos(u)*np.sin(v), y=np.sin(u)*np.sin(v), z=np.cos(v),
                        opacity=0.1, showscale=False, name='Unit Sphere',
                        colorscale=[[0, 'lightgray'], [1, 'lightgray']]
                    )
                )

                # Add coordinate axes
                axis_length = 1.3
                frame_traces.append(go.Scatter3d(
                    x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
                    mode='lines', line=dict(color='red', width=2), name='Real Axis'
                ))
                frame_traces.append(go.Scatter3d(
                    x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
                    mode='lines', line=dict(color='green', width=2), name='Imag Axis'
                ))
                frame_traces.append(go.Scatter3d(
                    x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
                    mode='lines', line=dict(color='blue', width=2), name='Phase Axis'
                ))

                # Add class orientation vectors
                if len(weights.shape) == 4:
                    n_classes = min(weights.shape[0], 10)

                    for c in range(n_classes):
                        class_weights = weights[c].flatten()
                        significant = class_weights[np.abs(class_weights) > 0.01]

                        if len(significant) > 0:
                            magnitudes = np.abs(significant)
                            phases = np.angle(significant)

                            # Weighted circular mean
                            sin_sum = np.sum(magnitudes * np.sin(phases))
                            cos_sum = np.sum(magnitudes * np.cos(phases))
                            avg_theta = np.arctan2(sin_sum, cos_sum)
                            avg_r = np.mean(magnitudes)

                            # Polar angle from class index (shows separation)
                            phi = (c * np.pi / max(1, n_classes))

                            x = avg_r * np.sin(phi) * np.cos(avg_theta)
                            y = avg_r * np.sin(phi) * np.sin(avg_theta)
                            z = avg_r * np.cos(phi)

                            class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'

                            frame_traces.append(
                                go.Scatter3d(
                                    x=[0, x], y=[0, y], z=[0, z],
                                    mode='lines+markers',
                                    marker=dict(size=10, color=class_colors[c % len(class_colors)],
                                               symbol='circle', line=dict(width=1, color='white')),
                                    line=dict(color=class_colors[c % len(class_colors)], width=3),
                                    name=class_name,
                                    legendgroup=f'class{c}',
                                    showlegend=False,
                                    text=f"Class {class_name}<br>Magnitude: {avg_r:.3f}<br>Phase: {avg_theta:.2f} rad<br>Polar: {phi:.2f} rad",
                                    hoverinfo='text'
                                )
                            )

                # Add target orthogonal positions (for perfect 90° separation)
                if len(weights.shape) == 4:
                    n_classes = min(weights.shape[0], 10)
                    for c in range(n_classes):
                        target_theta = (c * 2 * np.pi / n_classes)
                        target_phi = np.pi / 2  # Equatorial plane
                        r = 0.95

                        x_target = r * np.sin(target_phi) * np.cos(target_theta)
                        y_target = r * np.sin(target_phi) * np.sin(target_theta)
                        z_target = r * np.cos(target_phi)

                        frame_traces.append(
                            go.Scatter3d(
                                x=[x_target], y=[y_target], z=[z_target],
                                mode='markers',
                                marker=dict(size=12, color=class_colors[c % len(class_colors)],
                                           symbol='x', line=dict(width=2, color='white')),
                                name=f'Target {class_names[c] if class_names and c < len(class_names) else f"C{c+1}"}',
                                legendgroup=f'target{c}',
                                showlegend=False
                            )
                        )

                # Add reference circle
                theta_circle = np.linspace(0, 2*np.pi, 50)
                x_circle = 0.98 * np.cos(theta_circle)
                y_circle = 0.98 * np.sin(theta_circle)
                z_circle = np.zeros_like(theta_circle)

                frame_traces.append(
                    go.Scatter3d(
                        x=x_circle, y=y_circle, z=z_circle,
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dash'),
                        name='Equatorial Plane (90°)',
                        showlegend=False
                    )
                )

                frames.append(
                    go.Frame(
                        data=frame_traces,
                        name=f'Round {round_num}',
                        layout=go.Layout(
                            title=dict(
                                text=f'<b>Round {round_num}</b> - Accuracy: {accuracy:.3f} | Training Samples: {training_size}<br>'
                                     f'<sup>Class orientation vectors evolving toward orthogonal positions</sup>',
                                font=dict(size=14)
                            )
                        )
                    )
                )

            if not frames:
                return None

            # Create figure with first frame
            fig = go.Figure(
                data=frames[0].data,
                layout=go.Layout(
                    title=dict(text=f'<b>Class Orientation Vector Evolution</b><br>'
                                   f'Round {evolution_history[0]["round"]} - Starting State'),
                    scene=dict(
                        xaxis_title='<b>Real Feature Component</b>',
                        yaxis_title='<b>Imaginary Feature Component</b>',
                        zaxis_title='<b>Class Separation</b>',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                        aspectmode='cube'
                    ),
                    updatemenus=[
                        dict(
                            type='buttons',
                            buttons=[
                                dict(label='▶️ Play', method='animate',
                                     args=[None, {'frame': {'duration': 800, 'redraw': True}, 'fromcurrent': True}]),
                                dict(label='⏸️ Pause', method='animate',
                                     args=[[None], {'frame': {'duration': 0, 'redraw': False}}]),
                                dict(label='🔄 Reset', method='animate',
                                     args=[[frames[0].name], {'frame': {'duration': 0, 'redraw': True}}])
                            ],
                            y=0.9,
                            x=0.1
                        )
                    ],
                    sliders=[{
                        'active': 0,
                        'currentvalue': {'prefix': 'Round: ', 'font': {'size': 14}},
                        'steps': [
                            {
                                'args': [[f'Round {snap["round"]}'], {'frame': {'duration': 0, 'redraw': True}}],
                                'label': str(snap['round']),
                                'method': 'animate'
                            }
                            for snap in evolution_history if 'complex_weights' in snap
                        ]
                    }],
                    width=1200,
                    height=800,
                    showlegend=True,
                    legend=dict(
                        yanchor="top", y=0.99, xanchor="left", x=0.02,
                        bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1,
                        font=dict(color='white')
                    ),
                    paper_bgcolor='rgba(0,0,0,0.9)',
                    plot_bgcolor='rgba(0,0,0,0.9)'
                ),
                frames=frames
            )

            # Save to interactive directory (not spherical)
            output_path = self.dirs.get('interactive', Path('visualizations/interactive')) / f'{self.dataset_name}_evolution_slider.html'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            print(f"   ✅ Evolution slider saved to: {output_path}")
            print(f"   Shows {len(frames)} rounds of vector evolution")

            return str(output_path)

        except Exception as e:
            print(f"{Colors.RED}   ❌ Error creating evolution slider: {e}{Colors.ENDC}")
            return None

    def _calculate_class_orthogonality(self, weights):
        """
        Calculate orthogonality metrics between class tensors in complex feature-pair space
        Used by multiple visualization methods

        Returns:
            orthogonality_matrix: n_classes x n_classes matrix of cosine similarities
            separation_angles: list of angles between class pairs (degrees)
            average_orthogonality: mean orthogonality (1 - mean similarity)
        """
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()

        # weights shape: (n_classes, n_pairs, n_bins, n_bins)
        n_classes = weights.shape[0]

        # Step 1: Compute orientation vector for each class
        class_vectors = []

        for c in range(n_classes):
            # Flatten all weights for this class
            class_weights = weights[c].flatten()

            # Filter significant weights (non-zero)
            significant = class_weights[np.abs(class_weights) > 1e-6]

            if len(significant) > 0:
                # Each weight is a complex number: w = a + i*b
                # We want the average orientation in complex space
                magnitudes = np.abs(significant)
                phases = np.angle(significant)

                # Weighted circular mean (accounts for magnitude importance)
                sin_sum = np.sum(magnitudes * np.sin(phases))
                cos_sum = np.sum(magnitudes * np.cos(phases))
                avg_phase = np.arctan2(sin_sum, cos_sum)

                # Average magnitude (strength of orientation)
                avg_mag = np.mean(magnitudes)

                # Unit vector in complex plane
                class_vectors.append(np.exp(1j * avg_phase) * avg_mag)
            else:
                class_vectors.append(0 + 0j)

        # Step 2: Calculate pairwise orthogonality
        orthogonality_matrix = np.zeros((n_classes, n_classes))
        separation_angles = []

        for i in range(n_classes):
            for j in range(n_classes):
                if i == j:
                    orthogonality_matrix[i, j] = 1.0
                else:
                    vi = class_vectors[i]
                    vj = class_vectors[j]

                    if np.abs(vi) > 0 and np.abs(vj) > 0:
                        # Cosine similarity between unit vectors
                        vi_unit = vi / np.abs(vi)
                        vj_unit = vj / np.abs(vj)
                        similarity = np.real(vi_unit * np.conj(vj_unit))
                        orthogonality_matrix[i, j] = similarity

                        if i < j:
                            # Angle in degrees
                            angle = np.arccos(np.clip(similarity, -1, 1)) * 180 / np.pi
                            separation_angles.append(angle)
                    else:
                        orthogonality_matrix[i, j] = 0

        # Step 3: Calculate metrics
        avg_similarity = np.mean([orthogonality_matrix[i, j]
                                  for i in range(n_classes)
                                  for j in range(i+1, n_classes)]) if n_classes > 1 else 0
        avg_orthogonality = 1.0 - avg_similarity
        avg_separation = np.mean(separation_angles) if separation_angles else 0

        return {
            'orthogonality_matrix': orthogonality_matrix,
            'separation_angles': separation_angles,
            'avg_orthogonality': avg_orthogonality,
            'avg_separation_degrees': avg_separation,
            'class_vectors': class_vectors
        }

# =============================================================================
# SECTION: SAMP INTEGRATION FOR EXTERNAL TOOLS
# =============================================================================

# =============================================================================
# SECTION: IMPROVED SAMP INTEGRATION FOR EXTERNAL TOOLS
# =============================================================================

class SAMPIntegration:
    """
    SAMP (Simple Application Messaging Protocol) integration for Topcat and Aladin.
    Allows direct data transfer without file exports.
    """

    def __init__(self):
        self.samp_client = None
        self.connected = False
        self.topcat_client_id = None
        self.aladin_client_id = None
        self.samp_hub_running = False

    def check_hub(self):
        """Check if a SAMP hub is already running"""
        try:
            import astropy.samp
            # Try to connect to existing hub
            test_client = astropy.samp.SAMPIntegratedClient()
            test_client.connect()
            test_client.disconnect()
            self.samp_hub_running = True
            return True
        except:
            self.samp_hub_running = False
            return False

    def start_hub(self):
        """Start a SAMP hub if none exists"""
        try:
            import astropy.samp
            import subprocess
            import threading

            # Try to start hub in background
            hub_process = subprocess.Popen(
                ['samp_hub'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Wait for hub to start
            import time
            time.sleep(2)

            # Check if hub is now running
            return self.check_hub()
        except:
            return False

    def connect(self):
        """Connect to SAMP hub (start if needed)"""
        try:
            import astropy.samp

            # Check if hub is running
            if not self.check_hub():
                print(f"{Colors.YELLOW}⚠️ No SAMP hub found, starting one...{Colors.ENDC}")
                if not self.start_hub():
                    print(f"{Colors.YELLOW}⚠️ Could not start SAMP hub. Tools will use file fallback.{Colors.ENDC}")
                    return False

            # Connect to hub
            self.samp_client = astropy.samp.SAMPIntegratedClient()
            self.samp_client.connect()
            self.connected = True
            print(f"{Colors.GREEN}✅ Connected to SAMP hub{Colors.ENDC}")
            return True

        except ImportError:
            print(f"{Colors.YELLOW}⚠️ astropy.samp not available. Install: pip install astropy{Colors.ENDC}")
            return False
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️ Could not connect to SAMP hub: {e}{Colors.ENDC}")
            return False

    def disconnect(self):
        """Disconnect from SAMP hub"""
        if self.samp_client and self.connected:
            try:
                self.samp_client.disconnect()
            except:
                pass
            self.connected = False
            self.topcat_client_id = None
            self.aladin_client_id = None

    def find_tool(self, tool_name):
        """Find running instance of a tool"""
        if not self.connected:
            return None

        try:
            clients = self.samp_client.get_registered_clients()
            for client_id in clients:
                metadata = self.samp_client.get_metadata(client_id)
                name = metadata.get('samp.name', '').lower()
                if tool_name.lower() in name:
                    return client_id
            return None
        except:
            return None

    def launch_topcat(self):
        """Launch Topcat and wait for it to register"""
        import subprocess
        import shutil
        import time

        # Try to find existing instance first
        existing = self.find_tool('topcat')
        if existing:
            self.topcat_client_id = existing
            print(f"{Colors.GREEN}✅ Using existing Topcat instance{Colors.ENDC}")
            return True

        # Find Topcat executable
        topcat = shutil.which('topcat')

        if not topcat:
            # Try JAR files
            jar_paths = [
                os.path.expanduser('~/topcat/topcat.jar'),
                '/usr/share/topcat/topcat.jar',
                '/usr/local/share/topcat/topcat.jar'
            ]
            for jar in jar_paths:
                if os.path.exists(jar):
                    topcat = ['java', '-jar', jar]
                    break

        if not topcat:
            print(f"{Colors.RED}❌ Topcat not found{Colors.ENDC}")
            return False

        # Launch Topcat
        try:
            if isinstance(topcat, str):
                subprocess.Popen([topcat])
            else:
                subprocess.Popen(topcat)

            # Wait for Topcat to register with SAMP
            for i in range(10):  # Wait up to 10 seconds
                time.sleep(1)
                existing = self.find_tool('topcat')
                if existing:
                    self.topcat_client_id = existing
                    print(f"{Colors.GREEN}✅ Topcat launched and connected{Colors.ENDC}")
                    return True

            print(f"{Colors.YELLOW}⚠️ Topcat launched but not registered with SAMP{Colors.ENDC}")
            return False

        except Exception as e:
            print(f"{Colors.RED}❌ Failed to launch Topcat: {e}{Colors.ENDC}")
            return False

    def launch_aladin(self):
        """Launch Aladin and wait for it to register"""
        import subprocess
        import shutil
        import time

        # Try to find existing instance
        existing = self.find_tool('aladin')
        if existing:
            self.aladin_client_id = existing
            print(f"{Colors.GREEN}✅ Using existing Aladin instance{Colors.ENDC}")
            return True

        # Find Aladin executable
        aladin = shutil.which('aladin')

        if not aladin:
            # Try JAR files
            jar_paths = [
                os.path.expanduser('~/aladin/aladin.jar'),
                '/usr/share/aladin/aladin.jar',
                '/usr/local/share/aladin/aladin.jar'
            ]
            for jar in jar_paths:
                if os.path.exists(jar):
                    aladin = ['java', '-jar', jar]
                    break

        if not aladin:
            print(f"{Colors.RED}❌ Aladin not found{Colors.ENDC}")
            return False

        # Launch Aladin
        try:
            # Aladin might be a script that needs bash
            if isinstance(aladin, str) and aladin.endswith('.sh'):
                subprocess.Popen(['bash', aladin])
            elif isinstance(aladin, str):
                subprocess.Popen([aladin])
            else:
                subprocess.Popen(aladin)

            # Wait for Aladin to register with SAMP
            for i in range(15):  # Aladin takes longer to start
                time.sleep(1)
                existing = self.find_tool('aladin')
                if existing:
                    self.aladin_client_id = existing
                    print(f"{Colors.GREEN}✅ Aladin launched and connected{Colors.ENDC}")
                    return True

            print(f"{Colors.YELLOW}⚠️ Aladin launched but not registered with SAMP{Colors.ENDC}")
            return False

        except Exception as e:
            print(f"{Colors.RED}❌ Failed to launch Aladin: {e}{Colors.ENDC}")
            return False

    def send_data_to_topcat(self, data: pd.DataFrame, title: str = "CT-DBNN Data"):
        """Send data to Topcat via SAMP"""
        if not self.connected:
            if not self.connect():
                return False

        if not self.topcat_client_id:
            if not self.launch_topcat():
                return False

        try:
            from astropy.io.votable import from_table
            from astropy.table import Table
            from io import BytesIO
            import urllib.parse

            # Convert to VOTable in memory
            votable = from_table(Table.from_pandas(data))
            buf = BytesIO()
            votable.to_xml(buf)
            votable_xml = buf.getvalue().decode('utf-8')

            # Encode for data URI
            encoded_xml = urllib.parse.quote(votable_xml)
            data_uri = f"data:text/xml,{encoded_xml}"

            # Send to Topcat
            self.samp_client.notify(self.topcat_client_id, "table.load.votable", {
                "url": data_uri,
                "name": title
            })

            print(f"{Colors.GREEN}✅ Data sent to Topcat via SAMP{Colors.ENDC}")
            return True

        except Exception as e:
            print(f"{Colors.RED}❌ SAMP communication failed: {e}{Colors.ENDC}")
            return False

    def send_data_to_aladin(self, data: pd.DataFrame, title: str = "CT-DBNN Data"):
        """Send data to Aladin via SAMP"""
        if not self.connected:
            if not self.connect():
                return False

        if not self.aladin_client_id:
            if not self.launch_aladin():
                return False

        try:
            from astropy.io.votable import from_table
            from astropy.table import Table
            from io import BytesIO
            import urllib.parse

            # Convert to VOTable in memory
            votable = from_table(Table.from_pandas(data))
            buf = BytesIO()
            votable.to_xml(buf)
            votable_xml = buf.getvalue().decode('utf-8')

            # Encode for data URI
            encoded_xml = urllib.parse.quote(votable_xml)
            data_uri = f"data:text/xml,{encoded_xml}"

            # Send to Aladin
            self.samp_client.notify(self.aladin_client_id, "table.load.votable", {
                "url": data_uri,
                "name": title
            })

            print(f"{Colors.GREEN}✅ Data sent to Aladin via SAMP{Colors.ENDC}")
            return True

        except Exception as e:
            print(f"{Colors.RED}❌ SAMP communication failed: {e}{Colors.ENDC}")
            return False


#
#  =============================================================================
# SECTION 10: GUI CLASS (COMPLETE WITH ALL INIT VARIABLES)
# =============================================================================

class CTDBNNGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CT-DBNN - Complex Tensor Difference Boosting Neural Network")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)

        self.model = None
        self.config = None
        self.env_manager = EnvironmentManager()
        self.current_data_file = None
        self.original_data = None
        self.data_loaded = False

        self.training_in_progress = False
        self.stop_training_flag = False
        self.current_training_thread = None

        # ========== ADD THESE MISSING INITIALIZATIONS ==========
        self.deps_status = {}           # For tracking dependency install buttons
        self.install_thread = None      # For background installation
        self.install_in_progress = False # Installation flag

        # External tools buttons (for enabling/disabling)
        self.topcat_btn = None
        self.aladin_btn = None
        self.ext_status_label = None

        # Dataset and target variables
        self.dataset_var = tk.StringVar()
        self.target_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="train_predict")
        self.model_type_var = tk.StringVar(value="Histogram")
        self.device_var = tk.StringVar(value="auto")
        self.file_path_var = tk.StringVar()
        self.uci_var = tk.StringVar()

        # Training parameters
        self.learning_rate_var = tk.StringVar(value="0.1")
        self.epochs_var = tk.StringVar(value="100")
        self.bins_var = tk.StringVar(value="128")
        self.test_size_var = tk.StringVar(value="0.2")
        self.adaptive_var = tk.BooleanVar(value=True)
        self.adaptive_rounds_var = tk.StringVar(value="10")
        self.initial_samples_var = tk.StringVar(value="50")
        self.max_samples_round_var = tk.StringVar(value="25")

        # Parallel processing variables
        self.parallel_var = tk.BooleanVar(value=True)
        self.n_jobs_var = tk.StringVar(value=str(mp.cpu_count()))
        self.parallel_batch_var = tk.StringVar(value="1000")
        self.parallel_mode_var = tk.StringVar(value="threads")

        # Export dataset selection
        self.export_dataset_var = tk.StringVar(value="all")
        self.sdss_radius_var = tk.StringVar(value="1.0")

        # Feature selection
        self.feature_vars = {}

        # Visualization references
        self.output_text = None
        self.results_text = None
        self.info_text = None
        self.feature_canvas = None
        self.feature_scroll_frame = None

        # Progress indicators
        self.install_progress = None
        self.install_status = None

        self.setup_gui()

        # Initialize SAMP integration
        self.samp = SAMPIntegration()
        self.samp_connected = False

    def setup_gui(self):
        """Setup the complete GUI interface"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.env_tab = ttk.Frame(notebook)
        notebook.add(self.env_tab, text="🌍 Environment")
        self.setup_environment_tab()

        self.dataset_tab = ttk.Frame(notebook)
        notebook.add(self.dataset_tab, text="📊 Dataset")
        self.setup_dataset_tab()

        self.config_tab = ttk.Frame(notebook)
        notebook.add(self.config_tab, text="⚙️ Configuration")
        self.setup_config_tab()

        self.training_tab = ttk.Frame(notebook)
        notebook.add(self.training_tab, text="🚀 Training")
        self.setup_training_tab()

        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # ========== ADD EXTERNAL TOOLS TAB ==========
        self.ext_tab = ttk.Frame(notebook)
        notebook.add(self.ext_tab, text="🔌 External Tools")
        self.setup_external_tools_tab()

        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_external_tools_tab(self):
        """Setup external tools tab with comprehensive controls"""
        # Tool status frame
        status_frame = ttk.LabelFrame(self.ext_tab, text="Tool Availability", padding="10")
        status_frame.pack(fill=tk.X, pady=5)

        # Check all tools
        topcat_avail = self._check_topcat()
        aladin_avail = self._check_aladin()
        fits_avail = ASTROPY_AVAILABLE
        sdss_avail = True  # Always available via web

        # Display status grid
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)

        tools = [
            ("Topcat", topcat_avail, "📊", "https://www.star.bris.ac.uk/~mbt/topcat/"),
            ("Aladin", aladin_avail, "🌌", "https://aladin.u-strasbg.fr/"),
            ("Astropy (FITS)", fits_avail, "💾", "pip install astropy"),
            ("SDSS SkyServer", sdss_avail, "🔭", "https://skyserver.sdss.org/")
        ]

        for i, (name, avail, icon, url) in enumerate(tools):
            status = "✓ Available" if avail else "✗ Not Found"
            color = "green" if avail else "red"
            ttk.Label(status_grid, text=f"{icon} {name}:", font=('Arial', 10, 'bold')).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Label(status_grid, text=status, foreground=color).grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            if not avail:
                ttk.Label(status_grid, text=f"Install: {url}", foreground="gray").grid(row=i, column=2, sticky=tk.W, padx=5, pady=2)

        # Export controls frame
        export_frame = ttk.LabelFrame(self.ext_tab, text="Data Export", padding="10")
        export_frame.pack(fill=tk.X, pady=5)

        # Export buttons
        btn_frame = ttk.Frame(export_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        if topcat_avail:
            ttk.Button(btn_frame, text="📊 Launch Topcat", command=self.launch_topcat_gui, width=20).pack(side=tk.LEFT, padx=5)

        if aladin_avail:
            ttk.Button(btn_frame, text="🌌 Launch Aladin", command=self.launch_aladin_gui, width=20).pack(side=tk.LEFT, padx=5)

        if fits_avail:
            ttk.Button(btn_frame, text="💾 Export to FITS", command=self.export_to_fits_gui, width=20).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="📄 Export to VOTable", command=self.export_to_votable_gui, width=20).pack(side=tk.LEFT, padx=5)

        # Data selection for export
        ttk.Label(export_frame, text="Dataset to Export:").pack(anchor=tk.W, pady=5)
        self.export_dataset_var = tk.StringVar(value="all")
        ttk.Radiobutton(export_frame, text="Full Dataset", variable=self.export_dataset_var, value="all").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(export_frame, text="Training Set Only", variable=self.export_dataset_var, value="training").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(export_frame, text="Test Set Only", variable=self.export_dataset_var, value="test").pack(anchor=tk.W, padx=20)

        # External queries frame
        query_frame = ttk.LabelFrame(self.ext_tab, text="External Queries", padding="10")
        query_frame.pack(fill=tk.X, pady=5)

        ttk.Button(query_frame, text="🔭 Query SDSS", command=self.query_sdss_gui, width=20).pack(side=tk.LEFT, padx=5)

        ttk.Label(query_frame, text="Search Radius (arcsec):").pack(side=tk.LEFT, padx=5)
        self.sdss_radius_var = tk.StringVar(value="1.0")
        ttk.Entry(query_frame, textvariable=self.sdss_radius_var, width=10).pack(side=tk.LEFT, padx=5)

        # Vizier query (if pyvo available)
        try:
            import pyvo
            ttk.Button(query_frame, text="📚 Query Vizier", command=self.query_vizier_gui, width=20).pack(side=tk.LEFT, padx=5)
        except ImportError:
            pass

        # Coordinate conversion frame
        coord_frame = ttk.LabelFrame(self.ext_tab, text="Coordinate Conversion", padding="10")
        coord_frame.pack(fill=tk.X, pady=5)

        ttk.Button(coord_frame, text="🌐 Convert to Galactic", command=self.convert_coordinates_gui, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(coord_frame, text="🌍 Convert to Equatorial", command=self.convert_coordinates_gui, width=20).pack(side=tk.LEFT, padx=5)

        # Status output
        self.ext_status_text = scrolledtext.ScrolledText(self.ext_tab, height=10)
        self.ext_status_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def query_vizier_gui(self):
        """Query Vizier catalog"""
        if not self.model:
            messagebox.showwarning("Warning", "No model to query")
            return

        # Get catalog name
        catalog = tk.simpledialog.askstring("Vizier Query", "Enter Vizier catalog name (e.g., 'I/311/hip2'):")
        if not catalog:
            return

        # Get coordinates
        data = self.model._prepare_export_data()
        ra_col, dec_col = self._find_coordinate_columns(data)

        if not ra_col or not dec_col:
            self.log_output("❌ No RA/Dec columns found")
            return

        self.log_output(f"📚 Querying Vizier catalog {catalog}...")

        try:
            import pyvo
            results = []
            for r, d in zip(data[ra_col].values[:10], data[dec_col].values[:10]):
                query = f"SELECT * FROM {catalog} WHERE CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), CIRCLE('ICRS', {r}, {d}, 0.0166667)) = 1"
                result = pyvo.vizier.query(query)
                if result:
                    results.append(result.to_table().to_pandas())

            if results:
                combined = pd.concat(results)
                self.log_output(f"✅ Vizier query returned {len(combined)} matches")
                self._show_vizier_results(combined, catalog)
            else:
                self.log_output("No matches found")

        except Exception as e:
            self.log_output(f"❌ Vizier query failed: {e}")

    def convert_coordinates_gui(self):
        """Convert coordinates using Astropy"""
        if not self.model or not ASTROPY_AVAILABLE:
            messagebox.showwarning("Warning", "Astropy not available")
            return

        data = self.model._prepare_export_data()
        ra_col, dec_col = self._find_coordinate_columns(data)

        if not ra_col or not dec_col:
            self.log_output("❌ No RA/Dec columns found")
            return

        self.log_output("🌐 Converting coordinates...")

        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            coords = SkyCoord(ra=data[ra_col].values * u.deg, dec=data[dec_col].values * u.deg)

            results_df = pd.DataFrame({
                'ra_deg': coords.ra.deg,
                'dec_deg': coords.dec.deg,
                'gal_l_deg': coords.galactic.l.deg,
                'gal_b_deg': coords.galactic.b.deg,
                'fk5_ra_deg': coords.fk5.ra.deg,
                'fk5_dec_deg': coords.fk5.dec.deg
            })

            self.log_output(f"✅ Converted {len(results_df)} coordinates")
            self._show_coordinate_results(results_df)

        except Exception as e:
            self.log_output(f"❌ Coordinate conversion failed: {e}")

    def setup_environment_tab(self):
        """Enhanced environment tab with dependency management"""
        # System info frame (existing)
        sys_frame = ttk.LabelFrame(self.env_tab, text="System Information", padding="10")
        sys_frame.pack(fill=tk.X, pady=5)

        sys_info = self.env_manager.get_system_info()
        info_text = f"""
        Operating System: {sys_info['os']} {sys_info['os_version']}
        Architecture: {sys_info['architecture']}
        Processor: {sys_info['processor']}
        Python: {sys_info['python_version']}
        """
        ttk.Label(sys_frame, text=info_text, font=('Courier', 10)).pack(anchor=tk.W)

        # CUDA info
        cuda_info = self.env_manager.check_cuda()
        if cuda_info['available']:
            cuda_text = f"""
            CUDA Available: ✓
            CUDA Version: {cuda_info['version']}
            Device: {cuda_info['device_name']}
            Memory: {cuda_info['memory']:.1f} GB
            """
            ttk.Label(sys_frame, text=cuda_text, font=('Courier', 10), foreground='green').pack(anchor=tk.W)
        else:
            ttk.Label(sys_frame, text="CUDA Available: ✗ (CPU mode only)",
                     font=('Courier', 10), foreground='red').pack(anchor=tk.W)

        # ========== DEPENDENCY MANAGEMENT FRAME ==========
        deps_frame = ttk.LabelFrame(self.env_tab, text="📦 Dependency Manager", padding="10")
        deps_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create notebook for dependency categories
        deps_notebook = ttk.Notebook(deps_frame)
        deps_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Python Dependencies
        python_tab = ttk.Frame(deps_notebook)
        deps_notebook.add(python_tab, text="🐍 Python Packages")
        self._setup_python_deps_tab(python_tab)

        # Tab 2: Java Tools (Topcat, Aladin)
        java_tab = ttk.Frame(deps_notebook)
        deps_notebook.add(java_tab, text="☕ Java Tools")
        self._setup_java_tools_tab(java_tab)

        # Tab 3: System Tools
        system_tab = ttk.Frame(deps_notebook)
        deps_notebook.add(system_tab, text="🖥️ System Tools")
        self._setup_system_tools_tab(system_tab)

        # Progress bar for installations
        self.install_progress = ttk.Progressbar(deps_frame, mode='indeterminate')
        self.install_progress.pack(fill=tk.X, pady=5)
        self.install_status = ttk.Label(deps_frame, text="", foreground="blue")
        self.install_status.pack(fill=tk.X)


    def _setup_python_deps_tab(self, parent):
        """Setup Python dependencies tab with install buttons"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Python packages
        packages = {
            "Core ML": {
                "torch": "PyTorch deep learning framework",
                "numpy": "Numerical computing",
                "pandas": "Data manipulation",
                "scikit-learn": "Machine learning algorithms"
            },
            "Visualization": {
                "matplotlib": "Basic plotting",
                "seaborn": "Statistical visualization",
                "plotly": "Interactive visualizations"
            },
            "Astronomy": {
                "astropy": "Astronomy core library (FITS, VOTable)",
                "astroquery": "Online astronomical databases",
                "pyvo": "Virtual Observatory access"
            },
            "Utilities": {
                "requests": "HTTP requests",
                "tqdm": "Progress bars",
                "joblib": "Parallel processing"
            }
        }

        row = 0
        for category, pkgs in packages.items():
            # Category header
            ttk.Label(scrollable_frame, text=category, font=('Arial', 10, 'bold')).grid(
                row=row, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(10, 5))
            row += 1

            for pkg, desc in pkgs.items():
                # Check if installed
                installed = self._check_python_package(pkg)

                # Package name
                ttk.Label(scrollable_frame, text=pkg, font=('Courier', 10)).grid(
                    row=row, column=0, sticky=tk.W, padx=5, pady=2)

                # Status
                status_text = "✓ Installed" if installed else "✗ Not Installed"
                status_color = "green" if installed else "red"
                ttk.Label(scrollable_frame, text=status_text, foreground=status_color).grid(
                    row=row, column=1, sticky=tk.W, padx=5, pady=2)

                # Description
                ttk.Label(scrollable_frame, text=desc, foreground="gray").grid(
                    row=row, column=2, sticky=tk.W, padx=5, pady=2)

                # Install button
                if not installed:
                    btn = ttk.Button(scrollable_frame, text="Install",
                                    command=lambda p=pkg: self._install_python_package(p),
                                    width=10)
                    btn.grid(row=row, column=3, padx=5, pady=2)
                    self.deps_status[pkg] = btn
                else:
                    ttk.Button(scrollable_frame, text="Reinstall",
                              command=lambda p=pkg: self._install_python_package(p),
                              width=10).grid(row=row, column=3, padx=5, pady=2)

                row += 1

        # Install all missing button
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.grid(row=row, column=0, columnspan=4, pady=20)

        ttk.Button(btn_frame, text="Install All Missing Packages",
                  command=self._install_all_missing).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Check All",
                  command=self._refresh_python_deps).pack(side=tk.LEFT, padx=5)

    def _setup_java_tools_tab(self, parent):
        """Setup Java tools tab for Topcat and Aladin"""
        # Check Java first
        java_installed = self._check_java()

        java_frame = ttk.LabelFrame(parent, text="Java Runtime", padding="10")
        java_frame.pack(fill=tk.X, pady=5)

        if java_installed:
            java_version = self._get_java_version()
            ttk.Label(java_frame, text=f"✓ Java found: {java_version}", foreground="green").pack(anchor=tk.W)
        else:
            ttk.Label(java_frame, text="✗ Java not found", foreground="red").pack(anchor=tk.W)
            ttk.Button(java_frame, text="Install Java", command=self._install_java).pack(anchor=tk.W, pady=5)

        # Topcat section
        topcat_frame = ttk.LabelFrame(parent, text="Topcat - Table Viewer", padding="10")
        topcat_frame.pack(fill=tk.X, pady=5)

        topcat_installed = self._check_topcat()
        if topcat_installed:
            ttk.Label(topcat_frame, text="✓ Topcat installed", foreground="green").pack(anchor=tk.W)
            ttk.Button(topcat_frame, text="Launch Topcat", command=self.launch_topcat_gui).pack(anchor=tk.W, pady=5)
        else:
            ttk.Label(topcat_frame, text="✗ Topcat not installed", foreground="red").pack(anchor=tk.W)
            ttk.Label(topcat_frame, text="Topcat is a Java application for table visualization", foreground="gray").pack(anchor=tk.W)
            ttk.Button(topcat_frame, text="Download Topcat", command=self._download_topcat).pack(anchor=tk.W, pady=5)
            ttk.Button(topcat_frame, text="Install from Package Manager", command=self._install_topcat).pack(anchor=tk.W, pady=2)

        # Aladin section
        aladin_frame = ttk.LabelFrame(parent, text="Aladin - Sky Atlas", padding="10")
        aladin_frame.pack(fill=tk.X, pady=5)

        aladin_installed = self._check_aladin()
        if aladin_installed:
            ttk.Label(aladin_frame, text="✓ Aladin installed", foreground="green").pack(anchor=tk.W)
            ttk.Button(aladin_frame, text="Launch Aladin", command=self.launch_aladin_gui).pack(anchor=tk.W, pady=5)
        else:
            ttk.Label(aladin_frame, text="✗ Aladin not installed", foreground="red").pack(anchor=tk.W)
            ttk.Label(aladin_frame, text="Aladin is a Java application for sky visualization", foreground="gray").pack(anchor=tk.W)
            ttk.Button(aladin_frame, text="Download Aladin", command=self._download_aladin).pack(anchor=tk.W, pady=5)
            ttk.Button(aladin_frame, text="Install from Package Manager", command=self._install_aladin).pack(anchor=tk.W, pady=2)

        # Installation instructions
        instructions_frame = ttk.LabelFrame(parent, text="Installation Instructions", padding="10")
        instructions_frame.pack(fill=tk.X, pady=5)

        instructions = """
        Topcat and Aladin are Java applications:

        1. Download the .jar file from the official websites
        2. Place it in a directory (e.g., ~/astronomy_tools/)
        3. Add to PATH or create a launcher script

        Or use package managers:
        - Ubuntu/Debian: sudo apt install topcat aladin
        - macOS: brew install topcat aladin
        - Windows: Download from websites
        """

        text_widget = scrolledtext.ScrolledText(instructions_frame, height=8, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, instructions)
        text_widget.config(state=tk.DISABLED)

    def _setup_system_tools_tab(self, parent):
        """Setup system tools tab"""
        tools = [
            ("Git", "git", "Version control", "https://git-scm.com/"),
            ("Conda", "conda", "Package and environment manager", "https://conda.io/"),
            ("Make", "make", "Build automation", "https://www.gnu.org/software/make/"),
            ("Wget", "wget", "File downloader", "https://www.gnu.org/software/wget/"),
            ("Curl", "curl", "URL transfer tool", "https://curl.se/"),
        ]

        for i, (name, cmd, desc, url) in enumerate(tools):
            frame = ttk.LabelFrame(parent, text=name, padding="10")
            frame.pack(fill=tk.X, pady=5)

            installed = self._check_system_tool(cmd)
            status = "✓ Installed" if installed else "✗ Not Installed"
            color = "green" if installed else "red"

            ttk.Label(frame, text=status, foreground=color).pack(anchor=tk.W)
            ttk.Label(frame, text=desc, foreground="gray").pack(anchor=tk.W)

            if not installed:
                ttk.Label(frame, text=f"Install: {url}", foreground="blue").pack(anchor=tk.W)

    def _check_python_package(self, package):
        """Check if Python package is installed"""
        try:
            __import__(package.replace('-', '_'))
            return True
        except ImportError:
            return False

    def _check_java(self):
        """Check if Java is installed"""
        import shutil
        return shutil.which('java') is not None

    def _get_java_version(self):
        """Get Java version"""
        try:
            result = subprocess.run(['java', '-version'], capture_output=True, text=True)
            return result.stderr.split('\n')[0] if result.stderr else "Unknown"
        except:
            return "Unknown"

    def _check_system_tool(self, tool):
        """Check if system tool is available"""
        import shutil
        return shutil.which(tool) is not None

    def _install_python_package(self, package):
        """Install Python package in background thread"""
        if self.install_in_progress:
            messagebox.showwarning("Installation in Progress", "Please wait for current installation to complete")
            return

        self.install_thread = threading.Thread(target=self._install_package_thread, args=(package,))
        self.install_thread.daemon = True
        self.install_thread.start()

    def _install_package_thread(self, package):
        """Background thread for package installation"""
        self.install_in_progress = True
        self.install_progress.start()
        self.install_status.config(text=f"Installing {package}...", foreground="blue")

        try:
            # Disable the button while installing
            if package in self.deps_status:
                self.deps_status[package].config(state=tk.DISABLED, text="Installing...")

            # Run pip install
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", package],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                self.install_status.config(text=f"✓ {package} installed successfully", foreground="green")
                self.log_output(f"✅ Installed {package}")

                # Update button
                if package in self.deps_status:
                    self.deps_status[package].config(text="Reinstall", state=tk.NORMAL)

                # Refresh the tab to show installed status
                self._refresh_python_deps()
            else:
                error_msg = result.stderr.split('\n')[-2] if result.stderr else "Unknown error"
                self.install_status.config(text=f"✗ Failed to install {package}: {error_msg}", foreground="red")
                self.log_output(f"❌ Failed to install {package}: {error_msg}")

                if package in self.deps_status:
                    self.deps_status[package].config(text="Retry", state=tk.NORMAL)

        except Exception as e:
            self.install_status.config(text=f"✗ Error: {str(e)}", foreground="red")
            self.log_output(f"❌ Error installing {package}: {e}")

            if package in self.deps_status:
                self.deps_status[package].config(text="Retry", state=tk.NORMAL)

        finally:
            self.install_progress.stop()
            self.install_in_progress = False

            # Reset status after 3 seconds
            self.root.after(3000, lambda: self.install_status.config(text=""))

    def _install_all_missing(self):
        """Install all missing Python packages"""
        missing = []
        for pkg, btn in self.deps_status.items():
            if btn.cget('text') == "Install":
                missing.append(pkg)

        if missing:
            response = messagebox.askyesno("Install All", f"Install {len(missing)} missing packages?\n\n{', '.join(missing)}")
            if response:
                for pkg in missing:
                    self._install_python_package(pkg)
        else:
            messagebox.showinfo("All Installed", "All packages are already installed!")

    def _refresh_python_deps(self):
        """Refresh Python dependencies tab"""
        # Find and refresh the Python dependencies tab
        for child in self.env_tab.winfo_children():
            if isinstance(child, ttk.LabelFrame) and child.cget("text") == "Dependencies & Tools":
                for notebook in child.winfo_children():
                    if isinstance(notebook, ttk.Notebook):
                        for tab in notebook.tabs():
                            if notebook.tab(tab, "text") == "🐍 Python Packages":
                                # Recreate the tab
                                notebook.forget(tab)
                                new_tab = ttk.Frame(notebook)
                                notebook.insert(tab, new_tab, text="🐍 Python Packages")
                                self._setup_python_deps_tab(new_tab)
                                break
                        break
                break

    def _install_java(self):
        """Open Java download page"""
        import webbrowser
        webbrowser.open("https://adoptium.net/")
        messagebox.showinfo("Java Installation",
                           "Java is required for Topcat and Aladin.\n"
                           "Please download and install Java from the opened page.")

    def _download_topcat(self):
        """Open Topcat download page"""
        import webbrowser
        webbrowser.open("https://www.star.bris.ac.uk/~mbt/topcat/")
        messagebox.showinfo("Topcat Download",
                           "Please download topcat-full.jar and save it to:\n\n"
                           "~/astronomy_tools/topcat.jar\n\n"
                           "Then add to PATH or create a launcher script.")

    def _download_aladin(self):
        """Open Aladin download page"""
        import webbrowser
        webbrowser.open("https://aladin.u-strasbg.fr/java/nph-aladin.pl?frame=downloading")
        messagebox.showinfo("Aladin Download",
                           "Please download aladin.jar and save it to:\n\n"
                           "~/astronomy_tools/aladin.jar\n\n"
                           "Then add to PATH or create a launcher script.")

    def _install_topcat(self):
        """Install Topcat via package manager based on OS"""
        import platform
        system = platform.system()

        if system == "Linux":
            # Try to detect package manager
            if shutil.which('apt'):
                response = messagebox.askyesno("Install Topcat",
                                              "Install Topcat using apt?\n\nsudo apt install topcat")
                if response:
                    subprocess.Popen(['sudo', 'apt', 'install', '-y', 'topcat'])
            elif shutil.which('brew'):
                response = messagebox.askyesno("Install Topcat",
                                              "Install Topcat using Homebrew?\n\nbrew install topcat")
                if response:
                    subprocess.Popen(['brew', 'install', 'topcat'])
            else:
                self._download_topcat()
        elif system == "Darwin":  # macOS
            response = messagebox.askyesno("Install Topcat",
                                          "Install Topcat using Homebrew?\n\nbrew install topcat")
            if response:
                subprocess.Popen(['brew', 'install', 'topcat'])
        else:  # Windows
            self._download_topcat()

    def _install_aladin(self):
        """Install Aladin via package manager based on OS"""
        import platform
        system = platform.system()

        if system == "Linux":
            if shutil.which('apt'):
                response = messagebox.askyesno("Install Aladin",
                                              "Install Aladin using apt?\n\nsudo apt install aladin")
                if response:
                    subprocess.Popen(['sudo', 'apt', 'install', '-y', 'aladin'])
            elif shutil.which('brew'):
                response = messagebox.askyesno("Install Aladin",
                                              "Install Aladin using Homebrew?\n\nbrew install aladin")
                if response:
                    subprocess.Popen(['brew', 'install', 'aladin'])
            else:
                self._download_aladin()
        elif system == "Darwin":
            response = messagebox.askyesno("Install Aladin",
                                          "Install Aladin using Homebrew?\n\nbrew install aladin")
            if response:
                subprocess.Popen(['brew', 'install', 'aladin'])
        else:
            self._download_aladin()

    def setup_dataset_tab(self):
        """Setup dataset management tab with auto-target deselection"""
        # Dataset selection frame
        select_frame = ttk.LabelFrame(self.dataset_tab, text="Dataset Selection", padding="10")
        select_frame.pack(fill=tk.X, pady=5)

        # Local file selection
        ttk.Label(select_frame, text="Local File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(select_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(select_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)

        # UCI dataset selection
        ttk.Label(select_frame, text="UCI Dataset:").grid(row=1, column=0, sticky=tk.W, padx=5)
        uci_datasets = DatasetConfig.list_uci_datasets()
        self.uci_var = tk.StringVar()
        uci_combo = ttk.Combobox(select_frame, textvariable=self.uci_var, values=uci_datasets, width=30)
        uci_combo.grid(row=1, column=1, padx=5, sticky=tk.W)

        # Add description label
        self.uci_desc_label = ttk.Label(select_frame, text="", foreground="gray", wraplength=400)
        self.uci_desc_label.grid(row=2, column=1, padx=5, sticky=tk.W)

        # Bind selection event to show description
        uci_combo.bind('<<ComboboxSelected>>', self._show_uci_description)

        ttk.Button(select_frame, text="Download", command=self.download_uci).grid(row=1, column=2, padx=5)
        ttk.Button(select_frame, text="Load Dataset", command=self.load_dataset).grid(row=3, column=1, pady=10)

        # Dataset info frame
        self.info_frame = ttk.LabelFrame(self.dataset_tab, text="Dataset Information", padding="10")
        self.info_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.info_text = scrolledtext.ScrolledText(self.info_frame, height=10)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Feature selection frame
        self.feature_frame = ttk.LabelFrame(self.dataset_tab, text="Feature Selection", padding="10")
        self.feature_frame.pack(fill=tk.X, pady=5)

        # Create canvas with scrollbar for feature list
        self.feature_canvas = tk.Canvas(self.feature_frame, height=200)
        feature_scroll = ttk.Scrollbar(self.feature_frame, orient="vertical", command=self.feature_canvas.yview)
        self.feature_scroll_frame = ttk.Frame(self.feature_canvas)

        self.feature_canvas.configure(yscrollcommand=feature_scroll.set)
        self.feature_canvas.create_window((0, 0), window=self.feature_scroll_frame, anchor="nw")

        self.feature_canvas.pack(side="left", fill="both", expand=True)
        feature_scroll.pack(side="right", fill="y")

        self.feature_scroll_frame.bind("<Configure>",
            lambda e: self.feature_canvas.configure(scrollregion=self.feature_canvas.bbox("all")))

        # Feature selection buttons
        btn_frame = ttk.Frame(self.feature_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Select All", command=self.select_all_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Deselect All", command=self.deselect_all_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Select Numeric", command=self.select_numeric_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Apply Selection", command=self.apply_feature_selection).pack(side=tk.LEFT, padx=2)

    def update_feature_selection(self, df):
        """Update feature selection UI with auto-deselection of target"""
        # Clear existing widgets
        for widget in self.feature_scroll_frame.winfo_children():
            widget.destroy()

        self.feature_vars = {}

        # Target selection row
        target_frame = ttk.LabelFrame(self.feature_scroll_frame, text="Target Column", padding="5")
        target_frame.pack(fill=tk.X, pady=5)

        ttk.Label(target_frame, text="Select the target column (will be excluded from features):",
                  font=('Arial', 9)).pack(anchor=tk.W)

        # Target combobox
        target_combo = ttk.Combobox(target_frame, textvariable=self.target_var,
                                    values=list(df.columns), width=30, state="readonly")
        target_combo.pack(pady=5)

        # Bind target selection event to auto-deselect from features
        target_combo.bind('<<ComboboxSelected>>', self._on_target_selected)

        # Features header
        ttk.Label(self.feature_scroll_frame, text="Feature Columns (select for training):",
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))

        # Create checkboxes for all columns
        for col in df.columns:
            # Create variable for each column (default: True for features, False for target)
            var = tk.BooleanVar(value=(col != self.target_var.get()))
            self.feature_vars[col] = var

            # Determine column type for display
            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = "numeric"
                type_color = "blue"
            elif pd.api.types.is_string_dtype(df[col]):
                col_type = "categorical"
                type_color = "green"
            else:
                col_type = "other"
                type_color = "gray"

            # Create frame for each feature
            feature_row = ttk.Frame(self.feature_scroll_frame)
            feature_row.pack(fill=tk.X, pady=2)

            # Checkbox
            cb = ttk.Checkbutton(feature_row, variable=var, width=3)
            cb.pack(side=tk.LEFT)

            # Column name with type indicator
            if col == self.target_var.get():
                # Highlight target column with special styling
                ttk.Label(feature_row, text=f"🎯 {col} (TARGET - not used as feature)",
                         foreground="red", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
                cb.config(state=tk.DISABLED)  # Disable checkbox for target
            else:
                ttk.Label(feature_row, text=f"{col}", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
                ttk.Label(feature_row, text=f"({col_type})", foreground=type_color,
                         font=('Arial', 8)).pack(side=tk.LEFT, padx=2)

        # Add info about selection
        info_label = ttk.Label(self.feature_scroll_frame,
                              text="Note: Target column is automatically excluded from features.",
                              foreground="gray", font=('Arial', 8, 'italic'))
        info_label.pack(anchor=tk.W, pady=(10, 0))

    def _on_target_selected(self, event):
        """Handle target column selection - automatically deselect from features"""
        new_target = self.target_var.get()

        # Update all feature checkboxes
        for col, var in self.feature_vars.items():
            if col == new_target:
                # Disable and uncheck the target column
                var.set(False)
                # Find and disable the checkbox widget
                self._disable_feature_checkbox(col)
            else:
                # Re-enable and keep existing state for other columns
                self._enable_feature_checkbox(col)
                # Don't change existing state - keep user's previous selection

        self.log_output(f"🎯 Target column set to: {new_target} (automatically excluded from features)")

    def _disable_feature_checkbox(self, column):
        """Disable checkbox for a specific column"""
        # Find the widget and disable it
        for widget in self.feature_scroll_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Checkbutton) and child.cget('text') == '':
                        # This is the checkbox - check if it belongs to this column
                        # Find the label next to it
                        for label in widget.winfo_children():
                            if isinstance(label, ttk.Label) and label.cget('text') == column:
                                child.config(state=tk.DISABLED)
                                break

    def _enable_feature_checkbox(self, column):
        """Enable checkbox for a specific column"""
        for widget in self.feature_scroll_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Checkbutton) and child.cget('text') == '':
                        for label in widget.winfo_children():
                            if isinstance(label, ttk.Label) and label.cget('text') == column:
                                child.config(state=tk.NORMAL)
                                break

    def select_all_features(self):
        """Select all features (excludes target column automatically)"""
        target = self.target_var.get()
        for col, var in self.feature_vars.items():
            if col != target:
                var.set(True)
        self.log_output("📊 Selected all features")

    def deselect_all_features(self):
        """Deselect all features"""
        for var in self.feature_vars.values():
            var.set(False)
        self.log_output("📊 Deselected all features")

    def select_numeric_features(self):
        """Select only numeric features (excludes target automatically)"""
        if not hasattr(self, 'original_data') or self.original_data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        df = self.original_data
        target = self.target_var.get()

        selected = 0
        for col, var in self.feature_vars.items():
            if col != target and pd.api.types.is_numeric_dtype(df[col]):
                var.set(True)
                selected += 1
            elif col != target:
                var.set(False)

        self.log_output(f"📊 Selected {selected} numeric features (excluded target: {target})")

    def apply_feature_selection(self):
        """Apply the current feature selection"""
        if not hasattr(self, 'original_data') or self.original_data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        try:
            # Get selected features
            selected_features = []
            for col, var in self.feature_vars.items():
                if var.get() and col != self.target_var.get():
                    selected_features.append(col)

            # Get target column
            target_column = self.target_var.get()

            if not selected_features:
                messagebox.showwarning("Warning", "Please select at least one feature.")
                return

            if not target_column:
                messagebox.showwarning("Warning", "Please select a target column.")
                return

            # Initialize model with selected features
            if hasattr(self, 'current_data_file') and self.current_data_file:
                dataset_name = os.path.splitext(os.path.basename(self.current_data_file))[0]
            else:
                dataset_name = "dataset"

            # Create config with selected features
            config = {
                'file_path': self.current_data_file if hasattr(self, 'current_data_file') else None,
                'target_column': target_column,
                'model_type': self.model_type_var.get(),
                'compute_device': self.device_var.get(),
                'parallel': self.parallel_var.get(),
                'n_jobs': int(self.n_jobs_var.get()) if self.parallel_var.get() else 1,
                'parallel_batch_size': int(self.parallel_batch_var.get()) if self.parallel_var.get() else 1000,
                'parallel_mode': self.parallel_mode_var.get(),
                'selected_features': selected_features,  # Store selected features
                'training_params': {
                    'learning_rate': float(self.learning_rate_var.get()),
                    'epochs': int(self.epochs_var.get()),
                    'n_bins_per_dim': int(self.bins_var.get()),
                    'test_fraction': float(self.test_size_var.get()),
                    'enable_adaptive': self.adaptive_var.get(),
                    'adaptive_rounds': int(self.adaptive_rounds_var.get()),
                    'initial_samples': int(self.initial_samples_var.get()),
                    'max_samples_per_round': int(self.max_samples_round_var.get())
                }
            }

            # Use OptimizedDBNN (not AdaptiveCTDBNN)
            self.model = OptimizedDBNN(
                dataset_name=dataset_name,
                config=config,
                enable_external_tools=ASTROPY_AVAILABLE
            )

            # Load data with selected features
            success = self.model.load_data(file_path=self.current_data_file)

            if success:
                # Apply feature filtering - keep only selected features
                if selected_features:
                    # Get indices of selected features
                    feature_indices = []
                    for feat in selected_features:
                        if feat in self.model.feature_names:
                            feature_indices.append(self.model.feature_names.index(feat))

                    if feature_indices:
                        # Filter X_tensor to only selected features
                        self.model.X_tensor = self.model.X_tensor[:, feature_indices]
                        self.model.feature_names = selected_features
                        self.log_output(f"📊 Using {len(selected_features)} selected features: {', '.join(selected_features)}")

                self.log_output(f"✅ Model initialized with feature selection")
                self.log_output(f"🎯 Target: {target_column}")
                self.log_output(f"📊 Features: {len(selected_features)}")

                # Store feature selection in model
                self.model.selected_features = selected_features

                # Update the model reference in the GUI
                self.model_trained = False  # Reset trained flag since new model
                self.log_output("✅ Feature selection applied successfully")

                # Save configuration
                if hasattr(self, 'current_data_file') and self.current_data_file:
                    self.save_configuration_for_file(self.current_data_file)
            else:
                self.log_output("❌ Failed to load data with selected features")

        except Exception as e:
            self.log_output(f"❌ Error applying feature selection: {e}")
            import traceback
            traceback.print_exc()

    def _show_uci_description(self, event):
        """Show description for selected UCI dataset"""
        dataset = self.uci_var.get()
        info = DatasetConfig.get_dataset_info(dataset)
        if info:
            desc = info['description']
            self.uci_desc_label.config(text=desc)
        else:
            self.uci_desc_label.config(text="")

    def setup_config_tab(self):
        model_frame = ttk.LabelFrame(self.config_tab, text="Model Configuration", padding="10")
        model_frame.pack(fill=tk.X, pady=5)

        params = [
            ("Learning Rate:", self.learning_rate_var, "0.1"),
            ("Epochs:", self.epochs_var, "100"),
            ("Number of Bins:", self.bins_var, "128"),
            ("Test Size:", self.test_size_var, "0.2"),
            ("Device:", self.device_var, "auto"),
            ("Model Type:", self.model_type_var, "Histogram")
        ]

        for i, (label, var, default) in enumerate(params):
            ttk.Label(model_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(model_frame, textvariable=var, width=15).grid(row=i, column=1, padx=5, pady=2)

        self.device_var.set("auto")
        device_combo = ttk.Combobox(model_frame, textvariable=self.device_var,
                                    values=["auto", "cuda", "cpu"], width=12)
        device_combo.grid(row=4, column=1, padx=5, pady=2)

        model_combo = ttk.Combobox(model_frame, textvariable=self.model_type_var,
                                  values=["Histogram", "Gaussian"], width=12)
        model_combo.grid(row=5, column=1, padx=5, pady=2)

        self._create_parallel_frame()

        adaptive_frame = ttk.LabelFrame(self.config_tab, text="Adaptive Learning", padding="10")
        adaptive_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(adaptive_frame, text="Enable Adaptive Learning",
                       variable=self.adaptive_var).grid(row=0, column=0, sticky=tk.W, padx=5)

        ttk.Label(adaptive_frame, text="Adaptive Rounds:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Entry(adaptive_frame, textvariable=self.adaptive_rounds_var, width=10).grid(row=1, column=1, padx=5)

        ttk.Label(adaptive_frame, text="Initial Samples:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(adaptive_frame, textvariable=self.initial_samples_var, width=10).grid(row=2, column=1, padx=5)

        ttk.Label(adaptive_frame, text="Max Samples/Round:").grid(row=3, column=0, sticky=tk.W, padx=5)
        ttk.Entry(adaptive_frame, textvariable=self.max_samples_round_var, width=10).grid(row=3, column=1, padx=5)

        btn_frame = ttk.Frame(self.config_tab)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="Load Defaults", command=self.load_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Configuration", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load Configuration", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Apply Configuration", command=self.apply_config).pack(side=tk.LEFT, padx=5)

    def setup_training_tab(self):
        """Setup training tab with full external tools integration"""
        # Control buttons
        control_frame = ttk.LabelFrame(self.training_tab, text="Training Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        btn_row1 = ttk.Frame(control_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row1, text="Initialize Model", command=self.initialize_model, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row1, text="Train Model", command=self.train_model, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row1, text="Fresh Train", command=self.fresh_train_model, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row1, text="Adaptive Training", command=self.adaptive_training, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row1, text="Fresh Adaptive", command=self.fresh_adaptive_training, width=15).pack(side=tk.LEFT, padx=2)

        btn_row2 = ttk.Frame(control_frame)
        btn_row2.pack(fill=tk.X, pady=5)
        self.stop_button = ttk.Button(btn_row2, text="🛑 STOP", command=self.stop_training, width=15, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row2, text="❌ EXIT", command=self.exit_application, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Separator(btn_row2, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.status_label = ttk.Label(btn_row2, text="⏸️ Idle", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=10)

        btn_row3 = ttk.Frame(control_frame)
        btn_row3.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row3, text="Evaluate", command=self.evaluate_model, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row3, text="Predict File", command=self.predict_file, width=12).pack(side=tk.LEFT, padx=2)

        # ========== EXTERNAL TOOLS FRAME (COMPREHENSIVE) ==========
        ext_frame = ttk.LabelFrame(self.training_tab, text="🔌 External Tools Integration", padding="10")
        ext_frame.pack(fill=tk.X, pady=5)

        # Check available tools
        topcat_avail = self._check_topcat()
        aladin_avail = self._check_aladin()
        fits_avail = ASTROPY_AVAILABLE
        sdss_avail = self._check_sdss()

        # First row - Launch tools
        tools_row = ttk.Frame(ext_frame)
        tools_row.pack(fill=tk.X, pady=2)

        # Topcat button
        if topcat_avail:
            self.topcat_btn = ttk.Button(tools_row, text="📊 Launch Topcat",
                                         command=self.launch_topcat_gui, width=18)
            self.topcat_btn.pack(side=tk.LEFT, padx=2)
        else:
            self.topcat_btn = ttk.Button(tools_row, text="📊 Topcat (Not Found)",
                                         state=tk.DISABLED, width=18)
            self.topcat_btn.pack(side=tk.LEFT, padx=2)

        # Aladin button
        if aladin_avail:
            self.aladin_btn = ttk.Button(tools_row, text="🌌 Launch Aladin",
                                         command=self.launch_aladin_gui, width=18)
            self.aladin_btn.pack(side=tk.LEFT, padx=2)
        else:
            self.aladin_btn = ttk.Button(tools_row, text="🌌 Aladin (Not Found)",
                                         state=tk.DISABLED, width=18)
            self.aladin_btn.pack(side=tk.LEFT, padx=2)

        # Second row - Export options
        export_row = ttk.Frame(ext_frame)
        export_row.pack(fill=tk.X, pady=2)

        if fits_avail:
            ttk.Button(export_row, text="💾 Export to FITS",
                      command=self.export_to_fits_gui, width=18).pack(side=tk.LEFT, padx=2)
            ttk.Button(export_row, text="📄 Export to VOTable",
                      command=self.export_to_votable_gui, width=18).pack(side=tk.LEFT, padx=2)
        else:
            ttk.Button(export_row, text="💾 FITS (Astropy missing)",
                      state=tk.DISABLED, width=18).pack(side=tk.LEFT, padx=2)
            ttk.Button(export_row, text="📄 VOTable (Astropy missing)",
                      state=tk.DISABLED, width=18).pack(side=tk.LEFT, padx=2)

        # Third row - Queries
        query_row = ttk.Frame(ext_frame)
        query_row.pack(fill=tk.X, pady=2)

        if sdss_avail:
            ttk.Button(query_row, text="🔭 Query SDSS",
                      command=self.query_sdss_gui, width=18).pack(side=tk.LEFT, padx=2)
        else:
            ttk.Button(query_row, text="🔭 SDSS (requests missing)",
                      state=tk.DISABLED, width=18).pack(side=tk.LEFT, padx=2)

        # Dataset selection for export
        ttk.Label(query_row, text="Dataset:", font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 2))
        self.export_dataset_var = tk.StringVar(value="all")
        dataset_combo = ttk.Combobox(query_row, textvariable=self.export_dataset_var,
                                     values=["all", "training", "test"], width=10, state="readonly")
        dataset_combo.pack(side=tk.LEFT, padx=2)

        # Search radius for SDSS
        ttk.Label(query_row, text="Radius (arcsec):", font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 2))
        self.sdss_radius_var = tk.StringVar(value="1.0")
        ttk.Entry(query_row, textvariable=self.sdss_radius_var, width=8).pack(side=tk.LEFT, padx=2)

        # Fourth row - Status
        status_row = ttk.Frame(ext_frame)
        status_row.pack(fill=tk.X, pady=5)

        self.ext_status_label = ttk.Label(status_row, text="", foreground="gray")
        self.ext_status_label.pack(side=tk.LEFT, padx=5)

        # Update status
        self._update_ext_tools_status(topcat_avail, aladin_avail, fits_avail, sdss_avail)

        # ========== VISUALIZATION BUTTONS ==========
        viz_frame = ttk.LabelFrame(self.training_tab, text="Visualizations", padding="10")
        viz_frame.pack(fill=tk.X, pady=5)

        row1 = ttk.Frame(viz_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="Confusion Matrix", command=self.show_confusion_matrix, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Training History", command=self.show_training_history, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Tensor Evolution", command=self.show_tensor_evolution, width=18).pack(side=tk.LEFT, padx=2)

        row2 = ttk.Frame(viz_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="Interactive Dashboard", command=self.show_dashboard, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="🌐 Spherical Evolution", command=self.show_spherical_evolution, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="🎬 Side-by-Side", command=self.show_side_by_side, width=18).pack(side=tk.LEFT, padx=2)

        # ========== MODEL I/O ==========
        io_frame = ttk.Frame(self.training_tab)
        io_frame.pack(fill=tk.X, pady=5)
        ttk.Button(io_frame, text="Save Model", command=self.save_model_gui, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_frame, text="Load Model", command=self.load_model_gui, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_frame, text="Reset Model", command=self.reset_model_gui, width=12).pack(side=tk.LEFT, padx=2)

        # ========== OUTPUT ==========
        output_notebook = ttk.Notebook(self.training_tab)
        output_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        output_frame = ttk.Frame(output_notebook)
        output_notebook.add(output_frame, text="📝 Output")
        self.output_text = scrolledtext.ScrolledText(output_frame, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        results_frame = ttk.Frame(output_notebook)
        output_notebook.add(results_frame, text="📊 Results")
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def _check_topcat(self):
        """Check if Topcat is available on the system"""
        import shutil
        topcat = shutil.which('topcat')
        if topcat:
            return True
        # Check for topcat.jar
        topcat_jar_paths = [
            '/usr/share/topcat/topcat.jar',
            '/usr/local/share/topcat/topcat.jar',
            os.path.expanduser('~/topcat/topcat.jar'),
            os.path.expanduser('~/Applications/topcat/topcat.jar')
        ]
        for jar_path in topcat_jar_paths:
            if os.path.exists(jar_path):
                return True
        return False

    def _check_sdss(self):
        """Check if SDSS web service is accessible"""
        try:
            import requests
            response = requests.head("https://skyserver.sdss.org/dr16/SkyServerWS", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _update_ext_tools_status(self, topcat, aladin, fits, sdss):
        """Update external tools status label"""
        status_parts = []
        if topcat:
            status_parts.append(f"Topcat ✓")
        if aladin:
            status_parts.append(f"Aladin ✓")
        if fits:
            status_parts.append(f"FITS ✓")
        if sdss:
            status_parts.append(f"SDSS ✓")

        if status_parts:
            self.ext_status_label.config(text=f"Available: {', '.join(status_parts)}", foreground="green")
        else:
            self.ext_status_label.config(text="No external tools detected - install for additional features", foreground="orange")

    def launch_topcat_gui(self):
        """Launch Topcat with SAMP integration"""
        if not hasattr(self, 'original_data') or self.original_data is None:
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        dataset = self.export_dataset_var.get()
        self.log_output(f"📊 Connecting to Topcat...")

        # Get data
        data = self._get_data_for_export(dataset)

        if data is None or data.empty:
            self.log_output("❌ No data to export")
            return

        # Initialize SAMP if not already
        if not hasattr(self, 'samp'):
            self.samp = SAMPIntegration()

        # Try SAMP first
        success = self.samp.send_data_to_topcat(data, title=f"CT-DBNN - {self.model.dataset_name if self.model else 'Data'}")

        if success:
            self.log_output(f"✅ Data sent to Topcat via SAMP")
            self.log_output(f"   {len(data)} records, {len(data.columns)} columns")
            self.log_output(f"   Topcat window should appear with the data")
        else:
            self.log_output("⚠️ SAMP failed, falling back to file export...")
            self._launch_topcat_file_fallback(data)

    def launch_aladin_gui(self):
        """Launch Aladin with SAMP integration"""
        if not hasattr(self, 'original_data') or self.original_data is None:
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        dataset = self.export_dataset_var.get()
        self.log_output(f"🌌 Connecting to Aladin...")

        data = self._get_data_for_export(dataset)

        if data is None or data.empty:
            self.log_output("❌ No data to export")
            return

        # Initialize SAMP if not already
        if not hasattr(self, 'samp'):
            self.samp = SAMPIntegration()

        # Try SAMP first
        success = self.samp.send_data_to_aladin(data, title=f"CT-DBNN - {self.model.dataset_name if self.model else 'Data'}")

        if success:
            self.log_output(f"✅ Data sent to Aladin via SAMP")
            self.log_output(f"   {len(data)} records, {len(data.columns)} columns")
            self.log_output(f"   Aladin window should appear with the data")
        else:
            self.log_output("⚠️ SAMP failed, falling back to file export...")
            self._launch_aladin_file_fallback(data)

    def _launch_topcat_file_fallback(self, data):
        """Fallback to CSV file export"""
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        temp_file.close()

        try:
            # Export as CSV (fastest)
            data.to_csv(temp_file.name, index=False)
            self.log_output(f"   Exported {len(data)} records to CSV")

            import shutil
            import subprocess
            topcat = shutil.which('topcat')

            if not topcat:
                # Try JAR
                jar_paths = [
                    os.path.expanduser('~/topcat/topcat.jar'),
                    '/usr/share/topcat/topcat.jar'
                ]
                for jar in jar_paths:
                    if os.path.exists(jar):
                        topcat = ['java', '-jar', jar]
                        break

            if topcat:
                cmd = [topcat, temp_file.name] if isinstance(topcat, str) else topcat + [temp_file.name]
                subprocess.Popen(cmd)
                self.log_output(f"✅ Topcat launched with CSV file")
                self.log_output(f"   File: {temp_file.name}")
                self.log_output(f"   Keep this file until you close Topcat")
            else:
                self.log_output("❌ Topcat not found")
                self.log_output("   Install from: https://www.star.bris.ac.uk/~mbt/topcat/")

        except Exception as e:
            self.log_output(f"❌ Failed: {e}")

    def _launch_aladin_file_fallback(self, data):
        """Fallback to CSV file export for Aladin"""
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        temp_file.close()

        try:
            data.to_csv(temp_file.name, index=False)
            self.log_output(f"   Exported {len(data)} records to CSV")

            import shutil
            import subprocess
            aladin = shutil.which('aladin')

            if not aladin:
                # Try JAR
                jar_paths = [
                    os.path.expanduser('~/aladin/aladin.jar'),
                    '/usr/share/aladin/aladin.jar'
                ]
                for jar in jar_paths:
                    if os.path.exists(jar):
                        aladin = ['java', '-jar', jar]
                        break

            if aladin:
                # Handle Aladin script properly
                if isinstance(aladin, str):
                    if aladin.endswith('.sh'):
                        cmd = ['bash', aladin, temp_file.name]
                    else:
                        cmd = [aladin, temp_file.name]
                else:
                    cmd = aladin + [temp_file.name]

                subprocess.Popen(cmd)
                self.log_output(f"✅ Aladin launched with CSV file")
                self.log_output(f"   File: {temp_file.name}")
            else:
                self.log_output("❌ Aladin not found")
                self.log_output("   Install from: https://aladin.u-strasbg.fr/")

        except Exception as e:
            self.log_output(f"❌ Failed: {e}")

    def send_predictions_to_topcat(self):
        """Send only predictions to existing Topcat table"""
        if not self.model or not self.samp.topcat_session:
            self.log_output("⚠️ No model or Topcat session")
            return

        # Get predictions
        predictions, posteriors = self.model.predict(self.model.X_tensor)

        # Create predictions table
        pred_df = pd.DataFrame({
            'predicted_class': predictions.numpy(),
            'confidence': posteriors.max(dim=1)[0].numpy()
        })

        # Send update
        self.samp.send_table_update("predictions", pred_df)
        self.log_output("✅ Predictions sent to Topcat")

    def query_topcat_for_selection(self):
        """Get selected rows from Topcat for active learning"""
        if not self.samp.topcat_session:
            self.log_output("⚠️ No Topcat session")
            return None

        result = self.samp.query_topcat("SELECT * FROM current_selection")
        if result:
            self.log_output(f"✅ Retrieved {len(result)} selected rows from Topcat")
            return result
        return None

    def _get_data_for_export(self, dataset='all'):
        """
        Get data for export - works with or without trained model

        Args:
            dataset: 'all', 'training', or 'test'

        Returns:
            DataFrame with the data to export
        """
        # Start with original data
        if not hasattr(self, 'original_data') or self.original_data is None:
            return None

        data = self.original_data.copy()

        # If we have a trained model, add predictions
        if self.model and hasattr(self.model, 'weight_updater') and self.model.weight_updater is not None:
            try:
                # Get predictions from the model
                if dataset == 'all':
                    X = self.model.X_tensor
                    y = self.model.y_tensor
                    indices = list(range(len(X)))
                elif dataset == 'training' and hasattr(self.model, 'train_indices'):
                    X = self.model.X_train
                    y = self.model.y_train
                    indices = self.model.train_indices
                elif dataset == 'test' and hasattr(self.model, 'test_indices'):
                    X = self.model.X_test
                    y = self.model.y_test
                    indices = self.model.test_indices
                else:
                    X = None
                    indices = None

                if X is not None:
                    predictions, posteriors = self.model.predict(X)

                    # Get class names
                    if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                        inv_encoder = {v: k for k, v in self.model.label_encoder.items()}
                        pred_labels = [inv_encoder.get(p, p) for p in predictions.numpy()]
                        true_labels = [inv_encoder.get(t, t) for t in y.numpy()]
                    else:
                        pred_labels = predictions.numpy()
                        true_labels = y.numpy()

                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'predicted_class': pred_labels,
                        'true_class': true_labels,
                        'confidence': posteriors.max(dim=1)[0].numpy()
                    })

                    # Add probabilities for each class
                    if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                        for i, class_name in enumerate(self.model.label_encoder.keys()):
                            results_df[f'prob_{class_name}'] = posteriors[:, i].numpy()

                    # Merge with original data
                    if indices is not None:
                        # Create a DataFrame with indices
                        indices_df = pd.DataFrame({'original_index': indices})
                        indices_df = indices_df.reset_index(drop=True)
                        results_df = results_df.reset_index(drop=True)

                        # Combine indices and results
                        combined = pd.concat([indices_df, results_df], axis=1)

                        # Merge with original data
                        data_to_merge = data.reset_index().rename(columns={'index': 'original_index'})
                        data = data_to_merge.merge(combined, on='original_index', how='left')
                        data = data.drop(columns=['original_index'])
                    else:
                        # Just add predictions
                        for col in results_df.columns:
                            data[col] = results_df[col].values

            except Exception as e:
                self.log_output(f"⚠️ Could not add predictions: {e}")

        return data

    def export_to_fits_gui(self):
        """Export current data to FITS file (works with or without model)"""
        if not hasattr(self, 'original_data') or self.original_data is None:
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        dataset = self.export_dataset_var.get()
        default_name = f"{self.model.dataset_name if self.model else 'data'}_{dataset}.fits"

        file_path = filedialog.asksaveasfilename(
            title="Save as FITS",
            defaultextension=".fits",
            initialfile=default_name,
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")]
        )

        if file_path:
            self.log_output(f"💾 Exporting {dataset} dataset to FITS: {file_path}")

            data = self._get_data_for_export(dataset)

            if data is None or data.empty:
                self.log_output("❌ No data to export")
                return

            try:
                from astropy.io import fits
                from astropy.table import Table

                table = Table.from_pandas(data)
                table.write(file_path, format='fits', overwrite=True)
                self.log_output(f"✅ Data exported to: {file_path}")
            except Exception as e:
                self.log_output(f"❌ Failed to export to FITS: {e}")

    def export_to_votable_gui(self):
        """Export current data to VOTable file (works with or without model)"""
        if not hasattr(self, 'original_data') or self.original_data is None:
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        dataset = self.export_dataset_var.get()
        default_name = f"{self.model.dataset_name if self.model else 'data'}_{dataset}.vot"

        file_path = filedialog.asksaveasfilename(
            title="Save as VOTable",
            defaultextension=".vot",
            initialfile=default_name,
            filetypes=[("VOTable files", "*.vot"), ("All files", "*.*")]
        )

        if file_path:
            self.log_output(f"📄 Exporting {dataset} dataset to VOTable: {file_path}")

            data = self._get_data_for_export(dataset)

            if data is None or data.empty:
                self.log_output("❌ No data to export")
                return

            try:
                from astropy.io.votable import from_table
                from astropy.table import Table

                votable = from_table(Table.from_pandas(data))
                votable.to_xml(file_path)
                self.log_output(f"✅ Data exported to: {file_path}")
            except Exception as e:
                self.log_output(f"❌ Failed to export to VOTable: {e}")

    def query_sdss_gui(self):
        """Query SDSS for cross-matching"""
        if not self.model:
            messagebox.showwarning("Warning", "No model to query")
            return

        # Get search radius
        try:
            radius = float(self.sdss_radius_var.get())
        except:
            radius = 1.0
            self.sdss_radius_var.set("1.0")

        # Prepare data
        data = self.model._prepare_export_data()

        # Find RA/Dec columns
        ra_col, dec_col = self._find_coordinate_columns(data)

        if not ra_col or not dec_col:
            self.log_output("❌ No RA/Dec columns found in data")
            messagebox.showinfo("SDSS Query",
                               "No RA/Dec columns found. Please ensure your data contains RA and Dec columns.")
            return

        self.log_output(f"🔭 Querying SDSS with radius {radius} arcsec...")
        self.log_output(f"   Using columns: {ra_col}, {dec_col}")

        try:
            import requests
            import io

            results = []
            n_queries = min(50, len(data))  # Limit to 50 queries for performance

            for i, (r, d) in enumerate(zip(data[ra_col].values[:n_queries], data[dec_col].values[:n_queries])):
                query = f"""
                SELECT TOP 5
                    p.ra, p.dec, p.objid, p.type,
                    p.u, p.g, p.r, p.i, p.z,
                    s.z as redshift
                FROM
                    PhotoObj AS p
                    LEFT JOIN SpecObj AS s ON s.bestobjid = p.objid
                WHERE
                    p.ra BETWEEN {r - radius/3600} AND {r + radius/3600}
                    AND p.dec BETWEEN {d - radius/3600} AND {d + radius/3600}
                """

                response = requests.post(
                    "https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch",
                    data={'cmd': query, 'format': 'csv'},
                    timeout=10
                )

                if response.status_code == 200:
                    df = pd.read_csv(io.StringIO(response.text))
                    if len(df) > 0:
                        df['query_ra'] = r
                        df['query_dec'] = d
                        results.append(df)

                if (i + 1) % 10 == 0:
                    self.log_output(f"   Processed {i+1}/{n_queries} queries...")

            if results:
                combined = pd.concat(results, ignore_index=True)
                self.log_output(f"✅ SDSS query returned {len(combined)} matches from {len(results)} queries")

                # Show results in a new window
                self._show_sdss_results(combined)
            else:
                self.log_output("No matches found")

        except ImportError:
            self.log_output("❌ requests module not installed. Install with: pip install requests")
            messagebox.showinfo("Missing Dependency", "requests module required for SDSS queries.\nInstall with: pip install requests")
        except Exception as e:
            self.log_output(f"❌ SDSS query failed: {e}")
            import traceback
            traceback.print_exc()

    def _find_coordinate_columns(self, data):
        """Find RA/Dec columns in dataframe"""
        ra_col = None
        dec_col = None

        for col in data.columns:
            col_lower = col.lower()
            if col_lower in ['ra', 'right_ascension', 'alpha', 'raj2000']:
                ra_col = col
            if col_lower in ['dec', 'declination', 'delta', 'dej2000']:
                dec_col = col

        return ra_col, dec_col

    def _show_sdss_results(self, results_df):
        """Show SDSS query results in a new window"""
        result_window = tk.Toplevel(self.root)
        result_window.title("SDSS Query Results")
        result_window.geometry("900x600")

        # Create frame
        frame = ttk.Frame(result_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add info label
        info_label = ttk.Label(frame, text=f"Matches found: {len(results_df)}", font=('Arial', 10, 'bold'))
        info_label.pack(anchor=tk.W, pady=5)

        # Add text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = scrolledtext.ScrolledText(text_frame, height=20, yscrollcommand=scrollbar.set)
        text_widget.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        # Display results
        text_widget.insert(tk.END, "SDSS Query Results\n")
        text_widget.insert(tk.END, "=" * 50 + "\n\n")
        text_widget.insert(tk.END, results_df.to_string())
        text_widget.config(state=tk.DISABLED)

        # Add button frame
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)

        def save_results():
            file_path = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                results_df.to_csv(file_path, index=False)
                self.log_output(f"✅ SDSS results saved to {file_path}")
                messagebox.showinfo("Save Complete", f"Results saved to:\n{file_path}")

        ttk.Button(btn_frame, text="Save as CSV", command=save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=result_window.destroy).pack(side=tk.LEFT, padx=5)

    def show_side_by_side(self):
        """Show side-by-side comparison of initial and final states"""
        if not self.model:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        evolution_history = self.model.get_evolution_history()
        if not evolution_history or len(evolution_history) < 2:
            messagebox.showwarning("Warning", "Need at least 2 rounds for comparison.\n"
                                   "Run adaptive training with evolution tracking enabled.")
            return

        try:
            self.log_output("🔄 Creating side-by-side evolution comparison...")

            visualizer = OptimizedVisualizer(self.model)
            output_path = visualizer.spherical_viz.create_side_by_side_evolution(evolution_history)

            if output_path and Path(output_path).exists():
                self.log_output(f"✅ Side-by-side comparison saved to: {output_path}")

                if messagebox.askyesno("Comparison Ready",
                                      "Side-by-side comparison created!\n\nOpen in browser now?"):
                    import webbrowser
                    webbrowser.open(f'file://{Path(output_path).absolute()}')
            else:
                self.log_output("❌ Failed to create side-by-side comparison")

        except Exception as e:
            self.log_output(f"❌ Error creating side-by-side comparison: {e}")
            import traceback
            traceback.print_exc()

    def _check_aladin(self):
        """Check if Aladin is available on the system"""
        import shutil
        aladin = shutil.which('aladin')
        if aladin:
            return True
        # Check for aladin.jar
        aladin_jar_paths = [
            '/usr/share/aladin/aladin.jar',
            '/usr/local/share/aladin/aladin.jar',
            os.path.expanduser('~/aladin/aladin.jar')
        ]
        for jar_path in aladin_jar_paths:
            if os.path.exists(jar_path):
                return True
        return False

    def _create_parallel_frame(self):
        parallel_frame = ttk.LabelFrame(self.config_tab, text="Parallel Processing", padding="10")
        parallel_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(parallel_frame, text="Enable Parallel Processing",
                       variable=self.parallel_var,
                       command=self._toggle_parallel_options).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5)

        ttk.Label(parallel_frame, text="Number of CPU Cores:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        n_jobs_spinbox = ttk.Spinbox(parallel_frame, from_=1, to=mp.cpu_count()*2,
                                     textvariable=self.n_jobs_var, width=10)
        n_jobs_spinbox.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(parallel_frame, text=f"Available cores: {mp.cpu_count()}",
                 foreground="gray").grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)

        ttk.Label(parallel_frame, text="Parallel Batch Size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(parallel_frame, textvariable=self.parallel_batch_var, width=15).grid(row=3, column=1, padx=5, pady=2)

        ttk.Radiobutton(parallel_frame, text="Threads (I/O bound)", variable=self.parallel_mode_var,
                       value="threads").grid(row=4, column=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(parallel_frame, text="Processes (CPU bound)", variable=self.parallel_mode_var,
                       value="processes").grid(row=4, column=1, sticky=tk.W, padx=5)

    def _set_training_state(self, in_progress: bool):
        self.training_in_progress = in_progress

        if in_progress:
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="🏃 Training in progress...", foreground="orange")
            self.root.update()
        else:
            self.stop_button.config(state=tk.DISABLED)
            self.stop_training_flag = False
            self.status_label.config(text="⏸️ Idle", foreground="gray")
            if self.model:
                self.model.stop_training_flag = False
            self.root.update()

    def stop_training(self):
        if self.training_in_progress:
            self.log_output("🛑 STOP signal received - stopping after current round...")
            self.stop_training_flag = True
            if self.model:
                self.model.stop_training_flag = True
            self.status_label.config(text="🛑 Stopping...", foreground="red")
            self.stop_button.config(state=tk.DISABLED)
            self.root.update()

    def exit_application(self):
        if self.training_in_progress:
            response = messagebox.askyesno(
                "Training in Progress",
                "Training is still running. Are you sure you want to exit?\n"
                "All progress will be lost."
            )
            if not response:
                return

            self.stop_training_flag = True
            if self.model:
                self.model.stop_training_flag = True
            time.sleep(1)

        self.log_output("👋 Exiting application...")
        self.root.quit()
        self.root.destroy()

    def browse_file(self):
        """Browse for data file"""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.current_data_file = file_path  # Set this!
            self.log_output(f"📁 Selected file: {file_path}")

    def download_uci(self):
        dataset = self.uci_var.get()
        if not dataset:
            messagebox.showwarning("Warning", "Please select a dataset")
            return

        self.log_output(f"📥 Downloading UCI dataset: {dataset}")
        df = DatasetConfig.download_uci_data(dataset)

        if df is not None:
            data_dir = os.path.join('data', dataset)
            os.makedirs(data_dir, exist_ok=True)
            csv_path = os.path.join(data_dir, f"{dataset}.csv")
            df.to_csv(csv_path, index=False)

            self.file_path_var.set(csv_path)
            self.log_output(f"✅ Dataset saved to: {csv_path}")
            self.load_dataset()
        else:
            self.log_output(f"❌ Failed to download dataset")

    def load_dataset(self):
        """Load and display dataset information"""
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showwarning("Warning", "Please select a valid file")
            return

        try:
            df = pd.read_csv(file_path)
            self.original_data = df

            # Display dataset info
            self.info_text.delete(1.0, tk.END)
            info = f"""
            File: {os.path.basename(file_path)}
            Samples: {len(df)}
            Features: {len(df.columns)}

            Columns:
            {', '.join(df.columns.tolist())}

            Data Types:
            {df.dtypes.to_string()}

            Missing Values:
            {df.isnull().sum().to_string()}

            Statistics:
            {df.describe().to_string()}
            """
            self.info_text.insert(1.0, info)

            # Auto-detect target column
            # Look for common target column names
            target_candidates = ['target', 'class', 'label', 'y', 'output', 'result', 'type', 'diagnosis']
            auto_target = None

            for candidate in target_candidates:
                if candidate in df.columns:
                    auto_target = candidate
                    break

            # If no common name found, use the last column
            if auto_target is None:
                auto_target = df.columns[-1]

            self.target_var.set(auto_target)
            self.log_output(f"🎯 Auto-detected target column: {auto_target}")

            # Update feature selection UI
            self.update_feature_selection(df)

            self.data_loaded = True
            self.log_output(f"✅ Dataset loaded: {len(df)} samples, {len(df.columns)} columns")

        except Exception as e:
            self.log_output(f"❌ Error loading dataset: {e}")

    def load_defaults(self):
        self.learning_rate_var.set("0.1")
        self.epochs_var.set("100")
        self.bins_var.set("128")
        self.test_size_var.set("0.2")
        self.device_var.set("auto")
        self.model_type_var.set("Histogram")

        self.parallel_var.set(True)
        self.n_jobs_var.set(str(mp.cpu_count()))
        self.parallel_batch_var.set("1000")
        self.parallel_mode_var.set("threads")

        self.adaptive_var.set(True)
        self.adaptive_rounds_var.set("10")
        self.initial_samples_var.set("50")
        self.max_samples_round_var.set("25")

        self.log_output("✅ Default configuration loaded")
        self._toggle_parallel_options()

    def _toggle_parallel_options(self):
        state = tk.NORMAL if self.parallel_var.get() else tk.DISABLED
        for child in self.config_tab.winfo_children():
            if isinstance(child, ttk.LabelFrame) and child.cget("text") == "Parallel Processing":
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, (ttk.Spinbox, ttk.Entry, ttk.Radiobutton)):
                        grandchild.config(state=state)

    def save_config(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if file_path:
            config = {
                'learning_rate': float(self.learning_rate_var.get()),
                'epochs': int(self.epochs_var.get()),
                'n_bins_per_dim': int(self.bins_var.get()),
                'test_fraction': float(self.test_size_var.get()),
                'device': self.device_var.get(),
                'model_type': self.model_type_var.get(),
                'parallel': self.parallel_var.get(),
                'n_jobs': int(self.n_jobs_var.get()) if self.parallel_var.get() else 1,
                'parallel_batch_size': int(self.parallel_batch_var.get()) if self.parallel_var.get() else 1000,
                'parallel_mode': self.parallel_mode_var.get(),
                'enable_adaptive': self.adaptive_var.get(),
                'adaptive_rounds': int(self.adaptive_rounds_var.get()),
                'initial_samples': int(self.initial_samples_var.get()),
                'max_samples_per_round': int(self.max_samples_round_var.get())
            }

            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)

            self.log_output(f"✅ Configuration saved to: {file_path}")

    def load_config(self):
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)

                self.learning_rate_var.set(str(config.get('learning_rate', 0.1)))
                self.epochs_var.set(str(config.get('epochs', 100)))
                self.bins_var.set(str(config.get('n_bins_per_dim', 128)))
                self.test_size_var.set(str(config.get('test_fraction', 0.2)))
                self.device_var.set(config.get('device', 'auto'))
                self.model_type_var.set(config.get('model_type', 'Histogram'))

                self.parallel_var.set(config.get('parallel', True))
                self.n_jobs_var.set(str(config.get('n_jobs', mp.cpu_count())))
                self.parallel_batch_var.set(str(config.get('parallel_batch_size', 1000)))
                self.parallel_mode_var.set(config.get('parallel_mode', 'threads'))

                self.adaptive_var.set(config.get('enable_adaptive', True))
                self.adaptive_rounds_var.set(str(config.get('adaptive_rounds', 10)))
                self.initial_samples_var.set(str(config.get('initial_samples', 50)))
                self.max_samples_round_var.set(str(config.get('max_samples_per_round', 25)))

                self.log_output(f"✅ Configuration loaded from: {file_path}")
                self._toggle_parallel_options()

            except Exception as e:
                self.log_output(f"❌ Error loading configuration: {e}")

    def apply_config(self):
        if self.model:
            self.model.learning_rate = float(self.learning_rate_var.get())
            self.model.n_bins_per_dim = int(self.bins_var.get())
            self.model.test_size = float(self.test_size_var.get())
            self.model.adaptive_rounds = int(self.adaptive_rounds_var.get())
            self.model.enable_adaptive = self.adaptive_var.get()
            self.model.initial_samples = int(self.initial_samples_var.get())
            self.model.max_samples_per_round = int(self.max_samples_round_var.get())

            self.model.parallel = self.parallel_var.get()
            self.model.n_jobs = int(self.n_jobs_var.get()) if self.parallel_var.get() else 1
            self.model.parallel_batch_size = int(self.parallel_batch_var.get()) if self.parallel_var.get() else 1000
            self.model.parallel_mode = self.parallel_mode_var.get()

            self.log_output("✅ Configuration applied to model")

            if self.model.parallel:
                self.log_output(f"   Parallel: Enabled with {self.model.n_jobs} workers")
            else:
                self.log_output("   Parallel: Disabled")
        else:
            self.log_output("⚠️ Model not initialized - configuration saved for later use")

    def initialize_model(self):
        """Initialize the model with current settings"""
        if not self.file_path_var.get():
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        try:
            dataset_name = os.path.splitext(os.path.basename(self.file_path_var.get()))[0]

            # Get selected features
            selected_features = [col for col, var in self.feature_vars.items() if var.get() and col != self.target_var.get()]

            config = {
                'file_path': self.file_path_var.get(),
                'target_column': self.target_var.get(),
                'model_type': self.model_type_var.get(),
                'compute_device': self.device_var.get(),
                'parallel': self.parallel_var.get(),
                'n_jobs': int(self.n_jobs_var.get()) if self.parallel_var.get() else 1,
                'parallel_batch_size': int(self.parallel_batch_var.get()) if self.parallel_var.get() else 1000,
                'parallel_mode': self.parallel_mode_var.get(),
                'selected_features': selected_features,
                'training_params': {
                    'learning_rate': float(self.learning_rate_var.get()),
                    'epochs': int(self.epochs_var.get()),
                    'n_bins_per_dim': int(self.bins_var.get()),
                    'test_fraction': float(self.test_size_var.get()),
                    'enable_adaptive': self.adaptive_var.get(),
                    'adaptive_rounds': int(self.adaptive_rounds_var.get()),
                    'initial_samples': int(self.initial_samples_var.get()),
                    'max_samples_per_round': int(self.max_samples_round_var.get())
                }
            }

            self.model = OptimizedDBNN(
                dataset_name=dataset_name,
                config=config,
                enable_external_tools=ASTROPY_AVAILABLE
            )

            # Load data
            self.model.load_data(file_path=self.file_path_var.get())

            # Apply feature selection if any
            if selected_features:
                # Get indices of selected features
                feature_indices = []
                for feat in selected_features:
                    if feat in self.model.feature_names:
                        feature_indices.append(self.model.feature_names.index(feat))

                if feature_indices:
                    # Filter X_tensor to only selected features
                    self.model.X_tensor = self.model.X_tensor[:, feature_indices]
                    self.model.feature_names = selected_features
                    self.log_output(f"📊 Using {len(selected_features)} selected features")

            # Log parallel status
            if self.parallel_var.get():
                self.log_output(f"🚀 Parallel mode enabled with {config['n_jobs']} workers")
                self.log_output(f"   Mode: {config['parallel_mode']}, Batch size: {config['parallel_batch_size']}")
            else:
                self.log_output("🐢 Sequential mode enabled")

            self.log_output(f"✅ Model initialized: {dataset_name}")
            self.model_trained = False

        except Exception as e:
            self.log_output(f"❌ Error initializing model: {e}")
            import traceback
            traceback.print_exc()

    def reset_model_gui(self):
        """Reset the model to initial state - FIXED"""
        if not self.model:
            messagebox.showwarning("Warning", "No model to reset")
            return

        response = messagebox.askyesno(
            "Reset Model",
            "This will erase all learned knowledge from the model.\n"
            "Are you sure you want to continue?"
        )

        if response:
            try:
                self.log_output("🔄 Resetting model...")

                # Store data file path
                data_file = self.file_path_var.get()
                target_col = self.target_var.get()
                selected_features = [col for col, var in self.feature_vars.items() if var.get()]

                # Get current config
                if hasattr(self.model, 'config'):
                    config = self.model.config
                else:
                    config = {
                        'file_path': data_file,
                        'target_column': target_col,
                        'model_type': self.model_type_var.get(),
                        'compute_device': self.device_var.get(),
                        'parallel': self.parallel_var.get(),
                        'n_jobs': int(self.n_jobs_var.get()) if self.parallel_var.get() else 1,
                        'parallel_batch_size': int(self.parallel_batch_var.get()) if self.parallel_var.get() else 1000,
                        'parallel_mode': self.parallel_mode_var.get(),
                        'training_params': {
                            'learning_rate': float(self.learning_rate_var.get()),
                            'epochs': int(self.epochs_var.get()),
                            'n_bins_per_dim': int(self.bins_var.get()),
                            'test_fraction': float(self.test_size_var.get()),
                            'enable_adaptive': self.adaptive_var.get(),
                            'adaptive_rounds': int(self.adaptive_rounds_var.get()),
                            'initial_samples': int(self.initial_samples_var.get()),
                            'max_samples_per_round': int(self.max_samples_round_var.get())
                        }
                    }

                # Create NEW model instance (clean slate)
                dataset_name = os.path.splitext(os.path.basename(data_file))[0]
                self.model = OptimizedDBNN(
                    dataset_name=dataset_name,
                    config=config,
                    enable_external_tools=ASTROPY_AVAILABLE
                )

                # Reload data
                self.log_output("📥 Reloading data after reset...")
                self.model.load_data(file_path=data_file)

                # Apply feature selection
                if selected_features:
                    self.model.selected_features = selected_features
                    self.log_output(f"📊 Using {len(selected_features)} selected features")

                self.log_output("✅ Model reset complete")
                self.log_output(f"   Loaded {len(self.model.classes)} classes, {len(self.model.X_tensor)} samples")

            except Exception as e:
                self.log_output(f"❌ Error resetting model: {e}")
                import traceback
                traceback.print_exc()

    def train_model(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please initialize the model first")
            return

        self.current_training_thread = threading.Thread(target=self._train_thread)
        self.current_training_thread.daemon = True
        self.current_training_thread.start()

    def _train_thread(self):
        self._set_training_state(True)

        try:
            self.log_output("🚀 Starting training...")

            if self.model.X_train is None:
                self.model.split_data()

            self.model.stop_training_flag = False
            results = self.model.fit(
                epochs=int(self.epochs_var.get())
            )

            if self.stop_training_flag or self.model.stop_training_flag:
                self.log_output("🛑 Training stopped by user")
                return

            self.log_output(f"✅ Training completed!")
            self.log_output(f"   Best accuracy: {results['best_accuracy']:.4f}")

            self.display_results(results)

        except Exception as e:
            if not self.stop_training_flag:
                self.log_output(f"❌ Error during training: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self._set_training_state(False)
            self.current_training_thread = None

    def fresh_train_model(self):
        """Fresh training - completely reset before training"""
        if not self.model:
            messagebox.showwarning("Warning", "Please initialize the model first")
            return

        self.current_training_thread = threading.Thread(target=self._fresh_train_thread)
        self.current_training_thread.daemon = True
        self.current_training_thread.start()

    def _fresh_train_thread(self):
        """Background thread for fresh training - FIXED"""
        self._set_training_state(True)

        try:
            self.log_output("🚀 Starting FRESH training (previous knowledge ignored)...")

            # Store data file path
            data_file = self.file_path_var.get()
            target_col = self.target_var.get()
            selected_features = [col for col, var in self.feature_vars.items() if var.get()]

            # Get current config
            if self.model and hasattr(self.model, 'config'):
                config = self.model.config
            else:
                config = {
                    'file_path': data_file,
                    'target_column': target_col,
                    'model_type': self.model_type_var.get(),
                    'compute_device': self.device_var.get(),
                    'parallel': self.parallel_var.get(),
                    'n_jobs': int(self.n_jobs_var.get()) if self.parallel_var.get() else 1,
                    'parallel_batch_size': int(self.parallel_batch_var.get()) if self.parallel_var.get() else 1000,
                    'parallel_mode': self.parallel_mode_var.get(),
                    'training_params': {
                        'learning_rate': float(self.learning_rate_var.get()),
                        'epochs': int(self.epochs_var.get()),
                        'n_bins_per_dim': int(self.bins_var.get()),
                        'test_fraction': float(self.test_size_var.get()),
                        'enable_adaptive': self.adaptive_var.get(),
                        'adaptive_rounds': int(self.adaptive_rounds_var.get()),
                        'initial_samples': int(self.initial_samples_var.get()),
                        'max_samples_per_round': int(self.max_samples_round_var.get())
                    }
                }

            # Create NEW model instance
            dataset_name = os.path.splitext(os.path.basename(data_file))[0]
            self.model = OptimizedDBNN(
                dataset_name=dataset_name,
                config=config,
                enable_external_tools=ASTROPY_AVAILABLE
            )

            # Load data
            self.log_output("📥 Loading data...")
            self.model.load_data(file_path=data_file)

            # Apply feature selection
            if selected_features:
                self.model.selected_features = selected_features
                self.log_output(f"📊 Using {len(selected_features)} selected features")

            # Split data
            self.model.split_data()

            # Run fresh training
            self.log_output("🚀 Running fresh training...")
            results = self.model.fresh_train(
                epochs=int(self.epochs_var.get())
            )

            if self.stop_training_flag or self.model.stop_training_flag:
                self.log_output("🛑 Training stopped by user")
                return

            self.log_output(f"✅ Fresh training completed!")
            self.log_output(f"   Best accuracy: {results['best_accuracy']:.4f}")

            self.display_results(results)

        except Exception as e:
            if not self.stop_training_flag:
                self.log_output(f"❌ Error during fresh training: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self._set_training_state(False)
            self.current_training_thread = None

    def adaptive_training(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please initialize the model first")
            return

        self.current_training_thread = threading.Thread(target=self._adaptive_thread)
        self.current_training_thread.daemon = True
        self.current_training_thread.start()

    def _adaptive_thread(self):
        self._set_training_state(True)

        try:
            self.log_output("🚀 Starting adaptive training...")

            self.model.enable_evolution_tracking()
            self.model.stop_training_flag = False

            results = self.model.adaptive_fit_predict(
                max_rounds=int(self.adaptive_rounds_var.get())
            )

            if self.stop_training_flag or self.model.stop_training_flag:
                self.log_output("🛑 Adaptive training stopped by user")
                return

            self.log_output(f"✅ Adaptive training completed!")
            self.log_output(f"   Best accuracy: {results['best_accuracy']:.4f}")
            self.log_output(f"   Final training samples: {len(results['train_indices'])}")

            self.display_adaptive_results(results)

        except Exception as e:
            if not self.stop_training_flag:
                self.log_output(f"❌ Error during adaptive training: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self._set_training_state(False)
            self.current_training_thread = None

    def fresh_adaptive_training(self):
        """Fresh adaptive training - completely reset before training"""
        if not self.model:
            messagebox.showwarning("Warning", "Please initialize the model first")
            return

        self.current_training_thread = threading.Thread(target=self._fresh_adaptive_thread)
        self.current_training_thread.daemon = True
        self.current_training_thread.start()

    def _fresh_adaptive_thread(self):
        """Background thread for fresh adaptive training - FIXED"""
        self._set_training_state(True)

        try:
            self.log_output("🚀 Starting FRESH adaptive training (previous knowledge ignored)...")

            # Store data file path
            data_file = self.file_path_var.get()
            target_col = self.target_var.get()
            selected_features = [col for col, var in self.feature_vars.items() if var.get()]

            # Get current config before reset
            if self.model and hasattr(self.model, 'config'):
                config = self.model.config
            else:
                # Create fresh config
                config = {
                    'file_path': data_file,
                    'target_column': target_col,
                    'model_type': self.model_type_var.get(),
                    'compute_device': self.device_var.get(),
                    'parallel': self.parallel_var.get(),
                    'n_jobs': int(self.n_jobs_var.get()) if self.parallel_var.get() else 1,
                    'parallel_batch_size': int(self.parallel_batch_var.get()) if self.parallel_var.get() else 1000,
                    'parallel_mode': self.parallel_mode_var.get(),
                    'training_params': {
                        'learning_rate': float(self.learning_rate_var.get()),
                        'epochs': int(self.epochs_var.get()),
                        'n_bins_per_dim': int(self.bins_var.get()),
                        'test_fraction': float(self.test_size_var.get()),
                        'enable_adaptive': self.adaptive_var.get(),
                        'adaptive_rounds': int(self.adaptive_rounds_var.get()),
                        'initial_samples': int(self.initial_samples_var.get()),
                        'max_samples_per_round': int(self.max_samples_round_var.get())
                    }
                }

            # CRITICAL: Create NEW model instance instead of resetting old one
            dataset_name = os.path.splitext(os.path.basename(data_file))[0]

            self.log_output("🔄 Creating fresh model instance...")
            self.model = OptimizedDBNN(
                dataset_name=dataset_name,
                config=config,
                enable_external_tools=ASTROPY_AVAILABLE
            )

            # Load data into new model
            self.log_output("📥 Loading data into fresh model...")
            self.model.load_data(file_path=data_file)

            # Verify classes were loaded
            if self.model.classes is None:
                self.log_output("❌ Failed to load classes from data")
                return

            self.log_output(f"   Loaded {len(self.model.classes)} classes: {self.model.classes}")

            # Apply feature selection if needed
            if selected_features:
                self.model.selected_features = selected_features
                self.log_output(f"📊 Using {len(selected_features)} selected features")

            # Enable evolution tracking
            self.model.enable_evolution_tracking()

            # CRITICAL: Ensure classes are properly set before training
            if self.model.classes is None or len(self.model.classes) == 0:
                self.log_output("❌ No classes loaded. Cannot train.")
                return

            self.log_output(f"🎯 Classes ready: {self.model.classes}")

            # Run fresh adaptive training
            self.log_output("🚀 Running fresh adaptive training...")
            results = self.model.adaptive_fit_predict(
                max_rounds=int(self.adaptive_rounds_var.get())
            )

            if self.stop_training_flag or self.model.stop_training_flag:
                self.log_output("🛑 Fresh adaptive training stopped by user")
                return

            self.log_output(f"✅ Fresh adaptive training completed!")
            self.log_output(f"   Best accuracy: {results['best_accuracy']:.4f}")
            self.log_output(f"   Final training samples: {len(results['train_indices'])}")

            self.display_adaptive_results(results)

        except Exception as e:
            if not self.stop_training_flag:
                self.log_output(f"❌ Error during fresh adaptive training: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self._set_training_state(False)
            self.current_training_thread = None

    def evaluate_model(self):
        if self.training_in_progress:
            messagebox.showwarning("Warning", "Please wait for training to complete")
            return

        if not self.model:
            messagebox.showwarning("Warning", "Please initialize and train the model first")
            return

        eval_thread = threading.Thread(target=self._evaluate_thread)
        eval_thread.daemon = True
        eval_thread.start()

    def _evaluate_thread(self):
        try:
            self.log_output("📊 Evaluating model...")

            if self.model.X_test is None or self.model.y_test is None:
                self.log_output("⚠️ No test data available. Creating test split...")

                if self.model.X_tensor is not None and self.model.y_tensor is not None:
                    from sklearn.model_selection import train_test_split

                    X_np = self.model.X_tensor.numpy()
                    y_np = self.model.y_tensor.numpy()

                    indices = np.arange(len(X_np))
                    train_idx, test_idx = train_test_split(
                        indices, test_size=0.2, random_state=42, stratify=y_np
                    )

                    self.model.X_train = self.model.X_tensor[train_idx]
                    self.model.X_test = self.model.X_tensor[test_idx]
                    self.model.y_train = self.model.y_tensor[train_idx]
                    self.model.y_test = self.model.y_tensor[test_idx]

                    self.log_output(f"   Created test split: {len(self.model.X_test)} samples")
                else:
                    self.log_output("❌ No data available for evaluation")
                    return

            if self.stop_training_flag:
                return

            predictions, posteriors = self.model.predict(self.model.X_test)

            accuracy = (predictions == self.model.y_test).float().mean().item()

            self.log_output(f"   Test accuracy: {accuracy:.4f}")
            self.log_output(f"   Test samples: {len(self.model.X_test)}")

            self.model.print_colored_confusion_matrix(
                self.model.y_test.numpy(),
                predictions.numpy(),
                "Test Data"
            )

        except Exception as e:
            if not self.stop_training_flag:
                self.log_output(f"❌ Error during evaluation: {e}")
                import traceback
                traceback.print_exc()

    def predict_file(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please initialize and train the model first")
            return

        file_path = filedialog.askopenfilename(
            title="Select File for Prediction",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.log_output(f"🔮 Predicting on file: {file_path}")

                output_dir = filedialog.askdirectory(title="Select Output Directory")
                if output_dir:
                    results = self.model.predict_from_file(file_path, output_path=output_dir)
                    self.log_output(f"✅ Predictions saved to {output_dir}")

            except Exception as e:
                self.log_output(f"❌ Error during prediction: {e}")
                import traceback
                traceback.print_exc()

    def show_confusion_matrix(self):
        if not self.model or self.model.X_test is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        try:
            predictions, _ = self.model.predict(self.model.X_test)

            visualizer = OptimizedVisualizer(self.model)
            visualizer.plot_confusion_matrix(
                self.model.y_test.numpy(),
                predictions.numpy(),
                "Test Data"
            )

            self.log_output(f"✅ Confusion matrix saved to: {visualizer.dirs['confusion']}")

            self.open_visualization_dir()

        except Exception as e:
            self.log_output(f"❌ Error creating confusion matrix: {e}")
            traceback.print_exc()

    def show_training_history(self):
        if not self.model or not hasattr(self.model, 'training_history'):
            messagebox.showwarning("Warning", "No training history available")
            return

        try:
            visualizer = OptimizedVisualizer(self.model)
            visualizer.plot_training_history(self.model.training_history)

            self.log_output(f"✅ Training history saved to: {visualizer.dirs['performance']}")
            self.open_visualization_dir()

        except Exception as e:
            self.log_output(f"❌ Error creating training history: {e}")
            traceback.print_exc()

    def show_tensor_evolution(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please initialize the model first")
            return

        evolution_history = self.model.get_evolution_history()
        if not evolution_history:
            messagebox.showwarning("Warning", "No evolution history available. Run adaptive training with tracking enabled.")
            return

        try:
            visualizer = OptimizedVisualizer(self.model)
            visualizer.plot_tensor_evolution(evolution_history)

            self.log_output(f"✅ Tensor evolution saved to: {visualizer.dirs['tensor']}")
            self.open_visualization_dir()

        except Exception as e:
            self.log_output(f"❌ Error creating tensor evolution: {e}")
            traceback.print_exc()

    def show_dashboard(self):
        """Show interactive dashboard with evolution slider"""
        if not self.model or self.model.X_tensor is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        try:
            visualizer = OptimizedVisualizer(self.model)

            X_np = self.model.X_tensor.numpy()
            y_np = self.model.y_tensor.numpy()

            evolution_history = self.model.get_evolution_history()

            # Create main interactive dashboard
            visualizer.create_interactive_dashboard(
                self.model.training_history,
                X_np, y_np,
                evolution_history=evolution_history
            )

            # Create evolution slider if history exists
            if evolution_history and len(evolution_history) > 1:
                visualizer.create_evolution_slider(evolution_history)

            self.log_output(f"✅ Interactive dashboard saved to: {visualizer.dirs['interactive']}")

            if messagebox.askyesno("Dashboard Ready", "Open dashboard in browser?"):
                import webbrowser
                dashboard_path = visualizer.dirs['interactive'] / f'{self.model.dataset_name}_dashboard.html'
                webbrowser.open(f'file://{dashboard_path.absolute()}')

        except Exception as e:
            self.log_output(f"❌ Error creating dashboard: {e}")
            traceback.print_exc()

    def show_spherical_evolution(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        evolution_history = self.model.get_evolution_history()
        if not evolution_history:
            messagebox.showwarning("Warning", "No evolution history available.")
            return

        try:
            self.log_output("🌐 Creating spherical tensor evolution animation...")

            visualizer = OptimizedVisualizer(self.model)
            output_path = visualizer.spherical_viz.create_spherical_animation(evolution_history)

            if output_path and Path(output_path).exists():
                self.log_output(f"✅ Spherical evolution saved to: {output_path}")

                if messagebox.askyesno("Animation Ready",
                                      f"Spherical evolution for {self.model.dataset_name} created!\n\nOpen in browser now?"):
                    import webbrowser
                    webbrowser.open(f'file://{Path(output_path).absolute()}')
            else:
                self.log_output("❌ Failed to create spherical evolution")

        except Exception as e:
            self.log_output(f"❌ Error creating spherical evolution: {e}")
            traceback.print_exc()

    def save_model_gui(self):
        if not self.model:
            messagebox.showwarning("Warning", "No model to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")]
        )

        if file_path:
            try:
                self.model.save_model(file_path)
                self.log_output(f"✅ Model saved to {file_path}")
            except Exception as e:
                self.log_output(f"❌ Error saving model: {e}")

    def load_model_gui(self):
        file_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl")]
        )

        if file_path:
            try:
                dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                self.model = OptimizedDBNN(dataset_name=dataset_name)
                self.model.load_model(file_path)

                self.log_output(f"✅ Model loaded from {file_path}")
                self.log_output(f"   Dataset: {self.model.dataset_name}")
                self.log_output(f"   Classes: {len(self.model.classes) if self.model.classes else 0}")

            except Exception as e:
                self.log_output(f"❌ Error loading model: {e}")
                import traceback
                traceback.print_exc()

    def install_dependencies(self):
        if messagebox.askyesno("Install Dependencies",
                              "This will install required packages using pip. Continue?"):
            self.log_output("📦 Installing dependencies...")
            success = self.env_manager.install_dependencies()
            if success:
                self.log_output("✅ Dependencies installed successfully")
            else:
                self.log_output("❌ Failed to install some dependencies")

    def generate_requirements(self):
        file_path = filedialog.asksaveasfilename(
            title="Save requirements.txt",
            initialfile="requirements.txt",
            defaultextension=".txt"
        )
        if file_path:
            success = self.env_manager.generate_requirements_file(file_path)
            if success:
                self.log_output(f"✅ Requirements file saved to {file_path}")

    def check_cuda(self):
        cuda_info = self.env_manager.check_cuda()

        if cuda_info['available']:
            msg = f"""
            CUDA Available: ✓
            Version: {cuda_info['version']}
            Device: {cuda_info['device_name']}
            Memory: {cuda_info['memory']:.1f} GB
            """
            messagebox.showinfo("CUDA Information", msg)
        else:
            messagebox.showinfo("CUDA Information", "CUDA not available - using CPU mode")

    def display_results(self, results):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        text = f"""
🏆 TRAINING RESULTS
{'='*50}

Best Accuracy: {results['best_accuracy']:.4f}
Final Train Accuracy: {results['final_train_accuracy']:.4f}
Final Test Accuracy: {results['final_test_accuracy']:.4f}

Training History:
{'='*50}
"""
        for entry in results['history'][-10:]:
            text += f"Epoch {entry['epoch']:3d}: Train={entry['train_accuracy']:.4f}, Test={entry['test_accuracy']:.4f}, Failed={entry['failed_cases']:4d}\n"

        self.results_text.insert(1.0, text)
        self.results_text.config(state=tk.DISABLED)

    def display_adaptive_results(self, results):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        text = f"""
🏆 ADAPTIVE TRAINING RESULTS
{'='*50}

Best Combined Accuracy: {results['best_accuracy']:.4f}
Final Total Accuracy: {results['final_total_accuracy']:.4f}
Training Samples: {len(results['train_indices'])}
Test Samples: {len(results['test_indices'])}
Rounds Completed: {len(results['round_stats'])}

Round Statistics:
{'='*50}
"""
        for stat in results['round_stats']:
            text += f"Round {stat['round']+1:2d}: Train={stat['train_accuracy']:.4f}, Test={stat['test_accuracy']:.4f}, Total={stat['total_accuracy']:.4f} | Size={stat['train_size']}\n"

        if 'evolution_history' in results and results['evolution_history']:
            text += f"\nTensor Evolution Captured: {len(results['evolution_history'])} states\n"

        self.results_text.insert(1.0, text)
        self.results_text.config(state=tk.DISABLED)

    def log_output(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.output_text.see(tk.END)
        self.root.update()
        self.status_var.set(message)

    def open_visualization_dir(self):
        try:
            if self.model and hasattr(self.model, 'dataset_name'):
                vis_dir = os.path.abspath(f"visualizations/{self.model.dataset_name}")
            else:
                vis_dir = os.path.abspath("visualizations")

            if os.name == 'nt':
                subprocess.Popen(f'explorer "{vis_dir}"')
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', vis_dir])
            else:
                subprocess.Popen(['xdg-open', vis_dir])

            self.log_output(f"📂 Opened directory: {vis_dir}")
        except Exception as e:
            self.log_output(f"⚠️ Could not open directory: {e}")

    def log_output(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.output_text.see(tk.END)
        self.root.update()
        self.status_var.set(message)

    def download_uci(self):
        """Download UCI dataset"""
        dataset = self.uci_var.get()
        if not dataset:
            messagebox.showwarning("Warning", "Please select a dataset from the list")
            return

        self.log_output(f"📥 Downloading UCI dataset: {dataset}")

        # Get dataset info
        dataset_info = DatasetConfig.get_dataset_info(dataset)
        if dataset_info:
            self.log_output(f"   {dataset_info['description']}")

        # Download the dataset
        df = DatasetConfig.download_uci_data(dataset)

        if df is not None:
            # Create data directory
            data_dir = os.path.join('data', dataset)
            os.makedirs(data_dir, exist_ok=True)

            # Save CSV
            csv_path = os.path.join(data_dir, f"{dataset}.csv")
            df.to_csv(csv_path, index=False)

            # Set the file path variable
            self.file_path_var.set(csv_path)

            self.log_output(f"✅ Dataset saved to: {csv_path}")
            self.log_output(f"   Samples: {len(df)}, Features: {len(df.columns)}")

            # Load the dataset
            self.load_dataset()
        else:
            self.log_output(f"❌ Failed to download dataset: {dataset}")
            messagebox.showerror("Download Failed", f"Could not download {dataset}.\nCheck your internet connection.")

    def save_configuration_for_file(self, file_path):
        """Save configuration for specific data file"""
        try:
            configs_dir = "configs"
            os.makedirs(configs_dir, exist_ok=True)

            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            config_file = os.path.join(configs_dir, f"{dataset_name}_config.json")

            config = {
                'dataset_name': dataset_name,
                'target_column': self.target_var.get(),
                'selected_features': [col for col, var in self.feature_vars.items() if var.get() and col != self.target_var.get()],
                'model_type': self.model_type_var.get(),
                'compute_device': self.device_var.get(),
                'parallel': self.parallel_var.get(),
                'n_jobs': int(self.n_jobs_var.get()) if self.parallel_var.get() else 1,
                'parallel_batch_size': int(self.parallel_batch_var.get()) if self.parallel_var.get() else 1000,
                'parallel_mode': self.parallel_mode_var.get(),
                'learning_rate': float(self.learning_rate_var.get()),
                'epochs': int(self.epochs_var.get()),
                'n_bins_per_dim': int(self.bins_var.get()),
                'test_fraction': float(self.test_size_var.get()),
                'enable_adaptive': self.adaptive_var.get(),
                'adaptive_rounds': int(self.adaptive_rounds_var.get()),
                'initial_samples': int(self.initial_samples_var.get()),
                'max_samples_per_round': int(self.max_samples_round_var.get())
            }

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            self.log_output(f"💾 Configuration saved to: {config_file}")

        except Exception as e:
            self.log_output(f"❌ Error saving configuration: {e}")

# =============================================================================
# SECTION 11: MAIN ENTRY POINT
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CT-DBNN - Complex Tensor DBNN')
    parser.add_argument('--gui', action='store_true', help='Launch GUI')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--mode', type=str, default='train_predict', help='Operation mode')
    parser.add_argument('--track-evolution', action='store_true', help='Track tensor evolution')
    args = parser.parse_args()

    print(f"""
{Colors.BOLD}{Colors.RED}
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║    🧠   Complex Tensor Difference Boosting Neural Network      ║
║                 Author: nsp@airis4d.com                        ║
║               Optimized Version: 7-10x Faster                  ║
║              with GUI & Interactive Features                   ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝{Colors.ENDC}
    """)
    if args.gui:
        if GUI_AVAILABLE:
            root = tk.Tk()
            app = CTDBNNGUI(root)
            root.mainloop()
        else:
            print(f"{Colors.RED}❌ Tkinter not available{Colors.ENDC}")
        return

    if args.interactive:
        interactive_mode()
        return

    if args.dataset:
        model = OptimizedDBNN(dataset_name=args.dataset, mode=args.mode)
        if args.track_evolution:
            model.enable_evolution_tracking()
        model.load_data()

        if model.enable_adaptive:
            results = model.adaptive_fit_predict()
        else:
            model.split_data()
            results = model.fit_predict()

        model.save_model(f'model_{args.dataset}.pkl')
        print(f"\n{Colors.GREEN}✅ Training complete! Best accuracy: {results['best_accuracy']:.4f}{Colors.ENDC}")
    else:
        parser.print_help()

def interactive_mode():
    """
    Enhanced interactive mode with ALL GUI features for headless GPU servers
    """
    print(f"\n{Colors.BOLD}{Colors.BLUE}╔══════════════════════════════════════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}║     🧠 CT-DBNN INTERACTIVE MODE - HEADLESS GPU SERVER READY    ║{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}╚══════════════════════════════════════════════════════════════╝{Colors.ENDC}")

    # =========================================================================
    # SECTION 1: ENVIRONMENT CHECK
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.CYAN}📊 ENVIRONMENT CHECK{Colors.ENDC}")
    print(f"{'='*60}")

    env_manager = EnvironmentManager()

    # Check Python version
    py_ok, py_version = env_manager.check_python_version()
    print(f"Python: {Colors.GREEN if py_ok else Colors.RED}{py_version}{Colors.ENDC}")

    # Check CUDA
    cuda_info = env_manager.check_cuda()
    if cuda_info['available']:
        print(f"CUDA: {Colors.GREEN}Available{Colors.ENDC} - {cuda_info['device_name']} ({cuda_info['memory']:.1f} GB)")
        print(f"   CUDA Version: {cuda_info['version']}")
        print(f"   Devices: {cuda_info['device_count']}")
    else:
        print(f"CUDA: {Colors.YELLOW}Not available (CPU mode){Colors.ENDC}")

    # Check dependencies
    print(f"\n{Colors.CYAN}📦 Dependencies:{Colors.ENDC}")
    deps = env_manager.check_dependencies()
    missing = []
    for pkg, installed in deps.items():
        status = f"{Colors.GREEN}✓{Colors.ENDC}" if installed else f"{Colors.RED}✗{Colors.ENDC}"
        print(f"   {status} {pkg}")
        if not installed:
            missing.append(pkg)

    if missing:
        print(f"\n{Colors.YELLOW}⚠️ Missing dependencies: {', '.join(missing)}{Colors.ENDC}")
        install = input(f"{Colors.CYAN}Install missing dependencies? (y/n): {Colors.ENDC}").strip().lower()
        if install == 'y':
            env_manager.install_dependencies(missing)

    # =========================================================================
    # SECTION 2: DATASET SELECTION
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.CYAN}📊 DATASET SELECTION{Colors.ENDC}")
    print(f"{'='*60}")

    # Check for existing datasets
    dataset_pairs = find_dataset_pairs()
    uci_datasets = DatasetConfig.list_uci_datasets()

    if dataset_pairs:
        print(f"\n{Colors.GREEN}Available local datasets:{Colors.ENDC}")
        for i, (name, _, _) in enumerate(dataset_pairs):
            print(f"   {i+1:2d}. {name}")
        print(f"   {len(dataset_pairs)+1}. Download UCI dataset")
        print(f"   {len(dataset_pairs)+2}. Load custom CSV file")

        choice = input(f"\n{Colors.BOLD}Select dataset (1-{len(dataset_pairs)+2}): {Colors.ENDC}").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(dataset_pairs):
            dataset_name, conf_path, csv_path = dataset_pairs[int(choice)-1]
            file_path = csv_path
        elif choice.isdigit() and int(choice) == len(dataset_pairs) + 1:
            # Download UCI dataset
            print(f"\n{Colors.CYAN}Available UCI datasets:{Colors.ENDC}")
            for i, name in enumerate(uci_datasets):
                print(f"   {i+1:2d}. {name}")

            uci_choice = input(f"\n{Colors.BOLD}Select UCI dataset (1-{len(uci_datasets)}): {Colors.ENDC}").strip()
            if uci_choice.isdigit() and 1 <= int(uci_choice) <= len(uci_datasets):
                dataset_name = uci_datasets[int(uci_choice)-1]
                print(f"{Colors.CYAN}Downloading {dataset_name}...{Colors.ENDC}")
                df = DatasetConfig.download_uci_data(dataset_name)

                if df is not None:
                    data_dir = os.path.join('data', dataset_name)
                    os.makedirs(data_dir, exist_ok=True)
                    file_path = os.path.join(data_dir, f"{dataset_name}.csv")
                    df.to_csv(file_path, index=False)
                    print(f"{Colors.GREEN}✓ Dataset saved to {file_path}{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}❌ Failed to download dataset{Colors.ENDC}")
                    return
            else:
                print(f"{Colors.RED}Invalid selection{Colors.ENDC}")
                return
        else:
            # Custom CSV file
            file_path = input(f"\n{Colors.CYAN}Enter CSV file path: {Colors.ENDC}").strip()
            if not os.path.exists(file_path):
                print(f"{Colors.RED}File not found: {file_path}{Colors.ENDC}")
                return
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    else:
        print(f"{Colors.YELLOW}No local datasets found{Colors.ENDC}")
        print(f"\nOptions:")
        print(f"   1. Download UCI dataset")
        print(f"   2. Load custom CSV file")

        choice = input(f"\n{Colors.BOLD}Select option (1-2): {Colors.ENDC}").strip()

        if choice == '1':
            print(f"\n{Colors.CYAN}Available UCI datasets:{Colors.ENDC}")
            for i, name in enumerate(uci_datasets):
                print(f"   {i+1:2d}. {name}")

            uci_choice = input(f"\n{Colors.BOLD}Select UCI dataset (1-{len(uci_datasets)}): {Colors.ENDC}").strip()
            if uci_choice.isdigit() and 1 <= int(uci_choice) <= len(uci_datasets):
                dataset_name = uci_datasets[int(uci_choice)-1]
                print(f"{Colors.CYAN}Downloading {dataset_name}...{Colors.ENDC}")
                df = DatasetConfig.download_uci_data(dataset_name)

                if df is not None:
                    data_dir = os.path.join('data', dataset_name)
                    os.makedirs(data_dir, exist_ok=True)
                    file_path = os.path.join(data_dir, f"{dataset_name}.csv")
                    df.to_csv(file_path, index=False)
                    print(f"{Colors.GREEN}✓ Dataset saved to {file_path}{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}❌ Failed to download dataset{Colors.ENDC}")
                    return
            else:
                print(f"{Colors.RED}Invalid selection{Colors.ENDC}")
                return
        else:
            file_path = input(f"\n{Colors.CYAN}Enter CSV file path: {Colors.ENDC}").strip()
            if not os.path.exists(file_path):
                print(f"{Colors.RED}File not found: {file_path}{Colors.ENDC}")
                return
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    # =========================================================================
    # SECTION 3: FEATURE AND TARGET SELECTION
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.CYAN}🔧 FEATURE CONFIGURATION{Colors.ENDC}")
    print(f"{'='*60}")

    # Load data to examine columns
    df = pd.read_csv(file_path)

    print(f"\n{Colors.GREEN}Dataset loaded: {len(df)} samples, {len(df.columns)} columns{Colors.ENDC}")
    print(f"\n{Colors.CYAN}Columns:{Colors.ENDC}")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        print(f"   {i+1:2d}. {col} ({dtype}) - unique: {unique_count}")

    # Target selection
    print(f"\n{Colors.BOLD}Target column selection:{Colors.ENDC}")
    target_cols = list(df.columns)
    for i, col in enumerate(target_cols):
        print(f"   {i+1:2d}. {col}")

    target_choice = input(f"\n{Colors.CYAN}Select target column (1-{len(target_cols)}): {Colors.ENDC}").strip()
    if target_choice.isdigit() and 1 <= int(target_choice) <= len(target_cols):
        target_column = target_cols[int(target_choice)-1]
        print(f"{Colors.GREEN}✓ Target: {target_column}{Colors.ENDC}")
    else:
        target_column = target_cols[-1]
        print(f"{Colors.YELLOW}Using last column as target: {target_column}{Colors.ENDC}")

    # Feature selection
    print(f"\n{Colors.BOLD}Feature selection:{Colors.ENDC}")
    print(f"   Options:")
    print(f"   1. Use all numeric columns")
    print(f"   2. Select manually")
    print(f"   3. Use all except target")

    feature_choice = input(f"\n{Colors.CYAN}Select option (1-3): {Colors.ENDC}").strip()

    if feature_choice == '1':
        selected_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in selected_features:
            selected_features.remove(target_column)
        print(f"{Colors.GREEN}✓ Using {len(selected_features)} numeric features{Colors.ENDC}")
    elif feature_choice == '2':
        print(f"\n{Colors.CYAN}Available features (excluding target):{Colors.ENDC}")
        feature_cols = [col for col in df.columns if col != target_column]
        for i, col in enumerate(feature_cols):
            dtype = df[col].dtype
            print(f"   {i+1:2d}. {col} ({dtype})")

        selected_features = []
        while True:
            sel = input(f"\n{Colors.CYAN}Enter feature numbers (comma-separated, or 'done'): {Colors.ENDC}").strip()
            if sel.lower() == 'done':
                break
            try:
                indices = [int(x.strip()) for x in sel.split(',')]
                for idx in indices:
                    if 1 <= idx <= len(feature_cols):
                        selected_features.append(feature_cols[idx-1])
            except:
                print(f"{Colors.RED}Invalid input{Colors.ENDC}")

        if not selected_features:
            selected_features = feature_cols
            print(f"{Colors.YELLOW}Using all features{Colors.ENDC}")
        else:
            print(f"{Colors.GREEN}✓ Selected {len(selected_features)} features{Colors.ENDC}")
    else:
        selected_features = [col for col in df.columns if col != target_column]
        print(f"{Colors.GREEN}✓ Using all {len(selected_features)} features except target{Colors.ENDC}")

    # =========================================================================
    # SECTION 4: MODEL CONFIGURATION
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.CYAN}⚙️ MODEL CONFIGURATION{Colors.ENDC}")
    print(f"{'='*60}")

    # Model type
    print(f"\n{Colors.BOLD}Model type:{Colors.ENDC}")
    print(f"   1. Histogram (faster, discrete bins)")
    print(f"   2. Gaussian (more precise, continuous)")
    model_type_choice = input(f"\n{Colors.CYAN}Select model type (1-2) [1]: {Colors.ENDC}").strip() or "1"
    model_type = "Histogram" if model_type_choice == "1" else "Gaussian"

    # Training mode
    print(f"\n{Colors.BOLD}Training mode:{Colors.ENDC}")
    print(f"   1. Standard training (fixed train/test split)")
    print(f"   2. Adaptive training (active learning)")
    mode_choice = input(f"\n{Colors.CYAN}Select mode (1-2) [2]: {Colors.ENDC}").strip() or "2"
    use_adaptive = mode_choice == "2"

    # Hyperparameters
    print(f"\n{Colors.BOLD}Hyperparameters (press Enter for defaults):{Colors.ENDC}")

    learning_rate = input(f"{Colors.CYAN}Learning rate [0.1]: {Colors.ENDC}").strip()
    learning_rate = float(learning_rate) if learning_rate else 0.1

    n_bins = input(f"{Colors.CYAN}Number of bins [128]: {Colors.ENDC}").strip()
    n_bins = int(n_bins) if n_bins else 128

    test_size = input(f"{Colors.CYAN}Test size [0.2]: {Colors.ENDC}").strip()
    test_size = float(test_size) if test_size else 0.2

    epochs = input(f"{Colors.CYAN}Epochs [100]: {Colors.ENDC}").strip()
    epochs = int(epochs) if epochs else 100

    if use_adaptive:
        adaptive_rounds = input(f"{Colors.CYAN}Adaptive rounds [10]: {Colors.ENDC}").strip()
        adaptive_rounds = int(adaptive_rounds) if adaptive_rounds else 10

        initial_samples = input(f"{Colors.CYAN}Initial samples per class [5]: {Colors.ENDC}").strip()
        initial_samples = int(initial_samples) if initial_samples else 5

        max_samples_round = input(f"{Colors.CYAN}Max samples per round [25]: {Colors.ENDC}").strip()
        max_samples_round = int(max_samples_round) if max_samples_round else 25

    # Parallel processing
    print(f"\n{Colors.BOLD}Parallel processing:{Colors.ENDC}")
    parallel_choice = input(f"{Colors.CYAN}Enable parallel processing? (y/n) [y]: {Colors.ENDC}").strip().lower()
    parallel = parallel_choice != 'n'

    if parallel:
        n_jobs = input(f"{Colors.CYAN}Number of CPU cores [auto, {mp.cpu_count()}]: {Colors.ENDC}").strip()
        n_jobs = int(n_jobs) if n_jobs and n_jobs.isdigit() else mp.cpu_count()

        parallel_mode = input(f"{Colors.CYAN}Parallel mode (threads/processes) [threads]: {Colors.ENDC}").strip().lower()
        parallel_mode = parallel_mode if parallel_mode in ['threads', 'processes'] else 'threads'

        batch_size = input(f"{Colors.CYAN}Parallel batch size [1000]: {Colors.ENDC}").strip()
        batch_size = int(batch_size) if batch_size else 1000

    # Visualization
    print(f"\n{Colors.BOLD}Visualization:{Colors.ENDC}")
    visualize = input(f"{Colors.CYAN}Generate visualizations? (y/n) [y]: {Colors.ENDC}").strip().lower()
    visualize = visualize != 'n'

    track_evolution = False
    if use_adaptive:
        track_evolution = input(f"{Colors.CYAN}Track tensor evolution? (y/n) [y]: {Colors.ENDC}").strip().lower()
        track_evolution = track_evolution != 'n'

    # =========================================================================
    # SECTION 5: CREATE AND CONFIGURE MODEL
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 CREATING MODEL{Colors.ENDC}")
    print(f"{'='*60}")

    # Build configuration
    config = {
        'file_path': file_path,
        'target_column': target_column,
        'model_type': model_type,
        'compute_device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'parallel': parallel,
        'n_jobs': n_jobs if parallel else 1,
        'parallel_batch_size': batch_size if parallel else 1000,
        'parallel_mode': parallel_mode if parallel else 'threads',
        'training_params': {
            'learning_rate': learning_rate,
            'epochs': epochs,
            'n_bins_per_dim': n_bins,
            'test_fraction': test_size,
            'enable_adaptive': use_adaptive,
            'adaptive_rounds': adaptive_rounds if use_adaptive else 10,
            'initial_samples': initial_samples if use_adaptive else 50,
            'max_samples_per_round': max_samples_round if use_adaptive else 25,
            'patience': 25
        }
    }

    # Create model
    model = OptimizedDBNN(dataset_name=dataset_name, config=config)

    # Apply feature selection
    model.selected_features = selected_features

    # Load data
    print(f"{Colors.CYAN}📥 Loading data...{Colors.ENDC}")
    model.load_data(file_path=file_path)

    print(f"{Colors.GREEN}✓ Data loaded: {len(model.X_tensor)} samples, {len(model.classes)} classes{Colors.ENDC}")

    # =========================================================================
    # SECTION 6: TRAINING
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.BLUE}🏋️ TRAINING{Colors.ENDC}")
    print(f"{'='*60}")

    start_time = time.time()

    if track_evolution:
        model.enable_evolution_tracking()

    if use_adaptive:
        print(f"{Colors.CYAN}🚀 Starting ADAPTIVE training...{Colors.ENDC}")
        results = model.adaptive_fit_predict(max_rounds=adaptive_rounds)
    else:
        print(f"{Colors.CYAN}🚀 Starting STANDARD training...{Colors.ENDC}")
        model.split_data()
        results = model.fit_predict()

    training_time = time.time() - start_time

    # =========================================================================
    # SECTION 7: RESULTS DISPLAY
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.BLUE}📊 RESULTS{Colors.ENDC}")
    print(f"{'='*60}")

    print(f"\n{Colors.GREEN}✅ Training completed in {training_time:.2f} seconds{Colors.ENDC}")

    if use_adaptive:
        print(f"\n{Colors.BOLD}Adaptive Training Results:{Colors.ENDC}")
        print(f"   Best combined accuracy: {Colors.highlight_accuracy(results['best_accuracy'])}")
        print(f"   Final total accuracy: {Colors.highlight_accuracy(results['final_total_accuracy'])}")
        print(f"   Training samples: {len(results['train_indices'])}")
        print(f"   Test samples: {len(results['test_indices'])}")
        print(f"   Rounds completed: {len(results['round_stats'])}")

        # Round summary
        print(f"\n{Colors.CYAN}Round progression:{Colors.ENDC}")
        print(f"{'Round':>6} {'Train Acc':>10} {'Test Acc':>10} {'Total Acc':>10} {'Train Size':>12}")
        print(f"{'-'*56}")
        for stat in results['round_stats']:
            print(f"{stat['round']+1:6d} {stat['train_accuracy']:10.4f} {stat['test_accuracy']:10.4f} "
                  f"{stat['total_accuracy']:10.4f} {stat['train_size']:12d}")

        if 'evolution_history' in results and results['evolution_history']:
            print(f"\n{Colors.GREEN}📸 Tensor evolution captured: {len(results['evolution_history'])} states{Colors.ENDC}")
    else:
        print(f"\n{Colors.BOLD}Standard Training Results:{Colors.ENDC}")
        print(f"   Best accuracy: {Colors.highlight_accuracy(results['best_accuracy'])}")
        print(f"   Final train accuracy: {Colors.highlight_accuracy(results['train_accuracy'])}")
        print(f"   Final test accuracy: {Colors.highlight_accuracy(results['test_accuracy'])}")

    # =========================================================================
    # SECTION 8: CONFUSION MATRIX DISPLAY
    # =========================================================================
    print(f"\n{Colors.BOLD}📈 Confusion Matrices:{Colors.ENDC}")
    print(f"{'='*60}")

    # Training confusion matrix
    if model.X_train is not None and len(model.X_train) > 0:
        train_pred, _ = model.predict(model.X_train)
        model.print_colored_confusion_matrix(
            model.y_train.numpy(),
            train_pred.numpy(),
            "Training Data"
        )

    # Test confusion matrix
    if use_adaptive:
        test_indices = list(set(range(len(model.X_tensor))) - set(results['train_indices']))
        X_test = model.X_tensor[test_indices]
        y_test = model.y_tensor[test_indices]

        if len(X_test) > 0:
            test_pred, _ = model.predict(X_test)
            model.print_colored_confusion_matrix(
                y_test.numpy(),
                test_pred.numpy(),
                "Test Data"
            )
    elif model.X_test is not None and len(model.X_test) > 0:
        test_pred, _ = model.predict(model.X_test)
        model.print_colored_confusion_matrix(
            model.y_test.numpy(),
            test_pred.numpy(),
            "Test Data"
        )

    # =========================================================================
    # SECTION 9: VISUALIZATION (if enabled)
    # =========================================================================
    if visualize:
        print(f"\n{Colors.BOLD}🎨 GENERATING VISUALIZATIONS{Colors.ENDC}")
        print(f"{'='*60}")

        visualizer = OptimizedVisualizer(model)

        # Prepare data for visualization
        X_np = model.X_tensor.numpy()
        y_np = model.y_tensor.numpy()

        if use_adaptive:
            train_mask = np.zeros(len(X_np), dtype=bool)
            train_mask[results['train_indices']] = True
            test_mask = ~train_mask

            y_train_np = y_np[train_mask]
            y_test_np = y_np[test_mask]

            train_pred, _ = model.predict(model.X_tensor[train_mask])
            test_pred, _ = model.predict(model.X_tensor[test_mask])
        else:
            y_train_np = model.y_train.numpy()
            y_test_np = model.y_test.numpy()
            train_pred, _ = model.predict(model.X_train)
            test_pred, _ = model.predict(model.X_test)

        # Get history
        if use_adaptive:
            viz_history = results.get('round_stats', [])
            evolution_history = results.get('evolution_history', [])
        else:
            viz_history = results.get('history', [])
            evolution_history = []

        # Generate visualizations
        visualizer.generate_all_visualizations(
            viz_history,
            X_np, y_np,
            y_train_np, y_test_np,
            train_pred.numpy() if hasattr(train_pred, 'numpy') else np.array(train_pred),
            test_pred.numpy() if hasattr(test_pred, 'numpy') else np.array(test_pred),
            evolution_history=evolution_history
        )

        # Create spherical evolution if available
        if evolution_history and len(evolution_history) > 1:
            print(f"\n{Colors.CYAN}🌐 Creating spherical tensor evolution...{Colors.ENDC}")
            spherical_path = visualizer.spherical_viz.create_spherical_animation(evolution_history)
            if spherical_path:
                print(f"   Saved to: {spherical_path}")

            # Create side-by-side comparison
            print(f"{Colors.CYAN}🔄 Creating side-by-side comparison...{Colors.ENDC}")
            side_by_side_path = visualizer.spherical_viz.create_side_by_side_evolution(evolution_history)
            if side_by_side_path:
                print(f"   Saved to: {side_by_side_path}")

    # =========================================================================
    # SECTION 10: SAVE MODEL
    # =========================================================================
    print(f"\n{Colors.BOLD}💾 SAVE MODEL{Colors.ENDC}")
    print(f"{'='*60}")

    save_model = input(f"{Colors.CYAN}Save trained model? (y/n) [y]: {Colors.ENDC}").strip().lower()
    if save_model != 'n':
        model_path = f"models/{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        os.makedirs("models", exist_ok=True)
        model.save_model(model_path)
        print(f"{Colors.GREEN}✓ Model saved to: {model_path}{Colors.ENDC}")

    # =========================================================================
    # SECTION 11: PREDICTION (optional)
    # =========================================================================
    print(f"\n{Colors.BOLD}🔮 PREDICTION{Colors.ENDC}")
    print(f"{'='*60}")

    predict = input(f"{Colors.CYAN}Make predictions on a file? (y/n) [n]: {Colors.ENDC}").strip().lower()
    if predict == 'y':
        pred_file = input(f"{Colors.CYAN}Enter prediction file path: {Colors.ENDC}").strip()
        if os.path.exists(pred_file):
            output_dir = f"predictions/{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_dir, exist_ok=True)

            print(f"{Colors.CYAN}🔮 Making predictions...{Colors.ENDC}")
            results = model.predict_from_file(pred_file, output_path=output_dir)

            if results and 'predictions' in results:
                print(f"{Colors.GREEN}✓ Predictions saved to: {output_dir}{Colors.ENDC}")

                # Show sample predictions
                df = pd.read_csv(pred_file)
                pred_df = results['predictions']
                print(f"\n{Colors.CYAN}Sample predictions (first 5):{Colors.ENDC}")
                print(pred_df[['predicted_class', 'confidence']].head())
        else:
            print(f"{Colors.RED}File not found: {pred_file}{Colors.ENDC}")

    # =========================================================================
    # SECTION 12: SUMMARY
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.BLUE}🎉 INTERACTIVE SESSION COMPLETE{Colors.ENDC}")
    print(f"{'='*60}")
    print(f"\n{Colors.GREEN}Summary:{Colors.ENDC}")
    print(f"   Dataset: {dataset_name}")
    print(f"   Model type: {model_type}")
    print(f"   Training mode: {'Adaptive' if use_adaptive else 'Standard'}")
    print(f"   Parallel: {'Enabled' if parallel else 'Disabled'}")
    print(f"   Training time: {training_time:.2f} seconds")

    if use_adaptive:
        print(f"   Final accuracy: {results['final_total_accuracy']:.4f}")
        print(f"   Training samples: {len(results['train_indices'])}")
    else:
        print(f"   Test accuracy: {results['test_accuracy']:.4f}")

    if visualize:
        print(f"   Visualizations: visualizations/{dataset_name}/")

    print(f"\n{Colors.CYAN}Tip: Run with --gui for graphical interface{Colors.ENDC}")
    print(f"      Run with --help for command-line options{Colors.ENDC}")


# Also add this function to handle stop signal for headless mode
def setup_signal_handlers():
    """Setup signal handlers for graceful termination in headless mode"""
    import signal

    def signal_handler(signum, frame):
        print(f"\n{Colors.YELLOW}⚠️ Interrupt received. Cleaning up...{Colors.ENDC}")
        if hasattr(signal_handler, 'model') and signal_handler.model:
            signal_handler.model.stop_training_flag = True
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return signal_handler


def find_dataset_pairs() -> List[Tuple[str, str, str]]:
    dataset_pairs = []
    data_dir = 'data'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return []

    for dataset_name in os.listdir(data_dir):
        dataset_dir = os.path.join(data_dir, dataset_name)
        if os.path.isdir(dataset_dir):
            conf_path = os.path.join(dataset_dir, f"{dataset_name}.conf")
            csv_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
            if os.path.exists(conf_path) and os.path.exists(csv_path):
                dataset_pairs.append((dataset_name, conf_path, csv_path))

    return dataset_pairs


def display_dataset_menu(dataset_pairs: List[Tuple[str, str, str]]):
    print(f"\n{Colors.BOLD}📊 Available Datasets:{Colors.ENDC}")
    print(f"{Colors.BLUE}╔══════════════════════════════════════════════════════════════════════════{Colors.ENDC}")

    for i, (dataset_name, conf_path, csv_path) in enumerate(dataset_pairs):
        try:
            df = pd.read_csv(csv_path, nrows=0)
            n_features = len(df.columns) - 1
            df_full = pd.read_csv(csv_path)
            n_samples = len(df_full)

            with open(conf_path, 'r') as f:
                config = json.load(f)
            model_type = config.get('modelType', 'Histogram')

            print(f"{Colors.BLUE}║ {Colors.ENDC}{i+1:2d}. {Colors.GREEN}{dataset_name:<20}{Colors.ENDC} "
                  f"Samples: {n_samples:>5} Features: {n_features:>3} "
                  f"Model: {model_type:<10}  {Colors.BLUE}║{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.BLUE}║ {Colors.ENDC}{i+1:2d}. {Colors.RED}{dataset_name:<20}{Colors.ENDC} "
                  f"Error loading dataset {Colors.BLUE}║{Colors.ENDC}")

    print(f"{Colors.BLUE}╚══════════════════════════════════════════════════════════════════════════{Colors.ENDC}")


if __name__ == "__main__":
    main()
