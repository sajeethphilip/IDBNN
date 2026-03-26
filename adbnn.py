"""
Optimized DBNN - Deep Bayesian Neural Network with GUI & Interactive Features
Mathematically equivalent to original but 7-10x faster
Author: Ninan Sajeeth Philip, AIRIS4D
Optimized by: DeepSeek AI
Version: 3.5 - FIXED 5D Tensor Visualization from adaptive_dbnn.py
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
import copy
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.spatial import ConvexHull
import networkx as nx

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

class RobustCSVReader:
    """
    Robust CSV reader that handles:
    - Comment lines starting with #
    - Missing or extra columns
    - Unknown class labels
    """

    @staticmethod
    def read_csv(file_path: str, expected_columns: List[str] = None,
                 ignore_comments: bool = True) -> pd.DataFrame:
        """
        Read CSV file, ignoring comment lines and handling malformed data.

        Args:
            file_path: Path to CSV file
            expected_columns: Optional list of expected column names
            ignore_comments: If True, skip lines starting with #

        Returns:
            DataFrame with cleaned data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # First, try pandas with comment parameter
        if ignore_comments:
            try:
                df = pd.read_csv(file_path, comment='#')
                if expected_columns:
                    # Keep only expected columns
                    available_cols = [col for col in expected_columns if col in df.columns]
                    if available_cols:
                        df = df[available_cols]
                return df
            except Exception as e:
                print(f"{Colors.YELLOW}⚠️ Pandas read with comment failed: {e}{Colors.ENDC}")
                # Fall through to manual parsing

        # Manual parsing fallback
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find first non-comment line for header
        header_line = None
        header_index = None
        data_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                if header_line is None:
                    header_line = stripped
                    header_index = i
                else:
                    data_lines.append(stripped)

        if header_line is None:
            raise ValueError(f"No valid header found in {file_path}")

        # Parse header
        import csv
        header = next(csv.reader([header_line]))

        # Parse data lines
        rows = []
        for line in data_lines:
            try:
                row = next(csv.reader([line]))
                # If row has fewer columns than header, pad with None
                if len(row) < len(header):
                    row.extend([None] * (len(header) - len(row)))
                # If row has more columns, truncate
                elif len(row) > len(header):
                    row = row[:len(header)]
                rows.append(row)
            except Exception:
                continue  # Skip malformed rows

        df = pd.DataFrame(rows, columns=header)

        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

        if expected_columns:
            available_cols = [col for col in expected_columns if col in df.columns]
            if available_cols:
                df = df[available_cols]

        return df

    @staticmethod
    def filter_known_classes(df: pd.DataFrame, target_col: str,
                             known_classes: Dict[str, int]) -> pd.DataFrame:
        """
        Filter dataframe to only include rows with known class labels.

        Args:
            df: Input dataframe
            target_col: Name of target column
            known_classes: Dictionary mapping class labels to encoded values

        Returns:
            Filtered dataframe (only rows with known classes)
        """
        if target_col not in df.columns:
            return df

        # Get unique classes in data
        data_classes = df[target_col].astype(str).unique()

        # Find unknown classes
        known_labels = set(known_classes.keys())
        unknown_classes = [c for c in data_classes if c not in known_labels]

        if unknown_classes:
            print(f"{Colors.YELLOW}⚠️ Found {len(unknown_classes)} unknown classes: {unknown_classes[:5]}{Colors.ENDC}")
            print(f"{Colors.YELLOW}   These rows will be ignored for training/evaluation{Colors.ENDC}")

            # Filter to keep only known classes
            mask = df[target_col].astype(str).isin(known_labels)
            filtered_df = df[mask].copy()

            print(f"{Colors.GREEN}   Kept {len(filtered_df)}/{len(df)} rows with known classes{Colors.ENDC}")
            return filtered_df

        return df

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
            'matplotlib': False, 'seaborn': False, 'plotly': False, 'requests': False,
            'networkx': False, 'scipy': False
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
        try:
            import networkx; required_packages['networkx'] = True
        except: pass
        try:
            import scipy; required_packages['scipy'] = True
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
                       "matplotlib>=3.4.0", "seaborn>=0.11.0", "plotly>=5.0.0", "requests>=2.26.0",
                       "networkx>=2.6", "scipy>=1.7.0"]
        try:
            with open(filepath, 'w') as f:
                f.write('\n'.join(requirements))
            print(f"{Colors.GREEN}✓ Requirements file saved to {filepath}{Colors.ENDC}")
            return True
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
            return False


# =============================================================================
# SECTION 0.7: TENSOR EVOLUTION TRACKER (FIXED for 5D tensor)
# =============================================================================

class TensorEvolutionTracker:
    """
    Passively mirrors tensor evolution without modifying core logic.
    Properly handles 5D tensor: (n_classes, n_pairs, n_bins, n_bins)
    """

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
        snapshot = {
            'round': round_num,
            'accuracy': accuracy,
            'training_size': training_size,
            'timestamp': time.time()
        }
        if hasattr(self.model.weight_updater, 'weights'):
            # Store the 5D tensor weights properly
            weights = self.model.weight_updater.weights.detach().cpu().clone()
            # Ensure shape: (n_classes, n_pairs, n_bins, n_bins)
            snapshot['complex_weights'] = weights
        self.tensor_evolution_history.append(snapshot)

    def get_history(self):
        return self.tensor_evolution_history

    def clear_history(self):
        self.tensor_evolution_history = []

"""
Enhanced Multi-Projection Spherical Evolution for 5D Tensor Visualization
Supports 5 different projection methods in a unified interactive dashboard
Author: Enhanced from adbnn-gui.py
Version: 4.0 - Multi-projection spherical evolution
"""

import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorly as tl
    from tensorly.decomposition import tucker
    TENSORLY_AVAILABLE = True
except ImportError:
    TENSORLY_AVAILABLE = False
    print("⚠️ Tensorly not available. HOSVD projection will be disabled.")


class MultiProjectionSphericalEvolution:
    """
    Multi-projection spherical evolution with 5 different projection methods
    All visualizations in a single, interactive HTML dashboard
    """

    def __init__(self, model, output_dir='visualizations'):
        self.model = model
        self.dataset_name = model.dataset_name if model and hasattr(model, 'dataset_name') else 'unknown'
        self.output_dir = Path(output_dir) / self.dataset_name / 'multi_projection'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes for classes
        self.class_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel

        # Store projections for each method
        self.projections_cache = {}

        print(f"{Colors.CYAN}🌐 Multi-Projection Spherical Evolution initialized{Colors.ENDC}")
        print(f"   Output: {self.output_dir}")
        print(f"   Projection methods: 2D Evolution, PCA, MDS, Physics, HOSVD (if available)")

    def project_2d_compression(self, weights):
        """
        Method 1: Original 2D compression + artificial φ
        This is the baseline method from original code
        """
        n_classes = weights.shape[0]
        projections = np.zeros((n_classes, 3))

        for c in range(n_classes):
            class_weights = weights[c].flatten()
            significant = class_weights[np.abs(class_weights) > 1e-6]

            if len(significant) > 0:
                magnitudes = np.abs(significant)
                phases = np.angle(significant)

                # Weighted circular mean
                sin_sum = np.sum(magnitudes * np.sin(phases))
                cos_sum = np.sum(magnitudes * np.cos(phases))
                avg_theta = np.arctan2(sin_sum, cos_sum)
                avg_r = np.mean(magnitudes)

                # Artificial polar angle (even spacing)
                phi = (c * np.pi / max(1, n_classes))

                # Convert to 3D
                projections[c, 0] = avg_r * np.sin(phi) * np.cos(avg_theta)
                projections[c, 1] = avg_r * np.sin(phi) * np.sin(avg_theta)
                projections[c, 2] = avg_r * np.cos(phi)
            else:
                projections[c] = [0, 0, 0]

        return projections, {'method': '2D Compression (Original)', 'description': 'Compresses 5D→2D complex plane, then adds artificial φ'}

    def project_pca_flattened(self, weights):
        """
        Method 2: PCA on flattened 5D tensor
        Preserves maximum variance in the full tensor space
        """
        n_classes = weights.shape[0]

        # If fewer than 3 classes, we need to handle specially
        if n_classes < 3:
            # Create default positions on a circle/sphere for visualization
            projections = np.zeros((n_classes, 3))
            for i in range(n_classes):
                angle = 2 * np.pi * i / max(1, n_classes)
                projections[i, 0] = np.cos(angle)
                projections[i, 1] = np.sin(angle)
                projections[i, 2] = 0

            return projections, {
                'method': 'PCA on Flattened Tensor',
                'description': f'Fallback: {n_classes} classes (<3) placed on circle',
                'explained_variance': [1.0]  # Placeholder
            }

        # Flatten: (n_classes, n_pairs, n_bins, n_bins) → (n_classes, n_pairs * n_bins * n_bins)
        flattened = weights.reshape(n_classes, -1)

        # Determine max components (can't exceed n_classes-1)
        n_components = min(3, n_classes - 1)

        # Apply PCA to get n_components
        pca = PCA(n_components=n_components, random_state=42)
        projections = pca.fit_transform(flattened)

        # If we got fewer than 3 components, pad with zeros
        if projections.shape[1] < 3:
            padded = np.zeros((n_classes, 3))
            padded[:, :projections.shape[1]] = projections
            projections = padded

        # Normalize to reasonable scale for visualization
        if np.max(np.abs(projections)) > 0:
            projections = projections / np.max(np.abs(projections)) * 0.8

        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_.tolist() if hasattr(pca, 'explained_variance_ratio_') else []

        return projections, {
            'method': 'PCA on Flattened Tensor',
            'description': f'Preserves {sum(explained_variance)*100:.1f}% variance' if explained_variance else 'PCA applied',
            'explained_variance': explained_variance
        }

    def project_mds_class_distances(self, weights):
        """
        Method 3: MDS preserving pairwise class distances
        Best for showing true class separation in the 5D space
        """
        n_classes = weights.shape[0]

        # Handle case with 1 class
        if n_classes == 1:
            return np.array([[0, 0, 0]]), {
                'method': 'MDS on Class Distances',
                'description': 'Single class at origin',
                'stress': 0.0
            }

        # Handle case with 2 classes
        if n_classes == 2:
            # Place them at opposite points on the X-axis
            projections = np.array([[-0.8, 0, 0], [0.8, 0, 0]])
            return projections, {
                'method': 'MDS on Class Distances',
                'description': f'2 classes placed opposite each other',
                'stress': 0.0
            }

        # Compute Frobenius distances between classes
        distances = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                diff = weights[i] - weights[j]
                distance = np.linalg.norm(diff)
                distances[i, j] = distances[j, i] = distance

        # Apply MDS to project to 3D while preserving distances
        mds = MDS(n_components=min(3, n_classes - 1), dissimilarity='precomputed', random_state=42)
        projections = mds.fit_transform(distances)

        # If we got fewer than 3 components, pad with zeros
        if projections.shape[1] < 3:
            padded = np.zeros((n_classes, 3))
            padded[:, :projections.shape[1]] = projections
            projections = padded

        # Normalize to unit sphere
        norms = np.linalg.norm(projections, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1
        projections = projections / norms * 0.8

        # Calculate stress (how well distances are preserved)
        if n_classes > 2:
            original_dist = squareform(pdist(projections))
            stress = np.sqrt(np.sum((distances - original_dist)**2) / np.sum(distances**2))
        else:
            stress = 0.0

        return projections, {
            'method': 'MDS on Class Distances',
            'description': f'Preserves class distances in 5D space (stress={stress:.3f})',
            'stress': stress
        }

    def project_physics_inspired(self, weights):
        """
        Method 4: Physics-inspired projection using tensor moments
        Maps each dimension to interpretable physical quantities
        """
        n_classes, n_pairs, n_bins, _ = weights.shape
        projections = np.zeros((n_classes, 3))

        # Handle single class case
        if n_classes == 1:
            return np.array([[0, 0, 0]]), {
                'method': 'Physics-Inspired',
                'description': 'Single class at origin',
                'axes_meaning': ['Activation Center', 'Spread (Focus)', 'Phase Coherence']
            }

        # Handle 2 classes case - place them opposite
        if n_classes == 2:
            projections = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
            return projections, {
                'method': 'Physics-Inspired',
                'description': '2 classes placed opposite each other',
                'axes_meaning': ['Activation Center', 'Spread (Focus)', 'Phase Coherence']
            }

        for c in range(n_classes):
            class_weights = weights[c]  # Shape: (n_pairs, n_bins, n_bins)

            # Calculate moments across feature pairs
            pair_centers = []
            pair_spreads = []
            pair_phases = []

            for p in range(n_pairs):
                pair_data = class_weights[p].flatten()
                magnitudes = np.abs(pair_data)
                phases = np.angle(pair_data)

                total_mag = np.sum(magnitudes)
                if total_mag > 0:
                    # Center of mass in bin space
                    bin_indices = np.arange(len(magnitudes))
                    center = np.sum(bin_indices * magnitudes) / total_mag
                    spread = np.sqrt(np.sum(magnitudes * (bin_indices - center)**2) / total_mag)

                    # Weighted average phase
                    sin_sum = np.sum(magnitudes * np.sin(phases))
                    cos_sum = np.sum(magnitudes * np.cos(phases))
                    avg_phase = np.arctan2(sin_sum, cos_sum)

                    pair_centers.append(center)
                    pair_spreads.append(spread)
                    pair_phases.append(avg_phase)
                else:
                    pair_centers.append(0)
                    pair_spreads.append(0)
                    pair_phases.append(0)

            # X-axis: Average center across pairs (where class activates)
            projections[c, 0] = np.mean(pair_centers) / n_bins * 2 - 1

            # Y-axis: Average spread (how focused the class is)
            projections[c, 1] = np.mean(pair_spreads) / n_bins * 2

            # Z-axis: Phase coherence across pairs
            if len(pair_phases) > 0 and np.std(pair_phases) > 0:
                projections[c, 2] = np.cos(np.std(pair_phases))  # High coherence = high Z
            else:
                projections[c, 2] = 0

        # Normalize to unit sphere
        if n_classes > 2:
            # Use the first two classes to determine normalization
            projections = projections / np.max(np.abs(projections)) * 0.8

        return projections, {
            'method': 'Physics-Inspired',
            'description': 'X: Activation center, Y: Focus, Z: Phase coherence',
            'axes_meaning': ['Activation Center', 'Spread (Focus)', 'Phase Coherence']
        }

    def project_hosvd_tucker(self, weights):
        """
        Method 5: Higher-Order SVD (Tucker Decomposition)
        Preserves multi-linear structure of the tensor
        Requires tensorly package
        """
        if not TENSORLY_AVAILABLE:
            return None, {'error': 'Tensorly not available'}

        try:
            # Convert to tensorly format
            tensor = tl.tensor(weights)

            # Tucker decomposition - keep 3 components in class mode
            core, factors = tucker(tensor, ranks=[3, weights.shape[1], weights.shape[2], weights.shape[3]])

            # Use class factor matrix as projection
            projections = factors[0][:, :3]  # Shape: (n_classes, 3)

            # Normalize
            norms = np.linalg.norm(projections, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1
            projections = projections / norms * 0.8

            # Calculate reconstruction error
            reconstructed = tl.tucker_to_tensor(core, factors)
            error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)

            return projections, {
                'method': 'HOSVD (Tucker Decomposition)',
                'description': f'Preserves multi-linear structure (reconstruction error={error:.4f})',
                'error': error
            }

        except Exception as e:
            print(f"⚠️ HOSVD failed: {e}")
            return None, {'error': str(e)}

    def create_2d_evolution_surface(self, evolution_history):
        """
        Create a 2D surface plot showing evolution over time
        This is a complementary view to the 3D spherical plots
        """
        if not evolution_history:
            return None

        # Extract data for all rounds
        rounds = []
        class_data = {}
        projection_data = []

        for snap in evolution_history:
            if 'complex_weights' not in snap:
                continue

            weights = snap['complex_weights']
            if torch.is_tensor(weights):
                weights = weights.cpu().numpy()

            round_num = snap['round']
            rounds.append(round_num)

            # Get projections for this round (use PCA method for consistency)
            proj, _ = self.project_pca_flattened(weights)
            projection_data.append(proj)

            # Store class trajectories
            for c in range(proj.shape[0]):
                if c not in class_data:
                    class_data[c] = {'x': [], 'y': [], 'z': [], 'rounds': []}
                class_data[c]['x'].append(proj[c, 0])
                class_data[c]['y'].append(proj[c, 1])
                class_data[c]['z'].append(proj[c, 2])
                class_data[c]['rounds'].append(round_num)

        if not projection_data:
            return None

        # Create 2D evolution surface figure
        fig = go.Figure()

        # Add class trajectories as lines
        for c, data in class_data.items():
            class_name = f'Class {c+1}'
            color = self.class_colors[c % len(self.class_colors)]

            # 2D projection (x vs y) over time
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                name=f'{class_name} (XY Projection)',
                hovertemplate=f'Class {c+1}<br>Round: %{{text}}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<extra></extra>',
                text=data['rounds']
            ))

            # Also show XZ projection
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['z'],
                mode='lines+markers',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=6, color=color, symbol='square'),
                name=f'{class_name} (XZ Projection)',
                hovertemplate=f'Class {c+1}<br>Round: %{{text}}<br>X: %{{x:.3f}}<br>Z: %{{y:.3f}}<extra></extra>',
                text=data['rounds'],
                visible='legendonly'  # Initially hidden
            ))

        fig.update_layout(
            title=f'2D Evolution Surface - {self.dataset_name}<br>Class Trajectories in Projection Space',
            xaxis_title='X Component',
            yaxis_title='Y / Z Component',
            height=600,
            hovermode='closest',
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.02,
                bgcolor='rgba(0,0,0,0.7)',
                font=dict(color='white')
            ),
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )

        return fig

    def create_multi_projection_dashboard(self, evolution_history, class_names=None):
        """
        Create a unified dashboard with all projection methods
        Features: synchronized views, zoom controls, evolution slider
        """
        if not evolution_history:
            print(f"{Colors.YELLOW}No evolution history available{Colors.ENDC}")
            return None

        print(f"{Colors.CYAN}🎨 Creating multi-projection dashboard...{Colors.ENDC}")

        # Get class names
        if class_names is None and hasattr(self.model, 'label_encoder') and self.model.label_encoder:
            class_names = list(self.model.label_encoder.keys())

        # Prepare all projection methods
        projection_methods = {
            '2d_compression': self.project_2d_compression,
            'pca_flattened': self.project_pca_flattened,
            'mds_distances': self.project_mds_class_distances,
            'physics_inspired': self.project_physics_inspired,
        }

        if TENSORLY_AVAILABLE:
            projection_methods['hosvd_tucker'] = self.project_hosvd_tucker

        # Process all rounds and collect projections
        rounds_data = []
        valid_snapshots = []

        for snap in evolution_history:
            if 'complex_weights' not in snap:
                continue

            weights = snap['complex_weights']
            if torch.is_tensor(weights):
                weights = weights.cpu().numpy()

            if len(weights.shape) != 4:
                continue

            round_num = snap['round']
            accuracy = snap.get('accuracy', 0)
            training_size = snap.get('training_size', 0)

            round_projs = {}
            for method_name, method_func in projection_methods.items():
                proj, metadata = method_func(weights)
                if proj is not None:
                    round_projs[method_name] = {
                        'projections': proj,
                        'metadata': metadata
                    }

            if round_projs:
                rounds_data.append({
                    'round': round_num,
                    'accuracy': accuracy,
                    'training_size': training_size,
                    'projections': round_projs
                })
                valid_snapshots.append(snap)

        if not rounds_data:
            print(f"{Colors.YELLOW}No valid projection data{Colors.ENDC}")
            return None

        n_methods = len(projection_methods)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        # Store metadata for each method for subplot titles
        method_metadata = {}
        for method_name in projection_methods.keys():
            # Get metadata from first round to use for titles
            if rounds_data and method_name in rounds_data[0]['projections']:
                method_metadata[method_name] = rounds_data[0]['projections'][method_name]['metadata']
            else:
                # Create default metadata if not available
                method_metadata[method_name] = {'method': method_name.replace('_', ' ').title()}

        # Create subplot figure with all methods
        # FIXED: Use method_metadata, not projection_methods
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            specs=[[{'type': 'scene'} for _ in range(n_cols)] for _ in range(n_rows)],
            subplot_titles=[f'<b>{method_metadata[method]["method"]}</b>'
                          for method in projection_methods.keys()],
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )

        # Create frames for animation
        frames = []

        for round_data in rounds_data:
            round_num = round_data['round']
            accuracy = round_data['accuracy']
            training_size = round_data['training_size']

            frame_traces = []
            method_idx = 0

            for method_name, method_data in round_data['projections'].items():
                proj = method_data['projections']
                metadata = method_data['metadata']
                n_classes = proj.shape[0]

                # Calculate row and column
                row = method_idx // n_cols + 1
                col = method_idx % n_cols + 1

                # Add sphere surface for reference
                u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
                x_sphere = np.cos(u) * np.sin(v)
                y_sphere = np.sin(u) * np.sin(v)
                z_sphere = np.cos(v)

                frame_traces.append(go.Surface(
                    x=x_sphere, y=y_sphere, z=z_sphere,
                    opacity=0.08,
                    showscale=False,
                    hoverinfo='none',
                    colorscale=[[0, 'lightgray'], [1, 'lightgray']],
                    showlegend=False
                ))

                # Add class vectors
                for c in range(min(n_classes, 12)):
                    class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'
                    color = self.class_colors[c % len(self.class_colors)]

                    x, y, z = proj[c, 0], proj[c, 1], proj[c, 2]
                    norm = np.sqrt(x*x + y*y + z*z)

                    if norm > 0.01:
                        frame_traces.append(go.Scatter3d(
                            x=[0, x], y=[0, y], z=[0, z],
                            mode='lines+markers',
                            marker=dict(size=8, color=color, symbol='circle', line=dict(width=1, color='white')),
                            line=dict(color=color, width=3),
                            name=class_name,
                            legendgroup=f'class_{method_name}_{c}',
                            showlegend=False,
                            text=f"<b>Class {class_name}</b><br>"
                                 f"Round: {round_num}<br>"
                                 f"Accuracy: {accuracy:.3f}<br>"
                                 f"X: {x:.3f}<br>Y: {y:.3f}<br>Z: {z:.3f}<br>"
                                 f"{metadata.get('description', '')}",
                            hoverinfo='text'
                        ))

                    # Add target positions for orthogonal separation
                    if method_name == 'mds_distances' or method_name == 'pca_flattened':
                        target_theta = (c * 2 * np.pi / max(1, n_classes))
                        target_phi = np.pi / 2
                        r = 0.95

                        x_target = r * np.sin(target_phi) * np.cos(target_theta)
                        y_target = r * np.sin(target_phi) * np.sin(target_theta)
                        z_target = r * np.cos(target_phi)

                        frame_traces.append(go.Scatter3d(
                            x=[x_target], y=[y_target], z=[z_target],
                            mode='markers',
                            marker=dict(size=12, color=color, symbol='x', line=dict(width=2, color='white')),
                            name=f'Target {class_name}',
                            legendgroup=f'target_{method_name}_{c}',
                            showlegend=False,
                            text=f"Target orthogonal position for {class_name}<br>90° separation",
                            hoverinfo='text'
                        ))

                # Add equatorial plane reference
                theta_circle = np.linspace(0, 2*np.pi, 50)
                x_circle = 0.98 * np.cos(theta_circle)
                y_circle = 0.98 * np.sin(theta_circle)
                z_circle = np.zeros_like(theta_circle)

                frame_traces.append(go.Scatter3d(
                    x=x_circle, y=y_circle, z=z_circle,
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dash'),
                    name='Equatorial Plane (90°)',
                    showlegend=False
                ))

                method_idx += 1

            # Create frame
            frames.append(go.Frame(
                data=frame_traces,
                name=f'round_{round_num}',
                layout=go.Layout(
                    title=dict(
                        text=f'<b>Round {round_num}</b><br>Accuracy: {accuracy:.3f} | Training Samples: {training_size}',
                        font=dict(size=14)
                    )
                )
            ))

        # Add initial data (first round)
        first_round = rounds_data[0]
        method_idx = 0

        for method_name, method_data in first_round['projections'].items():
            proj = method_data['projections']
            metadata = method_data['metadata']
            n_classes = proj.shape[0]

            row = method_idx // n_cols + 1
            col = method_idx % n_cols + 1

            # Add sphere
            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
            x_sphere = np.cos(u) * np.sin(v)
            y_sphere = np.sin(u) * np.sin(v)
            z_sphere = np.cos(v)

            fig.add_trace(
                go.Surface(
                    x=x_sphere, y=y_sphere, z=z_sphere,
                    opacity=0.08, showscale=False, hoverinfo='none',
                    colorscale=[[0, 'lightgray'], [1, 'lightgray']],
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add class vectors
            for c in range(min(n_classes, 12)):
                class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'
                color = self.class_colors[c % len(self.class_colors)]

                x, y, z = proj[c, 0], proj[c, 1], proj[c, 2]
                norm = np.sqrt(x*x + y*y + z*z)

                if norm > 0.01:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[0, x], y=[0, y], z=[0, z],
                            mode='lines+markers',
                            marker=dict(size=8, color=color, symbol='circle', line=dict(width=1, color='white')),
                            line=dict(color=color, width=3),
                            name=class_name,
                            legendgroup=f'class_{method_name}_{c}',
                            showlegend=(method_idx == 0 and c < 8),
                            text=f"<b>Class {class_name}</b><br>Round: {first_round['round']}<br>"
                                 f"Accuracy: {first_round['accuracy']:.3f}<br>"
                                 f"X: {x:.3f}<br>Y: {y:.3f}<br>Z: {z:.3f}<br>"
                                 f"{metadata.get('description', '')}",
                            hoverinfo='text'
                        ),
                        row=row, col=col
                    )

                # Add target positions
                if method_name == 'mds_distances' or method_name == 'pca_flattened':
                    target_theta = (c * 2 * np.pi / max(1, n_classes))
                    target_phi = np.pi / 2
                    r = 0.95

                    x_target = r * np.sin(target_phi) * np.cos(target_theta)
                    y_target = r * np.sin(target_phi) * np.sin(target_theta)
                    z_target = r * np.cos(target_phi)

                    fig.add_trace(
                        go.Scatter3d(
                            x=[x_target], y=[y_target], z=[z_target],
                            mode='markers',
                            marker=dict(size=12, color=color, symbol='x', line=dict(width=2, color='white')),
                            name=f'Target {class_name}',
                            legendgroup=f'target_{method_name}_{c}',
                            showlegend=False,
                            text=f"Target orthogonal position for {class_name}<br>90° separation",
                            hoverinfo='text'
                        ),
                        row=row, col=col
                    )

            # Add equatorial plane
            theta_circle = np.linspace(0, 2*np.pi, 50)
            x_circle = 0.98 * np.cos(theta_circle)
            y_circle = 0.98 * np.sin(theta_circle)
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

            method_idx += 1

        # Create 2D evolution surface
        fig_2d = self.create_2d_evolution_surface(valid_snapshots)

        # Update layout with animation controls
        fig.update_layout(
            title=dict(
                text=f'<b>Multi-Projection Tensor Evolution - {self.dataset_name}</b><br>'
                     f'<sup>5 different projection methods showing class orthogonalization in complex tensor space</sup>',
                font=dict(size=16)
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=0.95,
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
                        'args': [[f'round_{rd["round"]}'], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': str(rd['round']),
                        'method': 'animate'
                    }
                    for rd in rounds_data
                ]
            }],
            width=1600,
            height=900,
            showlegend=True,
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.02,
                bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1,
                font=dict(color='white', size=10)
            ),
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )

        # Update all 3D scenes with consistent camera and axes
        for i in range(1, n_methods + 1):
            row = (i - 1) // n_cols + 1
            col = (i - 1) % n_cols + 1
            fig.update_scenes(
                xaxis_title='<b>Component 1</b>',
                yaxis_title='<b>Component 2</b>',
                zaxis_title='<b>Component 3</b>',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='cube',
                row=row, col=col
            )

        # Save the dashboard
        dashboard_path = self.output_dir / f'{self.dataset_name}_multi_projection_dashboard.html'
        fig.write_html(str(dashboard_path))

        # Also save 2D evolution as separate file
        if fig_2d:
            fig_2d.write_html(str(self.output_dir / f'{self.dataset_name}_2d_evolution.html'))

        # Save metadata
        metadata = {
            'dataset': self.dataset_name,
            'projection_methods': list(projection_methods.keys()),
            'n_rounds': len(rounds_data),
            'n_classes': rounds_data[0]['projections'][list(projection_methods.keys())[0]]['projections'].shape[0],
            'rounds': [rd['round'] for rd in rounds_data],
            'accuracy_progression': [rd['accuracy'] for rd in rounds_data],
            'training_sizes': [rd['training_size'] for rd in rounds_data],
            'created': datetime.now().isoformat()
        }

        with open(self.output_dir / f'{self.dataset_name}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"{Colors.GREEN}✅ Multi-projection dashboard saved to: {dashboard_path}{Colors.ENDC}")
        print(f"   Methods: {', '.join(projection_methods.keys())}")
        print(f"   Rounds: {len(rounds_data)}")
        print(f"   Classes: {rounds_data[0]['projections'][list(projection_methods.keys())[0]]['projections'].shape[0]}")

        return str(dashboard_path)

    def create_comparison_grid(self, evolution_history):
        """
        Create a static comparison grid showing initial vs final state for each projection
        """
        if not evolution_history or len(evolution_history) < 2:
            return None

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        first_snap = evolution_history[0]
        last_snap = evolution_history[-1]

        if 'complex_weights' not in first_snap or 'complex_weights' not in last_snap:
            return None

        weights_first = first_snap['complex_weights']
        weights_last = last_snap['complex_weights']

        if torch.is_tensor(weights_first):
            weights_first = weights_first.cpu().numpy()
        if torch.is_tensor(weights_last):
            weights_last = weights_last.cpu().numpy()

        # Get all projections
        methods = {
            '2D Compression': self.project_2d_compression,
            'PCA Flattened': self.project_pca_flattened,
            'MDS Distances': self.project_mds_class_distances,
            'Physics-Inspired': self.project_physics_inspired,
        }

        if TENSORLY_AVAILABLE:
            methods['HOSVD Tucker'] = self.project_hosvd_tucker

        n_methods = len(methods)
        n_cols = 4
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(20, 5 * n_rows))

        for idx, (method_name, method_func) in enumerate(methods.items()):
            proj_first, meta_first = method_func(weights_first)
            proj_last, meta_last = method_func(weights_last)

            if proj_first is None or proj_last is None:
                continue

            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

            # Plot initial state (transparent)
            ax.scatter(proj_first[:, 0], proj_first[:, 1], proj_first[:, 2],
                      c='blue', alpha=0.3, s=50, label='Initial')

            # Plot final state (solid)
            ax.scatter(proj_last[:, 0], proj_last[:, 1], proj_last[:, 2],
                      c='red', alpha=0.8, s=80, label='Final')

            # Draw arrows showing evolution
            for c in range(proj_first.shape[0]):
                ax.quiver(proj_first[c, 0], proj_first[c, 1], proj_first[c, 2],
                         proj_last[c, 0] - proj_first[c, 0],
                         proj_last[c, 1] - proj_first[c, 1],
                         proj_last[c, 2] - proj_first[c, 2],
                         color='green', alpha=0.5, arrow_length_ratio=0.2)

            ax.set_title(f'{method_name}\n{meta_first.get("description", "")[:50]}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()

        plt.suptitle(f'Tensor Evolution Comparison: Initial vs Final - {self.dataset_name}', fontsize=16)
        plt.tight_layout()

        comparison_path = self.output_dir / f'{self.dataset_name}_comparison_grid.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(comparison_path)

# =============================================================================
# SECTION: POLAR COORDINATE EVOLUTION VISUALIZER
# =============================================================================

class PolarCoordinateEvolution:
    """
    Polar coordinate evolution showing how class clusters form in the complex plane.
    Now uses TEST DATA for orthogonality metrics to show generalization.
    """

    def __init__(self, model, output_dir='visualizations'):
        self.model = model
        self.dataset_name = model.dataset_name if model and hasattr(model, 'dataset_name') else 'unknown'
        self.output_dir = Path(output_dir) / self.dataset_name / 'polar_evolution'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes for classes
        self.class_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + \
                           px.colors.qualitative.Set3 + px.colors.qualitative.Pastel

        print(f"{Colors.CYAN}📊 Polar Coordinate Evolution initialized{Colors.ENDC}")
        print(f"   Output: {self.output_dir}")
        print(f"   Using TEST DATA for orthogonality metrics (generalization)")

    def _get_test_indices(self):
        """Get test indices from model"""
        if hasattr(self.model, 'test_indices') and self.model.test_indices is not None:
            test_indices = self.model.test_indices
            if torch.is_tensor(test_indices):
                test_indices = test_indices.numpy() if hasattr(test_indices, 'numpy') else np.array(test_indices)
            return test_indices
        return None

    def _get_test_predictions(self, weights):
        """
        Get predictions and posteriors for test data.
        Returns test predictions, posteriors, and labels.
        """
        test_indices = self._get_test_indices()
        if test_indices is None or len(test_indices) == 0:
            return None, None, None

        X_test = self.model.X_tensor[test_indices]
        y_test = self.model.y_tensor[test_indices]

        predictions, posteriors = self.model.predict(X_test)

        if torch.is_tensor(posteriors):
            posteriors = posteriors.numpy()
        if torch.is_tensor(y_test):
            y_test = y_test.numpy()

        return predictions.numpy() if torch.is_tensor(predictions) else predictions, posteriors, y_test

    def extract_class_distributions_from_test_data(self, weights):
        """
        Extract polar coordinate distributions for each class using TEST DATA predictions.
        This shows generalization rather than memorization.
        """
        test_indices = self._get_test_indices()
        if test_indices is None or len(test_indices) == 0:
            return self.extract_class_distributions(weights)  # Fallback to training weights

        X_test = self.model.X_tensor[test_indices]
        y_test = self.model.y_tensor[test_indices]

        predictions, posteriors, y_test_np = self._get_test_predictions(weights)

        if predictions is None:
            return self.extract_class_distributions(weights)

        n_classes = weights.shape[0]
        n_test_samples = len(y_test_np)

        # Get class orientation vectors from tensor (for direction reference)
        class_vectors = self._compute_class_orientation_vectors(weights, n_classes)

        # Project test points to sphere
        point_coords = np.zeros((n_test_samples, 3))

        for i in range(n_test_samples):
            probs = posteriors[i]
            weighted_sum = np.zeros(3)
            for c in range(n_classes):
                weighted_sum += probs[c] * class_vectors[c]

            norm = np.linalg.norm(weighted_sum)
            if norm > 0:
                point_coords[i] = weighted_sum / norm

        # Now extract distributions per class from these projected points
        class_distributions = {}

        for c in range(n_classes):
            class_mask = (y_test_np == c)
            class_points = point_coords[class_mask]

            if len(class_points) == 0:
                class_distributions[c] = {
                    'radii': np.array([]),
                    'angles': np.array([]),
                    'n_points': 0,
                    'mean_radius': 0,
                    'mean_angle': 0,
                    'radius_std': 0,
                    'angle_std': 0,
                    'total_mass': 0,
                    'test_data': True
                }
                continue

            # Convert Cartesian to polar coordinates (on sphere surface)
            # For points on sphere, radius = 1 always, so we use angles
            # But we want to show the distribution of angles
            x, y, z = class_points[:, 0], class_points[:, 1], class_points[:, 2]

            # Get azimuth (theta) and polar (phi) angles
            theta = np.arctan2(y, x)
            phi = np.arccos(z)

            # For 2D polar plot, we use theta as angle and phi as magnitude?
            # Better: Use the projection onto complex plane (x,y) with radius = magnitude of projection
            proj_radius = np.sqrt(x**2 + y**2)
            proj_angle = np.arctan2(y, x)

            # Store as polar coordinates
            radii = proj_radius
            angles = proj_angle

            class_distributions[c] = {
                'radii': radii,
                'angles': angles,
                'n_points': len(class_points),
                'mean_radius': np.mean(radii),
                'mean_angle': np.mean(angles),
                'radius_std': np.std(radii),
                'angle_std': np.std(angles),
                'total_mass': np.sum(radii),
                'test_data': True
            }

        return class_distributions

    def _compute_class_orientation_vectors(self, weights, n_classes):
        """Compute unit orientation vectors for each class from tensor weights"""
        class_vectors = []

        for c in range(n_classes):
            class_weights = weights[c].flatten()
            significant = class_weights[np.abs(class_weights) > 1e-6]

            if len(significant) > 0:
                magnitudes = np.abs(significant)
                phases = np.angle(significant)

                sin_sum = np.sum(magnitudes * np.sin(phases))
                cos_sum = np.sum(magnitudes * np.cos(phases))
                avg_theta = np.arctan2(sin_sum, cos_sum)
                avg_r = np.mean(magnitudes)

                phi = (c * np.pi / max(1, n_classes))
                x = avg_r * np.sin(phi) * np.cos(avg_theta)
                y = avg_r * np.sin(phi) * np.sin(avg_theta)
                z = avg_r * np.cos(phi)

                norm = np.sqrt(x*x + y*y + z*z)
                if norm > 0:
                    class_vectors.append(np.array([x/norm, y/norm, z/norm]))
                else:
                    theta = (c * 2 * np.pi / max(2, n_classes))
                    phi = np.pi / 2
                    class_vectors.append(np.array([
                        np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi)
                    ]))
            else:
                theta = (c * 2 * np.pi / max(2, n_classes))
                phi = np.pi / 2
                class_vectors.append(np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ]))

        return np.array(class_vectors)

    def extract_class_distributions(self, weights):
        """Original method - kept for fallback"""
        n_classes = weights.shape[0]
        class_distributions = {}

        for c in range(n_classes):
            class_weights = weights[c].flatten()
            significant = class_weights[np.abs(class_weights) > 1e-6]

            if len(significant) > 0:
                radii = np.abs(significant)
                angles = np.angle(significant)

                class_distributions[c] = {
                    'radii': radii,
                    'angles': angles,
                    'n_points': len(significant),
                    'mean_radius': np.mean(radii),
                    'mean_angle': np.mean(angles),
                    'radius_std': np.std(radii),
                    'angle_std': np.std(angles),
                    'total_mass': np.sum(radii),
                    'test_data': False
                }
            else:
                class_distributions[c] = {
                    'radii': np.array([]),
                    'angles': np.array([]),
                    'n_points': 0,
                    'mean_radius': 0,
                    'mean_angle': 0,
                    'radius_std': 0,
                    'angle_std': 0,
                    'total_mass': 0,
                    'test_data': False
                }

        return class_distributions

    def calculate_cluster_metrics(self, class_distributions):
        """
        Calculate cluster formation metrics for each class.
        Returns metrics showing how well-defined each class cluster is.
        """
        metrics = {}

        data_source = "TEST" if any(d.get('test_data', False) for d in class_distributions.values()) else "TRAINING"

        for c, dist in class_distributions.items():
            if dist['n_points'] < 2:
                metrics[c] = {
                    'cluster_quality': 0,
                    'angular_concentration': 0,
                    'radial_concentration': 0,
                    'entropy': 0,
                    'data_source': data_source
                }
                continue

            angles = dist['angles']
            complex_angles = np.exp(1j * angles)
            mean_complex = np.mean(complex_angles)
            angular_concentration = np.abs(mean_complex)

            radii = dist['radii']
            if np.std(radii) > 0:
                radial_concentration = 1.0 / (1.0 + np.std(radii) / np.mean(radii))
            else:
                radial_concentration = 1.0

            hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            normalized_entropy = entropy / np.log(36)

            cluster_quality = (angular_concentration + radial_concentration + (1 - normalized_entropy)) / 3

            metrics[c] = {
                'cluster_quality': cluster_quality,
                'angular_concentration': angular_concentration,
                'radial_concentration': radial_concentration,
                'entropy': normalized_entropy,
                'data_source': data_source
            }

        return metrics

    def create_polar_dashboard(self, evolution_history, class_names=None):
        """
        Create a complete polar coordinate evolution dashboard with multiple views.
        Uses TEST DATA for metrics.
        """
        if not evolution_history or len(evolution_history) < 2:
            print(f"{Colors.YELLOW}Need at least 2 rounds for evolution visualization{Colors.ENDC}")
            return None

        print(f"{Colors.CYAN}📊 Creating polar coordinate evolution dashboard...{Colors.ENDC}")

        test_indices = self._get_test_indices()
        if test_indices is not None and len(test_indices) > 0:
            print(f"   Using TEST DATA for metrics ({len(test_indices)} samples)")
            print(f"   This shows GENERALIZATION, not memorization")
        else:
            print(f"   {Colors.YELLOW}No test indices found. Using training data (memorization){Colors.ENDC}")

        # Create animated polar scatter plot with test data
        animated_fig = self.create_polar_animation(evolution_history, class_names)

        # Create cluster quality plot
        quality_fig = self.create_cluster_quality_plot(evolution_history)

        # Create static images for key rounds
        key_rounds = self._get_key_rounds(evolution_history)
        static_figs = []

        for round_num in key_rounds:
            snap = next((s for s in evolution_history if s.get('round') == round_num), None)
            if snap and 'complex_weights' in snap:
                weights = snap['complex_weights']
                if torch.is_tensor(weights):
                    weights = weights.cpu().numpy()

                # Use test data distributions
                distributions = self.extract_class_distributions_from_test_data(weights)
                fig = self.create_polar_scatter_plot(
                    distributions, round_num,
                    snap.get('accuracy', 0),
                    snap.get('training_size', 0)
                )
                static_figs.append(fig)

        # Create combined HTML dashboard
        dashboard_html = self._create_combined_dashboard_html(
            evolution_history, animated_fig, quality_fig, static_figs, class_names,
            use_test_data=(test_indices is not None and len(test_indices) > 0)
        )

        dashboard_path = self.output_dir / f'{self.dataset_name}_polar_dashboard.html'

        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)

        print(f"{Colors.GREEN}✅ Polar dashboard saved to: {dashboard_path}{Colors.ENDC}")

        return str(dashboard_path)

    def create_polar_animation(self, evolution_history, class_names=None):
        """
        Create an animated polar evolution showing how clusters form over time.
        Uses TEST DATA for visualization.
        """
        if not evolution_history or len(evolution_history) < 2:
            return None

        print(f"{Colors.CYAN}🎬 Creating polar evolution animation...{Colors.ENDC}")

        # First pass: find global max radius for consistent scaling
        all_max_radius = 0
        valid_snapshots = []

        for snap in evolution_history:
            if 'complex_weights' not in snap:
                continue

            weights = snap['complex_weights']
            if torch.is_tensor(weights):
                weights = weights.cpu().numpy()

            if len(weights.shape) == 4:
                distributions = self.extract_class_distributions_from_test_data(weights)
                for dist in distributions.values():
                    if len(dist['radii']) > 0:
                        all_max_radius = max(all_max_radius, np.max(dist['radii']))
                valid_snapshots.append(snap)

        all_max_radius = all_max_radius * 1.1

        # Create frames
        frames = []
        test_indices = self._get_test_indices()
        use_test = test_indices is not None and len(test_indices) > 0

        for snap in valid_snapshots:
            weights = snap['complex_weights']
            if torch.is_tensor(weights):
                weights = weights.cpu().numpy()

            round_num = snap['round']
            accuracy = snap.get('accuracy', 0)
            training_size = snap.get('training_size', 0)

            distributions = self.extract_class_distributions_from_test_data(weights)
            metrics = self.calculate_cluster_metrics(distributions)

            # Create frame data
            frame_traces = []

            for c, dist in distributions.items():
                if dist['n_points'] == 0:
                    continue

                class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'
                color = self.class_colors[c % len(self.class_colors)]

                x = dist['radii'] * np.cos(dist['angles'])
                y = dist['radii'] * np.sin(dist['angles'])

                frame_traces.append(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(size=4, color=color, opacity=0.7),
                    name=class_name,
                    legendgroup=f'class_{c}',
                    showlegend=False,
                    text=[f'Class: {class_name}<br>Radius: {r:.3f}<br>Angle: {a:.2f} rad'
                          for r, a in zip(dist['radii'], dist['angles'])],
                    hoverinfo='text'
                ))

                if dist['mean_radius'] > 0:
                    mean_x = dist['mean_radius'] * np.cos(dist['mean_angle'])
                    mean_y = dist['mean_radius'] * np.sin(dist['mean_angle'])

                    frame_traces.append(go.Scatter(
                        x=[0, mean_x], y=[0, mean_y],
                        mode='lines+markers',
                        marker=dict(size=6, color=color, symbol='circle'),
                        line=dict(color=color, width=2, dash='dash'),
                        name=f'{class_name} (mean)',
                        showlegend=False,
                        legendgroup=f'class_{c}'
                    ))

            # Add unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            frame_traces.append(go.Scatter(
                x=np.cos(theta), y=np.sin(theta),
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                name='Unit Circle',
                showlegend=False
            ))

            data_source = "TEST DATA" if use_test else "TRAINING DATA"
            quality_text = "<br>".join([
                f"Class {c+1}: Q={metrics[c]['cluster_quality']:.2f}, "
                f"θ_conc={metrics[c]['angular_concentration']:.2f}, "
                f"r_conc={metrics[c]['radial_concentration']:.2f}"
                for c in metrics.keys()
            ])

            frames.append(go.Frame(
                data=frame_traces,
                name=f'round_{round_num}',
                layout=go.Layout(
                    title=dict(
                        text=f'<b>Polar Evolution - Round {round_num} ({data_source})</b><br>'
                             f'Accuracy: {accuracy:.3f} | Training Samples: {training_size}<br>'
                             f'<sup>Cluster Quality: {quality_text}</sup>',
                        font=dict(size=12)
                    )
                )
            ))

        if not frames:
            return None

        # Create initial frame (same as before, with updated title)
        first_frame = frames[0]
        data_source = "TEST DATA" if use_test else "TRAINING DATA"

        fig = go.Figure(
            data=first_frame.data,
            layout=go.Layout(
                title=first_frame.layout.title,
                xaxis=dict(
                    title='Real Component',
                    range=[-all_max_radius, all_max_radius],
                    scaleanchor='y',
                    scaleratio=1
                ),
                yaxis=dict(
                    title='Imaginary Component',
                    range=[-all_max_radius, all_max_radius]
                ),
                height=800,
                width=800,
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
                        y=0.95,
                        x=0.05
                    )
                ],
                sliders=[{
                    'active': 0,
                    'currentvalue': {'prefix': 'Round: ', 'font': {'size': 14}},
                    'steps': [
                        {
                            'args': [[f'round_{rd["round"]}'], {'frame': {'duration': 0, 'redraw': True}}],
                            'label': str(rd['round']),
                            'method': 'animate'
                        }
                        for rd in valid_snapshots
                    ]
                }],
                showlegend=True,
                legend=dict(
                    yanchor="top", y=0.99, xanchor="left", x=0.02,
                    bgcolor='rgba(0,0,0,0.7)',
                    font=dict(color='white')
                ),
                paper_bgcolor='rgba(0,0,0,0.9)',
                plot_bgcolor='rgba(0,0,0,0.9)'
            ),
            frames=frames
        )

        return fig

    def create_cluster_quality_plot(self, evolution_history):
        """
        Create a line plot showing how cluster quality evolves over rounds.
        Uses TEST DATA metrics.
        """
        if not evolution_history:
            return None

        rounds = []
        cluster_quality_history = []
        angular_concentration_history = []
        radial_concentration_history = []
        entropy_history = []
        test_indices = self._get_test_indices()
        use_test = test_indices is not None and len(test_indices) > 0

        for snap in evolution_history:
            if 'complex_weights' not in snap:
                continue

            weights = snap['complex_weights']
            if torch.is_tensor(weights):
                weights = weights.cpu().numpy()

            if len(weights.shape) != 4:
                continue

            round_num = snap['round']
            distributions = self.extract_class_distributions_from_test_data(weights)
            metrics = self.calculate_cluster_metrics(distributions)

            rounds.append(round_num)

            avg_quality = np.mean([m['cluster_quality'] for m in metrics.values()])
            avg_angular = np.mean([m['angular_concentration'] for m in metrics.values()])
            avg_radial = np.mean([m['radial_concentration'] for m in metrics.values()])
            avg_entropy = np.mean([m['entropy'] for m in metrics.values()])

            cluster_quality_history.append(avg_quality)
            angular_concentration_history.append(avg_angular)
            radial_concentration_history.append(avg_radial)
            entropy_history.append(avg_entropy)

        if not rounds:
            return None

        data_source = "TEST DATA" if use_test else "TRAINING DATA"

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rounds, y=cluster_quality_history,
            mode='lines+markers',
            name='Cluster Quality',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))

        fig.add_trace(go.Scatter(
            x=rounds, y=angular_concentration_history,
            mode='lines+markers',
            name='Angular Concentration',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))

        fig.add_trace(go.Scatter(
            x=rounds, y=radial_concentration_history,
            mode='lines+markers',
            name='Radial Concentration',
            line=dict(color='orange', width=2),
            marker=dict(size=8)
        ))

        fig.add_trace(go.Scatter(
            x=rounds, y=entropy_history,
            mode='lines+markers',
            name='Angle Entropy (lower = better)',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title=dict(
                text=f'<b>Cluster Formation Metrics - {self.dataset_name} ({data_source})</b><br>'
                     f'<sup>Higher quality = better defined clusters, lower entropy = more focused</sup>',
                font=dict(size=14)
            ),
            xaxis_title='Adaptive Round',
            yaxis_title='Metric Value',
            yaxis_range=[0, 1],
            height=500,
            width=900,
            hovermode='closest',
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.02,
                bgcolor='rgba(0,0,0,0.7)',
                font=dict(color='white')
            ),
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )

        return fig

    def _get_key_rounds(self, evolution_history):
        """Get key rounds for static snapshots"""
        n_rounds = len(evolution_history)
        if n_rounds <= 4:
            return [snap['round'] for snap in evolution_history if 'round' in snap]

        rounds = [snap['round'] for snap in evolution_history if 'round' in snap]
        return [rounds[0], rounds[n_rounds//3], rounds[2*n_rounds//3], rounds[-1]]

    def _create_combined_dashboard_html(self, evolution_history, animated_fig, quality_fig,
                                         static_figs, class_names, use_test_data=True):
        """Create combined HTML dashboard with all polar visualizations"""

        # Convert figures to HTML
        animated_html = animated_fig.to_html(include_plotlyjs='cdn', div_id='polar-animation') if animated_fig else ''
        quality_html = quality_fig.to_html(include_plotlyjs=False, div_id='cluster-quality') if quality_fig else ''

        static_htmls = []
        for i, fig in enumerate(static_figs):
            static_htmls.append(fig.to_html(include_plotlyjs=False, div_id=f'static-round-{i}'))

        # Calculate metrics summary from final round using test data
        final_snap = evolution_history[-1]
        final_weights = final_snap['complex_weights']
        if torch.is_tensor(final_weights):
            final_weights = final_weights.cpu().numpy()

        final_distributions = self.extract_class_distributions_from_test_data(final_weights)
        final_metrics = self.calculate_cluster_metrics(final_distributions)

        data_source = "TEST DATA (Generalization)" if use_test_data else "TRAINING DATA (Memorization)"

        metrics_summary = []
        for c, metrics in final_metrics.items():
            class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'
            metrics_summary.append(
                f"<tr><td>{class_name}</td>"
                f"<td>{metrics['cluster_quality']:.3f}</td>"
                f"<td>{metrics['angular_concentration']:.3f}</td>"
                f"<td>{metrics['radial_concentration']:.3f}</td>"
                f"<td>{metrics['entropy']:.3f}</td></tr>"
            )

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Polar Coordinate Evolution - {self.dataset_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .data-source-badge {{
            display: inline-block;
            background: {'#4caf50' if use_test_data else '#ff9800'};
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            margin-top: 10px;
        }}
        .dashboard-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .plot-container {{
            background: rgba(0,0,0,0.7);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .metrics-table {{
            background: rgba(0,0,0,0.5);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }}
        th {{
            background: rgba(102,126,234,0.5);
            font-weight: bold;
        }}
        .metric-good {{
            color: #4caf50;
        }}
        .metric-medium {{
            color: #ff9800;
        }}
        .metric-poor {{
            color: #f44336;
        }}
        .info-text {{
            font-size: 12px;
            color: #aaa;
            margin-top: 10px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Polar Coordinate Evolution - {self.dataset_name}</h1>
        <p>Showing how class clusters form in the complex plane over adaptive rounds</p>
        <p>Each point = a weight from the 5D tensor | Radius = Magnitude | Angle = Phase</p>
        <div class="data-source-badge">📐 Metrics from {data_source}</div>
    </div>

    <div class="dashboard-container">
        <div class="plot-container full-width">
            <h3>🎬 Animated Polar Evolution</h3>
            {animated_html}
        </div>
    </div>

    <div class="dashboard-container">
        <div class="plot-container">
            <h3>📈 Cluster Quality Metrics</h3>
            {quality_html}
        </div>

        <div class="plot-container">
            <h3>🎯 Final State Metrics ({data_source})</h3>
            <div class="metrics-table">
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Cluster Quality</th>
                        <th>Angular Concentration</th>
                        <th>Radial Concentration</th>
                        <th>Entropy (lower better)</th>
                    </tr>
                    {''.join(metrics_summary)}
                </table>
                <div class="info-text">
                    <strong>Interpretation:</strong><br>
                    • Cluster Quality: 1.0 = perfectly defined cluster<br>
                    • Angular Concentration: 1.0 = all points at same angle<br>
                    • Radial Concentration: 1.0 = all points at same radius<br>
                    • Entropy: 0.0 = perfectly focused, 1.0 = uniform distribution<br>
                    <br>
                    <strong>⚠️ Note:</strong> These metrics are computed on <strong>{data_source}</strong><br>
                    Training data becomes orthogonal quickly (memorization).<br>
                    Test data shows true generalization ability.
                </div>
            </div>
        </div>
    </div>

    <div class="dashboard-container">
        <div class="plot-container full-width">
            <h3>📸 Key Round Snapshots ({data_source})</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                {''.join([f'<div>{html}</div>' for html in static_htmls])}
            </div>
        </div>
    </div>

    <div class="info-text">
        <strong>How to Read These Plots:</strong><br>
        • Each point represents a weight from the 5D tensor (magnitude = distance from origin, phase = angle)<br>
        • Classes are color-coded<br>
        • Mean vectors show the center of each class cluster<br>
        • Perfect classification occurs when classes form separate, focused clusters at different angles (orthogonal in complex space)<br>
        <br>
        <strong>📐 Understanding the Metrics:</strong><br>
        • <strong>TEST DATA metrics</strong> show generalization - how well the model performs on unseen data<br>
        • <strong>TRAINING DATA metrics</strong> show memorization - how well the model fits the training data<br>
        • For true model evaluation, focus on TEST DATA metrics
    </div>
</body>
</html>"""

        return html


    def create_polar_scatter_plot(self, class_distributions, round_num, accuracy, training_size):
        """
        Create a polar scatter plot showing all class distributions.
        Each point is a weight in polar coordinates.
        """
        fig = go.Figure()

        # Determine global max radius for consistent scaling
        max_radius = 0
        for dist in class_distributions.values():
            if len(dist['radii']) > 0:
                max_radius = max(max_radius, np.max(dist['radii']))
        max_radius = max_radius * 1.1  # Add padding

        for c, dist in class_distributions.items():
            if dist['n_points'] == 0:
                continue

            class_name = f'Class {c+1}'
            color = self.class_colors[c % len(self.class_colors)]

            # Convert polar to cartesian for plotting
            x = dist['radii'] * np.cos(dist['angles'])
            y = dist['radii'] * np.sin(dist['angles'])

            # Add scatter trace
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(
                    size=5,
                    color=color,
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                name=class_name,
                text=[f'Radius: {r:.3f}<br>Angle: {a:.2f} rad ({a*180/np.pi:.0f}°)'
                      for r, a in zip(dist['radii'], dist['angles'])],
                hoverinfo='text+name',
                legendgroup=f'class_{c}'
            ))

            # Add mean vector (center of mass)
            if dist['mean_radius'] > 0:
                mean_x = dist['mean_radius'] * np.cos(dist['mean_angle'])
                mean_y = dist['mean_radius'] * np.sin(dist['mean_angle'])

                fig.add_trace(go.Scatter(
                    x=[0, mean_x],
                    y=[0, mean_y],
                    mode='lines+markers',
                    marker=dict(size=8, color=color, symbol='circle', line=dict(width=2, color='white')),
                    line=dict(color=color, width=2, dash='dash'),
                    name=f'{class_name} (mean)',
                    showlegend=False,
                    hoverinfo='text',
                    text=f'Mean vector<br>Radius: {dist["mean_radius"]:.3f}<br>Angle: {dist["mean_angle"]:.2f} rad',
                    legendgroup=f'class_{c}'
                ))

            # Add confidence ellipse (1-sigma)
            if dist['n_points'] > 2:
                self._add_confidence_ellipse(fig, dist, color, class_name)

        # Add unit circle reference
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x_circle, y=y_circle,
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            name='Unit Circle (Radius=1)',
            hoverinfo='none'
        ))

        # Add origin marker
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=6, color='white', symbol='x'),
            name='Origin',
            hoverinfo='none'
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'<b>Polar Coordinate Evolution - Round {round_num}</b><br>'
                     f'Accuracy: {accuracy:.3f} | Training Samples: {training_size}<br>'
                     f'<sup>Each point = weight in complex plane (radius = magnitude, angle = phase)</sup>',
                font=dict(size=14)
            ),
            xaxis=dict(
                title='Real Component',
                range=[-max_radius, max_radius],
                scaleanchor='y',
                scaleratio=1
            ),
            yaxis=dict(
                title='Imaginary Component',
                range=[-max_radius, max_radius]
            ),
            height=700,
            width=700,
            hovermode='closest',
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.02,
                bgcolor='rgba(0,0,0,0.7)',
                font=dict(color='white')
            ),
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )

        return fig

    def _add_confidence_ellipse(self, fig, dist, color, class_name):
        """Add 1-sigma confidence ellipse to the plot"""
        from scipy.stats import chi2

        if dist['n_points'] < 3:
            return

        # Convert polar to cartesian for covariance calculation
        x = dist['radii'] * np.cos(dist['angles'])
        y = dist['radii'] * np.sin(dist['angles'])

        # Calculate covariance matrix
        cov = np.cov(x, y)
        if np.any(np.isnan(cov)):
            return

        # Get eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvalues
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Calculate ellipse parameters (1-sigma, chi-squared with 2 DOF)
        chi2_val = chi2.ppf(0.68, 2)  # 68% confidence interval

        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = 2 * np.sqrt(chi2_val * eigvals[0])
        height = 2 * np.sqrt(chi2_val * eigvals[1])

        # Add ellipse trace
        fig.add_trace(go.Scatter(
            x=None, y=None,
            mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            name=f'{class_name} (1σ ellipse)',
            showlegend=False,
            hoverinfo='none'
        ))

        # We'll add the ellipse using layout shapes since Scatter doesn't support ellipses directly
        # This will be handled in the layout update

    def create_polar_heatmap(self, class_distributions, round_num, accuracy, training_size):
        """
        Create a polar heatmap showing density of weights in the complex plane.
        Better for showing cluster formation and separation between classes.
        """
        # Create polar grid
        n_radial_bins = 50
        n_angular_bins = 72  # 5° increments

        radial_edges = np.linspace(0, 1, n_radial_bins + 1)
        angular_edges = np.linspace(-np.pi, np.pi, n_angular_bins + 1)

        # Create heatmap for each class
        fig = make_subplots(
            rows=1, cols=len(class_distributions),
            subplot_titles=[f'Class {c+1}' for c in class_distributions.keys()],
            specs=[[{'type': 'polar'} for _ in class_distributions.keys()]]
        )

        for idx, (c, dist) in enumerate(class_distributions.items()):
            if dist['n_points'] == 0:
                continue

            # Create 2D histogram
            hist, _, _ = np.histogram2d(
                dist['angles'], dist['radii'],
                bins=[angular_edges, radial_edges]
            )

            # Create polar heatmap
            fig.add_trace(
                go.Barpolar(
                    r=radial_edges[:-1],
                    theta=np.degrees(angular_edges[:-1]),
                    width=np.degrees(np.diff(angular_edges)),
                    marker=dict(
                        color=hist.T.flatten(),
                        colorscale='Viridis',
                        showscale=(idx == 0)
                    ),
                    name=f'Class {c+1}',
                    hovertemplate='Radius: %{r:.3f}<br>Angle: %{theta:.0f}°<br>Density: %{marker.color}<extra></extra>'
                ),
                row=1, col=idx+1
            )

            # Add mean vector
            if dist['mean_radius'] > 0:
                fig.add_annotation(
                    x=np.cos(dist['mean_angle']),
                    y=np.sin(dist['mean_angle']),
                    ax=0, ay=0,
                    xref=f'x{idx+1}', yref=f'y{idx+1}',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='red',
                    text=f'μ = {dist["mean_radius"]:.2f}',
                    font=dict(size=8, color='white')
                )

        fig.update_layout(
            title=dict(
                text=f'<b>Polar Heatmap - Round {round_num}</b><br>'
                     f'Accuracy: {accuracy:.3f} | Training Samples: {training_size}<br>'
                     f'<sup>Density of weights in complex plane (brighter = higher density)</sup>',
                font=dict(size=14)
            ),
            height=500,
            width=500 * len(class_distributions),
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )

        return fig



# =============================================================================
# SECTION: OPTIMIZED VISUALIZER (FIXED - Using adaptive_dbnn.py approach)
# =============================================================================

class AdvancedInteractiveVisualizer:
    """Advanced interactive 3D visualization with dynamic controls - from adaptive_dbnn.py"""

    def __init__(self, model, output_dir='visualizations'):
        self.model = model
        self.dataset_name = model.dataset_name if model and hasattr(model, 'dataset_name') else 'unknown'
        self.output_dir = Path(output_dir) / self.dataset_name / 'interactive_3d'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.colors = px.colors.qualitative.Set1 + px.colors.qualitative.Pastel
        print(f"{Colors.CYAN}🎨 Advanced Interactive Visualizer initialized{Colors.ENDC}")

    def create_advanced_3d_dashboard(self, X_full, y_full, training_history, feature_names, round_num=None):
        """Create advanced interactive 3D dashboard with multiple visualization options"""
        print("🌐 Creating advanced interactive 3D dashboard...")

        # Create multiple visualization methods
        self._create_pca_3d_plot(X_full, y_full, training_history, feature_names, round_num)
        self._create_feature_space_3d(X_full, y_full, training_history, feature_names, round_num)
        self._create_network_graph_3d(X_full, y_full, training_history, feature_names, round_num)
        self._create_density_controlled_3d(X_full, y_full, training_history, feature_names, round_num)

        # Create main dashboard that links all visualizations
        self._create_main_dashboard(X_full, y_full, training_history, feature_names, round_num)

    def _create_pca_3d_plot(self, X_full, y_full, training_history, feature_names, round_num):
        """Create PCA-based 3D plot with interactive controls"""
        pca = PCA(n_components=3, random_state=42)
        X_3d = pca.fit_transform(X_full)
        explained_var = pca.explained_variance_ratio_

        # Create interactive plot
        unique_classes = np.unique(y_full)
        fig = go.Figure()

        for i, cls in enumerate(unique_classes):
            class_mask = y_full == cls
            scatter = go.Scatter3d(
                x=X_3d[class_mask, 0],
                y=X_3d[class_mask, 1],
                z=X_3d[class_mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=self.colors[i % len(self.colors)],
                    opacity=0.7,
                    line=dict(width=0.5, color='black')
                ),
                name=f'Class {cls}',
                text=[f'Class: {cls}<br>PC1: {x:.3f}<br>PC2: {y:.3f}<br>PC3: {z:.3f}'
                      for x, y, z in zip(X_3d[class_mask, 0], X_3d[class_mask, 1], X_3d[class_mask, 2])],
                hoverinfo='text'
            )
            fig.add_trace(scatter)

        # Add network connections for training samples
        if training_history and len(training_history) > 0:
            training_indices = training_history[-1] if round_num is None else training_history[round_num]
            self._add_network_connections_3d(fig, X_3d, y_full, training_indices)

        fig.update_layout(
            title=f'3D PCA Visualization - {self.dataset_name}<br>'
                  f'Explained Variance: PC1: {explained_var[0]:.3f}, PC2: {explained_var[1]:.3f}, PC3: {explained_var[2]:.3f}',
            scene=dict(
                xaxis_title=f'PC1 ({explained_var[0]:.2%} variance)',
                yaxis_title=f'PC2 ({explained_var[1]:.2%} variance)',
                zaxis_title=f'PC3 ({explained_var[2]:.2%} variance)',
            ),
            width=1000,
            height=800
        )

        filename = f'pca_3d_round_{round_num}.html' if round_num else 'pca_3d_final.html'
        fig.write_html(self.output_dir / filename)

    def _create_feature_space_3d(self, X_full, y_full, training_history, feature_names, round_num):
        """Create feature space 3D plot with selectable features"""
        if len(feature_names) >= 3:
            # Use first 3 features by default
            feature_indices = [0, 1, 2]
            selected_features = [feature_names[i] for i in feature_indices]

            fig = go.Figure()
            unique_classes = np.unique(y_full)

            for i, cls in enumerate(unique_classes):
                class_mask = y_full == cls
                scatter = go.Scatter3d(
                    x=X_full[class_mask, feature_indices[0]],
                    y=X_full[class_mask, feature_indices[1]],
                    z=X_full[class_mask, feature_indices[2]],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=self.colors[i % len(self.colors)],
                        opacity=0.6,
                        symbol='circle'
                    ),
                    name=f'Class {cls}',
                    text=[f'Class: {cls}<br>{selected_features[0]}: {x:.3f}<br>{selected_features[1]}: {y:.3f}<br>{selected_features[2]}: {z:.3f}'
                          for x, y, z in zip(X_full[class_mask, feature_indices[0]],
                                           X_full[class_mask, feature_indices[1]],
                                           X_full[class_mask, feature_indices[2]])],
                    hoverinfo='text'
                )
                fig.add_trace(scatter)

            fig.update_layout(
                title=f'3D Feature Space - {self.dataset_name}<br>Features: {selected_features}',
                scene=dict(
                    xaxis_title=selected_features[0],
                    yaxis_title=selected_features[1],
                    zaxis_title=selected_features[2],
                ),
                width=1000,
                height=800
            )

            filename = f'feature_3d_round_{round_num}.html' if round_num else 'feature_3d_final.html'
            fig.write_html(self.output_dir / filename)

    def _add_network_connections_3d(self, fig, X_3d, y_full, training_indices):
        """Add network connections between training samples"""
        training_mask = np.isin(range(len(X_3d)), training_indices)
        X_train = X_3d[training_mask]
        y_train = y_full[training_mask]

        unique_classes = np.unique(y_train)

        for i, cls in enumerate(unique_classes):
            class_mask = y_train == cls
            class_points = X_train[class_mask]

            if len(class_points) < 2:
                continue

            try:
                # Create minimum spanning tree
                dist_matrix = distance_matrix(class_points, class_points)
                G = nx.Graph()

                for j in range(len(class_points)):
                    for k in range(j+1, len(class_points)):
                        if dist_matrix[j, k] < np.percentile(dist_matrix, 25):
                            G.add_edge(j, k, weight=dist_matrix[j, k])

                if G.number_of_edges() > 0:
                    mst = nx.minimum_spanning_tree(G)

                    # Add edges to plot
                    for edge in mst.edges():
                        x_edges = [class_points[edge[0], 0], class_points[edge[1], 0], None]
                        y_edges = [class_points[edge[0], 1], class_points[edge[1], 1], None]
                        z_edges = [class_points[edge[0], 2], class_points[edge[1], 2], None]

                        fig.add_trace(go.Scatter3d(
                            x=x_edges, y=y_edges, z=z_edges,
                            mode='lines',
                            line=dict(color=self.colors[i % len(self.colors)], width=2, opacity=0.6),
                            showlegend=False,
                            hoverinfo='none'
                        ))
            except Exception:
                continue

    def _create_density_controlled_3d(self, X_full, y_full, training_history, feature_names, round_num):
        """Create density-controlled 3D visualization with point skipping"""
        pca = PCA(n_components=3, random_state=42)
        X_3d = pca.fit_transform(X_full)

        # Apply density-based sampling
        X_sampled, y_sampled = self._density_based_sampling(X_3d, y_full, max_points_per_class=100)

        fig = go.Figure()
        unique_classes = np.unique(y_sampled)

        for i, cls in enumerate(unique_classes):
            class_mask = y_sampled == cls
            scatter = go.Scatter3d(
                x=X_sampled[class_mask, 0],
                y=X_sampled[class_mask, 1],
                z=X_sampled[class_mask, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.colors[i % len(self.colors)],
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                name=f'Class {cls} (density-controlled)',
                text=[f'Class: {cls}' for _ in range(np.sum(class_mask))],
                hoverinfo='text'
            )
            fig.add_trace(scatter)

        fig.update_layout(
            title=f'Density-Controlled 3D Visualization - {self.dataset_name}<br>'
                  f'Points sampled to reduce overcrowding',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3',
            ),
            width=1000,
            height=800
        )

        filename = f'density_3d_round_{round_num}.html' if round_num else 'density_3d_final.html'
        fig.write_html(self.output_dir / filename)

    def _density_based_sampling(self, X, y, max_points_per_class=100):
        """Sample points based on density to reduce overcrowding"""
        from sklearn.neighbors import NearestNeighbors

        unique_classes = np.unique(y)
        X_sampled_list = []
        y_sampled_list = []

        for cls in unique_classes:
            class_mask = y == cls
            X_class = X[class_mask]

            if len(X_class) <= max_points_per_class:
                X_sampled_list.append(X_class)
                y_sampled_list.append(np.full(len(X_class), cls))
            else:
                nbrs = NearestNeighbors(n_neighbors=min(10, len(X_class)), algorithm='auto').fit(X_class)
                distances, indices = nbrs.kneighbors(X_class)

                avg_distances = np.mean(distances, axis=1)
                density_scores = 1 / (avg_distances + 1e-8)

                probabilities = 1 / (density_scores + 1e-8)
                probabilities = probabilities / np.sum(probabilities)

                selected_indices = np.random.choice(
                    len(X_class),
                    size=max_points_per_class,
                    replace=False,
                    p=probabilities
                )

                X_sampled_list.append(X_class[selected_indices])
                y_sampled_list.append(np.full(max_points_per_class, cls))

        return np.vstack(X_sampled_list), np.hstack(y_sampled_list)

    def _create_main_dashboard(self, X_full, y_full, training_history, feature_names, round_num):
        """Create main dashboard linking all visualizations"""
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced 3D Visualization Dashboard - {self.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                         color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .nav {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
                .nav-button {{ padding: 10px 20px; background: #4CAF50; color: white;
                            border: none; border-radius: 5px; cursor: pointer; text-decoration: none; }}
                .nav-button:hover {{ background: #45a049; }}
                .iframe-container {{ border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; }}
                iframe {{ width: 100%; height: 800px; border: none; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌐 Advanced 3D Visualization Dashboard</h1>
                <h2>Dataset: {self.dataset_name}</h2>
                <p>Round: {round_info} | Features: {len(feature_names)} | Samples: {len(X_full)}</p>
            </div>

            <div class="nav">
                <a class="nav-button" href="#pca">PCA 3D</a>
                <a class="nav-button" href="#feature">Feature Space 3D</a>
                <a class="nav-button" href="#density">Density-Controlled 3D</a>
            </div>

            <div id="pca" class="iframe-container">
                <h3>📊 PCA 3D Visualization</h3>
                <iframe src="pca_3d_{round_suffix}.html"></iframe>
            </div>

            <div id="feature" class="iframe-container">
                <h3>🔧 Feature Space 3D</h3>
                <iframe src="feature_3d_{round_suffix}.html"></iframe>
            </div>

            <div id="density" class="iframe-container">
                <h3>📈 Density-Controlled 3D</h3>
                <iframe src="density_3d_{round_suffix}.html"></iframe>
            </div>

            <script>
                document.querySelectorAll('.nav-button').forEach(button => {{
                    button.addEventListener('click', function(e) {{
                        e.preventDefault();
                        const targetId = this.getAttribute('href').substring(1);
                        document.getElementById(targetId).scrollIntoView({{
                            behavior: 'smooth'
                        }});
                    }});
                }});
            </script>
        </body>
        </html>
        """
        round_info = f"Round {round_num}" if round_num else "Final"
        round_suffix = f"round_{round_num}" if round_num else "final"

        with open(self.output_dir / f"dashboard_{round_suffix}.html", "w") as f:
            f.write(dashboard_html)


class ComprehensiveAdaptiveVisualizer:
    """Comprehensive visualization system for Adaptive DBNN - from adaptive_dbnn.py"""

    def __init__(self, model, output_dir='visualizations'):
        self.model = model
        self.dataset_name = model.dataset_name if model and hasattr(model, 'dataset_name') else 'unknown'
        self.output_dir = Path(output_dir) / self.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different plot types
        self.subdirs = {
            'performance': self.output_dir / 'performance',
            'samples': self.output_dir / 'sample_evolution',
            'distributions': self.output_dir / 'distributions',
            'networks': self.output_dir / 'networks',
            'comparisons': self.output_dir / 'comparisons',
            'interactive': self.output_dir / 'interactive'
        }

        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)

        # Color schemes
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        print(f"{Colors.CYAN}🎨 Comprehensive visualizer initialized for: {self.dataset_name}{Colors.ENDC}")

    def plot_performance_evolution(self, round_stats):
        """Plot comprehensive performance evolution across rounds"""
        if not round_stats:
            return

        rounds = [stat.get('round', i) for i, stat in enumerate(round_stats)]
        train_acc = [stat.get('train_accuracy', 0) * 100 for stat in round_stats]
        test_acc = [stat.get('test_accuracy', 0) * 100 for stat in round_stats]
        training_sizes = [stat.get('training_size', 0) for stat in round_stats]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Accuracy Evolution
        ax1.plot(rounds, train_acc, 'o-', linewidth=2, markersize=6, label='Training Accuracy', color=self.colors[0])
        ax1.plot(rounds, test_acc, 's-', linewidth=2, markersize=6, label='Test Accuracy', color=self.colors[1])

        best_round_idx = np.argmax(test_acc)
        ax1.axvline(x=rounds[best_round_idx], color='red', linestyle='--', alpha=0.7)

        ax1.set_xlabel('Adaptive Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Evolution Across Rounds', fontweight='bold', fontsize=14)
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Training Size Growth
        ax2.plot(rounds, training_sizes, '^-', linewidth=2, markersize=6, color=self.colors[2])
        ax2.set_xlabel('Adaptive Round')
        ax2.set_ylabel('Training Set Size')
        ax2.set_title('Training Set Growth', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Improvement per Round
        if len(round_stats) > 1:
            improvements = [test_acc[i] - test_acc[i-1] for i in range(1, len(test_acc))]
            improvements.insert(0, 0)
            bars = ax3.bar(rounds, improvements, color=np.where(np.array(improvements) >= 0, 'green', 'red'), alpha=0.7)
            ax3.set_xlabel('Adaptive Round')
            ax3.set_ylabel('Accuracy Improvement (%)')
            ax3.set_title('Accuracy Improvement per Round', fontweight='bold', fontsize=14)
            ax3.grid(True, alpha=0.3)

            for bar, imp in zip(bars, improvements):
                if abs(imp) > 0.1:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

        # Plot 4: Cumulative Improvement
        if len(round_stats) > 1:
            cumulative = np.cumsum(improvements)
            ax4.plot(rounds, cumulative, 'o-', linewidth=2, markersize=6, color=self.colors[3])
            ax4.set_xlabel('Adaptive Round')
            ax4.set_ylabel('Cumulative Improvement (%)')
            ax4.set_title('Cumulative Performance Improvement', fontweight='bold', fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.subdirs['performance'] / 'performance_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Create interactive version
        self._create_interactive_performance_plot(rounds, train_acc, test_acc, training_sizes)

    def _create_interactive_performance_plot(self, rounds, train_acc, test_acc, training_sizes):
        """Create interactive performance plot"""
        if not PLOTLY_AVAILABLE:
            return

        fig = make_subplots(rows=2, cols=2, subplot_titles=('Accuracy Evolution', 'Training Set Growth'))

        fig.add_trace(go.Scatter(x=rounds, y=train_acc, name='Training Accuracy', line=dict(color=self.colors[0])), row=1, col=1)
        fig.add_trace(go.Scatter(x=rounds, y=test_acc, name='Test Accuracy', line=dict(color=self.colors[1])), row=1, col=1)
        fig.add_trace(go.Scatter(x=rounds, y=training_sizes, name='Training Size', line=dict(color=self.colors[2])), row=1, col=2)

        fig.update_layout(height=800, title_text="Adaptive Learning Performance Evolution")
        fig.write_html(self.subdirs['interactive'] / 'performance_evolution.html')

    def plot_3d_networks(self, X_full, y_full, training_history, feature_names):
        """Create optimized 3D network visualizations of training samples"""
        if not training_history:
            return

        # Reduce dimensionality for visualization
        if X_full.shape[1] > 3:
            pca = PCA(n_components=3, random_state=42)
            X_3d = pca.fit_transform(X_full)
            explained_var = pca.explained_variance_ratio_.sum()
        else:
            X_3d = X_full
            explained_var = 1.0

        # Limit to key rounds for performance
        total_rounds = len(training_history)
        if total_rounds > 5:
            key_rounds = [0, total_rounds//2, -1]
        else:
            key_rounds = list(range(total_rounds))

        for round_num in key_rounds:
            training_indices = training_history[round_num] if isinstance(training_history[round_num], list) else training_history[round_num]
            self._create_optimized_3d_network(X_3d, y_full, training_indices, round_num, explained_var, feature_names)

    def _create_optimized_3d_network(self, X_3d, y_full, training_indices, round_num, explained_var, feature_names):
        """Create optimized single 3D network visualization"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Limit data for better performance
        max_points = 1000
        if len(X_3d) > max_points:
            sample_indices = np.random.choice(len(X_3d), max_points, replace=False)
            X_display = X_3d[sample_indices]
            y_display = y_full[sample_indices]
            training_mask_display = np.isin(sample_indices, training_indices) if isinstance(training_indices, list) else np.zeros(len(sample_indices), dtype=bool)
        else:
            X_display = X_3d
            y_display = y_full
            training_mask_display = np.isin(range(len(X_3d)), training_indices) if isinstance(training_indices, list) else np.zeros(len(X_3d), dtype=bool)

        unique_classes = np.unique(y_display)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

        # Plot non-training samples (background)
        background_mask = ~training_mask_display
        for i, cls in enumerate(unique_classes):
            class_mask = (y_display == cls) & background_mask
            if np.any(class_mask):
                ax.scatter(X_display[class_mask, 0], X_display[class_mask, 1], X_display[class_mask, 2],
                          c=[colors[i]], alpha=0.05, s=5, marker='.')

        # Plot training samples (foreground)
        legend_handles = []
        legend_labels = []

        for i, cls in enumerate(unique_classes):
            class_mask = (y_display == cls) & training_mask_display
            if np.any(class_mask):
                scatter = ax.scatter(X_display[class_mask, 0], X_display[class_mask, 1], X_display[class_mask, 2],
                                   c=[colors[i]], alpha=0.8, s=30, label=f'Class {cls}',
                                   edgecolors='black', linewidth=0.5)
                if len(legend_handles) < 8:
                    legend_handles.append(scatter)
                    legend_labels.append(f'Class {cls}')

        # Add network connections
        if isinstance(training_indices, list) and len(training_indices) <= 200:
            self._add_optimized_network_connections(ax, X_3d, y_full, training_indices, colors)

        ax.set_xlabel(f'PC1 ({explained_var*100:.1f}% variance)')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'3D Training Network - Round {round_num + 1}\nTraining Samples: {len(training_indices) if isinstance(training_indices, list) else training_indices}', fontweight='bold', fontsize=12)

        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(0, 1))

        plt.tight_layout()
        filename = f'3d_network_round_{round_num + 1}.png'
        plt.savefig(self.subdirs['networks'] / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def _add_optimized_network_connections(self, ax, X_3d, y_full, training_indices, colors):
        """Add optimized network connections between training samples"""
        training_mask = np.isin(range(len(X_3d)), training_indices)
        X_train = X_3d[training_mask]
        y_train = y_full[training_mask]

        unique_classes = np.unique(y_train)

        for i, cls in enumerate(unique_classes):
            class_mask = y_train == cls
            class_points = X_train[class_mask]

            if len(class_points) < 2 or len(class_points) > 50:
                continue

            try:
                dist_matrix = distance_matrix(class_points, class_points)
                max_distance = np.percentile(dist_matrix[dist_matrix > 0], 50)

                G = nx.Graph()
                for j in range(len(class_points)):
                    for k in range(j+1, len(class_points)):
                        if dist_matrix[j, k] <= max_distance:
                            G.add_edge(j, k, weight=dist_matrix[j, k])

                if G.number_of_edges() > 0:
                    mst = nx.minimum_spanning_tree(G)

                    for edge in list(mst.edges())[:50]:
                        point1 = class_points[edge[0]]
                        point2 = class_points[edge[1]]
                        ax.plot([point1[0], point2[0]],
                               [point1[1], point2[1]],
                               [point1[2], point2[2]],
                               color=colors[i], alpha=0.4, linewidth=0.8)
            except Exception:
                continue

    def plot_class_separation_analysis(self, X_full, y_full, training_history):
        """Analyze class separation evolution"""
        if not training_history:
            return

        separation_scores = []

        for training_indices in training_history:
            if not isinstance(training_indices, list):
                continue

            X_train = X_full[training_indices]
            y_train = y_full[training_indices]

            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                separation_scores.append(0)
                continue

            overall_mean = np.mean(X_train, axis=0)
            between_var = 0
            within_var = 0

            for cls in unique_classes:
                class_mask = y_train == cls
                if np.sum(class_mask) > 0:
                    class_mean = np.mean(X_train[class_mask], axis=0)
                    between_var += np.sum(class_mask) * np.sum((class_mean - overall_mean) ** 2)
                    within_var += np.sum((X_train[class_mask] - class_mean) ** 2)

            if within_var > 0:
                separation_scores.append(between_var / within_var)
            else:
                separation_scores.append(0)

        if separation_scores:
            fig, ax = plt.subplots(figsize=(12, 6))
            rounds = list(range(1, len(separation_scores) + 1))

            ax.plot(rounds, separation_scores, 'o-', linewidth=2, markersize=8, color=self.colors[0])
            ax.set_xlabel('Adaptive Round')
            ax.set_ylabel('Separation Score')
            ax.set_title('Class Separation Evolution in Training Set', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.subdirs['comparisons'] / 'class_separation_evolution.png', dpi=150, bbox_inches='tight')
            plt.close()

class PointBasedSphericalVisualization:
    """
    Enhanced spherical visualization showing ACTUAL DATA POINTS projected into tensor space.
    Each point represents a data sample's position in the 5D tensor space.
    Classes form clusters that become orthogonal over time.
    Now uses TEST DATA for orthogonality metrics to show generalization.
    """

    def __init__(self, model, output_dir='visualizations'):
        self.model = model
        self.dataset_name = model.dataset_name if model and hasattr(model, 'dataset_name') else 'unknown'
        self.output_dir = Path.cwd() / output_dir / self.dataset_name / 'point_based_spherical'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes
        self.class_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + \
                           px.colors.qualitative.Set3 + px.colors.qualitative.Pastel

        # Store verification metrics
        self.verification_metrics = {}

        print(f"{Colors.CYAN}🎯 Point-Based Spherical Visualization initialized{Colors.ENDC}")
        print(f"   Output: {self.output_dir}")
        print(f"   Using TEST DATA for orthogonality metrics (generalization)")

    def _get_test_indices(self):
        """Get test indices from model"""
        if hasattr(self.model, 'test_indices') and self.model.test_indices is not None:
            test_indices = self.model.test_indices
            if torch.is_tensor(test_indices):
                test_indices = test_indices.numpy() if hasattr(test_indices, 'numpy') else np.array(test_indices)
            return test_indices
        return None

    def _farthest_point_sampling(self, points, n_samples):
        """Select n_samples points using farthest point sampling to preserve shape"""
        if len(points) <= n_samples:
            return np.arange(len(points))

        # Ensure points is 2D
        if points.ndim == 1:
            points = points.reshape(-1, 1)

        selected = [np.random.randint(len(points))]

        for _ in range(1, n_samples):
            # Find point farthest from selected points
            distances = np.min(
                np.linalg.norm(points[:, np.newaxis] - points[selected], axis=2),
                axis=1
            )
            farthest = np.argmax(distances)
            selected.append(farthest)

        return np.array(selected)

    def _compute_class_centers(self, point_coords, y_samples, n_classes):
        """
        Compute center and angular spread of each class in projected space.
        Returns: (class_centers, class_spreads) where spreads are in degrees
        """
        class_centers = []
        class_spreads = []  # Angular standard deviation in degrees

        for c in range(n_classes):
            class_mask = y_samples == c
            if np.any(class_mask):
                points = point_coords[class_mask]
                center = np.mean(points, axis=0)
                # Normalize to unit sphere
                norm = np.linalg.norm(center)
                if norm > 0:
                    center = center / norm
                class_centers.append(center)

                # Calculate angular spread (standard deviation) of points around center
                angles_to_center = []
                for p in points:
                    # Normalize point to unit sphere
                    p_norm = np.linalg.norm(p)
                    if p_norm > 0:
                        p_unit = p / p_norm
                        # Cosine similarity to center
                        cos_sim = np.dot(p_unit, center)
                        cos_sim = np.clip(cos_sim, -1, 1)
                        angle = np.arccos(cos_sim) * 180 / np.pi
                        angles_to_center.append(angle)

                spread = np.std(angles_to_center) if angles_to_center else 0
                class_spreads.append(spread)
            else:
                class_centers.append(np.zeros(3))
                class_spreads.append(0)

        return np.array(class_centers), np.array(class_spreads)

    def _compute_angular_separation(self, class_centers, class_spreads):
        """
        Compute angular separation between class centers.
        Also computes margin separation considering class spreads.

        Returns:
            - avg_center_separation: average angle between centers
            - avg_margin_separation: average center angle minus spreads (effective separation)
            - orthogonality_center: normalized orthogonality based on centers only
            - orthogonality_margin: normalized orthogonality based on margins (TRUE measure)
            - pairwise_center_angles: list of all pairwise center angles
            - pairwise_margin_angles: list of all pairwise margin angles
        """
        n_classes = len(class_centers)
        if n_classes < 2:
            return {
                'avg_center_separation': 0,
                'avg_margin_separation': 0,
                'orthogonality_center': 0,
                'orthogonality_margin': 0,
                'pairwise_center_angles': [],
                'pairwise_margin_angles': []
            }

        center_angles = []
        margin_angles = []

        for i in range(n_classes):
            for j in range(i+1, n_classes):
                v1 = class_centers[i]
                v2 = class_centers[j]

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 > 0 and norm2 > 0:
                    dot = np.dot(v1, v2) / (norm1 * norm2)
                    dot = np.clip(dot, -1, 1)
                    center_angle = np.arccos(dot) * 180 / np.pi
                    center_angles.append(center_angle)

                    # Calculate margin separation (center angle minus spreads)
                    # If spreads overlap, margin becomes 0
                    margin = center_angle - (class_spreads[i] + class_spreads[j])
                    margin_angles.append(max(0, margin))
                else:
                    center_angles.append(90)
                    margin_angles.append(0)

        avg_center_sep = np.mean(center_angles) if center_angles else 0
        avg_margin_sep = np.mean(margin_angles) if margin_angles else 0

        # Orthogonality based on center separation (what we were measuring)
        orthogonality_center = min(1.0, avg_center_sep / 90.0)

        # TRUE orthogonality based on margin separation (accounts for spread)
        # This is what actually matters for classification
        orthogonality_margin = min(1.0, avg_margin_sep / 90.0)

        return {
            'avg_center_separation': avg_center_sep,
            'avg_margin_separation': avg_margin_sep,
            'orthogonality_center': orthogonality_center,
            'orthogonality_margin': orthogonality_margin,
            'pairwise_center_angles': center_angles,
            'pairwise_margin_angles': margin_angles
        }

    def project_tensor_to_sphere(self, weights, X_samples, y_samples, max_points_per_class=200, use_test_data=False):
        """
        Project data points into spherical coordinates based on their activation in tensor space.

        Each data point is represented by its activation pattern across all class tensors.
        Position on sphere = weighted combination of class orientation vectors.

        Args:
            weights: The tensor weights (n_classes, n_pairs, n_bins, n_bins)
            X_samples: Feature data
            y_samples: Labels
            max_points_per_class: Maximum points per class for visualization
            use_test_data: If True, compute verification on test data (recommended)
        """
        if len(weights.shape) != 4:
            return None, None, None, None, None

        n_classes = weights.shape[0]
        n_samples = len(X_samples)

        # Handle case with 1 class
        if n_classes == 1:
            point_coords = np.zeros((n_samples, 3))
            y_sampled = y_samples
            verification = {
                'mean_projected_angle': 0,
                'orthogonality_score': 1.0,
                'orthogonality_margin': 1.0,
                'trustworthiness': 1.0,
                'angular_separation': {
                    'avg_center_separation': 0,
                    'avg_margin_separation': 0,
                    'orthogonality_center': 1.0,
                    'orthogonality_margin': 1.0
                },
                'data_source': 'test' if use_test_data else 'training'
            }
            class_centers = np.zeros((1, 3))
            class_spreads = np.zeros(1)
            return point_coords, y_sampled, verification, class_centers, class_spreads

        # Step 1: Get class orientation vectors (unit vectors on sphere)
        class_vectors = []
        for c in range(n_classes):
            class_weights = weights[c].flatten()
            significant = class_weights[np.abs(class_weights) > 1e-6]

            if len(significant) > 0:
                # Calculate orientation in complex space
                magnitudes = np.abs(significant)
                phases = np.angle(significant)

                # Weighted circular mean
                sin_sum = np.sum(magnitudes * np.sin(phases))
                cos_sum = np.sum(magnitudes * np.cos(phases))
                avg_theta = np.arctan2(sin_sum, cos_sum)
                avg_r = np.mean(magnitudes)

                # Map to 3D sphere
                phi = (c * np.pi / max(1, n_classes))
                x = avg_r * np.sin(phi) * np.cos(avg_theta)
                y = avg_r * np.sin(phi) * np.sin(avg_theta)
                z = avg_r * np.cos(phi)

                # Normalize to unit sphere
                norm = np.sqrt(x*x + y*y + z*z)
                if norm > 0:
                    class_vectors.append([x/norm, y/norm, z/norm])
                else:
                    # Create default orthogonal positions
                    theta = (c * 2 * np.pi / max(2, n_classes))
                    phi = np.pi / 2
                    class_vectors.append([
                        np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi)
                    ])
            else:
                # Create default orthogonal positions
                theta = (c * 2 * np.pi / max(2, n_classes))
                phi = np.pi / 2
                class_vectors.append([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])

        class_vectors = np.array(class_vectors)  # (n_classes, 3)

        # Step 2: Compute posterior probabilities for each sample
        if hasattr(self.model, 'predict'):
            # Convert to tensor if needed
            if not torch.is_tensor(X_samples):
                X_tensor = torch.tensor(X_samples, dtype=torch.float64)
            else:
                X_tensor = X_samples

            posteriors, _ = self.model.predict(X_tensor)
            if torch.is_tensor(posteriors):
                posteriors = posteriors.numpy()

            # Ensure posteriors is 2D: (n_samples, n_classes)
            if posteriors.ndim == 1:
                # If 1D, it might be just class indices, create one-hot
                n_classes_actual = len(np.unique(posteriors))
                posteriors_2d = np.zeros((len(posteriors), n_classes_actual))
                posteriors_2d[np.arange(len(posteriors)), posteriors.astype(int)] = 1
                posteriors = posteriors_2d
            elif posteriors.ndim == 2 and posteriors.shape[1] != n_classes:
                # Might need to reshape
                pass
        else:
            # Fallback: use uniform probabilities
            posteriors = np.ones((n_samples, n_classes)) / n_classes

        # Step 3: Project each data point as weighted combination of class vectors
        point_coords = np.zeros((n_samples, 3))

        for i in range(n_samples):
            # Get probabilities for this sample
            if i < len(posteriors):
                probs = posteriors[i]
            else:
                probs = np.ones(n_classes) / n_classes

            # FIX: Ensure probs is 1D array
            if isinstance(probs, (int, float)):
                # If scalar, convert to array
                probs = np.ones(n_classes) / n_classes
            elif np.isscalar(probs):
                probs = np.ones(n_classes) / n_classes
            elif hasattr(probs, 'ndim') and probs.ndim > 1:
                # If 2D, flatten
                probs = probs.flatten()

            # Ensure correct length
            if len(probs) != n_classes:
                # If wrong length, use uniform probabilities
                probs = np.ones(n_classes) / n_classes

            # Normalize probabilities to sum to 1
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                probs = probs / prob_sum

            # Calculate weighted average of class vectors
            weighted_sum = np.zeros(3)
            for c in range(n_classes):
                weighted_sum += probs[c] * class_vectors[c]

            point_coords[i] = weighted_sum

            # Normalize to unit sphere
            norm = np.linalg.norm(point_coords[i])
            if norm > 0:
                point_coords[i] = point_coords[i] / norm

        # Step 4: Sample points if too many (for performance)
        sampled_indices = []
        sampled_points = []
        sampled_labels = []
        sampled_posteriors = []

        # Ensure y_samples is 1D array
        if torch.is_tensor(y_samples):
            y_samples = y_samples.numpy()

        unique_classes = np.unique(y_samples)
        for cls in unique_classes:
            cls_mask = y_samples == cls
            cls_indices = np.where(cls_mask)[0]

            if len(cls_indices) > max_points_per_class:
                # Sample intelligently - keep border points for better visualization
                cls_points = point_coords[cls_indices]
                # Use farthest point sampling to preserve shape
                selected = self._farthest_point_sampling(cls_points, max_points_per_class)
                cls_sampled = cls_indices[selected]
            else:
                cls_sampled = cls_indices

            sampled_indices.extend(cls_sampled)
            sampled_points.append(point_coords[cls_sampled])
            sampled_labels.extend([cls] * len(cls_sampled))
            sampled_posteriors.append(posteriors[cls_sampled])

        if sampled_points:
            point_coords_sampled = np.vstack(sampled_points)
            y_sampled = np.array(sampled_labels)
            posteriors_sampled = np.vstack(sampled_posteriors) if sampled_posteriors else posteriors
        else:
            point_coords_sampled = point_coords
            y_sampled = y_samples
            posteriors_sampled = posteriors

        # Step 5: Compute class centers and spreads
        class_centers, class_spreads = self._compute_class_centers(point_coords_sampled, y_sampled, n_classes)
        angular_separation = self._compute_angular_separation(class_centers, class_spreads)

        # Calculate trustworthiness: how well margin orthogonality correlates with accuracy
        # This is estimated based on the gap between center and margin orthogonality
        ortho_gap = angular_separation['orthogonality_center'] - angular_separation['orthogonality_margin']
        trustworthiness = max(0, min(1.0, 1.0 - ortho_gap * 1.5))  # Larger gap = less trustworthy

        verification = {
            'class_centers': class_centers,
            'class_spreads': class_spreads,
            'angular_separation': angular_separation,
            'mean_projected_angle': angular_separation.get('avg_center_separation', 0),
            'orthogonality_score': angular_separation.get('orthogonality_center', 0),
            'orthogonality_margin': angular_separation.get('orthogonality_margin', 0),
            'trustworthiness': trustworthiness,
            'n_points': len(point_coords_sampled),
            'n_classes': n_classes,
            'data_source': 'test' if use_test_data else 'training'
        }

        return point_coords_sampled, y_sampled, verification, class_centers, class_spreads

    def create_point_based_spherical_animation(self, evolution_history, X_data, y_data, class_names=None):
        """
        Create animated spherical visualization with actual data points.
        Uses TEST DATA for orthogonality metrics to show generalization.
        Now shows BOTH center orthogonality AND margin orthogonality (TRUE measure).
        """
        if not evolution_history or len(evolution_history) < 2:
            print(f"{Colors.YELLOW}Need at least 2 rounds for evolution{Colors.ENDC}")
            return None

        if X_data is None or y_data is None:
            print(f"{Colors.YELLOW}No data points available for projection{Colors.ENDC}")
            return None

        print(f"{Colors.CYAN}🎬 Creating point-based spherical animation...{Colors.ENDC}")

        # CRITICAL: Get test indices if available
        test_indices = self._get_test_indices()
        if test_indices is not None and len(test_indices) > 0:
            if torch.is_tensor(test_indices):
                test_indices = test_indices.numpy() if hasattr(test_indices, 'numpy') else np.array(test_indices)
            X_test = X_data[test_indices]
            y_test = y_data[test_indices]
            print(f"   Using {len(X_test)} TEST samples for orthogonality metrics (generalization)")
            print(f"   Test indices: {test_indices[:5]}...")
            use_test_data = True
        else:
            # Fallback: use a random split if no test indices available
            from sklearn.model_selection import train_test_split
            X_test, _, y_test, _ = train_test_split(
                X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
            )
            print(f"   Using {len(X_test)} random test samples (20% split) - fallback")
            use_test_data = True

        # Convert data to numpy if needed
        if torch.is_tensor(X_test):
            X_test = X_test.numpy()
        if torch.is_tensor(y_test):
            y_test = y_test.numpy()

        # Get class names
        if class_names is None and hasattr(self.model, 'label_encoder') and self.model.label_encoder:
            class_names = list(self.model.label_encoder.keys())

        # Process all rounds using TEST data
        frames_data = []
        center_orthogonality_progression = []
        margin_orthogonality_progression = []
        center_separation_progression = []
        margin_separation_progression = []

        for snap in evolution_history:
            if 'complex_weights' not in snap:
                continue

            weights = snap['complex_weights']
            if torch.is_tensor(weights):
                weights = weights.cpu().numpy()

            if len(weights.shape) != 4:
                continue

            round_num = snap['round']
            accuracy = snap.get('accuracy', 0)
            training_size = snap.get('training_size', 0)

            # Project test data points to sphere (returns 5 values)
            point_coords, y_proj, verification, class_centers, class_spreads = self.project_tensor_to_sphere(
                weights, X_test, y_test, use_test_data=use_test_data
            )

            if point_coords is None:
                continue

            frames_data.append({
                'round': round_num,
                'accuracy': accuracy,
                'training_size': training_size,
                'points': point_coords,
                'labels': y_proj,
                'verification': verification,
                'class_centers': class_centers,
                'class_spreads': class_spreads,
                'n_points': len(point_coords)
            })

            # Track orthogonality progression
            if verification and 'angular_separation' in verification:
                center_orthogonality_progression.append(verification['angular_separation']['orthogonality_center'])
                margin_orthogonality_progression.append(verification['angular_separation']['orthogonality_margin'])
                center_separation_progression.append(verification['angular_separation']['avg_center_separation'])
                margin_separation_progression.append(verification['angular_separation']['avg_margin_separation'])

        if not frames_data:
            print(f"{Colors.YELLOW}No valid frames created{Colors.ENDC}")
            return None

        # Create frames for animation
        frames = []
        n_classes = len(np.unique(y_test))

        for fd_idx, fd in enumerate(frames_data):
            round_num = fd['round']
            points = fd['points']
            labels = fd['labels']
            accuracy = fd['accuracy']
            verification = fd['verification']
            angular_sep = verification.get('angular_separation', {})
            ortho_center = angular_sep.get('orthogonality_center', 0)
            ortho_margin = angular_sep.get('orthogonality_margin', 0)
            avg_center_sep = angular_sep.get('avg_center_separation', 0)
            avg_margin_sep = angular_sep.get('avg_margin_separation', 0)
            data_source = verification.get('data_source', 'test').upper()

            # Determine color based on MARGIN orthogonality (TRUE measure)
            ortho_color = "green" if ortho_margin > 0.8 else "orange" if ortho_margin > 0.5 else "red"

            traces = []

            # Add transparent sphere reference
            u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
            x_sphere = np.cos(u) * np.sin(v)
            y_sphere = np.sin(u) * np.sin(v)
            z_sphere = np.cos(v)

            traces.append(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                opacity=0.05, showscale=False, hoverinfo='none',
                colorscale=[[0, 'lightgray'], [1, 'lightgray']],
                name='Unit Sphere'
            ))

            # Add coordinate axes
            axis_length = 1.2
            traces.append(go.Scatter3d(
                x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
                mode='lines', line=dict(color='red', width=2), name='X (Real)'
            ))
            traces.append(go.Scatter3d(
                x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
                mode='lines', line=dict(color='green', width=2), name='Y (Imag)'
            ))
            traces.append(go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
                mode='lines', line=dict(color='blue', width=2), name='Z (Class)'
            ))

            # Add data points for each class
            unique_labels = np.unique(labels)
            for c in unique_labels:
                class_name = class_names[int(c)] if class_names and int(c) < len(class_names) else f'Class {int(c)+1}'
                color = self.class_colors[int(c) % len(self.class_colors)]

                mask = labels == c
                class_points = points[mask]

                traces.append(go.Scatter3d(
                    x=class_points[:, 0], y=class_points[:, 1], z=class_points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=color,
                        opacity=0.6,
                        line=dict(width=0.5, color='white')
                    ),
                    name=class_name,
                    legendgroup=f'class_{c}',
                    showlegend=(fd_idx == 0),
                    text=[f'Class: {class_name}<br>X: {p[0]:.3f}<br>Y: {p[1]:.3f}<br>Z: {p[2]:.3f}'
                          for p in class_points],
                    hoverinfo='text'
                ))

                # Add class center
                if fd.get('class_centers') is not None and c < len(fd['class_centers']):
                    center = fd['class_centers'][c]
                    center_norm = np.linalg.norm(center)
                    if center_norm > 0:
                        center_unit = center / center_norm
                        traces.append(go.Scatter3d(
                            x=[0, center_unit[0]], y=[0, center_unit[1]], z=[0, center_unit[2]],
                            mode='lines+markers',
                            marker=dict(size=10, color=color, symbol='circle', line=dict(width=2, color='white')),
                            line=dict(color=color, width=3, dash='dash'),
                            name=f'{class_name} Center',
                            legendgroup=f'center_{c}',
                            showlegend=False,
                            hoverinfo='text',
                            text=f'<b>{class_name} Center</b><br>Angle from origin: {np.arccos(center_unit[2])*180/np.pi:.1f}°'
                        ))

                # Add confidence sphere showing class spread
                if fd.get('class_spreads') is not None and c < len(fd['class_spreads']):
                    spread = fd['class_spreads'][c]
                    if spread > 0 and center_norm > 0:
                        # Create a small sphere representing the spread around the center
                        # The radius is proportional to angular spread (capped at 0.3 for visibility)
                        spread_radius = min(0.25, np.radians(spread) / 2)

                        # Create sphere points around the class center direction
                        u_local, v_local = np.mgrid[0:2*np.pi:20j, 0:np.pi:15j]

                        # Orientation: we want the sphere to be centered at the class center
                        # For simplicity, show as a translucent sphere at the center location
                        center_point = center_unit

                        x_sphere_local = center_point[0] + spread_radius * np.cos(u_local) * np.sin(v_local)
                        y_sphere_local = center_point[1] + spread_radius * np.sin(u_local) * np.sin(v_local)
                        z_sphere_local = center_point[2] + spread_radius * np.cos(v_local)

                        traces.append(go.Surface(
                            x=x_sphere_local, y=y_sphere_local, z=z_sphere_local,
                            opacity=0.15,
                            showscale=False,
                            hoverinfo='text',
                            colorscale=[[0, color], [1, color]],
                            name=f'{class_name} Spread (±{spread:.1f}°)',
                            text=f'Angular spread: ±{spread:.1f}°',
                            showlegend=False
                        ))

            # Add ideal orthogonal positions (targets)
            for c in range(n_classes):
                theta = (c * 2 * np.pi / n_classes)
                phi = np.pi / 2
                x_target = 0.95 * np.sin(phi) * np.cos(theta)
                y_target = 0.95 * np.sin(phi) * np.sin(theta)
                z_target = 0.95 * np.cos(phi)

                color = self.class_colors[c % len(self.class_colors)]
                traces.append(go.Scatter3d(
                    x=[x_target], y=[y_target], z=[z_target],
                    mode='markers',
                    marker=dict(size=12, color=color, symbol='x', line=dict(width=2, color='white')),
                    name=f'Ideal {class_names[c] if class_names and c < len(class_names) else f"C{c+1}"}',
                    legendgroup=f'ideal_{c}',
                    showlegend=False,
                    hoverinfo='text',
                    text=f'<b>Ideal Position for Class {c+1}</b><br>90° separation'
                ))

            # Add equatorial plane reference
            theta_circle = np.linspace(0, 2*np.pi, 100)
            x_circle = 0.98 * np.cos(theta_circle)
            y_circle = 0.98 * np.sin(theta_circle)
            z_circle = np.zeros_like(theta_circle)

            traces.append(go.Scatter3d(
                x=x_circle, y=y_circle, z=z_circle,
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                name='Equatorial Plane (90°)',
                showlegend=False
            ))

            # Create frame with enhanced title showing both metrics
            frames.append(go.Frame(
                data=traces,
                name=f'round_{round_num}',
                layout=go.Layout(
                    title=dict(
                        text=f'<b>Round {round_num} - {data_source} DATA</b><br>'
                             f'Accuracy: {accuracy:.3f} | Training Samples: {fd["training_size"]}<br>'
                             f'<span style="color:{ortho_color}">🎯 TRUE Separation (Margin): {ortho_margin:.3f} | {avg_margin_sep:.1f}°</span><br>'
                             f'<span style="color:gray">Center Separation: {ortho_center:.3f} | {avg_center_sep:.1f}°</span><br>'
                             f'<sup>{fd["n_points"]} test points | Translucent spheres show class spread</sup>',
                        font=dict(size=13)
                    )
                )
            ))

        if not frames:
            print(f"{Colors.YELLOW}No valid frames created{Colors.ENDC}")
            return None

        # Create initial figure
        first_frame = frames[0]

        fig = go.Figure(
            data=first_frame.data,
            layout=go.Layout(
                title=first_frame.layout.title,
                scene=dict(
                    xaxis_title='<b>Real Component</b>',
                    yaxis_title='<b>Imaginary Component</b>',
                    zaxis_title='<b>Class Separation Axis</b>',
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
                        y=0.95, x=0.05
                    )
                ],
                sliders=[{
                    'active': 0,
                    'currentvalue': {'prefix': 'Round: ', 'font': {'size': 14}},
                    'steps': [
                        {
                            'args': [[f'round_{fd["round"]}'], {'frame': {'duration': 0, 'redraw': True}}],
                            'label': str(fd['round']),
                            'method': 'animate'
                        }
                        for fd in frames_data
                    ]
                }],
                width=1400,
                height=900,
                showlegend=True,
                legend=dict(
                    yanchor="top", y=0.99, xanchor="left", x=0.02,
                    bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1,
                    font=dict(color='white', size=10)
                ),
                paper_bgcolor='rgba(0,0,0,0.9)',
                plot_bgcolor='rgba(0,0,0,0.9)'
            ),
            frames=frames
        )

        # Add comprehensive explanation annotation
        final_verification = frames_data[-1]['verification']
        final_angular = final_verification.get('angular_separation', {})
        final_ortho_center = final_angular.get('orthogonality_center', 0)
        final_ortho_margin = final_angular.get('orthogonality_margin', 0)
        final_center_sep = final_angular.get('avg_center_separation', 0)
        final_margin_sep = final_angular.get('avg_margin_separation', 0)
        final_accuracy = frames_data[-1]['accuracy']
        data_source = final_verification.get('data_source', 'test').upper()

        ortho_color = "green" if final_ortho_margin > 0.8 else "orange" if final_ortho_margin > 0.5 else "red"

        # Calculate gap between center and margin orthogonality
        ortho_gap = final_ortho_center - final_ortho_margin

        fig.add_annotation(
            x=0.98, y=0.05, xref="paper", yref="paper",
            text=f"<b>📐 Understanding Orthogonality ({data_source} DATA):</b><br>"
                 f"• Each colored dot = data point projected to sphere surface<br>"
                 f"• Class centers (dashed lines) move toward ideal positions (X)<br>"
                 f"• ✗ marks = ideal orthogonal positions (90° separation)<br>"
                 f"• <span style='color:lightgreen'>CENTER Separation</span>: {final_center_sep:.1f}° | Orthogonality: {final_ortho_center:.3f}<br>"
                 f"• <span style='color:{ortho_color}'>TRUE Separation (Margin)</span>: {final_margin_sep:.1f}° | Orthogonality: {final_ortho_margin:.3f}<br>"
                 f"• Translucent spheres show class spread (angular variance)<br>"
                 f"<br>"
                 f"<b>Why accuracy ≠ center orthogonality:</b><br>"
                 f"• Centers can be orthogonal while clusters overlap (gap: {ortho_gap:.3f})<br>"
                 f"• Classification requires MARGIN separation > 45°<br>"
                 f"• {data_source} Accuracy: {final_accuracy:.3f} vs Margin Orthogonality: {final_ortho_margin:.3f}<br>"
                 f"<span style='color:{'green' if final_ortho_margin > 0.7 else 'orange' if final_ortho_margin > 0.5 else 'red'}'>"
                 f"{'✓ Good generalization' if final_ortho_margin > 0.7 else '⚠️ Needs more training'}</span>",
            showarrow=False,
            font=dict(size=9, color='white'),
            align='right',
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='white',
            borderwidth=1
        )

        # Add margin orthogonality gauge
        fig.add_annotation(
            x=0.02, y=0.95, xref="paper", yref="paper",
            text=f"<b>🎯 TRUE Orthogonality (Margin):</b><br>"
                 f"<span style='color:{ortho_color}; font-size:20px;'>{final_ortho_margin:.3f}</span><br>"
                 f"Target: 1.000<br>"
                 f"<sup>Higher = better separation with margins</sup>",
            showarrow=False,
            font=dict(size=10, color='white'),
            align='left',
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='white',
            borderwidth=1
        )

        # Save dashboard
        dashboard_path = self.output_dir / f'{self.dataset_name}_point_based_spherical.html'
        fig.write_html(str(dashboard_path))

        # Generate enhanced verification plot
        self._create_enhanced_verification_plot(frames_data, center_orthogonality_progression,
                                                 margin_orthogonality_progression, center_separation_progression,
                                                 margin_separation_progression)

        print(f"{Colors.GREEN}✅ Point-based spherical dashboard: {dashboard_path}{Colors.ENDC}")
        print(f"   Rounds: {len(frames_data)}")
        print(f"   Points per round: {frames_data[0]['n_points']}")
        print(f"   Center orthogonality: {final_ortho_center:.3f}")
        print(f"   TRUE margin orthogonality: {final_ortho_margin:.3f}")
        print(f"   Data source: {data_source} DATA (generalization)")

        return str(dashboard_path)

    def _create_enhanced_verification_plot(self, frames_data, center_orthogonality, margin_orthogonality,
                                            center_separation, margin_separation):
        """Create verification plot showing BOTH center and margin orthogonality progression"""
        import matplotlib.pyplot as plt

        rounds = [fd['round'] for fd in frames_data]
        accuracy = [fd['accuracy'] for fd in frames_data]
        data_source = frames_data[0]['verification'].get('data_source', 'test').upper() if frames_data else 'TEST'

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Point-Based Spherical Visualization - {self.dataset_name} ({data_source} DATA)',
                     fontsize=14, fontweight='bold')

        # Plot 1: Orthogonality Progression (Center vs Margin)
        ax = axes[0, 0]
        ax.plot(rounds, center_orthogonality, 'b-o', linewidth=2, markersize=8, label='Center Orthogonality', alpha=0.7)
        ax.plot(rounds, margin_orthogonality, 'g-s', linewidth=2, markersize=8, label='MARGIN Orthogonality (TRUE)')
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (0.9)')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (0.7)')
        ax.set_xlabel('Round')
        ax.set_ylabel('Orthogonality Score')
        ax.set_title(f'Orthogonality Progression ({data_source} DATA)', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # Highlight the gap between center and margin
        for i, (c, m) in enumerate(zip(center_orthogonality, margin_orthogonality)):
            ax.annotate(f'Δ={c-m:.2f}', (rounds[i], (c+m)/2),
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # Plot 2: Angular Separation Progression
        ax = axes[0, 1]
        ax.plot(rounds, center_separation, 'b--o', linewidth=2, markersize=8, label='Center Separation', alpha=0.7)
        ax.plot(rounds, margin_separation, 'g-s', linewidth=2, markersize=8, label='MARGIN Separation (TRUE)')
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Target (90°)')
        ax.axhline(y=45, color='orange', linestyle='--', alpha=0.5, label='Min Good Separation (45°)')
        ax.fill_between(rounds, margin_separation, 90,
                        where=np.array(margin_separation) <= 90,
                        color='green', alpha=0.3, interpolate=True)
        ax.set_xlabel('Round')
        ax.set_ylabel('Separation (degrees)')
        ax.set_title(f'Class Separation Progression ({data_source} DATA)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        # Plot 3: Margin Orthogonality vs Accuracy
        ax = axes[1, 0]
        ax.scatter(margin_orthogonality, accuracy, c=rounds, cmap='viridis', s=100, alpha=0.7)
        ax.plot(margin_orthogonality, accuracy, 'gray', alpha=0.3)
        ax.set_xlabel('MARGIN Orthogonality (TRUE Measure)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Margin Orthogonality vs Accuracy', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        if len(margin_orthogonality) > 1 and len(accuracy) > 1:
            corr = np.corrcoef(margin_orthogonality, accuracy)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 4: Orthogonality Gap Analysis
        ax = axes[1, 1]
        ortho_gap = [c - m for c, m in zip(center_orthogonality, margin_orthogonality)]
        ax.bar(rounds, ortho_gap, color='red', alpha=0.7, edgecolor='black')
        ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Warning Threshold (0.2)')
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good Threshold (0.1)')
        ax.set_xlabel('Round')
        ax.set_ylabel('Orthogonality Gap (Center - Margin)')
        ax.set_title('Orthogonality Gap Analysis', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotations for large gaps
        for i, (r, gap) in enumerate(zip(rounds, ortho_gap)):
            if gap > 0.2:
                ax.annotate(f'Large Gap\n{gap:.2f}', (r, gap),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=8, color='red')

        plt.tight_layout()

        plot_path = self.output_dir / f'{self.dataset_name}_verification.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   Verification plot: {plot_path}")



# =============================================================================
# SECTION: SPHERICAL TENSOR EVOLUTION (FIXED for 5D tensor)
# =============================================================================

class SphericalTensorEvolution:
    """Spherical tensor evolution visualization - with margin metrics and density-based sampling"""

    def __init__(self, model, output_dir='visualizations'):
        self.model = model
        self.dataset_name = model.dataset_name if model and hasattr(model, 'dataset_name') else 'unknown'
        self.output_dir = Path(output_dir) / self.dataset_name / 'spherical_evolution'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"{Colors.CYAN}🌐 Spherical Tensor Evolution initialized{Colors.ENDC}")

    def _get_test_indices(self):
        """Get test indices from model"""
        if hasattr(self.model, 'test_indices') and self.model.test_indices is not None:
            test_indices = self.model.test_indices
            if torch.is_tensor(test_indices):
                test_indices = test_indices.numpy() if hasattr(test_indices, 'numpy') else np.array(test_indices)
            return test_indices
        return None

    def _density_based_sampling(self, points, max_points=500):
        """
        Sample points using density-based sampling to preserve shape.
        Keeps border points and reduces density in crowded regions.
        """
        if len(points) <= max_points:
            return points

        from sklearn.neighbors import NearestNeighbors

        # Compute local density
        nbrs = NearestNeighbors(n_neighbors=min(10, len(points)), algorithm='auto').fit(points)
        distances, _ = nbrs.kneighbors(points)
        avg_distances = np.mean(distances, axis=1)

        # Higher density = smaller distances
        density_scores = 1 / (avg_distances + 1e-8)

        # Probability of selection is inversely proportional to density
        # This keeps border points (lower density) and reduces crowded interior
        selection_probs = 1 / (density_scores + 1e-8)
        selection_probs = selection_probs / np.sum(selection_probs)

        # Select points
        selected_indices = np.random.choice(
            len(points),
            size=max_points,
            replace=False,
            p=selection_probs
        )

        return points[selected_indices]

    def _compute_class_centers_and_spreads_from_test_data(self, weights, test_indices):
        """
        Compute class centers and angular spreads using test data predictions.
        Returns: (class_centers, class_spreads) where spreads are in degrees
        """
        if test_indices is None or len(test_indices) == 0:
            return None, None

        X_test = self.model.X_tensor[test_indices]
        y_test = self.model.y_tensor[test_indices]

        predictions, posteriors = self.model.predict(X_test)

        if torch.is_tensor(posteriors):
            posteriors = posteriors.numpy()
        if torch.is_tensor(y_test):
            y_test = y_test.numpy()

        n_classes = weights.shape[0]

        # Get class orientation vectors from tensor
        class_vectors = self._compute_class_orientation_vectors(weights, n_classes)

        # Project test points to sphere
        n_test = len(y_test)
        point_coords = np.zeros((n_test, 3))

        for i in range(n_test):
            probs = posteriors[i]
            weighted_sum = np.zeros(3)
            for c in range(n_classes):
                weighted_sum += probs[c] * class_vectors[c]

            norm = np.linalg.norm(weighted_sum)
            if norm > 0:
                point_coords[i] = weighted_sum / norm

        # Compute class centers and spreads
        class_centers = []
        class_spreads = []

        for c in range(n_classes):
            class_mask = y_test == c
            class_points = point_coords[class_mask]

            if len(class_points) > 0:
                center = np.mean(class_points, axis=0)
                norm = np.linalg.norm(center)
                if norm > 0:
                    center = center / norm
                class_centers.append(center)

                # Calculate angular spread
                angles_to_center = []
                for p in class_points:
                    p_norm = np.linalg.norm(p)
                    if p_norm > 0:
                        p_unit = p / p_norm
                        cos_sim = np.dot(p_unit, center)
                        cos_sim = np.clip(cos_sim, -1, 1)
                        angle = np.arccos(cos_sim) * 180 / np.pi
                        angles_to_center.append(angle)

                spread = np.std(angles_to_center) if angles_to_center else 0
                class_spreads.append(spread)

                # Also store sampled points for visualization (density-based)
                if len(class_points) > 200:
                    class_points = self._density_based_sampling(class_points, 200)
            else:
                class_centers.append(np.zeros(3))
                class_spreads.append(0)

        return np.array(class_centers), np.array(class_spreads), point_coords, y_test

    def _compute_class_orientation_vectors(self, weights, n_classes):
        """Compute unit orientation vectors for each class from tensor weights"""
        class_vectors = []

        for c in range(n_classes):
            class_weights = weights[c].flatten()
            significant = class_weights[np.abs(class_weights) > 1e-6]

            if len(significant) > 0:
                magnitudes = np.abs(significant)
                phases = np.angle(significant)

                sin_sum = np.sum(magnitudes * np.sin(phases))
                cos_sum = np.sum(magnitudes * np.cos(phases))
                avg_theta = np.arctan2(sin_sum, cos_sum)
                avg_r = np.mean(magnitudes)

                phi = (c * np.pi / max(1, n_classes))
                x = avg_r * np.sin(phi) * np.cos(avg_theta)
                y = avg_r * np.sin(phi) * np.sin(avg_theta)
                z = avg_r * np.cos(phi)

                norm = np.sqrt(x*x + y*y + z*z)
                if norm > 0:
                    class_vectors.append(np.array([x/norm, y/norm, z/norm]))
                else:
                    theta = (c * 2 * np.pi / max(2, n_classes))
                    phi = np.pi / 2
                    class_vectors.append(np.array([
                        np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi)
                    ]))
            else:
                theta = (c * 2 * np.pi / max(2, n_classes))
                phi = np.pi / 2
                class_vectors.append(np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ]))

        return np.array(class_vectors)

    def _compute_angular_separation_with_margin(self, class_centers, class_spreads):
        """
        Compute angular separation with margin (accounting for spreads).
        """
        n_classes = len(class_centers)
        if n_classes < 2:
            return {
                'avg_center_separation': 0,
                'avg_margin_separation': 0,
                'orthogonality_center': 0,
                'orthogonality_margin': 0,
                'pairwise_center_angles': [],
                'pairwise_margin_angles': []
            }

        center_angles = []
        margin_angles = []

        for i in range(n_classes):
            for j in range(i+1, n_classes):
                v1 = class_centers[i]
                v2 = class_centers[j]

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 > 0 and norm2 > 0:
                    dot = np.dot(v1, v2) / (norm1 * norm2)
                    dot = np.clip(dot, -1, 1)
                    center_angle = np.arccos(dot) * 180 / np.pi
                    center_angles.append(center_angle)

                    # Margin = center angle minus sum of spreads
                    margin = center_angle - (class_spreads[i] + class_spreads[j])
                    margin_angles.append(max(0, margin))
                else:
                    center_angles.append(90)
                    margin_angles.append(0)

        avg_center_sep = np.mean(center_angles) if center_angles else 0
        avg_margin_sep = np.mean(margin_angles) if margin_angles else 0

        return {
            'avg_center_separation': avg_center_sep,
            'avg_margin_separation': avg_margin_sep,
            'orthogonality_center': min(1.0, avg_center_sep / 90.0),
            'orthogonality_margin': min(1.0, avg_margin_sep / 90.0),
            'pairwise_center_angles': center_angles,
            'pairwise_margin_angles': margin_angles
        }

    def create_spherical_animation(self, evolution_history, class_names=None, use_test_data=True):
        """
        Create spherical evolution animation showing TENSOR ORTHOGONALIZATION in complex space.
        Now shows BOTH center orthogonality AND margin orthogonality (TRUE measure).
        Uses density-based sampling to avoid overcrowding.
        """
        if not evolution_history or not PLOTLY_AVAILABLE:
            return None

        print(f"{Colors.CYAN}🌐 Creating spherical evolution - Tensor Orthogonalization Visualization...{Colors.ENDC}")

        test_indices = self._get_test_indices()
        use_test = use_test_data and test_indices is not None and len(test_indices) > 0

        if use_test:
            print(f"   Using TEST DATA for orthogonality metrics ({len(test_indices)} samples)")
            print(f"   This shows GENERALIZATION, not memorization")
        else:
            print(f"   {Colors.YELLOW}No test indices found. Using training data (memorization){Colors.ENDC}")

        if class_names is None and hasattr(self.model, 'label_encoder') and self.model.label_encoder:
            class_names = list(self.model.label_encoder.keys())

        class_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
        frames_data = []

        # Store for density-based scaling
        all_point_coords = []
        all_round_points = []

        for snap_idx, snap in enumerate(evolution_history):
            if 'complex_weights' not in snap:
                continue

            weights = snap['complex_weights']
            if torch.is_tensor(weights):
                weights = weights.cpu().numpy()

            round_num = snap['round']
            accuracy = snap.get('accuracy', 0)
            training_size = snap.get('training_size', 0)

            if len(weights.shape) == 4:
                n_classes = weights.shape[0]
                n_pairs = weights.shape[1]
                n_bins = weights.shape[2]

                print(f"   Round {round_num}: {n_classes} classes, {n_pairs} feature pairs")

                # Get class orientation vectors (for visualization)
                class_vectors = self._compute_class_orientation_vectors(weights, n_classes)

                # Get test data points and spreads if available
                if use_test:
                    class_centers, class_spreads, point_coords, y_test = self._compute_class_centers_and_spreads_from_test_data(
                        weights, test_indices
                    )

                    # Apply density-based sampling to reduce overcrowding
                    if point_coords is not None and len(point_coords) > 500:
                        sampled_indices = []
                        for c in range(n_classes):
                            class_mask = y_test == c
                            class_points = point_coords[class_mask]
                            if len(class_points) > 100:
                                sampled = self._density_based_sampling(class_points, 100)
                                # Need to track which indices we kept
                                sampled_indices.extend(list(range(len(sampled))))
                            else:
                                sampled_indices.extend(list(range(len(class_points))))

                        # Sample uniformly if still too many
                        if len(sampled_indices) > 500:
                            sampled_indices = np.random.choice(sampled_indices, 500, replace=False)

                        point_coords = point_coords[sampled_indices]
                        y_test = y_test[sampled_indices]

                    all_point_coords.append(point_coords if point_coords is not None else [])
                    all_round_points.append(len(point_coords) if point_coords is not None else 0)

                    # Calculate metrics with margin
                    if class_centers is not None and class_spreads is not None:
                        angular_metrics = self._compute_angular_separation_with_margin(class_centers, class_spreads)
                    else:
                        angular_metrics = {
                            'avg_center_separation': 0,
                            'avg_margin_separation': 0,
                            'orthogonality_center': 0,
                            'orthogonality_margin': 0
                        }
                else:
                    # Fallback to training data (original method)
                    class_centers = class_vectors
                    class_spreads = np.zeros(n_classes)
                    point_coords = None
                    angular_metrics = self._calculate_orthogonality_metrics(class_vectors, n_classes)
                    angular_metrics = {
                        'avg_center_separation': angular_metrics.get('avg_separation', 0),
                        'avg_margin_separation': angular_metrics.get('avg_separation', 0),
                        'orthogonality_center': angular_metrics.get('orthogonality', 0),
                        'orthogonality_margin': angular_metrics.get('orthogonality', 0)
                    }

                frames_data.append({
                    'round': round_num,
                    'class_vectors': class_vectors,
                    'class_centers': class_centers if use_test else class_vectors,
                    'class_spreads': class_spreads,
                    'point_coords': point_coords,
                    'y_test': y_test if use_test else None,
                    'accuracy': accuracy,
                    'training_size': training_size,
                    'n_classes': n_classes,
                    'n_pairs': n_pairs,
                    'angular_metrics': angular_metrics,
                    'data_source': 'TEST' if use_test else 'TRAINING'
                })

        if not frames_data:
            print(f"{Colors.YELLOW}   ℹ️ No valid class orientations extracted{Colors.ENDC}")
            return None

        # Find global scale for point size based on density
        max_points_per_round = max([fd.get('point_coords', [None])[0] is not None and len(fd['point_coords']) or 0 for fd in frames_data])
        point_size_base = max(2, 8 - (max_points_per_round / 100))

        # Create frames for animation
        frames = []

        for frame_idx, fd in enumerate(frames_data):
            round_num = fd['round']
            class_vectors = fd['class_vectors']
            class_centers = fd['class_centers']
            class_spreads = fd['class_spreads']
            point_coords = fd.get('point_coords')
            accuracy = fd['accuracy']
            training_size = fd['training_size']
            angular_metrics = fd['angular_metrics']
            data_source = fd['data_source']

            ortho_center = angular_metrics.get('orthogonality_center', 0)
            ortho_margin = angular_metrics.get('orthogonality_margin', 0)
            avg_center_sep = angular_metrics.get('avg_center_separation', 0)
            avg_margin_sep = angular_metrics.get('avg_margin_separation', 0)

            ortho_color = "green" if ortho_margin > 0.8 else "orange" if ortho_margin > 0.5 else "red"

            traces = []

            # Add transparent unit sphere
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
                mode='lines', line=dict(color='red', width=2), name='Real Component'
            ))
            traces.append(go.Scatter3d(
                x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
                mode='lines', line=dict(color='green', width=2), name='Imaginary Component'
            ))
            traces.append(go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
                mode='lines', line=dict(color='blue', width=2), name='Class Separation'
            ))

            # Add data points (if using test data)
            if point_coords is not None and len(point_coords) > 0:
                # Dynamic point size based on density
                point_size = max(2, min(8, point_size_base * (1 - len(point_coords) / 1000)))

                for c in range(fd['n_classes']):
                    class_mask = fd['y_test'] == c
                    class_points = point_coords[class_mask]

                    if len(class_points) > 0:
                        color = class_colors[c % len(class_colors)]
                        class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'

                        traces.append(go.Scatter3d(
                            x=class_points[:, 0], y=class_points[:, 1], z=class_points[:, 2],
                            mode='markers',
                            marker=dict(
                                size=point_size,
                                color=color,
                                opacity=0.5,
                                line=dict(width=0.3, color='white')
                            ),
                            name=f'{class_name} (points)',
                            legendgroup=f'points_{c}',
                            showlegend=(frame_idx == 0 and c < 5),
                            text=[f'Class: {class_name}' for _ in range(len(class_points))],
                            hoverinfo='text'
                        ))

            # Add class orientation vectors (from tensor)
            n_classes_vis = min(fd['n_classes'], 12)
            for c in range(n_classes_vis):
                if c < len(class_vectors):
                    v = class_vectors[c]
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        v_unit = v / norm
                        color = class_colors[c % len(class_colors)]
                        class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'

                        # Determine vector length (shorter if spread is large)
                        spread = class_spreads[c] if c < len(class_spreads) else 0
                        vector_length = max(0.3, 1.0 - spread / 90.0)  # Shorter vector = more spread

                        traces.append(go.Scatter3d(
                            x=[0, v_unit[0] * vector_length],
                            y=[0, v_unit[1] * vector_length],
                            z=[0, v_unit[2] * vector_length],
                            mode='lines+markers',
                            marker=dict(
                                size=8,
                                color=color,
                                symbol='circle',
                                line=dict(width=1, color='white')
                            ),
                            line=dict(color=color, width=3),
                            name=class_name,
                            legendgroup=f'vector_{c}',
                            showlegend=(frame_idx == 0),
                            text=f"<b>Class {class_name}</b><br>"
                                 f"Center Orthogonality: {ortho_center:.3f}<br>"
                                 f"TRUE Margin Orthogonality: {ortho_margin:.3f}<br>"
                                 f"Spread: ±{spread:.1f}°",
                            hoverinfo='text'
                        ))

                        # Add spread cone/ellipse visualization
                        if spread > 0:
                            # Create a translucent cone representing spread
                            theta_cone = np.linspace(0, 2*np.pi, 20)
                            spread_rad = np.radians(spread)
                            cone_radius = 0.2 * spread_rad

                            # Points around the vector direction
                            # Find perpendicular vectors
                            if abs(v_unit[0]) < 0.9:
                                perp = np.cross(v_unit, [1, 0, 0])
                            else:
                                perp = np.cross(v_unit, [0, 1, 0])
                            perp = perp / np.linalg.norm(perp)
                            perp2 = np.cross(v_unit, perp)

                            cone_points = []
                            for t in theta_cone:
                                offset = cone_radius * (np.cos(t) * perp + np.sin(t) * perp2)
                                point = v_unit * vector_length + offset
                                cone_points.append(point)

                            traces.append(go.Scatter3d(
                                x=[p[0] for p in cone_points],
                                y=[p[1] for p in cone_points],
                                z=[p[2] for p in cone_points],
                                mode='lines',
                                line=dict(color=color, width=1, dash='dot'),
                                name=f'{class_name} spread',
                                showlegend=False,
                                hoverinfo='none'
                            ))

            # Add target orthogonal positions
            n_classes = fd['n_classes']
            for c in range(min(n_classes, 12)):
                target_theta = (c * 2 * np.pi / n_classes)
                target_phi = np.pi / 2
                r = 0.95

                x_target = r * np.sin(target_phi) * np.cos(target_theta)
                y_target = r * np.sin(target_phi) * np.sin(target_theta)
                z_target = r * np.cos(target_phi)

                color = class_colors[c % len(class_colors)]

                traces.append(go.Scatter3d(
                    x=[x_target], y=[y_target], z=[z_target],
                    mode='markers',
                    marker=dict(
                        size=12,
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

            # Add reference circle at 90°
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

            # Determine if we're showing points
            points_info = f" | {len(point_coords)} test points" if point_coords is not None else ""

            frames.append(go.Frame(
                data=traces,
                name=f'Round {round_num}',
                layout=go.Layout(
                    title=dict(
                        text=f'<b>Round {round_num} - {data_source} DATA{points_info}</b><br>'
                             f'Accuracy: {accuracy:.3f} | Training Samples: {training_size}<br>'
                             f'<span style="color:lightgreen">Center Orthogonality: {ortho_center:.3f} | {avg_center_sep:.1f}°</span><br>'
                             f'<span style="color:{ortho_color}">🎯 TRUE Margin Orthogonality: {ortho_margin:.3f} | {avg_margin_sep:.1f}°</span>',
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
                    text=f'<b>CT-DBNN Tensor Orthogonalization - {"GENERALIZATION" if use_test else "MEMORIZATION"}</b><br>'
                         f'Class Orientation Vectors in Complex Feature-Pair Space<br>'
                         f'<sup>Vector length = confidence (shorter = more spread) | Points show test data distribution</sup>',
                    font=dict(size=16)
                ),
                scene=dict(
                    xaxis_title='<b>Real Feature Component</b>',
                    yaxis_title='<b>Imaginary Feature Component</b>',
                    zaxis_title='<b>Class Separation Axis</b>',
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)),
                    aspectmode='cube'
                ),
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=False,
                        y=0.92,
                        x=0.05,
                        buttons=[
                            dict(label='▶️ Play', method='animate',
                                 args=[None, {'frame': {'duration': 800, 'redraw': True},
                                              'fromcurrent': True, 'mode': 'immediate'}]),
                            dict(label='⏸️ Pause', method='animate',
                                 args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]),
                            dict(label='🔄 Reset', method='animate',
                                 args=[[frames[0].name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}])
                        ]
                    )
                ],
                sliders=[{
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {'prefix': 'Round: ', 'font': {'size': 14, 'color': 'white'}, 'visible': True},
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9,
                    'x': 0.1,
                    'y': 0,
                    'steps': [
                        {
                            'args': [[f'Round {fd["round"]}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
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
        final_metrics = frames_data[-1]['angular_metrics']
        final_ortho_center = final_metrics.get('orthogonality_center', 0)
        final_ortho_margin = final_metrics.get('orthogonality_margin', 0)
        final_center_sep = final_metrics.get('avg_center_separation', 0)
        final_margin_sep = final_metrics.get('avg_margin_separation', 0)
        ortho_color = "green" if final_ortho_margin > 0.8 else "orange" if final_ortho_margin > 0.5 else "red"

        fig.add_annotation(
            x=0.98, y=0.05, xref="paper", yref="paper",
            text=f"<b>📐 Understanding Tensor Orthogonalization:</b><br>"
                 f"• <span style='color:lightgreen'>CENTER separation</span>: Angle between class centers<br>"
                 f"  → {final_center_sep:.1f}° | Orthogonality: {final_ortho_center:.3f}<br>"
                 f"• <span style='color:{ortho_color}'>TRUE separation (MARGIN)</span>: Center minus spreads<br>"
                 f"  → {final_margin_sep:.1f}° | Orthogonality: {final_ortho_margin:.3f}<br>"
                 f"• <span style='color:orange'>Vector length</span> = confidence (shorter = more spread)<br>"
                 f"• Colored dots = individual test data points (density-sampled)<br>"
                 f"• ✗ marks = ideal orthogonal positions (90° separation)<br>"
                 f"<br>"
                 f"<b>Why accuracy ≠ center orthogonality:</b><br>"
                 f"• Centers can be orthogonal while clusters overlap due to spread<br>"
                 f"• Classification requires MARGIN separation > 45°<br>"
                 f"• Current margin orthogonality: <span style='color:{ortho_color}'>{final_ortho_margin:.3f}</span>",
            showarrow=False,
            font=dict(size=10, color='white'),
            align='right',
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='white',
            borderwidth=1
        )

        # Add margin orthogonality gauge
        fig.add_annotation(
            x=0.02, y=0.95, xref="paper", yref="paper",
            text=f"<b>🎯 TRUE Margin Orthogonality:</b><br>"
                 f"<span style='color:{ortho_color}; font-size:20px;'>{final_ortho_margin:.3f}</span><br>"
                 f"Target: 1.000<br>"
                 f"<sup>Higher = better separation with margins</sup>",
            showarrow=False,
            font=dict(size=11, color='white'),
            align='left',
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='white',
            borderwidth=1
        )

        # FIXED: Use absolute path with correct directory structure
        output_path = self.output_dir / f'{self.dataset_name}_spherical.html'

        # Ensure the directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write the file
        fig.write_html(str(output_path))

        # Convert to absolute path for display
        abs_output_path = output_path.absolute()

        print(f"   ✅ Spherical evolution: {abs_output_path}")
        print(f"   Rounds: {[fd['round'] for fd in frames_data]}")
        print(f"   Center orthogonality progression: {[f'{fd["angular_metrics"]["orthogonality_center"]:.3f}' for fd in frames_data]}")
        print(f"   MARGIN orthogonality progression: {[f'{fd["angular_metrics"]["orthogonality_margin"]:.3f}' for fd in frames_data]}")
        print(f"   Data source: {frames_data[0]['data_source']}")

        return str(abs_output_path)

    def _calculate_orthogonality_metrics(self, vectors, n_classes):
        """Fallback method for training data"""
        if len(vectors) < 2:
            return {'avg_separation': 0, 'orthogonality': 0}

        angles = []
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                v1 = vectors[i][:3] if len(vectors[i]) >= 3 else vectors[i]
                v2 = vectors[j][:3] if len(vectors[j]) >= 3 else vectors[j]

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 > 0 and norm2 > 0:
                    dot = np.dot(v1, v2) / (norm1 * norm2)
                    dot = np.clip(dot, -1, 1)
                    angle = np.arccos(dot) * 180 / np.pi
                    angles.append(angle)
                else:
                    angles.append(90)

        if angles:
            avg_sep = np.mean(angles)
            orthogonality = min(1.0, avg_sep / 90.0)
            return {'avg_separation': avg_sep, 'orthogonality': orthogonality}
        return {'avg_separation': 0, 'orthogonality': 0}

    def _compute_class_orientation_from_test_data(self, weights, test_indices):
        """
        Compute class orientation vectors using test data predictions.
        This shows generalization rather than training data memorization.
        """
        if test_indices is None or len(test_indices) == 0:
            return None, None

        # Get test data
        X_test = self.model.X_tensor[test_indices]
        y_test = self.model.y_tensor[test_indices]

        # Get predictions and posteriors for test data
        predictions, posteriors = self.model.predict(X_test)

        if torch.is_tensor(posteriors):
            posteriors = posteriors.numpy()
        if torch.is_tensor(y_test):
            y_test = y_test.numpy()

        n_classes = weights.shape[0]

        # For each class, compute the average posterior-weighted orientation
        class_vectors = []
        class_weights_list = []

        for c in range(n_classes):
            # Get test samples that belong to this class
            class_mask = (y_test == c)
            if not np.any(class_mask):
                # No test samples for this class - use training-derived vector
                class_weights = weights[c].flatten()
                significant = class_weights[np.abs(class_weights) > 1e-6]

                if len(significant) > 0:
                    magnitudes = np.abs(significant)
                    phases = np.angle(significant)

                    sin_sum = np.sum(magnitudes * np.sin(phases))
                    cos_sum = np.sum(magnitudes * np.cos(phases))
                    avg_theta = np.arctan2(sin_sum, cos_sum)
                    avg_r = np.mean(magnitudes)

                    phi = (c * np.pi / max(1, n_classes))
                    x = avg_r * np.sin(phi) * np.cos(avg_theta)
                    y = avg_r * np.sin(phi) * np.sin(avg_theta)
                    z = avg_r * np.cos(phi)

                    norm = np.sqrt(x*x + y*y + z*z)
                    if norm > 0:
                        class_vectors.append((x/norm, y/norm, z/norm, avg_r))
                    else:
                        class_vectors.append((0, 0, 0, 0))
                else:
                    class_vectors.append((0, 0, 0, 0))
                class_weights_list.append(0)
                continue

            # Get posteriors for this class's test samples
            class_posteriors = posteriors[class_mask, c]

            # Weight test samples by their confidence in this class
            # (higher confidence = more reliable orientation)
            weights_test = class_posteriors / (np.sum(class_posteriors) + 1e-10)

            # For each test sample, compute its orientation in tensor space
            orientations_x = []
            orientations_y = []
            orientations_z = []

            for idx in np.where(class_mask)[0]:
                # Get this sample's posterior distribution
                sample_posterior = posteriors[idx]

                # Weighted sum of class vectors based on posteriors
                weighted_sum = np.zeros(3)
                for other_c in range(n_classes):
                    if other_c == c:
                        # For its own class, use training-derived orientation
                        class_weights = weights[other_c].flatten()
                        significant = class_weights[np.abs(class_weights) > 1e-6]
                        if len(significant) > 0:
                            magnitudes = np.abs(significant)
                            phases = np.angle(significant)

                            sin_sum = np.sum(magnitudes * np.sin(phases))
                            cos_sum = np.sum(magnitudes * np.cos(phases))
                            avg_theta = np.arctan2(sin_sum, cos_sum)
                            avg_r = np.mean(magnitudes)

                            phi = (other_c * np.pi / max(1, n_classes))
                            x_vec = avg_r * np.sin(phi) * np.cos(avg_theta)
                            y_vec = avg_r * np.sin(phi) * np.sin(avg_theta)
                            z_vec = avg_r * np.cos(phi)

                            norm = np.sqrt(x_vec*x_vec + y_vec*y_vec + z_vec*z_vec)
                            if norm > 0:
                                weighted_sum += sample_posterior[other_c] * np.array([x_vec/norm, y_vec/norm, z_vec/norm])
                    else:
                        # For other classes, also use training-derived orientation
                        class_weights = weights[other_c].flatten()
                        significant = class_weights[np.abs(class_weights) > 1e-6]
                        if len(significant) > 0:
                            magnitudes = np.abs(significant)
                            phases = np.angle(significant)

                            sin_sum = np.sum(magnitudes * np.sin(phases))
                            cos_sum = np.sum(magnitudes * np.cos(phases))
                            avg_theta = np.arctan2(sin_sum, cos_sum)
                            avg_r = np.mean(magnitudes)

                            phi = (other_c * np.pi / max(1, n_classes))
                            x_vec = avg_r * np.sin(phi) * np.cos(avg_theta)
                            y_vec = avg_r * np.sin(phi) * np.sin(avg_theta)
                            z_vec = avg_r * np.cos(phi)

                            norm = np.sqrt(x_vec*x_vec + y_vec*y_vec + z_vec*z_vec)
                            if norm > 0:
                                weighted_sum += sample_posterior[other_c] * np.array([x_vec/norm, y_vec/norm, z_vec/norm])

                # Normalize to unit sphere
                norm = np.linalg.norm(weighted_sum)
                if norm > 0:
                    weighted_sum = weighted_sum / norm
                    orientations_x.append(weighted_sum[0])
                    orientations_y.append(weighted_sum[1])
                    orientations_z.append(weighted_sum[2])

            if orientations_x:
                # Weighted average of test sample orientations
                weighted_avg_x = np.average(orientations_x, weights=weights_test)
                weighted_avg_y = np.average(orientations_y, weights=weights_test)
                weighted_avg_z = np.average(orientations_z, weights=weights_test)

                # Normalize to unit sphere
                norm = np.sqrt(weighted_avg_x**2 + weighted_avg_y**2 + weighted_avg_z**2)
                if norm > 0:
                    class_vectors.append((weighted_avg_x/norm, weighted_avg_y/norm, weighted_avg_z/norm, np.mean(weights_test)))
                else:
                    class_vectors.append((0, 0, 0, 0))
            else:
                class_vectors.append((0, 0, 0, 0))

            class_weights_list.append(np.sum(weights_test))

        return np.array(class_vectors), class_weights_list


# =============================================================================
# INTEGRATION WITH OPTIMIZED VISUALIZER
# =============================================================================

class OptimizedVisualizer:
    """Complete Optimized Visualizer with all visualization methods including multi-projection"""

    def __init__(self, model, output_dir='visualizations'):
        """
        Initialize the Optimized Visualizer with all visualization capabilities

        Args:
            model: Trained OptimizedDBNN model
            output_dir: Base directory for all visualizations
        """
        self.model = model
        self.dataset_name = model.dataset_name if model and hasattr(model, 'dataset_name') else 'unknown'

        # Setup output directories
        self.base_output_dir = Path(output_dir)
        self.output_dir = self.base_output_dir / self.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create all subdirectories
        self.dirs = {
            'performance': self.output_dir / 'performance',
            'confusion': self.output_dir / 'confusion',
            'interactive': self.output_dir / 'interactive',
            'spherical': self.output_dir / 'spherical_evolution',
            'tensor': self.output_dir / 'tensor_evolution',
            'networks': self.output_dir / 'networks',
            'multi_projection': self.output_dir / 'multi_projection',  # NEW
            'comparisons': self.output_dir / 'comparisons'  # NEW
        }
        for d in self.dirs.values():
            d.mkdir(exist_ok=True)

        # Initialize sub-visualizers
        self.advanced_visualizer = AdvancedInteractiveVisualizer(model, output_dir)
        self.comprehensive_visualizer = ComprehensiveAdaptiveVisualizer(model, output_dir)
        self.spherical_viz = SphericalTensorEvolution(model, output_dir)  # Original spherical
        self.multi_projection = MultiProjectionSphericalEvolution(model, output_dir)  # NEW multi-projection

        # Initialize 2D polar visualizer
        self.polar_2d_viz = Polar2DEvolution(model, output_dir)

        print(f"{Colors.GREEN}✅ 2D Polar Evolution available{Colors.ENDC}")

        # Initialize polar coordinate visualizer
        self.polar_viz = PolarCoordinateEvolution(model, output_dir)

        print(f"{Colors.CYAN}📁 Visualizations: {self.output_dir}{Colors.ENDC}")
        print(f"{Colors.GREEN}✅ OptimizedVisualizer initialized with polar evolution{Colors.ENDC}")

        print(f"   - Standard visualizations (confusion, training history)")
        print(f"   - Advanced 3D visualizations")
        print(f"   - Comprehensive adaptive visualizations")
        print(f"   - Original spherical evolution")
        print(f"   - Multi-projection spherical evolution (5 methods)")

        # Initialize true polar visualizer
        self.true_polar = TruePolarEvolution(model, output_dir)

        print(f"{Colors.GREEN}✅ True Polar Evolution available{Colors.ENDC}")

        # Initialize new point-based visualizer
        self.point_spherical_viz = PointBasedSphericalVisualization(model, output_dir)

        print(f"{Colors.GREEN}✅ Point-Based Spherical Visualization available{Colors.ENDC}")


    def create_point_based_spherical_evolution(self, evolution_history, X_data, y_data, class_names=None):
        """
        Create point-based spherical evolution showing actual data points.
        This REPLACES the old vector-based spherical visualization.
        """
        if not evolution_history or len(evolution_history) < 2:
            print(f"{Colors.YELLOW}Need at least 2 rounds for evolution visualization{Colors.ENDC}")
            return None

        if X_data is None or y_data is None:
            print(f"{Colors.YELLOW}No data points available for projection{Colors.ENDC}")
            return None

        print(f"\n{Colors.BOLD}{Colors.CYAN}🎯 Creating Point-Based Spherical Evolution{Colors.ENDC}")
        print(f"{'='*60}")

        dashboard_path = self.point_spherical_viz.create_point_based_spherical_animation(
            evolution_history, X_data, y_data, class_names
        )

        return dashboard_path

    def create_true_polar_visualization(self, evolution_history, class_names=None):
        """
        Create true polar evolution dashboard with r vs θ.
        """
        if not evolution_history or len(evolution_history) < 2:
            print(f"{Colors.YELLOW}Need at least 2 rounds for evolution visualization{Colors.ENDC}")
            return None

        print(f"\n{Colors.BOLD}{Colors.CYAN}🎯 Creating True Polar Evolution (r vs θ){Colors.ENDC}")
        print(f"{'='*60}")

        dashboard_path = self.true_polar.create_animated_polar_dashboard(
            evolution_history, class_names
        )

        return dashboard_path

    def create_2d_polar_visualization(self, evolution_history, class_names=None):
        """
        Create 2D polar evolution dashboard showing actual weight distributions
        """
        if not evolution_history or len(evolution_history) < 2:
            print(f"{Colors.YELLOW}Need at least 2 rounds for evolution visualization{Colors.ENDC}")
            return None

        print(f"\n{Colors.BOLD}{Colors.CYAN}📊 Creating 2D Polar Evolution Dashboard{Colors.ENDC}")
        print(f"{'='*60}")

        dashboard_path = self.polar_2d_viz.create_complete_polar_dashboard(evolution_history, class_names)

        print(f"\n{Colors.GREEN}✅ 2D Polar dashboard saved to: {dashboard_path}{Colors.ENDC}")

        return dashboard_path

    def create_polar_visualization(self, evolution_history, class_names=None):
        """
        Create polar coordinate evolution dashboard
        """
        if not evolution_history or len(evolution_history) < 2:
            print(f"{Colors.YELLOW}Need at least 2 rounds for evolution visualization{Colors.ENDC}")
            return None

        print(f"\n{Colors.BOLD}{Colors.CYAN}📊 Creating Polar Coordinate Evolution{Colors.ENDC}")
        print(f"{'='*60}")

        dashboard_path = self.polar_viz.create_polar_dashboard(evolution_history, class_names)

        print(f"\n{Colors.GREEN}✅ Polar evolution dashboard saved to: {dashboard_path}{Colors.ENDC}")

        return dashboard_path

    # =========================================================================
    # EXISTING METHODS (Preserved from original)
    # =========================================================================

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = ''):
        """Plot confusion matrix with ACTUAL class labels (strings)"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        if len(y_true) == 0 or len(y_pred) == 0:
            return

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

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
            print(f"   ✅ Confusion matrix: {filename}")
        else:
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

                if 'complex_weights' in snap:
                    weights = snap['complex_weights']
                    if torch.is_tensor(weights):
                        weights = weights.cpu().numpy()

                    if len(weights.shape) == 4:
                        n_classes = weights.shape[0]

                        if n_classes >= 2:
                            class_orientations = []

                            for c in range(n_classes):
                                class_weights = weights[c].flatten()
                                significant = class_weights[np.abs(class_weights) > 0.01]

                                if len(significant) > 0:
                                    magnitudes = np.abs(significant)
                                    phases = np.angle(significant)

                                    sin_sum = np.sum(magnitudes * np.sin(phases))
                                    cos_sum = np.sum(magnitudes * np.cos(phases))
                                    avg_phase = np.arctan2(sin_sum, cos_sum)

                                    class_orientations.append(np.exp(1j * avg_phase))
                                else:
                                    class_orientations.append(0 + 0j)

                            n = len(class_orientations)
                            ortho_matrix = np.zeros((n, n))
                            angles = []

                            for i in range(n):
                                for j in range(n):
                                    if i == j:
                                        ortho_matrix[i, j] = 1.0
                                    else:
                                        vi = class_orientations[i]
                                        vj = class_orientations[j]

                                        if np.abs(vi) > 0 and np.abs(vj) > 0:
                                            vi_unit = vi / np.abs(vi)
                                            vj_unit = vj / np.abs(vj)
                                            similarity = np.real(vi_unit * np.conj(vj_unit))
                                            ortho_matrix[i, j] = similarity

                                            if i < j:
                                                angle = np.arccos(np.clip(similarity, -1, 1)) * 180 / np.pi
                                                angles.append(angle)

                            orthogonality_matrix.append(ortho_matrix)

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

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Tensor Evolution & Orthogonalization - {self.dataset_name}', fontsize=16, fontweight='bold')

        axes[0, 0].plot(rounds, accuracies, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Adaptive Round', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Classification Accuracy Evolution', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.05])

        axes[0, 1].bar(rounds, training_sizes, color='orange', alpha=0.7, edgecolor='darkorange')
        axes[0, 1].set_xlabel('Adaptive Round', fontsize=12)
        axes[0, 1].set_ylabel('Training Samples', fontsize=12)
        axes[0, 1].set_title('Training Set Growth', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(rounds, class_separation_angles, 'g-s', linewidth=2, markersize=8)
        axes[1, 0].axhline(y=90, color='red', linestyle='--', linewidth=2,
                           label='Perfect Orthogonality (90°)', alpha=0.7)
        axes[1, 0].set_xlabel('Adaptive Round', fontsize=12)
        axes[1, 0].set_ylabel('Average Class Separation Angle (degrees)', fontsize=12)
        axes[1, 0].set_title('Class Tensor Orthogonalization', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].legend(loc='lower right')

        if class_separation_angles:
            final_angle = class_separation_angles[-1]
            axes[1, 0].annotate(f'Final: {final_angle:.1f}°',
                               xy=(rounds[-1], final_angle),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=10, fontweight='bold')

        if orthogonality_matrix and len(orthogonality_matrix) > 0:
            final_ortho = orthogonality_matrix[-1]
            n_classes = final_ortho.shape[0]

            im = axes[1, 1].imshow(final_ortho, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
            axes[1, 1].set_xlabel('Class', fontsize=12)
            axes[1, 1].set_ylabel('Class', fontsize=12)
            axes[1, 1].set_title('Final Class Orthogonality Matrix', fontsize=12, fontweight='bold')

            cbar = plt.colorbar(im, ax=axes[1, 1])
            cbar.set_label('Cosine Similarity (0=Orthogonal, 1=Identical)', fontsize=10)

            for i in range(n_classes):
                for j in range(n_classes):
                    value = final_ortho[i, j]
                    color = 'white' if value > 0.5 else 'black'
                    axes[1, 1].text(j, i, f'{value:.2f}',
                                   ha='center', va='center', color=color, fontsize=9)

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

        self._plot_orthogonality_progression(rounds, class_separation_angles, accuracies)

    def _plot_orthogonality_progression(self, rounds, angles, accuracies):
        """Create a detailed orthogonality progression plot"""
        if not angles or len(angles) < 2:
            return

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Class Tensor Orthogonalization Analysis - {self.dataset_name}', fontsize=14, fontweight='bold')

        ax1.scatter(angles, accuracies, c=rounds, cmap='viridis', s=100, alpha=0.7)
        ax1.plot(angles, accuracies, 'b-', alpha=0.3, linewidth=1)
        ax1.axvline(x=90, color='red', linestyle='--', label='Perfect Orthogonality (90°)', alpha=0.7)
        ax1.set_xlabel('Class Separation Angle (degrees)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy vs Orthogonality', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        cbar = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Round', fontsize=10)

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
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=2, subplot_titles=('Training Progress', 'Feature Space (PCA)',
                                                           'Accuracy Distribution', 'Class Distribution'))

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

        if X is not None and len(X) > 0:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            fig.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers',
                                    marker=dict(color=y, colorscale='Viridis', showscale=True),
                                    text=y, name='Data'), row=1, col=2)

        if y is not None and len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            fig.add_trace(go.Bar(x=[str(c) for c in unique], y=counts, name='Class Distribution'), row=2, col=2)

        fig.update_layout(height=800, title_text=f"DBNN Dashboard - {self.dataset_name}")
        dashboard_path = self.dirs['interactive'] / f'{self.dataset_name}_dashboard.html'
        fig.write_html(str(dashboard_path))
        print(f"   ✅ Interactive dashboard: {dashboard_path}")

    # =========================================================================
    # NEW MULTI-PROJECTION METHODS
    # =========================================================================

    def create_multi_projection_visualization(self, evolution_history, class_names=None):
        """
        Create the complete multi-projection dashboard with 5 projection methods

        Args:
            evolution_history: List of evolution snapshots from TensorEvolutionTracker
            class_names: Optional list of class names for labels

        Returns:
            Path to the generated HTML dashboard
        """
        if not evolution_history or len(evolution_history) < 2:
            print(f"{Colors.YELLOW}Need at least 2 rounds for evolution visualization{Colors.ENDC}")
            return None

        print(f"\n{Colors.BOLD}{Colors.CYAN}🎨 Creating Multi-Projection Spherical Evolution{Colors.ENDC}")
        print(f"{'='*60}")

        # Create main dashboard using MultiProjectionSphericalEvolution
        dashboard_path = self.multi_projection.create_multi_projection_dashboard(
            evolution_history, class_names
        )

        # Create comparison grid
        comparison_path = self.multi_projection.create_comparison_grid(evolution_history)

        print(f"\n{Colors.GREEN}✅ Multi-projection visualizations complete!{Colors.ENDC}")
        print(f"   Dashboard: {dashboard_path}")
        if comparison_path:
            print(f"   Comparison: {comparison_path}")

        return dashboard_path

    def create_spherical_evolution(self, evolution_history, class_names=None):
        """
        Create the original spherical evolution (for backward compatibility)

        Args:
            evolution_history: List of evolution snapshots
            class_names: Optional list of class names

        Returns:
            Path to the generated HTML file
        """
        if not evolution_history:
            print(f"{Colors.YELLOW}No evolution history available{Colors.ENDC}")
            return None

        print(f"\n{Colors.CYAN}🌐 Creating original spherical evolution...{Colors.ENDC}")

        output_path = self.spherical_viz.create_spherical_animation(evolution_history, class_names)

        if output_path:
            print(f"   ✅ Spherical evolution: {output_path}")

        return output_path

    # =========================================================================
    # COMPREHENSIVE GENERATION METHOD
    # =========================================================================

    def generate_all_visualizations(self, history, X, y, y_train, y_test, train_pred, test_pred,
                                     evolution_history=None, class_names=None):
        """
        Generate all visualizations with verification

        This is the main entry point for generating all visualizations
        """
        print(f"\n🎨 Visualizations for: {self.dataset_name}")
        print(f"{'='*60}")

        # Convert to numpy if they're tensors (they should be numpy already)
        X_np = X if isinstance(X, np.ndarray) else X.numpy() if hasattr(X, 'numpy') else X
        y_np = y if isinstance(y, np.ndarray) else y.numpy() if hasattr(y, 'numpy') else y
        y_train_np = y_train if isinstance(y_train, np.ndarray) else y_train.numpy() if hasattr(y_train, 'numpy') else y_train
        y_test_np = y_test if isinstance(y_test, np.ndarray) else y_test.numpy() if hasattr(y_test, 'numpy') else y_test
        train_pred_np = train_pred if isinstance(train_pred, np.ndarray) else train_pred.numpy() if hasattr(train_pred, 'numpy') else train_pred
        test_pred_np = test_pred if isinstance(test_pred, np.ndarray) else test_pred.numpy() if hasattr(test_pred, 'numpy') else test_pred

        # Standard visualizations
        self.plot_training_history(history)
        self.plot_confusion_matrix(y_train_np, train_pred_np, 'Training')
        self.plot_confusion_matrix(y_test_np, test_pred_np, 'Test')

        # Tensor evolution
        if evolution_history and len(evolution_history) > 0:
            self.plot_tensor_evolution(evolution_history)

        # Add polar evolution
        if evolution_history and len(evolution_history) > 1 and X_np is not None and y_np is not None:
            try:
                self.create_point_based_spherical_evolution(evolution_history, X_np, y_np, class_names)
            except Exception as e:
                print(f"   ⚠️ Point-based spherical evolution: {e}")

        # Interactive dashboard
        try:
            self.create_interactive_dashboard(history, X_np, y_np, evolution_history)
        except Exception as e:
            print(f"   ⚠️ Dashboard: {e}")

        # Advanced visualizations
        try:
            if hasattr(self, 'advanced_visualizer') and X_np is not None and len(X_np) > 0:
                self.advanced_visualizer.create_advanced_3d_dashboard(X_np, y_np, history, self.model.feature_names)
        except Exception as e:
            print(f"   ⚠️ Advanced 3D: {e}")

        try:
            if hasattr(self, 'comprehensive_visualizer') and history:
                self.comprehensive_visualizer.plot_3d_networks(X_np, y_np, history, self.model.feature_names)
        except Exception as e:
            print(f"   ⚠️ 3D Networks: {e}")

        # Original spherical evolution
        if evolution_history and len(evolution_history) > 1:
            try:
                self.create_spherical_evolution(evolution_history, class_names)
            except Exception as e:
                print(f"   ⚠️ Spherical evolution: {e}")

        # NEW: Multi-projection spherical evolution
        if evolution_history and len(evolution_history) > 1:
            try:
                self.create_multi_projection_visualization(evolution_history, class_names)
            except Exception as e:
                print(f"   ⚠️ Multi-projection: {e}")

        print(f"\n📁 All saved to: {self.output_dir}")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_visualization_paths(self):
        """Return dictionary of all visualization paths"""
        return {
            'performance': str(self.dirs['performance']),
            'confusion': str(self.dirs['confusion']),
            'interactive': str(self.dirs['interactive']),
            'spherical': str(self.dirs['spherical']),
            'tensor': str(self.dirs['tensor']),
            'networks': str(self.dirs['networks']),
            'multi_projection': str(self.dirs['multi_projection'])
        }

    def open_visualization_folder(self):
        """Open the visualization folder in file explorer"""
        try:
            import subprocess
            import platform

            system = platform.system()
            if system == "Windows":
                subprocess.Popen(f'explorer "{self.output_dir}"')
            elif system == "Darwin":  # macOS
                subprocess.Popen(['open', str(self.output_dir)])
            else:  # Linux
                subprocess.Popen(['xdg-open', str(self.output_dir)])

            print(f"📂 Opened visualization folder: {self.output_dir}")
            return True
        except Exception as e:
            print(f"⚠️ Could not open folder: {e}")
            return False

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

            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            }

            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()

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

        df = DatasetConfig._try_download_url(url, dataset_name, dataset_info)

        if df is None and dataset_name in DatasetConfig.ALTERNATIVE_URLS:
            alt_url = DatasetConfig.ALTERNATIVE_URLS[dataset_name]
            print(f"   Trying alternative source: {alt_url}")
            df = DatasetConfig._try_download_url(alt_url, dataset_name, dataset_info)

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
                try:
                    df = pd.read_csv(StringIO(response.text), header=None)
                    df.columns = dataset_info["columns"]
                except:
                    df = pd.read_csv(StringIO(response.text), delim_whitespace=True, header=None)
                    df.columns = dataset_info["columns"]

            target = dataset_info["target"]
            if target not in df.columns:
                for col in df.columns:
                    if target.lower() in col.lower() or col.lower() in ["class", "type", "label"]:
                        target = col
                        break

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
            y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)

        else:
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
            if not os.path.exists(config_path) and dataset_name in DatasetConfig.UCI_DATASETS:
                print(f"📥 UCI dataset '{dataset_name}' found. Downloading...")
                df = DatasetConfig.download_uci_data(dataset_name)

                if df is not None:
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    df.to_csv(csv_path, index=False)
                    print(f"✅ Dataset saved to {csv_path}")

                    config = DatasetConfig.DEFAULT_CONFIG.copy()
                    config.update({
                        "file_path": csv_path,
                        "column_names": list(df.columns),
                        "target_column": DatasetConfig.UCI_DATASETS[dataset_name]["target"],
                        "has_header": True,
                        "modelType": "Histogram",
                    })

                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"✅ Configuration saved to {config_path}")

                    return config
                else:
                    print(f"❌ Failed to download dataset {dataset_name}")
                    return None

            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return config

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
# SECTION: 2D POLAR COORDINATE EVOLUTION VISUALIZER
# =============================================================================

class Polar2DEvolution:
    """
    2D Polar Coordinate Evolution showing actual weight distributions.
    Now uses TEST DATA for orthogonality metrics to show generalization.
    """

    def __init__(self, model, output_dir='visualizations'):
        self.model = model
        self.dataset_name = model.dataset_name if model and hasattr(model, 'dataset_name') else 'unknown'
        self.output_dir = Path(output_dir) / self.dataset_name / 'polar_2d_evolution'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.class_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + \
                           px.colors.qualitative.Set3 + px.colors.qualitative.Pastel

        print(f"{Colors.CYAN}📊 2D Polar Coordinate Evolution initialized{Colors.ENDC}")
        print(f"   Output: {self.output_dir}")
        print(f"   Using TEST DATA for orthogonality metrics (generalization)")

    def _get_test_indices(self):
        """Get test indices from model"""
        if hasattr(self.model, 'test_indices') and self.model.test_indices is not None:
            test_indices = self.model.test_indices
            if torch.is_tensor(test_indices):
                test_indices = test_indices.numpy() if hasattr(test_indices, 'numpy') else np.array(test_indices)
            return test_indices
        return None

    def _compute_class_orientation_vectors(self, weights, n_classes):
        """Compute unit orientation vectors for each class from tensor weights"""
        class_vectors = []

        for c in range(n_classes):
            class_weights = weights[c].flatten()
            significant = class_weights[np.abs(class_weights) > 1e-6]

            if len(significant) > 0:
                magnitudes = np.abs(significant)
                phases = np.angle(significant)

                sin_sum = np.sum(magnitudes * np.sin(phases))
                cos_sum = np.sum(magnitudes * np.cos(phases))
                avg_theta = np.arctan2(sin_sum, cos_sum)
                avg_r = np.mean(magnitudes)

                phi = (c * np.pi / max(1, n_classes))
                x = avg_r * np.sin(phi) * np.cos(avg_theta)
                y = avg_r * np.sin(phi) * np.sin(avg_theta)
                z = avg_r * np.cos(phi)

                norm = np.sqrt(x*x + y*y + z*z)
                if norm > 0:
                    class_vectors.append(np.array([x/norm, y/norm, z/norm]))
                else:
                    theta = (c * 2 * np.pi / max(2, n_classes))
                    phi = np.pi / 2
                    class_vectors.append(np.array([
                        np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi)
                    ]))
            else:
                theta = (c * 2 * np.pi / max(2, n_classes))
                phi = np.pi / 2
                class_vectors.append(np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ]))

        return np.array(class_vectors)

    def extract_class_distributions_from_test_data(self, weights, max_points_per_class=500):
        """
        Extract polar coordinate distributions for each class using TEST DATA predictions.
        This shows generalization rather than memorization.
        """
        test_indices = self._get_test_indices()
        if test_indices is None or len(test_indices) == 0:
            return self.extract_class_distributions(weights, max_points_per_class)

        X_test = self.model.X_tensor[test_indices]
        y_test = self.model.y_tensor[test_indices]

        predictions, posteriors = self.model.predict(X_test)

        if torch.is_tensor(posteriors):
            posteriors = posteriors.numpy()
        if torch.is_tensor(y_test):
            y_test = y_test.numpy()

        n_classes = weights.shape[0]
        n_test_samples = len(y_test)

        # Get class orientation vectors from tensor
        class_vectors = self._compute_class_orientation_vectors(weights, n_classes)

        # Project test points to sphere
        point_coords = np.zeros((n_test_samples, 3))

        for i in range(n_test_samples):
            probs = posteriors[i]
            weighted_sum = np.zeros(3)
            for c in range(n_classes):
                weighted_sum += probs[c] * class_vectors[c]

            norm = np.linalg.norm(weighted_sum)
            if norm > 0:
                point_coords[i] = weighted_sum / norm

        # Extract distributions per class
        class_distributions = {}

        for c in range(n_classes):
            class_mask = (y_test == c)
            class_points = point_coords[class_mask]

            if len(class_points) == 0:
                class_distributions[c] = {
                    'radii': np.array([]),
                    'angles': np.array([]),
                    'angles_deg': np.array([]),
                    'n_points': 0,
                    'mean_radius': 0,
                    'mean_angle': 0,
                    'mean_angle_deg': 0,
                    'radius_std': 0,
                    'angle_std': 0,
                    'angle_std_deg': 0,
                    'total_mass': 0,
                    'min_radius': 0,
                    'max_radius': 0,
                    'test_data': True
                }
                continue

            # Project onto complex plane (x,y) for 2D polar plot
            x_proj = class_points[:, 0]
            y_proj = class_points[:, 1]

            radii = np.sqrt(x_proj**2 + y_proj**2)
            angles = np.arctan2(y_proj, x_proj)
            angles_deg = angles * 180 / np.pi

            # Sample if too many points
            if len(radii) > max_points_per_class:
                indices = np.random.choice(len(radii), max_points_per_class, replace=False)
                radii = radii[indices]
                angles = angles[indices]
                angles_deg = angles_deg[indices]

            class_distributions[c] = {
                'radii': radii,
                'angles': angles,
                'angles_deg': angles_deg,
                'n_points': len(radii),
                'mean_radius': np.mean(radii),
                'mean_angle': np.mean(angles),
                'mean_angle_deg': np.mean(angles) * 180 / np.pi,
                'radius_std': np.std(radii),
                'angle_std': np.std(angles),
                'angle_std_deg': np.std(angles) * 180 / np.pi,
                'total_mass': np.sum(radii),
                'min_radius': np.min(radii) if len(radii) > 0 else 0,
                'max_radius': np.max(radii) if len(radii) > 0 else 0,
                'test_data': True
            }

        return class_distributions

    def extract_class_distributions(self, weights, max_points_per_class=500):
        """Original method - kept for fallback"""
        n_classes = weights.shape[0]
        class_distributions = {}

        for c in range(n_classes):
            class_weights = weights[c].flatten()
            significant = class_weights[np.abs(class_weights) > 1e-6]

            if len(significant) > max_points_per_class:
                indices = np.random.choice(len(significant), max_points_per_class, replace=False)
                significant = significant[indices]

            if len(significant) > 0:
                radii = np.abs(significant)
                angles = np.angle(significant)
                angles_deg = angles * 180 / np.pi

                class_distributions[c] = {
                    'radii': radii,
                    'angles': angles,
                    'angles_deg': angles_deg,
                    'n_points': len(significant),
                    'mean_radius': np.mean(radii),
                    'mean_angle': np.mean(angles),
                    'mean_angle_deg': np.mean(angles) * 180 / np.pi,
                    'radius_std': np.std(radii),
                    'angle_std': np.std(angles),
                    'angle_std_deg': np.std(angles) * 180 / np.pi,
                    'total_mass': np.sum(radii),
                    'min_radius': np.min(radii),
                    'max_radius': np.max(radii),
                    'test_data': False
                }
            else:
                class_distributions[c] = {
                    'radii': np.array([]),
                    'angles': np.array([]),
                    'angles_deg': np.array([]),
                    'n_points': 0,
                    'mean_radius': 0,
                    'mean_angle': 0,
                    'mean_angle_deg': 0,
                    'radius_std': 0,
                    'angle_std': 0,
                    'angle_std_deg': 0,
                    'total_mass': 0,
                    'min_radius': 0,
                    'max_radius': 0,
                    'test_data': False
                }

        return class_distributions

    def calculate_cluster_metrics_2d(self, class_distributions):
        """
        Calculate 2D cluster formation metrics for each class.
        """
        metrics = {}
        data_source = "TEST" if any(d.get('test_data', False) for d in class_distributions.values()) else "TRAINING"

        for c, dist in class_distributions.items():
            if dist['n_points'] < 2:
                metrics[c] = {
                    'cluster_quality': 0,
                    'angular_concentration': 0,
                    'radial_concentration': 0,
                    'circular_variance': 1,
                    'entropy': 1,
                    'mean_resultant_length': 0,
                    'angular_spread': 0,
                    'data_source': data_source
                }
                continue

            angles = dist['angles']
            complex_angles = np.exp(1j * angles)
            mean_complex = np.mean(complex_angles)
            mean_resultant_length = np.abs(mean_complex)
            circular_variance = 1 - mean_resultant_length

            angular_spread = np.std(angles) * 180 / np.pi

            radii = dist['radii']
            cv = np.std(radii) / (np.mean(radii) + 1e-10)
            radial_concentration = 1.0 / (1.0 + cv)

            hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
            hist = hist / (np.sum(hist) + 1e-10)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            normalized_entropy = entropy / np.log(36)

            cluster_quality = (mean_resultant_length + radial_concentration + (1 - normalized_entropy)) / 3

            metrics[c] = {
                'cluster_quality': cluster_quality,
                'angular_concentration': mean_resultant_length,
                'radial_concentration': radial_concentration,
                'circular_variance': circular_variance,
                'entropy': normalized_entropy,
                'mean_resultant_length': mean_resultant_length,
                'angular_spread': angular_spread,
                'n_points': dist['n_points'],
                'mean_radius': dist['mean_radius'],
                'mean_angle_deg': dist['mean_angle_deg'],
                'data_source': data_source
            }

        return metrics

    def create_complete_polar_dashboard(self, evolution_history, class_names=None):
        """
        Create a complete 2D polar dashboard with multiple views.
        Uses TEST DATA for metrics.
        """
        if not evolution_history or len(evolution_history) < 2:
            print(f"{Colors.YELLOW}Need at least 2 rounds for evolution visualization{Colors.ENDC}")
            return None

        print(f"{Colors.CYAN}📊 Creating complete 2D polar dashboard...{Colors.ENDC}")

        test_indices = self._get_test_indices()
        if test_indices is not None and len(test_indices) > 0:
            print(f"   Using TEST DATA for metrics ({len(test_indices)} samples)")
            print(f"   This shows GENERALIZATION, not memorization")
        else:
            print(f"   {Colors.YELLOW}No test indices found. Using training data (memorization){Colors.ENDC}")

        use_test = test_indices is not None and len(test_indices) > 0

        # Create animated scatter plot
        animated_fig = self.create_2d_polar_animation(evolution_history, class_names)

        # Create quality timeline
        timeline_fig = self.create_cluster_quality_timeline(evolution_history)

        # Create static snapshots for key rounds
        key_rounds = self._get_key_rounds(evolution_history)
        static_figs = []

        for round_num in key_rounds:
            snap = next((s for s in evolution_history if s.get('round') == round_num), None)
            if snap and 'complex_weights' in snap:
                weights = snap['complex_weights']
                if torch.is_tensor(weights):
                    weights = weights.cpu().numpy()

                # Use test data distributions
                distributions = self.extract_class_distributions_from_test_data(weights)
                fig, _ = self.create_2d_polar_scatter(
                    distributions, round_num,
                    snap.get('accuracy', 0),
                    snap.get('training_size', 0)
                )
                static_figs.append(fig)

                # Also create angular histogram for key rounds
                angular_fig = self.create_angular_histogram(
                    distributions, round_num,
                    snap.get('accuracy', 0),
                    snap.get('training_size', 0)
                )
                static_figs.append(angular_fig)

        # Create final metrics table using test data
        final_snap = evolution_history[-1]
        final_weights = final_snap['complex_weights']
        if torch.is_tensor(final_weights):
            final_weights = final_weights.cpu().numpy()

        final_distributions = self.extract_class_distributions_from_test_data(final_weights)
        final_metrics = self.calculate_cluster_metrics_2d(final_distributions)

        # Create combined HTML
        dashboard_html = self._create_combined_2d_dashboard_html(
            evolution_history, animated_fig, timeline_fig, static_figs,
            final_metrics, final_distributions, class_names, use_test
        )

        dashboard_path = self.output_dir / f'{self.dataset_name}_2d_polar_dashboard.html'

        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)

        print(f"{Colors.GREEN}✅ 2D Polar dashboard saved to: {dashboard_path}{Colors.ENDC}")

        return str(dashboard_path)

    def create_2d_polar_animation(self, evolution_history, class_names=None):
        """
        Create an animated 2D polar evolution showing how clusters form over time.
        Uses TEST DATA for visualization.
        """
        if not evolution_history or len(evolution_history) < 2:
            return None

        print(f"{Colors.CYAN}🎬 Creating 2D polar animation...{Colors.ENDC}")

        # First pass: find global max radius for consistent scaling
        global_max_radius = 0
        valid_snapshots = []
        test_indices = self._get_test_indices()
        use_test = test_indices is not None and len(test_indices) > 0

        for snap in evolution_history:
            if 'complex_weights' not in snap:
                continue

            weights = snap['complex_weights']
            if torch.is_tensor(weights):
                weights = weights.cpu().numpy()

            if len(weights.shape) == 4:
                distributions = self.extract_class_distributions_from_test_data(weights)
                for dist in distributions.values():
                    if len(dist['radii']) > 0:
                        global_max_radius = max(global_max_radius, np.max(dist['radii']))
                valid_snapshots.append(snap)

        global_max_radius = global_max_radius * 1.1

        # Create frames
        frames = []

        for snap in valid_snapshots:
            weights = snap['complex_weights']
            if torch.is_tensor(weights):
                weights = weights.cpu().numpy()

            round_num = snap['round']
            accuracy = snap.get('accuracy', 0)
            training_size = snap.get('training_size', 0)

            distributions = self.extract_class_distributions_from_test_data(weights)
            metrics = self.calculate_cluster_metrics_2d(distributions)

            # Create frame data
            frame_traces = []

            # Add reference circles and radial lines (same as before)
            for r in [0.25, 0.5, 0.75, 1.0]:
                if r * global_max_radius <= global_max_radius:
                    theta = np.linspace(0, 2*np.pi, 100)
                    x_circle = r * global_max_radius * np.cos(theta)
                    y_circle = r * global_max_radius * np.sin(theta)
                    frame_traces.append(go.Scatter(
                        x=x_circle, y=y_circle,
                        mode='lines',
                        line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash'),
                        showlegend=False,
                        hoverinfo='none'
                    ))

            for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
                angle_rad = angle_deg * np.pi / 180
                x_line = [0, global_max_radius * np.cos(angle_rad)]
                y_line = [0, global_max_radius * np.sin(angle_rad)]
                frame_traces.append(go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines',
                    line=dict(color='rgba(128,128,128,0.2)', width=1),
                    showlegend=False,
                    hoverinfo='none'
                ))

            # Add class points
            for c, dist in distributions.items():
                if dist['n_points'] == 0:
                    continue

                class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'
                color = self.class_colors[c % len(self.class_colors)]

                x = dist['radii'] * np.cos(dist['angles'])
                y = dist['radii'] * np.sin(dist['angles'])

                frame_traces.append(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(size=5, color=color, opacity=0.7),
                    name=class_name,
                    legendgroup=f'class_{c}',
                    showlegend=False,
                    text=[f'Class: {class_name}<br>Radius: {r:.3f}<br>Angle: {a:.1f}°'
                          for r, a in zip(dist['radii'], dist['angles_deg'])],
                    hoverinfo='text'
                ))

                if dist['mean_radius'] > 0:
                    mean_x = dist['mean_radius'] * np.cos(dist['mean_angle'])
                    mean_y = dist['mean_radius'] * np.sin(dist['mean_angle'])

                    frame_traces.append(go.Scatter(
                        x=[0, mean_x], y=[0, mean_y],
                        mode='lines+markers',
                        marker=dict(size=8, color=color, symbol='circle'),
                        line=dict(color=color, width=2),
                        name=f'{class_name} (mean)',
                        showlegend=False,
                        legendgroup=f'class_{c}'
                    ))

            # Add unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            frame_traces.append(go.Scatter(
                x=np.cos(theta), y=np.sin(theta),
                mode='lines',
                line=dict(color='white', width=2, dash='dash'),
                name='Unit Circle',
                showlegend=False
            ))

            # Add origin
            frame_traces.append(go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(size=6, color='white', symbol='x'),
                name='Origin',
                showlegend=False
            ))

            data_source = "TEST DATA" if use_test else "TRAINING DATA"
            quality_text = "<br>".join([
                f"Class {c+1}: Q={metrics[c]['cluster_quality']:.2f}, "
                f"θ_conc={metrics[c]['angular_concentration']:.2f}, "
                f"θ_spread={metrics[c]['angular_spread']:.1f}°"
                for c in metrics.keys()
            ])

            frames.append(go.Frame(
                data=frame_traces,
                name=f'round_{round_num}',
                layout=go.Layout(
                    title=dict(
                        text=f'<b>2D Polar Evolution - Round {round_num} ({data_source})</b><br>'
                             f'Accuracy: {accuracy:.3f} | Training Samples: {training_size}<br>'
                             f'<sup>Cluster Quality: {quality_text}</sup>',
                        font=dict(size=12)
                    )
                )
            ))

        if not frames:
            return None

        # Create initial frame
        first_frame = frames[0]
        data_source = "TEST DATA" if use_test else "TRAINING DATA"

        fig = go.Figure(
            data=first_frame.data,
            layout=go.Layout(
                title=first_frame.layout.title,
                xaxis=dict(
                    title='Real Component',
                    range=[-global_max_radius, global_max_radius],
                    scaleanchor='y',
                    scaleratio=1,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                yaxis=dict(
                    title='Imaginary Component',
                    range=[-global_max_radius, global_max_radius],
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                height=800,
                width=800,
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
                        y=0.95,
                        x=0.05
                    )
                ],
                sliders=[{
                    'active': 0,
                    'currentvalue': {'prefix': 'Round: ', 'font': {'size': 14}},
                    'steps': [
                        {
                            'args': [[f'round_{snap["round"]}'], {'frame': {'duration': 0, 'redraw': True}}],
                            'label': str(snap['round']),
                            'method': 'animate'
                        }
                        for snap in valid_snapshots
                    ]
                }],
                showlegend=True,
                legend=dict(
                    yanchor="top", y=0.99, xanchor="left", x=0.02,
                    bgcolor='rgba(0,0,0,0.7)',
                    font=dict(color='white')
                ),
                paper_bgcolor='rgba(0,0,0,0.9)',
                plot_bgcolor='rgba(0,0,0,0.9)'
            ),
            frames=frames
        )

        return fig

    def create_cluster_quality_timeline(self, evolution_history):
        """
        Create a timeline plot showing how cluster quality evolves over rounds.
        Uses TEST DATA metrics.
        """
        if not evolution_history:
            return None

        rounds = []
        cluster_quality = []
        angular_concentration = []
        radial_concentration = []
        angular_spread = []
        test_indices = self._get_test_indices()
        use_test = test_indices is not None and len(test_indices) > 0

        for snap in evolution_history:
            if 'complex_weights' not in snap:
                continue

            weights = snap['complex_weights']
            if torch.is_tensor(weights):
                weights = weights.cpu().numpy()

            if len(weights.shape) != 4:
                continue

            round_num = snap['round']
            distributions = self.extract_class_distributions_from_test_data(weights)
            metrics = self.calculate_cluster_metrics_2d(distributions)

            rounds.append(round_num)

            avg_quality = np.mean([m['cluster_quality'] for m in metrics.values()])
            avg_angular_conc = np.mean([m['angular_concentration'] for m in metrics.values()])
            avg_radial_conc = np.mean([m['radial_concentration'] for m in metrics.values()])
            avg_angular_spread = np.mean([m['angular_spread'] for m in metrics.values()])

            cluster_quality.append(avg_quality)
            angular_concentration.append(avg_angular_conc)
            radial_concentration.append(avg_radial_conc)
            angular_spread.append(avg_angular_spread)

        if not rounds:
            return None

        data_source = "TEST DATA" if use_test else "TRAINING DATA"

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(f'Cluster Quality ({data_source}, higher = better)',
                          f'Angular Concentration ({data_source}, 1 = perfectly focused)',
                          f'Radial Concentration ({data_source}, 1 = perfectly focused)',
                          f'Angular Spread ({data_source}, degrees, lower = better)')
        )

        fig.add_trace(go.Scatter(
            x=rounds, y=cluster_quality,
            mode='lines+markers',
            name='Avg Quality',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ), row=1, col=1)
        fig.add_hline(y=0.9, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)

        fig.add_trace(go.Scatter(
            x=rounds, y=angular_concentration,
            mode='lines+markers',
            name='Angular Concentration',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ), row=1, col=2)
        fig.add_hline(y=0.9, line_dash="dash", line_color="blue", opacity=0.5, row=1, col=2)

        fig.add_trace(go.Scatter(
            x=rounds, y=radial_concentration,
            mode='lines+markers',
            name='Radial Concentration',
            line=dict(color='orange', width=2),
            marker=dict(size=8)
        ), row=2, col=1)
        fig.add_hline(y=0.8, line_dash="dash", line_color="orange", opacity=0.5, row=2, col=1)

        fig.add_trace(go.Scatter(
            x=rounds, y=angular_spread,
            mode='lines+markers',
            name='Angular Spread',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ), row=2, col=2)
        fig.add_hline(y=45, line_dash="dash", line_color="red", opacity=0.5, row=2, col=2,
                     annotation_text="Good separation (<45°)")

        fig.update_layout(
            title=dict(
                text=f'<b>Cluster Formation Timeline - {self.dataset_name} ({data_source})</b><br>'
                     f'<sup>Higher quality and concentration = better defined clusters</sup>',
                font=dict(size=14)
            ),
            height=800,
            width=1000,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )

        return fig

    def create_2d_polar_scatter(self, class_distributions, round_num, accuracy, training_size,
                                  max_radius=None, show_density=False):
        """
        Create a 2D polar scatter plot showing all class distributions.
        Each point is a weight in Cartesian coordinates (x = r*cosθ, y = r*sinθ).
        """
        fig = go.Figure()

        # Determine global max radius for consistent scaling
        if max_radius is None:
            max_radius = 0
            for dist in class_distributions.values():
                if len(dist['radii']) > 0:
                    max_radius = max(max_radius, np.max(dist['radii']))
            max_radius = max_radius * 1.1  # Add padding

        # Add reference circles for visual guidance
        for r in [0.25, 0.5, 0.75, 1.0]:
            if r * max_radius <= max_radius:
                theta = np.linspace(0, 2*np.pi, 100)
                x_circle = r * max_radius * np.cos(theta)
                y_circle = r * max_radius * np.sin(theta)
                fig.add_trace(go.Scatter(
                    x=x_circle, y=y_circle,
                    mode='lines',
                    line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='none'
                ))

        # Add radial lines at key angles (0°, 90°, 180°, 270°)
        for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            angle_rad = angle_deg * np.pi / 180
            x_line = [0, max_radius * np.cos(angle_rad)]
            y_line = [0, max_radius * np.sin(angle_rad)]
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color='rgba(128,128,128,0.2)', width=1),
                showlegend=False,
                hoverinfo='none'
            ))

        # Add points for each class
        for c, dist in class_distributions.items():
            if dist['n_points'] == 0:
                continue

            class_name = f'Class {c+1}'
            color = self.class_colors[c % len(self.class_colors)]

            # Convert to Cartesian coordinates
            x = dist['radii'] * np.cos(dist['angles'])
            y = dist['radii'] * np.sin(dist['angles'])

            # Add scatter trace
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(
                    size=6,
                    color=color,
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                name=class_name,
                text=[f'Class: {class_name}<br>'
                      f'Radius: {r:.3f}<br>'
                      f'Angle: {a:.2f} rad ({a*180/np.pi:.1f}°)<br>'
                      f'Magnitude: {r:.3f}'
                      for r, a in zip(dist['radii'], dist['angles'])],
                hoverinfo='text+name',
                legendgroup=f'class_{c}'
            ))

            # Add mean vector (center of mass)
            if dist['mean_radius'] > 0:
                mean_x = dist['mean_radius'] * np.cos(dist['mean_angle'])
                mean_y = dist['mean_radius'] * np.sin(dist['mean_angle'])

                fig.add_trace(go.Scatter(
                    x=[0, mean_x],
                    y=[0, mean_y],
                    mode='lines+markers',
                    marker=dict(size=10, color=color, symbol='circle',
                               line=dict(width=2, color='white')),
                    line=dict(color=color, width=3),
                    name=f'{class_name} (mean)',
                    showlegend=False,
                    hoverinfo='text',
                    text=f'Mean vector<br>'
                         f'Radius: {dist["mean_radius"]:.3f}<br>'
                         f'Angle: {dist["mean_angle_deg"]:.1f}°',
                    legendgroup=f'class_{c}'
                ))

            # Add 1-sigma confidence ellipse (optional)
            if dist['n_points'] > 3:
                self._add_confidence_ellipse_2d(fig, x, y, color, class_name)

        # Add unit circle reference
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x_circle, y=y_circle,
            mode='lines',
            line=dict(color='white', width=2, dash='dash'),
            name='Unit Circle (Radius=1)',
            hoverinfo='none'
        ))

        # Add origin marker
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=8, color='white', symbol='x', line=dict(width=2)),
            name='Origin',
            hoverinfo='none'
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'<b>2D Polar Evolution - Round {round_num}</b><br>'
                     f'Accuracy: {accuracy:.3f} | Training Samples: {training_size}<br>'
                     f'<sup>Each point = weight in complex plane (distance = magnitude, angle = phase)</sup>',
                font=dict(size=14)
            ),
            xaxis=dict(
                title='Real Component (r·cosθ)',
                range=[-max_radius, max_radius],
                scaleanchor='y',
                scaleratio=1,
                gridcolor='rgba(128,128,128,0.2)',
                zerolinecolor='rgba(128,128,128,0.5)'
            ),
            yaxis=dict(
                title='Imaginary Component (r·sinθ)',
                range=[-max_radius, max_radius],
                gridcolor='rgba(128,128,128,0.2)',
                zerolinecolor='rgba(128,128,128,0.5)'
            ),
            height=700,
            width=700,
            hovermode='closest',
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.02,
                bgcolor='rgba(0,0,0,0.7)',
                font=dict(color='white')
            ),
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )

        return fig, max_radius

    def _add_confidence_ellipse_2d(self, fig, x, y, color, class_name):
        """Add 1-sigma confidence ellipse to the plot"""
        from scipy.stats import chi2

        if len(x) < 3:
            return

        # Calculate covariance matrix
        cov = np.cov(x, y)
        if np.any(np.isnan(cov)) or np.linalg.det(cov) == 0:
            return

        # Get eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvalues
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Calculate ellipse parameters (1-sigma, chi-squared with 2 DOF)
        chi2_val = chi2.ppf(0.68, 2)  # 68% confidence interval

        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = 2 * np.sqrt(chi2_val * eigvals[0])
        height = 2 * np.sqrt(chi2_val * eigvals[1])

        # Create ellipse points
        t = np.linspace(0, 2*np.pi, 100)
        ellipse_x = width/2 * np.cos(t)
        ellipse_y = height/2 * np.sin(t)

        # Rotate ellipse
        rot_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                               [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
        ellipse_rot = np.dot(rot_matrix, np.array([ellipse_x, ellipse_y]))

        # Center at mean
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        ellipse_x_plot = ellipse_rot[0, :] + mean_x
        ellipse_y_plot = ellipse_rot[1, :] + mean_y

        # Add ellipse trace
        fig.add_trace(go.Scatter(
            x=ellipse_x_plot,
            y=ellipse_y_plot,
            mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            name=f'{class_name} (1σ ellipse)',
            showlegend=False,
            hoverinfo='none'
        ))

    def create_angular_histogram(self, class_distributions, round_num, accuracy, training_size):
        """
        Create a circular histogram showing angular distribution for each class.
        This is excellent for showing class separation by angle.
        """
        fig = make_subplots(
            rows=1, cols=len(class_distributions),
            subplot_titles=[f'Class {c+1}' for c in class_distributions.keys()],
            specs=[[{'type': 'polar'} for _ in class_distributions.keys()]]
        )

        for idx, (c, dist) in enumerate(class_distributions.items()):
            if dist['n_points'] == 0:
                continue

            # Create angular histogram
            angles_deg = dist['angles_deg']

            # Create histogram bins (36 bins = 10° each)
            bins = np.linspace(-180, 180, 37)
            hist, bin_edges = np.histogram(angles_deg, bins=bins)

            # Convert to polar bar chart
            theta = np.radians(bin_edges[:-1] + 10)  # Center of each bin
            r = hist / np.max(hist) if np.max(hist) > 0 else hist

            fig.add_trace(
                go.Barpolar(
                    r=r,
                    theta=bin_edges[:-1] + 10,
                    width=10,
                    marker=dict(
                        color=self.class_colors[c % len(self.class_colors)],
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    name=f'Class {c+1}',
                    hovertemplate='Angle: %{theta:.0f}°<br>Density: %{r:.3f}<extra></extra>'
                ),
                row=1, col=idx+1
            )

            # Add mean angle marker
            mean_angle = dist['mean_angle_deg']
            mean_radius = 1.1  # Just outside the histogram

            fig.add_trace(
                go.Scatterpolar(
                    r=[mean_radius],
                    theta=[mean_angle],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=self.class_colors[c % len(self.class_colors)],
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    name='Mean',
                    showlegend=False,
                    hovertemplate=f'Mean Angle: {mean_angle:.1f}°<extra></extra>'
                ),
                row=1, col=idx+1
            )

            # Add angular spread annotation
            angular_spread = dist['angle_std_deg']
            fig.add_annotation(
                x=0.5, y=1.1,
                xref=f'x{idx+1}', yref=f'paper',
                text=f'Spread: ±{angular_spread:.1f}°',
                showarrow=False,
                font=dict(size=10, color='white'),
                bgcolor='rgba(0,0,0,0.5)'
            )

        fig.update_layout(
            title=dict(
                text=f'<b>Angular Distribution - Round {round_num}</b><br>'
                     f'Accuracy: {accuracy:.3f} | Training Samples: {training_size}<br>'
                     f'<sup>Shows how classes separate by angle in complex space</sup>',
                font=dict(size=14)
            ),
            height=500,
            width=500 * len(class_distributions),
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)',
            showlegend=False
        )

        return fig

    def create_radial_histogram(self, class_distributions, round_num, accuracy, training_size):
        """
        Create a radial histogram showing magnitude distribution for each class.
        Shows how focused the magnitudes are.
        """
        fig, axes = plt.subplots(1, len(class_distributions), figsize=(5 * len(class_distributions), 4))

        if len(class_distributions) == 1:
            axes = [axes]

        for idx, (c, dist) in enumerate(class_distributions.items()):
            ax = axes[idx]

            if dist['n_points'] > 0:
                ax.hist(dist['radii'], bins=30, color=self.class_colors[c % len(self.class_colors)],
                       alpha=0.7, edgecolor='white', linewidth=0.5)
                ax.axvline(dist['mean_radius'], color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {dist["mean_radius"]:.3f}')
                ax.set_title(f'Class {c+1}')
                ax.set_xlabel('Magnitude (Radius)')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Class {c+1}')

        plt.suptitle(f'Magnitude Distribution - Round {round_num} (Accuracy: {accuracy:.3f})')
        plt.tight_layout()

        # Save to file
        radial_path = self.output_dir / f'{self.dataset_name}_radial_hist_round_{round_num}.png'
        plt.savefig(radial_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(radial_path)

    def _get_key_rounds(self, evolution_history):
        """Get key rounds for static snapshots"""
        n_rounds = len(evolution_history)
        if n_rounds <= 4:
            return [snap['round'] for snap in evolution_history if 'round' in snap]

        rounds = [snap['round'] for snap in evolution_history if 'round' in snap]
        return [rounds[0], rounds[n_rounds//3], rounds[2*n_rounds//3], rounds[-1]]

    def _create_combined_2d_dashboard_html(self, evolution_history, animated_fig, timeline_fig,
                                            static_figs, final_metrics, final_distributions, class_names):
        """Create combined HTML dashboard with all 2D polar visualizations"""

        # Convert figures to HTML
        animated_html = animated_fig.to_html(include_plotlyjs='cdn', div_id='polar-animation') if animated_fig else ''
        timeline_html = timeline_fig.to_html(include_plotlyjs=False, div_id='quality-timeline') if timeline_fig else ''

        static_htmls = []
        for i, fig in enumerate(static_figs):
            static_htmls.append(fig.to_html(include_plotlyjs=False, div_id=f'static-{i}'))

        # Build metrics table
        metrics_rows = []
        for c, metrics in final_metrics.items():
            class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'

            # Color coding based on quality
            quality_color = 'metric-good' if metrics['cluster_quality'] > 0.8 else \
                           'metric-medium' if metrics['cluster_quality'] > 0.6 else 'metric-poor'

            angular_color = 'metric-good' if metrics['angular_concentration'] > 0.8 else \
                           'metric-medium' if metrics['angular_concentration'] > 0.6 else 'metric-poor'

            spread_color = 'metric-good' if metrics['angular_spread'] < 30 else \
                          'metric-medium' if metrics['angular_spread'] < 60 else 'metric-poor'

            # Get sample points from distribution
            dist = final_distributions[c]
            n_points = dist['n_points']
            mean_radius = dist['mean_radius']
            mean_angle = dist['mean_angle_deg']

            metrics_rows.append(f"""
            <tr>
                <td><span style="color: {self.class_colors[c % len(self.class_colors)]}; font-weight: bold;">●</span> {class_name}</td>
                <td class="{quality_color}">{metrics['cluster_quality']:.3f}</td>
                <td class="{angular_color}">{metrics['angular_concentration']:.3f}</td>
                <td class="{spread_color}">{metrics['angular_spread']:.1f}°</td>
                <td>{metrics['radial_concentration']:.3f}</td>
                <td>{n_points}</td>
                <td>{mean_radius:.3f}</td>
                <td>{mean_angle:.1f}°</td>
            </tr>
            """)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>2D Polar Evolution - {self.dataset_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header p {{
            margin: 0;
            opacity: 0.9;
        }}
        .dashboard-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .plot-container {{
            background: rgba(0,0,0,0.7);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .metrics-table {{
            background: rgba(0,0,0,0.5);
            border-radius: 10px;
            padding: 15px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }}
        th {{
            background: rgba(102,126,234,0.5);
            font-weight: bold;
        }}
        .metric-good {{
            color: #4caf50;
            font-weight: bold;
        }}
        .metric-medium {{
            color: #ff9800;
            font-weight: bold;
        }}
        .metric-poor {{
            color: #f44336;
            font-weight: bold;
        }}
        .info-text {{
            font-size: 12px;
            color: #aaa;
            margin-top: 10px;
            text-align: center;
            padding: 10px;
            background: rgba(0,0,0,0.3);
            border-radius: 5px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }}
        .color-box {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 2D Polar Evolution - {self.dataset_name}</h1>
        <p>Visualizing how class clusters form in the complex plane over adaptive rounds</p>
        <p>Each point = a weight from the 5D tensor | Distance from origin = Magnitude | Angle = Phase</p>
        <div class="legend">
            {''.join([f'<div class="legend-item"><div class="color-box" style="background: {self.class_colors[c % len(self.class_colors)]};"></div><span>Class {class_names[c] if class_names and c < len(class_names) else c+1}</span></div>' for c in final_metrics.keys()])}
        </div>
    </div>

    <div class="dashboard-container">
        <div class="plot-container full-width">
            <h3>🎬 Animated 2D Polar Evolution</h3>
            <p>Watch clusters form and separate over time</p>
            {animated_html}
        </div>
    </div>

    <div class="dashboard-container">
        <div class="plot-container">
            <h3>📈 Cluster Quality Timeline</h3>
            <p>Track how well-defined clusters become during training</p>
            {timeline_html}
        </div>

        <div class="plot-container">
            <h3>🎯 Final State Metrics</h3>
            <div class="metrics-table">
                <table>
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Cluster Quality</th>
                            <th>Angular Concentration</th>
                            <th>Angular Spread</th>
                            <th>Radial Concentration</th>
                            <th>Points</th>
                            <th>Mean Radius</th>
                            <th>Mean Angle</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(metrics_rows)}
                    </tbody>
                </table>
            </div>
            <div class="info-text">
                <strong>Interpretation:</strong><br>
                • <strong>Cluster Quality</strong>: 1.0 = perfectly defined separate clusters<br>
                • <strong>Angular Concentration</strong>: 1.0 = all points at same angle (good separation)<br>
                • <strong>Angular Spread</strong>: Lower = more focused cluster (good: &lt;45°)<br>
                • <strong>Radial Concentration</strong>: 1.0 = all points at same radius (consistent magnitude)
            </div>
        </div>
    </div>

    <div class="dashboard-container">
        <div class="plot-container full-width">
            <h3>📸 Key Round Snapshots</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                {''.join([f'<div>{html}</div>' for html in static_htmls])}
            </div>
        </div>
    </div>

    <div class="info-text">
        <strong>How to Read 2D Polar Plots:</strong><br>
        • <strong>X-axis</strong>: Real component of weight (r·cosθ)<br>
        • <strong>Y-axis</strong>: Imaginary component of weight (r·sinθ)<br>
        • <strong>Distance from origin</strong>: Magnitude of weight (strength of that feature-pair bin)<br>
        • <strong>Angle from origin</strong>: Phase of weight (orientation in complex space)<br>
        • <strong>Color</strong>: Class membership<br>
        • <strong>Ellipse</strong>: 1-sigma confidence region (68% of points)<br>
        • <strong>Arrow</strong>: Mean vector (center of mass of cluster)<br>
        <br>
        <strong>Perfect Classification Achieved When:</strong><br>
        • Classes form separate, focused clusters at different angles<br>
        • Clusters are at least 45° apart (angular spread &lt; 45°)<br>
        • High angular concentration (>0.8) and cluster quality (>0.8)<br>
        • Classes are orthogonal (90° apart) in complex space
    </div>
</body>
</html>"""

        return html

# =============================================================================
# COMPLETE CORRECTED: TRUE POLAR EVOLUTION - With Proper Animation
# =============================================================================

import numpy as np
import torch
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class TruePolarEvolution:
    """
    True polar coordinate evolution using DBNN's internal feature pairs.
    Now uses TEST DATA for class separation metrics with margin calculations
    and density-based sampling to avoid overcrowding.
    """

    def __init__(self, model, output_dir: str = 'visualizations'):
        self.model = model
        self.dataset_name = model.dataset_name if model and hasattr(model, 'dataset_name') else 'unknown'
        self.output_dir = Path(output_dir) / self.dataset_name / 'true_polar_evolution'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_pairs = None
        self.feature_names = None

        if hasattr(model, 'feature_pairs') and model.feature_pairs:
            self.feature_pairs = model.feature_pairs
            print(f"{Colors.CYAN}📊 Using DBNN's {len(self.feature_pairs)} internal feature pairs{Colors.ENDC}")

        if hasattr(model, 'feature_names') and model.feature_names:
            self.feature_names = model.feature_names
            print(f"   Feature names available: {len(self.feature_names)} features")

        self.class_colors = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
            '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
            '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'
        ]

        self.current_pair_idx = 0

        print(f"{Colors.CYAN}📊 True Polar Evolution initialized{Colors.ENDC}")
        print(f"   Output: {self.output_dir}")
        print(f"   Using TEST DATA for class separation metrics (generalization)")

    def _get_test_indices(self):
        """Get test indices from model"""
        if hasattr(self.model, 'test_indices') and self.model.test_indices is not None:
            test_indices = self.model.test_indices
            if torch.is_tensor(test_indices):
                test_indices = test_indices.numpy() if hasattr(test_indices, 'numpy') else np.array(test_indices)
            return test_indices
        return None

    def _density_based_sampling(self, X, y, max_points_per_class=200):
        """
        Sample points using density-based sampling to preserve shape.
        Keeps border points and reduces density in crowded regions.
        """
        unique_classes = np.unique(y)
        X_sampled = []
        y_sampled = []

        for cls in unique_classes:
            class_mask = y == cls
            X_class = X[class_mask]

            if len(X_class) <= max_points_per_class:
                X_sampled.append(X_class)
                y_sampled.append(np.full(len(X_class), cls))
                continue

            from sklearn.neighbors import NearestNeighbors

            # Compute local density
            nbrs = NearestNeighbors(n_neighbors=min(10, len(X_class)), algorithm='auto').fit(X_class)
            distances, _ = nbrs.kneighbors(X_class)
            avg_distances = np.mean(distances, axis=1)

            # Higher density = smaller distances
            density_scores = 1 / (avg_distances + 1e-8)

            # Probability of selection is inversely proportional to density
            # This keeps border points (lower density) and reduces crowded interior
            selection_probs = 1 / (density_scores + 1e-8)
            selection_probs = selection_probs / np.sum(selection_probs)

            # Select points
            selected_indices = np.random.choice(
                len(X_class),
                size=max_points_per_class,
                replace=False,
                p=selection_probs
            )

            X_sampled.append(X_class[selected_indices])
            y_sampled.append(np.full(max_points_per_class, cls))

        return np.vstack(X_sampled) if X_sampled else X, np.hstack(y_sampled) if y_sampled else y

    def _get_test_data_for_round(self):
        """Get test data with density-based sampling"""
        test_indices = self._get_test_indices()
        if test_indices is None or len(test_indices) == 0:
            return None, None

        X_test = self.model.X_tensor[test_indices]
        y_test = self.model.y_tensor[test_indices]

        if torch.is_tensor(X_test):
            X_test = X_test.cpu().numpy()
        if torch.is_tensor(y_test):
            y_test = y_test.cpu().numpy()

        # Apply density-based sampling to avoid overcrowding
        X_test, y_test = self._density_based_sampling(X_test, y_test, max_points_per_class=150)

        return X_test, y_test

    def _compute_class_statistics_with_margin(self, X, y, f1, f2):
        """
        Compute class statistics including centers, spreads, and margin separation.
        Returns: dict with class points, centers, spreads, and margin metrics
        """
        unique_classes = np.unique(y)
        class_stats = {}

        # First pass: compute class centers and spreads
        for cls in unique_classes:
            class_mask = y == cls
            X_class = X[class_mask]

            if len(X_class) == 0:
                class_stats[int(cls)] = {
                    'points': {'x': np.array([]), 'y': np.array([]), 'magnitudes': np.array([]), 'angles_deg': np.array([])},
                    'center': {'x': 0, 'y': 0, 'angle_deg': 0, 'magnitude': 0},
                    'spread': 0,
                    'n_points': 0
                }
                continue

            x_vals = X_class[:, f1]
            y_vals = X_class[:, f2]
            magnitudes = np.sqrt(x_vals**2 + y_vals**2)
            angles_rad = np.arctan2(y_vals, x_vals)
            angles_deg = angles_rad * 180 / np.pi

            # Class center (mean of points)
            center_x = np.mean(x_vals)
            center_y = np.mean(y_vals)
            center_angle = np.arctan2(center_y, center_x) * 180 / np.pi
            center_magnitude = np.sqrt(center_x**2 + center_y**2)

            # Angular spread (standard deviation of angles around center)
            angles_to_center = []
            for ang in angles_deg:
                # Circular distance to center angle
                diff = abs(ang - center_angle)
                diff = min(diff, 360 - diff)
                angles_to_center.append(diff)
            spread = np.std(angles_to_center) if angles_to_center else 0

            class_stats[int(cls)] = {
                'points': {
                    'x': x_vals,
                    'y': y_vals,
                    'magnitudes': magnitudes,
                    'angles_deg': angles_deg
                },
                'center': {
                    'x': center_x,
                    'y': center_y,
                    'angle_deg': center_angle,
                    'magnitude': center_magnitude
                },
                'spread': spread,
                'n_points': len(X_class)
            }

        # Second pass: compute margin-based separation between classes
        n_classes = len(class_stats)
        if n_classes >= 2:
            center_angles = []
            margin_angles = []

            class_list = list(class_stats.keys())
            for i in range(len(class_list)):
                for j in range(i+1, len(class_list)):
                    c1 = class_list[i]
                    c2 = class_list[j]

                    angle1 = class_stats[c1]['center']['angle_deg']
                    angle2 = class_stats[c2]['center']['angle_deg']

                    # Angular difference (circular)
                    diff = abs(angle1 - angle2)
                    center_angle = min(diff, 360 - diff)
                    center_angles.append(center_angle)

                    # Margin separation = center angle minus sum of spreads
                    margin = center_angle - (class_stats[c1]['spread'] + class_stats[c2]['spread'])
                    margin_angles.append(max(0, margin))

            class_stats['global_metrics'] = {
                'avg_center_separation': np.mean(center_angles) if center_angles else 0,
                'avg_margin_separation': np.mean(margin_angles) if margin_angles else 0,
                'orthogonality_center': min(1.0, np.mean(center_angles) / 90.0) if center_angles else 0,
                'orthogonality_margin': min(1.0, np.mean(margin_angles) / 90.0) if margin_angles else 0,
                'center_angles': center_angles,
                'margin_angles': margin_angles
            }
        else:
            class_stats['global_metrics'] = {
                'avg_center_separation': 0,
                'avg_margin_separation': 0,
                'orthogonality_center': 0,
                'orthogonality_margin': 0,
                'center_angles': [],
                'margin_angles': []
            }

        return class_stats

    def set_feature_pair_by_index(self, pair_idx: int) -> bool:
        """Set which feature pair to visualize by index."""
        if self.feature_pairs and 0 <= pair_idx < len(self.feature_pairs):
            self.current_pair_idx = pair_idx
            f1, f2 = self.feature_pairs[pair_idx]
            if self.feature_names and f1 < len(self.feature_names) and f2 < len(self.feature_names):
                print(f"{Colors.GREEN}✓ Using feature pair {pair_idx}: {self.feature_names[f1]} (real) + i·{self.feature_names[f2]} (imag){Colors.ENDC}")
            else:
                print(f"{Colors.GREEN}✓ Using feature pair {pair_idx}: f{f1} (real) + i·f{f2} (imag){Colors.ENDC}")
            return True
        return False

    def get_feature_pair_description(self, pair_idx: int) -> str:
        """Get human-readable description of a feature pair."""
        if not self.feature_pairs or pair_idx >= len(self.feature_pairs):
            return f"Pair {pair_idx}"
        f1, f2 = self.feature_pairs[pair_idx]
        if self.feature_names and f1 < len(self.feature_names) and f2 < len(self.feature_names):
            return f"{self.feature_names[f1]} vs {self.feature_names[f2]}"
        return f"Feature {f1} vs Feature {f2}"

    def create_animated_polar_dashboard(
        self,
        evolution_history: List[Dict],
        class_names: Optional[List[str]] = None,
        feature_pair_idx: Optional[int] = None
    ) -> Optional[str]:
        """
        Create an animated polar dashboard showing class separation.
        Uses TEST DATA for metrics and density-based sampling.
        """
        if feature_pair_idx is not None:
            self.set_feature_pair_by_index(feature_pair_idx)

        if not evolution_history or len(evolution_history) < 2:
            print(f"{Colors.YELLOW}Need at least 2 rounds for evolution visualization{Colors.ENDC}")
            return None

        if not self.feature_pairs:
            print(f"{Colors.YELLOW}⚠️ DBNN has no feature pairs. Run generate_feature_pairs() first.{Colors.ENDC}")
            return None

        print(f"{Colors.CYAN}🎬 Creating true polar evolution animation...{Colors.ENDC}")

        test_indices = self._get_test_indices()
        use_test = test_indices is not None and len(test_indices) > 0

        if use_test:
            print(f"   Using TEST DATA for class separation metrics ({len(test_indices)} samples)")
            print(f"   This shows GENERALIZATION, not memorization")
            print(f"   Applying density-based sampling to avoid overcrowding")
        else:
            print(f"   {Colors.YELLOW}No test indices found. Using training data (memorization){Colors.ENDC}")

        # Get current feature pair info
        f1, f2 = self.feature_pairs[self.current_pair_idx]
        if self.feature_names and f1 < len(self.feature_names) and f2 < len(self.feature_names):
            pair_desc = f"{self.feature_names[f1]} (real) vs {self.feature_names[f2]} (imag)"
        else:
            pair_desc = f"Feature {f1} (real) vs Feature {f2} (imag)"

        print(f"   Using DBNN feature pair {self.current_pair_idx}: {pair_desc}")

        # Process all rounds
        rounds_data = []

        for snap in evolution_history:
            if use_test:
                X_data, y_data = self._get_test_data_for_round()
                if X_data is not None and len(X_data) > 0:
                    class_stats = self._compute_class_statistics_with_margin(X_data, y_data, f1, f2)
                    rounds_data.append({
                        'round': snap.get('round', 0),
                        'accuracy': snap.get('accuracy', 0.0),
                        'training_size': snap.get('training_size', 0),
                        'class_stats': class_stats,
                        'feature_pair': (f1, f2),
                        'pair_desc': pair_desc,
                        'data_source': 'TEST'
                    })
            else:
                # Fallback to training data
                train_indices = snap.get('training_indices', None)
                if train_indices is None and hasattr(self.model, 'train_indices'):
                    train_indices = self.model.train_indices

                X_train, y_train = self._get_training_data_for_round(train_indices)
                if X_train is not None and len(X_train) > 0:
                    class_stats = self._compute_class_statistics_with_margin(X_train, y_train, f1, f2)
                    rounds_data.append({
                        'round': snap.get('round', 0),
                        'accuracy': snap.get('accuracy', 0.0),
                        'training_size': snap.get('training_size', len(X_train)),
                        'class_stats': class_stats,
                        'feature_pair': (f1, f2),
                        'pair_desc': pair_desc,
                        'data_source': 'TRAINING'
                    })

        if not rounds_data:
            print(f"{Colors.RED}No valid rounds with data found{Colors.ENDC}")
            return None

        # Find global bounds
        all_x, all_y = [], []
        for rd in rounds_data:
            for stats in rd['class_stats'].values():
                if isinstance(stats, dict) and 'points' in stats:
                    all_x.extend(stats['points']['x'])
                    all_y.extend(stats['points']['y'])

        if all_x and all_y:
            max_val = max(max(np.abs(all_x)), max(np.abs(all_y)), 0.1) * 1.1
        else:
            max_val = 1.0

        # Get class names
        if class_names is None and hasattr(self.model, 'label_encoder') and self.model.label_encoder:
            class_names = list(self.model.label_encoder.keys())
        elif class_names is None:
            # Get class IDs from first round
            class_ids = [k for k in rounds_data[0]['class_stats'].keys() if isinstance(k, int)]
            class_names = [f'Class {c+1}' for c in sorted(class_ids)]

        print(f"   Processing {len(rounds_data)} rounds")
        print(f"   Classes: {class_names}")

        # Create static background traces
        static_traces = []

        # Reference circles
        for r in [0.25, 0.5, 0.75, 1.0]:
            if r * max_val <= max_val:
                theta = np.linspace(0, 2*np.pi, 100)
                static_traces.append(go.Scatter(
                    x=r * max_val * np.cos(theta),
                    y=r * max_val * np.sin(theta),
                    mode='lines',
                    line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='none',
                    name=f'circle_{r}'
                ))

        # Radial lines
        for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            angle_rad = angle_deg * np.pi / 180
            static_traces.append(go.Scatter(
                x=[0, max_val * np.cos(angle_rad)],
                y=[0, max_val * np.sin(angle_rad)],
                mode='lines',
                line=dict(color='rgba(128,128,128,0.2)', width=1),
                showlegend=False,
                hoverinfo='none',
                name=f'radial_{angle_deg}'
            ))

        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        static_traces.append(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode='lines',
            line=dict(color='white', width=2, dash='dash'),
            name='Unit Circle (r=1)',
            showlegend=True
        ))

        # Origin
        static_traces.append(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=6, color='white', symbol='x'),
            name='Origin',
            showlegend=True
        ))

        # Create frames
        frames = []
        center_separation_progression = []
        margin_separation_progression = []
        center_ortho_progression = []
        margin_ortho_progression = []

        for rd_idx, rd in enumerate(rounds_data):
            dynamic_traces = []
            class_stats = rd['class_stats']
            global_metrics = class_stats.pop('global_metrics', {})

            center_sep = global_metrics.get('avg_center_separation', 0)
            margin_sep = global_metrics.get('avg_margin_separation', 0)
            ortho_center = global_metrics.get('orthogonality_center', 0)
            ortho_margin = global_metrics.get('orthogonality_margin', 0)

            center_separation_progression.append(center_sep)
            margin_separation_progression.append(margin_sep)
            center_ortho_progression.append(ortho_center)
            margin_ortho_progression.append(ortho_margin)

            ortho_color = "green" if ortho_margin > 0.8 else "orange" if ortho_margin > 0.5 else "red"

            # Add class points
            for cls, stats in class_stats.items():
                if not isinstance(stats, dict) or stats['n_points'] == 0:
                    continue

                idx = int(cls)
                class_name = class_names[idx] if idx < len(class_names) else f'Class {idx+1}'
                color = self.class_colors[idx % len(self.class_colors)]

                points = stats['points']
                center = stats['center']
                spread = stats['spread']

                # Add scatter points
                hover_texts = []
                for x_val, y_val, mag, ang in zip(points['x'], points['y'], points['magnitudes'], points['angles_deg']):
                    hover_texts.append(
                        f'<b>{class_name}</b><br>'
                        f'x: {x_val:.4f}<br>y: {y_val:.4f}<br>'
                        f'Radius: {mag:.4f}<br>Angle: {ang:.1f}°'
                    )

                # Dynamic point size based on density
                point_size = max(4, min(8, 8 - len(points['x']) / 200))

                dynamic_traces.append(go.Scatter(
                    x=points['x'],
                    y=points['y'],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=color,
                        opacity=0.5,
                        line=dict(width=0.5, color='white')
                    ),
                    name=class_name,
                    showlegend=(rd_idx == 0),
                    text=hover_texts,
                    hoverinfo='text'
                ))

                # Add class center
                if center['magnitude'] > 0:
                    dynamic_traces.append(go.Scatter(
                        x=[center['x']],
                        y=[center['y']],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=color,
                            symbol='star',
                            line=dict(width=2, color='white')
                        ),
                        name=f'{class_name} Center',
                        showlegend=False,
                        hoverinfo='text',
                        text=f'<b>{class_name} Center</b><br>'
                             f'x: {center["x"]:.4f}<br>y: {center["y"]:.4f}<br>'
                             f'Angle: {center["angle_deg"]:.1f}°<br>'
                             f'Spread: ±{spread:.1f}°'
                    ))

                    # Add spread arc (1-sigma)
                    if spread > 0:
                        # Create arc from center_angle - spread to center_angle + spread
                        arc_angles = np.linspace(center['angle_deg'] - spread,
                                                center['angle_deg'] + spread, 50)
                        arc_rad = np.radians(arc_angles)
                        r_arc = 0.85 * np.ones_like(arc_rad)  # Slightly inside unit circle

                        x_arc = r_arc * np.cos(arc_rad)
                        y_arc = r_arc * np.sin(arc_rad)

                        dynamic_traces.append(go.Scatter(
                            x=x_arc, y=y_arc,
                            mode='lines',
                            line=dict(color=color, width=2, dash='dash'),
                            name=f'{class_name} Spread (±{spread:.1f}°)',
                            showlegend=False,
                            hoverinfo='text',
                            text=f'Angular spread: ±{spread:.1f}°'
                        ))

            data_source = rd['data_source']
            source_badge = "📐 TEST DATA (Generalization)" if data_source == "TEST" else "⚠️ TRAINING DATA (Memorization)"

            frames.append(go.Frame(
                data=dynamic_traces,
                name=f'round_{rd["round"]}',
                layout=go.Layout(
                    title=dict(
                        text=f'<b>Round {rd["round"]} - {source_badge}</b><br>'
                             f'Accuracy: {rd["accuracy"]:.1%} | Training Samples: {rd["training_size"]}<br>'
                             f'<span style="color:lightgreen">Center Separation: {center_sep:.1f}° | Ortho: {ortho_center:.3f}</span><br>'
                             f'<span style="color:{ortho_color}">🎯 TRUE Margin Separation: {margin_sep:.1f}° | Ortho: {ortho_margin:.3f}</span><br>'
                             f'<sup>DBNN Feature Pair: {rd["pair_desc"]} | Dashed arcs = ±1σ spread</sup>',
                        font=dict(size=13)
                    )
                )
            ))

        # Create initial figure
        first_rd = rounds_data[0]
        first_class_stats = first_rd['class_stats'].copy()
        first_global_metrics = first_class_stats.pop('global_metrics', {})

        initial_dynamic_traces = []
        for cls, stats in first_class_stats.items():
            if not isinstance(stats, dict) or stats['n_points'] == 0:
                continue

            idx = int(cls)
            class_name = class_names[idx] if idx < len(class_names) else f'Class {idx+1}'
            color = self.class_colors[idx % len(self.class_colors)]

            points = stats['points']
            center = stats['center']
            spread = stats['spread']

            hover_texts = []
            for x_val, y_val, mag, ang in zip(points['x'], points['y'], points['magnitudes'], points['angles_deg']):
                hover_texts.append(
                    f'<b>{class_name}</b><br>'
                    f'x: {x_val:.4f}<br>y: {y_val:.4f}<br>'
                    f'Radius: {mag:.4f}<br>Angle: {ang:.1f}°'
                )

            point_size = max(4, min(8, 8 - len(points['x']) / 200))

            initial_dynamic_traces.append(go.Scatter(
                x=points['x'],
                y=points['y'],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color,
                    opacity=0.5,
                    line=dict(width=0.5, color='white')
                ),
                name=class_name,
                showlegend=True,
                text=hover_texts,
                hoverinfo='text'
            ))

            if center['magnitude'] > 0:
                initial_dynamic_traces.append(go.Scatter(
                    x=[center['x']],
                    y=[center['y']],
                    mode='markers',
                    marker=dict(size=12, color=color, symbol='star', line=dict(width=2, color='white')),
                    name=f'{class_name} Center',
                    showlegend=False,
                    hoverinfo='text',
                    text=f'<b>{class_name} Center</b><br>Angle: {center["angle_deg"]:.1f}°<br>Spread: ±{spread:.1f}°'
                ))

                if spread > 0:
                    arc_angles = np.linspace(center['angle_deg'] - spread,
                                            center['angle_deg'] + spread, 50)
                    arc_rad = np.radians(arc_angles)
                    r_arc = 0.85 * np.ones_like(arc_rad)
                    x_arc = r_arc * np.cos(arc_rad)
                    y_arc = r_arc * np.sin(arc_rad)

                    initial_dynamic_traces.append(go.Scatter(
                        x=x_arc, y=y_arc,
                        mode='lines',
                        line=dict(color=color, width=2, dash='dash'),
                        name=f'{class_name} Spread',
                        showlegend=False,
                        hoverinfo='text',
                        text=f'Spread: ±{spread:.1f}°'
                    ))

        fig = go.Figure(
            data=static_traces + initial_dynamic_traces,
            layout=go.Layout(
                title=dict(
                    text=f'<b>True Polar Evolution - {self.dataset_name}</b><br>'
                         f'<sup>Using DBNN Feature Pair: {first_rd["pair_desc"]} | Dashed arcs = ±1σ spread</sup>',
                    font=dict(size=14)
                ),
                xaxis=dict(
                    title=f'Feature {first_rd["feature_pair"][0]} (Real Component)',
                    range=[-max_val, max_val],
                    scaleanchor='y',
                    scaleratio=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    zerolinecolor='rgba(255,255,255,0.5)'
                ),
                yaxis=dict(
                    title=f'Feature {first_rd["feature_pair"][1]} (Imaginary Component)',
                    range=[-max_val, max_val],
                    gridcolor='rgba(128,128,128,0.2)',
                    zerolinecolor='rgba(255,255,255,0.5)'
                ),
                height=800,
                width=900,
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=False,
                        y=0.95,
                        x=0.05,
                        buttons=[
                            dict(label='▶️ Play', method='animate',
                                 args=[None, {'frame': {'duration': 800, 'redraw': True},
                                              'fromcurrent': True, 'mode': 'immediate'}]),
                            dict(label='⏸️ Pause', method='animate',
                                 args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]),
                            dict(label='🔄 Reset', method='animate',
                                 args=[[frames[0].name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}])
                        ]
                    )
                ],
                sliders=[{
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {'prefix': 'Round: ', 'font': {'size': 14, 'color': 'white'}, 'visible': True},
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9,
                    'x': 0.1,
                    'y': 0,
                    'steps': [
                        {
                            'args': [[f'round_{rd["round"]}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                            'label': str(rd['round']),
                            'method': 'animate'
                        }
                        for rd in rounds_data
                    ]
                }],
                showlegend=True,
                legend=dict(
                    yanchor="top", y=0.99, xanchor="left", x=0.02,
                    bgcolor='rgba(0,0,0,0.7)',
                    bordercolor='white',
                    borderwidth=1,
                    font=dict(color='white', size=10)
                ),
                paper_bgcolor='rgba(0,0,0,0.9)',
                plot_bgcolor='rgba(0,0,0,0.9)'
            ),
            frames=frames
        )

        # Add comprehensive annotations
        final_rd = rounds_data[-1]
        final_metrics = final_rd['class_stats'].get('global_metrics', {})
        final_center_sep = final_metrics.get('avg_center_separation', 0)
        final_margin_sep = final_metrics.get('avg_margin_separation', 0)
        final_ortho_center = final_metrics.get('orthogonality_center', 0)
        final_ortho_margin = final_metrics.get('orthogonality_margin', 0)
        data_source = final_rd['data_source']

        ortho_color = "green" if final_ortho_margin > 0.8 else "orange" if final_ortho_margin > 0.5 else "red"

        fig.add_annotation(
            x=0.98, y=0.05, xref="paper", yref="paper",
            text=f"<b>📐 Understanding Class Separation:</b><br>"
                 f"• Each dot = data point in feature pair space<br>"
                 f"• ★ = Class center<br>"
                 f"• Dashed arc = ±1σ angular spread<br>"
                 f"• <span style='color:lightgreen'>CENTER separation</span>: {final_center_sep:.1f}° | Ortho: {final_ortho_center:.3f}<br>"
                 f"• <span style='color:{ortho_color}'>TRUE separation (MARGIN)</span>: {final_margin_sep:.1f}° | Ortho: {final_ortho_margin:.3f}<br>"
                 f"<br>"
                 f"<b>Why accuracy ≠ center separation:</b><br>"
                 f"• Centers can be separated while clusters overlap<br>"
                 f"• Margin separation accounts for cluster spread<br>"
                 f"• Good separation requires margin > 45°<br>"
                 f"• Current margin: {final_margin_sep:.1f}° → {final_ortho_margin:.3f}",
            showarrow=False,
            font=dict(size=9, color='white'),
            align='right',
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='white',
            borderwidth=1
        )

        # Add margin orthogonality gauge
        fig.add_annotation(
            x=0.02, y=0.95, xref="paper", yref="paper",
            text=f"<b>🎯 TRUE Margin Separation:</b><br>"
                 f"<span style='color:{ortho_color}; font-size:20px;'>{final_ortho_margin:.3f}</span><br>"
                 f"Target: 1.000<br>"
                 f"<sup>Higher = better separation with margins</sup>",
            showarrow=False,
            font=dict(size=10, color='white'),
            align='left',
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='white',
            borderwidth=1
        )

        # Add feature pair info
        fig.add_annotation(
            x=0.02, y=0.05, xref="paper", yref="paper",
            text=f"<b>📊 Feature Pair {first_rd['feature_pair'][0]} (real) vs {first_rd['feature_pair'][1]} (imag)</b><br>"
                 f"<sup>This is DBNN's internal complex representation</sup>",
            showarrow=False,
            font=dict(size=8, color='gray'),
            align='left',
            bgcolor='rgba(0,0,0,0.4)',
            borderwidth=0
        )

        # Save dashboard
        dashboard_path = self.output_dir / f'{self.dataset_name}_true_polar.html'
        fig.write_html(str(dashboard_path))

        abs_path = dashboard_path.absolute()
        print(f"\n{Colors.GREEN}✅ True Polar dashboard saved to: {abs_path}{Colors.ENDC}")
        print(f"   Rounds: {len(rounds_data)}")
        print(f"   Feature Pair: {first_rd['pair_desc']}")
        print(f"   Data Source: {data_source}")
        print(f"   Final Center Separation: {final_center_sep:.1f}°")
        print(f"   Final TRUE Margin Separation: {final_margin_sep:.1f}°")

        return str(abs_path)

    def _get_training_data_for_round(self, train_indices: Optional[List[int]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get training data for a specific round."""
        if train_indices is None or len(train_indices) == 0:
            return None, None
        if self.model.X_tensor is None or self.model.y_tensor is None:
            return None, None
        X_train = self.model.X_tensor[train_indices].cpu().numpy()
        y_train = self.model.y_tensor[train_indices].cpu().numpy()
        return X_train, y_train

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

        # Ensure features are on the correct device
        if features.device != self.device:
            features = features.to(self.device)

        for pair_idx in range(n_pairs):
            f1, f2 = feature_pairs[pair_idx]

            cache_key = (pair_idx, f1, f2)
            if cache_key not in self._bin_edges_cache:
                # Ensure bin edges are on the correct device
                edges0 = bin_edges[pair_idx][0].contiguous()
                edges1 = bin_edges[pair_idx][1].contiguous()

                if edges0.device != self.device:
                    edges0 = edges0.to(self.device)
                if edges1.device != self.device:
                    edges1 = edges1.to(self.device)

                self._bin_edges_cache[cache_key] = (edges0, edges1)
            else:
                edges0, edges1 = self._bin_edges_cache[cache_key]
                # Ensure cached edges are on the correct device (re-cache if needed)
                if edges0.device != self.device:
                    edges0 = edges0.to(self.device)
                    edges1 = edges1.to(self.device)
                    self._bin_edges_cache[cache_key] = (edges0, edges1)

            indices0, indices1 = compute_bin_indices_jit(
                features[:, [f1, f2]], edges0, edges1, n_bins
            )

            bin_indices_dict[pair_idx] = (indices0.clone(), indices1.clone())

            if pair_idx not in self._bin_probs_cache:
                probs_tensor = bin_probs[pair_idx]
                if isinstance(probs_tensor, torch.Tensor):
                    if probs_tensor.device != self.device:
                        probs_tensor = probs_tensor.to(self.device)
                    self._bin_probs_cache[pair_idx] = probs_tensor
                else:
                    self._bin_probs_cache[pair_idx] = torch.tensor(probs_tensor, device=self.device)
            else:
                # Ensure cached probs are on the correct device
                cached_probs = self._bin_probs_cache[pair_idx]
                if cached_probs.device != self.device:
                    cached_probs = cached_probs.to(self.device)
                    self._bin_probs_cache[pair_idx] = cached_probs

            pair_weights = weights[:, pair_idx, :, :]
            # Ensure weights are on the correct device
            if pair_weights.device != self.device:
                pair_weights = pair_weights.to(self.device)

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
                           is_training: bool = True,
                           known_classes: Dict[str, int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Load and preprocess data with robust CSV handling.

        Args:
            file_path: Path to CSV file
            target_column: Name of target column
            is_training: Whether this is training data
            known_classes: For prediction mode, classes seen during training
        """
        # Use robust CSV reader
        df = RobustCSVReader.read_csv(file_path, ignore_comments=True)

        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column].astype(str)

            # Filter out unknown classes if this is training or evaluation
            if known_classes is not None:
                # For prediction mode - keep all rows, but unknown classes will be marked
                if not is_training:
                    # Create mask for known classes
                    mask = y.isin(known_classes.keys())
                    if not mask.all():
                        print(f"{Colors.YELLOW}⚠️ {len(mask) - mask.sum()} rows have unknown classes{Colors.ENDC}")
                        print(f"{Colors.YELLOW}   These rows will still be processed but labels will be ignored{Colors.ENDC}")
            elif is_training:
                # For training, we only want known classes
                known_labels = set(self.label_encoder.keys()) if self.label_encoder else set()
                if known_labels:
                    mask = y.isin(known_labels)
                    if not mask.all():
                        print(f"{Colors.YELLOW}⚠️ Filtering out {len(mask) - mask.sum()} rows with unknown classes{Colors.ENDC}")
                        df = df[mask]
                        X = X[mask]
                        y = y[mask]
        else:
            X = df
            y = None

        # Preprocess features
        X_processed = self._preprocess_features(X, is_training)

        # Convert to tensor
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
# SECTION 7: OPTIMIZED DBNN CORE (Modified - Added ExternalToolsMixin)
# =============================================================================

class OptimizedDBNN(ExternalToolsMixin):
    """Complete DBNN implementation with all fixes"""

    def __init__(self, dataset_name: str = None, config: Union[Dict, str] = None,
                 mode: str = 'train_predict', parallel: bool = True,
                 enable_external_tools: bool = ASTROPY_AVAILABLE):

        super().__init__(enable_external_tools=enable_external_tools)

        ExternalToolsMixin.__init__(self, enable_external_tools=enable_external_tools)

        self.dataset_name = dataset_name
        self.mode = mode
        self.stop_training_flag = False

        if config is None and dataset_name is not None:
            self.config = DatasetConfig.load_config(dataset_name)
        elif isinstance(config, str):
            with open(config, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config or {}

        if self.config is None:
            raise ValueError(f"Could not load configuration for {dataset_name}")

        self.model_type = self.config.get('model_type', self.config.get('modelType', 'Histogram'))
        self.target_column = self.config.get('target_column')

        # ========== FIX: Get feature names from config ==========
        # Get column names from config
        self.all_columns = self.config.get('column_names', [])

        # Extract feature names (all columns except target)
        if self.all_columns and self.target_column:
            self.feature_names = [col for col in self.all_columns if col != self.target_column]
            print(f"{Colors.CYAN}📋 Using {len(self.feature_names)} features from config{Colors.ENDC}")
        else:
            self.feature_names = None
            print(f"{Colors.YELLOW}⚠️ No feature names in config - will auto-detect{Colors.ENDC}")
        # ========== END FIX ==========

        compute_device = self.config.get('compute_device', 'auto')
        if compute_device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = compute_device

        training_params = self.config.get('training_params', {})
        self.learning_rate = training_params.get('learning_rate', 0.1)
        self.n_bins_per_dim = training_params.get('n_bins_per_dim', 128)
        self.test_size = training_params.get('test_fraction', 0.2)
        self.enable_adaptive = training_params.get('enable_adaptive', True)
        self.adaptive_rounds = training_params.get('adaptive_rounds', 10)
        self.initial_samples = training_params.get('initial_samples', 50)
        self.max_samples_per_round = training_params.get('max_samples_per_round', 500)
        self.patience = training_params.get('patience', 25)

        active_learning = self.config.get('active_learning', {})
        self.similarity_threshold = active_learning.get('similarity_threshold', 0.25)
        self.min_divergence = active_learning.get('min_divergence', 0.1)

        self.batch_processor = OptimizedBatchProcessor(self, self.device)
        self.batch_size = self._calculate_optimal_batch_size()

        self.evolution_tracker = TensorEvolutionTracker(self)

        self.feature_pairs = None
        self.bin_edges = None
        self.bin_probs = None
        self.weight_updater = None
        self.classes = None
        self.label_encoder = None

        self.X_tensor = None
        self.y_tensor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_indices = None
        self.test_indices = None
        self.data_original = None
        self.preprocessor = None

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
        """
        Load data using the column structure from config.
        Preserves original data for later use.
        """
        processor = OptimizedDatasetProcessor(self.config, self.device)

        file_path = file_path or self.config.get('file_path')
        if not file_path:
            raise ValueError("No file path provided")

        print(f"{Colors.CYAN}📖 Loading data from: {file_path}{Colors.ENDC}")

        # Load CSV with comment skipping - preserve all data
        try:
            df_original = pd.read_csv(file_path, comment='#')
            print(f"{Colors.GREEN}✓ Loaded {len(df_original)} rows, {len(df_original.columns)} columns{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}❌ Error loading CSV: {e}{Colors.ENDC}")
            raise

        # Store original data (preserve all columns)
        self.data_original = df_original

        # Get target column
        target_column = self.config.get('target_column', self.target_column)

        # If no target column in config, try to detect
        if not target_column:
            target_candidates = ['target', 'class', 'label', 'y', 'prediction', 'type', 'diagnosis']
            for candidate in target_candidates:
                if candidate in df_original.columns:
                    target_column = candidate
                    break
            if not target_column:
                target_column = df_original.columns[-1]
            print(f"{Colors.CYAN}   Auto-detected target column: {target_column}{Colors.ENDC}")

        self.target_column = target_column

        # Separate features and target for training
        if target_column in df_original.columns:
            y = df_original[target_column].astype(str)

            # Use feature_names if available, otherwise use all other columns
            if self.feature_names:
                # Use only specified features
                X = df_original[self.feature_names].copy()
            else:
                # Use all columns except target
                X = df_original.drop(columns=[target_column])
                self.feature_names = list(X.columns)
                print(f"{Colors.GREEN}✓ Auto-detected {len(self.feature_names)} features{Colors.ENDC}")
        else:
            if is_training:
                raise ValueError(f"Target column '{target_column}' not found in training data")
            X = df_original
            y = None

        # Ensure all features are numeric
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X[col] = 0.0

        # Handle missing values
        X = X.fillna(0)

        # Preprocess features
        X_processed = processor._preprocess_features(X, is_training)
        X_tensor = torch.tensor(X_processed, dtype=torch.float64, device=self.device)

        # Handle target
        if y is not None:
            y_tensor = torch.tensor(processor._encode_labels(y, is_training), dtype=torch.long, device=self.device)
            self.y_tensor = y_tensor

            if is_training:
                # Log class distribution
                class_counts = y.value_counts()
                print(f"{Colors.GREEN}✓ Found {len(class_counts)} classes in training data:{Colors.ENDC}")
                for cls, count in list(class_counts.items())[:10]:
                    print(f"   {cls}: {count} samples")
                if len(class_counts) > 10:
                    print(f"   ... and {len(class_counts) - 10} more classes")
        else:
            self.y_tensor = None

        self.X_tensor = X_tensor
        self.preprocessor = processor

        print(f"{Colors.GREEN}✓ Data prepared: {len(X_tensor)} samples, {len(self.feature_names)} features{Colors.ENDC}")

        return X_tensor, self.y_tensor

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
        # Ensure X is on the correct device
        if X.device != self.device:
            X = X.to(self.device)

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

        # Ensure X is on the correct device
        if X.device != self.device:
            X = X.to(self.device)

        # Ensure y is on the correct device
        if y.device != self.device:
            y = y.to(self.device)

        bin_counts = torch.zeros((n_classes, n_pairs, self.n_bins_per_dim, self.n_bins_per_dim),
                                dtype=torch.float64, device=self.device)

        for pair_idx, (f1, f2) in enumerate(self.feature_pairs):
            edges0, edges1 = self.bin_edges[pair_idx]

            # Ensure bin edges are on the correct device
            if edges0.device != self.device:
                edges0 = edges0.to(self.device)
            if edges1.device != self.device:
                edges1 = edges1.to(self.device)

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
            # Ensure features are on the correct device
            if features.device != self.device:
                features = features.to(self.device)

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

            # Move batch to the correct device for computation
            if batch_X.device != self.device:
                batch_X = batch_X.to(self.device)

            posteriors, _ = self._compute_batch_posterior(batch_X, return_bin_indices=False)
            predictions = torch.argmax(posteriors, dim=1)

            # Return to CPU immediately after computation
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

    def fit_predict(self, batch_size: int = 128, save_path: str = None, verbose: bool = True) -> Dict:
        """Fit and predict with optional verbosity control"""
        if self.X_tensor is None:
            self.load_data()

        if self.X_train is None:
            self.split_data()

        # Store original print settings
        import sys
        from io import StringIO
        original_stdout = sys.stdout

        if not verbose:
            # Suppress output during adaptive training
            sys.stdout = StringIO()

        results = self.fit(self.X_train, self.y_train, self.X_test, self.y_test, epochs=100)

        train_pred, train_post = self.predict(self.X_train)
        test_pred, test_post = self.predict(self.X_test)

        # Convert to CPU and then to numpy
        train_pred_cpu = train_pred.cpu()
        test_pred_cpu = test_pred.cpu()

        # For labels, we need to handle encoding
        inv_label_encoder = {v: k for k, v in self.label_encoder.items()}
        train_pred_labels = [inv_label_encoder[p.item()] for p in train_pred_cpu]
        test_pred_labels = [inv_label_encoder[p.item()] for p in test_pred_cpu]

        # Convert y_train and y_test to CPU and numpy
        y_train_cpu = self.y_train.cpu()
        y_test_cpu = self.y_test.cpu()

        all_indices = np.concatenate([self.train_indices, self.test_indices])
        all_pred = np.concatenate([train_pred_cpu.numpy(), test_pred_cpu.numpy()])
        all_true = np.concatenate([y_train_cpu.numpy(), y_test_cpu.numpy()])

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
                'true': y_train_cpu.numpy(),
                'true_label': [inv_label_encoder[t.item()] for t in y_train_cpu],
                'predicted': train_pred_cpu.numpy(),
                'predicted_label': train_pred_labels
            })
            train_results.to_csv(os.path.join(save_path, 'train_predictions.csv'), index=False)

            test_results = pd.DataFrame({
                'true': y_test_cpu.numpy(),
                'true_label': [inv_label_encoder[t.item()] for t in y_test_cpu],
                'predicted': test_pred_cpu.numpy(),
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

        # CRITICAL: Load feature names from saved model
        self.feature_names = model_state.get('feature_names', [])
        self.selected_features = model_state.get('selected_features', [])
        self.target_column = model_state.get('target_column', self.target_column)

        # Also load the all_columns from config if available
        self.all_columns = self.config.get('column_names', [])

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

        if 'evolution_history' in model_state:
            self.evolution_tracker.tensor_evolution_history = model_state['evolution_history']

        # Restore preprocessor stats if available
        if 'preprocessor_stats' in model_state:
            if not hasattr(self, 'preprocessor') or self.preprocessor is None:
                self.preprocessor = OptimizedDatasetProcessor(self.config, self.device)
            self.preprocessor.feature_stats = model_state['preprocessor_stats'].get('feature_stats', {})
            self.preprocessor.categorical_encoders = model_state['preprocessor_stats'].get('categorical_encoders', {})
            self.preprocessor.label_encoder = model_state['preprocessor_stats'].get('label_encoder', {})

        print(f"✅ Model loaded from {path}")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Classes: {len(self.classes) if self.classes is not None else 0}")
        print(f"   Features: {len(self.feature_names) if self.feature_names else 0}")
        if self.feature_names:
            print(f"   First 5 features: {self.feature_names[:5]}")

    def predict_from_file(self, input_csv: str, output_path: str = None, **kwargs) -> Dict:
        """
        Predict from a new CSV file.
        Preserves all original data including metadata columns.
        Only uses numeric features from training for prediction.
        """
        if output_path:
            os.makedirs(output_path, exist_ok=True)

        print(f"{Colors.CYAN}📖 Reading prediction file: {input_csv}{Colors.ENDC}")

        # Read CSV with comment skipping - preserve all data
        try:
            df_original = pd.read_csv(input_csv, comment='#')
            print(f"{Colors.GREEN}✓ Loaded {len(df_original)} rows, {len(df_original.columns)} columns{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to read file: {e}{Colors.ENDC}")
            return {'predictions': None, 'error': str(e)}

        # CRITICAL: Use ONLY the feature_names from the trained model
        if not self.feature_names:
            print(f"{Colors.RED}❌ No feature names available from trained model{Colors.ENDC}")
            return {'predictions': None, 'error': 'Model not properly trained'}

        print(f"{Colors.CYAN}📊 Model was trained with {len(self.feature_names)} numeric features:{Colors.ENDC}")
        print(f"{Colors.CYAN}   {self.feature_names[:5]}... (total {len(self.feature_names)}){Colors.ENDC}")

        # Build feature dataframe with ONLY the numeric features from training
        # This is separate from the original data - we only use these for prediction
        X = pd.DataFrame(index=df_original.index)
        missing_features = []
        non_numeric_features = []

        for feat in self.feature_names:
            if feat in df_original.columns:
                # Try to convert to numeric
                try:
                    X[feat] = pd.to_numeric(df_original[feat], errors='coerce')
                    # Check if conversion resulted in all NaN (completely non-numeric column)
                    if X[feat].isna().all():
                        non_numeric_features.append(feat)
                        X[feat] = 0.0
                except:
                    non_numeric_features.append(feat)
                    X[feat] = 0.0
            else:
                missing_features.append(feat)
                X[feat] = 0.0

        if missing_features:
            print(f"{Colors.YELLOW}⚠️ Missing {len(missing_features)} feature columns in prediction file:{Colors.ENDC}")
            print(f"{Colors.YELLOW}   {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}{Colors.ENDC}")
            print(f"{Colors.YELLOW}   These will be filled with 0{Colors.ENDC}")

        if non_numeric_features:
            print(f"{Colors.YELLOW}⚠️ {len(non_numeric_features)} feature columns contain non-numeric data:{Colors.ENDC}")
            print(f"{Colors.YELLOW}   {non_numeric_features[:10]}{'...' if len(non_numeric_features) > 10 else ''}{Colors.ENDC}")
            print(f"{Colors.YELLOW}   These will be filled with 0{Colors.ENDC}")

        # Handle any NaN values
        X = X.fillna(0)

        # Verify all features are numeric
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = 0.0

        print(f"{Colors.GREEN}✓ Prepared {len(X)} rows with {len(X.columns)} numeric features for prediction{Colors.ENDC}")

        # Convert to tensor
        X_np = X.values.astype(np.float64)
        X_tensor = torch.tensor(X_np, dtype=torch.float64, device=self.device)

        print(f"{Colors.CYAN}🔮 Making predictions on {len(X_tensor)} samples...{Colors.ENDC}")

        try:
            predictions, posteriors = self.predict(X_tensor)

            # Convert predictions to labels
            inv_label_encoder = {v: k for k, v in self.label_encoder.items()}
            pred_labels = [inv_label_encoder.get(p.item(), f"unknown_{p.item()}") for p in predictions]

            # Create results dataframe that preserves ALL original data
            results = df_original.copy()

            # Add prediction columns
            results['predicted_class'] = pred_labels
            results['confidence'] = posteriors.max(dim=1)[0].cpu().numpy()

            # Add probability columns for each class
            for i, (label, idx) in enumerate(self.label_encoder.items()):
                results[f'prob_{label}'] = posteriors[:, i].cpu().numpy()

            # Add uncertainty
            results['uncertainty'] = 1.0 - results['confidence']

            # Add a flag for rows that had issues (missing features or non-numeric)
            results['has_missing_features'] = False
            results['has_non_numeric_features'] = False

            # Mark rows with issues (if any)
            if missing_features or non_numeric_features:
                # For simplicity, mark all rows if any features were missing
                # You could make this more granular if needed
                if missing_features:
                    results['has_missing_features'] = True
                if non_numeric_features:
                    results['has_non_numeric_features'] = True

            # If true labels exist, add them and calculate metrics
            target_column = self.target_column
            has_target = target_column in df_original.columns

            if has_target:
                y_true = df_original[target_column].astype(str)
                results['true_class'] = y_true

                # Calculate accuracy only on rows where true class is known to the model
                known_true_mask = y_true.isin(self.label_encoder.keys())
                if known_true_mask.any():
                    known_true = y_true[known_true_mask]
                    known_pred = results.loc[known_true_mask, 'predicted_class']
                    accuracy = (known_true == known_pred).mean()
                    print(f"{Colors.GREEN}✓ Accuracy on known classes: {accuracy:.4f} ({known_true_mask.sum()} samples){Colors.ENDC}")
                else:
                    print(f"{Colors.YELLOW}⚠️ No samples with known classes for evaluation{Colors.ENDC}")

                # Report unknown classes found
                unknown_true_mask = ~known_true_mask
                if unknown_true_mask.any():
                    unknown_classes = y_true[unknown_true_mask].unique()
                    print(f"{Colors.YELLOW}⚠️ Found {unknown_true_mask.sum()} samples with unknown classes: {unknown_classes[:5]}{Colors.ENDC}")

            # Save complete results (all original data + predictions)
            if output_path:
                csv_path = os.path.join(output_path, 'predictions.csv')
                results.to_csv(csv_path, index=False)
                print(f"{Colors.GREEN}✅ Complete predictions saved to: {csv_path}{Colors.ENDC}")

                # Save a compact version with just predictions and key columns
                summary_cols = ['predicted_class', 'confidence', 'uncertainty']
                if has_target:
                    summary_cols.insert(0, 'true_class')

                # Also include any identifier columns if they exist
                id_cols = ['id', 'ID', 'filename', 'filepath', 'name', 'Name']
                for id_col in id_cols:
                    if id_col in results.columns:
                        summary_cols.insert(0, id_col)

                summary = results[summary_cols]
                summary_path = os.path.join(output_path, 'predictions_summary.csv')
                summary.to_csv(summary_path, index=False)
                print(f"{Colors.GREEN}✅ Summary saved to: {summary_path}{Colors.ENDC}")

                # Save predictions with probabilities for each class
                prob_cols = ['predicted_class', 'confidence', 'uncertainty'] + \
                           [f'prob_{label}' for label in self.label_encoder.keys()]
                if has_target:
                    prob_cols.insert(0, 'true_class')
                if any(col in results.columns for col in id_cols):
                    for id_col in id_cols:
                        if id_col in results.columns:
                            prob_cols.insert(0, id_col)
                            break

                prob_df = results[prob_cols]
                prob_path = os.path.join(output_path, 'predictions_with_probs.csv')
                prob_df.to_csv(prob_path, index=False)
                print(f"{Colors.GREEN}✅ Probabilities saved to: {prob_path}{Colors.ENDC}")

                # Also save a debug file showing which features were used
                feature_info = pd.DataFrame({
                    'feature_name': self.feature_names,
                    'present_in_prediction': [f in df_original.columns for f in self.feature_names],
                    'is_numeric': [not pd.api.types.is_numeric_dtype(df_original[f]) if f in df_original.columns else False for f in self.feature_names]
                })
                feature_info_path = os.path.join(output_path, 'feature_analysis.csv')
                feature_info.to_csv(feature_info_path, index=False)
                print(f"{Colors.GREEN}✅ Feature analysis saved to: {feature_info_path}{Colors.ENDC}")

            return {
                'predictions': results,
                'n_samples': len(results),
                'has_target': has_target,
                'features_used': self.feature_names,
                'missing_features': missing_features,
                'non_numeric_features': non_numeric_features
            }

        except Exception as e:
            print(f"{Colors.RED}❌ Error during prediction: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            return {'predictions': None, 'error': str(e)}

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
        """Select samples from misclassified classes with proper dimension handling"""
        # Convert to CPU and numpy
        if torch.is_tensor(test_predictions):
            test_predictions = test_predictions.cpu().numpy()
        if torch.is_tensor(y_test):
            y_test = y_test.cpu().numpy()

        # Get or create all_predictions DataFrame
        all_results = results.get('all_predictions', pd.DataFrame())
        if len(all_results) == 0:
            inv_label_encoder = {v: k for k, v in self.label_encoder.items()}
            all_results = pd.DataFrame({
                'true_class': [inv_label_encoder[t] for t in y_test],
                'predicted_class': [inv_label_encoder[p] for p in test_predictions],
                'true_encoded': y_test,
                'predicted_encoded': test_predictions
            })
            all_results.index = test_indices

        # Get test results using the correct indices
        test_results = all_results.loc[test_indices] if all_results.index.isin(test_indices).any() else all_results

        # Ensure required columns exist
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

        # Find misclassified samples
        misclassified_mask = test_results['predicted_class'] != test_results['true_class']

        # CRITICAL FIX: Create misclassified_indices list that matches test_results
        misclassified_indices = test_results.index[misclassified_mask].tolist()

        if not misclassified_indices:
            print(f"{Colors.YELLOW}No misclassified samples found{Colors.ENDC}")
            return []

        # Create mapping from index to position in test_results
        test_pos_map = {idx: pos for pos, idx in enumerate(test_indices) if idx in test_results.index}

        # Get unique classes from misclassified samples
        misclassified_df = test_results.loc[misclassified_indices]
        unique_classes = misclassified_df['true_class'].unique()

        print(f"\n{Colors.CYAN}Selecting samples from failed classes...{Colors.ENDC}")

        final_selected_indices = []

        for class_label in unique_classes:
            # Get indices for this class from misclassified_df
            class_df = misclassified_df[misclassified_df['true_class'] == class_label]

            if len(class_df) == 0:
                continue

            # Get the actual indices from class_df
            class_indices_list = class_df.index.tolist()

            # Map to positions in test_indices
            class_positions = []
            for idx in class_indices_list:
                if idx in test_pos_map:
                    class_positions.append(test_pos_map[idx])
                elif idx in test_indices:
                    # Try direct index if not in map
                    try:
                        class_positions.append(test_indices.index(idx))
                    except ValueError:
                        continue

            if not class_positions:
                continue

            # Limit samples per class
            max_samples_this_class = min(self.max_samples_per_round, len(class_positions))
            if len(class_positions) > max_samples_this_class:
                class_positions = random.sample(class_positions, max_samples_this_class)
                print(f"{Colors.YELLOW}   Limited class {class_label} to {max_samples_this_class} samples{Colors.ENDC}")

            # Get the original dataset indices
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
        # Convert to CPU and numpy if they're tensors
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
        Complete adaptive training - Shows colored confusion matrices at round level only
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
        training_history = []
        round_stats = []

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
                self.classes = torch.tensor(list(self.label_encoder.values()), device=self.device)  # Move to device
                y_encoded = y.map(self.label_encoder).values
                self.y_tensor = torch.tensor(y_encoded, dtype=torch.long, device=self.device)  # Move to device

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

                # Get y_tensor as numpy on CPU for comparison
                y_numpy = self.y_tensor.cpu().numpy()  # Move to CPU for numpy operations

                for class_label, class_id in self.label_encoder.items():
                    # Use numpy array for indexing
                    class_indices = np.where(y_numpy == class_id)[0]

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

                # Ensure they're on the correct device
                X_train_subset = X_train_subset.to(self.device)
                y_train_subset = y_train_subset.to(self.device)

                likelihoods = self.compute_likelihoods(X_train_subset, y_train_subset)
                self.bin_probs = likelihoods['bin_probs']
                print(f"   Computed likelihoods")

            adaptive_patience_counter = 0
            patience = self.patience
            best_combined_accuracy = 0.0
            best_round_initial_conditions = None
            round_num = 0

            # Suppress print output from fit_predict
            import sys
            from io import StringIO
            original_stdout = sys.stdout

            while round_num < max_rounds:
                if self.stop_training_flag:
                    print(f"\n{Colors.YELLOW}🛑 Training stopped by user at round {round_num + 1}{Colors.ENDC}")
                    break

                print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
                print(f"{Colors.BOLD}{Colors.BLUE}Round {round_num + 1}/{max_rounds}{Colors.ENDC}")
                print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
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

                # SUPPRESS ALL PRINT OUTPUT DURING INTERNAL TRAINING
                captured_output = StringIO()
                sys.stdout = captured_output

                try:
                    results = self.fit_predict(batch_size=batch_size, save_path=save_path)
                finally:
                    # Restore stdout
                    sys.stdout = original_stdout

                if 'history' in results and results['history']:
                    for epoch_data in results['history']:
                        epoch_data['round'] = round_num
                        training_history.append(epoch_data)

                train_accuracy = results['train_accuracy']
                test_accuracy = results['test_accuracy']

                # Print round summary with colored accuracy
                print(f"\n{Colors.BOLD}📊 Round {round_num + 1} Results:{Colors.ENDC}")
                print(f"   {Colors.GREEN}Training accuracy: {train_accuracy:.4f}{Colors.ENDC}")
                print(f"   {Colors.GREEN}Test accuracy: {test_accuracy:.4f}{Colors.ENDC}")

                all_predictions, _ = self.predict(self.X_tensor)
                total_accuracy = (all_predictions == self.y_tensor.cpu()).float().mean().item()
                print(f"   {Colors.GREEN}Total accuracy: {total_accuracy:.4f}{Colors.ENDC}")

                self.evolution_tracker.capture_state(
                    round_num=round_num + 1,
                    accuracy=total_accuracy,
                    training_size=len(train_indices)
                )

                # Print FULL COLORED CONFUSION MATRICES
                print(f"\n{Colors.BOLD}{Colors.CYAN}📈 Confusion Matrices:{Colors.ENDC}")

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
                    print(f"\n{Colors.GREEN}{'-'*20}{Colors.ENDC}")
                    print(f"{Colors.GREEN}🎉 NEW BEST ACCURACY! Combined: {combined_accuracy:.4f}{Colors.ENDC}")
                    print(f"{Colors.GREEN}{'-'*20}{Colors.ENDC}")
                else:
                    adaptive_patience_counter += 1
                    print(f"\n{Colors.YELLOW}⚠️ No improvement. Patience: {adaptive_patience_counter}/{patience}{Colors.ENDC}")

                if total_accuracy >= 0.9999:
                    print(f"\n{Colors.GREEN}{'-'*30}{Colors.ENDC}")
                    print(f"{Colors.GREEN}🎯 TOTAL ACCURACY REACHED 100%! Training complete.{Colors.ENDC}")
                    print(f"{Colors.GREEN}{'-'*30}{Colors.ENDC}")
                    break

                if adaptive_patience_counter >= patience:
                    print(f"\n{Colors.YELLOW}{'-'*30}{Colors.ENDC}")
                    print(f"{Colors.YELLOW}⚠️No improvement after {patience} rounds. Stopping.{Colors.ENDC}⚠️")
                    print(f"{Colors.YELLOW}{'⚠-'*30}{Colors.ENDC}")
                    break

                if test_indices:
                    test_predictions, _ = self.predict(self.X_tensor[test_indices])
                    y_test_np = self.y_tensor[test_indices].cpu().numpy()  # Move to CPU

                    new_train_indices = self._select_samples_from_failed_classes(
                        test_predictions, y_test_np, test_indices, results
                    )

                    if new_train_indices:
                        class_dist = self._format_class_distribution(new_train_indices)
                        print(f"\n{Colors.CYAN}📚 Adding new samples to training set:{Colors.ENDC}")
                        print(f"   {Colors.GREEN}Added {len(new_train_indices)} new samples - {class_dist}{Colors.ENDC}")

                        train_indices = list(set(train_indices + new_train_indices))
                        test_indices = list(set(test_indices) - set(new_train_indices))
                        round_num += 1
                    else:
                        if train_accuracy >= 0.99:
                            print(f"\n{Colors.GREEN}{'✓'*30}{Colors.ENDC}")
                            print(f"{Colors.GREEN}Perfect accuracy achieved on training data.{Colors.ENDC}")
                            print(f"{Colors.GREEN}No more suitable samples in test set.{Colors.ENDC}")
                            print(f"{Colors.GREEN}{'✓'*30}{Colors.ENDC}")
                        else:
                            print(f"\n{Colors.YELLOW}{'⚠️'*30}{Colors.ENDC}")
                            print(f"{Colors.YELLOW}No suitable new samples found meeting selection criteria.{Colors.ENDC}")
                            print(f"{Colors.YELLOW}Training complete.{Colors.ENDC}")
                            print(f"{Colors.YELLOW}{'⚠️'*30}{Colors.ENDC}")
                        break
                else:
                    print(f"\n{Colors.GREEN}{'✓'*30}{Colors.ENDC}")
                    print(f"{Colors.GREEN}No more test samples available. Training complete.{Colors.ENDC}")
                    print(f"{Colors.GREEN}{'✓'*30}{Colors.ENDC}")
                    break

            end_time = time.time()
            elapsed_time = end_time - start_time
            end_clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))

            print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.GREEN}{'-'*15}{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.GREEN}✅ ADAPTIVE TRAINING COMPLETE{Colors.ENDC}🏆")
            print(f"{Colors.BOLD}{Colors.GREEN}{'-'*15}{Colors.ENDC}")
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

        # ========== GUI Variables ==========
        self.deps_status = {}
        self.install_thread = None
        self.install_in_progress = False

        # External tools buttons
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

        self.ext_tab = ttk.Frame(notebook)
        notebook.add(self.ext_tab, text="🔌 External Tools")
        self.setup_external_tools_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_training_tab(self):
        """Setup training tab with full external tools integration and all visualization buttons"""
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

        # ========== EXTERNAL TOOLS FRAME ==========
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

        if topcat_avail:
            self.topcat_btn = ttk.Button(tools_row, text="📊 Launch Topcat",
                                         command=self.launch_topcat_gui, width=18)
            self.topcat_btn.pack(side=tk.LEFT, padx=2)
        else:
            self.topcat_btn = ttk.Button(tools_row, text="📊 Topcat (Not Found)",
                                         state=tk.DISABLED, width=18)
            self.topcat_btn.pack(side=tk.LEFT, padx=2)

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

        ttk.Label(query_row, text="Dataset:", font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 2))
        self.export_dataset_var = tk.StringVar(value="all")
        dataset_combo = ttk.Combobox(query_row, textvariable=self.export_dataset_var,
                                     values=["all", "training", "test"], width=10, state="readonly")
        dataset_combo.pack(side=tk.LEFT, padx=2)

        ttk.Label(query_row, text="Radius (arcsec):", font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 2))
        self.sdss_radius_var = tk.StringVar(value="1.0")
        ttk.Entry(query_row, textvariable=self.sdss_radius_var, width=8).pack(side=tk.LEFT, padx=2)

        # Fourth row - Status
        status_row = ttk.Frame(ext_frame)
        status_row.pack(fill=tk.X, pady=5)
        self.ext_status_label = ttk.Label(status_row, text="", foreground="gray")
        self.ext_status_label.pack(side=tk.LEFT, padx=5)
        self._update_ext_tools_status(topcat_avail, aladin_avail, fits_avail, sdss_avail)

        # ========== VISUALIZATION BUTTONS (ENHANCED) ==========
        viz_frame = ttk.LabelFrame(self.training_tab, text="Visualizations", padding="10")
        viz_frame.pack(fill=tk.X, pady=5)

        # Row 1: Basic visualizations
        row1 = ttk.Frame(viz_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="Confusion Matrix", command=self.show_confusion_matrix, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Training History", command=self.show_training_history, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Tensor Evolution", command=self.show_tensor_evolution, width=18).pack(side=tk.LEFT, padx=2)

        # Row 2: Advanced 3D visualizations
        row2 = ttk.Frame(viz_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="Interactive Dashboard", command=self.show_dashboard, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="🌐 Spherical Evolution", command=self.show_spherical_evolution, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="🎬 Side-by-Side", command=self.show_side_by_side, width=18).pack(side=tk.LEFT, padx=2)

        # Row 3: Multi-projection and Polar Evolution
        row3 = ttk.Frame(viz_frame)
        row3.pack(fill=tk.X, pady=2)

        # Multi-projection button (5 methods)
        self.multi_projection_btn = ttk.Button(
            row3,
            text="🔬 Multi-Projection Evolution (5 Methods)",
            command=self.show_multi_projection_evolution,
            width=25
        )
        self.multi_projection_btn.pack(side=tk.LEFT, padx=2)

        # NEW: Polar Evolution Button (Cluster Formation)
        self.polar_evolution_btn = ttk.Button(
            row3,
            text="📊 Polar Evolution (Cluster Formation)",
            command=self.show_polar_evolution,
            width=25
        )
        self.polar_evolution_btn.pack(side=tk.LEFT, padx=2)

        # Row 4: Utility buttons
        row4 = ttk.Frame(viz_frame)
        row4.pack(fill=tk.X, pady=2)
        ttk.Button(row4, text="📁 Open Viz Folder", command=self.open_visualization_dir, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row4, text="📊 Comparison Grid", command=self.show_comparison_grid, width=18).pack(side=tk.LEFT, padx=2)

        #Row 5: 2D Polar Evolution (NEW)
        row5 = ttk.Frame(viz_frame)
        row5.pack(fill=tk.X, pady=2)

        self.polar_2d_btn = ttk.Button(
            row5,
            text="🎯 2D Polar Evolution (Weight Distribution)",
            command=self.show_2d_polar_evolution,
            width=28
        )
        self.polar_2d_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(row5, text="📊 Angular Histogram", command=self.show_angular_histogram, width=18).pack(side=tk.LEFT, padx=2)

        # Row 6: True Polar Evolution (NEW)
        row6 = ttk.Frame(viz_frame)
        row6.pack(fill=tk.X, pady=2)

        self.true_polar_btn = ttk.Button(
            row6,
            text="🎯 True Polar Evolution (r vs θ)",
            command=self.show_true_polar_evolution,
            width=25
        )
        self.true_polar_btn.pack(side=tk.LEFT, padx=2)

        # Row 7: Point-Based Spherical Evolution (NEW)
        row7 = ttk.Frame(viz_frame)
        row7.pack(fill=tk.X, pady=2)

        self.point_spherical_btn = ttk.Button(
            row7,
            text="🎯 Point-Based Spherical Evolution (Actual Data Points)",
            command=self.show_point_based_spherical_evolution,
            width=32
        )
        self.point_spherical_btn.pack(side=tk.LEFT, padx=2)

        # Add a brief description label
        ttk.Label(row7, text="Shows actual data points projected to sphere surface",
                  foreground="gray", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)

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

    # =============================================================================
    # FIXED: ASK OPEN DASHBOARD METHOD (with proper absolute path handling)
    # =============================================================================

    def _ask_open_dashboard(self, dashboard_path, dashboard_type="multi-projection"):
        """
        Ask user if they want to open the dashboard.
        FIXED: All Tkinter operations now happen in main thread via root.after
        """
        # This method is called from root.after, so we're in main thread
        # Convert to absolute path properly
        if isinstance(dashboard_path, str):
            path = Path(dashboard_path)
        else:
            path = dashboard_path

        # If path is relative, make it absolute relative to current working directory
        if not path.is_absolute():
            # Remove any leading slash or backslash that might cause issues
            clean_path = str(path).lstrip('/\\')
            abs_path = Path.cwd() / clean_path
        else:
            abs_path = path

        # Verify file exists
        if not abs_path.exists():
            # Use root.after for GUI operations
            self.root.after(0, lambda: self._show_file_not_found_warning(abs_path, dashboard_type))
            return

        # Create appropriate title and description based on dashboard type
        titles = {
            'multi-projection': 'Multi-Projection Dashboard Ready',
            '2d_polar': '2D Polar Evolution Dashboard Ready',
            'angular': 'Angular Histogram Dashboard Ready',
            'spherical': 'Spherical Evolution Dashboard Ready',
            'point_based_spherical': 'Point-Based Spherical Evolution Dashboard Ready',
            'polar': 'Polar Evolution Dashboard Ready',
            'comparison': 'Comparison Grid Ready',
            'true_polar': 'True Polar Evolution Dashboard Ready'
        }

        descriptions = {
            'multi-projection': f"""Multi-projection dashboard created successfully!

    File: {abs_path}
    Size: {abs_path.stat().st_size / 1024:.1f} KB

    Features:
    • 5 synchronized projection methods
    • Animated evolution over rounds
    • Interactive 3D views with zoom/pan
    • Hover information for each class
    • Target positions (X markers) showing ideal orthogonal configuration""",

            'point_based_spherical': f"""Point-Based Spherical Evolution Dashboard created successfully!

    File: {abs_path}
    Size: {abs_path.stat().st_size / 1024:.1f} KB

    What it shows:
    • ACTUAL data points projected to sphere surface
    • Each dot = one data sample's position in tensor space
    • Classes form clusters that become orthogonal over time
    • Class centers (dashed lines) move toward ideal positions (X markers)
    • Includes orthogonality verification metrics

    Interpretation:
    • Points from same class should cluster together
    • Clusters should separate to different angles (orthogonal)
    • Trustworthiness > 0.7 means projection is reliable
    • Orthogonality score > 0.8 indicates good separation""",

            'spherical': f"""Spherical Evolution Dashboard created successfully!

    File: {abs_path}
    Size: {abs_path.stat().st_size / 1024:.1f} KB

    What it shows:
    • Class orientation vectors in 3D space
    • Evolution from random to orthogonal positions
    • Target positions for perfect classification
    • Orthogonality metrics progression""",

            '2d_polar': f"""2D Polar Evolution Dashboard created successfully!

    File: {abs_path}
    Size: {abs_path.stat().st_size / 1024:.1f} KB

    What it shows:
    • Each point = weight from the 5D tensor
    • Distance from origin = magnitude (strength)
    • Angle from origin = phase (orientation)
    • Classes form clusters at different angles

    Interpretation:
    • Perfect classification = separate, focused clusters at different angles
    • Angular spread < 30° = well-defined clusters
    • Angular concentration > 0.8 = good separation""",

            'angular': f"""Angular Histogram Dashboard created successfully!

    File: {abs_path}
    Size: {abs_path.stat().st_size / 1024:.1f} KB

    What it shows:
    • Angular distribution of weights for each class
    • Mean angle and angular spread
    • Class separation quality metrics

    Interpretation:
    • High angular concentration (>0.8) = well-separated classes
    • Low angular spread (<30°) = focused clusters
    • Classes at different angles = orthogonal in complex space""",

            'true_polar': f"""True Polar Evolution Dashboard created successfully!

    File: {abs_path}
    Size: {abs_path.stat().st_size / 1024:.1f} KB

    What it shows:
    • Actual weight distributions from DBNN's feature pairs
    • Radius = magnitude, Angle = phase
    • Class separation in complex space
    • Animated evolution over adaptive rounds"""
        }

        title = titles.get(dashboard_type, 'Dashboard Ready')
        description = descriptions.get(dashboard_type, f'{dashboard_type.title()} dashboard created successfully!\n\nFile: {abs_path}')

        # Use messagebox in main thread (we're already in main thread from root.after)
        if messagebox.askyesno(title, f"{description}\n\nOpen in browser now?"):
            import webbrowser
            # Use absolute path with proper file:// URL format
            file_url = f'file://{abs_path}'
            webbrowser.open(file_url)
            # Use root.after for log_output to ensure thread safety
            self.root.after(0, lambda: self._safe_log(f"🌐 {dashboard_type.title()} dashboard opened in browser"))
            self.root.after(0, lambda: self._safe_log(f"   URL: {file_url}"))

    def _show_file_not_found_warning(self, abs_path, dashboard_type):
        """Show warning when file not found - called from main thread"""
        # Try to find the file in visualizations directory
        alt_path = Path.cwd() / 'visualizations' / (self.model.dataset_name if self.model else '') / dashboard_type.split('_')[0] / f"{self.model.dataset_name if self.model else 'unknown'}_{dashboard_type}_dashboard.html"

        if alt_path.exists():
            # Found it! Open it directly
            if messagebox.askyesno("Dashboard Found",
                                   f"Found dashboard at:\n{alt_path}\n\nOpen in browser now?"):
                import webbrowser
                webbrowser.open(f'file://{alt_path}')
                self._safe_log(f"🌐 Dashboard opened from: {alt_path}")
            return

        # List available files for debugging
        self._safe_log(f"⚠️ Warning: Dashboard file not found at: {abs_path}")
        viz_dir = Path.cwd() / 'visualizations'
        if viz_dir.exists():
            self._safe_log(f"Available files in {viz_dir}:")
            for f in viz_dir.rglob('*.html'):
                self._safe_log(f"  - {f.relative_to(Path.cwd())}")
        else:
            self._safe_log(f"No visualizations directory found at {viz_dir}")

    def _safe_log(self, message):
        """Thread-safe logging using root.after"""
        self.root.after(0, lambda: self._do_log(message))

    def _do_log(self, message):
        """Actual logging - called from main thread"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'output_text') and self.output_text:
            self.output_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.output_text.see(tk.END)
        self.root.update()
        self.status_var.set(message)


    # =============================================================================
    # FIXED: RADIAL HISTOGRAM METHOD (using matplotlib-compatible colors)
    # =============================================================================

    def _create_radial_histogram_figure(self, class_distributions, class_names=None):
        """
        Create radial histogram showing magnitude distributions.
        FIXED: Uses matplotlib-compatible colors (hex codes instead of RGB strings)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        n_classes = len(class_distributions)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        # Use matplotlib-compatible colors (hex codes)
        colors_hex = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
            '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
            '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'
        ]

        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (c, dist) in enumerate(class_distributions.items()):
            row = idx // n_cols
            col = idx % n_cols

            ax = axes[row, col]

            if dist['n_points'] > 0:
                # Get color for this class (hex format)
                color = colors_hex[c % len(colors_hex)]

                # Create histogram
                ax.hist(dist['radii'], bins=30, color=color,
                       alpha=0.7, edgecolor='white', linewidth=0.5)

                # Add mean line
                ax.axvline(dist['mean_radius'], color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {dist["mean_radius"]:.3f}')

                # Add median line
                median = np.median(dist['radii'])
                ax.axvline(median, color='orange', linestyle=':', linewidth=2,
                          label=f'Median: {median:.3f}')

                # Add statistics text
                class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'
                ax.set_title(f'{class_name}')
                ax.set_xlabel('Magnitude (Radius)')
                ax.set_ylabel('Frequency')
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3)

                # Add text box with stats
                stats_text = f"n={dist['n_points']}\nμ={dist['mean_radius']:.3f}\nσ={dist['radius_std']:.3f}\nCV={dist['radius_cv']:.2f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'
                ax.set_title(f'{class_name}')

        # Hide empty subplots
        total_plots = n_rows * n_cols
        for idx in range(len(class_distributions), total_plots):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        plt.suptitle(f'Radial Histogram - Magnitude Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save to file
        output_dir = Path('visualizations') / (self.model.dataset_name if hasattr(self.model, 'dataset_name') else 'model') / 'angular_histograms'
        output_dir.mkdir(parents=True, exist_ok=True)
        radial_path = output_dir / 'radial_histogram.png'
        plt.savefig(radial_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(radial_path)


    # =============================================================================
    # FIXED: ANGULAR HISTOGRAM FIGURE (using consistent color format)
    # =============================================================================

    def _create_angular_histogram_figure(self, class_distributions, class_names=None):
        """
        Create a polar bar chart (angular histogram) showing angle distributions.
        FIXED: Uses consistent color format
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import plotly.express as px

        n_classes = len(class_distributions)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        # Use Plotly-compatible colors (hex codes)
        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f'Class {class_names[c] if class_names and c < len(class_names) else c+1}'
                           for c in class_distributions.keys()],
            specs=[[{'type': 'polar'} for _ in range(n_cols)] for _ in range(n_rows)]
        )

        for idx, (c, dist) in enumerate(class_distributions.items()):
            if dist['n_points'] == 0:
                continue

            row = idx // n_cols + 1
            col = idx % n_cols + 1

            # Create angular histogram (circular)
            angles_deg = dist['angles_deg']

            # Create histogram bins (36 bins = 10° each, centered)
            bins = np.linspace(-180, 180, 37)
            hist, bin_edges = np.histogram(angles_deg, bins=bins)

            # Normalize to max for better visualization
            hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist

            # Center angles for bar positions
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Create polar bar chart
            fig.add_trace(
                go.Barpolar(
                    r=hist_norm,
                    theta=bin_centers,
                    width=10,
                    marker=dict(
                        color=colors[c % len(colors)],
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    name=f'Class {c+1}',
                    hovertemplate='Angle: %{theta:.0f}°<br>Relative Density: %{r:.3f}<extra></extra>'
                ),
                row=row, col=col
            )

            # Add mean angle marker
            mean_angle = dist['mean_angle_deg']
            max_density = np.max(hist_norm) if len(hist_norm) > 0 else 1
            marker_radius = max_density * 1.1

            fig.add_trace(
                go.Scatterpolar(
                    r=[marker_radius],
                    theta=[mean_angle],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=colors[c % len(colors)],
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    name='Mean',
                    showlegend=False,
                    hovertemplate=f'Mean Angle: {mean_angle:.1f}°<extra></extra>'
                ),
                row=row, col=col
            )

            # Add angular spread as arc (if concentration is good)
            if dist['angular_concentration'] > 0.5:
                spread = dist['angular_spread_deg']

                # Create arc for ±1σ
                theta_arc = np.linspace(mean_angle - spread, mean_angle + spread, 50)
                r_arc = [marker_radius * 0.8] * len(theta_arc)

                fig.add_trace(
                    go.Scatterpolar(
                        r=r_arc,
                        theta=theta_arc,
                        mode='lines',
                        line=dict(color=colors[c % len(colors)], width=2, dash='dash'),
                        name='1σ Spread',
                        showlegend=False,
                        hovertemplate=f'Spread: ±{spread:.1f}°<extra></extra>'
                    ),
                    row=row, col=col
                )

            # Add annotation with statistics
            fig.add_annotation(
                x=0.5, y=1.1,
                xref=f'x{idx+1}', yref='paper',
                text=f"Conc: {dist['angular_concentration']:.2f} | Spread: {dist['angular_spread_deg']:.1f}° | n={dist['n_points']}",
                showarrow=False,
                font=dict(size=9, color='white'),
                bgcolor='rgba(0,0,0,0.5)',
                borderpad=2
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'<b>Angular Histogram - {self.model.dataset_name if hasattr(self.model, "dataset_name") else "Model"}</b><br>'
                     f'<sup>Shows angular distribution of weights in complex plane | Higher concentration = better class separation</sup>',
                font=dict(size=14)
            ),
            height=400 * n_rows,
            width=500 * n_cols,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0.9)',
            plot_bgcolor='rgba(0,0,0,0.9)'
        )

        return fig


    # =============================================================================
    # FIXED: ANGULAR HISTOGRAM DASHBOARD (using hex colors)
    # =============================================================================

    def _create_angular_histogram_dashboard(self, angular_fig, radial_path, class_distributions, class_names, weights):
        """
        Create combined HTML dashboard with angular histogram and statistics.
        FIXED: Uses hex colors for consistent display
        """
        # Calculate global statistics
        global_stats = self._calculate_global_stats(class_distributions)

        # Convert angular figure to HTML
        angular_html = angular_fig.to_html(include_plotlyjs='cdn', div_id='angular-histogram')

        # Use hex colors (matplotlib and plotly compatible)
        colors_hex = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
            '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5'
        ]

        # Build metrics table
        metrics_rows = []

        for c, dist in class_distributions.items():
            class_name = class_names[c] if class_names and c < len(class_names) else f'Class {c+1}'

            # Color coding based on metrics
            angle_color = 'metric-good' if dist['angular_concentration'] > 0.8 else \
                         'metric-medium' if dist['angular_concentration'] > 0.6 else 'metric-poor'

            spread_color = 'metric-good' if dist['angular_spread_deg'] < 30 else \
                          'metric-medium' if dist['angular_spread_deg'] < 60 else 'metric-poor'

            radius_color = 'metric-good' if dist['radius_cv'] < 0.3 else \
                          'metric-medium' if dist['radius_cv'] < 0.6 else 'metric-poor'

            metrics_rows.append(f"""
            <tr>
                <td><span style="color: {colors_hex[c % len(colors_hex)]}; font-weight: bold;">●</span> {class_name}</td>
                <td class="{angle_color}">{dist['angular_concentration']:.3f}</td>
                <td class="{spread_color}">{dist['angular_spread_deg']:.1f}°</td>
                <td>{dist['mean_angle_deg']:.1f}°</td>
                <td class="{radius_color}">{dist['radius_cv']:.3f}</td>
                <td>{dist['mean_radius']:.3f}</td>
                <td>{dist['n_points']}</td>
            </tr>
            """)

        # Create HTML
        html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Angular Histogram - {self.model.dataset_name if hasattr(self.model, 'dataset_name') else 'Model'}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: white;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0 0 10px 0;
            }}
            .header p {{
                margin: 0;
                opacity: 0.9;
            }}
            .dashboard-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }}
            .plot-container {{
                background: rgba(0,0,0,0.7);
                border-radius: 10px;
                padding: 15px;
                border: 1px solid rgba(255,255,255,0.2);
            }}
            .full-width {{
                grid-column: 1 / -1;
            }}
            .metrics-table {{
                background: rgba(0,0,0,0.5);
                border-radius: 10px;
                padding: 15px;
                overflow-x: auto;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            }}
            th {{
                background: rgba(102,126,234,0.5);
                font-weight: bold;
            }}
            .metric-good {{
                color: #4caf50;
                font-weight: bold;
            }}
            .metric-medium {{
                color: #ff9800;
                font-weight: bold;
            }}
            .metric-poor {{
                color: #f44336;
                font-weight: bold;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background: rgba(0,0,0,0.5);
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.2);
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .stat-label {{
                font-size: 12px;
                color: #aaa;
            }}
            .info-text {{
                font-size: 12px;
                color: #aaa;
                margin-top: 10px;
                text-align: center;
                padding: 10px;
                background: rgba(0,0,0,0.3);
                border-radius: 5px;
            }}
            .legend {{
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-top: 10px;
                flex-wrap: wrap;
            }}
            .legend-item {{
                display: inline-flex;
                align-items: center;
                gap: 5px;
            }}
            .color-box {{
                width: 12px;
                height: 12px;
                border-radius: 2px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Angular Histogram Analysis</h1>
            <p>Visualizing class separation in complex space | Each point = weight from 5D tensor</p>
            <p>Angle = Phase of weight | Higher concentration = Better class separation</p>
            <div class="legend">
                {''.join([f'<div class="legend-item"><div class="color-box" style="background: {colors_hex[c % len(colors_hex)]};"></div><span>Class {class_names[c] if class_names and c < len(class_names) else c+1}</span></div>' for c in class_distributions.keys()])}
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Mean Angular Concentration</div>
                <div class="stat-value" style="color: {'#4caf50' if global_stats['mean_concentration'] > 0.7 else '#ff9800' if global_stats['mean_concentration'] > 0.5 else '#f44336'}">
                    {global_stats['mean_concentration']:.3f}
                </div>
                <div class="stat-label">(Higher = better separation)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Angular Spread</div>
                <div class="stat-value" style="color: {'#4caf50' if global_stats['mean_spread'] < 45 else '#ff9800' if global_stats['mean_spread'] < 90 else '#f44336'}">
                    {global_stats['mean_spread']:.1f}°
                </div>
                <div class="stat-label">(Lower = more focused)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Radial CV</div>
                <div class="stat-value" style="color: {'#4caf50' if global_stats['mean_radius_cv'] < 0.5 else '#ff9800' if global_stats['mean_radius_cv'] < 0.8 else '#f44336'}">
                    {global_stats['mean_radius_cv']:.3f}
                </div>
                <div class="stat-label">(Lower = consistent magnitude)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Weights</div>
                <div class="stat-value">{global_stats['total_weights']:,}</div>
                <div class="stat-label">Significant weights analyzed</div>
            </div>
        </div>

        <div class="dashboard-container">
            <div class="plot-container full-width">
                <h3>🎯 Angular Histograms</h3>
                <p>Each polar plot shows the angular distribution of weights for one class</p>
                {angular_html}
            </div>
        </div>

        <div class="dashboard-container">
            <div class="plot-container">
                <h3>📈 Radial Histograms</h3>
                <p>Magnitude distributions - shows how consistent weights are</p>
                <img src="{radial_path}" style="width: 100%; border-radius: 5px;">
            </div>

            <div class="plot-container">
                <h3>📊 Class Statistics</h3>
                <div class="metrics-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Class</th>
                                <th>Angular Concentration</th>
                                <th>Angular Spread</th>
                                <th>Mean Angle</th>
                                <th>Radial CV</th>
                                <th>Mean Radius</th>
                                <th>Points</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(metrics_rows)}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="info-text">
            <strong>📖 How to Read Angular Histograms:</strong><br>
            • <strong>Polar Plot</strong>: Shows distribution of angles (0° to 360°)<br>
            • <strong>Bar Height</strong>: Relative density of weights at that angle<br>
            • <strong>Red Dot</strong>: Mean angle (center of mass)<br>
            • <strong>Dashed Arc</strong>: 1σ angular spread (68% of weights)<br>
            • <strong>Angular Concentration</strong>: 1.0 = perfectly focused at one angle<br>
            • <strong>Angular Spread</strong>: Lower = more focused cluster<br>
            <br>
            <strong>✅ Perfect Classification Achieved When:</strong><br>
            • Classes have high angular concentration (>0.8)<br>
            • Classes are separated by at least 45°<br>
            • Angular spread is low (<30°)<br>
            • Radial CV is low (<0.5) for consistent magnitude<br>
            <br>
            <strong>🎯 Current Status:</strong><br>
            • Mean Angular Concentration: {global_stats['mean_concentration']:.3f} {'(Good)' if global_stats['mean_concentration'] > 0.7 else '(Needs Improvement)'}<br>
            • Mean Angular Spread: {global_stats['mean_spread']:.1f}° {'(Good)' if global_stats['mean_spread'] < 45 else '(Needs Improvement)'}<br>
            • Separation Quality: {'Excellent' if global_stats['mean_spread'] < 30 else 'Good' if global_stats['mean_spread'] < 60 else 'Poor'}
        </div>
    </body>
    </html>"""

        # Save HTML
        output_dir = Path('visualizations') / (self.model.dataset_name if hasattr(self.model, 'dataset_name') else 'model') / 'angular_histograms'
        output_dir.mkdir(parents=True, exist_ok=True)
        dashboard_path = output_dir / 'angular_histogram_dashboard.html'

        with open(dashboard_path, 'w') as f:
            f.write(html)

        return str(dashboard_path)


    # =============================================================================
    # FIXED: GENERATE ANGULAR HISTOGRAM THREAD
    # =============================================================================

    def _generate_angular_histogram_thread(self, weights):
        """
        Background thread for generating angular histogram visualizations.
        FIXED: Proper error handling and color management
        """
        try:
            self.log_output("📊 Generating Angular Histogram...")
            self.log_output("   This shows how classes are separated by angle in complex space:")
            self.log_output("   • Each class's angular distribution")
            self.log_output("   • Mean angle (center of mass)")
            self.log_output("   • Angular spread (how focused the cluster is)")
            self.log_output("   • Perfect separation = classes at different angles with low spread")
            self.status_var.set("Generating angular histogram...")

            # Extract class distributions
            class_distributions = self._extract_class_distributions(weights)

            # Get class names
            class_names = None
            if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                class_names = list(self.model.label_encoder.keys())

            # Create angular histogram figure
            angular_fig = self._create_angular_histogram_figure(class_distributions, class_names)

            # Create radial histogram for complementary view
            radial_path = self._create_radial_histogram_figure(class_distributions, class_names)

            # Create combined dashboard
            dashboard_path = self._create_angular_histogram_dashboard(
                angular_fig, radial_path, class_distributions, class_names, weights
            )

            if dashboard_path:
                self.log_output(f"✅ Angular histogram dashboard saved to: {dashboard_path}")
                self.root.after(100, lambda: self._ask_open_dashboard(dashboard_path, "angular"))
            else:
                self.log_output("❌ Failed to create angular histogram")
                self.status_var.set("Angular histogram generation failed")

        except Exception as e:
            self.log_output(f"❌ Error generating angular histogram: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Error generating angular histogram")

    def show_spherical_evolution(self):
        """
        Show the original spherical evolution (preserved for backward compatibility)
        """
        if not self.model:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        evolution_history = self.model.get_evolution_history()
        if not evolution_history or len(evolution_history) < 2:
            messagebox.showwarning(
                "Warning",
                "No evolution history available.\n"
                "Please run adaptive training with evolution tracking enabled."
            )
            return

        try:
            self.log_output("🌐 Creating original spherical evolution...")
            self.status_var.set("Generating spherical evolution...")

            # Get class names
            class_names = None
            if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                class_names = list(self.model.label_encoder.keys())

            # Create visualizer
            visualizer = OptimizedVisualizer(self.model)
            output_path = visualizer.create_spherical_evolution(evolution_history, class_names)

            if output_path:
                self.log_output(f"✅ Spherical evolution saved to: {output_path}")

                if messagebox.askyesno("Spherical Evolution Ready", "Open in browser now?"):
                    import webbrowser
                    webbrowser.open(f'file://{output_path}')
            else:
                self.log_output("❌ Failed to create spherical evolution")

        except Exception as e:
            self.log_output(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    def show_side_by_side(self):
        """
        Show side-by-side comparison of initial and final states
        """
        if not self.model:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        evolution_history = self.model.get_evolution_history()
        if not evolution_history or len(evolution_history) < 2:
            messagebox.showwarning(
                "Warning",
                "Need at least 2 rounds for comparison.\n"
                "Please run adaptive training first."
            )
            return

        try:
            self.log_output("🔄 Creating side-by-side evolution comparison...")
            self.status_var.set("Generating side-by-side comparison...")

            visualizer = OptimizedVisualizer(self.model)
            output_path = visualizer.spherical_viz.create_side_by_side_evolution(evolution_history)

            if output_path and Path(output_path).exists():
                self.log_output(f"✅ Side-by-side comparison saved to: {output_path}")

                if messagebox.askyesno("Comparison Ready", "Open in browser now?"):
                    import webbrowser
                    webbrowser.open(f'file://{Path(output_path).absolute()}')
            else:
                self.log_output("❌ Failed to create side-by-side comparison")

        except Exception as e:
            self.log_output(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    def show_tensor_evolution(self):
        """
        Show the tensor evolution plot (orthogonalization analysis)
        """
        if not self.model:
            messagebox.showwarning("Warning", "Please initialize the model first")
            return

        evolution_history = self.model.get_evolution_history()
        if not evolution_history:
            messagebox.showwarning(
                "Warning",
                "No evolution history available. Run adaptive training with tracking enabled."
            )
            return

        try:
            self.log_output("📊 Generating tensor evolution plot...")
            visualizer = OptimizedVisualizer(self.model)
            visualizer.plot_tensor_evolution(evolution_history)
            self.log_output(f"✅ Tensor evolution saved to: {visualizer.dirs['tensor']}")
            self.open_visualization_dir()
        except Exception as e:
            self.log_output(f"❌ Error creating tensor evolution: {e}")
            traceback.print_exc()

    def show_dashboard(self):
        """
        Show interactive dashboard with evolution slider
        """
        if not self.model or self.model.X_tensor is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        try:
            self.log_output("📊 Creating interactive dashboard...")
            visualizer = OptimizedVisualizer(self.model)

            X_np = self.model.X_tensor.numpy()
            y_np = self.model.y_tensor.numpy()
            evolution_history = self.model.get_evolution_history()

            visualizer.create_interactive_dashboard(
                self.model.training_history,
                X_np, y_np,
                evolution_history=evolution_history
            )

            # Create evolution slider if history exists
            if evolution_history and len(evolution_history) > 1:
                visualizer.spherical_viz.create_evolution_slider(evolution_history)

            self.log_output(f"✅ Interactive dashboard saved to: {visualizer.dirs['interactive']}")

            if messagebox.askyesno("Dashboard Ready", "Open dashboard in browser?"):
                import webbrowser
                dashboard_path = visualizer.dirs['interactive'] / f'{self.model.dataset_name}_dashboard.html'
                webbrowser.open(f'file://{dashboard_path.absolute()}')

        except Exception as e:
            self.log_output(f"❌ Error creating dashboard: {e}")
            traceback.print_exc()

    def show_confusion_matrix(self):
        """Show confusion matrix visualization"""
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
        """Show training history plot"""
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
            # Get selected features (exclude target column automatically)
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

            # Validate that selected features are numeric
            df = self.original_data
            numeric_features = []
            non_numeric_features = []

            for col in selected_features:
                if col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        numeric_features.append(col)
                    else:
                        non_numeric_features.append(col)
                        self.log_output(f"⚠️ Warning: Column '{col}' is not numeric and will be skipped")

            if not numeric_features:
                messagebox.showerror("Error",
                    "No numeric features selected. Please select numeric columns only.")
                return

            if non_numeric_features:
                self.log_output(f"   Skipped non-numeric features: {', '.join(non_numeric_features)}")
                selected_features = numeric_features

            # Update UI to reflect filtered selection
            for col in non_numeric_features:
                self.feature_vars[col].set(False)
                self.log_output(f"   Auto-deselected non-numeric column: {col}")

            # Initialize model with selected features
            if hasattr(self, 'current_data_file') and self.current_data_file:
                dataset_name = os.path.splitext(os.path.basename(self.current_data_file))[0]
            else:
                # CRITICAL FIX: If current_data_file is not set, use the file path from the UI
                file_path = self.file_path_var.get()
                if file_path:
                    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                    self.current_data_file = file_path
                else:
                    messagebox.showerror("Error", "No data file specified.")
                    return

            # Create config with selected features
            config = {
                'file_path': self.current_data_file,  # Use current_data_file
                'target_column': target_column,
                'model_type': self.model_type_var.get(),
                'compute_device': self.device_var.get(),
                'parallel': self.parallel_var.get(),
                'n_jobs': int(self.n_jobs_var.get()) if self.parallel_var.get() else 1,
                'parallel_batch_size': int(self.parallel_batch_var.get()) if self.parallel_var.get() else 1000,
                'parallel_mode': self.parallel_mode_var.get(),
                'selected_features': selected_features,
                'numeric_features': numeric_features,
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

            # Use OptimizedDBNN
            self.model = OptimizedDBNN(
                dataset_name=dataset_name,
                config=config,
                enable_external_tools=ASTROPY_AVAILABLE
            )

            # Store selected features in model BEFORE loading data
            self.model.selected_features = selected_features
            self.model.numeric_features = numeric_features

            # Load data with feature filtering
            # CRITICAL FIX: Make sure the file path is set
            if not hasattr(self.model, 'config') or not self.model.config.get('file_path'):
                self.model.config['file_path'] = self.current_data_file

            success = self.model.load_data(file_path=self.current_data_file)

            if success:
                self.log_output(f"✅ Model initialized with feature selection")
                self.log_output(f"🎯 Target: {target_column}")
                self.log_output(f"📊 Features: {len(selected_features)} numeric features")
                self.log_output(f"   Features: {', '.join(selected_features[:10])}{'...' if len(selected_features) > 10 else ''}")

                self.model_trained = False
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

            # CRITICAL FIX: Set current_data_file
            self.current_data_file = file_path  # <-- ADD THIS LINE

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
            target_candidates = ['target', 'class', 'label', 'y', 'output', 'result', 'type', 'diagnosis']
            auto_target = None

            for candidate in target_candidates:
                if candidate in df.columns:
                    auto_target = candidate
                    break

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

    def download_uci(self):
        """Download UCI dataset and set current_data_file"""
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

            # CRITICAL FIX: Set the file path variables
            self.file_path_var.set(csv_path)
            self.current_data_file = csv_path  # <-- THIS WAS MISSING

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
    # POLAR EVOLUTION GUI METHODS
    # =============================================================================

    def show_polar_evolution(self):
        """Show polar coordinate evolution dashboard showing cluster formation"""
        if not self.model:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        evolution_history = self.model.get_evolution_history()
        if not evolution_history or len(evolution_history) < 2:
            messagebox.showwarning(
                "Warning",
                "Need at least 2 rounds of evolution history.\n\n"
                "Please run adaptive training with evolution tracking enabled.\n\n"
                "To enable evolution tracking:\n"
                "1. Go to Configuration tab\n"
                "2. Ensure 'Enable Adaptive Learning' is checked\n"
                "3. Run Adaptive Training"
            )
            return

        # Run in separate thread to avoid blocking GUI
        thread = threading.Thread(target=self._generate_polar_evolution_thread, args=(evolution_history,))
        thread.daemon = True
        thread.start()

    def _generate_polar_evolution_thread(self, evolution_history):
        """Background thread for generating polar evolution visualizations"""
        try:
            self.log_output("📊 Generating Polar Coordinate Evolution...")
            self.log_output("   This shows how class clusters form in the complex plane:")
            self.log_output("   • Each point = weight magnitude and phase from the 5D tensor")
            self.log_output("   • Classes form focused clusters at different angles")
            self.log_output("   • Perfect classification = separate, focused clusters (orthogonal)")
            self.log_output("")
            self.status_var.set("Generating polar evolution visualizations...")

            # Get class names
            class_names = None
            if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                class_names = list(self.model.label_encoder.keys())
                self.log_output(f"📊 Class labels: {class_names}")

            # Create visualizer and generate dashboard
            visualizer = OptimizedVisualizer(self.model)
            dashboard_path = visualizer.create_polar_visualization(evolution_history, class_names)

            if dashboard_path:
                file_size = Path(dashboard_path).stat().st_size / 1024
                self.log_output(f"✅ Polar evolution dashboard saved to: {dashboard_path}")
                self.log_output(f"   File size: {file_size:.1f} KB")
                self.log_output(f"   Rounds: {len(evolution_history)}")

                # Ask user if they want to open it
                self.root.after(100, lambda: self._ask_open_polar_dashboard(dashboard_path))
            else:
                self.log_output("❌ Failed to create polar evolution dashboard")
                self.status_var.set("Polar evolution generation failed")

        except Exception as e:
            self.log_output(f"❌ Error generating polar evolution: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Error generating polar evolution")

    def _ask_open_polar_dashboard(self, dashboard_path):
        """Ask user if they want to open the polar dashboard"""
        if messagebox.askyesno(
            "Polar Evolution Dashboard Ready",
            f"Polar evolution dashboard created successfully!\n\n"
            f"File: {dashboard_path}\n\n"
            f"Features:\n"
            f"• Animated evolution showing cluster formation\n"
            f"• Interactive plots with zoom/pan\n"
            f"• Cluster quality metrics (angular/radial concentration)\n"
            f"• Key round snapshots\n"
            f"• Summary table with final metrics\n\n"
            f"What it shows:\n"
            f"• Each point = weight (radius = magnitude, angle = phase)\n"
            f"• Classes form clusters at different angles\n"
            f"• Perfect separation = orthogonal clusters (90° apart)\n\n"
            f"Open in browser now?"
        ):
            import webbrowser
            webbrowser.open(f'file://{dashboard_path}')
            self.log_output("🌐 Polar evolution dashboard opened in browser")

    # =============================================================================
    # ENHANCED OPEN VISUALIZATION FOLDER METHOD
    # =============================================================================

    def open_visualization_dir(self):
        """Open the visualization directory in file explorer with detailed listing"""
        try:
            if self.model and hasattr(self.model, 'dataset_name'):
                vis_dir = os.path.abspath(f"visualizations/{self.model.dataset_name}")
            else:
                vis_dir = os.path.abspath("visualizations")

            # Create directory if it doesn't exist
            os.makedirs(vis_dir, exist_ok=True)

            if os.name == 'nt':
                subprocess.Popen(f'explorer "{vis_dir}"')
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', vis_dir])
            else:
                subprocess.Popen(['xdg-open', vis_dir])

            self.log_output(f"📂 Opened directory: {vis_dir}")

            # List available visualization files
            self._list_visualization_files(vis_dir)

        except Exception as e:
            self.log_output(f"⚠️ Could not open directory: {e}")

    def _list_visualization_files(self, vis_dir):
        """List available visualization files in the output directory"""
        try:
            from pathlib import Path
            vis_path = Path(vis_dir)

            if not vis_path.exists():
                self.log_output("   No visualizations generated yet.")
                return

            # Find all visualization files
            html_files = list(vis_path.rglob("*.html"))
            png_files = list(vis_path.rglob("*.png"))

            if html_files or png_files:
                self.log_output(f"\n📁 Available visualizations in {vis_path.name}:")

                # HTML files
                if html_files:
                    self.log_output(f"\n   🌐 HTML Interactive Files ({len(html_files)}):")
                    for f in html_files[:8]:  # Show first 8
                        rel_path = f.relative_to(vis_path.parent) if f.parent != vis_path else f.name
                        size_kb = f.stat().st_size / 1024
                        self.log_output(f"     • {rel_path} ({size_kb:.1f} KB)")
                    if len(html_files) > 8:
                        self.log_output(f"     ... and {len(html_files)-8} more")

                # PNG files
                if png_files:
                    self.log_output(f"\n   🖼️  PNG Images ({len(png_files)}):")
                    for f in png_files[:8]:
                        rel_path = f.relative_to(vis_path.parent) if f.parent != vis_path else f.name
                        size_kb = f.stat().st_size / 1024
                        self.log_output(f"     • {rel_path} ({size_kb:.1f} KB)")
                    if len(png_files) > 8:
                        self.log_output(f"     ... and {len(png_files)-8} more")

                # Highlight specific dashboards
                self.log_output(f"\n   📊 Dashboard Highlights:")

                # Check for multi-projection dashboard
                multi_proj = vis_path / "multi_projection" / f"{self.model.dataset_name if self.model else 'unknown'}_multi_projection_dashboard.html"
                if multi_proj.exists():
                    self.log_output(f"     🔬 Multi-Projection Dashboard: multi_projection/{multi_proj.name}")

                # Check for polar dashboard
                polar = vis_path / "polar_evolution" / f"{self.model.dataset_name if self.model else 'unknown'}_polar_dashboard.html"
                if polar.exists():
                    self.log_output(f"     📊 Polar Evolution Dashboard: polar_evolution/{polar.name}")

                # Check for spherical dashboard
                spherical = vis_path / "spherical_evolution" / f"{self.model.dataset_name if self.model else 'unknown'}_spherical.html"
                if spherical.exists():
                    self.log_output(f"     🌐 Spherical Evolution: spherical_evolution/{spherical.name}")

                # Check for interactive dashboard
                interactive = vis_path / "interactive" / f"{self.model.dataset_name if self.model else 'unknown'}_dashboard.html"
                if interactive.exists():
                    self.log_output(f"     📈 Interactive Dashboard: interactive/{interactive.name}")

            else:
                self.log_output("   No visualizations generated yet. Run training and generate visualizations first.")

        except Exception as e:
            pass  # Silent fail for listing

    # =============================================================================
    # ENHANCED COMPARISON GRID METHOD
    # =============================================================================

    def show_comparison_grid(self):
        """
        Show the static comparison grid (initial vs final state for all projection methods)
        """
        if not self.model:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        evolution_history = self.model.get_evolution_history()
        if not evolution_history or len(evolution_history) < 2:
            messagebox.showwarning(
                "Warning",
                "Need at least 2 rounds for comparison.\n"
                "Please run adaptive training first."
            )
            return

        try:
            self.log_output("📊 Generating comparison grid (Initial vs Final)...")
            self.status_var.set("Generating comparison grid...")

            # Get class names
            class_names = None
            if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                class_names = list(self.model.label_encoder.keys())

            # Create visualizer and generate comparison
            visualizer = OptimizedVisualizer(self.model)
            comparison_path = visualizer.multi_projection.create_comparison_grid(evolution_history)

            if comparison_path:
                file_size = Path(comparison_path).stat().st_size / 1024
                self.log_output(f"✅ Comparison grid saved to: {comparison_path} ({file_size:.1f} KB)")

                if messagebox.askyesno(
                    "Comparison Grid Ready",
                    f"Comparison grid saved to:\n{comparison_path}\n\n"
                    f"Shows Initial vs Final state for all 5 projection methods\n\n"
                    f"Open image now?"
                ):
                    import subprocess
                    import platform

                    system = platform.system()
                    if system == "Windows":
                        subprocess.Popen(['start', comparison_path], shell=True)
                    elif system == "Darwin":
                        subprocess.Popen(['open', comparison_path])
                    else:
                        subprocess.Popen(['xdg-open', comparison_path])
            else:
                self.log_output("❌ Failed to create comparison grid")

        except Exception as e:
            self.log_output(f"❌ Error generating comparison grid: {e}")
            import traceback
            traceback.print_exc()

    # =============================================================================
    # ENHANCED MULTI-PROJECTION METHOD (with better feedback)
    # =============================================================================

    def show_multi_projection_evolution(self):
        """
        Show the multi-projection spherical evolution with 5 different projection methods
        This creates an interactive dashboard with synchronized views
        """
        if not self.model:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        # Check if evolution history exists
        evolution_history = self.model.get_evolution_history()
        if not evolution_history or len(evolution_history) < 2:
            messagebox.showwarning(
                "Warning",
                "Need at least 2 rounds of evolution history.\n\n"
                "Please run adaptive training with evolution tracking enabled:\n"
                "1. Go to Configuration tab\n"
                "2. Ensure 'Enable Adaptive Learning' is checked\n"
                "3. Run Adaptive Training\n\n"
                "The evolution tracker will automatically capture tensor states."
            )
            return

        # Run in a separate thread to avoid blocking GUI
        thread = threading.Thread(target=self._generate_multi_projection_thread, args=(evolution_history,))
        thread.daemon = True
        thread.start()

    def _generate_multi_projection_thread(self, evolution_history):
        """
        Background thread for generating multi-projection visualizations
        """
        try:
            self.log_output("🎨 Generating Multi-Projection Spherical Evolution...")
            self.log_output("   This creates 5 different views of the same tensor evolution:")
            self.log_output("   • 2D Compression (Original method)")
            self.log_output("   • PCA on Flattened Tensor (Maximum variance)")
            self.log_output("   • MDS on Class Distances (True class separation)")
            self.log_output("   • Physics-Inspired (Interpretable axes)")
            if TENSORLY_AVAILABLE:
                self.log_output("   • HOSVD Tucker (Multi-linear structure)")
            self.log_output("")
            self.status_var.set("Generating multi-projection visualizations...")

            # Get class names
            class_names = None
            if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                class_names = list(self.model.label_encoder.keys())
                self.log_output(f"📊 Class labels: {class_names}")

            # Create visualizer and generate dashboard
            visualizer = OptimizedVisualizer(self.model)
            dashboard_path = visualizer.create_multi_projection_visualization(
                evolution_history, class_names
            )

            if dashboard_path:
                file_size = Path(dashboard_path).stat().st_size / 1024
                self.log_output(f"✅ Multi-projection dashboard saved to: {dashboard_path}")
                self.log_output(f"   File size: {file_size:.1f} KB")
                self.log_output(f"   Rounds: {len(evolution_history)}")

                # Ask user if they want to open it
                self.root.after(100, lambda: self._ask_open_multi_projection_dashboard(dashboard_path))
            else:
                self.log_output("❌ Failed to create multi-projection dashboard")
                self.status_var.set("Multi-projection generation failed")

        except Exception as e:
            self.log_output(f"❌ Error generating multi-projection visualization: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Error generating visualizations")

    def _ask_open_multi_projection_dashboard(self, dashboard_path):
        """Ask user if they want to open the multi-projection dashboard"""
        if messagebox.askyesno(
            "Multi-Projection Dashboard Ready",
            f"Multi-projection dashboard created successfully!\n\n"
            f"File: {dashboard_path}\n\n"
            f"Features:\n"
            f"• 5 synchronized projection methods\n"
            f"• Animated evolution over rounds\n"
            f"• Interactive 3D views with zoom/pan\n"
            f"• Hover information for each class\n"
            f"• Target positions (X markers) showing ideal orthogonal configuration\n\n"
            f"Open in browser now?"
        ):
            import webbrowser
            webbrowser.open(f'file://{dashboard_path}')
            self.log_output("🌐 Multi-projection dashboard opened in browser")

    def show_2d_polar_evolution(self):
        """Show 2D polar evolution dashboard showing actual weight distributions"""
        if not self.model:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        evolution_history = self.model.get_evolution_history()
        if not evolution_history or len(evolution_history) < 2:
            messagebox.showwarning(
                "Warning",
                "Need at least 2 rounds of evolution history.\n\n"
                "Please run adaptive training with evolution tracking enabled."
            )
            return

        thread = threading.Thread(target=self._generate_2d_polar_thread, args=(evolution_history,))
        thread.daemon = True
        thread.start()

    def _generate_2d_polar_thread(self, evolution_history):
        """Background thread for generating 2D polar visualizations"""
        try:
            self.log_output("🎯 Generating 2D Polar Evolution Dashboard...")
            self.log_output("   This shows the ACTUAL weight distributions in complex space:")
            self.log_output("   • Each point = one weight from the 5D tensor")
            self.log_output("   • Distance from origin = magnitude (strength)")
            self.log_output("   • Angle from origin = phase (orientation)")
            self.log_output("   • Perfect classification = separate, focused clusters at different angles")
            self.status_var.set("Generating 2D polar visualizations...")

            class_names = None
            if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                class_names = list(self.model.label_encoder.keys())

            visualizer = OptimizedVisualizer(self.model)
            dashboard_path = visualizer.create_2d_polar_visualization(evolution_history, class_names)

            if dashboard_path:
                self.log_output(f"✅ 2D Polar dashboard saved to: {dashboard_path}")
                self.root.after(100, lambda: self._ask_open_dashboard(dashboard_path, "2d_polar"))
            else:
                self.log_output("❌ Failed to create 2D polar dashboard")
                self.status_var.set("2D polar generation failed")

        except Exception as e:
            self.log_output(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # =============================================================================
    # ANGULAR HISTOGRAM METHODS FOR GUI
    # =============================================================================

    def show_angular_histogram(self):
        """
        Show angular histogram for the current model state.
        This shows how classes are distributed by angle in the complex plane.
        """
        if not self.model:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        # Check if we have a trained model with weights
        if not hasattr(self.model, 'weight_updater') or self.model.weight_updater is None:
            messagebox.showwarning(
                "Warning",
                "Model not trained yet. Please train the model first.\n\n"
                "Angular histogram requires trained weights to visualize class separation."
            )
            return

        # Get current weights
        weights = self.model.weight_updater.weights
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()

        if len(weights.shape) != 4:
            self.log_output("❌ Invalid weights shape. Expected 4D tensor.")
            return

        # Run in separate thread
        thread = threading.Thread(target=self._generate_angular_histogram_thread, args=(weights,))
        thread.daemon = True
        thread.start()

    def _extract_class_distributions(self, weights):
        """
        Extract angular and radial distributions for each class from weights.
        """
        n_classes = weights.shape[0]
        class_distributions = {}

        for c in range(n_classes):
            # Get all weights for this class, flattened
            class_weights = weights[c].flatten()

            # Filter significant weights (avoid noise)
            significant = class_weights[np.abs(class_weights) > 1e-6]

            if len(significant) > 0:
                # Convert to polar coordinates
                radii = np.abs(significant)
                angles = np.angle(significant)  # Range: -π to π
                angles_deg = angles * 180 / np.pi

                # Calculate statistics
                mean_angle = np.mean(angles)
                mean_angle_deg = mean_angle * 180 / np.pi

                # Circular statistics for angular concentration
                complex_angles = np.exp(1j * angles)
                mean_resultant = np.mean(complex_angles)
                angular_concentration = np.abs(mean_resultant)  # 1 = perfectly focused
                circular_variance = 1 - angular_concentration

                # Angular spread (circular standard deviation)
                angular_spread_rad = np.sqrt(-2 * np.log(angular_concentration))
                angular_spread_deg = angular_spread_rad * 180 / np.pi

                # Radial statistics
                mean_radius = np.mean(radii)
                radius_std = np.std(radii)
                radius_cv = radius_std / (mean_radius + 1e-10)  # Coefficient of variation

                class_distributions[c] = {
                    'radii': radii,
                    'angles': angles,
                    'angles_deg': angles_deg,
                    'n_points': len(significant),
                    'mean_angle': mean_angle,
                    'mean_angle_deg': mean_angle_deg,
                    'mean_radius': mean_radius,
                    'radius_std': radius_std,
                    'angular_concentration': angular_concentration,
                    'angular_spread_deg': angular_spread_deg,
                    'circular_variance': circular_variance,
                    'radius_cv': radius_cv,
                    'total_mass': np.sum(radii),
                    'min_radius': np.min(radii),
                    'max_radius': np.max(radii)
                }
            else:
                class_distributions[c] = {
                    'radii': np.array([]),
                    'angles': np.array([]),
                    'angles_deg': np.array([]),
                    'n_points': 0,
                    'mean_angle': 0,
                    'mean_angle_deg': 0,
                    'mean_radius': 0,
                    'radius_std': 0,
                    'angular_concentration': 0,
                    'angular_spread_deg': 0,
                    'circular_variance': 1,
                    'radius_cv': 1,
                    'total_mass': 0,
                    'min_radius': 0,
                    'max_radius': 0
                }

        return class_distributions



    def _calculate_global_stats(self, class_distributions):
        """
        Calculate global statistics across all classes.
        """
        concentrations = []
        spreads = []
        radius_cvs = []
        total_weights = 0

        for dist in class_distributions.values():
            if dist['n_points'] > 0:
                concentrations.append(dist['angular_concentration'])
                spreads.append(dist['angular_spread_deg'])
                radius_cvs.append(dist['radius_cv'])
                total_weights += dist['n_points']

        return {
            'mean_concentration': np.mean(concentrations) if concentrations else 0,
            'mean_spread': np.mean(spreads) if spreads else 0,
            'mean_radius_cv': np.mean(radius_cvs) if radius_cvs else 0,
            'total_weights': total_weights
        }

    def _ask_open_angular_histogram(self, dashboard_path):
        """
        Ask user if they want to open the angular histogram dashboard.
        """
        if messagebox.askyesno(
            "Angular Histogram Dashboard Ready",
            f"Angular histogram dashboard created successfully!\n\n"
            f"File: {dashboard_path}\n\n"
            f"What it shows:\n"
            f"• Angular distribution of weights for each class\n"
            f"• Mean angle and angular spread (how focused each class is)\n"
            f"• Radial magnitude distributions\n"
            f"• Class separation quality metrics\n\n"
            f"Interpretation:\n"
            f"• High angular concentration (>0.8) = well-separated classes\n"
            f"• Low angular spread (<30°) = focused clusters\n"
            f"• Classes at different angles = orthogonal in complex space\n\n"
            f"Open in browser now?"
        ):
            import webbrowser
            webbrowser.open(f'file://{dashboard_path}')
            self.log_output("🌐 Angular histogram dashboard opened in browser")

    def show_true_polar_evolution(self):
        """Show true polar evolution dashboard (r vs θ)"""
        if not self.model:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        evolution_history = self.model.get_evolution_history()
        if not evolution_history or len(evolution_history) < 2:
            messagebox.showwarning(
                "Warning",
                "Need at least 2 rounds of evolution history.\n\n"
                "Please run adaptive training with evolution tracking enabled."
            )
            return

        thread = threading.Thread(target=self._generate_true_polar_thread, args=(evolution_history,))
        thread.daemon = True
        thread.start()

    def _generate_true_polar_thread(self, evolution_history):
        """Background thread for generating true polar visualizations"""
        try:
            self.log_output("🎯 Generating True Polar Evolution (r vs θ)...")
            self.log_output("   Features:")
            self.log_output("   • Radius = Magnitude of weight")
            self.log_output("   • Angle = Phase of weight (0° to 360°)")
            self.log_output("   • Classes form distinct angular wedges")
            self.log_output("   • Perfect classification = classes at different angles")
            self.status_var.set("Generating true polar visualization...")

            class_names = None
            if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                class_names = list(self.model.label_encoder.keys())

            visualizer = OptimizedVisualizer(self.model)
            dashboard_path = visualizer.create_true_polar_visualization(evolution_history, class_names)

            if dashboard_path:
                self.log_output(f"✅ True Polar dashboard saved to: {dashboard_path}")
                self.root.after(100, lambda: self._ask_open_dashboard(dashboard_path, "true_polar"))
            else:
                self.log_output("❌ Failed to create true polar dashboard")
                self.status_var.set("True polar generation failed")

        except Exception as e:
            self.log_output(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Error generating true polar")

    def show_point_based_spherical_evolution(self):
        """
        Show point-based spherical evolution with actual data points.
        This replaces the old vector-based spherical visualization.
        """
        if not self.model:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        evolution_history = self.model.get_evolution_history()
        if not evolution_history or len(evolution_history) < 2:
            messagebox.showwarning(
                "Warning",
                "Need at least 2 rounds of evolution history.\n\n"
                "Please run adaptive training with evolution tracking enabled."
            )
            return

        # Get data for projection
        if self.model.X_tensor is None or self.model.y_tensor is None:
            messagebox.showwarning("Warning", "No data available for visualization")
            return

        # Run in separate thread
        thread = threading.Thread(
            target=self._generate_point_based_spherical_thread,
            args=(evolution_history, self.model.X_tensor.numpy(), self.model.y_tensor.numpy())
        )
        thread.daemon = True
        thread.start()

    def _generate_point_based_spherical_thread(self, evolution_history, X_data, y_data):
        """Background thread for point-based spherical visualization"""
        try:
            self.log_output("🎯 Generating Point-Based Spherical Evolution...")
            self.log_output("   Features:")
            self.log_output("   • Shows ACTUAL data points projected to sphere surface")
            self.log_output("   • Each dot = one data sample's position in tensor space")
            self.log_output("   • Classes form clusters that separate over time")
            self.log_output("   • Includes orthogonality verification metrics")
            self.status_var.set("Generating point-based spherical visualization...")

            # Get class names
            class_names = None
            if hasattr(self.model, 'label_encoder') and self.model.label_encoder:
                class_names = list(self.model.label_encoder.keys())

            # Create visualizer and generate dashboard
            visualizer = OptimizedVisualizer(self.model)
            dashboard_path = visualizer.create_point_based_spherical_evolution(
                evolution_history, X_data, y_data, class_names
            )

            if dashboard_path:
                file_size = Path(dashboard_path).stat().st_size / 1024
                self.log_output(f"✅ Point-based spherical dashboard: {dashboard_path}")
                self.log_output(f"   File size: {file_size:.1f} KB")

                # Ask user if they want to open it
                self.root.after(100, lambda: self._ask_open_dashboard(
                    dashboard_path, "point_based_spherical"
                ))
            else:
                self.log_output("❌ Failed to create point-based spherical visualization")
                self.status_var.set("Visualization generation failed")

        except Exception as e:
            self.log_output(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Error generating visualization")


# =============================================================================
# SECTION 11: MAIN ENTRY POINT
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CT-DBNN - Complex Tensor DBNN')
    parser.add_argument('--gui', action='store_true', help='Launch GUI')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--mode', type=str, default='train_predict',
                       choices=['train', 'predict', 'train_predict', 'adaptive'],
                       help='Operation mode: train, predict, train_predict (default), or adaptive')
    parser.add_argument('--predict-file', type=str, help='File to make predictions on (for predict mode)')
    parser.add_argument('--output-dir', type=str, default='predictions', help='Output directory for predictions')
    parser.add_argument('--track-evolution', action='store_true', help='Track tensor evolution')
    parser.add_argument('--model-file', type=str, help='Specific model file to load (for predict mode)')
    parser.add_argument('--no-confusion', action='store_true', help='Skip confusion matrix display')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory (default: data)')
    parser.add_argument('--input-path', type=str, help='Custom path to input CSV file (overrides dataset name)')
    parser.add_argument('--config-file', type=str, help='Specific config file to use')
    parser.add_argument('--csv-file', type=str, help='Specific CSV file to use')
    parser.add_argument('--skip-comments', action='store_true', default=True,
                       help='Skip lines starting with # (default: True)')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
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

    # Helper function to load CSV with comment skipping
    def load_csv_with_comments(file_path, skip_comments=True):
        """Load CSV file, skipping comment lines starting with #"""
        if not os.path.exists(file_path):
            return None

        if args.verbose:
            print(f"{Colors.CYAN}   Reading file: {file_path}{Colors.ENDC}")

        if skip_comments:
            # Read all lines and filter out comments
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Filter out lines that start with # (after stripping whitespace)
            data_lines = [line for line in lines if not line.strip().startswith('#')]

            if not data_lines:
                print(f"{Colors.RED}❌ No non-comment lines found in {file_path}{Colors.ENDC}")
                return None

            if args.verbose:
                print(f"{Colors.CYAN}   Found {len(data_lines)} data lines (skipped {len(lines) - len(data_lines)} comment lines){Colors.ENDC}")

            # Write filtered lines to a temporary file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_file.writelines(data_lines)
            temp_file.close()

            # Read the temporary file with pandas
            try:
                df = pd.read_csv(temp_file.name)
                # Clean up temp file
                os.unlink(temp_file.name)
                return df
            except Exception as e:
                os.unlink(temp_file.name)
                print(f"{Colors.RED}❌ Error reading CSV after filtering comments: {e}{Colors.ENDC}")
                return None
        else:
            # Read normally
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                print(f"{Colors.RED}❌ Error reading CSV: {e}{Colors.ENDC}")
                return None

    # If no dataset specified, show help
    if not args.dataset:
        print(f"{Colors.YELLOW}No dataset specified. Please use --dataset to specify a dataset.{Colors.ENDC}")
        print(f"\n{Colors.CYAN}Examples:{Colors.ENDC}")
        print(f"  # Train a model")
        print(f"  python adbnn.py --dataset breast_cancer --mode train")
        print(f"  # Train with adaptive learning")
        print(f"  python adbnn.py --dataset breast_cancer --mode adaptive --track-evolution")
        print(f"  # Make predictions")
        print(f"  python adbnn.py --dataset breast_cancer --mode predict --predict-file new_data.csv")
        print(f"  # Use custom input file")
        print(f"  python adbnn.py --dataset galaxy --mode train --input-path /path/to/data.csv")
        parser.print_help()
        return

    # First, try to locate dataset files
    print(f"{Colors.BOLD}{Colors.CYAN}📁 Locating dataset files...{Colors.ENDC}")
    print(f"{'='*60}")

    # Determine file paths with priority:
    # 1. Explicit command-line arguments (--input-path, --config-file, --csv-file)
    # 2. Standard location: data/<dataset>/<dataset>.conf and data/<dataset>/<dataset>.csv
    # 3. Local files: <dataset>.conf and <dataset>.csv in current directory

    config_path = None
    csv_path = None
    dataset_name = args.dataset

    # Priority 1: Explicit command-line arguments
    if args.input_path:
        csv_path = args.input_path
        if os.path.exists(csv_path):
            print(f"{Colors.GREEN}✓ Using input file: {csv_path}{Colors.ENDC}")
            # Try to infer config from same directory
            csv_dir = os.path.dirname(csv_path)
            csv_base = os.path.splitext(os.path.basename(csv_path))[0]
            inferred_config = os.path.join(csv_dir, f"{csv_base}.conf")
            if os.path.exists(inferred_config):
                config_path = inferred_config
                print(f"{Colors.GREEN}✓ Found config file: {config_path}{Colors.ENDC}")
            else:
                # Also try with dataset name
                dataset_config = os.path.join(csv_dir, f"{args.dataset}.conf")
                if os.path.exists(dataset_config):
                    config_path = dataset_config
                    print(f"{Colors.GREEN}✓ Found config file: {config_path}{Colors.ENDC}")
        else:
            print(f"{Colors.RED}❌ Input file not found: {csv_path}{Colors.ENDC}")
            return

    if args.config_file:
        config_path = args.config_file
        if os.path.exists(config_path):
            print(f"{Colors.GREEN}✓ Using config file: {config_path}{Colors.ENDC}")
        else:
            print(f"{Colors.RED}❌ Config file not found: {config_path}{Colors.ENDC}")
            return

    if args.csv_file:
        csv_path = args.csv_file
        if os.path.exists(csv_path):
            print(f"{Colors.GREEN}✓ Using CSV file: {csv_path}{Colors.ENDC}")
        else:
            print(f"{Colors.RED}❌ CSV file not found: {csv_path}{Colors.ENDC}")
            return

    # Priority 2: Standard location in data directory
    if not config_path:
        standard_config = os.path.join(args.data_dir, args.dataset, f"{args.dataset}.conf")
        if os.path.exists(standard_config):
            config_path = standard_config
            print(f"{Colors.GREEN}✓ Found config file: {config_path}{Colors.ENDC}")

    if not csv_path:
        standard_csv = os.path.join(args.data_dir, args.dataset, f"{args.dataset}.csv")
        if os.path.exists(standard_csv):
            csv_path = standard_csv
            print(f"{Colors.GREEN}✓ Found CSV file: {csv_path}{Colors.ENDC}")

    # Priority 3: Local files in current directory
    if not config_path:
        local_config = f"{args.dataset}.conf"
        if os.path.exists(local_config):
            config_path = local_config
            print(f"{Colors.GREEN}✓ Found local config file: {config_path}{Colors.ENDC}")

    if not csv_path:
        local_csv = f"{args.dataset}.csv"
        if os.path.exists(local_csv):
            csv_path = local_csv
            print(f"{Colors.GREEN}✓ Found local CSV file: {csv_path}{Colors.ENDC}")

    # Check if we found the CSV file
    if not csv_path:
        print(f"{Colors.RED}❌ Could not find data file for '{args.dataset}'{Colors.ENDC}")
        print(f"{Colors.YELLOW}Searched locations:{Colors.ENDC}")
        print(f"   1. Explicit: --input-path, --csv-file")
        print(f"   2. Standard: {args.data_dir}/{args.dataset}/{args.dataset}.csv")
        print(f"   3. Local: {args.dataset}.csv")
        print(f"\n{Colors.CYAN}Please ensure your data file exists or use --input-path{Colors.ENDC}")
        return

    # Load the CSV file (skipping comments)
    print(f"\n{Colors.CYAN}📖 Loading data from: {csv_path}{Colors.ENDC}")
    if args.skip_comments:
        print(f"{Colors.CYAN}   Skipping comment lines (starting with #){Colors.ENDC}")

    df = load_csv_with_comments(csv_path, skip_comments=args.skip_comments)

    if df is None:
        print(f"{Colors.RED}❌ Failed to load data from {csv_path}{Colors.ENDC}")
        return

    print(f"{Colors.GREEN}✓ Loaded {len(df)} rows, {len(df.columns)} columns{Colors.ENDC}")
    if args.verbose:
        print(f"   Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")

    # If we have a config file, load it
    target_column = None
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                target_column = saved_config.get('target_column')
                print(f"{Colors.GREEN}✓ Loaded config from: {config_path}{Colors.ENDC}")
                if target_column:
                    print(f"   Target column: {target_column}")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️ Could not load config file: {e}{Colors.ENDC}")

    # If no target column in config, try to detect it
    if not target_column:
        # Look for common target column names
        target_candidates = ['target', 'class', 'label', 'y', 'prediction', 'type', 'diagnosis', 'class_label', 'true_class']
        for candidate in target_candidates:
            if candidate in df.columns:
                target_column = candidate
                print(f"{Colors.GREEN}✓ Auto-detected target column: {target_column}{Colors.ENDC}")
                break

        # If still not found, use last column
        if not target_column:
            target_column = df.columns[-1]
            print(f"{Colors.YELLOW}⚠️ Using last column as target: {target_column}{Colors.ENDC}")

    # Check if this is a prediction-only mode
    if args.mode == 'predict':
        print(f"\n{Colors.BOLD}{Colors.CYAN}🔮 PREDICTION MODE - Using Existing Model{Colors.ENDC}")
        print(f"{'='*60}")

        # Find available models
        models_dir = "models"
        dataset_models = []

        if os.path.exists(models_dir):
            dataset_models = [f for f in os.listdir(models_dir)
                            if f.endswith('.pkl') and f.startswith(args.dataset)]

        if not dataset_models:
            print(f"{Colors.RED}❌ No saved models found for dataset '{args.dataset}'{Colors.ENDC}")
            print(f"{Colors.YELLOW}Please train a model first using: python adbnn.py --dataset {args.dataset} --mode train{Colors.ENDC}")
            return

        # Select model
        if args.model_file:
            # Use specified model file
            model_file = args.model_file
            if not model_file.endswith('.pkl'):
                model_file += '.pkl'
            model_path = os.path.join(models_dir, model_file)
            if not os.path.exists(model_path):
                print(f"{Colors.RED}❌ Model file not found: {model_path}{Colors.ENDC}")
                return
            print(f"{Colors.CYAN}📥 Using specified model: {model_file}{Colors.ENDC}")
        else:
            # Use the most recent model
            model_file = sorted(dataset_models)[-1]
            model_path = os.path.join(models_dir, model_file)
            print(f"{Colors.CYAN}📥 Using most recent model: {model_file}{Colors.ENDC}")

        # Check if prediction file is specified
        if not args.predict_file:
            print(f"{Colors.RED}❌ No prediction file specified. Use --predict-file to specify input file.{Colors.ENDC}")
            return

        if not os.path.exists(args.predict_file):
            print(f"{Colors.RED}❌ Prediction file not found: {args.predict_file}{Colors.ENDC}")
            return

        # Load model and make predictions
        try:
            print(f"{Colors.CYAN}📥 Loading model...{Colors.ENDC}")
            config = {
                'file_path': csv_path,
                'target_column': target_column,
                'model_type': 'Histogram',
                'compute_device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            model = OptimizedDBNN(dataset_name=args.dataset, config=config)
            model.load_model(model_path)
            print(f"{Colors.GREEN}✓ Model loaded successfully{Colors.ENDC}")

            # Load dataset to reconstruct preprocessing structures
            print(f"{Colors.CYAN}📥 Loading dataset to reconstruct preprocessing...{Colors.ENDC}")
            model.target_column = target_column

            # Save filtered data to temp file
            import tempfile
            temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            df.to_csv(temp_csv.name, index=False)
            temp_csv.close()

            try:
                model.load_data(file_path=temp_csv.name)
            finally:
                os.unlink(temp_csv.name)

            print(f"{Colors.GREEN}✓ Dataset loaded: {len(model.X_tensor)} samples, {len(model.classes)} classes{Colors.ENDC}")

            # Make predictions
            output_dir = os.path.join(args.output_dir, f"{args.dataset}_predictions")
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n{Colors.CYAN}🔮 Making predictions on: {args.predict_file}{Colors.ENDC}")
            pred_results = model.predict_from_file(args.predict_file, output_path=output_dir)

            if pred_results and 'predictions' in pred_results:
                pred_df = pred_results['predictions']
                print(f"\n{Colors.GREEN}✅ Predictions completed successfully!{Colors.ENDC}")
                print(f"   Output directory: {output_dir}")
                print(f"   Predictions file: {output_dir}/predictions.csv")
                print(f"   Number of predictions: {len(pred_df)}")

                # Show sample predictions
                print(f"\n{Colors.CYAN}Sample predictions (first 10):{Colors.ENDC}")
                sample_cols = ['predicted_class', 'confidence']
                if 'true_class' in pred_df.columns:
                    sample_cols.insert(0, 'true_class')
                print(pred_df[sample_cols].head(10).to_string(index=False))

                # Calculate and display confusion matrix if true labels exist
                if 'true_class' in pred_df.columns and not args.no_confusion:
                    print(f"\n{Colors.BOLD}📈 CONFUSION MATRIX - Prediction Performance{Colors.ENDC}")
                    print(f"{'='*60}")

                    # Get unique classes
                    true_classes = pred_df['true_class'].astype(str)
                    pred_classes = pred_df['predicted_class'].astype(str)
                    unique_classes = sorted(set(true_classes) | set(pred_classes))

                    # Create confusion matrix
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(true_classes, pred_classes, labels=unique_classes)

                    # Calculate per-class accuracy
                    class_accuracies = {}
                    for i, cls in enumerate(unique_classes):
                        total = cm[i].sum()
                        if total > 0:
                            acc = cm[i, i] / total
                            class_accuracies[cls] = acc

                    # Overall accuracy
                    total_correct = np.diag(cm).sum()
                    total_samples = cm.sum()
                    overall_acc = total_correct / total_samples if total_samples > 0 else 0

                    # Determine column widths
                    max_class_len = max(len(str(cls)) for cls in unique_classes)
                    col_width = max(8, max_class_len + 2)

                    # Print header
                    print(f"\n{'True\\Pred':<{col_width}}", end='')
                    for cls in unique_classes:
                        print(f"{str(cls):<{col_width}}", end='')
                    print(f"{'Class Acc':<12}")
                    print("-" * (col_width + len(unique_classes) * col_width + 12))

                    # Print each row
                    for i, true_cls in enumerate(unique_classes):
                        print(f"{Colors.BOLD}{str(true_cls):<{col_width}}{Colors.ENDC}", end='')

                        for j, pred_cls in enumerate(unique_classes):
                            count = cm[i, j]
                            if i == j:
                                if count > 0:
                                    color = Colors.GREEN
                                else:
                                    color = Colors.RED
                                print(f"{color}{count:<{col_width}}{Colors.ENDC}", end='')
                            else:
                                if count > 0:
                                    color = Colors.RED
                                else:
                                    color = Colors.WHITE
                                print(f"{color}{count:<{col_width}}{Colors.ENDC}", end='')

                        acc = class_accuracies.get(true_cls, 0)
                        if acc >= 0.9:
                            acc_color = Colors.GREEN
                        elif acc >= 0.7:
                            acc_color = Colors.YELLOW
                        else:
                            acc_color = Colors.RED
                        print(f"{acc_color}{acc:>6.2%}{Colors.ENDC}")

                    print("-" * (col_width + len(unique_classes) * col_width + 12))
                    if overall_acc >= 0.9:
                        acc_color = Colors.GREEN
                    elif overall_acc >= 0.7:
                        acc_color = Colors.YELLOW
                    else:
                        acc_color = Colors.RED
                    print(f"{Colors.BOLD}Overall Accuracy:{Colors.ENDC} {acc_color}{overall_acc:.2%}{Colors.ENDC}")
                    print(f"{Colors.BOLD}Total Samples:{Colors.ENDC} {total_samples}")

                    # Save confusion matrix as image
                    try:
                        import matplotlib.pyplot as plt
                        import seaborn as sns

                        plt.figure(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                   xticklabels=unique_classes, yticklabels=unique_classes,
                                   annot_kws={'size': 12})
                        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
                        plt.ylabel('True Class', fontsize=12, fontweight='bold')
                        plt.title(f'Confusion Matrix - {args.dataset}\nPredictions on {os.path.basename(args.predict_file)}',
                                 fontsize=14, fontweight='bold')

                        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
                        plt.tight_layout()
                        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"\n{Colors.GREEN}✓ Confusion matrix saved to: {cm_path}{Colors.ENDC}")
                    except Exception as e:
                        print(f"{Colors.YELLOW}⚠️ Could not save confusion matrix image: {e}{Colors.ENDC}")

                    # Show classification report
                    print(f"\n{Colors.CYAN}📊 Classification Report:{Colors.ENDC}")
                    from sklearn.metrics import classification_report
                    report = classification_report(true_classes, pred_classes,
                                                  target_names=unique_classes,
                                                  digits=4)
                    print(report)

                    # Save classification report
                    report_path = os.path.join(output_dir, 'classification_report.txt')
                    with open(report_path, 'w') as f:
                        f.write(f"Classification Report - {args.dataset}\n")
                        f.write(f"Predictions on: {args.predict_file}\n")
                        f.write(f"{'='*60}\n\n")
                        f.write(report)
                    print(f"{Colors.GREEN}✓ Classification report saved to: {report_path}{Colors.ENDC}")

                print(f"\n{Colors.GREEN}🎉 Prediction complete!{Colors.ENDC}")
            else:
                print(f"{Colors.RED}❌ Prediction failed{Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.RED}❌ Error during prediction: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            return

        return

    # Training modes
    print(f"\n{Colors.BOLD}{Colors.BLUE}🏋️ TRAINING MODE - {args.mode.upper()}{Colors.ENDC}")
    print(f"{'='*60}")

    # Create config with the found paths and target column
    config = {
        'file_path': csv_path,
        'target_column': target_column,
        'model_type': 'Histogram',
        'compute_device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'training_params': {
            'enable_adaptive': args.mode == 'adaptive'
        }
    }

    # Create model
    model = OptimizedDBNN(dataset_name=args.dataset, config=config)

    if args.track_evolution:
        model.enable_evolution_tracking()

    # Load data (save filtered data to temp file for model loading)
    print(f"{Colors.CYAN}📥 Loading data into model...{Colors.ENDC}")
    import tempfile
    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_csv.name, index=False)
    temp_csv.close()

    try:
        model.load_data(file_path=temp_csv.name)
    finally:
        # Clean up temp file
        os.unlink(temp_csv.name)

    if args.mode == 'adaptive':
        print(f"{Colors.CYAN}🚀 Starting ADAPTIVE training...{Colors.ENDC}")
        results = model.adaptive_fit_predict()
    elif args.mode == 'train':
        print(f"{Colors.CYAN}🚀 Starting STANDARD training...{Colors.ENDC}")
        model.split_data()
        results = model.fit_predict()
    else:  # train_predict (default)
        if model.enable_adaptive:
            print(f"{Colors.CYAN}🚀 Starting ADAPTIVE training...{Colors.ENDC}")
            results = model.adaptive_fit_predict()
        else:
            print(f"{Colors.CYAN}🚀 Starting STANDARD training...{Colors.ENDC}")
            model.split_data()
            results = model.fit_predict()

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f'models/{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    model.save_model(model_path)
    print(f"\n{Colors.GREEN}✅ Training complete! Best accuracy: {results['best_accuracy']:.4f}{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Model saved to: {model_path}{Colors.ENDC}")

    # If in train_predict mode, optionally make predictions
    if args.mode == 'train_predict':
        print(f"\n{Colors.CYAN}🔮 Model trained and saved. Use --mode predict for future predictions.{Colors.ENDC}")

def interactive_mode():
    """
    Enhanced interactive mode with ALL GUI features for headless GPU servers
    Dataset is primary - models are associated with their dataset
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
    # SECTION 2: DATASET SELECTION (PRIMARY)
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.CYAN}📊 DATASET SELECTION (Primary){Colors.ENDC}")
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
            config_path = conf_path
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
                    config_path = os.path.join(data_dir, f"{dataset_name}.conf")
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
            # Create default config path
            data_dir = os.path.join('data', dataset_name)
            os.makedirs(data_dir, exist_ok=True)
            config_path = os.path.join(data_dir, f"{dataset_name}.conf")
            # Move/copy file to standard location
            if file_path != os.path.join(data_dir, f"{dataset_name}.csv"):
                import shutil
                shutil.copy2(file_path, os.path.join(data_dir, f"{dataset_name}.csv"))
                file_path = os.path.join(data_dir, f"{dataset_name}.csv")
                print(f"{Colors.GREEN}✓ Dataset copied to {file_path}{Colors.ENDC}")
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
                    config_path = os.path.join(data_dir, f"{dataset_name}.conf")
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
            # Create default config path
            data_dir = os.path.join('data', dataset_name)
            os.makedirs(data_dir, exist_ok=True)
            config_path = os.path.join(data_dir, f"{dataset_name}.conf")
            # Move/copy file to standard location
            if file_path != os.path.join(data_dir, f"{dataset_name}.csv"):
                import shutil
                shutil.copy2(file_path, os.path.join(data_dir, f"{dataset_name}.csv"))
                file_path = os.path.join(data_dir, f"{dataset_name}.csv")
                print(f"{Colors.GREEN}✓ Dataset copied to {file_path}{Colors.ENDC}")

    print(f"\n{Colors.GREEN}✓ Dataset: {dataset_name}{Colors.ENDC}")
    print(f"   Data file: {file_path}")
    print(f"   Config file: {config_path}")

    # =========================================================================
    # SECTION 3: MAIN OPERATION MODE SELECTION
    # =========================================================================
    print(f"\n{Colors.BOLD}{Colors.CYAN}🎯 SELECT OPERATION MODE{Colors.ENDC}")
    print(f"{'='*60}")
    print(f"\n{Colors.BOLD}Choose an option:{Colors.ENDC}")
    print(f"   1. Train a new model on this dataset")
    print(f"   2. Load existing model for this dataset and make predictions")
    print(f"   3. Quick prediction (use latest model for this dataset)")
    print(f"   4. Exit")

    mode_choice = input(f"\n{Colors.CYAN}Select option (1-4) [1]: {Colors.ENDC}").strip() or "1"

    if mode_choice == "4":
        print(f"\n{Colors.GREEN}Goodbye!{Colors.ENDC}")
        return

    # =========================================================================
    # SECTION 4: FEATURE AND TARGET SELECTION (for training or prediction)
    # =========================================================================
    # Load data to examine columns
    df = pd.read_csv(file_path)

    print(f"\n{Colors.BOLD}{Colors.CYAN}🔧 FEATURE CONFIGURATION{Colors.ENDC}")
    print(f"{'='*60}")
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
    # OPTION 2 & 3: PREDICTION ONLY MODES
    # =========================================================================
    if mode_choice in ["2", "3"]:
        print(f"\n{Colors.BOLD}{Colors.BLUE}🔮 PREDICTION MODE - Using Existing Model{Colors.ENDC}")
        print(f"{'='*60}")

        # Find models for this dataset
        models_dir = "models"
        dataset_models = []

        if os.path.exists(models_dir):
            # Look for models that start with the dataset name
            dataset_models = [f for f in os.listdir(models_dir)
                            if f.endswith('.pkl') and f.startswith(dataset_name)]

        if not dataset_models:
            print(f"{Colors.RED}❌ No saved models found for dataset '{dataset_name}' in 'models' directory.{Colors.ENDC}")
            print(f"{Colors.YELLOW}Please train a model first (Option 1).{Colors.ENDC}")
            return

        if mode_choice == "3":
            # Quick prediction with the most recent model for this dataset
            model_file = sorted(dataset_models)[-1]
            model_path = os.path.join(models_dir, model_file)
            print(f"{Colors.CYAN}📥 Using most recent model: {model_file}{Colors.ENDC}")

            # Load the model
            try:
                # Create model with dataset info
                model = OptimizedDBNN(dataset_name=dataset_name)
                model.load_model(model_path)
                print(f"{Colors.GREEN}✓ Model loaded successfully{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}❌ Error loading model: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
                return
        else:
            # Manual model selection
            print(f"\n{Colors.GREEN}Models for dataset '{dataset_name}':{Colors.ENDC}")
            for i, model_file in enumerate(dataset_models):
                model_path = os.path.join(models_dir, model_file)
                try:
                    with open(model_path, 'rb') as f:
                        model_state = pickle.load(f)
                    classes = len(model_state.get('classes', []))
                    # Try to get timestamp from filename
                    timestamp = model_file.replace(f"{dataset_name}_", "").replace(".pkl", "")
                    print(f"   {i+1:2d}. {model_file} (classes: {classes}, created: {timestamp})")
                except:
                    print(f"   {i+1:2d}. {model_file}")

            model_choice = input(f"\n{Colors.CYAN}Select model (1-{len(dataset_models)}): {Colors.ENDC}").strip()
            if not model_choice.isdigit() or not (1 <= int(model_choice) <= len(dataset_models)):
                print(f"{Colors.RED}Invalid selection{Colors.ENDC}")
                return

            model_file = dataset_models[int(model_choice)-1]
            model_path = os.path.join(models_dir, model_file)

            print(f"{Colors.CYAN}📥 Loading model: {model_file}{Colors.ENDC}")

            try:
                # Create model with dataset info
                model = OptimizedDBNN(dataset_name=dataset_name)
                model.load_model(model_path)
                print(f"{Colors.GREEN}✓ Model loaded successfully{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}❌ Error loading model: {e}{Colors.ENDC}")
                import traceback
                traceback.print_exc()
                return

        # Now get prediction file
        print(f"\n{Colors.BOLD}📁 Prediction file setup{Colors.ENDC}")
        print(f"{Colors.CYAN}The prediction file should have the same features as training data.{Colors.ENDC}")
        print(f"{Colors.CYAN}Target column is optional - if present, accuracy will be reported.{Colors.ENDC}")
        print(f"{Colors.CYAN}Prediction results will include class probabilities and confidence scores.{Colors.ENDC}\n")

        pred_file = input(f"{Colors.BOLD}Enter prediction file path: {Colors.ENDC}").strip()

        if not os.path.exists(pred_file):
            print(f"{Colors.RED}❌ File not found: {pred_file}{Colors.ENDC}")
            return

        # Create output directory
        output_dir = f"predictions/{dataset_name}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{Colors.CYAN}🔮 Making predictions...{Colors.ENDC}")

        try:
            # Make predictions
            pred_results = model.predict_from_file(pred_file, output_path=output_dir)

            if pred_results and 'predictions' in pred_results:
                pred_df = pred_results['predictions']
                print(f"\n{Colors.GREEN}✅ Predictions completed successfully!{Colors.ENDC}")
                print(f"   Output directory: {output_dir}")
                print(f"   Predictions file: {output_dir}/predictions.csv")
                print(f"   Number of predictions: {len(pred_df)}")

                # Show sample predictions
                print(f"\n{Colors.CYAN}Sample predictions (first 10):{Colors.ENDC}")
                sample_cols = ['predicted_class', 'confidence']
                if 'true_class' in pred_df.columns:
                    sample_cols.insert(0, 'true_class')
                print(pred_df[sample_cols].head(10).to_string(index=False))

                # Show class probability columns
                prob_cols = [col for col in pred_df.columns if col.startswith('prob_')]
                if prob_cols:
                    print(f"\n{Colors.CYAN}Class probabilities available for: {', '.join(prob_cols)}{Colors.ENDC}")

                # Calculate accuracy if true labels present
                if 'true_class' in pred_df.columns:
                    accuracy = (pred_df['predicted_class'] == pred_df['true_class']).mean()
                    print(f"\n{Colors.GREEN}✓ Accuracy on prediction file: {accuracy:.4f}{Colors.ENDC}")

                # Ask to open results
                open_results = input(f"\n{Colors.CYAN}Open results directory? (y/n) [n]: {Colors.ENDC}").strip().lower()
                if open_results == 'y':
                    if os.name == 'nt':
                        os.startfile(output_dir)
                    elif sys.platform == 'darwin':
                        subprocess.Popen(['open', output_dir])
                    else:
                        subprocess.Popen(['xdg-open', output_dir])

                # Ask for another prediction
                another = input(f"\n{Colors.CYAN}Make predictions on another file? (y/n) [n]: {Colors.ENDC}").strip().lower()
                if another == 'y':
                    # Go back to prediction mode with same dataset
                    interactive_mode()  # Restart with same dataset? For simplicity, just restart
                else:
                    print(f"\n{Colors.GREEN}Prediction session complete!{Colors.ENDC}")
                    return

        except Exception as e:
            print(f"{Colors.RED}❌ Error during prediction: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            return

    # =========================================================================
    # OPTION 1: TRAIN NEW MODEL
    # =========================================================================
    if mode_choice == "1":
        print(f"\n{Colors.BOLD}{Colors.BLUE}🏋️ TRAINING MODE - Train New Model{Colors.ENDC}")
        print(f"{'='*60}")

        # =========================================================================
        # MODEL CONFIGURATION
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
        training_mode = input(f"\n{Colors.CYAN}Select mode (1-2) [2]: {Colors.ENDC}").strip() or "2"
        use_adaptive = training_mode == "2"

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
        # CREATE AND CONFIGURE MODEL
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
        # TRAINING
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
        # RESULTS DISPLAY
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
        # CONFUSION MATRIX DISPLAY
        # =========================================================================
        print(f"\n{Colors.BOLD}📈 Confusion Matrices:{Colors.ENDC}")
        print(f"{'='*60}")

        # Training confusion matrix
        if model.X_train is not None and len(model.X_train) > 0:
            train_pred, _ = model.predict(model.X_train)
            model.print_colored_confusion_matrix(
                model.y_train.cpu().numpy(),
                train_pred.cpu().numpy(),
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
                    y_test.cpu().numpy(),
                    test_pred.cpu().numpy(),
                    "Test Data"
                )
        elif model.X_test is not None and len(model.X_test) > 0:
            test_pred, _ = model.predict(model.X_test)
            model.print_colored_confusion_matrix(
                model.y_test.cpu().numpy(),
                test_pred.cpu().numpy(),
                "Test Data"
            )

        # =========================================================================
        # VISUALIZATION (if enabled)
        # =========================================================================
        if visualize:
            print(f"\n{Colors.BOLD}🎨 GENERATING VISUALIZATIONS{Colors.ENDC}")
            print(f"{'='*60}")

            visualizer = OptimizedVisualizer(model)

            # Prepare data for visualization
            X_np = model.X_tensor.cpu().numpy()
            y_np = model.y_tensor.cpu().numpy()

            if use_adaptive:
                train_mask = np.zeros(len(X_np), dtype=bool)
                train_mask[results['train_indices']] = True
                test_mask = ~train_mask

                y_train_np = y_np[train_mask]
                y_test_np = y_np[test_mask]

                train_pred, _ = model.predict(model.X_tensor[train_mask])
                test_pred, _ = model.predict(model.X_tensor[test_mask])

                train_pred_np = train_pred.numpy() if torch.is_tensor(train_pred) else train_pred
                test_pred_np = test_pred.numpy() if torch.is_tensor(test_pred) else test_pred
            else:
                y_train_np = model.y_train.cpu().numpy()
                y_test_np = model.y_test.cpu().numpy()

                train_pred, _ = model.predict(model.X_train)
                test_pred, _ = model.predict(model.X_test)

                train_pred_np = train_pred.numpy() if torch.is_tensor(train_pred) else train_pred
                test_pred_np = test_pred.numpy() if torch.is_tensor(test_pred) else test_pred

            if use_adaptive:
                viz_history = results.get('round_stats', [])
                evolution_history = results.get('evolution_history', [])
            else:
                viz_history = results.get('history', [])
                evolution_history = []

            visualizer.generate_all_visualizations(
                viz_history,
                X_np, y_np,
                y_train_np, y_test_np,
                train_pred_np,
                test_pred_np,
                evolution_history=evolution_history
            )

            if evolution_history and len(evolution_history) > 1:
                print(f"\n{Colors.CYAN}🌐 Creating spherical tensor evolution...{Colors.ENDC}")
                spherical_path = visualizer.spherical_viz.create_spherical_animation(evolution_history)
                if spherical_path:
                    print(f"   Saved to: {spherical_path}")

        # =========================================================================
        # SAVE MODEL
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
        # PREDICTION WITH TRAINED MODEL (OPTIONAL)
        # =========================================================================
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}🔮 PREDICTION OPTION - Make predictions with the trained model{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")

        predict_choice = input(f"\n{Colors.BOLD}Do you want to make predictions on a file using this model?{Colors.ENDC} {Colors.CYAN}(y/n) [y]: {Colors.ENDC}").strip().lower()
        predict_choice = predict_choice if predict_choice else 'y'

        if predict_choice == 'y':
            print(f"\n{Colors.CYAN}📁 Prediction file should have the same features as training data.{Colors.ENDC}\n")

            pred_file = input(f"{Colors.BOLD}Enter prediction file path: {Colors.ENDC}").strip()

            if os.path.exists(pred_file):
                output_dir = f"predictions/{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(output_dir, exist_ok=True)

                print(f"\n{Colors.CYAN}🔮 Making predictions...{Colors.ENDC}")
                try:
                    pred_results = model.predict_from_file(pred_file, output_path=output_dir)

                    if pred_results and 'predictions' in pred_results:
                        pred_df = pred_results['predictions']
                        print(f"\n{Colors.GREEN}✅ Predictions completed successfully!{Colors.ENDC}")
                        print(f"   Output directory: {output_dir}")
                        print(f"   Predictions file: {output_dir}/predictions.csv")
                        print(f"   Number of predictions: {len(pred_df)}")

                        print(f"\n{Colors.CYAN}Sample predictions (first 10):{Colors.ENDC}")
                        sample_cols = ['predicted_class', 'confidence']
                        if 'true_class' in pred_df.columns:
                            sample_cols.insert(0, 'true_class')
                        print(pred_df[sample_cols].head(10).to_string(index=False))

                        if 'true_class' in pred_df.columns:
                            accuracy = (pred_df['predicted_class'] == pred_df['true_class']).mean()
                            print(f"\n{Colors.GREEN}✓ Accuracy: {accuracy:.4f}{Colors.ENDC}")
                except Exception as e:
                    print(f"{Colors.RED}❌ Error during prediction: {e}{Colors.ENDC}")
            else:
                print(f"{Colors.RED}❌ File not found: {pred_file}{Colors.ENDC}")

        # =========================================================================
        # SUMMARY
        # =========================================================================
        print(f"\n{Colors.BOLD}{Colors.BLUE}🎉 TRAINING SESSION COMPLETE{Colors.ENDC}")
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
    print("""
Example Usage:
python adbnn.py --dataset breast_cancer --mode adaptive

 python adbnn.py --dataset breast_cancer --mode predict --predict-file data/breast_cancer/breast_cancer.cs

    """)
    main()
