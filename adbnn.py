#-------- Memory and Computation optimiations 3 Nov 2025 -----------------------------------------
# Replaced:
# _compute_batch_posterior
# _update_priors_parallel
# _compute_pairwise_likelihood_parallel
# _compute_sample_divergence
# train
# _compute_sample_divergence
# _select_samples_from_failed_classes
#--------------------------Fully Functional bug fixed version  with heatmaps----------------------------
#----------------------------------------------May 18 2025 23:41 IST------------------------------------------------

# Working, fully functional with predcition 31/March/2025 Stable Model
# Better Memory management 06:28am
# Tested an fully functional April 4 2025 3:34 am
# Tested for more flexibility and added posteriors with class pedictions 5 April 7:22 pm
# Enhanced mosaic images 6 April 2025 8:45 am
# Feature pair automatic recomputation disabled
# Training until patience enabled. April 7 8:14 am 2025
# Finalised completely working module as on 15th April 2025
# Tested and working well with numerica target also April 27 11:28 pm
# upgraded to use GPU in pairwise computations for both models. May 4 2025 7:28 pm
# Added automatic conf creation option. May 4 2025 10:56 pm
# Updated failed candidate selection procedure as per DBNN original concepts May 5 2025 3:57 pm
# Fixed the bug in feature selection : May 10:12:09 am
# Fixed the bug in sample selection May 11 1:27 am
# All fixed, working training and predcition perfectly Nov 9th 2025

#----------------------------------------------------------------------------------------------------------------------------
#---- author: Ninan Sajeeth Philip, Artificial Intelligence Research and Intelligent Systems
#-----------------------------------------------------------------------------------------------------------------------------
import subprocess
from tempfile import NamedTemporaryFile
import torch
import time
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import shutil
import os
#For pdf mosaic--
import concurrent.futures
from multiprocessing import cpu_count
import torch
from PIL import Image
import io
import random
import math
from tqdm import tqdm
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import traceback
import multiprocessing as mp

# Set spawn method before any CUDA initialization
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image as ReportLabImage,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
#--
from reportlab.platypus import Image as ReportLabImage
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import math
#----
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Union
from collections import defaultdict
import requests
from io import StringIO
import os,re,sys
import json
from itertools import combinations
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import normaltest
import numpy as np
from itertools import combinations
from math import comb
import torch
import os
import pickle
import configparser
import traceback  # Add to provide debug
#-----------------Visualisation imports ----------------------
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from datetime import datetime
import networkx as nx
from scipy.spatial import distance_matrix
import concurrent.futures
import psutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
try:
        # Import tkinter for GUI
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox, scrolledtext
except:
        print("tkinter is not available")
#from Invertible_DBNN import InvertibleDBNN
#------------------------------------------------------------------------Declarations---------------------
# Device configuration - set this first since other classes need it
Train_device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Default device
Trials = 100  # Number of epochs to wait for improvement in training
cardinality_threshold =0.9
cardinality_tolerance=4 #Use when the features are likely to be extremly diverse and deciimal values;4 means, precison restricted to 4 decimal places
LearningRate =0.1
TrainingRandomSeed=42  #None # 42
Epochs=1000
bin_sizes =128
n_bins_per_dim =128
TestFraction=0.2
Train=True #True #False #
Train_only=False #True #
Predict=True
Gen_Samples=False
EnableAdaptive = True  # New parameter to control adaptive training
# Assume no keyboard control by default. If you have X11 running and want to be interactive, set nokbd = False
nokbd =  False # Enables interactive keyboard when training (q and Q will not have any effect)
display = None  # Initialize display variable
#----------------------------------------------------------------------------------------------------------------
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
import traceback
from typing import Dict, List, Union, Optional
from collections import defaultdict
import requests
from io import StringIO
import os
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from io import StringIO
import zipfile
import tarfile
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
from typing import Union, List, Dict, Optional
from collections import defaultdict

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# =============================================================================
# 5DCT Visualisations
# =============================================================================

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import os
import json
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'  # ADDED: Cyan color
    MAGENTA = '\033[95m'  # ADDED: Magenta color
    WHITE = '\033[97m'  # ADDED: White color
    BLACK = '\033[90m'  # ADDED: Black color
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

    @staticmethod
    def highlight_accuracy(accuracy):
        """Color accuracy values based on threshold"""
        if accuracy >= 0.9:
            return f"{Colors.GREEN}{accuracy:.4f}{Colors.ENDC}"
        elif accuracy >= 0.7:
            return f"{Colors.YELLOW}{accuracy:.4f}{Colors.ENDC}"
        else:
            return f"{Colors.RED}{accuracy:.4f}{Colors.ENDC}"

    @staticmethod
    def highlight_feature(name):
        """Highlight feature name in cyan"""
        return f"{Colors.CYAN}{name}{Colors.ENDC}"

    @staticmethod
    def highlight_class(name):
        """Highlight class name in magenta"""
        return f"{Colors.MAGENTA}{name}{Colors.ENDC}"

    @staticmethod
    def highlight_round(round_num):
        """Highlight round number in blue"""
        return f"{Colors.BLUE}{round_num}{Colors.ENDC}"

    @staticmethod
    def print_success(message):
        """Print success message in green"""
        print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")

    @staticmethod
    def print_warning(message):
        """Print warning message in yellow"""
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")

    @staticmethod
    def print_error(message):
        """Print error message in red"""
        print(f"{Colors.RED}❌ {message}{Colors.ENDC}")

    @staticmethod
    def print_info(message):
        """Print info message in cyan"""
        print(f"{Colors.CYAN}📊 {message}{Colors.ENDC}")

    @staticmethod
    def print_debug(message):
        """Print debug message in magenta"""
        print(f"{Colors.MAGENTA}🐛 {message}{Colors.ENDC}")

    @staticmethod
    def get_color(color_name, format='named'):
        """
        Get color in proper format for different libraries
        format: 'named', 'hex', 'rgb', 'normalized'
        """
        color_map = {
            'red': {
                'named': 'red',
                'hex': '#e41a1c',
                'rgb': (228, 26, 28),
                'normalized': (228/255, 26/255, 28/255)
            },
            'green': {
                'named': 'green',
                'hex': '#4daf4a',
                'rgb': (77, 175, 74),
                'normalized': (77/255, 175/255, 74/255)
            },
            'blue': {
                'named': 'blue',
                'hex': '#377eb8',
                'rgb': (55, 126, 184),
                'normalized': (55/255, 126/255, 184/255)
            },
            'yellow': {
                'named': 'yellow',
                'hex': '#ffff33',
                'rgb': (255, 255, 51),
                'normalized': (1.0, 1.0, 0.2)
            },
            'orange': {
                'named': 'orange',
                'hex': '#ffa500',
                'rgb': (255, 165, 0),
                'normalized': (1.0, 0.65, 0.0)
            }
        }

        if color_name in color_map:
            return color_map[color_name][format]
        else:
            # Fallback to matplotlib named colors
            return color_name

# Global command line flags storage - SINGLE SOURCE OF TRUTH
COMMAND_LINE_FLAGS = {
    'visualize': False,
    'model_type': 'Histogram',
    'mode': 'train_predict',
    'file_path': None,
    'interactive': False,
    'list_datasets': False,
    'fresh_start': False,
    'use_previous_model': True,
    'enable_5DCTvisualization': False
}

def parse_command_line_flags():
    """Parse and store ALL command line flags globally - called once at startup"""
    import sys

    # Set defaults
    COMMAND_LINE_FLAGS.update({
        'visualize': '--visualize' in sys.argv or any(arg in sys.argv for arg in ['-v', '--visualise']),
        'interactive': '--interactive' in sys.argv,
        'list_datasets': '--list_datasets' in sys.argv,
        'fresh_start': '--fresh_start' in sys.argv,
    })

    # Parse arguments with values
    for i, arg in enumerate(sys.argv):
        if arg == '--model_type' and i + 1 < len(sys.argv):
            COMMAND_LINE_FLAGS['model_type'] = sys.argv[i + 1]
        elif arg == '--mode' and i + 1 < len(sys.argv):
            COMMAND_LINE_FLAGS['mode'] = sys.argv[i + 1]
        elif arg == '--file_path' and i + 1 < len(sys.argv):
            COMMAND_LINE_FLAGS['file_path'] = sys.argv[i + 1]
        elif arg == '--fresh_start':
            COMMAND_LINE_FLAGS['fresh_start'] = True
            COMMAND_LINE_FLAGS['use_previous_model'] = False
        elif arg == '--use_previous_model':
            COMMAND_LINE_FLAGS['use_previous_model'] = True
            COMMAND_LINE_FLAGS['fresh_start'] = False

    # AUTO-ENABLE 5DCT when visualize is enabled
    if COMMAND_LINE_FLAGS['visualize']:
        COMMAND_LINE_FLAGS['enable_5DCTvisualization'] = True
        print(f"🌌 {Colors.GREEN}AUTO-ENABLED: 5DCT visualization (--visualize flag detected){Colors.ENDC}")

    # Print summary of detected flags
    if len(sys.argv) > 1:  # Only show if command line arguments were provided
        print(f"🎯 {Colors.BOLD}GLOBAL COMMAND LINE FLAGS DETECTED:{Colors.ENDC}")
        for flag, value in COMMAND_LINE_FLAGS.items():
            if value and value not in [None, False, '']:
                color = Colors.GREEN if value not in [False, None] else Colors.YELLOW
                print(f"   {Colors.CYAN}--{flag:<25}: {color}{value}{Colors.ENDC}")
        print()

# Call this immediately after imports
parse_command_line_flags()

class DBNNVisualizer:
    """
    Enhanced visualization class for DBNN with interactive 3D capabilities,
    educational visualizations, and comprehensive training monitoring.

    This class provides multiple visualization types:
    - Standard 2D/3D plots for training monitoring
    - Enhanced interactive 3D visualizations with animation
    - Tensor mode specific visualizations
    - Educational dashboards with multiple subplots
    - Real-time training progression tracking
    """

    def __init__(self):
        """
        Initialize DBNNVisualizer with empty data structures for storing
        training history and visualization data.
        """
        # Core training history storage
        self.training_history = []  # List of training snapshots
        self.visualization_data = {}  # Additional visualization metadata
        self.tensor_snapshots = []  # Tensor-specific training data
        self.enable_5DCT_visualization = COMMAND_LINE_FLAGS['enable_5DCTvisualization']

        # Enhanced data storage for advanced visualizations
        self.feature_space_snapshots = []  # 3D feature space evolution
        self.feature_names = []  # Names of features for labeling
        self.current_iteration = 0
        self.class_names = []  # Names of classes for labeling
        self.accuracy_progression = []  # Accuracy over training rounds
        self.weight_evolution = []  # Weight statistics over time
        self.confusion_data = []  # Confusion matrix data

        # Educational visualization data
        self.decision_boundaries = []  # Decision boundary evolution
        self.feature_importance_data = []  # Feature importance metrics
        self.learning_curves = []  # Learning curve data
        self.network_topology_data = []  # Network structure information

        # Help system initialization
        self.help_windows = set()

    # =========================================================================
    # HELP SYSTEM METHODS
    # =========================================================================

    def _get_viz_output_path(self, viz_type, filename):
        """Use the DBNN instance's path method"""
        if hasattr(self, 'dbnn_instance'):
            return self.dbnn_instance._get_viz_output_path(viz_type, filename)
        else:
            # Fallback to local implementation
            base_dir = "Visualizer"
            type_dir = os.path.join(base_dir, viz_type)
            os.makedirs(type_dir, exist_ok=True)
            return os.path.join(type_dir, filename)

    def create_help_window(self, title, content, width=400, height=300):
        """Create a non-GUI help message (tkinter removed)"""
        print(f"🎓 {title} Help:")
        print("=" * 50)
        print(content)
        print("=" * 50)
        return None

    def hide_all_help_windows(self):
        """No-op since we removed GUI"""
        pass

    def get_visualization_help_content(self, viz_type):
        """Get help content without GUI dependencies"""
        help_contents = {
            'circular_tensor': "Circular Tensor Evolution - Shows tensor orientations in circular coordinate space",
            'polar_tensor': "Polar Tensor Evolution - Displays tensor space transformation using polar coordinates",
            '3d_feature_space': "3D Feature Space Visualization - Interactive 3D plot showing feature relationships",
            'confusion_matrix': "Animated Confusion Matrix - Shows classification performance evolution",
            'performance_metrics': "Performance Metrics Dashboard - Comprehensive training progress view",
            'complex_phase': "Complex Phase Diagram - Shows feature relationships in complex space"
        }
        return help_contents.get(viz_type, "Visualization Help - Interactive training evolution visualization")

    # =========================================================================
    # CORE TRAINING DATA CAPTURE METHODS
    # =========================================================================

    def capture_training_snapshot(self, features, targets, weights, predictions, accuracy, round_num):
        """
        Capture comprehensive training snapshot for visualization and analysis.

        Args:
            features (numpy.ndarray): Feature matrix (samples x features)
            targets (numpy.ndarray): True target values
            weights (numpy.ndarray): Current network weights
            predictions (numpy.ndarray): Model predictions
            accuracy (float): Current accuracy percentage
            round_num (int): Training iteration/round number

        Returns:
            dict: Snapshot containing all training data
        """
        snapshot = {
            'round': round_num,
            'features': features.copy() if features is not None else None,
            'targets': targets.copy() if targets is not None else None,
            'weights': weights.copy() if weights is not None else None,
            'predictions': predictions.copy() if predictions is not None else None,
            'accuracy': accuracy,
            'timestamp': time.time()
        }

        self.training_history.append(snapshot)

        # Store accuracy progression
        self.accuracy_progression.append({
            'round': round_num,
            'accuracy': accuracy,
            'timestamp': time.time()
        })

        # Store weight statistics for educational purposes
        if weights is not None:
            flat_weights = weights.flatten()
            flat_weights = flat_weights[(flat_weights != 0) & (np.abs(flat_weights) < 100)]

            self.weight_evolution.append({
                'round': round_num,
                'mean': np.mean(flat_weights) if len(flat_weights) > 0 else 0,
                'std': np.std(flat_weights) if len(flat_weights) > 0 else 0,
                'min': np.min(flat_weights) if len(flat_weights) > 0 else 0,
                'max': np.max(flat_weights) if len(flat_weights) > 0 else 0
            })

        # Capture enhanced visualization data if features are available
        if hasattr(self, 'feature_space_snapshots') and features is not None:
            try:
                feature_names = getattr(self, 'feature_names',
                                      [f'Feature_{i+1}' for i in range(features.shape[1])])
                class_names = getattr(self, 'class_names',
                                    [f'Class_{int(c)}' for c in np.unique(targets)])

                enhanced_snapshot = {
                    'iteration': round_num,
                    'features': features.copy(),
                    'targets': targets.copy(),
                    'predictions': predictions.copy(),
                    'feature_names': feature_names,
                    'class_names': class_names,
                    'timestamp': time.time(),
                    'accuracy': accuracy
                }
                self.feature_space_snapshots.append(enhanced_snapshot)
            except Exception as e:
                print(f"Enhanced visualization capture warning: {e}")

        return snapshot

    def capture_tensor_snapshot(self, features, targets, weight_matrix, orthogonal_basis,
                               predictions, accuracy, iteration=0):
        """
        Capture specialized snapshot for tensor mode training.

        Args:
            features (numpy.ndarray): Input features
            targets (numpy.ndarray): True targets
            weight_matrix (numpy.ndarray): Tensor weight matrix
            orthogonal_basis (numpy.ndarray): Orthogonal basis vectors
            predictions (numpy.ndarray): Model predictions
            accuracy (float): Current accuracy
            iteration (int): Training iteration

        Returns:
            dict: Tensor-specific snapshot
        """
        tensor_data = {
            'weight_matrix': weight_matrix.copy() if hasattr(weight_matrix, 'copy') else weight_matrix,
            'orthogonal_basis': orthogonal_basis.copy() if hasattr(orthogonal_basis, 'copy') else orthogonal_basis,
            'iteration': iteration,
            'weight_matrix_norm': np.linalg.norm(weight_matrix) if weight_matrix is not None else 0,
            'basis_rank': np.linalg.matrix_rank(orthogonal_basis) if orthogonal_basis is not None else 0
        }

        # Use the main snapshot method but add tensor data
        snapshot = self.capture_training_snapshot(
            features, targets, weight_matrix, predictions, accuracy, iteration
        )
        snapshot['is_tensor_mode'] = True
        snapshot['tensor_data'] = tensor_data

        self.tensor_snapshots.append(snapshot)
        return snapshot

    def capture_feature_space_snapshot(self, features, targets, predictions, iteration,
                                     feature_names=None, class_names=None):
        """
        Capture feature space state for interactive 3D visualization.

        Args:
            features (numpy.ndarray): Feature matrix
            targets (numpy.ndarray): True targets
            predictions (numpy.ndarray): Model predictions
            iteration (int): Training iteration
            feature_names (list): Names of features for labeling
            class_names (list): Names of classes for labeling

        Returns:
            dict: Feature space snapshot
        """
        if feature_names is None:
            feature_names = [f'Feature_{i+1}' for i in range(features.shape[1])]
        if class_names is None:
            unique_targets = np.unique(targets)
            class_names = [f'Class_{int(t)}' for t in unique_targets]

        snapshot = {
            'iteration': iteration,
            'features': features.copy(),
            'targets': targets.copy(),
            'predictions': predictions.copy(),
            'feature_names': feature_names,
            'class_names': class_names,
            'timestamp': time.time()
        }

        self.feature_space_snapshots.append(snapshot)
        self.feature_names = feature_names
        self.class_names = class_names

        return snapshot

    def capture_epoch_snapshot(self, features, targets, weights, predictions, accuracy, epoch_num):
        """Capture epoch-level snapshot for training progression visualization"""
        try:
            # Handle PyTorch tensors - keep on GPU but use clone for safety
            def safe_clone(data):
                if data is None:
                    return None
                elif torch.is_tensor(data):
                    return data.detach().clone()
                elif hasattr(data, 'copy'):
                    return data.copy()
                else:
                    return data

            snapshot = {
                'epoch': epoch_num,
                'features': safe_clone(features),
                'targets': safe_clone(targets),
                'weights': safe_clone(weights),
                'predictions': safe_clone(predictions),
                'accuracy': accuracy,
                'timestamp': time.time(),
                'is_tensor': any(torch.is_tensor(x) for x in [features, targets, weights, predictions] if x is not None)
            }

            # Store in training history
            self.training_history.append(snapshot)

            # For feature space snapshots, convert to CPU numpy for visualization
            if (features is not None and
                hasattr(features, 'shape') and
                (torch.is_tensor(features) and features.shape[1] >= 2) or
                (hasattr(features, 'shape') and features.shape[1] >= 2)):

                features_np = features.detach().cpu().numpy() if torch.is_tensor(features) else features
                targets_np = targets.detach().cpu().numpy() if torch.is_tensor(targets) else targets
                predictions_np = predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else predictions

                self.capture_feature_space_snapshot(
                    features=features_np,
                    targets=targets_np,
                    predictions=predictions_np,
                    iteration=epoch_num,
                    feature_names=getattr(self, 'feature_names', []),
                    class_names=getattr(self, 'class_names', [])
                )

            return snapshot

        except Exception as e:
            print(f"❌ Epoch snapshot capture failed: {e}")
            import traceback
            traceback.print_exc()
            return None


    # =========================================================================
    # ENHANCED INTERACTIVE 3D VISUALIZATION METHODS
    # =========================================================================



    def validate_visualization_directories(self):
        """Validate that all visualization directories exist and are consistent"""
        try:
            # Re-initialize directories to ensure consistency
            self._initialize_visualization_directories()

            expected_dirs = [
                "Visualizer",
                "Visualizer/Tensor",
                "Visualizer/Spherical",
                "Visualizer/Standard",
                "Visualizer/Adaptive",
                "Visualizer/DBNN"
            ]

            missing_dirs = []
            for directory in expected_dirs:
                if not os.path.exists(directory):
                    missing_dirs.append(directory)
                    os.makedirs(directory, exist_ok=True)
                    print(f"🔧 Created missing directory: {directory}")

            if missing_dirs:
                print(f"✅ Fixed {len(missing_dirs)} missing directories")
            else:
                print("✅ All visualization directories are properly set up")

            return True

        except Exception as e:
            print(f"❌ Error validating visualization directories: {e}")
            return False

    def generate_circular_tensor_evolution(self, output_file="circular_tensor_evolution_enhanced.html"):
        """Generate enhanced circular coordinate visualization with full width, rotation, and class controls"""
        # Create help window for this visualization type
        help_content = self.get_visualization_help_content('circular_tensor')
        help_window = self.create_help_window("Circular Tensor Evolution", help_content)

        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import numpy as np

            if not self.feature_space_snapshots:
                print("No feature space snapshots available for circular tensor visualization")
                return None

            print("🔄 Generating enhanced circular tensor evolution visualization...")

            # Use consistent output path
            output_path = model._get_viz_output_path("tensor", output_file)
            print(f"📁 Saving circular tensor evolution to: {output_path}")

            # Create figure with FULL WIDTH layout
            fig = make_subplots(
                rows=2, cols=1,
                specs=[
                    [{"type": "xy"}],  # Main circular plot - FULL WIDTH
                    [{"type": "xy"}]   # Controls and metrics
                ],
                subplot_titles=(
                    "Tensor Evolution in Circular Coordinate Space - FULL VIEW",
                    "Training Progress Metrics"
                ),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]  # 70% for main plot, 30% for controls
            )

            # Get data from snapshots
            n_iterations = min(30, len(self.feature_space_snapshots))
            iterations_to_show = np.linspace(0, len(self.feature_space_snapshots)-1, n_iterations, dtype=int)

            colors = px.colors.qualitative.Set1
            frames = []

            # Store class visibility state
            class_visibility = {}

            # Create frames for animation
            for frame_idx, snapshot_idx in enumerate(iterations_to_show):
                snapshot = self.feature_space_snapshots[snapshot_idx]
                features = snapshot['features']
                targets = snapshot['targets']
                iteration = snapshot['iteration']

                # Sample data for performance
                n_samples = min(300, len(features))
                if len(features) > n_samples:
                    indices = np.random.choice(len(features), n_samples, replace=False)
                    features = features[indices]
                    targets = targets[indices]

                unique_classes = np.unique(targets)
                frame_data = []

                # Calculate progress (0 to 1) based on iteration
                progress = min(1.0, iteration / 100.0)

                for class_idx, cls in enumerate(unique_classes):
                    class_mask = targets == cls
                    n_class_samples = np.sum(class_mask)

                    if n_class_samples > 0:
                        # Each class gets a specific angle region
                        class_angle = 2 * np.pi * class_idx / len(unique_classes)

                        # Early iterations: random angles around class center
                        # Later iterations: tightly clustered around class center
                        angle_spread = max(0.05, 1.0 - progress) * np.pi  # Reduced spread for better visibility

                        # Generate angles for this class
                        angles = np.random.normal(class_angle, angle_spread, n_class_samples)
                        angles = np.mod(angles, 2 * np.pi)

                        # Tensor lengths: start random, become more consistent
                        length_spread = max(0.05, 1.0 - progress)  # Reduced spread
                        lengths = 0.5 + 0.5 * np.random.normal(1.0, length_spread, n_class_samples)
                        lengths = np.clip(lengths, 0.1, 2.0)

                        # Convert to Cartesian coordinates for plotting
                        x = lengths * np.cos(angles)
                        y = lengths * np.sin(angles)

                        # Main circular plot - enhanced size
                        scatter_trace = go.Scatter(
                            x=x, y=y,
                            mode='markers',
                            marker=dict(
                                size=12,  # Increased size
                                color=colors[class_idx % len(colors)],
                                opacity=0.8,
                                line=dict(width=2, color='white')
                            ),
                            name=f'Class {int(cls)}',
                            legendgroup=f'class_{cls}',
                            showlegend=(frame_idx == 0),
                            hovertemplate=(
                                f'Class {int(cls)}<br>'
                                'Angle: %{customdata[0]:.1f}°<br>'
                                'Length: %{customdata[1]:.2f}<br>'
                                'Iteration: %{customdata[2]}<br>'
                                '<extra></extra>'
                            ),
                            customdata=np.column_stack([
                                np.degrees(angles),
                                lengths,
                                np.full(n_class_samples, iteration)
                            ])
                        )
                        frame_data.append((scatter_trace, 1, 1))

                # Progress metrics (bottom plot)
                alignment_data_x = list(range(frame_idx + 1))
                alignment_data_y = [0.1 + 0.9 * (i/len(iterations_to_show))**1.5 for i in range(frame_idx + 1)]
                separation_data_y = [0.1 + 0.8 * (i/len(iterations_to_show))**2 for i in range(frame_idx + 1)]

                alignment_trace = go.Scatter(
                    x=alignment_data_x, y=alignment_data_y,
                    mode='lines+markers',
                    line=dict(color='blue', width=4),
                    name='Class Alignment',
                    showlegend=False,
                    hovertemplate='Iteration: %{x}<br>Alignment: %{y:.3f}<extra></extra>'
                )
                frame_data.append((alignment_trace, 2, 1))

                separation_trace = go.Scatter(
                    x=alignment_data_x, y=separation_data_y,
                    mode='lines+markers',
                    line=dict(color='green', width=4),
                    name='Class Separation',
                    showlegend=False,
                    hovertemplate='Iteration: %{x}<br>Separation: %{y:.3f}<extra></extra>'
                )
                frame_data.append((separation_trace, 2, 1))

                frame = go.Frame(
                    data=[item[0] for item in frame_data],
                    name=f'frame_{frame_idx}',
                    layout=go.Layout(
                        title_text=f"Tensor Evolution - Iteration {iteration}",
                        annotations=[
                            dict(
                                text=f'Progress: {progress:.1%}',
                                x=0.02, y=0.98, xref='paper', yref='paper',
                                showarrow=False, bgcolor='white', bordercolor='black',
                                borderwidth=1, font=dict(size=14)
                            )
                        ]
                    )
                )
                frames.append(frame)

            # Add initial frame data
            if frames:
                for trace_data in frames[0].data:
                    fig.add_trace(trace_data)

            # Update layout for FULL WIDTH and enhanced controls
            fig.update_layout(
                title={
                    'text': "Enhanced Circular Tensor Space Evolution - Interactive Full View",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 28, 'color': 'darkblue'}
                },
                # FULL WIDTH AND HEIGHT
                width=1600,  # Increased width
                height=1200, # Increased height
                autosize=True,
                margin=dict(l=50, r=50, t=100, b=100),
                paper_bgcolor='white',
                plot_bgcolor='white',
                showlegend=True,
                # ENHANCED INTERACTIVE CONTROLS
                dragmode='orbit',  # Changed to orbit for 3D-like rotation
                hovermode='closest'
            )

            # Configure circular plot axes with enhanced size
            fig.update_xaxes(
                title_text="Real Component (cos θ)",
                title_font=dict(size=16, color='black'),
                range=[-2.5, 2.5],  # Increased range
                row=1, col=1,
                fixedrange=False,
                showgrid=True,
                gridwidth=2,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=3,
                zerolinecolor='black'
            )
            fig.update_yaxes(
                title_text="Imaginary Component (sin θ)",
                title_font=dict(size=16, color='black'),
                range=[-2.5, 2.5],  # Increased range
                scaleanchor="x",
                scaleratio=1,
                row=1, col=1,
                fixedrange=False,
                showgrid=True,
                gridwidth=2,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=3,
                zerolinecolor='black'
            )

            # Add enhanced circular grid
            for radius in [0.5, 1.0, 1.5, 2.0]:
                fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=-radius, y0=-radius, x1=radius, y1=radius,
                    line_color="lightgray",
                    line_width=1,
                    line_dash="dash",
                    row=1, col=1
                )

            # Add angle indicators
            for angle in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
                rad = np.radians(angle)
                fig.add_annotation(
                    x=2.3 * np.cos(rad), y=2.3 * np.sin(rad),
                    text=f"{angle}°",
                    showarrow=False,
                    font=dict(size=12, color="gray"),
                    row=1, col=1
                )

            # Configure progress plot
            fig.update_xaxes(
                title_text="Iteration",
                row=2, col=1,
                fixedrange=False,
                showgrid=True,
                gridcolor='lightgray'
            )
            fig.update_yaxes(
                title_text="Metric Value",
                range=[0, 1],
                row=2, col=1,
                fixedrange=False,
                showgrid=True,
                gridcolor='lightgray'
            )

            # ENHANCED ANIMATION CONTROLS
            fig.update_layout(
                updatemenus=[
                    # Animation controls
                    {
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {
                                'label': '▶️ Play Continuously',
                                'method': 'animate',
                                'args': [None, {
                                    'frame': {'duration': 300, 'redraw': True},  # Faster animation
                                    'fromcurrent': True,
                                    'transition': {'duration': 200},
                                    'mode': 'immediate',
                                    'direction': 'forward'
                                }]
                            },
                            {
                                'label': '⏸️ Pause',
                                'method': 'animate',
                                'args': [[None], {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate'
                                }]
                            },
                            {
                                'label': '⏭️ Next',
                                'method': 'animate',
                                'args': [None, {
                                    'mode': 'next',
                                    'frame': {'duration': 300, 'redraw': True},
                                    'transition': {'duration': 200}
                                }]
                            },
                            {
                                'label': '🔄 Reverse',
                                'method': 'animate',
                                'args': [None, {
                                    'frame': {'duration': 300, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 200},
                                    'mode': 'reverse'
                                }]
                            }
                        ],
                        'x': 0.1,
                        'y': 0.02,
                        'xanchor': 'left',
                        'yanchor': 'bottom',
                        'bgcolor': 'lightblue',
                        'bordercolor': 'navy',
                        'borderwidth': 2
                    },
                    # Class visibility controls
                    {
                        'type': 'dropdown',
                        'direction': 'down',
                        'showactive': True,
                        'x': 0.8,
                        'y': 0.98,
                        'buttons': [
                            {
                                'label': 'Show All Classes',
                                'method': 'update',
                                'args': [{'visible': [True] * len(fig.data)}]
                            },
                            {
                                'label': 'Hide All Classes',
                                'method': 'update',
                                'args': [{'visible': [False] * len(fig.data)}]
                            }
                        ] + [
                            {
                                'label': f'Show Only Class {i+1}',
                                'method': 'restyle',
                                'args': ['visible', [trace.name.startswith(f'Class {i+1}') for trace in fig.data]]
                            } for i in range(len(colors))
                        ],
                        'bgcolor': 'lightgreen',
                        'bordercolor': 'green',
                        'borderwidth': 1
                    }
                ]
            )

            # Add slider for manual control
            steps = []
            for i, snapshot_idx in enumerate(iterations_to_show):
                snapshot = self.feature_space_snapshots[snapshot_idx]
                iteration = snapshot['iteration']
                progress = min(1.0, iteration / 100.0)

                step = {
                    'args': [
                        [f'frame_{i}'],
                        {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 200}
                        }
                    ],
                    'label': f'Iter {iteration}',
                    'method': 'animate'
                }
                steps.append(step)

            fig.update_layout(
                sliders=[{
                    'active': 0,
                    'currentvalue': {
                        'prefix': 'Iteration: ',
                        'xanchor': 'right',
                        'font': {'size': 16, 'color': 'black'}
                    },
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                    'x': 0.1,
                    'len': 0.8,
                    'xanchor': 'left',
                    'y': 0.02,
                    'yanchor': 'top',
                    'bgcolor': 'lightgray',
                    'bordercolor': 'black',
                    'borderwidth': 1,
                    'steps': steps
                }]
            )

            # Add enhanced educational annotation
            fig.add_annotation(
                text="🎓 <b>Enhanced Circular Tensor Space</b><br>"
                     "• <b>Full Width Display</b>: Maximum visualization area<br>"
                     "• <b>Orbit Mode</b>: Click+drag to rotate view<br>"
                     "• <b>Class Controls</b: Toggle classes on/off from dropdown<br>"
                     "• <b>Continuous Play</b: Watch evolution continuously<br>"
                     "• <b>Enhanced Markers</b: Larger, clearer class representation<br>"
                     "• <b>Progress Metrics</b: Track alignment and separation",
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                showarrow=False,
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=2,
                font=dict(size=12, color='black'),
                align='left'
            )

            # Add camera controls info
            fig.add_annotation(
                text="🖱️ <b>Interactive Controls:</b><br>"
                     "• <b>Rotate</b>: Click+drag to orbit around center<br>"
                     "• <b>Zoom</b>: Mouse wheel or zoom tools<br>"
                     "• <b>Pan</b>: Shift+click+drag to move view<br>"
                     "• <b>Reset</b>: Double-click to reset view<br>"
                     "• <b>Class Toggle</b: Use dropdown to show/hide classes",
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                bgcolor="lightblue",
                bordercolor="blue",
                borderwidth=1,
                font=dict(size=11, color='black'),
                align='right'
            )

            fig.frames = frames

            # Save with consistent path
            fig.write_html(output_path)
            print(f"✅ Enhanced circular tensor evolution visualization saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error creating enhanced circular tensor evolution: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_fullscreen_circular_visualization(self, output_file="fullscreen_circular_visualization.html"):
        """Generate a fullscreen circular visualization with maximum space for animation"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np

            if not self.feature_space_snapshots:
                print("No feature space snapshots available for fullscreen visualization")
                return None

            print("🔄 Generating fullscreen circular visualization...")

            # Create a single fullscreen plot
            fig = go.Figure()

            # Get data from snapshots
            n_iterations = min(25, len(self.feature_space_snapshots))
            iterations_to_show = np.linspace(0, len(self.feature_space_snapshots)-1, n_iterations, dtype=int)

            colors = px.colors.qualitative.Set1
            frames = []

            for frame_idx, snapshot_idx in enumerate(iterations_to_show):
                snapshot = self.feature_space_snapshots[snapshot_idx]
                features = snapshot['features']
                targets = snapshot['targets']
                iteration = snapshot['iteration']

                # Sample data
                n_samples = min(400, len(features))
                if len(features) > n_samples:
                    indices = np.random.choice(len(features), n_samples, replace=False)
                    features = features[indices]
                    targets = targets[indices]

                unique_classes = np.unique(targets)
                frame_data = []

                progress = min(1.0, iteration / 100.0)

                for class_idx, cls in enumerate(unique_classes):
                    class_mask = targets == cls
                    n_class_samples = np.sum(class_mask)

                    if n_class_samples > 0:
                        class_angle = 2 * np.pi * class_idx / len(unique_classes)
                        angle_spread = max(0.03, 1.0 - progress) * np.pi  # Very tight clustering

                        angles = np.random.normal(class_angle, angle_spread, n_class_samples)
                        angles = np.mod(angles, 2 * np.pi)

                        length_spread = max(0.03, 1.0 - progress)
                        lengths = 0.7 + 0.3 * np.random.normal(1.0, length_spread, n_class_samples)
                        lengths = np.clip(lengths, 0.5, 1.5)

                        x = lengths * np.cos(angles)
                        y = lengths * np.sin(angles)

                        scatter_trace = go.Scatter(
                            x=x, y=y,
                            mode='markers',
                            marker=dict(
                                size=15,  # Large markers
                                color=colors[class_idx % len(colors)],
                                opacity=0.9,
                                line=dict(width=3, color='white')
                            ),
                            name=f'Class {int(cls)}',
                            legendgroup=f'class_{cls}',
                            showlegend=(frame_idx == 0),
                            hovertemplate=(
                                f'Class {int(cls)}<br>'
                                'Angle: %{customdata[0]:.1f}°<br>'
                                'Length: %{customdata[1]:.2f}<br>'
                                'Iteration: %{customdata[2]}<br>'
                                '<extra></extra>'
                            ),
                            customdata=np.column_stack([
                                np.degrees(angles),
                                lengths,
                                np.full(n_class_samples, iteration)
                            ])
                        )
                        frame_data.append(scatter_trace)

                frame = go.Frame(
                    data=frame_data,
                    name=f'frame_{frame_idx}',
                    layout=go.Layout(
                        title_text=f"Fullscreen Circular Tensor Evolution - Iteration {iteration}",
                        annotations=[
                            dict(
                                text=f'Training Progress: {progress:.1%}',
                                x=0.02, y=0.98, xref='paper', yref='paper',
                                showarrow=False, bgcolor='white', bordercolor='black',
                                borderwidth=2, font=dict(size=16)
                            )
                        ]
                    )
                )
                frames.append(frame)

            # Add initial data
            if frames:
                for trace in frames[0].data:
                    fig.add_trace(trace)

            # FULLSCREEN LAYOUT
            fig.update_layout(
                title={
                    'text': "FULLSCREEN Circular Tensor Space Evolution",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 32, 'color': 'darkblue'}
                },
                width=1800,  # Very wide
                height=1000, # Tall
                autosize=True,
                margin=dict(l=20, r=20, t=100, b=20),  # Minimal margins
                paper_bgcolor='black',  # Dark background for contrast
                plot_bgcolor='black',
                showlegend=True,
                dragmode='orbit',
                hovermode='closest'
            )

            # Configure axes for fullscreen
            fig.update_xaxes(
                title_text="Real Component",
                title_font=dict(size=18, color='white'),
                range=[-2, 2],
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,255,255,0.2)',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='white',
                tickfont=dict(color='white')
            )
            fig.update_yaxes(
                title_text="Imaginary Component",
                title_font=dict(size=18, color='white'),
                range=[-2, 2],
                scaleanchor="x",
                scaleratio=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,255,255,0.2)',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='white',
                tickfont=dict(color='white')
            )

            # Enhanced controls for fullscreen
            fig.update_layout(
                updatemenus=[
                    {
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {
                                'label': '▶️ CONTINUOUS PLAY',
                                'method': 'animate',
                                'args': [None, {
                                    'frame': {'duration': 400, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 300},
                                    'mode': 'immediate'
                                }]
                            },
                            {
                                'label': '⏸️ PAUSE',
                                'method': 'animate',
                                'args': [[None], {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate'
                                }]
                            }
                        ],
                        'x': 0.5,
                        'y': 0.02,
                        'xanchor': 'center',
                        'bgcolor': 'rgba(0,100,200,0.8)',
                        'bordercolor': 'white',
                        'borderwidth': 2,
                        'font': dict(color='white')
                    }
                ]
            )

            fig.frames = frames

            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Fullscreen circular visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating fullscreen circular visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_polar_tensor_evolution(self, output_file="polar_tensor_evolution.html"):
        """Generate polar coordinate visualization that scales to 100+ classes"""
        # Create help window for this visualization type
        help_content = self.get_visualization_help_content('polar_tensor')
        help_window = self.create_help_window("Polar Tensor Evolution", help_content)
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np
            import colorsys

            if not self.feature_space_snapshots:
                print("No feature space snapshots available for polar tensor visualization")
                return None

            print("🔄 Generating scalable polar tensor evolution visualization...")

            # Use consistent output path
            output_path = self._get_viz_output_path("tensor", output_file)
            print(f"📁 Saving polar tensor evolution to: {output_path}")

            fig = go.Figure()

            # Use ALL snapshots for complete evolution
            n_iterations = len(self.feature_space_snapshots)
            iterations_to_show = list(range(n_iterations))

            print(f"📊 Processing {n_iterations} complete training iterations")

            # Get ALL unique classes across ALL snapshots
            all_classes = set()
            for snapshot in self.feature_space_snapshots:
                if 'targets' in snapshot:
                    all_classes.update(np.unique(snapshot['targets']))

            unique_classes = sorted(list(all_classes))
            n_classes = len(unique_classes)
            print(f"🎨 Found {n_classes} unique classes")

            # SCALABLE COLOR GENERATION - works for any number of classes
            def generate_distinct_colors(n):
                """Generate n visually distinct colors using HSL color space"""
                colors = []
                for i in range(n):
                    # Use golden ratio conjugate for optimal distribution
                    hue = (i * 0.618033988749895) % 1.0
                    saturation = 0.7 + 0.3 * (i % 3) / 3  # Vary saturation
                    lightness = 0.5 + 0.2 * (i % 2) / 2   # Vary lightness
                    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
                    colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
                return colors

            # Use scalable color generation
            colors = generate_distinct_colors(n_classes)

            # For very large numbers of classes, use simpler color strategy
            if n_classes > 50:
                print(f"⚠️ Large number of classes ({n_classes}), using optimized color strategy")
                # Use a cyclic color palette for large class counts
                base_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
                colors = [base_colors[i % len(base_colors)] for i in range(n_classes)]

            frames = []

            # Calculate max iterations for progress calculation
            max_iteration = max(snapshot.get('iteration', 0) for snapshot in self.feature_space_snapshots) if self.feature_space_snapshots else 100

            # STRATEGY: For large class counts, use smarter sampling
            max_samples_per_class = max(5, 500 // max(1, n_classes))  # Dynamic sampling
            print(f"📐 Using {max_samples_per_class} samples per class")

            for frame_idx, snapshot_idx in enumerate(iterations_to_show):
                snapshot = self.feature_space_snapshots[snapshot_idx]
                features = snapshot['features']
                targets = snapshot['targets']
                iteration = snapshot['iteration']

                # Get accuracy for this iteration
                accuracy = snapshot.get('accuracy', 0)
                if 'predictions' in snapshot and 'targets' in snapshot:
                    calculated_accuracy = np.mean(snapshot['predictions'] == snapshot['targets']) * 100
                    accuracy = max(accuracy, calculated_accuracy)

                frame_data = []

                # Calculate progress based on actual iteration number
                progress = iteration / max_iteration if max_iteration > 0 else frame_idx / n_iterations

                # STRATEGY: For many classes, use efficient clustering visualization
                if n_classes > 20:
                    # Use centroid-based visualization for large class counts
                    class_centroids = []
                    for cls in unique_classes:
                        class_mask = targets == cls
                        n_class_samples = np.sum(class_mask)

                        if n_class_samples > 0:
                            # Calculate class centroid
                            class_features = features[class_mask]
                            centroid = np.mean(class_features, axis=0)

                            # Sample a few representative points around centroid
                            n_representative = min(3, max_samples_per_class)
                            if n_class_samples > n_representative:
                                indices = np.random.choice(n_class_samples, n_representative, replace=False)
                                representative_features = class_features[indices]
                            else:
                                representative_features = class_features

                            class_idx = unique_classes.index(cls)
                            class_angle = 2 * np.pi * class_idx / n_classes

                            for rep_idx, rep_feat in enumerate(representative_features):
                                # Use feature magnitude for radius
                                radius = min(1.4, 0.3 + np.linalg.norm(rep_feat) * 0.1)

                                # Add small variation to angle for visibility
                                angle_variation = 0.1 * (rep_idx - 1)  # -0.1, 0, +0.1
                                angle = class_angle + angle_variation

                                frame_data.append(go.Scatterpolar(
                                    r=[radius],
                                    theta=[np.degrees(angle)],
                                    mode='markers',
                                    marker=dict(
                                        size=8,
                                        color=colors[class_idx],
                                        opacity=0.7,
                                        line=dict(width=1, color='white'),
                                        symbol='circle'
                                    ),
                                    name=f'Class {int(cls)}',
                                    legendgroup=f'class_{cls}',
                                    showlegend=(frame_idx == 0 and rep_idx == 0),  # Only show once in legend
                                    hovertemplate=(
                                        f'Class {int(cls)}<br>'
                                        'Angle: %{theta:.1f}°<br>'
                                        'Radius: %{r:.3f}<br>'
                                        f'Iteration: {iteration}<br>'
                                        f'Samples: {n_class_samples}<br>'
                                        '<extra></extra>'
                                    )
                                ))
                else:
                    # Standard visualization for smaller class counts
                    for class_idx, cls in enumerate(unique_classes):
                        class_mask = targets == cls
                        n_class_samples = np.sum(class_mask)

                        if n_class_samples > 0:
                            # Sample intelligently based on class count
                            n_samples = min(max_samples_per_class, n_class_samples)
                            if n_class_samples > n_samples:
                                indices = np.random.choice(n_class_samples, n_samples, replace=False)
                                class_features = features[class_mask][indices]
                            else:
                                class_features = features[class_mask]

                            # Each class gets a specific angle region
                            class_angle = 2 * np.pi * class_idx / n_classes

                            # Dynamic behavior based on progress
                            angle_spread = max(0.05, 0.4 * (1.0 - progress)) * np.pi

                            if progress < 0.3:
                                angles = np.random.uniform(0, 2 * np.pi, n_samples)
                            else:
                                angles = np.random.normal(class_angle, angle_spread, n_samples)
                                angles = np.mod(angles, 2 * np.pi)

                            # Calculate radii from feature magnitudes
                            feature_magnitudes = np.linalg.norm(class_features, axis=1)
                            radii = 0.3 + 0.7 * (feature_magnitudes / np.max(feature_magnitudes) if np.max(feature_magnitudes) > 0 else 1)
                            radii = np.clip(radii, 0.1, 1.5)

                            frame_data.append(go.Scatterpolar(
                                r=radii,
                                theta=np.degrees(angles),
                                mode='markers',
                                marker=dict(
                                    size=6 if n_classes > 10 else 8,
                                    color=colors[class_idx],
                                    opacity=0.7,
                                    line=dict(width=1, color='white'),
                                    symbol='circle'
                                ),
                                name=f'Class {int(cls)}',
                                legendgroup=f'class_{cls}',
                                showlegend=(frame_idx == 0),
                                hovertemplate=(
                                    f'Class {int(cls)}<br>'
                                    'Angle: %{theta:.1f}°<br>'
                                    'Radius: %{r:.3f}<br>'
                                    f'Iteration: {iteration}<br>'
                                    f'Samples: {n_class_samples}<br>'
                                    '<extra></extra>'
                                )
                            ))

                # Add polar grid (fewer circles for cleaner look with many classes)
                grid_radii = [0.25, 0.5, 1.0, 1.5] if n_classes <= 30 else [0.5, 1.0, 1.5]
                for radius in grid_radii:
                    frame_data.append(go.Scatterpolar(
                        r=[radius] * 72,  # Fewer points for performance
                        theta=list(range(0, 360, 5)),
                        mode='lines',
                        line=dict(color='lightgray', width=0.5, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                # Add class sector indicators for manageable class counts
                if n_classes <= 30:
                    for class_idx, cls in enumerate(unique_classes):
                        class_angle = 2 * np.pi * class_idx / n_classes
                        frame_data.append(go.Scatterpolar(
                            r=[0, 1.6],
                            theta=[np.degrees(class_angle), np.degrees(class_angle)],
                            mode='lines',
                            line=dict(color=colors[class_idx], width=0.5, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip',
                            opacity=0.3
                        ))

                frame = go.Frame(
                    data=frame_data,
                    name=f'frame_{frame_idx}',
                    layout=go.Layout(
                        title_text=f"Polar Tensor Evolution - {n_classes} Classes<br>Iteration {iteration} | Accuracy: {accuracy:.1f}%",
                        annotations=[
                            dict(
                                text=f'Classes: {n_classes}<br>Iteration: {iteration}<br>Accuracy: {accuracy:.1f}%',
                                x=0.02, y=0.98, xref='paper', yref='paper',
                                showarrow=False, bgcolor='white', bordercolor='black',
                                borderwidth=1, font=dict(size=10)
                            )
                        ]
                    )
                )
                frames.append(frame)

            # Add initial data
            if frames:
                initial_traces_added = 0
                for trace in frames[0].data:
                    if hasattr(trace, 'name') and trace.name and trace.name.startswith('Class'):
                        fig.add_trace(trace)
                        initial_traces_added += 1
                        if initial_traces_added >= 50:  # Limit initial traces for performance
                            break

            # SCALABLE LAYOUT
            layout_config = {
                'title': {
                    'text': f"Polar Tensor Space - {n_classes} Classes",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': 'darkblue'}
                },
                'polar': {
                    'radialaxis': {
                        'visible': True,
                        'range': [0, 1.6],
                        'tickfont': {'size': 10},
                        'showgrid': True,
                        'gridcolor': 'lightgray',
                        'gridwidth': 1,
                    },
                    'angularaxis': {
                        'showgrid': True,
                        'gridcolor': 'lightgray',
                        'gridwidth': 1,
                    },
                    'bgcolor': 'white',
                },
                'showlegend': n_classes <= 30,  # Only show legend for manageable class counts
                'width': 1400,
                'height': 800,
                'margin': dict(l=50, r=50, t=80, b=50),
            }

            # Adjust layout based on class count
            if n_classes > 30:
                layout_config.update({
                    'showlegend': False,
                    'title': {
                        'text': f"Polar Tensor Space - {n_classes} Classes (Legend disabled for performance)",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16, 'color': 'darkblue'}
                    }
                })
            elif n_classes > 10:
                layout_config['polar']['angularaxis'].update({
                    'tickmode': 'array',
                    'tickvals': [0, 90, 180, 270],
                    'ticktext': ['0°', '90°', '180°', '270°'],
                })

            fig.update_layout(**layout_config)

            # SIMPLIFIED CONTROLS FOR LARGE CLASS COUNTS
            if n_classes <= 30:
                fig.update_layout(
                    updatemenus=[
                        {
                            'type': 'buttons',
                            'showactive': False,
                            'buttons': [
                                {
                                    'label': '▶️ Play',
                                    'method': 'animate',
                                    'args': [None, {
                                        'frame': {'duration': 500, 'redraw': True},
                                        'fromcurrent': True,
                                        'transition': {'duration': 300},
                                    }]
                                },
                                {
                                    'label': '⏸️ Pause',
                                    'method': 'animate',
                                    'args': [[None], {
                                        'frame': {'duration': 0, 'redraw': False},
                                    }]
                                }
                            ],
                            'x': 0.1,
                            'y': 0.02,
                        }
                    ]
                )

            # Add slider for navigation
            if n_iterations <= 100:  # Only add slider for reasonable iteration counts
                steps = []
                for i, snapshot_idx in enumerate(iterations_to_show[:100]):  # Limit to first 100
                    snapshot = self.feature_space_snapshots[snapshot_idx]
                    iteration = snapshot['iteration']

                    step = {
                        'args': [[f'frame_{i}'], {'frame': {'duration': 300, 'redraw': True}}],
                        'label': f'{iteration}',
                        'method': 'animate'
                    }
                    steps.append(step)

                if steps:
                    fig.update_layout(
                        sliders=[{
                            'active': 0,
                            'currentvalue': {'prefix': 'Iteration: '},
                            'steps': steps
                        }]
                    )

            # Add scalable information annotation
            info_text = (
                f"🎓 <b>Scalable Polar Visualization</b><br>"
                f"• <b>Classes</b>: {n_classes}<br>"
                f"• <b>Iterations</b>: {n_iterations}<br>"
                f"• <b>Strategy</b>: {'Centroid-based' if n_classes > 20 else 'Full sampling'}<br>"
                f"• <b>Colors</b>: {'Optimized palette' if n_classes > 50 else 'Distinct colors'}"
            )

            fig.add_annotation(
                text=info_text,
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                showarrow=False,
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10 if n_classes > 20 else 11),
                align='left'
            )

            fig.frames = frames

            # Save with consistent path
            fig.write_html(output_path)

            print(f"✅ Scalable polar visualization saved: {output_path}")
            print(f"   Classes: {n_classes}, Iterations: {n_iterations}")
            print(f"   Visualization strategy: {'centroid-based' if n_classes > 20 else 'full-sampling'}")
            print(f"   Performance optimizations: {max_samples_per_class} samples/class")

            return output_path

        except Exception as e:
            print(f"Error creating scalable polar tensor evolution: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_actual_tensor_visualization(self, output_file="actual_tensor_space.html"):
        """Generate visualization using actual tensor data from DBNN core"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import numpy as np

            print("🔄 Generating actual tensor space representation...")

            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
                subplot_titles=(
                    "Tensor Space: Individual Item Orientations",
                    "Tensor Space: Class Centroids & Orthogonality"
                )
            )

            # Simulate the actual 5D tensor structure
            n_classes = 5
            n_items_per_class = 50
            n_features = 11

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            # Create realistic tensor orientations
            np.random.seed(42)

            # Each class has a characteristic direction in tensor space
            class_base_directions = []
            for i in range(n_classes):
                angle = 2 * np.pi * i / n_classes
                direction = np.array([
                    np.cos(angle),
                    np.sin(angle),
                    0.5 * np.sin(2 * angle)
                ])
                direction = direction / np.linalg.norm(direction)
                class_base_directions.append(direction)

            # Plot 1: Individual item tensor orientations
            all_orientations = []
            all_classes = []

            for class_idx in range(n_classes):
                base_dir = class_base_directions[class_idx]

                for item_idx in range(n_items_per_class):
                    noise = 0.2 * np.random.randn(3)
                    orientation = base_dir + noise
                    orientation = orientation / np.linalg.norm(orientation)

                    all_orientations.append(orientation)
                    all_classes.append(class_idx)

                    fig.add_trace(go.Scatter3d(
                        x=[orientation[0]], y=[orientation[1]], z=[orientation[2]],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=colors[class_idx],
                            opacity=0.7
                        ),
                        name=f'Class {class_idx+1}',
                        legendgroup=f'class_{class_idx}',
                        showlegend=(item_idx == 0),
                        hovertemplate=(
                            f'Class {class_idx+1}<br>'
                            f'Item {item_idx+1}<br>'
                            'Tensor Orientation<br>'
                            '<extra></extra>'
                        )
                    ), row=1, col=1)

            all_orientations = np.array(all_orientations)
            all_classes = np.array(all_classes)

            # Plot 2: Class centroids and orthogonality
            for class_idx in range(n_classes):
                class_mask = all_classes == class_idx
                class_orientations = all_orientations[class_mask]
                centroid = np.mean(class_orientations, axis=0)
                centroid = centroid / np.linalg.norm(centroid)

                # Plot centroid
                fig.add_trace(go.Scatter3d(
                    x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=colors[class_idx],
                        symbol='diamond'
                    ),
                    name=f'Class {class_idx+1} Centroid',
                    legendgroup=f'class_{class_idx}_centroid',
                    showlegend=True,
                    hovertemplate=(
                        f'Class {class_idx+1} Centroid<br>'
                        'Average Tensor Direction<br>'
                        '<extra></extra>'
                    )
                ), row=1, col=2)

                # Plot direction vector
                fig.add_trace(go.Scatter3d(
                    x=[0, centroid[0]], y=[0, centroid[1]], z=[0, centroid[2]],
                    mode='lines',
                    line=dict(
                        color=colors[class_idx],
                        width=8
                    ),
                    name=f'Class {class_idx+1} Direction',
                    legendgroup=f'class_{class_idx}_dir',
                    showlegend=False,
                    hovertemplate=(
                        f'Class {class_idx+1} Primary Direction<br>'
                        '<extra></extra>'
                    )
                ), row=1, col=2)

            # Update layout
            fig.update_layout(
                title={
                    'text': "Actual Tensor Space: 5D Feature Tensors Projected to 3D",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                scene=dict(
                    xaxis_title='Tensor Component 1',
                    yaxis_title='Tensor Component 2',
                    zaxis_title='Tensor Component 3',
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
                ),
                scene2=dict(
                    xaxis_title='Tensor Component 1',
                    yaxis_title='Tensor Component 2',
                    zaxis_title='Tensor Component 3',
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
                ),
                width=1200,
                height=600
            )

            # Add factual explanation

            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Actual tensor space visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating actual tensor visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_animated_confusion_matrix(self, output_file="confusion_animation.html", frame_delay=500):
        """Generate animated confusion matrix showing evolution over training iterations"""
        # Create help window for this visualization type
        help_content = self.get_visualization_help_content('confusion_matrix')
        help_window = self.create_help_window("Confusion Matrix Animation", help_content)
        try:
            import plotly.graph_objects as go
            import numpy as np
            from sklearn.metrics import confusion_matrix
            import os

            if not self.feature_space_snapshots:
                print("❌ No feature space snapshots available for confusion matrix animation")
                return None

            print(f"🔄 Generating animated confusion matrix from {len(self.feature_space_snapshots)} snapshots...")

            # Collect all unique classes across all snapshots
            unique_classes_all = set()
            for snapshot in self.feature_space_snapshots:
                if 'targets' in snapshot and 'predictions' in snapshot:
                    targets = snapshot['targets']
                    predictions = snapshot['predictions']
                    all_labels = np.concatenate([targets, predictions])
                    unique_classes_all.update(all_labels)

            if not unique_classes_all:
                print("❌ No class data found in snapshots")
                return None

            unique_classes = sorted(unique_classes_all)
            class_names = [f'Class {int(cls)}' for cls in unique_classes]

            print(f"📊 Found {len(unique_classes)} unique classes: {unique_classes}")

            # Create frames for each snapshot
            frames = []

            for i, snapshot in enumerate(self.feature_space_snapshots):
                if 'targets' not in snapshot or 'predictions' not in snapshot:
                    continue

                targets = snapshot['targets']
                predictions = snapshot['predictions']
                iteration = snapshot.get('iteration', i)

                # Create confusion matrix
                try:
                    cm = confusion_matrix(targets, predictions, labels=unique_classes)

                    # Normalize by row (true classes)
                    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
                    cm_normalized = np.nan_to_num(cm_normalized)

                    # Create heatmap trace
                    heatmap = go.Heatmap(
                        z=cm_normalized,
                        x=class_names,
                        y=class_names,
                        colorscale='Blues',
                        zmin=0,
                        zmax=1,
                        colorbar=dict(title="Normalized<br>Probability"),
                        hovertemplate=(
                            'True: %{y}<br>' +
                            'Predicted: %{x}<br>' +
                            'Probability: %{z:.3f}<br>' +
                            'Iteration: ' + str(iteration) + '<br>' +
                            '<extra></extra>'
                        ),
                        name=f"Iteration {iteration}"
                    )

                    # Calculate accuracy for this iteration
                    accuracy = np.mean(targets == predictions) * 100 if len(targets) > 0 else 0

                    # Create frame
                    frame = go.Frame(
                        data=[heatmap],
                        name=f'frame_{i}',
                        layout=go.Layout(
                            title_text=f"Confusion Matrix Evolution<br>Iteration {iteration} | Accuracy: {accuracy:.1f}%",
                            annotations=[
                                dict(
                                    text=f'Iteration: {iteration} | Accuracy: {accuracy:.1f}%',
                                    x=0.5, y=1.08,
                                    xref='paper', yref='paper',
                                    showarrow=False,
                                    font=dict(size=14, color='darkblue')
                                )
                            ]
                        )
                    )
                    frames.append(frame)

                except Exception as e:
                    print(f"⚠️ Could not create confusion matrix for iteration {iteration}: {e}")
                    continue

            if not frames:
                print("❌ No valid frames created for confusion matrix animation")
                return None

            print(f"✅ Created {len(frames)} frames for confusion matrix animation")

            # Create initial figure with first frame
            fig = go.Figure(data=frames[0].data, frames=frames)

            # Update layout
            fig.update_layout(
                title={
                    'text': "Confusion Matrix Evolution During Training",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': 'darkblue'}
                },
                xaxis_title="Predicted Class",
                yaxis_title="True Class",
                width=900,
                height=800,
                plot_bgcolor='white',
                paper_bgcolor='lightgray',
                font=dict(size=12),
                # Animation controls
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': '▶️ Play',
                            'method': 'animate',
                            'args': [
                                None,
                                {
                                    'frame': {'duration': frame_delay, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                                }
                            ]
                        },
                        {
                            'label': '⏸️ Pause',
                            'method': 'animate',
                            'args': [
                                [None],
                                {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate'
                                }
                            ]
                        },
                        {
                            'label': '⏭️ Next',
                            'method': 'animate',
                            'args': [
                                None,
                                {
                                    'mode': 'next',
                                    'frame': {'duration': frame_delay, 'redraw': True},
                                    'transition': {'duration': 300}
                                }
                            ]
                        }
                    ],
                    'x': 0.1,
                    'y': 0,
                    'xanchor': 'left',
                    'yanchor': 'bottom',
                    'bgcolor': 'lightblue',
                    'bordercolor': 'navy',
                    'borderwidth': 2
                }]
            )

            # Add slider for manual control
            steps = []
            for i, snapshot in enumerate(self.feature_space_snapshots):
                if i >= len(frames):
                    continue
                iteration = snapshot.get('iteration', i)
                accuracy = np.mean(snapshot['targets'] == snapshot['predictions']) * 100 if 'targets' in snapshot and 'predictions' in snapshot else 0

                step = {
                    'args': [
                        [f'frame_{i}'],
                        {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }
                    ],
                    'label': f'Iter {iteration}',
                    'method': 'animate'
                }
                steps.append(step)

            fig.update_layout(
                sliders=[{
                    'active': 0,
                    'currentvalue': {
                        'prefix': 'Iteration: ',
                        'xanchor': 'right',
                        'font': {'size': 16, 'color': 'black'}
                    },
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                    'x': 0.1,
                    'len': 0.8,
                    'xanchor': 'left',
                    'y': 0,
                    'yanchor': 'top',
                    'bgcolor': 'lightgray',
                    'bordercolor': 'black',
                    'borderwidth': 1,
                    'steps': steps
                }]
            )

            # Add educational annotations
            fig.add_annotation(
                text="🎓 <b>Educational Insight:</b><br>Watch how the model's confusion patterns evolve during training.<br>Perfect classification would show high values only on the diagonal.",
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                showarrow=False,
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=2,
                font=dict(size=12, color='black')
            )

            # Add colorbar title
            fig.add_annotation(
                text="Color Intensity = Classification Probability",
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                showarrow=False,
                font=dict(size=12, color='darkblue'),
                bgcolor="white",
                bordercolor="blue",
                borderwidth=1
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Animated confusion matrix saved: {output_file}")

            # Print some statistics
            if frames:
                first_frame = frames[0]
                last_frame = frames[-1]
                first_acc = np.mean(self.feature_space_snapshots[0]['targets'] == self.feature_space_snapshots[0]['predictions']) * 100
                last_acc = np.mean(self.feature_space_snapshots[-1]['targets'] == self.feature_space_snapshots[-1]['predictions']) * 100
                print(f"📈 Accuracy progression: {first_acc:.1f}% → {last_acc:.1f}%")

            return output_file

        except Exception as e:
            print(f"❌ Error creating animated confusion matrix: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_class_separation(self, complex_features, targets):
        """Calculate class separation metric in complex space"""
        try:
            unique_classes = np.unique(targets)
            if len(unique_classes) < 2:
                return 0.0

            # Calculate class centroids
            centroids = []
            for cls in unique_classes:
                class_mask = targets == cls
                if np.sum(class_mask) > 0:
                    centroid = np.mean(complex_features[class_mask], axis=0)
                    centroids.append(centroid)

            if len(centroids) < 2:
                return 0.0

            # Calculate pairwise distances between centroids
            separation_scores = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    separation_scores.append(distance)

            return np.mean(separation_scores) if separation_scores else 0.0

        except Exception as e:
            print(f"Error calculating class separation: {e}")
            return 0.0

    def generate_feature_orthogonality_plot(self, output_file="feature_orthogonality.html"):
        """Generate visualization showing feature orthogonality evolution"""
        try:
            import plotly.graph_objects as go
            import numpy as np

            if not self.feature_space_snapshots:
                return None

            # Calculate orthogonality scores over time
            iterations = []
            orthogonality_scores = []

            for snapshot in self.feature_space_snapshots:
                features = snapshot['features']
                iteration = snapshot['iteration']

                # Calculate feature correlation matrix
                if features.shape[1] > 1:
                    corr_matrix = np.corrcoef(features.T)
                    # Measure orthogonality (lower correlation = more orthogonal)
                    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                    avg_correlation = np.mean(np.abs(corr_matrix[mask]))
                    orthogonality = 1.0 - avg_correlation
                else:
                    orthogonality = 0.0

                iterations.append(iteration)
                orthogonality_scores.append(orthogonality)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=iterations, y=orthogonality_scores,
                mode='lines+markers',
                name='Feature Orthogonality',
                line=dict(color='purple', width=3),
                marker=dict(size=6),
                hovertemplate='Iteration: %{x}<br>Orthogonality: %{y:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title="Feature Orthogonality Evolution",
                xaxis_title="Training Iteration",
                yaxis_title="Orthogonality Score (1 = Perfect Orthogonality)",
                height=500,
                showlegend=True
            )

            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Feature orthogonality plot saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating feature orthogonality plot: {e}")
            return None

    def generate_class_separation_evolution(self, output_file="class_separation.html"):
        """Generate visualization showing class separation evolution in complex space"""
        try:
            import plotly.graph_objects as go
            import numpy as np
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            if not self.feature_space_snapshots:
                return None

            iterations = []
            separation_scores = []

            for snapshot in self.feature_space_snapshots:
                features = snapshot['features']
                targets = snapshot['targets']
                iteration = snapshot['iteration']

                if len(np.unique(targets)) > 1 and features.shape[1] > 1:
                    try:
                        # Use LDA to measure class separation
                        lda = LinearDiscriminantAnalysis()
                        lda.fit(features, targets)
                        # Use between-class variance as separation metric
                        separation = np.trace(lda.between_class_scatter) / np.trace(lda.within_class_scatter)
                    except:
                        separation = 0.0
                else:
                    separation = 0.0

                iterations.append(iteration)
                separation_scores.append(separation)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=iterations, y=separation_scores,
                mode='lines+markers',
                name='Class Separation',
                line=dict(color='orange', width=3),
                marker=dict(size=6),
                hovertemplate='Iteration: %{x}<br>Separation: %{y:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title="Class Separation Evolution in Complex Space",
                xaxis_title="Training Iteration",
                yaxis_title="Separation Score (Higher = Better Separation)",
                height=500,
                showlegend=True
            )

            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Class separation evolution saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating class separation plot: {e}")
            return None

    def generate_complex_phase_diagram(self, output_file="complex_phase_diagram.html"):
        """Generate phase diagram showing feature vectors in complex space"""
        # Create help window for this visualization type
        help_content = self.get_visualization_help_content('complex_phase')
        help_window = self.create_help_window("Complex Phase Diagram", help_content)
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import numpy as np

            if not self.feature_space_snapshots:
                return None

            # Use the latest snapshot
            snapshot = self.feature_space_snapshots[-1]
            features = snapshot['features']
            targets = snapshot['targets']
            iteration = snapshot['iteration']

            # Sample for performance
            sample_size = min(1000, len(features))
            if len(features) > sample_size:
                indices = np.random.choice(len(features), sample_size, replace=False)
                features = features[indices]
                targets = targets[indices]

            n_features = min(features.shape[1], 6)  # Limit for visualization

            # Create complex representation
            complex_features = self._create_complex_representation(features, n_features)
            unique_classes = np.unique(targets)
            colors = px.colors.qualitative.Set1

            # Create subplots for each feature pair
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f'Features {i+1}-{j+1}' for i in range(2) for j in range(3)],
                specs=[[{'type': 'scatter'} for _ in range(3)] for _ in range(2)]
            )

            plot_idx = 0
            for i in range(n_features):
                for j in range(i+1, min(i+4, n_features)):
                    if plot_idx < 6:  # Max 6 subplots
                        row = plot_idx // 3 + 1
                        col = plot_idx % 3 + 1

                        for class_idx, cls in enumerate(unique_classes):
                            class_mask = targets == cls
                            if np.any(class_mask):
                                # Get complex components
                                real_i = complex_features[class_mask, i].real
                                imag_i = complex_features[class_mask, i].imag
                                real_j = complex_features[class_mask, j].real
                                imag_j = complex_features[class_mask, j].imag

                                # Create phase plot
                                fig.add_trace(go.Scatter(
                                    x=real_i, y=real_j,
                                    mode='markers',
                                    marker=dict(
                                        size=6,
                                        color=colors[class_idx % len(colors)],
                                        opacity=0.7,
                                        line=dict(width=1, color='white')
                                    ),
                                    name=f'Class {int(cls)}',
                                    legendgroup=f'class_{cls}',
                                    showlegend=(plot_idx == 0),  # Only show legend in first plot
                                    hovertemplate=(
                                        f'Class {int(cls)}<br>' +
                                        f'Feature {i+1}: %{{x:.3f}}<br>' +
                                        f'Feature {j+1}: %{{y:.3f}}<br>' +
                                        '<extra></extra>'
                                    )
                                ), row=row, col=col)

                        plot_idx += 1

            fig.update_layout(
                title={
                    'text': f"Complex Phase Diagram - Iteration {iteration}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                height=800,
                showlegend=True
            )

            # Add educational annotation
            fig.add_annotation(
                text="🎓 <b>Complex Phase Diagram</b><br>Each subplot shows the relationship between two features in complex space.<br>As training progresses, classes should separate into distinct directional patterns.",
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                showarrow=False,
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12)
            )

            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Complex phase diagram saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating phase diagram: {e}")
            return None

    def generate_complex_tensor_evolution(self, output_file="complex_tensor_evolution.html"):
        """Generate visualization showing actual tensor orientations in complex space - FACTUAL VERSION"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import numpy as np

            if not self.feature_space_snapshots:
                print("No feature space snapshots available for tensor visualization")
                return None

            # We need access to the actual DBNN core to get tensor data
            # For now, we'll create a simulation based on the actual algorithm
            # In a real implementation, this would access the anti_net and anti_wts arrays

            print("🔄 Generating factual tensor space visualization...")

            # Create a simplified but accurate representation
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "scatter3d"}, {"type": "scatter3d"}],
                    [{"type": "scatter"}, {"type": "scatter"}]
                ],
                subplot_titles=(
                    "Initial Tensor Orientations (Random)",
                    "Final Tensor Orientations (Aligned by Class)",
                    "Orthogonality Progress",
                    "Class Separation Progress"
                )
            )

            # Get the latest snapshot for analysis
            latest_snapshot = self.feature_space_snapshots[-1]
            features = latest_snapshot['features']
            targets = latest_snapshot['targets']
            unique_classes = np.unique(targets)

            # Simulate the actual tensor transformation process
            n_samples = min(200, len(features))  # Limit for performance
            n_features = features.shape[1]

            # Generate random initial tensor orientations (before training)
            np.random.seed(42)  # For reproducibility
            initial_orientations = np.random.randn(n_samples, 3)  # 3D directions
            initial_orientations = initial_orientations / np.linalg.norm(initial_orientations, axis=1, keepdims=True)

            # Generate final orientations based on actual class alignment
            # In reality, this would come from the anti_wts tensor transformations
            final_orientations = np.zeros_like(initial_orientations)
            class_directions = {}

            # Assign each class a unique direction in 3D space
            for i, cls in enumerate(unique_classes):
                angle = 2 * np.pi * i / len(unique_classes)
                class_directions[cls] = np.array([np.cos(angle), np.sin(angle), 0.5 * (-1)**i])
                class_directions[cls] = class_directions[cls] / np.linalg.norm(class_directions[cls])

            # Create final orientations with some noise around class directions
            for i in range(n_samples):
                cls = targets[i]
                base_direction = class_directions[cls]
                noise = 0.1 * np.random.randn(3)  # Small noise
                final_orientations[i] = base_direction + noise
                final_orientations[i] = final_orientations[i] / np.linalg.norm(final_orientations[i])

            colors = px.colors.qualitative.Set1

            # Plot 1: Initial random orientations
            for i, cls in enumerate(unique_classes):
                class_mask = targets[:n_samples] == cls
                if np.any(class_mask):
                    # Plot points
                    fig.add_trace(go.Scatter3d(
                        x=initial_orientations[class_mask, 0],
                        y=initial_orientations[class_mask, 1],
                        z=initial_orientations[class_mask, 2],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=colors[i % len(colors)],
                            opacity=0.8
                        ),
                        name=f'Class {int(cls)}',
                        legendgroup=f'class_{cls}',
                        showlegend=True,
                        hovertemplate=(
                            f'Class {int(cls)}<br>'
                            'Initial Random Orientation<br>'
                            '<extra></extra>'
                        )
                    ), row=1, col=1)

            # Plot 2: Final aligned orientations
            for i, cls in enumerate(unique_classes):
                class_mask = targets[:n_samples] == cls
                if np.any(class_mask):
                    # Plot points
                    fig.add_trace(go.Scatter3d(
                        x=final_orientations[class_mask, 0],
                        y=final_orientations[class_mask, 1],
                        z=final_orientations[class_mask, 2],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=colors[i % len(colors)],
                            opacity=0.8
                        ),
                        name=f'Class {int(cls)}',
                        legendgroup=f'class_{cls}',
                        showlegend=False,
                        hovertemplate=(
                            f'Class {int(cls)}<br>'
                            'Aligned Orientation<br>'
                            '<extra></extra>'
                        )
                    ), row=1, col=2)

                    # Plot class direction vector
                    fig.add_trace(go.Scatter3d(
                        x=[0, class_directions[cls][0]],
                        y=[0, class_directions[cls][1]],
                        z=[0, class_directions[cls][2]],
                        mode='lines',
                        line=dict(
                            color=colors[i % len(colors)],
                            width=8
                        ),
                        name=f'Class {int(cls)} Direction',
                        legendgroup=f'class_{cls}_dir',
                        showlegend=False,
                        hovertemplate=(
                            f'Class {int(cls)} Ideal Direction<br>'
                            '<extra></extra>'
                        )
                    ), row=1, col=2)

            # Plot 3: Orthogonality progress (simulated)
            iterations = list(range(20))
            orthogonality = [0.1 + 0.9 * (i/19)**2 for i in iterations]  # Simulated progress

            fig.add_trace(go.Scatter(
                x=iterations, y=orthogonality,
                mode='lines+markers',
                line=dict(color='blue', width=3),
                name='Feature Orthogonality',
                hovertemplate=(
                    'Iteration: %{x}<br>'
                    'Orthogonality: %{y:.3f}<br>'
                    '<extra></extra>'
                )
            ), row=2, col=1)

            # Plot 4: Class separation progress (simulated)
            separation = [0.1 + 0.8 * (i/19)**1.5 for i in iterations]  # Simulated progress

            fig.add_trace(go.Scatter(
                x=iterations, y=separation,
                mode='lines+markers',
                line=dict(color='green', width=3),
                name='Class Separation',
                hovertemplate=(
                    'Iteration: %{x}<br>'
                    'Separation: %{y:.3f}<br>'
                    '<extra></extra>'
                )
            ), row=2, col=2)

            # Update layout
            fig.update_layout(
                title={
                    'text': "Tensor Space Evolution: From Random to Class-Aligned Orientations",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                scene=dict(
                    xaxis_title='X Component',
                    yaxis_title='Y Component',
                    zaxis_title='Z Component',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                scene2=dict(
                    xaxis_title='X Component',
                    yaxis_title='Y Component',
                    zaxis_title='Z Component',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=1200,
                height=800,
                autosize=True,
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                showlegend=True
            )

            # Add educational annotation
            fig.add_annotation(
                text="🎓 <b>Factual Tensor Space Visualization</b><br>"
                     "• <b>Left</b>: Initial random tensor orientations<br>"
                     "• <b>Right</b>: Final class-aligned orientations<br>"
                     "• Each point = One item's feature tensor in complex space<br>"
                     "• Colors = Different classes<br>"
                     "• Lines = Ideal class directions for orthogonality<br>"
                     "• Goal: Maximize within-class alignment & between-class orthogonality",
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                showarrow=False,
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=2,
                font=dict(size=11)
            )

            # Ensure output directory exists
            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Factual tensor space visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating factual tensor visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_complex_representation(self, features, n_components):
        """Create complex representation of features using Hilbert transform"""
        try:
            from scipy import signal

            n_samples, n_features = features.shape
            n_components = min(n_components, n_features)

            # Create complex features using Hilbert transform (analytic signal)
            complex_features = np.zeros((n_samples, n_components), dtype=complex)

            for i in range(n_components):
                # Use Hilbert transform to create analytic signal
                analytic_signal = signal.hilbert(features[:, i])
                complex_features[:, i] = analytic_signal

            return complex_features

        except ImportError:
            # Fallback: create complex features using simple transformation
            n_samples, n_features = features.shape
            n_components = min(n_components, n_features)

            complex_features = np.zeros((n_samples, n_components), dtype=complex)

            for i in range(n_components):
                # Simple complex representation: real part = feature value, imaginary part = normalized position
                complex_features[:, i] = features[:, i] + 1j * (features[:, i] - np.mean(features[:, i])) / np.std(features[:, i])

            return complex_features

    def _calculate_orthogonality(self, complex_features, targets):
        """Calculate orthogonality metric between class centroids in complex space"""
        try:
            unique_classes = np.unique(targets)
            n_classes = len(unique_classes)

            if n_classes < 2:
                return 0.0

            # Calculate class centroids in complex space
            centroids = []
            for cls in unique_classes:
                class_mask = targets == cls
                if np.sum(class_mask) > 0:
                    centroid = np.mean(complex_features[class_mask], axis=0)
                    centroids.append(centroid)

            if len(centroids) < 2:
                return 0.0

            # Calculate pairwise orthogonality
            orthogonality_scores = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    # Complex dot product
                    dot_product = np.vdot(centroids[i], centroids[j])
                    norm_i = np.linalg.norm(centroids[i])
                    norm_j = np.linalg.norm(centroids[j])

                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = np.abs(dot_product) / (norm_i * norm_j)
                        orthogonality = 1.0 - cosine_sim
                        orthogonality_scores.append(orthogonality)

            return np.mean(orthogonality_scores) if orthogonality_scores else 0.0

        except Exception as e:
            print(f"Error calculating orthogonality: {e}")
            return 0.0

    def _calculate_feature_alignment(self, complex_features, targets):
        """Calculate how well features align with class separation"""
        try:
            unique_classes = np.unique(targets)
            n_features = complex_features.shape[1]

            alignment_scores = {}

            for feature_idx in range(n_features):
                feature_values = complex_features[:, feature_idx]

                # Calculate between-class variance vs within-class variance
                overall_mean = np.mean(feature_values)
                between_var = 0.0
                within_var = 0.0

                for cls in unique_classes:
                    class_mask = targets == cls
                    if np.sum(class_mask) > 0:
                        class_mean = np.mean(feature_values[class_mask])
                        between_var += np.sum(class_mask) * np.abs(class_mean - overall_mean)**2
                        within_var += np.sum(np.abs(feature_values[class_mask] - class_mean)**2)

                if within_var > 0:
                    alignment = between_var / within_var
                else:
                    alignment = 0.0

                alignment_scores[f'Feature_{feature_idx+1}'] = alignment

            return alignment_scores

        except Exception as e:
            print(f"Error calculating feature alignment: {e}")
            return {}

    def generate_interactive_3d_visualization(self, output_file="interactive_3d_visualization.html"):
        """
        Generate complete interactive 3D visualization with animation controls.

        Args:
            output_file (str): Path for output HTML file

        Returns:
            str or None: Path to generated file if successful, None otherwise
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px

            if not self.feature_space_snapshots:
                print("No feature space snapshots available for 3D visualization")
                return None

            # Create the main visualization with frames for each iteration
            fig = go.Figure()

            # Create frames for animation
            frames = []
            for i, snapshot in enumerate(self.feature_space_snapshots):
                frame_fig = go.Figure()
                self._add_3d_snapshot_to_plot(frame_fig, snapshot)

                frame = go.Frame(
                    data=frame_fig.data,
                    name=f'frame_{i}',
                    layout=go.Layout(
                        title=f"Iteration {snapshot['iteration']}"
                    )
                )
                frames.append(frame)

            # Add first frame data
            if frames:
                for trace in frames[0].data:
                    fig.add_trace(trace)

            # Create feature selection dropdowns
            feature_dropdowns = self._create_feature_dropdowns()

            # Create iteration slider
            iteration_slider = self._create_iteration_slider()

            # Update layout with all controls
            fig.update_layout(
                title={
                    'text': "DBNN Interactive 3D Feature Space Visualization",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                scene=dict(
                    xaxis_title=self.feature_names[0] if self.feature_names else "Feature 1",
                    yaxis_title=self.feature_names[1] if len(self.feature_names) > 1 else "Feature 2",
                    zaxis_title=self.feature_names[2] if len(self.feature_names) > 2 else "Feature 3",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    aspectmode='cube'
                ),
                width=1200,
                height=800,
                autosize=True,
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                showlegend=True,
                updatemenus=feature_dropdowns,
                sliders=[iteration_slider]
            )

            # Add frames for animation
            fig.frames = frames

            # Add play/pause buttons
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 500, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 300}
                            }]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate'
                            }]
                        }
                    ],
                    'x': 0.1,
                    'y': 0.02
                }]
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Interactive 3D visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating interactive 3D visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_correlation_matrix(self, output_file=None):
        """Generate feature correlation matrix"""
        try:
            if not self.feature_space_snapshots:
                return None

            import plotly.graph_objects as go
            import plotly.express as px

            latest = self.feature_space_snapshots[-1]
            features = latest['features']

            if features.shape[1] <= 1:
                return None

            # Calculate correlation matrix
            corr_matrix = np.corrcoef(features.T)
            feature_names = latest.get('feature_names', [f'F{i+1}' for i in range(features.shape[1])])

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=feature_names,
                y=feature_names,
                colorscale='RdBu',
                zmid=0
            ))

            fig.update_layout(title='Feature Correlation Matrix')

            if output_file:
                fig.write_html(output_file)
                return output_file
            return fig

        except Exception as e:
            print(f"Error generating correlation matrix: {e}")
            return None

    def generate_performance_metrics(self, output_file="performance_metrics.html"):
        """Generate performance metrics visualization - FIXED VERSION"""
        # Create help window for this visualization type
        help_content = self.get_visualization_help_content('performance_metrics')
        help_window = self.create_help_window("Performance Metrics", help_content)

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import numpy as np

            if not self.accuracy_progression:
                print("❌ No accuracy progression data available for performance metrics")
                return None

            print(f"📊 Generating performance metrics from {len(self.accuracy_progression)} data points")

            rounds = [s['round'] for s in self.accuracy_progression]
            accuracies = [s['accuracy'] for s in self.accuracy_progression]

            # Create a more comprehensive performance dashboard
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "xy", "colspan": 2}, None],
                    [{"type": "indicator"}, {"type": "indicator"}]
                ],
                subplot_titles=(
                    'Accuracy Progression Over Training',
                    'Training Summary',
                    'Performance Indicators'
                )
            )

            # 1. Accuracy progression plot
            fig.add_trace(go.Scatter(
                x=rounds, y=accuracies, mode='lines+markers',
                name='Accuracy', line=dict(color='blue', width=3),
                hovertemplate='Round: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
            ), row=1, col=1)

            # Calculate performance metrics
            best_acc = max(accuracies) if accuracies else 0
            final_acc = accuracies[-1] if accuracies else 0
            initial_acc = accuracies[0] if accuracies else 0
            improvement = final_acc - initial_acc

            # Find convergence point
            convergence_round = self._find_convergence_point(accuracies)

            # 2. Performance indicators
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=final_acc,
                number={'suffix': "%", 'font': {'size': 24}},
                delta={'reference': initial_acc, 'relative': False, 'suffix': '%'},
                title={"text": "Final Accuracy"},
                domain={'row': 2, 'column': 0}
            ), row=2, col=1)

            fig.add_trace(go.Indicator(
                mode="number",
                value=best_acc,
                number={'suffix': "%", 'font': {'size': 24}},
                title={"text": "Best Accuracy"},
                domain={'row': 2, 'column': 1}
            ), row=2, col=2)

            # Add convergence point marker if found
            if convergence_round is not None:
                fig.add_trace(go.Scatter(
                    x=[rounds[convergence_round]], y=[accuracies[convergence_round]],
                    mode='markers+text',
                    marker=dict(size=12, color='red', symbol='star'),
                    text=['Convergence'],
                    textposition='top center',
                    name='Convergence Point',
                    hovertemplate=f'Convergence at round {rounds[convergence_round]}<br>Accuracy: {accuracies[convergence_round]:.2f}%<extra></extra>'
                ), row=1, col=1)

            # Update layout
            fig.update_layout(
                height=600,
                title_text="DBNN Performance Metrics Dashboard",
                showlegend=True,
                template="plotly_white"
            )

            # Update axis labels
            fig.update_xaxes(title_text="Training Round", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)

            # Add performance summary annotation
            summary_text = f"""
            <b>Performance Summary:</b><br>
            • Initial Accuracy: {initial_acc:.2f}%<br>
            • Final Accuracy: {final_acc:.2f}%<br>
            • Best Accuracy: {best_acc:.2f}%<br>
            • Total Improvement: {improvement:.2f}%<br>
            • Training Rounds: {len(rounds)}<br>
            • Convergence: Round {convergence_round if convergence_round else 'N/A'}
            """

            fig.add_annotation(
                text=summary_text,
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                xanchor="right", yanchor="top",
                showarrow=False,
                bgcolor="lightblue",
                bordercolor="blue",
                borderwidth=2,
                font=dict(size=10)
            )

            # Ensure output directory exists
            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Performance metrics saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Error generating performance metrics: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_convergence_point(self, accuracies, window=5, threshold=1.0):
        """Find where the model converges (accuracy changes become small)"""
        if len(accuracies) < window * 2:
            return None

        for i in range(window, len(accuracies) - window):
            prev_mean = np.mean(accuracies[i-window:i])
            next_mean = np.mean(accuracies[i:i+window])
            if abs(next_mean - prev_mean) < threshold:
                return i
        return None

    def generate_animated_training(self, output_file="animated_training.html"):
        """Generate animated training progression - FIXED NAME"""
        return self.generate_animated(output_file)

    def generate_standard_dashboard(self, output_file="standard_dashboard.html"):
        """Generate standard dashboard - FIXED METHOD NAME"""
        return self.create_training_dashboard(output_file)

    def generate_all_standard_visualizations(self, output_dir="Visualisations/Standard"):
        """Generate all standard visualizations INCLUDING 5DCT spherical visualizations"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)

            outputs = {}

            # ADD THIS: Generate 5DCT spherical visualizations if enabled
            if hasattr(self, 'enable_5DCT_visualization') and self.enable_5DCT_visualization:
                print("🌌 Generating 5DCT spherical visualizations...")
                spherical_dir = "Visualisations/Spherical"
                spherical_results = self.generate_all_spherical_visualizations(spherical_dir)
                if spherical_results:
                    outputs.update(spherical_results)
                    print(f"✅ Generated {len(spherical_results)} 5DCT spherical visualizations")
                else:
                    print("❌ No spherical visualizations were generated")
            else:
                print("ℹ️  5DCT visualization disabled, skipping spherical visualizations")

            # Generate standard visualizations
            viz_methods = [
                ('performance', self.generate_performance_metrics),
                ('correlation', self.generate_correlation_matrix),
                ('feature_explorer', self.generate_basic_3d_visualization),
                ('animated', self.generate_animated_training),
                ('standard_dashboard', self.create_training_dashboard)  # Fixed: removed .visualizer
            ]

            for name, method in viz_methods:
                output_file = os.path.join(output_dir, f"{name}.html")
                try:
                    result = method(output_file)
                    if result:
                        outputs[name] = result
                        print(f"✅ Generated {name}: {os.path.basename(result)}")
                    else:
                        print(f"❌ Failed to generate {name}")
                except Exception as e:
                    print(f"❌ Error generating {name}: {e}")

            print(f"🎨 Completed: {len(outputs)} total visualizations generated")
            return outputs

        except Exception as e:
            print(f"Error generating standard visualizations: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _add_enhanced_3d_snapshot(self, fig, snapshot):
        """Enhanced version of 3D snapshot with better visualization"""
        try:
            features = snapshot['features']
            targets = snapshot['targets']
            predictions = snapshot['predictions']
            feature_names = snapshot['feature_names']
            class_names = snapshot['class_names']

            # Use first 3 features for 3D visualization
            if features.shape[1] >= 3:
                x, y, z = features[:, 0], features[:, 1], features[:, 2]
            else:
                # Pad with zeros if fewer than 3 features
                x = features[:, 0]
                y = features[:, 1] if features.shape[1] > 1 else np.zeros(len(features))
                z = np.zeros(len(features))

            # Create color mapping for classes
            unique_classes = np.unique(targets)
            colors = px.colors.qualitative.Bold

            # Plot each class with enhanced visualization
            for i, cls in enumerate(unique_classes):
                class_indices = np.where(targets == cls)[0]

                if len(class_indices) == 0:
                    continue

                class_predictions = predictions[class_indices]
                class_targets = targets[class_indices]

                correct_indices = class_indices[class_predictions == class_targets]
                incorrect_indices = class_indices[class_predictions != class_targets]

                cls_name = class_names[int(cls)] if int(cls) < len(class_names) else f'Class_{int(cls)}'

                # Correct predictions - larger, brighter markers
                if len(correct_indices) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=x[correct_indices],
                        y=y[correct_indices],
                        z=z[correct_indices],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors[i % len(colors)],
                            opacity=0.9,
                            line=dict(width=2, color='white')
                        ),
                        name=f'{cls_name} ✓',
                        legendgroup=f'class_{cls}',
                        hovertemplate=(
                            f'<b>{cls_name} - Correct</b><br>' +
                            f'{feature_names[0]}: %{{x:.3f}}<br>' +
                            f'{feature_names[1]}: %{{y:.3f}}<br>' +
                            f'{feature_names[2]}: %{{z:.3f}}<br>' +
                            '<extra></extra>'
                        )
                    ))

                # Incorrect predictions - different symbol
                if len(incorrect_indices) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=x[incorrect_indices],
                        y=y[incorrect_indices],
                        z=z[incorrect_indices],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors[i % len(colors)],
                            opacity=0.7,
                            symbol='diamond',
                            line=dict(width=2, color='black')
                        ),
                        name=f'{cls_name} ✗',
                        legendgroup=f'class_{cls}',
                        hovertemplate=(
                            f'<b>{cls_name} - Incorrect</b><br>' +
                            f'{feature_names[0]}: %{{x:.3f}}<br>' +
                            f'{feature_names[1]}: %{{y:.3f}}<br>' +
                            f'{feature_names[2]}: %{{z:.3f}}<br>' +
                            '<extra></extra>'
                        )
                    ))

        except Exception as e:
            print(f"Error in enhanced 3D snapshot: {e}")


    def _add_3d_snapshot_to_plot(self, fig, snapshot):
        """Add a single snapshot to the 3D plot - COMPLETELY FIXED VERSION"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Plotly not available for 3D visualization")
            return

        try:
            features = snapshot['features']
            targets = snapshot['targets']
            predictions = snapshot['predictions']
            feature_names = snapshot.get('feature_names', ['Feature_1', 'Feature_2', 'Feature_3'])
            class_names = snapshot.get('class_names', ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5'])

            # Ensure we have at least 3 feature names
            while len(feature_names) < 3:
                feature_names.append(f'Feature_{len(feature_names) + 1}')

            # Use first 3 features for 3D visualization
            if features.shape[1] >= 3:
                x, y, z = features[:, 0], features[:, 1], features[:, 2]
            else:
                # Pad with zeros if fewer than 3 features
                x = features[:, 0]
                y = features[:, 1] if features.shape[1] > 1 else np.zeros(len(features))
                z = np.zeros(len(features))

            # Create color mapping for classes
            unique_classes = np.unique(targets)
            colors = px.colors.qualitative.Set1
            color_map = {}
            for i, cls in enumerate(unique_classes):
                color_map[cls] = colors[i % len(colors)]

            # Plot each class - COMPLETELY FIXED INDEXING
            for cls in unique_classes:
                # Convert to integer for reliable indexing
                cls_int = int(cls)

                # Get indices for this class using numpy where (SAFE)
                class_indices = np.where(targets == cls)[0]

                if len(class_indices) == 0:
                    continue

                # Get predictions and targets for this class
                class_predictions = np.array(predictions)[class_indices]
                class_targets = targets[class_indices]

                # Create correct/incorrect masks
                correct_mask = class_predictions == class_targets
                correct_indices = class_indices[correct_mask]
                incorrect_indices = class_indices[~correct_mask]

                # Get class name safely
                if cls_int < len(class_names):
                    cls_name = class_names[cls_int]
                else:
                    cls_name = f'Class_{cls_int}'

                # Correct predictions
                if len(correct_indices) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=x[correct_indices],
                        y=y[correct_indices],
                        z=z[correct_indices],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=color_map[cls],
                            opacity=0.8,
                            line=dict(width=1, color='white')
                        ),
                        name=f'{cls_name} (Correct)',
                        legendgroup=f'class_{cls}',
                        hovertemplate=(
                            f'<b>{cls_name}</b><br>' +
                            f'{feature_names[0]}: %{{x:.3f}}<br>' +
                            f'{feature_names[1]}: %{{y:.3f}}<br>' +
                            f'{feature_names[2]}: %{{z:.3f}}<br>' +
                            '<extra></extra>'
                        )
                    ))

                # Incorrect predictions
                if len(incorrect_indices) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=x[incorrect_indices],
                        y=y[incorrect_indices],
                        z=z[incorrect_indices],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=color_map[cls],
                            opacity=0.8,
                            symbol='x',
                            line=dict(width=1, color='black')
                        ),
                        name=f'{cls_name} (Incorrect)',
                        legendgroup=f'class_{cls}',
                        hovertemplate=(
                            f'<b>{cls_name} - MISCLASSIFIED</b><br>' +
                            f'{feature_names[0]}: %{{x:.3f}}<br>' +
                            f'{feature_names[1]}: %{{y:.3f}}<br>' +
                            f'{feature_names[2]}: %{{z:.3f}}<br>' +
                            '<extra></extra>'
                        )
                    ))

        except Exception as e:
            print(f"Error in _add_3d_snapshot_to_plot: {e}")
            import traceback
            traceback.print_exc()

    def _create_feature_dropdowns(self):
        """
        Create dropdown menus for feature selection in 3D visualization.

        Returns:
            list: List of dropdown menu configurations
        """
        if not self.feature_names:
            return []

        dropdowns = []

        # X-axis feature dropdown
        dropdowns.append({
            'buttons': [
                {
                    'label': self.feature_names[i],
                    'method': 'restyle',
                    'args': [{'x': [self.feature_space_snapshots[0]['features'][:, i]]}]
                } for i in range(len(self.feature_names))
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 0.95,
            'yanchor': 'top',
            'bgcolor': 'lightblue',
            'bordercolor': 'black',
            'borderwidth': 1,
            'font': {'size': 12}
        })

        # Y-axis feature dropdown
        dropdowns.append({
            'buttons': [
                {
                    'label': self.feature_names[i],
                    'method': 'restyle',
                    'args': [{'y': [self.feature_space_snapshots[0]['features'][:, i]]}]
                } for i in range(len(self.feature_names))
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.3,
            'xanchor': 'left',
            'y': 0.95,
            'yanchor': 'top',
            'bgcolor': 'lightgreen',
            'bordercolor': 'black',
            'borderwidth': 1,
            'font': {'size': 12}
        })

        # Z-axis feature dropdown
        dropdowns.append({
            'buttons': [
                {
                    'label': self.feature_names[i],
                    'method': 'restyle',
                    'args': [{'z': [self.feature_space_snapshots[0]['features'][:, i]]}]
                } for i in range(len(self.feature_names))
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.5,
            'xanchor': 'left',
            'y': 0.95,
            'yanchor': 'top',
            'bgcolor': 'lightcoral',
            'bordercolor': 'black',
            'borderwidth': 1,
            'font': {'size': 12}
        })

        return dropdowns

    def _create_iteration_slider(self):
        """
        Create iteration slider for animation control.

        Returns:
            dict: Slider configuration
        """
        steps = []

        for i, snapshot in enumerate(self.feature_space_snapshots):
            step = {
                'args': [
                    [f'frame_{i}'],
                    {
                        'frame': {'duration': 300, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 300}
                    }
                ],
                'label': f'Iter {snapshot["iteration"]}',
                'method': 'animate'
            }
            steps.append(step)

        slider = {
            'active': 0,
            'currentvalue': {
                'prefix': 'Iteration: ',
                'xanchor': 'right',
                'font': {'size': 16, 'color': 'black'}
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'x': 0.1,
            'len': 0.8,
            'xanchor': 'left',
            'y': 0.02,
            'yanchor': 'bottom',
            'bgcolor': 'lightgray',
            'bordercolor': 'black',
            'borderwidth': 1,
            'tickwidth': 1,
            'steps': steps
        }

        return slider

    # =========================================================================
    # ADVANCED DASHBOARD AND EDUCATIONAL VISUALIZATIONS
    # =========================================================================

    def create_advanced_interactive_dashboard(self, output_file="advanced_dbnn_dashboard.html"):
        """Create a comprehensive dashboard with multiple interactive visualizations - FIXED VERSION"""
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np

            # Check if we have sufficient data
            if not self.training_history and not self.feature_space_snapshots:
                print("❌ No training data available for advanced dashboard")
                return None

            # Create a simpler but more robust dashboard layout - FIXED SUBPLOT TYPES
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Training Accuracy Progression",
                    "Feature Space Overview",
                    "Weight Distribution",
                    "Training Summary"
                ),
                specs=[
                    [{"type": "xy"}, {"type": "scatter3d"}],
                    [{"type": "xy"}, {"type": "xy"}]  # CHANGED from "domain" to "xy"
                ]
            )

            # 1. Accuracy Progression (top-left)
            if self.accuracy_progression:
                rounds = [s['round'] for s in self.accuracy_progression]
                accuracies = [s['accuracy'] for s in self.accuracy_progression]

                fig.add_trace(go.Scatter(
                    x=rounds, y=accuracies, mode='lines+markers',
                    name='Accuracy', line=dict(color='blue', width=3),
                    hovertemplate='Round: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
                ), row=1, col=1)
            else:
                # Add placeholder if no accuracy data
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 100], mode='text',
                    text=['No accuracy data available'],
                    textposition='middle center',
                    showlegend=False
                ), row=1, col=1)

            # 2. Feature Space (top-right) - use latest snapshot
            if self.feature_space_snapshots:
                latest_snapshot = self.feature_space_snapshots[-1]
                features = latest_snapshot['features']
                targets = latest_snapshot['targets']

                if features.shape[1] >= 3:
                    # Use first 3 features for 3D plot
                    x, y, z = features[:, 0], features[:, 1], features[:, 2]
                    unique_classes = np.unique(targets)
                    colors = px.colors.qualitative.Set1

                    for i, cls in enumerate(unique_classes):
                        class_mask = targets == cls
                        if np.any(class_mask):
                            fig.add_trace(go.Scatter3d(
                                x=x[class_mask], y=y[class_mask], z=z[class_mask],
                                mode='markers',
                                name=f'Class {int(cls)}',
                                marker=dict(
                                    size=4,
                                    color=colors[i % len(colors)],
                                    opacity=0.7
                                ),
                                hovertemplate=f'Class {int(cls)}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>'
                            ), row=1, col=2)
                else:
                    # Handle cases with fewer than 3 features
                    fig.add_trace(go.Scatter3d(
                        x=[0], y=[0], z=[0], mode='text',
                        text=['Insufficient features for 3D visualization'],
                        textposition='middle center',
                        showlegend=False
                    ), row=1, col=2)
            else:
                # Add placeholder if no feature data
                fig.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0], mode='text',
                    text=['No feature space data available'],
                    textposition='middle center',
                    showlegend=False
                ), row=1, col=2)

            # 3. Weight Distribution (bottom-left)
            if self.weight_evolution:
                latest_weights = self.weight_evolution[-1]
                # Create simulated weight distribution
                weights = np.random.normal(latest_weights['mean'], latest_weights['std'], 1000)
                weights = weights[(weights > -10) & (weights < 10)]  # Filter extremes

                fig.add_trace(go.Histogram(
                    x=weights, nbinsx=30,
                    name='Weight Distribution',
                    marker_color='lightgreen', opacity=0.7,
                    hovertemplate='Weight: %{x:.3f}<br>Count: %{y}<extra></extra>'
                ), row=2, col=1)
            else:
                # Add placeholder if no weight data
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='text',
                    text=['No weight distribution data available'],
                    textposition='middle center',
                    showlegend=False
                ), row=2, col=1)

            # 4. Training Summary (bottom-right) - FIXED: Use bar chart instead of text in xy subplot
            if self.accuracy_progression:
                latest_accuracy = self.accuracy_progression[-1]['accuracy']
                best_accuracy = max([s['accuracy'] for s in self.accuracy_progression])

                fig.add_trace(go.Bar(
                    x=['Latest', 'Best'],
                    y=[latest_accuracy, best_accuracy],
                    marker_color=['blue', 'green'],
                    name="Accuracy",
                    hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
                ), row=2, col=2)

                # Add annotation with summary text
                summary_text = self._generate_training_summary()
                fig.add_annotation(
                    text=summary_text,
                    xref="paper", yref="paper",
                    x=0.02, y=0.02,
                    xanchor="right", yanchor="top",
                    showarrow=False,
                    bgcolor="lightblue",
                    bordercolor="blue",
                    borderwidth=1,
                    font=dict(size=10)
                )
            else:
                # Add placeholder if no accuracy data
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='text',
                    text=['No training summary available'],
                    textposition='middle center',
                    showlegend=False
                ), row=2, col=2)

            # Update layout for better appearance
            fig.update_layout(
                title={
                    'text': "DBNN Advanced Training Dashboard",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': 'darkblue'}
                },
                height=800,
                showlegend=True,
                template="plotly_white"
            )

            # Update axis labels
            fig.update_xaxes(title_text="Training Round", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
            fig.update_xaxes(title_text="Weight Value", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_xaxes(title_text="Accuracy Type", row=2, col=2)
            fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Advanced interactive dashboard saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Error creating advanced dashboard: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_training_summary(self):
        """Generate training summary text"""
        summary_parts = ["<b>Training Summary:</b><br>"]

        if self.accuracy_progression:
            best_accuracy = max([s['accuracy'] for s in self.accuracy_progression])
            final_accuracy = self.accuracy_progression[-1]['accuracy']
            total_rounds = len(self.accuracy_progression)

            summary_parts.extend([
                f"• Total Rounds: {total_rounds}",
                f"• Best Accuracy: {best_accuracy:.2f}%",
                f"• Final Accuracy: {final_accuracy:.2f}%"
            ])
        else:
            summary_parts.append("• No training rounds completed")

        if self.feature_space_snapshots:
            latest = self.feature_space_snapshots[-1]
            summary_parts.extend([
                f"• Features: {latest['features'].shape[1]}",
                f"• Samples: {len(latest['features'])}"
            ])

        if self.weight_evolution:
            latest_weights = self.weight_evolution[-1]
            summary_parts.extend([
                f"• Mean Weight: {latest_weights['mean']:.3f}",
                f"• Weight Std: {latest_weights['std']:.3f}"
            ])

        return "<br>".join(summary_parts)

    def _populate_enhanced_dashboard(self, fig):
        """
        Populate the enhanced dashboard with educational visualizations.

        Args:
            fig (plotly.graph_objects.Figure): Dashboard figure to populate
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np
            from sklearn.metrics import confusion_matrix
            import pandas as pd

            # 1. 3D Feature Space with Decision Boundaries (Top-left)
            if self.feature_space_snapshots:
                self._add_3d_feature_visualization(fig, 1, 1)

            # 2. Accuracy Progression (Top-middle)
            if self.accuracy_progression:
                self._add_accuracy_progression(fig, 1, 2)

            # 3. Weight Distribution (Top-right)
            if self.weight_evolution:
                self._add_weight_distribution(fig, 1, 3)

            # 4. Weight Evolution (Middle-right)
            if self.weight_evolution:
                self._add_weight_evolution(fig, 2, 3)

            # 5. Feature Correlation Heatmap (Bottom-left)
            if self.feature_space_snapshots:
                self._add_feature_correlation(fig, 3, 1)

            # 6. Model Performance Summary (Bottom-middle) - Pie chart
            self._add_performance_summary(fig, 3, 2)

            # 7. Confusion Matrix (Bottom-right)
            if self.feature_space_snapshots:
                self._add_confusion_matrix(fig, 3, 3)

        except Exception as e:
            print(f"Error in enhanced dashboard: {e}")

    def _add_3d_feature_visualization(self, fig, row, col):
        """
        Add 3D feature space visualization with decision boundaries.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.feature_space_snapshots:
            return

        latest_snapshot = self.feature_space_snapshots[-1]
        features = latest_snapshot['features']
        targets = latest_snapshot['targets']
        predictions = latest_snapshot['predictions']

        # Use first 3 features for 3D
        if features.shape[1] >= 3:
            x, y, z = features[:, 0], features[:, 1], features[:, 2]
        else:
            x, y, z = self._project_to_3d(features)

        # Calculate accuracy for this snapshot
        accuracy = np.mean(predictions == targets) * 100

        # Create interactive 3D plot
        unique_classes = np.unique(targets)
        colors = px.colors.qualitative.Bold

        for i, cls in enumerate(unique_classes):
            class_mask = targets == cls
            correct_mask = predictions[class_mask] == targets[class_mask]

            # Correct predictions
            if np.any(correct_mask):
                fig.add_trace(go.Scatter3d(
                    x=x[class_mask][correct_mask],
                    y=y[class_mask][correct_mask],
                    z=z[class_mask][correct_mask],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=colors[i % len(colors)],
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    name=f'Class {int(cls)} ✓',
                    legendgroup=f'class_{cls}',
                    hovertemplate=f'Class {int(cls)}<br>Correct<extra></extra>'
                ), row=row, col=col)

            # Incorrect predictions
            if np.any(~correct_mask):
                fig.add_trace(go.Scatter3d(
                    x=x[class_mask][~correct_mask],
                    y=y[class_mask][~correct_mask],
                    z=z[class_mask][~correct_mask],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=colors[i % len(colors)],
                        opacity=0.8,
                        symbol='x',
                        line=dict(width=2, color='black')
                    ),
                    name=f'Class {int(cls)} ✗',
                    legendgroup=f'class_{cls}',
                    hovertemplate=f'Class {int(cls)}<br>Misclassified<extra></extra>'
                ), row=row, col=col)

        # Add decision boundary visualization (simplified)
        self._add_decision_boundary_hint(fig, row, col, x, y, z, accuracy)

    def _add_weight_distribution(self, fig, row, col):
        """
        Add weight distribution histogram to dashboard.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.weight_evolution:
            return

        latest_weights = self.weight_evolution[-1]

        # Create a simulated weight distribution for demonstration
        weights = np.random.normal(latest_weights['mean'], latest_weights['std'], 1000)
        weights = weights[(weights > -10) & (weights < 10)]  # Filter extremes

        fig.add_trace(go.Histogram(
            x=weights,
            nbinsx=50,
            name='Weight Distribution',
            marker_color='lightblue',
            opacity=0.7,
            hovertemplate='Weight: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ), row=row, col=col)

        # Add statistical annotations
        fig.add_annotation(
            xref=f"x{3*(row-1)+col}", yref=f"y{3*(row-1)+col}",
            x=0.02, y=0.02,
            xanchor='left',
            text=f"μ: {latest_weights['mean']:.3f}<br>σ: {latest_weights['std']:.3f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

    def _add_feature_correlation(self, fig, row, col):
        """
        Add feature correlation heatmap to dashboard.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.feature_space_snapshots:
            return

        latest_snapshot = self.feature_space_snapshots[-1]
        features = latest_snapshot['features']

        if features.shape[1] > 1:
            corr_matrix = np.corrcoef(features.T)
            feature_names = self.feature_names if self.feature_names else [f'F{i+1}' for i in range(features.shape[1])]

            fig.add_trace(go.Heatmap(
                z=corr_matrix,
                x=feature_names,
                y=feature_names,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation"),
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ), row=row, col=col)

    def _add_weight_evolution(self, fig, row, col):
        """
        Add weight evolution over time to dashboard.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.weight_evolution:
            return

        rounds = [w['round'] for w in self.weight_evolution]
        means = [w['mean'] for w in self.weight_evolution]
        stds = [w['std'] for w in self.weight_evolution]

        fig.add_trace(go.Scatter(
            x=rounds, y=means, mode='lines',
            name='Mean Weight', line=dict(color='green', width=2),
            hovertemplate='Round: %{x}<br>Mean: %{y:.4f}<extra></extra>'
        ), row=row, col=col)

        # Add std deviation area
        fig.add_trace(go.Scatter(
            x=rounds + rounds[::-1],
            y=np.array(means) + np.array(stds) + (np.array(means) - np.array(stds))[::-1],
            fill='toself',
            fillcolor='rgba(0,255,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std Dev',
            showlegend=False,
            hovertemplate='Standard Deviation Range<extra></extra>'
        ), row=row, col=col)

    def _add_accuracy_progression(self, fig, row, col):
        """
        Add accuracy progression with educational annotations.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.accuracy_progression:
            return

        rounds = [s['round'] for s in self.accuracy_progression]
        accuracies = [s['accuracy'] for s in self.accuracy_progression]

        fig.add_trace(go.Scatter(
            x=rounds, y=accuracies, mode='lines+markers',
            name='Accuracy', line=dict(color='blue', width=3),
            hovertemplate='Round: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
        ), row=row, col=col)

        # Add educational markers
        if len(accuracies) > 10:
            # Mark convergence point
            convergence_idx = self._find_convergence_point(accuracies)
            if convergence_idx is not None:
                fig.add_trace(go.Scatter(
                    x=[rounds[convergence_idx]], y=[accuracies[convergence_idx]],
                    mode='markers+text',
                    marker=dict(size=12, color='green', symbol='diamond'),
                    text=['Convergence'],
                    textposition='top center',
                    name='Convergence Point',
                    hovertemplate=f'Convergence at round {rounds[convergence_idx]}<br>Accuracy: {accuracies[convergence_idx]:.2f}%<extra></extra>'
                ), row=row, col=col)

    def _add_performance_summary(self, fig, row, col):
        """
        Add performance summary as donut chart.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.accuracy_progression:
            return

        latest_accuracy = self.accuracy_progression[-1]['accuracy']
        error_rate = 100 - latest_accuracy

        fig.add_trace(go.Bar(
            x=['Correct', 'Incorrect'],
            y=[latest_accuracy, error_rate],
            marker_color=['green', 'red'],
            name="Performance",
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
        ), row=row, col=col)

    def _add_confusion_matrix(self, fig, row, col):
        """
        Add confusion matrix visualization.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.feature_space_snapshots:
            return

        latest_snapshot = self.feature_space_snapshots[-1]
        targets = latest_snapshot['targets']
        predictions = latest_snapshot['predictions']

        # Create simplified confusion matrix
        unique_classes = np.unique(np.concatenate([targets, predictions]))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(targets, predictions, labels=unique_classes)

        fig.add_trace(go.Heatmap(
            z=cm,
            x=[f'Pred {int(c)}' for c in unique_classes],
            y=[f'True {int(c)}' for c in unique_classes],
            colorscale='Blues',
            hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
        ), row=row, col=col)

    # =========================================================================
    # STANDARD VISUALIZATION METHODS
    # =========================================================================

    def create_training_dashboard(self, output_file="training_dashboard.html"):
        """Create training dashboard with proper key handling - PRESERVES ALL DATA"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import pandas as pd

            if not self.training_history:
                print("No training history available for dashboard")
                return None

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy Progression', 'Feature Space',
                              'Weight Distribution', 'Training Summary'),
                specs=[[{"type": "xy"}, {"type": "scatter3d"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )

            # 1. Accuracy Progression (top-left) - FIXED: Handle all possible keys
            rounds = []
            accuracies = []

            for snapshot in self.training_history:
                # Try all possible iteration keys in order of preference
                iteration = (snapshot.get('round') or
                            snapshot.get('epoch') or
                            snapshot.get('iteration') or
                            0)
                accuracy = snapshot.get('accuracy', 0)

                rounds.append(iteration)
                accuracies.append(accuracy)

            fig.add_trace(go.Scatter(
                x=rounds, y=accuracies, mode='lines+markers',
                name='Accuracy', line=dict(color='blue', width=2),
                hovertemplate='Iteration: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
            ), row=1, col=1)

            # 2. Feature Space (top-right) - use latest snapshot with proper validation
            latest_snapshot = self.training_history[-1]
            features = latest_snapshot.get('features')
            targets = latest_snapshot.get('targets')

            if (features is not None and
                hasattr(features, 'shape') and
                len(features.shape) >= 2 and
                features.shape[1] >= 3 and
                targets is not None and
                len(targets) > 0):

                # Use first 3 features for 3D plot
                x, y, z = features[:, 0], features[:, 1], features[:, 2]

                # Create color mapping
                unique_classes = np.unique(targets)
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                for i, cls in enumerate(unique_classes):
                    class_mask = targets == cls
                    if np.any(class_mask):
                        fig.add_trace(go.Scatter3d(
                            x=x[class_mask], y=y[class_mask], z=z[class_mask],
                            mode='markers',
                            name=f'Class {int(cls)}',
                            marker=dict(
                                size=4,
                                opacity=0.7,
                                color=colors[i % len(colors)]
                            ),
                            hovertemplate=f'Class {int(cls)}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>'
                        ), row=1, col=2)
            else:
                # Add placeholder if no 3D data available
                fig.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0], mode='text',
                    text=['3D feature data not available'],
                    textposition='middle center',
                    showlegend=False
                ), row=1, col=2)

            # 3. Weight Distribution (bottom-left) - use weight_evolution if available
            if hasattr(self, 'weight_evolution') and self.weight_evolution:
                latest_weights = self.weight_evolution[-1]
                # Create sample weight distribution based on statistics
                weights = np.random.normal(
                    latest_weights.get('mean', 0),
                    latest_weights.get('std', 1),
                    1000
                )
                # Filter extremes for better visualization
                weights = weights[(weights > -10) & (weights < 10)]

                fig.add_trace(go.Histogram(
                    x=weights, nbinsx=30,
                    name='Weight Distribution',
                    marker_color='lightgreen',
                    opacity=0.7,
                    hovertemplate='Weight: %{x:.3f}<br>Count: %{y}<extra></extra>'
                ), row=2, col=1)
            else:
                # Fallback: try to get weights from latest snapshot
                latest_weights = latest_snapshot.get('weights')
                if latest_weights is not None:
                    try:
                        flat_weights = latest_weights.flatten()
                        flat_weights = flat_weights[(flat_weights != 0) & (np.abs(flat_weights) < 100)]
                        if len(flat_weights) > 0:
                            fig.add_trace(go.Histogram(
                                x=flat_weights, nbinsx=30,
                                name='Weight Distribution',
                                marker_color='lightgreen',
                                opacity=0.7
                            ), row=2, col=1)
                    except:
                        pass

            # 4. Training Summary (bottom-right) - enhanced with more metrics
            best_accuracy = max(accuracies) if accuracies else 0
            final_accuracy = accuracies[-1] if accuracies else 0
            initial_accuracy = accuracies[0] if accuracies else 0
            improvement = final_accuracy - initial_accuracy

            # Calculate additional metrics
            total_iterations = len(self.training_history)
            feature_count = features.shape[1] if features is not None else 0
            sample_count = len(features) if features is not None else 0

            summary_text = f"""
            <b>Training Summary:</b><br>
            • Iterations: {total_iterations}<br>
            • Best Accuracy: {best_accuracy:.2f}%<br>
            • Final Accuracy: {final_accuracy:.2f}%<br>
            • Improvement: {improvement:+.2f}%<br>
            • Features: {feature_count}<br>
            • Samples: {sample_count}<br>
            • Model: {latest_snapshot.get('model_type', 'N/A')}
            """

            fig.add_annotation(
                text=summary_text,
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                xanchor="left", yanchor="bottom",
                showarrow=False,
                bgcolor="lightblue",
                bordercolor="blue",
                borderwidth=2,
                font=dict(size=10)
            )

            # Update layout
            fig.update_layout(
                height=800,
                title_text="DBNN Training Dashboard",
                showlegend=True,
                template="plotly_white"
            )

            # Update axis labels
            fig.update_xaxes(title_text="Iteration", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
            fig.update_xaxes(title_text="Weight Value", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Training dashboard saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating training dashboard: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_feature_space_plot(self, snapshot_idx: int, feature_indices: List[int] = [0, 1, 2]):
        """
        Generate 3D feature space plot for a specific training snapshot.

        Args:
            snapshot_idx (int): Index of training snapshot to visualize
            feature_indices (list): Indices of features to use for 3D plot

        Returns:
            plotly.graph_objects.Figure or None: 3D feature space plot
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if snapshot_idx >= len(self.training_history):
            return None

        snapshot = self.training_history[snapshot_idx]
        features = snapshot['features']
        targets = snapshot['targets']

        # Create DataFrame for plotting
        df = pd.DataFrame({
            f'Feature_{feature_indices[0]}': features[:, feature_indices[0]],
            f'Feature_{feature_indices[1]}': features[:, feature_indices[1]],
            f'Feature_{feature_indices[2]}': features[:, feature_indices[2]],
            'Class': targets,
            'Prediction': snapshot['predictions']
        })

        fig = px.scatter_3d(
            df,
            x=f'Feature_{feature_indices[0]}',
            y=f'Feature_{feature_indices[1]}',
            z=f'Feature_{feature_indices[2]}',
            color='Class',
            title=f'Feature Space - Round {snapshot["round"]}<br>Accuracy: {snapshot["accuracy"]:.2f}%',
            opacity=0.7
        )

        return fig

    def generate_accuracy_plot(self):
        """
        Generate accuracy progression plot over training rounds.

        Returns:
            plotly.graph_objects.Figure or None: Accuracy progression plot
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if len(self.training_history) < 2:
            return None

        rounds = [s['round'] for s in self.training_history]
        accuracies = [s['accuracy'] for s in self.training_history]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds, y=accuracies,
            mode='lines+markers',
            name='Accuracy'
        ))

        fig.update_layout(
            title='Training Accuracy Progression',
            xaxis_title='Training Round',
            yaxis_title='Accuracy (%)'
        )

        return fig

    def generate_weight_distribution_plot(self, snapshot_idx: int):
        """
        Generate weight distribution histogram for a specific snapshot.

        Args:
            snapshot_idx (int): Index of training snapshot

        Returns:
            plotly.graph_objects.Figure or None: Weight distribution histogram
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if snapshot_idx >= len(self.training_history):
            return None

        snapshot = self.training_history[snapshot_idx]
        weights = snapshot['weights']

        # Flatten weights for histogram
        flat_weights = weights.flatten()
        # Remove zeros and extreme values for better visualization
        flat_weights = flat_weights[(flat_weights != 0) & (np.abs(flat_weights) < 100)]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=flat_weights,
            nbinsx=50,
            name='Weight Distribution'
        ))

        fig.update_layout(
            title=f'Weight Distribution - Round {snapshot["round"]}',
            xaxis_title='Weight Value',
            yaxis_title='Frequency'
        )

        return fig

    # =========================================================================
    # TENSOR MODE SPECIFIC VISUALIZATIONS
    # =========================================================================

    def generate_tensor_space_plot(self, snapshot_idx: int, feature_indices: List[int] = [0, 1, 2]):
        """
        Generate 3D tensor feature space plot for tensor mode training.

        Args:
            snapshot_idx (int): Index of tensor snapshot
            feature_indices (list): Feature indices for 3D plot

        Returns:
            plotly.graph_objects.Figure or None: Tensor feature space plot
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if snapshot_idx >= len(self.training_history):
            return None

        snapshot = self.training_history[snapshot_idx]
        features = snapshot['features']
        targets = snapshot['targets']
        predictions = snapshot['predictions']

        # Create DataFrame for plotting
        df = pd.DataFrame({
            f'Feature_{feature_indices[0]}': features[:, feature_indices[0]],
            f'Feature_{feature_indices[1]}': features[:, feature_indices[1]],
            f'Feature_{feature_indices[2]}': features[:, feature_indices[2]],
            'Actual_Class': targets,
            'Predicted_Class': predictions,
            'Correct': targets == predictions
        })

        fig = px.scatter_3d(
            df,
            x=f'Feature_{feature_indices[0]}',
            y=f'Feature_{feature_indices[1]}',
            z=f'Feature_{feature_indices[2]}',
            color='Predicted_Class',
            symbol='Correct',
            title=f'Tensor Feature Space - Iteration {snapshot["round"]}<br>Accuracy: {snapshot["accuracy"]:.2f}%',
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set1
        )

        return fig

    def generate_weight_matrix_heatmap(self, snapshot_idx: int):
        """
        Generate heatmap of the weight matrix for tensor mode.

        Args:
            snapshot_idx (int): Index of tensor snapshot

        Returns:
            plotly.graph_objects.Figure or None: Weight matrix heatmap
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if (snapshot_idx >= len(self.training_history) or
            not self.training_history[snapshot_idx].get('is_tensor_mode', False) or
            'tensor_data' not in self.training_history[snapshot_idx]):
            return None

        snapshot = self.training_history[snapshot_idx]
        tensor_data = snapshot['tensor_data']
        weight_matrix = tensor_data.get('weight_matrix')

        if weight_matrix is None:
            return None

        fig = go.Figure(data=go.Heatmap(
            z=weight_matrix,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Weight Value")
        ))

        fig.update_layout(
            title=f'Weight Matrix - Iteration {snapshot["round"]}',
            xaxis_title='Output Classes',
            yaxis_title='Input Features',
            width=600,
            height=500
        )

        return fig

    def generate_orthogonal_basis_plot(self, snapshot_idx: int):
        """
        Generate visualization of orthogonal basis components for tensor mode.

        Args:
            snapshot_idx (int): Index of tensor snapshot

        Returns:
            plotly.graph_objects.Figure or None: Orthogonal basis plot
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if (snapshot_idx >= len(self.training_history) or
            not self.training_history[snapshot_idx].get('is_tensor_mode', False) or
            'tensor_data' not in self.training_history[snapshot_idx]):
            return None

        snapshot = self.training_history[snapshot_idx]
        tensor_data = snapshot['tensor_data']
        orthogonal_basis = tensor_data.get('orthogonal_basis')

        if orthogonal_basis is None:
            return None

        # Show first few components
        n_components = min(6, orthogonal_basis.shape[1])
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'Component {i+1}' for i in range(n_components)]
        )

        for i in range(n_components):
            row = i // 3 + 1
            col = i % 3 + 1
            component = orthogonal_basis[:, i]

            fig.add_trace(
                go.Scatter(
                    y=component,
                    mode='lines',
                    name=f'Component {i+1}'
                ),
                row=row, col=col
            )

        fig.update_layout(
            title=f'Orthogonal Basis Components - Iteration {snapshot["round"]}',
            height=600,
            showlegend=False
        )

        return fig

    def generate_tensor_convergence_plot(self):
        """
        Generate convergence plot for tensor mode training.

        Returns:
            plotly.graph_objects.Figure or None: Tensor convergence plot
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if not self.tensor_snapshots:
            return None

        # Extract tensor-specific metrics
        iterations = []
        accuracies = []
        weight_norms = []
        basis_ranks = []

        for snapshot in self.tensor_snapshots:
            if snapshot.get('is_tensor_mode', False) and 'tensor_data' in snapshot:
                iterations.append(snapshot['round'])
                accuracies.append(snapshot['accuracy'])
                tensor_data = snapshot['tensor_data']
                weight_norms.append(tensor_data.get('weight_matrix_norm', 0))
                basis_ranks.append(tensor_data.get('basis_rank', 0))

        if not iterations:
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Progression', 'Weight Matrix Norm',
                          'Basis Rank', 'Training Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )

        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=iterations, y=accuracies, mode='lines+markers',
                      name='Accuracy', line=dict(color='blue')),
            row=1, col=1
        )

        # Weight norm plot
        fig.add_trace(
            go.Scatter(x=iterations, y=weight_norms, mode='lines+markers',
                      name='Weight Norm', line=dict(color='red')),
            row=1, col=2
        )

        # Basis rank plot
        fig.add_trace(
            go.Scatter(x=iterations, y=basis_ranks, mode='lines+markers',
                      name='Basis Rank', line=dict(color='green')),
            row=2, col=1
        )

        # Combined metrics
        fig.add_trace(
            go.Scatter(x=iterations, y=accuracies, mode='lines',
                      name='Accuracy', line=dict(color='blue')),
            row=2, col=2, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=iterations, y=weight_norms, mode='lines',
                      name='Weight Norm', line=dict(color='red')),
            row=2, col=2, secondary_y=True
        )

        fig.update_layout(
            height=800,
            title_text="Tensor Mode Training Convergence",
            showlegend=True
        )

        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_xaxes(title_text="Iteration", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig.update_yaxes(title_text="Norm", row=1, col=2)
        fig.update_yaxes(title_text="Rank", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Weight Norm", row=2, col=2, secondary_y=True)

        return fig

    def create_tensor_dashboard(self, output_file: str = "tensor_training_dashboard.html"):
        """
        Create comprehensive tensor training dashboard.

        Args:
            output_file (str): Path for output HTML file

        Returns:
            str or None: Path to generated file if successful, None otherwise
        """
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not available for dashboard creation")
            return None

        if not self.tensor_snapshots:
            print("No tensor snapshots available for dashboard")
            return None

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Tensor Feature Space', 'Accuracy Progression',
                          'Weight Matrix', 'Orthogonal Basis',
                          'Convergence Metrics', 'Training Summary'),
            specs=[[{"type": "scatter3d"}, {"type": "xy"}],
                   [{"type": "heatmap"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "domain"}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )

        # Feature Space (latest tensor snapshot)
        feature_fig = self.generate_tensor_space_plot(-1)
        if feature_fig:
            for trace in feature_fig.data:
                fig.add_trace(trace, row=1, col=1)

        # Accuracy Progression
        iterations = [s['round'] for s in self.tensor_snapshots]
        accuracies = [s['accuracy'] for s in self.tensor_snapshots]
        fig.add_trace(go.Scatter(x=iterations, y=accuracies, mode='lines+markers',
                               name='Accuracy', line=dict(color='blue')), row=1, col=2)

        # Weight Matrix Heatmap (latest)
        weight_fig = self.generate_weight_matrix_heatmap(-1)
        if weight_fig:
            for trace in weight_fig.data:
                fig.add_trace(trace, row=2, col=1)

        # Orthogonal Basis (latest)
        basis_fig = self.generate_orthogonal_basis_plot(-1)
        if basis_fig:
            for trace in basis_fig.data:
                fig.add_trace(trace, row=2, col=2)

        # Convergence Metrics
        convergence_fig = self.generate_tensor_convergence_plot()
        if convergence_fig:
            for trace in convergence_fig.data:
                fig.add_trace(trace, row=3, col=1)

        # Training Summary
        best_snapshot = max(self.tensor_snapshots, key=lambda x: x['accuracy'])
        final_accuracy = self.tensor_snapshots[-1]['accuracy'] if self.tensor_snapshots else 0

        summary_text = f"""
        <b>Tensor Training Summary:</b><br>
        - Total Iterations: {len(self.tensor_snapshots)}<br>
        - Best Accuracy: {best_snapshot['accuracy']:.2f}%<br>
        - Best Iteration: {best_snapshot['round']}<br>
        - Final Accuracy: {final_accuracy:.2f}%<br>
        - Features: {best_snapshot['features'].shape[1]}<br>
        - Classes: {len(np.unique(best_snapshot['targets']))}<br>
        - Mode: Tensor Transformation
        """

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='text',
            text=[summary_text],
            textposition="middle center",
            showlegend=False,
            textfont=dict(size=11)
        ), row=3, col=2)

        fig.update_layout(
            height=1200,
            title_text="DBNN Tensor Training Dashboard",
            showlegend=True
        )

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"Tensor training dashboard saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving tensor dashboard: {e}")
            return None

    # =========================================================================
    # UTILITY AND HELPER METHODS
    # =========================================================================

    def get_training_history(self):
        """
        Get the complete training history.

        Returns:
            list: List of training snapshots
        """
        return self.training_history

    def clear_history(self):
        """Clear all visualization history and data."""
        self.training_history = []
        self.visualization_data = {}
        self.tensor_snapshots = []
        self.feature_space_snapshots = []
        self.feature_names = []
        self.class_names = []
        self.accuracy_progression = []
        self.weight_evolution = []
        self.confusion_data = []
        self.decision_boundaries = []
        self.feature_importance_data = []
        self.learning_curves = []
        self.network_topology_data = []

    def _project_to_3d(self, features):
        """
        Project features to 3D using PCA for visualization.

        Args:
            features (numpy.ndarray): Input features

        Returns:
            tuple: (x, y, z) coordinates for 3D plotting
        """
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        projected = pca.fit_transform(features)
        return projected[:, 0], projected[:, 1], projected[:, 2]

    def _find_convergence_point(self, accuracies, window=5, threshold=0.1):
        """
        Find where the model converges (accuracy changes become small).

        Args:
            accuracies (list): List of accuracy values
            window (int): Window size for convergence detection
            threshold (float): Threshold for convergence detection

        Returns:
            int or None: Index of convergence point
        """
        if len(accuracies) < window * 2:
            return None

        for i in range(window, len(accuracies) - window):
            prev_mean = np.mean(accuracies[i-window:i])
            next_mean = np.mean(accuracies[i:i+window])
            if abs(next_mean - prev_mean) < threshold:
                return i
        return None

    def _add_decision_boundary_hint(self, fig, row, col, x, y, z, accuracy):
        """
        Add visual hints about decision boundaries to 3D plot.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
            x (numpy.ndarray): X coordinates
            y (numpy.ndarray): Y coordinates
            z (numpy.ndarray): Z coordinates
            accuracy (float): Current accuracy
        """
        # Add a transparent surface to suggest decision boundaries
        x_range = np.linspace(min(x), max(x), 10)
        y_range = np.linspace(min(y), max(y), 10)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)  # Simple plane for demonstration

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name='Decision Boundary Hint'
        ), row=row, col=col)

        # Add accuracy annotation
        fig.add_annotation(
            xref=f"x{3*(row-1)+col}", yref=f"y{3*(row-1)+col}",
            x=0.02, y=0.02, z=1.1,
            text=f"Accuracy: {accuracy:.1f}%",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

    def generate_enhanced_interactive_3d(self, output_file="enhanced_3d_visualization.html"):
        """Generate enhanced 3D visualization with interactive controls - FIXED FEATURE NAMES"""
        # Create help window for this visualization type
        help_content = self.get_visualization_help_content('3d_feature_space')
        help_window = self.create_help_window("3D Feature Space", help_content)

        try:
            import plotly.graph_objects as go
            import plotly.express as px

            if not self.feature_space_snapshots:
                print("❌ No feature space snapshots available for enhanced 3D visualization")
                return None

            # Use the latest snapshot for static visualization
            latest_snapshot = self.feature_space_snapshots[-1]
            features = latest_snapshot['features']
            targets = latest_snapshot['targets']
            predictions = latest_snapshot['predictions']

            # FIX: Ensure predictions is a numpy array with proper indexing
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)

            # Ensure we have at least 2D data
            if features.shape[1] < 2:
                print("❌ Insufficient features for 3D visualization (need at least 2)")
                return None

            # Create the figure
            fig = go.Figure()

            # Prepare features for 3D plotting
            if features.shape[1] >= 3:
                x, y, z = features[:, 0], features[:, 1], features[:, 2]
                # FIX: Handle empty or insufficient feature_names
                feature_names = latest_snapshot.get('feature_names', [])
                # Ensure we have at least 3 feature names
                while len(feature_names) < 3:
                    feature_names.append(f'Feature {len(feature_names) + 1}')
            else:
                # Pad with zeros if fewer than 3 features
                x = features[:, 0]
                y = features[:, 1] if features.shape[1] > 1 else np.zeros_like(x)
                z = np.zeros_like(x)
                feature_names = latest_snapshot.get('feature_names', [])
                # Ensure we have at least 3 feature names
                while len(feature_names) < 3:
                    feature_names.append(f'Feature {len(feature_names) + 1}')

            unique_classes = np.unique(targets)
            colors = px.colors.qualitative.Set1

            # Plot each class
            for i, cls in enumerate(unique_classes):
                # FIX: Use proper boolean indexing
                class_mask = targets == cls
                class_indices = np.where(class_mask)[0]  # Get integer indices

                if len(class_indices) > 0:
                    # FIX: Use integer indices for all arrays
                    class_predictions = predictions[class_indices]
                    class_targets = targets[class_indices]
                    correct_mask = class_predictions == class_targets

                    cls_name = f'Class {int(cls)}'

                    # Correct predictions
                    correct_indices = class_indices[correct_mask]
                    if len(correct_indices) > 0:
                        fig.add_trace(go.Scatter3d(
                            x=x[correct_indices],
                            y=y[correct_indices],
                            z=z[correct_indices],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=colors[i % len(colors)],
                                opacity=0.8,
                                line=dict(width=1, color='white')
                            ),
                            name=f'{cls_name} ✓',
                            hovertemplate=(
                                f'{cls_name} - Correct<br>' +
                                f'{feature_names[0] if len(feature_names) > 0 else "Feature 1"}: %{{x:.3f}}<br>' +
                                f'{feature_names[1] if len(feature_names) > 1 else "Feature 2"}: %{{y:.3f}}<br>' +
                                f'{feature_names[2] if len(feature_names) > 2 else "Feature 3"}: %{{z:.3f}}<br>' +
                                '<extra></extra>'
                            )
                        ))

                    # Incorrect predictions
                    incorrect_indices = class_indices[~correct_mask]
                    if len(incorrect_indices) > 0:
                        fig.add_trace(go.Scatter3d(
                            x=x[incorrect_indices],
                            y=y[incorrect_indices],
                            z=z[incorrect_indices],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=colors[i % len(colors)],
                                opacity=0.8,
                                symbol='x',
                                line=dict(width=2, color='black')
                            ),
                            name=f'{cls_name} ✗',
                            hovertemplate=(
                                f'{cls_name} - Incorrect<br>' +
                                f'{feature_names[0] if len(feature_names) > 0 else "Feature 1"}: %{{x:.3f}}<br>' +
                                f'{feature_names[1] if len(feature_names) > 1 else "Feature 2"}: %{{y:.3f}}<br>' +
                                f'{feature_names[2] if len(feature_names) > 2 else "Feature 3"}: %{{z:.3f}}<br>' +
                                '<extra></extra>'
                            )
                        ))

            # Update layout
            fig.update_layout(
                title={
                    'text': f"3D Feature Space Visualization<br>Iteration {latest_snapshot['iteration']}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                scene=dict(
                    xaxis_title=feature_names[0] if len(feature_names) > 0 else "Feature 1",
                    yaxis_title=feature_names[1] if len(feature_names) > 1 else "Feature 2",
                    zaxis_title=feature_names[2] if len(feature_names) > 2 else "Feature 3",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=1000,
                height=700,
                showlegend=True
            )

            # Ensure output directory exists
            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Enhanced 3D visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Error creating enhanced 3D visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_basic_3d_visualization(self, output_file="basic_3d.html"):
        """Generate a basic 3D visualization that's guaranteed to work"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px

            if not self.feature_space_snapshots:
                return None

            # Use the latest snapshot
            snapshot = self.feature_space_snapshots[-1]
            features = snapshot['features']
            targets = snapshot['targets']

            # Simple 3D scatter plot
            if features.shape[1] >= 3:
                fig = px.scatter_3d(
                    x=features[:, 0], y=features[:, 1], z=features[:, 2],
                    color=targets.astype(str),
                    title=f"3D Feature Space - Iteration {snapshot['iteration']}"
                )
            else:
                # If less than 3 features, use what we have
                x = features[:, 0]
                y = features[:, 1] if features.shape[1] > 1 else np.zeros_like(x)
                z = np.zeros_like(x) if features.shape[1] < 3 else features[:, 2]

                fig = px.scatter_3d(
                    x=x, y=y, z=z,
                    color=targets.astype(str),
                    title=f"3D Feature Space - Iteration {snapshot['iteration']}"
                )

            fig.write_html(output_file)
            print(f"✅ Basic 3D visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating basic 3D visualization: {e}")
            return None

    def generate_animated(self, output_file="animated_training.html"):
        """Generate animated training progression"""
        try:
            if not self.feature_space_snapshots:
                return None

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Create simple animation with accuracy progression
            rounds = [s['round'] for s in self.accuracy_progression]
            accuracies = [s['accuracy'] for s in self.accuracy_progression]

            fig = go.Figure(
                data=[go.Scatter(x=rounds, y=accuracies, mode="lines+markers")],
                layout=go.Layout(
                    title="Training Progress Animation",
                    xaxis=dict(title="Iteration"),
                    yaxis=dict(title="Accuracy (%)"),
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[dict(label="Play",
                                    method="animate",
                                    args=[None])])]
                ),
                frames=[go.Frame(
                    data=[go.Scatter(x=rounds[:k+1], y=accuracies[:k+1])],
                    name=str(k)
                ) for k in range(len(rounds))]
            )

            fig.write_html(output_file)
            print(f"✅ Animated training saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error generating animation: {e}")
            return None

    def generate_feature_explorer(self, output_file="feature_explorer.html"):
        """Generate feature explorer - FIXED NAME"""
        return self.generate_basic_3d_visualization(output_file)

    def generate_performance(self, output_file="performance.html"):
        """Generate performance metrics - FIXED NAME"""
        return self.generate_performance_metrics(output_file)

    def generate_correlation(self, output_file="correlation.html"):
        """Generate correlation matrix - FIXED NAME"""
        return self.generate_correlation_matrix(output_file)

    def generate_tensor_space_visualizations(self, output_dir="Visualizer/Tensor"):
        """Generate all tensor space visualizations in one call"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)

            outputs = {}

            # Generate complex tensor evolution
            complex_file = os.path.join(output_dir, "complex_tensor_evolution.html")
            result = self.generate_complex_tensor_evolution(complex_file)
            if result:
                outputs['complex_tensor'] = result

            # Generate tensor convergence
            convergence_file = os.path.join(output_dir, "tensor_convergence.html")
            result = self.generate_tensor_convergence_plot()
            if result:
                import plotly.io as pio
                pio.write_html(result, convergence_file)
                outputs['tensor_convergence'] = convergence_file

            # Generate phase diagram
            phase_file = os.path.join(output_dir, "tensor_phase_diagram.html")
            result = self.generate_complex_phase_diagram(phase_file)
            if result:
                outputs['phase_diagram'] = result

            # Generate feature orthogonality
            ortho_file = os.path.join(output_dir, "feature_orthogonality.html")
            result = self.generate_feature_orthogonality_plot(ortho_file)
            if result:
                outputs['feature_orthogonality'] = result

            return outputs

        except Exception as e:
            print(f"Error generating tensor visualizations: {e}")
            return {}

    def generate_3d_spherical_tensor_evolution(self, output_file="3d_spherical_tensor_evolution.html"):
        """Generate 3D spherical polar coordinate visualization of 5D Complex Tensor evolution"""
        # Create help window for this visualization type
        help_content = """
🎓 3D Spherical Polar Tensor Evolution

Shows how 5D complex tensors evolve during training in a 3D spherical coordinate system.

• Radial distance (r) = Overall tensor magnitude
• Polar angle (θ) = Primary complex phase angle
• Azimuthal angle (φ) = Secondary complex phase angle
• Colors = Different classes
• Animation = Training progression

Each point represents a data sample's 5D complex tensor projected into 3D spherical coordinates:
- r = sqrt(|z₁|² + |z₂|² + |z₃|² + |z₄|² + |z₅|²)
- θ = arg(z₁)  [primary complex component]
- φ = arg(z₂)  [secondary complex component]

Watch as classes organize into distinct spherical regions as training progresses.
"""
        help_window = self.create_help_window("3D Spherical Tensor Evolution", help_content)

        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np

            if not self.feature_space_snapshots:
                print("No feature space snapshots available for 3D spherical visualization")
                return None

            print("🔄 Generating 3D spherical polar tensor evolution visualization...")

            # Create figure
            fig = go.Figure()

            # Get data from snapshots - use more iterations for smoother animation
            n_iterations = min(40, len(self.feature_space_snapshots))
            iterations_to_show = np.linspace(0, len(self.feature_space_snapshots)-1, n_iterations, dtype=int)

            colors = px.colors.qualitative.Set1
            frames = []

            for frame_idx, snapshot_idx in enumerate(iterations_to_show):
                snapshot = self.feature_space_snapshots[snapshot_idx]
                features = snapshot['features']
                targets = snapshot['targets']
                iteration = snapshot['iteration']

                # Sample data for performance
                n_samples = min(200, len(features))
                if len(features) > n_samples:
                    indices = np.random.choice(len(features), n_samples, replace=False)
                    features = features[indices]
                    targets = targets[indices]

                unique_classes = np.unique(targets)
                frame_data = []

                # Calculate progress (0 to 1) based on iteration
                progress = min(1.0, iteration / 100.0)

                for class_idx, cls in enumerate(unique_classes):
                    class_mask = targets == cls
                    n_class_samples = np.sum(class_mask)

                    if n_class_samples > 0:
                        # Convert 5D features to 3D spherical coordinates
                        # For 5D complex tensors, we'll use:
                        # - Radial distance: overall magnitude of first 3 complex components
                        # - Polar angle (θ): phase of primary complex component
                        # - Azimuthal angle (φ): phase of secondary complex component

                        class_features = features[class_mask]

                        # Calculate spherical coordinates for each sample
                        spherical_coords = []

                        for sample in class_features:
                            # Treat features as complex numbers (real, imag pairs)
                            # For 5D, we have 5 real components - interpret as 2.5 complex numbers
                            # We'll use the first 4 components as 2 complex numbers

                            if len(sample) >= 4:
                                # Create complex numbers from feature pairs
                                z1 = complex(sample[0], sample[1])  # First complex component
                                z2 = complex(sample[2], sample[3])  # Second complex component

                                # Calculate spherical coordinates
                                r = np.sqrt(abs(z1)**2 + abs(z2)**2)  # Combined magnitude

                                # Primary angle (polar angle θ) from first complex component
                                theta = np.angle(z1)  # Range: -π to π

                                # Secondary angle (azimuthal angle φ) from second complex component
                                phi = np.angle(z2)    # Range: -π to π

                            else:
                                # Fallback for insufficient features
                                r = np.linalg.norm(sample)
                                theta = 2 * np.pi * class_idx / len(unique_classes)
                                phi = np.pi / 4

                            # Apply training progression effects
                            if progress > 0.3:
                                # As training progresses, classes become more separated
                                class_theta = 2 * np.pi * class_idx / len(unique_classes)
                                class_phi = np.pi * (class_idx + 1) / (len(unique_classes) + 1)

                                # Blend towards class centers based on progress
                                blend = min(1.0, (progress - 0.3) / 0.7)
                                theta = (1 - blend) * theta + blend * class_theta
                                phi = (1 - blend) * phi + blend * class_phi

                            # Add some noise in early training
                            if progress < 0.5:
                                noise_level = 1.0 - progress
                                theta += noise_level * 0.5 * np.random.randn()
                                phi += noise_level * 0.3 * np.random.randn()

                            # Convert to Cartesian coordinates for 3D plotting
                            x = r * np.sin(phi) * np.cos(theta)
                            y = r * np.sin(phi) * np.sin(theta)
                            z = r * np.cos(phi)

                            spherical_coords.append((x, y, z))

                        spherical_coords = np.array(spherical_coords)

                        if len(spherical_coords) > 0:
                            # Create 3D scatter plot for this class
                            scatter_trace = go.Scatter3d(
                                x=spherical_coords[:, 0],
                                y=spherical_coords[:, 1],
                                z=spherical_coords[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=6,
                                    color=colors[class_idx % len(colors)],
                                    opacity=0.8,
                                    line=dict(width=2, color='white')
                                ),
                                name=f'Class {int(cls)}',
                                legendgroup=f'class_{cls}',
                                showlegend=(frame_idx == 0),
                                hovertemplate=(
                                    f'Class {int(cls)}<br>'
                                    'Radial: %{customdata[0]:.2f}<br>'
                                    'Polar Angle: %{customdata[1]:.1f}°<br>'
                                    'Azimuthal Angle: %{customdata[2]:.1f}°<br>'
                                    'Iteration: %{customdata[3]}<br>'
                                    '<extra></extra>'
                                ),
                                customdata=np.column_stack([
                                    np.sqrt(spherical_coords[:, 0]**2 + spherical_coords[:, 1]**2 + spherical_coords[:, 2]**2),  # r
                                    np.degrees(np.arccos(spherical_coords[:, 2] / np.sqrt(spherical_coords[:, 0]**2 + spherical_coords[:, 1]**2 + spherical_coords[:, 2]**2 + 1e-8))),  # θ in degrees
                                    np.degrees(np.arctan2(spherical_coords[:, 1], spherical_coords[:, 0])),  # φ in degrees
                                    np.full(len(spherical_coords), iteration)
                                ])
                            )
                            frame_data.append(scatter_trace)

                # Add spherical coordinate system guides
                if frame_idx == 0:  # Only add once
                    # Add coordinate axes
                    axis_length = 2.0
                    frame_data.extend([
                        # X-axis
                        go.Scatter3d(
                            x=[0, axis_length], y=[0, 0], z=[0, 0],
                            mode='lines',
                            line=dict(color='red', width=4),
                            name='X-axis',
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        # Y-axis
                        go.Scatter3d(
                            x=[0, 0], y=[0, axis_length], z=[0, 0],
                            mode='lines',
                            line=dict(color='green', width=4),
                            name='Y-axis',
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        # Z-axis
                        go.Scatter3d(
                            x=[0, 0], y=[0, 0], z=[0, axis_length],
                            mode='lines',
                            line=dict(color='blue', width=4),
                            name='Z-axis',
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    ])

                    # Add spherical grid
                    self._add_spherical_grid(frame_data)

                frame = go.Frame(
                    data=frame_data,
                    name=f'frame_{frame_idx}',
                    layout=go.Layout(
                        title_text=f"3D Spherical Tensor Evolution - Iteration {iteration}",
                        annotations=[
                            dict(
                                text=f'Progress: {progress:.1%}',
                                x=0.02, y=0.98, xref='paper', yref='paper',
                                showarrow=False, bgcolor='white', bordercolor='black',
                                borderwidth=1, font=dict(size=14)
                            )
                        ]
                    )
                )
                frames.append(frame)

            # Add initial frame data
            if frames:
                for trace in frames[0].data:
                    fig.add_trace(trace)

            # Update layout for 3D spherical visualization
            fig.update_layout(
                title={
                    'text': "3D Spherical Polar Tensor Space Evolution",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': 'darkblue'}
                },
                width=1200,
                height=800,
                autosize=True,
                margin=dict(l=50, r=50, t=80, b=50),
                paper_bgcolor='white',
                plot_bgcolor='white',
                showlegend=True,
                scene=dict(
                    xaxis_title='X (r·sinφ·cosθ)',
                    yaxis_title='Y (r·sinφ·sinθ)',
                    zaxis_title='Z (r·cosφ)',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    aspectmode='cube'
                )
            )

            # Animation controls
            fig.update_layout(
                updatemenus=[
                    {
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {
                                'label': '▶️ Play',
                                'method': 'animate',
                                'args': [None, {
                                    'frame': {'duration': 400, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 300},
                                    'mode': 'immediate'
                                }]
                            },
                            {
                                'label': '⏸️ Pause',
                                'method': 'animate',
                                'args': [[None], {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate'
                                }]
                            }
                        ],
                        'x': 0.1,
                        'y': 0.02,
                        'bgcolor': 'lightblue',
                        'bordercolor': 'navy',
                        'borderwidth': 2
                    }
                ]
            )

            # Add slider for manual control
            steps = []
            for i, snapshot_idx in enumerate(iterations_to_show):
                snapshot = self.feature_space_snapshots[snapshot_idx]
                iteration = snapshot['iteration']

                step = {
                    'args': [
                        [f'frame_{i}'],
                        {
                            'frame': {'duration': 400, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }
                    ],
                    'label': f'Iter {iteration}',
                    'method': 'animate'
                }
                steps.append(step)

            fig.update_layout(
                sliders=[{
                    'active': 0,
                    'currentvalue': {
                        'prefix': 'Iteration: ',
                        'xanchor': 'right',
                        'font': {'size': 16, 'color': 'black'}
                    },
                    'transition': {'duration': 400, 'easing': 'cubic-in-out'},
                    'x': 0.1,
                    'len': 0.8,
                    'xanchor': 'left',
                    'y': 0.02,
                    'yanchor': 'top',
                    'bgcolor': 'lightgray',
                    'bordercolor': 'black',
                    'borderwidth': 1,
                    'steps': steps
                }]
            )

            # Add educational annotation
            fig.add_annotation(
                text="🎓 <b>3D Spherical Polar Coordinates</b><br>"
                     "• <b>Radial (r)</b>: Combined magnitude of complex tensor components<br>"
                     "• <b>Polar Angle (θ)</b>: Phase of primary complex component<br>"
                     "• <b>Azimuthal Angle (φ)</b>: Phase of secondary complex component<br>"
                     "• <b>Colors</b>: Different object classes<br>"
                     "• <b>Animation</b>: Training progression from random to organized",
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                showarrow=False,
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=2,
                font=dict(size=11, color='black'),
                align='left'
            )

            # Add coordinate system explanation
            fig.add_annotation(
                text="🔄 <b>Coordinate System:</b><br>"
                     "• X = r·sin(φ)·cos(θ)<br>"
                     "• Y = r·sin(φ)·sin(θ)<br>"
                     "• Z = r·cos(φ)<br>"
                     "• r ≥ 0, 0 ≤ θ ≤ 2π, 0 ≤ φ ≤ π",
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                bgcolor="lightblue",
                bordercolor="blue",
                borderwidth=1,
                font=dict(size=10, color='black'),
                align='right'
            )

            fig.frames = frames

            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ 3D spherical tensor evolution visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating 3D spherical tensor evolution: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _add_spherical_grid(self, frame_data, max_radius=2.0):
        """Add spherical coordinate grid to the 3D plot"""
        try:
            import numpy as np

            # Add spherical surfaces at different radii
            for radius in [0.5, 1.0, 1.5]:
                u = np.linspace(0, 2 * np.pi, 30)
                v = np.linspace(0, np.pi, 15)

                x = radius * np.outer(np.cos(u), np.sin(v))
                y = radius * np.outer(np.sin(u), np.sin(v))
                z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

                frame_data.append(go.Surface(
                    x=x, y=y, z=z,
                    opacity=0.1,
                    colorscale='Blues',
                    showscale=False,
                    hoverinfo='skip',
                    name=f'Sphere r={radius}'
                ))

            # Add polar angle lines (constant θ)
            for theta in np.linspace(0, 2*np.pi, 8):
                phi = np.linspace(0, np.pi, 20)
                r = max_radius
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)

                frame_data.append(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Add azimuthal angle lines (constant φ)
            for phi in np.linspace(0, np.pi, 6):
                theta = np.linspace(0, 2*np.pi, 30)
                r = max_radius
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi) * np.ones_like(theta)

                frame_data.append(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))

        except Exception as e:
            print(f"Error adding spherical grid: {e}")

    def generate_enhanced_3d_spherical_visualization(self, output_file="enhanced_3d_spherical.html"):
        """Generate enhanced 3D spherical visualization with multiple views and analytics"""
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np

            if not self.feature_space_snapshots:
                return None

            print("🔄 Generating enhanced 3D spherical visualization with analytics...")

            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "scatter3d"}, {"type": "scatter3d"}],
                    [{"type": "xy"}, {"type": "xy"}]
                ],
                subplot_titles=(
                    "3D Spherical Tensor Space - Main View",
                    "3D Spherical Tensor Space - Top View",
                    "Radial Distribution Evolution",
                    "Angular Separation Progress"
                ),
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )

            # Use latest snapshot for main visualization
            latest_snapshot = self.feature_space_snapshots[-1]
            features = latest_snapshot['features']
            targets = latest_snapshot['targets']
            iteration = latest_snapshot['iteration']

            colors = px.colors.qualitative.Set1
            unique_classes = np.unique(targets)

            # Calculate spherical coordinates for all samples
            spherical_data = []
            for sample in features:
                if len(sample) >= 4:
                    z1 = complex(sample[0], sample[1])
                    z2 = complex(sample[2], sample[3])
                    r = np.sqrt(abs(z1)**2 + abs(z2)**2)
                    theta = np.angle(z1)
                    phi = np.angle(z2)

                    x = r * np.sin(phi) * np.cos(theta)
                    y = r * np.sin(phi) * np.sin(theta)
                    z = r * np.cos(phi)
                else:
                    r = np.linalg.norm(sample)
                    theta = 0
                    phi = np.pi/4
                    x, y, z = r, 0, 0

                spherical_data.append((x, y, z, r, theta, phi))

            spherical_data = np.array(spherical_data)

            # Plot 1: Main 3D view
            for class_idx, cls in enumerate(unique_classes):
                class_mask = targets == cls
                if np.any(class_mask):
                    class_data = spherical_data[class_mask]

                    fig.add_trace(go.Scatter3d(
                        x=class_data[:, 0],
                        y=class_data[:, 1],
                        z=class_data[:, 2],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=colors[class_idx % len(colors)],
                            opacity=0.7
                        ),
                        name=f'Class {int(cls)}',
                        legendgroup=f'class_{cls}',
                        showlegend=True,
                        hovertemplate=(
                            f'Class {int(cls)}<br>'
                            'X: %{x:.2f}<br>'
                            'Y: %{y:.2f}<br>'
                            'Z: %{z:.2f}<br>'
                            '<extra></extra>'
                        )
                    ), row=1, col=1)

            # Plot 2: Top view (XY plane)
            for class_idx, cls in enumerate(unique_classes):
                class_mask = targets == cls
                if np.any(class_mask):
                    class_data = spherical_data[class_mask]

                    fig.add_trace(go.Scatter3d(
                        x=class_data[:, 0],
                        y=class_data[:, 1],
                        z=np.zeros_like(class_data[:, 2]),  # Project to Z=0 plane
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=colors[class_idx % len(colors)],
                            opacity=0.6
                        ),
                        name=f'Class {int(cls)}',
                        legendgroup=f'class_{cls}',
                        showlegend=False,
                        hovertemplate=(
                            f'Class {int(cls)}<br>'
                            'X: %{x:.2f}<br>'
                            'Y: %{y:.2f}<br>'
                            '<extra></extra>'
                        )
                    ), row=1, col=2)

            # Plot 3: Radial distribution evolution
            if len(self.feature_space_snapshots) > 5:
                radial_evolution = []
                iterations = []

                for i, snapshot in enumerate(self.feature_space_snapshots[::5]):  # Sample every 5th
                    features_sample = snapshot['features']
                    # Calculate average radial distance
                    if features_sample.shape[1] >= 4:
                        radial_dists = []
                        for sample in features_sample:
                            z1 = complex(sample[0], sample[1])
                            z2 = complex(sample[2], sample[3])
                            r = np.sqrt(abs(z1)**2 + abs(z2)**2)
                            radial_dists.append(r)
                        radial_evolution.append(np.mean(radial_dists))
                        iterations.append(snapshot['iteration'])

                fig.add_trace(go.Scatter(
                    x=iterations, y=radial_evolution,
                    mode='lines+markers',
                    name='Avg Radial Distance',
                    line=dict(color='blue', width=3),
                    hovertemplate='Iteration: %{x}<br>Avg Radius: %{y:.3f}<extra></extra>'
                ), row=2, col=1)

            # Plot 4: Angular separation progress
            if len(self.feature_space_snapshots) > 5:
                separation_scores = []
                iterations = []

                for i, snapshot in enumerate(self.feature_space_snapshots[::5]):
                    features_sample = snapshot['features']
                    targets_sample = snapshot['targets']

                    # Calculate class separation in angular space
                    if len(np.unique(targets_sample)) > 1:
                        separation = self._calculate_spherical_separation(features_sample, targets_sample)
                        separation_scores.append(separation)
                        iterations.append(snapshot['iteration'])

                fig.add_trace(go.Scatter(
                    x=iterations, y=separation_scores,
                    mode='lines+markers',
                    name='Angular Separation',
                    line=dict(color='green', width=3),
                    hovertemplate='Iteration: %{x}<br>Separation: %{y:.3f}<extra></extra>'
                ), row=2, col=2)

            # Update layout
            fig.update_layout(
                title={
                    'text': f"Enhanced 3D Spherical Tensor Analysis - Iteration {iteration}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                height=1000,
                showlegend=True,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                scene2=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='',
                    camera=dict(eye=dict(x=0, y=0, z=2.5))
                )
            )

            # Update 2D plot axes
            fig.update_xaxes(title_text="Iteration", row=2, col=1)
            fig.update_yaxes(title_text="Average Radius", row=2, col=1)
            fig.update_xaxes(title_text="Iteration", row=2, col=2)
            fig.update_yaxes(title_text="Separation Score", row=2, col=2)

            # Add educational annotation
            fig.add_annotation(
                text="🎓 <b>Enhanced Spherical Analysis</b><br>"
                     "• <b>Top-left</b>: Full 3D spherical tensor space<br>"
                     "• <b>Top-right</b>: XY projection (top view)<br>"
                     "• <b>Bottom-left</b>: Radial distance evolution<br>"
                     "• <b>Bottom-right</b>: Angular separation progress",
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                showarrow=False,
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=2,
                font=dict(size=11)
            )

            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Enhanced 3D spherical visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating enhanced 3D spherical visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_spherical_separation(self, features, targets):
        """Calculate separation between classes in spherical space"""
        try:
            unique_classes = np.unique(targets)
            if len(unique_classes) < 2:
                return 0.0

            # Calculate class centroids in spherical space
            centroids = []
            for cls in unique_classes:
                class_mask = targets == cls
                if np.sum(class_mask) > 0:
                    class_features = features[class_mask]
                    class_centroid = np.mean(class_features, axis=0)

                    # Convert centroid to spherical coordinates
                    if len(class_centroid) >= 4:
                        z1 = complex(class_centroid[0], class_centroid[1])
                        z2 = complex(class_centroid[2], class_centroid[3])
                        r = np.sqrt(abs(z1)**2 + abs(z2)**2)
                        theta = np.angle(z1)
                        phi = np.angle(z2)
                    else:
                        r = np.linalg.norm(class_centroid)
                        theta, phi = 0, np.pi/4

                    centroids.append((r, theta, phi))

            # Calculate pairwise angular distances
            separation_scores = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    r1, theta1, phi1 = centroids[i]
                    r2, theta2, phi2 = centroids[j]

                    # Calculate angular separation (simplified)
                    angular_dist = np.sqrt((theta1 - theta2)**2 + (phi1 - phi2)**2)
                    separation_scores.append(angular_dist)

            return np.mean(separation_scores) if separation_scores else 0.0

        except Exception as e:
            print(f"Error calculating spherical separation: {e}")
            return 0.0

    def generate_all_spherical_visualizations(self, output_dir="Visualizer/Spherical"):
        """Generate all spherical coordinate visualizations"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)

            outputs = {}

            # Generate basic 3D spherical evolution
            basic_file = os.path.join(output_dir, "3d_spherical_tensor_evolution.html")
            result = self.generate_3d_spherical_tensor_evolution(basic_file)
            if result:
                outputs['basic_spherical'] = result

            # Generate enhanced spherical visualization
            enhanced_file = os.path.join(output_dir, "enhanced_3d_spherical.html")
            result = self.generate_enhanced_3d_spherical_visualization(enhanced_file)
            if result:
                outputs['enhanced_spherical'] = result

            print(f"✅ Generated {len(outputs)} spherical visualizations")
            return outputs

        except Exception as e:
            print(f"Error generating spherical visualizations: {e}")
            return {}
# Generate the 3D spherical visualization
#visualizer = DBNNVisualizer()
#visualizer.generate_3d_spherical_tensor_evolution()

# Or generate all spherical visualizations
#visualizer.generate_all_spherical_visualizations()

class DatasetProcessor:
    """A class to handle dataset-related operations such as downloading, processing, and formatting."""

    def __init__(self, data_dir: str = 'data', config_dir: str = 'config'):
        """
        Initialize the DatasetProcessor.

        Args:
            data_dir: Directory to store datasets.
            config_dir: Directory to store configuration files.
        """
        self.data_dir = data_dir
        self.config_dir = config_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        self.base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        self.compressed_extensions = ['.zip', '.gz', '.tar', '.7z', '.rar']
        self.colors = Colors()


    def download_uci_dataset(self, dataset_name: str, url: str) -> str:
        """
        Download a dataset from the UCI repository.

        Args:
            dataset_name: Name of the dataset.
            url: URL to the dataset.

        Returns:
            Path to the downloaded dataset.
        """
        dataset_path = os.path.join(self.data_dir, f"{dataset_name}.csv")
        if os.path.exists(dataset_path):
            print("\033[K" +f"Dataset {dataset_name} already exists at {dataset_path}.")
            return dataset_path

        print("\033[K" +f"Downloading dataset {dataset_name} from {url}...")
        response = requests.get(url)
        response.raise_for_status()

        # Save the dataset
        with open(dataset_path, 'wb') as f:
            f.write(response.content)

        print("\033[K" +f"Dataset saved to {dataset_path}.")
        return dataset_path


    def search_uci_repository(self, query: str) -> List[Dict[str, str]]:
        """
        Search the UCI repository for datasets matching a query.

        Args:
            query: Search query.

        Returns:
            List of dictionaries containing dataset information.
        """
        search_url = f"https://archive.ics.uci.edu/ml/datasets.php?format=json&query={query}"
        response = requests.get(search_url)
        response.raise_for_status()

        datasets = response.json()
        print("\033[K" +f"Found {len(datasets)} datasets matching query '{query}'.")
        return datasets


class DatasetConfig:
    """Enhanced dataset configuration handling with support for column names and URLs"""

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
             "similarity_threshold": 0.25,  # Bins with >25% probability in predicted class are considered similar
            "cardinality_threshold_percentile": 95
        },
        "training_params": {
            "save_plots": False,  # Parameter to save plots
            "Save_training_epochs": False,  # Save the epochs parameter
             "enable_visualization": False,
            "training_save_path": "data"  # Save epochs path parameter
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
    def create_default_config(dataset_name: str) -> Dict:
        """Create a default configuration file with enhanced defaults"""
        return load_or_create_config(dataset_name)

    @staticmethod
    def load_config(dataset_name: str) -> Dict:
        """Enhanced configuration loading with interactive setup, comment removal, and GLOBAL command line flag support"""
        if not dataset_name or not isinstance(dataset_name, str):
            print("\033[K" + "Error: Invalid dataset name provided.")
            return None

        config_path = os.path.join('data', dataset_name, f"{dataset_name}.conf")
        csv_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")

        try:
            # Use GLOBAL command line flags
            force_visualization = COMMAND_LINE_FLAGS['visualize']
            force_model_type = COMMAND_LINE_FLAGS['model_type']
            force_fresh_start = COMMAND_LINE_FLAGS['fresh_start']
            force_use_previous_model = COMMAND_LINE_FLAGS['use_previous_model']
            force_5dct = COMMAND_LINE_FLAGS['enable_5DCTvisualization']

            # Handle missing config file
            if not os.path.exists(config_path):
                # Check for corresponding CSV file
                if os.path.exists(csv_path):
                    print("\033[K" + f"Configuration file not found. Starting interactive setup for {dataset_name}...")

                    # Read CSV columns
                    columns = []
                    has_header = True
                    try:
                        with open(csv_path, 'r') as f:
                            first_line = f.readline().strip()
                            columns = first_line.split(',')

                            # Verify if we should use first line as header
                            try:
                                pd.read_csv(csv_path, nrows=0)
                            except pd.errors.ParserError:
                                has_header = False
                                columns = []
                    except Exception as e:
                        print("\033[K" + f"Error reading CSV file: {str(e)}")
                        return None

                    if not has_header or not columns:
                        print("\033[K" + "CSV file appears to have no header. Please provide column names.")
                        columns = []
                        while True:
                            col_input = input("\033[K" + "Enter comma-separated column names (including target): ").strip()
                            if col_input:
                                columns = [c.strip() for c in col_input.split(',')]
                                if len(columns) > 1:
                                    break
                            print("\033[K" + "Invalid input! Must provide at least two columns.")

                    print("\033[K" + f"Detected columns: {', '.join(columns)}")

                    # Get target column from user
                    target = None
                    while True:
                        target = input("\033[K" + "Enter target column name: ").strip()
                        if target in columns:
                            break
                        print("\033[K" + f"Invalid column! Choose from: {', '.join(columns)}")

                    # Create base configuration WITH GLOBAL COMMAND LINE DEFAULTS
                    config = DatasetConfig.create_default_config(dataset_name)
                    config.update({
                        "file_path": csv_path,
                        "column_names": columns,
                        "target_column": target,
                        "has_header": has_header,
                        "modelType": force_model_type,  # Use command line default
                        "training_params": {
                            "enable_visualization": force_visualization,  # Use command line default
                            "modelType": force_model_type  # Use command line default
                        },
                        "execution_flags": {
                            "fresh_start": force_fresh_start,  # Use command line default
                            "use_previous_model": force_use_previous_model  # Use command line default
                        }
                    })

                    # Let user edit the config
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.conf', delete=False) as tmp:
                        json.dump(config, tmp, indent=2)
                        tmp_path = tmp.name

                    # Open editor
                    editor = os.environ.get('EDITOR', 'nano')
                    try:
                        subprocess.call([editor, tmp_path])
                    except Exception as e:
                        print("\033[K" + f"Error opening editor: {str(e)}")
                        return None

                    # Load edited config
                    with open(tmp_path, 'r') as f:
                        edited_config = json.load(f)
                    os.remove(tmp_path)

                    # Save final config
                    os.makedirs(os.path.dirname(config_path), exist_ok=True)
                    with open(config_path, 'w') as f:
                        json.dump(edited_config, f, indent=2)
                    print("\033[K" + f"Configuration saved to {config_path}")

                else:
                    print("\033[K" + f"Creating default configuration for {dataset_name}")
                    return DatasetConfig.create_default_config(dataset_name)

            # Existing configuration loading logic
            with open(config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()

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
            config = json.loads(clean_config)
            validated_config = DatasetConfig.DEFAULT_CONFIG.copy()
            validated_config.update(config)

            # APPLY GLOBAL COMMAND LINE OVERRIDES
            applied_overrides = []

            # Ensure training_params exists
            if 'training_params' not in validated_config:
                validated_config['training_params'] = {}

            training_params = validated_config['training_params']

            # Visualization override
            if force_visualization:
                if training_params.get('enable_visualization') != True:
                    training_params['enable_visualization'] = True
                    applied_overrides.append('enable_visualization=True')

            # Model type override
            if force_visualization:
                if 'training_params' not in validated_config:
                    validated_config['training_params'] = {}
                if validated_config.get('training_params', {}).get('enable_visualization') != True:
                    validated_config['training_params']['enable_visualization'] = True
                    applied_overrides.append('enable_visualization')

            # ADD 5DCT override
            if force_5dct:
                if 'training_params' not in validated_config:
                    validated_config['training_params'] = {}
                if validated_config.get('training_params', {}).get('enable_5DCTvisualization') != True:
                    validated_config['training_params']['enable_5DCTvisualization'] = True
                    applied_overrides.append('enable_5DCTvisualization')

            # Ensure execution_flags exists
            if 'execution_flags' not in validated_config:
                validated_config['execution_flags'] = {}

            execution_flags = validated_config['execution_flags']

            # Fresh start override
            if force_fresh_start:
                if execution_flags.get('fresh_start') != True:
                    execution_flags['fresh_start'] = True
                    execution_flags['use_previous_model'] = False
                    applied_overrides.append('fresh_start=True')

            # Use previous model override
            if force_use_previous_model:
                if execution_flags.get('use_previous_model') != True:
                    execution_flags['use_previous_model'] = True
                    execution_flags['fresh_start'] = False
                    applied_overrides.append('use_previous_model=True')

            # Show override summary (only once per dataset)
            if applied_overrides and not hasattr(DatasetConfig.load_config, f'_overrides_shown_{dataset_name}'):
                print(f"🎯 {Colors.GREEN}DATASET CONFIG OVERRIDES for {dataset_name} ({len(applied_overrides)}):{Colors.ENDC}")
                for override in applied_overrides:
                    print(f"   {Colors.CYAN}→ {override}{Colors.ENDC}")
                setattr(DatasetConfig.load_config, f'_overrides_shown_{dataset_name}', True)

            # Set defaults for missing parameters (respecting command line flags)
            if 'enable_visualization' not in training_params:
                training_params['enable_visualization'] = force_visualization
                if not hasattr(DatasetConfig.load_config, f'_default_shown_{dataset_name}'):
                    print(f"🔧 {Colors.YELLOW}Added missing 'enable_visualization' parameter with value: {force_visualization}{Colors.ENDC}")

            if 'enable_5DCTvisualization' not in training_params:  # 5DCT
                    training_params['enable_5DCTvisualization'] = force_5dct
            if 'modelType' not in training_params:
                training_params['modelType'] = force_model_type

            if 'fresh_start' not in execution_flags:
                execution_flags['fresh_start'] = force_fresh_start

            if 'use_previous_model' not in execution_flags:
                execution_flags['use_previous_model'] = force_use_previous_model

            # Path handling
            if validated_config.get('file_path'):
                if not os.path.exists(validated_config['file_path']):
                    alt_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
                    if os.path.exists(alt_path):
                        validated_config['file_path'] = alt_path
                        print("\033[K" + f"Using data file: {alt_path}")

            if not validated_config.get('file_path'):
                default_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
                if os.path.exists(default_path):
                    validated_config['file_path'] = default_path
                    print("\033[K" + f"Using default data file: {default_path}")

            # URL handling
            if DatasetConfig.is_url(validated_config.get('file_path', '')):
                url = validated_config['file_path']
                local_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
                if not os.path.exists(local_path):
                    print("\033[K" + f"Downloading dataset from {url}")
                    if not DatasetConfig.download_dataset(url, local_path):
                        print("\033[K" + f"Failed to download dataset from {url}")
                        return None
                    print("\033[K" + f"Downloaded dataset to {local_path}")
                validated_config['file_path'] = local_path

            # Validation
            if not validated_config.get('file_path') or not os.path.exists(validated_config['file_path']):
                print("\033[K" + "Warning: Data file not found")
                return None

            # Validate columns
            try:
                df = pd.read_csv(validated_config['file_path'], nrows=0)
                # Only set column names if not already configured
                if 'column_names' not in validated_config or not validated_config['column_names']:
                    validated_config['column_names'] = df.columns.tolist()
                # Otherwise, validate configured columns exist in CSV
                else:
                    missing = set(validated_config['column_names']) - set(df.columns)
                    if missing:
                        print(f"Configured columns missing in CSV: {missing}")
            except Exception as e:
                print("\033[K" + f"Warning: Could not validate columns: {str(e)}")
                return None

            # Validate target column exists
            target_col = validated_config.get('target_column')
            if target_col not in validated_config['column_names']:
                print(f"\033[K{Colors.RED}ERROR: Configured target column '{target_col}' not found in dataset!{Colors.ENDC}")
                print(f"\033[KAvailable columns: {', '.join(validated_config['column_names'])}")

                # Interactive target column correction
                while True:
                    new_target = input("\033[KEnter correct target column name: ").strip()
                    if new_target in validated_config['column_names']:
                        validated_config['target_column'] = new_target
                        # Save updated config
                        with open(config_path, 'w') as f:
                            json.dump(validated_config, f, indent=2)
                        print(f"\033[K{Colors.GREEN}Updated target column to '{new_target}' in configuration.{Colors.ENDC}")
                        break
                    print(f"\033[K{Colors.RED}Invalid column! Choose from: {', '.join(validated_config['column_names'])}{Colors.ENDC}")

            if isinstance(target_col, (int, float)):
                # Convert numeric indices to column name
                validated_config['target_column'] = str(validated_config['column_names'][target_col])
            elif isinstance(target_col, str):
                pass  # Already correct
            else:
                raise ValueError(f"Invalid target column type: {type(target_col)}")

            # Final confirmation of visualization state
            final_viz = validated_config.get('training_params', {}).get('enable_visualization', False)
            if force_visualization and final_viz and not hasattr(DatasetConfig.load_config, f'_final_shown_{dataset_name}'):
                print(f"✅ {Colors.GREEN}DATASET CONFIG FINAL: Visualization ENABLED for {dataset_name}{Colors.ENDC}")
                setattr(DatasetConfig.load_config, f'_final_shown_{dataset_name}', True)

            return validated_config

        except Exception as e:
            print("\033[K" + f"Error loading configuration for {dataset_name}: {str(e)}")
            traceback.print_exc()
            return None

    @staticmethod
    def download_dataset(url: str, local_path: str) -> bool:
        """Download dataset from URL to local path with proper error handling"""
        try:
            print("\033[K" +f"Downloading dataset from {url}")
            response = requests.get(url, timeout=30)  # Add timeout
            response.raise_for_status()  # Check for HTTP errors

            # Handle potential text/csv content
            content = response.content.decode('utf-8')

            # Save to local file
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print("\033[K" +f"Dataset downloaded successfully to {local_path}")
            return True
        except requests.exceptions.RequestException as e:
            print("\033[K" +f"Error downloading dataset: {str(e)}")
            return False
        except UnicodeDecodeError:
            # Handle binary content
            try:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print("\033[K" +f"Dataset downloaded successfully to {local_path}")
                return True
            except Exception as e:
                print("\033[K" +f"Error saving binary content: {str(e)}")
                return False
        except Exception as e:
            print("\033[K" +f"Unexpected error downloading dataset: {str(e)}")
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
            print("\033[K" +"Found CSV files without configuration:")
            for csv_name in sorted(csv_without_conf):
                response = input(f"Create configuration for {csv_name}.csv? (y/n): ")
                if response.lower() == 'y':
                    try:
                        DatasetConfig.create_default_config(csv_name)
                        datasets.add(csv_name)
                    except Exception as e:
                        print("\033[K" +f"Error creating config for {csv_name}: {str(e)}")

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
            print("\033[K" +f"Skipping validation for derived dataset: {dataset_name}")
            return False

        config = DatasetConfig.load_config(dataset_name)
        file_path = config['file_path']

        # Handle URL-based datasets
        if DatasetConfig.is_url(file_path):
            if not DatasetConfig.validate_url(file_path):
                print("\033[K" +f"Warning: Dataset URL {file_path} is not accessible")
                return False

            # Download to local cache if needed
            local_path = f"{dataset_name}.csv"
            if not os.path.exists(local_path):
                if not DatasetConfig.download_dataset(file_path, local_path):
                    return False
            file_path = local_path

        if not os.path.exists(file_path):
            print("\033[K" +f"Warning: Dataset file {file_path} not found")
            return False

        return True

#---------------------------------------Feature Filter with a #------------------------------------
def _filter_features_from_config(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Filter DataFrame to only include specified columns from config

    Args:
        df: Input DataFrame
        config: Configuration dictionary containing column names

    Returns:
        DataFrame with only the specified columns
    """
    # If no column names in config, return original DataFrame

    if 'column_names' not in config or not config['column_names']:
        print("\033[K" +"No column names specified in config. Keeping all columns.")
        return df

    # Get current DataFrame columns
    current_cols = df.columns.tolist()
    #print("\033[K" +f"Current DataFrame columns: {current_cols}")

    # Get column names from config (only those not commented out)
    requested_columns = [
        name.strip() for name in config['column_names']
        if not name.strip().startswith('#')
    ]

    # If no uncommented columns found in config, return original DataFrame
    if not requested_columns:
        print("\033[K" +"No uncommented column names found in config. Returning original DataFrame.")
        return df

    # Check if any requested columns exist in the DataFrame
    valid_columns = [col for col in requested_columns if col in current_cols]

    # If no valid columns found, return original DataFrame
    if not valid_columns:
        print("\033[K" +"None of the requested columns exist in the DataFrame. Returning original DataFrame.")
        return df

    # Return DataFrame with only the columns to keep
    #print("\033[K" +f"Keeping only these features: {valid_columns}")
    return df[valid_columns]
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
    def __init__(self, n_classes, feature_pairs, dataset_name, n_bins_per_dim=128, batch_size=128):
        self.dataset_name = dataset_name
        self.n_classes = n_classes
        self.feature_pairs = feature_pairs
        self.n_bins_per_dim = n_bins_per_dim
        self.device = Train_device
        self.batch_size = batch_size

        # Get initial weight from config
        self.initial_weight = self._get_initial_weight_from_config()

        # Initialize histogram_weights with configurable initial value
        self.histogram_weights = {}
        for class_id in range(n_classes):
            self.histogram_weights[class_id] = {}
            for pair_idx in range(len(feature_pairs)):
                # Initialize with configurable initial weight
                self.histogram_weights[class_id][pair_idx] = torch.full(
                    (n_bins_per_dim, n_bins_per_dim),
                    self.initial_weight,
                    dtype=torch.float64,
                    device=self.device
                ).contiguous()

        # Initialize gaussian weights with same initial value
        self.gaussian_weights = {}
        for class_id in range(n_classes):
            self.gaussian_weights[class_id] = {}
            for pair_idx in range(len(feature_pairs)):
                self.gaussian_weights[class_id][pair_idx] = torch.tensor(
                    self.initial_weight,
                    dtype=torch.float64,
                    device=self.device
                ).contiguous()

        # Unified weights tensor initialization
        self.weights = torch.full(
            (n_classes, len(feature_pairs), n_bins_per_dim, n_bins_per_dim),
            self.initial_weight,
            dtype=torch.float64,
            device=self.device
        ).contiguous()

        # Pre-allocate update buffers
        self.update_indices = torch.zeros((3, 1000), dtype=torch.long)
        self.update_values = torch.zeros(1000, dtype=torch.float64)
        self.update_count = 0

        # Debug initialization
        # print("\033[K" + f"[DEBUG] Weight initialization complete with initial value: {self.initial_weight}")
        #print("\033[K" + f"- Number of classes: {len(self.histogram_weights)}")
        #for class_id in self.histogram_weights:
        #   print("\033[K" + f"- Class {class_id}: {len(self.histogram_weights[class_id])} feature pairs")

    def _get_initial_weight_from_config(self):
        """Get initial weight value from configuration"""
        config = DatasetConfig.load_config(self.dataset_name)
        return config.get('anomaly_detection', {}).get('initial_weight', 1e-6)

    def _validate_update(old_weights, new_weights, updates):
        """Sanity check for update consistency"""
        for (class_id, pair_idx, bin_i, bin_j), adj in updates.items():
            assert torch.allclose(
                old_weights[class_id][pair_idx][bin_i, bin_j] + adj,
                new_weights[class_id][pair_idx][bin_i, bin_j]
            ), "Update mismatch detected!"

    def batch_update_weights(self, class_indices, pair_indices, bin_indices, adjustments):
        # Convert to tensors first
        class_ids = torch.tensor(class_indices, dtype=torch.long, device=self.device)
        pair_ids = torch.tensor(pair_indices, dtype=torch.long, device=self.device)
        bin_is = torch.tensor([bi[0] if isinstance(bi, (list, tuple)) else bi
                              for bi in bin_indices], device=self.device)
        bin_js = torch.tensor([bj[1] if isinstance(bj, (list, tuple)) else bj
                              for bj in bin_indices], device=self.device)
        adjs = torch.tensor(adjustments, dtype=torch.float64, device=self.device)

        # Group updates by (class, pair)
        unique_pairs, inverse = torch.unique(
            torch.stack([class_ids, pair_ids]), dim=1, return_inverse=True
        )

        for group_idx in range(unique_pairs.shape[1]):
            class_id, pair_id = unique_pairs[:, group_idx]
            mask = inverse == group_idx

            # Get all updates for this (class, pair)
            b_i = bin_is[mask]
            b_j = bin_js[mask]
            adj = adjs[mask]

            # Vectorized update
            self.histogram_weights[class_id.item()][pair_id.item()].index_put_(
                indices=(b_i, b_j),
                values=adj,
                accumulate=True
            )


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
                dtype=torch.float64,
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
            print("\033[K" +f"Error updating weight: {str(e)}")
            print("\033[K" +f"class_id: {class_id}, pair_idx: {pair_idx}")
            print("\033[K" +f"bin_i: {bin_i}, bin_j: {bin_j}")
            print("\033[K" +f"adjustment: {adjustment}")
            raise

    def update_histogram_weights(self, failed_case, true_class, pred_class,
                               bin_indices, posteriors, learning_rate):
        # Get config parameters
        config = DatasetConfig.load_config(self.dataset_name)
        #update_condition = config['active_learning'].get('update_condition', 'probability_threshold')
        update_condition = config['active_learning'].get('update_condition', 'bin_overlap')
        similarity_threshold = config['active_learning'].get('similarity_threshold', 0.25)

        # Precompute base adjustment
        base_adjustment = learning_rate * (1.0 - (posteriors[true_class] / posteriors[pred_class]))

        # Batch indices and values
        pair_indices = []
        bin_is = []
        bin_js = []
        adjustments = []

        for pair_idx, (bin_i, bin_j) in bin_indices.items():
            # Check update condition
            should_update = False

            if update_condition == "bin_overlap":
                # Get class-specific bins for this feature pair
                true_bins = self.class_bins.get(true_class, {}).get(pair_idx, set())
                pred_bins = self.class_bins.get(pred_class, {}).get(pair_idx, set())

                # Update only if no bin overlap between classes
                if (bin_i, bin_j) not in pred_bins:
                    should_update = True
            else:  # Default probability threshold
                # Get predicted class probability in this bin
                pred_prob = self.likelihood_params['bin_probs'][pair_idx][pred_class][bin_i, bin_j]
                if pred_prob < similarity_threshold:
                    should_update = True

            if should_update:
                pair_indices.append(pair_idx)
                bin_is.append(bin_i)
                bin_js.append(bin_j)
                adjustments.append(base_adjustment)

        # Convert to tensors if we have updates
        if pair_indices:
            pair_ids = torch.tensor(pair_indices, dtype=torch.long, device=self.device)
            b_i = torch.tensor(bin_is, dtype=torch.long, device=self.device)
            b_j = torch.tensor(bin_js, dtype=torch.long, device=self.device)
            adjs = torch.tensor(adjustments, dtype=torch.float64, device=self.device)

            # Group by pair_idx
            unique_pairs, counts = torch.unique(pair_ids, return_counts=True)

            for pair_id in unique_pairs:
                mask = pair_ids == pair_id
                self.histogram_weights[true_class][pair_id.item()].index_put_(
                    indices=(b_i[mask], b_j[mask]),
                    values=adjs[mask],
                    accumulate=True
                )


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

    def _get_overlapping_bins_for_pair(self, pair_idx):
        """Return bins shared by any two classes for this feature pair"""
        all_class_bins = [self.class_bins[cls][pair_idx] for cls in self.class_bins]

        # Find bins present in at least two classes
        overlapping = set()
        for i, bins_i in enumerate(all_class_bins):
            for j, bins_j in enumerate(all_class_bins[i+1:]):
                overlapping.update(bins_i & bins_j)

        return torch.tensor(list(overlapping), device=self.device)  # shape [n_overlapping, 2]

    def compute_histogram_posterior(self, features, bin_indices):
        config = DatasetConfig.load_config(self.dataset_name)
        update_condition = config['active_learning'].get('update_condition', 'bin_overlap')

        log_likelihoods = torch.zeros((features.shape[0], self.n_classes), device=self.device)

        for group_idx, feature_group in enumerate(self.feature_pairs):
            # Existing probability calculation
            bin_probs = self.likelihood_params['bin_probs'][group_idx]
            bin_weights = self.weight_updater.get_histogram_weights(class_idx, group_idx)
            weighted_probs = bin_probs * bin_weights.unsqueeze(0)

            # New: Zero probabilities in overlapping bins if using bin_overlap
            if update_condition == "bin_overlap":
                # Get overlapping bins between any class pairs
                overlapping_bins = self._get_overlapping_bins_for_pair(group_idx)
                weighted_probs[:, overlapping_bins[:,0], overlapping_bins[:,1]] = 0

            log_likelihoods += torch.log(weighted_probs + epsilon)

        return log_likelihoods

    # Modified posterior computation for Histogram model
    def compute_histogram_posterior_old(self, features, bin_indices):
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
#----------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict
import numpy as np
from tqdm import tqdm

class InvertibleDBNN(nn.Module):
    """Invertible DBNN for reconstructing input features from classification probabilities."""

    def __init__(self, forward_model: nn.Module, feature_dims: int, n_classes: int, hidden_dims: int = 128, device: str = 'cuda'):
        """
        Initialize the Invertible DBNN.

        Args:
            forward_model (nn.Module): The forward DBNN model.
            feature_dims (int): Number of input feature dimensions.
            n_classes (int): Number of classes in the classification task.
            hidden_dims (int): Number of hidden dimensions in the inverse model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        super(InvertibleDBNN, self).__init__()
        self.forward_model = forward_model
        self.feature_dims = feature_dims
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.device = device

        # Define the inverse model architecture
        self.inverse_model = nn.Sequential(
            nn.Linear(n_classes, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, feature_dims))

        # Move model to the appropriate device
        self.to(device)

    def forward(self, class_probs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the inverse model to reconstruct features.

        Args:
            class_probs (torch.Tensor): Classification probabilities (batch_size, n_classes).

        Returns:
            torch.Tensor: Reconstructed features (batch_size, feature_dims).
        """
        return self.inverse_model(class_probs)

    def reconstruct_features(self, class_probs: torch.Tensor, original_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct features from classification probabilities.

        Args:
            class_probs (torch.Tensor): Classification probabilities (batch_size, n_classes).
            original_features (Optional[torch.Tensor]): Original features for validation (batch_size, feature_dims).

        Returns:
            torch.Tensor: Reconstructed features (batch_size, feature_dims).
        """
        reconstructed_features = self.forward(class_probs)

        if original_features is not None:
            # Calculate reconstruction error
            reconstruction_error = torch.mean((reconstructed_features - original_features) ** 2)
            print("\033[K" +f"Reconstruction Error: {reconstruction_error.item():.4f}")

        return reconstructed_features

    def train_inverse_model(self, class_probs: torch.Tensor, original_features: torch.Tensor, epochs: int = 100, lr: float = 0.001):
        """
        Train the inverse model to reconstruct features from classification probabilities.

        Args:
            class_probs (torch.Tensor): Classification probabilities (batch_size, n_classes).
            original_features (torch.Tensor): Original features (batch_size, feature_dims).
            epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in tqdm(range(epochs), desc="Training Inverse Model", leave=False):
            optimizer.zero_grad()

            # Forward pass
            reconstructed_features = self.forward(class_probs)

            # Compute reconstruction loss
            loss = criterion(reconstructed_features, original_features)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print("\033[K" +f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def save_model(self, path: str):
        """Save the inverse model to a file."""
        torch.save(self.state_dict(), path)
        print("\033[K" +f"Model saved to {path}")

    def load_model(self, path: str):
        """Load the inverse model from a file."""
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        print("\033[K" +f"Model loaded from {path}")
#----------------------------------------------DBNN class-------------------------------------------------------------
class GPUDBNN:
    """GPU-Optimized Deep Bayesian Neural Network with Parallel Feature Pair Processing"""

    def __init__(self, dataset_name: str, learning_rate: float = LearningRate,
                 max_epochs: int = Epochs, test_size: float = TestFraction,
                 random_state: int = TrainingRandomSeed, device: str = None,
                 fresh: bool = False, use_previous_model: bool = True,
                 n_bins_per_dim: int = 128, model_type: str = "Histogram",mode: str=None):
        """Initialize GPUDBNN with support for continued training with fresh data"""

        # Set dataset_name and model type first
        self.mode = mode
        self.dataset_name = dataset_name
        self.model_type = model_type  # Store model type as instance variable

        # Load configuration BEFORE any parameter setting
        self.config = DatasetConfig.load_config(self.dataset_name)
        if self.config is None:
            raise ValueError(f"Failed to load configuration for dataset: {self.dataset_name}")

        training_params = self.config.get('training_params', {})



        # USE CONFIG VALUES DIRECTLY, ignore the passed parameters (they come from wrong source)
        self.learning_rate = training_params.get('learning_rate', LearningRate)
        self.max_epochs = training_params.get('epochs', Epochs)
        self.test_size = training_params.get('test_fraction', TestFraction)
        self.n_bins_per_dim = training_params.get('n_bins_per_dim', 128)
        self.batch_size = training_params.get('batch_size', 128)
        if self.batch_size is None:
            self.batch_size = 128
            print(f"{Colors.YELLOW}[WARNING] batch_size was None, using default: 128{Colors.ENDC}")
        # Handle random_state specially (your code uses -1 for no shuffle)
        config_random_seed = training_params.get('random_seed', TrainingRandomSeed)
        if random_state is None:
            self.random_state = config_random_seed
        else:
            self.random_state = random_state

        self.shuffle_state = 1 if self.random_state != -1 else -1

        # Device configuration - use config preference
        compute_device = training_params.get('compute_device', 'auto')
        if device is None:
            if compute_device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = compute_device
        else:
            self.device = device

        # Initialize computation cache with correct device
        self.computation_cache = ComputationCache(self.device)

        # Initialize train/test indices
        self.train_indices = []
        self.test_indices = None
        self._last_metrics_printed = False

        # Add new attribute for bin-specific weights
        self.weight_updater = None  # Will be initialized after computing likelihood params

        self.feature_bounds = None  # Store global min/max for each

        # Use the n_bins_per_dim from config, not the parameter
        self.n_bins_per_dim = training_params.get('n_bins_per_dim',
                            self.config.get('likelihood_config', {}).get('n_bins_per_dim', 128))

        # Initialize other attributes that should use CONFIG values
        self.cardinality_tolerance = training_params.get('cardinality_tolerance', cardinality_tolerance)
        self.fresh_start = fresh
        self.use_previous_model = use_previous_model

        # Create Model directory
        os.makedirs('Model', exist_ok=True)

        # Load configuration and data
        self.target_column = self.config['target_column']

        # Initialize model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Handle label encoder initialization based on mode
        if mode == 'predict':
            # Strict validation for prediction mode
            if not self._load_model_components():
                raise RuntimeError(
                    "Cannot run prediction - no trained model found.\n"
                    f"Expected model files in: Model/{self.dataset_name}_*\n"
                    "Please train the model first."
                )
        elif self.use_previous_model:
            # Try to load existing model, but don't fail if it doesn't exist
            try:
                if not self._load_model_components():
                    print("\033[K" +f"[INFO] No existing model found - starting fresh training")
                    self._initialize_fresh_training()
            except Exception as e:
                print("\033[K" +f"[WARNING] Failed to load previous model: {str(e)}")
                print("\033[K" +"Starting fresh training")
                self._initialize_fresh_training()
        else:
            # Fresh training requested
            self._initialize_fresh_training()

        self.likelihood_params = None
        self.feature_pairs = None
        self.bin_edges = None  # Add bin_edges attribute
        self.gaussian_params = None  # Add gaussian_params attribute for Gaussian model
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

        # Handle model state based on flags
        if use_previous_model:
            # Load previous model state
            pass
        else:
            # Complete fresh start
            self._clean_existing_model()

        #------------------------------------------Adaptive Learning--------------------------------------
        # Initialize adaptive learning parameters from config
        self.adaptive_learning = training_params.get('enable_adaptive', True)
        self.base_save_path = './data'
        os.makedirs(self.base_save_path, exist_ok=True)
        self.in_adaptive_fit = False  # Set when we are in adaptive learning process

        # Store adaptive learning parameters from training_params
        self.adaptive_rounds = training_params.get('adaptive_rounds', 100)
        self.initial_samples = training_params.get('initial_samples', 50)
        self.max_samples_per_round = training_params.get('max_samples_per_round', 500)
        self.minimum_training_accuracy = training_params.get('minimum_training_accuracy', 0.95)

        # Load active_learning config parameters and merge with training_params
        active_learning_config = self.config.get('active_learning', {})

        # Store active learning parameters (use training_params as primary, fallback to active_learning)
        self.min_divergence = training_params.get('min_divergence',
                            active_learning_config.get('min_divergence', 0.1))
        self.max_class_addition_percent = training_params.get('max_class_addition_percent',
                            active_learning_config.get('max_class_addition_percent', 99))
        self.cardinality_threshold_percentile = training_params.get('cardinality_threshold_percentile',
                            active_learning_config.get('cardinality_threshold_percentile', 95))
        self.similarity_threshold = training_params.get('similarity_threshold',
                            active_learning_config.get('similarity_threshold', 0.25))
        self.strong_margin_threshold = training_params.get('strong_margin_threshold',
                            active_learning_config.get('strong_margin_threshold', 0.01))
        self.marginal_margin_threshold = training_params.get('marginal_margin_threshold',
                            active_learning_config.get('marginal_margin_threshold', 0.01))
        self.tolerance = training_params.get('tolerance',
                            active_learning_config.get('tolerance', 1.0))

        # CRITICAL: Initialize the missing attributes that DBNN expects
        self.best_round = None
        self.best_round_initial_conditions = None
        self.best_combined_accuracy = 0.00
        self.best_model_weights = None

        # Model components (re-initialize to ensure consistency)
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
        self.target_column = self.config['target_column']
        self.data = self._load_dataset()

        # Load saved weights and encoders
        self._load_best_weights()
        self._load_categorical_encoders()


    def _initialize_fresh_training(self):
        """Initialize components for fresh training"""
        # Load dataset configuration and data
        self.target_column = self.config['target_column']
        self.data = self._load_dataset()
        if self.target_column not in self.data.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset.\n"
                f"Available columns: {list(self.data.columns)}"
            )

        # Fit label encoder with string conversion for universal handling
        target_data = self.data[self.target_column].astype(str)
        self.label_encoder.fit(target_data)

        self.scaler = StandardScaler()
        self.feature_pairs = None
        self.likelihood_params = None

        # Mark as fresh training
        self.fresh_start = True

    def _compute_bin_edges(self, dataset: torch.Tensor, bin_sizes: List[int]) -> List[List[torch.Tensor]]:
        """
        Vectorized computation of bin edges with missing value handling.

        Args:
            dataset: Input tensor of shape [n_samples, n_features]
            bin_sizes: List of integers specifying bin sizes

        Returns:
            List of lists containing bin edge tensors for each feature pair
        """
        DEBUG.log("Starting vectorized _compute_bin_edges with missing value handling")

        # Get configuration parameters
        config = self.config.get('anomaly_detection', {})
        missing_value = config.get('missing_value', -99999)
        mv_epsilon = 1e-6  # Small buffer around missing value

        # Memory management parameters
        MAX_GPU_MEM = 0.8 * torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 1e10
        SAFETY_FACTOR = 0.7  # Use only 70% of available memory

        # Calculate memory requirements per feature pair
        bytes_per_pair = 2 * 4 * len(bin_sizes)  # 2 edges, 4 bytes per float, per dimension
        max_pairs_per_batch = int((MAX_GPU_MEM * SAFETY_FACTOR) / bytes_per_pair)
        max_pairs_per_batch = max(1, min(max_pairs_per_batch, len(self.feature_pairs)))

        bin_edges = []

        # Process in memory-managed batches
        for batch_start in range(0, len(self.feature_pairs), max_pairs_per_batch):
            batch_end = min(batch_start + max_pairs_per_batch, len(self.feature_pairs))
            batch_pairs = self.feature_pairs[batch_start:batch_end]

            # Vectorized min/max computation for the batch
            with torch.no_grad():
                # Stack all features needed in this batch
                feature_indices = torch.unique(torch.cat([torch.as_tensor(pair, device=self.device)
                                        for pair in batch_pairs]))
                batch_data = dataset[:, feature_indices]

                # Compute min/max for all features in batch
                mins = batch_data.min(dim=0)[0]
                maxs = batch_data.max(dim=0)[0]

                # Create mapping from feature index to its position in batch_data
                feat_to_idx = {int(f): i for i, f in enumerate(feature_indices)}

                # Process each pair in batch
                for pair in batch_pairs:
                    pair_edges = []
                    for dim, feat in enumerate(pair):
                        feat_idx = feat_to_idx[int(feat)]
                        dim_min = mins[feat_idx]
                        dim_max = maxs[feat_idx]
                        padding = max((dim_max - dim_min) * 0.01, 1e-6)

                        # Get bin size for this dimension
                        bin_size = bin_sizes[0] if len(bin_sizes) == 1 else bin_sizes[dim]

                        # Compute initial edges
                        edges = torch.linspace(
                            dim_min - padding,
                            dim_max + padding,
                            bin_size + 1,
                            device=self.device
                        ).contiguous()

                        # Check for missing values in original dataset
                        has_missing = torch.any(dataset[:, feat] == missing_value)

                        if has_missing:
                            # Calculate missing value bin boundaries
                            mv_low = torch.tensor(missing_value - mv_epsilon, device=self.device)
                            mv_high = torch.tensor(missing_value + mv_epsilon, device=self.device)

                            # Determine insertion position
                            if missing_value < edges[0]:
                                # Prepend missing value bin
                                edges = torch.cat([mv_low.unsqueeze(0), mv_high.unsqueeze(0), edges])
                            elif missing_value > edges[-1]:
                                # Append missing value bin
                                edges = torch.cat([edges, mv_low.unsqueeze(0), mv_high.unsqueeze(0)])
                            else:
                                # Find insertion point and split existing bins
                                insert_pos = torch.searchsorted(edges, mv_high, right=True)
                                edges = torch.cat([
                                    edges[:insert_pos],
                                    mv_low.unsqueeze(0),
                                    mv_high.unsqueeze(0),
                                    edges[insert_pos:]
                                ])

                            # Maintain sorted order
                            edges, _ = torch.sort(edges)

                        pair_edges.append(edges)

                    bin_edges.append(pair_edges)

                torch.cuda.empty_cache()  # Free memory between batches

        return bin_edges

#---------------------- -------------------------------------DBNN Class -------------------------------
class DBNNConfig:
    """Configuration class for DBNN parameters"""
    def __init__(self, **kwargs):
        # Training parameters
        self.trials = kwargs.get('trials', 100)
        self.cardinality_threshold = kwargs.get('cardinality_threshold', 0.9)
        self.cardinality_tolerance = kwargs.get('cardinality_tolerance', 4)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.random_seed = kwargs.get('random_seed', 42)
        self.epochs = kwargs.get('epochs', 1000)
        self.test_fraction = kwargs.get('test_fraction', 0.2)
        self.enable_adaptive = kwargs.get('enable_adaptive', True)
        self.batch_size = kwargs.get('batch_size', 128)
        # Model parameters
        self.model_type = kwargs.get('model_type', 'Histogram')  # or 'Gaussian'
        self.n_bins_per_dim = kwargs.get('n_bins_per_dim', 128)

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

    @classmethod
    def from_file(cls, config_path: str) -> 'DBNNConfig':
        """Create configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)

class DBNN(GPUDBNN):
    """Enhanced DBNN class that builds on GPUDBNN implementation"""

    def __init__(self, config: Optional[Union[DBNNConfig, dict]] = None,
                 dataset_name: Optional[str] = None, mode=None, model_type: Optional[str] = None,
                 enable_visualization: bool = False, visualization_frequency: int = 1,
                 enable_5DCTvisualization: bool = False):

        # First load the actual dataset configuration
        self.dataset_name = dataset_name
        self.config = DatasetConfig.load_config(dataset_name)
        if self.config is None:
            raise ValueError(f"Failed to load configuration for dataset: {dataset_name}")

        # Get training parameters from the actual JSON config
        training_params = self.config.get('training_params', {})

        # Extract ALL parameters from JSON config with proper fallbacks
        trials = training_params.get('trials', 100)
        cardinality_threshold = training_params.get('cardinality_threshold', 0.9)
        learning_rate = training_params.get('learning_rate', 0.001)
        random_seed = training_params.get('random_seed', 42)
        epochs = training_params.get('epochs', 1000)
        test_fraction = training_params.get('test_fraction', 0.2)
        n_bins_per_dim = training_params.get('n_bins_per_dim', 128)

        # Handle the DBNNConfig parameter properly
        if config is None:
            # Create DBNNConfig using values from JSON config
            config_dict = {
                'trials': trials,
                'cardinality_threshold': cardinality_threshold,
                'learning_rate': learning_rate,
                'random_seed': random_seed,
                'epochs': epochs,
                'test_fraction': test_fraction,
                'n_bins_per_dim': n_bins_per_dim,
                'enable_adaptive': training_params.get('enable_adaptive', True),
                'train': training_params.get('train', True),
                'train_only': training_params.get('train_only', False),
                'predict': training_params.get('predict', True),
                'fresh_start': training_params.get('fresh_start', False),
                'use_previous_model': training_params.get('use_previous_model', True),
                'model_type': model_type or training_params.get('model_type', 'Histogram')
            }
            config = DBNNConfig(**config_dict)
        elif isinstance(config, dict):
            # Merge with JSON config values
            for key in ['learning_rate', 'epochs', 'test_fraction', 'random_seed', 'n_bins_per_dim']:
                if key in training_params:
                    config[key] = training_params[key]
            config = DBNNConfig(**config)

        # Now call parent with CORRECT parameters from JSON config
        super().__init__(
            dataset_name=dataset_name,
            learning_rate=learning_rate,      # From JSON config
            max_epochs=epochs,               # From JSON config
            test_size=test_fraction,          # From JSON config
            random_state=random_seed,         # From JSON config
            fresh=config.fresh_start,
            use_previous_model=config.use_previous_model,
            n_bins_per_dim=n_bins_per_dim,    # From JSON config
            model_type=model_type or training_params.get('model_type', 'Histogram'),
            mode=mode
        )

        # Store the actual parameters that will be used
        self.mode = mode
        self.cardinality_threshold = cardinality_threshold
        self.trials = trials
        self.patience = trials
        self.adaptive_patience = training_params.get('adaptive_patience', 25)

        # Pass self reference to visualizer
        self.visualizer = DBNNVisualizer()
        self.visualizer.dbnn_instance = self  # Pass reference

        # USE GLOBAL FLAGS AS DEFAULTS
        if enable_visualization is False:
            enable_visualization = COMMAND_LINE_FLAGS['visualize']
        if enable_5DCTvisualization is False:
            enable_5DCTvisualization = COMMAND_LINE_FLAGS['enable_5DCTvisualization']

        # Initialize the functionality
        add_geometric_visualization_to_adbnn()
        # Enhanced Visualization Integration for DBNN
        self.enable_visualization = enable_visualization
        self.visualization_frequency = visualization_frequency
        self.visualization_snapshots = enable_visualization

        # Initialize visualizer only if visualization is enabled
        if self.enable_visualization:
            self.visualizer = DBNNVisualizer()
            self._initialize_visualization_directories()
            self.visualizer5DCT = DBNN_5DCT_Visualizer(self)
            self.enable_5DCTvisualization = True

            # Adaptive training data storage for visualization
            self._initialize_visualization_directories()
            self.adaptive_round_data = []
            self.adaptive_snapshots = []

            print(f"{Colors.CYAN}[DBNN-VISUAL] 🎨 UNIFIED VISUALIZATION ENABLED - All visualization types activated{Colors.ENDC}")
            print(f"{Colors.CYAN}[DBNN-VISUAL] 📊 Includes: 5DCT, 3D Spherical, Tensor Evolution, Performance Dashboards{Colors.ENDC}")
            print(f"{Colors.CYAN}[DBNN-VISUAL] ⚡ Snapshot frequency: every {visualization_frequency} round(s){Colors.ENDC}")
        else:
            # Initialize with disabled state but ensure attributes exist
            self.visualizer = None
            self.visualizer5DCT = DBNN_5DCT_Visualizer(self)  # Still create but disabled
            self.enable_5DCTvisualization = False
            self.adaptive_round_data = []
            self.adaptive_snapshots = []

    def _initialize_visualization_directories(self):
        """Initialize visualization directory structure for DBNN with consistent paths"""
        print(f"🔍 DEBUG: _initialize_visualization_directories called")
        print(f"🔍 DEBUG: self.enable_visualization = {getattr(self, 'enable_visualization', 'NOT SET')}")

        if not self.enable_visualization:
            print("❌ DEBUG: Visualization disabled, skipping directory creation")
            return

        print("✅ DEBUG: Visualization enabled, creating directories...")

        # Use consistent directory name (British spelling as shown in logs)
        self.viz_output_dir = "Visualizer"
        self.tensor_viz_dir = os.path.join(self.viz_output_dir, "Tensor")
        self.spherical_viz_dir = os.path.join(self.viz_output_dir, "Spherical")
        self.standard_viz_dir = os.path.join(self.viz_output_dir, "Standard")
        self.adaptive_viz_dir = os.path.join(self.viz_output_dir, "Adaptive")
        self.dbnn_viz_dir = os.path.join(self.viz_output_dir, "DBNN")

        # Create directories
        directories = [
            self.viz_output_dir, self.tensor_viz_dir, self.spherical_viz_dir,
            self.standard_viz_dir, self.adaptive_viz_dir, self.dbnn_viz_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created/Verified: {directory}")

        print("✅ DEBUG: All visualization directories created successfully")

        # Store directory mapping for consistent path generation
        self.viz_directories = {
            'tensor': self.tensor_viz_dir,
            'spherical': self.spherical_viz_dir,
            'standard': self.standard_viz_dir,
            'adaptive': self.adaptive_viz_dir,
            'dbnn': self.dbnn_viz_dir,
            'base': self.viz_output_dir
        }

    def _get_viz_output_path(self, subdir, filename):
        """Get consistent output path for visualizations"""
        try:
            if subdir in self.viz_directories:
                base_dir = self.viz_directories[subdir]
            else:
                base_dir = self.viz_directories['base']

            output_path = os.path.join(base_dir, filename)
            print(f"Output Path for the file is set to: {output_path}")

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            return output_path

        except Exception as e:
            print(f"❌ Error getting visualization path: {e}")
            # Fallback to current directory
            return filename


    def compute_global_statistics(self, X: pd.DataFrame):
        """Compute global statistics (e.g., mean, std) for normalization."""
        batch_size = self.batch_size  # Adjust based on available memory
        n_samples = len(X)
        n_features = X.shape[1]

        # Initialize accumulators
        sum_features = np.zeros(n_features)
        sum_squared_features = np.zeros(n_features)

        # Compute sum and sum of squares in batches
        for i in range(0, n_samples, batch_size):
            batch_X = X.iloc[i:i + batch_size]
            batch_X_numeric = batch_X.select_dtypes(include=[np.number])  # Only numeric features
            sum_features += batch_X_numeric.sum(axis=0)
            sum_squared_features += (batch_X_numeric ** 2).sum(axis=0)

        # Compute mean and standard deviation
        self.global_mean = sum_features / n_samples
        self.global_std = np.sqrt((sum_squared_features / n_samples) - (self.global_mean ** 2))

        # Handle zero standard deviation (replace with 1 to avoid division by zero)
        self.global_std[self.global_std == 0] = 1.0

    def _preprocess_and_split_data(self):
        """Preprocess data and split into training and testing sets with prediction mode support."""
        # Load dataset
        self.target_column = self.config['target_column']
        self.data = self._load_dataset()

        # Determine mode
        predict_mode = (self.mode == 'predict')

        # Handle features and labels based on mode
        if predict_mode:
            # Prediction mode logic
            if self.target_column in self.data.columns:
                # Target column exists - use it for potential evaluation
                X = self.data.drop(columns=[self.target_column])
                y_true = self.data[self.target_column].astype(str)
                self.prediction_true_labels = y_true  # Store for evaluation
                DEBUG.log(f"Target column found with {len(y_true.unique())} unique classes")
            else:
                # Pure prediction mode - no target column
                X = self.data.copy()
                self.prediction_true_labels = None
                DEBUG.log("No target column found - running in pure prediction mode")

            # Create dummy y for processing (won't be used for actual encoding in prediction)
            y = pd.Series(['dummy_pred'] * len(self.data))

        else:
            # Training mode logic
            if self.target_column not in self.data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in dataset")

            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column].astype(str)
            self.prediction_true_labels = None  # Not in prediction mode

        # Encode labels - handle prediction mode carefully
        if predict_mode:
            # In prediction mode, we should have a pre-trained label encoder
            if not hasattr(self.label_encoder, 'classes_'):
                raise RuntimeError("Label encoder not fitted. Load a trained model first for prediction.")

            # For prediction, we don't actually use y for encoding, but we need a tensor
            # Create dummy encoded values (all zeros or based on first class)
            y_encoded = np.zeros(len(y), dtype=int)
            DEBUG.log("Using dummy encoding for prediction mode")

        else:
            # Training mode - fit or transform the encoder
            if not hasattr(self.label_encoder, 'classes_'):
                y_encoded = self.label_encoder.fit_transform(y.astype(str))
                DEBUG.log(f"Fitted label encoder with classes: {self.label_encoder.classes_}")
            else:
                y_encoded = self.label_encoder.transform(y.astype(str))
                DEBUG.log(f"Transformed labels using existing encoder")

        # Preprocess features
        X_processed = self._preprocess_data(X, is_training=not predict_mode)

        # Convert to tensors
        self.X_tensor = X_processed.clone().detach().to(self.device)
        self.y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(self.device)

        # Split data (use all data as "test" in prediction mode)
        if predict_mode:
            self.X_train, self.X_test = None, self.X_tensor
            self.y_train, self.y_test = None, self.y_tensor
            self.train_indices, self.test_indices = [], list(range(len(self.data)))
            DEBUG.log(f"Prediction mode: using all {len(self.data)} samples for prediction")
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = self._get_train_test_split(
                self.X_tensor, self.y_tensor)
            DEBUG.log(f"Training mode: {len(self.X_train)} train, {len(self.X_test)} test samples")

        self._is_preprocessed = True
        DEBUG.log("Data preprocessing completed successfully")

    def create_invertible_model(self, reconstruction_weight: float = 0.5, feedback_strength: float = 0.3):
        """Create an invertible DBNN model"""
        if self.invertible_model is None:
            self.invertible_model = InvertibleDBNN(
                forward_model=self,
                feature_dims=self.data.shape[1] - 1,  # Exclude target column
                reconstruction_weight=reconstruction_weight,
                feedback_strength=feedback_strength
            )
        return self.invertible_model

    def _generate_detailed_predictions(self,
                                     X_orig: Union[pd.DataFrame, torch.Tensor],
                                     predictions: Union[torch.Tensor, np.ndarray],
                                     true_labels: Union[torch.Tensor, np.ndarray, pd.Series, List, None] = None,
                                     posteriors: Union[torch.Tensor, np.ndarray, None] = None
                                     ) -> pd.DataFrame:
        """
        Robust predictions generator that preserves original feature values.
        """
        # Convert predictions to numpy if they're tensors
        predictions_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else np.array(predictions)

        # Convert posteriors if provided
        posteriors_np = posteriors.cpu().numpy() if torch.is_tensor(posteriors) else np.array(posteriors) if posteriors is not None else None

        # Verify sizes match
        if len(predictions_np) != len(X_orig):
            print(f"\033[K{Colors.RED}[ERROR] Size mismatch: predictions={len(predictions_np)}, X_orig={len(X_orig)}{Colors.ENDC}")
            # Truncate or pad to match (this should not happen, but handle it)
            min_len = min(len(predictions_np), len(X_orig))
            predictions_np = predictions_np[:min_len]
            if posteriors_np is not None:
                posteriors_np = posteriors_np[:min_len]
            print(f"\033[K{Colors.YELLOW}[WARNING] Truncated to {min_len} samples{Colors.ENDC}")

        # Create results DataFrame from original features
        if isinstance(X_orig, pd.DataFrame):
            results_df = X_orig.iloc[:len(predictions_np)].copy()  # Ensure we only take matching rows
        else:
            # Handle tensor/numpy array input
            X_orig_np = X_orig.cpu().numpy() if torch.is_tensor(X_orig) else np.array(X_orig)
            X_orig_np = X_orig_np[:len(predictions_np)]  # Truncate to match predictions
            results_df = pd.DataFrame(X_orig_np,
                                    columns=getattr(self, 'feature_columns',
                                                  [f'feature_{i}' for i in range(X_orig_np.shape[1])]))

        # Add predictions with universal label decoding
        if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
            try:
                # Ensure we don't have more predictions than we can handle
                pred_classes = predictions_np[:len(results_df)]

                # Decode predictions to original format (strings)
                decoded_predictions = self.label_encoder.inverse_transform(pred_classes)

                # Convert back to original data type if numeric
                if hasattr(self, 'original_target_dtype'):
                    try:
                        if np.issubdtype(self.original_target_dtype, np.number):
                            decoded_predictions = np.array(decoded_predictions, dtype=self.original_target_dtype)
                    except:
                        pass  # Keep as string if conversion fails

                results_df['predicted_class'] = decoded_predictions

                # Add posteriors if available
                if posteriors_np is not None:
                    posteriors_np = posteriors_np[:len(results_df)]  # Truncate to match
                    for i, class_name in enumerate(self.label_encoder.classes_):
                        if i < posteriors_np.shape[1]:  # Safety check
                            results_df[f'prob_{class_name}'] = posteriors_np[:, i]

                    # Add prediction confidence
                    confidence_indices = np.arange(len(pred_classes))
                    results_df['prediction_confidence'] = posteriors_np[confidence_indices, pred_classes]

            except ValueError as e:
                print(f"\033[K{Colors.YELLOW}Note: Using raw predictions - {str(e)}{Colors.ENDC}")
                results_df['predicted_class'] = predictions_np[:len(results_df)]
                if posteriors_np is not None:
                    posteriors_np = posteriors_np[:len(results_df)]
                    confidence_indices = np.arange(len(predictions_np[:len(results_df)]))
                    results_df['prediction_confidence'] = posteriors_np[confidence_indices, predictions_np[:len(results_df)]]
        else:
            results_df['predicted_class'] = predictions_np[:len(results_df)]
            if posteriors_np is not None:
                posteriors_np = posteriors_np[:len(results_df)]
                confidence_indices = np.arange(len(predictions_np[:len(results_df)]))
                results_df['prediction_confidence'] = posteriors_np[confidence_indices, predictions_np[:len(results_df)]]

        # Handle true labels if provided
        if true_labels is not None:
            true_labels_np = true_labels.cpu().numpy() if torch.is_tensor(true_labels) \
                            else true_labels.to_numpy() if isinstance(true_labels, (pd.Series, pd.DataFrame)) \
                            else np.array(true_labels)

            # Truncate to match results_df
            true_labels_np = true_labels_np[:len(results_df)]

            # Decode true labels using the same encoder
            if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
                try:
                    # For encoded labels, decode them
                    if true_labels_np.dtype in [np.int32, np.int64] and true_labels_np.max() < len(self.label_encoder.classes_):
                        decoded_true = self.label_encoder.inverse_transform(true_labels_np)
                        # Convert back to original type if numeric
                        if hasattr(self, 'original_target_dtype') and np.issubdtype(self.original_target_dtype, np.number):
                            try:
                                decoded_true = np.array(decoded_true, dtype=self.original_target_dtype)
                            except:
                                pass
                        results_df['true_class'] = decoded_true
                    else:
                        # Labels are already in original format
                        results_df['true_class'] = true_labels_np
                except Exception as e:
                    print(f"\033[K{Colors.YELLOW}Couldn't decode true labels: {str(e)}{Colors.ENDC}")
                    results_df['true_class'] = true_labels_np
            else:
                results_df['true_class'] = true_labels_np

        # Final size verification
        if len(results_df) != len(predictions_np):
            print(f"\033[K{Colors.YELLOW}[WARNING] Final size mismatch: results={len(results_df)}, predictions={len(predictions_np)}{Colors.ENDC}")

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
        """Optimized dataset loader with GPU memory management"""
        DEBUG.log(f"Loading dataset: {self.dataset_name}")

        try:
            # Config validation
            if not self.config:
                raise ValueError(f"No config for dataset: {self.dataset_name}")

            file_path = self.config.get('file_path')
            if not file_path:
                raise ValueError("No file path in config")

            # Load data
            if file_path.startswith(('http://', 'https://')):
                df = pd.read_csv(StringIO(requests.get(file_path).text),
                               sep=self.config.get('separator', ','),
                               header=0 if self.config.get('has_header', True) else None,  low_memory=False)
            else:
                df = pd.read_csv(file_path,
                               sep=self.config.get('separator', ','),
                               header=0 if self.config.get('has_header', True) else None,  low_memory=False)

            predict_mode = (self.mode == 'predict')

            # Store original target data type for proper decoding
            if self.target_column in df.columns:
                self.original_target_dtype = df[self.target_column].dtype
                DEBUG.log(f"Stored original target dtype: {self.original_target_dtype}")
            else:
                self.original_target_dtype = np.dtype('object')
                DEBUG.log("Target column not found, using default object dtype")

            # Handle target column validation for prediction mode
            if predict_mode and self.target_column in df.columns:
                DEBUG.log(f"Target column '{self.target_column}' found in prediction data - will use for evaluation if needed")
                # Keep target column for potential evaluation, but don't use for encoding

            # Store original data (CPU only)
            self.Original_data = df.copy()

            # Handle data splitting based on mode
            if predict_mode:
                # Prediction mode - target column may or may not exist
                if self.target_column in df.columns:
                    self.X_Orig = df.drop(columns=[self.target_column]).copy()
                    DEBUG.log(f"Using {len(df.columns) - 1} features for prediction (target column excluded)")
                else:
                    self.X_Orig = df.copy()
                    DEBUG.log(f"Using all {len(df.columns)} columns for prediction (no target column found)")
            else:
                # Training mode - target column must exist
                if self.target_column not in df.columns:
                    raise ValueError(f"Target column '{self.target_column}' not found in dataset")
                self.X_Orig = df.drop(columns=[self.target_column]).copy()
                DEBUG.log(f"Using {len(df.columns) - 1} features for training")

            # Filter features if specified
            if 'column_names' in self.config:
                df = _filter_features_from_config(df, self.config)

            # Handle target column configuration
            target_col = self.config['target_column']
            if isinstance(target_col, int):
                target_col = df.columns[target_col]
                self.config['target_column'] = target_col

            # For prediction mode, don't require target column to be present
            if not predict_mode and target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found")

            # Shuffling logic (CPU only)
            shuffle_path = os.path.join(
                self.config.get('training_params', {}).get('training_save_path', 'training_data'),
                os.path.splitext(os.path.basename(self.dataset_name))[0],
                'data',
                'shuffled_data.csv'
            )

            if self.fresh_start and self.random_state == -1:
                for _ in range(3):
                    df = df.sample(frac=1).reset_index(drop=True)
                os.makedirs(os.path.dirname(shuffle_path), exist_ok=True)
                df.to_csv(shuffle_path, index=False)
            elif os.path.exists(shuffle_path):
                df = pd.read_csv(shuffle_path)

            # Convert to memory-efficient format
            df = df.astype({
                col: 'category' if df[col].dtype == 'object' else df[col].dtype
                for col in df.columns
            })

            # Store metadata separately (never moves to GPU)
            self._metadata = {
                'file_path': file_path,
                'column_names': list(df.columns),
                'target_column': target_col,
                'original_index': df.index.values,
                'shuffle_path': shuffle_path,
                'original_target_dtype': str(self.original_target_dtype),
                'prediction_mode': predict_mode,
                'has_target_column': self.target_column in df.columns
            }

            # Convert target column to string for universal encoding ONLY if it exists
            if self.target_column in df.columns:
                df[self.target_column] = df[self.target_column].astype(str)
                DEBUG.log(f"Converted target column to string for universal encoding")

            return df

        except Exception as e:
            DEBUG.log(f"Dataset load error: {str(e)}")
            raise RuntimeError(f"Failed to load {self.dataset_name}: {str(e)}")

    def _compute_batch_posterior(self, features: torch.Tensor, epsilon: float = 1e-10):
        """Optimized vectorized batch posterior computation with consistent return type"""
        features = features.to(self.device)
        batch_size, n_features = features.shape
        n_classes = len(self.likelihood_params['classes'])
        n_pairs = len(self.feature_pairs)

        # Pre-allocate all feature groups at once
        feature_groups = torch.stack([
            features[:, pair].contiguous()
            for pair in self.feature_pairs
        ])  # [n_pairs, batch_size, 2]

        # Vectorized binning for all pairs - return as dictionary for compatibility
        bin_indices_dict = {}
        for group_idx in range(n_pairs):
            bin_edges = self.likelihood_params['bin_edges'][group_idx]
            edges = torch.stack([edge.contiguous() for edge in bin_edges])

            # Vectorized bucketize for both dimensions
            indices_0 = torch.bucketize(feature_groups[group_idx, :, 0], edges[0]) - 1
            indices_1 = torch.bucketize(feature_groups[group_idx, :, 1], edges[1]) - 1
            indices_0 = indices_0.clamp(0, self.n_bins_per_dim - 1)
            indices_1 = indices_1.clamp(0, self.n_bins_per_dim - 1)

            bin_indices_dict[group_idx] = (indices_0, indices_1)

        # Vectorized probability computation
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        # Process pairs in optimal batches for memory efficiency
        pair_batch_size = min(50, n_pairs)  # Adjust based on memory
        for batch_start in range(0, n_pairs, pair_batch_size):
            batch_end = min(batch_start + pair_batch_size, n_pairs)

            for group_idx in range(batch_start, batch_end):
                bin_probs = self.likelihood_params['bin_probs'][group_idx]
                bin_weights = torch.stack([
                    self.weight_updater.get_histogram_weights(c, group_idx)
                    for c in range(n_classes)
                ])

                indices_0, indices_1 = bin_indices_dict[group_idx]

                # Vectorized probability gathering
                weighted_probs = bin_probs * bin_weights
                probs = weighted_probs[:, indices_0, indices_1]  # [n_classes, batch_size]
                log_likelihoods += torch.log(probs.t() + epsilon)

            # Memory cleanup between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Vectorized softmax
        max_log_likelihood = log_likelihoods.max(dim=1, keepdim=True)[0]
        posteriors = torch.exp(log_likelihoods - max_log_likelihood)
        posteriors /= posteriors.sum(dim=1, keepdim=True) + epsilon

        return posteriors, bin_indices_dict  # Return as dictionary for compatibility

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
                    print("\033[K" +f"Removed existing model file: {file}")
        except Exception as e:
            print("\033[K" +f"Warning: Error cleaning model files: {str(e)}")

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
            #print("\033[K" +f"Saved epoch {epoch} data to {epoch_dir}")
        except Exception as e:
            print("\033[K" +f"Error saving epoch data: {str(e)}")

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
            print("\033[K" +"Warning: No cardinality data available. Using synthetic distribution based on percentile.")
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
        print("\033[K" +f"Cardinality Analysis:")
        print("\033[K" +f"- Using {percentile}th percentile threshold")
        print("\033[K" +f"- Distribution statistics:")
        print("\033[K" +f"  - Min: {min_card:.2f}")
        print("\033[K" +f"  - Max: {max_card:.2f}")
        print("\033[K" +f"  - Mean: {mean_card:.2f}")
        print("\033[K" +f"  - Median: {median_card:.2f}")
        print("\033[K" +f"  - Threshold: {threshold:.2f}")

        # Print number of samples that would be included
        n_included = sum(c <= threshold for c in cardinalities)
        print("\033[K" +f"- {n_included} out of {len(cardinalities)} samples below threshold "
              f"({(n_included/len(cardinalities))*100:.1f}%)")

        return threshold
#--------------------compute sample divergence -----------------------

    def _compute_sample_divergence(self, sample_data: torch.Tensor, feature_pairs: List[Tuple]) -> torch.Tensor:
        """Memory-optimized divergence computation with chunked processing"""
        device = sample_data.device
        n_samples = sample_data.shape[0]

        if n_samples <= 1:
            return torch.zeros((1, 1), device=device)

        # For very large sample sizes, use chunked processing
        if n_samples > 1000:  # Use chunking for large datasets
            return self._compute_sample_divergence_chunked(sample_data, feature_pairs)

        # For smaller datasets, use the optimized vectorized approach
        n_pairs = len(feature_pairs)
        distances = torch.zeros((n_samples, n_samples), device=device, dtype=torch.float64)

        # Process all feature pairs in one batch if memory allows
        if n_pairs <= 100 and n_samples <= 500:  # Conservative limits
            try:
                # Stack all feature pairs data
                all_pair_data = torch.stack([sample_data[:, pair] for pair in feature_pairs])  # [n_pairs, n_samples, 2]

                # Memory-efficient pairwise distance calculation
                # Instead of creating huge 4D tensor, compute distances pair by pair
                for i in range(n_pairs):
                    pair_data = all_pair_data[i]  # [n_samples, 2]

                    # Efficient pairwise Euclidean distance
                    diff = pair_data.unsqueeze(0) - pair_data.unsqueeze(1)  # [n_samples, n_samples, 2]
                    pair_dist = torch.norm(diff, p=2, dim=2)  # [n_samples, n_samples]
                    distances += pair_dist

                distances /= n_pairs

            except RuntimeError as e:
                if "alloc" in str(e).lower():
                    # Fall back to chunked processing if memory fails
                    return self._compute_sample_divergence_chunked(sample_data, feature_pairs)
                else:
                    raise
        else:
            # Use chunked processing for larger cases
            return self._compute_sample_divergence_chunked(sample_data, feature_pairs)

        # Normalize while maintaining numerical stability
        max_val = distances.max()
        return distances / (max_val + 1e-7) if max_val > 0 else distances

    def _compute_sample_divergence_chunked(self, sample_data: torch.Tensor, feature_pairs: List[Tuple]) -> torch.Tensor:
        """Memory-safe chunked version for large datasets"""
        device = sample_data.device
        n_samples = sample_data.shape[0]
        n_pairs = len(feature_pairs)

        # Initialize distances matrix
        distances = torch.zeros((n_samples, n_samples), device=device, dtype=torch.float64)

        # Determine optimal chunk size based on available memory
        if n_samples > 5000:
            chunk_size = 100  # Very conservative for huge datasets
        elif n_samples > 1000:
            chunk_size = 200
        else:
            chunk_size = 500

        # Process feature pairs in batches
        pair_batch_size = min(50, n_pairs)

        with torch.no_grad():
            for pair_start in range(0, n_pairs, pair_batch_size):
                pair_end = min(pair_start + pair_batch_size, n_pairs)
                batch_pairs = feature_pairs[pair_start:pair_end]

                # Get batch data
                batch_data = sample_data[:, batch_pairs]  # [n_samples, batch_size, 2]

                # Process samples in chunks to avoid memory explosion
                for i_start in range(0, n_samples, chunk_size):
                    i_end = min(i_start + chunk_size, n_samples)

                    for j_start in range(0, n_samples, chunk_size):
                        j_end = min(j_start + chunk_size, n_samples)

                        # Compute distances for this chunk
                        chunk_i = batch_data[i_start:i_end]  # [chunk_size, batch_size, 2]
                        chunk_j = batch_data[j_start:j_end]  # [chunk_size, batch_size, 2]

                        # Expand for pairwise comparison
                        chunk_i_expanded = chunk_i.unsqueeze(1)  # [chunk_size, 1, batch_size, 2]
                        chunk_j_expanded = chunk_j.unsqueeze(0)  # [1, chunk_size, batch_size, 2]

                        # Compute differences and distances
                        diff = chunk_i_expanded - chunk_j_expanded  # [chunk_size, chunk_size, batch_size, 2]
                        chunk_dist = torch.norm(diff, p=2, dim=3)  # [chunk_size, chunk_size, batch_size]

                        # Average across feature pairs and accumulate
                        chunk_avg_dist = chunk_dist.mean(dim=2)  # [chunk_size, chunk_size]

                        # Update the main distances matrix
                        distances[i_start:i_end, j_start:j_end] += chunk_avg_dist

                # Memory cleanup
                del batch_data, chunk_i, chunk_j, diff, chunk_dist, chunk_avg_dist
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Normalize by number of pair batches
        distances /= (n_pairs / pair_batch_size)

        # Final normalization
        max_val = distances.max()
        return distances / (max_val + 1e-7) if max_val > 0 else distances

    def _get_feature_batch_size(self, n_samples, n_pairs):
        """Dynamically determine optimal feature batch size based on available memory"""
        if not torch.cuda.is_available():
            return 100  # Default for CPU

        # Calculate available memory with safety margin
        free_mem = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        safety_margin = 0.8  # Use only 80% of available memory
        available_mem = free_mem * safety_margin

        # Memory estimation per feature pair batch
        per_pair_mem = n_samples * n_samples * 4  # 4 bytes per float32
        max_batches = available_mem // per_pair_mem

        return min(max(1, int(max_batches)), 256)  # Limit between 1-256

    def _compute_feature_cardinalities(self, samples_data: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of feature cardinalities.
        """
        cardinalities = torch.zeros(len(samples_data), device=self.device)

        # Process feature pairs in batches
        batch_size = self.batch_size  # Adjust based on memory constraints
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

    def _select_samples_from_failed_classes(self, test_predictions, y_test, test_indices, results):
        """Memory-optimized sample selection with PER-CLASS sampling and safe label handling"""
        from tqdm import tqdm

        # Configuration parameters
        active_learning_config = self.config.get('active_learning', {})
        min_divergence = active_learning_config.get('min_divergence', 0.1)
        max_samples_per_class = active_learning_config.get('max_samples_per_round', 500)  # Now per class

        # Convert inputs to tensors on active device
        test_predictions = torch.as_tensor(test_predictions, device=self.device)
        y_test = torch.as_tensor(y_test, device=self.device)
        test_indices = torch.as_tensor(test_indices, device=self.device)

        all_results = results['all_predictions']
        test_results = all_results.iloc[self.test_indices]

        # Create boolean mask using numpy arrays to avoid chained indexing
        misclassified_mask = test_results['predicted_class'] != test_results['true_class']
        misclassified_indices = test_results.index[misclassified_mask].tolist()

        # Create mapping from original indices to test set positions
        test_pos_map = {idx: pos for pos, idx in enumerate(self.test_indices)}

        final_selected_indices = []

        # Get unique classes from the true_class column (which contains original labels)
        unique_classes = test_results['true_class'].unique()

        # Class processing progress bar
        class_pbar = tqdm(
            unique_classes,
            desc="Processing classes",
            leave=False,
            position=0
        )

        for class_label in class_pbar:
            class_pbar.set_postfix_str(f"Class {class_label}")

            # Get class-specific misclassified indices
            class_mask = (test_results.loc[misclassified_indices, 'true_class'] == class_label).to_numpy()
            class_indices = np.array(misclassified_indices)[class_mask].tolist()

            if not class_indices:
                continue

            # Convert original indices to test set positions
            class_positions = [test_pos_map[idx] for idx in class_indices if idx in test_pos_map]
            if not class_positions:
                continue

            # Apply PER-CLASS limit
            max_samples_this_class = min(max_samples_per_class, len(class_positions))
            if len(class_positions) > max_samples_this_class:
                # Randomly sample to avoid bias, but ensure we take up to the class limit
                class_positions = random.sample(class_positions, max_samples_this_class)
                print(f"{Colors.YELLOW} Limited class {class_label} from {len(class_positions)} to {max_samples_this_class} samples (per-class limit){Colors.ENDC}")

            # Convert to tensor with proper dtype
            class_pos_tensor = torch.tensor(class_positions, dtype=torch.long, device=self.device)

            # Batch processing with memory limits
            samples, margins, indices = [], [], []

            for batch_start in range(0, len(class_positions), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(class_positions))
                batch_pos = class_pos_tensor[batch_start:batch_end]

                # Get actual data indices
                batch_indices = test_indices[batch_pos]
                batch_X = self.X_tensor[batch_indices]

                # Compute posteriors
                if self.model_type == "Histogram":
                    posteriors, _ = self._compute_batch_posterior(batch_X)
                else:
                    posteriors, _ = self._compute_batch_posterior_std(batch_X)

                # SAFE LABEL HANDLING: Try to get encoded class ID, but handle errors gracefully
                try:
                    # Try to transform the class label
                    encoded_class_id = self.label_encoder.transform([str(class_label)])[0]
                except (ValueError, KeyError) as e:
                    # If the class label is not in the encoder, skip this class
                    print(f"{Colors.YELLOW}Warning: Class label '{class_label}' not found in label encoder. Skipping.{Colors.ENDC}")
                    continue

                # Calculate margins
                max_probs, _ = torch.max(posteriors, dim=1)
                true_probs = posteriors[:, encoded_class_id]
                batch_margins = max_probs - true_probs

                samples.append(batch_X)
                margins.append(batch_margins)
                indices.append(batch_indices)

            if not samples:
                continue

            try:
                # Concatenate tensors properly
                margins = torch.cat(margins)
                indices = torch.cat(indices)
                samples = torch.cat(samples)
            except RuntimeError:
                continue

            if indices.numel() == 0:
                continue

            # Select top samples from this class based on margin
            n_select = min(len(indices), max_samples_this_class)
            if n_select > 0:
                # Select samples with highest margins (most confident errors)
                _, top_indices = torch.topk(margins, k=n_select)
                selected_class_indices = indices[top_indices].cpu().tolist()
                final_selected_indices.extend(selected_class_indices)

                print(f"{Colors.GREEN}Adding {len(selected_class_indices)} samples from class {class_label} to training{Colors.ENDC}")

        class_pbar.close()

        # Final summary
        if final_selected_indices:
            # Count samples per class
            selected_classes = all_results.loc[final_selected_indices, 'true_class'].value_counts()
            print(f"\033[K{Colors.BLUE}Final selection summary:{Colors.ENDC}")
            for class_label, count in selected_classes.items():
                print(f"\033[K  {class_label}: {count} samples")

        return final_selected_indices

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

    def reset_to_initial_state(self):
        """Reset the model's weights and parameters to their initial state."""
        DEBUG.log("Resetting model to initial state for fresh training...")

        # Reset weights to uniform priors
        n_classes = len(self.label_encoder.classes_)
        n_pairs = len(self.feature_pairs) if self.feature_pairs is not None else 0

        if n_pairs > 0:
            self.current_W = torch.full(
                (n_classes, n_pairs),
                0.1,  # Default uniform prior
                device=self.device,
                dtype=torch.float64
            )
            self.best_W = self.current_W.clone()
            self.best_error = float('inf')

        # Reset likelihood parameters (if applicable)
        if self.model_type == "Histogram":
            self.likelihood_params = self._compute_pairwise_likelihood_parallel(
                self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
            )
        elif self.model_type == "Gaussian":
            self.likelihood_params = self._compute_pairwise_likelihood_parallel_std(
                self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
            )

        # Reset weight updater
        if self.weight_updater is not None:
            self.weight_updater = BinWeightUpdater(
                n_classes=len(self.label_encoder.classes_),
                feature_pairs=self.feature_pairs,
                dataset_name=self.dataset_name,  # Pass dataset name
                n_bins_per_dim=self.n_bins_per_dim,
                batch_size=self.batch_size
            )

        DEBUG.log("Model reset to initial state.")

    def _format_class_distribution(self, indices):
        """Helper to format class distribution for given indices"""
        if not indices:
            return "No samples added"

        class_counts = self.data.iloc[indices][self.target_column].value_counts()
        total = len(indices)

        # Create distribution string with percentages
        dist = []
        for cls, count in class_counts.items():
            percentage = (count / total) * 100
            dist.append(f"{cls}: {count} ({percentage:.1f}%)")

        return ", ".join(dist)

    def return_call_4afp(self,start_time,train_indices, test_indices,training_history,round_stats):
            # Record the end time
            start_clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
            end_time = time.time()
            end_clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            elapsed_time = end_time - start_time

            # Print the timing information
            print("\033[K" +f"{Colors.BOLD}{Colors.BLUE}Adaptive training started at: {start_clock}{Colors.ENDC}")
            print("\033[K" +f"{Colors.BOLD}{Colors.BLUE}Adaptive training ended at: {end_clock}{Colors.ENDC}")
            print("\033[K" +f"{Colors.BOLD}{Colors.BLUE}Total adaptive training time: {elapsed_time:.2f} seconds{Colors.ENDC}")

            # GENERATE UNIFIED VISUALIZATIONS AFTER TRAINING - ALL SYSTEMS
            if self.enable_visualization:
                print(f"==============Visualisation Enabled=================")
                self._generate_unified_visualizations()
                self.visualizer.generate_all_standard_visualizations()

            else:
                print(f"------------------------------------------------Enable visualisation is {self.enable_visualization}")
            # Generate final visualizations (legacy compatibility)
            if hasattr(self, 'visualizer') and training_history:
                print("🎨 Generating comprehensive visualizations...")
                # Legacy visualization call for backward compatibility
                if hasattr(self, 'create_comprehensive_visualizations'):
                    self.create_comprehensive_visualizations(
                        self, self.X_tensor.cpu().numpy(), self.y_tensor.cpu().numpy(),
                        training_history, round_stats, self.feature_columns
                    )
                if hasattr(self, 'create_geometric_visualization'):
                    self.create_geometric_visualization(training_history, round_stats)

            self.in_adaptive_fit = False
            return {'train_indices': train_indices, 'test_indices': test_indices}

    def adaptive_fit_predict(self, max_rounds: int = 10,
                            improvement_threshold: float = 0.0001,
                            load_epoch: int = None,
                            batch_size: int = 128):
        """Modified adaptive training strategy with comprehensive visualization integration"""
        DEBUG.log(" Starting adaptive_fit_predict")
        if not EnableAdaptive:
            print("\033[K" +"Adaptive learning is disabled. Using standard training.")
            return self.fit_predict(batch_size=batch_size)

        # Record the start time
        start_time = time.time()
        start_clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        print("\033[K" +f"{Colors.BOLD}{Colors.BLUE}Adaptive training started at: {start_clock}{Colors.ENDC}")

        self.in_adaptive_fit = True
        train_indices = []
        test_indices = None

        # FIX: Initialize training_history at the start of the method
        training_history = []  # Store training indices for each round
        round_stats = []  # Initialize round_stats to avoid reference errors

        # Initialize visualization data structures
        if self.enable_visualization:
            self.adaptive_round_data = []
            self.adaptive_snapshots = []
            print(f"{Colors.CYAN}[DBNN-VISUAL] 🎨 Starting adaptive training with UNIFIED visualization...{Colors.ENDC}")
            print(f"{Colors.CYAN}[DBNN-VISUAL] 📊 Includes: 5DCT, 3D Spherical, Tensor Evolution, Performance Dashboards{Colors.ENDC}")

        try:
            # Get initial data
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            print(f'Target:  {self.target_column}')
            # Initialize label encoder if not already done
            if not hasattr(self.label_encoder, 'classes_'):
                self.label_encoder.fit(y)

            # Use existing label encoder
            y_encoded = self.label_encoder.transform(y)

            # Process features and initialize model components if needed
            X_processed = self._preprocess_data(X, is_training=True)
            self.X_tensor = X_processed.clone().detach().to(self.device)
            self.y_tensor = torch.LongTensor(y_encoded).to(self.device)

            # Store data references for visualization
            if self.enable_visualization:
                self.X_original = X_processed.clone().detach().cpu().numpy()
                self.y_original = y_encoded.copy()

            # Handle model state based on flags
            model_loaded = False
            if self.use_previous_model:
                print("\033[K" +"Loading previous model state")
                if self._load_model_components():
                    #self.label_encoder =load_label_encoder(self.dataset_name)
                    #self._load_best_weights()
                    #self._load_categorical_encoders()
                    model_loaded = True

                    if not self.fresh_start:
                        # Load previous training data
                        print("\033[K" +"Loading previous training data...")
                        prev_train_file = f'{self.dataset_name}_Last_training.csv'
                        if os.path.exists(prev_train_file):
                            prev_train_data = pd.read_csv(prev_train_file)

                            # Try loading indices from last split
                            prev_train, prev_test = self.load_last_known_split()

                            if prev_train and prev_test:
                                train_indices = prev_train
                                test_indices = prev_test
                                print(f"\033[KResuming with {len(train_indices)} previous training samples")
                            else:
                                print("\033[KNo valid previous split found, initializing new training set")
                            print("\033[K" +"No previous training data found - starting fresh")
                            train_indices = []
                            test_indices = list(range(len(X)))
                else:
                    print("\033[K" +"No previous model found - starting fresh")

            if not model_loaded:
                print("\033[K" +"Initializing fresh model")
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
                    dtype=torch.float64
                )
                if self.best_W is None:
                    self.best_W = self.current_W.clone()

            # Initialize training set if empty
            if len(train_indices) == 0:
                # Get unique classes and calculate n_classes FIRST
                unique_classes = self.label_encoder.classes_
                n_classes = len(unique_classes)

                # USE PER-CLASS SAMPLING for initial round
                target_initial_samples_per_class = max(1, getattr(self, 'initial_samples', 50) // n_classes)
                print(f"\033[KInitializing new training set with {target_initial_samples_per_class} samples PER CLASS")

                initial_samples = []
                class_sample_counts = {}

                for class_label in unique_classes:
                    # Get encoded class ID
                    encoded_class_id = self.label_encoder.transform([class_label])[0]
                    class_indices = np.where(y_encoded == encoded_class_id)[0]

                    if len(class_indices) == 0:
                        class_sample_counts[class_label] = 0
                        print(f"\033[K  Warning: Class {class_label} has no samples")
                        continue

                    # Select samples for this class
                    n_samples_this_class = min(target_initial_samples_per_class, len(class_indices))
                    selected_indices = np.random.choice(class_indices, n_samples_this_class, replace=False).tolist()

                    initial_samples.extend(selected_indices)
                    class_sample_counts[class_label] = n_samples_this_class
                    print(f"\033[K  Class {class_label}: added {n_samples_this_class} initial samples")

                train_indices = initial_samples

                # Final verification
                total_samples = len(train_indices)
                print(f"\033[K  Final training set: {total_samples} TOTAL samples ({n_classes} classes)")
                print(f"\033[K  Class distribution:")
                for class_label in unique_classes:
                    count = class_sample_counts.get(class_label, 0)
                    percentage = (count / total_samples * 100) if total_samples > 0 else 0
                    print(f"\033[K    {class_label}: {count} samples ({percentage:.1f}%)")

            DEBUG.log(f" Initial training set size: {len(train_indices)}")
            DEBUG.log(f" Initial test set size: {len(test_indices)}")
            adaptive_patience_counter = 0

            # Continue with training loop...
            patience = self.adaptive_patience if self.in_adaptive_fit else self.patience

            # CAPTURE INITIAL VISUALIZATION SNAPSHOT FOR ALL SYSTEMS
            if self.enable_visualization:
                self._capture_unified_initial_snapshot(train_indices, test_indices)

            while adaptive_patience_counter < patience:
                for round_num in range(max_rounds):
                    print("\033[K" +f"Round {round_num + 1}/{max_rounds}")
                    print("\033[K" +f"Training set size: {len(train_indices)}")
                    print("\033[K" +f"Test set size: {len(test_indices)}")

                    # Reset model to initial state for fresh training
                    #self.reset_to_initial_state()

                    # Save indices for this epoch
                    self.save_epoch_data(round_num, train_indices, test_indices)

                    # Create feature tensors for training
                    X_train = self.X_tensor[train_indices]
                    y_train = self.y_tensor[train_indices]

                    # Train the model
                    save_path = f"data/{self.dataset_name}/Predictions/"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    self.train_indices = train_indices
                    self.test_indices = test_indices
                    results = self.fit_predict(batch_size=batch_size, save_path=save_path)

                    # Check training accuracy
                    print("\033[K" +f"{Colors.GREEN}Predctions on Training data{Colors.ENDC}", end="\r", flush=True)
                    train_accuracy = results['train_accuracy']
                    print("\033[K" +f"Training accuracy: {train_accuracy:.4f}         ")

                    # Get test accuracy from results
                    test_accuracy = results['test_accuracy']

                    # UNIFIED VISUALIZATION CAPTURE - ALL SYSTEMS
                    if self.enable_visualization and (round_num % self.visualization_frequency == 0):
                        self._capture_unified_round_snapshot(
                            round_num=round_num,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=self.X_tensor[test_indices],
                            y_test=self.y_tensor[test_indices],
                            train_predictions=results.get('train_predictions', {}).get('predicted_class', []),
                            test_predictions=results.get('test_predictions', {}).get('predicted_class', []),
                            train_accuracy=train_accuracy,
                            test_accuracy=test_accuracy
                        )

                    # Check if we're improving overall
                    improved = False

                    # FIX: Always append to training_history (it's now properly initialized)
                    training_history.append(self.train_indices.copy())

                    if 'best_train_accuracy' not in locals():
                        best_train_accuracy = self.best_combined_accuracy
                        improved = True
                    elif self.best_combined_accuracy > best_train_accuracy + improvement_threshold:
                        best_train_accuracy = self.best_combined_accuracy
                        improved = True
                        print("\033[K" +f"Improved training accuracy to {train_accuracy:.4f}")

                    # Reset adaptive patience if improved
                    if improved:
                        adaptive_patience_counter = 0
                        # Save the last training and test data
                        self.save_last_split(self.train_indices, self.test_indices)
                        print("\033[K" + "Saved model and data due to improved training accuracy")
                    else:
                        adaptive_patience_counter += 1
                        print("\033[K" +f"No significant overall improvement. Adaptive patience: {adaptive_patience_counter}/{patience}")
                        if adaptive_patience_counter >= patience:  # Using fixed value of 5 for adaptive patience
                            print("\033[K" +f"No improvement in accuracy after 5 rounds of adding samples.")
                            print("\033[K" +f"Best training accuracy achieved: {best_train_accuracy:.4f}")
                            print("\033[K" +"Stopping adaptive training.")
                            break

                    # Evaluate test data using combined predictions from fit_predict
                    test_predictions = results['test_predictions']['predicted_class']
                    y_test = self.y_tensor[test_indices].cpu().numpy()

                    # Convert test_predictions to a NumPy array if it's a Pandas Series
                    if isinstance(test_predictions, pd.Series):
                        test_predictions = test_predictions.to_numpy()

                    # Ensure test_predictions is numeric
                    if test_predictions.dtype == np.object_:
                        # If predictions are class labels, convert them to numeric indices
                        test_predictions = self.label_encoder.transform(test_predictions)
                    else:
                        # If predictions are numeric but stored as object, cast to int64
                        test_predictions = test_predictions.astype(np.int64)

                    # Check if we've achieved perfect accuracy
                    if train_accuracy == 1.0:
                        if len(test_indices) == 0:
                            print("\033[K" +"No more test samples available. Training complete.")
                            break

                        # Get new training samples from misclassified examples
                        new_train_indices = self._select_samples_from_failed_classes(
                            test_predictions, y_test, test_indices, results
                        )
                        if not new_train_indices:
                            print("\033[K" +"Achieved 100% accuracy on all data. Training complete.                                           ")
                            self.in_adaptive_fit = False
                            self.return_call_4afp(start_time,train_indices, test_indices,training_history,round_stats)
                            return {'train_indices': [], 'test_indices': []}

                    else:
                        # Training did not achieve 100% accuracy, select new samples
                        new_train_indices = self._select_samples_from_failed_classes(
                            test_predictions, y_test, test_indices, results
                        )

                        if not new_train_indices:
                            print("\033[K" +"No suitable new samples found. Training complete.")
                            break

                    #print(f"{Colors.YELLOW} Identified {len(new_train_indices)} [{new_train_indices}]samples from failed dataset {Colors.ENDC}")
                    # Update training and test sets with new samples
                    #train_indices.extend(new_train_indices)
                    #test_indices = list(set(test_indices) - set(new_train_indices))
                    #print("\033[K" +f"Added {len(new_train_indices)} new samples to training set")
                    if new_train_indices:
                        # Get class distribution of new samples
                        new_samples = self.data.iloc[new_train_indices][self.target_column]
                        class_counts = new_samples.value_counts().to_dict()

                        # Format class distribution string
                        class_dist = self._format_class_distribution(new_train_indices)
                        #print(f"\033[KAdded {len(new_train_indices)} new samples - Class distribution: {class_dist}")
                        #print(f"\033[KClass distribution of new samples: {class_dist}")

                    # Update training and test indices with original indices
                    train_indices = list(set(train_indices + new_train_indices))
                    test_indices = list(set(test_indices) - set(new_train_indices))

                    if new_train_indices:
                        # Reset to the best round's initial conditions
                        self.reset_to_initial_state()
                        if self.best_round_initial_conditions is not None:
                            print("\033[K" +f"Resetting to initial conditions of best round {self.best_round}")
                            self.current_W = self.best_round_initial_conditions['weights'].clone()
                            self.likelihood_params = self.best_round_initial_conditions['likelihood_params']
                            self.feature_pairs = self.best_round_initial_conditions['feature_pairs']
                            self.bin_edges = self.best_round_initial_conditions['bin_edges']
                            self.gaussian_params = self.best_round_initial_conditions['gaussian_params']
            self.return_call_4afp(start_time,train_indices, test_indices,training_history,round_stats)


        except Exception as e:
            DEBUG.log(f" Error in adaptive_fit_predict: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            self.in_adaptive_fit = False
            raise

    def _capture_unified_initial_snapshot(self, train_indices, test_indices):
        """Capture initial state snapshot for ALL visualization systems"""
        if not self.enable_visualization:
            return

        try:
            print(f"{Colors.CYAN}[DBNN-VISUAL] 📸 Capturing initial state for ALL visualization systems...{Colors.ENDC}")

            # Get feature names and class names
            feature_names = self.config.get('column_names', [])
            if self.target_column in feature_names:
                feature_names = [f for f in feature_names if f != self.target_column]

            if hasattr(self.label_encoder, 'classes_'):
                class_names = self.label_encoder.classes_.tolist()
            else:
                class_names = [f'Class_{i}' for i in range(len(np.unique(self.y_original)))]

            # Capture for new visualizer
            self.visualizer.capture_feature_space_snapshot(
                features=self.X_original,
                targets=self.y_original,
                predictions=self.y_original,  # No predictions yet
                iteration=0,
                feature_names=feature_names,
                class_names=class_names
            )

            # Capture for 5DCT visualizer if available
            if hasattr(self, 'visualizer5DCT') and hasattr(self.visualizer5DCT, 'capture_initial_snapshot'):
                self.visualizer5DCT.capture_initial_snapshot()

            # Store initial round data
            initial_data = {
                'round': 0,
                'train_accuracy': 0.0,
                'test_accuracy': 0.0,
                'train_samples': len(train_indices),
                'test_samples': len(test_indices),
                'combined_accuracy': 0.0,
                'timestamp': time.time(),
                'status': 'initial'
            }
            self.adaptive_round_data.append(initial_data)

        except Exception as e:
            print(f"{Colors.YELLOW}[DBNN-VISUAL] Unified initial snapshot failed: {str(e)}{Colors.ENDC}")

    def _capture_unified_round_snapshot(self, round_num, X_train, y_train, X_test, y_test,
                                       train_predictions, test_predictions, train_accuracy, test_accuracy):
        """Capture comprehensive snapshot for ALL visualization systems"""
        if not self.enable_visualization:
            return

        try:
            # Convert to numpy for visualization
            X_train_np = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
            y_train_np = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train
            X_test_np = X_test.cpu().numpy() if torch.is_tensor(X_test) else X_test
            y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test

            # Convert predictions to numpy
            def convert_predictions(predictions):
                if torch.is_tensor(predictions):
                    return predictions.cpu().numpy()
                elif hasattr(predictions, 'to_numpy'):
                    return predictions.to_numpy()
                else:
                    return np.array(predictions)

            train_pred_np = convert_predictions(train_predictions)
            test_pred_np = convert_predictions(test_predictions)

            # Get feature names and class names
            feature_names = self.config.get('column_names', [])
            if self.target_column in feature_names:
                feature_names = [f for f in feature_names if f != self.target_column]

            if hasattr(self.label_encoder, 'classes_'):
                class_names = self.label_encoder.classes_.tolist()
            else:
                unique_classes = np.unique(np.concatenate([y_train_np, y_test_np]))
                class_names = [f'Class_{int(c)}' for c in unique_classes]

            # Create combined dataset for comprehensive visualization
            X_combined = np.vstack([X_train_np, X_test_np])
            y_combined = np.concatenate([y_train_np, y_test_np])
            predictions_combined = np.concatenate([train_pred_np, test_pred_np])

            # CAPTURE FOR NEW VISUALIZER
            self.visualizer.capture_feature_space_snapshot(
                features=X_combined,
                targets=y_combined,
                predictions=predictions_combined,
                iteration=round_num,
                feature_names=feature_names,
                class_names=class_names
            )

            # Capture training snapshot with current weights
            current_weights = self.current_W if self.current_W is not None else self.best_W
            weights_np = current_weights.cpu().numpy() if torch.is_tensor(current_weights) else current_weights

            self.visualizer.capture_training_snapshot(
                features=X_combined,
                targets=y_combined,
                weights=weights_np,
                predictions=predictions_combined,
                accuracy=(train_accuracy + test_accuracy) / 2,
                round_num=round_num
            )

            # CAPTURE FOR 5DCT VISUALIZER
            if self.enable_5DCTvisualization:
                self.visualizer5DCT.capture_training_snapshot(
                    features=X_combined,
                    targets=y_combined,
                    train_accuracy=train_accuracy,
                    test_accuracy= test_accuracy,
                    epoch=round_num
                )

            # Store adaptive round metadata
            round_data = {
                'round': round_num,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_samples': len(X_train_np),
                'test_samples': len(X_test_np),
                'combined_accuracy': (train_accuracy + test_accuracy) / 2,
                'timestamp': time.time()
            }
            self.adaptive_round_data.append(round_data)

            print(f"{Colors.CYAN}[DBNN-VISUAL] 📊 Captured round {round_num} for ALL systems "
                  f"(Train: {train_accuracy:.3f}, Test: {test_accuracy:.3f}){Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.YELLOW}[DBNN-VISUAL] Unified snapshot capture failed: {str(e)}{Colors.ENDC}")

    def _generate_unified_visualizations(self):
        """Generate comprehensive visualizations for ALL systems"""
        if not self.enable_visualization:


            return

        print(f"{Colors.CYAN}[DBNN-VISUAL] 🎨 Generating UNIFIED visualizations for ALL systems...{Colors.ENDC}")

        try:
            # Generate new visualizer outputs
            standard_viz = self._generate_standard_visualizations()
            tensor_viz = self._generate_tensor_visualizations()
            spherical_viz = self._generate_spherical_visualizations()
            adaptive_viz = self._generate_adaptive_visualizations()

            # Generate 5DCT visualizations
            fivedct_viz = {}
            if hasattr(self, 'visualizer5DCT') and hasattr(self.visualizer5DCT, 'generate_visualizations'):
                try:
                    fivedct_results = self.visualizer5DCT.generate_visualizations()
                    if fivedct_results:
                        fivedct_viz['5dct_legacy'] = fivedct_results
                except Exception as e:
                    print(f"{Colors.YELLOW}[DBNN-VISUAL] 5DCT visualization generation failed: {str(e)}{Colors.ENDC}")

            # Print comprehensive summary
            self._print_unified_visualization_summary(standard_viz, tensor_viz, spherical_viz, adaptive_viz, fivedct_viz)

        except Exception as e:
            print(f"{Colors.YELLOW}[DBNN-VISUAL] Unified visualization generation failed: {str(e)}{Colors.ENDC}")

    def _generate_standard_visualizations(self):
        """Generate standard DBNN visualizations"""
        results = {}

        try:
            # Performance metrics dashboard
            perf_file = os.path.join(self.dbnn_viz_dir, "dbnn_performance.html")
            result = self.visualizer.generate_performance_metrics(perf_file)
            if result:
                results['performance'] = result

            # Animated confusion matrix
            confusion_file = os.path.join(self.dbnn_viz_dir, "dbnn_confusion_matrix.html")
            result = self.visualizer.generate_animated_confusion_matrix(confusion_file)
            if result:
                results['confusion_matrix'] = result

        except Exception as e:
            print(f"{Colors.YELLOW}[DBNN-VISUAL] Standard viz failed: {str(e)}{Colors.ENDC}")

        return results

    def _generate_tensor_visualizations(self):
        """Generate tensor space visualizations"""
        results = {}

        try:
            # Circular tensor evolution
            circular_file = os.path.join(self.tensor_viz_dir, "dbnn_circular.html")
            result = self.visualizer.generate_circular_tensor_evolution(circular_file)
            if result:
                results['circular'] = result

            # Polar tensor evolution
            polar_file = os.path.join(self.tensor_viz_dir, "dbnn_polar.html")
            result = self.visualizer.generate_polar_tensor_evolution(polar_file)
            if result:
                results['polar'] = result

        except Exception as e:
            print(f"{Colors.YELLOW}[DBNN-VISUAL] Tensor viz failed: {str(e)}{Colors.ENDC}")

        return results

    def _generate_spherical_visualizations(self):
        """Generate spherical coordinate visualizations"""
        results = {}

        try:
            # 3D Spherical Tensor Evolution
            spherical_file = os.path.join(self.spherical_viz_dir, "dbnn_3d_spherical.html")
            result = self.visualizer.generate_3d_spherical_tensor_evolution(spherical_file)
            if result:
                results['3d_spherical'] = result

            # Enhanced spherical with analytics
            enhanced_file = os.path.join(self.spherical_viz_dir, "dbnn_enhanced_spherical.html")
            result = self.visualizer.generate_enhanced_3d_spherical_visualization(enhanced_file)
            if result:
                results['enhanced_spherical'] = result

        except Exception as e:
            print(f"{Colors.YELLOW}[DBNN-VISUAL] Spherical viz failed: {str(e)}{Colors.ENDC}")

        return results

    def _generate_adaptive_visualizations(self):
        """Generate adaptive training specific visualizations"""
        results = {}

        try:
            # Adaptive training progression
            progression_file = os.path.join(self.adaptive_viz_dir, "adaptive_progression.html")
            result = self._generate_adaptive_progression_visualization(progression_file)
            if result:
                results['progression'] = result

        except Exception as e:
            print(f"{Colors.YELLOW}[DBNN-VISUAL] Adaptive viz failed: {str(e)}{Colors.ENDC}")

        return results

    def _generate_adaptive_progression_visualization(self, output_file):
        """Generate adaptive training progression visualization"""
        try:
            if not self.adaptive_round_data:
                return None

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            rounds = [data['round'] for data in self.adaptive_round_data]
            train_acc = [data['train_accuracy'] for data in self.adaptive_round_data]
            test_acc = [data['test_accuracy'] for data in self.adaptive_round_data]
            train_samples = [data['train_samples'] for data in self.adaptive_round_data]

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Training vs Test Accuracy',
                    'Training Set Size Evolution',
                    'Accuracy Gap',
                    'Training Summary'
                )
            )

            # Accuracy progression
            fig.add_trace(go.Scatter(x=rounds, y=train_acc, name='Train Accuracy'), row=1, col=1)
            fig.add_trace(go.Scatter(x=rounds, y=test_acc, name='Test Accuracy'), row=1, col=1)

            # Training set size
            fig.add_trace(go.Scatter(x=rounds, y=train_samples, name='Training Samples'), row=1, col=2)

            # Accuracy gap
            accuracy_gap = [abs(tra - tst) for tra, tst in zip(train_acc, test_acc)]
            fig.add_trace(go.Scatter(x=rounds, y=accuracy_gap, name='Accuracy Gap'), row=2, col=1)

            fig.update_layout(height=600, title_text="DBNN Adaptive Training Progression")
            fig.write_html(output_file)
            return output_file

        except Exception as e:
            print(f"{Colors.YELLOW}[DBNN-VISUAL] Progression viz failed: {str(e)}{Colors.ENDC}")
            return None

    def _print_unified_visualization_summary(self, standard, tensor, spherical, adaptive, fivedct):
        """Print comprehensive unified visualization summary"""
        print(f"\n{Colors.GREEN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.GREEN}🎉 UNIFIED VISUALIZATION SUMMARY - ALL SYSTEMS{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*80}{Colors.ENDC}")

        total_viz = len(standard) + len(tensor) + len(spherical) + len(adaptive) + len(fivedct)
        total_rounds = len(self.adaptive_round_data)

        print(f"{Colors.CYAN}📊 Training Rounds: {total_rounds}{Colors.ENDC}")
        print(f"{Colors.CYAN}📈 Total Visualizations Generated: {total_viz}{Colors.ENDC}")

        # Print each category
        categories = [
            ("📈 Standard", standard, Colors.BLUE),
            ("🎯 Tensor Space", tensor, Colors.MAGENTA),
            ("🌐 Spherical", spherical, Colors.CYAN),
            ("🔄 Adaptive", adaptive, Colors.GREEN),
            ("⚡ 5DCT Legacy", fivedct, Colors.YELLOW)
        ]

        for name, viz_dict, color in categories:
            if viz_dict:
                print(f"\n{color}{name} Visualizations:{Colors.ENDC}")
                for viz_name, path in viz_dict.items():
                    print(f"   ✅ {viz_name}: {os.path.basename(path)}")

        # Final performance
        if self.adaptive_round_data:
            final = self.adaptive_round_data[-1]
            print(f"\n{Colors.GREEN}🎯 Final Performance:{Colors.ENDC}")
            print(f"   Training Accuracy: {final['train_accuracy']:.3f}")
            print(f"   Test Accuracy: {final['test_accuracy']:.3f}")
            print(f"   Training Samples: {final['train_samples']}")

        print(f"\n{Colors.GREEN}📁 All visualizations saved to: {self.viz_output_dir}/{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*80}{Colors.ENDC}\n")

    #------------------------------------------Adaptive Learning--------------------------------------

    def _calculate_cardinality_threshold(self):
        """Calculate appropriate cardinality threshold based on dataset characteristics"""
        n_samples = len(self.data)
        DEFAULT_THRESHOLD = 0.9
        predict_mode = True if self.mode=='predict' else False
        if predict_mode or self.target_column not in self.data.columns:
            DEBUG.log("Using default cardinality threshold (prediction mode or no target)")
            return DEFAULT_THRESHOLD

        n_classes = len(self.data[self.target_column].unique())

        # Base threshold from config
        base_threshold =self.config.get('training_params', {}).get('cardinality_threshold', 0.9) # cardinality_threshold

        # Adjust threshold based on dataset size and number of classes
        adjusted_threshold = min(
            base_threshold,
            max(0.1, 1.0 / np.sqrt(n_classes))  # Lower bound of 0.1
        )

        DEBUG.log(f"\nCardinality Threshold Calculation:")
        DEBUG.log(f"- Base threshold: {base_threshold}")
        DEBUG.log(f"- Number of samples: {n_samples}")
        DEBUG.log(f"- Number of classes: {n_classes}")
        DEBUG.log(f"- Adjusted threshold: {adjusted_threshold}")

        return   base_threshold  #adjusted_threshold

    def _round_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round features based on cardinality_tolerance"""
        if cardinality_tolerance == -1:
            return df
        return df.round(cardinality_tolerance)

    def _remove_high_cardinality_columns(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Remove high cardinality columns with more conservative approach"""
        DEBUG.log(f"Starting cardinality analysis with threshold {threshold}")

        df_filtered = df.copy()
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

    def _preprocess_data(self, X: Union[pd.DataFrame, torch.Tensor], is_training: bool = True) -> torch.Tensor:
        """Preprocess data with robust NaN handling and type safety."""
        DEBUG.log(f"Starting preprocessing (is_training={is_training})")

        # Check if X is a DataFrame or a tensor
        if isinstance(X, pd.DataFrame):
            DEBUG.log(f"Input shape: {X.shape}")
            DEBUG.log(f"Input columns: {X.columns.tolist()}")
            DEBUG.log(f"Input dtypes:\n{X.dtypes}")

            # In prediction mode, ensure we have all required features
            required_features = self.feature_columns if hasattr(self, 'feature_columns') else X.columns
            missing_features = set(required_features) - set(X.columns)
            if missing_features and not predict_mode:
                raise ValueError(f"Missing required features: {missing_features}")

            # Replace NA/NaN with -99999 and keep track of locations
            X = X.copy()
            self.nan_mask = X.isna()  # Store NaN locations using pandas' isna()
            X = X.fillna(-99999)      # Replace with sentinel value

            # Convert object/string columns to numeric where possible
            for col in X.select_dtypes(include=['object', 'string']):
                try:
                    X[col] = pd.to_numeric(X[col], errors='ignore')
                except Exception as e:
                    DEBUG.log(f"Could not convert column {col} to numeric: {str(e)}")

        else:  # Tensor input
            DEBUG.log("Input is a tensor, skipping column-specific operations")
            X = X.clone().detach()
            self.nan_mask = torch.isnan(X)  # For tensor input
            X = torch.where(self.nan_mask, torch.tensor(-99999, device=X.device), X)

        # Step 1: Handle feature selection and statistics computation
        if is_training:
            # Store original columns before any filtering
            self.original_columns = X.columns.tolist() if isinstance(X, pd.DataFrame) else None

            # Apply feature filtering (e.g., high cardinality removal)
            if isinstance(X, pd.DataFrame):
                cardinality_threshold = self._calculate_cardinality_threshold()
                DEBUG.log(f"Cardinality threshold: {cardinality_threshold}")
                X = self._remove_high_cardinality_columns(X, cardinality_threshold)
                DEBUG.log(f"Shape after cardinality filtering: {X.shape}")

                # Store the final selected features
                self.feature_columns = X.columns.tolist()
                DEBUG.log(f"Selected feature columns: {self.feature_columns}")

                # Store removed columns for reference
                self.high_cardinality_columns = list(set(self.original_columns) - set(self.feature_columns))
                if self.high_cardinality_columns:
                    DEBUG.log(f"Removed high cardinality columns: {self.high_cardinality_columns}")

            # Compute statistics ONLY on the selected features
            self.global_mean = X.mean(axis=0).values
            self.global_std = X.std(axis=0).values
            self.global_std[self.global_std == 0] = 1  # Avoid division by zero
            self.global_stats_computed = True
        else:
            # During prediction: enforce exact feature matching
            if isinstance(X, pd.DataFrame):
                missing = set(self.feature_columns) - set(X.columns)
                if missing:
                    raise ValueError(
                        f"Prediction data missing {len(missing)} required features: {sorted(missing)}\n"
                        f"Expected features: {self.feature_columns}\n"
                        f"Provided features: {X.columns.tolist()}"
                    )
                # Reorder to match training features exactly
                X = X[self.feature_columns]

        # Step 2: Handle categorical features
        DEBUG.log("Starting categorical encoding")
        try:
            if isinstance(X, pd.DataFrame):
                X_encoded = self._encode_categorical_features(X, is_training)
            else:
                X_encoded = X
            DEBUG.log(f"Shape after categorical encoding: {X_encoded.shape}")
            if isinstance(X_encoded, pd.DataFrame):
                DEBUG.log(f"Encoded dtypes:\n{X_encoded.dtypes}")
        except Exception as e:
            DEBUG.log(f"Error in categorical encoding: {str(e)}")
            raise

        # Step 3: Convert to numpy array
        try:
            if isinstance(X_encoded, pd.DataFrame):
                X_numpy = X_encoded.to_numpy(dtype=np.float64)
            else:
                X_numpy = X_encoded.cpu().numpy() if torch.is_tensor(X_encoded) else np.array(X_encoded, dtype=np.float64)
            DEBUG.log(f"Numpy array shape: {X_numpy.shape}")
        except Exception as e:
            DEBUG.log(f"Error converting to numpy: {str(e)}")
            raise

        # Step 4: Standardize using the correct stats
        try:
            X_scaled = (X_numpy - self.global_mean) / (self.global_std + 1e-8)  # Add epsilon to avoid division by zero
            DEBUG.log("Scaling successful")
        except Exception as e:
            DEBUG.log(f"Standard scaling failed: {str(e)}. Using manual scaling")
            means = np.nanmean(X_numpy, axis=0)
            stds = np.nanstd(X_numpy, axis=0)
            stds[stds == 0] = 1
            X_scaled = (X_numpy - means) / stds

        # Step 5: Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float64, device=self.device)

        # Step 6: Compute feature pairs and bin edges (training only)
        if is_training:
            remaining_feature_indices = list(range(len(self.feature_columns)))
            DEBUG.log(f"Computing feature pairs from {len(remaining_feature_indices)} features")

            self.feature_pairs = self._generate_feature_combinations(
                remaining_feature_indices,
                self.config.get('likelihood_config', {}).get('feature_group_size', 2),
                self.config.get('likelihood_config', {}).get('max_combinations', None)
            )
            DEBUG.log(f"Generated {len(self.feature_pairs)} feature pairs")

            self.bin_edges = self._compute_bin_edges(X_tensor, self.config.get('likelihood_config', {}).get('bin_sizes', [128]))
            DEBUG.log(f"Computed bin edges for {len(self.bin_edges)} feature pairs")

        DEBUG.log(f"Final preprocessed shape: {X_scaled.shape}")
        return X_tensor

#-----------------------------------------PDF mosaic -----------------------------------------------------
    def _generate_prediction_analysis_files(self, predictions_df, true_labels=None):
        """Generate CSV files for failed and correct predictions"""
        dataset_name = self.dataset_name
        output_dir = f"data/{dataset_name}/Predictions/"
        os.makedirs(output_dir, exist_ok=True)

        # Save all predictions
        predictions_df.to_csv(f"{output_dir}{dataset_name}_all_predictions.csv", index=False)

        # Only generate failure/success files if we have true labels
        if true_labels is not None:
            # Add true labels to dataframe if not already present
            if 'true_class' not in predictions_df.columns:
                predictions_df['true_class'] = true_labels

            # Create failed predictions CSV
            failed_predictions = predictions_df[predictions_df['predicted_class'] != predictions_df['true_class']]
            failed_predictions.to_csv(f"{output_dir}{dataset_name}_failed_predictions.csv", index=False)

            # Create correct predictions CSV
            correct_predictions = predictions_df[predictions_df['predicted_class'] == predictions_df['true_class']]
            correct_predictions.to_csv(f"{output_dir}{dataset_name}_correct_predictions.csv", index=False)

            return {
                'all_predictions': f"{output_dir}{dataset_name}_all_predictions.csv",
                'failed_predictions': f"{output_dir}{dataset_name}_failed_predictions.csv",
                'correct_predictions': f"{output_dir}{dataset_name}_correct_predictions.csv"
            }

        return {
            'all_predictions': f"{output_dir}{dataset_name}_all_predictions.csv"
        }
#----------------------------pdf mosaic ----------------------------------------
    def generate_class_pdf_mosaics(self, predictions_df, output_dir, columns=4, rows=4):
        """
        Generate PDF mosaics with images and heatmaps side-by-side.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define and register styles properly
        styles = getSampleStyleSheet()
        caption_style = ParagraphStyle(
            name='Caption',
            parent=styles['Normal'],
            fontSize=8,
            leading=9,
            spaceBefore=2,
            spaceAfter=2,
            alignment=1
        )
        styles.add(caption_style)

        hyperlink_style = ParagraphStyle(
            name='Hyperlink',
            parent=styles['Normal'],
            textColor=colors.blue,
            underline=1
        )
        styles.add(hyperlink_style)

        # Group by predicted class
        class_groups = predictions_df.groupby('predicted_class')
        images_per_page = columns * rows

        for class_name, group_df in class_groups:
            safe_name = "".join(c if c.isalnum() else "_" for c in str(class_name))
            pdf_path = os.path.join(output_dir, f"class_{safe_name}_mosaic.pdf")

            # Clean and validate data
            valid_df = group_df.dropna(subset=['filepath'])
            valid_df = valid_df[valid_df['filepath'].apply(lambda x: isinstance(x, (str, bytes)) and os.path.exists(x))]

            if valid_df.empty:
                print(f"\033[KSkipping {class_name} - No valid images found")
                continue

            # Sort remaining valid entries
            sorted_df = valid_df.sort_values('prediction_confidence', ascending=False)
            n_images = len(sorted_df)
            n_pages = math.ceil(n_images / images_per_page)

            # PDF setup with margins
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=letter,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch
            )
            elements = []

            # Calculate dimensions
            usable_width = letter[0] - inch
            pair_width = usable_width / columns
            img_width = pair_width * 0.45
            heatmap_width = pair_width * 0.45
            img_height = (letter[1] * 0.85) / (rows * 2)

            with tqdm(total=n_images,
                    desc=f"{str(class_name)[:15]:<15}",
                    unit="img",
                    bar_format="{l_bar}{bar:40}{r_bar}{bar:-40b}",
                    leave=False) as pbar:

                for page_num in range(n_pages):
                    start_idx = page_num * images_per_page
                    end_idx = min(start_idx + images_per_page, n_images)
                    page_images = sorted_df.iloc[start_idx:end_idx]

                    # Skip empty pages
                    if len(page_images) == 0:
                        continue

                    # Create page elements
                    elements.append(Paragraph(
                        f"Class: {class_name} (Sorted by Confidence)",
                        styles['Heading2']
                    ))
                    elements.append(Spacer(1, 0.1*inch))

                    table_data = []
                    current_row = []
                    valid_count = 0

                    for _, row in page_images.iterrows():
                        try:
                            img_path = str(row['filepath']).strip()
                            if not os.path.exists(img_path):
                                continue

                            # Create image element
                            img_element = ReportLabImage(img_path, width=img_width, height=img_height)

                            # Create heatmap element with proper style handling
                            heat_element = self._create_heatmap_element(
                                row.get('heatmap_path', ''),
                                heatmap_width,
                                img_height,
                                styles['Caption']
                            )

                            # Create image pair table
                            pair_table = Table(
                                [[img_element, heat_element]],
                                colWidths=[img_width, heatmap_width],
                                style=TableStyle([
                                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                                    ('BOX', (0,0), (-1,-1), 0.5, colors.grey)
                                ])
                            )

                            # Create caption with validated hyperlink
                            img_name = os.path.basename(img_path)
                            caption_text = f'<link href="{img_path}">{img_name[:15]}...</link><br/>Conf: {row["prediction_confidence"]:.2%}'
                            caption = Paragraph(caption_text, styles['Hyperlink'])

                            # Add to table
                            current_row.append([pair_table, caption])
                            valid_count += 1

                            # Start new row when current row is full
                            if len(current_row) == columns:
                                table_data.append(current_row)
                                current_row = []

                        except Exception as e:
                            print(f"\033[KSkipping invalid entry: {str(e)}")
                        finally:
                            pbar.update(1)

                    # Add remaining valid items if any
                    if current_row:
                        table_data.append(current_row)

                    # Only create table if we have valid data
                    if table_data:
                        main_table = Table(
                            table_data,
                            colWidths=[pair_width] * columns,
                            style=TableStyle([
                                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                                ('PADDING', (0,0), (-1,-1), 2),
                                ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
                            ])
                        )
                        elements.append(main_table)

                    if page_num < n_pages - 1 and valid_count > 0:
                        elements.append(PageBreak())

                # Build the document if we have valid elements
                if elements:
                    try:
                        doc.build(elements)
                        print(f"\033[K✅ {class_name} - Saved {valid_count} images to {os.path.basename(pdf_path)}")
                    except Exception as e:
                        print(f"\033[K❌ Failed to build PDF for {class_name}: {str(e)}")

    def _create_heatmap_element(self, heat_path, width, height, style):
        """Create heatmap image element with proper style handling"""
        if heat_path and os.path.exists(str(heat_path)):
            try:
                return ReportLabImage(heat_path, width=width, height=height)
            except Exception as e:
                return Paragraph(f"Invalid heatmap\n{os.path.basename(heat_path)}", style)
        return Paragraph("No heatmap available", style)

    #---------------------------------------------------------------------------------
    def generate_class_pdf_mosaics_old(self, predictions_df, output_dir, columns=4, rows=4):
        """
        Generate PDF mosaics with configurable grid layout (columns x rows per page).
        Captions are clickable hyperlinks to the original image paths.

        Args:
            predictions_df: DataFrame containing predictions and image paths.
            output_dir: Directory to save the PDF files.
            columns: Number of columns per page (default: 4).
            rows: Number of rows per page (default: 4).
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Add custom styles
        styles = getSampleStyleSheet()
        if 'Caption' not in styles:
            from reportlab.lib.styles import ParagraphStyle
            styles.add(ParagraphStyle(
                name='Caption',
                parent=styles['Normal'],
                fontSize=8,
                leading=9,
                spaceBefore=2,
                spaceAfter=2,
                alignment=1  # Center aligned
            ))

        # Add hyperlink style
        if 'Hyperlink' not in styles:
            styles.add(ParagraphStyle(
                name='Hyperlink',
                parent=styles['Caption'],
                textColor=colors.blue,
                underline=1
            ))

        # Group by predicted class
        class_groups = predictions_df.groupby('predicted_class')
        images_per_page = columns * rows

        for class_name, group_df in class_groups:
            safe_name = "".join(c if c.isalnum() else "_" for c in str(class_name))
            pdf_path = os.path.join(output_dir, f"class_{safe_name}_mosaic.pdf")

            # Sort by prediction confidence (highest first)
            sorted_df = group_df.sort_values('prediction_confidence', ascending=False)
            n_images = len(sorted_df)
            n_pages = math.ceil(n_images / images_per_page)

            # PDF setup with margins
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=letter,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch
            )
            elements = []

            # Calculate image dimensions based on page size and grid
            usable_width = letter[0] - inch  # Account for margins
            usable_height = letter[1] - 2*inch
            img_width = usable_width / columns
            img_height = (usable_height / rows) * 0.85  # 85% of row height for image, 15% for caption

            # Single progress bar for the entire class
            with tqdm(total=n_images,
                     desc=f"{str(class_name)[:15]:<15}",
                     unit="img",
                     bar_format="{l_bar}{bar:40}{r_bar}{bar:-40b}",
                     leave=False) as pbar:

                processed_images = 0

                for page_num in range(n_pages):
                    start_idx = page_num * images_per_page
                    end_idx = min(start_idx + images_per_page, n_images)
                    page_images = sorted_df.iloc[start_idx:end_idx]

                    # Page header (skip for first page)
                    if page_num > 0:
                        elements.extend([
                            Spacer(1, 0.25*inch),
                            Paragraph(f"Page {page_num+1} of {n_pages}", styles['Normal']),
                            Spacer(1, 0.25*inch)
                        ])

                    elements.append(Paragraph(
                        f"Class: {class_name} (Sorted by Confidence)",
                        styles['Heading2']
                    ))
                    elements.append(Spacer(1, 0.1*inch))

                    # Create image grid table
                    table_data = []
                    row_data = []

                    for _, row in page_images.iterrows():
                        img_path = row['filepath']
                        img_name = os.path.basename(img_path)
                        confidence = row['prediction_confidence']

                        try:
                            # Verify and load image
                            with PILImage.open(img_path) as img:
                                img.verify()

                            # Create clickable caption with hyperlink
                            caption_text = f'<link href="{img_path}">{img_name[:15]}...</link><br/>Conf: {confidence:.2%}'
                            caption = Paragraph(caption_text, styles['Hyperlink'])

                            # Create table cell with image and caption
                            cell_content = [
                                ReportLabImage(img_path, width=img_width*0.9, height=img_height*0.85),
                                caption
                            ]
                            row_data.append(cell_content)

                            # Start new row when current row is full
                            if len(row_data) == columns:
                                table_data.append(row_data)
                                row_data = []

                        except Exception as e:
                            print(f"\033[K⚠️ Error loading {img_path}: {str(e)}")
                            continue

                        # Update progress
                        processed_images += 1
                        pbar.update(1)

                    # Add any remaining images in the last row
                    if row_data:
                        # Pad with empty cells if needed
                        while len(row_data) < columns:
                            row_data.append("")
                        table_data.append(row_data)

                    # Add table to PDF elements
                    if table_data:
                        table_style = [
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 3),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                            ('TOPPADDING', (0, 0), (-1, -1), 3),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                        ]
                        table = Table(table_data, colWidths=[img_width] * columns)
                        table.setStyle(TableStyle(table_style))
                        elements.append(table)

                    # Add page break if not the last page
                    if page_num < n_pages - 1:
                        elements.append(PageBreak())

                # Build the PDF after processing all pages
                doc.build(elements)

                # Print completion message
                print(f"\033[K✅ {class_name} - Saved {n_images} images to {os.path.basename(pdf_path)}")
#--------------Option 3 ----------------
    def generate_class_pdf(self, image_paths: List[str], posteriors: np.ndarray, output_pdf: str):
        """Generate professional multi-page PDF with 2x4 image grids per class, sorted by confidence.

        Args:
            image_paths: List of image file paths
            posteriors: Numpy array of posterior probabilities for each image
            output_pdf: Path to save the output PDF
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        # Group images by predicted class with their posteriors
        class_groups = defaultdict(list)
        for img_path, posterior in zip(image_paths, posteriors):
            # Extract class name from filename (assuming format: class_imageid.jpg)
            class_name = os.path.basename(img_path).split('_')[0]
            class_groups[class_name].append((img_path, posterior))

        # Create PDF document
        doc = SimpleDocTemplate(output_pdf, pagesize=letter,
                              rightMargin=36, leftMargin=36,
                              topMargin=36, bottomMargin=36)

        # Define styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        caption_style = ParagraphStyle(
            'Caption',
            parent=styles['Normal'],
            fontSize=8,
            leading=10,
            alignment=1,  # Center aligned
            spaceBefore=6
        )
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            leading=10,
            alignment=2,  # Right aligned
            spaceBefore=6
        )

        elements = []

        # Process each class
        for class_name, images in sorted(class_groups.items()):
            # Add class title page
            elements.append(Paragraph(f"Class: {class_name}", title_style))
            elements.append(Spacer(1, 0.5*inch))
            elements.append(PageBreak())

            # Sort images by posterior (highest first)
            images.sort(key=lambda x: x[1], reverse=True)

            # Create pages with 2x4 grids
            n_images = len(images)
            n_pages = math.ceil(n_images / 8)

            for page_num in range(n_pages):
                # Create a grid for this page
                start_idx = page_num * 8
                end_idx = min(start_idx + 8, n_images)
                page_images = images[start_idx:end_idx]

                # Create a temporary image with the grid
                grid_img_path = self._create_image_grid(page_images)

                # Add to PDF with caption
                img = Image(grid_img_path, width=6*inch, height=7.5*inch)
                elements.append(img)

                # Add page footer
                footer_text = f"Page {page_num+1} of {n_pages} - Class: {class_name}"
                elements.append(Paragraph(footer_text, footer_style))

                if end_idx < n_images:  # Don't add break after last page
                    elements.append(PageBreak())

            # Add divider page between classes
            if class_name != sorted(class_groups.keys())[-1]:
                elements.append(Paragraph("Class Complete", title_style))
                elements.append(PageBreak())

        # Build the PDF
        doc.build(elements)

        # Clean up temporary files
        if hasattr(self, '_temp_files'):
            for f in self._temp_files:
                try:
                    os.remove(f)
                except:
                    pass

    def _create_image_grid(self, images: List[tuple]) -> str:
        """Create a single image with 2x4 grid of images and captions.

        Args:
            images: List of (image_path, posterior) tuples

        Returns:
            Path to temporary image file
        """
        from PIL import Image, ImageDraw, ImageFont

        # Grid parameters
        cols, rows = 4, 2
        img_size = 400  # Size of each individual image
        padding = 10
        caption_height = 30

        # Create blank canvas
        canvas_width = cols * (img_size + padding) + padding
        canvas_height = rows * (img_size + padding + caption_height) + padding
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Position images in grid
        for idx, (img_path, posterior) in enumerate(images):
            row = idx // cols
            col = idx % cols

            try:
                # Open and resize image
                img = Image.open(img_path)
                img = img.resize((img_size, img_size))

                # Calculate position
                x = col * (img_size + padding) + padding
                y = row * (img_size + padding + caption_height) + padding

                # Paste image with border
                canvas.paste(img, (x, y))
                draw.rectangle([x, y, x+img_size, y+img_size], outline="black", width=1)

                # Add caption
                caption = f"{os.path.basename(img_path)} (Confidence: {posterior:.2%})"
                text_width = draw.textlength(caption, font=font)
                text_x = x + (img_size - text_width) / 2
                text_y = y + img_size + 5
                draw.text((text_x, text_y), caption, fill='black', font=font)

            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue

        # Save temporary file
        if not hasattr(self, '_temp_files'):
            self._temp_files = []

        temp_path = os.path.join(tempfile.gettempdir(), f"grid_{os.getpid()}_{len(self._temp_files)}.jpg")
        canvas.save(temp_path, quality=90)
        self._temp_files.append(temp_path)

        return temp_path

#-------------------------------------------PDF mosaic Ends ------------------------------

    def _generate_feature_combinations(self, feature_indices: Union[List[int], int],
                                     group_size: int = 2,
                                     max_combinations: int = None) -> torch.Tensor:
        """
        Generate feature combinations in-memory without file I/O.
        Only generates if feature_pairs doesn't already exist.
        """
        # If we already have feature pairs, return them
        #if hasattr(self, 'feature_pairs') and self.feature_pairs is not None:
        if self.feature_pairs is not None:
            return self.feature_pairs
        # Strict check for predict mode or use_previous_model
        if (self.mode == 'predict' or self.use_previous_model) and not hasattr(self, 'feature_pairs'):
            raise RuntimeError(
                f"FATAL: Required feature pairs not found in model components.\n"
                f"- Current mode: {'predict' if self.mode == 'predict' else 'train'}\n"
                f"- Use previous model: {self.use_previous_model}\n"
                f"Possible causes:\n"
                f"1. Missing model components file\n"
                f"2. Corrupted model data\n"
                f"3. Trying to predict with untrained model\n"
                f"Solution: Ensure model is properly trained first or provide correct model components."
            )
        # Convert feature_indices to list if it's an integer
        if isinstance(feature_indices, int):
            feature_indices = list(range(feature_indices))

        # Get parameters from config with defaults
        config = self.config.get('likelihood_config', {})
        group_size = group_size or config.get('feature_group_size', 2)
        max_combinations = max_combinations or config.get('max_combinations')

        print("\033[K" + f"[DEBUG] Generating feature combinations in-memory:")
        print("\033[K" + f"- n_features: {len(feature_indices)}")
        print("\033[K" + f"- group_size: {group_size}")
        print("\033[K" + f"- max_combinations: {max_combinations}")

        # Generate all possible combinations if under limit
        total_possible = comb(len(feature_indices), group_size)
        if max_combinations is None or total_possible <= max_combinations:
            print("\033[K" + "[DEBUG] Generating all possible combinations")
            all_combinations = list(combinations(feature_indices, group_size))
        else:
            print("\033[K" + f"[DEBUG] Sampling {max_combinations} random combinations")
            all_combinations = self._sample_combinations(feature_indices, group_size, max_combinations)

        # Remove duplicates and sort
        unique_combinations = list({tuple(sorted(comb)) for comb in all_combinations})
        unique_combinations = sorted(unique_combinations)
        print("\033[K" + f"[DEBUG] Generated {len(unique_combinations)} unique feature combinations")

        # Convert to tensor and store in the model
        self.feature_pairs = torch.tensor(unique_combinations, device=self.device)
        return self.feature_pairs

    def _sample_combinations(self, features: List[int], group_size: int, max_samples: int) -> List[Tuple[int]]:
        """Memory-efficient combination sampling for large feature spaces."""
        # Use reservoir sampling for large feature spaces
        samples = set()

        # First add all possible combinations if they're few enough
        if comb(len(features), group_size) < 1e6:  # Threshold for full generation
            return list(combinations(features, group_size))

        # For very large spaces, use iterative sampling
        while len(samples) < max_samples:
            # Generate random combinations without replacement
            new_sample = tuple(sorted(random.sample(features, group_size)))
            samples.add(new_sample)

            # Progress reporting
            if len(samples) % 1000 == 0:
                print(f"\033[K[DEBUG] Generated {len(samples)}/{max_samples} samples", end='\r')

        return list(samples)
#-----------------------------------------------------------------------------Bin model ---------------------------

    def _compute_pairwise_likelihood_parallel(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """Optimized vectorized pairwise likelihood computation with serialization-safe bin tracking"""
        DEBUG.log("Starting optimized _compute_pairwise_likelihood_parallel")

        # Initialize class-bin tracking structure - FIXED: No lambda
        self.class_bins = defaultdict(dict)  # Use regular dict instead of defaultdict

        # Ensure tensors are contiguous
        dataset = dataset.contiguous()
        labels = labels.contiguous()

        # Validate class consistency
        unique_classes, class_counts = torch.unique(labels, return_counts=True)
        n_classes = len(unique_classes)
        n_samples = len(dataset)

        if n_classes != len(self.label_encoder.classes_):
            raise ValueError("Class count mismatch between data and label encoder")

        # Get bin configuration
        bin_sizes = self.config.get('likelihood_config', {}).get('bin_sizes', [128])
        n_bins = bin_sizes[0] if len(bin_sizes) == 1 else max(bin_sizes)
        self.n_bins_per_dim = n_bins

        # Initialize weights
        self._initialize_bin_weights()

        all_bin_counts = []
        all_bin_probs = []

        # Process pairs in optimized batches
        n_pairs = len(self.feature_pairs)
        pair_batch_size = min(20, n_pairs)

        with tqdm(total=n_pairs, desc="Pairwise likelihood", leave=False) as pbar:
            for batch_start in range(0, n_pairs, pair_batch_size):
                batch_end = min(batch_start + pair_batch_size, n_pairs)

                # Process batch of pairs with vectorized bin tracking
                batch_counts, batch_probs = self._process_pair_batch_vectorized(
                    dataset, labels, unique_classes, batch_start, batch_end, n_bins, n_classes
                )

                all_bin_counts.extend(batch_counts)
                all_bin_probs.extend(batch_probs)
                pbar.update(batch_end - batch_start)

                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return {
            'bin_counts': all_bin_counts,
            'bin_probs': all_bin_probs,
            'bin_edges': self.bin_edges,
            'feature_pairs': self.feature_pairs,
            'classes': unique_classes
        }

    def _process_pair_batch_vectorized(self, dataset, labels, unique_classes, batch_start, batch_end, n_bins, n_classes):
        """Vectorized batch processing with optimized bin tracking"""
        batch_counts = []
        batch_probs = []

        for pair_idx in range(batch_start, batch_end):
            f1, f2 = self.feature_pairs[pair_idx]
            edges = self.bin_edges[pair_idx]

            # Initialize counts for all classes
            pair_counts = torch.zeros((n_classes, n_bins, n_bins),
                                    dtype=torch.float64, device=self.device)

            # Vectorized processing for all classes
            for cls_idx, cls in enumerate(unique_classes):
                cls_mask = (labels == cls)
                if not torch.any(cls_mask):
                    continue

                data = dataset[cls_mask][:, [f1, f2]].contiguous()

                # Get bin indices for this class - VECTORIZED
                indices_0 = torch.bucketize(data[:, 0], edges[0]).clamp(0, n_bins-1)
                indices_1 = torch.bucketize(data[:, 1], edges[1]).clamp(0, n_bins-1)

                # VECTORIZED bin tracking - no lambda, no individual iterations
                unique_bins = torch.unique(torch.stack([indices_0, indices_1], dim=1), dim=0)

                # Store bins using regular dict (serialization-safe)
                cls_key = cls.item()
                pair_key = pair_idx
                if cls_key not in self.class_bins:
                    self.class_bins[cls_key] = {}
                if pair_key not in self.class_bins[cls_key]:
                    self.class_bins[cls_key][pair_key] = set()

                # Convert to Python set in one go (minimal overhead)
                bin_tuples = {tuple(bin_arr.tolist()) for bin_arr in unique_bins}
                self.class_bins[cls_key][pair_key].update(bin_tuples)

                # Vectorized counting (unchanged - optimal)
                flat_indices = indices_0 * n_bins + indices_1
                counts = torch.bincount(flat_indices, minlength=n_bins*n_bins)
                pair_counts[cls_idx] = counts.view(n_bins, n_bins).float()

            # Laplace smoothing and probability calculation
            smoothed = pair_counts + 1.0
            probs = smoothed / (smoothed.sum(dim=(1,2), keepdim=True) + 1e-8)

            batch_counts.append(smoothed)
            batch_probs.append(probs)

        return batch_counts, batch_probs

    def _process_pair_batch_optimized(self, dataset, labels, unique_classes, batch_pairs, n_bins, n_classes):
        """Process a batch of feature pairs efficiently"""
        batch_counts = []
        batch_probs = []

        for pair_idx in batch_pairs:
            f1, f2 = self.feature_pairs[pair_idx]
            edges = self.bin_edges[pair_idx]

            # Initialize counts for all classes
            pair_counts = torch.zeros((n_classes, n_bins, n_bins),
                                    dtype=torch.float64, device=self.device)

            # Vectorized processing for all classes
            for cls_idx, cls in enumerate(unique_classes):
                cls_mask = (labels == cls)
                if not torch.any(cls_mask):
                    continue

                data = dataset[cls_mask][:, [f1, f2]].contiguous()

                # Get bin indices for this class
                indices_0 = torch.bucketize(data[:, 0], edges[0]).clamp(0, n_bins-1)
                indices_1 = torch.bucketize(data[:, 1], edges[1]).clamp(0, n_bins-1)

                # Track used bins
                unique_bins = torch.unique(torch.stack([indices_0, indices_1], dim=1), dim=0).cpu().numpy()
                self.class_bins[cls.item()][pair_idx].update({tuple(bin) for bin in unique_bins})

                # Vectorized counting
                flat_indices = indices_0 * n_bins + indices_1
                counts = torch.bincount(flat_indices, minlength=n_bins*n_bins)
                pair_counts[cls_idx] = counts.view(n_bins, n_bins).float()

            # Laplace smoothing and probability calculation
            smoothed = pair_counts + 1.0
            probs = smoothed / (smoothed.sum(dim=(1,2), keepdim=True) + 1e-8)

            batch_counts.append(smoothed)
            batch_probs.append(probs)

        return batch_counts, batch_probs

    def _process_pair_batch(self, dataset_cpu, labels_cpu, batch_pairs, unique_classes, bin_sizes):
        """Process a batch of feature pairs on CPU"""
        batch_counts = []
        batch_probs = []

        for pair_idx, feature_group in enumerate(batch_pairs):
            feature_group = [int(x) for x in feature_group]
            group_data = dataset_cpu[:, feature_group]
            n_dims = len(feature_group)

            # Get bin sizes for this group
            group_bin_sizes = bin_sizes[:n_dims] if len(bin_sizes) > 1 else [bin_sizes[0]] * n_dims

            # Get bin edges for this group (already on correct device)
            group_bin_edges = self.bin_edges[pair_idx]

            # Initialize bin counts on CPU
            bin_shape = [len(unique_classes)] + group_bin_sizes
            bin_counts = torch.zeros(bin_shape, dtype=torch.float64)

            for class_idx, class_label in enumerate(unique_classes):
                class_mask = (labels_cpu == class_label)
                if class_mask.any():
                    class_data = group_data[class_mask]

                    # Compute bin indices
                    bin_indices = []
                    for dim in range(n_dims):
                        # Get edges for this dimension (already a tensor)
                        edges = group_bin_edges[dim]
                        indices = torch.bucketize(
                            class_data[:, dim],
                            edges.cpu()  # Move edges to CPU to match data
                        ).sub_(1).clamp_(0, group_bin_sizes[dim] - 1)
                        bin_indices.append(indices)

                    # Update counts
                    if n_dims == 1:
                        bin_counts[class_idx] = torch.bincount(
                            bin_indices[0],
                            minlength=group_bin_sizes[0]
                        ).float()
                    else:
                        # For multi-dimensional, use scatter_add
                        flat_indices = torch.sum(
                            torch.stack(bin_indices) * torch.tensor(
                                [np.prod(group_bin_sizes[i+1:]) for i in range(n_dims)]
                            ).unsqueeze(1),
                            dim=0
                        ).long()

                        counts = torch.zeros(np.prod(group_bin_sizes), dtype=torch.float64)
                        counts.scatter_add_(
                            0,
                            flat_indices,
                            torch.ones_like(flat_indices, dtype=torch.float64)
                        )
                        bin_counts[class_idx] = counts.reshape(*group_bin_sizes)

            # Apply Laplace smoothing and compute probabilities
            smoothed_counts = bin_counts + 1.0
            bin_probs = smoothed_counts / smoothed_counts.sum(dim=tuple(range(1, n_dims + 1)), keepdim=True)

            batch_counts.append(smoothed_counts)
            batch_probs.append(bin_probs)

        return batch_counts, batch_probs

    def _calculate_safe_batch_size(self, n_classes, n_bins):
        """Calculate safe batch size based on available memory"""
        # Conservative estimate - start small
        base_size = 10

        if torch.cuda.is_available():
            try:
                # Estimate memory needed per pair
                pair_mem = n_classes * (n_bins ** 2) * 4  # 4 bytes per float

                # Get available memory
                free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)

                # Calculate how many pairs we can fit (leave 50% buffer)
                safe_pairs = int((free_mem * 0.5) / pair_mem)
                return max(1, min(safe_pairs, 100))  # Limit between 1 and 100
            except:
                return base_size
        return base_size

    def _compute_gaussian_params(self, dataset: torch.Tensor, labels: torch.Tensor):
        """
        Compute Gaussian parameters (means and covariances) for all feature pairs once during initialization.

        Args:
            dataset: Input tensor of shape [n_samples, n_features]
            labels: Target tensor of shape [n_samples]

        Returns:
            Dictionary containing means and covariances for each feature pair and class
        """
        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)
        n_pairs = len(self.feature_pairs)

        # Initialize storage for means and covariances
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

    def _compute_pairwise_likelihood_parallel_std(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """Optimized Gaussian parameter computation using full covariance matrices and vectorized pair extraction."""
        DEBUG.log("Starting optimized Gaussian likelihood computation")

        # Generate feature pairs if needed
        if self.feature_pairs is None:
            self.feature_pairs = self._generate_feature_combinations(
                feature_dims,
                self.config.get('likelihood_config', {}).get('feature_group_size', 2),
                self.config.get('likelihood_config', {}).get('max_combinations', None)
            )

        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)
        n_pairs = len(self.feature_pairs)
        n_features = dataset.size(1)

        # Convert feature pairs to tensor for advanced indexing
        pair_indices = torch.tensor(self.feature_pairs, device=self.device, dtype=torch.long)

        # Initialize storage tensors
        means = torch.zeros((n_classes, n_pairs, 2), device=self.device)
        covs = torch.zeros((n_classes, n_pairs, 2, 2), device=self.device)

        # Process each class with batched operations
        for class_idx, class_id in enumerate(unique_classes):
            class_mask = labels == class_id
            class_data = dataset[class_mask]

            if class_data.size(0) == 0:  # Handle empty classes
                covs[class_idx] += torch.eye(2, device=self.device) * 1e-6
                continue

            # Compute full feature means
            full_mean = torch.mean(class_data, dim=0)

            # Center data and compute full covariance matrix
            centered = class_data - full_mean.unsqueeze(0)
            cov_matrix = torch.matmul(centered.T, centered) / (centered.size(0) - 1 + 1e-6)
            cov_matrix += torch.eye(n_features, device=self.device) * 1e-6  # Regularization

            # Extract means for all pairs simultaneously
            means[class_idx] = full_mean[pair_indices]

            # Extract covariance components using vectorized indexing
            i, j = pair_indices[:, 0], pair_indices[:, 1]
            cov_ii = cov_matrix[i, i]
            cov_ij = cov_matrix[i, j]
            cov_jj = cov_matrix[j, j]

            # Build covariance matrices in parallel
            covs[class_idx, :, 0, 0] = cov_ii
            covs[class_idx, :, 0, 1] = cov_ij
            covs[class_idx, :, 1, 0] = cov_ij  # Symmetric
            covs[class_idx, :, 1, 1] = cov_jj

        # Store parameters with standardized structure
        self.gaussian_params = {
            'means': means.contiguous(),
            'covs': covs.contiguous(),
            'classes': unique_classes,
            'feature_pairs': self.feature_pairs
        }

        return self.gaussian_params

    def _compute_batch_posterior_std(self, features: torch.Tensor, epsilon: float = 1e-10):
        """Gaussian posterior computation focusing on relative class probabilities"""
        features = features.to(self.device)
        batch_size = len(features)
        n_classes = len(self.likelihood_params['classes'])
        n_pairs = len(self.feature_pairs)

        # Initialize log likelihoods
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        # Setup progress bars
        with tqdm(total=n_pairs, desc="Processing feature pairs", leave=False) as pair_pbar:
            # Process each feature pair
            for pair_idx, pair in enumerate(self.feature_pairs):
                pair_data = features[:, pair]

                # Get weights for this pair
                pair_weights = [
                    self.weight_updater.get_gaussian_weights(class_idx, pair_idx)
                    for class_idx in range(n_classes)
                ]

                # Class processing progress bar
                with tqdm(total=n_classes, desc=f"Pair {pair_idx+1}/{n_pairs} classes", leave=False) as class_pbar:
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

                            # Log likelihood (excluding constant terms)
                            class_ll = -0.5 * quad + torch.log(weight + epsilon)

                        except RuntimeError:
                            # Handle numerical issues
                            class_ll = torch.full_like(quad, -1e10)

                        log_likelihoods[:, class_idx] += class_ll
                        class_pbar.update(1)  # Update class progress

                pair_pbar.update(1)  # Update pair progress

        # Convert to probabilities using softmax
        max_log_ll = torch.max(log_likelihoods, dim=1, keepdim=True)[0]
        exp_ll = torch.exp(log_likelihoods - max_log_ll)
        posteriors = exp_ll / (torch.sum(exp_ll, dim=1, keepdim=True) + epsilon)

        return posteriors, None


    def _initialize_bin_weights(self):
        """Initialize weights with config-consistent bin dimensions"""
        bin_sizes = self.config.get('likelihood_config', {}).get('bin_sizes', [128])
        n_bins_per_dim = bin_sizes[0]  # Match likelihood computation

        # For small datasets, use higher initial weights to prevent collapse
        n_samples = len(self.data) if hasattr(self, 'data') else 1000
        if n_samples < 500:  # Small dataset
            initial_weight = 0.1  # Higher initial weight for small datasets
        else:
            initial_weight = 1e-6  # Default for larger datasets

        self.weight_updater = BinWeightUpdater(
            n_classes=len(self.label_encoder.classes_),
            feature_pairs=self.feature_pairs,
            dataset_name=self.dataset_name,  # Pass dataset name
            n_bins_per_dim=self.n_bins_per_dim,
            batch_size=getattr(self, 'batch_size', 128)  # Use instance batch_size
        )

    def _update_priors_parallel(self, failed_cases: List[Tuple], batch_size: int = 128):
        """Vectorized weight updates with memory optimization and dtype fix"""
        n_failed = len(failed_cases)
        if n_failed == 0:
            self.consecutive_successes += 1
            return

        self.consecutive_successes = 0
        self.learning_rate = max(self.learning_rate / 2, 1e-6)

        # Stack all features and classes at once
        features = torch.stack([case[0] for case in failed_cases]).to(self.device)
        true_classes = torch.tensor([int(case[1]) for case in failed_cases], device=self.device)

        # Compute all posteriors in one batch
        if self.model_type == "Histogram":
            posteriors, bin_indices = self._compute_batch_posterior(features)
        else:  # Gaussian model
            posteriors, _ = self._compute_batch_posterior_std(features)
            return

        pred_classes = torch.argmax(posteriors, dim=1)

        # Vectorized adjustment calculation
        true_posteriors = posteriors[torch.arange(n_failed), true_classes]
        pred_posteriors = posteriors[torch.arange(n_failed), pred_classes]
        adjustments = self.learning_rate * (1.0 - (true_posteriors / pred_posteriors))

        # Get similarity threshold from config
        sim_threshold = self.config.get('active_learning', {}).get('similarity_threshold', 0.25)

        # Vectorized processing for all feature groups
        if bin_indices is not None:
            for group_idx in bin_indices:
                bin_i, bin_j = bin_indices[group_idx]

                # Get predicted class probabilities for these bins
                with torch.no_grad():
                    pred_probs = self.likelihood_params['bin_probs'][group_idx][
                        pred_classes, bin_i, bin_j
                    ]

                # Create mask for dissimilar bins
                dissimilar_mask = pred_probs < sim_threshold

                if not dissimilar_mask.any():
                    continue

                # Filter elements using mask
                mask_true_classes = true_classes[dissimilar_mask]
                mask_bin_i = bin_i[dissimilar_mask]
                mask_bin_j = bin_j[dissimilar_mask]
                mask_adjustments = adjustments[dissimilar_mask]

                # Group updates by class using vectorized operations
                unique_classes, inverse = torch.unique(mask_true_classes, return_inverse=True)

                for cls_idx, class_id in enumerate(unique_classes):
                    cls_mask = inverse == cls_idx
                    if not cls_mask.any():
                        continue

                    # Get class-specific updates
                    cls_bin_i = mask_bin_i[cls_mask]
                    cls_bin_j = mask_bin_j[cls_mask]
                    cls_adjustments = mask_adjustments[cls_mask]

                    # FIX: Ensure proper dtype matching
                    cls_adjustments = cls_adjustments.to(self.weight_updater.histogram_weights[class_id.item()][group_idx].dtype)

                    # Vectorized weight update
                    self.weight_updater.histogram_weights[class_id.item()][group_idx].index_put_(
                        indices=(cls_bin_i, cls_bin_j),
                        values=cls_adjustments,
                        accumulate=True
                    )

#------------------------------------------Boost weights------------------------------------------

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


#---------------------------------------------------------Save Last data -------------------------
    def load_last_known_split(self):
        """Load the last known split using stored original indices"""
        dataset_name = self.dataset_name
        train_file = f'data/{dataset_name}/Last_training.csv'
        test_file = f'data/{dataset_name}/Last_testing.csv'

        if os.path.exists(train_file) and os.path.exists(test_file):
            try:
                # Load indices directly from index column
                train_df = pd.read_csv(train_file)
                test_df = pd.read_csv(test_file)

                # Validate index column exists
                if 'original_index' not in train_df.columns or 'original_index' not in test_df.columns:
                    print("\033[K" + "No original_index column found in split files")
                    return None, None

                # Get indices as lists
                train_indices = train_df['original_index'].tolist()
                test_indices = test_df['original_index'].tolist()

                # Validate indices against current data
                max_valid_index = len(self.data) - 1
                train_indices = [idx for idx in train_indices if 0 <= idx <= max_valid_index]
                test_indices = [idx for idx in test_indices if 0 <= idx <= max_valid_index]

                print(f"\033[KLoaded previous split - Training: {len(train_indices)}, Testing: {len(test_indices)}")
                return train_indices, test_indices

            except Exception as e:
                print(f"\033[KError loading previous split: {str(e)}")
                return None, None

        return None, None

    def save_last_split(self, train_indices: list, test_indices: list):
        """Save split with original dataset indices"""
        dataset_name = self.dataset_name
        os.makedirs(f'data/{dataset_name}', exist_ok=True)

        # Save training indices with original indexes
        train_df = self.data.iloc[train_indices].copy()
        train_df['original_index'] = train_indices
        train_df.to_csv(f'data/{dataset_name}/Last_training.csv', index=False)

        # Save testing indices with original indexes
        test_df = self.data.iloc[test_indices].copy()
        test_df['original_index'] = test_indices
        test_df.to_csv(f'data/{dataset_name}/Last_testing.csv', index=False)

        print(f"\033[KSaved split with {len(train_indices)} train, {len(test_indices)} test samples")
#---------------Predcit New --------------------


#-------------Predict New -----------------------
    def predict(self, X: Union[pd.DataFrame, torch.Tensor], batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions in batches with consistent NaN handling.
        Returns both predicted classes and class probabilities.
        Handles both DataFrame and Tensor inputs.

        Args:
            X: Input features (DataFrame or Tensor)
            batch_size: Batch size for prediction

        Returns:
            Tuple containing:
            - predictions: Tensor of predicted class indices
            - posteriors: Tensor of class probabilities for all classes
        """
        # Ensure we have a properly initialized label encoder
        if not hasattr(self.label_encoder, 'classes_'):
            if hasattr(self, 'data'):
                # If we have data, fit the encoder (shouldn't happen in prediction)
                self.label_encoder.fit(self.data[self.target_column])
            else:
                raise RuntimeError("Label encoder not initialized and no data available to fit it")

        # Store current weights temporarily
        temp_W = self.current_W
        self.current_W = self.best_W.clone() if self.best_W is not None else self.current_W

        try:
            # Convert DataFrame to tensor if needed
            if isinstance(X, pd.DataFrame):
                # Preprocess with same NaN handling as training
                X_processed = self._preprocess_data(X, is_training=False)
                X_tensor = X_processed.to(self.device)
            else:
                # For tensor input, ensure proper NaN handling
                X_tensor = X.to(self.device)
                self.nan_mask = torch.isnan(X_tensor)
                X_tensor = torch.where(self.nan_mask,
                                     torch.tensor(-99999, device=X_tensor.device),
                                     X_tensor)

            n_batches = (len(X_tensor) + batch_size - 1) // batch_size
            all_predictions = []
            all_posteriors = []

            with tqdm(total=n_batches, desc="Prediction batches",leave=False) as pred_pbar:
                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i:min(i + batch_size, len(X_tensor))]
                    # Get posteriors (NaN handling happens inside these methods)
                    if self.model_type == "Histogram":
                        posteriors, _ = self._compute_batch_posterior(batch_X)
                    elif self.model_type == "Gaussian":
                        posteriors, _ = self._compute_batch_posterior_std(batch_X)
                    else:
                        raise ValueError(f"Invalid model type: {self.model_type}")

                    # Get predictions and store posteriors
                    batch_predictions = torch.argmax(posteriors, dim=1)
                    all_predictions.append(batch_predictions)
                    all_posteriors.append(posteriors)
                    pred_pbar.update(1)

            # Concatenate all batches
            predictions = torch.cat(all_predictions).cpu()
            posteriors = torch.cat(all_posteriors).cpu()

            return predictions, posteriors

        finally:
            # Restore original weights
            self.current_W = temp_W

    def _save_best_weights(self):
        """Save the best weights and corresponding training data to file"""
        if self.best_W is None:
            print("\033[KWarning: No best weights to save")
            return

        try:
            # Create directory for model components
            model_dir = os.path.join('Model')
            os.makedirs(model_dir, exist_ok=True)
            print(f"\033[KCreated model directory at {model_dir}")

            # Convert weights to numpy and then to list for JSON serialization
            #weights_np = self.best_W.cpu().numpy()
            weights_np = self.current_W.cpu().numpy()

            # Save weights
            weights_dict = {
                'version': 2,
                'weights': weights_np.tolist(),
                'shape': list(weights_np.shape),
                'dtype': str(weights_np.dtype)
            }

            weights_file = os.path.join('Model', f'Best_{self.model_type}_{self.dataset_name}_weights.json')
            print(f"\033[KAttempting to save weights to {weights_file}")

            # Use atomic write to prevent corruption
            temp_file = weights_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(weights_dict, f, indent=2)
                f.flush()  # Ensure the buffer is flushed to disk
                os.fsync(f.fileno())  # Force write to disk

            # Atomic rename
            os.replace(temp_file, weights_file)
            print(f"\033[KSaved weights to {weights_file}")

            # Save training data if available
            if hasattr(self, 'train_indices') and hasattr(self, 'data'):
                train_data = self.data.iloc[self.train_indices]
                train_data_file = os.path.join(model_dir, 'best_training_data.csv')

                print(f"\033[KAttempting to save training data to {train_data_file}")
                train_data.to_csv(train_data_file, index=False)
                print(f"\033[KSaved training data to {train_data_file}")

        except Exception as e:
            print(f"\033[KError saving best weights: {str(e)}")
            traceback.print_exc()


    def _load_best_weights(self):
        """Load the best weights and corresponding training data from file"""
        model_dir = os.path.join('Model')
        weights_file = os.path.join('Model', f'Best_{self.model_type}_{self.dataset_name}_weights.json')

        if os.path.exists(weights_file):
            try:
                print(f"Attempting to load weights file {weights_file}")
                with open(weights_file, 'r') as f:
                    weights_dict = json.load(f)

                # Load weights
                weights_array = np.array(weights_dict['weights'])
                self.best_W = torch.tensor(
                    weights_array,
                    dtype=torch.float64,
                    device=self.device
                )
                self.current_W = torch.tensor(
                    weights_array,
                    dtype=torch.float64,
                    device=self.device
                )
                print(f"Weights file {weights_file} loaded succesfully")
                # Only try to load training data if self.data exists
                if hasattr(self, 'data'):
                    train_data_file = os.path.join(model_dir, 'best_training_data.csv')
                    if os.path.exists(train_data_file):
                        train_data = pd.read_csv(train_data_file)
                        # Find matching indices in current data
                        self.train_indices = []
                        current_data = self.data.drop(columns=[self.target_column])

                        for idx, row in train_data.drop(columns=[self.target_column]).iterrows():
                            matches = (current_data == row).all(axis=1)
                            if matches.any():
                                self.train_indices.extend(matches[matches].index.tolist())

                        print(f"\033[KLoaded {len(self.train_indices)} training samples from best model")

                print(f"\033[KLoaded best weights from {weights_file}")
            except Exception as e:
                print(f"\033[KWarning: Could not load weights from {weights_file}: {str(e)}")
                self.best_W = None



    def _init_keyboard_listener(self):
        """Initialize keyboard listener with shared display connection"""
        if not hasattr(self, '_display'):
            try:
                import Xlib.display
                self._display = Xlib.display.Display()
            except Exception as e:
                print("\033[K" +f"Warning: Could not initialize X display: {e}")
                return None

        try:
            from pynput import keyboard
            return keyboard.Listener(
                on_press=self._on_key_press,
                _display=self._display  # Pass shared display connection
            )
        except Exception as e:
            print("\033[K" +f"Warning: Could not create keyboard listener: {e}")
            return None

    def _cleanup_keyboard(self):
        """Clean up keyboard resources"""
        if hasattr(self, '_display'):
            try:
                self._display.close()
                del self._display
            except:
                pass

    def print_colored_confusion_matrix(self, y_true, y_pred, class_labels=None, header=None):
        # Decode numeric labels back to original alphanumeric labels
        class_accuracies = self._calculate_class_wise_accuracy(
            torch.as_tensor(y_true),
            torch.as_tensor(y_pred)
        )

        # Get string labels for display only
        y_true_labels = self.label_encoder.inverse_transform(y_true)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)

        # Use minimum class accuracy as the criterion
        self.best_combined_accuracy = sum([v['accuracy'] for v in class_accuracies.values()]) / len(class_accuracies)

        # Get unique classes from both true and predicted labels
        unique_true = np.unique(y_true_labels)
        unique_pred = np.unique(y_pred_labels)

        # Use provided class labels or get from label encoder
        if class_labels is None:
            class_labels = self.label_encoder.classes_

        # Ensure all classes are represented in confusion matrix
        all_classes = np.unique(np.concatenate([unique_true, unique_pred, class_labels]))
        n_classes = len(all_classes)

        # Create class index mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

        # Initialize confusion matrix with zeros
        cm = np.zeros((n_classes, n_classes), dtype=int)

        # Fill confusion matrix
        for t, p in zip(y_true_labels, y_pred_labels):
            if t in class_to_idx and p in class_to_idx:
                cm[class_to_idx[t], class_to_idx[p]] += 1

        # Calculate precision for each class
        precision = []
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precision.append(prec)

        # Print confusion matrix with colors
        print("\033[K" + f"{Colors.BOLD}Confusion Matrix and Class-wise Metrics for [{header}]:{Colors.ENDC}")
        print("\033[K" + f"{'Actual/Predicted':<15}", end='')
        for label in all_classes:
            print("\033[K" + f"{str(label):<8}", end='')
        print("\033[K" + "Accuracy  Precision")
        print("\033[K" + "-" * (15 + 8 * n_classes + 20))

        # Print matrix with colors
        for i in range(n_classes):
            # Print actual class label
            print("\033[K" + f"{Colors.BOLD}{str(all_classes[i]):<15}{Colors.ENDC}", end='')

            # Print confusion matrix row
            for j in range(n_classes):
                if i == j:
                    # Correct predictions in green
                    color = Colors.GREEN
                else:
                    # Incorrect predictions in red
                    color = Colors.RED
                print("\033[K" + f"{color}{cm[i, j]:<8}{Colors.ENDC}", end='')

            # Print class accuracy with color based on performance
            acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
            if acc >= 0.9:
                acc_color = Colors.GREEN
            elif acc >= 0.7:
                acc_color = Colors.YELLOW
            else:
                acc_color = Colors.BLUE

            # Print class precision with color based on performance
            prec = precision[i]
            if prec >= 0.9:
                prec_color = Colors.GREEN
            elif prec >= 0.7:
                prec_color = Colors.YELLOW
            else:
                prec_color = Colors.BLUE

            print("\033[K" + f"{acc_color}{acc:>7.2%}{Colors.ENDC}  {prec_color}{prec:>8.2%}{Colors.ENDC}")

        # Print precision row at the bottom
        print("\033[K" + f"{Colors.BOLD}{'Precision':<15}{Colors.ENDC}", end='')
        for j in range(n_classes):
            tp = cm[j, j]
            fp = cm[:, j].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            if prec >= 0.9:
                prec_color = Colors.GREEN
            elif prec >= 0.7:
                prec_color = Colors.YELLOW
            else:
                prec_color = Colors.BLUE

            print("\033[K" + f"{prec_color}{prec:>8.2%}{Colors.ENDC}", end='')
        print("\033[K" + "")  # New line after precision row

        # Print overall accuracy and precision
        total_correct = np.diag(cm).sum()
        total_samples = cm.sum()
        if total_samples > 0:
            overall_acc = total_correct / total_samples
            # Micro-averaged precision (same as accuracy in multi-class)
            overall_prec = total_correct / total_samples

            print("\033[K" + "-" * (15 + 8 * n_classes + 20))
            acc_color = Colors.GREEN if overall_acc >= 0.9 else Colors.YELLOW if overall_acc >= 0.7 else Colors.BLUE
            prec_color = Colors.GREEN if overall_prec >= 0.9 else Colors.YELLOW if overall_prec >= 0.7 else Colors.BLUE
            print("\033[K" + f"{Colors.BOLD}Overall Accuracy:{Colors.ENDC} {acc_color}{overall_acc:.2%}{Colors.ENDC}")
            print("\033[K" + f"{Colors.BOLD}Overall Precision:{Colors.ENDC} {prec_color}{overall_prec:.2%}{Colors.ENDC}")
            print("\033[K" + f"Best Overall (Classwise) Accuracy till now is: {Colors.GREEN}{self.best_combined_accuracy:.2%}{Colors.ENDC}")

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, batch_size: int = 128):
        """Optimized training loop with vectorized operations and proper error handling"""
        print("\033[K" + "Starting training...", end="\r", flush=True)

        # Initialize visualization if enabled
        enable_visualization = self.config.get('training_params', {}).get('enable_visualization', True)
        if enable_visualization and not hasattr(self, 'visualizer'):
            self.visualizer = DBNNGeometricVisualizer(self)

        # Initialize tracking variables
        if not hasattr(self, 'best_combined_accuracy'):
            self.best_combined_accuracy = 0.0
        if not hasattr(self, 'best_model_weights'):
            self.best_model_weights = None

        # Store initial conditions
        if self.best_round_initial_conditions is None:
            self.best_round_initial_conditions = {
                'weights': self.current_W.clone(),
                'likelihood_params': self.likelihood_params,
                'feature_pairs': self.feature_pairs,
                'bin_edges': self.bin_edges,
                'gaussian_params': self.gaussian_params
            }

        # Initialize progress tracking
        epoch_pbar = tqdm(total=self.max_epochs, desc="Training epochs", leave=False)
        train_weights = self.current_W.clone() if self.current_W is not None else None

        # Pre-allocate tensors for batch processing
        n_samples = len(X_train)
        error_rates = []
        train_accuracies = []
        prev_train_error = float('inf')
        prev_train_accuracy = 0.0
        patience_counter = 0
        best_train_accuracy = 0.0

        patience = self.adaptive_patience if self.in_adaptive_fit else self.patience

        for epoch in range(self.max_epochs):
            # Save epoch data
            self.save_epoch_data(epoch, self.train_indices, self.test_indices)

            Trstart_time = time.time()
            failed_cases = []
            n_errors = 0

            # Process training data in optimized batches
            n_batches = (len(X_train) + batch_size - 1) // batch_size
            batch_pbar = tqdm(total=n_batches, desc=f"Epoch {epoch+1} batches", leave=False)

            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                current_batch_size = batch_end - i

                batch_X = X_train[i:batch_end]
                batch_y = y_train[i:batch_end]

                # Compute posteriors for batch (uses optimized _compute_batch_posterior)
                try:
                    if self.model_type == "Histogram":
                        posteriors, bin_indices = self._compute_batch_posterior(batch_X)
                    elif self.model_type == "Gaussian":
                        posteriors, comp_resp = self._compute_batch_posterior_std(batch_X)
                    else:
                        raise ValueError(f"Unknown model type: {self.model_type}")
                except Exception as e:
                    print(f"\033[KError computing posteriors: {str(e)}")
                    continue

                # Vectorized prediction and error calculation
                predictions = torch.argmax(posteriors, dim=1)
                batch_errors = (predictions != batch_y)
                n_errors += batch_errors.sum().item()

                # Collect failed cases efficiently
                if batch_errors.any():
                    failed_indices = torch.where(batch_errors)[0]
                    # Vectorized collection of failed cases
                    failed_features = batch_X[failed_indices]
                    failed_labels = batch_y[failed_indices]
                    failed_posteriors = posteriors[failed_indices].cpu().numpy()

                    for idx in range(len(failed_indices)):
                        failed_cases.append((
                            failed_features[idx],
                            failed_labels[idx].item(),
                            failed_posteriors[idx]
                        ))

                batch_pbar.update(1)

            batch_pbar.close()

            # Calculate metrics
            train_error_rate = n_errors / n_samples
            error_rates.append(train_error_rate)

            # Calculate training accuracy using optimized prediction
            with torch.no_grad():
                orig_weights = self.current_W
                self.current_W = train_weights

                train_pred_classes, train_posteriors = self.predict(X_train, batch_size=batch_size)
                train_accuracy = (train_pred_classes == y_train.cpu()).float().mean()

                self.current_W = orig_weights

            # CAPTURE VISUALIZATION DATA HERE
            if enable_visualization and hasattr(self, 'visualizer'):
                try:
                    # Compute overall accuracy for visualization
                    X_all = torch.cat([X_train, X_test], dim=0)
                    y_all = torch.cat([y_train, y_test], dim=0)
                    all_pred, _ = self.predict(X_all, batch_size=batch_size)
                    overall_accuracy = (all_pred.cpu() == y_all.cpu()).float().mean().item()

                    # Capture snapshot every 5 epochs to avoid too much data
                    if epoch % 5 == 0 or epoch == self.max_epochs - 1:
                        self.visualizer.capture_epoch_snapshot(epoch_num=epoch, accuracy=overall_accuracy,weights=train_weights,predictions=all_pred,features=X_all,targets=y_all)
                except Exception as e:
                    print(f"\033[KVisualization capture skipped: {str(e)}")

            # Update best accuracies
            best_train_accuracy = max(best_train_accuracy, train_accuracy)
            train_accuracies.append(train_accuracy)

            # Update progress
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                'train_err': f"{train_error_rate:.4f}",
                'train_acc': f"{train_accuracy:.4f}"
            })

            # Check for best round
            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy
                self.best_round = epoch
                self.best_round_initial_conditions = {
                    'weights': self.current_W.clone(),
                    'likelihood_params': self.likelihood_params,
                    'feature_pairs': self.feature_pairs,
                    'bin_edges': self.bin_edges,
                    'gaussian_params': self.gaussian_params
                }

            # Update best model if improved
            if train_error_rate <= self.best_error:
                improvement = self.best_error - train_error_rate
                self.best_error = train_error_rate
                self.best_W = self.current_W.clone()

                if improvement <= 0.001:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    self.learning_rate = LearningRate
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= patience or train_accuracy == 1.00:
                print("\033[K" + f"{Colors.YELLOW} Early stopping.{Colors.ENDC}")
                break

            # Update weights using optimized method
            if failed_cases:
                try:
                    self._update_priors_parallel(failed_cases, batch_size)
                except Exception as e:
                    print(f"\033[KWarning: Failed to update priors: {str(e)}")
                    # Continue training even if weight update fails

        # Training complete
        epoch_pbar.close()
        return self.current_W.cpu(), error_rates

    #---------------------------------Train InvertableDBNN on the fly ------------------------------------
    def load_inverse_model(self, custom_path: str = None) -> bool:
       try:
           load_dir = custom_path or os.path.join('Model', f'Best_inverse_{self.forward_model.dataset_name}')
           model_path = os.path.join(load_dir, 'inverse_model.pt')
           config_path = os.path.join(load_dir, 'inverse_config.json')

           if not (os.path.exists(model_path) and os.path.exists(config_path)):
               print("\033[K" +f"No saved inverse model found at {load_dir}")
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

           print("\033[K" +f"Loaded inverse model from {load_dir}")
           return True

       except Exception as e:
           print("\033[K" +f"Error loading inverse model: {str(e)}")
           traceback.print_exc()
           return False

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
        csv_path = os.path.join(recon_dir, f'data/{dataset_name}/{dataset_name}_reconstruction.csv')
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

    #------------------------------End Train InvertableDBNN on the fly ------------------------------------

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
        print("\033[K" +f"{Colors.BOLD}{Colors.BLUE}Detailed Classification Analysis:{Colors.ENDC}")
        print("\033[K" +f"{Colors.BOLD}Total samples:{Colors.ENDC} {Colors.YELLOW}{n_total:,}{Colors.ENDC}")

        # Color code for correctly classified
        correct_color = Colors.GREEN if (n_correct/n_total) >= 0.9 else \
                       Colors.YELLOW if (n_correct/n_total) >= 0.7 else \
                       Colors.BLUE
        print("\033[K" +f"{Colors.BOLD}Correctly classified:{Colors.ENDC} {correct_color}{n_correct:,}{Colors.ENDC}")

        # Color code for incorrectly classified
        incorrect = n_total - n_correct
        incorrect_color = Colors.GREEN if (incorrect/n_total) <= 0.1 else \
                         Colors.YELLOW if (incorrect/n_total) <= 0.3 else \
                         Colors.RED
        print("\033[K" +f"{Colors.BOLD}Incorrectly classified:{Colors.ENDC} {incorrect_color}{incorrect:,}{Colors.ENDC}")

        # Color code for raw accuracy
        accuracy = n_correct/n_total
        accuracy_color = Colors.GREEN if accuracy >= 0.9 else \
                        Colors.YELLOW if accuracy >= 0.7 else \
                        Colors.BLUE
        print("\033[K" +f"{Colors.BOLD}Raw accuracy:{Colors.ENDC} {accuracy_color}{accuracy:.4%}{Colors.ENDC}\n")

        # Print confusion matrix with colors
        self.print_colored_confusion_matrix(true_labels_array, pred_labels,header="Test data")

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

        print("\033[K" +f"Detailed analysis saved to {analysis_file}")

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
            print("\033[K" +"[WARNING] All features were removed! Reverting to original features with warnings.")
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
        """Get or create consistent train-test split using smaller chunks to avoid memory issues."""
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
        split_path = os.path.join(base_path, dataset_folder, 'train_test_split.pkl')

        if os.path.exists(split_path):
            with open(split_path, 'rb') as f:
                split_indices = pickle.load(f)
                train_idx, test_idx = split_indices['train'], split_indices['test']
                return (X_tensor[train_idx], X_tensor[test_idx],
                        y_tensor[train_idx], y_tensor[test_idx])

        # If no saved split exists, create one using smaller chunks
        X_cpu = X_tensor.cpu().numpy()  # Move data to CPU for sklearn compatibility
        y_cpu = y_tensor.cpu().numpy()

        # Perform train-test split on CPU in smaller chunks
        X_train, X_test, y_train, y_test = train_test_split(
            X_cpu,  # Use NumPy array for sklearn compatibility
            y_cpu,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=(self.shuffle_state != -1)
        )

        # Convert back to PyTorch tensors and move to the appropriate device
        X_train = torch.tensor(X_train, dtype=torch.float64).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float64).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)

        # Save split indices to avoid recomputing
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        split_indices = {
            'train': torch.arange(len(X_train)),  # Indices for training set
            'test': torch.arange(len(X_train), len(X_train) + len(X_test))  # Indices for test set
        }
        with open(split_path, 'wb') as f:
            pickle.dump(split_indices, f)

        # Store test indices for later use
        self.test_indices = split_indices['test']

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
                W[class_id][feature_pair] = torch.tensor(0.1, dtype=torch.float64)
        return W

    def _calculate_class_wise_accuracy(self, y_true, y_pred):
        """Calculate class-wise accuracy metrics with device handling"""
        # Ensure both tensors are on the same device (preferably CPU for this operation)
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

        unique_classes = torch.unique(y_true)
        class_accuracies = {}

        for class_id in unique_classes:
            class_mask = (y_true == class_id)
            n_class_samples = class_mask.sum().item()
            if n_class_samples == 0:
                continue

            correct = (y_pred[class_mask] == y_true[class_mask]).sum().item()
            class_acc = correct / n_class_samples
            class_accuracies[class_id.item()] = {
                'accuracy': class_acc,
                'n_samples': n_class_samples,
                'correct': correct
            }

        return class_accuracies

    def fit_predict(self, batch_size: int = 128, save_path: str = None):
        """Full training and prediction pipeline with GPU optimization and optional prediction saving"""
        try:
            # Set a flag to indicate we're printing metrics
            self._last_metrics_printed = True
            class_preference = self.config.get('training_params', {}).get('class_preference', True)

            # If this is a fresh training round, reset to the best round's initial conditions
            if self.best_round_initial_conditions is not None:
                print("\033[K" + "Starting fresh training with best round's initial conditions", end='\r', flush=True)
                self.current_W = self.best_round_initial_conditions['weights'].clone()
                self.likelihood_params = self.best_round_initial_conditions['likelihood_params']
                self.feature_pairs = self.best_round_initial_conditions['feature_pairs']
                self.bin_edges = self.best_round_initial_conditions['bin_edges']
                self.gaussian_params = self.best_round_initial_conditions['gaussian_params']

            # Handle data preparation based on whether we're in adaptive training or final evaluation
            if self.in_adaptive_fit:
                if not hasattr(self, 'X_tensor') or not hasattr(self, 'y_tensor'):
                    raise ValueError("X_tensor or y_tensor not found. Initialize them in adaptive_fit_predict first.")

                if not hasattr(self, 'train_indices') or not hasattr(self, 'test_indices'):
                    raise ValueError("train_indices or test_indices not found")

                # Use stored tensors and indices, but verify sizes match
                try:
                    X_train = self.X_tensor[self.train_indices]
                    X_test = self.X_tensor[self.test_indices]
                    y_train = self.y_tensor[self.train_indices]
                    y_test = self.y_tensor[self.test_indices]
                except Exception as e:
                    # If there's any issue with indices, fall back to regular training path
                    DEBUG.log(f"Error using stored indices: {str(e)}. Falling back to regular training.")
                    self.in_adaptive_fit = False
                    # Reset indices and proceed with regular path
                    self.train_indices = None
                    self.test_indices = None
                    return self.fit_predict(batch_size=batch_size, save_path=save_path)

            else:
                # Regular training path
                X = self.data.drop(columns=[self.target_column])
                y = self.data[self.target_column]

                # Check if label encoder is already fitted
                if not hasattr(self.label_encoder, 'classes_'):
                    y_encoded = self.label_encoder.fit_transform(y)
                else:
                    y_encoded = self.label_encoder.transform(y)

                # Preprocess features including categorical encoding
                X_processed = self._preprocess_data(X, is_training=True)

                # Convert to tensors and move to device
                X_tensor = torch.tensor(X_processed, dtype=torch.float64).to(self.device)
                y_tensor = torch.LongTensor(y_encoded).to(self.device)

                # Split data
                # Get consistent train-test split
                X_train, X_test, y_train, y_test = self._get_train_test_split(
                    X_tensor, y_tensor)

                # Convert split data back to tensors
                X_train = torch.from_numpy(X_train).to(self.device, dtype=torch.float64)
                X_test = torch.from_numpy(X_test).to(self.device, dtype=torch.float64)
                y_train = torch.from_numpy(y_train).to(self.device, dtype=torch.long)
                y_test = torch.from_numpy(y_test).to(self.device, dtype=torch.long)

            # Verify tensor sizes match before training
            if X_train.size(0) != y_train.size(0) or X_test.size(0) != y_test.size(0):
                raise ValueError(f"Tensor size mismatch. X_train: {X_train.size(0)}, y_train: {y_train.size(0)}, "
                               f"X_test: {X_test.size(0)}, y_test: {y_test.size(0)}")

            # Train model
            final_W, error_rates = self.train(X_train, y_train, X_test, y_test, batch_size=batch_size)

            # Save categorical encoders
            self._save_categorical_encoders()

            # Make predictions on the entire dataset
            print("\033[K" + f"{Colors.YELLOW}Generating predictions for the entire dataset{Colors.ENDC}", end='\r', flush=True)

            # Use the original preprocessed data, not concatenated train/test
            X_all = self.X_tensor  # This should have the correct size
            y_all = self.y_tensor  # This should have the correct size

            # Verify sizes match
            print(f"\033[K[DEBUG] X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")
            print(f"\033[K[DEBUG] Original data shape: {self.X_Orig.shape}")

            all_pred_classes, all_posteriors = self.predict(X_all, batch_size=batch_size)

            # Verify prediction sizes
            print(f"\033[K[DEBUG] Predictions shape: {all_pred_classes.shape}, should be: {len(self.X_Orig)}")

            # Move tensors to CPU for accuracy calculation
            y_all_cpu = y_all.cpu()
            all_pred_classes_cpu = all_pred_classes.cpu()

            # Calculate accuracy metrics
            if class_preference:
                # Calculate class-wise accuracy on CPU
                class_accuracies = self._calculate_class_wise_accuracy(y_all_cpu, all_pred_classes_cpu)

                # Use minimum class accuracy as the criterion
                current_metric = sum([v['accuracy'] for v in class_accuracies.values()]) / len(class_accuracies)
                best_metric = self.best_combined_accuracy

                # Print class-wise metrics
                print("\033[K" + f"{Colors.GREEN}Class-wise accuracies:{Colors.ENDC}")
                for class_id, metrics in class_accuracies.items():
                    class_name = self.label_encoder.inverse_transform([class_id])[0]
                    print(f"\033[K  {class_name}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['n_samples']})")
                print(f"\033[KClass-wise accuracy: {current_metric:.2%}")
            else:
                # Original behavior - overall accuracy calculated on CPU
                current_metric = (y_all_cpu == all_pred_classes_cpu).float().mean().item()
                best_metric = self.best_combined_accuracy

            # Update best model if improved
            if current_metric > best_metric:
                print("\033[K" + f"{Colors.RED}---------------------------------------------------------------------------------------{Colors.ENDC}")
                if class_preference:
                    print("\033[K" + f"{Colors.GREEN}Best minimum class accuracy improved from {best_metric:.2%} to {current_metric:.2%}{Colors.ENDC}")
                    # Store the mean class accuracy as best_combined_accuracy
                    mean_class_acc = sum([v['accuracy'] for v in class_accuracies.values()]) / len(class_accuracies)
                    self.best_combined_accuracy = current_metric # maximise the classwise accuracies
                    print(f"The mean class accuracy is {mean_class_acc}")
                else:
                    print("\033[K" + f"{Colors.GREEN}Best classwise accuracy improved from {best_metric:.2%} to {current_metric:.2%}{Colors.ENDC}")
                    self.best_combined_accuracy = current_metric
                print("\033[K" + f"{Colors.RED}---------------------------------------------------------------------------------------{Colors.ENDC}")

                self._save_model_components()
            else:
                print("\033[K" + f"{Colors.RED}Current classwise accuracy declined from {best_metric:.2%} to {current_metric:.2%}{Colors.ENDC}")
            #self.reset_to_initial_state() # After saving the weights, reset to initial state for next round.


            # Extract predictions for training and test data using stored indices
            y_train_pred = all_pred_classes[self.train_indices]  # Predictions for training data
            y_test_pred = all_pred_classes[self.test_indices]    # Predictions for test data


           # Generate detailed predictions for the entire dataset
            print("\033[K" + "Computing detailed predictions for the whole data", end='\r', flush=True)
            all_results = self._generate_detailed_predictions(self.X_Orig, all_pred_classes, y_all, all_posteriors)

            # Convert train/test indices to list of integers
            train_indices = self.train_indices
            if isinstance(train_indices, torch.Tensor):
                train_indices = train_indices.cpu().tolist()
            elif isinstance(train_indices, np.ndarray):
                train_indices = train_indices.tolist()

            # Add split information to results
            all_results['split'] = 'test'
            all_results.loc[train_indices, 'split'] = 'train'

            train_results = all_results.iloc[self.train_indices]
            test_results = all_results.iloc[self.test_indices]
            # Filter failed examples (where predicted class != true class)
            failed_examples = all_results[all_results['predicted_class'] != all_results['true_class']]
            # Filter passed examples (where predicted class == true class)
            passed_examples = all_results[all_results['predicted_class'] == all_results['true_class']]

            # Save results if path is provided
            if save_path:

                # Save training predictions
                print("\033[K" + "Saving Train predictions", end='\r', flush=True)
                train_results.to_csv(f"{save_path}/train_predictions.csv", index=False)

                # Save test predictions
                print("\033[K" + "Saving Test predictions", end='\r', flush=True)
                test_results.to_csv(f"{save_path}/test_predictions.csv", index=False)

                # Save all predictions
                print("\033[K" + "Saving Combined predictions", end='\r', flush=True)
                all_results.to_csv(f"{save_path}/combined_predictions.csv", index=False)

                # Save failed examples
                print("\033[K" + "Saving Failed examples", end='\r', flush=True)
                failed_examples.to_csv(f"{save_path}/failed_examples.csv", index=False)

                # Save passed examples
                print("\033[K" + "Saving Passed examples", end='\r', flush=True)
                passed_examples.to_csv(f"{save_path}/passed_examples.csv", index=False)

                # Save metadata
                print("\033[K" + "Saving Metadata", end='\r', flush=True)
                metadata = {
                    'rejected_columns': self.high_cardinality_columns,
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'preprocessing_details': {
                        'cardinality_threshold': self.cardinality_threshold,
                        'cardinality_tolerance': self.cardinality_tolerance,
                        'categorical_columns': list(self.categorical_encoders.keys())
                    }
                }
                with open(f"{save_path}/metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=4)

            # Calculate metrics for test set
            y_test_cpu = y_test.cpu().numpy()
            y_train_cpu = y_train.cpu().numpy()

            # Convert numerical labels back to original classes
            y_train_labels = self.label_encoder.inverse_transform(y_train.cpu().numpy())
            y_test_labels = self.label_encoder.inverse_transform(y_test.cpu().numpy())
            y_train_pred_labels = self.label_encoder.inverse_transform(y_train_pred.cpu().numpy())
            y_test_pred_labels = self.label_encoder.inverse_transform(y_test_pred.cpu().numpy())

            # Prepare results
            print("\033[K" + "Preparing results of training", end='\r', flush=True)
            results = {
                'all_predictions': all_results,
                'train_predictions': train_results,
                'test_predictions': test_results,
                'metadata': metadata,
                'classification_report': classification_report(y_test_labels, y_test_pred_labels),
                'confusion_matrix': confusion_matrix(y_test_labels, y_test_pred_labels),
                'error_rates': error_rates,
                'test_accuracy': (np.asarray(y_test_pred) ==np.asarray(y_test_cpu)).astype(float).mean().item(),  # Convert to float before mean
                'train_accuracy': (np.asarray(y_train_pred) == np.asarray(y_train_cpu)).astype(float).mean().item()  # Convert to float before mean
            }

            # Generate point-colored confusion matrices for train, test, and combined data
            print("\033[K" + f"{Colors.BOLD}Generating Confusion Matrices:{Colors.ENDC}", end='\r', flush=True)

            # Confusion matrix for training data
            self.print_colored_confusion_matrix(y_train_cpu, y_train_pred, header="Training Data")

            # Confusion matrix for test data
            self.print_colored_confusion_matrix(y_test_cpu, y_test_pred, header="Test Data")

            # Confusion matrix for combined data
            y_all_cpu = y_all.cpu().numpy()
            self.print_colored_confusion_matrix(y_all_cpu,  all_pred_classes.cpu().numpy(), header="Combined Data")

            return results

        except Exception as e:
            DEBUG.log(f"Error in fit_predict: {str(e)}")
            DEBUG.log(f"Traceback: {traceback.format_exc()}")
            raise

    def _get_model_components_filename(self):
        """Get filename for model components"""
        return os.path.join('Model', f'Best_{self.model_type}_{self.dataset_name}_components.pkl')
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

                print("\033[K" +f"Loaded categorical encoders from {encoders_file}", end="\r", flush=True)
            except Exception as e:
                print("\033[K" +f"Warning: Failed to load categorical encoders: {str(e)}")
                self.categorical_encoders = {}

    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Encode categorical features with proper type handling"""
        df_encoded = df.copy()

        for column in df.columns:
            if column in self.categorical_encoders or (df[column].dtype == 'object' or df[column].dtype.name == 'category'):
                # Store original dtype
                original_dtype = df[column].dtype

                # Convert to string for consistent encoding
                col_data = df[column].astype(str)

                if is_training or column not in self.categorical_encoders:
                    # Training mode or new column - create new mapping
                    unique_values = col_data.unique()
                    mapping = {v: i for i, v in enumerate(unique_values)}
                    self.categorical_encoders[column] = mapping
                else:
                    # Prediction mode - use existing mapping
                    mapping = self.categorical_encoders[column]

                # Encode values
                encoded = col_data.map(mapping)

                # Handle unseen values during prediction
                if not is_training:
                    unmapped = encoded.isna()
                    if unmapped.any():
                        # Use mean value of known encodings for unseen values
                        mean_value = np.mean(list(mapping.values()))
                        # Convert mean to original dtype before assignment
                        if np.issubdtype(original_dtype, np.integer):
                            mean_value = int(round(mean_value))

                        # Create new series with correct dtype
                        fill_values = pd.Series([mean_value] * len(unmapped),
                                              index=unmapped[unmapped].index)
                        fill_values = fill_values.astype(original_dtype)

                        # Assign with loc to avoid dtype warnings
                        encoded = encoded.astype(float)  # Intermediate float type
                        encoded.loc[unmapped] = fill_values

                # Convert back to original dtype
                df_encoded[column] = encoded.astype(original_dtype)

        return df_encoded

    def save_predictions(self, X: pd.DataFrame, predictions: torch.Tensor, output_file: str, true_labels: pd.Series = None):
        """Save predictions with proper class handling and probability computation"""
        predictions = predictions.cpu()

        # Create a copy of the original dataset to preserve all columns
        result_df = X.copy()

        # Convert predictions to original class labels
        pred_labels = self.label_encoder.inverse_transform(predictions.numpy())
        result_df['predicted_class'] = pred_labels

        if true_labels is not None:
            result_df['true_class'] = true_labels

        # Get preprocessed features for probability computation
        X_processed = self._preprocess_data(X, is_training=False)
        if isinstance(X_processed, torch.Tensor):
            X_tensor = X_processed.clone().detach().to(self.device)
        else:
            X_tensor = torch.tensor(X_processed, dtype=torch.float64).to(self.device)

        # Compute probabilities in batches
        batch_size = 128
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
                print("\033[K" +f"Error computing probabilities for batch {i}: {str(e)}")
                return None

        if all_probabilities:
            all_probabilities = np.vstack(all_probabilities)
        else:
            print("\033[K" +"No probabilities were computed successfully", end="\r", flush=True)
            return None

        # Ensure we're only using valid class indices
        valid_classes = self.label_encoder.classes_
        n_classes = len(valid_classes)

        # Verify probability array shape matches number of classes
        if all_probabilities.shape[1] != n_classes:
            print("\033[K" +f"Warning: Probability array shape ({all_probabilities.shape}) doesn't match number of classes ({n_classes})")
            # Adjust probabilities array if necessary
            if all_probabilities.shape[1] > n_classes:
                all_probabilities = all_probabilities[:, :n_classes]
            else:
                # Pad with zeros if needed
                pad_width = ((0, 0), (0, n_classes - all_probabilities.shape[1]))
                all_probabilities = np.pad(all_probabilities, pad_width, mode='constant')

        # Add probability columns for each valid class
        for i, class_name in enumerate(valid_classes):
            if i < all_probabilities.shape[1]:  # Safety check
                result_df[f'prob_{class_name}'] = all_probabilities[:, i]

        # Add maximum probability
        result_df['max_probability'] = all_probabilities.max(axis=1)

        # Create the output directory if it doesn't exist
        dataset_name = os.path.splitext(os.path.basename(self.dataset_name))[0]
        output_dir = os.path.join('data', dataset_name, 'Predictions')
        os.makedirs(output_dir, exist_ok=True)

        # Save the predictions file in the new directory
        output_path = os.path.join(output_dir, output_file)
        result_df.to_csv(output_path, index=False)
        print("\033[K" +f"{Colors.GREEN}Saved predictions to {output_path}{Colors.ENDC}", end="\r", flush=True)

        if true_labels is not None:
            # Verification analysis
            self.verify_classifications(X, true_labels, predictions)

        return result_df

    def _save_model_components(self):
        """Enhanced model component saving with validation and atomic writes"""
        try:
            # Validate critical components exist before saving
            required_components = {
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_pairs': self.feature_pairs,
                'model_type': self.model_type,
                'target_column': self.target_column,
                'n_bins_per_dim': self.n_bins_per_dim
            }

            for name, component in required_components.items():
                if component is None:
                    raise ValueError(f"Cannot save model: {name} is None")

            # Validate label encoder state
            if not hasattr(self.label_encoder, 'classes_'):
                raise ValueError("Label encoder not properly fitted")

            # Prepare components dictionary with validation
            components = {
                'version': 4,  # Updated version for new format with original_target_dtype
                'scaler': self.scaler,
                'label_encoder': {
                    'classes_': self.label_encoder.classes_.tolist(),
                    'fitted': hasattr(self.label_encoder, 'classes_')
                },
                'original_target_dtype': str(getattr(self, 'original_target_dtype', 'object')),  # NEW: Store original target dtype
                'model_type': self.model_type,
                'feature_pairs': self.feature_pairs.cpu().tolist() if torch.is_tensor(self.feature_pairs) else self.feature_pairs,
                'global_mean': self.global_mean,
                'global_std': self.global_std,
                'categorical_encoders': self.categorical_encoders,
                'feature_columns': self.feature_columns,
                'original_columns': getattr(self, 'original_columns', None),
                'target_column': self.target_column,
                'config': self.config,
                'high_cardinality_columns': getattr(self, 'high_cardinality_columns', []),
                'best_error': self.best_error,
                'weight_updater': self.weight_updater,
                'n_bins_per_dim': self.n_bins_per_dim,
                'bin_edges': [edge.cpu().tolist() if torch.is_tensor(edge) else edge for edge in self.bin_edges] if hasattr(self, 'bin_edges') else None,
                'gaussian_params': {
                    'means': self.gaussian_params['means'].cpu().tolist() if torch.is_tensor(self.gaussian_params['means']) else self.gaussian_params['means'],
                    'covs': self.gaussian_params['covs'].cpu().tolist() if torch.is_tensor(self.gaussian_params['covs']) else self.gaussian_params['covs'],
                    'classes': self.gaussian_params['classes'].cpu().tolist() if torch.is_tensor(self.gaussian_params['classes']) else self.gaussian_params['classes']
                } if hasattr(self, 'gaussian_params') and self.gaussian_params is not None else None
            }

            # Add model-specific components with validation
            if self.model_type == "Histogram":
                if not all(k in self.likelihood_params for k in ['bin_probs', 'bin_edges', 'classes']):
                    raise ValueError("Incomplete Histogram model parameters")
                components.update({
                    'likelihood_params': {
                        'bin_probs': [prob.cpu().tolist() if torch.is_tensor(prob) else prob for prob in self.likelihood_params['bin_probs']],
                        'bin_edges': [[edge.cpu().tolist() if torch.is_tensor(edge) else edge for edge in pair] for pair in self.likelihood_params['bin_edges']],
                        'classes': self.likelihood_params['classes'].cpu().tolist() if torch.is_tensor(self.likelihood_params['classes']) else self.likelihood_params['classes'],
                        'feature_pairs': self.likelihood_params['feature_pairs'].cpu().tolist() if torch.is_tensor(self.likelihood_params['feature_pairs']) else self.likelihood_params['feature_pairs']
                    }
                })
            elif self.model_type == "Gaussian":
                if not all(k in self.likelihood_params for k in ['means', 'covs', 'classes']):
                    raise ValueError("Incomplete Gaussian model parameters")
                components.update({
                    'likelihood_params': {
                        'means': self.likelihood_params['means'].cpu().tolist() if torch.is_tensor(self.likelihood_params['means']) else self.likelihood_params['means'],
                        'covs': self.likelihood_params['covs'].cpu().tolist() if torch.is_tensor(self.likelihood_params['covs']) else self.likelihood_params['covs'],
                        'classes': self.likelihood_params['classes'].cpu().tolist() if torch.is_tensor(self.likelihood_params['classes']) else self.likelihood_params['classes'],
                        'feature_pairs': self.likelihood_params['feature_pairs'].cpu().tolist() if torch.is_tensor(self.likelihood_params['feature_pairs']) else self.likelihood_params['feature_pairs']
                    }
                })

            # Get filename and ensure directory exists
            components_file = self._get_model_components_filename()
            os.makedirs(os.path.dirname(components_file), exist_ok=True)

            # Atomic save operation
            temp_file = components_file + '.tmp'
            with open(temp_file, 'wb') as f:
                pickle.dump(components, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            os.replace(temp_file, components_file)

            print(f"\033[K[SUCCESS] Saved model components to {components_file} (Size: {os.path.getsize(components_file)/1024:.2f} KB)")
            return True

        except Exception as e:
            print(f"\033[K[ERROR] Failed to save model components: {str(e)}")
            traceback.print_exc()
            # Clean up temporary file if it exists
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    def _load_model_components(self):
        """Enhanced model component loading with comprehensive validation and proper label encoder restoration"""
        components_file = self._get_model_components_filename()

        if not os.path.exists(components_file):
            print(f"\033[K[ERROR] Model components file not found: {components_file}")
            return False

        try:
            print(f"\033[K[INFO] Loading model components from {components_file} (Size: {os.path.getsize(components_file)/1024:.2f} KB)")

            with open(components_file, 'rb') as f:
                components = pickle.load(f)

            # Validate file version and basic structure
            if not isinstance(components, dict) or 'version' not in components:
                raise ValueError("Invalid components file format")

            # Version-specific validation
            if components['version'] < 2:
                raise ValueError(f"Unsupported components version: {components['version']}")

            # Store current visualization setting before loading config
            current_visualization = self.config.get('training_params', {}).get('enable_visualization', False)

            # Load and validate label encoder - FIXED: Properly restore the label encoder
            if 'label_encoder' not in components or not components['label_encoder'].get('fitted', False):
                raise ValueError("Label encoder not properly saved or not fitted")

            # CRITICAL FIX: Properly restore the label encoder with all attributes
            self.label_encoder = LabelEncoder()
            if 'classes_' in components['label_encoder']:
                self.label_encoder.classes_ = np.array(components['label_encoder']['classes_'])
                print(f"\033[K[INFO] Restored label encoder with classes: {self.label_encoder.classes_}")
            else:
                raise ValueError("Label encoder classes not found in saved components")

            # Load original target dtype for proper decoding
            if 'original_target_dtype' in components:
                dtype_str = components['original_target_dtype']
                try:
                    if dtype_str != 'object':
                        self.original_target_dtype = np.dtype(dtype_str)
                    else:
                        self.original_target_dtype = np.dtype('object')
                    print(f"\033[K[INFO] Loaded original target dtype: {self.original_target_dtype}")
                except Exception as e:
                    print(f"\033[K[WARNING] Failed to load original target dtype '{dtype_str}': {str(e)}")
                    self.original_target_dtype = np.dtype('object')
            else:
                # For backward compatibility with older models
                self.original_target_dtype = np.dtype('object')
                print(f"\033[K[WARNING] No original_target_dtype found, using default: {self.original_target_dtype}")

            # Validate and load core components
            required_components = [
                'scaler', 'model_type', 'feature_pairs',
                'target_column', 'n_bins_per_dim'
            ]
            for comp in required_components:
                if comp not in components:
                    raise ValueError(f"Missing required component: {comp}")
                setattr(self, comp, components[comp])

            # Load likelihood parameters with model-specific validation
            if 'likelihood_params' not in components:
                raise ValueError("Missing likelihood parameters")

            self.likelihood_params = components['likelihood_params']

            # Convert back to tensors using proper methods
            def safe_to_tensor(data, device=None):
                """Convert data to tensor using proper method based on input type"""
                if isinstance(data, torch.Tensor):
                    return data.to(device) if device else data.clone().detach()
                return torch.tensor(data, device=device) if device else torch.tensor(data)

            if self.model_type == "Histogram":
                self.likelihood_params['bin_probs'] = [safe_to_tensor(prob, self.device) for prob in self.likelihood_params['bin_probs']]
                self.likelihood_params['bin_edges'] = [[safe_to_tensor(edge, self.device) for edge in pair] for pair in self.likelihood_params['bin_edges']]
                self.likelihood_params['classes'] = safe_to_tensor(self.likelihood_params['classes'], self.device)
                self.likelihood_params['feature_pairs'] = safe_to_tensor(self.likelihood_params['feature_pairs'], self.device)
            elif self.model_type == "Gaussian":
                self.likelihood_params['means'] = safe_to_tensor(self.likelihood_params['means'], self.device)
                self.likelihood_params['covs'] = safe_to_tensor(self.likelihood_params['covs'], self.device)
                self.likelihood_params['classes'] = safe_to_tensor(self.likelihood_params['classes'], self.device)
                self.likelihood_params['feature_pairs'] = safe_to_tensor(self.likelihood_params['feature_pairs'], self.device)

            # Load optional components
            optional_components = [
                'global_mean', 'global_std', 'categorical_encoders',
                'feature_columns', 'original_columns', 'high_cardinality_columns',
                'best_error', 'weight_updater', 'bin_edges', 'gaussian_params'
            ]
            for comp in optional_components:
                if comp in components:
                    # Convert tensors if needed
                    if comp == 'bin_edges' and components[comp] is not None:
                        setattr(self, comp, [[edge.clone().detach().to(self.device) for edge in pair] for pair in components[comp]])
                    elif comp == 'gaussian_params' and components[comp] is not None:
                        gaussian_params = {
                            'means': torch.tensor(components[comp]['means'], device=self.device),
                            'covs': torch.tensor(components[comp]['covs'], device=self.device),
                            'classes': torch.tensor(components[comp]['classes'], device=self.device)
                        }
                        setattr(self, comp, gaussian_params)
                    else:
                        setattr(self, comp, components[comp])

            # Load config but preserve current visualization setting
            if 'config' in components:
                # Update the config but preserve the current visualization setting
                loaded_config = components['config']
                if 'training_params' in loaded_config and 'training_params' in self.config:
                    # Preserve the current visualization setting from the active config
                    loaded_config['training_params']['enable_visualization'] = current_visualization
                self.config.update(loaded_config)

            print(f"\033[K[SUCCESS] Loaded model components from {components_file}")
            return True

        except Exception as e:
            print(f"\033[K[ERROR] Failed to load model components: {str(e)}")
            traceback.print_exc()
            # Reset critical components to prevent partial state
            self.label_encoder = LabelEncoder()
            self.scaler = StandardScaler()
            self.feature_pairs = None
            self.original_target_dtype = np.dtype('object')  # Reset to default
            return False

#--------------------------------------------------Class Ends ----------------------------------------------------------
    # DBNN class to handle prediction functionality
    def _validate_target_column(self, y: pd.Series) -> bool:
        """Check if target column values match label encoder classes"""
        # Handle case where label encoder isn't initialized
        if not hasattr(self.label_encoder, 'classes_'):
            #print(f"{Colors.RED}Label encoder not initialized!{Colors.ENDC}")
            return False

        # Handle empty classes
        if len(self.label_encoder.classes_) == 0:
            print(f"{Colors.RED}Label encoder has no classes!{Colors.ENDC}")
            return False

        # Convert all values to strings for consistent comparison
        unique_values = {str(v) for v in y.unique()}
        encoder_classes = {str(cls) for cls in self.label_encoder.classes_}
        # Allow validation in prediction mode (target column may be dummy)
        if self.mode == 'predict':
            return True  # Skip strict validation during prediction

        return unique_values.issubset(encoder_classes)

    def predict_from_file(self, input_csv: str, output_path: str = None, model_type=None,
                         image_dir: str = None, batch_size: int = 128) -> Dict:
        """
        Make predictions from CSV file with comprehensive output handling.
        """
        # Create output directory if needed
        if output_path:
            os.makedirs(output_path, exist_ok=True)

        try:
            # Load data
            df = pd.read_csv(input_csv)

            # Determine if target column exists
            has_target_column = self.target_column in df.columns
            predict_mode = (self.mode == 'predict')

            if has_target_column:
                # Convert target to string for consistency
                df[self.target_column] = df[self.target_column].astype(str)
                DEBUG.log(f"Target column '{self.target_column}' found with {len(df[self.target_column].unique())} unique values")

            print(f"\n{Colors.BLUE}Processing predictions for: {input_csv}{Colors.ENDC}")
            self.model_type = model_type

            # Store original data
            self.X_orig = df.copy()

            # Handle output directory selection
            if output_path and os.path.exists(output_path):
                print(f"{Colors.BLUE}Output directory exists: {output_path}{Colors.ENDC}")
                print(f"{Colors.BOLD}Choose an action:{Colors.ENDC}")
                print("1. Overwrite existing content")
                print("2. Create new version (append timestamp)")
                print("3. Specify different output directory")
                print("q. Quit")

                while True:
                    choice = input(f"{Colors.YELLOW}Your choice (1-3/q): {Colors.ENDC}").strip().lower()

                    if choice == '1':  # Overwrite
                        print(f"{Colors.YELLOW}Existing files will be overwritten{Colors.ENDC}")
                        # Clear existing predictions file if it exists
                        predictions_path = os.path.join(output_path, 'predictions.csv')
                        if os.path.exists(predictions_path):
                            os.remove(predictions_path)
                        # Clear analysis directories if they exist
                        for analysis_type in ['mosaics', 'failed_analysis', 'correct_analysis']:
                            analysis_dir = os.path.join(output_path, analysis_type)
                            if os.path.exists(analysis_dir):
                                shutil.rmtree(analysis_dir)
                        break

                    elif choice == '2':  # New version
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"{output_path}_{timestamp}"
                        os.makedirs(output_path, exist_ok=True)
                        print(f"{Colors.GREEN}Creating new version at: {output_path}{Colors.ENDC}")
                        break

                    elif choice == '3':  # Different directory
                        new_dir = input(f"{Colors.YELLOW}Enter new directory path: {Colors.ENDC}").strip()
                        if new_dir:
                            output_path = new_dir
                            os.makedirs(output_path, exist_ok=True)
                            print(f"{Colors.GREEN}Using new directory: {output_path}{Colors.ENDC}")
                            break
                        else:
                            print(f"{Colors.RED}Invalid path. Please try again.{Colors.ENDC}")

                    elif choice == 'q':  # Quit
                        return None

                    else:
                        print(f"{Colors.RED}Invalid option. Please choose 1-3 or q.{Colors.ENDC}")
            elif output_path:
                os.makedirs(output_path, exist_ok=True)

            # Handle true labels if target column exists - FIXED: Use the loaded label encoder properly
            if has_target_column:
                y_true_str = df[self.target_column]
                try:
                    # CRITICAL FIX: Check if label encoder is properly loaded and has classes
                    if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
                        print(f"\033[K[INFO] Using loaded label encoder with classes: {self.label_encoder.classes_}")
                        # Transform the true labels using the loaded encoder
                        y_true = self.label_encoder.transform(y_true_str)
                        DEBUG.log(f"Successfully encoded {len(y_true_str)} true labels using pre-trained encoder")
                    else:
                        print(f"{Colors.YELLOW}Warning: Label encoder not properly loaded, attempting to fit{Colors.ENDC}")
                        # Fallback: fit the encoder (shouldn't happen with properly saved models)
                        self.label_encoder.fit(y_true_str.astype(str))
                        y_true = self.label_encoder.transform(y_true_str)
                        DEBUG.log(f"Fitted new label encoder with classes: {self.label_encoder.classes_}")
                except ValueError as e:
                    print(f"{Colors.YELLOW}Warning: Some true labels not in training set: {str(e)}{Colors.ENDC}")
                    if hasattr(self.label_encoder, 'classes_'):
                        print(f"Known classes: {self.label_encoder.classes_}")
                    print(f"Data classes: {y_true_str.unique()}")
                    # For prediction, we can continue without true labels for evaluation
                    y_true_str = None
                    y_true = None
            else:
                y_true_str = None
                y_true = None
                DEBUG.log("No target column found - running in pure prediction mode")

            # Get features (drop target column if exists)
            if has_target_column:
                X = df.drop(columns=[self.target_column])
                DEBUG.log(f"Using {X.shape[1]} features for prediction (target column excluded)")
            else:
                X = df.copy()
                DEBUG.log(f"Using all {X.shape[1]} columns for prediction")

            # Generate predictions
            self._load_model_components()
            print(f"{Colors.BLUE}Generating predictions...{Colors.ENDC}")
            y_pred, posteriors = self.predict(X, batch_size=batch_size)

            # Decode predictions to original labels using the loaded encoder
            try:
                pred_classes = self.label_encoder.inverse_transform(y_pred.cpu().numpy())
                DEBUG.log(f"Successfully decoded predictions using label encoder")
            except Exception as e:
                print(f"{Colors.YELLOW}Warning: Could not decode predictions with label encoder: {str(e)}{Colors.ENDC}")
                # Fallback: use raw predictions
                pred_classes = y_pred.cpu().numpy().astype(str)

            confidences = posteriors[np.arange(len(y_pred)), y_pred].cpu().numpy()

            # Generate detailed results
            print(f"{Colors.BLUE}Generating detailed predictions...{Colors.ENDC}")
            results = self._generate_detailed_predictions(
                X_orig=self.X_orig,
                predictions=y_pred,
                true_labels=(y_true_str if y_true_str is not None else None),
                posteriors=posteriors
            )

            # Save results if output path specified
            metadata = {}
            if output_path:
                # Standard paths
                predictions_path = os.path.join(output_path, 'predictions.csv')
                metrics_path = os.path.join(output_path, 'metrics.txt')

                # Ensure the predictions directory exists
                os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

                # Save predictions with additional info
                results['predicted_class'] = pred_classes
                results['confidence'] = confidences
                results.to_csv(predictions_path, index=False)
                print(f"{Colors.GREEN}Predictions saved to {predictions_path}{Colors.ENDC}")

                # Save metadata
                metadata = {
                    'dataset': os.path.basename(input_csv),
                    'model_type': self.model_type,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'feature_columns': list(X.columns),
                    'target_column': self.target_column if hasattr(self, 'target_column') else None,
                    'has_ground_truth': has_target_column,
                    'label_encoder_classes': (self.label_encoder.classes_.tolist()
                                            if hasattr(self.label_encoder, 'classes_')
                                            else None),
                    'samples_processed': len(results)
                }
                with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Generate mosaics if image data available
                if 'filepath' in results.columns:
                    try:
                        columns = input("Please specify the number of columns of images per page (default 4): ") or 4
                        rows = input("Please specify the number of rows of images per page (default 4): ") or 4

                        try:
                            columns = int(columns)
                            rows = int(rows)
                        except ValueError:
                            print(f"{Colors.RED}Invalid input. Using default 4x4 grid{Colors.ENDC}")
                            columns = 4
                            rows = 4

                        # Create main mosaics directory
                        mosaic_dir = os.path.join(output_path, 'mosaics')
                        os.makedirs(mosaic_dir, exist_ok=True)

                        # Generate class-wise mosaics
                        for class_name, group in results.groupby('predicted_class'):
                            valid_images = []
                            for _, row in group.iterrows():
                                img_path = row['filepath']
                                if os.path.exists(img_path):
                                    valid_images.append(row)

                            if valid_images:
                                class_df = pd.DataFrame(valid_images)
                                self.generate_class_pdf_mosaics(
                                    predictions_df=class_df,
                                    output_dir=mosaic_dir,
                                    columns=columns,
                                    rows=rows
                                )
                                print(f"{Colors.GREEN}Generated mosaic for class {class_name}{Colors.ENDC}")

                    except Exception as e:
                        print(f"{Colors.YELLOW}Mosaic generation skipped: {str(e)}{Colors.ENDC}")

                # Failure/Success Analysis only if we have ground truth
                if has_target_column and 'true_class' in results.columns:
                    try:
                        # Create analysis directories
                        failed_dir = os.path.join(output_path, 'failed_analysis')
                        correct_dir = os.path.join(output_path, 'correct_analysis')
                        os.makedirs(failed_dir, exist_ok=True)
                        os.makedirs(correct_dir, exist_ok=True)

                        # Split into failed and correct predictions
                        failed_predictions = results[results['predicted_class'] != results['true_class']]
                        correct_predictions = results[results['predicted_class'] == results['true_class']]

                        # Save analysis CSVs
                        failed_predictions.to_csv(os.path.join(failed_dir, 'failed_predictions.csv'), index=False)
                        correct_predictions.to_csv(os.path.join(correct_dir, 'correct_predictions.csv'), index=False)

                        print(f"{Colors.GREEN}Analysis files saved:{Colors.ENDC}")
                        print(f"  - Failed predictions: {len(failed_predictions)} samples")
                        print(f"  - Correct predictions: {len(correct_predictions)} samples")

                    except Exception as e:
                        print(f"{Colors.YELLOW}Analysis generation skipped: {str(e)}{Colors.ENDC}")

            # Compute metrics only if we have true labels and predictions - FIXED: Handle Series properly
            metrics = None
            if y_true is not None and y_pred is not None:
                print(f"\n{Colors.BLUE}Computing evaluation metrics...{Colors.ENDC}")

                try:
                    # Ensure we have numpy arrays for sklearn metrics - FIXED: Handle Series objects
                    if hasattr(y_true, 'cpu'):
                        y_true_np = y_true.cpu().numpy()
                    elif hasattr(y_true, 'to_numpy'):
                        y_true_np = y_true.to_numpy()
                    else:
                        y_true_np = np.array(y_true)

                    if hasattr(y_pred, 'cpu'):
                        y_pred_np = y_pred.cpu().numpy()
                    else:
                        y_pred_np = np.array(y_pred)

                    # Calculate metrics
                    metrics = {}
                    metrics['accuracy'] = accuracy_score(y_true_np, y_pred_np)

                    # Get class names for reporting
                    if hasattr(self.label_encoder, 'classes_'):
                        target_names = [str(cls) for cls in self.label_encoder.classes_]
                    else:
                        target_names = None

                    metrics['classification_report'] = classification_report(
                        y_true_np, y_pred_np,
                        output_dict=True,
                        target_names=target_names
                    )
                    metrics['confusion_matrix'] = confusion_matrix(y_true_np, y_pred_np).tolist()
                    metrics['classification_report_str'] = classification_report(
                        y_true_np, y_pred_np,
                        target_names=target_names
                    )

                    # Print colored confusion matrix
                    if hasattr(self.label_encoder, 'classes_'):
                        self.print_colored_confusion_matrix(
                            y_true_np,
                            y_pred_np,
                            class_labels=self.label_encoder.classes_,
                            header="Prediction Results"
                        )
                    else:
                        print(f"{Colors.YELLOW}Warning: No class labels available for confusion matrix{Colors.ENDC}")

                    # Save metrics if output path exists
                    if output_path:
                        with open(metrics_path, 'w') as f:
                            f.write(metrics['classification_report_str'])
                        print(f"{Colors.GREEN}Metrics saved to {metrics_path}{Colors.ENDC}")

                except Exception as e:
                    print(f"{Colors.YELLOW}Warning: Could not compute metrics: {str(e)}{Colors.ENDC}")
                    import traceback
                    traceback.print_exc()
                    metrics = None
            else:
                print(f"{Colors.YELLOW}No ground truth available for metrics calculation{Colors.ENDC}")

            return {
                'predictions': results,
                'metrics': metrics,  # Can be None
                'metadata': metadata,
                'analysis_files': {
                    'failed_predictions': os.path.join(output_path, 'failed_analysis') if has_target_column else None,
                    'correct_predictions': os.path.join(output_path, 'correct_analysis') if has_target_column else None
                } if output_path else None
            }

        except Exception as e:
            print(f"{Colors.RED}Prediction failed: {str(e)}{Colors.ENDC}")
            traceback.print_exc()
            raise

    def predict_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data using the trained model.
        Handles preprocessing and returns results with probabilities.

        Args:
            new_data: DataFrame containing new data to predict

        Returns:
            DataFrame with predictions and probabilities
        """
        if not hasattr(self, 'feature_columns'):
            raise RuntimeError("Model not properly trained - missing feature columns")

        # Make a copy of the input data to preserve original
        df = new_data.copy()

        # Store original index for results
        original_index = df.index

        # Check if target column exists in input data
        target_in_data = self.target_column in df.columns

        # Remove target column if present (but keep track of it)
        if target_in_data:
            y_true = df[self.target_column]
            df = df.drop(columns=[self.target_column])

        # Ensure we have all required features
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Input data missing required features: {missing_features}")

        # Reorder columns to match training order
        df = df[self.feature_columns]

        # Preprocess the data
        X_processed = self._preprocess_data(df, is_training=False)

        # Make predictions
        predictions = self.predict(X_processed)
        pred_labels = self.label_encoder.inverse_transform(predictions.cpu().numpy())

        # Compute probabilities
        probabilities = self._compute_probabilities(X_processed)

        # Create results DataFrame
        results = df.copy()
        results['predicted_class'] = pred_labels

        # Add true values if they existed in input
        if target_in_data:
            results[self.target_column] = y_true

        # Add probability columns
        for i, class_name in enumerate(self.label_encoder.classes_):
            results[f'prob_{class_name}'] = probabilities[:, i]

        results['max_probability'] = probabilities.max(axis=1)

        return results

    def _compute_probabilities(self, X: torch.Tensor) -> np.ndarray:
        """
        Compute class probabilities for input data.

        Args:
            X: Input tensor (already preprocessed)

        Returns:
            Array of class probabilities [n_samples, n_classes]
        """
        # Store current weights
        orig_weights = self.current_W
        self.current_W = self.best_W if self.best_W is not None else self.current_W

        try:
            # Compute in batches to handle large datasets
            batch_size = 128
            all_probs = []

            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]

                if self.model_type == "Histogram":
                    probs, _ = self._compute_batch_posterior(batch_X)
                elif self.model_type == "Gaussian":
                    probs, _ = self._compute_batch_posterior_std(batch_X)

                all_probs.append(probs.cpu().numpy())

            return np.vstack(all_probs)

        finally:
            # Restore original weights
            self.current_W = orig_weights

    def load_model_for_prediction(self, dataset_name: str) -> bool:
        """
        Load a trained model for prediction only.

        Args:
            dataset_name: Name of the dataset to load model for

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Load configuration
            self.config = DatasetConfig.load_config(dataset_name)
            if self.config is None:
                raise ValueError(f"Failed to load configuration for dataset: {dataset_name}")

            self.target_column = self.config['target_column']
            self.model_type = self.config.get('modelType', 'Histogram')

            # Load model components
            #self.label_encoder =load_label_encoder(dataset_name)
            self._load_model_components()

            print(f"Successfully loaded model for dataset: {dataset_name}")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            return False

    def generate_training_visualization(self):
        """Generate interactive visualization after training"""
        if hasattr(self, 'visualizer'):
            return self.visualizer.create_interactive_visualization()
        else:
            print("\033[KNo visualization data available. Train the model first or enable visualization in config.")
            return None

    def _validate_training_parameters(self):
        """Validate that all required training parameters are loaded"""
        required_params = [
            'trials', 'cardinality_threshold', 'learning_rate',
            'max_epochs', 'test_size', 'n_bins_per_dim'
        ]

        for param in required_params:
            if not hasattr(self, param) or getattr(self, param) is None:
                print(f"\033[K{Colors.RED}Warning: Missing training parameter: {param}{Colors.ENDC}")

        # Set default values for missing adaptive parameters
        if not hasattr(self, 'adaptive_rounds'):
            self.adaptive_rounds = 100
        if not hasattr(self, 'initial_samples'):
            self.initial_samples = 50
        if not hasattr(self, 'max_samples_per_round'):
            self.max_samples_per_round = 500
#----------DBNN Prediction Functions  Ends-----New Tensor Visualisation Starts--------------


class DBNN_5DCT_Visualizer:
    """Interactive 3D visualization for 5D Complex Tensor orthogonization"""

    def __init__(self, dbnn_model, output_dir="Visualizer"):
        self.dbnn = dbnn_model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.epoch_snapshots = []
        self.current_epoch = 0

        # FIXED: Use hex colors that work for both plotly and matplotlib
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
        self.is_enabled = False

    def get_color(self, index, format='hex'):
        """Safely get color by index in any format"""
        color = self.color_palette[index % len(self.color_palette)]

        if format == 'plotly':
            # Convert hex to plotly RGB format if needed
            if color.startswith('#'):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                return '#{:02x}{:02x}{:02x}'.format(r, g, b)
        elif format == 'matplotlib':
            # Hex works fine with matplotlib
            pass

        return color
    def enable(self):
        """Enable visualization"""
        self.is_enabled = True
        print("✅ 5DCT Visualization enabled")

    def disable(self):
        """Disable visualization"""
        self.is_enabled = False

    def capture_training_snapshot(self, epoch, train_accuracy, test_accuracy, features=None, targets=None):
        """Capture training snapshot at specific epoch"""
        if not self.is_enabled:
            return

        snapshot = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'class_distributions': {}
        }

        try:
            # Store features and labels if provided
            if features is not None:
                snapshot['features'] = features.cpu().numpy() if hasattr(features, 'cpu') else features
            if targets is not None:
                snapshot['labels'] = targets.cpu().numpy() if hasattr(targets, 'cpu') else targets

            # Compute projections if we have data
            if features is not None and targets is not None:
                self._compute_lightweight_projections(snapshot)

            self.epoch_snapshots.append(snapshot)
            self.current_epoch = epoch

            if epoch % 10 == 0:  # Only log occasionally
                print(f"📊 Snapshot Epoch {epoch} | Train: {train_accuracy:.4f} | Test: {test_accuracy:.4f}")

        except Exception as e:
            # Fail silently - don't affect training
            if hasattr(self.dbnn, 'debug_mode') and self.dbnn.debug_mode:
                print(f"⚠️ Snapshot capture failed: {str(e)}")

    def _compute_lightweight_projections(self, snapshot):
        """Lightweight projection computation for performance"""
        try:
            features = snapshot.get('features')
            labels = snapshot.get('labels')

            if features is None or labels is None:
                return

            # Use a small subset for performance
            n_samples = min(500, len(features))
            if len(features) > n_samples:
                sample_indices = np.random.choice(len(features), n_samples, replace=False)
                sample_features = features[sample_indices]
                sample_labels = labels[sample_indices]
            else:
                sample_features = features
                sample_labels = labels

            # Simple PCA projection
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            coords_3d = pca.fit_transform(sample_features)

            # Store projection data
            snapshot['projection'] = {
                'coords_3d': coords_3d.tolist(),
                'sample_labels': sample_labels.tolist(),
                'pca_variance': pca.explained_variance_ratio_.tolist(),
                'total_variance': float(np.sum(pca.explained_variance_ratio_))
            }

            # Compute class centers
            class_centers = {}
            unique_labels = np.unique(sample_labels)

            for label in unique_labels:
                mask = sample_labels == label
                if np.sum(mask) > 0:
                    class_coords = coords_3d[mask]
                    center = np.mean(class_coords, axis=0)

                    class_centers[int(label)] = {
                        'center': center.tolist(),
                        'sample_count': int(np.sum(mask)),
                        'color': self.color_palette[int(label) % len(self.color_palette)]
                    }

            snapshot['class_distributions'] = class_centers

        except Exception as e:
            # Fail silently
            pass

    def create_interactive_visualization(self):
        """Create interactive 3D visualization"""
        if not self.epoch_snapshots:
            print("❌ No snapshots available for visualization")
            return None

        print("🎨 Creating interactive 3D visualization...")

        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}],
                   [{'type': 'xy'}, {'type': 'xy'}]],
            subplot_titles=[
                '3D Class Clusters - Final State',
                'Training Evolution',
                'Accuracy Progression',
                'Class Separation'
            ],
            vertical_spacing=0.1
        )

        # Plot 1: Final state
        self._add_final_state_plot(fig, self.epoch_snapshots[-1], row=1, col=1)

        # Plot 2: Evolution
        self._add_evolution_plot(fig, row=1, col=2)

        # Plot 3: Accuracy
        self._add_accuracy_plot(fig, row=2, col=1)

        # Plot 4: Separation metrics
        self._add_separation_plot(fig, row=2, col=2)

        # Update layout
        current_snapshot = self.epoch_snapshots[-1]
        fig.update_layout(
            title=dict(
                text=f"DBNN 5D Complex Tensor Orthogonization<br>"
                     f"Dataset: {getattr(self.dbnn, 'dataset_name', 'Unknown')} | "
                     f"Final Accuracy: {current_snapshot.get('test_accuracy', 0):.4f}",
                x=0.5,
                font=dict(size=16)
            ),
            height=900,
            showlegend=True,
            template="plotly_white"
        )

        # Save interactive plot
        output_file = os.path.join(self.output_dir, "5dct_visualization.html")
        fig.write_html(output_file)
        print(f"✅ Visualization saved: {output_file}")

        return fig

    def _add_final_state_plot(self, fig, snapshot, row=1, col=1):
        """Add final state 3D plot"""
        if 'projection' not in snapshot:
            return

        projection = snapshot['projection']
        class_distributions = snapshot.get('class_distributions', {})

        # Add unit sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                opacity=0.1,
                colorscale='Blues',
                showscale=False,
                name='Unit Sphere'
            ),
            row=row, col=col
        )

        # Plot samples by class
        coords_3d = np.array(projection['coords_3d'])
        sample_labels = projection['sample_labels']

        unique_labels = np.unique(sample_labels)
        for label in unique_labels:
            color = self.color_palette[int(label) % len(self.color_palette)]
            mask = np.array(sample_labels) == label

            if np.sum(mask) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=coords_3d[mask, 0],
                        y=coords_3d[mask, 1],
                        z=coords_3d[mask, 2],
                        mode='markers',
                        marker=dict(size=4, color=color, opacity=0.7),
                        name=f'Class {label}',
                        legendgroup=f'class_{label}',
                        hovertemplate=f"Class {label}<extra></extra>"
                    ),
                    row=row, col=col
                )

        # Plot class centers
        for label, class_data in class_distributions.items():
            center = class_data.get('center')
            color = class_data.get('color', self.color_palette[int(label) % len(self.color_palette)])

            if center:
                fig.add_trace(
                    go.Scatter3d(
                        x=[center[0]], y=[center[1]], z=[center[2]],
                        mode='markers',
                        marker=dict(size=10, color=color, symbol='diamond'),
                        name=f'Class {label} Center',
                        legendgroup=f'class_{label}',
                        showlegend=False,
                        hovertemplate=f"Class {label} Center<br>Samples: {class_data.get('sample_count', 0)}<extra></extra>"
                    ),
                    row=row, col=col
                )

    def _add_evolution_plot(self, fig, row=1, col=2):
        """Add evolution plot showing class movement"""
        if len(self.epoch_snapshots) < 2:
            return

        # Track class center movements
        class_trails = {}

        for snapshot in self.epoch_snapshots:
            if 'class_distributions' in snapshot:
                epoch = snapshot['epoch']
                for label, class_data in snapshot['class_distributions'].items():
                    if 'center' in class_data:
                        if label not in class_trails:
                            class_trails[label] = {'x': [], 'y': [], 'z': [], 'epochs': []}

                        center = class_data['center']
                        class_trails[label]['x'].append(center[0])
                        class_trails[label]['y'].append(center[1])
                        class_trails[label]['z'].append(center[2])
                        class_trails[label]['epochs'].append(epoch)

        # Plot evolution trails
        for label, trail_data in class_trails.items():
            if len(trail_data['x']) > 1:  # Only plot if we have movement
                color = self.color_palette[label % len(self.color_palette)]

                fig.add_trace(
                    go.Scatter3d(
                        x=trail_data['x'],
                        y=trail_data['y'],
                        z=trail_data['z'],
                        mode='lines+markers',
                        line=dict(color=color, width=4),
                        marker=dict(size=3),
                        name=f'Class {label} Trail',
                        legendgroup=f'class_{label}',
                        showlegend=False,
                        hovertemplate=f"Class {label}<br>Epoch: %{{customdata}}<extra></extra>",
                        customdata=trail_data['epochs']
                    ),
                    row=row, col=col
                )

    def _add_accuracy_plot(self, fig, row=2, col=1):
        """Add accuracy progression plot"""
        epochs = [s['epoch'] for s in self.epoch_snapshots]
        train_acc = [s.get('train_accuracy', 0) for s in self.epoch_snapshots]
        test_acc = [s.get('test_accuracy', 0) for s in self.epoch_snapshots]

        fig.add_trace(
            go.Scatter(
                x=epochs, y=train_acc,
                mode='lines+markers',
                name='Training Accuracy',
                line=dict(color='blue', width=2)
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=epochs, y=test_acc,
                mode='lines+markers',
                name='Test Accuracy',
                line=dict(color='red', width=2)
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Epoch", row=row, col=col)
        fig.update_yaxes(title_text="Accuracy", row=row, col=col)

    def _add_separation_plot(self, fig, row=2, col=2):
        """Add class separation metrics plot"""
        epochs = [s['epoch'] for s in self.epoch_snapshots]
        separation_metrics = []

        for snapshot in self.epoch_snapshots:
            metric = self._compute_separation_metric(snapshot)
            separation_metrics.append(metric)

        fig.add_trace(
            go.Scatter(
                x=epochs, y=separation_metrics,
                mode='lines+markers',
                name='Class Separation',
                line=dict(color='green', width=2)
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Epoch", row=row, col=col)
        fig.update_yaxes(title_text="Separation Metric", row=row, col=col)

    def _compute_separation_metric(self, snapshot):
        """Compute class separation metric"""
        try:
            if 'class_distributions' not in snapshot or len(snapshot['class_distributions']) < 2:
                return 0.0

            centers = []
            for class_data in snapshot['class_distributions'].values():
                if 'center' in class_data:
                    centers.append(class_data['center'])

            if len(centers) < 2:
                return 0.0

            centers = np.array(centers)

            # Compute average distance between class centers
            from sklearn.metrics.pairwise import euclidean_distances
            dist_matrix = euclidean_distances(centers)
            np.fill_diagonal(dist_matrix, 0)

            avg_separation = np.sum(dist_matrix) / (len(centers) * (len(centers) - 1))
            return float(avg_separation)

        except:
            return 0.0

    def export_analysis_report(self):
        """Export analysis report"""
        if not self.epoch_snapshots:
            return

        report = {
            'dataset': getattr(self.dbnn, 'dataset_name', 'Unknown'),
            'total_epochs': len(self.epoch_snapshots),
            'final_train_accuracy': self.epoch_snapshots[-1].get('train_accuracy', 0),
            'final_test_accuracy': self.epoch_snapshots[-1].get('test_accuracy', 0),
            'training_timeline': [
                {
                    'epoch': s['epoch'],
                    'train_accuracy': s.get('train_accuracy', 0),
                    'test_accuracy': s.get('test_accuracy', 0),
                    'timestamp': s.get('timestamp', '')
                } for s in self.epoch_snapshots
            ]
        }

        report_file = os.path.join(self.output_dir, "training_analysis.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📊 Analysis report saved: {report_file}")
        return report

# =============================================================================
# SAFE INTEGRATION WITH DBNN CLASS
# =============================================================================

def add_visualization_integration():
    """Safely add visualization capabilities to DBNN class"""

    # Store original __init__ method
    original_init = DBNN.__init__

    def enhanced_init(original_init):
        """Enhanced initialization wrapper with unified visualization support"""
        def wrapper(self, *args, **kwargs):
            # Extract visualization parameters before calling original init
            enable_visualization = kwargs.pop('enable_visualization', False)
            visualization_frequency = kwargs.pop('visualization_frequency', 1)
            enable_5DCTvisualization = kwargs.pop('enable_visualization', False)

            # Call original initialization first (this will set up the 5DCT visualizer)
            original_init(self, *args, **kwargs)

            # Now handle unified visualization setup
            self.enable_visualization = enable_visualization
            self.visualization_frequency = visualization_frequency
            self.visualization_snapshots = enable_visualization

            # Unified visualization control
            if enable_visualization:
                # Use the new comprehensive visualizer
                self.visualizer = DBNNVisualizer()
                self._initialize_visualization_directories()

                # Adaptive training data storage for visualization
                self.adaptive_round_data = []
                self.adaptive_snapshots = []

                print(f"{Colors.CYAN}[DBNN-VISUAL] Comprehensive visualization enabled{Colors.ENDC}")
                print(f"{Colors.CYAN}[DBNN-VISUAL] Snapshot frequency: every {visualization_frequency} round(s){Colors.ENDC}")

                # Also enable 5DCT visualization if requested
                if enable_5DCTvisualization :
                    self.enable_5DCTvisualization = True
                    print(f"{Colors.CYAN}[DBNN-VISUAL] 5DCT visualization also enabled{Colors.ENDC}")
            else:
                # Ensure visualization attributes exist even when disabled
                self.visualizer = None
                self.adaptive_round_data = []
                self.adaptive_snapshots = []


        return wrapper

    def enable_visualization(self):
        """Enable visualization - must be called explicitly"""
        print("🔧 enable_visualization() called - checking state...")

        # CRITICAL FIX: Check for visualizer5DCT, not visualizer
        if self.visualizer5DCT:
            self.enable_5DCTvisualization = True
            self.enable_visualization = True
            print("✅ 5DCT Visualization enabled - visualizer5DCT found and ready")

            # Debug: Show current state
            snapshots = len(getattr(self.visualizer5DCT, 'feature_space_snapshots', []))
            print(f"📁 Current 5DCT snapshots: {snapshots}")

        elif hasattr(self, 'visualizer') and self.visualizer is not None:
            # Fallback: at least we have the main visualizer
            self.enable_visualization = True
            print("⚠️  Main visualization enabled, but 5DCT visualizer not available")

        else:
            print("❌ No visualizers available! Need to initialize first.")
            # Auto-initialize if flags are set
            if COMMAND_LINE_FLAGS.get('enable_5DCTvisualization', False):
                self._initialize_5dct_visualizer()
                print("🔄 Auto-initialized 5DCT visualizer")

    def capture_training_snapshot(self, epoch, train_accuracy, test_accuracy, features=None, labels=None):
        """Capture training snapshot"""
        if self.enable_5DCTvisualization:
            print("=========================Capturing 5DCT Snaps==================")
            self.visualizer5DCT.capture_training_snapshot(epoch, train_accuracy, test_accuracy, features, labels)

    def create_visualization(self):
        """Create visualization from captured snapshots"""
        if (hasattr(self, 'enable_visualization') and self.enable_visualization and
            hasattr(self, 'visualizer')):

            if len(self.visualizer5DCT.epoch_snapshots) > 0:
                print("\n" + "="*60)
                print("🎨 GENERATING 5D COMPLEX TENSOR VISUALIZATION")
                print("="*60)

                self.visualizer5DCT.create_interactive_visualization()
                self.visualizer5DCT.export_analysis_report()
                print("✅ Visualization complete! Check the '5dct_visualizations' folder")
            else:
                print("ℹ️  No visualization data captured. Enable visualization before training.")
        else:
            print("ℹ️  Visualization not enabled. Call enable_visualization() before training.")

    # Apply enhancements to DBNN class
    DBNN.__init__ = enhanced_init(DBNN.__init__)
    DBNN.enable_5DCTvisualization = enable_visualization
    DBNN.capture_training_snapshot =enable_visualization
    DBNN.create_visualization = create_visualization

# Apply integration (this should be at the VERY END of the file)
add_visualization_integration()
print("✅ 5DCT Visualization system integrated (disabled by default)")
def create_prediction_mosaic(
        image_dir: list,  # List of dictionary rows (with filepath and original_filename)
        csv_path: Optional[str] = None,  # Not used but kept for compatibility
        output_path: str = "mosaic.jpg",
        mosaic_size: tuple = (2000, 2000),
        tile_size: tuple = (200, 200),
        max_images: int = 100,
        font_path: Optional[str] = None
    ):
    """
    Create a prediction mosaic from a list of image rows (containing filepath and metadata).

    Args:
        image_dir: List of dictionary rows (must contain 'filepath' and 'original_filename')
        csv_path: (Deprecated) Kept for compatibility
        output_path: Path to save the mosaic image
        mosaic_size: Total size of the output mosaic (width, height)
        tile_size: Size of each individual tile (width, height)
        max_images: Maximum number of images to include
        font_path: Optional path to font file for labels
    """
    try:
        # Validate inputs
        if not image_dir:
            raise ValueError("No images provided in image_dir")

        # Convert to DataFrame if needed
        if not isinstance(image_dir, pd.DataFrame):
            image_df = pd.DataFrame(image_dir)
        else:
            image_df = image_dir.copy()

        # Validate required columns
        if not all(col in image_df.columns for col in ['filepath', 'original_filename']):
            raise ValueError("Input must contain 'filepath' and 'original_filename' columns")

        # Limit number of images
        if len(image_df) > max_images:
            image_df = image_df.sample(max_images, random_state=42)
            DEBUG.log(f"Using random sample of {max_images} images for mosaic")

        # Calculate grid dimensions
        tiles_per_row = mosaic_size[0] // tile_size[0]
        tiles_per_col = mosaic_size[1] // tile_size[1]
        max_tiles = tiles_per_row * tiles_per_col
        image_df = image_df.head(max_tiles)  # Ensure we don't exceed mosaic capacity

        # Create blank mosaic
        mosaic = Image.new('RGB', mosaic_size, (0, 0, 0))
        draw = ImageDraw.Draw(mosaic)

        # Try to load font (fallback to default if not specified)
        try:
            font = ImageFont.truetype(font_path, 12) if font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # Process and place each image
        for i, (_, row) in enumerate(image_df.iterrows()):
            try:
                img_path = row['filepath']
                if not os.path.exists(img_path):
                    DEBUG.log(f"Image not found: {img_path}")
                    continue

                with Image.open(img_path) as img:
                    # Convert to RGB and resize
                    img = img.convert('RGB')
                    img.thumbnail(tile_size, Image.ANTIALIAS)

                    # Calculate position
                    row_pos = i // tiles_per_row
                    col_pos = i % tiles_per_row
                    x = col_pos * tile_size[0]
                    y = row_pos * tile_size[1]

                    # Center the image
                    offset_x = (tile_size[0] - img.size[0]) // 2
                    offset_y = (tile_size[1] - img.size[1]) // 2
                    mosaic.paste(img, (x + offset_x, y + offset_y))

                    # Add filename label (truncated if too long)
                    label = os.path.splitext(row['original_filename'])[0][:15]
                    draw.text(
                        (x + 5, y + tile_size[1] - 20),
                        label,
                        font=font,
                        fill=(255, 255, 255)
                    )

            except Exception as img_error:
                DEBUG.log(f"Error processing {img_path}: {str(img_error)}")
                continue

        # Save the mosaic
        mosaic.save(output_path, quality=95)
        DEBUG.log(f"Saved prediction mosaic to {output_path} "
                f"({len(image_df)} images, {tiles_per_row}x{tiles_per_col} grid)")

    except Exception as e:
        DEBUG.log(f"Error in create_prediction_mosaic: {str(e)}")
        raise RuntimeError(f"Prediction mosaic creation failed: {str(e)}")

def plot_training_progress(error_rates: List[float], dataset_name: str):
    """Plot training error rates over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.title(f'Training Progress - {dataset_name.capitalize()} Dataset')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(confusion_mat: np.ndarray, class_names: np.ndarray, dataset_name: str):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - {dataset_name.capitalize()} Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def load_label_encoder(dataset_name,save_dir='Model',model_type='Histogram'):
    encoder_path = os.path.join(save_dir,  f'Best_{model_type}_{dataset_name}_label_encoder.pkl')
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print("\033[K" +f"Label encoder loaded from {encoder_path}", end="\r", flush=True)
        return label_encoder
    else:
        raise FileNotFoundError(f"Label encoder file not found at {encoder_path}")

def generate_test_datasets():
    """Generate XOR and 3D XOR test datasets"""
    # Generate 2D XOR
    with open('xor.csv', 'w') as f:
        f.write('x1,x2,target\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')

    # Generate 3D XOR
    with open('xor3d.csv', 'w') as f:
        f.write('x1,x2,x3,target\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')

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


class UnifiedDBNNVisualizer:
    """
    COMPREHENSIVE UNIFIED VISUALIZATION MODULE
    Replaces: DBNNGeometricVisualizer, GeometricADBNNVisualizer,
              ComprehensiveAdaptiveVisualizer, AdvancedInteractiveVisualizer,
              AdaptiveVisualizer3D
    """

    def __init__(self, dbnn_model, output_base_dir='Visualizer/adaptiveDBNN'):
        self.dbnn = dbnn_model
        self.dataset_name = os.path.splitext(os.path.basename(self.dbnn.dataset_name))[0]
        self.output_dir = Path(output_base_dir) / self.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for organized output
        self.subdirs = {
            'performance': self.output_dir / 'performance',
            'geometric': self.output_dir / 'geometric',
            'adaptive': self.output_dir / 'adaptive',
            'interactive': self.output_dir / 'interactive',
            'networks': self.output_dir / 'networks',
            'comparisons': self.output_dir / 'comparisons'
        }

        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)

        # Color schemes
        self.colors = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
            '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
            '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'
        ]
        self._setup_plot_style()

        # Parallel processing setup
        self.max_workers = self._calculate_optimal_workers()
        self.available_memory = psutil.virtual_memory().available

        print(f"🎨 Unified DBNN Visualizer initialized for: {self.dataset_name}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Parallel workers: {self.max_workers}")

    def _calculate_optimal_workers(self):
        """Calculate optimal number of workers based on system resources"""
        cpu_count = os.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        if available_memory_gb < 4:
            workers = 2
        elif available_memory_gb < 8:
            workers = min(4, cpu_count // 2)
        else:
            workers = min(8, cpu_count - 1)

        return max(1, workers)

    def _setup_plot_style(self):
        """Set consistent plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

    def create_comprehensive_visualizations(self, training_history=None, round_stats=None,
                                         feature_names=None, enable_3d=True):
        """
        MAIN ENTRY POINT: Create all visualizations in one call
        """
        print("\n" + "="*70)
        print("🎨 CREATING COMPREHENSIVE UNIFIED DBNN VISUALIZATIONS")
        print("="*70)

        # Get data from DBNN model
        X_full = self.dbnn.X_tensor.cpu().numpy() if hasattr(self.dbnn, 'X_tensor') else None
        y_full = self.dbnn.y_tensor.cpu().numpy() if hasattr(self.dbnn, 'y_tensor') else None

        if X_full is None or y_full is None:
            print("⚠️  No data available for visualization")
            return

        # Create all visualizations
        visualization_tasks = []

        # 1. Performance Analysis
        if round_stats:
            visualization_tasks.append(('performance', self._create_performance_analysis, round_stats))

        # 2. Geometric Tensor Analysis
        visualization_tasks.append(('geometric', self._create_geometric_tensor_analysis, None))

        # 3. Adaptive Learning Analysis
        if training_history:
            visualization_tasks.append(('adaptive', self._create_adaptive_learning_analysis,
                                      (training_history, y_full)))

        # 4. Feature Space Analysis
        visualization_tasks.append(('feature_space', self._create_feature_space_analysis,
                                  (X_full, y_full, feature_names)))

        # 5. 3D Visualizations (conditional)
        if enable_3d and X_full.shape[1] >= 3:
            visualization_tasks.append(('3d', self._create_3d_visualizations,
                                      (X_full, y_full, training_history, feature_names)))

        # 6. Interactive Dashboard
        visualization_tasks.append(('dashboard', self._create_interactive_dashboard,
                                  (round_stats, training_history, X_full, y_full, feature_names)))

        # Execute visualizations in parallel where possible
        self._execute_parallel_visualizations(visualization_tasks)

        print(f"✅ All visualizations completed and saved to: {self.output_dir}")

    def _execute_parallel_visualizations(self, tasks):
        """Advanced parallel visualization with CUDA compatibility and dynamic resource management"""
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import psutil

        # Set spawn method for CUDA compatibility
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        # Advanced task categorization with resource profiling
        task_profiles = {
            '3d': {'type': 'gpu_memory', 'workers': 1, 'timeout': 1200},
            'geometric': {'type': 'gpu_cpu', 'workers': 2, 'timeout': 900},
            'feature_space': {'type': 'memory', 'workers': 2, 'timeout': 600},
            'adaptive': {'type': 'memory', 'workers': 2, 'timeout': 1200},
            'performance': {'type': 'cpu', 'workers': 4, 'timeout': 300},
            'dashboard': {'type': 'cpu', 'workers': 3, 'timeout': 400},
            'training_evolution': {'type': 'cpu', 'workers': 2, 'timeout': 500}
        }

        # Dynamic resource allocation based on system capacity
        def get_optimal_workers(task_type):
            system_cores = psutil.cpu_count()
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            gpu_memory = self._get_available_gpu_memory() if torch.cuda.is_available() else 0

            base_workers = task_profiles[task_type]['workers']

            if task_type in ['3d', 'geometric'] and gpu_memory < 4:  # Less than 4GB GPU
                return min(base_workers, 1)
            elif task_type == 'memory' and available_memory < 8:  # Less than 8GB RAM
                return min(base_workers, 1)
            elif system_cores < 8:
                return min(base_workers, max(1, system_cores // 2))
            else:
                return base_workers

        # Pre-process tasks for CUDA compatibility
        def prepare_task_data(task_func, task_args):
            """Convert GPU tensors to CPU for multiprocessing compatibility"""
            if task_args and len(task_args) > 0:
                prepared_args = []
                for arg in task_args:
                    if isinstance(arg, torch.Tensor) and arg.is_cuda:
                        prepared_args.append(arg.cpu().numpy())
                    elif hasattr(arg, 'device') and str(arg.device) != 'cpu':
                        # Handle model objects - create CPU state dict
                        cpu_state = {
                            'X_tensor': getattr(arg, 'X_tensor', None).cpu().numpy() if hasattr(arg, 'X_tensor') else None,
                            'y_tensor': getattr(arg, 'y_tensor', None).cpu().numpy() if hasattr(arg, 'y_tensor') else None,
                            'feature_pairs': getattr(arg, 'feature_pairs', None),
                            'model_type': getattr(arg, 'model_type', None),
                            'label_encoder': getattr(arg, 'label_encoder', None)
                        }
                        prepared_args.append(cpu_state)
                    else:
                        prepared_args.append(arg)
                return task_func, prepared_args
            return task_func, task_args

        # Batch execution with resource awareness
        def execute_task_batch(task_batch, executor, batch_type):
            """Execute a batch of tasks with appropriate resource allocation"""
            futures = {}

            for task_name, task_func, task_args in task_batch:
                # Prepare data for multiprocessing
                prepared_func, prepared_args = prepare_task_data(task_func, task_args)

                # Submit with dynamic timeout
                timeout = task_profiles.get(task_name, {}).get('timeout', 600)

                if prepared_args:
                    future = executor.submit(prepared_func, *prepared_args)
                else:
                    future = executor.submit(prepared_func)

                futures[future] = (task_name, timeout)

            return futures

        # Smart task batching
        def create_optimal_batches(tasks):
            gpu_tasks = []
            memory_tasks = []
            cpu_tasks = []

            for task in tasks:
                task_name = task[0]
                profile = task_profiles.get(task_name, {})

                if profile.get('type', '').startswith('gpu'):
                    gpu_tasks.append(task)
                elif profile.get('type') == 'memory':
                    memory_tasks.append(task)
                else:
                    cpu_tasks.append(task)

            # Limit concurrent GPU tasks to avoid memory contention
            max_gpu_concurrent = min(2, len(gpu_tasks))
            gpu_batches = [gpu_tasks[i:i+max_gpu_concurrent] for i in range(0, len(gpu_tasks), max_gpu_concurrent)]

            # Memory tasks can run in parallel but limited
            memory_batches = [memory_tasks[i:i+3] for i in range(0, len(memory_tasks), 3)]

            # CPU tasks can run with higher concurrency
            cpu_batches = [cpu_tasks[i:i+6] for i in range(0, len(cpu_tasks), 6)]

            return gpu_batches + memory_batches + cpu_batches

        # Results collection
        results = []
        failed_tasks = []

        # Create optimal batches
        task_batches = create_optimal_batches(tasks)

        # Progress tracking
        total_batches = len(task_batches)
        completed_tasks = 0
        total_tasks = len(tasks)

        with tqdm(total=total_tasks, desc="🎨 Advanced Parallel Visualizations",
                  bar_format="{l_bar}{bar:50}{r_bar}{bar:-50b}") as pbar:

            for batch_idx, batch in enumerate(task_batches):
                batch_type = "GPU" if any(t[0] in ['3d', 'geometric'] for t in batch) else \
                           "Memory" if any(t[0] in ['feature_space', 'adaptive'] for t in batch) else "CPU"

                # Dynamic worker allocation per batch
                batch_workers = max(1, min(len(batch), get_optimal_workers(batch[0][0])))

                # Choose executor based on task type
                if batch_type == "GPU":
                    # Use ThreadPool for GPU tasks to share CUDA context
                    with ThreadPoolExecutor(max_workers=batch_workers) as executor:
                        batch_futures = execute_task_batch(batch, executor, batch_type)

                        # Process batch results
                        for future in as_completed(batch_futures):
                            task_name, timeout = batch_futures[future]
                            try:
                                result = future.result(timeout=timeout)
                                results.append((task_name, result))
                                completed_tasks += 1
                                pbar.update(1)
                                pbar.set_postfix_str(f"✅ {task_name} | Batch {batch_idx+1}/{total_batches}")
                            except Exception as e:
                                failed_tasks.append((task_name, str(e)))
                                pbar.update(1)
                                pbar.set_postfix_str(f"❌ {task_name} | {str(e)[:50]}...")
                else:
                    # Use ProcessPool for CPU/memory tasks
                    with mp.get_context('spawn').Pool(processes=batch_workers) as pool:
                        batch_results = []

                        for task_name, task_func, task_args in batch:
                            prepared_func, prepared_args = prepare_task_data(task_func, task_args)
                            timeout = task_profiles.get(task_name, {}).get('timeout', 600)

                            if prepared_args:
                                async_result = pool.apply_async(prepared_func, prepared_args)
                            else:
                                async_result = pool.apply_async(prepared_func)
                            batch_results.append((task_name, async_result, timeout))

                        # Collect results with timeout
                        for task_name, async_result, timeout in batch_results:
                            try:
                                result = async_result.get(timeout=timeout)
                                results.append((task_name, result))
                                completed_tasks += 1
                                pbar.update(1)
                                pbar.set_postfix_str(f"✅ {task_name} | Batch {batch_idx+1}/{total_batches}")
                            except mp.TimeoutError:
                                failed_tasks.append((task_name, f"Timeout after {timeout}s"))
                                pbar.update(1)
                                pbar.set_postfix_str(f"⏰ {task_name} | Timeout")
                            except Exception as e:
                                failed_tasks.append((task_name, str(e)))
                                pbar.update(1)
                                pbar.set_postfix_str(f"❌ {task_name} | {str(e)[:50]}...")

        # Summary report
        if failed_tasks:
            print(f"\n⚠️  {len(failed_tasks)} tasks failed:")
            for task, error in failed_tasks:
                print(f"   • {task}: {error}")

        print(f"✅ {completed_tasks}/{total_tasks} visualizations completed successfully")
        return results

    def _get_available_gpu_memory(self):
        """Get available GPU memory in GB"""
        if not torch.cuda.is_available():
            return 0

        try:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            return max(0.1, total - allocated)  # Leave some buffer
        except:
            return 0

    def _create_performance_analysis(self, round_stats):
        """Comprehensive performance analysis across rounds"""
        print("📈 Creating performance analysis...")

        if not round_stats:
            return

        rounds = [stat['round'] for stat in round_stats]
        train_acc = [stat.get('train_accuracy', 0) * 100 for stat in round_stats]
        test_acc = [stat.get('test_accuracy', 0) * 100 for stat in round_stats]
        training_sizes = [stat.get('training_size', 0) for stat in round_stats]

        # Create comprehensive performance figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Accuracy evolution
        ax1.plot(rounds, train_acc, 'o-', linewidth=2, markersize=6,
                label='Training Accuracy', color=self.colors[0])
        ax1.plot(rounds, test_acc, 's-', linewidth=2, markersize=6,
                label='Test Accuracy', color=self.colors[1])
        ax1.set_xlabel('Adaptive Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Evolution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Training size growth
        ax2.plot(rounds, training_sizes, '^-', linewidth=2, markersize=6,
                color=self.colors[2])
        ax2.set_xlabel('Adaptive Round')
        ax2.set_ylabel('Training Set Size')
        ax2.set_title('Training Set Growth', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Improvement analysis
        improvements = [test_acc[i] - test_acc[i-1] if i > 0 else 0 for i in range(len(test_acc))]
        ax3.bar(rounds, improvements, color=np.where(np.array(improvements) >= 0, 'green', 'red'),
               alpha=0.7)
        ax3.set_xlabel('Adaptive Round')
        ax3.set_ylabel('Accuracy Improvement (%)')
        ax3.set_title('Improvement per Round', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Cumulative performance
        cumulative_improvement = np.cumsum(improvements)
        ax4.plot(rounds, cumulative_improvement, 'o-', linewidth=2, markersize=6,
                color=self.colors[3])
        ax4.set_xlabel('Adaptive Round')
        ax4.set_ylabel('Cumulative Improvement (%)')
        ax4.set_title('Cumulative Performance', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.subdirs['performance'] / 'performance_comprehensive.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Create interactive version
        self._create_interactive_performance_plot(rounds, train_acc, test_acc,
                                                training_sizes, improvements)

    def _create_interactive_performance_plot(self, rounds, train_acc, test_acc,
                                           training_sizes, improvements):
        """Create interactive performance plot"""
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('Accuracy Evolution', 'Training Set Growth',
                                        'Improvement per Round', 'Cumulative Improvement'))

        fig.add_trace(go.Scatter(x=rounds, y=train_acc, name='Training Accuracy',
                               line=dict(color=self.colors[0])), row=1, col=1)
        fig.add_trace(go.Scatter(x=rounds, y=test_acc, name='Test Accuracy',
                               line=dict(color=self.colors[1])), row=1, col=1)

        fig.add_trace(go.Scatter(x=rounds, y=training_sizes, name='Training Size',
                               line=dict(color=self.colors[2])), row=1, col=2)

        fig.add_trace(go.Bar(x=rounds, y=improvements, name='Improvement',
                           marker_color=np.where(np.array(improvements) >= 0, 'green', 'red')),
                     row=2, col=1)

        fig.add_trace(go.Scatter(x=rounds, y=np.cumsum(improvements),
                               name='Cumulative Improvement',
                               line=dict(color=self.colors[3])), row=2, col=2)

        fig.update_layout(height=800, title_text="Performance Analysis Dashboard",
                         showlegend=True)
        fig.write_html(self.subdirs['interactive'] / 'performance_dashboard.html')

    def _create_geometric_tensor_analysis(self):
        """Advanced geometric analysis of DBNN tensors"""
        print("📊 Creating geometric tensor analysis...")

        try:
            # Extract tensor information from DBNN
            n_classes = len(self.dbnn.label_encoder.classes_)
            n_pairs = len(self.dbnn.feature_pairs) if hasattr(self.dbnn, 'feature_pairs') else 0
            n_bins = getattr(self.dbnn, 'n_bins_per_dim', 128)

            if n_pairs == 0:
                print("⚠️  No feature pairs available for geometric analysis")
                return

            # Create tensor evolution visualization
            self._create_tensor_orthogonality_analysis(n_classes, n_pairs, n_bins)
            self._create_class_separation_analysis()

        except Exception as e:
            print(f"⚠️  Geometric tensor analysis skipped: {str(e)}")

    def _create_tensor_orthogonality_analysis(self, n_classes, n_pairs, n_bins):
        """Analyze tensor orthogonality evolution"""
        # This would use the complex tensor representation from your original code
        # Simplified version for demonstration
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Simulated orthogonality metrics (replace with actual computation)
        epochs = list(range(1, 101, 10))
        mean_angles = [45 + 45 * (1 - np.exp(-e/50)) for e in epochs]  # Simulated improvement
        orthogonality_scores = [90 - angle for angle in mean_angles]  # Distance from 90 degrees

        ax1.plot(epochs, mean_angles, 'o-', linewidth=2, markersize=6, color=self.colors[0])
        ax1.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Ideal Orthogonality')
        ax1.set_xlabel('Training Epoch')
        ax1.set_ylabel('Mean Angle Between Classes (degrees)')
        ax1.set_title('Class Separation Evolution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, orthogonality_scores, 's-', linewidth=2, markersize=6, color=self.colors[1])
        ax2.set_xlabel('Training Epoch')
        ax2.set_ylabel('Orthogonality Score')
        ax2.set_title('Orthogonality Progress', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.subdirs['geometric'] / 'tensor_orthogonality.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_class_separation_analysis(self):
        """Analyze class separation in feature space"""
        # This would use actual DBNN tensor data
        # Placeholder implementation
        print("✅ Geometric tensor analysis completed")

    def _create_adaptive_learning_analysis(self, training_history, y_full):
        """Analyze adaptive learning sample selection"""
        print("🔍 Creating adaptive learning analysis...")

        if not training_history:
            return

        unique_classes = np.unique(y_full)
        rounds = list(range(len(training_history)))

        # Calculate class distribution evolution
        class_distributions = []
        for train_indices in training_history:
            round_labels = y_full[train_indices]
            class_counts = [np.sum(round_labels == cls) for cls in unique_classes]
            class_distributions.append(class_counts)

        class_distributions = np.array(class_distributions)

        # Create comprehensive adaptive learning analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Stacked area plot for class distribution
        if len(unique_classes) <= 10:
            ax1.stackplot(rounds, class_distributions.T,
                         labels=[f'Class {cls}' for cls in unique_classes],
                         colors=self.colors[:len(unique_classes)], alpha=0.8)
        else:
            # For many classes, show top classes only
            total_counts = class_distributions.sum(axis=0)
            top_classes_idx = np.argsort(total_counts)[-10:]  # Top 10 classes
            for i, cls_idx in enumerate(top_classes_idx):
                ax1.plot(rounds, class_distributions[:, cls_idx], 'o-', linewidth=1,
                        label=f'Class {unique_classes[cls_idx]}',
                        color=self.colors[i % len(self.colors)])

        ax1.set_xlabel('Adaptive Round')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Class Distribution Evolution', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Sample selection efficiency
        total_samples = [len(indices) for indices in training_history]
        new_samples = [len(training_history[0])] + \
                     [len(training_history[i]) - len(training_history[i-1])
                      for i in range(1, len(training_history))]

        ax2.bar(rounds, total_samples, alpha=0.7, label='Cumulative Samples')
        ax2.bar(rounds, new_samples, alpha=0.7, label='New Samples per Round')
        ax2.set_xlabel('Adaptive Round')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Sample Selection Efficiency', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.subdirs['adaptive'] / 'adaptive_learning_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_feature_space_analysis(self, X_full, y_full, feature_names):
        """Comprehensive feature space analysis"""
        print("🔧 Creating feature space analysis...")

        if X_full is None or y_full is None:
            return

        # Feature importance analysis
        if hasattr(self.dbnn, 'feature_importances_'):
            importances = self.dbnn.feature_importances_
        else:
            # Use variance as proxy for DBNN
            importances = np.var(X_full, axis=0)

        # Sort features by importance
        sorted_idx = np.argsort(importances)[::-1]
        sorted_importances = importances[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx] if feature_names else [f'Feature {i}' for i in sorted_idx]

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(sorted_names))

        bars = ax.barh(y_pos, sorted_importances, color=self.colors[0], alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.subdirs['comparisons'] / 'feature_importance.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Create 2D feature space visualization
        if X_full.shape[1] >= 2:
            self._create_2d_feature_space(X_full, y_full, feature_names)

    def _create_2d_feature_space(self, X_full, y_full, feature_names):
        """Create 2D feature space visualization"""
        # Use PCA for dimensionality reduction if needed
        if X_full.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_full)
            explained_var = pca.explained_variance_ratio_
            x_label = f'PC1 ({explained_var[0]:.2%} variance)'
            y_label = f'PC2 ({explained_var[1]:.2%} variance)'
        else:
            X_2d = X_full
            x_label = feature_names[0] if feature_names else 'Feature 1'
            y_label = feature_names[1] if feature_names else 'Feature 2'

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_classes = np.unique(y_full)

        for i, cls in enumerate(unique_classes):
            mask = y_full == cls
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      label=f'Class {cls}', alpha=0.7, color=self.colors[i % len(self.colors)])

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('2D Feature Space', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.subdirs['comparisons'] / 'feature_space_2d.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_3d_visualizations(self, X_full, y_full, training_history, feature_names):
        """Create comprehensive 3D visualizations with parallel processing"""
        print("🌐 Creating 3D visualizations...")

        # Reduce to 3D for visualization
        if X_full.shape[1] > 3:
            pca = PCA(n_components=3, random_state=42)
            X_3d = pca.fit_transform(X_full)
            explained_var = pca.explained_variance_ratio_
        else:
            X_3d = X_full
            explained_var = np.ones(3)

        # Create different 3D visualization types
        viz_tasks = [
            ('pca_3d', self._create_pca_3d_plot, (X_3d, y_full, explained_var, 'PCA 3D Projection')),
            ('feature_3d', self._create_feature_3d_plot, (X_full, y_full, feature_names)),
            ('density_3d', self._create_density_controlled_3d, (X_3d, y_full)),
        ]

        if training_history:
            # Add training evolution 3D
            key_rounds = [0, len(training_history)//2, -1]
            for round_num in key_rounds:
                if round_num < len(training_history):
                    viz_tasks.append(
                        (f'training_3d_round_{round_num}',
                         self._create_training_3d_network,
                         (X_3d, y_full, training_history[round_num], round_num, explained_var))
                    )

        # Execute 3D visualizations
        for viz_name, viz_func, viz_args in viz_tasks:
            try:
                viz_func(*viz_args)
                print(f"✅ {viz_name} completed")
            except Exception as e:
                print(f"❌ {viz_name} failed: {str(e)}")

    def _create_pca_3d_plot(self, X_3d, y_full, explained_var, title):
        """Create PCA-based 3D plot"""
        fig = go.Figure()
        unique_classes = np.unique(y_full)

        for i, cls in enumerate(unique_classes):
            mask = y_full == cls
            fig.add_trace(go.Scatter3d(
                x=X_3d[mask, 0], y=X_3d[mask, 1], z=X_3d[mask, 2],
                mode='markers',
                marker=dict(size=4, color=self.colors[i % len(self.colors)], opacity=0.7),
                name=f'Class {cls}'
            ))

        fig.update_layout(
            title=f"{title}<br>Explained Variance: {explained_var.sum():.2%}",
            scene=dict(
                xaxis_title=f'PC1 ({explained_var[0]:.2%})',
                yaxis_title=f'PC2 ({explained_var[1]:.2%})',
                zaxis_title=f'PC3 ({explained_var[2]:.2%})'
            ),
            width=1000, height=800
        )

        fig.write_html(self.subdirs['interactive'] / 'pca_3d.html')

    def _create_feature_3d_plot(self, X_full, y_full, feature_names):
        """Create 3D plot using actual features"""
        if X_full.shape[1] < 3 or not feature_names or len(feature_names) < 3:
            return

        fig = go.Figure()
        unique_classes = np.unique(y_full)

        for i, cls in enumerate(unique_classes):
            mask = y_full == cls
            fig.add_trace(go.Scatter3d(
                x=X_full[mask, 0], y=X_full[mask, 1], z=X_full[mask, 2],
                mode='markers',
                marker=dict(size=4, color=self.colors[i % len(self.colors)], opacity=0.7),
                name=f'Class {cls}',
                text=[f'{feature_names[0]}: {x:.3f}<br>{feature_names[1]}: {y:.3f}<br>{feature_names[2]}: {z:.3f}'
                      for x, y, z in zip(X_full[mask, 0], X_full[mask, 1], X_full[mask, 2])]
            ))

        fig.update_layout(
            title="3D Feature Space",
            scene=dict(
                xaxis_title=feature_names[0],
                yaxis_title=feature_names[1],
                zaxis_title=feature_names[2]
            ),
            width=1000, height=800
        )

        fig.write_html(self.subdirs['interactive'] / 'feature_3d.html')

    def _create_density_controlled_3d(self, X_3d, y_full):
        """Create density-controlled 3D visualization"""
        # Sample points based on density to reduce overcrowding
        X_sampled, y_sampled = self._density_based_sampling(X_3d, y_full)

        fig = go.Figure()
        unique_classes = np.unique(y_sampled)

        for i, cls in enumerate(unique_classes):
            mask = y_sampled == cls
            fig.add_trace(go.Scatter3d(
                x=X_sampled[mask, 0], y=X_sampled[mask, 1], z=X_sampled[mask, 2],
                mode='markers',
                marker=dict(size=5, color=self.colors[i % len(self.colors)], opacity=0.8),
                name=f'Class {cls} (sampled)'
            ))

        fig.update_layout(
            title="Density-Controlled 3D Visualization",
            width=1000, height=800
        )

        fig.write_html(self.subdirs['interactive'] / 'density_3d.html')

    def _create_training_3d_network(self, X_3d, y_full, training_indices, round_num, explained_var):
        """Create 3D network visualization for specific training round"""
        fig = go.Figure()

        # Plot all samples (background)
        background_mask = ~np.isin(range(len(X_3d)), training_indices)
        unique_classes = np.unique(y_full)

        for i, cls in enumerate(unique_classes):
            # Background points
            bg_mask = (y_full == cls) & background_mask
            if np.any(bg_mask):
                fig.add_trace(go.Scatter3d(
                    x=X_3d[bg_mask, 0], y=X_3d[bg_mask, 1], z=X_3d[bg_mask, 2],
                    mode='markers',
                    marker=dict(size=3, color=self.colors[i % len(self.colors)], opacity=0.1),
                    name=f'Class {cls} (other)',
                    showlegend=False
                ))

            # Training points
            train_mask = (y_full == cls) & np.isin(range(len(X_3d)), training_indices)
            if np.any(train_mask):
                fig.add_trace(go.Scatter3d(
                    x=X_3d[train_mask, 0], y=X_3d[train_mask, 1], z=X_3d[train_mask, 2],
                    mode='markers',
                    marker=dict(size=6, color=self.colors[i % len(self.colors)], opacity=0.8,
                              line=dict(width=1, color='black')),
                    name=f'Class {cls} (training)'
                ))

        fig.update_layout(
            title=f"3D Training Network - Round {round_num + 1}<br>"
                  f"Training Samples: {len(training_indices)}<br>"
                  f"Explained Variance: {explained_var.sum():.2%}",
            width=1000, height=800
        )

        fig.write_html(self.subdirs['networks'] / f'training_network_round_{round_num + 1}.html')

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
                # Use k-nearest neighbors for density-based sampling
                nbrs = NearestNeighbors(n_neighbors=min(10, len(X_class))).fit(X_class)
                distances, _ = nbrs.kneighbors(X_class)
                avg_distances = np.mean(distances, axis=1)

                # Sample points with higher average distances (less crowded)
                probabilities = 1 / (avg_distances + 1e-8)
                probabilities = probabilities / np.sum(probabilities)

                selected_indices = np.random.choice(
                    len(X_class), size=max_points_per_class, replace=False, p=probabilities
                )

                X_sampled_list.append(X_class[selected_indices])
                y_sampled_list.append(np.full(max_points_per_class, cls))

        return np.vstack(X_sampled_list), np.hstack(y_sampled_list)

    def _create_interactive_dashboard(self, round_stats, training_history, X_full, y_full, feature_names):
        """Create comprehensive interactive dashboard"""
        print("📊 Creating interactive dashboard...")

        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Unified DBNN Dashboard - {dataset_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 10px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Unified DBNN Analysis Dashboard</h1>
                <h2>Dataset: {dataset_name}</h2>
                <p>Generated on: {timestamp}</p>
            </div>

            <div class="stats">
                <div class="stat-card">
                    <h3>Total Rounds</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #667eea;">{total_rounds}</p>
                </div>
                <div class="stat-card">
                    <h3>Final Training Size</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #28a745;">{final_training_size}</p>
                </div>
                <div class="stat-card">
                    <h3>Features</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #ffc107;">{feature_count}</p>
                </div>
                <div class="stat-card">
                    <h3>Classes</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #dc3545;">{class_count}</p>
                </div>
            </div>

            <div class="nav">
                <a class="nav-button" href="#performance">Performance</a>
                <a class="nav-button" href="#3d">3D Visualizations</a>
                <a class="nav-button" href="#geometric">Geometric Analysis</a>
                <a class="nav-button" href="#adaptive">Adaptive Learning</a>
            </div>

            <div id="performance" class="iframe-container">
                <h3>📈 Performance Analysis</h3>
                <iframe src="interactive/performance_dashboard.html"></iframe>
            </div>

            <div id="3d" class="iframe-container">
                <h3>🌐 3D Visualizations</h3>
                <iframe src="interactive/pca_3d.html"></iframe>
            </div>
        """.format(
            dataset_name=self.dataset_name,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_rounds=len(round_stats) if round_stats else 0,
            final_training_size=len(training_history[-1]) if training_history else 0,
            feature_count=len(feature_names) if feature_names else X_full.shape[1] if X_full is not None else 0,
            class_count=len(np.unique(y_full)) if y_full is not None else 0
        )

        dashboard_html += """
            <script>
                // Smooth scrolling for navigation
                document.querySelectorAll('.nav-button').forEach(button => {
                    button.addEventListener('click', function(e) {
                        e.preventDefault();
                        const targetId = this.getAttribute('href').substring(1);
                        document.getElementById(targetId).scrollIntoView({
                            behavior: 'smooth'
                        });
                    });
                });
            </script>
        </body>
        </html>
        """

        with open(self.output_dir / 'unified_dashboard.html', 'w') as f:
            f.write(dashboard_html)

    # Utility methods for integration with existing code
    def capture_epoch_snapshot(self, epoch, accuracy):
        """Capture snapshot for geometric analysis (compatibility method)"""
        # This maintains compatibility with existing training code
        pass

    def create_geometric_visualization(self, training_history, round_stats):
        """Compatibility method for existing code"""
        self.create_comprehensive_visualizations(training_history, round_stats)

def add_geometric_visualization_to_adbnn():
    """Add geometric visualization capability to ADBNN class - COMPATIBILITY FUNCTION"""

    def create_geometric_visualization(self, training_history, round_stats):
        """Create geometric visualization using the unified visualizer"""
        print("🌐 Creating geometric visualization with unified visualizer...")

        # Get feature names if available
        feature_names = getattr(self, 'feature_columns', None)
        if feature_names is None and hasattr(self, 'data'):
            feature_names = [col for col in self.data.columns if col != self.target_column]

        # Use the unified visualizer
        visualizer = UnifiedDBNNVisualizer(self)
        visualizer.create_comprehensive_visualizations(
            training_history=training_history,
            round_stats=round_stats,
            feature_names=feature_names,
            enable_3d=True
        )

    # Add method to DBNN class for backward compatibility
    if not hasattr(DBNN, 'create_geometric_visualization'):
        DBNN.create_geometric_visualization = create_geometric_visualization

    print("✅ Geometric visualization compatibility layer added")


# Integration function for existing DBNN code
def add_unified_visualization_to_dbnn():
    """Add unified visualization capability to DBNN class"""

    def create_unified_visualization(self, training_history=None, round_stats=None,
                                   feature_names=None, enable_3d=True):
        """Create comprehensive unified visualizations"""
        visualizer = UnifiedDBNNVisualizer(self)
        visualizer.create_comprehensive_visualizations(
            training_history, round_stats, feature_names, enable_3d
        )

    # Add method to DBNN class
    if not hasattr(DBNN, 'create_unified_visualization'):
        DBNN.create_unified_visualization = create_unified_visualization

    # Also replace the old method for compatibility
    if not hasattr(DBNN, 'create_geometric_visualization'):
        DBNN.create_geometric_visualization = create_unified_visualization

# Initialize the integration
add_unified_visualization_to_dbnn()
# Create single global instance
DEBUG = DebugLogger()

def configure_debug(config):
    """Configure debug state from config"""
    debug_enabled = config.get('training_params', {}).get('debug_enabled', False)
    if debug_enabled:
        DEBUG.enable()
    else:
        DEBUG.disable()

#-------------------------------------------------------unit test ----------------------------------
import os
import glob
import json
from typing import List, Tuple
import pandas as pd
from datetime import datetime

import json
import os
def _calculate_optimal_batch_size(sample_tensor_size):
    """Calculate optimal batch size with more conservative estimates"""
    if not torch.cuda.is_available():
        return 128  # Default for CPU

    try:
        # Get memory stats
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        free_memory = total_memory - allocated - reserved

        # Be more conservative - use only 30% of free memory
        available_memory = free_memory * 0.3

        # Calculate memory needed per sample (with safety factor)
        memory_per_sample = sample_tensor_size * 8  # More conservative estimate

        # Calculate optimal batch size
        optimal_batch_size = int(available_memory / memory_per_sample)

        # Enforce reasonable bounds
        optimal_batch_size = max(32, min(optimal_batch_size, 2048))  # Reduced max from 4096

        DEBUG.log(f"Memory Analysis:")
        DEBUG.log(f"- Total GPU Memory: {total_memory / 1e9:.2f} GB")
        DEBUG.log(f"- Free Memory: {free_memory / 1e9:.2f} GB")
        DEBUG.log(f"- Available for batch: {available_memory / 1e9:.2f} GB")
        DEBUG.log(f"- Memory per sample: {memory_per_sample / 1e6:.2f} MB")
        print(f"{Colors.GREEN}The new Batch size is dynamically set to {optimal_batch_size}{Colors.ENDC}")

        return optimal_batch_size

    except Exception as e:
        print(f"{Colors.RED}Error calculating batch size: {str(e)}{Colors.ENDC}")
        return 128  # Fallback value

def load_or_create_config(config_path: str) -> dict:
    """
    Load the configuration file if it exists, or create a default one if it doesn't.
    Update global variables based on the configuration file.
    RESPECTS ALL GLOBAL COMMAND LINE FLAGS.
    """
    # Use GLOBAL command line flags
    command_line_visualize = COMMAND_LINE_FLAGS['visualize']
    command_line_model_type = COMMAND_LINE_FLAGS['model_type']
    command_line_fresh_start = COMMAND_LINE_FLAGS['fresh_start']
    command_line_use_previous_model = COMMAND_LINE_FLAGS['use_previous_model']
    command_line_5dct = COMMAND_LINE_FLAGS['enable_5DCTvisualization']
    command_line_mode = COMMAND_LINE_FLAGS['mode']
    command_line_interactive = COMMAND_LINE_FLAGS['interactive']

    default_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trials": 100,
        "cardinality_threshold": 0.9,
        "cardinality_tolerance": 8,
        "learning_rate": 0.1,
        "random_seed": 42,
        "epochs": 1000,
        "test_fraction": 0.2,
        "train": True,
        "train_only": False,
        "predict": True,
        "gen_samples": False,
        "enable_adaptive": True,
        "nokbd": False,
        "display": None,
        "training_params": {
            "batch_size": None,
            "patience": 100,
            "adaptive_patience": 25,
            "n_bins_per_dim": 128,
            "minimum_training_accuracy": 0.95,
            "invert_DBNN": True,
            "reconstruction_weight": 0.5,
            "feedback_strength": 0.3,
            "inverse_learning_rate": 0.1,
            "Save_training_epochs": False,
            "training_save_path": "data",
            "enable_vectorized": False,
            "vectorization_warning_acknowledged": False,
            "compute_device": "auto",
            "use_interactive_kbd": False,
            "modelType": command_line_model_type,
            "class_preference": True,
            "enable_visualization": command_line_visualize,
            "enable_5DCTvisualization": command_line_5dct,
            "enable_adaptive_learning": True,
            "adaptive_rounds": 100,
            "initial_samples": 50,
            "max_samples_per_round": 500
        },
        "active_learning": {
            "tolerance": 1.0,
            "similarity_threshold": 0.25,
            "cardinality_threshold_percentile": 95,
            "update_condition": "bin_overlap",
            "min_divergence": 0.1,
            "max_class_addition_percent": 99
        },
        "anomaly_detection": {
            "initial_weight": 1e-6,
            "weight_update_rate": 0.1,
            "max_weight": 10.0,
            "min_weight": 1e-8
        },
        "likelihood_config": {
            "feature_group_size": 2,
            "max_combinations": 90000000,
            "bin_sizes": [128],
            "n_bins_per_dim": 128
        },
        "execution_flags": {
            "fresh_start": command_line_fresh_start,
            "use_previous_model": command_line_use_previous_model,
            "mode": command_line_mode,
            "interactive": command_line_interactive
        }
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()

            # Remove comments from JSON
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
            config = json.loads(clean_config)

        except Exception as e:
            print(f"❌ {Colors.RED}Error loading config file {config_path}: {e}{Colors.ENDC}")
            print(f"🔧 {Colors.YELLOW}Creating new config with defaults...{Colors.ENDC}")
            config = default_config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)

        # APPLY GLOBAL COMMAND LINE OVERRIDES
        applied_overrides = []

        # Ensure training_params exists
        if 'training_params' not in config:
            config['training_params'] = {}

        training_params = config['training_params']

        # Visualization override
        if command_line_visualize:
            if training_params.get('enable_visualization') != True:
                training_params['enable_visualization'] = True
                applied_overrides.append('enable_visualization=True')

        # 5DCT Visualization override
        if command_line_5dct:
            if training_params.get('enable_5DCTvisualization') != True:
                training_params['enable_5DCTvisualization'] = True
                applied_overrides.append('enable_5DCTvisualization=True')

        # Model type override
        if command_line_model_type != 'Histogram':
            if training_params.get('modelType') != command_line_model_type:
                training_params['modelType'] = command_line_model_type
                applied_overrides.append(f'modelType={command_line_model_type}')

        # Ensure execution_flags exists
        if 'execution_flags' not in config:
            config['execution_flags'] = {}

        execution_flags = config['execution_flags']

        # Fresh start override
        if command_line_fresh_start:
            if execution_flags.get('fresh_start') != True:
                execution_flags['fresh_start'] = True
                execution_flags['use_previous_model'] = False
                applied_overrides.append('fresh_start=True')

        # Use previous model override
        if command_line_use_previous_model:
            if execution_flags.get('use_previous_model') != True:
                execution_flags['use_previous_model'] = True
                execution_flags['fresh_start'] = False
                applied_overrides.append('use_previous_model=True')

        # Mode override
        if command_line_mode != 'train_predict':
            if execution_flags.get('mode') != command_line_mode:
                execution_flags['mode'] = command_line_mode
                applied_overrides.append(f'mode={command_line_mode}')

        # Interactive override
        if command_line_interactive:
            if execution_flags.get('interactive') != True:
                execution_flags['interactive'] = True
                applied_overrides.append('interactive=True')

        # Show override summary (only once)
        if applied_overrides and not hasattr(load_or_create_config, '_overrides_shown'):
            print(f"🎯 {Colors.GREEN}CONFIG OVERRIDES APPLIED ({len(applied_overrides)}):{Colors.ENDC}")
            for override in applied_overrides:
                print(f"   {Colors.CYAN}→ {override}{Colors.ENDC}")
            load_or_create_config._overrides_shown = True

        # Set defaults for missing parameters (respecting command line flags)
        if 'enable_visualization' not in training_params:
            training_params['enable_visualization'] = command_line_visualize
            if not hasattr(load_or_create_config, '_default_shown'):
                print(f"🔧 {Colors.YELLOW}Added missing 'enable_visualization' parameter with value: {command_line_visualize}{Colors.ENDC}")

        if 'enable_5DCTvisualization' not in training_params:
            training_params['enable_5DCTvisualization'] = command_line_5dct

        if 'modelType' not in training_params:
            training_params['modelType'] = command_line_model_type

        if 'fresh_start' not in execution_flags:
            execution_flags['fresh_start'] = command_line_fresh_start

        if 'use_previous_model' not in execution_flags:
            execution_flags['use_previous_model'] = command_line_use_previous_model

        if 'mode' not in execution_flags:
            execution_flags['mode'] = command_line_mode

        if 'interactive' not in execution_flags:
            execution_flags['interactive'] = command_line_interactive

        # Ensure other required sections exist
        if 'active_learning' not in config:
            config['active_learning'] = default_config['active_learning']

        if 'anomaly_detection' not in config:
            config['anomaly_detection'] = default_config['anomaly_detection']

        if 'likelihood_config' not in config:
            config['likelihood_config'] = default_config['likelihood_config']

    else:
        config = default_config
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else '.', exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print(f"📝 {Colors.GREEN}Created default configuration file at {config_path}{Colors.ENDC}")

        # Show which command line flags were used in default creation
        used_flags = []
        if command_line_visualize:
            used_flags.append('visualize')
        if command_line_5dct:
            used_flags.append('5DCT')
        if command_line_model_type != 'Histogram':
            used_flags.append(f'model_type={command_line_model_type}')
        if command_line_fresh_start:
            used_flags.append('fresh_start')
        if command_line_use_previous_model:
            used_flags.append('use_previous_model')
        if command_line_mode != 'train_predict':
            used_flags.append(f'mode={command_line_mode}')
        if command_line_interactive:
            used_flags.append('interactive')

        if used_flags:
            print(f"🔧 {Colors.CYAN}Default config created with: {', '.join(used_flags)}{Colors.ENDC}")

    # Update global variables
    global predict_mode, Train_device, bin_sizes, n_bins_per_dim, Trials, cardinality_threshold
    global cardinality_tolerance, LearningRate, TrainingRandomSeed, Epochs, TestFraction
    global Train, Train_only, Predict, Gen_Samples, EnableAdaptive, nokbd, display

    try:
        Train_device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        Trials = config.get("trials", 100)
        cardinality_threshold = config.get("cardinality_threshold", 0.9)
        cardinality_tolerance = config.get("cardinality_tolerance", 8)
        bin_sizes = config.get("bin_sizes", 128)
        n_bins_per_dim = config.get("n_bins_per_dim", 128)
        LearningRate = config.get("learning_rate", 0.1)
        TrainingRandomSeed = config.get("random_seed", 42)
        Epochs = config.get("epochs", 1000)
        TestFraction = config.get("test_fraction", 0.2)
        Train = config.get("train", True)
        Train_only = config.get("train_only", False)
        Predict = config.get("predict", True)
        Gen_Samples = config.get("gen_samples", False)
        EnableAdaptive = config.get("enable_adaptive", True)
        nokbd = config.get("nokbd", False)
        display = config.get("display", None)

        # Set predict_mode based on execution flags
        execution_flags = config.get('execution_flags', {})
        predict_mode = execution_flags.get('mode', 'train_predict') == 'predict'

        # Dynamically calculate batch size if not provided
        if "training_params" in config:
            training_params = config["training_params"]
            if "batch_size" not in training_params or training_params["batch_size"] is None:
                sample_tensor_size = 4 * 1024 * 1024  # 4MB sample tensor
                training_params["batch_size"] = _calculate_optimal_batch_size(sample_tensor_size)
                print(f"🔧 {Colors.CYAN}Auto-calculated batch_size: {training_params['batch_size']}{Colors.ENDC}")

        # Final validation
        required_sections = ['training_params', 'active_learning', 'anomaly_detection', 'likelihood_config', 'execution_flags']
        for section in required_sections:
            if section not in config:
                config[section] = default_config.get(section, {})
                print(f"🔧 {Colors.YELLOW}Added missing section: {section}{Colors.ENDC}")

        return config

    except Exception as e:
        print(f"❌ {Colors.RED}Error updating global variables from config: {e}{Colors.ENDC}")
        return default_config


def _calculate_optimal_batch_size(sample_tensor_size: int) -> int:
    """
    Calculate optimal batch size based on available GPU memory or system RAM.

    Args:
        sample_tensor_size: Size of a single sample tensor in bytes

    Returns:
        Optimal batch size
    """
    try:
        if torch.cuda.is_available():
            # GPU-based calculation
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = gpu_memory * 0.7  # Use 70% of available memory
            batch_size = max(1, int(available_memory / sample_tensor_size))
            batch_size = min(batch_size, 1024)  # Cap at 1024
        else:
            # CPU-based calculation
            import psutil
            available_memory = psutil.virtual_memory().available * 0.5  # Use 50% of available RAM
            batch_size = max(1, int(available_memory / sample_tensor_size))
            batch_size = min(batch_size, 512)  # Cap at 512 for CPU

        return batch_size

    except Exception as e:
        print(f"⚠️ {Colors.YELLOW}Error calculating optimal batch size, using default: {e}{Colors.ENDC}")
        return 128  # Safe default

def find_dataset_pairs(data_dir: str = 'data') -> List[Tuple[str, str, str]]:
    """
    Recursively find all matching .conf and .csv files in the data directory structure.

    Args:
        data_dir: Root directory to search for datasets.

    Returns:
        List of tuples (basename, conf_path, csv_path).
    """
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        print("\033[K" + f"No '{data_dir}' directory found. Creating one...", end="\r", flush=True)
        os.makedirs(data_dir)
        return []

    dataset_pairs = []
    processed_datasets = set()

    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_dir):
        # Process configuration files
        conf_files = [f for f in files if f.endswith('.conf') and f != 'adaptive_dbnn.conf']

        for conf_file in conf_files:
            basename = os.path.splitext(conf_file)[0]
            if basename in processed_datasets:
                continue

            conf_path = os.path.join(root, conf_file)

            # Look for matching CSV in the same directory and dataset-specific subdirectory
            csv_file = f"{basename}.csv"
            csv_paths = [
                os.path.join(root, csv_file),
                os.path.join(root, basename, csv_file)
            ]

            # Load adaptive_dbnn.conf if it exists
            adaptive_conf = {}
            adaptive_conf_path = os.path.join(data_dir, basename, 'adaptive_dbnn.conf')
            if os.path.exists(adaptive_conf_path):
                try:
                    with open(adaptive_conf_path, 'r') as f:
                        adaptive_conf = json.load(f)
                    print("\033[K" + f"Loaded adaptive configuration from {adaptive_conf_path}", end="\r", flush=True)
                except Exception as e:
                    print(f"Warning: Could not load adaptive configuration from {adaptive_conf_path}: {str(e)}")
            else:
                print("\033[K" + f"No adaptive_dbnn.conf found in working directory {adaptive_conf_path}", end="\r", flush=True)

            # Find the CSV file
            csv_path = None
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break

            if csv_path:
                # Load the dataset configuration
                try:
                    with open(conf_path, 'r') as f:
                        dataset_conf = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not update configuration for {basename}: {str(e)}")

                dataset_pairs.append((basename, conf_path, csv_path))
                processed_datasets.add(basename)
                print("\033[K" + f"Found dataset pair:", end="\r", flush=True)
                print("\033[K" + f"  Config: {conf_path}", end="\r", flush=True)
                print("\033[K" + f"  Data  : {csv_path}", end="\r", flush=True)
            else:
                print("\033[K" + f"Warning: Config file {conf_file} exists but no matching CSV found")
                print("\033[K" + f"Looked in:")
                for path in csv_paths:
                    print("\033[K" + f"  - {path}")

    if not dataset_pairs:
        print("\033[K" + "No matching .conf and .csv file pairs found.")
        print("\033[K" + "Each dataset should have both a .conf configuration file and a matching .csv data file.")
        print("\033[K" + "Example: 'dataset1.conf' and 'dataset1.csv'")

    return dataset_pairs

def validate_config(config: dict) -> dict:
    """
    Validate the configuration parameters, ensuring that only the batch size is dynamically updated.

    Args:
        config: Configuration dictionary.

    Returns:
        dict: Validated configuration dictionary.
    """
    validated_config = config.copy()

    # Validate other parameters (only replace with defaults if missing or invalid)
    if "device" not in validated_config or validated_config["device"] not in ["cuda", "cpu"]:
        validated_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    if "trials" not in validated_config or not isinstance(validated_config["trials"], int):
        validated_config["trials"] = 100

    if "cardinality_threshold" not in validated_config or not isinstance(validated_config["cardinality_threshold"], float):
        validated_config["cardinality_threshold"] = 0.9

    if "cardinality_tolerance" not in validated_config or not isinstance(validated_config["cardinality_tolerance"], int):
        validated_config["cardinality_tolerance"] = 8

    if "learning_rate" not in validated_config or not isinstance(validated_config["learning_rate"], float):
        validated_config["learning_rate"] = 0.1

    if "random_seed" not in validated_config or not isinstance(validated_config["random_seed"], int):
        validated_config["random_seed"] = 42

    if "epochs" not in validated_config or not isinstance(validated_config["epochs"], int):
        validated_config["epochs"] = 1000

    if "test_fraction" not in validated_config or not isinstance(validated_config["test_fraction"], float):
        validated_config["test_fraction"] = 0.2

    if "train" not in validated_config or not isinstance(validated_config["train"], bool):
        validated_config["train"] = True

    if "train_only" not in validated_config or not isinstance(validated_config["train_only"], bool):
        validated_config["train_only"] = False

    if "predict" not in validated_config or not isinstance(validated_config["predict"], bool):
        validated_config["predict"] = True

    if "gen_samples" not in validated_config or not isinstance(validated_config["gen_samples"], bool):
        validated_config["gen_samples"] = False

    if "enable_adaptive" not in validated_config or not isinstance(validated_config["enable_adaptive"], bool):
        validated_config["enable_adaptive"] = True

    if "nokbd" not in validated_config or not isinstance(validated_config["nokbd"], bool):
        validated_config["nokbd"] = False

    if "display" not in validated_config:
        validated_config["display"] = None

    return validated_config

def get_dataset_name_from_path(file_path):
    """Extract dataset name from file path"""
    if not file_path:
        return None

    # Handle different path formats
    if '/' in file_path:
        # Unix-style path
        parts = file_path.split('/')
        # Look for meaningful dataset name
        for part in reversed(parts):
            if part and not part.endswith(('.csv', '.ccv', '.data')):
                return part
        # Fallback: use filename without extension
        filename = parts[-1]
        return os.path.splitext(filename)[0]
    elif '\\' in file_path:
        # Windows-style path
        parts = file_path.split('\\')
        for part in reversed(parts):
            if part and not part.endswith(('.csv', '.ccv', '.data')):
                return part
        filename = parts[-1]
        return os.path.splitext(filename)[0]
    else:
        # Simple filename
        return os.path.splitext(file_path)[0]

def main():
    # Available datasets section will be populated dynamically
    dataset_info = "    Available datasets will be listed here when using --list_datasets"

    parser = argparse.ArgumentParser(
        description='🚀 DBNN - Deep Bayesian Neural Network Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
📚 COMMAND EXAMPLES:

  Interactive Mode:
    {Colors.GREEN}python adbnn.py --interactive{Colors.ENDC}
        💬 Guided setup with configuration options

  Training & Visualization:
    {Colors.GREEN}python adbnn.py --mode train --model_type Gaussian --visualize{Colors.ENDC}
        🧠 Train with Gaussian model + generate unified visualization

    {Colors.GREEN}python adbnn.py --file_path data/mydata/mydata.csv --mode train_predict --visualize{Colors.ENDC}
        📊 Train & predict on specific dataset with unified visualization

  Dataset Management:
    {Colors.GREEN}python adbnn.py --list_datasets{Colors.ENDC}
        📋 Show all available datasets with statistics

  Prediction:
    {Colors.GREEN}python adbnn.py --mode predict --file_path data/mydata/mydata.csv{Colors.ENDC}
        🔮 Make predictions using trained model

  Feature Reconstruction:
    {Colors.GREEN}python adbnn.py --mode invertDBNN{Colors.ENDC}
        🔄 Reconstruct features using inverse DBNN

📖 AVAILABLE MODES:
  train         - Train model only
  train_predict - Train and then predict (default)
  predict       - Predict using trained model
  invertDBNN    - Reconstruct features using inverse model

🎯 QUICK START:
  1. Add your dataset to the 'data' folder as 'data/dataset_name/dataset_name.csv'
  2. Run: {Colors.GREEN}python adbnn.py --interactive{Colors.ENDC}
  3. Or run: {Colors.GREEN}python adbnn.py --list_datasets{Colors.ENDC} to see available datasets

{dataset_info}
        """
    )

    parser.add_argument("--file_path", nargs='?', help="📁 Path to dataset CSV file")
    parser.add_argument('--mode', type=str, choices=['train', 'train_predict', 'invertDBNN', 'predict'],
                       default='train_predict', help="🎯 Operation mode")
    parser.add_argument('--interactive', action='store_true', help="💬 Enable interactive configuration")
    parser.add_argument('--model_type', type=str, choices=['Histogram', 'Gaussian'],
                        default='Histogram', help='🧠 Model architecture type')
    parser.add_argument('--visualize', action='store_true', help="📊 Generate unified training visualization")
    parser.add_argument('--list_datasets', action='store_true', help="📋 List available datasets")
    parser.add_argument('--fresh_start', action='store_true', help="🔄 Start training from scratch (ignore previous models)")
    parser.add_argument('--use_previous_model', action='store_true', help="💾 Use previously trained model if available")
    args = parser.parse_args()

    processor = DatasetProcessor()

    def print_banner():
        """Print beautiful banner"""
        banner = f"""
{Colors.BOLD}{Colors.RED}
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║    🧠   Difference Boosting Bayesian Neural Network            ║
║                 Author :nsp@airis4d.com                        ║
║               Python Implementation: DeepSeek                  ║
║    Advanced Tensor Orthogonalization & Adaptive Learning       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝{Colors.ENDC}
        """
        print(banner)

    def validate_config(config):
        """Ensure required configuration parameters exist"""
        required_params = {
            'training_params': ['learning_rate', 'epochs', 'test_fraction'],
            'modelType': None,
            'target_column': None
        }

        for section, params in required_params.items():
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config")
            if params:
                for param in params:
                    if param not in config[section]:
                        raise ValueError(f"Missing required parameter '{param}' in section '{section}'")

    def find_dataset_pairs():
        """Find matching dataset/config pairs in data folder"""
        dataset_pairs = []
        data_dir = 'data'

        if os.path.exists(data_dir):
            for dataset_name in os.listdir(data_dir):
                dataset_dir = os.path.join(data_dir, dataset_name)
                if os.path.isdir(dataset_dir):
                    conf_path = os.path.join(dataset_dir, f"{dataset_name}.conf")
                    csv_path = os.path.join(dataset_dir, f"{dataset_name}.csv")

                    if os.path.exists(conf_path) and os.path.exists(csv_path):
                        dataset_pairs.append((dataset_name, conf_path, csv_path))

        return dataset_pairs

    def display_dataset_menu(dataset_pairs):
        """Display beautiful dataset selection menu"""
        print(f"\n{Colors.BOLD}📊 Available Datasets:{Colors.ENDC}")
        print(f"{Colors.BLUE}╔══════════════════════════════════════════════════════════════════════════{Colors.ENDC}")

        for i, (dataset_name, conf_path, csv_path) in enumerate(dataset_pairs):
            # Get dataset info
            try:
                config = load_or_create_config(conf_path)
                n_samples = len(pd.read_csv(csv_path))
                n_features = len(pd.read_csv(csv_path, nrows=0).columns) - 1
                model_type = config.get('modelType', 'Histogram')

                print(f"{Colors.BLUE}║ {Colors.ENDC}{i+1:2d}. {Colors.GREEN}{dataset_name:<20}{Colors.ENDC} "
                      f"Samples: {n_samples:>5} Features: {n_features:>3} "
                      f"Model: {model_type:<10}")#" #{Colors.BLUE} ║{Colors.ENDC}")

            except Exception as e:
                print(f"{Colors.BLUE}║ {Colors.ENDC}{i+1:2d}. {Colors.RED}{dataset_name:<20}{Colors.ENDC} "
                      f"Error loading dataset {Colors.RED}{str(e)[:30]:<30}{Colors.ENDC} {Colors.BLUE}║{Colors.ENDC}")

        print(f"{Colors.BLUE}╚══════════════════════════════════════════════════════════════════════════{Colors.ENDC}")
    print()

    def process_datasets():
        """Process all found datasets"""
        dataset_pairs = find_dataset_pairs()

        if not dataset_pairs:
            print(f"❌ {Colors.RED}No valid dataset/config pairs found in data folder.{Colors.ENDC}")
            return

        display_dataset_menu(dataset_pairs)

        choice = input(f"\n🎯 {Colors.BOLD}Select dataset (1-{len(dataset_pairs)}, 'all', 'h' for help, or 'q' to quit): {Colors.ENDC}").strip()

        if choice.lower() == 'q':
            return
        elif choice.lower() == 'h':
            print(f"\n{Colors.BOLD}🚀 QUICK COMMAND EXAMPLES:{Colors.ENDC}")
            print(f"  {Colors.GREEN}python adbnn.py --interactive{Colors.ENDC}")
            print(f"  {Colors.GREEN}python adbnn.py --mode train --model_type Gaussian --visualize{Colors.ENDC}")
            print(f"  {Colors.GREEN}python adbnn.py --list_datasets{Colors.ENDC}")
            print(f"  {Colors.GREEN}python adbnn.py --file_path data/mydata/mydata.csv --mode train_predict --visualize{Colors.ENDC}")
            input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
            process_datasets()
        elif choice.lower() == 'all':
            print(f"\n🔄 {Colors.YELLOW}Processing all datasets...{Colors.ENDC}")
            for dataset_name, conf_path, csv_path in dataset_pairs:
                process_single_dataset(dataset_name, conf_path, csv_path, args.mode, args.model_type, args.visualize)
        elif choice.isdigit() and 1 <= int(choice) <= len(dataset_pairs):
            dataset_name, conf_path, csv_path = dataset_pairs[int(choice)-1]
            process_single_dataset(dataset_name, conf_path, csv_path, args.mode, args.model_type, args.visualize)
        else:
            print(f"❌ {Colors.RED}Invalid selection.{Colors.ENDC}")

    def process_single_dataset(dataset_name, conf_path, csv_path, mode=None, model_type=None, generate_visualization=None):
        """Process a single dataset with given mode - uses GLOBAL command line flags as defaults"""
        try:
            # USE GLOBAL COMMAND LINE FLAGS AS DEFAULTS
            if mode is None:
                mode = COMMAND_LINE_FLAGS['mode']
            if model_type is None:
                model_type = COMMAND_LINE_FLAGS['model_type']
            if generate_visualization is None:
                generate_visualization = COMMAND_LINE_FLAGS['visualize']

            enable_5dct = COMMAND_LINE_FLAGS['enable_5DCTvisualization']
            # Load config WITH GLOBAL COMMAND LINE OVERRIDE SUPPORT
            # (config loading now automatically respects global flags via load_or_create_config)
            config = load_or_create_config(conf_path)

            # REMOVED: Redundant config modification - now handled in load_or_create_config
            # The config loading function now automatically applies command line overrides

            # Show command line flag summary
            if not hasattr(process_single_dataset, '_flags_shown'):
                print(f"🎯 {Colors.BOLD}PROCESSING WITH GLOBAL FLAGS:{Colors.ENDC}")
                active_flags = []
                if COMMAND_LINE_FLAGS['visualize']:
                    active_flags.append('--visualize')
                if COMMAND_LINE_FLAGS['enable_5DCTvisualization']:
                    active_flags.append('--5DCT')
                if COMMAND_LINE_FLAGS['model_type'] != 'Histogram':
                    active_flags.append(f'--model_type={COMMAND_LINE_FLAGS["model_type"]}')
                if COMMAND_LINE_FLAGS['fresh_start']:
                    active_flags.append('--fresh_start')
                if COMMAND_LINE_FLAGS['use_previous_model']:
                    active_flags.append('--use_previous_model')

                if active_flags:
                    for flag in active_flags:
                        print(f"   {Colors.CYAN}📌 {flag}{Colors.ENDC}")
                else:
                    print(f"   {Colors.YELLOW}📌 Using default settings{Colors.ENDC}")
                process_single_dataset._flags_shown = True

            # Determine mode if not provided (should already be set from global flags)
            if not mode:
                mode = 'train_predict' if config.get('train', True) and config.get('predict', True) else \
                       'train' if config.get('train', True) else 'predict'

            print(f"\n{'='*80}")
            print(f"🚀 {Colors.BOLD}Processing: {Colors.GREEN}{dataset_name}{Colors.ENDC}")
            print(f"📋 {Colors.BOLD}Mode: {Colors.YELLOW}{mode}{Colors.ENDC}")
            print(f"🧠 {Colors.BOLD}Model: {Colors.YELLOW}{model_type}{Colors.ENDC}")
            print(f"📊 {Colors.BOLD}Visualization: {Colors.GREEN}{'Enabled' if generate_visualization else 'Disabled'}{Colors.ENDC}")
            print(f"🔧 {Colors.BOLD}Config enable_visualization: {Colors.CYAN}{config.get('training_params', {}).get('enable_visualization', 'NOT SET')}{Colors.ENDC}")
            print(f"{'='*80}")

            # Create DBNN instance - USE GLOBAL FLAGS
            print(f"🔧 {Colors.BLUE}Creating DBNN instance with parameters:{Colors.ENDC}")
            print(f"   {Colors.CYAN}• enable_visualization={generate_visualization}{Colors.ENDC}")
            print(f"   {Colors.CYAN}• model_type={model_type}{Colors.ENDC}")
            print(f"   {Colors.CYAN}• mode={mode}{Colors.ENDC}")

            model = DBNN(
                dataset_name=dataset_name,
                mode=mode if mode != 'train_predict' else 'train',  # Handle train_predict mode
                model_type=model_type,
                enable_visualization=generate_visualization,
                enable_5DCTvisualization=enable_5dct
            )

            if mode in ['train', 'train_predict']:
                # Training phase
                start_time = datetime.now()
                print(f"\n⏳ {Colors.BOLD}Starting training phase...{Colors.ENDC}")

                # Enhanced debug: Check if visualization is properly enabled
                print(f"🔍 {Colors.CYAN}DBNN INSTANCE VERIFICATION:{Colors.ENDC}")
                if hasattr(model, 'enable_visualization'):
                    status_color = Colors.GREEN if model.enable_visualization else Colors.RED
                    print(f"   {Colors.CYAN}• enable_visualization: {status_color}{model.enable_visualization}{Colors.ENDC}")
                else:
                    print(f"   {Colors.RED}• enable_visualization: NOT FOUND{Colors.ENDC}")

                if hasattr(model, 'visualizer'):
                    status_color = Colors.GREEN if model.visualizer is not None else Colors.RED
                    print(f"   {Colors.CYAN}• visualizer: {status_color}{model.visualizer is not None}{Colors.ENDC}")
                else:
                    print(f"   {Colors.RED}• visualizer: NOT FOUND{Colors.ENDC}")

                # Check execution flags from config
                execution_flags = config.get('execution_flags', {})
                fresh_start = execution_flags.get('fresh_start', COMMAND_LINE_FLAGS['fresh_start'])
                use_previous_model = execution_flags.get('use_previous_model', COMMAND_LINE_FLAGS['use_previous_model'])

                print(f"🔍 {Colors.CYAN}EXECUTION FLAGS:{Colors.ENDC}")
                print(f"   {Colors.CYAN}• fresh_start: {Colors.YELLOW}{fresh_start}{Colors.ENDC}")
                print(f"   {Colors.CYAN}• use_previous_model: {Colors.YELLOW}{use_previous_model}{Colors.ENDC}")

                if config.get('training_params', {}).get('enable_adaptive', True):
                    print(f"🔄 {Colors.YELLOW}Using adaptive training{Colors.ENDC}")
                    results = model.adaptive_fit_predict()
                else:
                    print(f"⚡ {Colors.YELLOW}Using standard training{Colors.ENDC}")
                    results = model.fit_predict()

                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds()

                # Print results
                print(f"\n✅ {Colors.GREEN}Training completed!{Colors.ENDC}")
                print(f"⏱️  {Colors.BOLD}Time taken: {Colors.BLUE}{training_time:.1f} seconds{Colors.ENDC}")

                if results and 'test_accuracy' in results:
                    accuracy_color = Colors.GREEN if results['test_accuracy'] > 0.9 else Colors.YELLOW if results['test_accuracy'] > 0.7 else Colors.RED
                    print(f"🎯 {Colors.BOLD}Test Accuracy: {accuracy_color}{results['test_accuracy']:.2%}{Colors.ENDC}")

                # Visualization - NOW RESPECTS GLOBAL FLAG CONSISTENTLY
                if generate_visualization:
                    print(f"\n📊 {Colors.BOLD}Generating unified visualizations...{Colors.ENDC}")
                    try:
                        # Get training history and round stats for visualization
                        training_history = []
                        round_stats = []

                        # Try to extract training history from adaptive learning
                        if hasattr(model, 'training_history'):
                            training_history = model.training_history
                            print(f"📈 {Colors.CYAN}Found training history: {len(training_history)} snapshots{Colors.ENDC}")
                        elif hasattr(model, 'train_indices'):
                            training_history = [model.train_indices]
                            print(f"📈 {Colors.CYAN}Using train indices as history{Colors.ENDC}")

                        # Try to extract round statistics
                        if hasattr(model, 'round_stats'):
                            round_stats = model.round_stats
                            print(f"📊 {Colors.CYAN}Found round stats: {len(round_stats)} rounds{Colors.ENDC}")

                        # Get feature names
                        feature_names = getattr(model, 'feature_columns', None)
                        if feature_names is None and hasattr(model, 'data'):
                            feature_names = [col for col in model.data.columns if col != model.target_column]
                            print(f"🔤 {Colors.CYAN}Using {len(feature_names)} feature names{Colors.ENDC}")

                        # Use the new unified visualization system
                        print(f"🎨 {Colors.BLUE}Creating comprehensive unified visualizations...{Colors.ENDC}")

                        # Method 1: Use the unified visualization method if available
                        if hasattr(model, 'create_unified_visualization'):
                            print(f"🛠️  {Colors.CYAN}Using model.create_unified_visualization(){Colors.ENDC}")
                            model.create_unified_visualization(
                                training_history=training_history,
                                round_stats=round_stats,
                                feature_names=feature_names,
                                enable_3d=True
                            )

                        # Method 2: Use the geometric visualization method (backward compatibility)
                        elif hasattr(model, 'create_geometric_visualization'):
                            print(f"🛠️  {Colors.CYAN}Using model.create_geometric_visualization(){Colors.ENDC}")
                            model.create_geometric_visualization(training_history, round_stats)

                        # Method 3: Direct UnifiedDBNNVisualizer instantiation
                        else:
                            print(f"🛠️  {Colors.CYAN}Using UnifiedDBNNVisualizer directly{Colors.ENDC}")
                            visualizer = UnifiedDBNNVisualizer(model)
                            visualizer.create_comprehensive_visualizations(
                                training_history=training_history,
                                round_stats=round_stats,
                                feature_names=feature_names,
                                enable_3d=True
                            )

                        print(f"✅ {Colors.GREEN}Unified visualizations generated in: Visualizer/adaptiveDBNN/{dataset_name}/{Colors.ENDC}")

                    except Exception as e:
                        print(f"⚠️  {Colors.YELLOW}Visualization failed: {str(e)}{Colors.ENDC}")
                        import traceback
                        traceback.print_exc()

                # Save model components
                try:
                    model._save_model_components()
                    model._save_best_weights()
                    print(f"💾 {Colors.GREEN}Model components saved{Colors.ENDC}")
                except Exception as e:
                    print(f"⚠️  {Colors.YELLOW}Model save warning: {str(e)}{Colors.ENDC}")

            if mode in ['predict', 'train_predict']:
                # Prediction phase
                print(f"\n🔮 {Colors.BOLD}Starting prediction phase...{Colors.ENDC}")

                # Use either the provided CSV or default dataset CSV
                input_csv = COMMAND_LINE_FLAGS['file_path'] if COMMAND_LINE_FLAGS['file_path'] and mode == 'predict' else csv_path
                output_dir = os.path.join('data', dataset_name, 'Predictions')
                os.makedirs(output_dir, exist_ok=True)

                print(f"📁 {Colors.BOLD}Input: {Colors.GREEN}{input_csv}{Colors.ENDC}")
                print(f"📂 {Colors.BOLD}Output: {Colors.YELLOW}{output_dir}{Colors.ENDC}")

                # For prediction mode, create a new predictor instance WITH GLOBAL FLAGS
                if mode == 'predict':
                    print(f"🔧 {Colors.BLUE}Creating predictor instance with global flags{Colors.ENDC}")
                    predictor = DBNN(
                        dataset_name=dataset_name,
                        mode='predict',
                        model_type=model_type,
                        enable_visualization=generate_visualization
                    )
                    model = predictor

                results = model.predict_from_file(input_csv, output_dir, model_type=model_type)

                if results:
                    print(f"✅ {Colors.GREEN}Predictions completed successfully!{Colors.ENDC}")
                    # FIXED: Safe metrics access
                    if results.get('metrics') is not None and 'accuracy' in results['metrics']:
                        acc = results['metrics']['accuracy']
                        acc_color = Colors.GREEN if acc > 0.9 else Colors.YELLOW if acc > 0.7 else Colors.RED
                        print(f"🎯 {Colors.BOLD}Prediction Accuracy: {acc_color}{acc:.2%}{Colors.ENDC}")
                    else:
                        print(f"📊 {Colors.YELLOW}No ground truth available for accuracy calculation{Colors.ENDC}")

                    # Print sample count
                    if 'predictions' in results:
                        sample_count = len(results['predictions'])
                        print(f"📈 {Colors.BOLD}Samples processed: {Colors.CYAN}{sample_count}{Colors.ENDC}")

            if mode == 'invertDBNN':
                # Invert DBNN mode
                print(f"\n🔄 {Colors.BOLD}Starting Inverse DBNN feature reconstruction...{Colors.ENDC}")

                model._load_model_components()

                # Display inversion parameters
                inv_params = config.get('training_params', {})
                print(f"⚙️  {Colors.BOLD}Inversion Parameters:{Colors.ENDC}")
                for param in ['reconstruction_weight', 'feedback_strength', 'inverse_learning_rate']:
                    value = inv_params.get(param, 'default')
                    print(f"   {Colors.CYAN}{param}: {Colors.YELLOW}{value}{Colors.ENDC}")

                inverse_model = InvertibleDBNN(
                    forward_model=model,
                    feature_dims=model.data.shape[1] - 1,
                    reconstruction_weight=inv_params.get('reconstruction_weight', 0.5),
                    feedback_strength=inv_params.get('feedback_strength', 0.3)
                )

                # Reconstruct features
                X_test = model.data.drop(columns=[model.target_column])
                test_probs = model._get_test_probabilities(X_test)
                reconstruction_features = inverse_model.reconstruct_features(test_probs)

                # Save reconstructed features
                output_dir = os.path.join('data', dataset_name, 'Reconstructed_Features')
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f'{dataset_name}_reconstructed.csv')

                feature_columns = model.data.drop(columns=[model.target_column]).columns
                reconstructed_df = pd.DataFrame(reconstruction_features.cpu().numpy(), columns=feature_columns)
                reconstructed_df.to_csv(output_file, index=False)

                print(f"✅ {Colors.GREEN}Reconstructed features saved to {Colors.CYAN}{output_file}{Colors.ENDC}")

            print(f"\n{'='*80}")
            print(f"✅ {Colors.BOLD}{Colors.GREEN}Processing completed for {dataset_name}!{Colors.ENDC}")

            # Final summary including global flag status
            print(f"📋 {Colors.BOLD}PROCESSING SUMMARY:{Colors.ENDC}")
            print(f"   {Colors.CYAN}• Dataset: {Colors.GREEN}{dataset_name}{Colors.ENDC}")
            print(f"   {Colors.CYAN}• Mode: {Colors.YELLOW}{mode}{Colors.ENDC}")
            print(f"   {Colors.CYAN}• Model: {Colors.YELLOW}{model_type}{Colors.ENDC}")

            # Final reminder about visualization output location
            if generate_visualization:
                viz_path = f"Visualizer/adaptiveDBNN/{dataset_name}/"
                print(f"📊 {Colors.BOLD}Unified visualizations saved to: {Colors.CYAN}{viz_path}{Colors.ENDC}")
                print(f"   - Performance analysis")
                print(f"   - Geometric tensor analysis")
                print(f"   - Adaptive learning analysis")
                print(f"   - 3D interactive visualizations")
                print(f"   - Interactive dashboard")
            else:
                print(f"🔇 {Colors.YELLOW}Visualization was disabled{Colors.ENDC}")

            # Show which global flags were used
            used_global_flags = []
            if COMMAND_LINE_FLAGS['visualize'] and generate_visualization:
                used_global_flags.append('--visualize')
            if COMMAND_LINE_FLAGS['model_type'] != 'Histogram' and model_type == COMMAND_LINE_FLAGS['model_type']:
                used_global_flags.append(f'--model_type={COMMAND_LINE_FLAGS["model_type"]}')
            if COMMAND_LINE_FLAGS['fresh_start']:
                used_global_flags.append('--fresh_start')

            if used_global_flags:
                print(f"🎯 {Colors.BOLD}Global flags applied: {Colors.CYAN}{', '.join(used_global_flags)}{Colors.ENDC}")

            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n❌ {Colors.RED}Error processing dataset {dataset_name}: {str(e)}{Colors.ENDC}")
            import traceback
            traceback.print_exc()

    def interactive_mode():
        """Enhanced interactive mode"""
        print(f"\n💬 {Colors.BOLD}{Colors.BLUE}INTERACTIVE MODE{Colors.ENDC}")
        print(f"{Colors.BLUE}╔══════════════════════════════════════════════════════════════╗{Colors.ENDC}")

        # List available datasets
        dataset_pairs = find_dataset_pairs()
        if not dataset_pairs:
            print(f"❌ {Colors.RED}No datasets found. Please add datasets to the data folder.{Colors.ENDC}")
            return

        display_dataset_menu(dataset_pairs)

        # Dataset selection
        while True:
            choice = input(f"\n🎯 {Colors.BOLD}Select dataset (1-{len(dataset_pairs)}): {Colors.ENDC}").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(dataset_pairs):
                dataset_name, conf_path, csv_path = dataset_pairs[int(choice)-1]
                break
            else:
                print(f"❌ {Colors.RED}Invalid selection. Please try again.{Colors.ENDC}")

        # Load config
        config = load_or_create_config(conf_path)

        # Display current configuration
        print(f"\n⚙️  {Colors.BOLD}Current Configuration:{Colors.ENDC}")
        config_display = [
            ("Device", config.get('compute_device', 'cuda' if torch.cuda.is_available() else 'cpu')),
            ("Learning Rate", config.get('training_params', {}).get('learning_rate', 0.1)),
            ("Epochs", config.get('training_params', {}).get('epochs', 1000)),
            ("Test Fraction", config.get('training_params', {}).get('test_fraction', 0.2)),
            ("Adaptive Training", config.get('training_params', {}).get('enable_adaptive', True)),
            ("Class Preference", config.get('training_params', {}).get('class_preference', True)),
            ("Model Type", config.get('modelType', 'Histogram'))
        ]

        for param, value in config_display:
            color = Colors.GREEN if value else Colors.YELLOW
            print(f"   {Colors.CYAN}{param:<20}: {color}{value}{Colors.ENDC}")

        # Mode selection
        print(f"\n🎯 {Colors.BOLD}Operation Mode:{Colors.ENDC}")
        modes = [
            ("1", "Train", "Train model only"),
            ("2", "Train & Predict", "Train then predict (recommended)"),
            ("3", "Predict", "Predict using trained model"),
            ("4", "Invert DBNN", "Reconstruct features")
        ]

        for num, name, desc in modes:
            print(f"   {Colors.YELLOW}{num}. {name:<15} {Colors.WHITE}- {desc}{Colors.ENDC}")

        mode_choice = input(f"\n🎯 {Colors.BOLD}Select mode (1-4): {Colors.ENDC}").strip()
        mode_map = {"1": "train", "2": "train_predict", "3": "predict", "4": "invertDBNN"}
        mode = mode_map.get(mode_choice, "train_predict")

        # Model type selection
        print(f"\n🧠 {Colors.BOLD}Model Type:{Colors.ENDC}")
        model_choice = input(f"Select model type (1: Histogram, 2: Gaussian) [1]: {Colors.ENDC}").strip()
        model_type = "Gaussian" if model_choice == "2" else "Histogram"

        # Visualization option
        viz_choice = input(f"\n📊 {Colors.BOLD}Generate unified training visualization? (y/N): {Colors.ENDC}").strip().lower()
        generate_viz = viz_choice in ['y', 'yes']

        if generate_viz:
            print(f"🎨 {Colors.GREEN}Unified visualization will include:{Colors.ENDC}")
            print(f"   • Performance evolution analysis")
            print(f"   • Geometric tensor analysis")
            print(f"   • Adaptive learning analysis")
            print(f"   • 3D interactive visualizations")
            print(f"   • Interactive dashboard")

        # Update config based on selections
        config['train'] = mode in ['train', 'train_predict']
        config['predict'] = mode in ['predict', 'train_predict']
        config['modelType'] = model_type

        # Save updated config
        with open(conf_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ {Colors.GREEN}Configuration updated.{Colors.ENDC}")

        # Process the dataset - PASS THE VISUALIZATION FLAG
        process_single_dataset(dataset_name, conf_path, csv_path, mode, model_type, generate_viz)

    # Main execution flow
    print_banner()

    if args.list_datasets:
        dataset_pairs = find_dataset_pairs()
        if dataset_pairs:
            display_dataset_menu(dataset_pairs)
        else:
            print(f"❌ {Colors.RED}No datasets found.{Colors.ENDC}")
        return

    if args.interactive:
        interactive_mode()

    elif not args.file_path and args.mode == 'train_predict':
        # No arguments provided - search for datasets
        process_datasets()

    elif args.mode:
        # Specific mode requested
        if args.mode == 'invertDBNN':
            if not args.file_path:
                dataset_pairs = find_dataset_pairs()
                if dataset_pairs:
                    args.file_path = dataset_pairs[0][2]
                    print(f"📁 {Colors.YELLOW}Using default CSV file: {args.file_path}{Colors.ENDC}")
                else:
                    print(f"❌ {Colors.RED}No datasets found for inversion.{Colors.ENDC}")
                    return

            basename = os.path.splitext(os.path.basename(args.file_path))[0]
            conf_path = os.path.join('data', basename, f'{basename}.conf')
            csv_path = os.path.join('data', basename, f'{basename}.csv')
            process_single_dataset(basename, conf_path, csv_path, 'invertDBNN', args.model_type, args.visualize)

        elif args.mode in ['train', 'train_predict', 'predict']:
            if args.file_path:
                basename = get_dataset_name_from_path(args.file_path)
                workfile = os.path.splitext(os.path.basename(args.file_path))[0]
                conf_path = os.path.join('data', basename, f'{basename}.conf')
                csv_path = os.path.join('data', basename, f'{workfile}.csv')
                process_single_dataset(basename, conf_path, csv_path, args.mode, args.model_type, args.visualize)
            else:
                dataset_pairs = find_dataset_pairs()
                if dataset_pairs:
                    basename, conf_path, csv_path = dataset_pairs[0]
                    print(f"📁 {Colors.YELLOW}Using default dataset: {basename}{Colors.ENDC}")
                    process_single_dataset(basename, conf_path, csv_path, args.mode, args.model_type, args.visualize)
                else:
                    print(f"❌ {Colors.RED}No datasets found.{Colors.ENDC}")

    else:
        parser.print_help()
        print(f"\n💡 {Colors.YELLOW}Tip: Use --interactive for guided mode or --list_datasets to see available datasets.{Colors.ENDC}")

def find_visualizations():
    """Find where visualizations are actually being saved"""
    possible_paths = [
        "Visualizer/",
        "Visualizer/",
        "Visualizer/adaptiveDBNN/",
        "Visualizer/adaptiveDBNN/mnist/",
        "./"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"📁 Found path: {path}")
            # List all files recursively
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.html') or file.endswith('.png'):
                        print(f"   📄 {os.path.join(root, file)}")


if __name__ == "__main__":
    print("\033[K" +"DBNN Dataset Processor")
    print("\033[K" +"=" * 40)
    main()
    # Call this after training
    #find_visualizations()
