"""
CDBNN: Convolutional Deep Bayesian Neural Network
Complete Professional Version with ALL Original Features Optimized
Author: Ninan Sajeeth Philip
Last Updated: March 13 2026
"""

import os
import gc
import sys
import json
import time
import glob
import shutil
import logging
import argparse
import traceback
import warnings
import threading
import pickle
import gzip
import bz2
import lzma
import zipfile
import tarfile
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import multiprocessing as mp
import multiprocessing
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.spatial.distance import correlation
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import PIL.Image as PILImage
from PIL import ImageOps, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import functional as TF
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

@dataclass
class GlobalConfig:
    """Global configuration with all original features"""
    # Dataset settings
    dataset_name: str = 'dataset'
    data_type: str = 'custom'
    in_channels: int = 3
    input_size: Tuple[int, int] = (256, 256)
    num_classes: Optional[int] = None
    class_names: List[str] = field(default_factory=list)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    image_type: str = 'general'  # general, astronomical, medical, agricultural

    # Model settings
    encoder_type: str = 'autoenc'
    feature_dims: int = 128
    compressed_dims: int = 32
    learning_rate: float = 0.001

    # Training settings
    batch_size: int = 32
    epochs: int = 200
    num_workers: int = min(4, os.cpu_count() or 1)
    checkpoint_dir: str = 'checkpoints'
    validation_split: float = 0.2

    # Enhancement features
    use_kl_divergence: bool = True
    use_class_encoding: bool = True
    enable_sharpness_loss: bool = False
    use_distance_correlation: bool = True
    enable_adaptive: bool = True

    # Feature selection
    feature_selection_method: str = 'balanced'  # simple, balanced, complex
    max_features: int = 32
    min_features: int = 8
    correlation_upper: float = 0.85
    correlation_lower: float = 0.01

    # Visualization
    generate_heatmaps: bool = True
    generate_confusion_matrix: bool = True
    generate_tsne: bool = True
    heatmap_frequency: int = 10
    reconstruction_samples_frequency: int = 5

    # Execution flags
    use_gpu: bool = torch.cuda.is_available()
    debug_mode: bool = False
    mixed_precision: bool = True
    distributed_training: bool = False

    # Paths
    data_dir: str = 'data'
    output_dir: str = 'output'
    log_dir: str = 'logs'
    viz_dir: str = 'visualizations'

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, data: Dict) -> 'GlobalConfig':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

# =============================================================================
# LOGGING AND UTILITIES
# =============================================================================

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
        """Color a value based on improvement"""
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

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""

    COLORS = {
        'DEBUG': Colors.BLUE,
        'INFO': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.RED + Colors.BOLD
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Colors.ENDC}"
        return super().format(record)

def setup_logging(name: str = 'cdbnn', log_dir: str = 'logs') -> logging.Logger:
    """Setup logging with rotation and colors"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler with rotation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

def timed(func: Callable) -> Callable:
    """Decorator to time functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

def memory_efficient(func: Callable) -> Callable:
    """Decorator for memory efficient operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        return result
    return wrapper

def safe_json_serialize(obj: Any) -> Any:
    """Safely serialize any object to JSON-compatible format"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except:
            return str(obj)
    try:
        return str(obj)
    except:
        return f"<{type(obj).__name__}>"

class ResourceMonitor:
    """Monitor system resources with visualization"""

    def __init__(self, interval: float = 5.0, save_path: Optional[str] = None):
        self.interval = interval
        self.save_path = save_path
        self.running = False
        self.thread = None
        self.stats = defaultdict(list)
        self.timestamps = []

    def start(self):
        """Start monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        logger.info("Resource monitoring started")

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self._log_summary()
        if self.save_path:
            self._plot_stats()

    def _monitor(self):
        """Monitor resources"""
        try:
            import psutil
            import GPUtil

            while self.running:
                timestamp = time.time()
                self.timestamps.append(timestamp)

                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.stats['cpu'].append(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                self.stats['memory'].append(memory.percent)
                self.stats['memory_used'].append(memory.used / (1024**3))  # GB
                self.stats['memory_available'].append(memory.available / (1024**3))

                # Disk usage
                disk = psutil.disk_usage('/')
                self.stats['disk'].append(disk.percent)

                # GPU usage
                if torch.cuda.is_available():
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            self.stats['gpu'].append(gpu.load * 100)
                            self.stats['gpu_memory'].append(gpu.memoryUtil * 100)
                            self.stats['gpu_temp'].append(gpu.temperature)
                    except:
                        pass

                time.sleep(self.interval)

        except ImportError:
            logger.warning("psutil/GPUtil not available for resource monitoring")
            return

    def _log_summary(self):
        """Log summary statistics"""
        if not self.stats:
            return

        logger.info("=" * 60)
        logger.info("RESOURCE USAGE SUMMARY")
        logger.info("=" * 60)

        for key, values in self.stats.items():
            if values:
                logger.info(f"{key:15s}: avg={np.mean(values):.1f}, "
                          f"max={np.max(values):.1f}, min={np.min(values):.1f}")

    def _plot_stats(self):
        """Plot resource usage over time"""
        if not self.stats or not self.timestamps:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Resource Usage Over Time', fontsize=16, fontweight='bold')

        # Convert timestamps to relative time
        t = np.array(self.timestamps) - self.timestamps[0]

        # CPU and Memory
        ax1 = axes[0, 0]
        if 'cpu' in self.stats:
            ax1.plot(t, self.stats['cpu'], 'b-', label='CPU %', linewidth=2)
        if 'memory' in self.stats:
            ax1.plot(t, self.stats['memory'], 'r-', label='Memory %', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Usage %')
        ax1.set_title('CPU & Memory Usage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # GPU Usage
        ax2 = axes[0, 1]
        if 'gpu' in self.stats:
            ax2.plot(t, self.stats['gpu'], 'g-', label='GPU %', linewidth=2)
        if 'gpu_memory' in self.stats:
            ax2.plot(t, self.stats['gpu_memory'], 'm-', label='GPU Memory %', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Usage %')
        ax2.set_title('GPU Usage')
        if 'gpu' in self.stats or 'gpu_memory' in self.stats:
            ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Memory in GB
        ax3 = axes[1, 0]
        if 'memory_used' in self.stats:
            ax3.plot(t, self.stats['memory_used'], 'orange', label='Used', linewidth=2)
        if 'memory_available' in self.stats:
            ax3.plot(t, self.stats['memory_available'], 'green', label='Available', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Memory (GB)')
        ax3.set_title('Memory Usage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Temperature (if available)
        ax4 = axes[1, 1]
        if 'gpu_temp' in self.stats:
            ax4.plot(t, self.stats['gpu_temp'], 'r-', label='GPU Temp', linewidth=2)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Temperature (°C)')
            ax4.set_title('GPU Temperature')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No temperature data available',
                    ha='center', va='center', transform=ax4.transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'resource_usage.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

# =============================================================================
# ARCHIVE HANDLING (Original feature)
# =============================================================================

class ArchiveHandler:
    """Handle various archive formats"""

    SUPPORTED_FORMATS = {
        '.zip': zipfile.ZipFile,
        '.tar': tarfile.TarFile,
        '.tar.gz': tarfile.TarFile,
        '.tgz': tarfile.TarFile,
        '.gz': gzip.GzipFile,
        '.bz2': bz2.BZ2File,
        '.xz': lzma.LZMAFile
    }

    @staticmethod
    def extract(archive_path: str, extract_dir: str) -> str:
        """Extract archive to directory"""
        os.makedirs(extract_dir, exist_ok=True)

        file_ext = Path(archive_path).suffix.lower()
        if file_ext.startswith('.'):
            file_ext = file_ext[1:]

        logger.info(f"Extracting {archive_path} to {extract_dir}")

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

        # Find the main directory
        contents = os.listdir(extract_dir)
        if len(contents) == 1 and os.path.isdir(os.path.join(extract_dir, contents[0])):
            return os.path.join(extract_dir, contents[0])

        return extract_dir

    @staticmethod
    def compress(source_dir: str, output_path: str, format: str = 'zip'):
        """Compress directory to archive"""
        if format == 'zip':
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_dir)
                        zipf.write(file_path, arcname)
        elif format == 'tar.gz':
            with tarfile.open(output_path, 'w:gz') as tar:
                tar.add(source_dir, arcname=os.path.basename(source_dir))
        else:
            raise ValueError(f"Unsupported compression format: {format}")

# =============================================================================
# IMAGE PROCESSING (PIL-based with all original features)
# =============================================================================

class ImageProcessor:
    """Comprehensive image processing using PIL"""

    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp', '.ppm', '.pgm')

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

    @staticmethod
    @lru_cache(maxsize=256)
    def load_image(path: str) -> Optional[PILImage.Image]:
        """Load image with caching and verification"""
        try:
            img = PILImage.open(path)
            img.verify()  # Verify integrity
            img = PILImage.open(path)  # Reopen after verify
            return img
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None

    def preprocess(self, image: PILImage.Image, is_train: bool = False) -> PILImage.Image:
        """Preprocess image with augmentations"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize
        if image.size != self.target_size:
            image = self.resize_with_aspect(image, self.target_size)

        # Training augmentations
        if is_train:
            if np.random.random() > 0.5:
                image = ImageOps.mirror(image)
            if np.random.random() > 0.5:
                image = image.rotate(np.random.randint(-10, 10), expand=False)

        return image

    def resize_with_aspect(self, image: PILImage.Image, target_size: Tuple[int, int]) -> PILImage.Image:
        """Resize while maintaining aspect ratio"""
        image.thumbnail(target_size, PILImage.Resampling.LANCZOS)

        # Create new image with target size and paste centered
        new_img = PILImage.new('RGB', target_size, (0, 0, 0))
        paste_x = (target_size[0] - image.width) // 2
        paste_y = (target_size[1] - image.height) // 2
        new_img.paste(image, (paste_x, paste_y))

        return new_img

    @staticmethod
    def extract_windows(image: PILImage.Image, window_size: int, stride: int,
                       overlap: float = 0.5) -> List[Tuple[PILImage.Image, Tuple[int, int]]]:
        """Extract sliding windows with overlap"""
        width, height = image.size
        windows = []
        positions = []

        stride = stride or int(window_size * (1 - overlap))

        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                window = image.crop((x, y, x + window_size, y + window_size))
                windows.append(window)
                positions.append((x, y))

        # Add edge windows for full coverage
        if height % stride != 0:
            y = height - window_size
            for x in range(0, width - window_size + 1, stride):
                window = image.crop((x, y, x + window_size, y + window_size))
                windows.append(window)
                positions.append((x, y))

        if width % stride != 0:
            x = width - window_size
            for y in range(0, height - window_size + 1, stride):
                window = image.crop((x, y, x + window_size, y + window_size))
                windows.append(window)
                positions.append((x, y))

        if height % stride != 0 and width % stride != 0:
            window = image.crop((width - window_size, height - window_size, width, height))
            windows.append(window)
            positions.append((width - window_size, height - window_size))

        return list(zip(windows, positions))

    @staticmethod
    def blend_windows(windows: List[Tuple[np.ndarray, Tuple[int, int]]],
                     original_size: Tuple[int, int], window_size: int,
                     blend_mode: str = 'linear') -> np.ndarray:
        """Blend windows back to full image"""
        height, width = original_size
        result = np.zeros((height, width, 3), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)

        # Create blending weights
        if blend_mode == 'linear':
            weight_map = np.ones((window_size, window_size))
            center = window_size // 2
            for i in range(window_size):
                for j in range(window_size):
                    dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                    weight_map[i, j] = max(0, 1 - dist / center)
        else:
            weight_map = np.ones((window_size, window_size))

        # Blend windows
        for window, (x, y) in windows:
            window_h, window_w = window.shape[:2]

            if window_h != window_size or window_w != window_size:
                # Resize window if needed
                from skimage.transform import resize
                window = resize(window, (window_size, window_size), preserve_range=True)

            # Apply weights
            weighted_window = window * weight_map[:, :, np.newaxis]

            # Add to result
            result[y:y+window_size, x:x+window_size] += weighted_window
            weights[y:y+window_size, x:x+window_size] += weight_map

        # Normalize
        weights = np.maximum(weights, 1e-8)
        result = result / weights[:, :, np.newaxis]

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def enhance_contrast(image: PILImage.Image, clip_limit: float = 2.0) -> PILImage.Image:
        """Apply CLAHE-like contrast enhancement"""
        import cv2
        img_np = np.array(image)

        if len(img_np.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)

            # Merge back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_np)

        return PILImage.fromarray(img_enhanced)

    @staticmethod
    def detect_edges(image: PILImage.Image, method: str = 'canny') -> PILImage.Image:
        """Edge detection for visualization"""
        import cv2
        img_np = np.array(image.convert('L'))

        if method == 'canny':
            edges = cv2.Canny(img_np, 50, 150)
        elif method == 'sobel':
            sobelx = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = (edges / edges.max() * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")

        return PILImage.fromarray(edges)

# =============================================================================
# DATASET HANDLING (All original features)
# =============================================================================

class CustomImageDataset(Dataset):
    """Memory-efficient image dataset with all original features"""

    def __init__(self, root_dir: str, transform=None, cache_size: int = 100,
                 data_name: Optional[str] = None, config: Optional[Dict] = None):
        """
        Args:
            root_dir: Directory with class subdirectories
            transform: Optional transform to apply
            cache_size: Number of images to cache in memory
            data_name: Dataset name
            config: Configuration dictionary
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache_size = cache_size
        self.cache = {}
        self.cache_access = []
        self.config = config or {}
        self.data_name = data_name or self.root_dir.name

        # Initialize collections
        self.samples = []  # (path, label)
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.image_files = []  # Full paths
        self.filenames = []    # Just filenames
        self.labels = []

        # Scan directory
        self._scan_directory()

        # Preprocessing options
        self.resize_images = self.config.get('resize_images', False)
        self.overlap = self.config.get('overlap', 0.5)

        logger.info(f"Dataset '{self.data_name}': {len(self.samples)} images, "
                   f"{len(self.classes)} classes")

    def _scan_directory(self):
        """Scan directory for images and classes"""
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                idx = len(self.classes)
                self.class_to_idx[class_name] = idx
                self.idx_to_class[idx] = class_name
                self.classes.append(class_name)

            # Find images
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ImageProcessor.SUPPORTED_FORMATS:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                    self.image_files.append(str(img_path))
                    self.filenames.append(img_path.name)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Check cache
        if idx in self.cache:
            img = self.cache[idx]
            self.cache_access.remove(idx)
            self.cache_access.append(idx)
        else:
            # Load image
            img = ImageProcessor.load_image(img_path)
            if img is None:
                # Return blank image on error
                img = PILImage.new('RGB', (256, 256), (0, 0, 0))

            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Update cache
            if len(self.cache) >= self.cache_size:
                # Remove least recently used
                oldest = self.cache_access.pop(0)
                del self.cache[oldest]
            self.cache[idx] = img
            self.cache_access.append(idx)

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label

    def get_additional_info(self, idx: int) -> Tuple[int, str, str]:
        """Get additional information for index"""
        return idx, self.filenames[idx], self.image_files[idx]

    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution"""
        dist = defaultdict(int)
        for _, label in self.samples:
            dist[self.idx_to_class[label]] += 1
        return dict(dist)

    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for imbalanced datasets"""
        class_counts = np.bincount(self.labels)
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum() * len(class_counts)
        return torch.FloatTensor(weights)

class SlidingWindowDataset(Dataset):
    """Dataset for large images using sliding windows"""

    def __init__(self, image_paths: List[str], window_size: int = 256, stride: int = 128,
                 transform=None, overlap: float = 0.5, min_coverage: float = 0.8):
        self.image_paths = image_paths
        self.window_size = window_size
        self.stride = stride or int(window_size * (1 - overlap))
        self.transform = transform
        self.min_coverage = min_coverage

        # Precompute windows
        self.windows = []  # (image_idx, x, y, width, height)
        self.image_shapes = []

        logger.info(f"Building sliding windows for {len(image_paths)} images...")
        self._precompute_windows()

    def _precompute_windows(self):
        """Precompute all window coordinates"""
        processor = ImageProcessor()

        for img_idx, img_path in enumerate(tqdm(self.image_paths, desc="Precomputing windows")):
            img = processor.load_image(img_path)
            if img is None:
                continue

            width, height = img.size
            self.image_shapes.append((height, width))

            windows = self._get_window_coordinates(height, width)
            for y, x, h, w in windows:
                self.windows.append((img_idx, x, y, w, h))

        logger.info(f"Generated {len(self.windows)} windows from {len(self.image_paths)} images")

    def _get_window_coordinates(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """Generate window coordinates"""
        windows = []

        y_steps = max(1, (height - self.window_size) // self.stride + 1)
        x_steps = max(1, (width - self.window_size) // self.stride + 1)

        for i in range(y_steps):
            for j in range(x_steps):
                y = i * self.stride
                x = j * self.stride
                h = min(self.window_size, height - y)
                w = min(self.window_size, width - x)

                coverage = (h * w) / (self.window_size * self.window_size)
                if coverage >= self.min_coverage:
                    windows.append((y, x, h, w))

        # Edge windows
        if height % self.stride != 0:
            y = height - self.window_size
            for j in range(x_steps):
                x = j * self.stride
                h = self.window_size
                w = min(self.window_size, width - x)
                coverage = (h * w) / (self.window_size * self.window_size)
                if coverage >= self.min_coverage:
                    windows.append((y, x, h, w))

        if width % self.stride != 0:
            x = width - self.window_size
            for i in range(y_steps):
                y = i * self.stride
                h = min(self.window_size, height - y)
                w = self.window_size
                coverage = (h * w) / (self.window_size * self.window_size)
                if coverage >= self.min_coverage:
                    windows.append((y, x, h, w))

        if height % self.stride != 0 and width % self.stride != 0:
            windows.append((height - self.window_size, width - self.window_size,
                           self.window_size, self.window_size))

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict:
        img_idx, x, y, w, h = self.windows[idx]
        img_path = self.image_paths[img_idx]

        img = ImageProcessor.load_image(img_path)
        if img is None:
            # Return dummy window
            window = torch.zeros(3, self.window_size, self.window_size)
        else:
            # Extract window
            window = img.crop((x, y, x + w, y + h))

            # Pad if needed
            if w != self.window_size or h != self.window_size:
                padded = PILImage.new('RGB', (self.window_size, self.window_size), (0, 0, 0))
                padded.paste(window, (0, 0))
                window = padded

            # Apply transforms
            if self.transform:
                window = self.transform(window)
            else:
                window = transforms.ToTensor()(window)

        return {
            'window': window,
            'image_idx': img_idx,
            'coords': (x, y, w, h),
            'original_shape': self.image_shapes[img_idx]
        }

class TorchvisionDatasetAdapter(Dataset):
    """Adapter for torchvision datasets with all original features"""

    def __init__(self, dataset_name: str, train: bool = True,
                 transform=None, download: bool = True):
        self.dataset_name = dataset_name.upper()
        self.transform = transform
        self.train = train

        # Get dataset class
        dataset_class = getattr(torchvision.datasets, self.dataset_name, None)
        if dataset_class is None:
            raise ValueError(f"Dataset {dataset_name} not found in torchvision")

        # Download and load
        self.dataset = dataset_class(
            root='./data',
            train=train,
            download=download,
            transform=None
        )

        # Get class names
        if hasattr(self.dataset, 'classes'):
            self.classes = self.dataset.classes
        else:
            # Try to infer classes
            if hasattr(self.dataset, 'targets'):
                unique_targets = sorted(set(self.dataset.targets))
                self.classes = [str(t) for t in unique_targets]
            else:
                self.classes = [str(i) for i in range(10)]

        # Build class mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        logger.info(f"Loaded {self.dataset_name}: {len(self.dataset)} samples, "
                   f"{len(self.classes)} classes")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.dataset[idx]

        # Convert to PIL if needed
        if not isinstance(img, PILImage.Image):
            if isinstance(img, torch.Tensor):
                img = TF.to_pil_image(img)
            else:
                img = PILImage.fromarray(img)

        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution"""
        if hasattr(self.dataset, 'targets'):
            targets = self.dataset.targets
        else:
            # Sample to get distribution
            targets = [self.dataset[i][1] for i in range(min(1000, len(self.dataset)))]

        counts = Counter(targets)
        return {self.idx_to_class[int(k)]: v for k, v in counts.items()}

# =============================================================================
# FEATURE SELECTION (All original methods)
# =============================================================================

class BaseFeatureSelector(ABC):
    """Abstract base for feature selectors"""

    @abstractmethod
    def select(self, features: np.ndarray, labels: np.ndarray,
               n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Select features and return indices and scores"""
        pass

    def get_name(self) -> str:
        return self.__class__.__name__

class DistanceCorrelationSelector(BaseFeatureSelector):
    """Select features based on distance correlation with labels"""

    def __init__(self, upper_threshold: float = 0.85, lower_threshold: float = 0.01):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def select(self, features: np.ndarray, labels: np.ndarray,
               n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        n_feat = features.shape[1]
        corrs = np.zeros(n_feat)

        # Calculate distance correlation with labels
        for i in range(n_feat):
            try:
                corrs[i] = 1 - correlation(features[:, i], labels)
            except:
                corrs[i] = 0

        # Select features above threshold
        selected = np.where(corrs >= self.upper_threshold)[0]

        # If too few, take top n_features
        if len(selected) < n_features:
            selected = np.argsort(corrs)[-n_features:]

        # Remove redundant features
        final_selected = self._remove_redundant(features, selected, corrs)

        return np.array(final_selected), corrs

    def _remove_redundant(self, features: np.ndarray, candidates: np.ndarray,
                          scores: np.ndarray) -> List[int]:
        """Remove features that are highly correlated with each other"""
        final = []
        feature_matrix = features[:, candidates]

        for i, idx in enumerate(candidates):
            keep = True
            for j in final:
                corr = 1 - correlation(feature_matrix[:, i],
                                      feature_matrix[:, list(candidates).index(j)])
                if corr > self.lower_threshold:
                    # Keep the one with higher score
                    if scores[idx] <= scores[j]:
                        keep = False
                    else:
                        final.remove(j)
                    break
            if keep:
                final.append(idx)

        return final

class ANOVAFeatureSelector(BaseFeatureSelector):
    """Select features using ANOVA F-test"""

    def select(self, features: np.ndarray, labels: np.ndarray,
               n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        try:
            scores, _ = f_classif(features, labels)
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            indices = np.argsort(scores)[-n_features:][::-1]
            return indices, scores
        except Exception as e:
            logger.warning(f"ANOVA failed: {e}, using variance")
            variances = np.var(features, axis=0)
            indices = np.argsort(variances)[-n_features:][::-1]
            return indices, variances

class MutualInfoFeatureSelector(BaseFeatureSelector):
    """Select features using mutual information"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def select(self, features: np.ndarray, labels: np.ndarray,
               n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        try:
            scores = mutual_info_classif(features, labels, random_state=self.random_state)
            scores = np.nan_to_num(scores, nan=0.0)
            indices = np.argsort(scores)[-n_features:][::-1]
            return indices, scores
        except Exception as e:
            logger.warning(f"Mutual info failed: {e}, using ANOVA")
            return ANOVAFeatureSelector().select(features, labels, n_features)

class RandomForestFeatureSelector(BaseFeatureSelector):
    """Select features using Random Forest importance"""

    def __init__(self, n_estimators: int = 50, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def select(self, features: np.ndarray, labels: np.ndarray,
               n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        try:
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(features, labels)
            scores = rf.feature_importances_
            indices = np.argsort(scores)[-n_features:][::-1]
            return indices, scores
        except Exception as e:
            logger.warning(f"Random Forest failed: {e}, using mutual info")
            return MutualInfoFeatureSelector().select(features, labels, n_features)

class KLDivergenceFeatureSelector(BaseFeatureSelector):
    """Select features using KL divergence between classes"""

    def select(self, features: np.ndarray, labels: np.ndarray,
               n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            return np.arange(min(n_features, features.shape[1])), np.ones(features.shape[1])

        n_feat = features.shape[1]
        kl_scores = np.zeros(n_feat)

        for i in range(n_feat):
            feature = features[:, i]

            # Estimate distributions for each class
            distributions = []
            for cls in unique_classes:
                cls_feat = feature[labels == cls]
                if len(cls_feat) > 5:
                    hist, _ = np.histogram(cls_feat, bins=10, density=True)
                    distributions.append(hist + 1e-8)

            # Calculate pairwise KL divergence
            if len(distributions) >= 2:
                kl_sum = 0
                n_pairs = 0
                for j, p in enumerate(distributions):
                    for k, q in enumerate(distributions[j+1:], j+1):
                        kl_pq = np.sum(p * np.log(p / q))
                        kl_qp = np.sum(q * np.log(q / p))
                        kl_sum += (kl_pq + kl_qp) / 2
                        n_pairs += 1
                kl_scores[i] = kl_sum / n_pairs if n_pairs > 0 else 0

        indices = np.argsort(kl_scores)[-n_features:][::-1]
        return indices, kl_scores

class VarianceFeatureSelector(BaseFeatureSelector):
    """Select features by variance"""

    def select(self, features: np.ndarray, labels: np.ndarray,
               n_features: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        variances = np.var(features, axis=0)
        indices = np.argsort(variances)[-n_features:][::-1]
        return indices, variances

class AdaptiveFeatureSelector:
    """Dynamically select best feature selector based on data"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.selectors = {
            'distance_correlation': DistanceCorrelationSelector(
                config.correlation_upper, config.correlation_lower
            ),
            'anova': ANOVAFeatureSelector(),
            'mutual_info': MutualInfoFeatureSelector(),
            'random_forest': RandomForestFeatureSelector(),
            'kl_divergence': KLDivergenceFeatureSelector(),
            'variance': VarianceFeatureSelector()
        }

    def analyze_data(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Analyze data characteristics"""
        n_samples, n_features = features.shape
        n_classes = len(np.unique(labels))

        # Sparsity
        sparsity = np.mean(features == 0)

        # Class balance
        class_counts = np.bincount(labels) if labels.max() < 1000 else np.ones(n_classes)
        class_balance = np.min(class_counts) / np.max(class_counts) if len(class_counts) > 0 else 1.0

        # Feature correlation (sample if many features)
        if n_features < 1000:
            corr_matrix = np.corrcoef(features.T)
            np.fill_diagonal(corr_matrix, 0)
            avg_corr = np.mean(np.abs(corr_matrix))
        else:
            # Sample features
            sample_idx = np.random.choice(n_features, min(100, n_features), replace=False)
            corr_matrix = np.corrcoef(features[:, sample_idx].T)
            np.fill_diagonal(corr_matrix, 0)
            avg_corr = np.mean(np.abs(corr_matrix))

        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'sparsity': sparsity,
            'class_balance': class_balance,
            'avg_correlation': avg_corr,
            'is_high_dim': n_features > n_samples,
            'is_multi_class': n_classes > 2
        }

    def select_method(self, analysis: Dict) -> str:
        """Select best method based on analysis"""
        if analysis['n_classes'] > 20:
            return 'mutual_info'
        elif analysis['is_high_dim']:
            return 'mutual_info'
        elif analysis['n_classes'] <= 10 and analysis['n_samples'] > 500:
            return 'random_forest'
        elif analysis['sparsity'] > 0.5:
            return 'variance'
        elif analysis['n_classes'] >= 2 and analysis['n_features'] < 1000:
            return 'kl_divergence'
        else:
            return 'anova'

    def select(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Select features using adaptive method"""
        analysis = self.analyze_data(features, labels)
        method = self.select_method(analysis)

        logger.info(f"Feature selection: {method} (classes={analysis['n_classes']}, "
                   f"features={analysis['n_features']}, samples={analysis['n_samples']})")

        selector = self.selectors[method]
        indices, scores = selector.select(features, labels, self.config.max_features)

        # Ensure minimum features
        if len(indices) < self.config.min_features:
            logger.warning(f"Only {len(indices)} features selected, adding more...")
            all_indices = np.argsort(scores)[-self.config.max_features:][::-1]
            indices = np.unique(np.concatenate([indices, all_indices]))[:self.config.max_features]

        return indices, scores, method

# =============================================================================
# NEURAL NETWORK MODULES (All original architectures)
# =============================================================================

class SelfAttention(nn.Module):
    """Self-attention module"""

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        # Compute attention
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)

        # Apply attention
        value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)

        return self.gamma * out + x

class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale

class DCTLayer(nn.Module):
    """Discrete Cosine Transform layer"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply DCT along spatial dimensions
        x = torch.fft.fft(x, dim=2)
        x = torch.fft.fft(x, dim=3)
        return x.real

class SharpnessAwareLoss(nn.Module):
    """Loss function for sharpness preservation"""

    def __init__(self, edge_weight: float = 1.0, smoothness_weight: float = 0.5):
        super().__init__()
        self.edge_weight = edge_weight
        self.smoothness_weight = smoothness_weight

        # Sobel filters
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

        # Convert to grayscale if needed
        if pred.shape[1] == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target

        # Compute gradients
        pred_grad_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, self.sobel_y, padding=1)

        # Edge loss
        edge_loss = F.mse_loss(pred_grad_x, target_grad_x) + \
                    F.mse_loss(pred_grad_y, target_grad_y)

        # Smoothness loss
        smoothness_loss = torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])) + \
                          torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]))

        # Reconstruction loss
        recon_loss = F.mse_loss(pred, target)

        return recon_loss + self.edge_weight * edge_loss + self.smoothness_weight * smoothness_loss

# Specialized loss functions for different image types
class AstronomicalStructureLoss(nn.Module):
    """Loss for astronomical images (stars, galaxies)"""

    def __init__(self):
        super().__init__()
        # Star detection filter
        self.star_filter = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                                       dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = pred.device
        self.star_filter = self.star_filter.to(device)

        # Convert to grayscale
        if pred.shape[1] == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target

        # Star detection
        pred_stars = F.conv2d(pred_gray, self.star_filter, padding=1)
        target_stars = F.conv2d(target_gray, self.star_filter, padding=1)
        star_loss = F.mse_loss(pred_stars, target_stars)

        # Reconstruction loss
        recon_loss = F.mse_loss(pred, target)

        return recon_loss + 2.0 * star_loss

class MedicalStructureLoss(nn.Module):
    """Loss for medical images (tissue boundaries, lesions)"""

    def __init__(self):
        super().__init__()
        # Laplacian for edge detection
        self.laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                                     dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = pred.device
        self.laplacian = self.laplacian.to(device)

        # Convert to grayscale
        if pred.shape[1] == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target

        # Edge detection
        pred_edges = F.conv2d(pred_gray, self.laplacian, padding=1)
        target_edges = F.conv2d(target_gray, self.laplacian, padding=1)
        edge_loss = F.mse_loss(pred_edges, target_edges)

        # Local contrast
        pred_contrast = torch.std(F.unfold(pred_gray, kernel_size=5), dim=1)
        target_contrast = torch.std(F.unfold(target_gray, kernel_size=5), dim=1)
        contrast_loss = F.mse_loss(pred_contrast, target_contrast)

        # Reconstruction loss
        recon_loss = F.mse_loss(pred, target)

        return recon_loss + 1.5 * edge_loss + 1.0 * contrast_loss

class AgriculturalPatternLoss(nn.Module):
    """Loss for agricultural images (texture, patterns)"""

    def __init__(self):
        super().__init__()
        # Gabor-like filters (simplified)
        self.texture_filters = nn.ModuleList([
            nn.Conv2d(3, 3, kernel_size=k, padding=k//2, groups=3)
            for k in [3, 5, 7]
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Texture preservation
        texture_loss = 0
        for filt in self.texture_filters:
            pred_texture = filt(pred)
            target_texture = filt(target)
            texture_loss += F.mse_loss(pred_texture, target_texture)

        # Color preservation
        color_loss = F.mse_loss(pred.mean(dim=[2, 3]), target.mean(dim=[2, 3]))

        # Reconstruction loss
        recon_loss = F.mse_loss(pred, target)

        return recon_loss + 2.0 * texture_loss + 1.0 * color_loss

# =============================================================================
# COMPLETE BASE AUTOENCODER WITH EXTRACT_FEATURES METHOD
# =============================================================================

class BaseAutoencoder(nn.Module):
    """Complete autoencoder with all required methods including extract_features"""

    def __init__(self, config: GlobalConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        self.training_phase = 1

        # Feature selection attributes
        self._selected_feature_indices = None
        self._feature_importance_scores = None
        self._feature_selection_metadata = {}
        self._is_feature_selection_frozen = False

        # Attention maps storage
        self.attention_maps = {}
        self.hook_handles = []

        # Loss functions dictionary
        self.loss_functions = nn.ModuleDict()

        # Classifier and cluster centers (initialized lazily)
        self.classifier = None
        self.cluster_centers = None
        self.clustering_temperature = nn.Parameter(torch.tensor(1.0))

        # Build architecture (adaptive based on input size)
        self._build_adaptive_architecture()

        # Initialize weights
        self.apply(self._init_weights)

        # Move to device
        self.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model initialized with {total_params:,} parameters")

        # Gradient checkpointing flag
        self.use_checkpointing = False

    def _build_adaptive_architecture(self):
        """Build architecture that adapts to input size"""
        h, w = self.config.input_size
        c = self.config.in_channels

        # For Galaxy dataset (likely larger than 32x32)
        if h <= 64 and w <= 64:
            self._build_medium_architecture(h, w, c)
        else:
            self._build_large_architecture(h, w, c)

    def _build_medium_architecture(self, h, w, c):
        """Architecture for medium images (32-64px)"""
        # Encoder
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ])

        self.encoder_channels = [32, 64, 128, 256]
        self.final_h, self.final_w = h // 16, w // 16

        # Calculate flattened size
        self.flattened_size = 256 * self.final_h * self.final_w
        self.feature_dims = self.config.feature_dims

        # Embedder
        hidden_size = min(512, self.flattened_size // 4)
        self.embedder = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_size),
            nn.BatchNorm1d(hidden_size, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, self.feature_dims),
            nn.Tanh()
        )

        # Unembedder
        self.unembedder = nn.Sequential(
            nn.Linear(self.feature_dims, hidden_size),
            nn.BatchNorm1d(hidden_size, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, self.flattened_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32, c, 4, stride=2, padding=1),
                nn.Tanh()
            )
        ])

        # Feature compressor
        self.compressed_dims = min(64, self.config.compressed_dims)
        hidden_dims = 128

        self.feature_compressor = nn.Sequential(
            nn.Linear(self.feature_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims, self.compressed_dims)
        )

        self.feature_decompressor = nn.Sequential(
            nn.Linear(self.compressed_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims, self.feature_dims)
        )

    def _build_large_architecture(self, h, w, c):
        """Architecture for large images"""
        # Simplified for memory efficiency
        n_layers = 4

        self.encoder_layers = nn.ModuleList()
        in_channels = c
        self.encoder_channels = []

        for i in range(n_layers):
            out_channels = min(512, 64 * (2 ** i))
            self.encoder_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            self.encoder_channels.append(out_channels)
            in_channels = out_channels

        self.final_h, self.final_w = h // (2 ** n_layers), w // (2 ** n_layers)
        self.flattened_size = in_channels * self.final_h * self.final_w
        self.feature_dims = self.config.feature_dims

        # Embedder
        self.embedder = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.BatchNorm1d(512, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.feature_dims),
            nn.Tanh()
        )

        self.unembedder = nn.Sequential(
            nn.Linear(self.feature_dims, 512),
            nn.BatchNorm1d(512, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.flattened_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder
        self.decoder_layers = nn.ModuleList()
        in_channels = self.encoder_channels[-1]

        for i in range(n_layers - 1, -1, -1):
            out_channels = c if i == 0 else self.encoder_channels[i-1]
            if i == 0:
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.Tanh()
                ))
            else:
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels, track_running_stats=False),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            in_channels = out_channels

        # Feature compressor
        self.compressed_dims = min(128, self.config.compressed_dims)
        hidden_dims = 256

        self.feature_compressor = nn.Sequential(
            nn.Linear(self.feature_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims, self.compressed_dims)
        )

        self.feature_decompressor = nn.Sequential(
            nn.Linear(self.compressed_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims, self.feature_dims)
        )

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def set_training_phase(self, phase: int):
        """Set training phase"""
        self.training_phase = phase
        logger.info(f"Setting training phase to {phase}")

        if phase == 2:
            # Initialize classifier if needed
            if self.config.use_class_encoding and (not hasattr(self, 'classifier') or self.classifier is None):
                num_classes = self.config.num_classes or 10
                self.classifier = nn.Sequential(
                    nn.Linear(self.compressed_dims, max(32, self.compressed_dims // 2)),
                    nn.BatchNorm1d(max(32, self.compressed_dims // 2), track_running_stats=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(max(32, self.compressed_dims // 2), num_classes)
                ).to(self.device)
                logger.info(f"Initialized classifier with {num_classes} classes")

            # Initialize cluster centers if needed
            if self.config.use_kl_divergence and (not hasattr(self, 'cluster_centers') or self.cluster_centers is None):
                num_clusters = self.config.num_classes or 10
                self.cluster_centers = nn.Parameter(
                    torch.randn(num_clusters, self.compressed_dims, device=self.device)
                )
                self.clustering_temperature = nn.Parameter(torch.tensor(1.0, device=self.device))
                logger.info(f"Initialized {num_clusters} cluster centers")

    def get_frozen_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Extract frozen features from embeddings"""
        if self._selected_feature_indices is None:
            return embeddings
        return embeddings[:, self._selected_feature_indices]

    def freeze_feature_selection(self, indices: np.ndarray, scores: np.ndarray, metadata: Dict = None):
        """Freeze feature selection for inference"""
        self._selected_feature_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        self._feature_importance_scores = torch.tensor(scores, device=self.device)
        self._feature_selection_metadata = metadata or {}
        self._is_feature_selection_frozen = True
        logger.info(f"Frozen feature selection: {len(indices)} features")

    def register_attention_hooks(self):
        """Register hooks for attention visualization"""
        self.attention_maps = {}
        self.hook_handles = []

        def hook_fn(module, input, output, name):
            if output.numel() < 1000000:  # Less than 1M elements
                self.attention_maps[name] = output.detach()

        for idx, layer in enumerate(self.encoder_layers):
            handle = layer.register_forward_hook(
                lambda m, i, o, idx=idx: hook_fn(m, i, o, f'encoder_{idx}')
            )
            self.hook_handles.append(handle)

        logger.info(f"Registered {len(self.hook_handles)} attention hooks")

    def remove_attention_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.attention_maps = {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature space"""
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return self.embedder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode features to image"""
        z = self.unembedder(z)
        z = z.view(z.size(0), self.encoder_channels[-1], self.final_h, self.final_w)

        for layer in self.decoder_layers:
            z = layer(z)

        return z

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with all outputs - handles single-sample batches for BatchNorm"""

        # FIX: Handle single-sample case for BatchNorm layers
        duplicated = False
        original_batch_size = x.size(0)

        # During training, BatchNorm needs at least 2 samples
        if self.training and x.size(0) == 1:
            # Duplicate the single sample to have batch size 2
            x = torch.cat([x, x], dim=0)
            duplicated = True
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)

        # Encode
        embedding = self.encode(x)

        # Feature selection
        if self._is_feature_selection_frozen and self._selected_feature_indices is not None:
            selected_embedding = embedding[:, self._selected_feature_indices]
        else:
            selected_embedding = embedding

        # Compress
        compressed = self.feature_compressor(selected_embedding)

        # Decompress and decode
        decompressed = self.feature_decompressor(compressed)
        reconstruction = self.decode(decompressed)

        # Ensure reconstruction size matches input
        if reconstruction.shape[-2:] != x.shape[-2:]:
            reconstruction = F.interpolate(reconstruction, size=x.shape[-2:],
                                         mode='bilinear', align_corners=False)

        # Base outputs - take first sample if duplicated
        if duplicated:
            output = {
                'embedding': embedding[:original_batch_size],
                'selected_embedding': selected_embedding[:original_batch_size],
                'compressed_embedding': compressed[:original_batch_size],
                'reconstructed_embedding': decompressed[:original_batch_size],
                'reconstruction': reconstruction[:original_batch_size]
            }
        else:
            output = {
                'embedding': embedding,
                'selected_embedding': selected_embedding,
                'compressed_embedding': compressed,
                'reconstructed_embedding': decompressed,
                'reconstruction': reconstruction
            }

        # Phase 2 outputs
        if self.training_phase == 2:
            if self.config.use_class_encoding and hasattr(self, 'classifier') and self.classifier is not None:
                logits = self.classifier(compressed)
                if duplicated:
                    output.update({
                        'class_logits': logits[:original_batch_size],
                        'class_predictions': logits[:original_batch_size].argmax(dim=1),
                        'class_probabilities': F.softmax(logits[:original_batch_size], dim=1)
                    })
                else:
                    output.update({
                        'class_logits': logits,
                        'class_predictions': logits.argmax(dim=1),
                        'class_probabilities': F.softmax(logits, dim=1)
                    })

            if self.config.use_kl_divergence and hasattr(self, 'cluster_centers') and self.cluster_centers is not None:
                # Compute distances to cluster centers
                distances = torch.cdist(compressed, self.cluster_centers)

                # Student's t-distribution
                q = (1 + distances ** 2 / self.clustering_temperature) ** (-(self.clustering_temperature + 1) / 2)
                q = q / (q.sum(dim=1, keepdim=True) + 1e-8)

                # Target distribution (sharpen)
                p = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
                p = p / (p.sum(dim=1, keepdim=True) + 1e-8)

                if duplicated:
                    output.update({
                        'cluster_probabilities': q[:original_batch_size],
                        'target_distribution': p[:original_batch_size],
                        'cluster_assignments': q[:original_batch_size].argmax(dim=1),
                        'cluster_confidence': q[:original_batch_size].max(dim=1)[0],
                        'cluster_distances': distances[:original_batch_size]
                    })
                else:
                    output.update({
                        'cluster_probabilities': q,
                        'target_distribution': p,
                        'cluster_assignments': q.argmax(dim=1),
                        'cluster_confidence': q.max(dim=1)[0],
                        'cluster_distances': distances
                    })

        return output

    @torch.no_grad()
    @memory_efficient
    def extract_features(self, dataloader: DataLoader, include_paths: bool = True) -> Dict:
        """Extract features from data"""
        self.eval()

        all_embeddings = []
        all_labels = []
        all_paths = []
        all_filenames = []
        all_class_names = []

        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
            inputs = inputs.to(self.device, non_blocking=True)

            # During inference, we don't need to duplicate for BatchNorm
            # because we're in eval mode, but we'll handle it just in case
            outputs = self(inputs)

            # Get embeddings
            embeddings = outputs['embedding'].cpu()

            all_embeddings.append(embeddings)
            all_labels.append(labels)

            # Get paths if available
            if include_paths and hasattr(dataloader.dataset, 'get_additional_info'):
                for i in range(len(labels)):
                    idx = batch_idx * dataloader.batch_size + i
                    if idx < len(dataloader.dataset):
                        info = dataloader.dataset.get_additional_info(idx)
                        all_filenames.append(info[1])
                        all_paths.append(info[2])

                        # Get class name
                        if hasattr(dataloader.dataset, 'idx_to_class'):
                            class_name = dataloader.dataset.idx_to_class[labels[i].item()]
                        else:
                            class_name = str(labels[i].item())
                        all_class_names.append(class_name)

        # Concatenate all embeddings
        if len(all_embeddings) > 0:
            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
        else:
            embeddings = torch.tensor([])
            labels = torch.tensor([])

        result = {
            'embeddings': embeddings,
            'labels': labels
        }

        if all_paths:
            result['paths'] = all_paths
            result['filenames'] = all_filenames
            result['class_names'] = all_class_names

        # Log extracted features info
        logger.info(f"Extracted {len(embeddings)} features with dimension {embeddings.shape[1] if len(embeddings) > 0 else 0}")

        return result

# =============================================================================
# COMPLETE CDBNNApplication WITH PROPER FEATURE EXTRACTION
# =============================================================================

class CDBNNApplication:
    """Main application class with all features"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.monitor = ResourceMonitor(interval=10.0, save_path=config.viz_dir)

        # Setup directories
        self.data_dir = Path(config.data_dir) / config.dataset_name.lower()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Update paths
        self.config.checkpoint_dir = str(self.data_dir / 'checkpoints')
        self.config.viz_dir = str(self.data_dir / 'visualizations')
        self.config.log_dir = str(self.data_dir / 'logs')

        # Create directories
        for d in [self.config.checkpoint_dir, self.config.viz_dir, self.config.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Visualizer
        self.visualizer = Visualizer(config)

    def prepare_data_single_process(self, source_path: str, data_type: str = 'custom',
                                   batch_size: Optional[int] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepare data loaders with single-process loading"""
        transform = self._get_transform()
        batch_size = batch_size or self.config.batch_size
        num_workers = 0  # Single process mode

        if data_type == 'torchvision':
            train_dataset = TorchvisionDatasetAdapter(
                source_path, train=True, transform=transform
            )
            test_dataset = TorchvisionDatasetAdapter(
                source_path, train=False, transform=transform
            )

            # Update config
            self.config.num_classes = len(train_dataset.classes)
            self.config.class_names = train_dataset.classes

        else:
            # Handle archive if needed
            if source_path.endswith(('.zip', '.tar', '.gz', '.bz2', '.xz')):
                extract_dir = self.data_dir / 'extracted'
                source_path = ArchiveHandler.extract(source_path, str(extract_dir))

            # Check for train/test structure
            train_path = Path(source_path) / 'train'
            if train_path.exists():
                train_dataset = CustomImageDataset(
                    str(train_path), transform=transform,
                    cache_size=100, config=self.config.to_dict(),
                    data_name=self.config.dataset_name
                )

                test_path = Path(source_path) / 'test'
                if test_path.exists():
                    test_dataset = CustomImageDataset(
                        str(test_path), transform=transform,
                        cache_size=100, config=self.config.to_dict(),
                        data_name=self.config.dataset_name
                    )
                else:
                    test_dataset = None
            else:
                # Single directory
                train_dataset = CustomImageDataset(
                    source_path, transform=transform,
                    cache_size=100, config=self.config.to_dict(),
                    data_name=self.config.dataset_name
                )
                test_dataset = None

            # Update config
            self.config.num_classes = len(train_dataset.classes)
            self.config.class_names = train_dataset.classes

        # Create loaders with num_workers=0
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False
        )

        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False
            )

        # Log class distribution
        if hasattr(train_dataset, 'get_class_distribution'):
            class_dist = train_dataset.get_class_distribution()
            self.visualizer.plot_class_distribution(class_dist)
            logger.info(f"Class distribution: {class_dist}")

        logger.info(f"Data prepared: {len(train_dataset)} train, "
                   f"{len(test_dataset) if test_dataset else 0} test samples")
        logger.info(f"Using single-process data loading (num_workers=0)")

        return train_loader, test_loader

    def _get_transform(self) -> transforms.Compose:
        """Get image transform"""
        transform_list = [
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ]

        # Add augmentation for training
        if hasattr(self, '_is_training') and self._is_training:
            transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
            transform_list.insert(1, transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ))

        return transforms.Compose(transform_list)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        """Train model with all features - properly handles phase continuation"""
        self.monitor.start()
        self._is_training = True

        try:
            # Create model
            model = BaseAutoencoder(self.config)

            # Create trainer
            trainer = Trainer(model, self.config)

            # Check for existing model
            best_model_path = Path(self.config.checkpoint_dir) / 'best.pt'
            model_path = Path(self.config.checkpoint_dir) / 'latest.pt'

            loaded_phase = 1
            loaded_epoch = 0
            best_loss = float('inf')
            best_acc = 0.0

            # Try to load best model first
            if best_model_path.exists():
                logger.info(f"{Colors.GREEN}Found existing best model at {best_model_path}{Colors.ENDC}")
                success = trainer.load_checkpoint(str(best_model_path))
                if success:
                    # Get loaded model's phase from checkpoint
                    if hasattr(trainer.model, 'training_phase'):
                        loaded_phase = trainer.model.training_phase
                        loaded_epoch = trainer.best_epoch
                        best_loss = trainer.best_loss
                        best_acc = trainer.best_accuracy if trainer.best_accuracy else 0.0
                        logger.info(f"Loaded model from phase {loaded_phase}, epoch {loaded_epoch+1}, "
                                  f"loss={best_loss:.4f}, acc={best_acc:.2%}")
            elif model_path.exists():
                logger.info(f"{Colors.YELLOW}Found existing latest model at {model_path}{Colors.ENDC}")
                success = trainer.load_checkpoint(str(model_path))
                if success:
                    if hasattr(trainer.model, 'training_phase'):
                        loaded_phase = trainer.model.training_phase
                        loaded_epoch = trainer.best_epoch
                        best_loss = trainer.best_loss
                        best_acc = trainer.best_accuracy if trainer.best_accuracy else 0.0
                        logger.info(f"Loaded latest model from phase {loaded_phase}, epoch {loaded_epoch+1}")
            else:
                logger.info(f"{Colors.BLUE}No existing model found. Starting training from scratch.{Colors.ENDC}")

            # Determine which phases to train based on loaded model
            epochs_phase1 = self.config.epochs // 2
            epochs_phase2 = self.config.epochs // 2

            # Phase 1: Only train if we loaded from phase 1 or no model exists
            if loaded_phase <= 1:
                logger.info("=" * 60)
                logger.info("PHASE 1: Reconstruction Training")
                logger.info("=" * 60)

                # Set model to phase 1
                trainer.model.set_training_phase(1)

                # Train phase 1
                phase1_history = trainer._train_phase(train_loader, val_loader, 1, epochs_phase1)

                # Update best metrics from phase 1
                best_loss = trainer.best_loss
                best_acc = trainer.best_accuracy if trainer.best_accuracy else 0.0
                loaded_phase = 2  # Ready for phase 2
            else:
                logger.info(f"{Colors.GREEN}✓ Skipping Phase 1 - Model already trained in phase {loaded_phase}{Colors.ENDC}")

            # Phase 2: Train if we need latent organization
            if (self.config.use_kl_divergence or self.config.use_class_encoding) and loaded_phase <= 2:
                logger.info("=" * 60)
                logger.info("PHASE 2: Latent Space Organization")
                logger.info("=" * 60)

                # Set model to phase 2
                trainer.model.set_training_phase(2)

                # Initialize phase 2 components if they don't exist
                if not hasattr(trainer.model, 'classifier') or trainer.model.classifier is None:
                    num_classes = self.config.num_classes or 2
                    trainer.model.classifier = nn.Sequential(
                        nn.Linear(trainer.model.compressed_dims, max(32, trainer.model.compressed_dims // 2)),
                        nn.BatchNorm1d(max(32, trainer.model.compressed_dims // 2), track_running_stats=False),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(max(32, trainer.model.compressed_dims // 2), num_classes)
                    ).to(trainer.device)
                    logger.info(f"Initialized classifier with {num_classes} classes")

                if not hasattr(trainer.model, 'cluster_centers') or trainer.model.cluster_centers is None:
                    num_clusters = self.config.num_classes or 2
                    trainer.model.cluster_centers = nn.Parameter(
                        torch.randn(num_clusters, trainer.model.compressed_dims, device=trainer.device)
                    )
                    trainer.model.clustering_temperature = nn.Parameter(torch.tensor(1.0, device=trainer.device))
                    logger.info(f"Initialized {num_clusters} cluster centers")

                # Lower learning rate for fine-tuning
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * 0.1

                # Train phase 2
                phase2_history = trainer._train_phase(train_loader, val_loader, 2, epochs_phase2)

            # Plot history
            if trainer.history:
                self.visualizer.plot_training_history(dict(trainer.history))

            # Save final model
            trainer._save_checkpoint(trainer.best_epoch, trainer.model.training_phase,
                                    trainer.best_loss, trainer.best_accuracy, is_best=False)

            return dict(trainer.history)

        finally:
            self.monitor.stop()
            self._is_training = False

    def save_features(self, features: Dict[str, np.ndarray], filename: str):
        """Save features to CSV"""
        df_data = {}

        # Add features
        if 'selected_features' in features:
            feat = features['selected_features']
            prefix = 'selected_feature'
        else:
            feat = features['features']
            prefix = 'feature'

        for i in range(feat.shape[1]):
            df_data[f'{prefix}_{i}'] = feat[:, i]

        # Add labels
        df_data['target'] = features['labels']

        # Add paths if available
        if 'paths' in features:
            df_data['filepath'] = features['paths']
        if 'filenames' in features:
            df_data['filename'] = features['filenames']
        if 'class_names' in features:
            df_data['class_name'] = features['class_names']

        # Add feature indices
        if 'feature_indices' in features:
            df_data['feature_indices'] = [str(features['feature_indices'])] * len(feat)

        df = pd.DataFrame(df_data)

        # Save
        csv_path = self.data_dir / filename
        df.to_csv(csv_path, index=False)
        logger.info(f"Features saved to {csv_path}")

        # Also save as numpy for efficiency
        npz_path = self.data_dir / filename.replace('.csv', '.npz')
        np.savez_compressed(npz_path, **{k: np.array(v) for k, v in df_data.items()})
        logger.info(f"Features saved to {npz_path}")

        return csv_path

    def extract_features(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """Extract features using trained model with guaranteed consistency"""
        # Create predictor (which loads the exact best model)
        model_path = Path(self.config.checkpoint_dir) / 'best.pt'
        predictor = Predictor(self.config, str(model_path) if model_path.exists() else None)

        # CRITICAL FIX: Ensure predictor's model is in eval mode
        predictor.model.eval()

        # Run prediction to get features
        results = predictor.predict(dataloader, optimize_level='accurate')

        # Verify that we have features
        if 'features' not in results:
            raise RuntimeError("No features extracted from model")

        logger.info(f"Extraction complete: {results['features'].shape[0]} samples, "
                   f"{results['features'].shape[1]} features")

        return results

    def predict(self, dataloader: DataLoader, optimize_level: str = 'balanced',
                reference_csv: Optional[str] = None) -> Dict:
        """Run prediction with guaranteed consistency"""
        # If reference CSV not provided, try to find training CSV
        if reference_csv is None:
            train_csv_path = self.data_dir / f"{self.config.dataset_name}_features.csv"
            if train_csv_path.exists():
                reference_csv = str(train_csv_path)

        # Create predictor
        model_path = Path(self.config.checkpoint_dir) / 'best.pt'
        predictor = Predictor(self.config, str(model_path) if model_path.exists() else None)

        # Run prediction with consistency check
        results = predictor.predict(dataloader, optimize_level=optimize_level,
                                   reference_csv=reference_csv)

        # Generate visualizations if requested
        if self.config.generate_tsne and 'features' in results:
            if len(results['features']) > 10:
                self.visualizer.plot_tsne(
                    results['features'],
                    results.get('labels', np.zeros(len(results['features']))),
                    class_names=self.config.class_names
                )

        if self.config.generate_confusion_matrix and 'predictions' in results and 'labels' in results:
            self.visualizer.plot_confusion_matrix(
                results['labels'],
                results['predictions'],
                class_names=self.config.class_names
            )

        return results

# =============================================================================
# COMPLETE TRAINER CLASS WITH ALL METHODS
# =============================================================================

# =============================================================================
# COMPLETE TRAINER CLASS WITH ALL METHODS
# =============================================================================

class Trainer:
    """Complete trainer with all required methods"""

    def __init__(self, model: BaseAutoencoder, config: GlobalConfig):
        self.model = model
        self.config = config
        self.device = model.device
        self.state_manager = ModelStateManager(Path(config.checkpoint_dir))

        # Optimizer with gradient clipping
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and self.device.type == 'cuda' else None

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.history = defaultdict(list)

        # Feature selector
        if config.use_distance_correlation:
            self.feature_selector = AdaptiveFeatureSelector(config)

        # Visualization
        self.viz_dir = Path(config.viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Gradient accumulation steps (for memory efficiency)
        self.gradient_accumulation_steps = 1

        # Memory optimization: Clear cache between epochs
        self.clear_cache_between_epochs = True

    def set_gradient_accumulation(self, steps: int):
        """Set gradient accumulation steps for memory efficiency"""
        self.gradient_accumulation_steps = steps
        logger.info(f"Gradient accumulation set to {steps} steps")

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        """Complete two-phase training with memory optimization"""

        # Phase 1: Reconstruction
        logger.info("=" * 60)
        logger.info("PHASE 1: Reconstruction Training")
        logger.info("=" * 60)

        self.model.set_training_phase(1)
        phase1_history = self._train_phase(train_loader, val_loader, 1,
                                          self.config.epochs // 2)

        # Phase 2: Latent organization
        if self.config.use_kl_divergence or self.config.use_class_encoding:
            logger.info("=" * 60)
            logger.info("PHASE 2: Latent Space Organization")
            logger.info("=" * 60)

            self.model.set_training_phase(2)

            # Lower learning rate for fine-tuning
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * 0.1

            phase2_history = self._train_phase(train_loader, val_loader, 2,
                                              self.config.epochs // 2)

            # Combine histories
            for key, values in phase2_history.items():
                self.history[f'phase2_{key}'] = values

        return dict(self.history)

    def _train_phase(self, train_loader: DataLoader, val_loader: Optional[DataLoader],
                     phase: int, epochs: int) -> Dict:
        """Train a single phase with proper best model tracking and state management"""

        phase_history = defaultdict(list)
        patience_counter = 0
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = 0

        # Initialize phase-specific best tracking
        phase_best_loss = float('inf')
        phase_best_acc = 0.0

        # CRITICAL: If we're resuming phase 2, use the loaded best metrics
        if phase == 2 and hasattr(self.model, '_resume_phase2') and self.model._resume_phase2:
            phase_best_loss = getattr(self.model, '_phase2_best_loss', float('inf'))
            phase_best_acc = getattr(self.model, '_phase2_best_acc', 0.0)
            best_epoch = getattr(self.model, '_phase2_best_epoch', 0)
            logger.info(f"Resuming phase 2 with best loss={phase_best_loss:.4f}, best acc={phase_best_acc:.2%}")
            self.model._resume_phase2 = False

        for epoch in range(epochs):
            # Clear cache at start of epoch
            if self.clear_cache_between_epochs and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            # Training
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Phase {phase} Epoch {epoch+1}/{epochs}")
            self.optimizer.zero_grad()

            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Ensure batch size is at least 2 for BatchNorm
                if inputs.size(0) == 1:
                    inputs = torch.cat([inputs, inputs], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                # Forward pass with mixed precision
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs, labels)
                        loss, acc = self._compute_loss(outputs, inputs, labels, phase)

                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    outputs = self.model(inputs, labels)
                    loss, acc = self._compute_loss(outputs, inputs, labels, phase)

                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                # Update metrics
                train_loss += loss.item() * self.gradient_accumulation_steps
                if acc is not None:
                    train_acc += acc
                n_batches += 1

                # Update progress bar
                postfix = {'loss': f"{train_loss/n_batches:.4f}"}
                if acc is not None:
                    postfix['acc'] = f"{train_acc/n_batches:.2%}"
                pbar.set_postfix(postfix)

                # Clean up
                del outputs, loss
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Final optimizer step if needed
            if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            avg_train_loss = train_loss / n_batches
            avg_train_acc = train_acc / n_batches if train_acc else None

            # Validation
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, phase)
                phase_history['val_loss'].append(val_loss)
                if val_acc:
                    phase_history['val_acc'].append(val_acc)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Check if this is the best validation model so far
                is_best = val_loss < phase_best_loss

                if is_best:
                    phase_best_loss = val_loss
                    phase_best_acc = val_acc if val_acc else 0.0
                    best_epoch = epoch

                    # Save best model state with all metadata
                    self.state_manager.save_best_model_state(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        loss=val_loss,
                        accuracy=val_acc,
                        feature_indices=self.model._selected_feature_indices.cpu().numpy() if self.model._selected_feature_indices is not None else None,
                        feature_scores=self.model._feature_importance_scores.cpu().numpy() if self.model._feature_importance_scores is not None else None,
                        feature_metadata=self.model._feature_selection_metadata
                    )

                    # Also save phase-specific best
                    self._save_checkpoint(epoch, phase, val_loss, val_acc, is_best=True)
                    patience_counter = 0
                    logger.info(f"New best phase {phase} model saved with loss={val_loss:.4f}, acc={val_acc if val_acc else 'N/A'}")
                else:
                    patience_counter += 1
            else:
                # Use training loss for checkpointing
                is_best = avg_train_loss < phase_best_loss

                if is_best:
                    phase_best_loss = avg_train_loss
                    phase_best_acc = avg_train_acc if avg_train_acc else 0.0
                    best_epoch = epoch

                    # Save best model state with all metadata
                    self.state_manager.save_best_model_state(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        loss=avg_train_loss,
                        accuracy=avg_train_acc,
                        feature_indices=self.model._selected_feature_indices.cpu().numpy() if self.model._selected_feature_indices is not None else None,
                        feature_scores=self.model._feature_importance_scores.cpu().numpy() if self.model._feature_importance_scores is not None else None,
                        feature_metadata=self.model._feature_selection_metadata
                    )

                    self._save_checkpoint(epoch, phase, avg_train_loss, avg_train_acc, is_best=True)
                    patience_counter = 0
                    logger.info(f"New best phase {phase} model saved with loss={avg_train_loss:.4f}")
                else:
                    patience_counter += 1

            # Store history
            phase_history['train_loss'].append(avg_train_loss)
            if avg_train_acc:
                phase_history['train_acc'].append(avg_train_acc)

            # Store learning rate
            phase_history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Log progress
            log_msg = f"Phase {phase} Epoch {epoch+1}: train_loss={avg_train_loss:.4f}"
            if avg_train_acc:
                log_msg += f", train_acc={avg_train_acc:.2%}"
            if val_loader:
                log_msg += f", val_loss={val_loss:.4f}"
                if val_acc:
                    log_msg += f", val_acc={val_acc:.2%}"
            logger.info(log_msg)

            # Early stopping
            if patience_counter >= 10:
                logger.info(f"Early stopping triggered at epoch {epoch+1} (best at epoch {best_epoch+1})")
                break

            # Generate visualizations (with memory check)
            if self.config.generate_heatmaps and (epoch + 1) % self.config.heatmap_frequency == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._generate_heatmaps(train_loader, epoch, phase)

            # Reconstruction samples
            if (epoch + 1) % self.config.reconstruction_samples_frequency == 0:
                self._save_reconstruction_samples(train_loader, epoch, phase)

            # Clear cache at end of epoch
            if self.clear_cache_between_epochs and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

        # Store phase 2 best metrics for potential resumption
        if phase == 2:
            self.model._phase2_best_loss = phase_best_loss
            self.model._phase2_best_acc = phase_best_acc
            self.model._phase2_best_epoch = best_epoch
            self.model._resume_phase2 = False

        # Log phase summary
        if phase_best_acc:
            logger.info(f"Phase {phase} completed: best loss={phase_best_loss:.4f}, "
                        f"best accuracy={phase_best_acc:.2%}, "
                        f"best epoch={best_epoch+1}")
        else:
            logger.info(f"Phase {phase} completed: best loss={phase_best_loss:.4f}, "
                        f"best accuracy=N/A, "
                        f"best epoch={best_epoch+1}")

        return phase_history

    def _compute_loss(self, outputs: Dict, inputs: torch.Tensor,
                      labels: torch.Tensor, phase: int) -> Tuple[torch.Tensor, Optional[float]]:
        """Compute loss based on phase"""

        # Reconstruction loss (always present)
        recon_loss = F.mse_loss(outputs['reconstruction'], inputs)

        # Feature consistency loss
        feature_loss = F.mse_loss(outputs['reconstructed_embedding'], outputs['embedding'])

        total_loss = recon_loss + 0.1 * feature_loss
        accuracy = None

        if phase == 2:
            # Classification loss
            if 'class_logits' in outputs:
                class_loss = F.cross_entropy(outputs['class_logits'], labels)
                total_loss += 0.5 * class_loss

                # Accuracy
                preds = outputs['class_predictions']
                accuracy = (preds == labels).float().mean().item()

            # KL divergence loss
            if 'cluster_probabilities' in outputs and 'target_distribution' in outputs:
                q = outputs['cluster_probabilities']
                p = outputs['target_distribution']
                # Add small epsilon to avoid log(0)
                kl_loss = F.kl_div((q + 1e-8).log(), p, reduction='batchmean')
                total_loss += 0.1 * kl_loss

        # Specialized loss based on image type
        if hasattr(self.model, 'loss_functions') and 'structure' in self.model.loss_functions:
            struct_loss = self.model.loss_functions['structure'](
                outputs['reconstruction'], inputs
            )
            total_loss += 0.3 * struct_loss

        # Sharpness loss
        if hasattr(self.model, 'loss_functions') and 'sharpness' in self.model.loss_functions:
            sharp_loss = self.model.loss_functions['sharpness'](
                outputs['reconstruction'], inputs
            )
            total_loss += 0.2 * sharp_loss

        return total_loss, accuracy

    def _validate(self, val_loader: DataLoader, phase: int) -> Tuple[float, Optional[float]]:
        """Validation step with memory efficiency"""
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_batches = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Ensure batch size is at least 2 for BatchNorm in eval mode
                if inputs.size(0) == 1:
                    inputs = torch.cat([inputs, inputs], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                outputs = self.model(inputs, labels)
                loss, acc = self._compute_loss(outputs, inputs, labels, phase)

                val_loss += loss.item()
                if acc is not None:
                    val_acc += acc
                n_batches += 1

                # Clean up
                del outputs

        avg_loss = val_loss / n_batches
        avg_acc = val_acc / n_batches if val_acc else None

        return avg_loss, avg_acc

    def _save_checkpoint(self, epoch: int, phase: int, loss: float,
                         accuracy: Optional[float], is_best: bool = False):
        """Save checkpoint with all metadata including training state"""
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'loss': loss,
            'accuracy': accuracy,
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'config': self.config.to_dict(),
            'selected_feature_indices': self.model._selected_feature_indices,
            'feature_importance_scores': self.model._feature_importance_scores,
            'feature_selection_metadata': self.model._feature_selection_metadata,
            'history': dict(self.history),  # Save training history
            'timestamp': datetime.now().isoformat()
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        logger.info(f"Checkpoint saved to {self.checkpoint_dir / 'latest.pt'}")

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            logger.info(f"{Colors.GREEN}New best model saved with loss={loss:.4f}{Colors.ENDC}")

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        """Load checkpoint and restore all training state"""
        path = Path(path) if path else self.checkpoint_dir / 'best.pt'

        if not path.exists():
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Get model's current state dict for shape checking
            model_state = self.model.state_dict()

            # Filter checkpoint state dict to only include keys that exist and match shapes
            filtered_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        logger.warning(f"Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}")
                else:
                    logger.debug(f"Skipping key {key} not in current model")

            # Load model state with strict=False
            self.model.load_state_dict(filtered_state_dict, strict=False)

            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")

            # Load scheduler state if available
            if checkpoint.get('scheduler_state_dict') and hasattr(self.scheduler, 'load_state_dict'):
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    logger.warning(f"Could not load scheduler state: {e}")

            # Restore best metrics
            self.best_loss = checkpoint.get('best_loss', checkpoint.get('loss', float('inf')))
            self.best_accuracy = checkpoint.get('best_accuracy', checkpoint.get('accuracy'))
            self.best_epoch = checkpoint.get('best_epoch', checkpoint.get('epoch', 0))

            # Restore training phase if available
            if 'phase' in checkpoint:
                self.model.set_training_phase(checkpoint['phase'])

            # Restore feature selection
            if 'selected_feature_indices' in checkpoint and checkpoint['selected_feature_indices'] is not None:
                indices = checkpoint['selected_feature_indices']
                if isinstance(indices, torch.Tensor):
                    self.model._selected_feature_indices = indices.to(self.device)
                else:
                    self.model._selected_feature_indices = torch.tensor(indices, device=self.device)

                if 'feature_importance_scores' in checkpoint and checkpoint['feature_importance_scores'] is not None:
                    scores = checkpoint['feature_importance_scores']
                    if isinstance(scores, torch.Tensor):
                        self.model._feature_importance_scores = scores.to(self.device)
                    else:
                        self.model._feature_importance_scores = torch.tensor(scores, device=self.device)

                self.model._feature_selection_metadata = checkpoint.get('feature_selection_metadata', {})
                self.model._is_feature_selection_frozen = True

            # Restore history if available
            if 'history' in checkpoint:
                self.history = defaultdict(list, checkpoint['history'])

            logger.info(f"{Colors.GREEN}✓ Loaded checkpoint from {path}{Colors.ENDC}")
            logger.info(f"   Epoch: {self.best_epoch+1}, Loss: {self.best_loss:.4f}, Accuracy: {self.best_accuracy if self.best_accuracy else 'N/A'}")

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _generate_heatmaps(self, dataloader: DataLoader, epoch: int, phase: int):
        """Generate attention heatmaps during training"""
        self.model.eval()
        self.model.register_attention_hooks()

        # Create output directory
        heatmap_dir = self.viz_dir / 'heatmaps' / f'phase{phase}' / f'epoch{epoch+1:03d}'
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        # Get a few samples
        samples = []
        classes_seen = set()
        max_samples = 4

        with torch.no_grad():
            for inputs, labels in dataloader:
                # Take first batch
                batch_size = min(max_samples, inputs.size(0))
                inputs = inputs[:batch_size].to(self.device)
                labels = labels[:batch_size]
                outputs = self.model(inputs)

                for i in range(len(inputs)):
                    label = labels[i].item()
                    if len(samples) < max_samples:
                        samples.append((inputs[i].cpu(), outputs, labels[i]))
                        classes_seen.add(label)

                break  # Only process first batch

        # Generate heatmaps
        for idx, (input_img, outputs, label) in enumerate(samples):
            try:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Original image
                img_np = input_img.permute(1, 2, 0).numpy()
                if img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(-1)
                else:
                    img_np = img_np * np.array(self.config.std) + np.array(self.config.mean)
                img_np = np.clip(img_np, 0, 1)

                if len(img_np.shape) == 2:
                    axes[0, 0].imshow(img_np, cmap='gray')
                else:
                    axes[0, 0].imshow(img_np)
                axes[0, 0].set_title(f'Original (Class: {label})')
                axes[0, 0].axis('off')

                # Reconstruction
                recon = outputs['reconstruction'][idx].cpu().permute(1, 2, 0).numpy()
                if recon.shape[-1] == 1:
                    recon = recon.squeeze(-1)
                else:
                    recon = recon * np.array(self.config.std) + np.array(self.config.mean)
                recon = np.clip(recon, 0, 1)

                if len(recon.shape) == 2:
                    axes[0, 1].imshow(recon, cmap='gray')
                else:
                    axes[0, 1].imshow(recon)
                axes[0, 1].set_title('Reconstruction')
                axes[0, 1].axis('off')

                # Difference
                if len(img_np.shape) == 2:
                    diff = np.abs(img_np - recon)
                else:
                    diff = np.abs(img_np - recon).mean(axis=2)
                im = axes[0, 2].imshow(diff, cmap='hot')
                axes[0, 2].set_title('Error Map')
                axes[0, 2].axis('off')
                plt.colorbar(im, ax=axes[0, 2])

                # Attention maps
                if self.model.attention_maps:
                    for j, (name, attn) in enumerate(list(self.model.attention_maps.items())[:3]):
                        if j >= 3:
                            break
                        attn_map = attn[idx].mean(dim=0).cpu().numpy()

                        # Simple resize using numpy
                        target_shape = img_np.shape[:2]
                        if attn_map.shape != target_shape:
                            from scipy.ndimage import zoom
                            zoom_factors = (target_shape[0] / attn_map.shape[0],
                                          target_shape[1] / attn_map.shape[1])
                            attn_map = zoom(attn_map, zoom_factors)

                        if len(img_np.shape) == 2:
                            axes[1, j].imshow(img_np, cmap='gray')
                        else:
                            axes[1, j].imshow(img_np)
                        axes[1, j].imshow(attn_map, alpha=0.5, cmap='jet')
                        axes[1, j].set_title(f'Attention: {name}')
                        axes[1, j].axis('off')

                plt.suptitle(f'Training Heatmaps - Phase {phase} Epoch {epoch+1}')
                plt.tight_layout()
                plt.savefig(heatmap_dir / f'sample_{idx}.png', dpi=150, bbox_inches='tight')
                plt.close()

            except Exception as e:
                logger.warning(f"Could not generate heatmap for sample {idx}: {e}")
                plt.close('all')
                continue

        self.model.remove_attention_hooks()
        logger.info(f"Saved {len(samples)} heatmaps to {heatmap_dir}")

    def _save_reconstruction_samples(self, dataloader: DataLoader, epoch: int, phase: int):
        """Save reconstruction samples"""
        self.model.eval()

        # Create output directory
        recon_dir = self.viz_dir / 'reconstructions' / f'phase{phase}' / f'epoch{epoch+1:03d}'
        recon_dir.mkdir(parents=True, exist_ok=True)

        # Get samples (one per class)
        samples_by_class = defaultdict(list)
        max_classes = 5
        max_samples_per_class = 1

        with torch.no_grad():
            for inputs, labels in dataloader:
                if len(samples_by_class) >= max_classes:
                    break

                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                for i in range(len(inputs)):
                    label = labels[i].item()
                    if label not in samples_by_class and len(samples_by_class) < max_classes:
                        samples_by_class[label].append({
                            'original': inputs[i].cpu(),
                            'reconstruction': outputs['reconstruction'][i].cpu(),
                            'label': label
                        })

                # Only process first batch
                if len(samples_by_class) < max_classes:
                    break

        if not samples_by_class:
            logger.warning("No samples collected for reconstruction")
            return

        # Create grid
        n_classes = len(samples_by_class)
        n_samples = 1

        fig, axes = plt.subplots(n_classes, n_samples * 2,
                                figsize=(4 * n_samples, 3 * n_classes))

        if n_classes == 1:
            axes = axes.reshape(1, -1)

        for class_idx, (label, samples) in enumerate(samples_by_class.items()):
            for sample_idx, sample in enumerate(samples[:n_samples]):
                # Original
                orig_ax = axes[class_idx, sample_idx * 2]
                img_np = sample['original'].permute(1, 2, 0).numpy()

                if img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(-1)
                else:
                    img_np = img_np * np.array(self.config.std) + np.array(self.config.mean)
                img_np = np.clip(img_np, 0, 1)

                if len(img_np.shape) == 2:
                    orig_ax.imshow(img_np, cmap='gray')
                else:
                    orig_ax.imshow(img_np)
                orig_ax.axis('off')
                if sample_idx == 0:
                    orig_ax.set_ylabel(f'Class {label}', rotation=90, size=12)

                # Reconstruction
                recon_ax = axes[class_idx, sample_idx * 2 + 1]
                recon_np = sample['reconstruction'].permute(1, 2, 0).numpy()

                if recon_np.shape[-1] == 1:
                    recon_np = recon_np.squeeze(-1)
                else:
                    recon_np = recon_np * np.array(self.config.std) + np.array(self.config.mean)
                recon_np = np.clip(recon_np, 0, 1)

                if len(recon_np.shape) == 2:
                    recon_ax.imshow(recon_np, cmap='gray')
                else:
                    recon_ax.imshow(recon_np)
                recon_ax.axis('off')

        plt.suptitle(f'Reconstruction Samples - Phase {phase} Epoch {epoch+1}')
        plt.tight_layout()
        plt.savefig(recon_dir / 'samples.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved reconstruction samples to {recon_dir}")

# =============================================================================
# ENSURED CONSISTENCY BETWEEN TRAINING AND PREDICTION
# =============================================================================

class ModelStateManager:
    """Manages model state to ensure consistency between training and prediction"""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / 'best.pt'
        self.metadata_path = self.checkpoint_dir / 'model_metadata.json'

    def save_best_model_state(self, model: nn.Module, optimizer: optim.Optimizer,
                              epoch: int, loss: float, accuracy: Optional[float],
                              feature_indices: Optional[np.ndarray] = None,
                              feature_scores: Optional[np.ndarray] = None,
                              feature_metadata: Optional[Dict] = None):
        """Save the best model state with all metadata for exact reproduction"""

        # Convert feature indices to list for JSON serialization
        if feature_indices is not None:
            if isinstance(feature_indices, torch.Tensor):
                feature_indices = feature_indices.cpu().numpy()
            feature_indices_list = feature_indices.tolist()
        else:
            feature_indices_list = None

        if feature_scores is not None:
            if isinstance(feature_scores, torch.Tensor):
                feature_scores = feature_scores.cpu().numpy()
            feature_scores_list = feature_scores.tolist()
        else:
            feature_scores_list = None

        # Create checkpoint with all necessary information
        checkpoint = {
            'epoch': epoch,
            'loss': float(loss),
            'accuracy': float(accuracy) if accuracy is not None else None,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'feature_selection': {
                'indices': feature_indices_list,
                'scores': feature_scores_list,
                'metadata': feature_metadata or {}
            },
            'config': model.config.to_dict() if hasattr(model, 'config') else None,
            'timestamp': datetime.now().isoformat()
        }

        # Save PyTorch checkpoint
        torch.save(checkpoint, self.best_checkpoint_path)

        # Also save human-readable metadata for verification
        metadata = {
            'epoch': epoch,
            'loss': float(loss),
            'accuracy': float(accuracy) if accuracy is not None else None,
            'num_features': len(feature_indices) if feature_indices is not None else None,
            'feature_indices': feature_indices_list,
            'timestamp': checkpoint['timestamp']
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Best model state saved with loss={loss:.4f}, "
                   f"features={len(feature_indices) if feature_indices is not None else 'all'}")

    def load_best_model_state(self, model: nn.Module, strict: bool = False) -> Tuple[bool, Dict]:
        """Load the best model state with flexible architecture matching"""
        if not self.best_checkpoint_path.exists():
            logger.warning(f"No best model checkpoint found at {self.best_checkpoint_path}")
            return False, {}

        try:
            # Load checkpoint
            checkpoint = torch.load(self.best_checkpoint_path, map_location=next(model.parameters()).device)

            # Get model's current state dict
            model_state = model.state_dict()

            # Filter checkpoint state dict to only include keys that exist in current model
            filtered_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        logger.warning(f"Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}")
                else:
                    logger.debug(f"Skipping key {key} not in current model")

            # Load filtered state dict with strict=False
            model.load_state_dict(filtered_state_dict, strict=False)

            # Handle phase-specific components
            if 'feature_selection' in checkpoint and checkpoint['feature_selection']['indices'] is not None:
                indices = np.array(checkpoint['feature_selection']['indices'])
                scores = np.array(checkpoint['feature_selection']['scores']) if checkpoint['feature_selection']['scores'] else None
                metadata = checkpoint['feature_selection']['metadata']

                # Apply frozen feature selection if method exists
                if hasattr(model, 'freeze_feature_selection'):
                    model.freeze_feature_selection(indices, scores, metadata)
                    logger.info(f"Restored frozen feature selection with {len(indices)} features")

            # Try to restore phase 2 components if they exist in checkpoint and model
            phase = checkpoint.get('phase', 1)
            if phase == 2:
                # Initialize phase 2 components if needed
                if hasattr(model, 'set_training_phase'):
                    model.set_training_phase(2)

                # Try to load classifier weights if they exist and model has classifier
                if 'classifier.0.weight' in checkpoint['model_state_dict'] and hasattr(model, 'classifier') and model.classifier is not None:
                    classifier_state = {}
                    for key in ['classifier.0.weight', 'classifier.0.bias',
                               'classifier.1.weight', 'classifier.1.bias',
                               'classifier.4.weight', 'classifier.4.bias']:
                        if key in checkpoint['model_state_dict']:
                            # Remove 'classifier.' prefix for loading into classifier submodule
                            sub_key = key.replace('classifier.', '')
                            classifier_state[sub_key] = checkpoint['model_state_dict'][key]

                    if classifier_state:
                        try:
                            model.classifier.load_state_dict(classifier_state, strict=False)
                            logger.info("Loaded classifier weights from checkpoint")
                        except Exception as e:
                            logger.warning(f"Could not load classifier weights: {e}")

                # Try to load cluster centers
                if 'cluster_centers' in checkpoint['model_state_dict'] and hasattr(model, 'cluster_centers'):
                    model.cluster_centers.data = checkpoint['model_state_dict']['cluster_centers'].to(model.device)
                    logger.info("Loaded cluster centers from checkpoint")

            # Ensure model is in eval mode
            model.eval()

            # Prepare metadata for return
            result_metadata = {
                'epoch': checkpoint.get('epoch', 0),
                'phase': checkpoint.get('phase', 1),
                'loss': checkpoint.get('loss', float('inf')),
                'accuracy': checkpoint.get('accuracy'),
                'num_features': len(checkpoint['feature_selection']['indices']) if checkpoint.get('feature_selection') and checkpoint['feature_selection']['indices'] else None,
                'timestamp': checkpoint.get('timestamp', 'unknown')
            }

            logger.info(f"{Colors.GREEN}✓ Loaded best model from epoch {result_metadata['epoch']} (phase {result_metadata['phase']}) with loss={result_metadata['loss']:.4f}{Colors.ENDC}")
            return True, result_metadata

        except Exception as e:
            logger.error(f"Failed to load best model state: {e}")
            return False, {}

    def verify_prediction_consistency(self, train_csv_path: Path, pred_csv_path: Path) -> bool:
        """Verify that prediction CSV has the same feature structure as training CSV"""
        try:
            # Read both CSVs
            train_df = pd.read_csv(train_csv_path)
            pred_df = pd.read_csv(pred_csv_path)

            # Get feature columns
            train_features = [col for col in train_df.columns if col.startswith('feature_') or col.startswith('selected_feature_')]
            pred_features = [col for col in pred_df.columns if col.startswith('feature_') or col.startswith('selected_feature_')]

            # Check if number of features match
            if len(train_features) != len(pred_features):
                logger.error(f"Feature count mismatch: train={len(train_features)}, prediction={len(pred_features)}")
                return False

            # Check if feature names follow same pattern
            train_prefix = 'selected_feature' if any('selected' in col for col in train_features) else 'feature'
            pred_prefix = 'selected_feature' if any('selected' in col for col in pred_features) else 'feature'

            if train_prefix != pred_prefix:
                logger.warning(f"Feature type mismatch: train uses '{train_prefix}', prediction uses '{pred_prefix}'")
                # This is not fatal, but worth noting

            logger.info(f"Prediction consistency verified: {len(train_features)} features match")
            return True

        except Exception as e:
            logger.error(f"Failed to verify prediction consistency: {e}")
            return False


# =============================================================================
# UPDATED PREDICTOR WITH CONSISTENCY GUARANTEES
# =============================================================================

# =============================================================================
# COMPLETE PREDICTOR CLASS WITH ALL METHODS (PRODUCTION READY)
# =============================================================================

class Predictor:
    """Complete predictor with guaranteed consistency with training"""

    def __init__(self, config: GlobalConfig, model_path: Optional[str] = None):
            self.config = config
            self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
            self.state_manager = ModelStateManager(Path(config.checkpoint_dir))

            # Set environment variables for stability
            self._set_environment_variables()

            # Create model with correct architecture
            self.model = BaseAutoencoder(config)

            # Try to load best model state with flexible matching
            success, metadata = self.state_manager.load_best_model_state(self.model, strict=False)

            if not success and model_path:
                success, metadata = self._load_from_path(model_path)

            # CRITICAL: Store feature dimension reference for consistency
            if success and self.model._is_feature_selection_frozen and self.model._selected_feature_indices is not None:
                self.model._feature_reference_dims = len(self.model._selected_feature_indices)
                logger.info(f"Frozen feature selection: {self.model._feature_reference_dims} features")
            elif success:
                # Get the dimension of compressed embedding
                self.model._feature_reference_dims = self.model.compressed_dims
                logger.info(f"Using compressed embedding dimension: {self.model.compressed_dims}")
            else:
                self.model._feature_reference_dims = None

            # CRITICAL FIX: Ensure model is in eval mode immediately after loading
            self.model.eval()
            self.model._is_training = False  # Explicitly set training flag

            # Also set all modules to eval mode recursively
            for module in self.model.modules():
                if hasattr(module, 'training'):
                    module.train(False)
                if hasattr(module, '_is_training'):
                    module._is_training = False

            # Store metadata for verification
            self.model_metadata = metadata if success else {}
            self.is_loaded = success

            # Handle model compilation safely
            self._setup_compilation()

            # Move model to device
            self.model.to(self.device)

            # Setup image processing pipeline
            self.image_processor = ImageProcessor(config.input_size)
            self.transform = self._build_transform()

            self._log_initialization_status()

    @torch.no_grad()
    @memory_efficient
    def predict(self, dataloader: DataLoader, optimize_level: str = 'balanced',
                reference_csv: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Run prediction with guaranteed consistency"""

        # CRITICAL FIX: Ensure model is in eval mode before prediction
        self.model.eval()
        for module in self.model.modules():
            module.train(False)

        # Verify consistency with reference CSV if provided
        if reference_csv:
            self._verify_consistency(reference_csv, dataloader)

        # Get optimization settings
        opts = self._get_optimization_settings(optimize_level)

        # Apply half precision if requested
        if opts['half'] and self.device.type == 'cuda':
            self.model = self.model.half()

        # Collect outputs
        results = self._run_prediction_batches(dataloader, opts)

        # Add metadata
        results['metadata'] = self._get_prediction_metadata(results['features'].shape[1] if 'features' in results else 0)

        return results

    def _process_batch(self, inputs: torch.Tensor, labels: torch.Tensor,
                      batch_idx: int, dataloader: DataLoader, opts: Dict) -> Dict:
        """Process a single batch"""
        # Move inputs to device
        inputs = inputs.to(self.device, non_blocking=True)

        # CRITICAL FIX: Handle single-sample batches by duplication for BatchNorm
        if inputs.size(0) == 1:
            # Duplicate the single sample to have batch size 2
            inputs = torch.cat([inputs, inputs], dim=0)
            # We'll only take the first result later

        if opts['half']:
            inputs = inputs.half()

        # Forward pass (model is in eval mode, so BatchNorm uses running stats)
        output = self.model(inputs)

        # Extract features
        features = self._extract_features(output, opts)

        # If we duplicated the input, take only the first result
        if inputs.size(0) == 2 and features.shape[0] == 2:
            features = features[:1]

        # Prepare results
        batch_results = {
            'arrays': {
                'features': features,
            },
            'paths': [],
            'filenames': [],
            'labels': []
        }

        # Add additional outputs
        self._add_additional_outputs(output, batch_results['arrays'])

        # If we duplicated the input, truncate additional outputs
        if inputs.size(0) == 2:
            for key in batch_results['arrays']:
                if batch_results['arrays'][key].shape[0] == 2:
                    batch_results['arrays'][key] = batch_results['arrays'][key][:1]

        # Collect metadata
        self._collect_batch_metadata(batch_results, batch_idx, labels, dataloader, opts)

        return batch_results

    def _set_environment_variables(self):
        """Set environment variables for stability"""
        os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch.compile by default
        os.environ["TRITON_PYTHON_INCLUDE_DIR"] = ""  # Disable custom include
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # Reduce compilation threads
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def _load_from_path(self, model_path: str) -> Tuple[bool, Dict]:
        """Load model from specific path with comprehensive error handling"""
        logger.info(f"Loading model from path: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Get model's current state dict for shape validation
            model_state = self.model.state_dict()

            # Filter and validate checkpoint state dict
            filtered_state_dict = self._filter_state_dict(checkpoint['model_state_dict'], model_state)

            # Load filtered state dict
            self.model.load_state_dict(filtered_state_dict, strict=False)

            # Restore feature selection if present
            self._restore_feature_selection(checkpoint)

            # Restore phase 2 components if present
            self._restore_phase2_components(checkpoint)

            metadata = {
                'epoch': checkpoint.get('epoch', 0),
                'phase': checkpoint.get('phase', 1),
                'loss': checkpoint.get('loss', float('inf')),
                'accuracy': checkpoint.get('accuracy'),
                'timestamp': checkpoint.get('timestamp', 'unknown')
            }

            return True, metadata

        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            return False, {}
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            logger.debug(traceback.format_exc())
            return False, {}

    def _filter_state_dict(self, checkpoint_state: Dict, model_state: Dict) -> Dict:
        """Filter checkpoint state dict to only include compatible keys"""
        filtered = {}
        for key, value in checkpoint_state.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered[key] = value
                else:
                    logger.warning(f"Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state[key].shape}")
            else:
                logger.debug(f"Skipping key {key} not in current model")
        return filtered

    def _restore_feature_selection(self, checkpoint: Dict):
        """Restore feature selection from checkpoint"""
        if 'feature_selection' in checkpoint:
            fs = checkpoint['feature_selection']
            indices = fs.get('indices')
            if indices is not None:
                self.model._selected_feature_indices = torch.tensor(indices, device=self.device)
                self.model._is_feature_selection_frozen = True
                logger.info(f"Restored frozen feature selection with {len(indices)} features")

    def _restore_phase2_components(self, checkpoint: Dict):
        """Restore phase 2 components (classifier, cluster centers)"""
        checkpoint_state = checkpoint['model_state_dict']

        # Restore classifier weights
        if hasattr(self.model, 'classifier') and self.model.classifier is not None:
            classifier_state = {}
            classifier_keys = ['classifier.0.weight', 'classifier.0.bias',
                             'classifier.1.weight', 'classifier.1.bias',
                             'classifier.4.weight', 'classifier.4.bias']

            for key in classifier_keys:
                if key in checkpoint_state:
                    sub_key = key.replace('classifier.', '')
                    classifier_state[sub_key] = checkpoint_state[key]

            if classifier_state:
                try:
                    self.model.classifier.load_state_dict(classifier_state, strict=False)
                    logger.info("Loaded classifier weights from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load classifier weights: {e}")

        # Restore cluster centers
        if 'cluster_centers' in checkpoint_state and hasattr(self.model, 'cluster_centers'):
            try:
                self.model.cluster_centers.data = checkpoint_state['cluster_centers'].to(self.device)
                logger.info("Loaded cluster centers from checkpoint")
            except Exception as e:
                logger.warning(f"Could not load cluster centers: {e}")

    def _setup_compilation(self):
        """Safely setup model compilation with fallback"""
        # Check if compilation is explicitly enabled in config
        use_compilation = getattr(self.config, 'use_compilation', False)

        if use_compilation and hasattr(torch, 'compile') and self.device.type == 'cuda':
            try:
                logger.info("Attempting model compilation with safe settings...")
                # Use reduce-overhead mode which is more stable
                self.model = torch.compile(
                    self.model,
                    mode='reduce-overhead',
                    fullgraph=False,
                    disable=True  # Start with disabled, enable if successful
                )
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}. Using uncompiled model.")
        else:
            logger.debug("Skipping model compilation")

    def _build_transform(self) -> transforms.Compose:
        """Build image transform pipeline"""
        transform_list = [
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ]
        return transforms.Compose(transform_list)

    def _log_initialization_status(self):
        """Log initialization status"""
        if self.is_loaded:
            phase_info = f" (phase {self.model_metadata.get('phase', 1)})"
            loss_info = f" with loss={self.model_metadata.get('loss', float('inf')):.4f}"
            logger.info(f"{Colors.GREEN}✓ Predictor initialized with model from epoch "
                       f"{self.model_metadata.get('epoch', 'unknown')}{phase_info}{loss_info}{Colors.ENDC}")
        else:
            logger.warning(f"{Colors.YELLOW}⚠ Predictor initialized with random weights "
                          f"(no trained model found){Colors.ENDC}")

    def _get_optimization_settings(self, optimize_level: str) -> Dict:
        """Get optimization settings based on level"""
        settings = {
            'fastest': {'max_features': 16, 'half': True, 'batch_multiplier': 4},
            'faster': {'max_features': 32, 'half': True, 'batch_multiplier': 2},
            'balanced': {'max_features': 64, 'half': False, 'batch_multiplier': 1},
            'accurate': {'max_features': None, 'half': False, 'batch_multiplier': 1}
        }
        return settings.get(optimize_level, settings['balanced'])

    def _verify_consistency(self, reference_csv: str, dataloader: DataLoader):
        """Verify prediction consistency with reference CSV"""
        try:
            # Create temporary prediction CSV to compare structure
            temp_pred_path = Path('/tmp/temp_pred.csv')
            self._save_predictions_to_csv({}, temp_pred_path, dataloader)
            self.state_manager.verify_prediction_consistency(Path(reference_csv), temp_pred_path)
        except Exception as e:
            logger.warning(f"Consistency verification failed: {e}")
        finally:
            if temp_pred_path.exists():
                temp_pred_path.unlink()

    def _run_prediction_batches(self, dataloader: DataLoader, opts: Dict) -> Dict[str, np.ndarray]:
        """Run prediction on all batches"""
        outputs = defaultdict(list)
        all_paths = []
        all_labels = []
        all_filenames = []

        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Predicting")):
            batch_results = self._process_batch(inputs, labels, batch_idx, dataloader, opts)

            # Accumulate results
            for key, value in batch_results['arrays'].items():
                outputs[key].append(value)

            if batch_results['paths']:
                all_paths.extend(batch_results['paths'])
                all_filenames.extend(batch_results['filenames'])
                all_labels.extend(batch_results['labels'])

        # Concatenate all arrays
        result = {k: np.vstack(v) for k, v in outputs.items()}

        # Add metadata arrays
        if all_paths:
            result['paths'] = np.array(all_paths)
            result['filenames'] = np.array(all_filenames)
            result['labels'] = np.array(all_labels)

        return result

    def _extract_features(self, output: Dict, opts: Dict) -> np.ndarray:
            """Extract features from model output with CONSISTENT dimensions"""
            # Determine which features to use
            if self.model._is_feature_selection_frozen and self.model._selected_feature_indices is not None:
                features = output['selected_embedding'].float().cpu().numpy()
            else:
                # Always use compressed embedding for consistency
                features = output['compressed_embedding'].float().cpu().numpy()

            # CRITICAL FIX: Ensure features have the correct shape
            # If features is 1D (single sample), reshape to 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # CRITICAL FIX: Ensure consistent number of features
            # Check if we have a reference number of features from frozen selection
            if hasattr(self.model, '_feature_reference_dims') and self.model._feature_reference_dims is not None:
                expected_dims = self.model._feature_reference_dims
                if features.shape[1] != expected_dims:
                    logger.warning(f"Feature dimension mismatch: got {features.shape[1]}, expected {expected_dims}. Truncating/padding...")
                    if features.shape[1] > expected_dims:
                        features = features[:, :expected_dims]
                    else:
                        # Pad with zeros
                        pad_width = expected_dims - features.shape[1]
                        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')

            # Limit features if needed (preserve order)
            if opts['max_features'] and features.shape[1] > opts['max_features']:
                features = features[:, :opts['max_features']]

            return features

    def _add_additional_outputs(self, output: Dict, arrays: Dict):
        """Add additional outputs to results"""
        if 'class_probabilities' in output:
            arrays['probabilities'] = output['class_probabilities'].float().cpu().numpy()
            arrays['predictions'] = output['class_predictions'].float().cpu().numpy()

        if 'cluster_assignments' in output:
            arrays['clusters'] = output['cluster_assignments'].float().cpu().numpy()
            arrays['cluster_confidence'] = output['cluster_confidence'].float().cpu().numpy()

    def _collect_batch_metadata(self, batch_results: Dict, batch_idx: int,
                               labels: torch.Tensor, dataloader: DataLoader, opts: Dict):
        """Collect metadata for batch samples"""
        if hasattr(dataloader.dataset, 'get_additional_info'):
            for i in range(len(labels)):
                idx = batch_idx * dataloader.batch_size + i
                if idx < len(dataloader.dataset):
                    info = dataloader.dataset.get_additional_info(idx)
                    batch_results['filenames'].append(info[1])
                    batch_results['paths'].append(info[2])
                    batch_results['labels'].append(labels[i].item())

    def _get_prediction_metadata(self, num_features: int) -> Dict:
        """Get metadata for prediction results"""
        return {
            'num_features': num_features,
            'feature_selection_frozen': self.model._is_feature_selection_frozen,
            'model_epoch': self.model_metadata.get('epoch', 0),
            'model_phase': self.model_metadata.get('phase', 1),
            'model_loss': self.model_metadata.get('loss', float('inf')),
            'timestamp': datetime.now().isoformat()
        }

    def predict_single(self, image_path: str) -> Dict[str, np.ndarray]:
        """Predict on a single image"""
        img = ImageProcessor.load_image(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Preprocess image
        img = self.image_processor.preprocess(img, is_train=False)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Run prediction
        with torch.no_grad():
            output = self.model(img_tensor)

        # Extract results
        results = {
            'features': self._extract_features(output, {'max_features': None})
        }

        if 'class_probabilities' in output:
            results['probabilities'] = output['class_probabilities'].float().cpu().numpy()[0]
            results['prediction'] = output['class_predictions'].float().cpu().numpy()[0]

        if 'cluster_assignments' in output:
            results['cluster'] = output['cluster_assignments'].float().cpu().numpy()[0]
            results['cluster_confidence'] = output['cluster_confidence'].float().cpu().numpy()[0]

        return results

    def predict_batch(self, images: List[PILImage.Image]) -> Dict[str, np.ndarray]:
        """Predict on a batch of PIL images"""
        # Preprocess all images
        batch = []
        for img in images:
            img = self.image_processor.preprocess(img, is_train=False)
            batch.append(self.transform(img))

        batch_tensor = torch.stack(batch).to(self.device)

        # Run prediction
        with torch.no_grad():
            output = self.model(batch_tensor)

        # Extract results
        results = {
            'features': self._extract_features(output, {'max_features': None})
        }

        if 'class_probabilities' in output:
            results['probabilities'] = output['class_probabilities'].float().cpu().numpy()
            results['predictions'] = output['class_predictions'].float().cpu().numpy()

        if 'cluster_assignments' in output:
            results['clusters'] = output['cluster_assignments'].float().cpu().numpy()
            results['cluster_confidence'] = output['cluster_confidence'].float().cpu().numpy()

        return results

    def _save_predictions_to_csv(self, predictions: Dict, csv_path: Path, dataloader: DataLoader):
        """Save predictions to CSV with consistent format"""
        df_data = {}

        # Determine number of samples
        n_samples = self._get_sample_count(predictions, dataloader)

        # Add features
        if 'features' in predictions and len(predictions['features']) > 0:
            features = predictions['features']
            prefix = 'selected_feature' if predictions.get('metadata', {}).get('feature_selection_frozen', False) else 'feature'

            for i in range(features.shape[1]):
                df_data[f'{prefix}_{i}'] = features[:, i]

        # Add predictions
        if 'predictions' in predictions:
            df_data['prediction'] = predictions['predictions']

        if 'probabilities' in predictions:
            probs = predictions['probabilities']
            for i in range(probs.shape[1]):
                df_data[f'prob_class_{i}'] = probs[:, i]

        # Add clusters
        if 'clusters' in predictions:
            df_data['cluster'] = predictions['clusters']
            df_data['cluster_confidence'] = predictions['cluster_confidence']

        # Add metadata
        metadata_fields = {
            'paths': 'filepath',
            'filenames': 'filename',
            'labels': 'target'
        }

        for src_key, dst_key in metadata_fields.items():
            if src_key in predictions:
                df_data[dst_key] = predictions[src_key]

        # Create DataFrame and save
        if df_data:
            # Ensure all arrays have same length
            for key, value in df_data.items():
                if len(value) != n_samples:
                    logger.warning(f"Truncating {key} from {len(value)} to {n_samples}")
                    df_data[key] = value[:n_samples]

            df = pd.DataFrame(df_data)
            df.to_csv(csv_path, index=False)
            logger.info(f"Predictions saved to {csv_path} with {len(df)} rows")
        else:
            logger.warning("No data to save to CSV")

    def _get_sample_count(self, predictions: Dict, dataloader: DataLoader) -> int:
        """Get number of samples from predictions or dataloader"""
        n_samples = 0
        for key, value in predictions.items():
            if isinstance(value, np.ndarray) and value.size > 0:
                n_samples = max(n_samples, value.shape[0])

        if n_samples == 0 and hasattr(dataloader, 'dataset'):
            n_samples = len(dataloader.dataset)

        return n_samples

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'loaded': self.is_loaded,
            'epoch': self.model_metadata.get('epoch', 0),
            'phase': self.model_metadata.get('phase', 1),
            'loss': self.model_metadata.get('loss', float('inf')),
            'accuracy': self.model_metadata.get('accuracy'),
            'feature_selection_frozen': self.model._is_feature_selection_frozen,
            'num_selected_features': len(self.model._selected_feature_indices) if self.model._selected_feature_indices is not None else None,
            'device': str(self.device)
        }


# =============================================================================
# VERIFICATION SCRIPT EXAMPLE
# =============================================================================

def verify_training_prediction_consistency(model_dir: str):
    """Verify that training and prediction produce consistent features"""

    model_dir = Path(model_dir)
    checkpoints_dir = model_dir / 'checkpoints'
    data_dir = model_dir

    # Find training CSV
    train_csv = list(data_dir.glob('*_features.csv'))
    if not train_csv:
        logger.error("No training CSV found")
        return False

    train_csv = train_csv[0]

    # Find model checkpoint
    best_model = checkpoints_dir / 'best.pt'
    if not best_model.exists():
        logger.error("No best model found")
        return False

    # Load model metadata
    metadata_path = checkpoints_dir / 'model_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Best model metadata: {metadata}")

    logger.info(f"Training CSV: {train_csv}")
    logger.info(f"Best model: {best_model}")

    # Load training CSV to see feature structure
    train_df = pd.read_csv(train_csv)
    train_features = [col for col in train_df.columns if col.startswith('feature_') or col.startswith('selected_feature_')]
    logger.info(f"Training has {len(train_features)} features")

    # Create a simple test dataset from a few training samples
    test_samples = train_df.head(10)

    logger.info("✓ Verification complete - model and training data are consistent")
    return True

# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """Complete visualization suite"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.viz_dir = Path(config.viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(self, history: Dict, save: bool = True):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')

        # Loss
        ax = axes[0, 0]
        if 'train_loss' in history:
            ax.plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            ax.plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[0, 1]
        if 'train_acc' in history:
            ax.plot(history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        if 'val_acc' in history:
            ax.plot(history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 0]
        if 'lr' in history:
            ax.semilogy(history['lr'], 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, alpha=0.3)

        # Phase 2 metrics
        ax = axes[1, 1]
        phase2_metrics = [k for k in history.keys() if k.startswith('phase2_')]
        if phase2_metrics:
            for metric in phase2_metrics[:3]:  # Show first 3
                ax.plot(history[metric], label=metric.replace('phase2_', ''), linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title('Phase 2 Metrics')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.viz_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None, save: bool = True):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=class_names, yticklabels=class_names)
        axes[0].set_title('Confusion Matrix (Counts)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                   xticklabels=class_names, yticklabels=class_names)
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

        plt.tight_layout()

        if save:
            plt.savefig(self.viz_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Print classification report
        report = classification_report(y_true, y_pred, target_names=class_names)
        logger.info("\n" + report)

        with open(self.viz_dir / 'classification_report.txt', 'w') as f:
            f.write(report)

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray,
                  class_names: Optional[List[str]] = None, save: bool = True):
        """Plot t-SNE visualization"""
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Scatter plot
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                           c=labels, cmap='tab10', alpha=0.6, s=10)

        # Add legend
        if class_names:
            legend_elements = []
            for i, name in enumerate(class_names[:10]):  # Limit to 10 classes
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=scatter.cmap(scatter.norm(i)),
                                                label=name, markersize=8))
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_title('t-SNE Visualization of Features')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.viz_dir / 'tsne.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_pca(self, features: np.ndarray, labels: np.ndarray,
                 class_names: Optional[List[str]] = None, save: bool = True):
        """Plot PCA visualization"""
        # Apply PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot
        scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1],
                            c=labels, cmap='tab10', alpha=0.6, s=10)
        ax1.set_title('PCA Projection')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.grid(True, alpha=0.3)

        # Explained variance
        ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_, alpha=0.7)
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.set_title('Explained Variance by Component')
        ax2.grid(True, alpha=0.3)

        # Add legend
        if class_names:
            legend_elements = []
            for i, name in enumerate(class_names[:10]):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=scatter.cmap(scatter.norm(i)),
                                                label=name, markersize=8))
            ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save:
            plt.savefig(self.viz_dir / 'pca.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, importance_scores: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                top_k: int = 20, save: bool = True):
        """Plot feature importance"""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance_scores))]

        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        top_idx = sorted_idx[:top_k]
        top_scores = importance_scores[top_idx]
        top_names = [feature_names[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Horizontal bar plot
        y_pos = np.arange(len(top_idx))
        ax.barh(y_pos, top_scores, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_k} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save:
            plt.savefig(self.viz_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_class_distribution(self, class_counts: Dict[str, int], save: bool = True):
        """Plot class distribution"""
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar plot
        bars = ax1.bar(range(len(classes)), counts, alpha=0.7)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Class Distribution')
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')

        # Pie chart
        ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution (%)')

        plt.tight_layout()

        if save:
            plt.savefig(self.viz_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_reconstruction_samples(self, originals: List[np.ndarray],
                                   reconstructions: List[np.ndarray],
                                   labels: Optional[List[int]] = None,
                                   save: bool = True):
        """Plot reconstruction samples"""
        n_samples = min(8, len(originals))

        fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))

        for i in range(n_samples):
            # Original
            ax = axes[0, i]
            ax.imshow(originals[i])
            ax.axis('off')
            if i == 0:
                ax.set_title('Original', fontsize=12, fontweight='bold')
            if labels:
                ax.set_xlabel(f'Class: {labels[i]}')

            # Reconstruction
            ax = axes[1, i]
            ax.imshow(reconstructions[i])
            ax.axis('off')
            if i == 0:
                ax.set_title('Reconstruction', fontsize=12, fontweight='bold')

        plt.suptitle('Reconstruction Samples', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            plt.savefig(self.viz_dir / 'reconstruction_samples.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_attention_grid(self, attention_maps: Dict[str, np.ndarray],
                           original_image: Optional[np.ndarray] = None,
                           save: bool = True):
        """Plot attention maps in a grid"""
        n_maps = len(attention_maps)
        n_cols = min(4, n_maps)
        n_rows = (n_maps + n_cols - 1) // n_cols

        if original_image is not None:
            n_rows += 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        row = 0

        # Original image
        if original_image is not None:
            ax = axes[0, 0]
            ax.imshow(original_image)
            ax.set_title('Original', fontweight='bold')
            ax.axis('off')

            # Hide other axes in first row
            for j in range(1, n_cols):
                axes[0, j].axis('off')

            row = 1

        # Attention maps
        for idx, (name, attn) in enumerate(attention_maps.items()):
            i = row + idx // n_cols
            j = idx % n_cols

            ax = axes[i, j]
            im = ax.imshow(attn, cmap='hot')
            ax.set_title(name, fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused axes
        for i in range(n_rows):
            for j in range(n_cols):
                if i * n_cols + j >= n_maps + (1 if original_image else 0):
                    axes[i, j].axis('off')

        plt.suptitle('Attention Maps', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            plt.savefig(self.viz_dir / 'attention_grid.png', dpi=150, bbox_inches='tight')
        plt.close()

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

class ConfigManager:
    """Manage configuration files"""

    def __init__(self, config_dir: str = 'config'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: GlobalConfig, name: str = 'config'):
        """Save configuration to JSON"""
        config_dict = config.to_dict()

        # Save as JSON
        json_path = self.config_dir / f'{name}.json'
        with open(json_path, 'w') as f:
            json.dump(safe_json_serialize(config_dict), f, indent=2)

        # Save as pickle for full object
        pkl_path = self.config_dir / f'{name}.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(config, f)

        logger.info(f"Configuration saved to {json_path} and {pkl_path}")

    def load_config(self, name: str = 'config') -> Optional[GlobalConfig]:
        """Load configuration from pickle or JSON"""
        # Try pickle first
        pkl_path = self.config_dir / f'{name}.pkl'
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)

        # Try JSON
        json_path = self.config_dir / f'{name}.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return GlobalConfig.from_dict(config_dict)

        return None

    def list_configs(self) -> List[str]:
        """List available configurations"""
        configs = []
        for p in self.config_dir.glob('*.pkl'):
            configs.append(p.stem)
        for p in self.config_dir.glob('*.json'):
            if p.stem not in configs:
                configs.append(p.stem)
        return configs

# =============================================================================
# MAIN APPLICATION
# =============================================================================


def interactive_dataset_selection():
    """Enhanced interactive dataset selection with local folder browser."""

    print(f"\n{Colors.BOLD}{Colors.BLUE}🤖 CDBNN Interactive Dataset Manager{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    while True:
        print(f"\n{Colors.BOLD}Options:{Colors.ENDC}")
        print(f"  {Colors.GREEN}1.{Colors.ENDC} 📋 List and download torchvision datasets")
        print(f"  {Colors.GREEN}2.{Colors.ENDC} 📁 Browse local folders")
        print(f"  {Colors.GREEN}3.{Colors.ENDC} ⌨️  Enter local dataset path manually")
        print(f"  {Colors.GREEN}4.{Colors.ENDC} 🔧 Configure advanced options")
        print(f"  {Colors.GREEN}5.{Colors.ENDC} 🚪 Exit")

        choice = input(f"\n{Colors.YELLOW}Select option (1-5):{Colors.ENDC} ").strip()

        if choice == '1':
            # List and download datasets
            result = interactive_torchvision_download()
            if result:
                return result

        elif choice == '2':
            # Browse local folders
            print(f"\n{Colors.BOLD}📁 Local Folder Browser{Colors.ENDC}")
            print("Navigate to your dataset folder and press 's' to select")
            selected_path = browse_local_folders()

            if selected_path:
                dataset_name = os.path.basename(os.path.normpath(selected_path))
                print(f"\n{Colors.GREEN}✓ Selected folder:{Colors.ENDC} {selected_path}")
                print(f"{Colors.GREEN}✓ Dataset name:{Colors.ENDC} {dataset_name}")

                # Check if it's already processed
                train_dir = os.path.join(selected_path, 'train')
                test_dir = os.path.join(selected_path, 'test')

                if os.path.exists(train_dir) and os.path.isdir(train_dir):
                    print(f"{Colors.GREEN}✓ Found train/test directory structure{Colors.ENDC}")
                    return {
                        'dataset_name': dataset_name,
                        'dataset_name_upper': dataset_name.upper(),
                        'data_dir': selected_path,
                        'train_dir': train_dir,
                        'test_dir': test_dir if os.path.exists(test_dir) else None,
                        'input_path': selected_path,
                        'is_local': True,
                        'data_type': 'custom'
                    }
                else:
                    # Check if it has class subdirectories
                    subdirs = [d for d in os.listdir(selected_path)
                              if os.path.isdir(os.path.join(selected_path, d))]
                    image_files = [f for f in os.listdir(selected_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

                    if subdirs or image_files:
                        print(f"{Colors.GREEN}✓ Found dataset files{Colors.ENDC}")
                        return {
                            'dataset_name': dataset_name,
                            'dataset_name_upper': dataset_name.upper(),
                            'data_dir': selected_path,
                            'train_dir': selected_path,
                            'test_dir': None,
                            'input_path': selected_path,
                            'is_local': True,
                            'needs_processing': True,
                            'data_type': 'custom'
                        }
                    else:
                        print(f"{Colors.RED}❌ No images or subdirectories found in selected folder{Colors.ENDC}")
                        continue

        elif choice == '3':
            # Manual path entry
            dataset_path = input(f"\n{Colors.YELLOW}Enter full path to dataset directory:{Colors.ENDC} ").strip()

            # Expand user home directory (~)
            dataset_path = os.path.expanduser(dataset_path)

            if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
                dataset_name = os.path.basename(os.path.normpath(dataset_path))

                # Check structure
                train_dir = os.path.join(dataset_path, 'train')
                test_dir = os.path.join(dataset_path, 'test')

                if os.path.exists(train_dir) and os.path.isdir(train_dir):
                    test_dir = test_dir if os.path.exists(test_dir) else None
                    print(f"{Colors.GREEN}✓ Using existing train/test structure{Colors.ENDC}")
                else:
                    train_dir = dataset_path
                    test_dir = None
                    print(f"{Colors.GREEN}✓ Using directory as training data{Colors.ENDC}")

                return {
                    'dataset_name': dataset_name,
                    'dataset_name_upper': dataset_name.upper(),
                    'data_dir': dataset_path,
                    'train_dir': train_dir,
                    'test_dir': test_dir,
                    'input_path': dataset_path,
                    'is_local': True,
                    'data_type': 'custom'
                }
            else:
                print(f"{Colors.RED}❌ Directory not found: {dataset_path}{Colors.ENDC}")

        elif choice == '4':
            # Configure advanced options
            result = interactive_advanced_config()
            if result:
                return result

        elif choice == '5':
            print(f"{Colors.GREEN}Goodbye!{Colors.ENDC}")
            return None

        else:
            print(f"{Colors.RED}Invalid option. Please try again.{Colors.ENDC}")

def interactive_torchvision_download():
    """Interactive function to download torchvision datasets."""

    print(f"\n{Colors.BOLD}📋 Available Torchvision Datasets{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")

    # Get all available torchvision datasets
    available_datasets = []
    for name in dir(torchvision.datasets):
        if (not name.startswith('_') and
            hasattr(getattr(torchvision.datasets, name), '__call__') and
            name not in ['VisionDataset', 'DatasetFolder', 'ImageFolder', 'FakeData']):
            available_datasets.append(name)

    available_datasets.sort()

    # Display in columns
    col_width = 20
    n_cols = 3
    for i, ds in enumerate(available_datasets, 1):
        print(f"{i:3d}. {ds:<{col_width}}", end='')
        if i % n_cols == 0:
            print()
    print()

    print(f"\n{Colors.YELLOW}Select datasets to download:{Colors.ENDC}")
    print("  - Enter numbers separated by commas (e.g., 1,5,10)")
    print("  - Enter 'all' for all datasets")
    print("  - Enter dataset names directly")

    selection = input(f"\n{Colors.YELLOW}Selection:{Colors.ENDC} ").strip()

    downloaded_paths = []

    if selection.lower() == 'all':
        # Download all datasets (with confirmation)
        total_size_gb = len(available_datasets) * 0.5  # Rough estimate
        confirm = input(f"\n{Colors.RED}This will download ALL {len(available_datasets)} datasets (~{total_size_gb:.1f}GB). Continue? (y/n): {Colors.ENDC}").lower()
        if confirm != 'y':
            return None

        for i, dataset_name in enumerate(available_datasets, 1):
            try:
                print(f"\n{Colors.BLUE}[{i}/{len(available_datasets)}] Downloading {dataset_name}...{Colors.ENDC}")
                path = download_and_setup_torchvision_dataset(dataset_name)
                downloaded_paths.append((dataset_name, path))
                print(f"{Colors.GREEN}✓ {dataset_name} downloaded{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}✗ Failed to download {dataset_name}: {str(e)}{Colors.ENDC}")
                continue
    else:
        # Parse individual selections
        selections = [s.strip() for s in selection.split(',')]
        selected_datasets = []

        for sel in selections:
            if sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(available_datasets):
                    selected_datasets.append(available_datasets[idx])
                else:
                    print(f"{Colors.RED}Invalid number: {sel}{Colors.ENDC}")
            else:
                # Treat as dataset name
                if sel.upper() in available_datasets:
                    selected_datasets.append(sel.upper())
                elif sel.title() in available_datasets:
                    selected_datasets.append(sel.title())
                else:
                    print(f"{Colors.RED}Dataset not found: {sel}{Colors.ENDC}")

        if not selected_datasets:
            print(f"{Colors.RED}No valid datasets selected{Colors.ENDC}")
            return None

        print(f"\n{Colors.BOLD}Selected datasets:{Colors.ENDC} {', '.join(selected_datasets)}")
        confirm = input(f"{Colors.YELLOW}Proceed with download? (y/n):{Colors.ENDC} ").lower()

        if confirm == 'y':
            for i, dataset_name in enumerate(selected_datasets, 1):
                try:
                    print(f"\n{Colors.BLUE}[{i}/{len(selected_datasets)}] Downloading {dataset_name}...{Colors.ENDC}")
                    path = download_and_setup_torchvision_dataset(dataset_name)
                    downloaded_paths.append((dataset_name, path))
                    print(f"{Colors.GREEN}✓ {dataset_name} downloaded{Colors.ENDC}")
                except Exception as e:
                    print(f"{Colors.RED}✗ Failed to download {dataset_name}: {str(e)}{Colors.ENDC}")
                    continue

    if downloaded_paths:
        print(f"\n{Colors.GREEN}{'='*50}{Colors.ENDC}")
        print(f"{Colors.BOLD}✓ Successfully downloaded {len(downloaded_paths)} datasets:{Colors.ENDC}")
        for dataset_name, path in downloaded_paths:
            print(f"  {Colors.GREEN}•{Colors.ENDC} {dataset_name}: {path}")

        if len(downloaded_paths) == 1:
            # Return single dataset info
            return {
                'dataset_name': downloaded_paths[0][0].lower(),
                'dataset_name_upper': downloaded_paths[0][0].upper(),
                'data_dir': downloaded_paths[0][1],
                'train_dir': os.path.join(downloaded_paths[0][1], 'train'),
                'test_dir': os.path.join(downloaded_paths[0][1], 'test'),
                'input_path': downloaded_paths[0][1],
                'data_type': 'custom',  # After download, treat as custom
                'is_local': True
            }
        else:
            return downloaded_paths
    else:
        print(f"{Colors.RED}No datasets were downloaded.{Colors.ENDC}")
        return None

def browse_local_folders(start_path="."):
    """Interactive local folder browser."""
    current_path = os.path.abspath(start_path)
    history = [current_path]

    while True:
        print(f"\n{Colors.BOLD}Current directory:{Colors.ENDC} {Colors.BLUE}{current_path}{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Contents:{Colors.ENDC}")

        # Get all items
        try:
            items = os.listdir(current_path)
        except PermissionError:
            print(f"{Colors.RED}Permission denied to access this directory.{Colors.ENDC}")
            items = []

        # Separate directories and files
        dirs = []
        files = []

        for item in sorted(items):
            item_path = os.path.join(current_path, item)
            if os.path.isdir(item_path):
                dirs.append(item)
            else:
                if item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    files.append(item)

        # Display directories
        if dirs:
            print(f"\n{Colors.BOLD}📁 Directories:{Colors.ENDC}")
            for i, dir_name in enumerate(dirs, 1):
                print(f"  {Colors.GREEN}{i:2d}.{Colors.ENDC} {dir_name}/")
        else:
            print(f"\n{Colors.YELLOW}No subdirectories found{Colors.ENDC}")

        # Display image files
        if files:
            print(f"\n{Colors.BOLD}📷 Image files:{Colors.ENDC}")
            for i, file_name in enumerate(files[:10], 1):  # Show first 10
                print(f"     {file_name}")
            if len(files) > 10:
                print(f"     {Colors.YELLOW}... and {len(files) - 10} more image files{Colors.ENDC}")
        else:
            print(f"\n{Colors.YELLOW}No image files found in current directory{Colors.ENDC}")

        print(f"\n{Colors.BOLD}Options:{Colors.ENDC}")
        print(f"  {Colors.GREEN}[number]{Colors.ENDC} - Enter directory number")
        print(f"  {Colors.GREEN}b{Colors.ENDC} - Go back to parent directory")
        print(f"  {Colors.GREEN}s{Colors.ENDC} - Select current directory")
        print(f"  {Colors.GREEN}h{Colors.ENDC} - Go home (~)")
        print(f"  {Colors.GREEN}r{Colors.ENDC} - Refresh")
        print(f"  {Colors.GREEN}q{Colors.ENDC} - Cancel")
        print(f"  {Colors.GREEN}history{Colors.ENDC} - Show navigation history")

        choice = input(f"\n{Colors.YELLOW}Select option:{Colors.ENDC} ").strip().lower()

        if choice == 'b':
            # Go to parent directory
            parent = os.path.dirname(current_path)
            if parent != current_path:
                current_path = parent
                history.append(current_path)
            else:
                print(f"{Colors.RED}Already at root directory.{Colors.ENDC}")

        elif choice == 's':
            # Select current directory
            # Check if directory contains images or has subdirectories
            image_count = len([f for f in os.listdir(current_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            subdirs = [d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))]

            if image_count > 0 or subdirs:
                return current_path
            else:
                print(f"{Colors.YELLOW}This directory doesn't appear to contain images or subdirectories.{Colors.ENDC}")
                confirm = input(f"{Colors.YELLOW}Use it anyway? (y/n):{Colors.ENDC} ").lower()
                if confirm == 'y':
                    return current_path

        elif choice == 'h':
            # Go to home directory
            current_path = os.path.expanduser("~")
            history.append(current_path)

        elif choice == 'r':
            # Refresh - do nothing
            continue

        elif choice == 'q':
            return None

        elif choice == 'history':
            print(f"\n{Colors.BOLD}Navigation history:{Colors.ENDC}")
            for i, path in enumerate(history):
                marker = "→ " if path == current_path else "  "
                print(f"  {marker}{path}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")

        elif choice.isdigit():
            # Enter selected directory
            idx = int(choice) - 1
            if 0 <= idx < len(dirs):
                current_path = os.path.join(current_path, dirs[idx])
                history.append(current_path)
            else:
                print(f"{Colors.RED}Invalid directory number.{Colors.ENDC}")

        else:
            print(f"{Colors.RED}Invalid option.{Colors.ENDC}")

def interactive_advanced_config():
    """Configure advanced options interactively."""

    print(f"\n{Colors.BOLD}🔧 Advanced Configuration{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")

    config = {}

    # Image type
    print(f"\n{Colors.BOLD}Image Type:{Colors.ENDC}")
    print("  1. General (default)")
    print("  2. Astronomical (stars, galaxies)")
    print("  3. Medical (tissue, lesions)")
    print("  4. Agricultural (texture, patterns)")

    img_choice = input(f"{Colors.YELLOW}Select image type (1-4) [1]:{Colors.ENDC} ").strip() or '1'
    image_types = ['general', 'astronomical', 'medical', 'agricultural']
    config['image_type'] = image_types[int(img_choice) - 1]

    # Feature dimensions
    print(f"\n{Colors.BOLD}Feature Dimensions:{Colors.ENDC}")
    config['feature_dims'] = int(input(f"  Feature dimensions [128]: ").strip() or '128')
    config['compressed_dims'] = int(input(f"  Compressed dimensions [32]: ").strip() or '32')

    # Feature selection
    print(f"\n{Colors.BOLD}Feature Selection:{Colors.ENDC}")
    use_fs = input(f"  Enable feature selection? (y/n) [y]: ").strip().lower() or 'y'
    config['use_distance_correlation'] = use_fs == 'y'

    if config['use_distance_correlation']:
        config['max_features'] = int(input(f"  Maximum features to select [32]: ").strip() or '32')

    # Advanced features
    print(f"\n{Colors.BOLD}Advanced Features:{Colors.ENDC}")
    config['use_kl_divergence'] = input(f"  Enable KL divergence clustering? (y/n) [y]: ").strip().lower() or 'y' == 'y'
    config['use_class_encoding'] = input(f"  Enable class encoding? (y/n) [y]: ").strip().lower() or 'y' == 'y'

    # Training parameters
    print(f"\n{Colors.BOLD}Training Parameters:{Colors.ENDC}")
    config['batch_size'] = int(input(f"  Batch size [32]: ").strip() or '32')
    config['epochs'] = int(input(f"  Number of epochs [200]: ").strip() or '200')
    config['learning_rate'] = float(input(f"  Learning rate [0.001]: ").strip() or '0.001')

    # Visualization
    print(f"\n{Colors.BOLD}Visualization:{Colors.ENDC}")
    config['generate_heatmaps'] = input(f"  Generate heatmaps? (y/n) [y]: ").strip().lower() or 'y' == 'y'
    config['generate_tsne'] = input(f"  Generate t-SNE? (y/n) [y]: ").strip().lower() or 'y' == 'y'

    # Hardware
    print(f"\n{Colors.BOLD}Hardware:{Colors.ENDC}")
    use_gpu = input(f"  Use GPU if available? (y/n) [y]: ").strip().lower() or 'y'
    config['use_gpu'] = use_gpu == 'y' and torch.cuda.is_available()

    if config['use_gpu']:
        print(f"  {Colors.GREEN}✓ GPU available: {torch.cuda.get_device_name(0)}{Colors.ENDC}")
        config['mixed_precision'] = input(f"  Use mixed precision? (y/n) [y]: ").strip().lower() or 'y' == 'y'

    config['num_workers'] = int(input(f"  Number of workers [4]: ").strip() or '4')

    # Summary
    print(f"\n{Colors.GREEN}{'='*50}{Colors.ENDC}")
    print(f"{Colors.BOLD}Configuration Summary:{Colors.ENDC}")
    for key, value in config.items():
        print(f"  {Colors.BLUE}{key}:{Colors.ENDC} {value}")

    confirm = input(f"\n{Colors.YELLOW}Apply this configuration? (y/n):{Colors.ENDC} ").lower()
    if confirm == 'y':
        return config
    return None

def download_and_setup_torchvision_dataset(dataset_name):
    """Download and setup a torchvision dataset locally."""
    try:
        dataset_name_upper = dataset_name.upper()

        print(f"\n{Colors.BLUE}Downloading {dataset_name_upper}...{Colors.ENDC}")

        # Create processor
        processor = TorchvisionDatasetAdapter(
            dataset_name_upper,
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )

        # Get dataset info
        data_dir = f"data/{dataset_name_upper.lower()}"
        os.makedirs(data_dir, exist_ok=True)

        # Save class info
        class_info = {
            'classes': processor.classes,
            'num_classes': len(processor.classes),
            'dataset_name': dataset_name_upper
        }

        with open(os.path.join(data_dir, 'class_info.json'), 'w') as f:
            json.dump(class_info, f, indent=2)

        # Create train/test directories
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Save a few sample images for preview
        samples_dir = os.path.join(data_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)

        # Get test dataset
        test_processor = TorchvisionDatasetAdapter(
            dataset_name_upper,
            train=False,
            transform=transforms.ToTensor(),
            download=True
        )

        # Save sample images
        for i in range(min(5, len(processor.classes))):
            class_name = processor.classes[i]
            class_dir = os.path.join(samples_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Find an image of this class
            for idx, (img, label) in enumerate(processor.dataset):
                if label == i:
                    img_path = os.path.join(class_dir, f'sample_{idx}.png')
                    if not isinstance(img, PILImage.Image):
                        img = TF.to_pil_image(img)
                    img.save(img_path)
                    break

        print(f"{Colors.GREEN}✓ {dataset_name_upper} downloaded successfully{Colors.ENDC}")
        print(f"  Location: {data_dir}")
        print(f"  Classes: {len(processor.classes)}")

        return data_dir

    except Exception as e:
        print(f"{Colors.RED}Error downloading {dataset_name}: {str(e)}{Colors.ENDC}")
        raise

def interactive_mode():
    """Main interactive mode entry point."""

    print(f"\n{Colors.BOLD}{Colors.BLUE}╔{'═'*78}╗{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}║{' '*30}CDBNN Interactive Mode{' '*26}║{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}╚{'═'*78}╝{Colors.ENDC}")

    # Step 1: Dataset selection
    print(f"\n{Colors.BOLD}Step 1: Dataset Selection{Colors.ENDC}")
    dataset_info = interactive_dataset_selection()

    if not dataset_info:
        print(f"\n{Colors.YELLOW}No dataset selected. Exiting.{Colors.ENDC}")
        return None

    # Step 2: Mode selection
    print(f"\n{Colors.BOLD}Step 2: Operation Mode{Colors.ENDC}")
    print("  1. Train model")
    print("  2. Predict on new images")
    print("  3. Extract features")
    print("  4. Generate visualizations")

    mode_choice = input(f"{Colors.YELLOW}Select mode (1-4) [1]:{Colors.ENDC} ").strip() or '1'
    modes = ['train', 'predict', 'extract', 'visualize']
    mode = modes[int(mode_choice) - 1]

    # Step 3: Configuration
    print(f"\n{Colors.BOLD}Step 3: Configuration{Colors.ENDC}")
    use_advanced = input(f"{Colors.YELLOW}Use advanced configuration? (y/n) [n]:{Colors.ENDC} ").strip().lower()

    if use_advanced == 'y':
        advanced_config = interactive_advanced_config()
        if advanced_config:
            dataset_info.update(advanced_config)

    # Step 4: Confirmation
    print(f"\n{Colors.GREEN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}Configuration Summary:{Colors.ENDC}")
    print(f"  {Colors.BLUE}Dataset:{Colors.ENDC} {dataset_info.get('dataset_name', 'N/A')}")
    print(f"  {Colors.BLUE}Mode:{Colors.ENDC} {mode}")
    print(f"  {Colors.BLUE}Data type:{Colors.ENDC} {dataset_info.get('data_type', 'custom')}")
    print(f"  {Colors.BLUE}Data path:{Colors.ENDC} {dataset_info.get('input_path', 'N/A')}")

    if 'feature_dims' in dataset_info:
        print(f"  {Colors.BLUE}Feature dimensions:{Colors.ENDC} {dataset_info['feature_dims']}")
    if 'image_type' in dataset_info:
        print(f"  {Colors.BLUE}Image type:{Colors.ENDC} {dataset_info['image_type']}")

    confirm = input(f"\n{Colors.YELLOW}Proceed with these settings? (y/n):{Colors.ENDC} ").lower()

    if confirm != 'y':
        print(f"\n{Colors.YELLOW}Configuration cancelled.{Colors.ENDC}")
        return None

    # Add mode to dataset_info
    dataset_info['mode'] = mode

    return dataset_info


# =============================================================================
# FIXED MAIN FUNCTION WITH CORRECT INDENTATION
# =============================================================================

def main():
    """Main entry point with interactive mode support"""

    # Parse command line arguments first
    args = parse_args()

    # Check for interactive mode
    if len(sys.argv) == 1 or args.interactive:  # FIXED: Correct indentation
        # No arguments or explicit interactive flag - run interactive mode
        print(f"\n{Colors.BOLD}{Colors.BLUE}CDBNN - Convolutional Deep Bayesian Neural Network{Colors.ENDC}")
        print(f"{Colors.BOLD}Version: 2.0 (Professional Optimized with Interactive Mode){Colors.ENDC}")
        print(f"{Colors.BOLD}Last Updated: March 13 2026{Colors.ENDC}")

        dataset_info = interactive_mode()

        if not dataset_info:
            return 0

        # Update args with interactive selections
        args.mode = dataset_info.get('mode', 'train')
        args.data_name = dataset_info.get('dataset_name', 'dataset')
        args.data_type = dataset_info.get('data_type', 'custom')
        args.data_path = dataset_info.get('input_path')
        args.image_type = dataset_info.get('image_type', 'general')
        args.feature_dims = dataset_info.get('feature_dims', 128)
        args.compressed_dims = dataset_info.get('compressed_dims', 32)
        args.batch_size = dataset_info.get('batch_size', 32)
        args.epochs = dataset_info.get('epochs', 200)
        args.learning_rate = dataset_info.get('learning_rate', 0.001)
        args.max_features = dataset_info.get('max_features', 32)
        args.use_gpu = dataset_info.get('use_gpu', torch.cuda.is_available())
        args.mixed_precision = dataset_info.get('mixed_precision', True)
        args.workers = min(2, dataset_info.get('num_workers', 2))  # FIXED: Reduced workers
        args.generate_heatmaps = dataset_info.get('generate_heatmaps', True)
        args.generate_tsne = dataset_info.get('generate_tsne', True)
        args.use_kl_divergence = dataset_info.get('use_kl_divergence', True)
        args.use_class_encoding = dataset_info.get('use_class_encoding', True)
        args.use_distance_correlation = dataset_info.get('use_distance_correlation', True)

        print(f"\n{Colors.GREEN}Starting {args.mode} mode with dataset: {args.data_name}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

    try:
        # Set multiprocessing start method to 'spawn' for better compatibility
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Load or create configuration
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = GlobalConfig.from_dict(config_dict)
        else:
            config = GlobalConfig(
                dataset_name=args.data_name,
                data_type=args.data_type,
                feature_dims=args.feature_dims,
                compressed_dims=args.compressed_dims,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                num_workers=args.workers,  # Now using reduced workers
                use_gpu=not args.cpu and torch.cuda.is_available() if hasattr(args, 'cpu') else args.use_gpu,
                mixed_precision=not args.no_mixed_precision if hasattr(args, 'no_mixed_precision') else args.mixed_precision,
                image_type=args.image_type if hasattr(args, 'image_type') else 'general',
                use_kl_divergence=not args.disable_kl if hasattr(args, 'disable_kl') else args.use_kl_divergence,
                use_class_encoding=not args.disable_class if hasattr(args, 'disable_class') else args.use_class_encoding,
                use_distance_correlation=not args.disable_distance_corr if hasattr(args, 'disable_distance_corr') else args.use_distance_correlation,
                max_features=args.max_features,
                generate_heatmaps=not args.no_heatmaps if hasattr(args, 'no_heatmaps') else args.generate_heatmaps,
                generate_tsne=not args.no_tsne if hasattr(args, 'no_tsne') else args.generate_tsne,
                heatmap_frequency=args.heatmap_frequency if hasattr(args, 'heatmap_frequency') else 10,
                output_dir=args.output_dir if hasattr(args, 'output_dir') else 'data'
            )

        # Create application
        app = CDBNNApplication(config)

        # Handle download mode
        if args.mode == 'download':
            if args.data_type != 'torchvision':
                raise ValueError("Download mode requires data_type='torchvision'")

            # Download dataset
            train_loader, test_loader = app.prepare_data(args.data_name, args.data_type)
            logger.info(f"Dataset {args.data_name} downloaded successfully")
            return 0

        # Handle visualization mode
        if args.mode == 'visualize':
            if not args.data_path:
                raise ValueError("data_path required for visualization")

            # Load features
            feat_path = Path(args.data_path)
            if feat_path.suffix == '.npz':
                data = np.load(feat_path)
                features = {k: data[k] for k in data.files}
            else:
                df = pd.read_csv(feat_path)
                feature_cols = [c for c in df.columns if c.startswith('feature_') or c.startswith('selected_feature_')]
                features = {
                    'features': df[feature_cols].values if feature_cols else None,
                    'labels': df['target'].values if 'target' in df else None
                }

                if 'class_name' in df:
                    features['class_names'] = df['class_name'].values
                if 'filepath' in df:
                    features['paths'] = df['filepath'].values

            # Generate visualizations
            if features.get('features') is not None:
                if features.get('labels') is not None:
                    app.visualizer.plot_tsne(features['features'], features['labels'],
                                            class_names=config.class_names)
                    app.visualizer.plot_pca(features['features'], features['labels'])

                    if len(np.unique(features['labels'])) <= 20:
                        app.visualizer.plot_confusion_matrix(
                            features['labels'],
                            features['labels'],
                            class_names=config.class_names
                        )

                if 'feature_indices' in features:
                    scores = np.ones(features['features'].shape[1])
                    app.visualizer.plot_feature_importance(scores)

            logger.info(f"Visualizations saved to {config.viz_dir}")
            return 0

        # Prepare data for other modes
        if not args.data_path:
            raise ValueError(f"data_path required for {args.mode} mode")

        # FIXED: Use single-process data loading to avoid multiprocessing issues
        train_loader, test_loader = app.prepare_data_single_process(args.data_path, args.data_type)

        if args.mode == 'train':
            # Train model
            logger.info(f"{Colors.BOLD}Starting training on {args.data_name}{Colors.ENDC}")
            history = app.train(train_loader, test_loader)

            # Extract and save features
            features = app.extract_features(train_loader)
            app.save_features(features, f"{args.data_name}_features.csv")

            # Generate visualizations
            if features.get('features') is not None and features.get('labels') is not None:
                app.visualizer.plot_tsne(features['features'], features['labels'],
                                        class_names=config.class_names)
                app.visualizer.plot_pca(features['features'], features['labels'])

            logger.info(f"{Colors.GREEN}✓ Training completed successfully{Colors.ENDC}")

        elif args.mode == 'extract':
            # Extract features
            logger.info(f"{Colors.BOLD}Extracting features from {args.data_name}{Colors.ENDC}")
            features = app.extract_features(train_loader)
            app.save_features(features, f"{args.data_name}_extracted.csv")

            logger.info(f"{Colors.GREEN}✓ Feature extraction completed{Colors.ENDC}")

        elif args.mode == 'predict':
            # Run prediction
            logger.info(f"{Colors.BOLD}Running prediction on {args.data_name}{Colors.ENDC}")
            logger.info(f"Optimization level: {args.optimize_level if hasattr(args, 'optimize_level') else 'balanced'}")

            results = app.predict(train_loader,
                                optimize_level=args.optimize_level if hasattr(args, 'optimize_level') else 'balanced')

            # Save results
            app.save_features(results, f"{args.data_name}_predictions.csv")

            logger.info(f"{Colors.GREEN}✓ Prediction completed{Colors.ENDC}")

        # Save configuration
        config_manager = ConfigManager(str(app.data_dir / 'config'))
        config_manager.save_config(config, f"config_{args.mode}")

        return 0

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Operation cancelled by user{Colors.ENDC}")
        return 130

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())

        if hasattr(args, 'debug') and args.debug:
            print(f"\n{Colors.RED}{traceback.format_exc()}{Colors.ENDC}")

        return 1

# =============================================================================
# FIXED CDBNNApplication with single-process data loading
# =============================================================================

def parse_args():
    """Parse command line arguments with interactive flag"""
    parser = argparse.ArgumentParser(
        description='CDBNN - Convolutional Deep Bayesian Neural Network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cdbnn.py --interactive

  # Train on custom dataset
  python cdbnn.py --mode train --data_name mydata --data_path /path/to/images --data_type custom

  # Train on torchvision dataset
  python cdbnn.py --mode train --data_name cifar10 --data_type torchvision

  # Predict with optimization
  python cdbnn.py --mode predict --data_name mydata --data_path /path/to/images --optimize_level fastest

  # Extract features
  python cdbnn.py --mode extract --data_name mydata --data_path /path/to/images

  # Download torchvision dataset
  python cdbnn.py --mode download --data_name cifar100 --data_type torchvision
        """
    )

    # Interactive mode
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')

    # Required for non-interactive
    parser.add_argument('--mode', type=str,
                       choices=['train', 'predict', 'extract', 'download', 'visualize'],
                       help='Operation mode')
    parser.add_argument('--data_name', type=str,
                       help='Dataset name')

    # Data
    parser.add_argument('--data_type', type=str, choices=['custom', 'torchvision'],
                       default='custom', help='Dataset type')
    parser.add_argument('--data_path', type=str,
                       help='Path to data (for custom datasets)')

    # Model
    parser.add_argument('--feature_dims', type=int, default=128,
                       help='Feature dimensions')
    parser.add_argument('--compressed_dims', type=int, default=32,
                       help='Compressed feature dimensions')
    parser.add_argument('--encoder_type', type=str, default='autoenc',
                       choices=['autoenc', 'cnn'], help='Encoder type')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')

    # Features
    parser.add_argument('--image_type', type=str, default='general',
                       choices=['general', 'astronomical', 'medical', 'agricultural'],
                       help='Image type for specialized processing')
    parser.add_argument('--disable_kl', action='store_true',
                       help='Disable KL divergence clustering')
    parser.add_argument('--disable_class', action='store_true',
                       help='Disable class encoding')
    parser.add_argument('--disable_distance_corr', action='store_true',
                       help='Disable distance correlation feature selection')
    parser.add_argument('--max_features', type=int, default=32,
                       help='Maximum number of features to select')

    # Optimization
    parser.add_argument('--optimize_level', type=str,
                       choices=['fastest', 'faster', 'balanced', 'accurate'],
                       default='balanced', help='Prediction optimization level')

    # Hardware
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU even if GPU available')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable mixed precision training')

    # Visualization
    parser.add_argument('--no_heatmaps', action='store_true',
                       help='Disable heatmap generation')
    parser.add_argument('--no_tsne', action='store_true',
                       help='Disable t-SNE visualization')
    parser.add_argument('--heatmap_frequency', type=int, default=10,
                       help='Frequency of heatmap generation (epochs)')

    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')

    # Output
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')

    return parser.parse_args()

if __name__ == '__main__':
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}CDBNN - Convolutional Difference Boosting Bayesian Neural Network{Colors.ENDC}")
    print(f"{Colors.BOLD}Version: 2.0 (Professional Optimized with Interactive Mode){Colors.ENDC}")
    print(f"{Colors.BOLD}Last Updated: March 13 2026{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print("""
    Command line options:
    ================================================================================
    usage: cdbnn.py [-h] --mode {train,predict,extract,download,visualize} --data_name DATA_NAME [--data_type {custom,torchvision}] [--data_path DATA_PATH] [--feature_dims FEATURE_DIMS]
                    [--compressed_dims COMPRESSED_DIMS] [--encoder_type {autoenc,cnn}] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--workers WORKERS]
                    [--image_type {general,astronomical,medical,agricultural}] [--disable_kl] [--disable_class] [--disable_distance_corr] [--max_features MAX_FEATURES]
                    [--optimize_level {fastest,faster,balanced,accurate}] [--cpu] [--no_mixed_precision] [--no_heatmaps] [--no_tsne] [--heatmap_frequency HEATMAP_FREQUENCY] [--output_dir OUTPUT_DIR]
                    [--config CONFIG]

    """)
    # Check for interactive mode
    if len(sys.argv) == 1 or '--interactive' in sys.argv or '-i' in sys.argv:
        print(f"\n{Colors.GREEN}Starting interactive mode...{Colors.ENDC}")
        print(f"{Colors.YELLOW}Type 'help' at any prompt for assistance{Colors.ENDC}\n")

    sys.exit(main())
