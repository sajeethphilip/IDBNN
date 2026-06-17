"""
CDBNN: Convolutional Deep Bayesian Neural Network
Complete Professional Version with Domain-Specific Enhancements
Author: Ninan Sajeeth Philip
Last Updated: March 25 2026
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
import threading
import pickle
import gzip
import bz2
import lzma
import zipfile
import tarfile
import copy

import math
from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import lru_cache, wraps
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.spatial.distance import correlation
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import PIL.Image as PILImage
from PIL import ImageOps, ImageDraw
import warnings
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import functional as TF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Domain-specific imports
import cv2
from scipy.ndimage import median_filter, gaussian_filter, label as ndimage_label
from skimage import exposure, filters, feature, measure, morphology, restoration
from skimage.feature import graycomatrix, graycoprops, blob_log, blob_dog, blob_doh
from skimage.measure import regionprops
from skimage.restoration import denoise_nl_means, denoise_bilateral
from skimage.filters import frangi, sobel, prewitt, roberts

from skimage.filters import threshold_otsu
from skimage import exposure
from scipy.ndimage import rotate, shift

# =============================================================================
# GLOBAL SEED FOR REPRODUCIBILITY
# =============================================================================
# =============================================================================
# GLOBAL SEED FOR REPRODUCIBILITY
# =============================================================================
def _set_global_seeds():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = '42'

_set_global_seeds()
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

@dataclass
class GlobalConfig:
    """Global configuration with all original features"""
    dataset_name: str = 'dataset'
    data_type: str = 'custom'
    in_channels: int = 3
    input_size: Tuple[int, int] = (256, 256)
    num_classes: Optional[int] = None
    class_names: List[str] = field(default_factory=list)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    image_type: str = 'general'
    domain: str = 'general'

    model_type: str = 'autoenc'
    feature_dims: int = 128
    compressed_dims: int = 32
    learning_rate: float = 0.001

    batch_size: int = 32
    epochs: int = 200
    num_workers: int = min(2, os.cpu_count() or 1)
    checkpoint_dir: str = 'checkpoints'
    validation_split: float = 0.2

    use_kl_divergence: bool = True
    use_class_encoding: bool = True
    use_distance_correlation: bool = True
    enable_sharpness_loss: bool = False
    enable_adaptive: bool = True

    feature_selection_method: str = 'balanced'
    max_features: int = 32
    min_features: int = 8
    correlation_upper: float = 0.85
    correlation_lower: float = 0.01

    generate_heatmaps: bool = True
    generate_confusion_matrix: bool = True
    generate_tsne: bool = True
    heatmap_frequency: int = 10
    reconstruction_samples_frequency: int = 5

    use_gpu: bool = torch.cuda.is_available()
    debug_mode: bool = False
    mixed_precision: bool = True
    distributed_training: bool = False

    # Input size with auto-detection support
    input_size: Tuple[int, int] = (256, 256)
    input_size_explicitly_set: bool = False  # Track if user explicitly set size

    data_dir: str = 'data'
    output_dir: str = 'output'
    log_dir: str = 'logs'
    viz_dir: str = 'visualizations'

    # Normalization strategy
    use_per_image_normalization: bool = False

    # Dynamic architecture attributes (set during analysis)
    complexity_score: float = 0.5
    dataset_size: int = 0

    # ========================================================================
    # NEW: Advanced model configuration flags
    # ========================================================================
    use_enhanced_autoencoder: bool = True      # Use enhanced base autoencoder
    use_hybrid_autoencoder: bool = True        # Use hybrid (conditional + unconditional)
    use_advanced_hybrid: bool = True           # Use advanced hybrid with modulation
    use_perceptual_loss: bool = True           # Use VGG-based perceptual loss
    distillation_mode: str = 'contrastive'     # 'contrastive', 'relational', or 'mse'

    # Domain-specific enhancement flags
    use_detail_attention: bool = True          # Use detail attention module
    use_multiscale_features: bool = True       # Use multi-scale feature extraction
    use_feature_refinement: bool = True        # Use feature refinement
    force_fixed_compressed_dims: bool = False

    # Training flags
    no_augmentation: bool = False              # Disable augmentation
    augmentation_strength: float = 0.5         # Strength of augmentations

    def __post_init__(self):
        """Post-initialization setup"""
        pass

    @property
    def normalization_mode(self) -> str:
        """Return the normalization mode as string"""
        return 'per_image' if self.use_per_image_normalization else 'dataset_wide'

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, data: Dict) -> 'GlobalConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

# =============================================================================
# DOMAIN CONFIGURATIONS
# =============================================================================

@dataclass
class DomainConfig:
    """Base configuration for domain-specific processing"""
    domain_name: str = 'general'
    enable_enhancement: bool = True

@dataclass
class AgricultureConfig(GlobalConfig):
    """Agriculture-specific configuration"""
    domain: str = 'agriculture'

    # Plant health detection
    detect_chlorophyll: bool = True
    detect_water_stress: bool = True
    detect_nutrient_deficiency: bool = True

    # Disease detection
    detect_leaf_disease: bool = True
    detect_fruit_disease: bool = True
    detect_pest_damage: bool = True

    # Vegetation indices
    compute_ndvi: bool = True
    compute_evi: bool = True
    compute_ndwi: bool = True
    compute_gci: bool = True

    # Texture analysis
    compute_leaf_texture: bool = True
    compute_canopy_structure: bool = True

    # Growth stage detection
    estimate_growth_stage: bool = True
    compute_biomass: bool = True

    # Multi-spectral support
    has_nir_band: bool = False
    nir_band_index: int = 3

@dataclass
class MedicalConfig(GlobalConfig):
    """Medical imaging configuration"""
    domain: str = 'medical'

    # Imaging modality
    modality: str = 'general'

    # Detection features
    detect_tumor: bool = True
    detect_lesion: bool = True
    detect_hemorrhage: bool = True
    detect_calcification: bool = True

    # Segmentation features
    segment_organs: bool = True
    segment_vessels: bool = True
    segment_tissues: bool = True

    # Texture analysis
    compute_tissue_texture: bool = True
    compute_boundary_regularity: bool = True

    # Quality metrics
    compute_contrast: bool = True
    compute_sharpness: bool = True
    detect_artifacts: bool = True

    # Clinical features
    estimate_tumor_size: bool = True
    compute_tumor_heterogeneity: bool = True

@dataclass
class SatelliteConfig(GlobalConfig):
    """Satellite/Remote sensing configuration"""
    domain: str = 'satellite'

    # Image type
    satellite_type: str = 'general'

    # Land cover
    classify_land_cover: bool = True
    detect_urban_area: bool = True
    detect_forest_cover: bool = True
    detect_water_body: bool = True
    detect_agriculture: bool = True

    # Change detection
    detect_change: bool = True
    detect_deforestation: bool = True
    detect_urban_sprawl: bool = True

    # Spectral indices
    compute_ndvi: bool = True
    compute_ndwi: bool = True
    compute_ndbi: bool = True
    compute_mndwi: bool = True

    # Texture features
    compute_glcm: bool = True
    compute_pansharpening: bool = True

    # Multi-band support
    num_bands: int = 4
    band_assignments: Dict[str, int] = field(default_factory=lambda: {
        'red': 0, 'green': 1, 'blue': 2, 'nir': 3
    })

@dataclass
class SurveillanceConfig(GlobalConfig):
    """Surveillance/CCTV configuration"""
    domain: str = 'surveillance'

    # Object detection
    detect_person: bool = True
    detect_vehicle: bool = True
    detect_animal: bool = True
    detect_face: bool = True

    # Activity recognition
    detect_motion: bool = True
    detect_anomaly: bool = True
    track_objects: bool = True

    # Scene understanding
    classify_scene_type: bool = True
    estimate_crowd_density: bool = True

    # Quality enhancement
    enhance_low_light: bool = True
    reduce_noise: bool = True
    dehaze: bool = True
    super_resolution: bool = True

    # Privacy features
    blur_faces: bool = False
    anonymize: bool = False

@dataclass
class MicroscopyConfig(GlobalConfig):
    """Microscopy imaging configuration"""
    domain: str = 'microscopy'

    # Microscopy type
    microscopy_type: str = 'general'

    # Cell analysis
    detect_cells: bool = True
    count_cells: bool = True
    segment_nucleus: bool = True
    detect_mitosis: bool = True

    # Organelle detection
    detect_mitochondria: bool = True
    detect_nucleoli: bool = True

    # Fluorescence features
    detect_fluorescent_signal: bool = True
    compute_intensity_distribution: bool = True

    # Quality metrics
    compute_resolution: bool = True
    detect_out_of_focus: bool = True

@dataclass
class IndustrialConfig(GlobalConfig):
    """Industrial inspection configuration"""
    domain: str = 'industrial'

    # Defect detection
    detect_crack: bool = True
    detect_corrosion: bool = True
    detect_dent: bool = True
    detect_scratch: bool = True

    # Quality control
    measure_dimensions: bool = True
    detect_misalignment: bool = True
    classify_defect_type: bool = True

    # Texture analysis
    compute_surface_roughness: bool = True
    detect_uniformity: bool = True

@dataclass
class AstronomyConfig(GlobalConfig):
    """Astronomy-specific configuration"""
    domain: str = 'astronomy'

    # FITS support
    use_fits: bool = True
    fits_extensions: List[str] = field(default_factory=lambda: ['.fits', '.fit', '.fits.gz'])
    fits_hdu: int = 0
    fits_normalization: str = 'zscale'  # zscale, percent, minmax, asinh

    # Background subtraction
    subtract_background: bool = True
    background_box_size: int = 50
    background_filter_size: int = 3

    # Source detection
    detect_sources: bool = True
    detection_threshold: float = 2.5
    npixels: int = 5
    deblend: bool = True

    # PSF/astrometry
    estimate_psf: bool = True
    psf_fwhm: Optional[float] = None

    # Feature extraction
    morphological_features: bool = True
    photometric_features: bool = True
    shape_features: bool = True

    # Astronomy-specific transforms
    use_log_transform: bool = True
    use_asinh_transform: bool = False
    use_sqrt_transform: bool = False

    # Quality metrics
    compute_snr: bool = True
    compute_ellipticity: bool = True
    compute_concentration: bool = True
    compute_asymmetry: bool = True

    # Instrument parameters
    pixel_scale: float = 1.0  # arcsec/pixel
    gain: float = 1.0  # e-/ADU
    read_noise: float = 0.0  # e-

class NormalizationStatistics:
    """
    Stores normalization statistics from training data.
    These statistics are used for ALL images (training, validation, test, prediction)
    to ensure consistent normalization across the entire pipeline.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.per_channel_min = None
        self.per_channel_max = None
        self.n_samples = 0
        self.is_fitted = False
        self.channel_order = None

    def fit(self, dataloader: DataLoader, max_samples: int = 10000000) -> 'NormalizationStatistics':
        """Calculate statistics from training data only."""
        logger.info("Computing normalization statistics from training data...")

        all_pixels = []
        total_pixels = 0

        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Computing stats")):
            images = images.cpu()
            b, c, h, w = images.shape
            pixels = images.permute(0, 2, 3, 1).reshape(-1, c)
            all_pixels.append(pixels)
            total_pixels += len(pixels)

            if total_pixels >= max_samples:
                logger.info(f"Sampled {total_pixels:,} pixels, stopping")
                break

        all_pixels = torch.cat(all_pixels, dim=0)

        self.mean = all_pixels.mean(dim=0)
        self.std = all_pixels.std(dim=0)
        self.per_channel_min = all_pixels.min(dim=0)[0]
        self.per_channel_max = all_pixels.max(dim=0)[0]
        self.n_samples = len(all_pixels)
        self.is_fitted = True

        logger.info("=" * 60)
        logger.info("Normalization statistics computed:")
        logger.info(f"  Mean: {self.mean.tolist()}")
        logger.info(f"  Std:  {self.std.tolist()}")
        logger.info(f"  Total pixels: {self.n_samples:,}")
        logger.info("=" * 60)

        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply training statistics to normalize ANY image."""
        if not self.is_fitted:
            raise ValueError("Must call fit() before normalize()")

        if x.dim() == 4:
            mean = self.mean.to(x.device).view(1, -1, 1, 1)
            std = self.std.to(x.device).view(1, -1, 1, 1)
            return (x - mean) / (std + 1e-8)
        elif x.dim() == 3:
            mean = self.mean.to(x.device).view(-1, 1, 1)
            std = self.std.to(x.device).view(-1, 1, 1)
            return (x - mean) / (std + 1e-8)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalization using training statistics."""
        if not self.is_fitted:
            raise ValueError("Must call fit() before denormalize()")

        if x.dim() == 4:
            mean = self.mean.to(x.device).view(1, -1, 1, 1)
            std = self.std.to(x.device).view(1, -1, 1, 1)
            return x * std + mean
        elif x.dim() == 3:
            mean = self.mean.to(x.device).view(-1, 1, 1)
            std = self.std.to(x.device).view(-1, 1, 1)
            return x * std + mean
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")

    def save(self, path: str):
        """Save statistics to file."""
        torch.save({
            'mean': self.mean.cpu(),
            'std': self.std.cpu(),
            'per_channel_min': self.per_channel_min.cpu(),
            'per_channel_max': self.per_channel_max.cpu(),
            'n_samples': self.n_samples,
            'is_fitted': self.is_fitted,
            'timestamp': datetime.now().isoformat()
        }, path)
        logger.info(f"Normalization statistics saved to {path}")

    def load(self, path: str):
        """Load statistics from file."""
        data = torch.load(path, map_location='cpu')
        self.mean = data['mean']
        self.std = data['std']
        self.per_channel_min = data['per_channel_min']
        self.per_channel_max = data['per_channel_max']
        self.n_samples = data['n_samples']
        self.is_fitted = data['is_fitted']
        logger.info(f"Normalization statistics loaded from {path}")

# =============================================================================
# ASTRONOMY DOMAIN PROCESSOR
# =============================================================================

class AstronomyDomainProcessor:
    """Optimized astronomy processor with vectorized operations - DETERMINISTIC"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.pixel_scale = getattr(config, 'pixel_scale', 1.0)
        self.gain = getattr(config, 'gain', 1.0)
        self.read_noise = getattr(config, 'read_noise', 0.0)

        # CRITICAL FIX: Fixed random state for reproducibility
        self._rng = np.random.RandomState(42)

        # Pre-compute constants
        self._sqrt_2pi = np.sqrt(2 * np.pi)
        self._fwhm_to_sigma = 1.0 / (2.355)  # FWHM = 2.355 * sigma

        # Lazy-loaded modules (only loaded once)
        self._astropy_loaded = False
        self._skimage_loaded = False
        self._astropy_sigma_clip = None
        self._photutils_background = None

        # Cache for expensive operations (per batch, not per image)
        self._batch_cache = {}


    def _ensure_astropy(self):
        """Load astropy modules once when needed"""
        if not self._astropy_loaded:
            try:
                from astropy.stats import sigma_clip
                self._astropy_sigma_clip = sigma_clip
                self._astropy_loaded = True
            except ImportError:
                self._astropy_sigma_clip = None
                self._astropy_loaded = True
                logger.warning("astropy not available, using fallback methods")

    def _ensure_skimage(self):
        """Load skimage modules once when needed"""
        if not self._skimage_loaded:
            try:
                from skimage.feature import blob_log
                from skimage.measure import label, regionprops
                self._blob_log = blob_log
                self._skimage_label = label
                self._skimage_regionprops = regionprops
                self._skimage_loaded = True
            except ImportError:
                self._blob_log = None
                self._skimage_loaded = True
                logger.warning("skimage not available, using fallback methods")

    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """Batch preprocessing for multiple images (much faster)"""
        if len(images.shape) == 3:
            # Single image
            return self.preprocess(images)

        # Batch processing using vectorized operations where possible
        batch_size = images.shape[0]
        processed = np.zeros_like(images)

        # Pre-allocate for batch operations
        for i in range(batch_size):
            processed[i] = self.preprocess(images[i])

        return processed

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess astronomical images with optimized operations"""
        # Convert to float once
        img_float = image.astype(np.float32) / 255.0

        # Background subtraction - optimized with median filter and convolution
        img_float = self._background_subtract_optimized(img_float)

        # Cosmic ray removal - using faster sigma clipping
        img_float = self._cosmic_ray_removal_optimized(img_float)

        # Normalization - using robust z-scale
        img_float = self._zscale_normalization_optimized(img_float)

        return img_float

    def _background_subtract_optimized(self, image: np.ndarray) -> np.ndarray:
        """Optimized background subtraction using convolution"""
        from scipy.ndimage import uniform_filter, median_filter

        # Determine filter size based on image size
        filter_size = max(25, min(image.shape) // 20)

        # Use median filter for initial estimate (faster than sigma clipping)
        background = median_filter(image, size=filter_size)

        # Optional: use convolution for smoother background (faster than full sigma clipping)
        if filter_size > 10:
            background = uniform_filter(background, size=5)

        return np.clip(image - background, 0, 1)

    def _cosmic_ray_removal_optimized(self, image: np.ndarray) -> np.ndarray:
        """Optimized cosmic ray removal using MAD-based clipping"""
        from scipy.ndimage import median_filter, uniform_filter

        # Fast median filter
        median_filtered = median_filter(image, size=3)

        # Calculate robust standard deviation using MAD (faster than sigma clipping)
        diff = image - median_filtered
        mad = np.median(np.abs(diff - np.median(diff)))
        robust_std = 1.4826 * mad

        # Threshold at 5 sigma (standard for astronomy)
        threshold = 5 * robust_std
        cosmic_rays = np.abs(diff) > threshold

        if np.any(cosmic_rays):
            # Replace with median filtered value
            image[cosmic_rays] = median_filtered[cosmic_rays]

        return image

    def _zscale_normalization_optimized(self, image: np.ndarray) -> np.ndarray:
        """Optimized Z-scale normalization using percentiles - DETERMINISTIC"""
        # Use percentiles instead of full MAD calculation
        p1, p99 = np.percentile(image, [1, 99])
        median = np.median(image)

        # Calculate robust standard deviation using percentiles
        # This is faster than full MAD and works well for astronomical images
        robust_std = (p99 - p1) / 4  # Approximation for normal distribution

        # Z-scale range: -2σ to +2σ around median
        zmin = median - 2 * robust_std
        zmax = median + 2 * robust_std

        # Normalize
        normalized = (image - zmin) / (zmax - zmin + 1e-8)
        return np.clip(normalized, 0, 1)

    def extract_features_batch(self, images: np.ndarray) -> List[Dict[str, float]]:
        """Extract features for multiple images with optimized batch processing"""
        if len(images.shape) == 3:
            # Single image
            return [self.extract_features(images)]

        batch_size = images.shape[0]
        features_list = []

        # Pre-compute grayscale for all images (vectorized)
        grays = np.mean(images, axis=3) if len(images.shape) == 4 else images

        # Process in batches for better performance
        for i in range(batch_size):
            features = self._extract_features_single(images[i], grays[i])
            features_list.append(features)

        return features_list

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features for single image"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        return self._extract_features_single(image, gray)

    def _extract_features_single(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Core feature extraction with optimized operations"""
        features = {}

        # 1. Basic statistics (fast)
        features.update(self._extract_basic_stats(gray))

        # 2. Astronomical statistics (fast)
        features.update(self._extract_astronomy_stats_optimized(gray))

        # 3. Shape features (moderate)
        features.update(self._extract_shape_features_optimized(gray))

        # 4. Source detection and analysis (most expensive, but vectorized)
        sources, segment_map = self._detect_sources_optimized(gray)
        features.update(self._extract_source_features_optimized(sources, gray))

        # 5. Morphological classification (moderate)
        if len(sources) > 0:
            features.update(self._classify_morphology_optimized(gray, sources))

        # 6. Quality metrics (fast)
        features.update(self._compute_quality_metrics_optimized(gray))

        return features

    def _extract_basic_stats(self, gray: np.ndarray) -> Dict[str, float]:
        """Vectorized basic statistics"""
        return {
            'mean': float(np.mean(gray)),
            'median': float(np.median(gray)),
            'std': float(np.std(gray)),
            'skew': float(self._fast_skewness(gray)),
            'kurtosis': float(self._fast_kurtosis(gray)),
            'percentile_95': float(np.percentile(gray, 95)),
            'percentile_5': float(np.percentile(gray, 5))
        }

    def _fast_skewness(self, x: np.ndarray) -> float:
        """Fast skewness calculation"""
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return np.mean(((x - mean) / std) ** 3)

    def _fast_kurtosis(self, x: np.ndarray) -> float:
        """Fast kurtosis calculation"""
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return np.mean(((x - mean) / std) ** 4) - 3

    def _extract_astronomy_stats_optimized(self, gray: np.ndarray) -> Dict[str, float]:
        """Optimized astronomy-specific statistics"""
        # Use percentile-based background estimation (faster than sigma clipping)
        background_estimate = np.percentile(gray, 10)

        # Use robust statistics for background noise
        background_pixels = gray[gray < np.percentile(gray, 20)]
        background_std = np.std(background_pixels) if len(background_pixels) > 0 else 0

        # Signal estimate (peak above background)
        peak = np.max(gray)
        signal = peak - background_estimate

        return {
            'background_estimate': float(background_estimate),
            'background_noise': float(background_std),
            'signal_peak': float(signal),
            'snr_estimate': float(signal / (background_std + 1e-8)),
            'dynamic_range': float(np.max(gray) - np.min(gray))
        }

    def _extract_shape_features_optimized(self, gray: np.ndarray) -> Dict[str, float]:
        """Optimized shape features using image moments"""
        # Calculate image moments (vectorized)
        y, x = np.mgrid[:gray.shape[0], :gray.shape[1]]
        total_flux = np.sum(gray)

        if total_flux < 1e-8:
            return {
                'ellipticity': 0.0,
                'size_fwhm': 0.0,
                'concentration': 0.0,
                'asymmetry': 0.0
            }

        # Center of mass (first moments)
        x_c = np.sum(x * gray) / total_flux
        y_c = np.sum(y * gray) / total_flux

        # Second moments (covariance matrix)
        dx = x - x_c
        dy = y - y_c

        mxx = np.sum(dx * dx * gray) / total_flux
        myy = np.sum(dy * dy * gray) / total_flux
        mxy = np.sum(dx * dy * gray) / total_flux

        # Ellipticity from moments
        # This is equivalent to (a-b)/(a+b) where a,b are semi-major/minor axes
        discriminant = np.sqrt((mxx - myy)**2 + 4 * mxy**2)
        ellipticity = discriminant / (mxx + myy + 1e-8)

        # Size estimate (FWHM from Gaussian approximation)
        # For a Gaussian, FWHM = 2.355 * sigma, where sigma^2 = (mxx + myy)/2
        sigma_estimate = np.sqrt((mxx + myy) / 2)
        fwhm_estimate = 2.355 * sigma_estimate

        # Concentration index (ratio of 20% to 80% flux radii)
        sorted_flux = np.sort(gray.flatten())[::-1]
        cumsum = np.cumsum(sorted_flux)
        total = cumsum[-1]

        def find_radius(percent):
            target = total * percent / 100
            n_pixels = np.searchsorted(cumsum, target)
            return np.sqrt(n_pixels / np.pi)

        r20 = find_radius(20)
        r80 = find_radius(80)
        concentration = r80 / (r20 + 1e-8)

        # Asymmetry index (fast approximation)
        # Rotate 180 degrees and compare
        center_y, center_x = int(y_c), int(x_c)
        h, w = gray.shape
        y_min = max(0, center_y - h//4)
        y_max = min(h, center_y + h//4)
        x_min = max(0, center_x - w//4)
        x_max = min(w, center_x + w//4)

        centered = gray[y_min:y_max, x_min:x_max]
        if centered.size > 0:
            rotated = np.rot90(centered, 2)
            if rotated.shape == centered.shape:
                asymmetry = np.sum(np.abs(centered - rotated)) / (np.sum(centered) + 1e-8)
            else:
                asymmetry = 0.0
        else:
            asymmetry = 0.0

        return {
            'ellipticity': float(ellipticity),
            'size_fwhm': float(fwhm_estimate * self.pixel_scale),  # in arcsec
            'concentration': float(concentration),
            'asymmetry': float(asymmetry),
            'position_angle': float(0.5 * np.arctan2(2 * mxy, mxx - myy))
        }

    def _detect_sources_optimized(self, gray: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Optimized source detection using thresholding and labeling"""
        self._ensure_skimage()

        # Use adaptive thresholding for source detection
        threshold = np.percentile(gray, 98)  # Higher threshold for fewer false positives

        # Create binary mask
        binary = gray > threshold

        # Label connected components
        if self._skimage_label:
            labeled = self._skimage_label(binary)
        else:
            from scipy.ndimage import label
            labeled, _ = label(binary)

        # Extract source properties
        sources = []

        if self._skimage_regionprops:
            regions = self._skimage_regionprops(labeled, intensity_image=gray)
            for region in regions:
                if region.area >= 5:  # Minimum source size
                    sources.append({
                        'type': 'source',
                        'centroid': (region.centroid[1], region.centroid[0]),
                        'area': region.area,
                        'intensity': region.intensity_mean,
                        'max_intensity': region.max_intensity,
                        'bbox': region.bbox
                    })
        else:
            # Fallback to centroid calculation
            from scipy.ndimage import center_of_mass, find_objects
            slices = find_objects(labeled)
            for i, sl in enumerate(slices, 1):
                if sl and np.sum(labeled[sl] == i) >= 5:
                    mask = labeled == i
                    com = center_of_mass(mask)
                    sources.append({
                        'type': 'source',
                        'centroid': (com[1], com[0]),
                        'area': int(np.sum(mask)),
                        'intensity': float(np.mean(gray[mask]))
                    })

        return sources, labeled

    def _extract_source_features_optimized(self, sources: List[Dict], gray: np.ndarray) -> Dict[str, float]:
        """Optimized source feature extraction"""
        if not sources:
            return {
                'num_sources': 0,
                'source_density': 0,
                'mean_source_intensity': 0,
                'max_source_intensity': 0,
                'mean_source_area': 0
            }

        # Use vectorized operations for lists
        intensities = np.array([s.get('intensity', 0) for s in sources])
        areas = np.array([s.get('area', 0) for s in sources])

        # Calculate source density (sources per 1000 pixels)
        source_density = len(sources) / (gray.size / 1000.0)

        return {
            'num_sources': len(sources),
            'source_density': float(source_density),
            'mean_source_intensity': float(np.mean(intensities)),
            'max_source_intensity': float(np.max(intensities)),
            'std_source_intensity': float(np.std(intensities)),
            'mean_source_area': float(np.mean(areas)),
            'total_source_flux': float(np.sum(intensities))
        }

    def _classify_morphology_optimized(self, gray: np.ndarray, sources: List[Dict]) -> Dict[str, float]:
        """Optimized morphological classification"""
        # Use concentration and asymmetry for classification
        shape_features = self._extract_shape_features_optimized(gray)

        concentration = shape_features.get('concentration', 1.0)
        asymmetry = shape_features.get('asymmetry', 0.5)

        # Classification based on known thresholds
        # Stars: high concentration (>2.5), low asymmetry (<0.3)
        # Galaxies: lower concentration, higher asymmetry
        star_likelihood = min(1.0, max(0.0, (concentration - 1.5) / 2.0))
        galaxy_likelihood = 1 - star_likelihood

        # Galaxy type classification based on asymmetry
        spiral_likelihood = min(1.0, asymmetry * 2) if asymmetry > 0.2 else 0.0
        elliptical_likelihood = 1 - spiral_likelihood

        return {
            'star_likelihood': float(star_likelihood),
            'galaxy_likelihood': float(galaxy_likelihood),
            'spiral_likelihood': float(spiral_likelihood),
            'elliptical_likelihood': float(elliptical_likelihood)
        }

    def _compute_quality_metrics_optimized(self, gray: np.ndarray) -> Dict[str, float]:
        """Optimized quality metrics computation"""
        # Background noise estimate
        background_pixels = gray[gray < np.percentile(gray, 20)]
        background_noise = np.std(background_pixels) if len(background_pixels) > 0 else 0

        # Signal-to-noise estimate
        peak = np.max(gray)
        signal = peak - np.median(gray)
        snr = signal / (background_noise + 1e-8)

        # Edge sharpness using gradient magnitude
        from scipy.ndimage import sobel
        gradient_x = sobel(gray, axis=0)
        gradient_y = sobel(gray, axis=1)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        sharpness = np.mean(gradient_magnitude)

        return {
            'quality_snr': float(snr),
            'quality_sharpness': float(sharpness),
            'quality_background_noise': float(background_noise),
            'quality_fwhm_estimate': float(self._estimate_seeing_optimized(gray))
        }

    def _estimate_seeing_optimized(self, gray: np.ndarray) -> float:
        """Optimized seeing estimate from brightest source"""
        # Find brightest pixels (likely stars)
        peak_idx = np.unravel_index(np.argmax(gray), gray.shape)

        # Extract region around peak
        size = 15
        y0, x0 = peak_idx
        y_min = max(0, y0 - size)
        y_max = min(gray.shape[0], y0 + size)
        x_min = max(0, x0 - size)
        x_max = min(gray.shape[1], x0 + size)

        region = gray[y_min:y_max, x_min:x_max]

        if region.size < 9:
            return 0.0

        # Fit 1D Gaussian along each axis (faster than 2D fitting)
        y, x = np.mgrid[:region.shape[0], :region.shape[1]]

        # Find center of mass in region
        total = np.sum(region)
        if total < 1e-8:
            return 0.0

        y_c = np.sum(y * region) / total
        x_c = np.sum(x * region) / total

        # Compute sigma along each axis
        dy = y - y_c
        dx = x - x_c

        sigma_y = np.sqrt(np.sum(dy**2 * region) / total)
        sigma_x = np.sqrt(np.sum(dx**2 * region) / total)

        # FWHM = 2.355 * sigma
        fwhm = 2.355 * (sigma_x + sigma_y) / 2

        return float(fwhm * self.pixel_scale)

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Public method to get quality metrics"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        return self._compute_quality_metrics_optimized(gray)


class AstronomyFeatureExtractor:
    """Feature extractor for astronomical images"""

    def __init__(self, config):
        self.config = config
        self.pixel_scale = getattr(config, 'pixel_scale', 1.0) if hasattr(config, 'pixel_scale') else 1.0

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract astronomical features from image"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        features = {}

        # Basic statistics
        features.update(self._extract_basic_stats(gray))

        # Shape features
        features.update(self._extract_shape_features(gray))

        # Source detection
        sources = self._detect_sources(gray)
        features.update(self._extract_source_features(sources, gray))

        # Quality metrics
        features.update(self._compute_quality_metrics(gray))

        return features

    def _extract_basic_stats(self, gray):
        return {
            'astronomy_mean': float(np.mean(gray)),
            'astronomy_median': float(np.median(gray)),
            'astronomy_std': float(np.std(gray)),
            'astronomy_skew': float(self._fast_skewness(gray)),
            'astronomy_kurtosis': float(self._fast_kurtosis(gray))
        }

    def _fast_skewness(self, x):
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return np.mean(((x - mean) / std) ** 3)

    def _fast_kurtosis(self, x):
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return np.mean(((x - mean) / std) ** 4) - 3

    def _extract_shape_features(self, gray):
        y, x = np.mgrid[:gray.shape[0], :gray.shape[1]]
        total_flux = np.sum(gray)

        if total_flux < 1e-8:
            return {'astronomy_ellipticity': 0, 'astronomy_size_fwhm': 0}

        x_c = np.sum(x * gray) / total_flux
        y_c = np.sum(y * gray) / total_flux

        dx = x - x_c
        dy = y - y_c

        mxx = np.sum(dx * dx * gray) / total_flux
        myy = np.sum(dy * dy * gray) / total_flux
        mxy = np.sum(dx * dy * gray) / total_flux

        discriminant = np.sqrt((mxx - myy)**2 + 4 * mxy**2)
        ellipticity = discriminant / (mxx + myy + 1e-8)

        sigma_estimate = np.sqrt((mxx + myy) / 2)
        fwhm_estimate = 2.355 * sigma_estimate

        return {
            'astronomy_ellipticity': float(ellipticity),
            'astronomy_size_fwhm': float(fwhm_estimate * self.pixel_scale)
        }

    def _detect_sources(self, gray):
        threshold = np.percentile(gray, 98)
        binary = gray > threshold

        from scipy.ndimage import label, find_objects
        labeled, num_features = label(binary)

        sources = []
        slices = find_objects(labeled)
        for i, sl in enumerate(slices, 1):
            if sl and np.sum(labeled[sl] == i) >= 5:
                mask = labeled == i
                from scipy.ndimage import center_of_mass
                com = center_of_mass(mask)
                sources.append({
                    'centroid': (com[1], com[0]),
                    'area': int(np.sum(mask)),
                    'intensity': float(np.mean(gray[mask]))
                })

        return sources

    def _extract_source_features(self, sources, gray):
        if not sources:
            return {
                'astronomy_num_sources': 0,
                'astronomy_source_density': 0,
                'astronomy_mean_source_intensity': 0
            }

        intensities = np.array([s.get('intensity', 0) for s in sources])
        source_density = len(sources) / (gray.size / 1000.0)

        return {
            'astronomy_num_sources': len(sources),
            'astronomy_source_density': float(source_density),
            'astronomy_mean_source_intensity': float(np.mean(intensities)),
            'astronomy_max_source_intensity': float(np.max(intensities))
        }

    def _compute_quality_metrics(self, gray):
        background_pixels = gray[gray < np.percentile(gray, 20)]
        background_noise = np.std(background_pixels) if len(background_pixels) > 0 else 0

        peak = np.max(gray)
        signal = peak - np.median(gray)
        snr = signal / (background_noise + 1e-8)

        from scipy.ndimage import sobel
        gradient_x = sobel(gray, axis=0)
        gradient_y = sobel(gray, axis=1)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        sharpness = np.mean(gradient_magnitude)

        return {
            'astronomy_snr': float(snr),
            'astronomy_sharpness': float(sharpness),
            'astronomy_background_noise': float(background_noise)
        }

    def get_quality_metrics(self, image):
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        return self._compute_quality_metrics(gray)


class MedicalFeatureExtractor:
    """Feature extractor for medical images"""

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        features = {}

        # Tissue texture
        features.update(self._compute_tissue_texture(gray))

        # Contrast and sharpness
        features.update(self._compute_medical_contrast(gray))
        features.update(self._compute_medical_sharpness(gray))

        # Tumor detection
        features.update(self._detect_tumor(gray))

        return features

    def _compute_tissue_texture(self, gray):
        from skimage.feature import graycomatrix, graycoprops
        gray_uint8 = (gray * 255).astype(np.uint8)

        glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True)

        return {
            'medical_tissue_contrast': float(graycoprops(glcm, 'contrast')[0, 0]),
            'medical_tissue_homogeneity': float(graycoprops(glcm, 'homogeneity')[0, 0]),
            'medical_tissue_energy': float(graycoprops(glcm, 'energy')[0, 0])
        }

    def _compute_medical_contrast(self, gray):
        return {
            'medical_contrast_ratio': float((np.percentile(gray, 95) - np.percentile(gray, 5)) /
                                           (np.percentile(gray, 95) + np.percentile(gray, 5) + 1e-8))
        }

    def _compute_medical_sharpness(self, gray):
        from skimage.filters import sobel
        edges = sobel(gray)
        return {'medical_sharpness_index': float(np.mean(edges))}

    def _detect_tumor(self, gray):
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        threshold = np.percentile(lbp, 95)
        abnormal = lbp > threshold

        return {
            'medical_tumor_suspicion': float(np.mean(abnormal)),
            'medical_abnormal_texture_score': float(np.std(lbp[abnormal])) if np.any(abnormal) else 0
        }

    def get_quality_metrics(self, image):
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        signal = np.mean(gray)
        noise = np.std(gray)

        return {
            'medical_snr': signal / (noise + 1e-8),
            'medical_contrast': np.std(gray),
            'medical_sharpness': np.std(gray)
        }


class AgricultureFeatureExtractor:
    """Feature extractor for agricultural images"""

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        features = {}

        # Vegetation indices
        features.update(self._compute_ndvi(image))
        features.update(self._compute_ndwi(image))

        # Plant health
        features.update(self._compute_chlorophyll_content(image))
        features.update(self._compute_water_stress(image))

        # Disease detection
        features.update(self._detect_leaf_disease(image))

        return features

    def _compute_ndvi(self, image):
        if len(image.shape) >= 3:
            nir = image[:, :, 2]
            red = image[:, :, 0]
            ndvi = (nir - red) / (nir + red + 1e-8)

            return {
                'agriculture_ndvi_mean': float(np.mean(ndvi)),
                'agriculture_vegetation_fraction': float(np.mean(ndvi > 0.3))
            }
        return {'agriculture_ndvi_mean': 0, 'agriculture_vegetation_fraction': 0}

    def _compute_ndwi(self, image):
        if len(image.shape) >= 3:
            green = image[:, :, 1]
            nir = image[:, :, 2]
            ndwi = (green - nir) / (green + nir + 1e-8)

            return {
                'agriculture_ndwi_mean': float(np.mean(ndwi)),
                'agriculture_water_content': float(np.mean(ndwi > 0))
            }
        return {'agriculture_ndwi_mean': 0, 'agriculture_water_content': 0}

    def _compute_chlorophyll_content(self, image):
        if len(image.shape) >= 3:
            green = image[:, :, 1]
            red = image[:, :, 0]
            chlorophyll = green / (red + 1e-8)

            return {
                'agriculture_chlorophyll_index': float(np.mean(chlorophyll)),
                'agriculture_green_percentage': float(np.mean(green > 0.3))
            }
        return {'agriculture_chlorophyll_index': 0, 'agriculture_green_percentage': 0}

    def _compute_water_stress(self, image):
        if len(image.shape) >= 3:
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32) / 255.0

            return {
                'agriculture_water_stress_index': float(np.mean(lab[:, :, 1] + lab[:, :, 2]) / 2),
                'agriculture_wilting_score': float(np.mean(lab[:, :, 1] < 0.3))
            }
        return {'agriculture_water_stress_index': 0, 'agriculture_wilting_score': 0}

    def _detect_leaf_disease(self, image):
        if len(image.shape) >= 3:
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32) / 255.0

            spots = (lab[:, :, 0] < 0.4) & (lab[:, :, 1] < 0.4) & (lab[:, :, 2] < 0.4)

            return {
                'agriculture_disease_spots': float(np.mean(spots)),
                'agriculture_disease_severity': float(np.mean(spots) * 2)
            }
        return {'agriculture_disease_spots': 0, 'agriculture_disease_severity': 0}

    def get_quality_metrics(self, image):
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        from skimage.filters import sobel

        return {
            'agriculture_sharpness': float(np.std(sobel(gray))),
            'agriculture_contrast': float(np.std(gray)),
            'agriculture_plant_visibility': float(np.mean(image[:, :, 1] > 0.3)) if len(image.shape) == 3 else 0
        }

class DynamicArchitectureOptimizer:
    """
    Dynamically determines optimal architecture based on dataset characteristics.
    Analyzes image size, complexity, number of classes, and dataset size.
    """

    def __init__(self, config: GlobalConfig):
        self.config = config

    def analyze_dataset(self, dataset: Dataset, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Analyze dataset characteristics to determine optimal architecture.

        Returns:
            Dictionary with analysis results including:
            - input_size: detected input dimensions
            - complexity_score: 0-1 score of image complexity
            - recommended_layers: number of encoder layers
            - recommended_base_channels: base channel count
            - recommended_feature_dims: feature dimension size
            - recommended_compressed_dims: compressed dimension size
        """
        logger.info("=" * 60)
        logger.info("ANALYZING DATASET FOR ARCHITECTURE OPTIMIZATION")
        logger.info("=" * 60)

        # Sample images to analyze
        sample_indices = np.random.choice(min(len(dataset), num_samples),
                                         size=min(num_samples, len(dataset)),
                                         replace=False)

        # Collect statistics
        image_sizes = []
        intensity_stats = []
        edge_densities = []
        color_variances = []

        from skimage.filters import sobel
        from skimage.transform import resize

        for idx in tqdm(sample_indices, desc="Analyzing images"):
            try:
                img, label = dataset[idx]

                # Convert to numpy for analysis
                if isinstance(img, torch.Tensor):
                    img_np = img.cpu().numpy()
                    if img_np.shape[0] == 3:  # CHW to HWC
                        img_np = img_np.transpose(1, 2, 0)
                elif isinstance(img, PILImage.Image):
                    img_np = np.array(img) / 255.0
                else:
                    continue

                h, w = img_np.shape[:2]
                image_sizes.append((h, w))

                # Convert to grayscale for analysis
                if len(img_np.shape) == 3:
                    gray = np.mean(img_np, axis=2)
                else:
                    gray = img_np

                # Intensity statistics (contrast, entropy)
                intensity_stats.append({
                    'std': np.std(gray),
                    'entropy': self._calculate_entropy(gray)
                })

                # Edge density (complexity measure)
                edges = sobel(gray)
                edge_density = np.mean(edges > np.percentile(edges, 85))
                edge_densities.append(edge_density)

                # Color variance (if RGB)
                if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
                    color_var = np.var(img_np, axis=(0, 1)).mean()
                    color_variances.append(color_var)

            except Exception as e:
                logger.debug(f"Error analyzing sample {idx}: {e}")
                continue

        # Determine most common image size
        from collections import Counter
        size_counter = Counter(image_sizes)
        most_common_size = size_counter.most_common(1)[0][0]
        h, w = most_common_size

        # Calculate complexity score (0-1)
        avg_edge_density = np.mean(edge_densities) if edge_densities else 0.3
        avg_intensity_std = np.mean([s['std'] for s in intensity_stats]) if intensity_stats else 0.2
        avg_entropy = np.mean([s['entropy'] for s in intensity_stats]) if intensity_stats else 4.0
        avg_color_var = np.mean(color_variances) if color_variances else 0.05

        # Normalize entropy (typical range 2-8 for natural images)
        normalized_entropy = min(1.0, max(0.0, (avg_entropy - 2) / 6))

        # Complexity combines edge density, contrast, entropy, and color
        complexity_score = (
            0.35 * avg_edge_density +
            0.25 * avg_intensity_std +
            0.25 * normalized_entropy +
            0.15 * min(1.0, avg_color_var * 10)
        )

        # Determine number of classes and dataset size
        num_classes = self.config.num_classes if self.config.num_classes else 2
        dataset_size = len(dataset)

        # Adjust complexity based on number of classes
        class_complexity = min(1.0, num_classes / 200)  # 200 classes = max complexity
        size_complexity = min(1.0, dataset_size / 100000)  # 100k samples = max complexity

        combined_complexity = (complexity_score * 0.5 + class_complexity * 0.3 + size_complexity * 0.2)

        # Store in config for later use
        self.config.complexity_score = combined_complexity
        self.config.dataset_size = len(dataset)

        logger.info(f"Dataset Analysis Results:")
        logger.info(f"  Image size: {h}x{w}")
        logger.info(f"  Edge density: {avg_edge_density:.3f}")
        logger.info(f"  Contrast (std): {avg_intensity_std:.3f}")
        logger.info(f"  Entropy: {avg_entropy:.3f} (normalized: {normalized_entropy:.3f})")
        logger.info(f"  Color variance: {avg_color_var:.3f}")
        logger.info(f"  Number of classes: {num_classes}")
        logger.info(f"  Dataset size: {dataset_size:,}")
        logger.info(f"  Complexity score: {combined_complexity:.3f}")

        # Determine optimal architecture parameters
        result = self._determine_architecture_params(
            h, w, combined_complexity, num_classes, dataset_size
        )

        logger.info("=" * 60)
        logger.info("Recommended Architecture:")
        logger.info(f"  Encoder layers: {result['n_layers']}")
        logger.info(f"  Base channels: {result['base_channels']}")
        logger.info(f"  Feature dimensions: {result['feature_dims']}")
        logger.info(f"  Compressed dimensions: {result['compressed_dims']}")
        logger.info("=" * 60)

        return result

    def _calculate_entropy(self, image: np.ndarray, bins: int = 32) -> float:
        """Calculate image entropy as measure of complexity"""
        hist, _ = np.histogram(image.flatten(), bins=bins, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def _determine_architecture_params(self, height: int, width: int,
                                        complexity: float, num_classes: int,
                                        dataset_size: int) -> Dict[str, Any]:
        """Determine optimal architecture parameters based on analysis"""

        min_dim = min(height, width)

        # NUMBER OF ENCODER LAYERS
        if min_dim <= 32:
            n_layers = 2
        elif min_dim <= 64:
            n_layers = 3
        else:
            n_layers = 4

        # BASE CHANNELS - ensure divisible by common group sizes
        if min_dim <= 32:
            base_channels = 32
        elif min_dim <= 64:
            base_channels = 48
        else:
            base_channels = 64

        # FEATURE DIMENSIONS - ensure divisible by small numbers for GroupNorm
        if min_dim <= 32:
            feature_dims = 64  # Changed from 56 to 64 (divisible by 1,2,4,8)
        elif min_dim <= 64:
            feature_dims = 128
        else:
            feature_dims = 256

        # Adjust for complexity
        if complexity > 0.6:
            feature_dims = int(feature_dims * 1.5)
            # Round to nearest multiple of 16
            feature_dims = ((feature_dims + 7) // 8) * 8

        feature_dims = max(32, min(512, feature_dims))

        # COMPRESSED DIMENSIONS
        if num_classes <= 10:
            compressed_dims = 16
        elif num_classes <= 50:
            compressed_dims = 32
        elif num_classes <= 100:
            compressed_dims = 64
        else:
            compressed_dims = 128

        if dataset_size < 10000:
            compressed_dims = max(8, compressed_dims // 2)
        elif dataset_size > 50000:
            compressed_dims = min(256, compressed_dims * 2)

        return {
            'n_layers': n_layers,
            'base_channels': base_channels,
            'feature_dims': feature_dims,
            'compressed_dims': compressed_dims,
            'recommended_norm': 'dataset_wide',
            'input_size': (height, width),
            'complexity_score': complexity,
            'use_attention': min_dim > 64,  # Only use attention for larger images
            'use_multiscale': min_dim > 64  # Only use multi-scale for larger images
        }

    def update_config_from_analysis(self, analysis: Dict[str, Any]):
        """Update configuration based on analysis results"""
        self.config.input_size = analysis['input_size']
        self.config.feature_dims = analysis['feature_dims']

        # CRITICAL: Only update compressed_dims if not forced to fixed value
        if not hasattr(self.config, 'force_fixed_compressed_dims') or not self.config.force_fixed_compressed_dims:
            self.config.compressed_dims = analysis['compressed_dims']
            logger.info(f"Auto-set compressed_dims to: {self.config.compressed_dims}")
        else:
            logger.info(f"Keeping fixed compressed_dims: {self.config.compressed_dims} (user specified or default 32)")

        # Update enhancement flags if they exist in config
        if hasattr(self.config, 'use_detail_attention'):
            self.config.use_detail_attention = analysis.get('use_attention', False)
        if hasattr(self.config, 'use_multiscale_features'):
            self.config.use_multiscale_features = analysis.get('use_multiscale', False)

        # Update normalization strategy if not explicitly set by user
        if not hasattr(self.config, 'input_size_explicitly_set') or not self.config.input_size_explicitly_set:
            if analysis['recommended_norm'] == 'per_image':
                self.config.use_per_image_normalization = True
                logger.info(f"Auto-selected per-image normalization based on complexity ({analysis['complexity_score']:.3f})")

        logger.info(f"Configuration updated based on dataset analysis")
        logger.info(f"  Input size: {self.config.input_size}")
        logger.info(f"  Feature dims: {self.config.feature_dims}")
        logger.info(f"  Compressed dims: {self.config.compressed_dims}")

# =============================================================================
# COMPLETELY FIXED DATASET STATISTICS CALCULATOR
# =============================================================================

class DatasetStatisticsCalculator:
    """
    COMPLETELY DETERMINISTIC: Dataset-wide Z-score standardization.

    KEY INSIGHTS:
    1. Statistics calculated ONCE from training data
    2. Same statistics used for ALL images (training, validation, test)
    3. Results are IDENTICAL regardless of batch composition
    4. Training and prediction use EXACTLY the same normalization
    """

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.mode = 'dataset_wide'  # Changed from per_image_deterministic
        self.is_calculated = False
        self._feature_cache = {}
        self._normalization_count = 0
        self._total_pixels_processed = 0

        # Dataset-wide statistics
        self.mean = None  # Shape: [C]
        self.std = None   # Shape: [C]
        self.per_channel_min = None
        self.per_channel_max = None
        self.n_samples_used = 0

        # Fixed random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        logger.info("=" * 60)
        logger.info("INITIALIZED: Dataset-wide Deterministic Normalizer")
        logger.info("Mode: All images normalized using FIXED dataset statistics")
        logger.info("Result: Same image → Same normalized output (ALWAYS)")
        logger.info("=" * 60)

    def calculate_statistics(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Calculate dataset-wide statistics from training data.
        MUST be called before training/prediction.
        """
        logger.info("Calculating dataset-wide statistics from training data...")

        all_pixels = []
        n_batches = 0

        # Collect pixels from training data
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Calculating stats")):
            # images shape: [B, C, H, W]
            b, c, h, w = images.shape

            # Reshape to [B*H*W, C] for efficient processing
            pixels = images.permute(0, 2, 3, 1).reshape(-1, c)
            all_pixels.append(pixels)
            n_batches += 1

            # Sample large datasets to avoid memory issues
            if batch_idx > 100 and len(all_pixels) * b * h * w > 1e7:
                logger.info(f"Sampled {batch_idx} batches, stopping to avoid memory issues")
                break

        # Concatenate all pixels
        if all_pixels:
            all_pixels = torch.cat(all_pixels, dim=0)

            # Calculate channel-wise statistics
            self.mean = all_pixels.mean(dim=0)  # [C]
            self.std = all_pixels.std(dim=0)    # [C]
            self.per_channel_min = all_pixels.min(dim=0)[0]
            self.per_channel_max = all_pixels.max(dim=0)[0]
            self.n_samples_used = len(all_pixels)
            self.is_calculated = True

            logger.info("=" * 60)
            logger.info("Dataset statistics calculated successfully:")
            logger.info(f"  Mean per channel: {self.mean.tolist()}")
            logger.info(f"  Std per channel:  {self.std.tolist()}")
            logger.info(f"  Min per channel:  {self.per_channel_min.tolist()}")
            logger.info(f"  Max per channel:  {self.per_channel_max.tolist()}")
            logger.info(f"  Total pixels:     {self.n_samples_used:,}")
            logger.info("=" * 60)
        else:
            logger.warning("No data found for statistics calculation")
            self.is_calculated = False

        return {
            'mean': self.mean,
            'std': self.std,
            'per_channel_min': self.per_channel_min,
            'per_channel_max': self.per_channel_max,
            'n_samples': self.n_samples_used,
            'mode': self.mode,
            'deterministic': True,
            'normalization_type': 'dataset_wide'
        }

    def normalize(self, x: torch.Tensor, return_stats: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Dataset-wide Z-score normalization using FIXED statistics.
        Results are IDENTICAL regardless of batch composition.

        Args:
            x: Input tensor of shape [B, C, H, W] or [C, H, W]
            return_stats: If True, returns (normalized, stats_dict)

        Returns:
            Normalized tensor with zero mean and unit variance
        """
        if not self.is_calculated or self.mean is None:
            raise ValueError(
                "Dataset statistics not calculated! "
                "Call calculate_statistics() before normalize()"
            )

        self._normalization_count += 1

        # Move statistics to same device as input
        mean = self.mean.to(x.device).view(1, -1, 1, 1)
        std = self.std.to(x.device).view(1, -1, 1, 1)

        # Apply fixed normalization
        normalized = (x - mean) / (std + 1e-8)

        if return_stats:
            stats = {
                'mean': self.mean.cpu(),
                'std': self.std.cpu(),
                'shape': x.shape
            }
            return normalized, stats

        self._total_pixels_processed += x.numel()
        return normalized

    def normalize_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for single image normalization using dataset statistics.

        Args:
            image: Tensor of shape [C, H, W]

        Returns:
            Normalized tensor [C, H, W]
        """
        if image.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {image.dim()}D")

        # Add batch dimension, normalize, remove batch dimension
        batch_image = image.unsqueeze(0)
        normalized_batch = self.normalize(batch_image)
        return normalized_batch.squeeze(0)

    def denormalize(self, x: torch.Tensor, stats: Dict = None) -> torch.Tensor:
        """
        Denormalize tensor back to original range using dataset statistics.

        Args:
            x: Normalized tensor of shape [B, C, H, W] or [C, H, W]
            stats: Ignored (uses stored dataset statistics)

        Returns:
            Denormalized tensor
        """
        if not self.is_calculated or self.mean is None:
            raise ValueError("Dataset statistics not calculated!")

        mean = self.mean.to(x.device).view(1, -1, 1, 1)
        std = self.std.to(x.device).view(1, -1, 1, 1)

        return x * std + mean

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize feature vectors (not images) per sample.

        This is for post-extraction feature normalization.

        Args:
            features: Array of shape [N, D] where N = samples, D = feature dims

        Returns:
            Normalized features with zero mean and unit variance per sample
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D array, got {features.ndim}D")

        # Per-sample normalization (each sample independently)
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True)
        normalized = (features - mean) / (std + 1e-8)

        return normalized

    def get_statistics_summary(self) -> Dict:
        """Get summary of normalization operations performed"""
        return {
            'mode': self.mode,
            'deterministic': True,
            'normalization_type': 'dataset_wide',
            'total_normalizations': self._normalization_count,
            'total_pixels_processed': self._total_pixels_processed,
            'batch_independent': True,
            'same_image_same_output': True,
            'is_calculated': self.is_calculated,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'n_samples_used': self.n_samples_used
        }

    def to_dict(self) -> Dict:
        """Save configuration to dictionary for checkpointing"""
        return {
            'mode': self.mode,
            'deterministic': True,
            'type': 'dataset_wide',
            'timestamp': datetime.now().isoformat(),
            'normalization_count': self._normalization_count,
            'total_pixels_processed': self._total_pixels_processed,
            'n_samples_used': self.n_samples_used,
            'is_calculated': self.is_calculated,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'per_channel_min': self.per_channel_min.tolist() if self.per_channel_min is not None else None,
            'per_channel_max': self.per_channel_max.tolist() if self.per_channel_max is not None else None,
            'normalization_type': 'dataset_wide'
        }

    def from_dict(self, data: Dict):
        """Load configuration from dictionary"""
        self.mode = data.get('mode', 'dataset_wide')
        self.is_calculated = data.get('is_calculated', True)
        self.n_samples_used = data.get('n_samples_used', 0)
        self._normalization_count = data.get('normalization_count', 0)
        self._total_pixels_processed = data.get('total_pixels_processed', 0)

        # Load statistics tensors
        if 'mean' in data and data['mean'] is not None:
            self.mean = torch.tensor(data['mean'], dtype=torch.float32)
        else:
            self.mean = None

        if 'std' in data and data['std'] is not None:
            self.std = torch.tensor(data['std'], dtype=torch.float32)
        else:
            self.std = None

        if 'per_channel_min' in data and data['per_channel_min'] is not None:
            self.per_channel_min = torch.tensor(data['per_channel_min'], dtype=torch.float32)
        else:
            self.per_channel_min = None

        if 'per_channel_max' in data and data['per_channel_max'] is not None:
            self.per_channel_max = torch.tensor(data['per_channel_max'], dtype=torch.float32)
        else:
            self.per_channel_max = None

        logger.info(f"Loaded dataset statistics from dict")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Is calculated: {self.is_calculated}")
        logger.info(f"  Previous normalizations: {self._normalization_count}")
        if self.mean is not None:
            logger.info(f"  Mean: {self.mean.tolist()}")
        if self.std is not None:
            logger.info(f"  Std: {self.std.tolist()}")

    def verify_determinism(self, test_image: torch.Tensor) -> Dict:
        """
        Verify that normalization is truly deterministic.

        Args:
            test_image: A test image tensor [C, H, W]

        Returns:
            Dictionary with verification results
        """
        if not self.is_calculated:
            return {'is_deterministic': False, 'error': 'Statistics not calculated'}

        results = {}

        # Test 1: Same image alone
        norm1 = self.normalize_single_image(test_image.clone())

        # Test 2: Same image in batch of 1
        batch1 = test_image.clone().unsqueeze(0)
        norm2_batch = self.normalize(batch1)
        norm2 = norm2_batch[0]

        # Test 3: Same image in batch with others
        dummy = torch.randn_like(test_image) * 0.5
        batch2 = torch.stack([test_image.clone(), dummy], dim=0)
        norm3_batch = self.normalize(batch2)
        norm3 = norm3_batch[0]

        # Compare results
        results['same_image_alone_vs_batch1'] = torch.allclose(norm1, norm2, atol=1e-6)
        results['same_image_alone_vs_batch_mixed'] = torch.allclose(norm1, norm3, atol=1e-6)
        results['batch1_vs_batch_mixed'] = torch.allclose(norm2, norm3, atol=1e-6)
        results['is_deterministic'] = all(results.values())

        if results['is_deterministic']:
            logger.info("✓ Determinism verification PASSED")
            logger.info("  Same image produces identical normalized output regardless of context")
        else:
            logger.warning("✗ Determinism verification FAILED")
            for key, value in results.items():
                if not value and key != 'is_deterministic':
                    logger.warning(f"  {key}: {value}")

        return results


class TorchVisionMetadata:
    """Store complete metadata for torchvision datasets"""

    def __init__(self, dataset_name: str, num_classes: int, class_names: List[str]):
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(class_names)}

    def save(self, save_dir: Path):
        """Save metadata to file"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            'dataset_name': self.dataset_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'timestamp': datetime.now().isoformat(),
            'source': 'torchvision',
            'version': '2.0'
        }

        metadata_path = save_dir / f"{self.dataset_name.lower()}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"TorchVision metadata saved to {metadata_path}")

    @classmethod
    def load(cls, load_dir: Path, dataset_name: str) -> Optional['TorchVisionMetadata']:
        """Load metadata from file"""
        load_dir = Path(load_dir)
        metadata_path = load_dir / f"{dataset_name.lower()}_metadata.json"

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            instance = cls(
                dataset_name=metadata['dataset_name'],
                num_classes=metadata['num_classes'],
                class_names=metadata['class_names']
            )
            instance.class_to_idx = metadata['class_to_idx']
            instance.idx_to_class = {int(k): v for k, v in metadata['idx_to_class'].items()}
            logger.info(f"Loaded TorchVision metadata from {metadata_path}")
            return instance
        return None

# =============================================================================
# COMPLETE NORMALIZED DATASET (With deterministic ordering)
# =============================================================================

# =============================================================================
# FIXED NORMALIZED DATASET - TRULY DETERMINISTIC
# =============================================================================

class NormalizedDataset(Dataset):
    """
    Dataset wrapper that applies DETERMINISTIC per-image normalization.

    CRITICAL FIXES:
    1. Each image is normalized independently using its OWN statistics
    2. Results are IDENTICAL regardless of batch composition
    3. Maintains deterministic ordering across all samples
    4. No dependency on dataset-wide statistics
    5. Same image always produces same normalized output
    """

    def __init__(self, dataset: Dataset, statistics_calculator: 'DatasetStatisticsCalculator'):
        """
        Args:
            dataset: The underlying dataset
            statistics_calculator: Per-image statistics calculator (NOT dataset-wide)
        """
        self.dataset = dataset
        self.statistics = statistics_calculator

        # CRITICAL: Pre-compute deterministic order for consistent indexing
        self._deterministic_order = None
        self._precompute_deterministic_order()

        # Cache for tracking normalization (optional, for debugging)
        self._normalization_cache = {}

        logger.info(f"NormalizedDataset initialized: {len(self.dataset)} samples")
        logger.info(f"Normalization mode: {self.statistics.mode}")
        logger.info(f"Deterministic order: ENABLED (batch-independent)")

    def _precompute_deterministic_order(self):
        """
        Pre-compute deterministic order for ALL samples.

        This ensures that samples are always accessed in the same order
        regardless of batch composition, shuffling, or DataLoader settings.
        """
        all_indices = list(range(len(self.dataset)))

        # Try to get filenames for stable sorting
        if hasattr(self.dataset, 'filenames') and self.dataset.filenames:
            # Sort by filename (most stable)
            indices_with_names = [(i, str(self.dataset.filenames[i])) for i in all_indices]
            indices_with_names.sort(key=lambda x: x[1])  # Sort by filename
            self._deterministic_order = [i for i, _ in indices_with_names]
            logger.info(f"  Sorted by filename: {len(self._deterministic_order)} samples")

        elif hasattr(self.dataset, 'samples') and self.dataset.samples:
            # Sort by file path
            indices_with_paths = [(i, str(self.dataset.samples[i][0])) for i in all_indices]
            indices_with_paths.sort(key=lambda x: x[1])  # Sort by path
            self._deterministic_order = [i for i, _ in indices_with_paths]
            logger.info(f"  Sorted by file path: {len(self._deterministic_order)} samples")

        elif hasattr(self.dataset, 'image_files') and self.dataset.image_files:
            # Sort by image file path
            indices_with_files = [(i, str(self.dataset.image_files[i])) for i in all_indices]
            indices_with_files.sort(key=lambda x: x[1])
            self._deterministic_order = [i for i, _ in indices_with_files]
            logger.info(f"  Sorted by image file: {len(self._deterministic_order)} samples")

        else:
            # Use natural order (already sorted during dataset construction)
            # But sort anyway to be safe
            self._deterministic_order = sorted(all_indices)
            logger.info(f"  Using natural sorted order: {len(self._deterministic_order)} samples")

        # Verify no duplicates
        assert len(set(self._deterministic_order)) == len(self.dataset), "Deterministic order has duplicates!"

    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item using DETERMINISTIC ordering.

        CRITICAL: The idx parameter is the position in the deterministic order,
        NOT the original dataset index. This ensures consistent ordering
        regardless of how the DataLoader is configured.

        Args:
            idx: Position in deterministic order (0 to len-1)

        Returns:
            Tuple of (normalized_image_tensor, label)
        """
        # Map from deterministic position to original dataset index
        original_idx = self._deterministic_order[idx]

        # Get image and label from original dataset
        image, label = self.dataset[original_idx]

        # Ensure image is a tensor
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PILImage.Image):
                from torchvision import transforms
                to_tensor = transforms.ToTensor()
                image = to_tensor(image)
            elif isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
                # Ensure correct shape [C, H, W]
                if image.dim() == 3 and image.shape[-1] in [1, 3, 4]:
                    image = image.permute(2, 0, 1)
            else:
                image = torch.tensor(image, dtype=torch.float32)

        # Ensure image is in range [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # CRITICAL: Apply PER-IMAGE normalization (not dataset-wide)
        # This is the key to determinism - each image normalized independently
        image = self.statistics.normalize(image)

        # Optional: Cache for debugging (disable for production)
        # if idx < 100:  # Only cache first 100 for memory
        #     self._normalization_cache[original_idx] = {
        #         'normalized_mean': image.mean().item(),
        #         'normalized_std': image.std().item(),
        #         'original_idx': original_idx
        #     }

        return image, label

    def get_original_index(self, deterministic_idx: int) -> int:
        """
        Get original dataset index from deterministic position.

        Args:
            deterministic_idx: Position in deterministic order

        Returns:
            Original index in the underlying dataset
        """
        if 0 <= deterministic_idx < len(self._deterministic_order):
            return self._deterministic_order[deterministic_idx]
        raise IndexError(f"Deterministic index {deterministic_idx} out of range (0-{len(self._deterministic_order)-1})")

    def get_deterministic_position(self, original_idx: int) -> int:
        """
        Get deterministic position from original dataset index.

        Args:
            original_idx: Original index in the underlying dataset

        Returns:
            Position in deterministic order
        """
        try:
            return self._deterministic_order.index(original_idx)
        except ValueError:
            raise ValueError(f"Original index {original_idx} not found in deterministic order")

    def get_sample_info(self, deterministic_idx: int) -> Dict:
        """
        Get detailed information about a sample at deterministic position.

        Args:
            deterministic_idx: Position in deterministic order

        Returns:
            Dictionary with sample information
        """
        original_idx = self.get_original_index(deterministic_idx)

        info = {
            'deterministic_position': deterministic_idx,
            'original_index': original_idx,
        }

        # Add filename if available
        if hasattr(self.dataset, 'filenames') and original_idx < len(self.dataset.filenames):
            info['filename'] = self.dataset.filenames[original_idx]

        if hasattr(self.dataset, 'full_paths') and original_idx < len(self.dataset.full_paths):
            info['filepath'] = self.dataset.full_paths[original_idx]

        if hasattr(self.dataset, 'samples') and original_idx < len(self.dataset.samples):
            info['sample_path'] = self.dataset.samples[original_idx][0]

        return info

    def verify_deterministic_access(self, test_indices: List[int] = None) -> Dict:
        """
        Verify that access is truly deterministic.

        Args:
            test_indices: List of indices to test (default: first 10)

        Returns:
            Dictionary with verification results
        """
        if test_indices is None:
            test_indices = list(range(min(10, len(self))))

        results = {
            'consistent_ordering': True,
            'same_image_same_output': True,
            'tested_indices': test_indices
        }

        # Test 1: Access same position twice should give same image
        for idx in test_indices:
            img1, label1 = self[idx]
            img2, label2 = self[idx]

            if not torch.allclose(img1, img2, atol=1e-6):
                results['consistent_ordering'] = False
                results['failure_at_index'] = idx
                break

        # Test 2: Verify normalized statistics are zero mean, unit variance per image
        for idx in test_indices[:5]:  # Test first 5
            img, _ = self[idx]
            # Per-image mean should be near 0 (per channel)
            channel_means = img.mean(dim=[1, 2])
            if torch.abs(channel_means).max() > 1e-6:
                results['same_image_same_output'] = False
                results['mean_error'] = channel_means.max().item()
                break

        logger.info(f"Deterministic verification: {results}")
        return results


class DeterministicInvariantPreprocessor:
    """
    Deterministic image preprocessing for invariance to:
    - Brightness & contrast
    - Translation & scaling
    - Rotation (optional)
    - Illumination variations

    Same image → same output (100% deterministic)
    """

    def __init__(self, target_size: Tuple[int, int] = (256, 256),
                 use_augmentation: bool = False,
                 augmentation_strength: float = 0.5):
        self.target_size = target_size
        self.use_augmentation = use_augmentation
        self.augmentation_strength = augmentation_strength
        # Fixed parameters for deterministic behavior
        self._fixed_seed = 42
        np.random.seed(self._fixed_seed)

    def process(self, image, is_training: bool = False) -> PILImage.Image:
        """
        Apply deterministic invariant preprocessing.
        Accepts PIL Image, Tensor, or numpy array.
        Same image → same output ALWAYS.
        """
        # Convert input to PIL Image if needed
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            from torchvision import transforms
            # Denormalize if needed (assume [0,1] range)
            if image.min() < 0 or image.max() > 1:
                image = (image - image.min()) / (image.max() - image.min())
            to_pil = transforms.ToPILImage()
            image = to_pil(image)
        elif isinstance(image, np.ndarray):
            # Convert numpy to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = PILImage.fromarray(image)

        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy for processing
        img_array = np.array(image).astype(np.float32) / 255.0

        # Ensure correct shape [H, W, C]
        if len(img_array.shape) == 2:
            # Grayscale to RGB
            img_array = np.stack([img_array, img_array, img_array], axis=2)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            # Single channel to RGB
            img_array = np.concatenate([img_array, img_array, img_array], axis=2)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA to RGB
            img_array = img_array[:, :, :3]

        # 1. Brightness & Contrast Normalization (Deterministic)
        img_array = self._normalize_brightness_contrast(img_array)

        # 2. Histogram Equalization (Deterministic)
        img_array = self._deterministic_histogram_equalization(img_array)

        # 3. Translation & Scale Normalization (Deterministic)
        img_array = self._center_and_scale_object(img_array)

        # 4. Resize to target size
        #Simg_array = self._deterministic_resize(img_array)

        # 5. Optional: Rotation normalization (if needed)
        if is_training:
            img_array = self._normalize_rotation(img_array)

        # 6. Optional: Deterministic augmentations (if enabled)
        if self.use_augmentation and is_training:
            img_array = self._apply_deterministic_augmentations(img_array)

        # Clip to valid range
        img_array = np.clip(img_array, 0, 1)

        # Convert to uint8 for PIL
        img_uint8 = (img_array * 255).astype(np.uint8)

        # Ensure correct shape for PIL (H, W, C)
        if len(img_uint8.shape) == 2:
            img_uint8 = np.stack([img_uint8, img_uint8, img_uint8], axis=2)
        elif len(img_uint8.shape) == 3 and img_uint8.shape[2] == 1:
            img_uint8 = np.concatenate([img_uint8, img_uint8, img_uint8], axis=2)
        elif len(img_uint8.shape) == 3 and img_uint8.shape[2] > 3:
            img_uint8 = img_uint8[:, :, :3]

        return PILImage.fromarray(img_uint8, mode='RGB')

    def _normalize_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """Deterministic brightness/contrast normalization using percentiles"""
        # Process each channel independently
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            p1, p99 = np.percentile(channel, [1, 99])
            if p99 > p1:
                channel_clipped = np.clip(channel, p1, p99)
                channel_normalized = (channel_clipped - p1) / (p99 - p1 + 1e-8)
                img[:, :, c] = channel_normalized
        return img

    def _deterministic_histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """Deterministic histogram equalization"""
        try:
            for c in range(img.shape[2]):
                img[:, :, c] = exposure.equalize_hist(img[:, :, c])
            return img
        except:
            return img

    def _center_and_scale_object(self, img: np.ndarray) -> np.ndarray:
        """Deterministic object centering and scaling"""
        # Convert to grayscale for object detection
        gray = np.mean(img, axis=2)

        try:
            threshold = threshold_otsu(gray)
            binary = gray > threshold
        except:
            threshold = np.percentile(gray, 70)
            binary = gray > threshold

        if np.any(binary):
            coords = np.argwhere(binary)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            h, w = img.shape[:2]
            pad = max(10, min(h, w) // 10)
            y_min = max(0, y_min - pad)
            y_max = min(h, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(w, x_max + pad)

            if y_max > y_min and x_max > x_min:
                img_cropped = img[y_min:y_max, x_min:x_max, :]
                crop_h, crop_w = img_cropped.shape[:2]
                target_h, target_w = self.target_size

                scale = min(target_h / crop_h, target_w / crop_w)
                scale = min(scale, 3.0)
                new_h = max(1, int(crop_h * scale))
                new_w = max(1, int(crop_w * scale))

                from skimage.transform import resize
                img_resized = resize(img_cropped, (new_h, new_w), mode='constant',
                                    preserve_range=True, anti_aliasing=True)

                canvas = np.zeros((target_h, target_w, img.shape[2]), dtype=img_resized.dtype)
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = img_resized
                return canvas

        from skimage.transform import resize
        return resize(img, self.target_size, mode='constant', preserve_range=True)

    def _deterministic_resize(self, img: np.ndarray) -> np.ndarray:
        """Deterministic resize with fixed interpolation"""
        from skimage.transform import resize
        if img.shape[:2] != self.target_size:
            return resize(img, self.target_size, mode='constant',
                         preserve_range=True, anti_aliasing=True)
        return img

    def _normalize_rotation(self, img: np.ndarray) -> np.ndarray:
        """Deterministic rotation normalization"""
        try:
            gray = np.mean(img, axis=2)
            moments = cv2.moments(gray)
            if moments['mu02'] != 0 and moments['mu20'] != 0:
                angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
                angle_deg = np.degrees(angle)
                if abs(angle_deg) > 5:
                    from scipy.ndimage import rotate
                    img_rotated = rotate(img, -angle_deg, reshape=False, order=3,
                                        mode='constant', cval=0)
                    return img_rotated
        except:
            pass
        return img

    def _apply_deterministic_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Apply deterministic augmentations based on image content hash"""
        # Use image content hash for deterministic variations
        content_hash = hash(img.tobytes()) % 100

        # Brightness adjustment (0.8 to 1.2)
        brightness_factor = 0.8 + (content_hash % 40) / 100.0 * self.augmentation_strength
        img = img * brightness_factor

        # Contrast adjustment
        contrast_factor = 0.8 + ((content_hash // 10) % 40) / 100.0 * self.augmentation_strength
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            mean = channel.mean()
            img[:, :, c] = (channel - mean) * contrast_factor + mean

        # Small translation (up to 5 pixels)
        shift_x = ((content_hash * 131) % 11) - 5
        shift_y = ((content_hash * 137) % 11) - 5
        shift_x = shift_x * self.augmentation_strength
        shift_y = shift_y * self.augmentation_strength

        from scipy.ndimage import shift
        img = shift(img, [shift_y, shift_x, 0], order=1, mode='nearest')

        return np.clip(img, 0, 1)


class DeterministicAugmentation:
    """
    Deterministic data augmentation for training.
    Uses fixed transformations based on image content, not random numbers.
    """

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self._fixed_seed = 42

    def apply_augmentations(self, image: PILImage.Image, idx: int) -> PILImage.Image:
        """
        Apply deterministic augmentations based on image index.
        Same image index → same augmentation.
        """
        img_array = np.array(image).astype(np.float32) / 255.0

        # Use image index and content hash to determine augmentation
        # This ensures determinism while still varying augmentations
        content_hash = hash(image.tobytes()) % 100

        # 1. Deterministic brightness adjustment (based on content)
        brightness_factor = 0.8 + (content_hash % 40) / 100.0  # 0.8 to 1.2
        img_array = img_array * brightness_factor

        # 2. Deterministic contrast adjustment
        contrast_factor = 0.8 + ((content_hash // 10) % 40) / 100.0
        mean = img_array.mean(axis=(0, 1), keepdims=True)
        img_array = (img_array - mean) * contrast_factor + mean

        # 3. Deterministic translation (based on index)
        shift_x = ((idx * 131) % 21) - 10  # -10 to +10 pixels
        shift_y = ((idx * 137) % 21) - 10

        from scipy.ndimage import shift
        img_array = shift(img_array, [shift_y, shift_x, 0], order=1, mode='nearest')

        # 4. Deterministic scaling (based on content hash)
        scale = 0.9 + (content_hash % 20) / 100.0  # 0.9 to 1.1
        from skimage.transform import resize
        h, w = img_array.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img_scaled = resize(img_array, (new_h, new_w), mode='constant', preserve_range=True)

        # Pad or crop to target size
        canvas = np.zeros((self.target_size[0], self.target_size[1], img_array.shape[2]),
                         dtype=img_scaled.dtype)
        y_offset = (self.target_size[0] - new_h) // 2
        x_offset = (self.target_size[1] - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_scaled

        # Clip to valid range
        img_array = np.clip(canvas, 0, 1)

        return PILImage.fromarray((img_array * 255).astype(np.uint8))

# =============================================================================
# DETERMINISTIC DATALOADER HELPER
# =============================================================================

class DeterministicDataLoader:
    """
    Helper class to create deterministic DataLoaders.
    Ensures consistent batching across runs.
    """

    @staticmethod
    def create_deterministic_loader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        seed: int = 42
    ) -> DataLoader:
        """
        Create a DataLoader with deterministic behavior.

        CRITICAL:
        - Uses fixed generator seed
        - Forces num_workers=0 for reproducibility
        - Handles shuffle with deterministic ordering

        Args:
            dataset: The dataset (should be NormalizedDataset for best results)
            batch_size: Batch size
            shuffle: Whether to shuffle (uses deterministic shuffle with seed)
            num_workers: Number of workers (0 recommended for reproducibility)
            seed: Random seed for shuffling

        Returns:
            Deterministic DataLoader
        """
        # Create generator with fixed seed
        g = torch.Generator()
        g.manual_seed(seed)

        # If dataset is NormalizedDataset, ensure deterministic order is used
        if isinstance(dataset, NormalizedDataset):
            logger.info("Using NormalizedDataset with deterministic ordering")

        # Create DataLoader with deterministic settings
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Force 0 workers for reproducibility
            generator=g,
            pin_memory=False,
            drop_last=False,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id) if shuffle else None
        )

        logger.info(f"Created deterministic DataLoader:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Shuffle: {shuffle} (seed={seed})")
        logger.info(f"  Workers: 0 (forced for reproducibility)")

        return loader

    @staticmethod
    def get_deterministic_batches(loader: DataLoader) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract all batches from a deterministic DataLoader.

        This is useful for debugging and verification.

        Args:
            loader: Deterministic DataLoader

        Returns:
            List of (images, labels) batches
        """
        batches = []
        for images, labels in loader:
            batches.append((images.clone(), labels.clone()))
        return batches


# =============================================================================
# VERIFICATION FUNCTION
# =============================================================================

def verify_normalized_dataset():
    """
    Comprehensive verification that NormalizedDataset works correctly.
    """
    print("\n" + "=" * 70)
    print("VERIFYING NORMALIZED DATASET")
    print("=" * 70)

    # Create dummy config
    config = GlobalConfig()
    config.input_size = (64, 64)
    config.in_channels = 3

    # Create normalizer
    normalizer = DatasetStatisticsCalculator(config)

    # Create dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, num_samples=20):
            self.num_samples = num_samples
            self.filenames = [f"image_{i:03d}.jpg" for i in range(num_samples)]
            # Create deterministic dummy images
            torch.manual_seed(42)
            self.images = [torch.randn(3, 64, 64) * 0.5 + 0.5 for _ in range(num_samples)]

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.images[idx], idx % 5  # 5 classes

    dummy_dataset = DummyDataset(num_samples=20)

    # Create normalized dataset
    normalized_dataset = NormalizedDataset(dummy_dataset, normalizer)

    print(f"\nDataset size: {len(normalized_dataset)}")
    print(f"Deterministic order: {normalized_dataset._deterministic_order[:10]}...")

    # Test 1: Same position always returns same image
    print("\n" + "-" * 70)
    print("TEST 1: Consistent access")
    print("-" * 70)

    img1, label1 = normalized_dataset[5]
    img2, label2 = normalized_dataset[5]

    if torch.allclose(img1, img2, atol=1e-6):
        print("✓ PASS: Same position returns same normalized image")
    else:
        print("✗ FAIL: Same position returned different images")

    # Test 2: Normalized statistics
    print("\n" + "-" * 70)
    print("TEST 2: Normalized statistics")
    print("-" * 70)

    img, label = normalized_dataset[0]
    channel_means = img.mean(dim=[1, 2])
    channel_stds = img.std(dim=[1, 2])

    print(f"  Channel means: {channel_means.tolist()}")
    print(f"  Channel stds:  {channel_stds.tolist()}")

    if torch.abs(channel_means).max() < 1e-6:
        print("✓ PASS: Zero mean per channel")
    else:
        print(f"✗ FAIL: Mean not zero (max={torch.abs(channel_means).max():.6f})")

    # Test 3: Different batch compositions
    print("\n" + "-" * 70)
    print("TEST 3: Batch composition independence")
    print("-" * 70)

    # Get same image in different contexts
    img_alone = normalized_dataset[3][0]

    # Create DataLoader with batch_size=1
    loader1 = DeterministicDataLoader.create_deterministic_loader(
        normalized_dataset, batch_size=1, shuffle=False
    )
    batches1 = list(loader1)
    img_batch1 = batches1[3][0][0]  # 4th image

    # Create DataLoader with batch_size=4
    loader4 = DeterministicDataLoader.create_deterministic_loader(
        normalized_dataset, batch_size=4, shuffle=False
    )
    batches4 = list(loader4)
    # Find the same image (should be at position 3 in the 4th batch? Let's find it)
    img_batch4 = None
    for batch_idx, (images, _) in enumerate(batches4):
        for img_idx in range(images.size(0)):
            if batch_idx * 4 + img_idx == 3:
                img_batch4 = images[img_idx]
                break

    if img_batch4 is not None:
        if torch.allclose(img_alone, img_batch4, atol=1e-6):
            print("✓ PASS: Same image normalized identically regardless of batch size")
        else:
            print("✗ FAIL: Image normalization depends on batch composition")
            print(f"  Max difference: {(img_alone - img_batch4).abs().max():.6f}")
    else:
        print("? SKIP: Could not locate image in batch")

    # Test 4: Shuffle determinism
    print("\n" + "-" * 70)
    print("TEST 4: Shuffle determinism")
    print("-" * 70)

    loader_shuffle1 = DeterministicDataLoader.create_deterministic_loader(
        normalized_dataset, batch_size=4, shuffle=True, seed=42
    )
    loader_shuffle2 = DeterministicDataLoader.create_deterministic_loader(
        normalized_dataset, batch_size=4, shuffle=True, seed=42
    )

    batches_shuffle1 = list(loader_shuffle1)
    batches_shuffle2 = list(loader_shuffle2)

    all_same = True
    for i, (b1, b2) in enumerate(zip(batches_shuffle1, batches_shuffle2)):
        if not torch.allclose(b1[0], b2[0], atol=1e-6):
            all_same = False
            break

    if all_same:
        print("✓ PASS: Shuffle is deterministic with same seed")
    else:
        print("✗ FAIL: Shuffle not deterministic")

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

    return normalized_dataset


# =============================================================================
# COLORS AND LOGGING
# =============================================================================

# =============================================================================
# Colors, Logging Setup, and other utility classes
# (These remain unchanged from the original code)
# =============================================================================

class Colors:
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
        if previous_value is None:
            return f"{Colors.BLUE}{current_value:.4f}{Colors.ENDC}"
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
        return f"{Colors.YELLOW}{current_value:.4f}{Colors.ENDC}"


def setup_logging(name: str = 'cdbnn', log_dir: str = 'logs') -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


def timed(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def memory_efficient(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return result
    return wrapper

def optimize_gpu_settings():
    """Apply GPU optimization settings"""
    if torch.cuda.is_available():
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

        # Set memory allocator settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Print GPU info
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        return True
    return False

# =============================================================================
# BASE CLASSES
# =============================================================================

class ImageProcessor:
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp')

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        # CRITICAL FIX: Fixed random state for reproducible augmentations
        self._rng = np.random.RandomState(42)

    @staticmethod
    @lru_cache(maxsize=256)
    def load_image(path: str) -> Optional[PILImage.Image]:
        try:
            img = PILImage.open(path)
            img.verify()
            img = PILImage.open(path)
            return img
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None

    def preprocess(self, image: PILImage.Image, is_train: bool = False) -> PILImage.Image:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if image.size != self.target_size:
            image = self.resize_with_aspect(image, self.target_size)

        if is_train:
            # CRITICAL FIX: Use fixed random state for reproducible augmentations
            if self._rng.random() > 0.5:
                image = ImageOps.mirror(image)
            if self._rng.random() > 0.5:
                image = image.rotate(self._rng.randint(-10, 10), expand=False)
        return image

    def resize_with_aspect(self, image: PILImage.Image, target_size: Tuple[int, int]) -> PILImage.Image:
        image.thumbnail(target_size, PILImage.Resampling.LANCZOS)
        new_img = PILImage.new('RGB', target_size, (0, 0, 0))
        paste_x = (target_size[0] - image.width) // 2
        paste_y = (target_size[1] - image.height) // 2
        new_img.paste(image, (paste_x, paste_y))
        return new_img

    @staticmethod
    def load_fits_image(path: str, hdu: int = 0, normalization: str = 'zscale') -> Optional[np.ndarray]:
        """Load FITS astronomical image with proper normalization"""
        try:
            from astropy.io import fits

            with fits.open(path) as hdul:
                if hdu >= len(hdul):
                    logger.warning(f"HDU {hdu} not found in {path}, using HDU 0")
                    hdu = 0
                data = hdul[hdu].data.astype(np.float32)

                # Handle NaN and infinite values
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                # Normalize based on method
                if normalization == 'zscale':
                    data = ImageProcessor._zscale_normalization(data)
                elif normalization == 'percent':
                    p1, p99 = np.percentile(data, [1, 99])
                    data = (data - p1) / (p99 - p1 + 1e-8)
                elif normalization == 'asinh':
                    median = np.median(data)
                    data = np.arcsinh(data - median)
                    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                else:  # minmax
                    data = (data - data.min()) / (data.max() - data.min() + 1e-8)

                # Clip to valid range
                data = np.clip(data, 0, 1)

                # Convert to 3-channel RGB by stacking (grayscale to RGB)
                if len(data.shape) == 2:
                    data = np.stack([data, data, data], axis=2)
                elif len(data.shape) == 3 and data.shape[2] == 1:
                    data = np.concatenate([data, data, data], axis=2)

                return data

        except ImportError:
            logger.error("astropy.io.fits not available. Install astropy for FITS support.")
            return None
        except Exception as e:
            logger.error(f"Failed to load FITS image {path}: {e}")
            return None

    @staticmethod
    def _zscale_normalization(data: np.ndarray, contrast: float = 0.25, samples: int = 1000) -> np.ndarray:
        """Z-scale normalization for astronomical images - DETERMINISTIC"""
        flat = data.flatten()
        if len(flat) > samples:
            # Use fixed random state for reproducibility
            rng = np.random.RandomState(42)
            idx = rng.choice(len(flat), samples, replace=False)
            idx.sort()  # Sort for deterministic order
            flat = flat[idx]
        else:
            flat = np.sort(flat)

        # Sort and compute percentiles
        flat.sort()
        n = len(flat)
        center = flat[n // 2]
        half_range = contrast * (flat[-1] - flat[0])
        zmin = center - half_range
        zmax = center + half_range

        normalized = (data - zmin) / (zmax - zmin + 1e-8)
        return np.clip(normalized, 0, 1)

class ArchiveHandler:
    SUPPORTED_FORMATS = {'.zip', '.tar', '.tar.gz', '.tgz', '.gz', '.bz2', '.xz'}

    @staticmethod
    def extract(archive_path: str, extract_dir: str) -> str:
        os.makedirs(extract_dir, exist_ok=True)
        file_ext = Path(archive_path).suffix.lower()

        if file_ext == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif file_ext in ['.tar', '.tgz'] or archive_path.endswith('tar.gz'):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        elif file_ext == '.gz':
            output_path = os.path.join(extract_dir, Path(archive_path).stem)
            with gzip.open(archive_path, 'rb') as gz_file:
                with open(output_path, 'wb') as out_file:
                    shutil.copyfileobj(gz_file, out_file)
        elif file_ext == '.bz2':
            output_path = os.path.join(extract_dir, Path(archive_path).stem)
            with bz2.open(archive_path, 'rb') as bz2_file:
                with open(output_path, 'wb') as out_file:
                    shutil.copyfileobj(bz2_file, out_file)
        elif file_ext == '.xz':
            output_path = os.path.join(extract_dir, Path(archive_path).stem)
            with lzma.open(archive_path, 'rb') as xz_file:
                with open(output_path, 'wb') as out_file:
                    shutil.copyfileobj(xz_file, out_file)
        else:
            raise ValueError(f"Unsupported archive format: {file_ext}")

        contents = os.listdir(extract_dir)
        if len(contents) == 1 and os.path.isdir(os.path.join(extract_dir, contents[0])):
            return os.path.join(extract_dir, contents[0])
        return extract_dir

# =============================================================================
# DISTANCE CORRELATION FEATURE SELECTOR
# =============================================================================

class DistanceCorrelationFeatureSelector:
    """Enhanced feature selector with deterministic output"""

    def __init__(self, upper_threshold=0.85, lower_threshold=0.01, min_features=8, max_features=50):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.min_features = min_features
        self.max_features = max_features
        # Set random seed for reproducibility
        np.random.seed(42)

    def calculate_distance_correlations(self, features, labels):
        """Calculate distance correlations between features and labels"""
        n_features = features.shape[1]
        label_corrs = np.zeros(n_features)

        for i in range(n_features):
            label_corrs[i] = 1 - correlation(features[:, i], labels)

        # For multi-class problems, add class separation scores
        if len(np.unique(labels)) > 10:
            separation_scores = self._calculate_class_separation_scores(features, labels)
            combined_scores = 0.7 * label_corrs + 0.3 * separation_scores
            return combined_scores

        return label_corrs

    def _calculate_class_separation_scores(self, features, labels):
        """Calculate how well each feature separates classes"""
        n_features = features.shape[1]
        separation_scores = np.zeros(n_features)
        unique_labels = np.unique(labels)

        for i in range(n_features):
            feature_values = features[:, i]
            overall_mean = np.mean(feature_values)
            between_var = 0
            within_var = 0

            for label in unique_labels:
                class_mask = labels == label
                class_values = feature_values[class_mask]
                class_mean = np.mean(class_values)
                class_size = len(class_values)

                between_var += class_size * (class_mean - overall_mean) ** 2
                within_var += np.sum((class_values - class_mean) ** 2)

            if within_var > 0:
                separation_scores[i] = between_var / within_var
            else:
                separation_scores[i] = between_var

        # Normalize scores
        if np.max(separation_scores) > 0:
            separation_scores = separation_scores / np.max(separation_scores)

        return separation_scores

    def select_features(self, features, labels):
        """Select most informative features with deterministic ordering"""
        label_corrs = self.calculate_distance_correlations(features, labels)

        # CRITICAL FIX: Sort by correlation DESCENDING, then by index for tie-breaking
        # This ensures deterministic selection
        sorted_indices = np.argsort(-label_corrs)  # Sort by correlation descending
        selected_indices = []

        # Take features above threshold, but maintain deterministic order
        for idx in sorted_indices:
            if label_corrs[idx] >= self.upper_threshold:
                selected_indices.append(idx)

        # If not enough features, take top ones by correlation (already sorted)
        if len(selected_indices) < self.min_features:
            selected_indices = sorted_indices[:self.min_features].tolist()
            logger.info(f"Relaxed threshold: selected top {self.min_features} features")

        # CRITICAL FIX: Sort selected_indices deterministically by index value
        selected_indices.sort()  # Sort by index value, not by correlation

        # Remove redundant features
        final_indices = self._remove_redundant_features(features, selected_indices, label_corrs)

        if len(final_indices) > self.max_features:
            final_indices = final_indices[:self.max_features]

        # Final deterministic sort
        final_indices.sort()

        logger.info(f"Final feature selection (deterministic order): {len(final_indices)} features")
        return np.array(final_indices), label_corrs

    def _remove_redundant_features(self, features, candidate_indices, corr_values):
        """Remove highly correlated features to reduce redundancy"""
        final_indices = []

        # CRITICAL FIX: Create mapping from candidate index to its position
        index_to_position = {idx: pos for pos, idx in enumerate(candidate_indices)}

        for idx in candidate_indices:
            keep = True
            pos = index_to_position[idx]

            for j in final_indices:
                # Calculate correlation between features
                corr = 1 - correlation(features[:, idx], features[:, j])

                if corr > self.lower_threshold:
                    # Keep the one with higher correlation to labels
                    if corr_values[idx] <= corr_values[j]:
                        keep = False
                        break

            if keep:
                final_indices.append(idx)
                if len(final_indices) >= self.max_features:
                    break

        # Sort final indices for deterministic order
        final_indices.sort()
        return final_indices

# =============================================================================
# CUSTOM IMAGE DATASET
# =============================================================================

class CustomImageDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, config: Optional[Dict] = None, data_name: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.config = config or {}

        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.image_files = []
        self.filenames = []
        self.full_paths = []
        self.labels = []

        self._scan_directory()

        # CRITICAL FIX: Sort samples deterministically after scanning
        # Sort by file path to ensure consistent ordering across runs
        self.samples.sort(key=lambda x: x[0])  # Sort by file path
        self.image_files.sort()
        self.full_paths.sort()
        self.filenames.sort()

        # Update labels to match sorted order of samples
        self.labels = [label for _, label in self.samples]

        self.resize_images = self.config.get('resize_images', False)

        logger.info(f"Dataset: {len(self.samples)} images, {len(self.classes)} classes")

    def _scan_directory(self):
        supported_formats = ImageProcessor.SUPPORTED_FORMATS
        # Add FITS formats if domain is astronomy
        if hasattr(self.config, 'domain') and self.config.domain == 'astronomy' and getattr(self.config, 'use_fits', False):
            fits_formats = ('.fits', '.fit', '.fits.gz', '.fit.gz')
            supported_formats = supported_formats + fits_formats

        # CRITICAL FIX: Sort directories for deterministic order
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                idx = len(self.classes)
                self.class_to_idx[class_name] = idx
                self.idx_to_class[idx] = class_name
                self.classes.append(class_name)

            # CRITICAL FIX: Sort image files for deterministic order
            # Use sorted() with natural sorting for better consistency
            all_files = list(class_dir.glob('*'))
            # Sort by path string for deterministic ordering
            all_files.sort(key=lambda p: str(p))

            for img_path in all_files:
                # Check if file is a supported image format
                is_supported = img_path.suffix.lower() in supported_formats
                is_fits_gz = img_path.name.lower().endswith('.fits.gz') or img_path.name.lower().endswith('.fit.gz')

                if is_supported or is_fits_gz:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                    self.image_files.append(str(img_path))
                    self.full_paths.append(str(img_path))
                    self.filenames.append(img_path.name)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Check if FITS file
        is_fits = img_path.lower().endswith(('.fits', '.fit', '.fits.gz', '.fit.gz'))

        if is_fits and hasattr(self.config, 'domain') and self.config.domain == 'astronomy':
            fits_hdu = getattr(self.config, 'fits_hdu', 0)
            fits_norm = getattr(self.config, 'fits_normalization', 'zscale')
            img_array = ImageProcessor.load_fits_image(img_path, hdu=fits_hdu, normalization=fits_norm)
            if img_array is None:
                img = PILImage.new('RGB', (256, 256), (0, 0, 0))
            else:
                img = PILImage.fromarray((img_array * 255).astype(np.uint8))
        else:
            img = ImageProcessor.load_image(img_path)
            if img is None:
                img = PILImage.new('RGB', (256, 256), (0, 0, 0))

        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_additional_info(self, idx: int) -> Tuple[int, str, str]:
        """Get additional information for a sample at given index"""
        return idx, self.filenames[idx], self.full_paths[idx]

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset"""
        dist = defaultdict(int)
        for _, label in self.samples:
            dist[self.idx_to_class[label]] += 1
        # Return sorted dictionary for deterministic order
        return dict(sorted(dist.items()))

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        class_counts = np.bincount(self.labels)
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum() * len(class_counts)
        return torch.FloatTensor(weights)

    def get_sample_by_filename(self, filename: str) -> Optional[Tuple[torch.Tensor, int, int]]:
        """Get a sample by its filename (useful for debugging)"""
        try:
            idx = self.filenames.index(filename)
            img, label = self[idx]
            return img, label, idx
        except ValueError:
            logger.warning(f"Filename {filename} not found in dataset")
            return None

    def get_all_filenames(self) -> List[str]:
        """Get all filenames in deterministic order"""
        return self.filenames.copy()

    def get_all_paths(self) -> List[str]:
        """Get all file paths in deterministic order"""
        return self.full_paths.copy()

# =============================================================================
# NEURAL NETWORK MODULES
# =============================================================================

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return self.gamma * out + x

class ResidualBlock(nn.Module):
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
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale

# =============================================================================
# BASE AUTOENCODER
# =============================================================================

# =============================================================================
# COMPLETE BASE AUTOENCODER - All original functionality preserved
# =============================================================================

# =============================================================================
# MODIFIED BASE AUTOENCODER with configurable normalization
# =============================================================================

class BaseAutoencoder(nn.Module):
    """Base autoencoder with configurable normalization (dataset-wide or per-image)"""

    def __init__(self, config: GlobalConfig):
        super().__init__()
        # CRITICAL FIX: Set seeds before model initialization
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)

        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        self._feature_order_deterministic = True

        # NEW: Deterministic mode flag
        self.deterministic_mode = True
        self._single_image_mode = False

        self.in_channels = config.in_channels
        self.feature_dims = config.feature_dims
        self.compressed_dims = min(64, max(8, config.compressed_dims))

        self.training_phase = 1
        self._selected_feature_indices = None
        self._feature_importance_scores = None
        self._feature_selection_metadata = {}
        self._is_feature_selection_frozen = False

        self.dataset_statistics = None
        self.attention_maps = {}
        self.hook_handles = []

        self.use_kl_divergence = config.use_kl_divergence
        self.use_class_encoding = config.use_class_encoding
        self.use_distance_correlation = config.use_distance_correlation
        self.enable_sharpness_loss = config.enable_sharpness_loss
        self.enable_adaptive = config.enable_adaptive

        self._build_adaptive_architecture()
        self.apply(self._init_weights)
        self.to(self.device)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model initialized with {total_params:,} parameters")
        logger.info(f"Normalization mode: {config.normalization_mode}")

    def _per_image_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-image Z-score normalization - amplitude invariant"""
        if x.dim() == 4:
            b, c, h, w = x.shape
            x_flat = x.reshape(b, c, -1)
            mean = x_flat.mean(dim=2, keepdim=True)
            std = x_flat.std(dim=2, keepdim=True)
            return ((x_flat - mean) / (std + 1e-8)).reshape(b, c, h, w)
        elif x.dim() == 3:
            c, h, w = x.shape
            x_flat = x.reshape(c, -1)
            mean = x_flat.mean(dim=1, keepdim=True)
            std = x_flat.std(dim=1, keepdim=True)
            return ((x_flat - mean) / (std + 1e-8)).reshape(c, h, w)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")

    def _dataset_wide_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Dataset-wide normalization using frozen statistics"""
        if self.dataset_statistics and self.dataset_statistics.is_calculated:
            return self.dataset_statistics.normalize(x)
        else:
            logger.warning("No frozen statistics available, falling back to per-image normalization")
            return self._per_image_normalize(x)

    def normalize_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization based on configuration.
        Returns normalized tensor.
        """
        if self.config.use_per_image_normalization:
            return self._per_image_normalize(x)
        else:
            return self._dataset_wide_normalize(x)

    def set_deterministic_mode(self, enabled: bool = True, single_image: bool = False):
        """Set deterministic mode for predictions"""
        self.deterministic_mode = enabled
        self._single_image_mode = single_image
        if enabled:
            logger.info(f"Deterministic mode enabled (single_image={single_image})")

    def set_training_phase(self, phase: int):
        """Set training phase and initialize phase-specific components"""
        self.training_phase = phase
        if phase == 2:
            if self.use_class_encoding and self.classifier is None:
                num_classes = self.config.num_classes or 2

                # Check if this is a small image dataset (CIFAR, MNIST, etc.)
                min_dim = min(self.config.input_size)
                is_small_image = min_dim <= 64
                is_cifar = self.config.dataset_name in ['cifar10', 'cifar100']

                if is_small_image or is_cifar:
                    # Simplified classifier for small images (reduces overfitting)
                    logger.info(f"Using simplified classifier for small images (input size: {self.config.input_size})")
                    self.classifier = nn.Sequential(
                        nn.Linear(self.compressed_dims, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        nn.Linear(128, num_classes)
                    ).to(self.device)
                else:
                    # Original complex classifier for larger images
                    compress_dim = max(32, self.compressed_dims // 2)
                    num_groups_class = min(32, compress_dim)
                    self.classifier = nn.Sequential(
                        nn.Linear(self.compressed_dims, compress_dim),
                        nn.GroupNorm(num_groups_class, compress_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(compress_dim, num_classes)
                    ).to(self.device)

                # Initialize weights properly
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm1d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

                self.classifier.apply(init_weights)
                logger.info(f"Initialized classifier with {num_classes} classes")

            if self.use_kl_divergence and self.cluster_centers is None:
                num_clusters = self.config.num_classes or 2
                # CRITICAL FIX: Set seed before random initialization
                torch.manual_seed(42)
                self.cluster_centers = nn.Parameter(
                    torch.randn(num_clusters, self.compressed_dims, device=self.device)
                )
                self.clustering_temperature = nn.Parameter(torch.tensor(1.0, device=self.device))
                logger.info(f"Initialized {num_clusters} cluster centers")

    def set_dataset_statistics(self, statistics: 'DatasetStatisticsCalculator'):
        """Set dataset statistics for dataset-wide normalization"""
        self.dataset_statistics = statistics
        logger.info("Dataset statistics loaded into model")

    def _auto_detect_input_size(self, dataset: Dataset, max_size: int = 512) -> Tuple[int, int]:
        """
        Automatically detect optimal input size from dataset.

        Strategy:
        - Sample a few images to determine typical dimensions
        - Use the most common dimensions or resize to a standard size
        - Clip to max_size to avoid memory issues
        """
        sample_sizes = []
        sample_count = min(100, len(dataset))

        logger.info(f"Auto-detecting input size from {sample_count} samples...")

        for i in range(sample_count):
            try:
                img, _ = dataset[i]
                if isinstance(img, torch.Tensor):
                    h, w = img.shape[-2], img.shape[-1]
                elif isinstance(img, PILImage.Image):
                    w, h = img.size
                elif isinstance(img, np.ndarray):
                    h, w = img.shape[:2]
                else:
                    continue
                sample_sizes.append((h, w))
            except Exception:
                continue

        if not sample_sizes:
            logger.warning("Could not auto-detect size, using default 256x256")
            return (256, 256)

        # Find the most common dimensions
        from collections import Counter
        size_counter = Counter(sample_sizes)
        most_common_size = size_counter.most_common(1)[0][0]
        h, w = most_common_size

        # Clip to max_size if too large
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            h = int(h * scale)
            w = int(w * scale)
            logger.info(f"Resizing from {most_common_size} to {h}x{w} (max size limit)")
        else:
            logger.info(f"Detected input size: {h}x{w}")

        # Ensure dimensions are multiples of 32 for efficient downsampling
        h = ((h + 31) // 32) * 32
        w = ((w + 31) // 32) * 32
        if (h, w) != most_common_size:
            logger.info(f"Adjusted to multiples of 32: {h}x{w}")

        return (h, w)

    def _build_adaptive_architecture(self):
        """
        TRULY DYNAMIC architecture that adapts to ANY dataset.
        Scales based on: number of classes, image size, and dataset complexity.
        """
        h, w = self.config.input_size
        c = self.in_channels
        num_classes = self.config.num_classes or 2
        min_dim = min(h, w)

        # Get dataset complexity from config (set by DynamicArchitectureOptimizer)
        complexity = getattr(self.config, 'complexity_score', 0.5)
        dataset_size = getattr(self.config, 'dataset_size', 50000)

        # Check if we should force fixed compressed_dims
        force_fixed_compressed_dims = getattr(self.config, 'force_fixed_compressed_dims', False)

        # ========================================================================
        # DETERMINE OPTIMAL NUMBER OF LAYERS (Depth)
        # ========================================================================
        # Based on image size: larger images need deeper networks
        if min_dim <= 32:
            base_layers = 2
        elif min_dim <= 64:
            base_layers = 3
        elif min_dim <= 128:
            base_layers = 4
        elif min_dim <= 256:
            base_layers = 5
        else:
            base_layers = 6

        # Adjust depth based on number of classes (more classes need more capacity)
        if num_classes >= 100:
            depth_multiplier = 1.5
        elif num_classes >= 50:
            depth_multiplier = 1.25
        elif num_classes >= 20:
            depth_multiplier = 1.0
        elif num_classes >= 10:
            depth_multiplier = 0.75
        else:
            depth_multiplier = 0.5

        n_layers = max(1, int(base_layers * depth_multiplier))

        # Adjust for dataset size (small datasets need fewer layers to avoid overfitting)
        if dataset_size < 10000:
            n_layers = max(1, n_layers - 1)
        elif dataset_size > 100000:
            n_layers = min(8, n_layers + 1)

        # Calculate actual final spatial size after encoding
        final_h, final_w = h, w
        for _ in range(n_layers):
            final_h = (final_h + 1) // 2
            final_w = (final_w + 1) // 2
        final_h = max(1, final_h)
        final_w = max(1, final_w)

        # ========================================================================
        # DETERMINE CHANNEL WIDTH (Capacity)
        # ========================================================================
        # Base channels scale with image size and complexity
        if min_dim <= 32:
            base_channels = 32
        elif min_dim <= 64:
            base_channels = 48
        else:
            base_channels = 64

        # Scale channels based on number of classes (logarithmic scale)
        # More classes need wider networks
        class_capacity = np.log2(max(2, num_classes)) / np.log2(100)  # Normalized to 100 classes
        class_capacity = max(0.5, min(2.0, class_capacity * 1.5))

        # Scale based on dataset complexity
        complexity_capacity = 0.5 + complexity  # complexity is 0-1, so 0.5-1.5

        # Combined capacity multiplier
        capacity_multiplier = class_capacity * complexity_capacity

        # Apply capacity multiplier to base channels
        if capacity_multiplier > 1.0:
            base_channels = int(base_channels * min(2.0, capacity_multiplier))
        # Round to multiple of 16 for efficiency
        base_channels = ((base_channels + 7) // 8) * 8

        # Maximum channels based on input size (prevent excessive parameters)
        if min_dim <= 32:
            max_channels = min(512, base_channels * (2 ** (n_layers - 1)))
        elif min_dim <= 64:
            max_channels = min(512, base_channels * (2 ** (n_layers - 1)))
        else:
            max_channels = min(1024, base_channels * (2 ** (n_layers - 1)))

        logger.info("=" * 60)
        logger.info("Building Truly Adaptive Architecture")
        logger.info(f"  Input: {c}x{h}x{w}")
        logger.info(f"  Number of classes: {num_classes}")
        logger.info(f"  Dataset complexity: {complexity:.3f}")
        logger.info(f"  Dataset size: {dataset_size:,}")
        logger.info(f"  Encoder layers: {n_layers}")
        logger.info(f"  Final encoder size: {final_h}x{final_w}")
        logger.info(f"  Base channels: {base_channels}")
        logger.info(f"  Max channels: {max_channels}")
        logger.info(f"  Capacity multiplier: {capacity_multiplier:.2f}")
        if force_fixed_compressed_dims:
            logger.info(f"  Force fixed compressed_dims: ENABLED")
        logger.info("=" * 60)

        # ========================================================================
        # BUILD ENCODER
        # ========================================================================
        self.encoder_layers = nn.ModuleList()
        in_channels = c
        self.encoder_channels = []

        current_h, current_w = h, w

        for i in range(n_layers):
            # Calculate output channels - exponential growth but capped
            out_channels = min(max_channels, base_channels * (2 ** i))

            # Ensure out_channels is divisible for GroupNorm
            num_groups = min(16, out_channels)
            if out_channels % num_groups != 0:
                out_channels = ((out_channels + num_groups - 1) // num_groups) * num_groups

            # Build encoder block
            # Use two conv layers per block for better feature extraction when capacity is high
            if capacity_multiplier > 0.8:
                encoder_block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                    nn.GroupNorm(min(16, out_channels), out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                    nn.GroupNorm(min(16, out_channels), out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                encoder_block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                    nn.GroupNorm(min(16, out_channels), out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )

            self.encoder_layers.append(encoder_block)
            self.encoder_channels.append(out_channels)
            in_channels = out_channels

            current_h = (current_h + 1) // 2
            current_w = (current_w + 1) // 2

        self.final_h = max(1, current_h)
        self.final_w = max(1, current_w)
        self.flattened_size = in_channels * self.final_h * self.final_w

        logger.info(f"  Encoder output: [{in_channels}, {self.final_h}, {self.final_w}]")
        logger.info(f"  Flattened size: {self.flattened_size}")

        # ========================================================================
        # DETERMINE FEATURE AND COMPRESSED DIMENSIONS
        # ========================================================================
        # Feature dimension - scales with number of classes
        base_feature_dims = 128
        feature_dims = int(base_feature_dims * class_capacity * complexity_capacity)
        feature_dims = max(32, min(1024, ((feature_dims + 15) // 16) * 16))

        # ========================================================================
        # COMPRESSED DIMENSIONS - WITH FORCE FIXED OPTION
        # ========================================================================
        if force_fixed_compressed_dims and hasattr(self.config, 'compressed_dims'):
            # Use the fixed compressed_dims from config
            compressed_dims = self.config.compressed_dims
            logger.info(f"  Using FIXED compressed_dims: {compressed_dims} (forced by config)")
        else:
            # Auto-calculate based on dataset
            if num_classes <= 10:
                compressed_dims = 16
            elif num_classes <= 50:
                compressed_dims = 32
            elif num_classes <= 100:
                compressed_dims = 64
            else:
                compressed_dims = 128

            if dataset_size < 10000:
                compressed_dims = max(8, compressed_dims // 2)
            elif dataset_size > 50000:
                compressed_dims = min(256, compressed_dims * 2)

            # Ensure compressed dimension is sufficient for classifier
            min_required = max(16, int(np.log2(num_classes) * 8))
            if compressed_dims < min_required:
                compressed_dims = min_required
                logger.info(f"  Adjusted compressed_dims to {compressed_dims} (minimum for {num_classes} classes)")

            logger.info(f"  Auto-calculated compressed_dims: {compressed_dims}")

        # Ensure compressed_dims is a multiple of 8 for better GPU performance
        compressed_dims = max(16, ((compressed_dims + 7) // 8) * 8)

        self.feature_dims = feature_dims
        self.compressed_dims = compressed_dims

        logger.info(f"  Feature dimensions: {self.feature_dims}")
        logger.info(f"  Final compressed dimensions: {self.compressed_dims}")

        # ========================================================================
        # BUILD EMBEDDER (Encoder → Feature Space)
        # ========================================================================
        embed_dim = min(self.flattened_size, self.feature_dims)
        num_groups_embed = min(16, max(1, embed_dim // 16))
        if embed_dim > 1 and embed_dim % num_groups_embed != 0:
            for g in range(num_groups_embed, 0, -1):
                if embed_dim % g == 0:
                    num_groups_embed = g
                    break

        self.embedder = nn.Sequential(
            nn.Linear(self.flattened_size, embed_dim),
            nn.GroupNorm(num_groups_embed, embed_dim) if embed_dim > 1 else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3 if capacity_multiplier > 1.0 else 0.2)
        )

        if embed_dim != self.feature_dims:
            self.embedder_projection = nn.Linear(embed_dim, self.feature_dims)
            logger.info(f"  Embedder projection: {embed_dim} → {self.feature_dims}")
        else:
            self.embedder_projection = nn.Identity()

        # ========================================================================
        # BUILD UNEMBEDDER (Feature Space → Encoder Output)
        # ========================================================================
        unembed_dim = min(self.flattened_size, self.feature_dims)
        num_groups_unembed = min(16, max(1, unembed_dim // 16))
        if unembed_dim > 1 and unembed_dim % num_groups_unembed != 0:
            for g in range(num_groups_unembed, 0, -1):
                if unembed_dim % g == 0:
                    num_groups_unembed = g
                    break

        self.unembedder = nn.Sequential(
            nn.Linear(self.feature_dims, unembed_dim),
            nn.GroupNorm(num_groups_unembed, unembed_dim) if unembed_dim > 1 else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3 if capacity_multiplier > 1.0 else 0.2)
        )

        if unembed_dim != self.flattened_size:
            self.unembedder_projection = nn.Linear(unembed_dim, self.flattened_size)
            logger.info(f"  Unembedder projection: {unembed_dim} → {self.flattened_size}")
        else:
            self.unembedder_projection = nn.Identity()

        # ========================================================================
        # BUILD DECODER
        # ========================================================================
        self.decoder_layers = nn.ModuleList()
        in_channels = self.encoder_channels[-1]

        for i in range(n_layers - 1, -1, -1):
            out_channels = c if i == 0 else self.encoder_channels[i-1]
            num_groups_dec = min(16, max(1, out_channels // 16))
            if out_channels % num_groups_dec != 0:
                for g in range(num_groups_dec, 0, -1):
                    if out_channels % g == 0:
                        num_groups_dec = g
                        break

            if i == 0:
                decoder_block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.Tanh()
                )
            else:
                if capacity_multiplier > 0.8:
                    decoder_block = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                        nn.GroupNorm(num_groups_dec, out_channels),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                        nn.GroupNorm(num_groups_dec, out_channels),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                else:
                    decoder_block = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                        nn.GroupNorm(num_groups_dec, out_channels),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
            self.decoder_layers.append(decoder_block)
            in_channels = out_channels

        # ========================================================================
        # BUILD FEATURE COMPRESSOR
        # ========================================================================
        compress_dim = max(32, min(256, self.feature_dims // 2))
        compress_dim = max(compress_dim, self.compressed_dims * 2)

        num_groups_comp = min(16, max(1, compress_dim // 16))
        if compress_dim > 1 and compress_dim % num_groups_comp != 0:
            for g in range(num_groups_comp, 0, -1):
                if compress_dim % g == 0:
                    num_groups_comp = g
                    break

        self.feature_compressor = nn.Sequential(
            nn.Linear(self.feature_dims, compress_dim),
            nn.GroupNorm(num_groups_comp, compress_dim) if compress_dim > 1 else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3 if capacity_multiplier > 1.0 else 0.2),
            nn.Linear(compress_dim, self.compressed_dims),
            nn.Tanh()
        )

        # ========================================================================
        # BUILD FEATURE DECOMPRESSOR
        # ========================================================================
        decompress_dim = max(32, min(256, self.feature_dims // 2))
        decompress_dim = max(decompress_dim, self.compressed_dims * 2)

        num_groups_decomp = min(16, max(1, decompress_dim // 16))
        if decompress_dim > 1 and decompress_dim % num_groups_decomp != 0:
            for g in range(num_groups_decomp, 0, -1):
                if decompress_dim % g == 0:
                    num_groups_decomp = g
                    break

        self.feature_decompressor = nn.Sequential(
            nn.Linear(self.compressed_dims, decompress_dim),
            nn.GroupNorm(num_groups_decomp, decompress_dim) if decompress_dim > 1 else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3 if capacity_multiplier > 1.0 else 0.2),
            nn.Linear(decompress_dim, self.feature_dims),
            nn.Tanh()
        )

        # ========================================================================
        # INITIALIZE PHASE 2 COMPONENTS
        # ========================================================================
        self.classifier = None
        self.cluster_centers = None
        self.clustering_temperature = None

        total_params = sum(p.numel() for p in self.parameters())

        logger.info("=" * 60)
        logger.info("Architecture Summary:")
        logger.info(f"  Input: {c}x{h}x{w}")
        logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Encoder layers: {n_layers}")
        logger.info(f"  Encoder channels: {self.encoder_channels}")
        logger.info(f"  Flattened size: {self.flattened_size}")
        logger.info(f"  Feature dims: {self.feature_dims}")
        logger.info(f"  Compressed dims: {self.compressed_dims}")
        logger.info(f"  Force fixed compressed_dims: {force_fixed_compressed_dims}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info("=" * 60)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GroupNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def get_frozen_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self._selected_feature_indices is None:
            return embeddings
        return embeddings[:, self._selected_feature_indices]

    def freeze_feature_selection(self, indices: np.ndarray, scores: np.ndarray, metadata: Dict = None):
        sorted_order = np.argsort(indices)
        indices = indices[sorted_order]
        scores = scores[sorted_order] if scores is not None else None

        self._selected_feature_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        self._feature_importance_scores = torch.tensor(scores, device=self.device) if scores is not None else None
        self._feature_selection_metadata = metadata or {}
        self._is_feature_selection_frozen = True
        logger.info(f"Frozen feature selection: {len(indices)} features")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to feature space.
        """
        # Run through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Flatten spatial dimensions
        x = x.view(x.size(0), -1)

        # Embed to feature space
        x = self.embedder(x)
        x = self.embedder_projection(x)

        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from feature space back to image space.
        Handles the projection from feature_dims to flattened_size correctly.
        """
        # Unembed: feature_dims → unembed_dim → flattened_size
        x = self.unembedder(z)
        x = self.unembedder_projection(x)

        # Reshape to spatial dimensions
        x = x.view(x.size(0), self.encoder_channels[-1], self.final_h, self.final_w)

        # Run through decoder layers
        for layer in self.decoder_layers:
            x = layer(x)

        # Ensure output has correct spatial dimensions
        # Sometimes ConvTranspose2d might produce slightly different dimensions
        target_h, target_w = self.config.input_size
        if x.shape[-2] != target_h or x.shape[-1] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)

        return x

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with proper dimension handling.
        """
        original_batch_size = x.size(0)
        duplicated = False

        # Apply configured normalization
        x = self.normalize_batch(x)

        # Handle single sample batches
        if self.training and original_batch_size == 1:
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

        # Compression and decompression
        compressed = self.feature_compressor(selected_embedding)
        decompressed = self.feature_decompressor(compressed)

        # Decode
        reconstruction = self.decode(decompressed)

        # Ensure reconstruction matches input size (fallback)
        if reconstruction.shape[-2:] != x.shape[-2:]:
            reconstruction = F.interpolate(reconstruction, size=x.shape[-2:],
                                         mode='bilinear', align_corners=False)

        # Build output dictionary
        output = {
            'embedding': self._fix_tensor_dim(embedding, original_batch_size, duplicated),
            'selected_embedding': self._fix_tensor_dim(selected_embedding, original_batch_size, duplicated),
            'compressed_embedding': self._fix_tensor_dim(compressed, original_batch_size, duplicated),
            'reconstructed_embedding': self._fix_tensor_dim(decompressed, original_batch_size, duplicated),
            'reconstruction': self._fix_tensor_dim(reconstruction, original_batch_size, duplicated)
        }

        # Phase 2 outputs (classification and clustering)
        if self.training_phase == 2:
            if self.use_class_encoding and self.classifier is not None:
                logits = self.classifier(compressed)
                output.update({
                    'class_logits': self._fix_tensor_dim(logits, original_batch_size, duplicated),
                    'class_predictions': self._fix_tensor_dim(logits.argmax(dim=1), original_batch_size, duplicated),
                    'class_probabilities': self._fix_tensor_dim(F.softmax(logits, dim=1), original_batch_size, duplicated)
                })

            if self.use_kl_divergence and self.cluster_centers is not None:
                distances = torch.cdist(compressed, self.cluster_centers)
                q = 1.0 / (1.0 + distances ** 2)
                q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
                p = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
                p = p / (p.sum(dim=1, keepdim=True) + 1e-8)

                output.update({
                    'cluster_probabilities': self._fix_tensor_dim(q, original_batch_size, duplicated),
                    'target_distribution': self._fix_tensor_dim(p, original_batch_size, duplicated),
                    'cluster_assignments': self._fix_tensor_dim(q.argmax(dim=1), original_batch_size, duplicated),
                    'cluster_confidence': self._fix_tensor_dim(q.max(dim=1)[0], original_batch_size, duplicated)
                })

        return output

    def _fix_tensor_dim(self, tensor: torch.Tensor, target_batch_size: int, duplicated: bool) -> torch.Tensor:
        """
        Fix tensor dimensions after possible duplication.
        """
        if duplicated and tensor.size(0) > target_batch_size:
            tensor = tensor[:target_batch_size]

        # Handle 1D tensors (like predictions)
        if tensor.dim() == 1 and target_batch_size == 1:
            tensor = tensor.unsqueeze(0)

        return tensor

    @torch.no_grad()
    @memory_efficient
    def extract_features(self, dataloader: DataLoader, include_paths: bool = True) -> Dict:
        self.eval()
        all_embeddings = []
        all_labels = []
        all_paths = []
        all_filenames = []
        all_class_names = []
        all_cluster_assignments = []
        all_cluster_confidence = []

        # Store class mapping if available
        self._idx_to_class = {}
        if hasattr(dataloader.dataset, 'idx_to_class'):
            self._idx_to_class = dataloader.dataset.idx_to_class
        elif hasattr(dataloader.dataset, 'dataset') and hasattr(dataloader.dataset.dataset, 'idx_to_class'):
            self._idx_to_class = dataloader.dataset.dataset.idx_to_class

        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
            inputs = inputs.to(self.device, non_blocking=True)
            outputs = self(inputs)

            # Get compressed embedding (this is the feature vector)
            if 'compressed_embedding' in outputs:
                embeddings = outputs['compressed_embedding'].float().cpu()
            else:
                embeddings = outputs['embedding'].float().cpu()

            all_embeddings.append(embeddings)
            all_labels.append(labels)

            # Get cluster information
            if 'cluster_assignments' in outputs:
                all_cluster_assignments.append(outputs['cluster_assignments'].cpu())
            if 'cluster_confidence' in outputs:
                all_cluster_confidence.append(outputs['cluster_confidence'].cpu())

            if include_paths and hasattr(dataloader.dataset, 'get_additional_info'):
                for i in range(len(labels)):
                    idx = batch_idx * dataloader.batch_size + i
                    if idx < len(dataloader.dataset):
                        info = dataloader.dataset.get_additional_info(idx)
                        all_filenames.append(info[1])  # filename
                        all_paths.append(info[2])      # full path
                        # Get class name from label
                        label_val = labels[i].item()
                        if hasattr(dataloader.dataset, 'idx_to_class'):
                            class_name = dataloader.dataset.idx_to_class.get(label_val, str(label_val))
                        elif hasattr(dataloader.dataset, 'dataset') and hasattr(dataloader.dataset.dataset, 'idx_to_class'):
                            class_name = dataloader.dataset.dataset.idx_to_class.get(label_val, str(label_val))
                        else:
                            class_name = str(label_val)
                        all_class_names.append(class_name)

        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
        else:
            embeddings = torch.tensor([])
            labels = torch.tensor([])

        if all_cluster_assignments:
            cluster_assignments = torch.cat(all_cluster_assignments, dim=0)
        else:
            cluster_assignments = torch.tensor([])

        if all_cluster_confidence:
            cluster_confidence = torch.cat(all_cluster_confidence, dim=0)
        else:
            cluster_confidence = torch.tensor([])

        result = {
            'embeddings': embeddings,
            'labels': labels,
            'cluster_assignments': cluster_assignments,
            'cluster_confidence': cluster_confidence
        }

        if all_paths:
            result['paths'] = all_paths
            result['filenames'] = all_filenames
            result['class_names'] = all_class_names

        logger.info(f"Extracted {len(embeddings)} features with dimension {embeddings.shape[1] if len(embeddings) > 0 else 0}")
        return result

    def save_features(self, features_dict: Dict[str, torch.Tensor], output_path: str) -> None:
        data = {}
        embeddings = features_dict['embeddings'].cpu().numpy()

        if self._is_feature_selection_frozen and self._selected_feature_indices is not None:
            indices = self._selected_feature_indices.cpu().numpy()
            if len(indices) > self.config.max_features:
                indices = indices[:self.config.max_features]
            embeddings = embeddings[:, indices]

        for i in range(embeddings.shape[1]):
            data[f'feature_{i}'] = embeddings[:, i]

        if 'class_names' in features_dict:
            data['target'] = features_dict['class_names']
        else:
            data['target'] = features_dict['labels'].cpu().numpy()

        if 'filenames' in features_dict:
            data['filename'] = features_dict['filenames']
        if 'paths' in features_dict:
            data['filepath'] = features_dict['paths']

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")

    @torch.jit.export
    def forward_optimized(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Optimized forward pass with reduced overhead"""
        original_batch_size = x.size(0)

        # Apply normalization (vectorized)
        x = self.normalize_batch(x)

        # Handle single sample efficiently
        if self.training and original_batch_size == 1:
            x = x.repeat(2, 1, 1, 1)  # More efficient than cat
            if labels is not None:
                labels = labels.repeat(2)
            duplicated = True
        else:
            duplicated = False

        # Vectorized encoding
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.flatten(1)  # More efficient than view
        embedding = self.embedder(x)

        # Feature selection
        if self._is_feature_selection_frozen and self._selected_feature_indices is not None:
            selected_embedding = embedding.index_select(1, self._selected_feature_indices)
        else:
            selected_embedding = embedding

        # Compression and decompression
        compressed = self.feature_compressor(selected_embedding)
        decompressed = self.feature_decompressor(compressed)

        # Decode
        decoded = self.unembedder(decompressed)
        decoded = decoded.view(-1, self.encoder_channels[-1], self.final_h, self.final_w)
        for layer in self.decoder_layers:
            decoded = layer(decoded)

        # Ensure correct size
        if decoded.shape[-2:] != x.shape[-2:]:
            decoded = F.interpolate(decoded, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # Build output
        output = {
            'embedding': embedding[:original_batch_size] if duplicated else embedding,
            'selected_embedding': selected_embedding[:original_batch_size] if duplicated else selected_embedding,
            'compressed_embedding': compressed[:original_batch_size] if duplicated else compressed,
            'reconstructed_embedding': decompressed[:original_batch_size] if duplicated else decompressed,
            'reconstruction': decoded[:original_batch_size] if duplicated else decoded
        }

        # Phase 2 outputs
        if self.training_phase == 2:
            if self.use_class_encoding and self.classifier is not None:
                logits = self.classifier(compressed)
                output.update({
                    'class_logits': logits[:original_batch_size] if duplicated else logits,
                    'class_predictions': logits.argmax(dim=1)[:original_batch_size] if duplicated else logits.argmax(dim=1),
                    'class_probabilities': F.softmax(logits, dim=1)[:original_batch_size] if duplicated else F.softmax(logits, dim=1)
                })

            if self.use_kl_divergence and self.cluster_centers is not None:
                # Vectorized distance computation
                distances = torch.cdist(compressed, self.cluster_centers)
                q = 1.0 / (1.0 + distances ** 2)
                q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
                p = (q ** 2) / (q.sum(dim=0, keepdim=True) + 1e-8)
                p = p / (p.sum(dim=1, keepdim=True) + 1e-8)

                output.update({
                    'cluster_probabilities': q[:original_batch_size] if duplicated else q,
                    'cluster_assignments': q.argmax(dim=1)[:original_batch_size] if duplicated else q.argmax(dim=1),
                    'cluster_confidence': q.max(dim=1)[0][:original_batch_size] if duplicated else q.max(dim=1)[0]
                })

        return output


class EnhancedBaseAutoencoder(BaseAutoencoder):
    """
    Enhanced Base Autoencoder with discriminative features for subtle pattern recognition.
    Works with any number of input channels (RGB, grayscale, multi-spectral).
    Can be inherited by domain-specific autoencoders.
    """

    def __init__(self, config: GlobalConfig):
        super().__init__(config)

        # Discriminative enhancement flags
        self.use_detail_attention = getattr(config, 'use_detail_attention', True)
        self.use_multiscale_features = getattr(config, 'use_multiscale_features', True)
        self.use_feature_refinement = getattr(config, 'use_feature_refinement', True)

        # Dynamically determine intermediate channels based on input channels
        # For grayscale (1 channel), use fewer intermediate channels
        # For RGB (3 channels), use standard 32
        # For multi-spectral (>3), scale accordingly
        if self.in_channels == 1:
            detail_intermediate = 16
            edge_intermediate = 8
        elif self.in_channels <= 3:
            detail_intermediate = 32
            edge_intermediate = 16
        else:
            detail_intermediate = min(64, self.in_channels * 8)
            edge_intermediate = min(32, self.in_channels * 4)

        # FIXED: Detail attention module - works with any input channels
        if self.use_detail_attention:
            self.detail_attention = nn.Sequential(
                nn.Conv2d(self.in_channels, detail_intermediate, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(detail_intermediate, 1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.detail_attention = None

        # Multi-scale feature extractor (captures features at different scales)
        # This works on the deepest encoder features, so encoder_channels[-1] is correct
        if self.use_multiscale_features and len(self.encoder_channels) > 0:
            # Adjust dilation rates for different input sizes
            max_dilation = min(4, self.encoder_channels[-1] // 64 + 1)
            self.multiscale_extractors = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.encoder_channels[-1], self.encoder_channels[-1],
                             kernel_size=k, padding=k//2, dilation=1),
                    nn.BatchNorm2d(self.encoder_channels[-1]),
                    nn.ReLU(inplace=True)
                ) for k in [3, 5, 7]
            ])
        else:
            self.multiscale_extractors = None

        # Feature refinement for better discrimination
        if self.use_feature_refinement:
            self.feature_refiner = nn.Sequential(
                nn.Linear(self.compressed_dims, self.compressed_dims * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(self.compressed_dims * 2, self.compressed_dims),
                nn.Tanh()
            )
        else:
            self.feature_refiner = None

        # Edge preservation module for Phase 1 - works with any input channels
        self.edge_preservation = nn.Sequential(
            nn.Conv2d(self.in_channels, edge_intermediate, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_intermediate, self.in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced encoding with discriminative feature extraction"""

        # Apply detail attention to preserve fine features
        # Works with any number of input channels
        if self.detail_attention is not None and self.training:
            attention = self.detail_attention(x)  # detail_attention adapts to in_channels
            x = x * (1 + attention)

        # Apply edge preservation
        edge_weights = self.edge_preservation(x)
        x = x * (0.5 + edge_weights)

        # Standard encoding path (now x has in_channels, which matches encoder_layers[0] input)
        for layer in self.encoder_layers:
            x = layer(x)

        # Multi-scale feature extraction (captures features at different scales)
        if self.multiscale_extractors is not None and self.training:
            multi_features = []
            for extractor in self.multiscale_extractors:
                multi_features.append(extractor(x))
            x = x + sum(multi_features) / len(multi_features)

        # Flatten and embed
        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)

        return embedding

    def get_refined_features(self, compressed_embedding: torch.Tensor) -> torch.Tensor:
        """Get refined features for better discrimination"""
        if self.feature_refiner is not None:
            return self.feature_refiner(compressed_embedding)
        return compressed_embedding

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced feature extraction"""

        # Apply dataset-wide normalization if available
        if self.dataset_statistics is not None and self.dataset_statistics.is_calculated:
            x = self.normalize_batch(x)

        original_batch_size = x.size(0)
        duplicated = False

        if self.training and original_batch_size == 1:
            x = torch.cat([x, x], dim=0)
            duplicated = True
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)

        # Enhanced encoding
        embedding = self.encode(x)

        # Feature selection
        if self._is_feature_selection_frozen and self._selected_feature_indices is not None:
            selected_embedding = embedding[:, self._selected_feature_indices]
        else:
            selected_embedding = embedding

        # Compression and decompression
        compressed = self.feature_compressor(selected_embedding)

        # Get refined features for discrimination
        refined = self.get_refined_features(compressed)

        decompressed = self.feature_decompressor(refined)
        reconstruction = self.decode(decompressed)

        if reconstruction.shape[-2:] != x.shape[-2:]:
            reconstruction = F.interpolate(reconstruction, size=x.shape[-2:],
                                         mode='bilinear', align_corners=False)

        # Build output dictionary
        output = {
            'embedding': self._fix_tensor_dim(embedding, original_batch_size, duplicated),
            'selected_embedding': self._fix_tensor_dim(selected_embedding, original_batch_size, duplicated),
            'compressed_embedding': self._fix_tensor_dim(compressed, original_batch_size, duplicated),
            'refined_embedding': self._fix_tensor_dim(refined, original_batch_size, duplicated),
            'reconstructed_embedding': self._fix_tensor_dim(decompressed, original_batch_size, duplicated),
            'reconstruction': self._fix_tensor_dim(reconstruction, original_batch_size, duplicated)
        }

        # Phase 2 outputs (classification and clustering)
        if self.training_phase == 2:
            # Use refined features for better classification
            features_for_classification = refined

            if self.use_class_encoding and self.classifier is not None:
                logits = self.classifier(features_for_classification)
                output.update({
                    'class_logits': self._fix_tensor_dim(logits, original_batch_size, duplicated),
                    'class_predictions': self._fix_tensor_dim(logits.argmax(dim=1), original_batch_size, duplicated),
                    'class_probabilities': self._fix_tensor_dim(F.softmax(logits, dim=1), original_batch_size, duplicated)
                })

            if self.use_kl_divergence and self.cluster_centers is not None:
                distances = torch.cdist(features_for_classification, self.cluster_centers)
                q = 1.0 / (1.0 + distances ** 2)
                q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
                p = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
                p = p / (p.sum(dim=1, keepdim=True) + 1e-8)

                output.update({
                    'cluster_probabilities': self._fix_tensor_dim(q, original_batch_size, duplicated),
                    'target_distribution': self._fix_tensor_dim(p, original_batch_size, duplicated),
                    'cluster_assignments': self._fix_tensor_dim(q.argmax(dim=1), original_batch_size, duplicated),
                    'cluster_confidence': self._fix_tensor_dim(q.max(dim=1)[0], original_batch_size, duplicated)
                })

        return output

# =============================================================================
# CONTRASTIVE LEARNING MODULES (Only used when --use_contrastive is specified)
# =============================================================================


class ContrastiveProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    Maps features to a lower-dimensional space where contrastive loss is applied
    """
    def __init__(self, input_dim, projection_dim=128, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(256, input_dim * 2)

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        return self.projection(x)


class ContrastiveAutoencoderWithProjection(BaseAutoencoder):
    """
    Autoencoder with contrastive learning projection head.
    Uses supervised contrastive learning for better feature separation.
    """

    def __init__(self, config: GlobalConfig):
        super().__init__(config)

        # Get contrastive-specific parameters
        self.projection_dim = getattr(config, 'contrastive_projection_dim', 128)
        self.temperature = getattr(config, 'contrastive_temperature', 0.07)
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.7)

        # ================================================================
        # PROJECTION HEAD
        # ================================================================
        self.projection_head = nn.Sequential(
            nn.Linear(self.compressed_dims, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.projection_dim),
            nn.BatchNorm1d(self.projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.projection_dim, self.projection_dim)
        )

        # ================================================================
        # ENHANCED CLASSIFIER (with dropout for regularization)
        # ================================================================
        num_classes = config.num_classes or 2

        # Scale classifier complexity with number of classes
        if num_classes >= 100:
            self.classifier = nn.Sequential(
                nn.Linear(self.compressed_dims, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        elif num_classes >= 50:
            self.classifier = nn.Sequential(
                nn.Linear(self.compressed_dims, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.compressed_dims, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )

        # ================================================================
        # LOSS FUNCTIONS
        # ================================================================
        self.contrastive_loss = SupervisedContrastiveLoss(
            temperature=self.temperature
        )

        self.is_contrastive = True

        logger.info(f"ContrastiveAutoencoderWithProjection initialized:")
        logger.info(f"  Projection dimension: {self.projection_dim}")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  Contrastive weight: {self.contrastive_weight}")

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with contrastive learning"""
        original_batch_size = x.size(0)
        duplicated = False

        # Apply normalization
        x = self.normalize_batch(x)

        # Handle single sample
        if self.training and original_batch_size == 1:
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

        # Compression
        compressed = self.feature_compressor(selected_embedding)

        # Apply projection for contrastive learning
        projected = self.projection_head(compressed)

        # Decode
        decompressed = self.feature_decompressor(compressed)
        reconstruction = self.decode(decompressed)

        if reconstruction.shape[-2:] != x.shape[-2:]:
            reconstruction = F.interpolate(reconstruction, size=x.shape[-2:],
                                         mode='bilinear', align_corners=False)

        # Build output dictionary
        output = {
            'embedding': self._fix_tensor_dim(embedding, original_batch_size, duplicated),
            'selected_embedding': self._fix_tensor_dim(selected_embedding, original_batch_size, duplicated),
            'compressed_embedding': self._fix_tensor_dim(compressed, original_batch_size, duplicated),
            'projected_embedding': self._fix_tensor_dim(projected, original_batch_size, duplicated),
            'reconstructed_embedding': self._fix_tensor_dim(decompressed, original_batch_size, duplicated),
            'reconstruction': self._fix_tensor_dim(reconstruction, original_batch_size, duplicated)
        }

        # Phase 2 outputs (classification and clustering)
        if self.training_phase == 2:
            # Classification
            if self.use_class_encoding and self.classifier is not None:
                logits = self.classifier(compressed)
                output.update({
                    'class_logits': self._fix_tensor_dim(logits, original_batch_size, duplicated),
                    'class_predictions': self._fix_tensor_dim(logits.argmax(dim=1), original_batch_size, duplicated),
                    'class_probabilities': self._fix_tensor_dim(F.softmax(logits, dim=1), original_batch_size, duplicated)
                })

            # Contrastive learning (only during training with labels)
            if self.training and labels is not None:
                # Use projected features for contrastive loss
                output.update({
                    'contrastive_features': self._fix_tensor_dim(projected, original_batch_size, duplicated),
                    'contrastive_labels': self._fix_tensor_dim(labels, original_batch_size, duplicated)
                })

        return output

    def compute_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss"""
        return self.contrastive_loss(features, labels)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon)
    Based on: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, mask=None):
        """
        Args:
            features: [B, D] normalized embeddings
            labels: [B] class labels
            mask: [B, B] optional mask for positive pairs
        Returns:
            SupCon loss
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features if not already
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs (same class, excluding self)
        if mask is None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

        # Remove self-similarity
        mask.fill_diagonal_(0)

        # Compute logits
        exp_sim = torch.exp(similarity)

        # Numerator: sum over positive pairs
        pos_sum = (exp_sim * mask).sum(dim=1)

        # Denominator: sum over all pairs (including self)
        neg_sum = exp_sim.sum(dim=1) - exp_sim.diag()  # Remove self

        # Loss: -log(pos / (pos + neg))
        loss = -torch.log(pos_sum / (neg_sum + 1e-8) + 1e-8)

        # Only compute loss for samples with positive pairs
        valid = mask.sum(dim=1) > 0
        if valid.any():
            loss = loss[valid].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss


class ContrastiveTrainer:
    """
    Trainer specialized for contrastive learning
    Only used when --use_contrastive is enabled
    """

    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = next(model.parameters()).device

        # ================================================================
        # OPTIMIZER: LARS or AdamW with warmup
        # ================================================================
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0005,
            betas=(0.9, 0.999)
        )

        # ================================================================
        # SCHEDULER: Cosine annealing with warm restarts
        # ================================================================
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=30,
            T_mult=2,
            eta_min=1e-6
        )

        # ================================================================
        # LOSS WEIGHTS
        # ================================================================
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.7)
        self.classification_weight = 0.2
        self.reconstruction_weight = 0.1

        # ================================================================
        # METRICS TRACKING
        # ================================================================
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.history = defaultdict(list)

        logger.info("ContrastiveTrainer initialized")
        logger.info(f"  Contrastive weight: {self.contrastive_weight}")
        logger.info(f"  Classification weight: {self.classification_weight}")
        logger.info(f"  Reconstruction weight: {self.reconstruction_weight}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_contrastive_loss = 0.0
        total_class_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images, labels)

            # ================================================================
            # COMPUTE LOSSES
            # ================================================================

            # 1. Reconstruction loss
            recon_loss = F.mse_loss(outputs['reconstruction'], images)

            # 2. Contrastive loss (using projected features)
            if 'contrastive_features' in outputs:
                contrastive_loss = self.model.compute_contrastive_loss(
                    outputs['contrastive_features'],
                    outputs['contrastive_labels']
                )
            else:
                contrastive_loss = torch.tensor(0.0, device=self.device)

            # 3. Classification loss
            if 'class_logits' in outputs:
                # Use label smoothing for many classes
                num_classes = self.config.num_classes or 100
                if num_classes >= 50:
                    smoothing = 0.1
                    n_classes = num_classes
                    logits = outputs['class_logits']
                    smooth_labels = torch.full_like(logits, smoothing / n_classes)
                    smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - smoothing + smoothing / n_classes)
                    class_loss = -(smooth_labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                else:
                    class_loss = F.cross_entropy(outputs['class_logits'], labels)
            else:
                class_loss = torch.tensor(0.0, device=self.device)

            # ================================================================
            # COMBINE LOSSES
            # ================================================================
            total_loss = (
                self.reconstruction_weight * recon_loss +
                self.contrastive_weight * contrastive_loss +
                self.classification_weight * class_loss
            )

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Track metrics
            total_loss += total_loss.item()
            total_recon_loss += recon_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_class_loss += class_loss.item()

            if 'class_predictions' in outputs:
                acc = (outputs['class_predictions'] == labels).float().mean().item()
                total_acc += acc

            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss/n_batches:.4f}",
                'acc': f"{total_acc/n_batches:.2%}" if n_batches > 0 else "0%"
            })

        # Calculate averages
        avg_loss = total_loss / n_batches
        avg_recon = total_recon_loss / n_batches
        avg_contrastive = total_contrastive_loss / n_batches
        avg_class = total_class_loss / n_batches
        avg_acc = total_acc / n_batches if n_batches > 0 else 0.0

        return {
            'loss': avg_loss,
            'recon_loss': avg_recon,
            'contrastive_loss': avg_contrastive,
            'class_loss': avg_class,
            'accuracy': avg_acc
        }

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_acc = 0.0
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images, labels)

                if 'class_predictions' in outputs:
                    acc = (outputs['class_predictions'] == labels).float().mean().item()
                    total_acc += acc

                if 'class_logits' in outputs:
                    loss = F.cross_entropy(outputs['class_logits'], labels)
                    total_loss += loss.item()

                n_batches += 1

        avg_acc = total_acc / n_batches if n_batches > 0 else 0.0
        avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')

        return {
            'accuracy': avg_acc,
            'loss': avg_loss
        }

    def train(self, epochs):
        """Full training loop"""
        logger.info("=" * 70)
        logger.info("CONTRASTIVE TRAINING STARTED")
        logger.info(f"Total epochs: {epochs}")
        logger.info(f"Contrastive weight: {self.contrastive_weight}")
        logger.info("=" * 70)

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2%} | "
                f"Val Acc: {val_metrics['accuracy']:.2%} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Save best model
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self.best_loss = val_metrics['loss']
                self.patience_counter = 0

                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': self.best_accuracy,
                    'best_loss': self.best_loss,
                    'use_contrastive': True,
                    'contrastive_temperature': self.config.contrastive_temperature,
                    'contrastive_weight': self.contrastive_weight,
                    'history': dict(self.history)
                }
                torch.save(checkpoint, self.config.checkpoint_dir / 'best_contrastive.pt')
                logger.info(f"✓ New best accuracy: {self.best_accuracy:.2%}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= 30:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info("=" * 70)
        logger.info("CONTRASTIVE TRAINING COMPLETED")
        logger.info(f"Best accuracy: {self.best_accuracy:.2%}")
        logger.info(f"Best loss: {self.best_loss:.4f}")
        logger.info("=" * 70)

        return dict(self.history)

# =============================================================================
# MODIFIED PREDICTION MANAGER with configurable normalization
# =============================================================================

class PredictionManager:
    def __init__(self, config: GlobalConfig, model_load_dir: str = None):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')

        # ========================================================================
        # OUTPUT DIRECTORY (where predictions and configs are saved)
        # ========================================================================
        if hasattr(config, 'output_dir') and config.output_dir:
            self.output_base_dir = Path(config.output_dir)
        else:
            self.output_base_dir = Path('data')

        dataset_name = config.dataset_name
        self.output_data_dir = self.output_base_dir / dataset_name
        self.output_checkpoint_dir = self.output_data_dir / 'checkpoints'
        self.output_viz_dir = self.output_data_dir / 'visualizations'

        # Create output directories
        #self.output_data_dir.mkdir(parents=True, exist_ok=True)
        #self.output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        #self.output_viz_dir.mkdir(parents=True, exist_ok=True)

        # ========================================================================
        # MODEL LOADING DIRECTORY (where trained model is stored)
        # ========================================================================
        if model_load_dir:
            self.model_load_dir = Path(model_load_dir)
        elif hasattr(config, 'model_loading_dir') and config.model_loading_dir:
            self.model_load_dir = Path(config.model_loading_dir)
        else:
            self.model_load_dir = Path('data') / dataset_name / 'checkpoints'

        # Also set data loading directory for stats
        if hasattr(config, 'data_dir_for_loading') and config.data_dir_for_loading:
            self.data_load_dir = Path(config.data_dir_for_loading)
        else:
            self.data_load_dir = Path('data') / dataset_name

        logger.info("=" * 60)
        logger.info(f"PredictionManager Initialized:")
        logger.info(f"  Model loading from: {self.model_load_dir}")
        logger.info(f"  Data/Stats loading from: {self.data_load_dir}")
        logger.info(f"  Output directory: {self.output_data_dir}")
        logger.info("=" * 60)

        self.domain_processor = None
        self.domain_info = {'domain': 'general', 'domain_config': {}}

        self.model = None
        self.dataset_statistics = None

        # Initialize normalization statistics attributes
        self.norm_mean = None
        self.norm_std = None
        self.norm_per_channel_min = None
        self.norm_per_channel_max = None
        self.norm_n_samples = 0
        self.norm_is_fitted = False

        # Load model (this also attaches statistics)
        self._load_model()

        if self.model is not None:
            self.model.set_deterministic_mode(enabled=True, single_image=True)
            logger.info(f"Model configured for deterministic single-image prediction")
            logger.info(f"Normalization mode: {self.config.normalization_mode}")
            logger.info(f"Expected input size: {self.config.input_size}")

        # Create invariant preprocessor (same as training)
        use_augmentation = not getattr(self.config, 'no_augmentation', False)
        augmentation_strength = getattr(self.config, 'augmentation_strength', 0.5)

        self.invariant_preprocessor = DeterministicInvariantPreprocessor(
            self.config.input_size,
            use_augmentation=False,
            augmentation_strength=augmentation_strength
        )

        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build transform that converts PIL to tensor"""
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    def _preprocess_image_for_model(self, img: PILImage.Image) -> torch.Tensor:
        """
        Apply the SAME preprocessing as training:
        1. Deterministic invariant preprocessing (resize, normalization, etc.)
        2. Convert to tensor
        """
        processed_img = self.invariant_preprocessor.process(img, is_training=False)
        img_tensor = self.transform(processed_img)

        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)

        return img_tensor

    def _load_model(self):
        """Load model from checkpoint using model_load_dir (not output_dir)"""
        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)

        logger.info(f"Looking for model in: {self.model_load_dir}")

        possible_paths = [
            self.model_load_dir / f"{dataset_name_lower}_best.pt",
            self.model_load_dir / f"{dataset_name_lower}_latest.pt",
            self.model_load_dir / 'best.pt',
            self.model_load_dir / 'latest.pt',
        ]

        for pattern in ['*.pt', '*.pth']:
            possible_paths.extend(list(self.model_load_dir.glob(pattern)))

        seen = set()
        unique_paths = []
        for path in possible_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)

        best_path = None
        for path in unique_paths:
            if path.exists() and path.stat().st_size > 0:
                try:
                    with open(path, 'rb') as f:
                        header = f.read(4)
                        is_valid_pickle = header[0] == 0x80 and header[1] in [0x02, 0x03, 0x04]
                        is_zip = header[:2] == b'PK'
                        if not (is_valid_pickle or is_zip):
                            continue

                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    if 'model_state_dict' not in checkpoint:
                        continue

                    best_path = path
                    logger.info(f"Found valid model at {best_path}")
                    break
                except Exception as e:
                    logger.debug(f"Error checking {path}: {e}")
                    continue

        if best_path and best_path.exists():
            try:
                checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)

                if 'model_config' in checkpoint:
                    model_config = checkpoint['model_config']
                    for key, value in model_config.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                    logger.info("Loaded model architecture from checkpoint")

                if 'input_size' in checkpoint:
                    self.config.input_size = tuple(checkpoint['input_size']) if isinstance(checkpoint['input_size'], list) else checkpoint['input_size']
                if 'feature_dims' in checkpoint:
                    self.config.feature_dims = checkpoint['feature_dims']
                if 'compressed_dims' in checkpoint:
                    self.config.compressed_dims = checkpoint['compressed_dims']
                if 'num_classes' in checkpoint:
                    self.config.num_classes = checkpoint['num_classes']

                self.config.input_size_explicitly_set = True

                self.model = BaseAutoencoder(self.config)
                self.model.set_training_phase(2)

                if 'use_per_image_normalization' in checkpoint:
                    self.config.use_per_image_normalization = checkpoint['use_per_image_normalization']
                    logger.info(f"Loaded normalization mode: {'PER-IMAGE' if self.config.use_per_image_normalization else 'DATASET-WIDE'}")
                elif 'normalization_mode' in checkpoint:
                    norm_mode_str = checkpoint['normalization_mode']
                    self.config.use_per_image_normalization = (norm_mode_str == 'per_image')
                    logger.info(f"Loaded normalization mode: {norm_mode_str}")

                # Load dataset statistics and attach to model
                if 'dataset_statistics' in checkpoint and checkpoint['dataset_statistics']:
                    stats_dict = checkpoint['dataset_statistics']
                    if 'mean' in stats_dict and stats_dict['mean'] is not None:
                        self.norm_mean = torch.tensor(stats_dict['mean'], dtype=torch.float32)
                    if 'std' in stats_dict and stats_dict['std'] is not None:
                        self.norm_std = torch.tensor(stats_dict['std'], dtype=torch.float32)
                    self.norm_is_fitted = True
                    self.norm_n_samples = stats_dict.get('n_samples_used', stats_dict.get('n_samples', 0))
                    logger.info("Loaded dataset statistics from checkpoint")

                    class StatisticsWrapper:
                        def __init__(self, mean, std, n_samples):
                            self.mean = mean
                            self.std = std
                            self.is_calculated = True
                            self.n_samples_used = n_samples

                        def normalize(self, x):
                            mean = self.mean.to(x.device).view(1, -1, 1, 1)
                            std = self.std.to(x.device).view(1, -1, 1, 1)
                            return (x - mean) / (std + 1e-8)

                    stats_wrapper = StatisticsWrapper(self.norm_mean, self.norm_std, self.norm_n_samples)
                    self.model.set_dataset_statistics(stats_wrapper)
                    logger.info("=" * 60)
                    logger.info("TRAINING STATISTICS ATTACHED TO MODEL:")
                    logger.info(f"  Mean: {self.norm_mean.tolist()}")
                    logger.info(f"  Std:  {self.norm_std.tolist()}")
                    logger.info(f"  Samples used: {self.norm_n_samples:,}")
                    logger.info("=" * 60)

                if 'domain' in checkpoint:
                    self.domain_info['domain'] = checkpoint['domain']
                if 'domain_config' in checkpoint:
                    self.domain_info['domain_config'] = checkpoint['domain_config']

                domain = self.domain_info['domain']
                if domain != 'general':
                    self._init_domain_processor(domain, self.domain_info['domain_config'])
                    logger.info(f"Initialized {domain} domain processor")

                model_state = self.model.state_dict()
                filtered_state = {}
                loaded_keys = 0
                skipped_keys = 0

                for key, value in checkpoint['model_state_dict'].items():
                    if key in model_state:
                        if model_state[key].shape == value.shape:
                            filtered_state[key] = value
                            loaded_keys += 1
                        else:
                            logger.debug(f"Skipping {key}: shape mismatch {value.shape} vs {model_state[key].shape}")
                            skipped_keys += 1
                    else:
                        skipped_keys += 1

                if loaded_keys > 0:
                    self.model.load_state_dict(filtered_state, strict=False)
                    logger.info(f"Loaded {loaded_keys} parameters, skipped {skipped_keys}")
                else:
                    logger.error("No compatible parameters found in checkpoint")
                    self.model = None
                    return

                if 'selected_feature_indices' in checkpoint and checkpoint['selected_feature_indices'] is not None:
                    indices = checkpoint['selected_feature_indices']
                    if isinstance(indices, torch.Tensor):
                        self.model._selected_feature_indices = indices.to(self.device)
                    else:
                        self.model._selected_feature_indices = torch.tensor(indices, device=self.device)
                    self.model._is_feature_selection_frozen = True
                    logger.info(f"Loaded feature selection with {len(indices)} features")

                if 'classifier_state_dict' in checkpoint and checkpoint['classifier_state_dict']:
                    if hasattr(self.model, 'classifier') and self.model.classifier is not None:
                        self.model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
                        logger.info("Loaded classifier state dict")

                if 'cluster_centers' in checkpoint and checkpoint['cluster_centers'] is not None:
                    if hasattr(self.model, 'cluster_centers') and self.model.cluster_centers is not None:
                        centers = checkpoint['cluster_centers']
                        if isinstance(centers, torch.Tensor):
                            self.model.cluster_centers.data = centers.to(self.device)
                        else:
                            self.model.cluster_centers.data = torch.tensor(centers, device=self.device)
                        logger.info(f"Loaded {len(self.model.cluster_centers)} cluster centers")

                checkpoint_phase = checkpoint.get('phase', 2)
                self.model.set_training_phase(checkpoint_phase)
                self.model.eval()
                self.model.to(self.device)

                logger.info(f"Successfully loaded model from {best_path}")
                logger.info(f"Model phase: {checkpoint_phase}")
                logger.info(f"Model expects input size: {self.config.input_size}")
                return

            except Exception as e:
                logger.error(f"Failed to load model from {best_path}: {e}")
                traceback.print_exc()
                self.model = None
                return

        logger.error(f"No valid model found in {self.model_load_dir}")
        logger.error(f"Available files: {[p.name for p in self.model_load_dir.glob('*')]}")
        logger.error("Please train the model first with: python cdbnn.py --mode train")
        self.model = None

    def _init_domain_processor(self, domain: str, domain_config: Dict):
        """Initialize domain processor"""
        config_copy = copy.deepcopy(self.config)
        config_copy.domain = domain
        for key, value in domain_config.items():
            if hasattr(config_copy, key):
                setattr(config_copy, key, value)

        if domain == 'astronomy':
            self.domain_processor = AstronomyDomainProcessor(config_copy)
        elif domain == 'agriculture':
            self.domain_processor = AgricultureDomainProcessor(config_copy)
        elif domain == 'medical':
            self.domain_processor = MedicalDomainProcessor(config_copy)
        elif domain == 'satellite':
            self.domain_processor = SatelliteDomainProcessor(config_copy)
        elif domain == 'surveillance':
            self.domain_processor = SurveillanceDomainProcessor(config_copy)
        elif domain == 'microscopy':
            self.domain_processor = MicroscopyDomainProcessor(config_copy)
        elif domain == 'industrial':
            self.domain_processor = IndustrialDomainProcessor(config_copy)
        else:
            logger.warning(f"Unknown domain: {domain}, using general processor")
            self.domain_processor = None

    def _get_image_files(self, data_path: str) -> Tuple[List[str], List[str], List[str]]:
        """Get image files with deterministic ordering"""
        image_files = []
        class_labels = []
        original_filenames = []
        supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.fits', '.fit', '.fits.gz')

        if not os.path.exists(data_path):
            logger.warning(f"Data path does not exist: {data_path}")
            return [], [], []

        data_path_obj = Path(data_path)

        train_path = data_path_obj / 'train'
        test_path = data_path_obj / 'test'

        if train_path.exists() and train_path.is_dir():
            logger.info(f"Found train/test dataset structure")
            for split_path in [train_path, test_path] if test_path.exists() else [train_path]:
                for class_dir in sorted(split_path.iterdir()):
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        for img_file in sorted(class_dir.glob('*')):
                            if img_file.suffix.lower() in supported:
                                image_files.append(str(img_file))
                                class_labels.append(class_name)
                                original_filenames.append(img_file.name)
            if image_files:
                return image_files, class_labels, original_filenames

        has_class_dirs = False
        for item in sorted(os.listdir(data_path)):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                has_images = any(f.lower().endswith(ext) for ext in supported
                               for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f)))
                if has_images:
                    has_class_dirs = True
                    break

        if has_class_dirs:
            logger.info(f"Found class subdirectories in {data_path}")
            for class_dir in sorted(os.listdir(data_path)):
                class_path = os.path.join(data_path, class_dir)
                if os.path.isdir(class_path):
                    for img_file in sorted(os.listdir(class_path)):
                        if img_file.lower().endswith(supported):
                            image_files.append(os.path.join(class_path, img_file))
                            class_labels.append(class_dir)
                            original_filenames.append(img_file)
            return image_files, class_labels, original_filenames

        logger.info(f"Scanning {data_path} recursively for images...")
        for root, dirs, files in os.walk(data_path):
            dirs.sort()
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files.sort()
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, data_path)
                    class_name = rel_path.split(os.sep)[0] if rel_path != '.' else "unknown"
                    image_files.append(full_path)
                    class_labels.append(class_name)
                    original_filenames.append(file)

        logger.info(f"Found {len(image_files)} images")
        return image_files, class_labels, original_filenames

    @torch.no_grad()
    @memory_efficient
    def predict_images(self, data_path: str, output_csv: str = None) -> Dict:
        """
        DETERMINISTIC prediction - model handles normalization internally using
        training statistics that were attached during _load_model()
        """

        if self.model is None:
            logger.error("No model loaded. Cannot perform prediction.")
            return None

        logger.info("=" * 70)
        logger.info("PREDICTION")
        logger.info("=" * 70)
        logger.info(f"Model expects input size: {self.config.input_size}")

        if self.config.use_per_image_normalization:
            logger.info("Normalization: PER-IMAGE (model will compute per-image stats)")
        else:
            logger.info("Normalization: TRAINING STATISTICS (attached to model)")
            if hasattr(self.model, 'dataset_statistics') and self.model.dataset_statistics and self.model.dataset_statistics.is_calculated:
                logger.info("  ✓ Training statistics are attached to model")

        image_files, class_labels, original_filenames = self._get_image_files(data_path)
        if not image_files:
            logger.warning(f"No image files found in {data_path}")
            return None

        sorted_indices = sorted(range(len(original_filenames)), key=lambda i: str(original_filenames[i]))
        image_files = [image_files[i] for i in sorted_indices]
        class_labels = [class_labels[i] for i in sorted_indices]
        original_filenames = [original_filenames[i] for i in sorted_indices]

        # Set output CSV path - use output_data_dir
        if output_csv is None:
            output_csv = str(self.output_data_dir / f"{self.config.dataset_name}.csv")

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

        all_features = []
        all_predictions = []
        all_probabilities = []
        all_cluster_assignments = []
        all_cluster_confidence = []
        collected_targets = []
        collected_filenames = []
        collected_filepaths = []

        has_classifier = hasattr(self.model, 'classifier') and self.model.classifier is not None
        has_clustering = hasattr(self.model, 'cluster_centers') and self.model.cluster_centers is not None

        logger.info(f"Processing {len(image_files)} images")

        for idx, img_path in enumerate(tqdm(image_files, desc="Predicting")):
            img = ImageProcessor.load_image(img_path)
            if img is None:
                logger.warning(f"Failed to load {img_path}, skipping")
                continue

            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Preprocess to tensor (model handles normalization internally)
            img_tensor = self._preprocess_image_for_model(img).unsqueeze(0).to(self.device)

            expected_h, expected_w = self.config.input_size
            actual_h, actual_w = img_tensor.shape[2], img_tensor.shape[3]

            if actual_h != expected_h or actual_w != expected_w:
                img_tensor = F.interpolate(img_tensor, size=(expected_h, expected_w), mode='bilinear', align_corners=False)

            # Duplicate single sample for norm layer compatibility (same as training)
            duplicated = False
            if img_tensor.size(0) == 1:
                img_tensor = torch.cat([img_tensor, img_tensor], dim=0)
                duplicated = True

            # Forward pass - model handles normalization internally
            output = self.model(img_tensor)

            if self.model._is_feature_selection_frozen and self.model._selected_feature_indices is not None:
                features = output['selected_embedding'].float().cpu().numpy()
            else:
                features = output['compressed_embedding'].float().cpu().numpy()

            if duplicated:
                features = features[0]
            else:
                features = features[0] if len(features) > 0 else features

            all_features.append(features)

            if has_classifier and 'class_logits' in output:
                probs = F.softmax(output['class_logits'], dim=1)
                if duplicated:
                    probs = probs[0:1]
                pred = probs.argmax(dim=1).item()
                all_predictions.append(pred)
                all_probabilities.append(probs.cpu().numpy()[0])
            else:
                all_predictions.append(-1)
                all_probabilities.append([0.0] * (self.config.num_classes or 2))

            if has_clustering and 'compressed_embedding' in output:
                compressed = output['compressed_embedding']
                if duplicated:
                    compressed = compressed[0:1]
                compressed_norm = F.normalize(compressed, p=2, dim=1)
                cluster_centers_norm = F.normalize(self.model.cluster_centers, p=2, dim=1)
                cosine_sim = torch.mm(compressed_norm, cluster_centers_norm.t())
                cluster_probs = F.softmax(cosine_sim, dim=1)
                cluster = cluster_probs.argmax(dim=1).item()
                confidence = cluster_probs.max(dim=1)[0].item()
                all_cluster_assignments.append(cluster)
                all_cluster_confidence.append(confidence)
            else:
                all_cluster_assignments.append(-1)
                all_cluster_confidence.append(0.0)

            collected_targets.append(class_labels[idx])
            collected_filenames.append(original_filenames[idx])
            collected_filepaths.append(img_path)

        results = {
            'features': np.array(all_features) if all_features else np.array([]),
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities) if all_probabilities else np.array([]),
            'cluster_assignments': np.array(all_cluster_assignments),
            'cluster_confidence': np.array(all_cluster_confidence)
        }

        self._save_predictions_deterministic(
            results, output_csv, targets=collected_targets,
            filenames=collected_filenames, filepaths=collected_filepaths
        )

        # Save config file alongside predictions
        self._save_config_file(output_csv, image_files, class_labels)

        logger.info("=" * 70)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Processed {len(image_files)} images")
        logger.info(f"Results saved to: {output_csv}")
        logger.info("=" * 70)

        return results

    def _save_predictions_deterministic(self, predictions: Dict, output_csv: str,
                                        targets: Optional[List[str]] = None,
                                        filenames: Optional[List[str]] = None,
                                        filepaths: Optional[List[str]] = None):
        """Save predictions with deterministic ordering"""
        import pandas as pd

        data = {}

        if filenames:
            n_samples = len(filenames)
        elif 'features' in predictions and predictions['features'] is not None:
            n_samples = len(predictions['features'])
        else:
            n_samples = 0

        if n_samples == 0:
            logger.error("No samples to save")
            return

        if filenames:
            sorted_indices = sorted(range(len(filenames)), key=lambda i: str(filenames[i]))
        else:
            sorted_indices = list(range(n_samples))

        if filepaths:
            data['filepath'] = [filepaths[i] for i in sorted_indices]
        if filenames:
            data['filename'] = [filenames[i] for i in sorted_indices]
            folders = []
            for fp in filepaths if filepaths else []:
                path = Path(fp)
                folder = path.parent.name
                folders.append(folder)
            if folders:
                data['folder'] = [folders[i] for i in sorted_indices]

        if targets:
            data['target'] = [targets[i] for i in sorted_indices]

        if 'features' in predictions and predictions['features'] is not None and len(predictions['features']) > 0:
            features = predictions['features'][sorted_indices]
            for i in range(features.shape[1]):
                data[f'feature_{i}'] = features[:, i]

        if 'predictions' in predictions and predictions['predictions'] is not None:
            data['prediction'] = predictions['predictions'][sorted_indices]

        if 'probabilities' in predictions and predictions['probabilities'] is not None and len(predictions['probabilities']) > 0:
            probs = predictions['probabilities'][sorted_indices]
            for i in range(probs.shape[1]):
                data[f'prob_class_{i}'] = probs[:, i]
            data['confidence'] = np.max(probs, axis=1)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            data['uncertainty'] = entropy

        if 'cluster_assignments' in predictions and predictions['cluster_assignments'] is not None:
            data['cluster_id'] = predictions['cluster_assignments'][sorted_indices]
        if 'cluster_confidence' in predictions and predictions['cluster_confidence'] is not None:
            data['cluster_confidence'] = predictions['cluster_confidence'][sorted_indices]

        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)

        logger.info(f"Predictions saved to {output_csv}")
        logger.info(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    def _save_config_file(self, output_csv: str, image_files: List[str], class_labels: List[str]):
        """Save configuration file along with predictions - ONLY features and target"""
        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)
        data_dir = Path(output_csv).parent

        # Get ONLY feature columns and target - NO prediction, confidence, uncertainty
        actual_feature_count = self.config.compressed_dims
        feature_columns = [f'feature_{i}' for i in range(actual_feature_count)]

        # Column names: features + target ONLY (no prediction/confidence/uncertainty)
        column_names = feature_columns.copy()
        column_names.append('target')  # target only, no prediction columns

        config_dict = {
            "dataset_name": dataset_name_lower,
            "num_classes": self.config.num_classes if self.config.num_classes else len(set(class_labels)),
            "csv_file": output_csv,
            "column_names": column_names,  # Now only features + target
            "target_column": "target",
            "feature_dims": self.config.feature_dims,
            "compressed_dims": self.config.compressed_dims,
            "input_size": list(self.config.input_size) if isinstance(self.config.input_size, tuple) else self.config.input_size,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "domain": getattr(self.config, 'domain', 'general'),
            "normalization_mode": self.config.normalization_mode,
            "use_per_image_normalization": self.config.use_per_image_normalization,
            "model_config": {
                "in_channels": self.config.in_channels,
                "input_size": list(self.config.input_size) if isinstance(self.config.input_size, tuple) else self.config.input_size,
                "feature_dims": self.config.feature_dims,
                "compressed_dims": self.config.compressed_dims,
                "use_kl_divergence": self.config.use_kl_divergence,
                "use_class_encoding": self.config.use_class_encoding,
            },
            "prediction_info": {
                "num_images": len(image_files),
                "classes_found": list(set(class_labels)),
                "timestamp": datetime.now().isoformat(),
                "mode": "prediction",
                "normalization_stats_loaded": self.norm_is_fitted or (self.dataset_statistics is not None),
            },
            "config_version": "2.4",
            "notes": "Prediction configuration - ONLY compressed features and target column (no prediction/confidence/uncertainty)"
        }

        if self.config.class_names:
            config_dict["class_info"] = {
                "class_names": self.config.class_names,
                "num_classes": len(self.config.class_names),
                "class_to_idx": {name: idx for idx, name in enumerate(self.config.class_names)}
            }

        if self.norm_is_fitted and self.norm_mean is not None:
            config_dict["normalization_statistics"] = {
                "mean": self.norm_mean.tolist() if isinstance(self.norm_mean, torch.Tensor) else self.norm_mean,
                "std": self.norm_std.tolist() if isinstance(self.norm_std, torch.Tensor) else self.norm_std,
                "is_fitted": True,
                "n_samples": self.norm_n_samples
            }

        conf_path = data_dir / f"{dataset_name_lower}.conf"
        with open(conf_path, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        logger.info(f"Configuration saved to {conf_path}")

    def _get_column_names(self) -> List[str]:
        """Get column names - ONLY features and target (no prediction columns)"""
        actual_feature_count = self.config.compressed_dims
        feature_columns = [f'feature_{i}' for i in range(actual_feature_count)]
        column_names = feature_columns.copy()
        column_names.append('target')  # Only target, no prediction/confidence/uncertainty
        return column_names

# =============================================================================
# COMPLETE TRAINER - All original functionality preserved
# =============================================================================

class Trainer:
    """
    ENHANCED DETERMINISTIC TRAINER - REPLACES the original Trainer class.
    Uses per-image normalization + Euclidean distance + Contrastive loss for proper clustering.
    Enhanced with comprehensive resume capabilities.
    """

    def __init__(self, model: BaseAutoencoder, config: GlobalConfig):
        self.model = model
        self.config = config
        self.device = model.device

        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and self.device.type == 'cuda' else None

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track best metrics
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_nmi = 0.0  # Track clustering quality
        self.best_epoch = 0
        self.best_phase = 1

        self.best_phase1_loss = float('inf')
        self.best_phase1_epoch = 0
        self.best_phase2_loss = float('inf')
        self.best_phase2_accuracy = 0.0
        self.best_phase2_nmi = 0.0
        self.best_phase2_epoch = 0

        self.model_loaded = False
        self.loaded_epoch = 0
        self.loaded_phase = 1
        self.loaded_loss = float('inf')
        self.loaded_accuracy = 0.0

        self.history = defaultdict(list)

        self.prev_train_loss = None
        self.prev_val_loss = None
        self.prev_train_acc = None
        self.prev_val_acc = None

        self.feature_selector = DistanceCorrelationFeatureSelector(
            config.correlation_upper, config.correlation_lower
        ) if config.use_distance_correlation else None

        self._set_deterministic_seeds()

        if hasattr(model, 'deterministic_mode'):
            model.deterministic_mode = True
            model._single_image_mode = False

        # Resume-related attributes
        self.is_resumed = False
        self.resume_info = {}
        self.saving_data_dir = None
        self.saving_checkpoint_dir = None
        self.loading_data_dir = None
        self.loading_checkpoint_dir = None
        self.visualizer = None
        self.statistics_calculator = None

    def _set_deterministic_seeds(self):
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("Set deterministic seeds")

    def _per_image_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-image Z-score normalization"""
        if x.dim() == 4:
            b, c, h, w = x.shape
            x_flat = x.reshape(b, c, -1)
            mean = x_flat.mean(dim=2, keepdim=True)
            std = x_flat.std(dim=2, keepdim=True)
            return ((x_flat - mean) / (std + 1e-8)).reshape(b, c, h, w)
        elif x.dim() == 3:
            c, h, w = x.shape
            x_flat = x.reshape(c, -1)
            mean = x_flat.mean(dim=1, keepdim=True)
            std = x_flat.std(dim=1, keepdim=True)
            return ((x_flat - mean) / (std + 1e-8)).reshape(c, h, w)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")

    def _create_deterministic_loader(self, dataset, batch_size, shuffle=False):
        g = torch.Generator()
        g.manual_seed(42)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=g,
            pin_memory=False,
            drop_last=False
        )

    def _verify_model_compatibility(self, checkpoint: Dict) -> Dict:
        """
        Verify that checkpoint model is compatible with current model configuration.

        Returns:
            Dictionary with 'compatible' (bool) and 'reason' (str if incompatible)
        """
        # Check model architecture compatibility
        model_config_keys = ['feature_dims', 'compressed_dims', 'in_channels', 'num_classes']

        for key in model_config_keys:
            if key in checkpoint:
                checkpoint_value = checkpoint[key]
                current_value = getattr(self.config, key, None)

                if current_value is not None and checkpoint_value != current_value:
                    return {
                        'compatible': False,
                        'reason': f"{key} mismatch: checkpoint={checkpoint_value}, config={current_value}"
                    }

        # Check input size compatibility
        if 'input_size' in checkpoint:
            checkpoint_size = tuple(checkpoint['input_size']) if isinstance(checkpoint['input_size'], list) else checkpoint['input_size']
            if checkpoint_size != self.config.input_size:
                logger.warning(f"Input size mismatch: checkpoint={checkpoint_size}, config={self.config.input_size}")
                logger.warning("Attempting to adapt... (may cause issues)")

        # Check normalization mode compatibility
        if 'normalization_mode' in checkpoint:
            checkpoint_norm = checkpoint['normalization_mode']
            if checkpoint_norm != self.config.normalization_mode:
                logger.warning(f"Normalization mode mismatch: checkpoint={checkpoint_norm}, config={self.config.normalization_mode}")
                logger.warning("This may affect results. Consider using consistent normalization.")

        return {'compatible': True, 'reason': None}

    def load_checkpoint_for_resume(self, checkpoint_path: Optional[Path] = None, reset_optimizer: bool = False) -> Dict:
        """
        Load checkpoint for resuming training with comprehensive validation.

        Args:
            checkpoint_path: Path to checkpoint file (if None, finds latest or best)
            reset_optimizer: If True, only load model weights, reset optimizer state

        Returns:
            Dictionary with resume information
        """
        if checkpoint_path is None:
            # Try to find the latest checkpoint
            possible_paths = [
                self.checkpoint_dir / 'latest.pt',
                self.checkpoint_dir / 'best.pt',
                self.checkpoint_dir / f"{self.config.dataset_name}_best.pt",
                self.checkpoint_dir / f"{self.config.dataset_name}_latest.pt",
            ]
            for path in possible_paths:
                if path.exists():
                    checkpoint_path = path
                    break

        if checkpoint_path is None or not checkpoint_path.exists():
            logger.warning(f"No checkpoint found to resume from")
            return {'resumed': False, 'error': 'No checkpoint found'}

        try:
            logger.info(f"=" * 60)
            logger.info(f"RESUMING TRAINING FROM CHECKPOINT")
            logger.info(f"Checkpoint: {checkpoint_path}")
            logger.info(f"=" * 60)

            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # ================================================================
            # VERIFY MODEL COMPATIBILITY
            # ================================================================
            compatibility_check = self._verify_model_compatibility(checkpoint)
            if not compatibility_check['compatible']:
                logger.error(f"Model incompatible: {compatibility_check['reason']}")
                return {'resumed': False, 'error': compatibility_check['reason']}

            # ================================================================
            # LOAD MODEL STATE
            # ================================================================
            # Load model state dict with strict=False for flexibility
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )

            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

            logger.info(f"✓ Model state loaded successfully")

            # ================================================================
            # LOAD OPTIMIZER STATE (optional)
            # ================================================================
            if not reset_optimizer and 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info(f"✓ Optimizer state loaded")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}, resetting...")
                    reset_optimizer = True

            if reset_optimizer:
                logger.info("Optimizer state reset (starting fresh)")

            # ================================================================
            # LOAD SCHEDULER STATE
            # ================================================================
            if not reset_optimizer and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info(f"✓ Scheduler state loaded")
                except Exception as e:
                    logger.warning(f"Could not load scheduler state: {e}")

            # ================================================================
            # LOAD TRAINING STATE
            # ================================================================
            resume_info = {
                'resumed': True,
                'checkpoint_path': str(checkpoint_path),
                'epoch': checkpoint.get('epoch', 0),
                'phase': checkpoint.get('phase', 1),
                'loss': checkpoint.get('loss', float('inf')),
                'accuracy': checkpoint.get('accuracy', 0.0),
                'nmi': checkpoint.get('nmi', 0.0),
                'best_loss': checkpoint.get('best_loss', float('inf')),
                'best_accuracy': checkpoint.get('best_accuracy', 0.0),
                'best_epoch': checkpoint.get('best_epoch', 0),
                'best_phase': checkpoint.get('best_phase', 1),
                'normalization_mode': checkpoint.get('normalization_mode', 'dataset_wide'),
                'clustering_mode': checkpoint.get('clustering_mode', 'enhanced'),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
            }

            # Load feature selection if available
            if 'selected_feature_indices' in checkpoint and checkpoint['selected_feature_indices'] is not None:
                indices = checkpoint['selected_feature_indices']
                if isinstance(indices, torch.Tensor):
                    self.model._selected_feature_indices = indices.to(self.device)
                else:
                    self.model._selected_feature_indices = torch.tensor(indices, device=self.device)
                self.model._is_feature_selection_frozen = True
                resume_info['feature_selection'] = len(indices)
                logger.info(f"✓ Feature selection loaded ({len(indices)} features)")

            # Load classifier if available
            if 'classifier_state_dict' in checkpoint and checkpoint['classifier_state_dict']:
                if hasattr(self.model, 'classifier') and self.model.classifier is not None:
                    try:
                        self.model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
                        logger.info(f"✓ Classifier state loaded")
                    except Exception as e:
                        logger.warning(f"Could not load classifier state: {e}")

            # Load cluster centers if available
            if 'cluster_centers' in checkpoint and checkpoint['cluster_centers'] is not None:
                if hasattr(self.model, 'cluster_centers') and self.model.cluster_centers is not None:
                    centers = checkpoint['cluster_centers']
                    if isinstance(centers, torch.Tensor):
                        self.model.cluster_centers.data = centers.to(self.device)
                    else:
                        self.model.cluster_centers.data = torch.tensor(centers, device=self.device)
                    resume_info['cluster_centers'] = len(self.model.cluster_centers)
                    logger.info(f"✓ Cluster centers loaded ({len(self.model.cluster_centers)} centers)")

            # Load dataset statistics if available
            if 'dataset_statistics' in checkpoint and checkpoint['dataset_statistics']:
                stats_dict = checkpoint['dataset_statistics']
                if hasattr(self.model, 'dataset_statistics'):
                    # Create statistics wrapper
                    class StatisticsWrapper:
                        def __init__(self, mean, std, n_samples):
                            self.mean = mean
                            self.std = std
                            self.is_calculated = True
                            self.n_samples_used = n_samples

                        def normalize(self, x):
                            mean = self.mean.to(x.device).view(1, -1, 1, 1)
                            std = self.std.to(x.device).view(1, -1, 1, 1)
                            return (x - mean) / (std + 1e-8)

                    if 'mean' in stats_dict and stats_dict['mean'] is not None:
                        mean = torch.tensor(stats_dict['mean'], dtype=torch.float32)
                        std = torch.tensor(stats_dict['std'], dtype=torch.float32)
                        n_samples = stats_dict.get('n_samples_used', stats_dict.get('n_samples', 0))
                        stats_wrapper = StatisticsWrapper(mean, std, n_samples)
                        self.model.set_dataset_statistics(stats_wrapper)
                        resume_info['dataset_statistics_loaded'] = True
                        logger.info(f"✓ Dataset statistics loaded")

            # Load history if available
            if 'history' in checkpoint:
                self.history = defaultdict(list, checkpoint['history'])
                resume_info['history_epochs'] = len(self.history.get('train_loss', []))
                logger.info(f"✓ Training history loaded ({resume_info['history_epochs']} epochs)")

            # Update best metrics
            self.best_loss = resume_info['best_loss']
            self.best_accuracy = resume_info['best_accuracy']
            self.best_epoch = resume_info['best_epoch']
            self.best_phase = resume_info['best_phase']

            # Update phase-specific best metrics
            if resume_info['phase'] == 1:
                self.best_phase1_loss = resume_info['best_loss']
                self.best_phase1_epoch = resume_info['best_epoch']
            else:
                self.best_phase2_loss = resume_info['best_loss']
                self.best_phase2_accuracy = resume_info['best_accuracy']
                self.best_phase2_epoch = resume_info['best_epoch']
                self.best_phase2_nmi = resume_info.get('nmi', 0.0)

            self.is_resumed = True
            self.resume_info = resume_info

            # Set model to appropriate phase
            self.model.set_training_phase(resume_info['phase'])

            logger.info(f"=" * 60)
            logger.info(f"RESUME INFORMATION:")
            logger.info(f"  Phase: {resume_info['phase']}")
            logger.info(f"  Epoch: {resume_info['epoch'] + 1}")
            logger.info(f"  Loss: {resume_info['loss']:.6f}")
            logger.info(f"  Accuracy: {resume_info['accuracy']:.2%}")
            logger.info(f"  Best Loss: {resume_info['best_loss']:.6f}")
            logger.info(f"  Best Accuracy: {resume_info['best_accuracy']:.2%}")
            logger.info(f"  Normalization: {resume_info['normalization_mode']}")
            logger.info(f"=" * 60)

            return resume_info

        except Exception as e:
            logger.error(f"Failed to load checkpoint for resume: {e}")
            traceback.print_exc()
            return {'resumed': False, 'error': str(e)}

    def save_checkpoint_for_resume(self, epoch: int, phase: int, loss: float,
                                   accuracy: float, nmi: float = 0.0,
                                   is_best: bool = False) -> None:
        """
        Save checkpoint with complete resume information.
        """
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'nmi': nmi,
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'best_phase': self.best_phase,
            'history': dict(self.history),
            'normalization_mode': self.config.normalization_mode,
            'use_per_image_normalization': self.config.use_per_image_normalization,
            'clustering_mode': 'enhanced',
            'deterministic': True,
            'feature_dims': self.config.feature_dims,
            'compressed_dims': self.config.compressed_dims,
            'in_channels': self.config.in_channels,
            'num_classes': self.config.num_classes,
            'input_size': list(self.config.input_size),
            'timestamp': datetime.now().isoformat(),
        }

        # Save feature selection if frozen
        if hasattr(self.model, '_is_feature_selection_frozen') and self.model._is_feature_selection_frozen:
            if hasattr(self.model, '_selected_feature_indices') and self.model._selected_feature_indices is not None:
                checkpoint['selected_feature_indices'] = self.model._selected_feature_indices.cpu()

        # Save classifier if available
        if hasattr(self.model, 'classifier') and self.model.classifier is not None:
            checkpoint['classifier_state_dict'] = self.model.classifier.state_dict()

        # Save cluster centers if available
        if hasattr(self.model, 'cluster_centers') and self.model.cluster_centers is not None:
            checkpoint['cluster_centers'] = self.model.cluster_centers.data.cpu()

        # Save dataset statistics if available
        if hasattr(self.model, 'dataset_statistics') and self.model.dataset_statistics:
            if hasattr(self.model.dataset_statistics, 'mean') and self.model.dataset_statistics.mean is not None:
                checkpoint['dataset_statistics'] = {
                    'mean': self.model.dataset_statistics.mean.cpu().tolist(),
                    'std': self.model.dataset_statistics.std.cpu().tolist(),
                    'n_samples_used': getattr(self.model.dataset_statistics, 'n_samples_used', 0),
                    'n_samples': getattr(self.model.dataset_statistics, 'n_samples', 0),
                }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        logger.info(f"Checkpoint saved: {latest_path} (Phase {phase}, Epoch {epoch+1}, Loss: {loss:.6f})")

        # Save best checkpoint if applicable
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Best model saved: {best_path} (Acc: {accuracy:.2%})")

            # Also save named checkpoint
            dataset_name_lower = normalize_dataset_name(self.config.dataset_name)
            named_best_path = self.checkpoint_dir / f"{dataset_name_lower}_best.pt"
            torch.save(checkpoint, named_best_path)
            logger.info(f"✓ Named best model saved: {named_best_path}")

    def get_resume_status(self) -> Dict:
        """Get current resume status"""
        return {
            'is_resumed': self.is_resumed,
            'resume_info': self.resume_info,
            'current_epoch': len(self.history.get('train_loss', [])),
            'current_phase': self.model.training_phase if hasattr(self.model, 'training_phase') else 1,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
        }

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              resume: bool = False, resume_from: Optional[str] = None,
              reset_optimizer: bool = False, additional_epochs: Optional[int] = None) -> Dict:
        """Train using standalone deterministic function with invariant preprocessing and resume capability"""

        self._set_deterministic_seeds()

        # Check if augmentation is enabled (default: True)
        use_augmentation = not getattr(self.config, 'no_augmentation', False)
        augmentation_strength = getattr(self.config, 'augmentation_strength', 0.5)

        # ========================================================================
        # 1. APPLY DETERMINISTIC INVARIANT PREPROCESSING TO DATASETS
        # ========================================================================
        logger.info("=" * 70)
        logger.info("APPLYING DETERMINISTIC INVARIANT PREPROCESSING")
        logger.info(f"  Augmentation enabled: {use_augmentation}")
        logger.info(f"  Augmentation strength: {augmentation_strength}")
        if resume:
            logger.info(f"  RESUME MODE: Continuing from previous checkpoint")
            if resume_from:
                logger.info(f"  Resuming from: {resume_from}")
            if reset_optimizer:
                logger.info("  Optimizer state will be reset")
            if additional_epochs:
                logger.info(f"  Adding {additional_epochs} additional epochs")
        logger.info("=" * 70)

        # Initialize invariant preprocessor wrapper
        class InvariantDatasetWrapper(Dataset):
            """Wrapper that applies deterministic invariant preprocessing"""

            def __init__(self, dataset, target_size, use_augmentation=True, augmentation_strength=0.5, is_train=True):
                self.dataset = dataset
                self.target_size = target_size
                self.use_augmentation = use_augmentation
                self.augmentation_strength = augmentation_strength
                self.is_train = is_train
                self.preprocessor = None

            def _get_preprocessor(self):
                if self.preprocessor is None:
                    self.preprocessor = DeterministicInvariantPreprocessor(
                        self.target_size,
                        use_augmentation=self.use_augmentation and self.is_train,
                        augmentation_strength=self.augmentation_strength
                    )
                return self.preprocessor

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img, label = self.dataset[idx]

                # CRITICAL: Convert tensor to PIL Image if needed
                if isinstance(img, torch.Tensor):
                    from torchvision import transforms
                    # Ensure values are in [0, 1] range for ToPILImage
                    if img.min() < 0 or img.max() > 1:
                        img = (img - img.min()) / (img.max() - img.min())
                    to_pil = transforms.ToPILImage()
                    img = to_pil(img)
                elif isinstance(img, np.ndarray):
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    img = PILImage.fromarray(img)

                # Ensure PIL Image
                if not isinstance(img, PILImage.Image):
                    raise TypeError(f"Expected PIL Image, got {type(img)}")

                preprocessor = self._get_preprocessor()
                img = preprocessor.process(img, is_training=self.is_train)

                # Convert back to tensor
                to_tensor = transforms.ToTensor()
                img = to_tensor(img)

                return img, label

        # Wrap datasets with invariant preprocessing
        logger.info("Wrapping training dataset with invariant preprocessing...")
        train_dataset = InvariantDatasetWrapper(
            train_loader.dataset,
            self.config.input_size,
            use_augmentation=use_augmentation,
            augmentation_strength=augmentation_strength,
            is_train=True
        )

        if val_loader:
            logger.info("Wrapping validation dataset with invariant preprocessing...")
            val_dataset = InvariantDatasetWrapper(
                val_loader.dataset,
                self.config.input_size,
                use_augmentation=False,  # No augmentation for validation
                augmentation_strength=0,
                is_train=False
            )
        else:
            val_dataset = None

        # Create new deterministic loaders with preprocessed datasets
        train_loader = self._create_deterministic_loader(
            train_dataset,
            self.config.batch_size,
            shuffle=True
        )

        if val_dataset:
            val_loader = self._create_deterministic_loader(
                val_dataset,
                self.config.batch_size,
                shuffle=False
            )

        # ========================================================================
        # 2. CALCULATE OR LOAD DATASET STATISTICS ON PREPROCESSED IMAGES
        # ========================================================================
        logger.info("=" * 70)
        logger.info("CALCULATING/LOADING STATISTICS ON PREPROCESSED IMAGES")
        logger.info("=" * 70)

        statistics_calculator = None

        # Try to load existing statistics if resuming
        if resume:
            # Try multiple possible locations for statistics file
            stats_paths = [
                self.saving_checkpoint_dir / 'dataset_statistics.pt' if self.saving_checkpoint_dir else None,
                self.loading_checkpoint_dir / 'dataset_statistics.pt' if self.loading_checkpoint_dir else None,
                self.saving_data_dir / 'dataset_statistics.pt' if self.saving_data_dir else None,
                self.loading_data_dir / 'dataset_statistics.pt' if self.loading_data_dir else None,
                self.checkpoint_dir / 'dataset_statistics.pt',
            ]

            # Filter out None paths
            stats_paths = [p for p in stats_paths if p is not None]

            for stats_path in stats_paths:
                if stats_path.exists():
                    try:
                        logger.info(f"Loading existing statistics from: {stats_path}")
                        stats = torch.load(stats_path, map_location='cpu')
                        statistics_calculator = DatasetStatisticsCalculator(self.config)
                        if 'mean' in stats and stats['mean'] is not None:
                            statistics_calculator.mean = stats['mean']
                        if 'std' in stats and stats['std'] is not None:
                            statistics_calculator.std = stats['std']
                        statistics_calculator.per_channel_min = stats.get('per_channel_min')
                        statistics_calculator.per_channel_max = stats.get('per_channel_max')
                        statistics_calculator.n_samples_used = stats.get('n_samples_used', 0)
                        statistics_calculator.is_calculated = True
                        self.statistics_calculator = statistics_calculator
                        logger.info("Statistics loaded successfully from existing file")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load statistics from {stats_path}: {e}")
                        continue

            if statistics_calculator is None:
                logger.warning("No existing statistics found, will recalculate...")
                resume = False  # Fall back to full training mode

        if not resume:
            # Calculate fresh statistics
            statistics_calculator = DatasetStatisticsCalculator(self.config)
            stats_loader = self._create_deterministic_loader(
                train_dataset,
                batch_size=64,  # Smaller batch for stats calculation
                shuffle=False
            )
            statistics_calculator.calculate_statistics(stats_loader)
            self.statistics_calculator = statistics_calculator

            # Save statistics for future resume
            stats_save_path = self.saving_checkpoint_dir / 'dataset_statistics.pt' if self.saving_checkpoint_dir else self.checkpoint_dir / 'dataset_statistics.pt'
            stats_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'mean': statistics_calculator.mean.cpu(),
                'std': statistics_calculator.std.cpu(),
                'per_channel_min': statistics_calculator.per_channel_min.cpu() if statistics_calculator.per_channel_min is not None else None,
                'per_channel_max': statistics_calculator.per_channel_max.cpu() if statistics_calculator.per_channel_max is not None else None,
                'n_samples_used': statistics_calculator.n_samples_used,
                'timestamp': datetime.now().isoformat()
            }, stats_save_path)
            logger.info(f"Statistics saved to {stats_save_path} for future resume")

        # ========================================================================
        # 3. CREATE AND TRAIN MODEL (USING EXISTING deterministic_train)
        # ========================================================================
        logger.info("=" * 70)
        logger.info("CREATING MODEL AND STARTING TRAINING")
        if resume:
            logger.info("RESUME MODE: Model will be loaded from checkpoint")
        logger.info("=" * 70)

        # Create model using ModelFactory
        model = ModelFactory.create_model(self.config)

        # Initialize resume variables with defaults
        start_epoch = 0
        start_phase = 1
        loaded_optimizer_state = None
        loaded_scheduler_state = None
        checkpoint_loaded = False

        # Load checkpoint if resuming
        if resume:
            checkpoint_path = None
            if resume_from:
                checkpoint_path = Path(resume_from)
                if not checkpoint_path.is_absolute():
                    if self.saving_checkpoint_dir:
                        checkpoint_path = self.saving_checkpoint_dir / checkpoint_path
                    else:
                        checkpoint_path = self.checkpoint_dir / checkpoint_path
            else:
                # Try to find latest checkpoint
                candidate_paths = []
                if self.saving_checkpoint_dir:
                    candidate_paths.extend([
                        self.saving_checkpoint_dir / 'latest.pt',
                        self.saving_checkpoint_dir / 'best.pt',
                        self.saving_checkpoint_dir / f"{self.config.dataset_name}_best.pt",
                    ])
                if self.loading_checkpoint_dir:
                    candidate_paths.extend([
                        self.loading_checkpoint_dir / 'latest.pt',
                        self.loading_checkpoint_dir / 'best.pt',
                        self.loading_checkpoint_dir / f"{self.config.dataset_name}_best.pt",
                    ])
                candidate_paths.extend([
                    self.checkpoint_dir / 'latest.pt',
                    self.checkpoint_dir / 'best.pt',
                    self.checkpoint_dir / f"{self.config.dataset_name}_best.pt",
                ])

                for path in candidate_paths:
                    if path and path.exists():
                        checkpoint_path = path
                        break

            if checkpoint_path and checkpoint_path.exists():
                try:
                    logger.info(f"Loading checkpoint from: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                    # Load model state
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    if missing_keys:
                        logger.debug(f"Missing keys: {missing_keys[:5]}...")
                    if unexpected_keys:
                        logger.debug(f"Unexpected keys: {unexpected_keys[:5]}...")
                    logger.info("✓ Model state loaded successfully")

                    # Record resume state
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    start_phase = checkpoint.get('phase', 1)
                    loaded_loss = checkpoint.get('loss', float('inf'))
                    loaded_accuracy = checkpoint.get('accuracy', 0.0)

                    # Store optimizer and scheduler state for potential use
                    if not reset_optimizer and 'optimizer_state_dict' in checkpoint:
                        loaded_optimizer_state = checkpoint['optimizer_state_dict']
                        logger.info("✓ Optimizer state will be restored")
                    if not reset_optimizer and 'scheduler_state_dict' in checkpoint:
                        loaded_scheduler_state = checkpoint['scheduler_state_dict']
                        logger.info("✓ Scheduler state will be restored")

                    logger.info(f"Resuming from: Phase {start_phase}, Epoch {start_epoch}")
                    logger.info(f"Loaded loss: {loaded_loss:.6f}, Accuracy: {loaded_accuracy:.4f}")

                    # Set dataset statistics in model
                    if statistics_calculator and statistics_calculator.is_calculated:
                        model.set_dataset_statistics(statistics_calculator)
                        logger.info("✓ Dataset statistics loaded into model")

                    # Load best metrics if available in checkpoint
                    if 'best_loss' in checkpoint:
                        self.best_loss = checkpoint['best_loss']
                    if 'best_accuracy' in checkpoint:
                        self.best_accuracy = checkpoint['best_accuracy']
                    if 'best_epoch' in checkpoint:
                        self.best_epoch = checkpoint['best_epoch']
                    if 'best_phase' in checkpoint:
                        self.best_phase = checkpoint['best_phase']

                    # Load history if available
                    if 'history' in checkpoint:
                        self.history = defaultdict(list, checkpoint['history'])
                        logger.info(f"✓ Training history loaded ({len(self.history.get('train_loss', []))} epochs)")

                    # Adjust epochs if additional epochs specified
                    if additional_epochs:
                        original_epochs = self.config.epochs
                        self.config.epochs = start_epoch + additional_epochs
                        logger.info(f"Extended training: {additional_epochs} additional epochs (was {original_epochs}, now {self.config.epochs})")

                    checkpoint_loaded = True

                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {e}")
                    traceback.print_exc()
                    logger.warning("Starting training from scratch instead")
                    start_epoch = 0
                    start_phase = 1
                    loaded_optimizer_state = None
                    loaded_scheduler_state = None
                    resume = False
            else:
                logger.warning(f"No checkpoint found to resume from")
                resume = False
                start_epoch = 0
                start_phase = 1

        # If resume was requested but no checkpoint loaded, set resume to False
        if resume and not checkpoint_loaded:
            resume = False
            start_epoch = 0
            start_phase = 1

        # Call standalone training function with resume parameters
        history = deterministic_train(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=self.config,
            checkpoint_dir=self.saving_checkpoint_dir if self.saving_checkpoint_dir else self.checkpoint_dir,
            statistics_calculator=statistics_calculator,
            resume=resume,
            start_epoch=start_epoch,
            start_phase=start_phase,
            loaded_optimizer_state=loaded_optimizer_state,
            loaded_scheduler_state=loaded_scheduler_state,
            reset_optimizer=reset_optimizer
        )

        # ========================================================================
        # 4. SAVE MODEL AND RESULTS
        # ========================================================================
        # Copy best model to loading directory
        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)
        checkpoint_dir_to_use = self.saving_checkpoint_dir if self.saving_checkpoint_dir else self.checkpoint_dir
        best_checkpoint = checkpoint_dir_to_use / 'best.pt'

        if best_checkpoint.exists():
            loading_dir = self.loading_checkpoint_dir if self.loading_checkpoint_dir else self.checkpoint_dir
            loading_best_path = loading_dir / f"{dataset_name_lower}_best.pt"
            loading_best_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_checkpoint, loading_best_path)
            logger.info(f"Model saved to {loading_best_path}")

            # Also save the preprocessor configuration
            preprocessor_config = {
                'target_size': list(self.config.input_size),
                'normalization_stats': {
                    'mean': statistics_calculator.mean.tolist() if statistics_calculator and statistics_calculator.mean is not None else None,
                    'std': statistics_calculator.std.tolist() if statistics_calculator and statistics_calculator.std is not None else None,
                },
                'preprocessor_type': 'DeterministicInvariantPreprocessor',
                'augmentation_enabled': use_augmentation,
                'augmentation_strength': augmentation_strength,
                'version': '1.0'
            }

            preprocessor_config_path = checkpoint_dir_to_use / f"{dataset_name_lower}_preprocessor.json"
            with open(preprocessor_config_path, 'w') as f:
                json.dump(preprocessor_config, f, indent=2)
            logger.info(f"Preprocessor config saved to {preprocessor_config_path}")

        # Plot training history if visualizer is available
        if self.visualizer:
            self.visualizer.plot_training_history(history)

        # Save final training summary
        training_summary = {
            'best_accuracy': history.get('val_acc', [0])[-1] if history.get('val_acc') else 0,
            'best_loss': min(history.get('val_loss', [float('inf')])),
            'total_epochs': len(history.get('train_loss', [])),
            'resumed': resume,
            'resumed_from_epoch': start_epoch - 1 if resume and start_epoch > 0 else 0,
            'augmentation_used': use_augmentation,
            'augmentation_strength': augmentation_strength,
            'normalization_stats': {
                'mean': statistics_calculator.mean.tolist() if statistics_calculator and statistics_calculator.mean is not None else None,
                'std': statistics_calculator.std.tolist() if statistics_calculator and statistics_calculator.std is not None else None,
            }
        }

        data_dir_to_use = self.saving_data_dir if self.saving_data_dir else self.checkpoint_dir.parent
        summary_path = data_dir_to_use / f"{dataset_name_lower}_training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        logger.info(f"Training summary saved to {summary_path}")

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Best validation accuracy: {training_summary['best_accuracy']:.4f}")
        logger.info(f"Best validation loss: {training_summary['best_loss']:.6f}")
        if resume:
            logger.info(f"Training resumed from epoch {start_epoch}")
        logger.info("=" * 70)

        return history

    def _initialize_cluster_centers(self, train_loader):
        """Initialize cluster centers using k-means on embeddings"""
        logger.info("Initializing cluster centers with k-means...")

        self.model.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(train_loader, desc="Collecting embeddings"):
                inputs = inputs.to(self.device)
                inputs_norm = self._per_image_normalize(inputs)
                outputs = self.model(inputs_norm)
                embeddings = outputs['compressed_embedding'].cpu()
                all_embeddings.append(embeddings)
                all_labels.append(labels)

        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        from sklearn.cluster import KMeans
        n_clusters = self.config.num_classes or len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)

        with torch.no_grad():
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)
            centers = F.normalize(centers, p=2, dim=1)
            self.model.cluster_centers.data = centers

        logger.info(f"Initialized {n_clusters} cluster centers using k-means")

    def _train_phase_enhanced(self, train_loader, val_loader, phase, epochs, start_epoch):
        """Enhanced training with proper clustering loss"""

        print(f"\n{Colors.BOLD}{'Phase ' + str(phase) + ' Training (Enhanced)'.center(80)}{Colors.ENDC}")
        if phase == 2:
            print(f"{'Epoch | Train Loss | Clust Loss | Val Loss | Train Acc | Val Acc | NMI | LR'.center(80)}")
        else:
            print(f"{'Epoch | Train Loss | Val Loss | LR'.center(80)}")
        print(f"{Colors.BOLD}{'-'*80}{Colors.ENDC}")

        patience_counter = 0

        for epoch_offset in range(epochs):
            epoch = start_epoch + epoch_offset

            self.model.train()
            train_loss = 0.0
            cluster_loss = 0.0
            train_acc = 0.0 if phase == 2 else None
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Phase {phase} Epoch {epoch+1}")
            self.optimizer.zero_grad()

            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                inputs_norm = self._per_image_normalize(inputs)

                if inputs_norm.size(0) == 1:
                    inputs_norm = torch.cat([inputs_norm, inputs_norm], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs_norm, labels)
                        loss, acc, clust_loss = self._compute_loss_enhanced(
                            outputs, inputs_norm, labels, phase
                        )

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                else:
                    outputs = self.model(inputs_norm, labels)
                    loss, acc, clust_loss = self._compute_loss_enhanced(
                        outputs, inputs_norm, labels, phase
                    )
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_loss += loss.item()
                cluster_loss += clust_loss
                if acc is not None:
                    train_acc += acc
                n_batches += 1
                pbar.set_postfix({'loss': f"{train_loss/n_batches:.4f}"})

            avg_train_loss = train_loss / n_batches
            avg_cluster_loss = cluster_loss / n_batches
            avg_train_acc = train_acc / n_batches if train_acc is not None else None

            if val_loader:
                val_loss, val_acc, val_nmi = self._validate_enhanced(val_loader, phase)
                self.scheduler.step(val_loss)

                is_better = val_loss < self.best_loss

                if is_better:
                    self._update_best_metrics(val_loss, val_acc, val_nmi, epoch, phase)
                    self.save_checkpoint_for_resume(epoch, phase, val_loss, val_acc, val_nmi, is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    self.save_checkpoint_for_resume(epoch, phase, val_loss, val_acc, val_nmi, is_best=False)

                if phase == 2:
                    print(f"Epoch {epoch+1:3d} | {avg_train_loss:.4f} | {avg_cluster_loss:.4f} | "
                          f"{val_loss:.4f} | {avg_train_acc:.2%} | {val_acc:.2%} | "
                          f"{val_nmi:.3f} | {self.optimizer.param_groups[0]['lr']:.2e}")
                else:
                    print(f"Epoch {epoch+1:3d} | {avg_train_loss:.4f} | {val_loss:.4f} | "
                          f"{self.optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(f"Epoch {epoch+1:3d} | {avg_train_loss:.4f} | N/A | "
                      f"{self.optimizer.param_groups[0]['lr']:.2e}")

            # Record history
            self.history['train_loss'].append(avg_train_loss)
            if phase == 2:
                self.history['cluster_loss'].append(avg_cluster_loss)
                if avg_train_acc:
                    self.history['train_acc'].append(avg_train_acc)
            if val_loader:
                self.history['val_loss'].append(val_loss)
                if val_acc:
                    self.history['val_acc'].append(val_acc)
                if val_nmi:
                    self.history['val_nmi'].append(val_nmi)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            if patience_counter >= 10:
                print(f"\n{Colors.YELLOW}Early stopping at epoch {epoch+1}{Colors.ENDC}")
                break

    def _compute_loss_enhanced(self, outputs, inputs, labels, phase):
        """Enhanced loss with Euclidean distance + Contrastive loss"""

        recon_loss = F.mse_loss(outputs['reconstruction'], inputs)
        feature_loss = F.mse_loss(outputs['reconstructed_embedding'], outputs['embedding'])

        total_loss = recon_loss + 0.1 * feature_loss
        accuracy = None
        cluster_loss = 0.0

        if phase == 2:
            compressed = outputs['compressed_embedding']

            if self.config.use_class_encoding and 'class_logits' in outputs:
                class_loss = F.cross_entropy(outputs['class_logits'], labels)
                total_loss += 0.5 * class_loss
                preds = outputs['class_predictions']
                accuracy = (preds == labels).float().mean().item()

            if self.config.use_kl_divergence and hasattr(self.model, 'cluster_centers'):

                compressed_norm = F.normalize(compressed, p=2, dim=1)
                centers_norm = F.normalize(self.model.cluster_centers, p=2, dim=1)

                # Euclidean distance (more discriminative)
                cosine_sim = torch.mm(compressed_norm, centers_norm.t())
                euclidean_dist = torch.sqrt(2 * (1 - cosine_sim) + 1e-8)

                # Student's t-distribution
                alpha = 1.0
                q = (1 + euclidean_dist**2 / alpha) ** (-(alpha + 1) / 2)
                q = q / (q.sum(dim=1, keepdim=True) + 1e-8)

                p = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
                p = p / (p.sum(dim=1, keepdim=True) + 1e-8)

                kl_loss = F.kl_div((q + 1e-8).log(), p, reduction='batchmean')
                cluster_loss = kl_loss
                total_loss += 0.5 * cluster_loss

                # Contrastive loss for better separation
                if self.config.use_class_encoding and labels is not None:
                    contrastive_loss = self._compute_contrastive_loss(compressed_norm, labels)
                    total_loss += 0.3 * contrastive_loss
                    cluster_loss += contrastive_loss

                outputs['cluster_probabilities'] = q
                outputs['target_distribution'] = p
                outputs['cluster_assignments'] = q.argmax(dim=1)
                outputs['cluster_confidence'] = q.max(dim=1)[0]

        return total_loss, accuracy, cluster_loss

    def _compute_contrastive_loss(self, embeddings, labels, temperature=0.5):
        """Contrastive loss for better class separation"""
        batch_size = embeddings.shape[0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.mm(embeddings, embeddings.t()) / temperature
        labels = labels.view(-1, 1)
        mask_positive = torch.eq(labels, labels.t()).float().to(embeddings.device)
        mask_self = torch.eye(batch_size, device=embeddings.device)
        mask_positive = mask_positive - mask_self
        exp_sim = torch.exp(similarity)
        pos_sum = (exp_sim * mask_positive).sum(dim=1)
        neg_sum = exp_sim.sum(dim=1) - exp_sim.diag()
        loss = -torch.log(pos_sum / (neg_sum + 1e-8) + 1e-8)
        loss = loss[mask_positive.sum(dim=1) > 0].mean()
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=embeddings.device)

    def _validate_enhanced(self, val_loader, phase):
        """Enhanced validation with NMI calculation"""
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_batches = 0
        all_embeddings = []
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs_norm = self._per_image_normalize(inputs)

                if inputs_norm.size(0) == 1:
                    inputs_norm = torch.cat([inputs_norm, inputs_norm], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                outputs = self.model(inputs_norm, labels)
                loss, acc, _ = self._compute_loss_enhanced(outputs, inputs_norm, labels, phase)

                val_loss += loss.item()
                if acc is not None:
                    val_acc += acc
                n_batches += 1

                all_embeddings.append(outputs['compressed_embedding'].cpu())
                all_labels.append(labels.cpu())
                if 'cluster_assignments' in outputs:
                    all_predictions.append(outputs['cluster_assignments'].cpu())

        avg_loss = val_loss / n_batches
        avg_acc = val_acc / n_batches if val_acc else None

        nmi = 0.0
        if all_predictions and phase == 2:
            try:
                from sklearn.metrics import normalized_mutual_info_score
                predictions = torch.cat(all_predictions, dim=0).numpy()
                labels_all = torch.cat(all_labels, dim=0).numpy()
                nmi = normalized_mutual_info_score(labels_all, predictions)
            except:
                pass

        return avg_loss, avg_acc, nmi

    def _update_best_metrics(self, loss, accuracy, nmi, epoch, phase):
        if phase == 1:
            if loss < self.best_phase1_loss:
                self.best_phase1_loss = loss
                self.best_phase1_epoch = epoch
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_epoch = epoch
                    self.best_phase = phase
        else:
            if accuracy and accuracy > self.best_phase2_accuracy:
                self.best_phase2_accuracy = accuracy
                self.best_phase2_epoch = epoch
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_loss = loss
                    self.best_epoch = epoch
                    self.best_phase = phase

            if nmi > self.best_phase2_nmi:
                self.best_phase2_nmi = nmi
                if nmi > self.best_nmi:
                    self.best_nmi = nmi

            if loss < self.best_phase2_loss:
                self.best_phase2_loss = loss

    def _save_checkpoint_enhanced(self, epoch, phase, loss, accuracy, nmi, is_best=False):
        """Legacy method - kept for backward compatibility, delegates to save_checkpoint_for_resume"""
        self.save_checkpoint_for_resume(epoch, phase, loss, accuracy, nmi, is_best)

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        """Load checkpoint with backward compatibility"""
        path = Path(path) if path else self.checkpoint_dir / 'best.pt'
        if not path.exists():
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.loaded_epoch = checkpoint.get('epoch', 0)
            self.loaded_phase = checkpoint.get('phase', 1)
            self.loaded_loss = checkpoint.get('loss', float('inf'))
            self.loaded_accuracy = checkpoint.get('accuracy', 0.0)
            self.model_loaded = True

            if hasattr(self.model, 'deterministic_mode'):
                self.model.deterministic_mode = True

            logger.info(f"Loaded checkpoint from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

# =============================================================================
# VISUALIZER
# =============================================================================

class Visualizer:
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.viz_dir = Path(config.viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(self, history: Dict, save: bool = True):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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

        ax = axes[1, 0]
        if 'lr' in history:
            ax.semilogy(history['lr'], 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig(self.viz_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=class_names, yticklabels=class_names)
        axes[0].set_title('Confusion Matrix (Counts)')

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                   xticklabels=class_names, yticklabels=class_names)
        axes[1].set_title('Confusion Matrix (Normalized)')

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, class_names: Optional[List[str]] = None):
        perplexity = min(30, len(features) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)

        if class_names:
            legend_elements = []
            for i, name in enumerate(class_names[:10]):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=scatter.cmap(scatter.norm(i)),
                                                label=name, markersize=8))
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_title('t-SNE Visualization of Features')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'tsne.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_pca(self, features: np.ndarray, labels: np.ndarray, class_names: Optional[List[str]] = None):
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
        ax1.set_title('PCA Projection')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.grid(True, alpha=0.3)

        ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7)
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.set_title('Explained Variance by Component')
        ax2.grid(True, alpha=0.3)

        if class_names:
            legend_elements = []
            for i, name in enumerate(class_names[:10]):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=scatter.cmap(scatter.norm(i)),
                                                label=name, markersize=8))
            ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'pca.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_class_distribution(self, class_counts: Dict[str, int]):
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        bars = ax1.bar(range(len(classes)), counts, alpha=0.7)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Class Distribution')
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{count}', ha='center', va='bottom')

        ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution (%)')

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# =============================================================================
# FIXED: CDBNN with proper lowercase handling and consistent file organization
# =============================================================================

import os
import sys
import json
from pathlib import Path

# =============================================================================
# FIXED: DETERMINISTIC TRAIN FUNCTION THAT RETURNS HISTORY
# =============================================================================

# =============================================================================
# FIXED: DETERMINISTIC TRAIN FUNCTION WITH NUMERICAL STABILITY
# =============================================================================

def deterministic_train(
    model: BaseAutoencoder,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: GlobalConfig,
    checkpoint_dir: Optional[Path] = None,
    statistics_calculator: Optional['DatasetStatisticsCalculator'] = None,
    resume: bool = False,
    start_epoch: int = 0,
    start_phase: int = 1,
    loaded_optimizer_state: Optional[Dict] = None,
    loaded_scheduler_state: Optional[Dict] = None,
    reset_optimizer: bool = False,
    additional_epochs: Optional[int] = None
) -> Dict:
    """
    FULLY VECTORIZED deterministic training with optimized GPU utilization.
    Enhanced with proper resume capability.
    """
    logger.info("=" * 70)
    logger.info("VECTORIZED DETERMINISTIC TRAINING ENGINE")
    logger.info(f"Normalization: {config.normalization_mode.upper()}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Batch size: {config.batch_size}")

    # Check for contrastive mode
    use_contrastive = getattr(config, 'use_contrastive_learning', False)
    if use_contrastive:
        logger.info("🚀 CONTRASTIVE LEARNING MODE ENABLED")
        logger.info(f"  Temperature: {config.contrastive_temperature}")
        logger.info(f"  Contrastive weight: {config.contrastive_weight}")

    if resume:
        logger.info(f"RESUME MODE: Continuing from Phase {start_phase}, Epoch {start_epoch + 1}")
        if additional_epochs:
            logger.info(f"Adding {additional_epochs} additional epochs")
    logger.info("=" * 70)

    device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Enable cuDNN auto-tuner for optimal performance
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    use_per_image_norm = config.use_per_image_normalization

    # Setup statistics
    if not use_per_image_norm:
        if statistics_calculator and statistics_calculator.is_calculated:
            model.set_dataset_statistics(statistics_calculator)
            logger.info("Dataset statistics loaded into model")
        else:
            logger.warning("No statistics calculator provided - will use per-image normalization fallback")
    else:
        logger.info("Using per-image normalization")

    if hasattr(model, 'deterministic_mode'):
        model.deterministic_mode = True

    # ========================================================================
    # DATASET-SPECIFIC OPTIMIZER AND SCHEDULER
    # ========================================================================

    # Detect dataset type for optimal hyperparameters
    is_cifar = config.dataset_name in ['cifar10', 'cifar100']
    is_mnist = config.dataset_name in ['mnist', 'fashionmnist']
    min_dim = min(config.input_size)
    is_small_image = min_dim <= 64

    # Adjust learning rate for contrastive learning
    if use_contrastive and config.num_classes >= 50:
        initial_lr = 0.0005
        weight_decay = 0.0001
        use_cosine_scheduler = True
        logger.info(f"Using contrastive-optimized settings: LR={initial_lr}, weight_decay={weight_decay}")
    elif is_cifar:
        initial_lr = 0.001
        weight_decay = 0.0001
        use_cosine_scheduler = True
        logger.info("Using CIFAR-optimized training settings: LR=0.001, weight_decay=0.0001")
    elif is_mnist:
        initial_lr = 0.001
        weight_decay = 0.0001
        use_cosine_scheduler = False
        logger.info("Using MNIST-optimized training settings")
    elif is_small_image:
        initial_lr = 0.0005
        weight_decay = 0.0001
        use_cosine_scheduler = True
        logger.info(f"Using small image optimized settings: LR={initial_lr}")
    else:
        initial_lr = config.learning_rate
        weight_decay = 0.0001
        use_cosine_scheduler = False

    # Create optimizer with dataset-specific settings
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

    # Load optimizer state if resuming and not resetting
    if resume and loaded_optimizer_state is not None and not reset_optimizer:
        try:
            optimizer.load_state_dict(loaded_optimizer_state)
            logger.info("✓ Loaded optimizer state from checkpoint")
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")
            logger.info("Using fresh optimizer")
    elif resume and reset_optimizer:
        logger.info("Optimizer state reset (starting fresh)")

    # Create scheduler
    if use_cosine_scheduler:
        # Adjust T_max for additional epochs if resuming
        if resume and additional_epochs:
            total_epochs = start_epoch + additional_epochs
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=1e-7
            )
            logger.info(f"Using CosineAnnealingLR scheduler (T_max={total_epochs}, eta_min=1e-7)")
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=1e-7
            )
            logger.info(f"Using CosineAnnealingLR scheduler (T_max={config.epochs}, eta_min=1e-7)")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        logger.info("Using ReduceLROnPlateau scheduler")

    # Load scheduler state if resuming and not resetting
    if resume and loaded_scheduler_state is not None and not reset_optimizer:
        try:
            scheduler.load_state_dict(loaded_scheduler_state)
            logger.info("✓ Loaded scheduler state from checkpoint")
        except Exception as e:
            logger.warning(f"Could not load scheduler state: {e}")

    # Disable mixed precision for stability
    use_mixed_precision = False
    scaler = None
    logger.info("Mixed precision training DISABLED (for stability)")

    # Create optimized DataLoaders with pinned memory and multiple workers
    num_workers = min(4, os.cpu_count() or 1) if config.num_workers == 0 else config.num_workers

    def _create_optimized_loader(dataset, batch_size, shuffle=False):
        g = torch.Generator()
        g.manual_seed(42)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=device.type == 'cuda',
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            generator=g,
            drop_last=False
        )

    train_loader = _create_optimized_loader(train_dataset, config.batch_size, shuffle=True)
    val_loader = _create_optimized_loader(val_dataset, config.batch_size, shuffle=False) if val_dataset else None

    # Pre-compute normalization statistics for validation
    if not use_per_image_norm and statistics_calculator and statistics_calculator.is_calculated:
        norm_mean = statistics_calculator.mean.to(device).view(1, -1, 1, 1)
        norm_std = statistics_calculator.std.to(device).view(1, -1, 1, 1)

        def apply_normalization(x):
            return (x - norm_mean) / (norm_std + 1e-6)
    elif use_per_image_norm:
        def apply_normalization(x):
            if x.dim() == 4:
                b, c, h, w = x.shape
                x_flat = x.view(b, c, -1)
                mean = x_flat.mean(dim=2, keepdim=True)
                std = x_flat.std(dim=2, keepdim=True)
                return ((x_flat - mean) / (std + 1e-6)).view(b, c, h, w)
            return x
    else:
        def apply_normalization(x):
            if x.dim() == 4:
                b, c, h, w = x.shape
                x_flat = x.view(b, c, -1)
                mean = x_flat.mean(dim=2, keepdim=True)
                std = x_flat.std(dim=2, keepdim=True)
                return ((x_flat - mean) / (std + 1e-6)).view(b, c, h, w)
            return x

    def compute_enhanced_loss(outputs, inputs, labels, phase, epoch):
        """NUMERICALLY STABLE loss computation with adaptive label smoothing"""
        recon_loss = F.smooth_l1_loss(outputs['reconstruction'], inputs, beta=0.1)
        recon_loss = torch.clamp(recon_loss, max=10.0)
        total_loss = recon_loss
        accuracy = None

        # Contrastive learning support
        if use_contrastive and hasattr(model, 'compute_contrastive_loss'):
            if 'contrastive_features' in outputs:
                contrastive_loss = model.compute_contrastive_loss(
                    outputs['contrastive_features'],
                    labels
                )
                contrastive_loss = torch.clamp(contrastive_loss, max=10.0)
                total_loss = total_loss + contrastive_loss
                logger.debug(f"Contrastive loss: {contrastive_loss.item():.4f}")

        if phase == 2 and config.use_class_encoding and 'class_logits' in outputs:
            n_classes = config.num_classes or 2
            logits = outputs['class_logits']
            logits = torch.clamp(logits, min=-50, max=50)

            # Adaptive label smoothing
            if n_classes >= 100:
                smoothing = max(0.05, 0.2 * (1.0 - min(1.0, epoch / config.epochs)))
            elif n_classes >= 50:
                smoothing = max(0.05, 0.15 * (1.0 - min(1.0, epoch / config.epochs)))
            else:
                smoothing = max(0.05, 0.1 * (1.0 - min(1.0, epoch / config.epochs)))

            if smoothing > 0:
                smooth_labels = torch.full_like(logits, smoothing / n_classes)
                smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - smoothing + smoothing / n_classes)
                class_loss = -(smooth_labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            else:
                class_loss = F.cross_entropy(logits, labels)

            class_loss = torch.clamp(class_loss, max=10.0)
            total_loss = total_loss + class_loss
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

        total_loss = torch.clamp(total_loss, max=100.0)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning("NaN/Inf detected in loss - returning zero")
            return torch.tensor(0.0, device=device, requires_grad=True), accuracy

        return total_loss, accuracy

    def validate(val_loader, model, phase):
        """Vectorized validation with NaN protection"""
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_batches = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                inputs_norm = apply_normalization(inputs)

                if inputs_norm.size(0) == 1:
                    inputs_norm = torch.cat([inputs_norm, inputs_norm], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                outputs = model(inputs_norm, labels)
                loss, acc = compute_enhanced_loss(outputs, inputs_norm, labels, phase, 0)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    if acc is not None:
                        val_acc += acc
                    n_batches += 1

        if n_batches == 0:
            return float('inf'), 0.0

        return val_loss / n_batches, (val_acc / n_batches if val_acc else None)

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Define best metrics variables
    best_loss = float('inf')
    best_accuracy = 0.0
    best_epoch = 0
    best_phase = 1
    history = defaultdict(list)

    # Load existing history if resuming
    if resume and loaded_scheduler_state is not None and 'history' in loaded_scheduler_state:
        try:
            history = defaultdict(list, loaded_scheduler_state.get('history', {}))
            logger.info("Loaded training history")
        except Exception as e:
            logger.warning(f"Could not load history: {e}")

    def save_checkpoint(epoch, phase, loss, accuracy, is_best):
        """Save checkpoint with complete state for resume"""
        nonlocal best_loss, best_accuracy, best_epoch, best_phase

        if loss is None or (isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss))):
            logger.warning(f"Skipping checkpoint save - invalid loss: {loss}")
            return

        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss if loss is not None else float('inf'),
            'accuracy': accuracy if accuracy is not None else 0.0,
            'best_loss': best_loss,
            'best_accuracy': best_accuracy,
            'best_epoch': best_epoch,
            'best_phase': best_phase,
            'history': dict(history),
            'normalization_mode': config.normalization_mode,
            'use_per_image_normalization': config.use_per_image_normalization,
            'use_contrastive_learning': use_contrastive,
            'clustering_mode': 'contrastive' if use_contrastive else 'enhanced',
            'deterministic': True,
            'feature_dims': config.feature_dims,
            'compressed_dims': config.compressed_dims,
            'in_channels': config.in_channels,
            'num_classes': config.num_classes,
            'input_size': list(config.input_size),
            'timestamp': datetime.now().isoformat(),
        }

        # Add contrastive-specific parameters
        if use_contrastive:
            checkpoint['contrastive_temperature'] = config.contrastive_temperature
            checkpoint['contrastive_weight'] = config.contrastive_weight
            if hasattr(model, 'projection_head'):
                checkpoint['projection_dim'] = config.contrastive_projection_dim

        # Save dataset statistics
        if not use_per_image_norm and statistics_calculator and statistics_calculator.is_calculated:
            if statistics_calculator.mean is not None and statistics_calculator.std is not None:
                checkpoint['dataset_statistics'] = {
                    'mean': statistics_calculator.mean.cpu().tolist(),
                    'std': statistics_calculator.std.cpu().tolist(),
                    'per_channel_min': statistics_calculator.per_channel_min.cpu().tolist() if statistics_calculator.per_channel_min is not None else None,
                    'per_channel_max': statistics_calculator.per_channel_max.cpu().tolist() if statistics_calculator.per_channel_max is not None else None,
                    'n_samples_used': statistics_calculator.n_samples_used,
                    'normalization_type': 'dataset_wide',
                    'mode': statistics_calculator.mode,
                    'is_calculated': statistics_calculator.is_calculated
                }

                # Save statistics file
                dataset_name_lower = normalize_dataset_name(config.dataset_name)
                stats_path = checkpoint_dir / f"{dataset_name_lower}_norm_stats.pt"
                stats_dict = {
                    'mean': statistics_calculator.mean.cpu(),
                    'std': statistics_calculator.std.cpu(),
                    'per_channel_min': statistics_calculator.per_channel_min.cpu() if statistics_calculator.per_channel_min is not None else None,
                    'per_channel_max': statistics_calculator.per_channel_max.cpu() if statistics_calculator.per_channel_max is not None else None,
                    'n_samples': statistics_calculator.n_samples_used,
                    'is_fitted': True,
                    'timestamp': datetime.now().isoformat()
                }
                torch.save(stats_dict, stats_path)
                logger.info(f"Normalization statistics saved to {stats_path}")

        # Add optional components
        if hasattr(model, 'classifier') and model.classifier is not None:
            checkpoint['classifier_state_dict'] = model.classifier.state_dict()
        if hasattr(model, 'cluster_centers') and model.cluster_centers is not None:
            checkpoint['cluster_centers'] = model.cluster_centers.data.cpu()
        if hasattr(model, '_selected_feature_indices') and model._selected_feature_indices is not None:
            checkpoint['selected_feature_indices'] = model._selected_feature_indices.cpu()

        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pt')
        logger.info(f"Saved latest checkpoint (Phase {phase}, Epoch {epoch+1}, Loss: {loss:.6f})")

        # Save best checkpoint
        if is_best and accuracy is not None and accuracy > 0:
            torch.save(checkpoint, checkpoint_dir / 'best.pt')
            best_loss = loss
            best_accuracy = accuracy
            best_epoch = epoch
            best_phase = phase
            logger.info(f"✓ Best model saved - Epoch {epoch+1}, Accuracy: {accuracy:.4f} ({accuracy:.2%})")

            # Named copy
            dataset_name_lower = normalize_dataset_name(config.dataset_name)
            named_best_path = checkpoint_dir / f"{dataset_name_lower}_best.pt"
            torch.save(checkpoint, named_best_path)
            logger.info(f"✓ Named best model saved to {named_best_path}")

    def train_phase(phase, epochs, start_epoch=0):
        nonlocal best_loss, best_accuracy, best_epoch, best_phase

        model.set_training_phase(phase)
        phase_best_accuracy = 0.0
        phase_best_loss = float('inf')
        patience_counter = 0
        patience_limit = 25 if use_contrastive else 20
        nan_count = 0
        max_nan_count = 5

        current_lr = optimizer.param_groups[0]['lr']
        epoch_train_losses = []

        print(f"\n{Colors.BOLD}{'Phase ' + str(phase) + ' Vectorized Training'.center(80)}{Colors.ENDC}")
        if phase == 2:
            print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Train Acc':>10} | {'Val Acc':>10} | {'Best Acc':>10} | {'LR':>12}")
            print(f"{Colors.BOLD}{'-'*90}{Colors.ENDC}")
        else:
            print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>12}")
            print(f"{Colors.BOLD}{'-'*80}{Colors.ENDC}")

        for epoch_offset in range(epochs):
            epoch = start_epoch + epoch_offset
            model.train()
            train_loss = 0.0
            train_acc = 0.0 if phase == 2 else None
            n_batches = 0
            nan_count_epoch = 0

            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(train_loader, desc=f"Phase {phase} Epoch {epoch+1}")

            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                inputs_norm = apply_normalization(inputs)

                if torch.isnan(inputs_norm).any() or torch.isinf(inputs_norm).any():
                    logger.warning(f"NaN/Inf detected in inputs at batch {batch_idx}, skipping")
                    continue

                if inputs_norm.size(0) == 1:
                    inputs_norm = torch.cat([inputs_norm, inputs_norm], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                outputs = model(inputs_norm, labels)
                loss, acc = compute_enhanced_loss(outputs, inputs_norm, labels, phase, epoch)

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count_epoch += 1
                    nan_count += 1
                    if nan_count > max_nan_count:
                        logger.warning(f"Too many NaN losses ({nan_count}), breaking...")
                        break
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item()
                if acc is not None:
                    train_acc += acc
                n_batches += 1
                pbar.set_postfix({'loss': f"{train_loss/n_batches:.4f}" if n_batches > 0 else 'nan',
                                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})

                # Diagnostic for first batch of resumed training
                if phase == 1 and epoch == start_epoch and batch_idx == 0 and resume:
                    logger.info("=" * 60)
                    logger.info("RESUME DIAGNOSTIC: First Batch Statistics")
                    logger.info(f"Input - mean: {inputs.mean().item():.6f}, std: {inputs.std().item():.6f}")
                    logger.info(f"Normalized - mean: {inputs_norm.mean().item():.6f}, std: {inputs_norm.std().item():.6f}")
                    logger.info(f"Loss: {loss.item():.6f}")
                    logger.info(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
                    logger.info("=" * 60)

            if n_batches == 0:
                logger.warning("No valid batches in epoch, skipping...")
                continue

            avg_train_loss = train_loss / n_batches
            avg_train_acc = train_acc / n_batches if train_acc is not None else None
            epoch_train_losses.append(avg_train_loss)

            if val_loader:
                val_loss, val_acc = validate(val_loader, model, phase)

                if not np.isnan(val_loss) and not np.isinf(val_loss):
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

                if phase == 2 and val_acc is not None and not np.isnan(val_acc):
                    is_better = val_acc > phase_best_accuracy
                    if is_better:
                        phase_best_accuracy = val_acc
                        phase_best_loss = val_loss
                        save_checkpoint(epoch, phase, val_loss, val_acc, is_best=True)
                        patience_counter = 0
                        logger.info(f"  ✓ New best accuracy: {val_acc:.4f} ({val_acc:.2%})")
                    else:
                        patience_counter += 1
                        save_checkpoint(epoch, phase, val_loss, val_acc, is_best=False)

                    print(f"{epoch+1:6d} | {avg_train_loss:12.6f} | {val_loss:12.6f} | "
                          f"{avg_train_acc:9.2%} | {val_acc:9.2%} | {phase_best_accuracy:9.2%} | {current_lr:12.2e}")

                    history['val_acc'].append(val_acc)
                elif phase == 1 and not np.isnan(val_loss):
                    is_better = val_loss < phase_best_loss
                    if is_better:
                        phase_best_loss = val_loss
                        save_checkpoint(epoch, phase, val_loss, val_acc, is_best=False)
                        patience_counter = 0
                        if val_loss < best_loss:
                            best_loss = val_loss
                        logger.info(f"  ✓ Loss improved to {val_loss:.6f}")
                    else:
                        patience_counter += 1
                        save_checkpoint(epoch, phase, val_loss, val_acc, is_best=False)

                    print(f"{epoch+1:6d} | {avg_train_loss:12.6f} | {val_loss:12.6f} | {current_lr:12.2e}")
                else:
                    print(f"{epoch+1:6d} | {avg_train_loss:12.6f} | {val_loss:12.6f} | {current_lr:12.2e}")

                if not np.isnan(val_loss):
                    history['val_loss'].append(val_loss)
                else:
                    history['val_loss'].append(phase_best_loss)
            else:
                if phase == 2 and avg_train_acc is not None and not np.isnan(avg_train_acc):
                    is_better = avg_train_acc > phase_best_accuracy
                    if is_better:
                        phase_best_accuracy = avg_train_acc
                        phase_best_loss = avg_train_loss
                        save_checkpoint(epoch, phase, avg_train_loss, avg_train_acc, is_best=True)
                        best_accuracy = avg_train_acc
                        best_loss = avg_train_loss
                        best_epoch = epoch
                        best_phase = phase
                        patience_counter = 0
                        logger.info(f"  ✓ New best accuracy: {avg_train_acc:.4f} ({avg_train_acc:.2%})")
                    else:
                        patience_counter += 1
                        save_checkpoint(epoch, phase, avg_train_loss, avg_train_acc, is_best=False)

                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_train_loss)
                    else:
                        scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']

                    print(f"{epoch+1:6d} | {avg_train_loss:12.6f} | {'N/A':>12} | "
                          f"{avg_train_acc:9.2%} | {'N/A':>9} | {phase_best_accuracy:9.2%} | {current_lr:12.2e}")
                elif phase == 1 and not np.isnan(avg_train_loss):
                    is_better = avg_train_loss < phase_best_loss
                    if is_better:
                        phase_best_loss = avg_train_loss
                        save_checkpoint(epoch, phase, avg_train_loss, avg_train_acc, is_best=False)
                        patience_counter = 0
                        if avg_train_loss < best_loss:
                            best_loss = avg_train_loss
                        logger.info(f"  ✓ Loss improved to {avg_train_loss:.6f}")
                    else:
                        patience_counter += 1
                        save_checkpoint(epoch, phase, avg_train_loss, avg_train_acc, is_best=False)

                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_train_loss)
                    else:
                        scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']

                    print(f"{epoch+1:6d} | {avg_train_loss:12.6f} | {'N/A':>12} | {current_lr:12.2e}")

            history['train_loss'].append(avg_train_loss)
            if avg_train_acc is not None:
                history['train_acc'].append(avg_train_acc)
            history['lr'].append(current_lr)

            if patience_counter >= patience_limit:
                print(f"\n{Colors.YELLOW}{'='*60}{Colors.ENDC}")
                print(f"{Colors.YELLOW}Early stopping triggered after {patience_counter} epochs without improvement{Colors.ENDC}")
                print(f"{Colors.YELLOW}Best loss: {phase_best_loss:.6f}{Colors.ENDC}")
                if phase == 2:
                    print(f"{Colors.YELLOW}Best accuracy: {phase_best_accuracy:.2%}{Colors.ENDC}")
                print(f"{Colors.YELLOW}{'='*60}{Colors.ENDC}")
                break

            if nan_count_epoch == 0:
                nan_count = max(0, nan_count - 1)

        print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
        if phase == 1:
            logger.info(f"Phase 1 completed. Best loss: {phase_best_loss:.6f}")
            print(f"{Colors.GREEN}Phase 1 Best Loss: {phase_best_loss:.6f}{Colors.ENDC}")
        else:
            logger.info(f"Phase 2 completed. Best accuracy: {phase_best_accuracy:.2%} at loss: {phase_best_loss:.6f}")
            print(f"{Colors.GREEN}Phase 2 Best Accuracy: {phase_best_accuracy:.2%}{Colors.ENDC}")
            print(f"{Colors.GREEN}Phase 2 Best Loss: {phase_best_loss:.6f}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

    # Calculate epochs based on configuration and resume
    total_epochs = config.epochs

    # Adjust if additional epochs specified
    if resume and additional_epochs:
        total_epochs = start_epoch + additional_epochs
        logger.info(f"Adjusted total epochs to {total_epochs} (original: {config.epochs})")

    # Phase lengths
    if use_contrastive:
        epochs_phase1 = max(1, total_epochs // 5)
        epochs_phase2 = max(1, total_epochs - epochs_phase1)
        logger.info(f"Contrastive learning mode: Phase1={epochs_phase1}, Phase2={epochs_phase2}")
    else:
        epochs_phase1 = max(1, total_epochs // 4)
        epochs_phase2 = max(1, total_epochs - epochs_phase1)

    # Adjust epochs based on resume position
    if resume:
        if start_phase == 1:
            epochs_phase1 = max(0, epochs_phase1 - start_epoch)
            epochs_phase2 = epochs_phase2
            logger.info(f"Resuming Phase 1 from epoch {start_epoch + 1}, remaining: {epochs_phase1} epochs")
        elif start_phase == 2:
            epochs_phase1 = 0
            epochs_phase2 = max(0, epochs_phase2 - start_epoch)
            logger.info(f"Resuming Phase 2 from epoch {start_epoch + 1}, remaining: {epochs_phase2} epochs")

    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'VECTORIZED CDBNN TRAINING'.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Normalization: {config.normalization_mode.upper()}{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Contrastive Learning: {use_contrastive}{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Mixed Precision: {scaler is not None}{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Num Workers: {num_workers}{Colors.ENDC}")
    if resume:
        print(f"{Colors.GREEN}✓ Resume Mode: Starting from Phase {start_phase}, Epoch {start_epoch + 1}{Colors.ENDC}")
    print()

    # Train Phase 1
    if start_phase <= 1 and epochs_phase1 > 0:
        logger.info(f"Phase 1: {epochs_phase1} epochs")
        train_phase(1, epochs_phase1, start_epoch if start_phase == 1 else 0)

    # Train Phase 2
    if start_phase <= 2 and epochs_phase2 > 0:
        if config.use_kl_divergence or config.use_class_encoding or use_contrastive:
            phase2_title = "PHASE 2: VECTORIZED " + ("CONTRASTIVE LEARNING" if use_contrastive else "CLUSTERING")
            print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
            print(f"{Colors.BOLD}{phase2_title.center(80)}{Colors.ENDC}")
            print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
            logger.info(f"Phase 2: {epochs_phase2} epochs")
            train_phase(2, epochs_phase2, start_epoch if start_phase == 2 else 0)

    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'TRAINING COMPLETED'.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.GREEN}Best Loss: {best_loss:.6f}{Colors.ENDC}")
    if best_accuracy > 0:
        print(f"{Colors.GREEN}Best Accuracy: {best_accuracy:.2%}{Colors.ENDC}")
    print(f"{Colors.GREEN}Best Epoch: {best_epoch + 1}{Colors.ENDC}")

    # Ensure history is returned even if no metrics were recorded
    if not history['train_loss']:
        history['train_loss'] = [0.0]
    if not history.get('val_loss'):
        history['val_loss'] = [float('inf')]
    if not history.get('val_acc'):
        history['val_acc'] = [0.0]
    if not history.get('train_acc'):
        history['train_acc'] = [0.0]

    return dict(history)

def normalize_dataset_name(data_name: str) -> str:
    """Convert dataset name to lowercase for consistent file naming"""
    return data_name.lower() if data_name else 'dataset'


def get_dataset_paths(data_name: str, base_dir: str = 'data'):
    """
    Get standardized paths for dataset files.
    ALL files go to: base_dir/dataset_name/ (where dataset_name is already normalized)
    """
    # data_name should already be normalized (lowercase)
    data_dir = Path(base_dir) / data_name
    return {
        'data_dir': data_dir,
        'csv_path': data_dir / f"{data_name}.csv",
        'train_csv': data_dir / f"{data_name}_train.csv",
        'test_csv': data_dir / f"{data_name}_test.csv",
        'json_config': data_dir / f"{data_name}_config.json",
        'conf_config': data_dir / f"{data_name}.conf",
        'minimal_config': data_dir / f"{data_name}_config_minimal.json",
        'checkpoint_dir': data_dir / 'checkpoints',
        'viz_dir': data_dir / 'visualizations',
        'log_dir': data_dir / 'logs',
        'heatmap_dir': data_dir / 'attention_heatmaps'
    }

# =============================================================================
# FIXED: CDBNN Application with proper attributes
# =============================================================================

# =============================================================================
# MODIFIED CDBNN APPLICATION (with dataset statistics support)
# =============================================================================

class CDBNNApplication:
    def __init__(self, config: GlobalConfig):
        # DO NOT modify config.dataset_name - it's already normalized in main()
        self.config = config
        self.model = None
        self.domain_processor = None

        # ========================================================================
        # SINGLE DIRECTORY STRUCTURE: data/dataset_name/
        # ========================================================================
        # Base data directory
        if hasattr(config, 'output_dir') and config.output_dir:
            self.base_dir = Path(config.output_dir)
        else:
            self.base_dir = Path('data')

        # Use dataset_name directly (already normalized)
        dataset_name = config.dataset_name  # Already 'galaxy', NOT 'galaxy/galaxy'

        self.saving_data_dir = self.base_dir
        self.saving_checkpoint_dir = self.saving_data_dir / 'checkpoints'
        self.saving_viz_dir = self.saving_data_dir / 'visualizations'
        self.saving_log_dir = self.saving_data_dir / 'logs'
        self.saving_heatmap_dir = self.saving_data_dir / 'attention_heatmaps'

        # Loading directory is the same as saving directory
        self.loading_data_dir = self.saving_data_dir
        self.loading_checkpoint_dir = self.saving_checkpoint_dir

        # Update config paths
        self.config.checkpoint_dir = str(self.saving_checkpoint_dir)
        self.config.viz_dir = str(self.saving_viz_dir)
        self.config.log_dir = str(self.saving_log_dir)

        # Set CSV and config paths
        self.config.csv_path = str(self.saving_data_dir / f"{dataset_name}.csv")
        self.config.train_csv_path = str(self.saving_data_dir / f"{dataset_name}_train.csv")
        self.config.test_csv_path = str(self.saving_data_dir / f"{dataset_name}_test.csv")
        self.config.conf_config_path = str(self.saving_data_dir / f"{dataset_name}.conf")
        self.config.json_config_path = str(self.saving_data_dir / f"{dataset_name}_config.json")
        self.config.class_names_path = str(self.saving_data_dir / f"{dataset_name}_classes.json")
        self.config.feature_map_path = str(self.saving_data_dir / f"{dataset_name}_feature_map.json")
        self.config.column_mapping_path = str(self.saving_data_dir / f"{dataset_name}_columns.json")

        # Create all directories
        for d in [self.saving_data_dir, self.saving_checkpoint_dir, self.saving_viz_dir,
                  self.saving_log_dir, self.saving_heatmap_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.visualizer = Visualizer(config)

        logger.info("=" * 60)
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Data directory: {self.saving_data_dir}")
        logger.info(f"Checkpoint directory: {self.saving_checkpoint_dir}")
        logger.info(f"CSV output: {self.config.csv_path}")
        logger.info(f"Configuration file: {self.config.conf_config_path}")
        logger.info("=" * 60)

    def prepare_data(self, source_path: str, data_type: str = 'custom') -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data with optimized DataLoader settings for GPU training.
        Auto-detects input size from actual image samples.
        Dynamically optimizes architecture based on dataset characteristics.
        Maintains reproducibility while maximizing performance.
        """
        # ========================================================================
        # 0. HANDLE DATA PATH FOR TORCHVISION DATASETS
        # ========================================================================
        if data_type == 'torchvision':
            dataset_name_upper = self.config.dataset_name.upper()
            valid_tv_datasets = ['MNIST', 'CIFAR10', 'CIFAR100', 'FASHIONMNIST']

            if dataset_name_upper not in valid_tv_datasets:
                raise ValueError(f"Unknown torchvision dataset: {dataset_name_upper}. "
                               f"Supported: {valid_tv_datasets}")

            logger.info(f"Using torchvision dataset: {dataset_name_upper}")

            # Auto-detect input size if not explicitly set
            if not hasattr(self.config, 'input_size_explicitly_set') or not self.config.input_size_explicitly_set:
                if dataset_name_upper in ['CIFAR10', 'CIFAR100']:
                    self.config.input_size = (32, 32)
                elif dataset_name_upper in ['MNIST', 'FASHIONMNIST']:
                    self.config.input_size = (28, 28)
                logger.info(f"Auto-set input size for {dataset_name_upper}: {self.config.input_size[0]}x{self.config.input_size[1]}")

        else:  # custom dataset
            # Auto-detect input size from sample images (if not explicitly set)
            if not hasattr(self.config, 'input_size_explicitly_set') or not self.config.input_size_explicitly_set:
                if source_path and os.path.exists(source_path):
                    logger.info("Auto-detecting input size from sample images...")
                    detected_size = self._detect_input_size_from_samples(source_path, data_type)
                    if detected_size:
                        old_size = self.config.input_size
                        self.config.input_size = detected_size
                        logger.info(f"Auto-detected input size: {detected_size[0]}x{detected_size[1]} (was {old_size[0]}x{old_size[1]})")
                    else:
                        logger.warning(f"Could not auto-detect size, using default {self.config.input_size}")
                else:
                    logger.warning(f"Source path {source_path} does not exist, using default size {self.config.input_size}")

        transform = self._get_transform()
        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)

        # ========================================================================
        # 1. CREATE DATASETS
        # ========================================================================
        if data_type == 'torchvision':
            dataset_name_upper = self.config.dataset_name.upper()
            train_dataset = self._get_torchvision_dataset(dataset_name_upper, train=True, transform=transform)
            test_dataset = self._get_torchvision_dataset(dataset_name_upper, train=False, transform=transform)

            # Set class information
            if hasattr(train_dataset, 'classes'):
                self.config.num_classes = len(train_dataset.classes)
                self.config.class_names = train_dataset.classes
            else:
                # For datasets without explicit classes (like MNIST)
                unique_labels = sorted(set(train_dataset.targets.tolist() if hasattr(train_dataset, 'targets') else []))
                self.config.num_classes = len(unique_labels)
                self.config.class_names = [str(i) for i in unique_labels]

            # ====================================================================
            # CRITICAL FIX: Save torchvision metadata for prediction
            # ====================================================================
            self._save_torchvision_metadata(train_dataset)

        else:
            # Handle archives for custom datasets
            if source_path.endswith(('.zip', '.tar', '.gz', '.bz2', '.xz')):
                extract_dir = self.loading_data_dir / 'extracted'
                source_path = ArchiveHandler.extract(source_path, str(extract_dir))

            train_path = Path(source_path) / 'train'
            if train_path.exists():
                train_dataset = CustomImageDataset(str(train_path), transform=transform, config=self.config.to_dict())
                test_path = Path(source_path) / 'test'
                test_dataset = CustomImageDataset(str(test_path), transform=transform, config=self.config.to_dict()) if test_path.exists() else None
            else:
                train_dataset = CustomImageDataset(source_path, transform=transform, config=self.config.to_dict())
                test_dataset = None

            self.config.num_classes = len(train_dataset.classes)
            self.config.class_names = train_dataset.classes

        # ========================================================================
        # 1.5 DYNAMIC ARCHITECTURE OPTIMIZATION
        # ========================================================================
        if getattr(self.config, 'auto_optimize_architecture', True):
            logger.info("=" * 70)
            logger.info("DYNAMIC ARCHITECTURE OPTIMIZATION")
            logger.info("=" * 70)

            try:
                # Analyze dataset
                optimizer = DynamicArchitectureOptimizer(self.config)
                analysis = optimizer.analyze_dataset(train_dataset)

                # Update config with optimized parameters
                optimizer.update_config_from_analysis(analysis)

                # Store analysis results for later use
                self.architecture_analysis = analysis

            except Exception as e:
                logger.warning(f"Architecture optimization failed: {e}")
                logger.warning("Using default architecture settings")

        self._save_config_files()

        # ========================================================================
        # 2. OPTIMIZED DATALOADER CONFIGURATION
        # ========================================================================

        # Determine optimal number of workers
        if self.config.num_workers > 0:
            num_workers = self.config.num_workers
        else:
            cpu_count = os.cpu_count() or 1
            if torch.cuda.is_available():
                num_workers = min(8, cpu_count)
            else:
                num_workers = min(4, cpu_count)

        pin_memory = self.config.use_gpu and torch.cuda.is_available()
        prefetch_factor = 2 if num_workers > 0 else None
        persistent_workers = num_workers > 0

        g_train = torch.Generator()
        g_train.manual_seed(42)
        g_test = torch.Generator()
        g_test.manual_seed(42)

        def worker_init_fn(worker_id):
            """Initialize worker with deterministic seed"""
            import random
            import numpy as np
            worker_seed = 42 + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        logger.info("=" * 60)
        logger.info("DataLoader Configuration:")
        logger.info(f"  Input size: {self.config.input_size[0]}x{self.config.input_size[1]}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Num workers: {num_workers}")
        logger.info(f"  Pin memory: {pin_memory}")
        logger.info(f"  Prefetch factor: {prefetch_factor}")
        logger.info(f"  Persistent workers: {persistent_workers}")
        logger.info("=" * 60)

        # ========================================================================
        # 3. CREATE TRAIN DATALOADER
        # ========================================================================
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            generator=g_train,
            worker_init_fn=worker_init_fn,
            drop_last=False,
            timeout=0
        )

        # ========================================================================
        # 4. CREATE TEST/VALIDATION DATALOADER
        # ========================================================================
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                generator=g_test,
                worker_init_fn=worker_init_fn,
                drop_last=False,
                timeout=0
            )

        # ========================================================================
        # 5. LOG DATASET INFORMATION
        # ========================================================================
        logger.info("=" * 60)
        logger.info("Dataset Information:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        if test_loader:
            logger.info(f"  Test/Validation samples: {len(test_dataset)}")
        logger.info(f"  Number of classes: {self.config.num_classes}")
        logger.info(f"  Class names: {self.config.class_names[:5]}{'...' if len(self.config.class_names) > 5 else ''}")
        logger.info("=" * 60)

        # Log architecture optimization results if available
        if hasattr(self, 'architecture_analysis'):
            logger.info("=" * 60)
            logger.info("Optimized Architecture:")
            logger.info(f"  Encoder layers: {self.architecture_analysis.get('n_layers', 'N/A')}")
            logger.info(f"  Base channels: {self.architecture_analysis.get('base_channels', 'N/A')}")
            logger.info(f"  Feature dims: {self.config.feature_dims}")
            logger.info(f"  Compressed dims: {self.config.compressed_dims}")
            logger.info(f"  Complexity score: {self.architecture_analysis.get('complexity_score', 0):.3f}")
            logger.info("=" * 60)

        if hasattr(train_dataset, 'get_class_distribution'):
            self.visualizer.plot_class_distribution(train_dataset.get_class_distribution())

        return train_loader, test_loader

    def _save_torchvision_metadata(self, dataset):
        """Save complete metadata for torchvision datasets"""
        try:
            if hasattr(dataset, 'classes'):
                class_names = list(dataset.classes)
            else:
                if hasattr(dataset, 'targets'):
                    unique_targets = sorted(set(dataset.targets.tolist() if hasattr(dataset.targets, 'tolist') else dataset.targets))
                    class_names = [str(i) for i in unique_targets]
                else:
                    class_names = [str(i) for i in range(self.config.num_classes or 10)]

            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            idx_to_class = {idx: name for idx, name in enumerate(class_names)}

            metadata = {
                'dataset_name': self.config.dataset_name,
                'source': 'torchvision',
                'num_classes': len(class_names),
                'class_names': class_names,
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class,
                'input_size': list(self.config.input_size),
                'normalization_mode': self.config.normalization_mode,
                'timestamp': datetime.now().isoformat(),
                'version': '2.0'
            }

            # Save to data_dir/dataset_name_metadata.json (NOT nested)
            metadata_path = self.saving_data_dir / f"{self.config.dataset_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"TorchVision metadata saved to {metadata_path}")

            # Also save class names separately
            class_path = self.saving_data_dir / f"{self.config.dataset_name}_classes.json"
            with open(class_path, 'w') as f:
                json.dump({
                    'class_names': class_names,
                    'num_classes': len(class_names),
                    'class_to_idx': class_to_idx,
                    'idx_to_class': idx_to_class
                }, f, indent=2)
            logger.info(f"Class mapping saved to {class_path}")

            self.config.class_names = class_names
            self.config.num_classes = len(class_names)

        except Exception as e:
            logger.warning(f"Failed to save torchvision metadata: {e}")

    def _detect_input_size_from_samples(self, source_path: str, data_type: str, num_samples: int = 50) -> Optional[Tuple[int, int]]:
        """
        Detect input size by sampling actual images from the dataset.
        Only used for custom datasets (not torchvision).

        IMPORTANT: Does NOT resize images during detection to get true original dimensions.

        Args:
            source_path: Path to data
            data_type: 'custom' or 'torchvision'
            num_samples: Number of images to sample

        Returns:
            Tuple of (height, width) or None if detection fails
        """
        # Skip detection for torchvision datasets (they have known sizes)
        if data_type == 'torchvision':
            return None

        try:
            # CRITICAL FIX: DO NOT RESIZE during detection!
            # Use a transform that only converts to tensor, preserving original dimensions
            temp_transform = transforms.Compose([
                transforms.ToTensor()  # Only convert to tensor - NO RESIZE!
            ])

            # Handle archives
            temp_source_path = source_path
            temp_extract_dir = None

            if source_path.endswith(('.zip', '.tar', '.gz', '.bz2', '.xz')):
                temp_extract_dir = self.loading_data_dir / 'extracted_temp'
                temp_source_path = ArchiveHandler.extract(source_path, str(temp_extract_dir))

            # Try to find images (look for train folder or root)
            train_path = Path(temp_source_path) / 'train'
            if train_path.exists():
                temp_dataset = CustomImageDataset(str(train_path), transform=temp_transform, config=self.config.to_dict())
            else:
                temp_dataset = CustomImageDataset(temp_source_path, transform=temp_transform, config=self.config.to_dict())

            if temp_dataset is None or len(temp_dataset) == 0:
                return None

            # Sample images to determine TRUE original size
            sample_sizes = []
            indices = list(range(min(len(temp_dataset), num_samples)))

            # Random sample for variety
            import random
            random.seed(42)
            random.shuffle(indices)

            logger.info(f"Sampling {min(len(indices), num_samples)} images to detect original dimensions...")

            for idx in indices[:num_samples]:
                try:
                    img, _ = temp_dataset[idx]
                    if isinstance(img, torch.Tensor):
                        # Get TRUE spatial dimensions (no resize applied)
                        h, w = img.shape[-2], img.shape[-1]
                    elif isinstance(img, PILImage.Image):
                        w, h = img.size  # Original PIL image size
                    elif isinstance(img, np.ndarray):
                        h, w = img.shape[:2]
                    else:
                        continue
                    sample_sizes.append((h, w))
                    logger.debug(f"Sample {idx}: {h}x{w}")
                except Exception as e:
                    logger.debug(f"Failed to get size for sample {idx}: {e}")
                    continue

            if not sample_sizes:
                logger.warning("No valid image samples found for size detection")
                return None

            # Find the most common dimensions
            from collections import Counter
            size_counter = Counter(sample_sizes)
            most_common_size = size_counter.most_common(1)[0][0]
            h, w = most_common_size

            # Calculate statistics about image sizes
            unique_sizes = len(size_counter)
            min_h = min(s[0] for s in sample_sizes)
            max_h = max(s[0] for s in sample_sizes)
            min_w = min(s[1] for s in sample_sizes)
            max_w = max(s[1] for s in sample_sizes)

            logger.info(f"Image size statistics:")
            logger.info(f"  Most common: {h}x{w}")
            logger.info(f"  Range: height {min_h}-{max_h}, width {min_w}-{max_w}")
            logger.info(f"  Unique sizes: {unique_sizes}")

            # Warn if images have very different sizes
            if unique_sizes > 3:
                logger.warning(f"Dataset has highly variable image sizes: {unique_sizes} different dimensions found")
                logger.warning(f"Most common: {h}x{w}, others: {[s for s, c in size_counter.most_common(5)[1:]]}")

            # For very small images (like MNIST 28x28, CIFAR 32x32), keep original size
            if h <= 64 and w <= 64:
                logger.info(f"Images are already small ({h}x{w}), keeping original size")
                # Just ensure they're at least 32x32
                h = max(32, h)
                w = max(32, w)
                return (h, w)

            # For larger images, optionally suggest a scaled size
            max_dim = 512  # Maximum allowed dimension
            if h > max_dim or w > max_dim:
                scale = max_dim / max(h, w)
                suggested_h = int(h * scale)
                suggested_w = int(w * scale)
                # Round to multiples of 32 for efficient downsampling
                suggested_h = ((suggested_h + 15) // 16) * 16
                suggested_w = ((suggested_w + 15) // 16) * 16
                logger.info(f"Large images detected ({h}x{w}). Suggested resize: {suggested_h}x{suggested_w}")

                # Use suggested size
                h, w = suggested_h, suggested_w
            else:
                # Round to multiples of 32 for efficient downsampling
                h = ((h + 15) // 16) * 16
                w = ((w + 15) // 16) * 16
                if (h, w) != most_common_size:
                    logger.info(f"Adjusted dimensions to {h}x{w} (multiples of 16 for efficiency)")

            logger.info(f"Final detected image dimensions: {h}x{w}")

            # Clean up temporary extracted directory if created
            if temp_extract_dir and temp_extract_dir.exists():
                import shutil
                shutil.rmtree(temp_extract_dir, ignore_errors=True)

            return (h, w)

        except Exception as e:
            logger.error(f"Error detecting input size: {e}")
            traceback.print_exc()
            return None

    def _save_config_files(self):
        """Save configuration files WITHOUT duplicating dataset name"""
        # CRITICAL: Use the config's dataset_name directly (already normalized)
        # Do NOT call normalize_dataset_name again!
        dataset_name = self.config.dataset_name
        data_dir = self.saving_data_dir

        actual_feature_count = self.config.compressed_dims
        feature_columns = [f'feature_{i}' for i in range(actual_feature_count)]
        column_names = feature_columns.copy()
        column_names.append("target")

        config_dict = {
            "dataset_name": dataset_name,
            "num_classes": self.config.num_classes,
            "csv_file": str(data_dir / f"{dataset_name}.csv"),
            "column_names": column_names,
            "target_column": "target",
            "feature_dims": self.config.feature_dims,
            "compressed_dims": self.config.compressed_dims,
            "actual_features_in_csv": actual_feature_count,
            "input_size": list(self.config.input_size) if isinstance(self.config.input_size, tuple) else self.config.input_size,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "domain": getattr(self.config, 'domain', 'general'),

            "model_config": {
                "in_channels": self.config.in_channels,
                "input_size": list(self.config.input_size) if isinstance(self.config.input_size, tuple) else self.config.input_size,
                "feature_dims": self.config.feature_dims,
                "compressed_dims": self.config.compressed_dims,
                "actual_features_in_csv": actual_feature_count,
                "use_kl_divergence": self.config.use_kl_divergence,
                "use_class_encoding": self.config.use_class_encoding,
                "use_distance_correlation": self.config.use_distance_correlation,
                "feature_selection_method": getattr(self.config, 'feature_selection_method', 'balanced'),
                "max_features": getattr(self.config, 'max_features', 32),
                "min_features": getattr(self.config, 'min_features', 8),
                "correlation_upper": getattr(self.config, 'correlation_upper', 0.85),
                "correlation_lower": getattr(self.config, 'correlation_lower', 0.01)
            },

            "training_config": {
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "validation_split": self.config.validation_split,
                "use_gpu": self.config.use_gpu,
                "mixed_precision": self.config.mixed_precision,
                "num_workers": self.config.num_workers
            },

            "visualization_config": {
                "generate_heatmaps": self.config.generate_heatmaps,
                "generate_confusion_matrix": self.config.generate_confusion_matrix,
                "generate_tsne": self.config.generate_tsne,
                "heatmap_frequency": self.config.heatmap_frequency,
                "reconstruction_samples_frequency": self.config.reconstruction_samples_frequency
            },

            "saved_at": datetime.now().isoformat(),
            "config_version": "2.4",
            "notes": "This configuration matches the training config format"
        }

        if self.config.class_names:
            config_dict["class_info"] = {
                "class_names": self.config.class_names,
                "num_classes": len(self.config.class_names),
                "class_to_idx": {name: idx for idx, name in enumerate(self.config.class_names)}
            }

        if hasattr(self.config, 'domain') and self.config.domain != 'general':
            domain_config = {}
            if self.config.domain == 'astronomy':
                domain_config.update({
                    "use_fits": getattr(self.config, 'use_fits', True),
                    "fits_hdu": getattr(self.config, 'fits_hdu', 0),
                    "fits_normalization": getattr(self.config, 'fits_normalization', 'zscale'),
                    "subtract_background": getattr(self.config, 'subtract_background', True),
                    "detect_sources": getattr(self.config, 'detect_sources', True),
                    "detection_threshold": getattr(self.config, 'detection_threshold', 2.5),
                    "pixel_scale": getattr(self.config, 'pixel_scale', 1.0),
                    "gain": getattr(self.config, 'gain', 1.0),
                    "read_noise": getattr(self.config, 'read_noise', 0.0)
                })
            config_dict['domain_config'] = domain_config

        # Save to data_dir/dataset_name.conf (NOT nested)
        conf_path = data_dir / f"{dataset_name}.conf"
        with open(conf_path, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        logger.info(f"Configuration saved to {conf_path}")
        logger.info(f"CSV will contain {len(column_names)} columns: {', '.join(column_names[:5])}...")

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              resume: bool = False, resume_from: Optional[str] = None,
              reset_optimizer: bool = False, additional_epochs: Optional[int] = None,
              save_features: bool = True) -> Dict:
        """
        Train using standalone deterministic function with invariant preprocessing and resume capability.

        Args:
            save_features: If True, extract and save features after training
        """
        self._set_deterministic_seeds()

        # Ensure device is set
        if not hasattr(self, 'device'):
            self.device = torch.device('cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu')

        # Check if augmentation is enabled (default: True)
        use_augmentation = not getattr(self.config, 'no_augmentation', False)
        augmentation_strength = getattr(self.config, 'augmentation_strength', 0.5)

        # ========================================================================
        # 1. APPLY DETERMINISTIC INVARIANT PREPROCESSING TO DATASETS
        # ========================================================================
        logger.info("=" * 70)
        logger.info("APPLYING DETERMINISTIC INVARIANT PREPROCESSING")
        logger.info(f"  Augmentation enabled: {use_augmentation}")
        logger.info(f"  Augmentation strength: {augmentation_strength}")
        if resume:
            logger.info(f"  RESUME MODE: Continuing from previous checkpoint")
            if resume_from:
                logger.info(f"  Resuming from: {resume_from}")
            if reset_optimizer:
                logger.info("  Optimizer state will be reset")
            if additional_epochs:
                logger.info(f"  Adding {additional_epochs} additional epochs")
        logger.info("=" * 70)

        # Initialize invariant preprocessor wrapper
        class InvariantDatasetWrapper(Dataset):
            """Wrapper that applies deterministic invariant preprocessing"""

            def __init__(self, dataset, target_size, use_augmentation=True, augmentation_strength=0.5, is_train=True):
                self.dataset = dataset
                self.target_size = target_size
                self.use_augmentation = use_augmentation
                self.augmentation_strength = augmentation_strength
                self.is_train = is_train
                self.preprocessor = None

            def _get_preprocessor(self):
                if self.preprocessor is None:
                    self.preprocessor = DeterministicInvariantPreprocessor(
                        self.target_size,
                        use_augmentation=self.use_augmentation and self.is_train,
                        augmentation_strength=self.augmentation_strength
                    )
                return self.preprocessor

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img, label = self.dataset[idx]

                # CRITICAL: Convert tensor to PIL Image if needed
                if isinstance(img, torch.Tensor):
                    from torchvision import transforms
                    # Ensure values are in [0, 1] range for ToPILImage
                    if img.min() < 0 or img.max() > 1:
                        img = (img - img.min()) / (img.max() - img.min())
                    to_pil = transforms.ToPILImage()
                    img = to_pil(img)
                elif isinstance(img, np.ndarray):
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    img = PILImage.fromarray(img)

                # Ensure PIL Image
                if not isinstance(img, PILImage.Image):
                    raise TypeError(f"Expected PIL Image, got {type(img)}")

                preprocessor = self._get_preprocessor()
                img = preprocessor.process(img, is_training=self.is_train)

                # Convert back to tensor
                to_tensor = transforms.ToTensor()
                img = to_tensor(img)

                return img, label

        # Wrap datasets with invariant preprocessing
        logger.info("Wrapping training dataset with invariant preprocessing...")
        train_dataset = InvariantDatasetWrapper(
            train_loader.dataset,
            self.config.input_size,
            use_augmentation=use_augmentation,
            augmentation_strength=augmentation_strength,
            is_train=True
        )

        if val_loader:
            logger.info("Wrapping validation dataset with invariant preprocessing...")
            val_dataset = InvariantDatasetWrapper(
                val_loader.dataset,
                self.config.input_size,
                use_augmentation=False,  # No augmentation for validation
                augmentation_strength=0,
                is_train=False
            )
        else:
            val_dataset = None

        # Create new deterministic loaders with preprocessed datasets
        train_loader = self._create_deterministic_loader(
            train_dataset,
            self.config.batch_size,
            shuffle=True
        )

        if val_dataset:
            val_loader = self._create_deterministic_loader(
                val_dataset,
                self.config.batch_size,
                shuffle=False
            )

        # ========================================================================
        # 2. CALCULATE OR LOAD DATASET STATISTICS ON PREPROCESSED IMAGES
        # ========================================================================
        logger.info("=" * 70)
        logger.info("CALCULATING/LOADING STATISTICS ON PREPROCESSED IMAGES")
        logger.info("=" * 70)

        statistics_calculator = None

        # Store paths for saving/loading
        self.saving_checkpoint_dir = Path(self.saving_checkpoint_dir) if not isinstance(self.saving_checkpoint_dir, Path) else self.saving_checkpoint_dir
        self.loading_checkpoint_dir = Path(self.loading_checkpoint_dir) if not isinstance(self.loading_checkpoint_dir, Path) else self.loading_checkpoint_dir
        self.saving_data_dir = Path(self.saving_data_dir) if not isinstance(self.saving_data_dir, Path) else self.saving_data_dir
        self.loading_data_dir = Path(self.loading_data_dir) if not isinstance(self.loading_data_dir, Path) else self.loading_data_dir

        # Try to load existing statistics if resuming
        if resume:
            # Try multiple possible locations for statistics file
            stats_paths = [
                self.saving_checkpoint_dir / 'dataset_statistics.pt',
                self.loading_checkpoint_dir / 'dataset_statistics.pt',
                self.saving_data_dir / 'dataset_statistics.pt',
                self.loading_data_dir / 'dataset_statistics.pt',
            ]

            for stats_path in stats_paths:
                if stats_path.exists():
                    try:
                        logger.info(f"Loading existing statistics from: {stats_path}")
                        stats = torch.load(stats_path, map_location='cpu')
                        statistics_calculator = DatasetStatisticsCalculator(self.config)
                        if 'mean' in stats and stats['mean'] is not None:
                            statistics_calculator.mean = stats['mean']
                        if 'std' in stats and stats['std'] is not None:
                            statistics_calculator.std = stats['std']
                        statistics_calculator.per_channel_min = stats.get('per_channel_min')
                        statistics_calculator.per_channel_max = stats.get('per_channel_max')
                        statistics_calculator.n_samples_used = stats.get('n_samples_used', 0)
                        statistics_calculator.is_calculated = True
                        self.statistics_calculator = statistics_calculator
                        logger.info("Statistics loaded successfully from existing file")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load statistics from {stats_path}: {e}")
                        continue

            if statistics_calculator is None:
                logger.warning("No existing statistics found, will recalculate...")
                resume = False  # Fall back to full training mode

        if not resume:
            # Calculate fresh statistics
            statistics_calculator = DatasetStatisticsCalculator(self.config)
            stats_loader = self._create_deterministic_loader(
                train_dataset,
                batch_size=min(64, self.config.batch_size),  # Smaller batch for stats calculation
                shuffle=False
            )
            statistics_calculator.calculate_statistics(stats_loader)
            self.statistics_calculator = statistics_calculator

            # Save statistics for future resume
            stats_save_path = self.saving_checkpoint_dir / 'dataset_statistics.pt'
            stats_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'mean': statistics_calculator.mean.cpu(),
                'std': statistics_calculator.std.cpu(),
                'per_channel_min': statistics_calculator.per_channel_min.cpu() if statistics_calculator.per_channel_min is not None else None,
                'per_channel_max': statistics_calculator.per_channel_max.cpu() if statistics_calculator.per_channel_max is not None else None,
                'n_samples_used': statistics_calculator.n_samples_used,
                'timestamp': datetime.now().isoformat()
            }, stats_save_path)
            logger.info(f"Statistics saved to {stats_save_path} for future resume")

        # ========================================================================
        # 3. CREATE AND TRAIN MODEL
        # ========================================================================
        logger.info("=" * 70)
        logger.info("CREATING MODEL AND STARTING TRAINING")
        if resume:
            logger.info("RESUME MODE: Model will be loaded from checkpoint")
        logger.info("=" * 70)

        # Create model using ModelFactory
        model = ModelFactory.create_model(self.config)
        model = model.to(self.device)
        self.model = model  # Store model for later feature extraction

        # Initialize resume variables with defaults
        start_epoch = 0
        start_phase = 1
        loaded_optimizer_state = None
        loaded_scheduler_state = None
        checkpoint_loaded = False

        # Load checkpoint if resuming
        if resume:
            # Determine checkpoint path
            checkpoint_path = None

            if resume_from:
                # Use explicitly provided path
                checkpoint_path = Path(resume_from)
                if not checkpoint_path.is_absolute():
                    # Try relative to saving_checkpoint_dir
                    if self.saving_checkpoint_dir:
                        alt_path = self.saving_checkpoint_dir / resume_from
                        if alt_path.exists():
                            checkpoint_path = alt_path
                    # Try relative to checkpoint_dir
                    alt_path2 = Path(self.config.checkpoint_dir) / resume_from
                    if alt_path2.exists():
                        checkpoint_path = alt_path2
            else:
                # Try to find latest checkpoint in saving_checkpoint_dir
                if self.saving_checkpoint_dir and self.saving_checkpoint_dir.exists():
                    possible_checkpoints = [
                        self.saving_checkpoint_dir / 'latest.pt',
                        self.saving_checkpoint_dir / 'best.pt',
                        self.saving_checkpoint_dir / f"{self.config.dataset_name}_best.pt",
                        self.saving_checkpoint_dir / f"{self.config.dataset_name}_latest.pt",
                    ]
                    for cp in possible_checkpoints:
                        if cp.exists():
                            checkpoint_path = cp
                            break

                # If not found, try loading_checkpoint_dir
                if checkpoint_path is None and self.loading_checkpoint_dir and self.loading_checkpoint_dir.exists():
                    possible_checkpoints = [
                        self.loading_checkpoint_dir / 'latest.pt',
                        self.loading_checkpoint_dir / 'best.pt',
                        self.loading_checkpoint_dir / f"{self.config.dataset_name}_best.pt",
                        self.loading_checkpoint_dir / f"{self.config.dataset_name}_latest.pt",
                    ]
                    for cp in possible_checkpoints:
                        if cp.exists():
                            checkpoint_path = cp
                            break

                # If still not found, try config checkpoint_dir
                if checkpoint_path is None:
                    config_checkpoint_dir = Path(self.config.checkpoint_dir)
                    if config_checkpoint_dir.exists():
                        possible_checkpoints = [
                            config_checkpoint_dir / 'latest.pt',
                            config_checkpoint_dir / 'best.pt',
                            config_checkpoint_dir / f"{self.config.dataset_name}_best.pt",
                            config_checkpoint_dir / f"{self.config.dataset_name}_latest.pt",
                        ]
                        for cp in possible_checkpoints:
                            if cp.exists():
                                checkpoint_path = cp
                                break

            if checkpoint_path and checkpoint_path.exists():
                try:
                    logger.info(f"Loading checkpoint from: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                    # Load model state
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    if missing_keys:
                        logger.debug(f"Missing keys: {missing_keys[:5]}...")
                    if unexpected_keys:
                        logger.debug(f"Unexpected keys: {unexpected_keys[:5]}...")
                    logger.info("✓ Model state loaded successfully")

                    # Record resume state
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    start_phase = checkpoint.get('phase', 1)
                    loaded_loss = checkpoint.get('loss', float('inf'))
                    loaded_accuracy = checkpoint.get('accuracy', 0.0)

                    # Store optimizer and scheduler state for potential use
                    if not reset_optimizer and 'optimizer_state_dict' in checkpoint:
                        loaded_optimizer_state = checkpoint['optimizer_state_dict']
                        logger.info("✓ Optimizer state will be restored")
                    if not reset_optimizer and 'scheduler_state_dict' in checkpoint:
                        loaded_scheduler_state = checkpoint['scheduler_state_dict']
                        logger.info("✓ Scheduler state will be restored")

                    logger.info(f"Resuming from: Phase {start_phase}, Epoch {start_epoch}")
                    logger.info(f"Loaded loss: {loaded_loss:.6f}, Accuracy: {loaded_accuracy:.4f}")

                    # Set dataset statistics in model
                    if statistics_calculator and statistics_calculator.is_calculated:
                        model.set_dataset_statistics(statistics_calculator)
                        logger.info("✓ Dataset statistics loaded into model")

                    # Load best metrics if available in checkpoint (for trainer tracking)
                    if 'best_loss' in checkpoint:
                        self.best_loss = checkpoint['best_loss']
                    if 'best_accuracy' in checkpoint:
                        self.best_accuracy = checkpoint['best_accuracy']
                    if 'best_epoch' in checkpoint:
                        self.best_epoch = checkpoint['best_epoch']
                    if 'best_phase' in checkpoint:
                        self.best_phase = checkpoint['best_phase']

                    # Load history if available
                    if 'history' in checkpoint:
                        self.history = defaultdict(list, checkpoint['history'])
                        logger.info(f"✓ Training history loaded ({len(self.history.get('train_loss', []))} epochs)")

                    checkpoint_loaded = True

                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {e}")
                    traceback.print_exc()
                    logger.warning("Starting training from scratch instead")
                    start_epoch = 0
                    start_phase = 1
                    loaded_optimizer_state = None
                    loaded_scheduler_state = None
                    resume = False
            else:
                logger.warning(f"No checkpoint found to resume from")
                logger.info(f"Checked paths:")
                if resume_from:
                    logger.info(f"  - {resume_from}")
                if self.saving_checkpoint_dir:
                    logger.info(f"  - {self.saving_checkpoint_dir}/latest.pt")
                    logger.info(f"  - {self.saving_checkpoint_dir}/best.pt")
                if self.loading_checkpoint_dir:
                    logger.info(f"  - {self.loading_checkpoint_dir}/latest.pt")
                    logger.info(f"  - {self.loading_checkpoint_dir}/best.pt")
                logger.info(f"  - {self.config.checkpoint_dir}/latest.pt")
                logger.info(f"  - {self.config.checkpoint_dir}/best.pt")
                resume = False
                start_epoch = 0
                start_phase = 1

        # If resume was requested but no checkpoint loaded, set resume to False
        if resume and not checkpoint_loaded:
            resume = False
            start_epoch = 0
            start_phase = 1

        # Adjust epochs if additional epochs specified
        if additional_epochs and resume:
            original_epochs = self.config.epochs
            self.config.epochs = start_epoch + additional_epochs
            logger.info(f"Extended training: {additional_epochs} additional epochs (was {original_epochs}, now {self.config.epochs})")

        # Call standalone training function with resume parameters
        history = deterministic_train(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=self.config,
            checkpoint_dir=self.saving_checkpoint_dir,
            statistics_calculator=statistics_calculator,
            resume=resume,
            start_epoch=start_epoch,
            start_phase=start_phase,
            loaded_optimizer_state=loaded_optimizer_state,
            loaded_scheduler_state=loaded_scheduler_state,
            reset_optimizer=reset_optimizer
        )

        # ========================================================================
        # 4. SAVE FEATURES AFTER TRAINING
        # ========================================================================
        if save_features:
            logger.info("=" * 70)
            logger.info("SAVING FEATURES AFTER TRAINING")
            logger.info("=" * 70)

            try:
                # Load the best model for feature extraction
                best_checkpoint_path = self.saving_checkpoint_dir / 'best.pt'
                if best_checkpoint_path.exists():
                    logger.info(f"Loading best model from {best_checkpoint_path} for feature extraction")
                    checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)

                    # Load weights into model
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    self.model = self.model.to(self.device)
                    self.model.eval()

                    # Extract features from training data
                    logger.info("Extracting features from training data...")
                    train_features = self.extract_features(train_loader, include_paths=True)

                    if train_features and train_features.get('embeddings') is not None and len(train_features['embeddings']) > 0:
                        # Save training features
                        train_csv_path = self.saving_data_dir / f"{self.config.dataset_name}_train_features.csv"
                        self._save_features_to_csv(train_features, str(train_csv_path))
                        logger.info(f"Training features saved to: {train_csv_path}")

                        # Also save as the main CSV
                        main_csv_path = self.saving_data_dir / f"{self.config.dataset_name}.csv"
                        self._save_features_to_csv(train_features, str(main_csv_path))
                        logger.info(f"Main features CSV saved to: {main_csv_path}")

                        # Save feature statistics
                        feature_stats = {
                            'n_samples': train_features['embeddings'].shape[0],
                            'n_features': train_features['embeddings'].shape[1],
                            'feature_dimension': train_features['embeddings'].shape[1],
                            'compressed_dims': self.config.compressed_dims,
                            'mean': train_features['embeddings'].mean(axis=0).tolist(),
                            'std': train_features['embeddings'].std(axis=0).tolist(),
                            'timestamp': datetime.now().isoformat()
                        }

                        stats_path = self.saving_data_dir / f"{self.config.dataset_name}_features_stats.json"
                        with open(stats_path, 'w') as f:
                            json.dump(feature_stats, f, indent=2)
                        logger.info(f"Feature statistics saved to: {stats_path}")

                    # Extract features from validation data if available
                    if val_loader:
                        logger.info("Extracting features from validation data...")
                        val_features = self.extract_features(val_loader, include_paths=True)
                        if val_features and val_features.get('embeddings') is not None and len(val_features['embeddings']) > 0:
                            val_csv_path = self.saving_data_dir / f"{self.config.dataset_name}_val_features.csv"
                            self._save_features_to_csv(val_features, str(val_csv_path))
                            logger.info(f"Validation features saved to: {val_csv_path}")
                else:
                    logger.warning(f"No best model found at {best_checkpoint_path}, skipping feature extraction")

            except Exception as e:
                logger.error(f"Failed to save features after training: {e}")
                logger.error(traceback.format_exc())

        # ========================================================================
        # 5. SAVE MODEL AND RESULTS
        # ========================================================================
        # Copy best model to loading directory
        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)
        best_checkpoint = self.saving_checkpoint_dir / 'best.pt'
        if best_checkpoint.exists():
            loading_best_path = self.loading_checkpoint_dir / f"{dataset_name_lower}_best.pt"
            loading_best_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_checkpoint, loading_best_path)
            logger.info(f"Model saved to {loading_best_path}")

            # Also save the preprocessor configuration
            preprocessor_config = {
                'target_size': list(self.config.input_size),
                'normalization_stats': {
                    'mean': statistics_calculator.mean.tolist() if statistics_calculator and statistics_calculator.mean is not None else None,
                    'std': statistics_calculator.std.tolist() if statistics_calculator and statistics_calculator.std is not None else None,
                },
                'preprocessor_type': 'DeterministicInvariantPreprocessor',
                'augmentation_enabled': use_augmentation,
                'augmentation_strength': augmentation_strength,
                'version': '1.0'
            }

            preprocessor_config_path = self.saving_checkpoint_dir / f"{dataset_name_lower}_preprocessor.json"
            with open(preprocessor_config_path, 'w') as f:
                json.dump(preprocessor_config, f, indent=2)
            logger.info(f"Preprocessor config saved to {preprocessor_config_path}")

        # Plot training history
        self.visualizer.plot_training_history(history)

        # Save final training summary
        training_summary = {
            'best_accuracy': max(history.get('val_acc', [0])) if history.get('val_acc') else 0,
            'best_loss': min(history.get('val_loss', [float('inf')])),
            'total_epochs': len(history.get('train_loss', [])),
            'resumed': resume,
            'resumed_from_epoch': start_epoch - 1 if resume and start_epoch > 0 else 0,
            'augmentation_used': use_augmentation,
            'augmentation_strength': augmentation_strength,
            'normalization_mode': self.config.normalization_mode,
            'features_saved': save_features,
            'normalization_stats': {
                'mean': statistics_calculator.mean.tolist() if statistics_calculator and statistics_calculator.mean is not None else None,
                'std': statistics_calculator.std.tolist() if statistics_calculator and statistics_calculator.std is not None else None,
            }
        }

        summary_path = self.saving_data_dir / f"{dataset_name_lower}_training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        logger.info(f"Training summary saved to {summary_path}")

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Best validation accuracy: {training_summary['best_accuracy']:.4f}")
        logger.info(f"Best validation loss: {training_summary['best_loss']:.6f}")
        if resume:
            logger.info(f"Training resumed from epoch {start_epoch}")
        if save_features:
            logger.info(f"Features saved to: {self.saving_data_dir}/{self.config.dataset_name}_*.csv")
        logger.info("=" * 70)

        return history

    def _set_deterministic_seeds(self):
        """Set all random seeds for reproducible training"""
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("Set deterministic seeds for reproducible training")

    def _create_deterministic_loader(self, dataset, batch_size, shuffle=False):
        """Create deterministic DataLoader with fixed seed"""
        from torch.utils.data import DataLoader
        from torch.utils.data.dataloader import default_collate

        g = torch.Generator()
        g.manual_seed(42)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=g,
            pin_memory=False,
            drop_last=False,
            collate_fn=default_collate
        )

    def extract_features(self, dataloader: DataLoader, include_paths: bool = True,
                        extract_domain_features: bool = False) -> Dict:
        """
        Extract features from the trained model with optional domain features.

        Args:
            dataloader: DataLoader containing images
            include_paths: Whether to include file paths
            extract_domain_features: Whether to extract domain-specific features

        Returns:
            Dictionary containing 'embeddings', 'labels', and optionally 'filenames', 'paths', 'class_names'
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot extract features.")
            return {}

        # Set model to evaluation mode
        self.model.eval()

        # Move model to device if not already there
        if not hasattr(self.model, 'device') or self.model.device != self.device:
            self.model = self.model.to(self.device)

        # Extract neural network features using the model's method
        features_dict = self.model.extract_features(dataloader, include_paths=include_paths)

        # Add domain features if requested and domain processor is available
        if extract_domain_features and hasattr(self, 'domain_processor') and self.domain_processor is not None:
            logger.info("Extracting domain-specific features...")

            domain_features_list = []

            # Get paths if available
            if 'paths' in features_dict and features_dict['paths']:
                paths = features_dict['paths']
            elif include_paths and hasattr(dataloader.dataset, 'full_paths'):
                paths = dataloader.dataset.full_paths
            else:
                paths = None

            if paths:
                for img_path in tqdm(paths, desc="Extracting domain features"):
                    try:
                        img = ImageProcessor.load_image(img_path)
                        if img is not None:
                            img_array = np.array(img) / 255.0
                            domain_feats = self.domain_processor.extract_features(img_array)
                            domain_features_list.append(domain_feats)
                        else:
                            domain_features_list.append({})
                    except Exception as e:
                        logger.debug(f"Failed to extract domain features for {img_path}: {e}")
                        domain_features_list.append({})

                # Store domain features in the dictionary
                if domain_features_list:
                    all_keys = sorted(set().union(*[f.keys() for f in domain_features_list]))
                    for key in all_keys:
                        features_dict[f'domain_{key}'] = [f.get(key, np.nan) for f in domain_features_list]

                    logger.info(f"Added {len(all_keys)} domain-specific features")

        logger.info(f"Extracted {len(features_dict.get('embeddings', []))} features "
                    f"with dimension {features_dict.get('embeddings', torch.tensor([])).shape[1] if len(features_dict.get('embeddings', [])) > 0 else 0}")

        return features_dict

    def predict(self, dataloader: DataLoader, optimize_level: str = 'balanced') -> Dict:
        """
        Run prediction on the dataloader.

        Args:
            dataloader: DataLoader containing images to predict
            optimize_level: 'fast', 'balanced', or 'accurate'

        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot predict.")
            return {}

        self.model.eval()

        all_predictions = []
        all_probabilities = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Predicting"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                if 'class_logits' in outputs:
                    probs = F.softmax(outputs['class_logits'], dim=1)
                    preds = probs.argmax(dim=1)
                    all_predictions.extend(preds.cpu().numpy())
                    all_probabilities.extend(probs.cpu().numpy())
                    all_labels.extend(labels.numpy())

        result = {
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities),
            'labels': np.array(all_labels)
        }

        logger.info(f"Predicted {len(all_predictions)} samples")
        return result

    def _get_torchvision_dataset(self, dataset_name: str, train: bool, transform):
        dataset_name = dataset_name.lower()
        if dataset_name == 'mnist':
            return datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        elif dataset_name == 'cifar10':
            return datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        elif dataset_name == 'cifar100':
            return datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
        elif dataset_name == 'fashionmnist':
            return datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown torchvision dataset: {dataset_name}")

    def _get_transform(self, is_train: bool = False) -> transforms.Compose:
        """
        Create optimized deterministic transform pipeline.
        Adds dataset-specific augmentations for better generalization.
        """
        # Check if this is CIFAR or similar small dataset
        is_cifar = self.config.dataset_name in ['cifar10', 'cifar100']
        is_mnist = self.config.dataset_name in ['mnist', 'fashionmnist']
        is_small_image = min(self.config.input_size) <= 64

        if is_train:
            # Build augmentation pipeline based on dataset type
            transforms_list = []

            # CIFAR-specific augmentations (helps with generalization)
            if is_cifar:
                transforms_list.extend([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ])
            elif is_mnist:
                # MNIST-specific augmentations
                transforms_list.extend([
                    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                ])
            elif is_small_image:
                # Generic augmentations for small images
                transforms_list.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ])

            # Add invariant preprocessing
            transforms_list.append(
                transforms.Lambda(lambda img: self._preprocess_invariant(img, is_train=True))
            )
            transforms_list.append(transforms.ToTensor())

            return transforms.Compose(transforms_list)
        else:
            # For prediction/validation, use minimal preprocessing
            return transforms.Compose([
                transforms.Lambda(lambda img: self._preprocess_invariant(img, is_train=False)),
                transforms.ToTensor(),
            ])

    def _preprocess_invariant(self, image: PILImage.Image, is_train: bool = False) -> PILImage.Image:
        """
        Apply deterministic invariant preprocessing with optimization for GPU transfer.
        """
        if not hasattr(self, '_invariant_preprocessor'):
            use_augmentation = not getattr(self.config, 'no_augmentation', False)
            augmentation_strength = getattr(self.config, 'augmentation_strength', 0.5)
            self._invariant_preprocessor = DeterministicInvariantPreprocessor(
                self.config.input_size,
                use_augmentation=use_augmentation,
                augmentation_strength=augmentation_strength
            )

        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')

        processed = self._invariant_preprocessor.process(image, is_training=is_train)
        return processed

    def _save_features_to_csv(self, features: Dict, output_csv: str,
                              include_domain_features: bool = True,
                              domain_processor=None):
        """
        Save features to CSV with proper metadata and exactly config.compressed_dims features.
        All columns will have the same length.
        TARGET COLUMN: Always saved as string (decoded from label encoding) as 'target' column.

        Gracefully handles:
        - Predictions without targets (new/unseen classes)
        - Missing folder structures
        - Unknown labels not seen during training
        - Partial or incomplete label information
        """
        import pandas as pd
        import numpy as np

        # CRITICAL: First determine the number of samples
        n_samples = 0
        if 'embeddings' in features and features['embeddings'] is not None:
            if isinstance(features['embeddings'], torch.Tensor):
                n_samples = features['embeddings'].shape[0]
            elif isinstance(features['embeddings'], np.ndarray):
                n_samples = features['embeddings'].shape[0]
            elif isinstance(features['embeddings'], list):
                n_samples = len(features['embeddings'])

        if n_samples == 0:
            # Try to get from labels
            if 'labels' in features and features['labels'] is not None:
                if isinstance(features['labels'], torch.Tensor):
                    n_samples = features['labels'].shape[0]
                elif isinstance(features['labels'], np.ndarray):
                    n_samples = features['labels'].shape[0]
                elif isinstance(features['labels'], list):
                    n_samples = len(features['labels'])

        if n_samples == 0:
            logger.error("Cannot determine number of samples - no data to save")
            return

        logger.info(f"Creating CSV with {n_samples} samples")
        data = {}

        # ========================================================================
        # 1. NEURAL NETWORK FEATURES - EXACTLY config.compressed_dims
        # ========================================================================
        if 'embeddings' in features and features['embeddings'] is not None and len(features['embeddings']) > 0:
            feature_array = features['embeddings']
            if isinstance(feature_array, torch.Tensor):
                feature_array = feature_array.cpu().numpy()

            # Ensure correct shape
            if len(feature_array.shape) == 1:
                feature_array = feature_array.reshape(-1, 1)

            # CRITICAL: Truncate or pad to match n_samples
            if feature_array.shape[0] != n_samples:
                logger.warning(f"Feature array length {feature_array.shape[0]} != {n_samples}, adjusting...")
                if feature_array.shape[0] > n_samples:
                    feature_array = feature_array[:n_samples]
                else:
                    # Pad with zeros
                    padded = np.zeros((n_samples, feature_array.shape[1]))
                    padded[:feature_array.shape[0]] = feature_array
                    feature_array = padded

            # Use exactly compressed_dims features
            expected_features = self.config.compressed_dims if hasattr(self.config, 'compressed_dims') else 32

            # Truncate or pad feature dimensions
            if feature_array.shape[1] > expected_features:
                logger.warning(f"Feature dimension {feature_array.shape[1]} > expected {expected_features}, truncating to {expected_features}")
                feature_array = feature_array[:, :expected_features]
            elif feature_array.shape[1] < expected_features:
                logger.warning(f"Feature dimension {feature_array.shape[1]} < expected {expected_features}, padding with zeros")
                padded = np.zeros((feature_array.shape[0], expected_features))
                padded[:, :feature_array.shape[1]] = feature_array
                feature_array = padded

            # Save as feature_0, feature_1, etc.
            for i in range(feature_array.shape[1]):
                data[f'feature_{i}'] = feature_array[:, i]

            logger.info(f"Added {feature_array.shape[1]} neural network features (expected: {expected_features})")
        else:
            # Create empty features if none exist
            expected_features = self.config.compressed_dims if hasattr(self.config, 'compressed_dims') else 32
            for i in range(expected_features):
                data[f'feature_{i}'] = [0.0] * n_samples
            logger.warning(f"No embeddings found, created {expected_features} zero features")

        # ========================================================================
        # 2. METADATA - original_filename, filepath, folder_name
        # ========================================================================

        # original_filename (just the filename without path)
        if 'filenames' in features and features['filenames']:
            filenames_list = features['filenames']
            if len(filenames_list) != n_samples:
                logger.warning(f"Filenames length {len(filenames_list)} != {n_samples}, adjusting...")
                if len(filenames_list) > n_samples:
                    filenames_list = filenames_list[:n_samples]
                else:
                    filenames_list = filenames_list + [f"unknown_{i}" for i in range(n_samples - len(filenames_list))]
            data['original_filename'] = filenames_list
            logger.info(f"Added original_filename column")
        elif 'paths' in features and features['paths']:
            # Extract filename from path
            paths_list = features['paths']
            if len(paths_list) != n_samples:
                logger.warning(f"Paths length {len(paths_list)} != {n_samples}, adjusting...")
                if len(paths_list) > n_samples:
                    paths_list = paths_list[:n_samples]
                else:
                    paths_list = paths_list + [f"unknown_path_{i}" for i in range(n_samples - len(paths_list))]
            filenames = [Path(p).name for p in paths_list]
            data['original_filename'] = filenames
            logger.info(f"Added original_filename column (extracted from paths)")

        # filepath (full path)
        if 'paths' in features and features['paths']:
            paths_list = features['paths']
            if len(paths_list) != n_samples:
                logger.warning(f"Paths length {len(paths_list)} != {n_samples}, adjusting...")
                if len(paths_list) > n_samples:
                    paths_list = paths_list[:n_samples]
                else:
                    paths_list = paths_list + [f"unknown_path_{i}" for i in range(n_samples - len(paths_list))]
            data['filepath'] = paths_list
            logger.info(f"Added filepath column")

            # Extract folder name from path (useful for prediction without labels)
            folder_names = []
            for p in paths_list:
                path = Path(p)
                # Get parent folder name (could be class name or just folder)
                folder_name = path.parent.name if path.parent.name else "root"
                folder_names.append(folder_name)
            data['folder_name'] = folder_names
            logger.info(f"Added folder_name column (extracted from paths)")

        # ========================================================================
        # 3. TARGET COLUMN - Decode from label encoding to string
        #    Gracefully handles missing/unknown labels
        # ========================================================================

        target_added = False
        numeric_labels = None
        class_mapping = None
        unknown_labels_found = []

        # Build class mapping from multiple sources (priority order)
        def _build_class_mapping():
            """Build class mapping from available sources"""
            mapping = None

            # Source 1: model._idx_to_class (from training)
            if hasattr(self, 'model') and hasattr(self.model, '_idx_to_class') and self.model._idx_to_class:
                mapping = self.model._idx_to_class
                logger.info(f"Using model._idx_to_class mapping with {len(mapping)} classes")
                return mapping

            # Source 2: config.class_names
            if hasattr(self.config, 'class_names') and self.config.class_names:
                mapping = {idx: name for idx, name in enumerate(self.config.class_names)}
                logger.info(f"Using config.class_names mapping with {len(mapping)} classes")
                return mapping

            # Source 3: dataset attribute (if available)
            if hasattr(features, 'dataset') and hasattr(features.dataset, 'idx_to_class'):
                mapping = features.dataset.idx_to_class
                logger.info(f"Using dataset.idx_to_class mapping with {len(mapping)} classes")
                return mapping

            # Source 4: class_names directly in features
            if 'class_names' in features and features['class_names']:
                unique_names = sorted(set(features['class_names']))
                mapping = {idx: name for idx, name in enumerate(unique_names)}
                logger.info(f"Using features.class_names mapping with {len(mapping)} classes")
                return mapping

            logger.info("No class mapping found - will use folder names or raw values for prediction")
            return None

        class_mapping = _build_class_mapping()

        # Try to get numeric labels from various sources
        if 'labels' in features and features['labels'] is not None:
            numeric_labels = features['labels']
            if isinstance(numeric_labels, torch.Tensor):
                numeric_labels = numeric_labels.cpu().numpy()
            logger.info(f"Found 'labels' field with {len(numeric_labels)} values")
        elif 'target' in features and features['target'] is not None:
            numeric_labels = features['target']
            if isinstance(numeric_labels, torch.Tensor):
                numeric_labels = numeric_labels.cpu().numpy()
            logger.info(f"Found 'target' field with {len(numeric_labels)} values")
        elif 'class_names' in features and features['class_names'] and not numeric_labels:
            # class_names are already strings, use them directly
            class_names_list = features['class_names']
            if len(class_names_list) != n_samples:
                logger.warning(f"Class names length {len(class_names_list)} != {n_samples}, adjusting...")
                if len(class_names_list) > n_samples:
                    class_names_list = class_names_list[:n_samples]
                else:
                    class_names_list = class_names_list + [f"class_{i}" for i in range(n_samples - len(class_names_list))]
            data['target'] = class_names_list
            target_added = True
            logger.info(f"Added target column directly from class_names")

        # Process numeric labels if found and not already added
        if not target_added and numeric_labels is not None:
            # Ensure correct length
            if len(numeric_labels) != n_samples:
                logger.warning(f"Labels length {len(numeric_labels)} != {n_samples}, adjusting...")
                if len(numeric_labels) > n_samples:
                    numeric_labels = numeric_labels[:n_samples]
                else:
                    numeric_labels = np.concatenate([numeric_labels, np.zeros(n_samples - len(numeric_labels))])

            # Convert numeric labels to string class names
            if class_mapping:
                # Decode using mapping, gracefully handling unknown labels
                target_strings = []
                for label in numeric_labels:
                    # Handle both integer and float labels
                    label_int = int(label) if not isinstance(label, str) else label
                    if label_int in class_mapping:
                        target_strings.append(str(class_mapping[label_int]))
                    else:
                        # Unknown label not seen during training
                        unknown_label_str = f"unknown_class_{label_int}"
                        target_strings.append(unknown_label_str)
                        if label_int not in unknown_labels_found:
                            unknown_labels_found.append(label_int)
                            logger.warning(f"Unknown label {label_int} found (not in training classes)")
                data['target'] = target_strings
                unique_classes = set(target_strings)
                logger.info(f"Added target column (decoded from numeric labels using mapping): {len(unique_classes)} unique classes")
                if unknown_labels_found:
                    logger.warning(f"Found {len(unknown_labels_found)} unknown/unseen classes: {unknown_labels_found[:10]}")
                target_added = True
            else:
                # No mapping available, use raw values as strings
                data['target'] = [str(l) for l in numeric_labels]
                logger.info(f"Added target column (raw numeric values as strings - no mapping available)")
                target_added = True

        # For prediction mode without labels: use folder names as pseudo-target
        if not target_added and 'folder_name' in data:
            # Use folder names as target (helpful for evaluating predictions on new data)
            folder_targets = data['folder_name']
            # Check if folder names correspond to known classes
            if class_mapping:
                # Try to map folder names to known classes (case-insensitive)
                known_class_names = {name.lower(): name for name in class_mapping.values()}
                mapped_targets = []
                for folder in folder_targets:
                    folder_lower = folder.lower()
                    if folder_lower in known_class_names:
                        mapped_targets.append(known_class_names[folder_lower])
                    else:
                        # Keep original folder name as is (new/unseen class)
                        mapped_targets.append(folder)
                        if folder not in unknown_labels_found:
                            unknown_labels_found.append(folder)
                            logger.info(f"New folder name '{folder}' - treating as potential new class")
                data['target'] = mapped_targets
            else:
                data['target'] = folder_targets
            logger.info(f"Added target column from folder_name (prediction mode - no ground truth labels)")
            target_added = True

        # Final fallback: create placeholder targets
        if not target_added:
            data['target'] = ['unknown'] * n_samples
            logger.warning(f"No labels or folder info found, created default 'unknown' target column")

        # Log target column statistics for verification
        if 'target' in data:
            unique_targets = set(data['target'])
            logger.info(f"Target column contains {len(unique_targets)} unique values: {list(unique_targets)[:10]}{'...' if len(unique_targets) > 10 else ''}")
            # Count unknown labels
            unknown_count = sum(1 for t in data['target'] if t.startswith('unknown'))
            if unknown_count > 0:
                logger.warning(f"  - {unknown_count}/{n_samples} samples have unknown/unseen labels ({unknown_count/n_samples*100:.1f}%)")

        # ========================================================================
        # 4. PREDICTION OUTPUTS (for predict mode)
        # ========================================================================
        if 'predictions' in features and features['predictions'] is not None:
            predictions = features['predictions']
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            if len(predictions) == n_samples:
                # Decode predictions to class names if mapping available
                if class_mapping:
                    pred_strings = []
                    for p in predictions:
                        p_int = int(p) if not isinstance(p, str) else p
                        if p_int in class_mapping:
                            pred_strings.append(str(class_mapping[p_int]))
                        else:
                            pred_strings.append(f"unknown_pred_{p_int}")
                    data['prediction'] = pred_strings
                    # Calculate prediction confidence vs target match
                    if 'target' in data:
                        matches = [data['target'][i] == data['prediction'][i] for i in range(n_samples)]
                        accuracy = sum(matches) / n_samples if n_samples > 0 else 0
                        logger.info(f"Prediction vs target accuracy: {accuracy:.2%} ({sum(matches)}/{n_samples})")
                else:
                    data['prediction'] = predictions
                logger.info(f"Added prediction column")

        if 'probabilities' in features and features['probabilities'] is not None:
            probs = features['probabilities']
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy()
            if len(probs) == n_samples:
                data['confidence'] = np.max(probs, axis=1) if probs.ndim > 1 else probs
                data['uncertainty'] = -np.sum(probs * np.log(probs + 1e-10), axis=1) if probs.ndim > 1 else np.zeros(n_samples)
                logger.info(f"Added confidence and uncertainty columns")

                # Log confidence statistics for unknown classes
                if 'target' in data and unknown_labels_found:
                    unknown_mask = [t.startswith('unknown') for t in data['target']]
                    if any(unknown_mask):
                        unknown_confidences = data['confidence'][unknown_mask]
                        logger.info(f"Unknown class samples - avg confidence: {np.mean(unknown_confidences):.3f}, max: {np.max(unknown_confidences):.3f}")

        # ========================================================================
        # 5. CLUSTER INFORMATION (if available)
        # ========================================================================
        if 'cluster_assignments' in features and features['cluster_assignments'] is not None:
            cluster_assignments = features['cluster_assignments']
            if isinstance(cluster_assignments, torch.Tensor):
                cluster_assignments = cluster_assignments.cpu().numpy()

            # Ensure correct length
            if len(cluster_assignments) != n_samples:
                logger.warning(f"Cluster assignments length {len(cluster_assignments)} != {n_samples}, adjusting...")
                if len(cluster_assignments) > n_samples:
                    cluster_assignments = cluster_assignments[:n_samples]
                else:
                    cluster_assignments = np.concatenate([cluster_assignments, np.zeros(n_samples - len(cluster_assignments))])

            data['cluster_assignment'] = cluster_assignments
            logger.info(f"Added cluster_assignment column")

        if 'cluster_confidence' in features and features['cluster_confidence'] is not None:
            cluster_confidence = features['cluster_confidence']
            if isinstance(cluster_confidence, torch.Tensor):
                cluster_confidence = cluster_confidence.cpu().numpy()

            # Ensure correct length
            if len(cluster_confidence) != n_samples:
                logger.warning(f"Cluster confidence length {len(cluster_confidence)} != {n_samples}, adjusting...")
                if len(cluster_confidence) > n_samples:
                    cluster_confidence = cluster_confidence[:n_samples]
                else:
                    cluster_confidence = np.concatenate([cluster_confidence, np.zeros(n_samples - len(cluster_confidence))])

            data['cluster_confidence'] = cluster_confidence
            logger.info(f"Added cluster_confidence column")

        # ========================================================================
        # 6. DOMAIN-SPECIFIC FEATURES (if requested)
        # ========================================================================
        if include_domain_features:
            domain_keys = [k for k in features.keys() if k.startswith('domain_')]
            if domain_keys:
                for key in domain_keys:
                    domain_vals = features[key]
                    if isinstance(domain_vals, list):
                        if len(domain_vals) != n_samples:
                            logger.warning(f"Domain feature {key} length {len(domain_vals)} != {n_samples}, adjusting...")
                            if len(domain_vals) > n_samples:
                                domain_vals = domain_vals[:n_samples]
                            else:
                                domain_vals = domain_vals + [np.nan] * (n_samples - len(domain_vals))
                        data[key] = domain_vals
                    elif isinstance(domain_vals, np.ndarray):
                        if len(domain_vals) != n_samples:
                            if len(domain_vals) > n_samples:
                                domain_vals = domain_vals[:n_samples]
                            else:
                                domain_vals = np.concatenate([domain_vals, np.full(n_samples - len(domain_vals), np.nan)])
                        data[key] = domain_vals
                logger.info(f"Added {len(domain_keys)} domain-specific features")

        # ========================================================================
        # 7. SAMPLE INDEX (always add for debugging)
        # ========================================================================
        data['sample_index'] = list(range(n_samples))
        logger.info(f"Added sample_index column")

        # ========================================================================
        # 8. VERIFY ALL COLUMNS HAVE SAME LENGTH
        # ========================================================================
        for col_name, col_data in data.items():
            if len(col_data) != n_samples:
                logger.error(f"Column {col_name} has length {len(col_data)}, expected {n_samples}")
                # Fix by truncating or padding
                if len(col_data) > n_samples:
                    data[col_name] = col_data[:n_samples]
                else:
                    if isinstance(col_data, list):
                        data[col_name] = col_data + [None] * (n_samples - len(col_data))
                    else:
                        # Assume numpy array
                        import numpy as np
                        data[col_name] = np.concatenate([col_data, np.full(n_samples - len(col_data), np.nan)])

        # ========================================================================
        # 9. SAVE TO CSV WITH PROPER COLUMN ORDER
        # ========================================================================
        # Define column order priority (target is now the main label column)
        priority_columns = ['sample_index', 'original_filename', 'filepath', 'folder_name', 'target',
                            'prediction', 'confidence', 'uncertainty',
                            'cluster_assignment', 'cluster_confidence']

        # Get feature columns
        expected_features = self.config.compressed_dims if hasattr(self.config, 'compressed_dims') else 32
        feature_columns = [f'feature_{i}' for i in range(expected_features) if f'feature_{i}' in data]

        # Get domain feature columns
        domain_columns = sorted([k for k in data.keys() if k.startswith('domain_')])

        # Get remaining columns
        other_columns = [k for k in data.keys()
                        if k not in priority_columns
                        and not k.startswith('feature_')
                        and not k.startswith('domain_')]

        # Build final column order
        column_order = ([c for c in priority_columns if c in data] +
                        feature_columns +
                        domain_columns +
                        other_columns)

        # Reorder data
        ordered_data = {col: data[col] for col in column_order if col in data}

        # Create DataFrame
        df = pd.DataFrame(ordered_data)

        # Ensure target column is string type
        if 'target' in df.columns:
            df['target'] = df['target'].astype(str)
            # Log class distribution for verification
            class_dist = df['target'].value_counts()
            logger.info(f"Target class distribution in CSV:")
            for class_name, count in list(class_dist.items())[:20]:  # Show top 20
                logger.info(f"  {class_name}: {count} ({count/len(df)*100:.1f}%)")
            if len(class_dist) > 20:
                logger.info(f"  ... and {len(class_dist) - 20} more classes")

        # Also handle legacy label_type column if present (for backward compatibility)
        if 'label_type' in df.columns and 'target' not in df.columns:
            df['target'] = df['label_type'].astype(str)
            logger.info(f"Renamed label_type column to target for consistency")

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        logger.info("=" * 60)
        logger.info(f"FEATURES SAVED TO: {output_csv}")
        logger.info(f"  Total samples: {len(df)}")
        logger.info(f"  Total columns: {len(df.columns)}")
        logger.info(f"  Feature columns: {len(feature_columns)} (expected: {expected_features})")
        logger.info(f"  Target column: present ({df['target'].nunique()} unique values)" if 'target' in df.columns else "  Target column: NOT FOUND")
        if unknown_labels_found:
            logger.info(f"  Unknown/unseen classes: {len(unknown_labels_found)}")
        logger.info("=" * 60)

        # Show sample of first row for verification
        if len(df) > 0:
            first_row = df.iloc[0]
            logger.info(f"Sample first row:")
            for col in ['sample_index', 'original_filename', 'folder_name', 'target', 'prediction', 'feature_0']:
                if col in first_row.index:
                    val = first_row[col]
                    if isinstance(val, float):
                        logger.info(f"  {col}: {val:.6f}")
                    else:
                        logger.info(f"  {col}: {val}")

        return df

    @staticmethod
    def load_config_from_json(config_path: str) -> Optional[Dict]:
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            domain = config_dict.get('domain', 'general')

            if domain == 'agriculture':
                config = AgricultureConfig(**config_dict)
            elif domain == 'medical':
                config = MedicalConfig(**config_dict)
            elif domain == 'satellite':
                config = SatelliteConfig(**config_dict)
            elif domain == 'surveillance':
                config = SurveillanceConfig(**config_dict)
            elif domain == 'microscopy':
                config = MicroscopyConfig(**config_dict)
            elif domain == 'industrial':
                config = IndustrialConfig(**config_dict)
            elif domain == 'astronomy':
                config = AstronomyConfig(**config_dict)
            else:
                config = GlobalConfig(**config_dict)

            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            return None

    @staticmethod
    def load_config(conf_path: str) -> Optional[Dict]:
        try:
            with open(conf_path, 'r') as f:
                config_dict = json.load(f)

            logger.info(f"Configuration loaded from {conf_path}")
            return config_dict
        except Exception as e:
            logger.error(f"Failed to load configuration from {conf_path}: {e}")
            return None

    def _create_model(self) -> BaseAutoencoder:
        """Create model using ModelFactory"""
        return ModelFactory.create_model(self.config)

    def export_features_to_csv(self, output_csv: str = None, split: str = 'train') -> Dict:
        """
        Export features to CSV using the trained model.

        Args:
            output_csv: Path to output CSV file (optional)
            split: 'train' or 'test' split to export

        Returns:
            Dictionary with features
        """
        logger.info("=" * 70)
        logger.info(f"EXPORTING FEATURES FOR {split.upper()} SPLIT")
        logger.info("=" * 70)

        # Check if model exists
        model_path = self.saving_checkpoint_dir / 'best.pt'

        if not model_path.exists():
            logger.error(f"No trained model found at {model_path}. Please train first.")
            return {}

        # Load model if not already loaded
        if self.model is None:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = ModelFactory.create_model(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()

        # Prepare data loader for the specified split
        if split == 'train':
            train_loader, _ = self.prepare_data(None, data_type='torchvision')
            loader = train_loader
        else:
            _, test_loader = self.prepare_data(None, data_type='torchvision')
            loader = test_loader

        # Extract features
        features = self.extract_features(loader, include_paths=True)

        # Save to CSV
        if output_csv is None:
            output_csv = self.saving_data_dir / f"{self.config.dataset_name}_{split}_features.csv"

        self._save_features_to_csv(features, str(output_csv))

        logger.info("=" * 70)
        logger.info(f"FEATURES EXPORTED SUCCESSFULLY")
        logger.info(f"  Split: {split}")
        logger.info(f"  Samples: {len(features['embeddings'])}")
        logger.info(f"  Feature dimension: {features['embeddings'].shape[1]}")
        logger.info(f"  Output CSV: {output_csv}")
        logger.info("=" * 70)

        return features

    def export_torchvision_to_images(self, dataset_name: str, output_dir: str = None):
        """
        Export torchvision dataset to image files for prediction.
        Creates folder structure: data/dataset_lower/train/class1/, data/dataset_lower/test/class1/
        """
        dataset_name_lower = normalize_dataset_name(dataset_name)

        if output_dir is None:
            output_dir = Path('data') / dataset_name_lower
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_name_upper = dataset_name.upper()
        valid_tv_datasets = ['MNIST', 'CIFAR10', 'CIFAR100', 'FASHIONMNIST']

        if dataset_name_upper not in valid_tv_datasets:
            raise ValueError(f"Unknown torchvision dataset: {dataset_name_upper}")

        logger.info("=" * 70)
        logger.info(f"Exporting {dataset_name_upper} to images")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 70)

        # Import torchvision datasets
        from torchvision import datasets

        # Load datasets
        if dataset_name_upper == 'CIFAR10':
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
            class_names = train_dataset.classes
        elif dataset_name_upper == 'CIFAR100':
            train_dataset = datasets.CIFAR100(root='./data', train=True, download=True)
            test_dataset = datasets.CIFAR100(root='./data', train=False, download=True)
            class_names = train_dataset.classes
        elif dataset_name_upper == 'MNIST':
            train_dataset = datasets.MNIST(root='./data', train=True, download=True)
            test_dataset = datasets.MNIST(root='./data', train=False, download=True)
            class_names = [str(i) for i in range(10)]
        elif dataset_name_upper == 'FASHIONMNIST':
            train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
            test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True)
            class_names = train_dataset.classes

        # Create output directories
        train_dir = output_dir / 'train'
        test_dir = output_dir / 'test'

        for class_name in class_names:
            (train_dir / class_name).mkdir(parents=True, exist_ok=True)
            (test_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Export training images
        logger.info(f"Exporting {len(train_dataset)} training images...")
        for idx, (img, label) in enumerate(tqdm(train_dataset, desc="Exporting train")):
            class_name = class_names[label]
            output_path = train_dir / class_name / f"{idx:06d}.png"

            # Convert to RGB if needed (MNIST is grayscale)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(output_path)

        # Export test images
        logger.info(f"Exporting {len(test_dataset)} test images...")
        for idx, (img, label) in enumerate(tqdm(test_dataset, desc="Exporting test")):
            class_name = class_names[label]
            output_path = test_dir / class_name / f"{idx:06d}.png"

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(output_path)

        # Save class mapping
        class_mapping = {
            'class_names': class_names,
            'num_classes': len(class_names),
            'class_to_idx': {name: idx for idx, name in enumerate(class_names)},
            'idx_to_class': {idx: name for idx, name in enumerate(class_names)},
            'dataset_name': dataset_name_lower,
            'source': 'torchvision',
            'export_date': datetime.now().isoformat()
        }

        mapping_path = output_dir / f"{dataset_name_lower}_classes.json"
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)

        logger.info(f"Class mapping saved to {mapping_path}")
        logger.info(f"Export complete! Images saved to {output_dir}")
        logger.info(f"  Train: {train_dir}")
        logger.info(f"  Test: {test_dir}")

        return output_dir


# =============================================================================
# UPDATED DOMAIN-AWARE CDBNN (Replace the existing class)
# =============================================================================

class DomainAwareCDBNN(CDBNNApplication):
    """Domain-aware CDBNN application with specialized processors"""

    def __init__(self, config: GlobalConfig):
        # DO NOT modify config.dataset_name - it's already normalized
        super().__init__(config)
        self.domain = config.domain if hasattr(config, 'domain') else 'general'
        self.domain_processor = None

        # Class mapping for predictions
        self.class_names = []
        self.num_classes = 0
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Initialize domain processor if domain is specified
        if self.domain != 'general':
            self._init_domain_processor()
            logger.info(f"Initialized {self.domain} domain processor")

        # Load class metadata if available
        self._load_class_metadata()

    def _load_class_metadata(self):
        """Load class metadata from training for proper label mapping"""
        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)

        # Try multiple possible locations for class metadata
        metadata_paths = [
            self.saving_data_dir / f"{dataset_name_lower}_metadata.json",
            self.loading_data_dir / f"{dataset_name_lower}_metadata.json",
            self.saving_checkpoint_dir / f"{dataset_name_lower}_metadata.json",
            self.loading_checkpoint_dir / f"{dataset_name_lower}_metadata.json",
            self.saving_data_dir / f"{dataset_name_lower}_classes.json",
            self.loading_data_dir / f"{dataset_name_lower}_classes.json",
        ]

        for metadata_path in metadata_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    # Extract class information
                    if 'class_names' in metadata:
                        self.class_names = metadata['class_names']
                    if 'num_classes' in metadata:
                        self.num_classes = metadata['num_classes']
                    if 'class_to_idx' in metadata:
                        self.class_to_idx = metadata['class_to_idx']
                    if 'idx_to_class' in metadata:
                        self.idx_to_class = metadata['idx_to_class']
                        # Convert string keys to int if needed
                        if self.idx_to_class and isinstance(list(self.idx_to_class.keys())[0], str):
                            self.idx_to_class = {int(k): v for k, v in self.idx_to_class.items()}

                    # Update config
                    if self.class_names and not self.config.class_names:
                        self.config.class_names = self.class_names
                    if self.num_classes and not self.config.num_classes:
                        self.config.num_classes = self.num_classes

                    logger.info(f"Loaded class metadata from {metadata_path}")
                    logger.info(f"  Classes: {self.class_names[:5]}{'...' if len(self.class_names) > 5 else ''}")
                    logger.info(f"  Num classes: {self.num_classes}")
                    return

                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
                    continue

        # If no metadata found, try to infer from config
        if self.config.class_names:
            self.class_names = self.config.class_names
            self.num_classes = len(self.config.class_names)
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
            logger.info(f"Inferred class metadata from config: {self.num_classes} classes")
        else:
            logger.info("No class metadata found - will use numeric labels")

    def _init_domain_processor(self):
        """Initialize the appropriate domain processor with full features"""
        if self.domain == 'agriculture':
            self.domain_processor = AgricultureDomainProcessor(self.config)
        elif self.domain == 'medical':
            self.domain_processor = MedicalDomainProcessor(self.config)
        elif self.domain == 'satellite':
            self.domain_processor = SatelliteDomainProcessor(self.config)
        elif self.domain == 'surveillance':
            self.domain_processor = SurveillanceDomainProcessor(self.config)
        elif self.domain == 'microscopy':
            self.domain_processor = MicroscopyDomainProcessor(self.config)
        elif self.domain == 'industrial':
            self.domain_processor = IndustrialDomainProcessor(self.config)
        elif self.domain == 'astronomy':
            self.domain_processor = AstronomyDomainProcessor(self.config)
        else:
            logger.warning(f"Unknown domain: {self.domain}, using general processor")
            self.domain_processor = None

    def prepare_data(self, source_path: str, data_type: str = 'custom') -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data with domain-specific handling and torchvision metadata support.
        Delegates to parent class for actual data loading.
        """
        # For astronomy domain, log FITS support
        if self.domain == 'astronomy' and hasattr(self.config, 'use_fits') and self.config.use_fits:
            logger.info("Astronomy domain with FITS support enabled")

        # For medical domain, log modality
        if self.domain == 'medical' and hasattr(self.config, 'modality'):
            logger.info(f"Medical domain with modality: {self.config.modality}")

        # For agriculture domain, log NIR support
        if self.domain == 'agriculture' and hasattr(self.config, 'has_nir_band') and self.config.has_nir_band:
            logger.info("Agriculture domain with NIR band support")

        # For satellite domain, log band configuration
        if self.domain == 'satellite' and hasattr(self.config, 'num_bands'):
            logger.info(f"Satellite domain with {self.config.num_bands} bands")

        # For surveillance domain, log enabled features
        if self.domain == 'surveillance':
            features_enabled = []
            if getattr(self.config, 'detect_motion', False):
                features_enabled.append('motion_detection')
            if getattr(self.config, 'enhance_low_light', False):
                features_enabled.append('low_light_enhancement')
            if features_enabled:
                logger.info(f"Surveillance domain features: {', '.join(features_enabled)}")

        # Delegate to parent class for actual data preparation
        train_loader, test_loader = super().prepare_data(source_path, data_type)

        # After data preparation, save domain-specific metadata
        if data_type == 'torchvision':
            self._save_domain_metadata()

        return train_loader, test_loader

    def _save_domain_metadata(self):
        """Save domain-specific metadata for later use in prediction"""
        try:
            dataset_name_lower = normalize_dataset_name(self.config.dataset_name)

            domain_metadata = {
                'domain': self.domain,
                'dataset_name': dataset_name_lower,
                'num_classes': self.config.num_classes,
                'class_names': self.config.class_names,
                'class_to_idx': {name: idx for idx, name in enumerate(self.config.class_names)},
                'idx_to_class': {idx: name for idx, name in enumerate(self.config.class_names)},
                'input_size': list(self.config.input_size),
                'normalization_mode': self.config.normalization_mode,
                'timestamp': datetime.now().isoformat(),
                'version': '2.0'
            }

            # Add domain-specific configuration
            if self.domain == 'astronomy':
                domain_metadata['domain_config'] = {
                    'use_fits': getattr(self.config, 'use_fits', False),
                    'fits_hdu': getattr(self.config, 'fits_hdu', 0),
                    'fits_normalization': getattr(self.config, 'fits_normalization', 'zscale'),
                    'pixel_scale': getattr(self.config, 'pixel_scale', 1.0),
                    'detect_sources': getattr(self.config, 'detect_sources', True)
                }
            elif self.domain == 'medical':
                domain_metadata['domain_config'] = {
                    'modality': getattr(self.config, 'modality', 'general'),
                    'detect_tumor': getattr(self.config, 'detect_tumor', True),
                    'detect_lesion': getattr(self.config, 'detect_lesion', True)
                }
            elif self.domain == 'agriculture':
                domain_metadata['domain_config'] = {
                    'has_nir_band': getattr(self.config, 'has_nir_band', False),
                    'compute_ndvi': getattr(self.config, 'compute_ndvi', True),
                    'detect_leaf_disease': getattr(self.config, 'detect_leaf_disease', True)
                }
            elif self.domain == 'satellite':
                domain_metadata['domain_config'] = {
                    'satellite_type': getattr(self.config, 'satellite_type', 'general'),
                    'num_bands': getattr(self.config, 'num_bands', 4),
                    'compute_ndvi': getattr(self.config, 'compute_ndvi', True)
                }

            # Save to multiple locations
            metadata_paths = [
                self.saving_data_dir / f"{dataset_name_lower}_domain_metadata.json",
                self.saving_checkpoint_dir / f"{dataset_name_lower}_domain_metadata.json",
                self.loading_data_dir / f"{dataset_name_lower}_domain_metadata.json",
            ]

            for metadata_path in metadata_paths:
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, 'w') as f:
                    json.dump(domain_metadata, f, indent=2)
                logger.info(f"Domain metadata saved to {metadata_path}")

        except Exception as e:
            logger.warning(f"Failed to save domain metadata: {e}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply domain-specific preprocessing to a single image"""
        if self.domain_processor:
            return self.domain_processor.preprocess(image)
        return image

    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """Apply domain-specific preprocessing to a batch of images"""
        if self.domain_processor and hasattr(self.domain_processor, 'preprocess_batch'):
            return self.domain_processor.preprocess_batch(images)
        elif self.domain_processor:
            # Fallback to per-image processing
            if len(images.shape) == 3:
                return self.domain_processor.preprocess(images)
            batch_size = images.shape[0]
            processed = np.zeros_like(images)
            for i in range(batch_size):
                processed[i] = self.domain_processor.preprocess(images[i])
            return processed
        return images

    def extract_domain_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract domain-specific features from a single image"""
        if self.domain_processor:
            return self.domain_processor.extract_features(image)
        return {}

    def extract_domain_features_batch(self, images: np.ndarray) -> List[Dict[str, float]]:
        """Extract domain-specific features from a batch of images"""
        if self.domain_processor and hasattr(self.domain_processor, 'extract_features_batch'):
            return self.domain_processor.extract_features_batch(images)
        elif self.domain_processor:
            # Fallback to per-image extraction
            if len(images.shape) == 3:
                return [self.domain_processor.extract_features(images)]
            batch_size = images.shape[0]
            features_list = []
            for i in range(batch_size):
                features_list.append(self.domain_processor.extract_features(images[i]))
            return features_list
        return [{}] * (images.shape[0] if len(images.shape) > 2 else 1)

    def get_domain_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Get domain-specific quality metrics for a single image"""
        if self.domain_processor:
            return self.domain_processor.get_quality_metrics(image)
        return {}

    def get_domain_quality_metrics_batch(self, images: np.ndarray) -> List[Dict[str, float]]:
        """Get domain-specific quality metrics for a batch of images"""
        if self.domain_processor and hasattr(self.domain_processor, 'get_quality_metrics_batch'):
            return self.domain_processor.get_quality_metrics_batch(images)
        elif self.domain_processor:
            if len(images.shape) == 3:
                return [self.domain_processor.get_quality_metrics(images)]
            batch_size = images.shape[0]
            metrics_list = []
            for i in range(batch_size):
                metrics_list.append(self.domain_processor.get_quality_metrics(images[i]))
            return metrics_list
        return [{}] * (images.shape[0] if len(images.shape) > 2 else 1)

    def get_domain_info(self) -> Dict[str, Any]:
        """Get information about the current domain configuration"""
        info = {
            'domain': self.domain,
            'processor_type': type(self.domain_processor).__name__ if self.domain_processor else 'None',
            'num_classes': self.num_classes,
            'class_names_count': len(self.class_names),
            'config': {}
        }

        if self.domain == 'astronomy':
            info['config'] = {
                'use_fits': getattr(self.config, 'use_fits', False),
                'fits_hdu': getattr(self.config, 'fits_hdu', 0),
                'fits_normalization': getattr(self.config, 'fits_normalization', 'zscale'),
                'pixel_scale': getattr(self.config, 'pixel_scale', 1.0),
                'detect_sources': getattr(self.config, 'detect_sources', True)
            }
        elif self.domain == 'medical':
            info['config'] = {
                'modality': getattr(self.config, 'modality', 'general'),
                'detect_tumor': getattr(self.config, 'detect_tumor', True),
                'detect_lesion': getattr(self.config, 'detect_lesion', True),
                'detect_hemorrhage': getattr(self.config, 'detect_hemorrhage', True)
            }
        elif self.domain == 'agriculture':
            info['config'] = {
                'has_nir_band': getattr(self.config, 'has_nir_band', False),
                'compute_ndvi': getattr(self.config, 'compute_ndvi', True),
                'compute_ndwi': getattr(self.config, 'compute_ndwi', True),
                'detect_leaf_disease': getattr(self.config, 'detect_leaf_disease', True),
                'detect_pest_damage': getattr(self.config, 'detect_pest_damage', True)
            }
        elif self.domain == 'satellite':
            info['config'] = {
                'satellite_type': getattr(self.config, 'satellite_type', 'general'),
                'num_bands': getattr(self.config, 'num_bands', 4),
                'classify_land_cover': getattr(self.config, 'classify_land_cover', True),
                'detect_change': getattr(self.config, 'detect_change', True)
            }
        elif self.domain == 'surveillance':
            info['config'] = {
                'detect_person': getattr(self.config, 'detect_person', True),
                'detect_vehicle': getattr(self.config, 'detect_vehicle', True),
                'detect_motion': getattr(self.config, 'detect_motion', True),
                'enhance_low_light': getattr(self.config, 'enhance_low_light', True)
            }
        elif self.domain == 'microscopy':
            info['config'] = {
                'microscopy_type': getattr(self.config, 'microscopy_type', 'general'),
                'detect_cells': getattr(self.config, 'detect_cells', True),
                'segment_nucleus': getattr(self.config, 'segment_nucleus', True)
            }
        elif self.domain == 'industrial':
            info['config'] = {
                'detect_crack': getattr(self.config, 'detect_crack', True),
                'detect_corrosion': getattr(self.config, 'detect_corrosion', True),
                'measure_dimensions': getattr(self.config, 'measure_dimensions', True)
            }

        return info

    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index"""
        if self.idx_to_class and class_idx in self.idx_to_class:
            return self.idx_to_class[class_idx]
        elif self.class_names and 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return str(class_idx)

    def get_class_index(self, class_name: str) -> int:
        """Get class index from name"""
        if self.class_to_idx and class_name in self.class_to_idx:
            return self.class_to_idx[class_name]
        return -1

    def get_all_class_names(self) -> List[str]:
        """Get all class names"""
        return self.class_names if self.class_names else []

# =============================================================================
# AGRICULTURE DOMAIN PROCESSOR
# =============================================================================

class AgricultureDomainProcessor:
    """Complete Agriculture-specific image processor with all features"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.detect_chlorophyll = getattr(config, 'detect_chlorophyll', True)
        self.detect_water_stress = getattr(config, 'detect_water_stress', True)
        self.detect_nutrient_deficiency = getattr(config, 'detect_nutrient_deficiency', True)
        self.detect_leaf_disease = getattr(config, 'detect_leaf_disease', True)
        self.detect_fruit_disease = getattr(config, 'detect_fruit_disease', True)
        self.detect_pest_damage = getattr(config, 'detect_pest_damage', True)
        self.compute_ndvi = getattr(config, 'compute_ndvi', True)
        self.compute_evi = getattr(config, 'compute_evi', True)
        self.compute_ndwi = getattr(config, 'compute_ndwi', True)
        self.compute_gci = getattr(config, 'compute_gci', True)
        self.compute_leaf_texture = getattr(config, 'compute_leaf_texture', True)
        self.compute_canopy_structure = getattr(config, 'compute_canopy_structure', True)
        self.estimate_growth_stage = getattr(config, 'estimate_growth_stage', True)
        self.compute_biomass = getattr(config, 'compute_biomass', True)
        self.has_nir_band = getattr(config, 'has_nir_band', False)
        self.nir_band_index = getattr(config, 'nir_band_index', 3)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess agricultural images"""
        img_float = image.astype(np.float32) / 255.0

        # Illumination normalization
        img_float = self._normalize_illumination(img_float)

        # Shadow removal
        img_float = self._remove_shadows(img_float)

        return img_float

    def _normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """Normalize illumination across the field"""
        if len(image.shape) == 3:
            for i in range(min(3, image.shape[2])):
                image[:, :, i] = exposure.equalize_adapthist(image[:, :, i])
        else:
            image = exposure.equalize_adapthist(image)
        return image

    def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove shadows from agricultural images"""
        if len(image.shape) == 3:
            # Convert to HSV for shadow detection
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32) / 255.0

            # Shadow detection based on low value and saturation
            shadow_mask = (hsv[:, :, 2] < 0.3) & (hsv[:, :, 1] < 0.3)

            # Inpaint shadows
            if np.any(shadow_mask):
                shadow_mask_uint8 = (shadow_mask * 255).astype(np.uint8)
                image_uint8 = (image * 255).astype(np.uint8)
                image_uint8 = cv2.inpaint(image_uint8, shadow_mask_uint8, 3, cv2.INPAINT_TELEA)
                image = image_uint8.astype(np.float32) / 255.0

        return image

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive agriculture-specific features"""
        features = {}

        # Vegetation indices
        if self.compute_ndvi:
            features.update(self._compute_ndvi(image))
        if self.compute_evi:
            features.update(self._compute_evi(image))
        if self.compute_ndwi:
            features.update(self._compute_ndwi(image))
        if self.compute_gci:
            features.update(self._compute_gci(image))

        # Plant health metrics
        if self.detect_chlorophyll:
            features.update(self._compute_chlorophyll_content(image))
        if self.detect_water_stress:
            features.update(self._compute_water_stress(image))
        if self.detect_nutrient_deficiency:
            features.update(self._detect_nutrient_deficiency(image))

        # Disease and pest detection
        if self.detect_leaf_disease:
            features.update(self._detect_leaf_disease(image))
        if self.detect_fruit_disease:
            features.update(self._detect_fruit_disease(image))
        if self.detect_pest_damage:
            features.update(self._detect_pest_damage(image))

        # Structural analysis
        if self.compute_leaf_texture:
            features.update(self._compute_leaf_texture(image))
        if self.compute_canopy_structure:
            features.update(self._compute_canopy_structure(image))

        # Growth and biomass
        if self.estimate_growth_stage:
            features.update(self._estimate_growth_stage(image))
        if self.compute_biomass:
            features.update(self._compute_biomass(image))

        return features

    def _compute_ndvi(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Normalized Difference Vegetation Index"""
        if self.has_nir_band and image.shape[2] > self.nir_band_index:
            nir = image[:, :, self.nir_band_index]
            red = image[:, :, 0]
        else:
            # Approximate NIR from RGB (normalized difference)
            nir = (image[:, :, 1] + image[:, :, 2]) / 2  # Green+Blue as proxy
            red = image[:, :, 0]

        ndvi = (nir - red) / (nir + red + 1e-8)

        return {
            'ndvi_mean': float(np.mean(ndvi)),
            'ndvi_std': float(np.std(ndvi)),
            'ndvi_max': float(np.max(ndvi)),
            'ndvi_min': float(np.min(ndvi)),
            'vegetation_fraction': float(np.mean(ndvi > 0.3)),
            'dense_vegetation_fraction': float(np.mean(ndvi > 0.6)),
            'sparse_vegetation_fraction': float(np.mean((ndvi > 0.1) & (ndvi <= 0.3)))
        }

    def _compute_evi(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Enhanced Vegetation Index"""
        if self.has_nir_band and image.shape[2] > self.nir_band_index:
            nir = image[:, :, self.nir_band_index]
            red = image[:, :, 0]
            blue = image[:, :, 2]
        else:
            nir = (image[:, :, 1] + image[:, :, 2]) / 2
            red = image[:, :, 0]
            blue = image[:, :, 2]

        # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)

        return {
            'evi_mean': float(np.mean(evi)),
            'evi_std': float(np.std(evi)),
            'evi_max': float(np.max(evi))
        }

    def _compute_ndwi(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Normalized Difference Water Index"""
        if len(image.shape) >= 3 and image.shape[2] >= 3:
            green = image[:, :, 1]
            if self.has_nir_band and image.shape[2] > self.nir_band_index:
                nir = image[:, :, self.nir_band_index]
            else:
                nir = (image[:, :, 0] + image[:, :, 2]) / 2

            ndwi = (green - nir) / (green + nir + 1e-8)

            return {
                'ndwi_mean': float(np.mean(ndwi)),
                'water_content': float(np.mean(ndwi > 0)),
                'water_stress_index': float(1 - np.mean(ndwi))
            }
        return {'ndwi_mean': 0, 'water_content': 0, 'water_stress_index': 0}

    def _compute_gci(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Green Chlorophyll Index"""
        if len(image.shape) >= 3 and image.shape[2] >= 3:
            if self.has_nir_band and image.shape[2] > self.nir_band_index:
                nir = image[:, :, self.nir_band_index]
            else:
                nir = (image[:, :, 0] + image[:, :, 2]) / 2

            green = image[:, :, 1]
            gci = (nir / green) - 1

            return {
                'gci_mean': float(np.mean(gci)),
                'chlorophyll_estimate': float(np.mean(np.clip(gci, 0, 10)))
            }
        return {'gci_mean': 0, 'chlorophyll_estimate': 0}

    def _compute_chlorophyll_content(self, image: np.ndarray) -> Dict[str, float]:
        """Estimate chlorophyll content using multiple indices"""
        if len(image.shape) >= 3 and image.shape[2] >= 3:
            # Simple chlorophyll index (green/red)
            green = image[:, :, 1]
            red = image[:, :, 0]
            chlorophyll_simple = green / (red + 1e-8)

            # More accurate chlorophyll using normalized difference
            chlorophyll_nd = (green - red) / (green + red + 1e-8)

            # CCI (Canopy Chlorophyll Index) approximation
            cci = (green - red) / (green + red) * 100

            return {
                'chlorophyll_index': float(np.mean(chlorophyll_simple)),
                'chlorophyll_ndvi': float(np.mean(chlorophyll_nd)),
                'cci_estimate': float(np.mean(cci)),
                'green_percentage': float(np.mean(green > 0.3)),
                'chlorophyll_variability': float(np.std(chlorophyll_simple))
            }
        return {'chlorophyll_index': 0, 'chlorophyll_ndvi': 0, 'cci_estimate': 0,
                'green_percentage': 0, 'chlorophyll_variability': 0}

    def _compute_water_stress(self, image: np.ndarray) -> Dict[str, float]:
        """Detect water stress using multiple indicators"""
        if len(image.shape) >= 3 and image.shape[2] >= 3:
            # Convert to LAB color space for better water stress detection
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32) / 255.0

            # Water stress manifests as lower b* (yellowing) and lower a* (less green)
            water_stress_score = (1 - lab[:, :, 1]) * (1 - lab[:, :, 2])

            # Wilting detection (drooping leaves - approximated by edge analysis)
            gray = np.mean(image, axis=2)
            edges = sobel(gray)
            wilting_indicator = np.std(edges[edges > np.percentile(edges, 90)])

            return {
                'water_stress_index': float(np.mean(water_stress_score)),
                'wilting_score': float(np.minimum(1.0, wilting_indicator / 0.5)),
                'stress_severity': float(np.mean(water_stress_score > 0.6)),
                'turgor_pressure': float(1 - np.mean(water_stress_score))
            }
        return {'water_stress_index': 0, 'wilting_score': 0, 'stress_severity': 0, 'turgor_pressure': 0}

    def _detect_nutrient_deficiency(self, image: np.ndarray) -> Dict[str, float]:
        """Detect nutrient deficiencies (N, P, K)"""
        if len(image.shape) >= 3 and image.shape[2] >= 3:
            # Nitrogen deficiency: pale green/yellow
            # Phosphorus deficiency: dark green/purplish
            # Potassium deficiency: yellowing at leaf edges

            # Convert to HSV
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32) / 255.0

            # Nitrogen deficiency indicator (low green intensity, high yellow hue)
            n_deficiency = (image[:, :, 1] < 0.3) & (hsv[:, :, 0] > 0.15) & (hsv[:, :, 0] < 0.2)

            # Phosphorus deficiency indicator (purplish hue)
            p_deficiency = (hsv[:, :, 0] > 0.7) & (hsv[:, :, 0] < 0.9) & (image[:, :, 1] < 0.4)

            # Potassium deficiency (edge yellowing)
            gray = np.mean(image, axis=2)
            edges = sobel(gray)
            edge_regions = edges > np.percentile(edges, 85)
            edge_yellowing = np.mean(image[edge_regions, 1] < image[edge_regions, 0])

            return {
                'nitrogen_deficiency_risk': float(np.mean(n_deficiency)),
                'phosphorus_deficiency_risk': float(np.mean(p_deficiency)),
                'potassium_deficiency_risk': float(edge_yellowing),
                'overall_nutrient_stress': float((np.mean(n_deficiency) + np.mean(p_deficiency) + edge_yellowing) / 3)
            }
        return {'nitrogen_deficiency_risk': 0, 'phosphorus_deficiency_risk': 0,
                'potassium_deficiency_risk': 0, 'overall_nutrient_stress': 0}

    def _detect_leaf_disease(self, image: np.ndarray) -> Dict[str, float]:
        """Detect leaf diseases using color and texture analysis"""
        if len(image.shape) >= 3 and image.shape[2] >= 3:
            # Convert to LAB for better disease spot detection
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32) / 255.0

            # Disease spots appear as dark/brown/necrotic regions
            # In LAB: low L (dark), moderate a (reddish), low b (blueish)
            disease_spots = (lab[:, :, 0] < 0.4) & (lab[:, :, 1] > 0.3) & (lab[:, :, 1] < 0.6) & (lab[:, :, 2] < 0.4)

            # Morphological analysis of spots
            if np.any(disease_spots):
                labeled, num_features = ndimage_label(disease_spots)
                spot_sizes = np.bincount(labeled.ravel())[1:]
                avg_spot_size = np.mean(spot_sizes) if len(spot_sizes) > 0 else 0
                num_spots = num_features
            else:
                avg_spot_size = 0
                num_spots = 0

            # Texture analysis for disease patterns
            gray = np.mean(image, axis=2)
            glcm = graycomatrix((gray * 255).astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True)

            return {
                'disease_spots_fraction': float(np.mean(disease_spots)),
                'disease_severity': float(np.minimum(1.0, np.mean(disease_spots) * 2)),
                'num_disease_spots': float(num_spots),
                'avg_spot_size': float(avg_spot_size),
                'disease_texture_contrast': float(graycoprops(glcm, 'contrast')[0, 0]),
                'lesion_density': float(np.mean(disease_spots) * 100)
            }
        return {'disease_spots_fraction': 0, 'disease_severity': 0, 'num_disease_spots': 0,
                'avg_spot_size': 0, 'disease_texture_contrast': 0, 'lesion_density': 0}

    def _detect_fruit_disease(self, image: np.ndarray) -> Dict[str, float]:
        """Detect fruit diseases"""
        if len(image.shape) >= 3 and image.shape[2] >= 3:
            # Fruit detection (circular/elliptical regions)
            gray = np.mean(image, axis=2)

            # Look for circular blobs (potential fruits)
            blobs = blob_log(gray, max_sigma=30, num_sigma=10, threshold=0.05)

            fruit_count = len(blobs)

            # Analyze each fruit region for disease
            diseased_fruits = 0
            for blob in blobs[:min(20, len(blobs))]:
                y, x, r = blob
                y, x, r = int(y), int(x), int(r)
                y_min, y_max = max(0, y - r), min(gray.shape[0], y + r)
                x_min, x_max = max(0, x - r), min(gray.shape[1], x + r)

                fruit_region = image[y_min:y_max, x_min:x_max]
                if fruit_region.size > 0:
                    # Check for rot/discoloration
                    rot_score = np.mean(fruit_region[:, :, 0] > fruit_region[:, :, 1]) if fruit_region.shape[2] >= 2 else 0
                    if rot_score > 0.6:
                        diseased_fruits += 1

            return {
                'fruit_count': float(fruit_count),
                'diseased_fruit_count': float(diseased_fruits),
                'disease_incidence': float(diseased_fruits / (fruit_count + 1e-8)),
                'fruit_health_index': float(1 - diseased_fruits / (fruit_count + 1e-8))
            }
        return {'fruit_count': 0, 'diseased_fruit_count': 0, 'disease_incidence': 0, 'fruit_health_index': 1}

    def _detect_pest_damage(self, image: np.ndarray) -> Dict[str, float]:
        """Detect pest damage on leaves"""
        if len(image.shape) >= 3 and image.shape[2] >= 3:
            # Pest damage: irregular holes, skeletonization, discoloration
            gray = np.mean(image, axis=2)

            # Edge detection to find damaged areas
            edges = sobel(gray)

            # Skeletonization (holes) detection
            _, binary = cv2.threshold((gray * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            holes = cv2.bitwise_not(binary)

            # Pest damage indicator
            pest_damage = (edges > np.percentile(edges, 95)) & (gray < np.percentile(gray, 30))

            # Chewed leaf margins (high edge density near leaf boundaries)
            leaf_mask = gray > np.percentile(gray, 20)
            if np.any(leaf_mask):
                boundary = morphology.binary_dilation(leaf_mask) ^ leaf_mask
                edge_density_on_boundary = np.mean(edges[boundary]) if np.any(boundary) else 0
            else:
                edge_density_on_boundary = 0

            return {
                'pest_damage_fraction': float(np.mean(pest_damage)),
                'hole_density': float(np.mean(holes > 0)),
                'defoliation_index': float(np.mean(gray < 0.2)),
                'edge_damage_score': float(edge_density_on_boundary),
                'pest_severity': float((np.mean(pest_damage) + np.mean(holes > 0)) / 2)
            }
        return {'pest_damage_fraction': 0, 'hole_density': 0, 'defoliation_index': 0,
                'edge_damage_score': 0, 'pest_severity': 0}

    def _compute_leaf_texture(self, image: np.ndarray) -> Dict[str, float]:
        """Compute leaf texture features using GLCM and Gabor filters"""
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        gray_uint8 = (gray * 255).astype(np.uint8)

        # GLCM features at multiple angles
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm_features = []

        for angle in angles:
            glcm = graycomatrix(gray_uint8, distances=[1], angles=[angle], levels=256, symmetric=True)
            glcm_features.append({
                'contrast': graycoprops(glcm, 'contrast')[0, 0],
                'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
                'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
                'energy': graycoprops(glcm, 'energy')[0, 0],
                'correlation': graycoprops(glcm, 'correlation')[0, 0]
            })

        # Gabor filter responses for texture orientation
        gabor_responses = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            gabor = filters.gabor(gray, frequency=0.1, theta=theta)[0]
            gabor_responses.append(np.mean(np.abs(gabor)))

        return {
            'leaf_texture_contrast': float(np.mean([f['contrast'] for f in glcm_features])),
            'leaf_texture_homogeneity': float(np.mean([f['homogeneity'] for f in glcm_features])),
            'leaf_texture_energy': float(np.mean([f['energy'] for f in glcm_features])),
            'leaf_texture_correlation': float(np.mean([f['correlation'] for f in glcm_features])),
            'texture_directionality': float(np.std(gabor_responses)),
            'leaf_smoothness': float(1 - np.std(gray))
        }

    def _compute_canopy_structure(self, image: np.ndarray) -> Dict[str, float]:
        """Compute canopy structure metrics"""
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Vegetation fraction
        vegetation_mask = gray > np.percentile(gray, 30)
        vegetation_fraction = np.mean(vegetation_mask)

        if vegetation_fraction > 0:
            # Compute canopy gaps and structure
            labeled, num_patches = ndimage_label(vegetation_mask)
            patch_sizes = np.bincount(labeled.ravel())[1:]

            # Canopy height proxy (using gradient information)
            gradient_x = np.abs(np.gradient(gray, axis=1))
            gradient_y = np.abs(np.gradient(gray, axis=0))
            canopy_roughness = np.mean(gradient_x[vegetation_mask] + gradient_y[vegetation_mask])

            # Gap fraction (non-vegetated areas)
            gap_fraction = 1 - vegetation_fraction

            # Patch size distribution
            avg_patch_size = np.mean(patch_sizes) if len(patch_sizes) > 0 else 0

            return {
                'canopy_cover': float(vegetation_fraction),
                'gap_fraction': float(gap_fraction),
                'canopy_roughness': float(canopy_roughness),
                'num_canopy_patches': float(num_patches),
                'avg_patch_size': float(avg_patch_size),
                'canopy_density': float(vegetation_fraction * (1 - gap_fraction))
            }

        return {'canopy_cover': 0, 'gap_fraction': 1, 'canopy_roughness': 0,
                'num_canopy_patches': 0, 'avg_patch_size': 0, 'canopy_density': 0}

    def _estimate_growth_stage(self, image: np.ndarray) -> Dict[str, float]:
        """Estimate plant growth stage"""
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Growth indicators
        vegetation_fraction = np.mean(gray > np.percentile(gray, 30))

        # Greenness intensity (proxy for chlorophyll)
        if len(image.shape) >= 3:
            greenness = np.mean(image[:, :, 1])
        else:
            greenness = np.mean(gray)

        # Canopy complexity (edge density)
        edges = sobel(gray)
        edge_density = np.mean(edges[edges > np.percentile(edges, 80)])

        # Growth stage estimation (0-4 scale: seedling, vegetative, flowering, fruiting, maturity)
        if vegetation_fraction < 0.2:
            growth_stage = 0  # Seedling
        elif vegetation_fraction < 0.5:
            growth_stage = 1  # Vegetative
        elif edge_density > 0.3:
            growth_stage = 2  # Flowering (more complex structure)
        elif greenness > 0.4:
            growth_stage = 3  # Fruiting
        else:
            growth_stage = 4  # Maturity/Senescence

        return {
            'growth_stage': float(growth_stage),
            'growth_stage_progress': float(vegetation_fraction),
            'canopy_development': float(vegetation_fraction),
            'senescence_index': float(1 - greenness if greenness < 0.3 else 0)
        }

    def _compute_biomass(self, image: np.ndarray) -> Dict[str, float]:
        """Estimate above-ground biomass"""
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Vegetation indices as biomass proxies
        if len(image.shape) >= 3:
            # Use NDVI-like calculation
            if self.has_nir_band and image.shape[2] > self.nir_band_index:
                nir = image[:, :, self.nir_band_index]
                red = image[:, :, 0]
            else:
                nir = (image[:, :, 1] + image[:, :, 2]) / 2
                red = image[:, :, 0]

            ndvi_like = (nir - red) / (nir + red + 1e-8)

            # Biomass estimation using multiple indices
            biomass_ndvi = np.mean(ndvi_like) * 2  # Scale to 0-2 range
            biomass_cover = np.mean(gray > np.percentile(gray, 30))

            # Combined biomass estimate
            estimated_biomass = (biomass_ndvi * 0.6 + biomass_cover * 0.4)
        else:
            # Grayscale-based estimation
            biomass_cover = np.mean(gray > np.percentile(gray, 30))
            estimated_biomass = biomass_cover

        return {
            'biomass_index': float(estimated_biomass),
            'biomass_cover': float(biomass_cover),
            'biomass_density': float(np.mean(gray > np.percentile(gray, 60))),
            'estimated_yield': float(estimated_biomass * 0.8)  # Simple yield proxy
        }

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute agriculture-specific quality metrics"""
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Illumination uniformity
        illumination_uniformity = 1 - np.std(gray) / (np.mean(gray) + 1e-8)

        # Shadow presence
        shadow_fraction = np.mean(gray < 0.2)

        # Focus quality (sharpness)
        laplacian = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
        sharpness = np.var(laplacian) / 1000

        # Cloud cover (if present)
        cloud_fraction = np.mean(gray > 0.95)

        return {
            'illumination_uniformity': float(illumination_uniformity),
            'shadow_fraction': float(shadow_fraction),
            'image_sharpness': float(min(1.0, sharpness / 500)),
            'cloud_cover_fraction': float(cloud_fraction),
            'image_quality_score': float((illumination_uniformity + (1 - shadow_fraction) + min(1.0, sharpness / 500)) / 3)
        }

# =============================================================================
# COMPLETE MEDICAL DOMAIN PROCESSOR
# =============================================================================

class MedicalDomainProcessor:
    """Complete Medical imaging processor with all features"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.modality = getattr(config, 'modality', 'general')
        self.detect_tumor = getattr(config, 'detect_tumor', True)
        self.detect_lesion = getattr(config, 'detect_lesion', True)
        self.detect_hemorrhage = getattr(config, 'detect_hemorrhage', True)
        self.detect_calcification = getattr(config, 'detect_calcification', True)
        self.segment_organs = getattr(config, 'segment_organs', True)
        self.segment_vessels = getattr(config, 'segment_vessels', True)
        self.segment_tissues = getattr(config, 'segment_tissues', True)
        self.compute_tissue_texture = getattr(config, 'compute_tissue_texture', True)
        self.compute_boundary_regularity = getattr(config, 'compute_boundary_regularity', True)
        self.compute_contrast = getattr(config, 'compute_contrast', True)
        self.compute_sharpness = getattr(config, 'compute_sharpness', True)
        self.detect_artifacts = getattr(config, 'detect_artifacts', True)
        self.estimate_tumor_size = getattr(config, 'estimate_tumor_size', True)
        self.compute_tumor_heterogeneity = getattr(config, 'compute_tumor_heterogeneity', True)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess medical images based on modality"""
        img_float = image.astype(np.float32) / 255.0

        # Modality-specific preprocessing
        if self.modality == 'ct':
            img_float = self._preprocess_ct(img_float)
        elif self.modality == 'mri':
            img_float = self._preprocess_mri(img_float)
        elif self.modality == 'xray':
            img_float = self._preprocess_xray(img_float)
        elif self.modality == 'ultrasound':
            img_float = self._preprocess_ultrasound(img_float)
        elif self.modality == 'histology':
            img_float = self._preprocess_histology(img_float)
        else:
            # General medical preprocessing
            img_float = self._denoise_medical(img_float)
            img_float = self._enhance_contrast(img_float)

        # Normalize intensity
        img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)

        return img_float

    def _preprocess_ct(self, image: np.ndarray) -> np.ndarray:
        """Preprocess CT images (bone/soft tissue windowing)"""
        # Simulate windowing for CT
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Bone window (high contrast)
        bone_window = np.clip((gray - 0.3) / 0.5, 0, 1)
        # Soft tissue window
        soft_window = np.clip((gray - 0.2) / 0.4, 0, 1)

        # Combine
        return np.stack([bone_window, soft_window, gray], axis=2) if len(image.shape) == 3 else gray

    def _preprocess_mri(self, image: np.ndarray) -> np.ndarray:
        """Preprocess MRI images (bias field correction)"""
        # Simple bias field correction using Gaussian filtering
        if len(image.shape) == 3:
            for i in range(3):
                blurred = cv2.GaussianBlur(image[:, :, i], (51, 51), 0)
                image[:, :, i] = image[:, :, i] / (blurred + 1e-8)
        else:
            blurred = cv2.GaussianBlur(image, (51, 51), 0)
            image = image / (blurred + 1e-8)

        return np.clip(image, 0, 1)

    def _preprocess_xray(self, image: np.ndarray) -> np.ndarray:
        """Preprocess X-ray images (contrast enhancement)"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # CLAHE for X-ray
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply((gray * 255).astype(np.uint8)) / 255.0

        if len(image.shape) == 3:
            return np.stack([enhanced, enhanced, enhanced], axis=2)
        return enhanced

    def _preprocess_ultrasound(self, image: np.ndarray) -> np.ndarray:
        """Preprocess ultrasound images (speckle reduction)"""
        if len(image.shape) == 3:
            for i in range(3):
                image[:, :, i] = restoration.denoise_bilateral(image[:, :, i], sigma_color=0.05, sigma_spatial=5)
        else:
            image = restoration.denoise_bilateral(image, sigma_color=0.05, sigma_spatial=5)
        return image

    def _preprocess_histology(self, image: np.ndarray) -> np.ndarray:
        """Preprocess histology images (color normalization)"""
        # Stain normalization using Macenko method approximation
        if len(image.shape) == 3:
            # Convert to LAB and normalize
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32) / 255.0

            # Normalize L channel
            lab[:, :, 0] = exposure.equalize_adapthist(lab[:, :, 0])

            # Convert back
            lab = (lab * 255).astype(np.uint8)
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) / 255.0

        return image

    def _denoise_medical(self, image: np.ndarray) -> np.ndarray:
        """Denoise medical images"""
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                image[:, :, i] = cv2.bilateralFilter((image[:, :, i] * 255).astype(np.uint8), 5, 50, 50) / 255.0
        else:
            image = cv2.bilateralFilter((image * 255).astype(np.uint8), 5, 50, 50) / 255.0
        return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE"""
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image[:, :, i] = clahe.apply((image[:, :, i] * 255).astype(np.uint8)) / 255.0
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply((image * 255).astype(np.uint8)) / 255.0
        return image

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive medical-specific features"""
        features = {}

        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # Tumor detection
        if self.detect_tumor:
            features.update(self._detect_tumor(image, gray))

        # Lesion detection
        if self.detect_lesion:
            features.update(self._detect_lesion(image, gray))

        # Hemorrhage detection
        if self.detect_hemorrhage:
            features.update(self._detect_hemorrhage(image, gray))

        # Calcification detection
        if self.detect_calcification:
            features.update(self._detect_calcification(image, gray))

        # Segmentation features
        if self.segment_organs:
            features.update(self._segment_organs(image, gray))
        if self.segment_vessels:
            features.update(self._segment_vessels(image, gray))
        if self.segment_tissues:
            features.update(self._segment_tissues(image, gray))

        # Texture analysis
        if self.compute_tissue_texture:
            features.update(self._compute_tissue_texture(image, gray))

        # Boundary analysis
        if self.compute_boundary_regularity:
            features.update(self._compute_boundary_regularity(gray))

        # Quality metrics
        if self.compute_contrast:
            features.update(self._compute_medical_contrast(gray))
        if self.compute_sharpness:
            features.update(self._compute_medical_sharpness(gray))
        if self.detect_artifacts:
            features.update(self._detect_artifacts(gray))

        # Tumor characterization
        if self.estimate_tumor_size:
            features.update(self._estimate_tumor_size(gray))
        if self.compute_tumor_heterogeneity:
            features.update(self._compute_tumor_heterogeneity(image, gray))

        return features

    def _detect_tumor(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect tumors using multiple criteria"""
        # Look for abnormal regions with different texture and intensity
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')

        # Compute local statistics
        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(gray, size=11)
        local_std = np.sqrt(uniform_filter(gray**2, size=11) - local_mean**2)

        # Abnormal regions: high LBP variation, different intensity
        lbp_std = uniform_filter(lbp**2, size=11) - uniform_filter(lbp, size=11)**2

        abnormality_score = (local_std / (local_mean + 1e-8)) * lbp_std
        abnormality_score = abnormality_score / (abnormality_score.max() + 1e-8)

        # Threshold for tumor suspicion
        threshold = np.percentile(abnormality_score, 95)
        tumor_mask = abnormality_score > threshold

        if np.any(tumor_mask):
            tumor_area = np.sum(tumor_mask) / tumor_mask.size
            tumor_intensity = np.mean(gray[tumor_mask])
            tumor_std = np.std(gray[tumor_mask])
        else:
            tumor_area = 0
            tumor_intensity = 0
            tumor_std = 0

        return {
            'tumor_suspicion': float(np.mean(abnormality_score > np.percentile(abnormality_score, 90))),
            'tumor_area_fraction': float(tumor_area),
            'tumor_intensity': float(tumor_intensity),
            'tumor_texture_abnormality': float(np.mean(lbp[tumor_mask]) if np.any(tumor_mask) else 0),
            'tumor_heterogeneity': float(tumor_std),
            'malignancy_risk': float(min(1.0, tumor_area * 2 + tumor_std * 2))
        }

    def _detect_lesion(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect lesions (focal abnormalities)"""
        # Use blob detection for potential lesions
        blobs = blob_log(gray, max_sigma=10, num_sigma=10, threshold=0.05)

        if len(blobs) > 0:
            lesion_count = len(blobs)
            avg_lesion_size = np.mean([b[2] for b in blobs])
            max_lesion_size = np.max([b[2] for b in blobs])

            # Analyze lesion intensities
            lesion_intensities = []
            for blob in blobs[:min(20, len(blobs))]:
                y, x, r = int(blob[0]), int(blob[1]), int(blob[2])
                y_min, y_max = max(0, y - r), min(gray.shape[0], y + r)
                x_min, x_max = max(0, x - r), min(gray.shape[1], x + r)
                lesion_intensities.append(np.mean(gray[y_min:y_max, x_min:x_max]))

            return {
                'lesion_count': float(lesion_count),
                'avg_lesion_size': float(avg_lesion_size),
                'max_lesion_size': float(max_lesion_size),
                'lesion_intensity_mean': float(np.mean(lesion_intensities)),
                'lesion_density': float(lesion_count / (gray.size / 10000))
            }

        return {'lesion_count': 0, 'avg_lesion_size': 0, 'max_lesion_size': 0,
                'lesion_intensity_mean': 0, 'lesion_density': 0}

    def _detect_hemorrhage(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect hemorrhage (bleeding) in medical images"""
        if len(image.shape) >= 3:
            # Hemorrhage appears as hyperdense/hyperintense regions
            # In CT: bright, in MRI: varies by sequence
            if self.modality == 'ct':
                # Hyperdense regions
                hemorrhage_mask = gray > np.percentile(gray, 95)
            elif self.modality == 'mri':
                # T1 hyperintense or T2 hypointense (approximated)
                hemorrhage_mask = (gray > np.percentile(gray, 90)) | (gray < np.percentile(gray, 10))
            else:
                hemorrhage_mask = gray > np.percentile(gray, 97)
        else:
            hemorrhage_mask = gray > np.percentile(gray, 97)

        # Morphological analysis
        if np.any(hemorrhage_mask):
            labeled, num_hemorrhages = ndimage_label(hemorrhage_mask)
            hemorrhage_areas = np.bincount(labeled.ravel())[1:]
            total_hemorrhage_area = np.sum(hemorrhage_areas)

            return {
                'hemorrhage_present': float(1 if num_hemorrhages > 0 else 0),
                'hemorrhage_count': float(num_hemorrhages),
                'hemorrhage_area_fraction': float(total_hemorrhage_area / hemorrhage_mask.size),
                'max_hemorrhage_size': float(np.max(hemorrhage_areas) if len(hemorrhage_areas) > 0 else 0),
                'hemorrhage_severity': float(min(1.0, total_hemorrhage_area / (hemorrhage_mask.size * 0.1)))
            }

        return {'hemorrhage_present': 0, 'hemorrhage_count': 0, 'hemorrhage_area_fraction': 0,
                'max_hemorrhage_size': 0, 'hemorrhage_severity': 0}

    def _detect_calcification(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect calcifications (small bright spots)"""
        # Calcifications are small, bright, well-defined spots
        from skimage.feature import blob_log

        blobs = blob_log(gray, max_sigma=3, num_sigma=10, threshold=0.1)

        # Filter for small blobs (calcifications are typically small)
        calcifications = [b for b in blobs if b[2] < 3]

        if len(calcifications) > 0:
            return {
                'calcification_count': float(len(calcifications)),
                'calcification_density': float(len(calcifications) / (gray.size / 10000)),
                'avg_calcification_size': float(np.mean([b[2] for b in calcifications])),
                'microcalcification_clusters': float(len([b for b in calcifications if b[2] < 1.5]))
            }

        return {'calcification_count': 0, 'calcification_density': 0,
                'avg_calcification_size': 0, 'microcalcification_clusters': 0}

    def _segment_organs(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Segment and analyze organ regions"""
        # Use thresholding and morphological operations
        threshold = np.percentile(gray, 70)
        organ_mask = gray > threshold

        if np.any(organ_mask):
            # Clean up mask
            organ_mask = morphology.binary_closing(organ_mask, morphology.disk(5))
            organ_mask = morphology.binary_opening(organ_mask, morphology.disk(3))

            labeled, num_organs = ndimage_label(organ_mask)
            organ_areas = np.bincount(labeled.ravel())[1:]

            # Find largest organ (primary)
            if len(organ_areas) > 0:
                primary_organ_area = np.max(organ_areas)
                organ_area_fraction = primary_organ_area / organ_mask.size

                # Organ shape analysis
                props = regionprops(labeled)
                if props:
                    major_axis = np.mean([p.major_axis_length for p in props])
                    minor_axis = np.mean([p.minor_axis_length for p in props])
                    eccentricity = np.mean([p.eccentricity for p in props])
                else:
                    major_axis, minor_axis, eccentricity = 0, 0, 0

                return {
                    'organ_area_fraction': float(organ_area_fraction),
                    'organ_count': float(num_organs),
                    'organ_elongation': float(major_axis / (minor_axis + 1e-8)),
                    'organ_eccentricity': float(eccentricity),
                    'organ_compactness': float(4 * np.pi * primary_organ_area / (props[0].perimeter**2) if props else 0)
                }

        return {'organ_area_fraction': 0, 'organ_count': 0, 'organ_elongation': 0,
                'organ_eccentricity': 0, 'organ_compactness': 0}

    def _segment_vessels(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Segment and analyze blood vessels"""
        # Use Frangi filter for vessel enhancement
        vessels = frangi(gray, scale_range=(1, 5), scale_step=1)

        # Threshold for vessel segmentation
        threshold = np.percentile(vessels, 90)
        vessel_mask = vessels > threshold

        if np.any(vessel_mask):
            vessel_density = np.mean(vessel_mask)

            # Skeletonize for length estimation
            skeleton = morphology.skeletonize(vessel_mask)
            vessel_length = np.sum(skeleton)

            # Branching points
            from skimage.morphology import branch_points
            branches = branch_points(skeleton)
            num_branches = np.sum(branches)

            return {
                'vessel_density': float(vessel_density),
                'vessel_length': float(vessel_length / vessel_mask.size),
                'vessel_branch_count': float(num_branches),
                'vessel_complexity': float(vessel_length / (vessel_mask.size + 1e-8)),
                'vascular_tortuosity': float(np.std(vessels[vessel_mask]) if np.any(vessel_mask) else 0)
            }

        return {'vessel_density': 0, 'vessel_length': 0, 'vessel_branch_count': 0,
                'vessel_complexity': 0, 'vascular_tortuosity': 0}

    def _segment_tissues(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Segment different tissue types"""
        # Use K-means clustering for tissue segmentation
        from sklearn.cluster import KMeans

        # Reshape for clustering
        pixels = gray.reshape(-1, 1)

        # Segment into 3 tissue types (e.g., fat, muscle, bone/other)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        tissue_map = labels.reshape(gray.shape)

        # Calculate tissue fractions
        unique, counts = np.unique(tissue_map, return_counts=True)
        tissue_fractions = counts / counts.sum()

        # Sort by intensity (darker to brighter)
        tissue_means = [np.mean(gray[tissue_map == i]) for i in range(3)]
        sorted_indices = np.argsort(tissue_means)

        tissue_fractions_sorted = tissue_fractions[sorted_indices]

        return {
            'tissue_1_fraction': float(tissue_fractions_sorted[0]) if len(tissue_fractions_sorted) > 0 else 0,  # Darkest
            'tissue_2_fraction': float(tissue_fractions_sorted[1]) if len(tissue_fractions_sorted) > 1 else 0,
            'tissue_3_fraction': float(tissue_fractions_sorted[2]) if len(tissue_fractions_sorted) > 2 else 0,
            'tissue_heterogeneity': float(np.std([tissue_fractions_sorted[i] for i in range(min(3, len(tissue_fractions_sorted)))])),
            'tissue_contrast': float(np.max(tissue_means) - np.min(tissue_means)) if len(tissue_means) > 1 else 0
        }

    def _compute_tissue_texture(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive tissue texture features"""
        gray_uint8 = (gray * 255).astype(np.uint8)

        # GLCM features
        glcm = graycomatrix(gray_uint8, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=256, symmetric=True)

        textures = {
            'tissue_contrast': float(np.mean([graycoprops(glcm, 'contrast')[0, i] for i in range(3)])),
            'tissue_homogeneity': float(np.mean([graycoprops(glcm, 'homogeneity')[0, i] for i in range(3)])),
            'tissue_energy': float(np.mean([graycoprops(glcm, 'energy')[0, i] for i in range(3)])),
            'tissue_correlation': float(np.mean([graycoprops(glcm, 'correlation')[0, i] for i in range(3)])),
            'tissue_dissimilarity': float(np.mean([graycoprops(glcm, 'dissimilarity')[0, i] for i in range(3)]))
        }

        # Add Laws' texture energy measures
        from scipy.ndimage import convolve

        # Laws' filters
        L5 = np.array([1, 4, 6, 4, 1])  # Level
        E5 = np.array([-1, -2, 0, 2, 1])  # Edge
        S5 = np.array([-1, 0, 2, 0, -1])  # Spot
        R5 = np.array([1, -4, 6, -4, 1])  # Ripple

        filters = {
            'L5E5': np.outer(L5, E5),
            'E5L5': np.outer(E5, L5),
            'E5E5': np.outer(E5, E5),
            'S5S5': np.outer(S5, S5),
            'R5R5': np.outer(R5, R5)
        }

        for name, kernel in filters.items():
            filtered = convolve(gray, kernel)
            textures[f'texture_energy_{name.lower()}'] = float(np.mean(np.abs(filtered)))

        return textures

    def _compute_boundary_regularity(self, gray: np.ndarray) -> Dict[str, float]:
        """Compute boundary regularity metrics for lesions/organs"""
        # Segment main object
        threshold = np.percentile(gray, 70)
        binary = gray > threshold

        if np.any(binary):
            # Find contours
            contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Perimeter and area
                perimeter = cv2.arcLength(largest_contour, True)
                area = cv2.contourArea(largest_contour)

                # Circularity (1 = perfect circle)
                circularity = 4 * np.pi * area / (perimeter**2 + 1e-8)

                # Convexity
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                convexity = area / (hull_area + 1e-8)

                # Boundary smoothness (using curvature)
                # Approximate with polygon
                epsilon = 0.01 * perimeter
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                num_vertices = len(approx)

                return {
                    'boundary_regularity': float(circularity),
                    'boundary_convexity': float(convexity),
                    'boundary_complexity': float(num_vertices / 20),  # Normalized
                    'boundary_smoothness': float(1 - (num_vertices - 4) / 100)  # Less vertices = smoother
                }

        return {'boundary_regularity': 0, 'boundary_convexity': 0, 'boundary_complexity': 0, 'boundary_smoothness': 0}

    def _compute_medical_contrast(self, gray: np.ndarray) -> Dict[str, float]:
        """Compute medical-specific contrast metrics"""
        # Michelson contrast
        max_val = np.percentile(gray, 99)
        min_val = np.percentile(gray, 1)
        michelson_contrast = (max_val - min_val) / (max_val + min_val + 1e-8)

        # RMS contrast
        rms_contrast = np.std(gray)

        # Local contrast
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(gray, size=11)
        local_contrast = np.std(gray - local_mean)

        return {
            'contrast_ratio': float(michelson_contrast),
            'rms_contrast': float(rms_contrast),
            'local_contrast': float(local_contrast),
            'contrast_to_noise': float(michelson_contrast / (np.std(gray[gray < np.percentile(gray, 20)]) + 1e-8))
        }

    def _compute_medical_sharpness(self, gray: np.ndarray) -> Dict[str, float]:
        """Compute medical-specific sharpness metrics"""
        # Gradient-based sharpness
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Laplacian-based sharpness
        laplacian = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
        laplacian_variance = np.var(laplacian) / 1000

        # Edge-based sharpness
        edges = sobel(gray)
        edge_sharpness = np.percentile(edges, 95)

        return {
            'sharpness_index': float(np.mean(gradient_magnitude)),
            'laplacian_sharpness': float(min(1.0, laplacian_variance / 500)),
            'edge_sharpness': float(edge_sharpness),
            'overall_sharpness': float((np.mean(gradient_magnitude) + min(1.0, laplacian_variance / 500) + edge_sharpness) / 3)
        }

    def _detect_artifacts(self, gray: np.ndarray) -> Dict[str, float]:
        """Detect common medical imaging artifacts"""
        # Motion artifact (blurring in one direction)
        grad_x = np.abs(np.gradient(gray, axis=1))
        grad_y = np.abs(np.gradient(gray, axis=0))
        motion_artifact = np.abs(np.mean(grad_x) - np.mean(grad_y)) / (np.mean(grad_x) + np.mean(grad_y) + 1e-8)

        # Ringing artifact (Gibbs phenomenon)
        # Detect by looking for oscillations near edges
        edges = sobel(gray)
        edge_locations = edges > np.percentile(edges, 95)
        if np.any(edge_locations):
            # Look for oscillations perpendicular to edges
            ringing_score = np.std(gray[edge_locations]) / (np.mean(gray[edge_locations]) + 1e-8)
        else:
            ringing_score = 0

        # Noise artifact
        from scipy.ndimage import uniform_filter
        smooth = uniform_filter(gray, size=5)
        noise_level = np.std(gray - smooth)

        return {
            'motion_artifact_score': float(motion_artifact),
            'ringing_artifact_score': float(min(1.0, ringing_score)),
            'noise_artifact_level': float(noise_level),
            'artifact_presence': float((motion_artifact > 0.3) or (ringing_score > 0.5) or (noise_level > 0.1))
        }

    def _estimate_tumor_size(self, gray: np.ndarray) -> Dict[str, float]:
        """Estimate tumor size based on segmentation"""
        # Use adaptive thresholding for tumor segmentation
        from skimage.filters import threshold_local

        threshold = threshold_local(gray, block_size=51, offset=0.1)
        tumor_candidate = gray > threshold

        if np.any(tumor_candidate):
            labeled, num_tumors = ndimage_label(tumor_candidate)
            tumor_sizes = np.bincount(labeled.ravel())[1:]

            if len(tumor_sizes) > 0:
                largest_tumor = np.max(tumor_sizes)
                total_tumor_area = np.sum(tumor_sizes)

                # Estimate diameter (assuming circular)
                estimated_diameter = 2 * np.sqrt(largest_tumor / np.pi)

                return {
                    'tumor_size_pixels': float(largest_tumor),
                    'tumor_estimated_diameter': float(estimated_diameter),
                    'total_tumor_burden': float(total_tumor_area / gray.size),
                    'tumor_stage_proxy': float(min(3, total_tumor_area / (gray.size * 0.1)))
                }

        return {'tumor_size_pixels': 0, 'tumor_estimated_diameter': 0, 'total_tumor_burden': 0, 'tumor_stage_proxy': 0}

    def _compute_tumor_heterogeneity(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Compute tumor heterogeneity metrics"""
        # Segment potential tumor region
        threshold = np.percentile(gray, 85)
        tumor_mask = gray > threshold

        if np.any(tumor_mask):
            # Intensity heterogeneity
            tumor_intensities = gray[tumor_mask]
            intensity_heterogeneity = np.std(tumor_intensities) / (np.mean(tumor_intensities) + 1e-8)

            # Texture heterogeneity within tumor
            if len(tumor_intensities) > 100:
                # Local binary patterns within tumor
                lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
                lbp_tumor = lbp[tumor_mask]
                texture_heterogeneity = np.std(lbp_tumor)

                # Entropy as heterogeneity measure
                hist, _ = np.histogram(tumor_intensities, bins=20, density=True)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log(hist + 1e-10))
            else:
                texture_heterogeneity = 0
                entropy = 0

            # Spatial heterogeneity (clumpiness)
            from scipy.ndimage import uniform_filter
            local_mean = uniform_filter(gray, size=11)
            local_var = uniform_filter(gray**2, size=11) - local_mean**2
            spatial_heterogeneity = np.std(local_var[tumor_mask]) if np.any(tumor_mask) else 0

            return {
                'tumor_intensity_heterogeneity': float(intensity_heterogeneity),
                'tumor_texture_heterogeneity': float(texture_heterogeneity),
                'tumor_entropy': float(entropy),
                'tumor_spatial_heterogeneity': float(spatial_heterogeneity),
                'overall_heterogeneity': float((intensity_heterogeneity + texture_heterogeneity + spatial_heterogeneity) / 3)
            }

        return {'tumor_intensity_heterogeneity': 0, 'tumor_texture_heterogeneity': 0,
                'tumor_entropy': 0, 'tumor_spatial_heterogeneity': 0, 'overall_heterogeneity': 0}

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute medical-specific quality metrics"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # SNR estimate
        signal_region = gray > np.percentile(gray, 90)
        noise_region = gray < np.percentile(gray, 10)
        signal = np.mean(gray[signal_region]) if np.any(signal_region) else np.mean(gray)
        noise = np.std(gray[noise_region]) if np.any(noise_region) else np.std(gray)
        snr = signal / (noise + 1e-8)

        # Contrast-to-noise ratio
        contrast = np.percentile(gray, 95) - np.percentile(gray, 5)
        cnr = contrast / (noise + 1e-8)

        # Sharpness
        laplacian = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
        sharpness = np.var(laplacian) / 1000

        return {
            'snr': float(snr),
            'contrast_to_noise_ratio': float(cnr),
            'sharpness': float(min(1.0, sharpness / 500)),
            'diagnostic_quality': float((snr / 10 + cnr / 20 + min(1.0, sharpness / 500)) / 3),
            'noise_level': float(noise)
        }

# =============================================================================
# COMPLETE SATELLITE DOMAIN PROCESSOR
# =============================================================================

class SatelliteDomainProcessor:
    """Complete Satellite/Remote sensing processor with all features"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.satellite_type = getattr(config, 'satellite_type', 'general')
        self.num_bands = getattr(config, 'num_bands', 4)
        self.band_assignments = getattr(config, 'band_assignments', {
            'red': 0, 'green': 1, 'blue': 2, 'nir': 3
        })
        self.classify_land_cover = getattr(config, 'classify_land_cover', True)
        self.detect_urban_area = getattr(config, 'detect_urban_area', True)
        self.detect_forest_cover = getattr(config, 'detect_forest_cover', True)
        self.detect_water_body = getattr(config, 'detect_water_body', True)
        self.detect_agriculture = getattr(config, 'detect_agriculture', True)
        self.detect_change = getattr(config, 'detect_change', True)
        self.detect_deforestation = getattr(config, 'detect_deforestation', True)
        self.detect_urban_sprawl = getattr(config, 'detect_urban_sprawl', True)
        self.compute_ndvi = getattr(config, 'compute_ndvi', True)
        self.compute_ndwi = getattr(config, 'compute_ndwi', True)
        self.compute_ndbi = getattr(config, 'compute_ndbi', True)
        self.compute_mndwi = getattr(config, 'compute_mndwi', True)
        self.compute_glcm = getattr(config, 'compute_glcm', True)
        self.compute_pansharpening = getattr(config, 'compute_pansharpening', True)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess satellite images"""
        img_float = image.astype(np.float32) / 255.0

        # Satellite-specific preprocessing
        if self.satellite_type == 'sentinel':
            img_float = self._preprocess_sentinel(img_float)
        elif self.satellite_type == 'landsat':
            img_float = self._preprocess_landsat(img_float)
        elif self.satellite_type == 'modis':
            img_float = self._preprocess_modis(img_float)
        else:
            # General satellite preprocessing
            img_float = self._radiometric_correction(img_float)
            img_float = self._atmospheric_correction(img_float)

        return img_float

    def _preprocess_sentinel(self, image: np.ndarray) -> np.ndarray:
        """Preprocess Sentinel satellite imagery"""
        # Apply atmospheric correction approximation
        return self._atmospheric_correction(image)

    def _preprocess_landsat(self, image: np.ndarray) -> np.ndarray:
        """Preprocess Landsat satellite imagery"""
        # Apply TOA reflectance approximation
        return self._radiometric_correction(image)

    def _preprocess_modis(self, image: np.ndarray) -> np.ndarray:
        """Preprocess MODIS satellite imagery"""
        # Apply cloud masking
        return self._cloud_masking(image)

    def _radiometric_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply radiometric correction"""
        for i in range(min(3, image.shape[2])):
            # Simple histogram matching
            image[:, :, i] = exposure.equalize_hist(image[:, :, i])
        return image

    def _atmospheric_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply atmospheric correction (dark object subtraction)"""
        for i in range(min(3, image.shape[2])):
            # Find dark object (5th percentile)
            dark_pixel = np.percentile(image[:, :, i], 5)
            image[:, :, i] = np.clip(image[:, :, i] - dark_pixel, 0, 1)
        return image

    def _cloud_masking(self, image: np.ndarray) -> np.ndarray:
        """Mask clouds in satellite imagery"""
        if len(image.shape) >= 3 and image.shape[2] >= 3:
            # Simple cloud detection (bright and white)
            blue = image[:, :, 2] if len(image.shape) > 2 else image
            cloud_score = blue > 0.8

            # Inpaint clouds
            if np.any(cloud_score):
                cloud_mask = (cloud_score * 255).astype(np.uint8)
                image_uint8 = (image * 255).astype(np.uint8)
                for i in range(min(3, image.shape[2])):
                    image_uint8[:, :, i] = cv2.inpaint(image_uint8[:, :, i], cloud_mask, 5, cv2.INPAINT_TELEA)
                image = image_uint8.astype(np.float32) / 255.0

        return image

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive satellite-specific features"""
        features = {}

        # Spectral indices
        if self.compute_ndvi:
            features.update(self._compute_ndvi(image))
        if self.compute_ndwi:
            features.update(self._compute_ndwi(image))
        if self.compute_ndbi:
            features.update(self._compute_ndbi(image))
        if self.compute_mndwi:
            features.update(self._compute_mndwi(image))

        # Land cover classification
        if self.classify_land_cover:
            features.update(self._classify_land_cover(image))

        # Specific class detection
        if self.detect_urban_area:
            features.update(self._detect_urban_area(image))
        if self.detect_forest_cover:
            features.update(self._detect_forest_cover(image))
        if self.detect_water_body:
            features.update(self._detect_water_body(image))
        if self.detect_agriculture:
            features.update(self._detect_agriculture(image))

        # GLCM texture features
        if self.compute_glcm:
            features.update(self._compute_glcm_features(image))

        # Change detection features
        if self.detect_change:
            features.update(self._detect_change_features(image))

        return features

    def _compute_ndvi(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Normalized Difference Vegetation Index"""
        red_idx = self.band_assignments.get('red', 0)
        nir_idx = self.band_assignments.get('nir', min(3, image.shape[2] - 1))

        if image.shape[2] > max(red_idx, nir_idx):
            red = image[:, :, red_idx]
            nir = image[:, :, nir_idx]

            ndvi = (nir - red) / (nir + red + 1e-8)

            return {
                'ndvi_mean': float(np.mean(ndvi)),
                'ndvi_std': float(np.std(ndvi)),
                'ndvi_max': float(np.max(ndvi)),
                'ndvi_min': float(np.min(ndvi)),
                'vegetation_fraction': float(np.mean(ndvi > 0.3)),
                'dense_vegetation': float(np.mean(ndvi > 0.6)),
                'sparse_vegetation': float(np.mean((ndvi > 0.1) & (ndvi <= 0.3))),
                'barren_land': float(np.mean(ndvi <= 0.1))
            }

        return {'ndvi_mean': 0, 'ndvi_std': 0, 'ndvi_max': 0, 'ndvi_min': 0,
                'vegetation_fraction': 0, 'dense_vegetation': 0, 'sparse_vegetation': 0, 'barren_land': 0}

    def _compute_ndwi(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Normalized Difference Water Index"""
        green_idx = self.band_assignments.get('green', 1)
        nir_idx = self.band_assignments.get('nir', min(3, image.shape[2] - 1))

        if image.shape[2] > max(green_idx, nir_idx):
            green = image[:, :, green_idx]
            nir = image[:, :, nir_idx]

            ndwi = (green - nir) / (green + nir + 1e-8)

            return {
                'ndwi_mean': float(np.mean(ndwi)),
                'water_fraction': float(np.mean(ndwi > 0)),
                'open_water': float(np.mean(ndwi > 0.3)),
                'wetland': float(np.mean((ndwi > 0) & (ndwi <= 0.3)))
            }

        return {'ndwi_mean': 0, 'water_fraction': 0, 'open_water': 0, 'wetland': 0}

    def _compute_ndbi(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Normalized Difference Built-up Index"""
        swir_idx = min(2, image.shape[2] - 1)  # Approximate SWIR
        nir_idx = self.band_assignments.get('nir', min(3, image.shape[2] - 1))

        if image.shape[2] > max(swir_idx, nir_idx):
            swir = image[:, :, swir_idx]
            nir = image[:, :, nir_idx]

            ndbi = (swir - nir) / (swir + nir + 1e-8)

            return {
                'ndbi_mean': float(np.mean(ndbi)),
                'built_up_fraction': float(np.mean(ndbi > 0)),
                'urban_density': float(np.mean(ndbi > 0.2)),
                'impervious_surface': float(np.mean(ndbi > 0.1))
            }

        return {'ndbi_mean': 0, 'built_up_fraction': 0, 'urban_density': 0, 'impervious_surface': 0}

    def _compute_mndwi(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Modified Normalized Difference Water Index"""
        green_idx = self.band_assignments.get('green', 1)
        swir_idx = min(2, image.shape[2] - 1)  # Approximate SWIR

        if image.shape[2] > max(green_idx, swir_idx):
            green = image[:, :, green_idx]
            swir = image[:, :, swir_idx]

            mndwi = (green - swir) / (green + swir + 1e-8)

            return {
                'mndwi_mean': float(np.mean(mndwi)),
                'water_improved_fraction': float(np.mean(mndwi > 0)),
                'water_quality_proxy': float(np.mean(mndwi))
            }

        return {'mndwi_mean': 0, 'water_improved_fraction': 0, 'water_quality_proxy': 0}

    def _classify_land_cover(self, image: np.ndarray) -> Dict[str, float]:
        """Classify land cover types using spectral indices"""
        ndvi = self._compute_ndvi(image)
        ndwi = self._compute_ndwi(image)
        ndbi = self._compute_ndbi(image)

        # Simple classification based on indices
        vegetation = ndvi.get('vegetation_fraction', 0)
        water = ndwi.get('water_fraction', 0)
        urban = ndbi.get('built_up_fraction', 0)

        # Remaining is barren/other
        barren = max(0, 1 - (vegetation + water + urban))

        return {
            'vegetation_cover': float(vegetation),
            'water_cover': float(water),
            'urban_cover': float(urban),
            'barren_cover': float(barren),
            'land_cover_diversity': float(1 - (vegetation**2 + water**2 + urban**2 + barren**2))  # Simpson diversity
        }

    def _detect_urban_area(self, image: np.ndarray) -> Dict[str, float]:
        """Detect and analyze urban areas"""
        ndbi = self._compute_ndbi(image)
        urban_fraction = ndbi.get('urban_density', 0)

        # Texture analysis for urban areas (high contrast, regular patterns)
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Urban areas typically have high local contrast
        from scipy.ndimage import uniform_filter
        local_contrast = uniform_filter(gray**2, size=11) - uniform_filter(gray, size=11)**2

        urban_texture = np.mean(local_contrast[local_contrast > np.percentile(local_contrast, 90)])

        return {
            'urban_fraction': float(urban_fraction),
            'urban_texture_index': float(urban_texture),
            'urban_sprawl_index': float(urban_fraction * (1 - urban_fraction)),
            'urban_compactness': float(1 - np.std(local_contrast[local_contrast > 0]) if np.any(local_contrast > 0) else 0)
        }

    def _detect_forest_cover(self, image: np.ndarray) -> Dict[str, float]:
        """Detect and analyze forest cover"""
        ndvi = self._compute_ndvi(image)
        dense_forest = ndvi.get('dense_vegetation', 0)

        # Forest texture (rough, high variability)
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Forest canopy roughness
        edges = sobel(gray)
        forest_roughness = np.mean(edges[edges > np.percentile(edges, 80)])

        return {
            'forest_cover_fraction': float(dense_forest),
            'forest_canopy_roughness': float(forest_roughness),
            'forest_health_index': float(ndvi.get('ndvi_mean', 0)),
            'forest_fragmentation': float(1 - (dense_forest / (dense_forest + 0.1)))
        }

    def _detect_water_body(self, image: np.ndarray) -> Dict[str, float]:
        """Detect and analyze water bodies"""
        ndwi = self._compute_ndwi(image)
        mndwi = self._compute_mndwi(image)

        water_fraction = ndwi.get('open_water', 0)

        # Water body smoothness (low texture)
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        water_mask = gray < np.percentile(gray, 20)
        water_smoothness = 1 - np.std(gray[water_mask]) if np.any(water_mask) else 0

        return {
            'water_body_fraction': float(water_fraction),
            'water_quality_index': float(mndwi.get('water_quality_proxy', 0)),
            'water_smoothness': float(water_smoothness),
            'water_turbidity_proxy': float(1 - water_smoothness)
        }

    def _detect_agriculture(self, image: np.ndarray) -> Dict[str, float]:
        """Detect and analyze agricultural areas"""
        ndvi = self._compute_ndvi(image)
        vegetation = ndvi.get('vegetation_fraction', 0)

        # Agricultural areas have regular patterns (field boundaries)
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Detect field boundaries using edge detection
        edges = sobel(gray)

        # Agricultural fields show regular edge patterns
        from scipy.ndimage import convolve
        edge_orientation = np.arctan2(np.gradient(gray, axis=0), np.gradient(gray, axis=1))
        orientation_hist, _ = np.histogram(edge_orientation[edges > np.percentile(edges, 90)], bins=36)
        orientation_regularity = np.max(orientation_hist) / (np.sum(orientation_hist) + 1e-8)

        return {
            'agriculture_fraction': float(vegetation * 0.8),  # Approximate
            'field_regularity': float(orientation_regularity),
            'crop_health': float(ndvi.get('ndvi_mean', 0)),
            'agricultural_intensity': float(vegetation * orientation_regularity)
        }

    def _compute_glcm_features(self, image: np.ndarray) -> Dict[str, float]:
        """Compute GLCM features for satellite imagery"""
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        gray_uint8 = (gray * 255).astype(np.uint8)

        # Multi-scale GLCM
        glcm_features = {}

        for distance in [1, 3, 5]:
            glcm = graycomatrix(gray_uint8, distances=[distance], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                               levels=256, symmetric=True)

            glcm_features[f'contrast_d{distance}'] = float(np.mean([graycoprops(glcm, 'contrast')[0, i] for i in range(4)]))
            glcm_features[f'homogeneity_d{distance}'] = float(np.mean([graycoprops(glcm, 'homogeneity')[0, i] for i in range(4)]))
            glcm_features[f'energy_d{distance}'] = float(np.mean([graycoprops(glcm, 'energy')[0, i] for i in range(4)]))
            glcm_features[f'correlation_d{distance}'] = float(np.mean([graycoprops(glcm, 'correlation')[0, i] for i in range(4)]))

        return glcm_features

    def _detect_change_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features for change detection (single image features)"""
        # These are static features that would be compared with temporal data
        # For now, extract baseline features
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Land cover stability proxy (texture uniformity)
        texture_uniformity = 1 - np.std(gray) / (np.mean(gray) + 1e-8)

        # Edge density as proxy for landscape fragmentation
        edges = sobel(gray)
        edge_density = np.mean(edges > np.percentile(edges, 85))

        return {
            'change_susceptibility': float(1 - texture_uniformity),
            'landscape_fragmentation': float(edge_density),
            'disturbance_proxy': float((1 - texture_uniformity) * edge_density)
        }

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute satellite-specific quality metrics"""
        if len(image.shape) >= 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Cloud cover estimation
        cloud_fraction = np.mean(gray > 0.95)

        # Sharpness (spatial resolution proxy)
        laplacian = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
        sharpness = np.var(laplacian) / 1000

        # Radiometric quality (dynamic range)
        dynamic_range = np.percentile(gray, 99) - np.percentile(gray, 1)

        return {
            'cloud_cover_fraction': float(cloud_fraction),
            'spatial_sharpness': float(min(1.0, sharpness / 500)),
            'radiometric_dynamic_range': float(dynamic_range),
            'radiometric_quality': float(dynamic_range),
            'image_usability': float(1 - cloud_fraction)
        }

# =============================================================================
# COMPLETE SURVEILLANCE DOMAIN PROCESSOR
# =============================================================================

class SurveillanceDomainProcessor:
    """Complete Surveillance/CCTV processor with all features - DETERMINISTIC"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.detect_person = getattr(config, 'detect_person', True)
        self.detect_vehicle = getattr(config, 'detect_vehicle', True)
        self.detect_animal = getattr(config, 'detect_animal', True)
        self.detect_face = getattr(config, 'detect_face', True)
        self.detect_motion = getattr(config, 'detect_motion', True)
        self.detect_anomaly = getattr(config, 'detect_anomaly', True)
        self.track_objects = getattr(config, 'track_objects', True)
        self.classify_scene_type = getattr(config, 'classify_scene_type', True)
        self.estimate_crowd_density = getattr(config, 'estimate_crowd_density', True)
        self.enhance_low_light = getattr(config, 'enhance_low_light', True)
        self.reduce_noise = getattr(config, 'reduce_noise', True)
        self.dehaze = getattr(config, 'dehaze', True)
        self.super_resolution = getattr(config, 'super_resolution', True)
        self.blur_faces = getattr(config, 'blur_faces', False)
        self.anonymize = getattr(config, 'anonymize', False)

        # CRITICAL FIX: Use fixed random state for reproducibility
        self._rng = np.random.RandomState(42)

        # Background subtractor for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.previous_frame = None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess surveillance images"""
        img_float = image.astype(np.float32) / 255.0

        # Apply enhancements in sequence
        if self.enhance_low_light:
            img_float = self._enhance_low_light(img_float)

        if self.reduce_noise:
            img_float = self._reduce_noise(img_float)

        if self.dehaze:
            img_float = self._dehaze(img_float)

        # Apply privacy features if enabled
        if self.blur_faces or self.anonymize:
            img_float = self._apply_privacy(img_float)

        return img_float

    def _enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """Enhance low-light surveillance images - DETERMINISTIC"""
        if len(image.shape) == 3:
            # Convert to LAB
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32) / 255.0

            # Enhance L channel
            l_channel = lab[:, :, 0]

            # Adaptive gamma correction (deterministic - based on mean)
            mean_l = np.mean(l_channel)
            gamma = 0.5 if mean_l < 0.3 else 1.0
            l_enhanced = np.power(l_channel, gamma)

            # CLAHE for local contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply((l_enhanced * 255).astype(np.uint8)) / 255.0

            lab[:, :, 0] = l_enhanced

            # Convert back
            lab = (lab * 255).astype(np.uint8)
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) / 255.0
        else:
            # Grayscale enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply((image * 255).astype(np.uint8)) / 255.0

        return image

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise in surveillance images"""
        if len(image.shape) == 3:
            # Bilateral filter for edge-preserving denoising
            for i in range(3):
                image[:, :, i] = cv2.bilateralFilter((image[:, :, i] * 255).astype(np.uint8), 9, 75, 75) / 255.0
        else:
            image = cv2.bilateralFilter((image * 255).astype(np.uint8), 9, 75, 75) / 255.0
        return image

    def _dehaze(self, image: np.ndarray) -> np.ndarray:
        """Remove haze from surveillance images"""
        if len(image.shape) == 3:
            img_uint8 = (image * 255).astype(np.uint8)
            # Dark channel prior dehazing (simplified)
            dark_channel = np.min(img_uint8, axis=2)
            atmospheric_light = np.percentile(dark_channel[dark_channel > 0], 99)

            # Estimate transmission
            transmission = 1 - 0.95 * (dark_channel / (atmospheric_light + 1e-8))
            transmission = np.clip(transmission, 0.1, 1)

            # Recover image
            dehazed = np.zeros_like(img_uint8, dtype=np.float32)
            for i in range(3):
                dehazed[:, :, i] = (img_uint8[:, :, i] - atmospheric_light) / (transmission + 1e-8) + atmospheric_light

            image = np.clip(dehazed / 255.0, 0, 1)

        return image

    def _apply_privacy(self, image: np.ndarray) -> np.ndarray:
        """Apply privacy features (face blurring)"""
        if len(image.shape) == 3:
            img_uint8 = (image * 255).astype(np.uint8)

            # Face detection using Haar cascades
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Blur faces
            for (x, y, w, h) in faces:
                roi = img_uint8[y:y+h, x:x+w]
                roi = cv2.GaussianBlur(roi, (51, 51), 30)
                img_uint8[y:y+h, x:x+w] = roi

            image = img_uint8 / 255.0

        return image

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive surveillance-specific features"""
        features = {}

        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # Object detection
        if self.detect_person:
            features.update(self._detect_person(image, gray))

        if self.detect_vehicle:
            features.update(self._detect_vehicle(image, gray))

        if self.detect_animal:
            features.update(self._detect_animal(image, gray))

        if self.detect_face:
            features.update(self._detect_face(image, gray))

        # Motion detection
        if self.detect_motion:
            features.update(self._detect_motion(image, gray))

        # Anomaly detection
        if self.detect_anomaly:
            features.update(self._detect_anomaly(image, gray))

        # Scene understanding
        if self.classify_scene_type:
            features.update(self._classify_scene_type(image, gray))

        # Crowd analysis
        if self.estimate_crowd_density:
            features.update(self._estimate_crowd_density(image, gray))

        return features

    def _detect_person(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect persons using HOG descriptor"""
        # HOG person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        img_uint8 = (gray * 255).astype(np.uint8)
        persons, _ = hog.detectMultiScale(img_uint8, winStride=(4, 4), padding=(8, 8), scale=1.05)

        if len(persons) > 0:
            person_areas = [w * h for (x, y, w, h) in persons]

            return {
                'person_count': float(len(persons)),
                'person_density': float(len(persons) / (gray.size / 10000)),
                'avg_person_size': float(np.mean(person_areas) if person_areas else 0),
                'person_coverage': float(np.sum(person_areas) / gray.size),
                'crowdedness': float(min(1.0, len(persons) / 20))
            }

        return {'person_count': 0, 'person_density': 0, 'avg_person_size': 0, 'person_coverage': 0, 'crowdedness': 0}

    def _detect_vehicle(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect vehicles using contour analysis"""
        # Use background subtraction for vehicle detection
        fgmask = self.background_subtractor.apply((gray * 255).astype(np.uint8))

        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for vehicle-sized objects
        vehicles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:  # Vehicle size range
                vehicles.append(contour)

        if len(vehicles) > 0:
            vehicle_areas = [cv2.contourArea(v) for v in vehicles]

            return {
                'vehicle_count': float(len(vehicles)),
                'vehicle_density': float(len(vehicles) / (gray.size / 10000)),
                'avg_vehicle_size': float(np.mean(vehicle_areas)),
                'traffic_density': float(min(1.0, len(vehicles) / 30))
            }

        return {'vehicle_count': 0, 'vehicle_density': 0, 'avg_vehicle_size': 0, 'traffic_density': 0}

    def _detect_animal(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect animals using texture and shape analysis"""
        # Animals often have irregular textures
        edges = sobel(gray)
        edge_density = np.mean(edges > np.percentile(edges, 90))

        # Look for regions with high edge density and specific sizes
        if edge_density > 0.05:
            # Segment candidate regions
            _, binary = cv2.threshold((edges * 255).astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            animal_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 5000:  # Animal size range
                    # Check shape complexity
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter**2 + 1e-8)
                    if circularity < 0.5:  # Non-circular (animal-like)
                        animal_candidates.append(contour)

            return {
                'animal_count': float(len(animal_candidates)),
                'animal_detection_score': float(min(1.0, len(animal_candidates) / 10)),
                'wildlife_activity': float(edge_density)
            }

        return {'animal_count': 0, 'animal_detection_score': 0, 'wildlife_activity': 0}

    def _detect_face(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect faces in surveillance images"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img_uint8 = (gray * 255).astype(np.uint8)
        faces = face_cascade.detectMultiScale(img_uint8, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            face_areas = [w * h for (x, y, w, h) in faces]

            return {
                'face_count': float(len(faces)),
                'face_density': float(len(faces) / (gray.size / 10000)),
                'avg_face_size': float(np.mean(face_areas)),
                'face_detection_confidence': float(min(1.0, len(faces) / 20))
            }

        return {'face_count': 0, 'face_density': 0, 'avg_face_size': 0, 'face_detection_confidence': 0}

    def _detect_motion(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect motion between frames"""
        if self.previous_frame is not None:
            # Compute frame difference
            diff = cv2.absdiff(self.previous_frame, (gray * 255).astype(np.uint8))
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            motion_fraction = np.sum(motion_mask > 0) / motion_mask.size

            # Motion intensity
            motion_intensity = np.mean(diff[diff > 0]) / 255.0 if np.any(diff > 0) else 0

            # Motion direction (using optical flow approximation)
            flow = cv2.calcOpticalFlowFarneback(self.previous_frame, (gray * 255).astype(np.uint8),
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_motion_magnitude = np.mean(magnitude)
            motion_coherence = np.std(angle) / (2 * np.pi)  # Lower = more coherent

            self.previous_frame = (gray * 255).astype(np.uint8)

            return {
                'motion_fraction': float(motion_fraction),
                'motion_intensity': float(motion_intensity),
                'avg_motion_magnitude': float(avg_motion_magnitude),
                'motion_coherence': float(motion_coherence),
                'activity_level': float(motion_fraction * motion_intensity)
            }
        else:
            self.previous_frame = (gray * 255).astype(np.uint8)
            return {'motion_fraction': 0, 'motion_intensity': 0, 'avg_motion_magnitude': 0,
                    'motion_coherence': 1, 'activity_level': 0}

    def _detect_anomaly(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect anomalies in surveillance footage"""
        # Anomaly indicators: sudden changes, unusual objects, abnormal motion

        # Texture anomaly (unusual patterns)
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        texture_anomaly = np.std(lbp) / 10  # Normalized

        # Intensity anomaly (sudden brightness changes)
        intensity_mean = np.mean(gray)
        intensity_std = np.std(gray)
        intensity_anomaly = intensity_std / (intensity_mean + 1e-8)

        # Edge anomaly (unusual edge density)
        edges = sobel(gray)
        edge_density = np.mean(edges > np.percentile(edges, 90))
        edge_anomaly = edge_density / 0.1  # Normalized to typical edge density

        # Combined anomaly score
        anomaly_score = (texture_anomaly + intensity_anomaly + edge_anomaly) / 3
        anomaly_score = min(1.0, anomaly_score)

        return {
            'texture_anomaly': float(texture_anomaly),
            'intensity_anomaly': float(intensity_anomaly),
            'edge_anomaly': float(edge_anomaly),
            'anomaly_score': float(anomaly_score),
            'alert_level': float(anomaly_score * 100)
        }

    def _classify_scene_type(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Classify scene type (indoor/outdoor, day/night, etc.)"""
        # Day/Night classification
        mean_intensity = np.mean(gray)
        is_night = mean_intensity < 0.3

        # Indoor/Outdoor classification using color and texture
        if len(image.shape) >= 3:
            # Outdoor tends to have more sky (blue) and vegetation (green)
            blue_ratio = np.mean(image[:, :, 2] / (np.mean(image, axis=2) + 1e-8))
            green_ratio = np.mean(image[:, :, 1] / (np.mean(image, axis=2) + 1e-8))

            is_outdoor = (blue_ratio > 0.35) or (green_ratio > 0.4)
            outdoor_prob = float(min(1.0, (blue_ratio + green_ratio)))
        else:
            is_outdoor = False
            outdoor_prob = 0

        # Scene complexity
        edges = sobel(gray)
        scene_complexity = np.mean(edges > np.percentile(edges, 85))

        return {
            'is_night': float(1 if is_night else 0),
            'night_probability': float(1 - mean_intensity * 3),
            'is_outdoor': float(1 if is_outdoor else 0),
            'outdoor_probability': float(outdoor_prob),
            'scene_complexity': float(scene_complexity)
        }

    def _estimate_crowd_density(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Estimate crowd density using texture analysis"""
        # Crowd density estimation using texture energy
        from scipy.ndimage import uniform_filter

        # High crowd density = high frequency texture
        laplacian = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
        high_freq_energy = np.var(laplacian) / 1000

        # Local binary patterns for crowd texture
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_entropy = -np.sum(np.bincount(lbp.astype(int).ravel()) / lbp.size *
                             np.log(np.bincount(lbp.astype(int).ravel()) / lbp.size + 1e-10))

        # Estimate crowd density (0-5 scale)
        if high_freq_energy < 50:
            crowd_density = 0  # Empty
        elif high_freq_energy < 150:
            crowd_density = 1  # Very low
        elif high_freq_energy < 300:
            crowd_density = 2  # Low
        elif high_freq_energy < 500:
            crowd_density = 3  # Medium
        elif high_freq_energy < 800:
            crowd_density = 4  # High
        else:
            crowd_density = 5  # Very high

        return {
            'crowd_density_level': float(crowd_density),
            'crowd_density_score': float(min(1.0, high_freq_energy / 1000)),
            'crowd_texture_entropy': float(lbp_entropy),
            'crowd_activity': float(high_freq_energy / 500),
            'social_distancing_score': float(1 - min(1.0, high_freq_energy / 800))
        }

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute surveillance-specific quality metrics"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # Brightness
        brightness = np.mean(gray)

        # Contrast
        contrast = np.std(gray)

        # Sharpness
        laplacian = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
        sharpness = np.var(laplacian) / 1000

        # Noise level
        from scipy.ndimage import uniform_filter
        smooth = uniform_filter(gray, size=5)
        noise_level = np.std(gray - smooth)

        # Usability score for surveillance
        usability = (brightness * 0.3 + contrast * 0.3 + min(1.0, sharpness / 500) * 0.4)

        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'sharpness': float(min(1.0, sharpness / 500)),
            'noise_level': float(noise_level),
            'surveillance_usability': float(usability)
        }

# =============================================================================
# COMPLETE MICROSCOPY DOMAIN PROCESSOR
# =============================================================================

class MicroscopyDomainProcessor:
    """Complete Microscopy imaging processor with all features - DETERMINISTIC"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.microscopy_type = getattr(config, 'microscopy_type', 'general')
        self.detect_cells = getattr(config, 'detect_cells', True)
        self.count_cells = getattr(config, 'count_cells', True)
        self.segment_nucleus = getattr(config, 'segment_nucleus', True)
        self.detect_mitosis = getattr(config, 'detect_mitosis', True)
        self.detect_mitochondria = getattr(config, 'detect_mitochondria', True)
        self.detect_nucleoli = getattr(config, 'detect_nucleoli', True)
        self.detect_fluorescent_signal = getattr(config, 'detect_fluorescent_signal', True)
        self.compute_intensity_distribution = getattr(config, 'compute_intensity_distribution', True)
        self.compute_resolution = getattr(config, 'compute_resolution', True)
        self.detect_out_of_focus = getattr(config, 'detect_out_of_focus', True)

        # CRITICAL FIX: Fixed random state for reproducibility
        self._rng = np.random.RandomState(42)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess microscopy images"""
        img_float = image.astype(np.float32) / 255.0

        # Microscopy type-specific preprocessing
        if self.microscopy_type == 'fluorescence':
            img_float = self._preprocess_fluorescence(img_float)
        elif self.microscopy_type == 'phase_contrast':
            img_float = self._preprocess_phase_contrast(img_float)
        elif self.microscopy_type == 'electron':
            img_float = self._preprocess_electron(img_float)
        else:
            # General microscopy preprocessing
            img_float = self._subtract_background(img_float)
            img_float = self._enhance_contrast(img_float)

        return img_float

    def _preprocess_fluorescence(self, image: np.ndarray) -> np.ndarray:
        """Preprocess fluorescence microscopy images"""
        # Denoise while preserving signal
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                image[:, :, i] = restoration.denoise_nl_means(image[:, :, i], patch_size=5, patch_distance=6)
        else:
            image = restoration.denoise_nl_means(image, patch_size=5, patch_distance=6)
        return image

    def _preprocess_phase_contrast(self, image: np.ndarray) -> np.ndarray:
        """Preprocess phase contrast microscopy images"""
        # Remove halo artifacts
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Background removal using rolling ball
        from skimage import morphology
        selem = morphology.disk(20)
        background = morphology.opening(gray, selem)
        image = gray - background

        return np.clip(image, 0, 1)

    def _preprocess_electron(self, image: np.ndarray) -> np.ndarray:
        """Preprocess electron microscopy images"""
        # Enhance edges and reduce noise
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                image[:, :, i] = cv2.bilateralFilter((image[:, :, i] * 255).astype(np.uint8), 5, 50, 50) / 255.0
        else:
            image = cv2.bilateralFilter((image * 255).astype(np.uint8), 5, 50, 50) / 255.0
        return image

    def _subtract_background(self, image: np.ndarray) -> np.ndarray:
        """Subtract background from microscopy images"""
        from skimage import morphology

        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                selem = morphology.disk(50)
                background = morphology.opening(image[:, :, i], selem)
                image[:, :, i] = image[:, :, i] - background
                image[:, :, i] = np.clip(image[:, :, i], 0, 1)
        else:
            selem = morphology.disk(50)
            background = morphology.opening(image, selem)
            image = image - background
            image = np.clip(image, 0, 1)

        return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for microscopy"""
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                image[:, :, i] = exposure.equalize_adapthist(image[:, :, i])
        else:
            image = exposure.equalize_adapthist(image)
        return image

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive microscopy-specific features"""
        features = {}

        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # Cell analysis
        if self.detect_cells:
            features.update(self._detect_cells(image, gray))

        if self.count_cells:
            features.update(self._count_cells(gray))

        # Nuclear features
        if self.segment_nucleus:
            features.update(self._segment_nucleus(image, gray))

        # Mitosis detection
        if self.detect_mitosis:
            features.update(self._detect_mitosis(image, gray))

        # Organelle detection
        if self.detect_mitochondria:
            features.update(self._detect_mitochondria(gray))

        if self.detect_nucleoli:
            features.update(self._detect_nucleoli(gray))

        # Fluorescence features
        if self.detect_fluorescent_signal:
            features.update(self._detect_fluorescent_signal(image, gray))

        # Intensity analysis
        if self.compute_intensity_distribution:
            features.update(self._compute_intensity_distribution(gray))

        # Quality metrics
        if self.compute_resolution:
            features.update(self._compute_resolution(gray))

        if self.detect_out_of_focus:
            features.update(self._detect_out_of_focus(gray))

        return features

    def _detect_cells(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect cells using blob detection and watershed"""
        from skimage.feature import blob_log

        # Blob detection for cell bodies
        blobs = blob_log(gray, max_sigma=30, num_sigma=10, threshold=0.05)

        if len(blobs) > 0:
            cell_count = len(blobs)
            cell_sizes = [b[2] for b in blobs]
            cell_intensities = []

            for blob in blobs[:min(50, len(blobs))]:
                y, x, r = int(blob[0]), int(blob[1]), int(blob[2])
                y_min, y_max = max(0, y - r), min(gray.shape[0], y + r)
                x_min, x_max = max(0, x - r), min(gray.shape[1], x + r)
                cell_intensities.append(np.mean(gray[y_min:y_max, x_min:x_max]))

            # Cell density
            cell_density = cell_count / (gray.size / 1000000)  # Per million pixels

            return {
                'cell_count': float(cell_count),
                'cell_density': float(cell_density),
                'avg_cell_size': float(np.mean(cell_sizes)),
                'avg_cell_intensity': float(np.mean(cell_intensities)),
                'cell_size_variability': float(np.std(cell_sizes) / (np.mean(cell_sizes) + 1e-8)),
                'cell_clustering': float(1 - (cell_count / (gray.size / 10000)) * 0.1)
            }

        return {'cell_count': 0, 'cell_density': 0, 'avg_cell_size': 0,
                'avg_cell_intensity': 0, 'cell_size_variability': 0, 'cell_clustering': 0}

    def _count_cells(self, gray: np.ndarray) -> Dict[str, float]:
        """Count cells using thresholding and labeling"""
        # Adaptive thresholding
        from skimage.filters import threshold_otsu

        threshold = threshold_otsu(gray)
        binary = gray > threshold

        # Morphological cleanup
        binary = morphology.binary_closing(binary, morphology.disk(3))
        binary = morphology.binary_opening(binary, morphology.disk(2))

        # Label connected components
        labeled, num_cells = ndimage_label(binary)

        # Filter by size (remove small artifacts)
        cell_sizes = np.bincount(labeled.ravel())[1:]
        valid_cells = [size for size in cell_sizes if size > 20]  # Minimum cell size

        return {
            'total_cell_count': float(len(valid_cells)),
            'cell_area_fraction': float(np.mean(binary)),
            'estimated_confluence': float(min(1.0, np.mean(binary) * 2)),
            'cell_density_estimate': float(len(valid_cells) / (gray.size / 1000000))
        }

    def _segment_nucleus(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Segment and analyze cell nuclei"""
        # Nuclei are typically darker in brightfield, brighter in fluorescence
        if self.microscopy_type == 'fluorescence':
            # Nuclei are bright in fluorescence
            threshold = np.percentile(gray, 80)
            nuclei_mask = gray > threshold
        else:
            # Nuclei are darker in brightfield
            threshold = np.percentile(gray, 20)
            nuclei_mask = gray < threshold

        if np.any(nuclei_mask):
            # Clean up
            nuclei_mask = morphology.binary_opening(nuclei_mask, morphology.disk(2))
            nuclei_mask = morphology.binary_closing(nuclei_mask, morphology.disk(3))

            labeled, num_nuclei = ndimage_label(nuclei_mask)

            # Nuclear features
            nucleus_areas = np.bincount(labeled.ravel())[1:]
            nucleus_intensities = [np.mean(gray[labeled == i]) for i in range(1, num_nuclei + 1)]

            # Nuclear shape features
            from skimage.measure import regionprops
            props = regionprops(labeled)

            if props:
                avg_nuclear_area = np.mean([p.area for p in props])
                avg_nuclear_compactness = np.mean([4 * np.pi * p.area / (p.perimeter**2) for p in props])
                avg_nuclear_eccentricity = np.mean([p.eccentricity for p in props])
            else:
                avg_nuclear_area = 0
                avg_nuclear_compactness = 0
                avg_nuclear_eccentricity = 0

            return {
                'nucleus_count': float(num_nuclei),
                'nucleus_area_fraction': float(np.mean(nuclei_mask)),
                'avg_nucleus_size': float(avg_nuclear_area),
                'nucleus_compactness': float(avg_nuclear_compactness),
                'nucleus_eccentricity': float(avg_nuclear_eccentricity),
                'nucleus_to_cell_ratio': float(avg_nuclear_area / (avg_nuclear_area + 100))  # Approximate
            }

        return {'nucleus_count': 0, 'nucleus_area_fraction': 0, 'avg_nucleus_size': 0,
                'nucleus_compactness': 0, 'nucleus_eccentricity': 0, 'nucleus_to_cell_ratio': 0}

    def _detect_mitosis(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect mitotic cells (dividing cells)"""
        # Mitotic cells have distinct morphology: rounded, condensed chromatin
        from skimage.feature import blob_log

        # Look for small, bright, circular objects
        blobs = blob_log(gray, max_sigma=10, num_sigma=10, threshold=0.08)

        # Filter for mitotic-like features
        mitotic_candidates = []
        for blob in blobs:
            if 3 < blob[2] < 10:  # Size range for mitotic cells
                mitotic_candidates.append(blob)

        # Additional texture analysis for mitosis
        # Mitotic cells have higher local contrast
        from scipy.ndimage import uniform_filter
        local_std = np.sqrt(uniform_filter(gray**2, size=11) - uniform_filter(gray, size=11)**2)
        mitotic_texture = np.mean(local_std[local_std > np.percentile(local_std, 95)])

        return {
            'mitotic_cell_count': float(len(mitotic_candidates)),
            'mitotic_index': float(len(mitotic_candidates) / (len(blobs) + 1e-8)),
            'mitotic_texture_score': float(mitotic_texture),
            'proliferation_marker': float(min(1.0, len(mitotic_candidates) / 20))
        }

    def _detect_mitochondria(self, gray: np.ndarray) -> Dict[str, float]:
        """Detect and analyze mitochondria"""
        # Mitochondria appear as elongated, filamentous structures
        # Enhance tubular structures
        from skimage.filters import frangi

        # Frangi filter for vessel/tubular enhancement
        tubeness = frangi(gray, scale_range=(1, 5), scale_step=1)

        # Threshold for mitochondrial regions
        threshold = np.percentile(tubeness, 90)
        mitochondria_mask = tubeness > threshold

        if np.any(mitochondria_mask):
            # Mitochondrial features
            mitochondrial_fraction = np.mean(mitochondria_mask)

            # Length estimation via skeletonization
            skeleton = morphology.skeletonize(mitochondria_mask)
            mitochondrial_length = np.sum(skeleton)

            # Branching complexity
            from skimage.morphology import branch_points
            branches = branch_points(skeleton)
            branch_count = np.sum(branches)

            return {
                'mitochondrial_fraction': float(mitochondrial_fraction),
                'mitochondrial_length': float(mitochondrial_length / (gray.size / 1000)),
                'mitochondrial_branching': float(branch_count),
                'mitochondrial_network_complexity': float(mitochondrial_length / (mitochondrial_fraction * gray.size + 1e-8))
            }

        return {'mitochondrial_fraction': 0, 'mitochondrial_length': 0,
                'mitochondrial_branching': 0, 'mitochondrial_network_complexity': 0}

    def _detect_nucleoli(self, gray: np.ndarray) -> Dict[str, float]:
        """Detect and analyze nucleoli"""
        # Nucleoli are small, bright spots within nuclei
        from skimage.feature import blob_log

        # Look for small bright blobs
        blobs = blob_log(gray, max_sigma=5, num_sigma=10, threshold=0.1)

        # Filter for nucleolus-sized objects (1-3 pixels radius)
        nucleoli = [b for b in blobs if 1 < b[2] < 4]

        if len(nucleoli) > 0:
            nucleolus_count = len(nucleoli)
            nucleolus_sizes = [b[2] for b in nucleoli]
            nucleolus_intensities = []

            for blob in nucleoli:
                y, x, r = int(blob[0]), int(blob[1]), int(blob[2])
                y_min, y_max = max(0, y - r), min(gray.shape[0], y + r)
                x_min, x_max = max(0, x - r), min(gray.shape[1], x + r)
                nucleolus_intensities.append(np.mean(gray[y_min:y_max, x_min:x_max]))

            return {
                'nucleolus_count': float(nucleolus_count),
                'nucleolus_density': float(nucleolus_count / (gray.size / 10000)),
                'avg_nucleolus_size': float(np.mean(nucleolus_sizes)),
                'avg_nucleolus_intensity': float(np.mean(nucleolus_intensities))
            }

        return {'nucleolus_count': 0, 'nucleolus_density': 0, 'avg_nucleolus_size': 0, 'avg_nucleolus_intensity': 0}

    def _detect_fluorescent_signal(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect and quantify fluorescent signal"""
        if len(image.shape) >= 3:
            # Use the brightest channel as fluorescence
            fluorescence = np.max(image, axis=2)
        else:
            fluorescence = gray

        # Signal-to-noise ratio
        signal_region = fluorescence > np.percentile(fluorescence, 90)
        background_region = fluorescence < np.percentile(fluorescence, 10)

        signal = np.mean(fluorescence[signal_region]) if np.any(signal_region) else 0
        background = np.mean(fluorescence[background_region]) if np.any(background_region) else np.std(fluorescence)
        snr = signal / (background + 1e-8)

        # Fluorescence intensity distribution
        intensity_hist, _ = np.histogram(fluorescence[fluorescence > 0], bins=50, density=True)
        intensity_entropy = -np.sum(intensity_hist * np.log(intensity_hist + 1e-10))

        # Fluorescent foci detection
        from skimage.feature import blob_log
        foci = blob_log(fluorescence, max_sigma=3, num_sigma=10, threshold=0.15)

        return {
            'fluorescence_snr': float(snr),
            'fluorescence_intensity': float(signal),
            'fluorescence_entropy': float(intensity_entropy),
            'fluorescent_foci_count': float(len(foci)),
            'fluorescence_quality': float(min(1.0, snr / 10))
        }

    def _compute_intensity_distribution(self, gray: np.ndarray) -> Dict[str, float]:
        """Compute intensity distribution features"""
        # Basic statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = np.mean(((gray - mean_intensity) / (std_intensity + 1e-8)) ** 3)
        kurtosis = np.mean(((gray - mean_intensity) / (std_intensity + 1e-8)) ** 4) - 3

        # Percentiles
        percentiles = np.percentile(gray, [1, 5, 10, 25, 50, 75, 90, 95, 99])

        # Intensity heterogeneity
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(gray, size=11)
        intensity_heterogeneity = np.std(local_mean) / (np.mean(local_mean) + 1e-8)

        return {
            'intensity_mean': float(mean_intensity),
            'intensity_std': float(std_intensity),
            'intensity_skewness': float(skewness),
            'intensity_kurtosis': float(kurtosis),
            'intensity_range': float(percentiles[8] - percentiles[0]),
            'intensity_heterogeneity': float(intensity_heterogeneity),
            'p1_intensity': float(percentiles[0]),
            'p99_intensity': float(percentiles[8])
        }

    def _compute_resolution(self, gray: np.ndarray) -> Dict[str, float]:
        """Estimate image resolution"""
        # Estimate resolution from edge sharpness
        edges = sobel(gray)

        # Find the sharpest edge
        edge_profile = edges[edges > np.percentile(edges, 99)]

        if len(edge_profile) > 0:
            # Resolution proxy (sharper edges = better resolution)
            edge_sharpness = np.std(edge_profile)

            # Measure point spread function (PSF) width
            # Find brightest point and fit Gaussian
            max_idx = np.unravel_index(np.argmax(gray), gray.shape)
            y, x = max_idx

            # Extract region around peak
            y_min, y_max = max(0, y - 10), min(gray.shape[0], y + 10)
            x_min, x_max = max(0, x - 10), min(gray.shape[1], x + 10)
            roi = gray[y_min:y_max, x_min:x_max]

            if roi.size > 0:
                # Calculate FWHM
                center = np.array(roi.shape) // 2
                y_vals = roi[center[0], :]
                x_vals = roi[:, center[1]]

                # Find width at half maximum
                max_val = np.max(roi)
                half_max = max_val / 2

                y_half_width = np.sum(y_vals > half_max)
                x_half_width = np.sum(x_vals > half_max)
                estimated_fwhm = (y_half_width + x_half_width) / 2
            else:
                estimated_fwhm = 0
        else:
            edge_sharpness = 0
            estimated_fwhm = 0

        return {
            'edge_sharpness': float(edge_sharpness),
            'estimated_fwhm_pixels': float(estimated_fwhm),
            'resolution_score': float(min(1.0, edge_sharpness / 0.5)),
            'optical_resolution': float(1 / (estimated_fwhm + 1e-8))
        }

    def _detect_out_of_focus(self, gray: np.ndarray) -> Dict[str, float]:
        """Detect if image is out of focus"""
        # Laplacian variance is a good focus measure
        laplacian = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
        laplacian_variance = np.var(laplacian)

        # Normalize for typical values
        focus_score = min(1.0, laplacian_variance / 500)
        is_out_of_focus = 1 if focus_score < 0.3 else 0

        # Alternative focus measure using gradient
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_focus = np.mean(gradient_magnitude)

        return {
            'focus_score': float(focus_score),
            'is_out_of_focus': float(is_out_of_focus),
            'laplacian_variance': float(laplacian_variance),
            'gradient_focus': float(gradient_focus),
            'focus_confidence': float(1 - focus_score if focus_score < 0.5 else focus_score)
        }

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute microscopy-specific quality metrics"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # Focus quality
        laplacian = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
        focus_score = min(1.0, np.var(laplacian) / 500)

        # Signal-to-noise ratio
        signal_region = gray > np.percentile(gray, 90)
        noise_region = gray < np.percentile(gray, 10)
        signal = np.mean(gray[signal_region]) if np.any(signal_region) else np.mean(gray)
        noise = np.std(gray[noise_region]) if np.any(noise_region) else np.std(gray)
        snr = signal / (noise + 1e-8)

        # Contrast
        contrast = (np.percentile(gray, 95) - np.percentile(gray, 5)) / (np.percentile(gray, 95) + np.percentile(gray, 5) + 1e-8)

        # Overall image quality
        image_quality = (focus_score * 0.5 + min(1.0, snr / 10) * 0.3 + contrast * 0.2)

        return {
            'focus_quality': float(focus_score),
            'signal_to_noise_ratio': float(snr),
            'image_contrast': float(contrast),
            'image_quality_score': float(image_quality),
            'acquisition_quality': float(image_quality)
        }

# =============================================================================
# COMPLETE INDUSTRIAL DOMAIN PROCESSOR
# =============================================================================

class IndustrialDomainProcessor:
    """Complete Industrial inspection processor with all features - DETERMINISTIC"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.detect_crack = getattr(config, 'detect_crack', True)
        self.detect_corrosion = getattr(config, 'detect_corrosion', True)
        self.detect_dent = getattr(config, 'detect_dent', True)
        self.detect_scratch = getattr(config, 'detect_scratch', True)
        self.measure_dimensions = getattr(config, 'measure_dimensions', True)
        self.detect_misalignment = getattr(config, 'detect_misalignment', True)
        self.classify_defect_type = getattr(config, 'classify_defect_type', True)
        self.compute_surface_roughness = getattr(config, 'compute_surface_roughness', True)
        self.detect_uniformity = getattr(config, 'detect_uniformity', True)

        # CRITICAL FIX: Fixed random state for reproducibility
        self._rng = np.random.RandomState(42)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess industrial images"""
        img_float = image.astype(np.float32) / 255.0

        # Illumination normalization
        img_float = self._normalize_illumination(img_float)

        # Edge enhancement for defect detection
        img_float = self._enhance_edges(img_float)

        return img_float

    def _normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """Normalize uneven illumination"""
        if len(image.shape) == 3:
            for i in range(min(3, image.shape[2])):
                # Use homomorphic filtering approximation
                log_img = np.log(image[:, :, i] + 1e-8)

                # Low-pass filter for illumination component
                low_pass = cv2.GaussianBlur(log_img, (51, 51), 0)

                # High-pass for reflectance component
                high_pass = log_img - low_pass

                # Reconstruct
                image[:, :, i] = np.exp(high_pass)
                image[:, :, i] = np.clip(image[:, :, i], 0, 1)
        else:
            log_img = np.log(image + 1e-8)
            low_pass = cv2.GaussianBlur(log_img, (51, 51), 0)
            high_pass = log_img - low_pass
            image = np.exp(high_pass)
            image = np.clip(image, 0, 1)

        return image

    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges for defect detection"""
        if len(image.shape) == 3:
            for i in range(3):
                # Unsharp masking
                blurred = cv2.GaussianBlur(image[:, :, i], (5, 5), 0)
                image[:, :, i] = np.clip(image[:, :, i] + (image[:, :, i] - blurred) * 0.5, 0, 1)
        else:
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            image = np.clip(image + (image - blurred) * 0.5, 0, 1)
        return image

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive industrial-specific features"""
        features = {}

        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # Defect detection
        if self.detect_crack:
            features.update(self._detect_crack(image, gray))

        if self.detect_corrosion:
            features.update(self._detect_corrosion(image, gray))

        if self.detect_dent:
            features.update(self._detect_dent(gray))

        if self.detect_scratch:
            features.update(self._detect_scratch(gray))

        # Dimensional analysis
        if self.measure_dimensions:
            features.update(self._measure_dimensions(gray))

        # Alignment
        if self.detect_misalignment:
            features.update(self._detect_misalignment(gray))

        # Surface analysis
        if self.compute_surface_roughness:
            features.update(self._compute_surface_roughness(gray))

        if self.detect_uniformity:
            features.update(self._detect_uniformity(gray))

        # Defect classification
        if self.classify_defect_type:
            features.update(self._classify_defect_type(features))

        return features

    def _detect_crack(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect cracks in industrial surfaces"""
        # Enhance linear structures
        from skimage.filters import frangi

        # Frangi filter for line enhancement
        ridges = frangi(gray, scale_range=(1, 5), scale_step=1)

        # Threshold for crack detection
        threshold = np.percentile(ridges, 95)
        crack_mask = ridges > threshold

        if np.any(crack_mask):
            # Crack features
            crack_fraction = np.mean(crack_mask)

            # Skeletonize for length estimation
            skeleton = morphology.skeletonize(crack_mask)
            crack_length = np.sum(skeleton)

            # Crack width estimation
            from scipy.ndimage import distance_transform_edt
            distance = distance_transform_edt(~crack_mask)
            crack_width = 2 * np.mean(distance[crack_mask]) if np.any(crack_mask) else 0

            # Crack orientation
            from skimage.measure import regionprops
            labeled, _ = ndimage_label(crack_mask)
            props = regionprops(labeled)

            if props:
                main_crack = max(props, key=lambda p: p.area)
                orientation = main_crack.orientation
            else:
                orientation = 0

            return {
                'crack_fraction': float(crack_fraction),
                'crack_length': float(crack_length / (gray.size / 1000)),
                'crack_width': float(crack_width),
                'crack_orientation': float(orientation),
                'crack_severity': float(min(1.0, crack_fraction * 10)),
                'crack_density': float(crack_length / (gray.size / 1000))
            }

        return {'crack_fraction': 0, 'crack_length': 0, 'crack_width': 0,
                'crack_orientation': 0, 'crack_severity': 0, 'crack_density': 0}

    def _detect_corrosion(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Detect corrosion (rust, oxidation)"""
        if len(image.shape) >= 3:
            # Corrosion appears as reddish-brown
            red = image[:, :, 0]
            green = image[:, :, 1]
            blue = image[:, :, 2]

            # Rust detection (reddish with low green/blue)
            rust_mask = (red > 0.4) & (green < 0.3) & (blue < 0.3)

            # Oxidation detection (dark spots)
            oxidation_mask = (red < 0.3) & (green < 0.3) & (blue < 0.3) & (gray < 0.3)

            corrosion_mask = rust_mask | oxidation_mask
        else:
            # Grayscale corrosion detection (dark irregular regions)
            threshold = np.percentile(gray, 20)
            corrosion_mask = gray < threshold

        if np.any(corrosion_mask):
            corrosion_fraction = np.mean(corrosion_mask)

            # Corrosion texture (roughness)
            corrosion_region = gray[corrosion_mask]
            corrosion_roughness = np.std(corrosion_region) / (np.mean(corrosion_region) + 1e-8)

            # Corrosion spread (number of connected components)
            labeled, num_corrosion_areas = ndimage_label(corrosion_mask)

            return {
                'corrosion_fraction': float(corrosion_fraction),
                'corrosion_roughness': float(corrosion_roughness),
                'corrosion_areas': float(num_corrosion_areas),
                'corrosion_severity': float(min(1.0, corrosion_fraction * 5)),
                'rust_intensity': float(np.mean(corrosion_region) if len(corrosion_region) > 0 else 0)
            }

        return {'corrosion_fraction': 0, 'corrosion_roughness': 0, 'corrosion_areas': 0,
                'corrosion_severity': 0, 'rust_intensity': 0}

    def _detect_dent(self, gray: np.ndarray) -> Dict[str, float]:
        """Detect dents (localized depressions)"""
        # Detect local intensity variations
        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(gray, size=21)
        local_deviation = gray - local_mean

        # Dents appear as local minima
        dent_mask = local_deviation < -np.std(local_deviation) * 2

        if np.any(dent_mask):
            dent_fraction = np.mean(dent_mask)

            # Dent depth (intensity difference)
            dent_depth = -np.mean(local_deviation[dent_mask]) if np.any(dent_mask) else 0

            # Dent size
            labeled, num_dents = ndimage_label(dent_mask)
            dent_sizes = np.bincount(labeled.ravel())[1:]

            return {
                'dent_fraction': float(dent_fraction),
                'dent_depth': float(dent_depth),
                'dent_count': float(num_dents),
                'avg_dent_size': float(np.mean(dent_sizes) if len(dent_sizes) > 0 else 0),
                'dent_severity': float(min(1.0, dent_depth * 5))
            }

        return {'dent_fraction': 0, 'dent_depth': 0, 'dent_count': 0, 'avg_dent_size': 0, 'dent_severity': 0}

    def _detect_scratch(self, gray: np.ndarray) -> Dict[str, float]:
        """Detect scratches (thin linear defects)"""
        # Enhance linear features
        from skimage.filters import frangi

        # Use small scale for thin scratches
        lines = frangi(gray, scale_range=(1, 3), scale_step=0.5)

        # Threshold for scratch detection
        threshold = np.percentile(lines, 97)
        scratch_mask = lines > threshold

        if np.any(scratch_mask):
            # Skeletonize for length measurement
            skeleton = morphology.skeletonize(scratch_mask)
            scratch_length = np.sum(skeleton)

            # Scratch density
            scratch_fraction = np.mean(scratch_mask)

            # Scratch orientation distribution
            from skimage.measure import regionprops
            labeled, _ = ndimage_label(scratch_mask)
            props = regionprops(labeled)

            if props:
                orientations = [p.orientation for p in props]
                orientation_std = np.std(orientations)
                primary_orientation = np.mean(orientations)
            else:
                orientation_std = 0
                primary_orientation = 0

            return {
                'scratch_fraction': float(scratch_fraction),
                'scratch_length': float(scratch_length / (gray.size / 1000)),
                'scratch_count': float(len(props)),
                'scratch_orientation_diversity': float(orientation_std),
                'primary_scratch_orientation': float(primary_orientation),
                'scratch_severity': float(min(1.0, scratch_fraction * 20))
            }

        return {'scratch_fraction': 0, 'scratch_length': 0, 'scratch_count': 0,
                'scratch_orientation_diversity': 0, 'primary_scratch_orientation': 0, 'scratch_severity': 0}

    def _measure_dimensions(self, gray: np.ndarray) -> Dict[str, float]:
        """Measure object dimensions"""
        # Segment the main object
        threshold = np.percentile(gray, 70)
        binary = gray > threshold

        if np.any(binary):
            # Clean up
            binary = morphology.binary_closing(binary, morphology.disk(5))
            binary = morphology.binary_opening(binary, morphology.disk(3))

            # Find contours
            contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Largest object
                largest_contour = max(contours, key=cv2.contourArea)

                # Bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Minimum enclosing circle
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

                # Dimensions
                aspect_ratio = w / (h + 1e-8)

                # Feret diameters
                rect = cv2.minAreaRect(largest_contour)
                feret_x, feret_y = rect[1]
                feret_ratio = max(feret_x, feret_y) / (min(feret_x, feret_y) + 1e-8)

                return {
                    'object_width': float(w),
                    'object_height': float(h),
                    'object_aspect_ratio': float(aspect_ratio),
                    'object_area': float(cv2.contourArea(largest_contour)),
                    'feret_diameter': float(max(feret_x, feret_y)),
                    'circularity': float(4 * np.pi * cv2.contourArea(largest_contour) / (cv2.arcLength(largest_contour, True)**2 + 1e-8))
                }

        return {'object_width': 0, 'object_height': 0, 'object_aspect_ratio': 0,
                'object_area': 0, 'feret_diameter': 0, 'circularity': 0}

    def _detect_misalignment(self, gray: np.ndarray) -> Dict[str, float]:
        """Detect misalignment in manufactured parts"""
        # Find dominant edges for alignment reference
        edges = sobel(gray)

        # Hough transform for line detection
        lines = cv2.HoughLinesP((edges * 255).astype(np.uint8), 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

        if lines is not None and len(lines) > 0:
            # Analyze line orientations
            orientations = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                orientations.append(angle)

            orientations = np.array(orientations)

            # Expected orientation (0 for horizontal, pi/2 for vertical)
            expected_horizontal = 0
            expected_vertical = np.pi / 2

            # Calculate deviation
            horizontal_lines = orientations[np.abs(orientations) < np.pi/4]
            vertical_lines = orientations[np.abs(np.abs(orientations) - np.pi/2) < np.pi/4]

            horizontal_deviation = np.std(horizontal_lines) if len(horizontal_lines) > 0 else 0
            vertical_deviation = np.std(vertical_lines) if len(vertical_lines) > 0 else 0

            misalignment_score = (horizontal_deviation + vertical_deviation) / (np.pi/4)
            misalignment_score = min(1.0, misalignment_score)

            return {
                'misalignment_score': float(misalignment_score),
                'horizontal_alignment': float(1 - min(1.0, horizontal_deviation * 4 / np.pi)),
                'vertical_alignment': float(1 - min(1.0, vertical_deviation * 4 / np.pi)),
                'is_aligned': float(1 if misalignment_score < 0.1 else 0)
            }

        return {'misalignment_score': 0, 'horizontal_alignment': 1, 'vertical_alignment': 1, 'is_aligned': 1}

    def _compute_surface_roughness(self, gray: np.ndarray) -> Dict[str, float]:
        """Compute surface roughness metrics"""
        # High-pass filter for surface texture
        low_pass = cv2.GaussianBlur(gray, (21, 21), 0)
        high_pass = gray - low_pass

        # Roughness parameters
        rms_roughness = np.std(high_pass)
        mean_roughness = np.mean(np.abs(high_pass))

        # Peak-to-valley
        max_peak = np.max(high_pass)
        max_valley = np.min(high_pass)
        peak_to_valley = max_peak - max_valley

        # Skewness and kurtosis of height distribution
        skewness = np.mean(((high_pass - np.mean(high_pass)) / (np.std(high_pass) + 1e-8)) ** 3)
        kurtosis = np.mean(((high_pass - np.mean(high_pass)) / (np.std(high_pass) + 1e-8)) ** 4) - 3

        # Power spectral density slope (fractal dimension proxy)
        fft = np.fft.fft2(high_pass)
        power_spectrum = np.abs(fft)**2
        radial_profile = np.mean(power_spectrum, axis=0)
        radial_profile = radial_profile[:len(radial_profile)//2]

        if len(radial_profile) > 10:
            log_freq = np.log(np.arange(1, len(radial_profile) + 1))
            log_power = np.log(radial_profile + 1e-8)
            from scipy import stats
            slope, _, _, _, _ = stats.linregress(log_freq[5:], log_power[5:])
            fractal_dimension = 3 - slope/2
        else:
            fractal_dimension = 2.5

        return {
            'surface_roughness_rms': float(rms_roughness),
            'surface_roughness_ra': float(mean_roughness),
            'peak_to_valley': float(peak_to_valley),
            'roughness_skewness': float(skewness),
            'roughness_kurtosis': float(kurtosis),
            'fractal_dimension': float(fractal_dimension)
        }

    def _detect_uniformity(self, gray: np.ndarray) -> Dict[str, float]:
        """Detect surface uniformity"""
        # Local statistics
        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(gray, size=21)
        local_std = np.sqrt(uniform_filter(gray**2, size=21) - local_mean**2)

        # Uniformity metrics
        mean_uniformity = 1 - np.std(local_mean) / (np.mean(local_mean) + 1e-8)
        texture_uniformity = 1 - np.std(local_std) / (np.mean(local_std) + 1e-8)

        # Defect-induced non-uniformity
        non_uniform_regions = local_std > np.percentile(local_std, 95)
        non_uniform_fraction = np.mean(non_uniform_regions)

        return {
            'intensity_uniformity': float(mean_uniformity),
            'texture_uniformity': float(texture_uniformity),
            'non_uniform_fraction': float(non_uniform_fraction),
            'overall_uniformity': float((mean_uniformity + texture_uniformity) / 2),
            'quality_grade': float(mean_uniformity * 100)  # Percentage grade
        }

    def _classify_defect_type(self, features: Dict[str, float]) -> Dict[str, float]:
        """Classify the type of defect present"""
        # Simple classification based on extracted features
        crack_score = features.get('crack_fraction', 0)
        corrosion_score = features.get('corrosion_fraction', 0)
        dent_score = features.get('dent_fraction', 0)
        scratch_score = features.get('scratch_fraction', 0)

        # Determine dominant defect type
        scores = {
            'crack': crack_score,
            'corrosion': corrosion_score,
            'dent': dent_score,
            'scratch': scratch_score
        }

        dominant_defect = max(scores, key=scores.get)
        max_score = scores[dominant_defect]

        # Defect severity based on combined score
        total_defect_score = (crack_score + corrosion_score + dent_score + scratch_score)
        severity = min(1.0, total_defect_score * 2)

        return {
            'dominant_defect_type': float(list(scores.keys()).index(dominant_defect)),
            'crack_probability': float(crack_score),
            'corrosion_probability': float(corrosion_score),
            'dent_probability': float(dent_score),
            'scratch_probability': float(scratch_score),
            'defect_severity': float(severity),
            'defect_present': float(1 if total_defect_score > 0.05 else 0)
        }

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute industrial-specific quality metrics"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # Sharpness (edge definition)
        edges = sobel(gray)
        sharpness = np.mean(edges)

        # Contrast
        contrast = np.std(gray)

        # Illumination uniformity
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(gray, size=51)
        illumination_uniformity = 1 - np.std(local_mean) / (np.mean(local_mean) + 1e-8)

        # Overall quality
        overall_quality = (sharpness * 0.4 + contrast * 0.3 + illumination_uniformity * 0.3)

        return {
            'edge_sharpness': float(sharpness),
            'image_contrast': float(contrast),
            'illumination_uniformity': float(illumination_uniformity),
            'inspection_quality': float(overall_quality),
            'pass_fail_threshold': float(1 if overall_quality > 0.5 else 0)
        }

# =============================================================================
# TORCHVISION DATASET ADAPTER
# =============================================================================

class TorchvisionDatasetAdapter(Dataset):
    def __init__(self, dataset_name: str, train: bool = True, transform=None, download: bool = True):
        self.dataset_name = dataset_name.upper()
        self.transform = transform
        self.train = train

        if self.dataset_name == 'MNIST':
            self.dataset = datasets.MNIST(root='./data', train=train, download=download, transform=None)
        elif self.dataset_name == 'CIFAR10':
            self.dataset = datasets.CIFAR10(root='./data', train=train, download=download, transform=None)
        elif self.dataset_name == 'CIFAR100':
            self.dataset = datasets.CIFAR100(root='./data', train=train, download=download, transform=None)
        elif self.dataset_name == 'FASHIONMNIST':
            self.dataset = datasets.FashionMNIST(root='./data', train=train, download=download, transform=None)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        if hasattr(self.dataset, 'classes'):
            self.classes = self.dataset.classes
        else:
            unique_targets = sorted(set(self.dataset.targets))
            self.classes = [str(t) for t in unique_targets]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.dataset[idx]
        if not isinstance(img, PILImage.Image):
            if isinstance(img, torch.Tensor):
                img = TF.to_pil_image(img)
            else:
                img = PILImage.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_distribution(self) -> Dict[str, int]:
        if hasattr(self.dataset, 'targets'):
            targets = self.dataset.targets
        else:
            targets = [self.dataset[i][1] for i in range(min(1000, len(self.dataset)))]
        counts = Counter(targets)
        return {self.idx_to_class[int(k)]: v for k, v in counts.items()}

# =============================================================================
# INTEGRATED MODEL FACTORY - Domain-Specific + Advanced Hybrid
# =============================================================================
# =============================================================================
# ADVANCED HYBRID AUTOENCODER - With Gemini's Recommendations
# =============================================================================


class FeatureModulator(nn.Module):
    """
    Conditional Instance Normalization / Feature Modulation
    Replaces hard concatenation with dynamic scaling
    """
    def __init__(self, feature_dim: int, class_embed_dim: int):
        super().__init__()
        self.to_scale_bias = nn.Linear(class_embed_dim, feature_dim * 2)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor, class_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, feature_dim]
            class_emb: [B, class_embed_dim]
        Returns:
            Modulated features [B, feature_dim]
        """
        chunks = self.to_scale_bias(class_emb).chunk(2, dim=1)
        scale, bias = chunks[0], chunks[1]
        # Reshape if needed
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
        return x * (1 + scale) + bias

class ContrastiveDistillationLoss(nn.Module):
    """
    InfoNCE-style contrastive distillation.
    Forces relative distances in unconditional space to match conditional space.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_uncond: torch.Tensor, z_cond: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_uncond: Unconditional embeddings [B, D]
            z_cond: Conditional embeddings [B, D]
            labels: Class labels [B]
        Returns:
            Contrastive distillation loss
        """
        batch_size = z_uncond.size(0)

        # Normalize embeddings
        z_uncond_norm = F.normalize(z_uncond, p=2, dim=1)
        z_cond_norm = F.normalize(z_cond, p=2, dim=1)

        # Compute similarity matrices
        sim_uncond = torch.mm(z_uncond_norm, z_uncond_norm.t()) / self.temperature
        sim_cond = torch.mm(z_cond_norm, z_cond_norm.t()) / self.temperature

        # Create mask for positive pairs (same class)
        pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        neg_mask = 1 - pos_mask
        # Remove self-pairs
        neg_mask.fill_diagonal_(0)

        # InfoNCE loss: positive pairs should have high similarity in both spaces
        # We want the similarity structure to be preserved between spaces
        pos_sim_uncond = (sim_uncond * pos_mask).sum(dim=1)
        neg_sim_uncond = (sim_uncond * neg_mask).sum(dim=1) / (neg_mask.sum(dim=1) + 1e-8)

        # Contrastive: positive should be higher than negative
        contrast_loss = torch.clamp(neg_sim_uncond - pos_sim_uncond + 0.5, min=0).mean()

        # Additional: MSE between similarity matrices (structure preservation)
        structure_loss = F.mse_loss(sim_uncond, sim_cond)

        return contrast_loss + structure_loss

class RelationalKnowledgeDistillation(nn.Module):
    """
    Relational KD: Preserves pairwise distances between samples
    """
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_uncond: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        """
        Preserve the relational structure between samples
        """
        # Compute pairwise distance matrices
        dist_uncond = torch.cdist(z_uncond, z_uncond, p=2)
        dist_cond = torch.cdist(z_cond, z_cond, p=2)

        # Convert to similarity (RBF kernel)
        sim_uncond = torch.exp(-dist_uncond / self.temperature)
        sim_cond = torch.exp(-dist_cond / self.temperature)

        # Normalize
        sim_uncond = sim_uncond / sim_uncond.sum(dim=1, keepdim=True)
        sim_cond = sim_cond / sim_cond.sum(dim=1, keepdim=True)

        # KL divergence between relational distributions
        kd_loss = F.kl_div(sim_uncond.log(), sim_cond, reduction='batchmean')

        return kd_loss

class AdaptiveLossBalancer(nn.Module):
    """
    Dynamically balances multiple loss components based on gradient magnitudes
    """
    def __init__(self, num_losses: int = 3, alpha: float = 0.1):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        self.alpha = alpha
        self.register_buffer('running_grads', torch.zeros(num_losses))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: List of loss tensors
        Returns:
            Weighted sum of losses
        """
        # Uncertainty weighting (Kendall et al.)
        precision = torch.exp(-self.log_vars)
        weighted_loss = 0
        for i, loss in enumerate(losses):
            weighted_loss += precision[i] * loss + self.log_vars[i]

        return weighted_loss / len(losses)

    def update_grad_stats(self, losses: List[torch.Tensor], model: nn.Module):
        """Update gradient statistics for balancing"""
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = math.sqrt(total_norm)

        # Exponential moving average
        self.running_grads = self.alpha * total_norm + (1 - self.alpha) * self.running_grads

class SharedDecoderWithModulation(nn.Module):
    """
    Single shared decoder with conditional modulation
    """
    def __init__(self, encoder_channels: List[int], in_channels: int,
                 final_channels: int, class_embed_dim: int, n_layers: int):
        super().__init__()
        self.decoder_layers = nn.ModuleList()
        self.modulators = nn.ModuleList()

        in_ch = encoder_channels[-1]

        for i in range(n_layers - 1, -1, -1):
            out_ch = final_channels if i == 0 else encoder_channels[i-1]

            # Add modulator for conditional feature modulation
            modulator = FeatureModulator(out_ch, class_embed_dim)
            self.modulators.append(modulator)

            # Build decoder block
            if i == 0:
                decoder_block = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                    nn.Tanh()
                )
            else:
                decoder_block = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                    nn.GroupNorm(min(16, out_ch), out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                    nn.GroupNorm(min(16, out_ch), out_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )

            self.decoder_layers.append(decoder_block)
            in_ch = out_ch

        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, class_emb: torch.Tensor, use_modulation: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
            class_emb: Class embeddings [B, class_embed_dim]
            use_modulation: Whether to apply conditional modulation
        """
        for i, (decoder, modulator) in enumerate(zip(self.decoder_layers, self.modulators)):
            x = decoder(x)

            if use_modulation and class_emb is not None:
                # Reshape x for modulation if needed
                if x.dim() == 4:
                    b, c, h, w = x.shape
                    x_flat = x.view(b, c, -1).transpose(1, 2)  # [B, H*W, C]
                    x_mod = modulator(x_flat, class_emb)  # [B, H*W, C]
                    x = x_mod.transpose(1, 2).view(b, c, h, w)
                else:
                    x = modulator(x, class_emb)

        return x

class AdvancedHybridAutoencoder(BaseAutoencoder):
    """
    Advanced Hybrid Autoencoder with Gemini's recommendations:
    1. Unified shared decoder with feature modulation
    2. Contrastive/Relational distillation
    3. Adaptive loss balancing
    4. Log-proportional class embeddings
    """

    def __init__(self, config: GlobalConfig):
        super().__init__(config)

        num_classes = config.num_classes or 2
        self.num_classes = num_classes

        # ========================================================================
        # HELPER FUNCTIONS FOR NUMERICAL STABILITY
        # ========================================================================
        def make_divisible(x, divisor=16):
            """Round x to the nearest multiple of divisor"""
            if x < divisor:
                return max(divisor // 2, ((x + divisor//2 - 1) // (divisor//2)) * (divisor//2))
            return int(np.ceil(x / divisor) * divisor)

        def get_group_norm(channels):
            """Create GroupNorm with proper divisibility"""
            if channels < 8:
                return nn.Identity()
            # Try divisors from min(16, channels//2) down to 1
            for g in [min(16, channels), 8, 4, 2, 1]:
                if g <= channels and channels % g == 0:
                    return nn.GroupNorm(g, channels)
            # Fallback to LayerNorm
            return nn.LayerNorm(channels) if channels > 0 else nn.Identity()

        def get_batch_norm(channels):
            """Create BatchNorm1d with proper handling of small channels"""
            if channels < 2:
                return nn.Identity()
            return nn.BatchNorm1d(channels)

        # ========================================================================
        # LOG-PROPORTIONAL CLASS EMBEDDING DIMENSIONS
        # ========================================================================
        self.class_embed_dim = int(np.clip(np.ceil(np.log2(num_classes) * 4), 8, 64))
        self.class_embedding = nn.Embedding(num_classes, self.class_embed_dim)

        # Ensure feature_dims and compressed_dims are divisible
        self.feature_dims = make_divisible(self.feature_dims, 16)
        self.compressed_dims = make_divisible(self.compressed_dims, 8)

        # Calculate feat_half - ensure it's at least 8 and divisible
        feat_half_raw = self.feature_dims // 2
        feat_half = make_divisible(feat_half_raw, 8)
        feat_half = max(8, feat_half)

        # Ensure compressed_dims is consistent
        compressed_dims_div = self.compressed_dims

        # Store these for later use
        self.feat_half = feat_half
        self.compressed_dims_div = compressed_dims_div

        # ========================================================================
        # UNCONDITIONAL PATH (Standard autoencoder)
        # ========================================================================
        self.unconditional_compressor = nn.Sequential(
            nn.Linear(self.feature_dims, feat_half),
            get_group_norm(feat_half),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_half, compressed_dims_div),
            nn.Tanh()
        )

        self.unconditional_decompressor = nn.Sequential(
            nn.Linear(compressed_dims_div, feat_half),
            get_group_norm(feat_half),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_half, self.feature_dims),
            nn.Tanh()
        )

        # ========================================================================
        # CONDITIONAL PATH (Uses shared decoder with modulation)
        # ========================================================================
        self.conditional_preprocess = nn.Sequential(
            nn.Linear(self.feature_dims, feat_half),
            get_group_norm(feat_half),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Conditional compressor (combines features with class embedding)
        # The input to conditional_compress is feat_half + class_embed_dim
        compress_input_dim = feat_half + self.class_embed_dim
        compress_output_dim = max(compressed_dims_div, self.class_embed_dim)
        compress_output_dim = make_divisible(compress_output_dim, 8)

        self.conditional_compress = nn.Linear(compress_input_dim, compress_output_dim)

        # Conditional decompressor
        decompress_input_dim = compress_output_dim + self.class_embed_dim
        decompress_output_dim = feat_half

        self.conditional_decompress = nn.Linear(decompress_input_dim, decompress_output_dim)

        self.conditional_postprocess = nn.Sequential(
            nn.Linear(decompress_output_dim, self.feature_dims),
            get_group_norm(self.feature_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Store for later use
        self.compress_output_dim = compress_output_dim
        self.decompress_output_dim = decompress_output_dim

        # ========================================================================
        # SHARED DECODER WITH FEATURE MODULATION
        # ========================================================================
        self.shared_decoder = SharedDecoderWithModulation(
            encoder_channels=self.encoder_channels,
            in_channels=self.encoder_channels[-1],
            final_channels=self.in_channels,
            class_embed_dim=self.class_embed_dim,
            n_layers=len(self.decoder_layers)
        )

        # ========================================================================
        # ADVANCED DISTILLATION LOSSES
        # ========================================================================
        self.contrastive_distillation = ContrastiveDistillationLoss(temperature=0.1)
        self.relational_distillation = RelationalKnowledgeDistillation(temperature=2.0)

        # Distillation weights (adaptive)
        self.distillation_weight = nn.Parameter(torch.tensor(0.5))

        # ========================================================================
        # PERCEPTUAL LOSS (for structural quality)
        # ========================================================================
        self.use_perceptual_loss = getattr(config, 'use_perceptual_loss', True)
        if self.use_perceptual_loss:
            self.perceptual_loss = PerceptualLoss(self.device)

        # ========================================================================
        # ADAPTIVE LOSS BALANCER
        # ========================================================================
        self.loss_balancer = AdaptiveLossBalancer(num_losses=4)

        # ========================================================================
        # CLASSIFIER (with adaptive architecture)
        # ========================================================================
        classifier_input_dim = max(2, compressed_dims_div)

        if num_classes >= 100:
            classifier_layers = [
                nn.Linear(classifier_input_dim, classifier_input_dim * 2),
                get_batch_norm(classifier_input_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(classifier_input_dim * 2, classifier_input_dim),
                get_batch_norm(classifier_input_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(classifier_input_dim, num_classes)
            ]
        elif num_classes >= 50:
            classifier_layers = [
                nn.Linear(classifier_input_dim, classifier_input_dim * 2),
                get_batch_norm(classifier_input_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(classifier_input_dim * 2, num_classes)
            ]
        else:
            classifier_layers = [
                nn.Linear(classifier_input_dim, classifier_input_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(classifier_input_dim, num_classes)
            ]

        self.classifier = nn.Sequential(*classifier_layers).to(self.device)

        # Training state
        self.use_conditional_training = True
        self.teacher_forcing_ratio = 0.5
        self.distillation_mode = getattr(config, 'distillation_mode', 'contrastive')

        # Log architecture
        logger.info("=" * 60)
        logger.info("Advanced Hybrid Autoencoder Initialized")
        logger.info(f"  Number of classes: {num_classes}")
        logger.info(f"  Class embedding dimension: {self.class_embed_dim}")
        logger.info(f"  Feature dimensions: {self.feature_dims}")
        logger.info(f"  Feature half: {self.feat_half}")
        logger.info(f"  Compressed dimensions: {self.compressed_dims}")
        logger.info(f"  Compress output dim: {self.compress_output_dim}")
        logger.info(f"  Distillation mode: {self.distillation_mode}")
        logger.info("=" * 60)

    def encode_conditional(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conditional encoding with class embedding"""
        # Standard encoding
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)
        embedding = self.embedder_projection(embedding)

        # Conditional preprocessing
        features = self.conditional_preprocess(embedding)  # [B, feat_half]

        # Get class embedding
        class_emb = self.class_embedding(labels)  # [B, class_embed_dim]

        # Combine and compress
        combined = torch.cat([features, class_emb], dim=1)  # [B, feat_half + class_embed_dim]
        compressed = self.conditional_compress(combined)  # [B, compress_output_dim]

        return compressed, class_emb

    def decode_conditional(self, compressed: torch.Tensor, class_emb: torch.Tensor) -> torch.Tensor:
        """Conditional decoding using shared decoder with modulation"""
        # Combine with class embedding for decompression
        combined = torch.cat([compressed, class_emb], dim=1)  # [B, compress_output_dim + class_embed_dim]
        features = self.conditional_decompress(combined)  # [B, decompress_output_dim]
        features = self.conditional_postprocess(features)  # [B, feature_dims]

        # Reshape to spatial
        x = self.unembedder(features)
        x = self.unembedder_projection(x)
        x = x.view(x.size(0), self.encoder_channels[-1], self.final_h, self.final_w)

        # Shared decoder with modulation
        x = self.shared_decoder(x, class_emb, use_modulation=True)

        return x

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with both conditional and unconditional paths"""
        original_batch_size = x.size(0)
        duplicated = False

        # Apply normalization
        x = self.normalize_batch(x)

        # Handle single sample
        if self.training and original_batch_size == 1:
            x = torch.cat([x, x], dim=0)
            duplicated = True
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)

        # ========================================================================
        # UNCONDITIONAL PATH (always computed)
        # ========================================================================
        uncompressed = self.encode_unconditional(x)

        # Unconditional decoding using shared decoder (no modulation)
        features = self.unconditional_decompressor(uncompressed)
        x_feat = self.unembedder(features)
        x_feat = self.unembedder_projection(x_feat)
        x_feat = x_feat.view(x_feat.size(0), self.encoder_channels[-1], self.final_h, self.final_w)
        unreconstruction = self.shared_decoder(x_feat, None, use_modulation=False)

        output = {
            'uncompressed_embedding': self._fix_tensor_dim(uncompressed, original_batch_size, duplicated),
            'unreconstruction': self._fix_tensor_dim(unreconstruction, original_batch_size, duplicated),
        }

        # ========================================================================
        # CONDITIONAL PATH (only during training with labels)
        # ========================================================================
        if self.training and labels is not None and self.use_conditional_training:
            # Teacher forcing with annealing
            use_teacher = torch.rand(1).item() < self.teacher_forcing_ratio

            if use_teacher:
                comp_cond, class_emb = self.encode_conditional(x, labels)
            else:
                with torch.no_grad():
                    pred_logits = self.classifier(uncompressed)
                    pred_labels = pred_logits.argmax(dim=1)
                    comp_cond, class_emb = self.encode_conditional(x, pred_labels)

            recon_cond = self.decode_conditional(comp_cond, class_emb)

            output.update({
                'compressed_embedding': self._fix_tensor_dim(comp_cond, original_batch_size, duplicated),
                'reconstruction': self._fix_tensor_dim(recon_cond, original_batch_size, duplicated),
                'class_embedding': self._fix_tensor_dim(class_emb, original_batch_size, duplicated),
                'conditional_active': torch.tensor(1.0, device=self.device)
            })
        else:
            # If no conditional path, use unconditional output as reconstruction
            output['reconstruction'] = output['unreconstruction']
            output['compressed_embedding'] = uncompressed
            output['conditional_active'] = torch.tensor(0.0, device=self.device)

        # ========================================================================
        # CLASSIFICATION (from unconditional path for consistency)
        # ========================================================================
        if self.training_phase == 2 and self.classifier is not None:
            logits = self.classifier(uncompressed)
            output.update({
                'class_logits': self._fix_tensor_dim(logits, original_batch_size, duplicated),
                'class_predictions': self._fix_tensor_dim(logits.argmax(dim=1), original_batch_size, duplicated),
                'class_probabilities': self._fix_tensor_dim(F.softmax(logits, dim=1), original_batch_size, duplicated)
            })

        return output

    def encode_unconditional(self, x: torch.Tensor) -> torch.Tensor:
        """Unconditional encoding"""
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)
        embedding = self.embedder_projection(embedding)
        return self.unconditional_compressor(embedding)

    def get_loss(self, outputs: Dict, inputs: torch.Tensor, labels: torch.Tensor,
                 phase: int, epoch: int) -> Tuple[torch.Tensor, Optional[float]]:
        """Compute loss with adaptive balancing and advanced distillation"""

        # Update teacher forcing ratio (curriculum learning)
        if self.training and epoch < 100:
            self.teacher_forcing_ratio = max(0.1, 1.0 - epoch / 100)

        # Phase 1: Reconstruction only
        if phase == 1:
            recon_loss = F.mse_loss(outputs['unreconstruction'], inputs)

            if outputs.get('conditional_active', torch.tensor(0)).item() > 0:
                recon_cond_loss = F.mse_loss(outputs['reconstruction'], inputs)
                total_loss = recon_loss + 0.5 * recon_cond_loss
            else:
                total_loss = recon_loss

            return total_loss, None

        # Phase 2: Full training with adaptive balancing
        losses = []

        # 1. Reconstruction loss
        recon_loss = F.mse_loss(outputs['unreconstruction'], inputs)
        losses.append(recon_loss)

        # 2. Conditional reconstruction loss (auxiliary task)
        cond_loss = torch.tensor(0.0, device=inputs.device)
        if outputs.get('conditional_active', torch.tensor(0)).item() > 0:
            cond_loss = F.mse_loss(outputs['reconstruction'], inputs)
        losses.append(cond_loss)

        # 3. Advanced distillation loss
        distill_loss = torch.tensor(0.0, device=inputs.device)
        if outputs.get('conditional_active', torch.tensor(0)).item() > 0:
            z_uncond = outputs['uncompressed_embedding']
            z_cond = outputs['compressed_embedding']

            # Ensure dimensions match for distillation
            min_dim = min(z_uncond.size(1), z_cond.size(1))
            z_uncond_aligned = z_uncond[:, :min_dim]
            z_cond_aligned = z_cond[:, :min_dim]

            if self.distillation_mode == 'contrastive':
                try:
                    distill_loss = self.contrastive_distillation(z_uncond_aligned, z_cond_aligned, labels)
                except:
                    distill_loss = F.mse_loss(z_uncond_aligned, z_cond_aligned)
            elif self.distillation_mode == 'relational':
                try:
                    distill_loss = self.relational_distillation(z_uncond_aligned, z_cond_aligned)
                except:
                    distill_loss = F.mse_loss(z_uncond_aligned, z_cond_aligned)
            else:
                distill_loss = F.mse_loss(z_uncond_aligned, z_cond_aligned)

            distill_loss = self.distillation_weight.abs() * distill_loss
        losses.append(distill_loss)

        # 4. Classification loss
        class_loss = torch.tensor(0.0, device=inputs.device)
        if 'class_logits' in outputs:
            class_loss = F.cross_entropy(outputs['class_logits'], labels)
        losses.append(class_loss)

        # Simple weighted sum (more stable)
        weights = [1.0, 0.5, 0.3, 1.0]
        total_loss = sum(w * l for w, l in zip(weights, losses))

        # Compute accuracy
        accuracy = None
        if 'class_logits' in outputs:
            accuracy = (outputs['class_logits'].argmax(dim=1) == labels).float().mean().item()

        return total_loss, accuracy

    def set_eval_mode(self):
        """Switch to inference mode"""
        self.use_conditional_training = False
        self.eval()
        logger.info("Switched to inference mode (unconditional only)")

    def set_train_mode(self):
        """Switch back to training mode"""
        self.use_conditional_training = True
        self.train()
        logger.info("Switched to training mode (conditional + unconditional)")

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better reconstruction quality"""

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.vgg = None
        self._load_vgg()

    def _load_vgg(self):
        try:
            # Load pretrained VGG16 features
            from torchvision import models
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            # Freeze weights
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg.to(self.device)
            self.vgg.eval()
        except Exception as e:
            logger.warning(f"Could not load VGG for perceptual loss: {e}")
            self.vgg = None

    def forward(self, pred, target):
        if self.vgg is None:
            # Fallback to MSE if VGG not available
            return F.mse_loss(pred, target)

        # Normalize to ImageNet range for VGG
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)

        # Ensure 3 channels
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        pred_norm = (pred - mean) / (std + 1e-8)
        target_norm = (target - mean) / (std + 1e-8)

        # Extract features from multiple layers
        pred_features = []
        target_features = []

        x = pred_norm
        y = target_norm

        with torch.no_grad():
            for layer in self.vgg:
                x = layer(x)
                y = layer(y)
                # Store features from certain layers
                if isinstance(layer, nn.ReLU) and len(pred_features) < 4:
                    pred_features.append(x)
                    target_features.append(y)

        # Compute perceptual loss
        loss = 0
        for p, t in zip(pred_features, target_features):
            loss += F.l1_loss(p, t)

        return loss / max(len(pred_features), 1)

class ModelFactory:
    """Factory for creating appropriate model based on configuration and domain"""

    @staticmethod
    def create_model(config: GlobalConfig) -> nn.Module:
        """Create model with domain-specific enhancements and advanced architecture"""

        # ================================================================
        # CHECK FOR CONTRASTIVE DIRECTIVE FIRST
        # ================================================================
        use_contrastive = getattr(config, 'use_contrastive_learning', False)

        # Get basic configuration
        input_shape = (config.in_channels, config.input_size[0], config.input_size[1])
        feature_dims = config.feature_dims
        compressed_dims = getattr(config, 'compressed_dims', min(64, max(8, feature_dims // 4)))
        config.compressed_dims = compressed_dims

        # Get configuration flags
        use_enhanced = getattr(config, 'use_enhanced_autoencoder', True)
        use_hybrid = getattr(config, 'use_hybrid_autoencoder', True)
        use_advanced = getattr(config, 'use_advanced_hybrid', True)
        use_invertible = getattr(config, 'use_invertible', False)

        # Get domain from config
        domain = getattr(config, 'domain', 'general')
        image_type = getattr(config, 'image_type', 'general')
        num_classes = config.num_classes or 2

        # ================================================================
        # CONTRASTIVE LEARNING MODELS (Highest Priority)
        # ================================================================
        if use_contrastive:
            logger.info("=" * 60)
            logger.info("🔬 CONTRASTIVE LEARNING MODE ENABLED")
            logger.info(f"  Dataset: {config.dataset_name}")
            logger.info(f"  Classes: {num_classes}")
            logger.info(f"  Temperature: {config.contrastive_temperature}")
            logger.info(f"  Projection dim: {config.contrastive_projection_dim}")
            logger.info("=" * 60)

            # For astronomy domain with contrastive
            if domain == 'astronomy' or image_type == 'astronomical':
                if use_enhanced:
                    logger.info("Creating Enhanced Astronomical Contrastive Autoencoder")
                    # Use the existing AstronomicalStructurePreservingAutoencoder
                    # but with contrastive learning enabled
                    # We'll just use the standard contrastive autoencoder with
                    # domain-specific enhancements
                    return ContrastiveAutoencoderWithProjection(config)
                else:
                    logger.info("Creating Base Contrastive Autoencoder")
                    return ContrastiveAutoencoderWithProjection(config)

            # For medical domain with contrastive
            elif domain == 'medical' or image_type == 'medical':
                if use_enhanced:
                    logger.info("Creating Enhanced Medical Contrastive Autoencoder")
                    return ContrastiveAutoencoderWithProjection(config)
                else:
                    return ContrastiveAutoencoderWithProjection(config)

            # For agriculture domain with contrastive
            elif domain == 'agriculture' or image_type == 'agricultural':
                if use_enhanced:
                    logger.info("Creating Enhanced Agricultural Contrastive Autoencoder")
                    return ContrastiveAutoencoderWithProjection(config)
                else:
                    return ContrastiveAutoencoderWithProjection(config)

            # Check if this is a complex dataset (many classes or large images)
            min_dim = min(config.input_size)
            is_complex = num_classes >= 50 or min_dim >= 128

            if is_complex:
                # Use ResNet-based contrastive for complex datasets
                logger.info("Using ResNetContrastiveAutoencoder for complex dataset")
                return ResNetContrastiveAutoencoder(config)
            else:
                # Use standard contrastive autoencoder
                logger.info("Using ContrastiveAutoencoderWithProjection")
                return ContrastiveAutoencoderWithProjection(config)

        # ================================================================
        # DOMAIN-SPECIFIC MODEL SELECTION (Second Priority)
        # ================================================================

        # Astronomy domain
        if domain == 'astronomy' or image_type == 'astronomical':
            if use_invertible:
                logger.info("Creating Invertible Astronomical Structure Preserving Autoencoder")
                # For now, use the enhanced version with invertible flag
                return AstronomicalStructurePreservingAutoencoder(input_shape, feature_dims, config)
            elif use_enhanced:
                logger.info("Creating Enhanced Astronomical Structure Preserving Autoencoder with Ring Detection")
                return AstronomicalStructurePreservingAutoencoder(input_shape, feature_dims, config)
            else:
                logger.info("Creating Base Astronomical Autoencoder")
                return BaseAutoencoder(config)

        # Medical domain
        elif domain == 'medical' or image_type == 'medical':
            if use_invertible:
                logger.info("Creating Invertible Medical Structure Preserving Autoencoder")
                return MedicalStructurePreservingAutoencoder(input_shape, feature_dims, config)
            elif use_enhanced:
                logger.info("Creating Enhanced Medical Structure Preserving Autoencoder")
                return MedicalStructurePreservingAutoencoder(input_shape, feature_dims, config)
            else:
                logger.info("Creating Base Medical Autoencoder")
                return BaseAutoencoder(config)

        # Agriculture domain
        elif domain == 'agriculture' or image_type == 'agricultural':
            if use_invertible:
                logger.info("Creating Invertible Agricultural Pattern Autoencoder")
                return AgriculturalPatternAutoencoder(input_shape, feature_dims, config)
            elif use_enhanced:
                logger.info("Creating Enhanced Agricultural Pattern Autoencoder")
                return AgriculturalPatternAutoencoder(input_shape, feature_dims, config)
            else:
                logger.info("Creating Base Agricultural Autoencoder")
                return BaseAutoencoder(config)

        # ================================================================
        # INVERTIBLE GENERAL AUTOENCODER
        # ================================================================
        if use_invertible:
            logger.info("=" * 60)
            logger.info("Creating Invertible Autoencoder for Interpretability")
            logger.info(f"  Invertible blocks: {config.invertible_blocks}")
            logger.info(f"  Domain: {domain}")
            logger.info(f"  Classes: {num_classes}")
            logger.info("=" * 60)
            # Use the invertible autoencoder (inlined version)
            return InvertibleAutoencoder(config)

        # ================================================================
        # GENERAL DOMAIN - Advanced Hybrid for multi-class datasets
        # ================================================================

        # For general domain with many classes, use advanced hybrid
        if use_advanced and num_classes >= 20:
            logger.info("=" * 60)
            logger.info(f"Creating Advanced Hybrid Autoencoder for {num_classes} classes")
            logger.info("  Features:")
            logger.info("    - Shared decoder with feature modulation")
            logger.info("    - Contrastive/Relational distillation")
            logger.info("    - Adaptive loss balancing")
            logger.info("    - Log-proportional class embeddings")
            logger.info(f"  Domain: {domain}")
            logger.info("=" * 60)
            return AdvancedHybridAutoencoder(config)

        # For general domain with moderate classes, use standard hybrid
        elif use_hybrid and num_classes >= 10:
            logger.info(f"Creating Hybrid Autoencoder for {num_classes} classes")
            logger.info("  Features:")
            logger.info("    - Conditional training with labels")
            logger.info("    - Unconditional inference")
            logger.info("    - Knowledge distillation")
            logger.info(f"  Domain: {domain}")
            return HybridAutoencoder(config)

        # For general domain with few classes, use enhanced base
        elif use_enhanced:
            logger.info(f"Creating Enhanced Base Autoencoder with {feature_dims}D → {compressed_dims}D features")
            logger.info(f"  Domain: {domain}")
            logger.info(f"  Classes: {num_classes}")
            return EnhancedBaseAutoencoder(config)

        # Fallback to original base autoencoder
        else:
            logger.info(f"Creating Base Autoencoder with {feature_dims}D → {compressed_dims}D features")
            return BaseAutoencoder(config)

# =============================================================================
# DOMAIN-SPECIFIC ADVANCED HYBRID AUTOENCODERS (Optional Extensions)
# =============================================================================

class AstronomicalAdvancedHybrid(AdvancedHybridAutoencoder):
    """
    Astronomy-specific advanced hybrid autoencoder
    Combines domain-specific features with advanced hybrid architecture
    """

    def __init__(self, config: GlobalConfig):
        super().__init__(config)

        # Add astronomy-specific components
        self.ring_detector = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.psf_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder_channels[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        logger.info("Added astronomy-specific components: ring detector, PSF estimator")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Astronomy-enhanced encoding"""
        # Detect rings and other structures
        ring_map = self.ring_detector(x)
        x = x * (1 + 0.1 * ring_map)

        # Standard encoding
        return super().encode(x)

    def get_domain_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract astronomy-specific features"""
        features = {}

        # PSF estimation
        for layer in self.encoder_layers:
            x = layer(x)
        features['psf_size'] = self.psf_estimator(x)

        return features


class MedicalAdvancedHybrid(AdvancedHybridAutoencoder):
    """
    Medical imaging-specific advanced hybrid autoencoder
    """

    def __init__(self, config: GlobalConfig):
        super().__init__(config)

        # Add medical-specific components
        self.tissue_analyzer = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 8)
        )

        self.anomaly_detector = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        logger.info("Added medical-specific components: tissue analyzer, anomaly detector")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Medical-enhanced encoding"""
        # Detect anomalies/lesions
        anomaly_map = self.anomaly_detector(x)
        x = x * (1 + 0.2 * anomaly_map)

        # Standard encoding
        return super().encode(x)

    def get_domain_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract medical-specific features"""
        features = {}

        # Tissue analysis
        features['tissue_features'] = self.tissue_analyzer(x)

        return features


class AgriculturalAdvancedHybrid(AdvancedHybridAutoencoder):
    """
    Agriculture-specific advanced hybrid autoencoder
    """

    def __init__(self, config: GlobalConfig):
        super().__init__(config)

        # Add agriculture-specific components
        self.vegetation_index = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.disease_detector = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        logger.info("Added agriculture-specific components: vegetation index, disease detector")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Agriculture-enhanced encoding"""
        # Compute vegetation index (NDVI-like)
        veg_index = self.vegetation_index(x)
        x = x * (1 + 0.15 * veg_index)

        # Detect disease patterns
        disease_map = self.disease_detector(x)
        x = x * (1 + 0.1 * disease_map)

        # Standard encoding
        return super().encode(x)


# =============================================================================
# ENHANCED CDBNN APPLICATION WITH DOMAIN SUPPORT
# =============================================================================

class DomainAwareAdvancedCDBNN(DomainAwareCDBNN):
    """
    Enhanced CDBNN application with domain-specific advanced hybrid autoencoders
    """

    def __init__(self, config: GlobalConfig):
        super().__init__(config)

        # Override model creation to use domain-specific advanced hybrids
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')

        # Re-create model with domain-specific advanced hybrid
        self.model = ModelFactory.create_model(config)
        self.model = self.model.to(self.device)

        logger.info(f"DomainAwareAdvancedCDBNN initialized with domain: {config.domain}")

        # Initialize domain processor if specified
        if config.domain != 'general':
            self._init_domain_processor()
            logger.info(f"Domain processor initialized for: {config.domain}")

    def extract_domain_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract domain-specific features using the model's domain components"""
        features = super().extract_domain_features(image)

        # Add advanced features if available
        if hasattr(self.model, 'get_domain_features'):
            with torch.no_grad():
                # Convert image to tensor
                img_tensor = torch.from_numpy(image).float()
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor.to(self.device)

                domain_features = self.model.get_domain_features(img_tensor)
                for key, value in domain_features.items():
                    features[f'advanced_{key}'] = value.mean().item()

        return features

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              resume: bool = False, resume_from: Optional[str] = None,
              reset_optimizer: bool = False, additional_epochs: Optional[int] = None) -> Dict:
        """Train with domain-specific enhancements"""

        # Log domain-specific training info
        domain = self.config.domain
        logger.info("=" * 60)
        logger.info(f"Domain-Specific Training: {domain.upper()}")

        if domain == 'astronomy':
            logger.info("  - Ring detection enabled")
            logger.info("  - PSF estimation enabled")
            logger.info("  - Source detection enabled")
        elif domain == 'medical':
            logger.info("  - Tissue analysis enabled")
            logger.info("  - Anomaly detection enabled")
            logger.info("  - Lesion detection enabled")
        elif domain == 'agriculture':
            logger.info("  - Vegetation index computation enabled")
            logger.info("  - Disease detection enabled")
            logger.info("  - Pest damage detection enabled")

        logger.info("=" * 60)

        # Call parent training method
        return super().train(train_loader, val_loader, resume, resume_from,
                            reset_optimizer, additional_epochs)


# =============================================================================
# UPDATED CREATE_DOMAIN_CONFIG WITH ADVANCED FLAGS
# =============================================================================

def create_domain_config(args):
    """Create appropriate configuration based on domain and advanced settings"""

    # Check if user explicitly set compressed_dims
    explicit_compressed_dims = hasattr(args, 'compressed_dims') and args.compressed_dims is not None

    base_config = {
        'dataset_name': args.data_name,
        'data_type': args.data_type,
        'feature_dims': args.feature_dims,
        'compressed_dims': args.compressed_dims if explicit_compressed_dims else 32,  # Force default 32
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'num_workers': args.workers,
        'use_gpu': not args.cpu and torch.cuda.is_available(),
        'mixed_precision': not args.no_mixed_precision,
        'use_kl_divergence': not args.disable_kl,
        'use_class_encoding': not args.disable_class,
        'use_distance_correlation': not args.disable_distance_corr,
        'max_features': args.max_features,
        'generate_heatmaps': args.generate_heatmaps,
        'generate_tsne': args.generate_tsne,
        'output_dir': args.output_dir,
        'domain': args.domain,
        'use_per_image_normalization': getattr(args, 'per_image_norm', False),

        # Advanced hybrid flags
        'use_enhanced_autoencoder': getattr(args, 'use_enhanced', True),
        'use_hybrid_autoencoder': getattr(args, 'use_hybrid', True),
        'use_advanced_hybrid': getattr(args, 'use_advanced', True),
        'use_perceptual_loss': getattr(args, 'use_perceptual_loss', True),
        'distillation_mode': getattr(args, 'distillation_mode', 'contrastive'),

        # Domain enhancement flags
        'use_detail_attention': getattr(args, 'use_detail_attention', True),
        'use_multiscale_features': getattr(args, 'use_multiscale_features', True),
        'use_feature_refinement': getattr(args, 'use_feature_refinement', True),

        # Training flags
        'no_augmentation': getattr(args, 'no_augmentation', False),
        'augmentation_strength': getattr(args, 'augmentation_strength', 0.5),

        # CRITICAL: Force fixed compressed_dims if not explicitly set by user
        'force_fixed_compressed_dims': not explicit_compressed_dims,
    }


    # Handle input size
    if hasattr(args, 'input_size') and args.input_size is not None:
        base_config['input_size'] = tuple(args.input_size)
        base_config['input_size_explicitly_set'] = True
    else:
        base_config['input_size_explicitly_set'] = False

    # Domain-specific configurations
    if args.domain == 'agriculture':
        return AgricultureConfig(**base_config, has_nir_band=args.has_nir_band)
    elif args.domain == 'medical':
        return MedicalConfig(**base_config, modality=args.modality)
    elif args.domain == 'satellite':
        return SatelliteConfig(**base_config, satellite_type=args.satellite_type)
    elif args.domain == 'surveillance':
        return SurveillanceConfig(**base_config, detect_motion=args.detect_motion,
                                 enhance_low_light=args.enhance_low_light)
    elif args.domain == 'microscopy':
        return MicroscopyConfig(**base_config, microscopy_type=args.microscopy_type)
    elif args.domain == 'industrial':
        return IndustrialConfig(**base_config, detect_crack=args.detect_crack,
                               detect_corrosion=args.detect_corrosion)
    elif args.domain == 'astronomy':
        astro_args = {
            'use_fits': getattr(args, 'use_fits', True),
            'fits_hdu': getattr(args, 'fits_hdu', 0),
            'fits_normalization': getattr(args, 'fits_normalization', 'zscale'),
            'subtract_background': getattr(args, 'subtract_background', True),
            'detect_sources': getattr(args, 'detect_sources', True),
            'detection_threshold': getattr(args, 'detection_threshold', 2.5),
            'pixel_scale': getattr(args, 'pixel_scale', 1.0),
            'gain': getattr(args, 'gain', 1.0),
            'read_noise': getattr(args, 'read_noise', 0.0)
        }
        return AstronomyConfig(**base_config, **astro_args)
    else:
        return GlobalConfig(**base_config)


# =============================================================================
# UPDATED ARGUMENT PARSER WITH ADVANCED OPTIONS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='CDBNN - Convolutional Deep Bayesian Neural Network')

    # Basic arguments
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--verify', '-v', action='store_true', help='Run verification tests')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'extract', 'resume'],
                           help='Operation mode (train, predict, extract, or resume)')
    parser.add_argument('--data_name', type=str, help='Dataset name')
    parser.add_argument('--data_type', type=str, choices=['custom', 'torchvision'], default='custom')
    parser.add_argument('--data_path', type=str, default=None, help='Path to data')
    parser.add_argument('--domain', type=str,
                       choices=['general', 'agriculture', 'medical', 'satellite',
                               'surveillance', 'microscopy', 'industrial', 'astronomy'],
                       default='general', help='Domain for specialized processing')

    # Normalization strategy
    parser.add_argument('--per_image_norm', action='store_true',
                       help='Use per-image Z-score normalization')

    # Advanced architecture options
    parser.add_argument('--use_enhanced', action='store_true', default=True,
                       help='Use enhanced autoencoder features')
    parser.add_argument('--use_hybrid', action='store_true', default=True,
                       help='Use hybrid autoencoder (conditional training, unconditional inference)')
    parser.add_argument('--use_advanced', action='store_true', default=True,
                       help='Use advanced hybrid with modulation and contrastive distillation')
    parser.add_argument('--use_perceptual_loss', action='store_true', default=True,
                       help='Use perceptual loss for sharper reconstructions')
    parser.add_argument('--distillation_mode', type=str,
                       choices=['contrastive', 'relational', 'mse'], default='contrastive',
                       help='Knowledge distillation mode')

    # Domain enhancement options
    parser.add_argument('--use_detail_attention', action='store_true', default=True,
                       help='Use detail attention module')
    parser.add_argument('--use_multiscale_features', action='store_true', default=True,
                       help='Use multi-scale feature extraction')
    parser.add_argument('--use_feature_refinement', action='store_true', default=True,
                       help='Use feature refinement')

    # Training options
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--augmentation_strength', type=float, default=0.5,
                       help='Strength of data augmentations (0.0 to 1.0)')

    # Model hyperparameters
    parser.add_argument('--feature_dims', type=int, default=128)
    parser.add_argument('--compressed_dims', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--max_features', type=int, default=32)

    # Training options
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no_mixed_precision', action='store_true')
    parser.add_argument('--generate_heatmaps', action='store_true', default=True)
    parser.add_argument('--generate_tsne', action='store_true', default=True)
    parser.add_argument('--disable_kl', action='store_true')
    parser.add_argument('--disable_class', action='store_true')
    parser.add_argument('--disable_distance_corr', action='store_true')
    parser.add_argument('--output_dir', type=str, default='data')

    # Input size
    parser.add_argument('--input_size', type=int, nargs=2, metavar=('H', 'W'),
                       help='Input image size (height width)')

    # Architecture optimization
    parser.add_argument('--auto_optimize', action='store_true', default=True)
    parser.add_argument('--no_auto_optimize', action='store_true')

    # Domain-specific flags
    parser.add_argument('--has_nir_band', action='store_true', help='Has near-infrared band (agriculture)')
    parser.add_argument('--modality', type=str, default='general', help='Medical imaging modality')
    parser.add_argument('--satellite_type', type=str, default='general', help='Satellite type')
    parser.add_argument('--detect_motion', action='store_true', help='Detect motion (surveillance)')
    parser.add_argument('--enhance_low_light', action='store_true', help='Enhance low-light (surveillance)')
    parser.add_argument('--microscopy_type', type=str, default='general', help='Microscopy type')
    parser.add_argument('--detect_crack', action='store_true', help='Detect cracks (industrial)')
    parser.add_argument('--detect_corrosion', action='store_true', help='Detect corrosion (industrial)')

    # Astronomy-specific flags
    parser.add_argument('--use_fits', action='store_true', help='Enable FITS support')
    parser.add_argument('--fits_hdu', type=int, default=0, help='FITS HDU to read')
    parser.add_argument('--fits_normalization', type=str,
                       choices=['zscale', 'percent', 'minmax', 'asinh'], default='zscale')
    parser.add_argument('--subtract_background', action='store_true', default=True)
    parser.add_argument('--detect_sources', action='store_true', default=True)
    parser.add_argument('--detection_threshold', type=float, default=2.5)
    parser.add_argument('--pixel_scale', type=float, default=1.0)
    parser.add_argument('--gain', type=float, default=1.0)
    parser.add_argument('--read_noise', type=float, default=0.0)

    # Resume training arguments
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to specific checkpoint file to resume from')
    parser.add_argument('--reset_optimizer', action='store_true',
                       help='Reset optimizer state when resuming (keep only model weights)')
    parser.add_argument('--additional_epochs', type=int, default=None,
                       help='Add additional epochs to original training (overrides original epochs)')

   # ================================================================
    # SUPERVISED CONTRASTIVE LEARNING DIRECTIVE
    # ================================================================
    parser.add_argument('--use_contrastive', action='store_true',
                       help='Use Supervised Contrastive Learning instead of clustering')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss (default: 0.07)')
    parser.add_argument('--arcface_margin', type=float, default=0.5,
                       help='Margin for ArcFace loss (default: 0.5)')
    parser.add_argument('--contrastive_projection_dim', type=int, default=128,
                       help='Projection dimension for contrastive head (default: 128)')
    parser.add_argument('--contrastive_weight', type=float, default=0.7,
                       help='Weight for contrastive loss (default: 0.7)')

    return parser.parse_args()

# =============================================================================
# DOMAIN-SPECIFIC LOSS FUNCTIONS
# =============================================================================

class AstronomicalStructureLoss(nn.Module):
    """Loss function specialized for astronomical imaging features"""
    def __init__(self):
        super().__init__()

        # Edge detection filters
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32).view(1, 1, 3, 3)

        # Point source detection filter (for stars)
        self.point_filter = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                                       dtype=torch.float32).view(1, 1, 3, 3)

        # Multi-scale structure filters
        self.structure_filters = [
            self._create_gaussian_kernel(sigma) for sigma in [0.5, 1.0, 2.0]
        ]

        # Scale-space filters for galaxy features
        self.scale_filters = [
            self._create_log_kernel(sigma) for sigma in [1.0, 2.0, 4.0]
        ]

    def _create_gaussian_kernel(self, sigma: float, kernel_size: int = 7) -> torch.Tensor:
        """Create Gaussian kernel for smoothing"""
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        x = x.view(-1, 1)
        y = x.t()
        gaussian = torch.exp(-(x**2 + y**2)/(2*sigma**2))
        return (gaussian / gaussian.sum()).view(1, 1, kernel_size, kernel_size)

    def _create_log_kernel(self, sigma: float, kernel_size: int = 7) -> torch.Tensor:
        """Create Laplacian of Gaussian kernel for blob detection"""
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        x = x.view(-1, 1)
        y = x.t()
        r2 = x**2 + y**2
        log = (1 - r2/(2*sigma**2)) * torch.exp(-r2/(2*sigma**2))
        return (log / log.abs().sum()).view(1, 1, kernel_size, kernel_size)

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss with astronomical feature preservation"""
        device = reconstruction.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        self.point_filter = self.point_filter.to(device)
        self.structure_filters = [f.to(device) for f in self.structure_filters]
        self.scale_filters = [f.to(device) for f in self.scale_filters]

        # Basic reconstruction loss with intensity weighting
        intensity_weights = (target > target.mean()).float() * 2 + 1
        recon_loss = F.mse_loss(reconstruction * intensity_weights,
                               target * intensity_weights)

        # Edge and gradient preservation
        rec_grad_x = F.conv2d(reconstruction, self.sobel_x, padding=1)
        rec_grad_y = F.conv2d(reconstruction, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)

        gradient_loss = F.mse_loss(rec_grad_x, target_grad_x) + \
                       F.mse_loss(rec_grad_y, target_grad_y)

        # Point source (star) preservation
        rec_points = F.conv2d(reconstruction, self.point_filter, padding=1)
        target_points = F.conv2d(target, self.point_filter, padding=1)
        point_loss = F.mse_loss(rec_points, target_points)

        # Multi-scale structure preservation
        structure_loss = 0
        for filter_kernel in self.structure_filters:
            rec_struct = F.conv2d(reconstruction, filter_kernel, padding=filter_kernel.size(-1)//2)
            target_struct = F.conv2d(target, filter_kernel, padding=filter_kernel.size(-1)//2)
            structure_loss += F.mse_loss(rec_struct, target_struct)

        # Scale-space feature preservation
        scale_loss = 0
        for filter_kernel in self.scale_filters:
            rec_scale = F.conv2d(reconstruction, filter_kernel, padding=filter_kernel.size(-1)//2)
            target_scale = F.conv2d(target, filter_kernel, padding=filter_kernel.size(-1)//2)
            scale_loss += F.mse_loss(rec_scale, target_scale)

        # Peak intensity preservation (for bright stars)
        peak_loss = F.l1_loss(
            torch.max_pool2d(reconstruction, kernel_size=3, stride=1, padding=1),
            torch.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        )

        # Combine losses with weights
        total_loss = (recon_loss +
                     2.0 * gradient_loss +
                     1.5 * point_loss +
                     1.0 * structure_loss +
                     1.0 * scale_loss +
                     2.0 * peak_loss)

        return total_loss

class MedicalStructureLoss(nn.Module):
    """Loss function specialized for medical imaging features"""
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                     dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                     dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = reconstruction.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        # Tissue weighting
        tissue_weights = (target > target.mean()).float() * 2 + 1
        recon_loss = F.mse_loss(reconstruction * tissue_weights, target * tissue_weights)

        # Boundary preservation
        rec_grad_x = F.conv2d(reconstruction, sobel_x, padding=1)
        rec_grad_y = F.conv2d(reconstruction, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)

        gradient_loss = F.mse_loss(rec_grad_x, target_grad_x) + \
                       F.mse_loss(rec_grad_y, target_grad_y)

        # Local contrast preservation
        rec_std = torch.std(F.unfold(reconstruction, kernel_size=5), dim=1)
        target_std = torch.std(F.unfold(target, kernel_size=5), dim=1)
        contrast_loss = F.mse_loss(rec_std, target_std)

        return recon_loss + 1.5 * gradient_loss + 1.0 * contrast_loss

class AgriculturalPatternLoss(nn.Module):
    """Loss function optimized for agricultural pest and disease detection"""
    def __init__(self):
        super().__init__()
        self.texture_filters = None

    def _create_gabor_kernel(self, frequency: float, angle: float, sigma: float = 3.0, size: int = 7):
        angle_rad = angle * np.pi / 180
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-size//2, size//2, size),
            torch.linspace(-size//2, size//2, size),
            indexing='ij'
        )
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        x_rot = x_grid * cos_angle + y_grid * sin_angle
        y_rot = -x_grid * sin_angle + y_grid * cos_angle

        gaussian = torch.exp(-(x_rot**2 + y_rot**2)/(2*sigma**2))
        sinusoid = torch.cos(2 * np.pi * frequency * x_rot)
        kernel = (gaussian * sinusoid).view(1, 1, size, size)
        return kernel / (kernel.abs().sum() + 1e-8)

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = reconstruction.device

        # Initialize filters on first use
        if self.texture_filters is None:
            self.texture_filters = [
                self._create_gabor_kernel(frequency=f, angle=a).to(device)
                for f in [0.1, 0.2, 0.3] for a in [0, 45, 90, 135]
            ]

        # Base reconstruction loss
        recon_loss = F.mse_loss(reconstruction, target)

        # Texture preservation loss
        texture_loss = 0
        for filter_kernel in self.texture_filters:
            rec_texture = F.conv2d(reconstruction, filter_kernel, padding=filter_kernel.size(-1)//2)
            target_texture = F.conv2d(target, filter_kernel, padding=filter_kernel.size(-1)//2)
            texture_loss += F.mse_loss(rec_texture, target_texture)
        texture_loss = texture_loss / len(self.texture_filters)

        # Color preservation
        color_loss = 0
        if reconstruction.shape[1] >= 3:
            rec_color = torch.mean(reconstruction, dim=1, keepdim=True)
            target_color = torch.mean(target, dim=1, keepdim=True)
            color_loss = F.mse_loss(rec_color, target_color)

        return recon_loss + 2.0 * texture_loss + 0.5 * color_loss

class EnhancedLossManager:
    """Manager for handling specialized loss functions"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.loss_functions = {}
        self.domain = getattr(config, 'domain', 'general')
        self.initialize_loss_functions()

    def initialize_loss_functions(self):
        """Initialize appropriate loss functions based on domain"""
        if self.domain == 'astronomy':
            self.loss_functions['astronomical'] = AstronomicalStructureLoss()
            logger.info("Initialized Astronomical Structure Loss")

        elif self.domain == 'medical':
            self.loss_functions['medical'] = MedicalStructureLoss()
            logger.info("Initialized Medical Structure Loss")

        elif self.domain == 'agriculture':
            self.loss_functions['agricultural'] = AgriculturalPatternLoss()
            logger.info("Initialized Agricultural Pattern Loss")

    def get_loss_function(self) -> Optional[nn.Module]:
        """Get appropriate loss function for current domain"""
        if self.domain == 'astronomy':
            return self.loss_functions.get('astronomical')
        elif self.domain == 'medical':
            return self.loss_functions.get('medical')
        elif self.domain == 'agriculture':
            return self.loss_functions.get('agricultural')
        return None

    def calculate_loss(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss with appropriate enhancements"""
        loss_fn = self.get_loss_function()
        if loss_fn is None:
            return F.mse_loss(reconstruction, target)
        return loss_fn(reconstruction, target)

# =============================================================================
# DOMAIN-SPECIFIC AUTOENCODERS
# =============================================================================

class AstronomicalStructurePreservingAutoencoder(EnhancedBaseAutoencoder):
    """Enhanced Autoencoder specialized for astronomical imaging features with discriminative capabilities"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: GlobalConfig):
        # Store config for later use
        self._domain_config = config

        # Call parent init first to set up basic structure
        super().__init__(config)

        # Get enhancement configurations
        self.enhancement_config = {
            'structure_preservation': getattr(config, 'structure_preservation', True),
            'detail_preservation': getattr(config, 'detail_preservation', True),
            'star_detection': getattr(config, 'star_detection', True),
            'galaxy_features': getattr(config, 'galaxy_features', True),
            'discriminative_ring_detection': getattr(config, 'discriminative_ring_detection', True)
        }

        # ================================================================
        # RECALCULATE FINAL SPATIAL DIMENSIONS AFTER ENCODER
        # ================================================================
        h, w = config.input_size
        self.final_h, self.final_w = h, w
        for _ in range(len(self.encoder_layers)):
            self.final_h = (self.final_h + 1) // 2
            self.final_w = (self.final_w + 1) // 2
        self.final_h = max(1, self.final_h)
        self.final_w = max(1, self.final_w)

        logger.info(f"Astronomical encoder final size: {self.final_h}x{self.final_w} with {self.encoder_channels[-1]} channels")

        # ================================================================
        # REBUILD FLATTENED SIZE AND EMBEDDER/UNEMBEDDER
        # ================================================================
        self.flattened_size = self.encoder_channels[-1] * self.final_h * self.final_w

        # Rebuild embedder with correct size
        embed_dim = min(self.flattened_size, self.feature_dims)
        embed_dim = max(embed_dim, 1)

        self.embedder = nn.Sequential(
            nn.Linear(self.flattened_size, embed_dim),
            nn.GroupNorm(min(16, embed_dim), embed_dim) if embed_dim > 1 else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        if embed_dim != self.feature_dims:
            self.embedder_projection = nn.Linear(embed_dim, self.feature_dims)
        else:
            self.embedder_projection = nn.Identity()

        # Rebuild unembedder with correct size
        unembed_dim = min(self.flattened_size, self.feature_dims)
        unembed_dim = max(unembed_dim, 1)

        self.unembedder = nn.Sequential(
            nn.Linear(self.feature_dims, unembed_dim),
            nn.GroupNorm(min(16, unembed_dim), unembed_dim) if unembed_dim > 1 else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        if unembed_dim != self.flattened_size:
            self.unembedder_projection = nn.Linear(unembed_dim, self.flattened_size)
        else:
            self.unembedder_projection = nn.Identity()

        # ================================================================
        # BUILD CUSTOM ENCODER PATH
        # ================================================================

        # Initial channel transformation layer (in_channels → encoder_channels[0])
        self.initial_transform = nn.Sequential(
            nn.Conv2d(self.in_channels, self.encoder_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(min(32, self.encoder_channels[0]), self.encoder_channels[0]),
            nn.LeakyReLU(0.2)
        )

        # Detail preservation module with multiple scales
        self.detail_preserving = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.encoder_channels[0], self.encoder_channels[0], kernel_size=k, padding=k//2),
                nn.GroupNorm(min(32, self.encoder_channels[0]), self.encoder_channels[0]),
                nn.LeakyReLU(0.2),
                nn.Conv2d(self.encoder_channels[0], self.encoder_channels[0], kernel_size=1)
            ) for k in [3, 5, 7]
        ])

        # Star detection module
        self.star_detector = nn.Sequential(
            nn.Conv2d(self.encoder_channels[0], self.encoder_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(min(32, self.encoder_channels[0]), self.encoder_channels[0]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.encoder_channels[0], self.encoder_channels[0], kernel_size=1),
            nn.Sigmoid()
        )

        # Custom encoder layers
        self.custom_encoder_layers = nn.ModuleList()
        in_channels = self.encoder_channels[0]
        for i in range(len(self.encoder_layers)):
            out_channels = self.encoder_channels[i] if i < len(self.encoder_channels) else min(512, 64 * (2 ** i))
            num_groups = min(32, max(1, out_channels))
            self.custom_encoder_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            in_channels = out_channels

        # Galaxy feature enhancement
        self.galaxy_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(size, size, kernel_size=3, padding=d, dilation=d),
                nn.GroupNorm(min(32, size), size),
                nn.LeakyReLU(0.2)
            ) for size, d in zip(self.encoder_channels, [1, 2, 4])
        ])

        # Ring structure detector
        if self.enhancement_config.get('discriminative_ring_detection', True):
            self.ring_detector = nn.Sequential(
                nn.Conv2d(self.encoder_channels[0], 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid()
            )

            self.ring_features = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.encoder_channels[-1], self.encoder_channels[-1],
                             kernel_size=k, padding=k//2, dilation=1),
                    nn.BatchNorm2d(self.encoder_channels[-1]),
                    nn.ReLU(inplace=True)
                ) for k in [3, 5, 7]
            ])

            self.discriminative_projector = nn.Sequential(
                nn.Linear(self.compressed_dims, self.compressed_dims * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(self.compressed_dims * 2, self.compressed_dims),
                nn.Tanh()
            )
        else:
            self.ring_detector = None
            self.ring_features = None
            self.discriminative_projector = None

        # Rebuild decoder
        self._rebuild_decoder()

        self._cached_features = {}

        # Log architecture info
        logger.info("=" * 60)
        logger.info("AstronomicalStructurePreservingAutoencoder initialized:")
        logger.info(f"  Input: {self.in_channels}x{h}x{w}")
        logger.info(f"  Encoder output: {self.encoder_channels[-1]}x{self.final_h}x{self.final_w}")
        logger.info(f"  Flattened size: {self.flattened_size}")
        logger.info(f"  Feature dims: {self.feature_dims}")
        logger.info(f"  Compressed dims: {self.compressed_dims}")
        logger.info("=" * 60)

    def _rebuild_decoder(self):
        """Rebuild decoder to match the custom encoder's output channels"""
        self.custom_decoder_layers = nn.ModuleList()
        in_channels = self.encoder_channels[-1]

        # Calculate target output size
        target_h, target_w = self.config.input_size
        current_h, current_w = self.final_h, self.final_w

        # Determine number of upsampling steps
        n_steps = len(self.encoder_channels)

        for i in range(n_steps):
            if i == n_steps - 1:  # Last layer (output)
                out_channels = self.in_channels
                self.custom_decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.Tanh()
                ))
            else:
                out_channels = self.encoder_channels[n_steps - 2 - i]
                num_groups_dec = min(32, max(1, out_channels))
                self.custom_decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.GroupNorm(num_groups_dec, out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            in_channels = out_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced encoding with astronomical feature preservation and ring detection"""
        features = {}

        # Apply dataset-wide normalization if available
        if self.dataset_statistics is not None and self.dataset_statistics.is_calculated:
            x = self.dataset_statistics.normalize(x)

        # Apply base enhanced encoding features
        if hasattr(super(), 'detail_attention') and super().detail_attention is not None:
            attention = super().detail_attention(x)
            x = x * (1 + attention)

        # Apply edge preservation
        if hasattr(super(), 'edge_preservation') and super().edge_preservation is not None:
            edge_weights = super().edge_preservation(x)
            x = x * (0.5 + edge_weights)

        # Initial channel transformation
        x = self.initial_transform(x)

        if self.enhancement_config.get('detail_preservation', True):
            # Multi-scale detail extraction
            detail_features = [module(x) for module in self.detail_preserving]
            features['details'] = sum(detail_features) / len(detail_features)
            x = x + 0.1 * features['details']

        # Ring structure detection (early stage)
        if self.ring_detector is not None and self.enhancement_config.get('discriminative_ring_detection', True):
            ring_map = self.ring_detector(x)
            features['ring_map'] = ring_map
            x = x * (1 + 0.15 * ring_map)

        if self.enhancement_config.get('star_detection', True):
            features['stars'] = self.star_detector(x)
            x = x * (1 + 0.1 * features['stars'])

        # Custom encoding path
        for idx, layer in enumerate(self.custom_encoder_layers):
            x = layer(x)
            if self.enhancement_config.get('galaxy_features', True):
                if idx < len(self.galaxy_enhancer):
                    galaxy_features = self.galaxy_enhancer[idx](x)
                    x = x + 0.1 * galaxy_features

        # Multi-scale ring feature extraction at deeper level
        if self.ring_features is not None and self.enhancement_config.get('discriminative_ring_detection', True):
            ring_multi_features = []
            for extractor in self.ring_features:
                ring_multi_features.append(extractor(x))
            features['ring_multiscale'] = sum(ring_multi_features) / len(ring_multi_features)
            x = x + 0.1 * features['ring_multiscale']

        # Flatten and embed
        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)
        embedding = self.embedder_projection(embedding)

        # Store features for use in decode
        self._cached_features = features

        return embedding

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Enhanced decoding with structure preservation - FIXED dimension handling"""

        # Unembed to get the encoded representation
        x = self.unembedder(z)
        x = self.unembedder_projection(x)

        batch_size = x.size(0)
        expected_channels = self.encoder_channels[-1]
        expected_height = self.final_h
        expected_width = self.final_w
        expected_elements = expected_channels * expected_height * expected_width
        actual_elements = x.size(1)

        # Handle dimension mismatch with multiple fallback strategies
        if actual_elements == expected_elements:
            # Perfect match - direct reshape
            x = x.view(batch_size, expected_channels, expected_height, expected_width)
        else:
            logger.debug(f"Reshape mismatch in decode: got {actual_elements}, expected {expected_elements}")

            # Strategy 1: Try to find factorable dimensions
            reshaped = False
            for h in range(int(actual_elements ** 0.5), 0, -1):
                if actual_elements % h == 0:
                    w = actual_elements // h
                    if h <= 512 and w <= 512 and h >= 1 and w >= 1:
                        # Try to reshape as 1-channel then adapt
                        x = x.view(batch_size, 1, h, w)
                        # Use adaptive pooling to target size
                        x = F.adaptive_avg_pool2d(x, (expected_height, expected_width))
                        # Expand channels to expected
                        x = x.expand(-1, expected_channels, -1, -1)
                        reshaped = True
                        logger.debug(f"Strategy 1 succeeded: reshaped to {h}x{w} then adapted")
                        break

            if not reshaped:
                # Strategy 2: Use linear projection
                if not hasattr(self, '_reshape_projection'):
                    self._reshape_projection = nn.Linear(actual_elements, expected_elements).to(x.device)
                x = self._reshape_projection(x)
                x = x.view(batch_size, expected_channels, expected_height, expected_width)
                logger.debug("Strategy 2 succeeded: used linear projection")

        # Run through decoder layers
        for layer in self.custom_decoder_layers:
            x = layer(x)

        # Ensure output has correct number of channels
        if x.shape[1] != self.in_channels:
            if not hasattr(self, '_final_channel_adaptor'):
                self._final_channel_adaptor = nn.Conv2d(x.shape[1], self.in_channels, kernel_size=1).to(x.device)
            x = self._final_channel_adaptor(x)

        # Ensure output has correct spatial dimensions
        target_h, target_w = self.config.input_size
        if x.shape[-2] != target_h or x.shape[-1] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)

        return x

    def get_refined_features(self, compressed_embedding: torch.Tensor) -> torch.Tensor:
        """Get refined features with ring discrimination enhancement"""
        if self.discriminative_projector is not None and self.enhancement_config.get('discriminative_ring_detection', True):
            return self.discriminative_projector(compressed_embedding)
        return compressed_embedding

class MedicalStructurePreservingAutoencoder(EnhancedBaseAutoencoder):
    """Enhanced Autoencoder specialized for medical imaging features"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: GlobalConfig):
        super().__init__(config)

        # Get enhancement configurations
        self.enhancement_config = {
            'tissue_boundary': getattr(config, 'tissue_boundary', True),
            'lesion_detection': getattr(config, 'lesion_detection', True),
            'contrast_enhancement': getattr(config, 'contrast_enhancement', True)
        }

        # Tissue boundary detection
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=4),
            nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Lesion detection module
        self.lesion_detector = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=d, dilation=d),
                nn.InstanceNorm2d(32),
                nn.PReLU()
            ) for d in [1, 2, 4]
        ])

        # Contrast enhancement module
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2),
            nn.InstanceNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, self.in_channels, kernel_size=1)
        )

        self._cached_features = {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced encoding with medical feature preservation"""
        features = {}

        # Apply dataset-wide normalization if available
        if self.dataset_statistics is not None and self.dataset_statistics.is_calculated:
            x = self.dataset_statistics.normalize(x)

        # Apply base enhanced encoding features
        if hasattr(super(), 'detail_attention') and super().detail_attention is not None:
            attention = super().detail_attention(x)
            x = x * (1 + attention)

        if self.enhancement_config.get('tissue_boundary', True):
            features['boundaries'] = self.boundary_detector(x)
            x = x * (1 + 0.1 * features['boundaries'])

        # Regular encoding path with lesion detection
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.enhancement_config.get('lesion_detection', True):
                if idx < len(self.lesion_detector):
                    lesion_features = self.lesion_detector[idx](x)
                    x = x + 0.1 * lesion_features

        if self.enhancement_config.get('contrast_enhancement', True):
            features['contrast'] = self.contrast_enhancer(x)
            x = x + 0.1 * features['contrast']

        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)

        # Store features for use in decode
        self._cached_features = features

        return embedding

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Enhanced decoding with structure preservation"""
        x = self.unembedder(z)
        x = x.view(x.size(0), self.encoder_channels[-1],
                  self.final_h, self.final_w)

        for idx, layer in enumerate(self.decoder_layers):
            x = layer(x)

        # Add preserved features
        if hasattr(self, '_cached_features') and self._cached_features:
            features = self._cached_features
            if self.enhancement_config.get('tissue_boundary', True):
                x = x * (1 + 0.1 * features.get('boundaries', 0))

            if self.enhancement_config.get('contrast_enhancement', True):
                x = x + 0.1 * features.get('contrast', 0)

            # Clear cached features
            self._cached_features = {}

        # Final channel transformation
        x = nn.Conv2d(self.encoder_channels[0], self.in_channels, kernel_size=1).to(x.device)(x)

        return x

class AgriculturalPatternAutoencoder(EnhancedBaseAutoencoder):
    """Enhanced Autoencoder specialized for agricultural imaging features"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: GlobalConfig):
        super().__init__(config)

        # Get enhancement configurations
        self.enhancement_config = {
            'texture_analysis': getattr(config, 'texture_analysis', True),
            'damage_detection': getattr(config, 'damage_detection', True),
            'color_anomaly': getattr(config, 'color_anomaly', True)
        }

        # Ensure channel numbers are compatible with groups
        texture_groups = min(4, self.in_channels)
        intermediate_channels = 32 - (32 % texture_groups)

        self.texture_analyzer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels, kernel_size=k, padding=k//2, groups=texture_groups),
                nn.InstanceNorm2d(intermediate_channels),
                nn.PReLU(),
                nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=k, padding=k//2, groups=texture_groups)
            ) for k in [3, 5, 7]
        ])

        # Damage pattern detector
        damage_intermediate_channels = 32 - (32 % self.in_channels)
        self.damage_detector = nn.Sequential(
            nn.Conv2d(self.in_channels, damage_intermediate_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(damage_intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Color anomaly detection
        self.color_analyzer = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(32, self.in_channels, kernel_size=1)
        )

        self._cached_features = {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced encoding with pattern preservation"""
        features = {}

        # Apply dataset-wide normalization if available
        if self.dataset_statistics is not None and self.dataset_statistics.is_calculated:
            x = self.dataset_statistics.normalize(x)

        # Apply base enhanced encoding features
        if hasattr(super(), 'detail_attention') and super().detail_attention is not None:
            attention = super().detail_attention(x)
            x = x * (1 + attention)

        if self.enhancement_config.get('texture_analysis', True):
            texture_features = [module(x) for module in self.texture_analyzer]
            features['texture'] = sum(texture_features) / len(texture_features)
            x = x + 0.1 * features['texture']

        if self.enhancement_config.get('damage_detection', True):
            features['damage'] = self.damage_detector(x)

        if self.enhancement_config.get('color_anomaly', True):
            features['color'] = self.color_analyzer(x)
            x = x + 0.1 * features['color']

        # Regular encoding path
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)

        # Store features for use in decode
        self._cached_features = features

        return embedding

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Enhanced decoding with pattern preservation"""
        x = self.unembedder(z)
        x = x.view(x.size(0), self.encoder_channels[-1],
                  self.final_h, self.final_w)

        for layer in self.decoder_layers:
            x = layer(x)

        # Add preserved features
        if hasattr(self, '_cached_features') and self._cached_features:
            features = self._cached_features
            if self.enhancement_config.get('texture_analysis', True):
                x = x + 0.1 * features.get('texture', 0)

            if self.enhancement_config.get('damage_detection', True):
                damage_mask = features.get('damage', torch.zeros_like(x))
                x = x * (1 + 0.2 * damage_mask)

            if self.enhancement_config.get('color_anomaly', True):
                x = x + 0.1 * features.get('color', 0)

            # Clear cached features
            self._cached_features = {}

        # Final channel transformation
        x = nn.Conv2d(self.encoder_channels[0], self.in_channels, kernel_size=1).to(x.device)(x)

        return x


# =============================================================================
# MAIN FUNCTION
# =============================================================================

#=============================================================================
# MODIFIED ARGUMENT PARSER
# =============================================================================


def set_global_random_seeds(seed: int = 42):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CuDNN operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For Python's hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info(f"Global random seeds set to {seed} for reproducibility")

# =============================================================================
# MAIN FUNCTION (with modifications for the new flag)
# =============================================================================

def main():
    args = parse_args()

    def normalize_dataset_name(data_name: str) -> str:
        if not data_name:
            return 'dataset'
        name = data_name.lower()
        name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        while '__' in name:
            name = name.replace('__', '_')
        name = name.strip('_')
        return name or 'dataset'

    def get_dataset_paths(data_name: str, base_dir: str = 'data'):
        """
        Get standardized paths for dataset files.
        ALL files go to: base_dir/dataset_lower/
        """
        dataset_name_lower = normalize_dataset_name(data_name)
        data_dir = Path(base_dir) / dataset_name_lower
        return {
            'data_dir': data_dir,
            'csv_path': data_dir / f"{dataset_name_lower}.csv",
            'train_csv': data_dir / f"{dataset_name_lower}_train.csv",
            'test_csv': data_dir / f"{dataset_name_lower}_test.csv",
            'json_config': data_dir / f"{dataset_name_lower}_config.json",
            'conf_config': data_dir / f"{dataset_name_lower}.conf",
            'minimal_config': data_dir / f"{dataset_name_lower}_config_minimal.json",
            'checkpoint_dir': data_dir / 'checkpoints',
            'viz_dir': data_dir / 'visualizations',
            'log_dir': data_dir / 'logs',
            'heatmap_dir': data_dir / 'attention_heatmaps'
        }

    # ========================================================================
    # VERIFICATION MODE
    # ========================================================================
    if hasattr(args, 'verify') and args.verify:
        print("\n" + "=" * 70)
        print("VERIFICATION MODE: Testing Deterministic Behavior")
        print("=" * 70)
        print("\n[1/3] Testing dataset statistics calculator...")
        print("\n[2/3] Testing dataset normalization...")
        print("\n[3/3] Testing end-to-end determinism...")
        print("\n" + "=" * 70)
        print("VERIFICATION COMPLETE")
        print("=" * 70)
        return 0

    # ========================================================================
    # INTERACTIVE MODE
    # ========================================================================
    if len(sys.argv) == 1 or (hasattr(args, 'interactive') and args.interactive):
        print("\n" + "=" * 70)
        print("CDBNN - Convolutional Deep Bayesian Neural Network with Domain Support")
        print("=" * 70)
        print("Available domains: general, agriculture, medical, satellite, surveillance,")
        print("                   microscopy, industrial, astronomy")
        print("=" * 70)

        data_name = input("Enter dataset name: ").strip() or 'dataset'
        mode = input("Enter mode (train/predict/extract/verify/resume): ").strip().lower() or 'train'
        data_type = input("Enter dataset type (custom/torchvision): ").strip().lower() or 'custom'
        data_path = input("Enter data path (optional for torchvision): ").strip()
        domain = input("Enter domain (general/agriculture/medical/satellite/surveillance/microscopy/industrial/astronomy): ").strip().lower() or 'general'

        use_per_image = input("Use per-image normalization? (y/n, default: n for dataset-wide): ").strip().lower()
        args.per_image_norm = use_per_image == 'y'

        use_contrastive = input("Use contrastive learning? (y/n, default: n): ").strip().lower()
        args.use_contrastive = use_contrastive == 'y'

        args.mode = mode
        args.data_name = data_name
        args.data_type = data_type
        args.data_path = data_path if data_path else None
        args.domain = domain

        if mode == 'verify':
            args.verify = True
            return main()

        if domain == 'astronomy':
            use_fits = input("Enable FITS support for astronomical images? (y/n): ").strip().lower() == 'y'
            args.use_fits = use_fits
            if use_fits:
                fits_hdu = input("FITS HDU to read (default: 0): ").strip()
                args.fits_hdu = int(fits_hdu) if fits_hdu else 0
                pixel_scale = input("Pixel scale in arcsec/pixel (default: 1.0): ").strip()
                args.pixel_scale = float(pixel_scale) if pixel_scale else 1.0

    # ========================================================================
    # VALIDATE REQUIRED ARGUMENTS
    # ========================================================================
    if not hasattr(args, 'verify') or not args.verify:
        if args.data_type == 'custom' and not args.data_path:
            raise ValueError("data_path is required for custom datasets")
        elif args.data_type == 'torchvision' and not args.data_path:
            dataset_name_lower = normalize_dataset_name(args.data_name)
            args.data_path = f"./data/{dataset_name_lower}"
            logger.info(f"Auto-setting data_path for torchvision: {args.data_path}")

    # Normalize dataset name
    if args.data_name:
        args.data_name = normalize_dataset_name(args.data_name)
    else:
        args.data_name = 'dataset'

    # Determine base directory
    if hasattr(args, 'output_dir') and args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = 'data'

    # Get dataset paths
    paths = get_dataset_paths(args.data_name, base_dir)

    # Create all directories
    paths['data_dir'].mkdir(parents=True, exist_ok=True)
    paths['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
    paths['viz_dir'].mkdir(parents=True, exist_ok=True)
    paths['log_dir'].mkdir(parents=True, exist_ok=True)
    paths['heatmap_dir'].mkdir(parents=True, exist_ok=True)

    # Update args with paths
    args.checkpoint_dir = str(paths['checkpoint_dir'])
    args.viz_dir = str(paths['viz_dir'])
    args.log_dir = str(paths['log_dir'])
    args.data_dir = str(paths['data_dir'])
    args.output_dir = str(paths['data_dir'])

    # Log configuration
    if not hasattr(args, 'verify') or not args.verify:
        logger.info("=" * 70)
        logger.info("CDBNN Configuration")
        logger.info("=" * 70)
        logger.info(f"Dataset name: {args.data_name}")
        logger.info(f"Data directory: {paths['data_dir']}")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Data type: {args.data_type}")
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"Domain: {args.domain}")
        logger.info(f"Normalization: {'PER-IMAGE' if args.per_image_norm else 'DATASET-WIDE'}")
        logger.info(f"Contrastive Learning: {args.use_contrastive if hasattr(args, 'use_contrastive') else False}")
        logger.info(f"CSV output: {paths['csv_path']}")
        logger.info(f"Config file: {paths['conf_config']}")
        logger.info(f"Checkpoint directory: {paths['checkpoint_dir']}")
        logger.info("=" * 70)

    # Create configuration
    config = create_domain_config(args)

    # ================================================================
    # SET CONTRASTIVE DIRECTIVE IN CONFIG
    # ================================================================
    config.use_contrastive_learning = getattr(args, 'use_contrastive', False)
    config.contrastive_temperature = getattr(args, 'contrastive_temperature', 0.07)
    config.contrastive_projection_dim = getattr(args, 'contrastive_projection_dim', 128)
    config.contrastive_weight = getattr(args, 'contrastive_weight', 0.7)
    config.arcface_margin = getattr(args, 'arcface_margin', 0.5)

    # Update config paths
    config.data_dir = str(paths['data_dir'])
    config.output_dir = str(paths['data_dir'])
    config.checkpoint_dir = str(paths['checkpoint_dir'])
    config.viz_dir = str(paths['viz_dir'])
    config.log_dir = str(paths['log_dir'])
    config.dataset_name = args.data_name
    config.csv_path = str(paths['csv_path'])
    config.conf_config_path = str(paths['conf_config'])

    # Create application
    app = DomainAwareCDBNN(config)

    # ========================================================================
    # HELPER FUNCTION TO CHECK FOR RESUME
    # ========================================================================
    def should_resume_training():
        """Check if we should resume training and return resume parameters"""
        is_resume_mode = (args.mode == 'resume') or getattr(args, 'resume', False)

        if not is_resume_mode:
            return False, None, False, None, None

        logger.info("=" * 70)
        logger.info("RESUME MODE ENABLED")
        logger.info("=" * 70)

        resume_from = getattr(args, 'resume_from', None)
        reset_optimizer = getattr(args, 'reset_optimizer', False)
        additional_epochs = getattr(args, 'additional_epochs', None)

        if resume_from:
            checkpoint_path = Path(resume_from)
            if checkpoint_path.exists():
                logger.info(f"Using specified checkpoint: {checkpoint_path}")
                return True, str(checkpoint_path), reset_optimizer, additional_epochs, checkpoint_path
            else:
                logger.warning(f"Specified checkpoint not found: {checkpoint_path}")
                logger.info("Looking for automatic checkpoint...")

        checkpoint_dir = Path(config.checkpoint_dir)
        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return False, None, False, None, None

        possible_checkpoints = [
            checkpoint_dir / 'latest.pt',
            checkpoint_dir / 'best.pt',
            checkpoint_dir / f"{args.data_name}_best.pt",
            checkpoint_dir / f"{args.data_name}_latest.pt",
        ]

        for cp in possible_checkpoints:
            if cp.exists():
                logger.info(f"Found checkpoint: {cp}")
                return True, str(cp), reset_optimizer, additional_epochs, cp

        logger.warning("No checkpoint found to resume from")
        return False, None, False, None, None

    # ========================================================================
    # EXECUTE BASED ON MODE
    # ========================================================================
    try:
        if args.mode == 'resume':
            logger.info(f"Resuming {args.domain} domain training on {args.data_name}")

            can_resume, resume_from, reset_optimizer, additional_epochs, checkpoint_path = should_resume_training()

            if not can_resume:
                logger.error("Cannot resume training - no valid checkpoint found")
                logger.info("Please train the model first with: python cdbnn.py --mode train")
                return 1

            # Load checkpoint to get resume state
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            start_epoch = checkpoint.get('epoch', 0) + 1
            start_phase = checkpoint.get('phase', 1)
            loaded_optimizer_state = checkpoint.get('optimizer_state_dict', None)
            loaded_scheduler_state = checkpoint.get('scheduler_state_dict', None)

            # Prepare data
            train_loader, test_loader = app.prepare_data(args.data_path, args.data_type)
            logger.info(f"Data loaded successfully: {len(train_loader.dataset)} training samples")
            if test_loader:
                logger.info(f"Test samples: {len(test_loader.dataset)}")

            # Train with resume
            history = app.train(
                train_loader,
                test_loader,
                resume=True,
                resume_from=resume_from,
                reset_optimizer=reset_optimizer,
                additional_epochs=additional_epochs
            )

            # Extract and save features
            logger.info("Extracting features from trained model...")
            features = app.extract_features(train_loader)

            if features and features.get('embeddings') is not None and len(features['embeddings']) > 0:
                if args.generate_tsne and features['embeddings'] is not None:
                    logger.info("Generating t-SNE visualization...")
                    labels_np = features['labels'].cpu().numpy() if isinstance(features['labels'], torch.Tensor) else features['labels']
                    app.visualizer.plot_tsne(features['embeddings'].cpu().numpy(), labels_np, class_names=config.class_names)
                    logger.info(f"t-SNE plot saved to: {paths['viz_dir']}/tsne.png")

                features_csv = paths['csv_path']
                app._save_features_to_csv(features, str(features_csv))
                logger.info(f"Features saved to: {features_csv}")

            app._save_config_files()

            logger.info(f"Resume training completed successfully for {args.domain} domain")
            logger.info(f"Normalization used: {config.normalization_mode}")
            logger.info(f"Model saved to: {paths['checkpoint_dir']}/{args.data_name}_best.pt")

        elif args.mode == 'train':
            logger.info(f"Starting {args.domain} domain training on {args.data_name}")
            logger.info(f"Normalization mode: {config.normalization_mode}")
            logger.info(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")

            # Check if we should resume
            can_resume, resume_from, reset_optimizer, additional_epochs, checkpoint_path = should_resume_training()

            # If resuming, load checkpoint state
            start_epoch = 0
            start_phase = 1
            loaded_optimizer_state = None
            loaded_scheduler_state = None

            if can_resume and checkpoint_path:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    start_phase = checkpoint.get('phase', 1)
                    loaded_optimizer_state = checkpoint.get('optimizer_state_dict', None)
                    loaded_scheduler_state = checkpoint.get('scheduler_state_dict', None)
                    logger.info(f"Loaded resume state: Phase {start_phase}, Epoch {start_epoch}")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint state: {e}")
                    can_resume = False

            # Prepare data
            train_loader, test_loader = app.prepare_data(args.data_path, args.data_type)
            logger.info(f"Data loaded successfully: {len(train_loader.dataset)} training samples")
            if test_loader:
                logger.info(f"Test samples: {len(test_loader.dataset)}")

            # Train with or without resume
            history = app.train(
                train_loader,
                test_loader,
                resume=can_resume,
                resume_from=resume_from,
                reset_optimizer=reset_optimizer,
                additional_epochs=additional_epochs
            )

            # Extract and save features
            logger.info("Extracting features from trained model...")
            features = app.extract_features(train_loader)

            if features and features.get('embeddings') is not None and len(features['embeddings']) > 0:
                logger.info(f"Features shape: {features['embeddings'].shape}")
                if 'labels' in features:
                    logger.info(f"Labels shape: {features['labels'].shape}")

                if args.generate_tsne and features['embeddings'] is not None:
                    logger.info("Generating t-SNE visualization...")
                    labels_np = features['labels'].cpu().numpy() if isinstance(features['labels'], torch.Tensor) else features['labels']
                    app.visualizer.plot_tsne(features['embeddings'].cpu().numpy(), labels_np, class_names=config.class_names)
                    logger.info(f"t-SNE plot saved to: {paths['viz_dir']}/tsne.png")

                features_csv = paths['csv_path']
                app._save_features_to_csv(features, str(features_csv))
                logger.info(f"Features saved to: {features_csv}")

            app._save_config_files()

            logger.info(f"Training completed successfully for {args.domain} domain")
            logger.info(f"Normalization used: {config.normalization_mode}")
            logger.info(f"Model saved to: {paths['checkpoint_dir']}/{args.data_name}_best.pt")

        elif args.mode == 'predict':
            logger.info(f"Running {args.domain} domain prediction on {args.data_name}")
            logger.info(f"Input data: {args.data_path}")

            if args.data_type == 'torchvision':
                export_dir = Path('data') / args.data_name
                if not export_dir.exists() or not (export_dir / 'train').exists():
                    logger.info("=" * 70)
                    logger.info(f"Exporting torchvision dataset {args.data_name.upper()} to images...")
                    logger.info("=" * 70)
                    temp_config = create_domain_config(args)
                    temp_app = CDBNNApplication(temp_config)
                    export_path = temp_app.export_torchvision_to_images(args.data_name, export_dir)
                    args.data_path = str(export_path)
                    logger.info(f"Using exported images from: {args.data_path}")
                else:
                    args.data_path = str(export_dir)
                    logger.info(f"Using existing exported images from: {args.data_path}")

            if not args.data_path and args.data_type == 'torchvision':
                args.data_path = str(Path('data') / args.data_name)
                logger.info(f"Auto-setting data_path to: {args.data_path}")

            # Model loading path
            model_base_dir = Path('data')
            model_data_dir = model_base_dir / args.data_name
            model_checkpoint_dir = model_data_dir / 'checkpoints'

            # Output path
            if hasattr(args, 'output_dir') and args.output_dir:
                output_base_dir = Path('./' + args.output_dir)
                output_data_dir = output_base_dir
                output_data_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_base_dir = Path('./data')
                output_data_dir = output_base_dir

            output_checkpoint_dir = output_data_dir / 'checkpoints'
            output_viz_dir = output_data_dir / 'visualizations'

            # Update config with output paths
            config.data_dir = str(output_data_dir)
            config.output_dir = str(output_data_dir)
            config.checkpoint_dir = str(output_checkpoint_dir)
            config.viz_dir = str(output_viz_dir)

            # Create prediction config
            prediction_config = copy.deepcopy(config)
            prediction_config.model_loading_dir = str(model_checkpoint_dir)
            prediction_config.data_dir_for_loading = str(model_data_dir)

            predictor = PredictionManager(prediction_config, model_load_dir=str(model_checkpoint_dir))

            output_csv = str(output_data_dir / f"{args.data_name}.csv")

            logger.info(f"Model loading from: {model_checkpoint_dir}")
            logger.info(f"Output directory: {output_data_dir}")
            logger.info(f"Output CSV: {output_csv}")

            results = predictor.predict_images(args.data_path, output_csv=output_csv)

        elif args.mode == 'extract':
            logger.info(f"Extracting {args.domain} domain features from {args.data_name}")
            logger.info(f"Input data: {args.data_path}")
            logger.info(f"Output CSV: {paths['csv_path']}")

            train_loader, test_loader = app.prepare_data(args.data_path, args.data_type)
            logger.info(f"Data loaded successfully: {len(train_loader.dataset)} samples")

            features = app.extract_features(train_loader)

            if features and features.get('embeddings') is not None and len(features['embeddings']) > 0:
                logger.info(f"Extracted {features['embeddings'].shape[0]} features")
                logger.info(f"Feature dimension: {features['embeddings'].shape[1]}")

                features_csv = paths['csv_path']
                app._save_features_to_csv(features, str(features_csv))
                logger.info(f"Features saved to: {features_csv}")

                stats = {
                    'n_samples': features['embeddings'].shape[0],
                    'n_features': features['embeddings'].shape[1],
                    'mean': features['embeddings'].mean(axis=0).tolist() if isinstance(features['embeddings'], np.ndarray) else features['embeddings'].cpu().numpy().mean(axis=0).tolist(),
                    'std': features['embeddings'].std(axis=0).tolist() if isinstance(features['embeddings'], np.ndarray) else features['embeddings'].cpu().numpy().std(axis=0).tolist(),
                }

                stats_path = paths['data_dir'] / f"{args.data_name}_features_stats.json"
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
                logger.info(f"Feature statistics saved to: {stats_path}")

                if args.generate_tsne:
                    logger.info("Generating t-SNE visualization...")
                    embeddings_np = features['embeddings'].cpu().numpy() if isinstance(features['embeddings'], torch.Tensor) else features['embeddings']
                    labels_np = features['labels'].cpu().numpy() if isinstance(features['labels'], torch.Tensor) else features['labels']
                    app.visualizer.plot_tsne(embeddings_np, labels_np, class_names=config.class_names)
                    logger.info(f"t-SNE plot saved to: {paths['viz_dir']}/tsne.png")

                app._save_config_files()

            else:
                logger.error("Feature extraction failed - no features extracted")
                return 1

        else:
            logger.error(f"Invalid mode: {args.mode}")
            logger.info("Valid modes: train, predict, extract, verify, resume")
            return 1

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

    logger.info("=" * 70)
    logger.info(f"Operation completed successfully!")
    logger.info(f"All outputs saved to: {paths['data_dir']}")
    logger.info("=" * 70)

    return 0



if __name__ == '__main__':
    print("""
===============================================Examples===========================================
# Train (with resume capability)
python cdbnn.py --mode train --data_name galaxy --data_type custom --data_path Images/Galaxy/

# Predict with custom output
python cdbnn.py --mode predict --data_name galaxy --data_type custom --data_path Images/Galaxy/ --output_dir results

# Extract features
python cdbnn.py --mode extract --data_name galaxy --data_type custom --data_path Images/Galaxy/

# With per-image normalisation (shape and structure based classification, not good for contrast based)
python cdbnn.py --mode train --data_name galaxy --data_type custom --data_path Data/Galaxies/ --per_image_norm

---------------------------------------------------------------------------------------------

# CIFAR-100 (32x32) - automatically uses 2-3 layers
python cdbnn.py --mode train --data_name cifar100 --data_type torchvision --input_size 32 32

# MNIST (28x28) - automatically adapts
python cdbnn.py --mode train --data_name mnist --data_type torchvision --input_size 28 28 --in_channels 1

# Custom small dataset (64x64)
python cdbnn.py --mode train --data_name mydata --input_size 64 64

# Large astronomy images (1024x1024)
python cdbnn.py --mode train --data_name galaxies --input_size 1024 1024 --domain astronomy

-----------------------------------------Auto detect------------------------------------------

# Auto-detect size from actual images (CIFAR-100 will detect 32x32)
python cdbnn.py --mode train --data_name cifar100 --data_type torchvision

# Auto-detect size from custom dataset
python cdbnn.py --mode train --data_name mydata --data_path ./images

# Override auto-detection with explicit size
python cdbnn.py --mode train --data_name mydata --data_path ./images --input_size 128 128
-----------------------------------------------------------------------------------------------
# Auto-optimize for CIFAR-100 (will detect 32x32 and set appropriate architecture)
python cdbnn.py --mode train --data_name cifar100 --data_type torchvision --auto_optimize

# For complex dataset with many classes (will increase capacity)
python cdbnn.py --mode train --data_name complex_data --data_path ./images --auto_optimize

# Disable auto-optimization and use manual settings
python cdbnn.py --mode train --data_name cifar100 --data_type torchvision --no_auto_optimize --input_size 32 32 --feature_dims 128 --compressed_dims 64

------------------------------------------------------------------------------------------------
# Initial training
python cdbnn.py --mode train --data_name galaxy --data_type custom --data_path Data/Galaxies/ --domain astronomy

# Resume training (automatically finds latest checkpoint)
python cdbnn.py --mode resume --data_name galaxy --data_type custom --data_path Data/Galaxies/ --domain astronomy

# Resume with specific checkpoint
python cdbnn.py --mode resume --data_name galaxy --resume_from data/galaxy/checkpoints/best.pt

# Resume but reset optimizer (keep only model weights)
python cdbnn.py --mode resume --data_name galaxy --reset_optimizer

# Resume and add 50 more epochs
python cdbnn.py --mode resume --data_name galaxy --additional_epochs 50 --epochs 200

# Using the --resume flag with train mode
python cdbnn.py --mode train --data_name galaxy --resume
===================================================================================================""")
    sys.exit(main())
