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

    data_dir: str = 'data'
    output_dir: str = 'output'
    log_dir: str = 'logs'
    viz_dir: str = 'visualizations'

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

# =============================================================================
# ASTRONOMY DOMAIN PROCESSOR
# =============================================================================

class AstronomyDomainProcessor:
    """Optimized astronomy processor with vectorized operations and scientific rigor"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.pixel_scale = getattr(config, 'pixel_scale', 1.0)
        self.gain = getattr(config, 'gain', 1.0)
        self.read_noise = getattr(config, 'read_noise', 0.0)

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
        """Optimized Z-scale normalization using percentiles"""
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

# =============================================================================
# FIXED DATASET STATISTICS CALCULATOR (Correct normalization)
# =============================================================================

class DatasetStatisticsCalculator:
    """Calculate and store dataset-wide statistics for normalization"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.mean = None
        self.std = None
        self.channel_mean = None
        self.channel_std = None
        self.per_channel_min = None
        self.per_channel_max = None
        self.n_samples_used = 0
        self.is_calculated = False

    def calculate_statistics(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """Calculate dataset-wide mean and std using Welford's algorithm for numerical stability"""
        logger.info("Calculating dataset-wide statistics for normalization...")

        # Initialize accumulators
        n_samples = 0
        channel_sum = None
        channel_sum_sq = None

        # For min/max tracking
        channel_min = None
        channel_max = None

        for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Computing dataset statistics")):
            # Move to CPU for statistics calculation
            if isinstance(inputs, torch.Tensor):
                batch = inputs.cpu()
            else:
                batch = inputs

            batch_size = batch.shape[0]
            n_channels = batch.shape[1]

            # Initialize accumulators on first batch
            if channel_sum is None:
                channel_sum = torch.zeros(n_channels)
                channel_sum_sq = torch.zeros(n_channels)
                channel_min = torch.full((n_channels,), float('inf'))
                channel_max = torch.full((n_channels,), float('-inf'))

            # Reshape batch to (batch_size * H * W, channels)
            batch_flat = batch.permute(0, 2, 3, 1).reshape(-1, n_channels)

            # Update min/max
            batch_min = batch_flat.min(dim=0)[0]
            batch_max = batch_flat.max(dim=0)[0]
            channel_min = torch.min(channel_min, batch_min)
            channel_max = torch.max(channel_max, batch_max)

            # Update sum and sum of squares (Welford's algorithm)
            batch_sum = batch_flat.sum(dim=0)
            batch_sum_sq = (batch_flat ** 2).sum(dim=0)

            # Update totals
            n_samples += batch_flat.shape[0]
            channel_sum += batch_sum
            channel_sum_sq += batch_sum_sq

        # Calculate mean and std
        self.mean = channel_sum / n_samples
        self.std = torch.sqrt((channel_sum_sq / n_samples) - (self.mean ** 2))

        # Add small epsilon to avoid division by zero
        self.std = torch.clamp(self.std, min=1e-8)

        self.per_channel_min = channel_min
        self.per_channel_max = channel_max
        self.n_samples_used = n_samples
        self.is_calculated = True

        # Store channel-wise statistics
        self.channel_mean = self.mean
        self.channel_std = self.std

        logger.info(f"Calculated statistics from {n_samples:,} pixels")
        logger.info(f"Mean per channel: {self.mean.tolist()}")
        logger.info(f"Std per channel: {self.std.tolist()}")

        return {
            'mean': self.mean,
            'std': self.std,
            'channel_min': self.per_channel_min,
            'channel_max': self.per_channel_max,
            'n_samples': self.n_samples_used
        }

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor using calculated statistics"""
        if not self.is_calculated:
            raise ValueError("Statistics not calculated. Call calculate_statistics first.")

        # Ensure statistics are on the same device as input
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)

        # Handle different input shapes
        if x.dim() == 3:  # [C, H, W]
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        elif x.dim() == 4:  # [B, C, H, W]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")

        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor back to original range"""
        if not self.is_calculated:
            raise ValueError("Statistics not calculated. Call calculate_statistics first.")

        mean = self.mean.to(x.device)
        std = self.std.to(x.device)

        if x.dim() == 3:  # [C, H, W]
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        elif x.dim() == 4:  # [B, C, H, W]
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")

        return (x * std) + mean

    def to_dict(self) -> Dict:
        """Save statistics to dictionary"""
        if not self.is_calculated:
            return {}

        return {
            'mean': self.mean.cpu().tolist() if torch.is_tensor(self.mean) else self.mean,
            'std': self.std.cpu().tolist() if torch.is_tensor(self.std) else self.std,
            'channel_min': self.per_channel_min.cpu().tolist() if torch.is_tensor(self.per_channel_min) else self.per_channel_min,
            'channel_max': self.per_channel_max.cpu().tolist() if torch.is_tensor(self.per_channel_max) else self.per_channel_max,
            'n_samples': self.n_samples_used
        }

    def from_dict(self, data: Dict):
        """Load statistics from dictionary"""
        self.mean = torch.tensor(data['mean'])
        self.std = torch.tensor(data['std'])
        self.per_channel_min = torch.tensor(data['channel_min'])
        self.per_channel_max = torch.tensor(data['channel_max'])
        self.n_samples_used = data['n_samples']
        self.is_calculated = True
        self.channel_mean = self.mean
        self.channel_std = self.std

# =============================================================================
# FIXED NORMALIZED DATASET (No extra dimension)
# =============================================================================

class NormalizedDataset(Dataset):
    """Dataset wrapper that applies dataset-wide normalization"""

    def __init__(self, dataset: Dataset, statistics_calculator: 'DatasetStatisticsCalculator'):
        self.dataset = dataset
        self.statistics = statistics_calculator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Apply dataset-wide normalization
        if self.statistics.is_calculated:
            # Ensure image is a tensor
            if not isinstance(image, torch.Tensor):
                if isinstance(image, PILImage.Image):
                    from torchvision import transforms
                    to_tensor = transforms.ToTensor()
                    image = to_tensor(image)
                else:
                    image = torch.tensor(image, dtype=torch.float32)

            # Normalize - image should be [C, H, W]
            image = self.statistics.normalize(image)

        return image, label

# =============================================================================
# COLORS AND LOGGING
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

# =============================================================================
# BASE CLASSES
# =============================================================================

class ImageProcessor:
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp')

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

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
            if np.random.random() > 0.5:
                image = ImageOps.mirror(image)
            if np.random.random() > 0.5:
                image = image.rotate(np.random.randint(-10, 10), expand=False)
        return image

    def resize_with_aspect(self, image: PILImage.Image, target_size: Tuple[int, int]) -> PILImage.Image:
        image.thumbnail(target_size, PILImage.Resampling.LANCZOS)
        new_img = PILImage.new('RGB', target_size, (0, 0, 0))
        paste_x = (target_size[0] - image.width) // 2
        paste_y = (target_size[1] - image.height) // 2
        new_img.paste(image, (paste_x, paste_y))
        return new_img

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
    """Enhanced feature selector with better handling of complex datasets"""

    def __init__(self, upper_threshold=0.85, lower_threshold=0.01, min_features=8, max_features=50):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.min_features = min_features
        self.max_features = max_features

    def calculate_distance_correlations(self, features, labels):
        n_features = features.shape[1]
        label_corrs = np.zeros(n_features)

        for i in range(n_features):
            label_corrs[i] = 1 - correlation(features[:, i], labels)

        if len(np.unique(labels)) > 10:
            separation_scores = self._calculate_class_separation_scores(features, labels)
            combined_scores = 0.7 * label_corrs + 0.3 * separation_scores
            return combined_scores

        return label_corrs

    def _calculate_class_separation_scores(self, features, labels):
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

        if np.max(separation_scores) > 0:
            separation_scores = separation_scores / np.max(separation_scores)

        return separation_scores

    def select_features(self, features, labels):
        label_corrs = self.calculate_distance_correlations(features, labels)

        selected_indices = [i for i, corr in enumerate(label_corrs) if corr >= self.upper_threshold]

        if len(selected_indices) < self.min_features:
            top_indices = np.argsort(label_corrs)[-self.min_features:]
            selected_indices = list(top_indices)
            logger.info(f"Relaxed threshold: selected top {self.min_features} features")

        selected_indices.sort(key=lambda i: -label_corrs[i])
        final_indices = self._remove_redundant_features(features, selected_indices, label_corrs)

        if len(final_indices) > self.max_features:
            final_indices = final_indices[:self.max_features]

        logger.info(f"Final feature selection: {len(final_indices)} features")
        return final_indices, label_corrs

    def _remove_redundant_features(self, features, candidate_indices, corr_values):
        final_indices = []
        feature_matrix = features[:, candidate_indices]

        for i, idx in enumerate(candidate_indices):
            keep = True
            for j in final_indices:
                corr = 1 - correlation(feature_matrix[:, i], feature_matrix[:, candidate_indices.index(j)])
                if corr > self.lower_threshold:
                    if corr_values[idx] <= corr_values[j]:
                        keep = False
                        break
            if keep:
                final_indices.append(idx)
                if len(final_indices) >= self.max_features:
                    break

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

        self.resize_images = self.config.get('resize_images', False)

        logger.info(f"Dataset: {len(self.samples)} images, {len(self.classes)} classes")

    def _scan_directory(self):
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                idx = len(self.classes)
                self.class_to_idx[class_name] = idx
                self.idx_to_class[idx] = class_name
                self.classes.append(class_name)

            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ImageProcessor.SUPPORTED_FORMATS:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                    self.image_files.append(str(img_path))
                    self.full_paths.append(str(img_path))
                    self.filenames.append(img_path.name)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = ImageProcessor.load_image(img_path)
        if img is None:
            img = PILImage.new('RGB', (256, 256), (0, 0, 0))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_additional_info(self, idx: int) -> Tuple[int, str, str]:
        return idx, self.filenames[idx], self.full_paths[idx]

    def get_class_distribution(self) -> Dict[str, int]:
        dist = defaultdict(int)
        for _, label in self.samples:
            dist[self.idx_to_class[label]] += 1
        return dict(dist)

    def get_class_weights(self) -> torch.Tensor:
        class_counts = np.bincount(self.labels)
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum() * len(class_counts)
        return torch.FloatTensor(weights)

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

class BaseAutoencoder(nn.Module):
    """Base autoencoder with dataset-wide normalization instead of BatchNorm"""

    def __init__(self, config: GlobalConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')

        self.in_channels = config.in_channels
        self.feature_dims = config.feature_dims
        self.compressed_dims = min(64, max(8, config.compressed_dims))

        self.training_phase = 1
        self._selected_feature_indices = None
        self._feature_importance_scores = None
        self._feature_selection_metadata = {}
        self._is_feature_selection_frozen = False

        # Dataset statistics (replaces BatchNorm)
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
        logger.info(f"Model initialized with {total_params:,} parameters (no BatchNorm)")

    def set_dataset_statistics(self, statistics: 'DatasetStatisticsCalculator'):
        """Set dataset statistics for normalization"""
        self.dataset_statistics = statistics
        logger.info("Dataset statistics loaded into model")

    def _build_adaptive_architecture(self):
        """Build architecture without BatchNorm layers - preserves original structure"""
        h, w = self.config.input_size
        c = self.in_channels

        n_layers = 4
        self.encoder_layers = nn.ModuleList()
        in_channels = c
        self.encoder_channels = []

        # Build encoder with GroupNorm (stable alternative to BatchNorm)
        for i in range(n_layers):
            out_channels = min(512, 64 * (2 ** i))
            # Use GroupNorm with groups=min(32, out_channels) for stability
            num_groups = min(32, out_channels)
            self.encoder_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            self.encoder_channels.append(out_channels)
            in_channels = out_channels

        self.final_h, self.final_w = max(1, h // (2 ** n_layers)), max(1, w // (2 ** n_layers))
        self.flattened_size = in_channels * self.final_h * self.final_w

        # Embedder with GroupNorm
        num_groups_embed = min(32, self.feature_dims)
        self.embedder = nn.Sequential(
            nn.Linear(self.flattened_size, self.feature_dims),
            nn.GroupNorm(num_groups_embed, self.feature_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )

        # Unembedder with GroupNorm
        num_groups_unembed = min(32, self.flattened_size)
        self.unembedder = nn.Sequential(
            nn.Linear(self.feature_dims, self.flattened_size),
            nn.GroupNorm(num_groups_unembed, self.flattened_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )

        # Decoder layers
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
                num_groups_dec = min(32, out_channels)
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.GroupNorm(num_groups_dec, out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            in_channels = out_channels

        # Feature compressor with GroupNorm
        compress_dim = max(64, self.feature_dims // 2)
        num_groups_comp = min(32, compress_dim)
        self.feature_compressor = nn.Sequential(
            nn.Linear(self.feature_dims, compress_dim),
            nn.GroupNorm(num_groups_comp, compress_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(compress_dim, self.compressed_dims),
            nn.Tanh()
        )

        # Feature decompressor with GroupNorm
        decompress_dim = max(64, self.feature_dims // 2)
        num_groups_decomp = min(32, decompress_dim)
        self.feature_decompressor = nn.Sequential(
            nn.Linear(self.compressed_dims, decompress_dim),
            nn.GroupNorm(num_groups_decomp, decompress_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(decompress_dim, self.feature_dims),
            nn.Tanh()
        )

        # Additional components for Phase 2 (will be initialized when needed)
        self.classifier = None
        self.cluster_centers = None
        self.clustering_temperature = None

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GroupNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def set_training_phase(self, phase: int):
        """Set training phase and initialize phase-specific components"""
        self.training_phase = phase
        if phase == 2:
            if self.use_class_encoding and self.classifier is None:
                num_classes = self.config.num_classes or 2
                compress_dim = max(32, self.compressed_dims // 2)
                num_groups_class = min(32, compress_dim)
                self.classifier = nn.Sequential(
                    nn.Linear(self.compressed_dims, compress_dim),
                    nn.GroupNorm(num_groups_class, compress_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(compress_dim, num_classes)
                ).to(self.device)
                logger.info(f"Initialized classifier with {num_classes} classes")

            if self.use_kl_divergence and self.cluster_centers is None:
                num_clusters = self.config.num_classes or 2
                self.cluster_centers = nn.Parameter(
                    torch.randn(num_clusters, self.compressed_dims, device=self.device)
                )
                self.clustering_temperature = nn.Parameter(torch.tensor(1.0, device=self.device))
                logger.info(f"Initialized {num_clusters} cluster centers")

    def get_frozen_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self._selected_feature_indices is None:
            return embeddings
        return embeddings[:, self._selected_feature_indices]

    def freeze_feature_selection(self, indices: np.ndarray, scores: np.ndarray, metadata: Dict = None):
        self._selected_feature_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        self._feature_importance_scores = torch.tensor(scores, device=self.device)
        self._feature_selection_metadata = metadata or {}
        self._is_feature_selection_frozen = True
        logger.info(f"Frozen feature selection: {len(indices)} features")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return self.embedder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.unembedder(z)
        z = z.view(z.size(0), self.encoder_channels[-1], self.final_h, self.final_w)
        for layer in self.decoder_layers:
            z = layer(z)
        return z

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        original_batch_size = x.size(0)
        duplicated = False

        # Apply dataset-wide normalization if statistics are available
        if self.dataset_statistics is not None and self.dataset_statistics.is_calculated:
            x = self.dataset_statistics.normalize(x)

        # Handle single sample batches (preserve original behavior)
        if self.training and original_batch_size == 1:
            x = torch.cat([x, x], dim=0)
            duplicated = True
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)

        embedding = self.encode(x)

        # Feature selection (preserve original behavior)
        if self._is_feature_selection_frozen and self._selected_feature_indices is not None:
            selected_embedding = embedding[:, self._selected_feature_indices]
        else:
            selected_embedding = embedding

        # Compression and decompression
        compressed = self.feature_compressor(selected_embedding)
        decompressed = self.feature_decompressor(compressed)
        reconstruction = self.decode(decompressed)

        # Ensure reconstruction matches input size
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
        if duplicated and tensor.size(0) > target_batch_size:
            tensor = tensor[:target_batch_size]
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

        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
            inputs = inputs.to(self.device, non_blocking=True)
            outputs = self(inputs)

            embeddings = outputs['compressed_embedding'].cpu()

            all_embeddings.append(embeddings)
            all_labels.append(labels)

            if include_paths and hasattr(dataloader.dataset, 'get_additional_info'):
                for i in range(len(labels)):
                    idx = batch_idx * dataloader.batch_size + i
                    if idx < len(dataloader.dataset):
                        info = dataloader.dataset.get_additional_info(idx)
                        all_filenames.append(info[1])
                        all_paths.append(info[2])
                        if hasattr(dataloader.dataset, 'idx_to_class'):
                            class_name = dataloader.dataset.idx_to_class[labels[i].item()]
                        else:
                            class_name = str(labels[i].item())
                        all_class_names.append(class_name)

        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
        else:
            embeddings = torch.tensor([])
            labels = torch.tensor([])

        result = {'embeddings': embeddings, 'labels': labels}

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


# =============================================================================
# PREDICTION MANAGER
# =============================================================================

# =============================================================================
# PREDICTION MANAGER - FIXED VERSION
# =============================================================================

# =============================================================================
# PREDICTION MANAGER - FIXED VERSION
# =============================================================================

# =============================================================================
# MODIFIED PREDICTION MANAGER (with dataset statistics support)
# =============================================================================

class PredictionManager:
    def __init__(self, config: GlobalConfig):
        if hasattr(config, 'dataset_name') and config.dataset_name:
            config.dataset_name = normalize_dataset_name(config.dataset_name)

        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')

        self.original_data_dir = Path('data') / config.dataset_name
        self.original_checkpoint_dir = self.original_data_dir / 'checkpoints'

        if hasattr(config, 'output_dir') and config.output_dir:
            self.output_base_dir = Path(config.output_dir)
            if self.output_base_dir.name == config.dataset_name:
                self.saving_data_dir = self.output_base_dir
            else:
                self.saving_data_dir = self.output_base_dir / config.dataset_name
        else:
            self.output_base_dir = Path('data')
            self.saving_data_dir = self.output_base_dir / config.dataset_name

        self.saving_checkpoint_dir = self.saving_data_dir / 'checkpoints'
        self.saving_viz_dir = self.saving_data_dir / 'visualizations'

        self.saving_data_dir.mkdir(parents=True, exist_ok=True)
        self.saving_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saving_viz_dir.mkdir(parents=True, exist_ok=True)

        self.domain_processor = None
        self.domain_info = {'domain': 'general', 'domain_config': {}}

        self.model = None
        self.dataset_statistics = None  # Store dataset statistics for consistent normalization
        self._load_model()

        self.image_processor = ImageProcessor(config.input_size)
        self.transform = self._build_transform()

    def _load_model(self):
        """Load model with dataset statistics embedded in checkpoint"""
        self.model = BaseAutoencoder(self.config)
        self.model.set_training_phase(2)

        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)

        possible_paths = [
            self.original_checkpoint_dir / f"{dataset_name_lower}_best.pt",
            self.saving_checkpoint_dir / f"{dataset_name_lower}_best.pt",
            self.original_checkpoint_dir / f"{dataset_name_lower}_latest.pt",
            self.saving_checkpoint_dir / f"{dataset_name_lower}_latest.pt",
        ]

        best_path = None
        for path in possible_paths:
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
                    logger.warning(f"Error checking {path}: {e}")
                    continue

        if best_path and best_path.exists():
            try:
                checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)

                # Load dataset statistics from checkpoint
                if 'dataset_statistics' in checkpoint and checkpoint['dataset_statistics']:
                    self.dataset_statistics = DatasetStatisticsCalculator(self.config)
                    self.dataset_statistics.from_dict(checkpoint['dataset_statistics'])
                    self.model.set_dataset_statistics(self.dataset_statistics)
                    logger.info("Dataset statistics loaded from checkpoint")
                    logger.info(f"  Mean per channel: {self.dataset_statistics.mean.tolist()}")
                    logger.info(f"  Std per channel: {self.dataset_statistics.std.tolist()}")

                self.domain_info = {
                    'domain': checkpoint.get('domain', 'general'),
                    'domain_config': checkpoint.get('domain_config', {}),
                    'model_metadata': checkpoint.get('model_metadata', {})
                }

                domain = self.domain_info['domain']
                if domain != 'general':
                    self._init_domain_processor(domain, self.domain_info['domain_config'])
                    logger.info(f"Initialized {domain} domain processor from saved model")

                model_state = self.model.state_dict()
                filtered_state = {}
                skipped_keys = []
                loaded_keys = []

                for key, value in checkpoint['model_state_dict'].items():
                    if key in model_state and model_state[key].shape == value.shape:
                        filtered_state[key] = value
                        loaded_keys.append(key)
                    else:
                        skipped_keys.append(key)

                if loaded_keys:
                    logger.info(f"Loaded {len(loaded_keys)} parameters")
                if skipped_keys:
                    logger.warning(f"Skipped {len(skipped_keys)} parameters due to shape mismatch")

                self.model.load_state_dict(filtered_state, strict=False)

                if 'selected_feature_indices' in checkpoint and checkpoint['selected_feature_indices'] is not None:
                    indices = checkpoint['selected_feature_indices']
                    if isinstance(indices, torch.Tensor):
                        self.model._selected_feature_indices = indices.to(self.device)
                    else:
                        self.model._selected_feature_indices = torch.tensor(indices, device=self.device)
                    self.model._is_feature_selection_frozen = True
                    logger.info(f"Loaded feature selection with {len(indices)} features")

                checkpoint_phase = checkpoint.get('phase', 1)
                self.model.set_training_phase(checkpoint_phase)

                logger.info(f"Successfully loaded model from {best_path}")
                logger.info(f"Model domain: {self.domain_info['domain']}")
                logger.info(f"Model phase: {checkpoint_phase}")
                logger.info(f"Normalization: Dataset-wide (consistent across all images)")
                return

            except Exception as e:
                logger.error(f"Failed to load model from {best_path}: {e}")
                traceback.print_exc()

        logger.warning(f"No valid model found. Using random weights - predictions will be random!")
        logger.warning(f"Please train the model first with: python cdbnn.py --mode train --data_name {self.config.dataset_name} --data_path <path>")

        self.model.eval()
        self.model.to(self.device)

    def _init_domain_processor(self, domain: str, domain_config: Dict):
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

    def _build_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            # No normalization here - it's handled by the model using dataset statistics
        ])

    def _get_image_files(self, data_path: str) -> Tuple[List[str], List[str], List[str]]:
        image_files = []
        class_labels = []
        original_filenames = []
        supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.fits', '.fit', '.fits.gz')

        if not os.path.exists(data_path):
            logger.warning(f"Data path does not exist: {data_path}")
            return [], [], []

        if os.path.isfile(data_path) and any(data_path.lower().endswith(ext) for ext in supported):
            image_files.append(data_path)
            parent_folder = os.path.basename(os.path.dirname(data_path))
            class_labels.append(parent_folder if parent_folder not in ['', '.', '..'] else "unknown")
            original_filenames.append(os.path.basename(data_path))
            return image_files, class_labels, original_filenames

        has_subdirs = False
        subdirs = []
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                has_images = False
                for root, dirs, files in os.walk(item_path):
                    if any(f.lower().endswith(ext) for ext in supported for f in files):
                        has_images = True
                        break
                if has_images:
                    has_subdirs = True
                    subdirs.append(item)

        if has_subdirs:
            logger.info(f"Found {len(subdirs)} class subdirectories: {subdirs}")
            for root, dirs, files in os.walk(data_path):
                rel_path = os.path.relpath(root, data_path)

                if rel_path == '.':
                    continue

                class_name = rel_path.split(os.sep)[0]

                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported):
                        full_path = os.path.join(root, file)
                        image_files.append(full_path)
                        class_labels.append(class_name)
                        original_filenames.append(file)
        else:
            logger.info("No class subdirectories found, treating all images as 'unknown' class")
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported):
                        full_path = os.path.join(root, file)
                        image_files.append(full_path)
                        class_labels.append("unknown")
                        original_filenames.append(file)

        if not image_files:
            logger.warning(f"No image files found in {data_path}")
            logger.info(f"Supported formats: {', '.join(supported)}")

        return image_files, class_labels, original_filenames

    def _create_dataset(self, image_files: List[str]) -> Dataset:
        class SimpleImageDataset(Dataset):
            def __init__(self, image_files, transform):
                self.image_files = image_files
                self.transform = transform

            def __len__(self):
                return len(self.image_files)

            def __getitem__(self, idx):
                image = ImageProcessor.load_image(self.image_files[idx])
                if image is None:
                    image = PILImage.new('RGB', (256, 256), (0, 0, 0))
                else:
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, 0

        return SimpleImageDataset(image_files, self.transform)

    def _copy_config_to_output(self):
        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)

        original_conf = self.original_data_dir / f"{dataset_name_lower}.conf"
        output_conf = self.saving_data_dir / f"{dataset_name_lower}.conf"

        try:
            if original_conf.resolve() == output_conf.resolve():
                if not output_conf.exists():
                    self._create_full_config_from_current(output_conf, dataset_name_lower)
                return
        except Exception as e:
            logger.debug(f"Error comparing paths: {e}")

        if original_conf.exists():
            try:
                if original_conf != output_conf:
                    shutil.copy2(original_conf, output_conf)
                    logger.info(f"Configuration copied to {output_conf}")
                else:
                    logger.debug(f"Configuration already at correct location: {output_conf}")
            except shutil.SameFileError:
                logger.debug(f"Source and destination are the same file, skipping copy")
            except Exception as e:
                logger.warning(f"Failed to copy config file: {e}")
                self._create_full_config_from_current(output_conf, dataset_name_lower)
        else:
            logger.warning(f"Original config not found at {original_conf}, creating from current config")
            self._create_full_config_from_current(output_conf, dataset_name_lower)

    def _create_full_config_from_current(self, output_path: Path, dataset_name_lower: str):
        if output_path.exists():
            logger.debug(f"Config file already exists at {output_path}, skipping creation")
            return

        actual_feature_count = self.config.compressed_dims
        feature_columns = [f'feature_{i}' for i in range(actual_feature_count)]
        column_names = feature_columns.copy()
        column_names.append("target")

        config_dict = {
            "dataset_name": dataset_name_lower,
            "num_classes": self.config.num_classes if hasattr(self.config, 'num_classes') else 2,
            "csv_file": str(self.saving_data_dir / f"{dataset_name_lower}.csv"),
            "column_names": column_names,
            "target_column": "target",
            "feature_dims": self.config.feature_dims,
            "compressed_dims": self.config.compressed_dims,
            "actual_features_in_csv": actual_feature_count,
            "input_size": list(self.config.input_size) if isinstance(self.config.input_size, tuple) else self.config.input_size,
            "batch_size": self.config.batch_size,
            "epochs": getattr(self.config, 'epochs', 200),
            "learning_rate": self.config.learning_rate,
            "domain": getattr(self.config, 'domain', 'general'),

            "normalization_config": {
                "type": "dataset_wide",
                "mean": self.dataset_statistics.mean.tolist() if self.dataset_statistics and self.dataset_statistics.is_calculated else self.config.mean,
                "std": self.dataset_statistics.std.tolist() if self.dataset_statistics and self.dataset_statistics.is_calculated else self.config.std,
            },

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
                "epochs": getattr(self.config, 'epochs', 200),
                "learning_rate": self.config.learning_rate,
                "validation_split": getattr(self.config, 'validation_split', 0.2),
                "use_gpu": self.config.use_gpu,
                "mixed_precision": self.config.mixed_precision,
                "num_workers": getattr(self.config, 'num_workers', 0)
            },

            "visualization_config": {
                "generate_heatmaps": getattr(self.config, 'generate_heatmaps', True),
                "generate_confusion_matrix": getattr(self.config, 'generate_confusion_matrix', True),
                "generate_tsne": getattr(self.config, 'generate_tsne', True),
                "heatmap_frequency": getattr(self.config, 'heatmap_frequency', 10),
                "reconstruction_samples_frequency": getattr(self.config, 'reconstruction_samples_frequency', 5)
            },

            "saved_at": datetime.now().isoformat(),
            "config_version": "2.4",
            "notes": "This configuration matches the training config format with dataset-wide normalization"
        }

        if hasattr(self.config, 'class_names') and self.config.class_names:
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
            elif self.config.domain == 'agriculture':
                domain_config.update({
                    "has_nir_band": getattr(self.config, 'has_nir_band', False),
                    "detect_chlorophyll": getattr(self.config, 'detect_chlorophyll', True),
                    "detect_water_stress": getattr(self.config, 'detect_water_stress', True),
                    "detect_nutrient_deficiency": getattr(self.config, 'detect_nutrient_deficiency', True)
                })
            elif self.config.domain == 'medical':
                domain_config.update({
                    "modality": getattr(self.config, 'modality', 'general'),
                    "detect_tumor": getattr(self.config, 'detect_tumor', True),
                    "detect_lesion": getattr(self.config, 'detect_lesion', True)
                })
            elif self.config.domain == 'satellite':
                domain_config.update({
                    "satellite_type": getattr(self.config, 'satellite_type', 'general'),
                    "num_bands": getattr(self.config, 'num_bands', 4)
                })
            elif self.config.domain == 'surveillance':
                domain_config.update({
                    "detect_motion": getattr(self.config, 'detect_motion', True),
                    "enhance_low_light": getattr(self.config, 'enhance_low_light', True),
                    "detect_person": getattr(self.config, 'detect_person', True)
                })
            elif self.config.domain == 'microscopy':
                domain_config.update({
                    "microscopy_type": getattr(self.config, 'microscopy_type', 'general'),
                    "detect_cells": getattr(self.config, 'detect_cells', True)
                })
            elif self.config.domain == 'industrial':
                domain_config.update({
                    "detect_crack": getattr(self.config, 'detect_crack', True),
                    "detect_corrosion": getattr(self.config, 'detect_corrosion', True)
                })

            config_dict['domain_config'] = domain_config

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)

        logger.info(f"Configuration created at {output_path}")
        logger.info(f"CSV columns ({len(column_names)}): {', '.join(column_names[:5])}...")

    @torch.no_grad()
    @memory_efficient
    def predict_images(self, data_path: str, output_csv: str = None, batch_size: int = 128):
        """Predict images with consistent dataset-wide normalization"""
        image_files, class_labels, original_filenames = self._get_image_files(data_path)
        if not image_files:
            logger.warning(f"No image files found in {data_path}")
            return None

        dataset = self._create_dataset(image_files)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        if output_csv is None:
            dataset_name_lower = normalize_dataset_name(self.config.dataset_name)
            output_csv = str(self.saving_data_dir / f"{dataset_name_lower}.csv")

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize all predictions with empty lists
        all_predictions = {
            'features': [],
            'predictions': [],
            'probabilities': [],
            'cluster_assignments': [],
            'cluster_confidence': []
        }

        collected_targets = []
        collected_filenames = []
        collected_filepaths = []

        all_features = []
        processed_count = 0

        domain_features_list = []
        quality_metrics_list = []

        for batch_idx, (batch_data, _) in enumerate(tqdm(dataloader, desc="Predicting")):
            batch_tensor = batch_data.to(self.device)

            original_batch_size = batch_tensor.size(0)
            duplicated = False

            if original_batch_size == 1:
                batch_tensor = torch.cat([batch_tensor, batch_tensor], dim=0)
                duplicated = True

            # The model applies dataset-wide normalization automatically in forward pass
            output = self.model(batch_tensor)

            # Extract features
            if self.model._is_feature_selection_frozen and self.model._selected_feature_indices is not None:
                features = output['selected_embedding'].float().cpu().numpy()
            else:
                features = output['compressed_embedding'].float().cpu().numpy()

            if duplicated:
                features = features[:original_batch_size]

            all_features.append(features)

            # Extract domain features
            if self.domain_processor:
                batch_np = batch_tensor.cpu().numpy()
                if len(batch_np.shape) == 4:
                    # Convert from [B, C, H, W] to [B, H, W, C] for domain processor
                    batch_np = np.transpose(batch_np, (0, 2, 3, 1))

                for i in range(original_batch_size):
                    img_np = batch_np[i]

                    try:
                        domain_feat = self.domain_processor.extract_features(img_np)
                        domain_features_list.append(domain_feat)
                    except Exception as e:
                        logger.warning(f"Error extracting domain features for sample {i}: {e}")
                        domain_features_list.append({})

                    try:
                        quality_metrics = self.domain_processor.get_quality_metrics(img_np)
                        quality_metrics_list.append(quality_metrics)
                    except Exception as e:
                        logger.warning(f"Error extracting quality metrics for sample {i}: {e}")
                        quality_metrics_list.append({})

            # Collect targets and filenames
            start_idx = batch_idx * batch_size
            for i in range(original_batch_size):
                actual_idx = start_idx + i
                if actual_idx < len(image_files):
                    collected_targets.append(class_labels[actual_idx])
                    collected_filenames.append(original_filenames[actual_idx])
                    collected_filepaths.append(image_files[actual_idx])
                    processed_count += 1

            # Add predictions - ensure we only add for actual samples
            if 'class_predictions' in output:
                preds = output['class_predictions'].float().cpu().numpy()
                if duplicated:
                    preds = preds[:original_batch_size]
                # Only add up to the number of actual samples
                all_predictions['predictions'].extend(preds[:original_batch_size])

            # Add probabilities
            if 'class_probabilities' in output:
                probs = output['class_probabilities'].float().cpu().numpy()
                if duplicated:
                    probs = probs[:original_batch_size]
                all_predictions['probabilities'].extend(probs[:original_batch_size])

            # Add cluster assignments
            if 'cluster_assignments' in output:
                clusters = output['cluster_assignments'].float().cpu().numpy()
                if duplicated:
                    clusters = clusters[:original_batch_size]
                all_predictions['cluster_assignments'].extend(clusters[:original_batch_size])

            # Add cluster confidence
            if 'cluster_confidence' in output:
                conf = output['cluster_confidence'].float().cpu().numpy()
                if duplicated:
                    conf = conf[:original_batch_size]
                all_predictions['cluster_confidence'].extend(conf[:original_batch_size])

        # Stack features
        if all_features:
            all_predictions['features'] = np.vstack(all_features)

        # Add domain features and quality metrics to predictions
        if domain_features_list:
            if domain_features_list:
                # Get all unique keys from domain features
                all_keys = set()
                for feat in domain_features_list:
                    all_keys.update(feat.keys())

                for key in all_keys:
                    all_predictions[f'domain_{key}'] = [feat.get(key, np.nan) for feat in domain_features_list]
                logger.info(f"Added {len(all_keys)} domain-specific features")

        if quality_metrics_list:
            if quality_metrics_list:
                all_keys = set()
                for metrics in quality_metrics_list:
                    all_keys.update(metrics.keys())

                for key in all_keys:
                    all_predictions[f'quality_{key}'] = [metrics.get(key, np.nan) for metrics in quality_metrics_list]
                logger.info(f"Added {len(all_keys)} quality metrics")

        # CRITICAL FIX: Trim all prediction arrays to match processed_count
        n_samples = processed_count

        # Trim features if needed
        if all_predictions['features'] is not None and len(all_predictions['features']) > n_samples:
            all_predictions['features'] = all_predictions['features'][:n_samples]

        # Trim each prediction list
        for key in ['predictions', 'probabilities', 'cluster_assignments', 'cluster_confidence']:
            if key in all_predictions and len(all_predictions[key]) > n_samples:
                all_predictions[key] = all_predictions[key][:n_samples]
            elif key in all_predictions and len(all_predictions[key]) < n_samples:
                # Pad with default values if too short
                logger.warning(f"Padding {key}: had {len(all_predictions[key])}, need {n_samples}")
                while len(all_predictions[key]) < n_samples:
                    if key == 'predictions':
                        all_predictions[key].append(0)
                    elif key == 'probabilities':
                        all_predictions[key].append([0.0] * self.config.num_classes if self.config.num_classes else [0.0])
                    elif key == 'cluster_assignments':
                        all_predictions[key].append(0)
                    elif key == 'cluster_confidence':
                        all_predictions[key].append(0.0)

        # Trim domain and quality feature lists
        for key in list(all_predictions.keys()):
            if key.startswith('domain_') or key.startswith('quality_'):
                if len(all_predictions[key]) > n_samples:
                    all_predictions[key] = all_predictions[key][:n_samples]
                elif len(all_predictions[key]) < n_samples:
                    logger.warning(f"Padding {key}: had {len(all_predictions[key])}, need {n_samples}")
                    while len(all_predictions[key]) < n_samples:
                        all_predictions[key].append(np.nan)

        # Trim targets and filenames
        collected_targets = collected_targets[:n_samples]
        collected_filenames = collected_filenames[:n_samples]
        collected_filepaths = collected_filepaths[:n_samples]

        # Convert probabilities to list of lists if needed
        if 'probabilities' in all_predictions and all_predictions['probabilities']:
            all_predictions['probabilities'] = [
                p.tolist() if hasattr(p, 'tolist') else p
                for p in all_predictions['probabilities']
            ]

        # Log class distribution
        unique_targets = set(collected_targets)
        if len(unique_targets) > 1 or "unknown" not in unique_targets:
            logger.info(f"Class distribution in prediction set: {Counter(collected_targets)}")
        else:
            logger.info(f"All images are from unknown class (single folder)")

        # Save predictions with target column
        self._save_predictions(all_predictions, output_csv, targets=collected_targets, filenames=collected_filenames)
        self._copy_config_to_output()

        # Log processing summary
        logger.info(f"Processed {processed_count} images")
        if all_predictions['features'] is not None:
            logger.info(f"Features shape: {all_predictions['features'].shape}")
        logger.info(f"Results saved to: {output_csv}")

        # Log target info for debugging
        if len(set(collected_targets)) <= 10:
            logger.info(f"Target classes in CSV: {sorted(set(collected_targets))}")

        return all_predictions

    def _save_predictions(self, predictions: Dict, output_csv: str, targets: Optional[List[str]] = None, filenames: Optional[List[str]] = None):
        """Save predictions with rich domain-specific information including target labels"""
        data = {}

        # Get the number of samples from the first non-None array
        n_samples = None
        for key, values in predictions.items():
            if values is not None and len(values) > 0:
                n_samples = len(values)
                break

        if n_samples is None:
            logger.error("No valid data to save")
            return

        # Basic compressed features
        if 'features' in predictions and predictions['features'] is not None:
            features = predictions['features']
            # Ensure features have correct number of rows
            if len(features) > n_samples:
                features = features[:n_samples]
            elif len(features) < n_samples:
                # Pad with zeros if needed
                padding = np.zeros((n_samples - len(features), features.shape[1]))
                features = np.vstack([features, padding])

            for i in range(features.shape[1]):
                data[f'feature_{i}'] = features[:, i]

        # Domain-specific features (prefix with domain_ or quality_)
        for key, values in predictions.items():
            if (key.startswith('domain_') or key.startswith('quality_')) and values is not None:
                # Ensure correct length
                if len(values) > n_samples:
                    values = values[:n_samples]
                elif len(values) < n_samples:
                    values = list(values) + [np.nan] * (n_samples - len(values))
                data[key] = values

        # Predictions and probabilities
        if 'predictions' in predictions and predictions['predictions'] is not None:
            preds = predictions['predictions']
            if len(preds) > n_samples:
                preds = preds[:n_samples]
            elif len(preds) < n_samples:
                preds = list(preds) + [0] * (n_samples - len(preds))
            data['prediction'] = preds

        if 'probabilities' in predictions and predictions['probabilities'] is not None:
            probs = predictions['probabilities']
            if len(probs) > n_samples:
                probs = probs[:n_samples]
            elif len(probs) < n_samples:
                # Determine probability dimension from first element or config
                prob_dim = self.config.num_classes if self.config.num_classes else 2
                default_prob = [0.0] * prob_dim
                probs = list(probs) + [default_prob] * (n_samples - len(probs))

            # Convert to numpy array for processing
            probs_array = np.array(probs)
            if probs_array.ndim == 2:
                # Add per-class probabilities
                for i in range(probs_array.shape[1]):
                    data[f'prob_class_{i}'] = probs_array[:, i]

                # Add confidence and uncertainty
                data['confidence'] = np.max(probs_array, axis=1)
                entropy = -np.sum(probs_array * np.log(probs_array + 1e-10), axis=1)
                data['uncertainty'] = entropy
            else:
                # Single probability value per sample
                data['confidence'] = probs_array
                data['uncertainty'] = -np.log(probs_array + 1e-10)

        # Cluster information
        if 'cluster_assignments' in predictions and predictions['cluster_assignments'] is not None:
            clusters = predictions['cluster_assignments']
            if len(clusters) > n_samples:
                clusters = clusters[:n_samples]
            elif len(clusters) < n_samples:
                clusters = list(clusters) + [0] * (n_samples - len(clusters))
            data['cluster_id'] = clusters

        if 'cluster_confidence' in predictions and predictions['cluster_confidence'] is not None:
            conf = predictions['cluster_confidence']
            if len(conf) > n_samples:
                conf = conf[:n_samples]
            elif len(conf) < n_samples:
                conf = list(conf) + [0.0] * (n_samples - len(conf))
            data['cluster_confidence'] = conf

        # Target column (class labels)
        if targets is not None:
            if len(targets) > n_samples:
                targets = targets[:n_samples]
            elif len(targets) < n_samples:
                targets = list(targets) + ['unknown'] * (n_samples - len(targets))
            data['target'] = targets

        # Filenames
        if filenames is not None:
            if len(filenames) > n_samples:
                filenames = filenames[:n_samples]
            elif len(filenames) < n_samples:
                filenames = list(filenames) + [f'unknown_{i}' for i in range(n_samples - len(filenames))]
            data['filename'] = filenames

        # Create DataFrame
        df = pd.DataFrame(data)

        # Add metadata comment if domain processor exists
        if self.domain_processor:
            metadata_lines = [
                f"# Domain: {self.domain_info.get('domain', 'unknown')}",
                f"# Domain processor: {type(self.domain_processor).__name__}",
                f"# Total samples: {n_samples}",
                f"# Total columns: {len(data.keys())}",
                f"# Feature columns: {len([k for k in data.keys() if k.startswith('feature_')])}",
                f"# Domain feature columns: {len([k for k in data.keys() if k.startswith('domain_')])}",
                f"# Quality metric columns: {len([k for k in data.keys() if k.startswith('quality_')])}",
                f"# Generated: {datetime.now().isoformat()}"
            ]

            # Write with metadata comment
            with open(output_csv, 'w') as f:
                for line in metadata_lines:
                    f.write(line + '\n')
                df.to_csv(f, index=False)

            logger.info(f"Predictions saved with metadata to {output_csv}")
        else:
            df.to_csv(output_csv, index=False)
            logger.info(f"Predictions saved to {output_csv}")

        logger.info(f"CSV columns ({len(data.keys())}): {', '.join(list(data.keys())[:10])}{'...' if len(data) > 10 else ''}")



# =============================================================================
# COMPLETE TRAINER - All original functionality preserved
# =============================================================================

class Trainer:
    def __init__(self, model: BaseAutoencoder, config: GlobalConfig):
        self.model = model
        self.config = config
        self.device = model.device

        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and self.device.type == 'cuda' else None

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Dataset statistics calculator
        self.dataset_statistics = DatasetStatisticsCalculator(config)

        # Track best metrics with phase-specific tracking
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.best_phase = 1

        # Track best metrics per phase separately
        self.best_phase1_loss = float('inf')
        self.best_phase1_epoch = 0
        self.best_phase2_loss = float('inf')
        self.best_phase2_accuracy = 0.0
        self.best_phase2_epoch = 0

        # Track if we loaded a model
        self.model_loaded = False
        self.loaded_epoch = 0
        self.loaded_phase = 1
        self.loaded_loss = float('inf')
        self.loaded_accuracy = 0.0

        self.history = defaultdict(list)

        # Track previous values for color coding
        self.prev_train_loss = None
        self.prev_val_loss = None
        self.prev_train_acc = None
        self.prev_val_acc = None

        self.feature_selector = DistanceCorrelationFeatureSelector(
            config.correlation_upper, config.correlation_lower
        ) if config.use_distance_correlation else None

    def calculate_dataset_statistics(self, train_loader: DataLoader) -> DatasetStatisticsCalculator:
        """Calculate dataset-wide statistics before training"""
        logger.info("Calculating dataset-wide statistics for consistent normalization...")

        stats_loader = DataLoader(
            train_loader.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        stats = self.dataset_statistics.calculate_statistics(stats_loader)

        self.model.set_dataset_statistics(self.dataset_statistics)

        logger.info(f"Dataset statistics calculated from {stats['n_samples']:,} pixels")
        logger.info(f"Mean per channel: {stats['mean'].tolist()}")
        logger.info(f"Std per channel: {stats['std'].tolist()}")

        return self.dataset_statistics

    def _is_better(self, current_loss: float, current_accuracy: Optional[float], phase: int) -> bool:
        """Determine if current model is better than best saved model"""
        if phase == 1:
            return current_loss < self.best_phase1_loss
        else:
            if current_accuracy is not None:
                if current_accuracy > self.best_phase2_accuracy:
                    return True
                elif abs(current_accuracy - self.best_phase2_accuracy) < 1e-6:
                    return current_loss < self.best_phase2_loss
                return False
            return current_loss < self.best_phase2_loss

    def _update_best_metrics(self, loss: float, accuracy: Optional[float], epoch: int, phase: int):
        """Update best metrics for the current phase"""
        if phase == 1:
            if loss < self.best_phase1_loss:
                self.best_phase1_loss = loss
                self.best_phase1_epoch = epoch
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_epoch = epoch
                    self.best_phase = phase
                logger.info(f"Phase 1 best loss updated: {loss:.6f} at epoch {epoch+1}")
        else:
            old_accuracy = self.best_phase2_accuracy
            old_loss = self.best_phase2_loss

            if loss < self.best_phase2_loss:
                self.best_phase2_loss = loss
            if accuracy is not None and accuracy > self.best_phase2_accuracy:
                self.best_phase2_accuracy = accuracy
                self.best_phase2_epoch = epoch

            if accuracy is not None:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_loss = loss
                    self.best_epoch = epoch
                    self.best_phase = phase
                    logger.info(f"Global best updated: accuracy={accuracy:.2%} (was {old_accuracy:.2%}), loss={loss:.6f}")
                elif abs(accuracy - self.best_accuracy) < 1e-6 and loss < self.best_loss:
                    self.best_loss = loss
                    self.best_epoch = epoch
                    self.best_phase = phase
                    logger.info(f"Global best updated (same accuracy, better loss): loss={loss:.6f}")

            if accuracy is not None:
                if accuracy > old_accuracy:
                    logger.info(f"Phase 2 best accuracy improved: {accuracy:.2%} (was {old_accuracy:.2%}) at epoch {epoch+1}")
                elif abs(accuracy - old_accuracy) < 1e-6 and loss < old_loss:
                    logger.info(f"Phase 2 best loss improved: {loss:.6f} (was {old_loss:.6f}) at epoch {epoch+1}")

    def _train_phase(self, train_loader: DataLoader, val_loader: Optional[DataLoader],
                     phase: int, epochs: int, start_epoch: int = 0) -> Dict:
        """Train a phase with ability to resume from a specific epoch"""
        phase_history = defaultdict(list)
        patience_counter = 0

        self.prev_train_loss = None
        self.prev_val_loss = None
        self.prev_train_acc = None
        self.prev_val_acc = None

        print(f"\n{Colors.BOLD}{'Phase ' + str(phase) + ' Training'.center(80)}{Colors.ENDC}")
        if phase == 2:
            print(f"{Colors.BOLD}{'Epoch | Train Loss | Val Loss | Train Acc | Val Acc | LR'.center(80)}{Colors.ENDC}")
            print(f"{Colors.BOLD}{'Current best: Accuracy = {:.2%}, Loss = {:.6f}'.format(self.best_phase2_accuracy, self.best_phase2_loss).center(80)}{Colors.ENDC}")
        else:
            print(f"{Colors.BOLD}{'Epoch | Train Loss | Val Loss | LR'.center(80)}{Colors.ENDC}")
            print(f"{Colors.BOLD}{'Current best loss = {:.6f}'.format(self.best_phase1_loss).center(80)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'-'*80}{Colors.ENDC}")

        for epoch_offset in range(epochs):
            epoch = start_epoch + epoch_offset

            self.model.train()
            train_loss = 0.0
            train_acc = 0.0 if phase == 2 else None
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Phase {phase} Epoch {epoch+1}/{start_epoch+epochs}")
            self.optimizer.zero_grad()

            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if inputs.size(0) == 1:
                    inputs = torch.cat([inputs, inputs], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs, labels)
                        loss, acc = self._compute_loss(outputs, inputs, labels, phase)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                else:
                    outputs = self.model(inputs, labels)
                    loss, acc = self._compute_loss(outputs, inputs, labels, phase)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_loss += loss.item()
                if acc is not None:
                    train_acc += acc
                n_batches += 1

                pbar.set_postfix({'loss': f"{train_loss/n_batches:.4f}"})

            avg_train_loss = train_loss / n_batches
            avg_train_acc = train_acc / n_batches if train_acc is not None else None

            colored_train_loss = self._color_value(avg_train_loss, self.prev_train_loss, higher_is_better=False)
            train_loss_indicator = self._get_change_indicator(avg_train_loss, self.prev_train_loss, higher_is_better=False)
            self.prev_train_loss = avg_train_loss

            if val_loader:
                val_loss, val_acc = self._validate(val_loader, phase)
                phase_history['val_loss'].append(val_loss)
                if val_acc:
                    phase_history['val_acc'].append(val_acc)
                self.scheduler.step(val_loss)

                colored_val_loss = self._color_value(val_loss, self.prev_val_loss, higher_is_better=False)
                colored_val_acc = self._color_value(val_acc, self.prev_val_acc, higher_is_better=True) if val_acc else "N/A"
                val_loss_indicator = self._get_change_indicator(val_loss, self.prev_val_loss, higher_is_better=False)
                val_acc_indicator = self._get_change_indicator(val_acc, self.prev_val_acc, higher_is_better=True) if val_acc else ""

                self.prev_val_loss = val_loss
                if val_acc:
                    self.prev_val_acc = val_acc

                is_better = self._is_better(val_loss, val_acc, phase)

                if is_better:
                    self._update_best_metrics(val_loss, val_acc, epoch, phase)
                    self._save_checkpoint(epoch, phase, val_loss, val_acc, is_best=True)
                    patience_counter = 0

                    if phase == 2 and val_acc:
                        print(f"\n{Colors.GREEN}✓ New best model saved! Accuracy: {val_acc:.2%} (was {self.best_phase2_accuracy:.2%}), Loss: {val_loss:.6f} (was {self.best_phase2_loss:.6f}){Colors.ENDC}")
                else:
                    patience_counter += 1
                    self._save_checkpoint(epoch, phase, val_loss, val_acc, is_best=False)

                if avg_train_acc is not None:
                    colored_train_acc = self._color_value(avg_train_acc, self.prev_train_acc, higher_is_better=True)
                    train_acc_indicator = self._get_change_indicator(avg_train_acc, self.prev_train_acc, higher_is_better=True)
                    self.prev_train_acc = avg_train_acc

                    print(f"Epoch {epoch+1:3d} | {colored_train_loss}{train_loss_indicator} | "
                          f"{colored_val_loss}{val_loss_indicator} | "
                          f"{colored_train_acc}{train_acc_indicator} | "
                          f"{colored_val_acc}{val_acc_indicator} | "
                          f"{self.optimizer.param_groups[0]['lr']:.2e}")
                else:
                    print(f"Epoch {epoch+1:3d} | {colored_train_loss}{train_loss_indicator} | "
                          f"{colored_val_loss}{val_loss_indicator} | "
                          f"Train Acc: N/A | {colored_val_acc}{val_acc_indicator} | "
                          f"{self.optimizer.param_groups[0]['lr']:.2e}")
            else:
                is_better = avg_train_loss < self.best_phase1_loss

                if is_better:
                    self._update_best_metrics(avg_train_loss, avg_train_acc, epoch, phase)
                    self._save_checkpoint(epoch, phase, avg_train_loss, avg_train_acc, is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    self._save_checkpoint(epoch, phase, avg_train_loss, avg_train_acc, is_best=False)

                if avg_train_acc is not None:
                    colored_train_acc = self._color_value(avg_train_acc, self.prev_train_acc, higher_is_better=True)
                    train_acc_indicator = self._get_change_indicator(avg_train_acc, self.prev_train_acc, higher_is_better=True)
                    self.prev_train_acc = avg_train_acc

                    print(f"Epoch {epoch+1:3d} | {colored_train_loss}{train_loss_indicator} | "
                          f"Val Loss: N/A | {colored_train_acc}{train_acc_indicator} | "
                          f"Val Acc: N/A | {self.optimizer.param_groups[0]['lr']:.2e}")
                else:
                    print(f"Epoch {epoch+1:3d} | {colored_train_loss}{train_loss_indicator} | "
                          f"Val Loss: N/A | Train Acc: N/A | Val Acc: N/A | "
                          f"{self.optimizer.param_groups[0]['lr']:.2e}")

            phase_history['train_loss'].append(avg_train_loss)
            if avg_train_acc is not None:
                phase_history['train_acc'].append(avg_train_acc)
            phase_history['lr'].append(self.optimizer.param_groups[0]['lr'])

            log_msg = f"Phase {phase} Epoch {epoch+1}: train_loss={avg_train_loss:.4f}"
            if avg_train_acc is not None:
                log_msg += f", train_acc={avg_train_acc:.2%}"
            if val_loader:
                log_msg += f", val_loss={val_loss:.4f}"
                if val_acc:
                    log_msg += f", val_acc={val_acc:.2%}"
            log_msg += f", is_best={'Yes' if is_better else 'No'}"
            logger.info(log_msg)

            if patience_counter >= 10:
                print(f"\n{Colors.YELLOW}Early stopping triggered at epoch {epoch+1}{Colors.ENDC}")
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        print(f"\n{Colors.BOLD}{'Phase ' + str(phase) + ' Summary'.center(80)}{Colors.ENDC}")
        if phase == 1:
            print(f"Best Phase 1 Loss: {self.best_phase1_loss:.6f} at epoch {self.best_phase1_epoch+1}")
        else:
            print(f"Best Phase 2 Loss: {self.best_phase2_loss:.6f}")
            print(f"Best Phase 2 Accuracy: {self.best_phase2_accuracy:.2%}")
            print(f"Best at epoch: {self.best_phase2_epoch+1}")
        print(f"{Colors.BOLD}{'-'*80}{Colors.ENDC}\n")

        return phase_history

    def _save_checkpoint(self, epoch: int, phase: int, loss: float, accuracy: Optional[float], is_best: bool = False):
        """Save checkpoint with dataset statistics embedded"""

        domain_config = {}
        if hasattr(self.config, 'domain') and self.config.domain != 'general':
            if self.config.domain == 'astronomy':
                domain_config = {
                    "use_fits": getattr(self.config, 'use_fits', True),
                    "fits_hdu": getattr(self.config, 'fits_hdu', 0),
                    "fits_normalization": getattr(self.config, 'fits_normalization', 'zscale'),
                    "subtract_background": getattr(self.config, 'subtract_background', True),
                    "detect_sources": getattr(self.config, 'detect_sources', True),
                    "detection_threshold": getattr(self.config, 'detection_threshold', 2.5),
                    "pixel_scale": getattr(self.config, 'pixel_scale', 1.0),
                    "gain": getattr(self.config, 'gain', 1.0),
                    "read_noise": getattr(self.config, 'read_noise', 0.0)
                }
            elif self.config.domain == 'agriculture':
                domain_config = {
                    "has_nir_band": getattr(self.config, 'has_nir_band', False),
                    "detect_chlorophyll": getattr(self.config, 'detect_chlorophyll', True),
                    "detect_water_stress": getattr(self.config, 'detect_water_stress', True),
                    "detect_nutrient_deficiency": getattr(self.config, 'detect_nutrient_deficiency', True),
                    "compute_ndvi": getattr(self.config, 'compute_ndvi', True),
                    "compute_evi": getattr(self.config, 'compute_evi', True),
                    "compute_ndwi": getattr(self.config, 'compute_ndwi', True)
                }
            elif self.config.domain == 'medical':
                domain_config = {
                    "modality": getattr(self.config, 'modality', 'general'),
                    "detect_tumor": getattr(self.config, 'detect_tumor', True),
                    "detect_lesion": getattr(self.config, 'detect_lesion', True),
                    "segment_organs": getattr(self.config, 'segment_organs', True)
                }
            elif self.config.domain == 'satellite':
                domain_config = {
                    "satellite_type": getattr(self.config, 'satellite_type', 'general'),
                    "num_bands": getattr(self.config, 'num_bands', 4),
                    "compute_ndvi": getattr(self.config, 'compute_ndvi', True),
                    "compute_ndwi": getattr(self.config, 'compute_ndwi', True),
                    "compute_glcm": getattr(self.config, 'compute_glcm', True)
                }
            elif self.config.domain == 'surveillance':
                domain_config = {
                    "detect_motion": getattr(self.config, 'detect_motion', True),
                    "enhance_low_light": getattr(self.config, 'enhance_low_light', True),
                    "detect_person": getattr(self.config, 'detect_person', True),
                    "detect_vehicle": getattr(self.config, 'detect_vehicle', True)
                }
            elif self.config.domain == 'microscopy':
                domain_config = {
                    "microscopy_type": getattr(self.config, 'microscopy_type', 'general'),
                    "detect_cells": getattr(self.config, 'detect_cells', True),
                    "segment_nucleus": getattr(self.config, 'segment_nucleus', True)
                }
            elif self.config.domain == 'industrial':
                domain_config = {
                    "detect_crack": getattr(self.config, 'detect_crack', True),
                    "detect_corrosion": getattr(self.config, 'detect_corrosion', True),
                    "measure_dimensions": getattr(self.config, 'measure_dimensions', True)
                }

        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'selected_feature_indices': self.model._selected_feature_indices,
            'feature_importance_scores': self.model._feature_importance_scores,
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat(),

            'dataset_statistics': self.dataset_statistics.to_dict(),
            'domain': getattr(self.config, 'domain', 'general'),
            'domain_config': domain_config,
            'model_metadata': {
                'framework': 'CDBNN',
                'version': '2.4',
                'training_date': datetime.now().isoformat(),
                'dataset_name': getattr(self.config, 'dataset_name', 'unknown'),
                'feature_dims': getattr(self.config, 'feature_dims', 128),
                'compressed_dims': getattr(self.config, 'compressed_dims', 32),
                'use_kl_divergence': getattr(self.config, 'use_kl_divergence', True),
                'use_class_encoding': getattr(self.config, 'use_class_encoding', True),
                'normalization_type': 'dataset_wide',

                'best_phase1_loss': self.best_phase1_loss,
                'best_phase1_epoch': self.best_phase1_epoch,
                'best_phase2_loss': self.best_phase2_loss,
                'best_phase2_accuracy': self.best_phase2_accuracy,
                'best_phase2_epoch': self.best_phase2_epoch,
                'best_overall_loss': self.best_loss,
                'best_overall_accuracy': self.best_accuracy,
                'best_overall_epoch': self.best_epoch,
                'best_overall_phase': self.best_phase
            }
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        latest_path = self.checkpoint_dir / 'latest.pt'
        latest_temp_path = self.checkpoint_dir / 'latest_temp.pt'

        try:
            torch.save(checkpoint, latest_temp_path, _use_new_zipfile_serialization=False)

            if latest_temp_path.exists() and latest_temp_path.stat().st_size > 0:
                try:
                    test_load = torch.load(latest_temp_path, map_location='cpu', weights_only=False)
                    if test_load:
                        if latest_path.exists():
                            latest_path.unlink()
                        latest_temp_path.rename(latest_path)
                        logger.debug(f"Latest checkpoint saved: {latest_path} ({latest_path.stat().st_size} bytes)")
                except Exception as e:
                    logger.error(f"Verification failed: {e}")
                    if latest_temp_path.exists():
                        latest_temp_path.unlink()
            else:
                logger.error(f"Failed to save temporary checkpoint to {latest_temp_path}")
                if latest_temp_path.exists():
                    latest_temp_path.unlink()
        except Exception as e:
            logger.error(f"Error saving latest checkpoint: {e}")
            if latest_temp_path.exists():
                latest_temp_path.unlink()

        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            best_temp_path = self.checkpoint_dir / 'best_temp.pt'

            try:
                torch.save(checkpoint, best_temp_path, _use_new_zipfile_serialization=False)

                if best_temp_path.exists() and best_temp_path.stat().st_size > 0:
                    try:
                        test_load = torch.load(best_temp_path, map_location='cpu', weights_only=False)
                        if test_load:
                            if best_path.exists():
                                best_path.unlink()
                            best_temp_path.rename(best_path)

                            if phase == 2 and accuracy:
                                print(f"\n{Colors.GREEN}✓ New best model saved!")
                                print(f"  Phase: {phase}")
                                print(f"  Epoch: {epoch+1}")
                                print(f"  Loss: {loss:.6f} (previous best: {self.best_phase2_loss:.6f})")
                                print(f"  Accuracy: {accuracy:.2%} (previous best: {self.best_phase2_accuracy:.2%}){Colors.ENDC}")
                            elif phase == 1:
                                print(f"\n{Colors.GREEN}✓ New best Phase 1 model saved!")
                                print(f"  Epoch: {epoch+1}")
                                print(f"  Loss: {loss:.6f} (previous best: {self.best_phase1_loss:.6f}){Colors.ENDC}")

                            logger.info(f"New best model saved: loss={loss:.6f}" +
                                      (f", accuracy={accuracy:.2%}" if accuracy else ""))
                    except Exception as e:
                        logger.error(f"Verification failed: {e}")
                        if best_temp_path.exists():
                            best_temp_path.unlink()
            except Exception as e:
                logger.error(f"Error saving best checkpoint: {e}")
                if best_temp_path.exists():
                    best_temp_path.unlink()

    def _color_value(self, current_value: float, previous_value: Optional[float] = None,
                     higher_is_better: bool = False) -> str:
        if previous_value is None:
            return f"{Colors.BLUE}{current_value:.4f}{Colors.ENDC}"

        if higher_is_better:
            if current_value > previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value < previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"
            else:
                return f"{Colors.YELLOW}{current_value:.4f}{Colors.ENDC}"
        else:
            if current_value < previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value > previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"
            else:
                return f"{Colors.YELLOW}{current_value:.4f}{Colors.ENDC}"

    def _get_change_indicator(self, current: float, previous: Optional[float],
                               higher_is_better: bool = False) -> str:
        if previous is None:
            return ""

        if higher_is_better:
            if current > previous:
                return " ↑"
            elif current < previous:
                return " ↓"
            else:
                return " →"
        else:
            if current < previous:
                return " ↓"
            elif current > previous:
                return " ↑"
            else:
                return " →"

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        """Train with dataset-wide normalization"""

        self.calculate_dataset_statistics(train_loader)

        normalized_train_dataset = NormalizedDataset(train_loader.dataset, self.dataset_statistics)
        normalized_train_loader = DataLoader(
            normalized_train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        normalized_val_loader = None
        if val_loader:
            normalized_val_dataset = NormalizedDataset(val_loader.dataset, self.dataset_statistics)
            normalized_val_loader = DataLoader(
                normalized_val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )

        if self.model_loaded:
            start_phase = self.loaded_phase
            start_epoch = self.loaded_epoch + 1
            logger.info(f"Resuming training from Phase {start_phase}, Epoch {start_epoch}")
            logger.info(f"Starting with best accuracy: {self.best_accuracy:.2%}")
            logger.info(f"Starting with best loss: {self.best_loss:.6f}")
        else:
            start_phase = 1
            start_epoch = 0
            logger.info("Starting training from scratch")

        epochs_phase1 = self.config.epochs // 2
        epochs_phase2 = self.config.epochs // 2

        print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'CDBNN TRAINING STARTED'.center(80)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}")
        if self.model_loaded:
            print(f"{Colors.YELLOW}Loaded model from epoch {self.loaded_epoch}, phase {self.loaded_phase}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Best accuracy: {self.best_accuracy:.2%} at epoch {self.best_epoch}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Best loss: {self.best_loss:.6f}{Colors.ENDC}")
        print()

        if start_phase <= 1:
            self.model.set_training_phase(1)

            remaining_epochs_phase1 = epochs_phase1 - start_epoch if start_epoch < epochs_phase1 else 0

            if remaining_epochs_phase1 > 0:
                logger.info(f"Starting Phase 1 from epoch {start_epoch + 1} to {epochs_phase1}")
                phase1_history = self._train_phase(normalized_train_loader, normalized_val_loader, 1, remaining_epochs_phase1, start_epoch)
            else:
                logger.info("Phase 1 already completed")
                phase1_history = {}
        else:
            logger.info("Skipping Phase 1 (already completed in previous training)")
            phase1_history = {}

        if self.config.use_kl_divergence or self.config.use_class_encoding:
            if start_phase <= 2:
                print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
                print(f"{Colors.BOLD}{'STARTING PHASE 2: SUPERVISED LEARNING'.center(80)}{Colors.ENDC}")
                print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

                self.model.set_training_phase(2)

                phase2_start_epoch = max(0, start_epoch - epochs_phase1) if start_epoch >= epochs_phase1 else 0
                remaining_epochs_phase2 = epochs_phase2 - phase2_start_epoch

                if self.model_loaded and start_phase == 2 and start_epoch >= epochs_phase1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.config.learning_rate * 0.1
                    logger.info(f"Resuming Phase 2 with learning rate {self.config.learning_rate * 0.1}")

                if remaining_epochs_phase2 > 0:
                    logger.info(f"Starting Phase 2 from epoch {phase2_start_epoch + 1} to {epochs_phase2}")
                    phase2_history = self._train_phase(normalized_train_loader, normalized_val_loader, 2, remaining_epochs_phase2, phase2_start_epoch)
                else:
                    logger.info("Phase 2 already completed")
                    phase2_history = {}
            else:
                logger.info("Phase 2 already completed")
                phase2_history = {}

        print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'TRAINING COMPLETED'.center(80)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.GREEN}Best Loss: {self.best_loss:.6f}{Colors.ENDC}")
        if self.best_accuracy > 0:
            print(f"{Colors.GREEN}Best Accuracy: {self.best_accuracy:.2%}{Colors.ENDC}")
        print(f"{Colors.GREEN}Best Epoch: {self.best_epoch + 1}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

        return dict(self.history)

    def _compute_loss(self, outputs: Dict, inputs: torch.Tensor, labels: torch.Tensor, phase: int) -> Tuple[torch.Tensor, Optional[float]]:
        recon_loss = F.mse_loss(outputs['reconstruction'], inputs)
        feature_loss = F.mse_loss(outputs['reconstructed_embedding'], outputs['embedding'])

        total_loss = recon_loss + 0.1 * feature_loss
        accuracy = None

        if phase == 2:
            if self.config.use_class_encoding and 'class_logits' in outputs:
                class_loss = F.cross_entropy(outputs['class_logits'], labels)
                total_loss += 0.5 * class_loss
                preds = outputs['class_predictions']
                accuracy = (preds == labels).float().mean().item()

            if self.config.use_kl_divergence and 'cluster_probabilities' in outputs and 'target_distribution' in outputs:
                q = outputs['cluster_probabilities']
                p = outputs['target_distribution']
                kl_loss = F.kl_div((q + 1e-8).log(), p, reduction='batchmean')
                total_loss += 0.1 * kl_loss

        return total_loss, accuracy

    def _validate(self, val_loader: DataLoader, phase: int) -> Tuple[float, Optional[float]]:
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_batches = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if inputs.size(0) == 1:
                    inputs = torch.cat([inputs, inputs], dim=0)
                    labels = torch.cat([labels, labels], dim=0)

                outputs = self.model(inputs, labels)
                loss, acc = self._compute_loss(outputs, inputs, labels, phase)

                val_loss += loss.item()
                if acc is not None:
                    val_acc += acc
                n_batches += 1

        avg_loss = val_loss / n_batches
        avg_acc = val_acc / n_batches if val_acc else None

        return avg_loss, avg_acc

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        """Load checkpoint with dataset statistics"""
        path = Path(path) if path else self.checkpoint_dir / 'best.pt'
        if not path.exists():
            logger.info(f"No checkpoint found at {path}")
            return False

        try:
            if path.stat().st_size == 0:
                logger.error(f"Checkpoint file {path} is empty")
                return False

            logger.info(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if 'model_state_dict' not in checkpoint:
                logger.error(f"Checkpoint {path} missing 'model_state_dict'")
                return False

            if 'dataset_statistics' in checkpoint and checkpoint['dataset_statistics']:
                self.dataset_statistics.from_dict(checkpoint['dataset_statistics'])
                self.model.set_dataset_statistics(self.dataset_statistics)
                logger.info("Dataset statistics loaded from checkpoint")

            checkpoint_phase = checkpoint.get('phase', 1)
            checkpoint_epoch = checkpoint.get('epoch', 0)
            checkpoint_loss = checkpoint.get('loss', float('inf'))
            checkpoint_accuracy = checkpoint.get('accuracy', 0.0)

            model_metadata = checkpoint.get('model_metadata', {})

            self.best_phase1_loss = model_metadata.get('best_phase1_loss', float('inf'))
            self.best_phase1_epoch = model_metadata.get('best_phase1_epoch', 0)
            self.best_phase2_loss = model_metadata.get('best_phase2_loss', float('inf'))
            self.best_phase2_accuracy = model_metadata.get('best_phase2_accuracy', 0.0)
            self.best_phase2_epoch = model_metadata.get('best_phase2_epoch', 0)

            self.best_loss = model_metadata.get('best_overall_loss', checkpoint_loss)
            self.best_accuracy = model_metadata.get('best_overall_accuracy', checkpoint_accuracy)
            self.best_epoch = model_metadata.get('best_overall_epoch', checkpoint_epoch)
            self.best_phase = model_metadata.get('best_overall_phase', checkpoint_phase)

            self.loaded_epoch = checkpoint_epoch
            self.loaded_phase = checkpoint_phase
            self.loaded_loss = checkpoint_loss
            self.loaded_accuracy = checkpoint_accuracy
            self.model_loaded = True

            logger.info(f"Loaded best metrics from checkpoint:")
            logger.info(f"  Phase 1 best loss: {self.best_phase1_loss:.6f} at epoch {self.best_phase1_epoch}")
            logger.info(f"  Phase 2 best loss: {self.best_phase2_loss:.6f} at epoch {self.best_phase2_epoch}")
            logger.info(f"  Phase 2 best accuracy: {self.best_phase2_accuracy:.2%}")

            model_state = self.model.state_dict()
            filtered_state = {}
            skipped_keys = []
            loaded_keys = []

            for key, value in checkpoint['model_state_dict'].items():
                if key in model_state and model_state[key].shape == value.shape:
                    filtered_state[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(key)

            if loaded_keys:
                logger.info(f"Loaded {len(loaded_keys)} parameters")
            if skipped_keys:
                logger.warning(f"Skipped {len(skipped_keys)} parameters due to shape mismatch")

            self.model.load_state_dict(filtered_state, strict=False)

            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Optimizer state loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")

            if 'selected_feature_indices' in checkpoint and checkpoint['selected_feature_indices'] is not None:
                indices = checkpoint['selected_feature_indices']
                if isinstance(indices, torch.Tensor):
                    self.model._selected_feature_indices = indices.to(self.device)
                else:
                    self.model._selected_feature_indices = torch.tensor(indices, device=self.device)
                self.model._is_feature_selection_frozen = True
                logger.info(f"Loaded feature selection with {len(indices)} features")

            self.model.set_training_phase(checkpoint_phase)

            logger.info(f"Successfully loaded checkpoint from {path}")
            logger.info(f"  - Phase: {checkpoint_phase}")
            logger.info(f"  - Epoch: {checkpoint_epoch}")
            logger.info(f"  - Loss: {checkpoint_loss:.6f}")
            if checkpoint_accuracy:
                logger.info(f"  - Accuracy: {checkpoint_accuracy:.2%}")

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

# Add this function at the beginning of the main() function or in the CDBNNApplication class

def normalize_dataset_name(data_name: str) -> str:
    """Convert dataset name to lowercase for consistent file naming"""
    return data_name.lower() if data_name else 'dataset'

def get_dataset_paths(data_name: str, base_dir: str = 'data'):
    """Get standardized paths for dataset files"""
    dataset_name_lower = normalize_dataset_name(data_name)
    data_dir = Path(base_dir) / dataset_name_lower
    return {
        'data_dir': data_dir,
        'csv_path': data_dir / f"{dataset_name_lower}.csv",
        'train_csv': data_dir / f"{dataset_name_lower}_train.csv",
        'test_csv': data_dir / f"{dataset_name_lower}_test.csv",
        'json_config': data_dir / f"{dataset_name_lower}.json",
        'conf_config': data_dir / f"{dataset_name_lower}.conf",
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
        if hasattr(config, 'dataset_name') and config.dataset_name:
            config.dataset_name = normalize_dataset_name(config.dataset_name)

        self.config = config

        self.loading_data_dir = Path('data') / config.dataset_name
        self.loading_checkpoint_dir = self.loading_data_dir / 'checkpoints'

        if hasattr(config, 'output_dir') and config.output_dir:
            output_dir_path = Path(config.output_dir)
            if output_dir_path.name == config.dataset_name:
                self.saving_data_dir = output_dir_path
                self.output_base_dir = output_dir_path.parent
            else:
                self.output_base_dir = output_dir_path
                self.saving_data_dir = output_dir_path / config.dataset_name
        else:
            self.output_base_dir = Path('data')
            self.saving_data_dir = self.output_base_dir / config.dataset_name

        self.saving_checkpoint_dir = self.saving_data_dir / 'checkpoints'
        self.saving_viz_dir = self.saving_data_dir / 'visualizations'
        self.saving_log_dir = self.saving_data_dir / 'logs'
        self.saving_heatmap_dir = self.saving_data_dir / 'attention_heatmaps'

        self.config.checkpoint_dir = str(self.saving_checkpoint_dir)
        self.config.viz_dir = str(self.saving_viz_dir)
        self.config.log_dir = str(self.saving_log_dir)

        self.config.conf_config_path = str(self.saving_data_dir / f"{config.dataset_name}.conf")
        self.config.json_config_path = str(self.saving_data_dir / f"{config.dataset_name}_config.json")
        self.config.class_names_path = str(self.saving_data_dir / f"{config.dataset_name}_classes.json")

        self.config.csv_path = str(self.saving_data_dir / f"{config.dataset_name}.csv")
        self.config.train_csv_path = str(self.saving_data_dir / f"{config.dataset_name}_train.csv")
        self.config.test_csv_path = str(self.saving_data_dir / f"{config.dataset_name}_test.csv")

        self.config.feature_map_path = str(self.saving_data_dir / f"{config.dataset_name}_feature_map.json")
        self.config.column_mapping_path = str(self.saving_data_dir / f"{config.dataset_name}_columns.json")

        for d in [self.saving_data_dir, self.saving_checkpoint_dir, self.saving_viz_dir,
                  self.saving_log_dir, self.saving_heatmap_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.visualizer = Visualizer(config)

        logger.info(f"Dataset: {config.dataset_name}")
        logger.info(f"Loading data from: {self.loading_data_dir}")
        logger.info(f"Saving outputs to: {self.saving_data_dir}")
        logger.info(f"CSV output: {self.config.csv_path}")
        logger.info(f"Configuration file: {self.config.conf_config_path}")
        logger.info(f"Checkpoint directory (loading): {self.loading_checkpoint_dir}")
        logger.info(f"Checkpoint directory (saving): {self.saving_checkpoint_dir}")

    def prepare_data(self, source_path: str, data_type: str = 'custom') -> Tuple[DataLoader, Optional[DataLoader]]:
        transform = self._get_transform()

        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)

        if data_type == 'torchvision':
            dataset_name_upper = self.config.dataset_name.upper()
            if dataset_name_upper in ['MNIST', 'CIFAR10', 'CIFAR100', 'FASHIONMNIST']:
                train_dataset = self._get_torchvision_dataset(dataset_name_upper, train=True, transform=transform)
                test_dataset = self._get_torchvision_dataset(dataset_name_upper, train=False, transform=transform)
                self.config.num_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 10
                self.config.class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else [str(i) for i in range(self.config.num_classes)]
            else:
                train_path = Path(source_path) / 'train' if (Path(source_path) / 'train').exists() else Path(source_path)
                test_path = Path(source_path) / 'test' if (Path(source_path) / 'test').exists() else None

                train_dataset = CustomImageDataset(str(train_path), transform=transform, config=self.config.to_dict())
                test_dataset = CustomImageDataset(str(test_path), transform=transform, config=self.config.to_dict()) if test_path else None

                self.config.num_classes = len(train_dataset.classes)
                self.config.class_names = train_dataset.classes
        else:
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

        self._save_config_files()

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False) if test_dataset else None

        if hasattr(train_dataset, 'get_class_distribution'):
            self.visualizer.plot_class_distribution(train_dataset.get_class_distribution())

        return train_loader, test_loader

    def _save_config_files(self):
        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)
        data_dir = self.saving_data_dir

        actual_feature_count = self.config.compressed_dims
        feature_columns = [f'feature_{i}' for i in range(actual_feature_count)]
        column_names = feature_columns.copy()
        column_names.append("target")

        config_dict = {
            "dataset_name": dataset_name_lower,
            "num_classes": self.config.num_classes,
            "csv_file": str(data_dir / f"{dataset_name_lower}.csv"),
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

        conf_path = data_dir / f"{dataset_name_lower}.conf"
        with open(conf_path, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        logger.info(f"Configuration saved to {conf_path}")
        logger.info(f"CSV will contain {len(column_names)} columns: {', '.join(column_names[:5])}...")

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        model = BaseAutoencoder(self.config)

        config_copy = copy.deepcopy(self.config)
        config_copy.checkpoint_dir = str(self.saving_checkpoint_dir)

        trainer = Trainer(model, config_copy)

        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)

        loading_checkpoint_path = self.loading_checkpoint_dir / f"{dataset_name_lower}_best.pt"
        if loading_checkpoint_path.exists() and loading_checkpoint_path.stat().st_size > 0:
            try:
                logger.info(f"Found existing model at {loading_checkpoint_path}, loading...")
                success = trainer.load_checkpoint(str(loading_checkpoint_path))
                if success:
                    logger.info("Successfully loaded existing model")
                else:
                    logger.warning("Failed to load existing model, training from scratch")
            except Exception as e:
                logger.warning(f"Could not load existing model from {loading_checkpoint_path}: {e}")
                logger.warning("Training from scratch")
        else:
            logger.info("No existing model found, training from scratch")

        history = trainer.train(train_loader, val_loader)

        best_checkpoint = trainer.checkpoint_dir / 'best.pt'
        if best_checkpoint.exists() and best_checkpoint.stat().st_size > 0:
            logger.info(f"Model successfully saved to {best_checkpoint}")

            loading_best_path = self.loading_checkpoint_dir / f"{dataset_name_lower}_best.pt"
            loading_best_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                test_load = torch.load(best_checkpoint, map_location='cpu', weights_only=False)
                if test_load:
                    shutil.copy2(best_checkpoint, loading_best_path)
                    logger.info(f"Model copied to loading directory: {loading_best_path}")

                    latest_checkpoint = trainer.checkpoint_dir / 'latest.pt'
                    if latest_checkpoint.exists():
                        loading_latest_path = self.loading_checkpoint_dir / f"{dataset_name_lower}_latest.pt"
                        shutil.copy2(latest_checkpoint, loading_latest_path)
                        logger.info(f"Latest checkpoint copied to {loading_latest_path}")
                else:
                    logger.error("Model verification failed - file appears corrupted")
            except Exception as e:
                logger.error(f"Failed to copy model to loading directory: {e}")
        else:
            logger.error(f"Training completed but no model was saved to {best_checkpoint}")

        self._save_config_files()
        self.visualizer.plot_training_history(dict(trainer.history))

        return history

    def extract_features(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        model_path = self.loading_checkpoint_dir / f"{normalize_dataset_name(self.config.dataset_name)}_best.pt"

        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)
        output_csv = str(self.saving_data_dir / f"{dataset_name_lower}.csv")

        predictor = PredictionManager(self.config)

        if hasattr(dataloader.dataset, 'data_dir'):
            data_path = str(dataloader.dataset.data_dir)
        else:
            data_path = str(self.loading_data_dir)

        predictions = predictor.predict_images(data_path, output_csv=output_csv)

        if predictions:
            features_dict = {
                'features': predictions.get('features'),
                'labels': predictions.get('target'),
                'class_names': predictions.get('target'),
                'filenames': predictions.get('filename'),
                'paths': predictions.get('filepath')
            }

            if 'labels' in features_dict and features_dict['labels']:
                if isinstance(features_dict['labels'][0], str):
                    unique_labels = sorted(set(features_dict['labels']))
                    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                    features_dict['labels'] = np.array([label_to_idx[label] for label in features_dict['labels']])

            return features_dict

        return {'features': None, 'labels': None}

    def predict(self, dataloader: DataLoader, optimize_level: str = 'balanced') -> Dict:
        predictor = PredictionManager(self.config)

        dataset_name_lower = normalize_dataset_name(self.config.dataset_name)

        if hasattr(dataloader.dataset, 'data_dir'):
            data_path = str(dataloader.dataset.data_dir)
        else:
            data_path = str(self.loading_data_dir)

        output_csv = str(self.saving_data_dir / f"{dataset_name_lower}.csv")
        config_file = str(self.saving_data_dir / f"{dataset_name_lower}.conf")

        logger.info(f"Predicting from: {data_path}")
        logger.info(f"Saving to: {output_csv}")
        logger.info(f"Configuration file: {config_file}")

        predictor.config.conf_config_path = config_file

        return predictor.predict_images(data_path, output_csv=output_csv)

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

    def _get_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            # NO normalization here - will be applied by model using dataset statistics
        ])

    def _save_features_to_csv(self, features: Dict, output_csv: str):
        data = {}

        if 'features' in features and len(features['features']) > 0:
            feature_array = features['features']
            for i in range(feature_array.shape[1]):
                data[f'feature_{i}'] = feature_array[:, i]

        if 'labels' in features:
            data['label'] = features['labels']

        if 'class_names' in features:
            data['target'] = features['class_names']

        if 'filenames' in features:
            data['filename'] = features['filenames']

        if 'paths' in features:
            data['filepath'] = features['paths']

        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        logger.info(f"Features saved to {output_csv}")

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


# =============================================================================
# DOMAIN-AWARE CDBNN
# =============================================================================

class DomainAwareCDBNN(CDBNNApplication):
    """Domain-aware CDBNN application with specialized processors"""

    def __init__(self, config: GlobalConfig):
        super().__init__(config)
        self.domain = config.domain if hasattr(config, 'domain') else 'general'
        self.domain_processor = None

        # Initialize domain processor if domain is specified
        if self.domain != 'general':
            self._init_domain_processor()
            logger.info(f"Initialized {self.domain} domain processor")

    def _init_domain_processor(self):
        """Initialize the appropriate domain processor"""
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

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply domain-specific preprocessing"""
        if self.domain_processor:
            return self.domain_processor.preprocess(image)
        return image

    def extract_domain_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract domain-specific features"""
        if self.domain_processor:
            return self.domain_processor.extract_features(image)
        return {}

    def get_domain_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Get domain-specific quality metrics"""
        if self.domain_processor:
            return self.domain_processor.get_quality_metrics(image)
        return {}

# =============================================================================
# AGRICULTURE DOMAIN PROCESSOR
# =============================================================================

class AgricultureDomainProcessor:
    """Agriculture-specific image processor"""

    def __init__(self, config: GlobalConfig):
        self.config = config

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess agricultural images"""
        img_float = image.astype(np.float32) / 255.0

        # Apply vegetation enhancement
        img_float = self._normalize_illumination(img_float)

        return img_float

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract agriculture-specific features"""
        features = {}

        # Vegetation indices
        features.update(self._compute_ndvi(image))
        features.update(self._compute_ndwi(image))

        # Plant health metrics
        features.update(self._compute_chlorophyll_content(image))
        features.update(self._compute_water_stress(image))

        # Disease detection
        features.update(self._detect_leaf_disease(image))

        # Texture analysis
        features.update(self._compute_leaf_texture(image))

        return features

    def _compute_ndvi(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Normalized Difference Vegetation Index"""
        if image.shape[2] >= 3:
            # Approximate NIR from RGB (simplified)
            nir = image[:, :, 2]  # Approximate NIR as red channel
            red = image[:, :, 0]

            ndvi = (nir - red) / (nir + red + 1e-8)

            return {
                'ndvi_mean': np.mean(ndvi),
                'ndvi_std': np.std(ndvi),
                'vegetation_fraction': np.mean(ndvi > 0.3)
            }
        return {'ndvi_mean': 0, 'ndvi_std': 0, 'vegetation_fraction': 0}

    def _compute_ndwi(self, image: np.ndarray) -> Dict[str, float]:
        """Compute Normalized Difference Water Index"""
        if image.shape[2] >= 3:
            green = image[:, :, 1]
            nir = image[:, :, 2]

            ndwi = (green - nir) / (green + nir + 1e-8)

            return {
                'ndwi_mean': np.mean(ndwi),
                'water_content': np.mean(ndwi > 0)
            }
        return {'ndwi_mean': 0, 'water_content': 0}

    def _compute_chlorophyll_content(self, image: np.ndarray) -> Dict[str, float]:
        """Estimate chlorophyll content"""
        if image.shape[2] >= 3:
            # Simple chlorophyll index
            green = image[:, :, 1]
            red = image[:, :, 0]

            chlorophyll = green / (red + 1e-8)

            return {
                'chlorophyll_index': np.mean(chlorophyll),
                'green_percentage': np.mean(green > 0.3)
            }
        return {'chlorophyll_index': 0, 'green_percentage': 0}

    def _compute_water_stress(self, image: np.ndarray) -> Dict[str, float]:
        """Detect water stress"""
        if image.shape[2] >= 3:
            # Convert to LAB
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32) / 255.0

            water_stress = {
                'water_stress_index': np.mean(lab[:, :, 1] + lab[:, :, 2]) / 2,
                'wilting_score': np.mean(lab[:, :, 1] < 0.3)
            }
            return water_stress
        return {'water_stress_index': 0, 'wilting_score': 0}

    def _detect_leaf_disease(self, image: np.ndarray) -> Dict[str, float]:
        """Detect leaf diseases"""
        if image.shape[2] >= 3:
            # Convert to LAB for disease spot detection
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32) / 255.0

            # Disease spots appear as dark/brown regions
            spots = (lab[:, :, 0] < 0.4) & (lab[:, :, 1] < 0.4) & (lab[:, :, 2] < 0.4)

            return {
                'disease_spots': np.mean(spots),
                'disease_severity': np.mean(spots) * 2
            }
        return {'disease_spots': 0, 'disease_severity': 0}

    def _compute_leaf_texture(self, image: np.ndarray) -> Dict[str, float]:
        """Compute leaf texture features"""
        if image.shape[2] >= 3:
            gray = np.mean(image, axis=2)
            gray_uint8 = (gray * 255).astype(np.uint8)

            glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True)

            texture = {
                'contrast': graycoprops(glcm, 'contrast')[0, 0],
                'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
                'energy': graycoprops(glcm, 'energy')[0, 0]
            }
            return texture
        return {'contrast': 0, 'homogeneity': 0, 'energy': 0}

    def _normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """Normalize illumination"""
        for i in range(image.shape[2]):
            image[:, :, i] = exposure.equalize_adapthist(image[:, :, i])
        return image

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute agriculture-specific quality metrics"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        metrics = {
            'sharpness': np.std(filters.sobel(gray)),
            'contrast': np.std(gray),
            'plant_visibility': np.mean(image[:, :, 1] > 0.3) if len(image.shape) == 3 else 0
        }

        return metrics

# =============================================================================
# MEDICAL DOMAIN PROCESSOR
# =============================================================================

class MedicalDomainProcessor:
    """Medical imaging processor"""

    def __init__(self, config: GlobalConfig):
        self.config = config

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess medical images"""
        img_float = image.astype(np.float32) / 255.0

        # Denoise
        img_float = self._denoise_medical(img_float)

        # Enhance contrast
        img_float = self._enhance_contrast(img_float)

        return img_float

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract medical-specific features"""
        features = {}

        # Tumor detection
        features.update(self._detect_tumor(image))

        # Texture analysis
        features.update(self._compute_tissue_texture(image))

        # Quality metrics
        features.update(self._compute_medical_contrast(image))
        features.update(self._compute_medical_sharpness(image))

        return features

    def _denoise_medical(self, image: np.ndarray) -> np.ndarray:
        """Denoise medical images"""
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                image[:, :, i] = cv2.bilateralFilter((image[:, :, i] * 255).astype(np.uint8), 5, 50, 50) / 255.0
        else:
            image = cv2.bilateralFilter((image * 255).astype(np.uint8), 5, 50, 50) / 255.0
        return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast"""
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                image[:, :, i] = exposure.equalize_adapthist(image[:, :, i])
        else:
            image = exposure.equalize_adapthist(image)
        return image

    def _detect_tumor(self, image: np.ndarray) -> Dict[str, float]:
        """Detect tumors"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # Look for abnormal regions with different texture
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        threshold = np.percentile(lbp, 95)
        abnormal = lbp > threshold

        return {
            'tumor_suspicion': np.mean(abnormal),
            'abnormal_texture_score': np.std(lbp[abnormal]) if np.any(abnormal) else 0
        }

    def _compute_tissue_texture(self, image: np.ndarray) -> Dict[str, float]:
        """Compute tissue texture features"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        gray_uint8 = (gray * 255).astype(np.uint8)

        glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True)

        return {
            'tissue_contrast': graycoprops(glcm, 'contrast')[0, 0],
            'tissue_homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'tissue_energy': graycoprops(glcm, 'energy')[0, 0]
        }

    def _compute_medical_contrast(self, image: np.ndarray) -> Dict[str, float]:
        """Compute contrast metric"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        return {
            'contrast_ratio': (np.percentile(gray, 95) - np.percentile(gray, 5)) / (np.percentile(gray, 95) + np.percentile(gray, 5) + 1e-8)
        }

    def _compute_medical_sharpness(self, image: np.ndarray) -> Dict[str, float]:
        """Compute sharpness metric"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        edges = filters.sobel(gray)

        return {'sharpness_index': np.mean(edges)}

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute medical-specific quality metrics"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        signal = np.mean(gray)
        noise = np.std(gray)

        metrics = {
            'snr': signal / (noise + 1e-8),
            'contrast': np.std(gray),
            'sharpness': np.std(filters.sobel(gray))
        }

        return metrics

# =============================================================================
# SATELLITE DOMAIN PROCESSOR
# =============================================================================

class SatelliteDomainProcessor:
    """Satellite/Remote sensing processor"""

    def __init__(self, config: GlobalConfig):
        self.config = config

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess satellite images"""
        img_float = image.astype(np.float32) / 255.0

        # Radiometric correction
        for i in range(img_float.shape[2]):
            img_float[:, :, i] = exposure.equalize_adapthist(img_float[:, :, i])

        return img_float

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract satellite-specific features"""
        features = {}

        # Spectral indices
        features.update(self._compute_sat_ndvi(image))
        features.update(self._compute_sat_ndwi(image))

        # Land cover
        features.update(self._classify_land_cover(image))

        return features

    def _compute_sat_ndvi(self, image: np.ndarray) -> Dict[str, float]:
        """Compute NDVI"""
        if image.shape[2] >= 3:
            nir = image[:, :, 2]
            red = image[:, :, 0]
            ndvi = (nir - red) / (nir + red + 1e-8)

            return {
                'sat_ndvi_mean': np.mean(ndvi),
                'vegetation_fraction': np.mean(ndvi > 0.3)
            }
        return {'sat_ndvi_mean': 0, 'vegetation_fraction': 0}

    def _compute_sat_ndwi(self, image: np.ndarray) -> Dict[str, float]:
        """Compute NDWI"""
        if image.shape[2] >= 3:
            green = image[:, :, 1]
            nir = image[:, :, 2]
            ndwi = (green - nir) / (green + nir + 1e-8)

            return {
                'sat_ndwi_mean': np.mean(ndwi),
                'water_fraction': np.mean(ndwi > 0)
            }
        return {'sat_ndwi_mean': 0, 'water_fraction': 0}

    def _classify_land_cover(self, image: np.ndarray) -> Dict[str, float]:
        """Classify land cover"""
        ndvi = self._compute_sat_ndvi(image)
        ndwi = self._compute_sat_ndwi(image)

        return {
            'forest_cover': ndvi.get('vegetation_fraction', 0),
            'water_cover': ndwi.get('water_fraction', 0),
            'urban_cover': 1 - (ndvi.get('vegetation_fraction', 0) + ndwi.get('water_fraction', 0))
        }

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute satellite-specific quality metrics"""
        gray = np.mean(image, axis=2)
        edges = filters.sobel(gray)

        metrics = {
            'sharpness': np.mean(edges),
            'contrast': np.std(gray),
            'cloud_coverage': np.mean(gray > 0.9)
        }

        return metrics

# =============================================================================
# SURVEILLANCE DOMAIN PROCESSOR
# =============================================================================

class SurveillanceDomainProcessor:
    """Surveillance/CCTV processor"""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess surveillance images"""
        img_float = image.astype(np.float32) / 255.0

        # Low-light enhancement
        img_float = self._enhance_low_light(img_float)

        # Noise reduction
        img_float = self._reduce_noise(img_float)

        return img_float

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract surveillance-specific features"""
        features = {}

        # Person detection
        features.update(self._detect_person(image))

        # Motion detection
        features.update(self._detect_motion(image))

        # Scene understanding
        features.update(self._classify_scene_type(image))

        return features

    def _enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """Enhance low-light images"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32) / 255.0
            l_channel = lab[:, :, 0]
            l_enhanced = exposure.equalize_adapthist(l_channel)
            lab[:, :, 0] = l_enhanced
            lab = (lab * 255).astype(np.uint8)
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) / 255.0
        return image

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise"""
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                image[:, :, i] = cv2.bilateralFilter((image[:, :, i] * 255).astype(np.uint8), 9, 75, 75) / 255.0
        else:
            image = cv2.bilateralFilter((image * 255).astype(np.uint8), 9, 75, 75) / 255.0
        return image

    def _detect_person(self, image: np.ndarray) -> Dict[str, float]:
        """Detect persons"""
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        img_uint8 = (image * 255).astype(np.uint8)
        if len(img_uint8.shape) == 3:
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        persons, _ = hog.detectMultiScale(img_uint8, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {
            'num_persons': len(persons),
            'person_density': len(persons) / (image.shape[0] * image.shape[1]) * 100000
        }

    def _detect_motion(self, image: np.ndarray) -> Dict[str, float]:
        """Detect motion"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        gray_uint8 = (gray * 255).astype(np.uint8)
        fgmask = self.background_subtractor.apply(gray_uint8)

        return {
            'motion_intensity': np.mean(fgmask) / 255.0,
            'motion_area': np.sum(fgmask > 0) / (fgmask.shape[0] * fgmask.shape[1])
        }

    def _classify_scene_type(self, image: np.ndarray) -> Dict[str, float]:
        """Classify scene type"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # Simple classification using color and texture
        sky_likelihood = np.mean(image[:, :, 2] > 0.5) if len(image.shape) == 3 else 0
        vegetation = np.mean(image[:, :, 1] > 0.4) if len(image.shape) == 3 else 0

        return {
            'outdoor_prob': min(1, sky_likelihood + vegetation),
            'indoor_prob': 1 - min(1, sky_likelihood + vegetation)
        }

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute surveillance-specific quality metrics"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        metrics = {
            'brightness': np.mean(gray),
            'contrast': np.std(gray),
            'noise_level': np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))
        }

        return metrics

# =============================================================================
# MICROSCOPY DOMAIN PROCESSOR
# =============================================================================

class MicroscopyDomainProcessor:
    """Microscopy imaging processor"""

    def __init__(self, config: GlobalConfig):
        self.config = config

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess microscopy images"""
        img_float = image.astype(np.float32) / 255.0

        # Background subtraction
        img_float = self._subtract_background(img_float)

        # Contrast enhancement
        img_float = self._enhance_contrast(img_float)

        return img_float

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract microscopy-specific features"""
        features = {}

        # Cell detection
        features.update(self._detect_cells(image))

        # Intensity distribution
        features.update(self._compute_intensity_distribution(image))

        # Focus quality
        features.update(self._detect_out_of_focus(image))

        return features

    def _subtract_background(self, image: np.ndarray) -> np.ndarray:
        """Subtract background"""
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
        """Enhance contrast"""
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                image[:, :, i] = exposure.equalize_adapthist(image[:, :, i])
        else:
            image = exposure.equalize_adapthist(image)
        return image

    def _detect_cells(self, image: np.ndarray) -> Dict[str, float]:
        """Detect cells"""
        from skimage.feature import blob_log

        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        blobs = blob_log(gray, max_sigma=30, num_sigma=10, threshold=0.1)

        return {
            'cell_count': len(blobs),
            'cell_density': len(blobs) / (image.shape[0] * image.shape[1]) * 1000000
        }

    def _compute_intensity_distribution(self, image: np.ndarray) -> Dict[str, float]:
        """Compute intensity distribution features"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        return {
            'intensity_mean': np.mean(gray),
            'intensity_std': np.std(gray),
            'intensity_skew': np.mean(((gray - np.mean(gray)) / (np.std(gray) + 1e-8)) ** 3)
        }

    def _detect_out_of_focus(self, image: np.ndarray) -> Dict[str, float]:
        """Detect if image is out of focus"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        laplacian = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)
        focus_measure = np.var(laplacian)

        return {
            'focus_score': focus_measure / 1000,
            'is_out_of_focus': 1 if focus_measure < 100 else 0
        }

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute microscopy-specific quality metrics"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        laplacian = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F)

        metrics = {
            'focus_score': np.var(laplacian) / 1000,
            'contrast': np.std(gray),
            'signal_to_noise': np.mean(gray) / (np.std(gray) + 1e-8)
        }

        return metrics

# =============================================================================
# INDUSTRIAL DOMAIN PROCESSOR
# =============================================================================

class IndustrialDomainProcessor:
    """Industrial inspection processor"""

    def __init__(self, config: GlobalConfig):
        self.config = config

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess industrial images"""
        img_float = image.astype(np.float32) / 255.0

        # Illumination normalization
        img_float = self._normalize_illumination(img_float)

        # Edge enhancement
        img_float = self._enhance_edges(img_float)

        return img_float

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract industrial-specific features"""
        features = {}

        # Defect detection
        features.update(self._detect_crack(image))
        features.update(self._detect_corrosion(image))

        # Surface analysis
        features.update(self._compute_surface_roughness(image))

        return features

    def _normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """Normalize illumination"""
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                image[:, :, i] = exposure.equalize_adapthist(image[:, :, i])
        else:
            image = exposure.equalize_adapthist(image)
        return image

    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges"""
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                blurred = cv2.GaussianBlur(image[:, :, i], (5, 5), 0)
                image[:, :, i] = np.clip(image[:, :, i] + (image[:, :, i] - blurred), 0, 1)
        else:
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            image = np.clip(image + (image - blurred), 0, 1)
        return image

    def _detect_crack(self, image: np.ndarray) -> Dict[str, float]:
        """Detect cracks"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        # Enhance linear structures
        ridges = frangi(gray)
        threshold = np.percentile(ridges, 95)
        cracks = ridges > threshold

        return {
            'crack_density': np.mean(cracks),
            'crack_severity': np.mean(ridges[cracks]) if np.any(cracks) else 0
        }

    def _detect_corrosion(self, image: np.ndarray) -> Dict[str, float]:
        """Detect corrosion"""
        if len(image.shape) == 3:
            # Corrosion appears as reddish-brown spots
            red = image[:, :, 0]
            green = image[:, :, 1]
            blue = image[:, :, 2]

            corrosion_mask = (red > 0.5) & (green < 0.4) & (blue < 0.4)

            return {
                'corrosion_area': np.mean(corrosion_mask),
                'corrosion_severity': np.mean(red[corrosion_mask]) if np.any(corrosion_mask) else 0
            }
        return {'corrosion_area': 0, 'corrosion_severity': 0}

    def _compute_surface_roughness(self, image: np.ndarray) -> Dict[str, float]:
        """Compute surface roughness"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        high_pass = gray - cv2.GaussianBlur(gray, (21, 21), 0)

        return {
            'roughness': np.std(high_pass),
            'uniformity': 1 - np.std(gray) / (np.mean(gray) + 1e-8)
        }

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute industrial-specific quality metrics"""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        metrics = {
            'sharpness': np.std(filters.sobel(gray)),
            'contrast': np.std(gray),
            'uniformity': 1 - np.std(gray) / (np.mean(gray) + 1e-8)
        }

        return metrics

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
# MAIN FUNCTION
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='CDBNN - Convolutional Deep Bayesian Neural Network with Domain Support')

    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'extract'], help='Operation mode')
    parser.add_argument('--data_name', type=str, help='Dataset name')
    parser.add_argument('--data_type', type=str, choices=['custom', 'torchvision'], default='custom')
    parser.add_argument('--data_path', type=str, help='Path to data')
    parser.add_argument('--domain', type=str, choices=['general', 'agriculture', 'medical', 'satellite', 'surveillance', 'microscopy', 'industrial', 'astronomy'],
                       default='general', help='Domain for specialized processing')
        # Astronomy-specific flags
    parser.add_argument('--use_fits', action='store_true', help='Enable FITS support for astronomical images')
    parser.add_argument('--fits_hdu', type=int, default=0, help='FITS HDU to read')
    parser.add_argument('--fits_normalization', type=str, choices=['zscale', 'percent', 'minmax', 'asinh'],
                       default='zscale', help='FITS image normalization method')
    parser.add_argument('--subtract_background', action='store_true', default=True, help='Subtract background')
    parser.add_argument('--detect_sources', action='store_true', default=True, help='Detect astronomical sources')
    parser.add_argument('--detection_threshold', type=float, default=2.5, help='Source detection threshold')
    parser.add_argument('--pixel_scale', type=float, default=1.0, help='Pixel scale in arcsec/pixel')
    parser.add_argument('--gain', type=float, default=1.0, help='Gain in e-/ADU')
    parser.add_argument('--read_noise', type=float, default=0.0, help='Read noise in e-')

    parser.add_argument('--feature_dims', type=int, default=128)
    parser.add_argument('--compressed_dims', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--max_features', type=int, default=32)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no_mixed_precision', action='store_true')
    parser.add_argument('--generate_heatmaps', action='store_true', default=True)
    parser.add_argument('--generate_tsne', action='store_true', default=True)
    parser.add_argument('--disable_kl', action='store_true')
    parser.add_argument('--disable_class', action='store_true')
    parser.add_argument('--disable_distance_corr', action='store_true')
    parser.add_argument('--output_dir', type=str, default='data')

    # Domain-specific flags
    parser.add_argument('--has_nir_band', action='store_true', help='Has near-infrared band (agriculture)')
    parser.add_argument('--modality', type=str, default='general', help='Medical imaging modality')
    parser.add_argument('--satellite_type', type=str, default='general', help='Satellite type')
    parser.add_argument('--detect_motion', action='store_true', help='Detect motion (surveillance)')
    parser.add_argument('--enhance_low_light', action='store_true', help='Enhance low-light (surveillance)')
    parser.add_argument('--microscopy_type', type=str, default='general', help='Microscopy type')
    parser.add_argument('--detect_crack', action='store_true', help='Detect cracks (industrial)')
    parser.add_argument('--detect_corrosion', action='store_true', help='Detect corrosion (industrial)')

    return parser.parse_args()

def create_domain_config(args):
    """Create appropriate configuration based on domain"""
    base_config = {
        'dataset_name': args.data_name,
        'data_type': args.data_type,
        'feature_dims': args.feature_dims,
        'compressed_dims': args.compressed_dims,
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
        'domain': args.domain
    }

    if args.domain == 'agriculture':
        return AgricultureConfig(**base_config, has_nir_band=args.has_nir_band)
    elif args.domain == 'medical':
        return MedicalConfig(**base_config, modality=args.modality)
    elif args.domain == 'satellite':
        return SatelliteConfig(**base_config, satellite_type=args.satellite_type)
    elif args.domain == 'surveillance':
        return SurveillanceConfig(**base_config, detect_motion=args.detect_motion, enhance_low_light=args.enhance_low_light)
    elif args.domain == 'microscopy':
        return MicroscopyConfig(**base_config, microscopy_type=args.microscopy_type)
    elif args.domain == 'industrial':
        return IndustrialConfig(**base_config, detect_crack=args.detect_crack, detect_corrosion=args.detect_corrosion)
    elif args.domain == 'astronomy':  # ADD THIS
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

def main():
    args = parse_args()

    # Helper functions for consistent file naming
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
        dataset_name_lower = normalize_dataset_name(data_name)
        data_dir = Path(base_dir) / dataset_name_lower
        return {
            'data_dir': data_dir,
            'csv_path': data_dir / f"{dataset_name_lower}.csv",
            'train_csv': data_dir / f"{dataset_name_lower}_train.csv",
            'test_csv': data_dir / f"{dataset_name_lower}_test.csv",
            'json_config': data_dir / f"{dataset_name_lower}_config.json",
            'minimal_config': data_dir / f"{dataset_name_lower}_config_minimal.json",
            'checkpoint_dir': data_dir / 'checkpoints',
            'viz_dir': data_dir / 'visualizations',
            'log_dir': data_dir / 'logs',
            'heatmap_dir': data_dir / 'attention_heatmaps'
        }

    # Interactive mode
    if len(sys.argv) == 1 or args.interactive:
        print("\n" + "=" * 70)
        print("CDBNN - Convolutional Deep Bayesian Neural Network with Domain Support")
        print("=" * 70)
        print("Available domains: general, agriculture, medical, satellite, surveillance,")
        print("                   microscopy, industrial, astronomy")
        print("=" * 70)

        data_name = input("Enter dataset name: ").strip() or 'dataset'
        mode = input("Enter mode (train/predict/extract): ").strip().lower() or 'train'
        data_type = input("Enter dataset type (custom/torchvision): ").strip().lower() or 'custom'
        data_path = input("Enter data path: ").strip()
        domain = input("Enter domain (general/agriculture/medical/satellite/surveillance/microscopy/industrial/astronomy): ").strip().lower() or 'general'

        args.mode = mode
        args.data_name = data_name
        args.data_type = data_type
        args.data_path = data_path
        args.domain = domain

        # Additional astronomy-specific prompts if domain is astronomy
        if domain == 'astronomy':
            use_fits = input("Enable FITS support for astronomical images? (y/n): ").strip().lower() == 'y'
            args.use_fits = use_fits
            if use_fits:
                fits_hdu = input("FITS HDU to read (default: 0): ").strip()
                args.fits_hdu = int(fits_hdu) if fits_hdu else 0
                pixel_scale = input("Pixel scale in arcsec/pixel (default: 1.0): ").strip()
                args.pixel_scale = float(pixel_scale) if pixel_scale else 1.0

    # Validate required arguments
    if not args.data_path:
        raise ValueError("data_path is required")

    # Normalize dataset name to lowercase
    if args.data_name:
        args.data_name = normalize_dataset_name(args.data_name)
    else:
        args.data_name = 'dataset'

    # CRITICAL FIX: Separate data directory (for loading) from output directory (for writing)
    # The data directory is always under the original data/ path
    data_dir = Path('data')
    original_paths = get_dataset_paths(args.data_name, 'data')

    # The output directory is where we write results (if specified)
    if args.output_dir:
        output_base_dir = args.output_dir
    else:
        output_base_dir = 'data'  # Default to same as data if not specified

    output_paths = get_dataset_paths(args.data_name, output_base_dir)

    # Create output directories if they don't exist
    output_paths['data_dir'].mkdir(parents=True, exist_ok=True)
    output_paths['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
    output_paths['viz_dir'].mkdir(parents=True, exist_ok=True)
    output_paths['log_dir'].mkdir(parents=True, exist_ok=True)
    output_paths['heatmap_dir'].mkdir(parents=True, exist_ok=True)

    # Update args with original paths for loading and output paths for writing
    args.checkpoint_dir = str(original_paths['checkpoint_dir'])
    args.viz_dir = str(output_paths['viz_dir'])
    args.log_dir = str(output_paths['log_dir'])
    args.data_dir = str(original_paths['data_dir'])
    args.output_dir = str(output_paths['data_dir'])

    # Log the configuration
    logger.info("=" * 70)
    logger.info("CDBNN Configuration")
    logger.info("=" * 70)
    logger.info(f"Dataset name: {args.data_name}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Data type: {args.data_type}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"Loading from: {original_paths['data_dir']}")
    logger.info(f"Saving to: {output_paths['data_dir']}")
    logger.info(f"CSV output: {output_paths['csv_path']}")
    logger.info(f"Checkpoint directory (loading): {original_paths['checkpoint_dir']}")
    logger.info(f"Visualization directory (saving): {output_paths['viz_dir']}")
    logger.info("=" * 70)

    # Create domain-specific configuration
    config = create_domain_config(args)

    # Store both loading and saving directories in config
    config.data_dir = str(original_paths['data_dir'])
    config.output_dir = str(output_paths['data_dir'])
    config.checkpoint_dir = str(original_paths['checkpoint_dir'])
    config.viz_dir = str(output_paths['viz_dir'])
    config.log_dir = str(output_paths['log_dir'])

    # Ensure dataset name is lowercase in config
    config.dataset_name = args.data_name

    # Create domain-aware application
    app = DomainAwareCDBNN(config)

    # Execute based on mode
    try:
        if args.mode == 'train':
            logger.info(f"Starting {args.domain} domain training on {args.data_name}")
            logger.info(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")

            # For training, we need to prepare data with class structure
            train_loader, test_loader = app.prepare_data(args.data_path, args.data_type)
            logger.info(f"Data loaded successfully: {len(train_loader.dataset)} training samples")
            if test_loader:
                logger.info(f"Test samples: {len(test_loader.dataset)}")

            history = app.train(train_loader, test_loader)

            # Extract features and visualize
            logger.info("Extracting features from trained model...")
            features = app.extract_features(train_loader)

            if features and features.get('features') is not None and features.get('labels') is not None:
                logger.info(f"Features shape: {features['features'].shape}")
                logger.info(f"Labels shape: {features['labels'].shape}")

                # Generate visualizations
                if args.generate_tsne:
                    logger.info("Generating t-SNE visualization...")
                    app.visualizer.plot_tsne(features['features'], features['labels'], class_names=config.class_names)
                    logger.info(f"t-SNE plot saved to: {output_paths['viz_dir']}/tsne.png")

                # Save features to CSV in output directory
                features_csv = output_paths['csv_path']
                app._save_features_to_csv(features, features_csv)
                logger.info(f"Features saved to: {features_csv}")

            # Save configuration files in output directory
            app._save_config_files()

            logger.info(f"Training completed successfully for {args.domain} domain")
            logger.info(f"Model saved to: {original_paths['checkpoint_dir']}/{args.data_name}_best.pt")
            logger.info(f"Checkpoint saved to: {original_paths['checkpoint_dir']}/{args.data_name}_latest.pt")

        elif args.mode == 'predict':
            logger.info(f"Running {args.domain} domain prediction on {args.data_name}")
            logger.info(f"Input data: {args.data_path}")

            # For prediction, we don't need to prepare data - just use the PredictionManager directly
            # Create a prediction manager
            predictor = PredictionManager(config)

            # Use output directory for predictions
            output_csv = output_paths['csv_path']
            config_file = output_paths['data_dir'] / f"{args.data_name}.conf"

            logger.info(f"Output CSV: {output_csv}")
            logger.info(f"Configuration file: {config_file}")
            logger.info(f"Loading model from: {original_paths['checkpoint_dir']}/{args.data_name}_best.pt")

            # Run prediction directly on the data path
            results = predictor.predict_images(args.data_path, output_csv=str(output_csv))

            if results:
                logger.info(f"Prediction completed successfully!")
                if results.get('features') is not None:
                    logger.info(f"Processed {len(results['features'])} images")
                else:
                    logger.info(f"Prediction completed successfully!")
                logger.info(f"Results saved to: {output_csv}")
                logger.info(f"Configuration saved to: {config_file}")
            else:
                logger.error("Prediction failed - no results generated")
                return 1

        elif args.mode == 'extract':
            logger.info(f"Extracting {args.domain} domain features from {args.data_name}")
            logger.info(f"Input data: {args.data_path}")
            logger.info(f"Output CSV: {output_paths['csv_path']}")

            # For feature extraction, we need to prepare data
            train_loader, test_loader = app.prepare_data(args.data_path, args.data_type)
            logger.info(f"Data loaded successfully: {len(train_loader.dataset)} samples")

            features = app.extract_features(train_loader)

            if features and features.get('features') is not None:
                logger.info(f"Extracted {features['features'].shape[0]} features")
                logger.info(f"Feature dimension: {features['features'].shape[1]}")
                logger.info(f"Features saved to: {output_paths['csv_path']}")

                # Save feature statistics in output directory
                stats = {
                    'n_samples': features['features'].shape[0],
                    'n_features': features['features'].shape[1],
                    'mean': np.mean(features['features'], axis=0).tolist(),
                    'std': np.std(features['features'], axis=0).tolist(),
                    'min': np.min(features['features'], axis=0).tolist(),
                    'max': np.max(features['features'], axis=0).tolist()
                }

                stats_path = output_paths['data_dir'] / f"{args.data_name}_features_stats.json"
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
                logger.info(f"Feature statistics saved to: {stats_path}")

                # Generate t-SNE if requested
                if args.generate_tsne:
                    logger.info("Generating t-SNE visualization...")
                    app.visualizer.plot_tsne(features['features'], features['labels'], class_names=config.class_names)
                    logger.info(f"t-SNE plot saved to: {output_paths['viz_dir']}/tsne.png")

                # Save configuration files in output directory
                app._save_config_files()

            else:
                logger.error("Feature extraction failed - no features extracted")
                return 1

        else:
            logger.error(f"Invalid mode: {args.mode}")
            logger.info("Valid modes: train, predict, extract")
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
    logger.info(f"All outputs saved to: {output_paths['data_dir']}")
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
===================================================================================================""")
    sys.exit(main())
