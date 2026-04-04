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
        """Z-scale normalization for astronomical images"""
        # Flatten and sample
        flat = data.flatten()
        if len(flat) > samples:
            idx = np.random.choice(len(flat), samples, replace=False)
            flat = flat[idx]

        # Sort and compute percentiles
        flat.sort()

        # Find median and compute scale
        n = len(flat)
        center = flat[n // 2]

        # Use contrast to determine range
        half_range = contrast * (flat[-1] - flat[0])
        zmin = center - half_range
        zmax = center + half_range

        # Normalize
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

        # CRITICAL FIX: Sort selected_indices deterministically
        selected_indices.sort()  # Sort by index value, not by correlation

        # Now sort by correlation for importance ranking, but maintain index reference
        sorted_by_corr = sorted(selected_indices, key=lambda i: -label_corrs[i])
        final_indices = self._remove_redundant_features(features, sorted_by_corr, label_corrs)

        if len(final_indices) > self.max_features:
            final_indices = final_indices[:self.max_features]

        # Final deterministic sort
        final_indices.sort()

        logger.info(f"Final feature selection (deterministic order): {len(final_indices)} features")
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
        supported_formats = ImageProcessor.SUPPORTED_FORMATS
        # Add FITS formats if domain is astronomy
        if hasattr(self.config, 'domain') and self.config.domain == 'astronomy' and getattr(self.config, 'use_fits', False):
            fits_formats = ('.fits', '.fit', '.fits.gz', '.fit.gz')
            supported_formats = supported_formats + fits_formats

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
                if img_path.suffix.lower() in supported_formats or img_path.name.lower().endswith('.fits.gz'):
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
        self._feature_order_deterministic = True  # Ensure deterministic feature order

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
        # Sort indices to ensure deterministic order across runs
        sorted_order = np.argsort(indices)  # Sort by index value
        indices = indices[sorted_order]
        scores = scores[sorted_order] if scores is not None else None

        self._selected_feature_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        self._feature_importance_scores = torch.tensor(scores, device=self.device) if scores is not None else None
        self._feature_selection_metadata = metadata or {}
        self._is_feature_selection_frozen = True
        logger.info(f"Frozen feature selection (deterministic order): {len(indices)} features")

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

        # Check what outputs the model provides
        has_classifier = hasattr(self.model, 'classifier') and self.model.classifier is not None
        has_clustering = hasattr(self.model, 'cluster_centers') and self.model.cluster_centers is not None
        model_phase = getattr(self.model, 'training_phase', 1)

        logger.info(f"Model phase: {model_phase}, has_classifier: {has_classifier}, has_clustering: {has_clustering}")

        for batch_idx, (batch_data, _) in enumerate(tqdm(dataloader, desc="Predicting")):
            batch_tensor = batch_data.to(self.device)

            original_batch_size = batch_tensor.size(0)
            duplicated = False

            if original_batch_size == 1:
                batch_tensor = torch.cat([batch_tensor, batch_tensor], dim=0)
                duplicated = True

            # The model applies dataset-wide normalization automatically in forward pass
            output = self.model(batch_tensor)

            # Extract features - this should always be available
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
                    img_np = batch_np[i] if i < len(batch_np) else batch_np[0]

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

            # Collect targets and filenames for ALL samples in batch
            start_idx = batch_idx * batch_size
            for i in range(original_batch_size):
                actual_idx = start_idx + i
                if actual_idx < len(image_files):
                    collected_targets.append(class_labels[actual_idx])
                    collected_filenames.append(original_filenames[actual_idx])
                    collected_filepaths.append(image_files[actual_idx])
                    processed_count += 1

            # Add predictions - ONLY if they exist in output
            if 'class_predictions' in output and output['class_predictions'] is not None:
                preds = output['class_predictions']
                if hasattr(preds, 'float'):
                    preds = preds.float().cpu().numpy()
                if duplicated:
                    preds = preds[:original_batch_size]
                all_predictions['predictions'].extend(preds[:original_batch_size])
            else:
                # No classifier predictions - extend with placeholder
                all_predictions['predictions'].extend([-1] * original_batch_size)

            # Add probabilities
            if 'class_probabilities' in output and output['class_probabilities'] is not None:
                probs = output['class_probabilities']
                if hasattr(probs, 'float'):
                    probs = probs.float().cpu().numpy()
                if duplicated:
                    probs = probs[:original_batch_size]
                all_predictions['probabilities'].extend(probs[:original_batch_size])
            else:
                # No probabilities - extend with placeholder
                num_classes = self.config.num_classes if self.config.num_classes else 2
                all_predictions['probabilities'].extend([[0.0] * num_classes] * original_batch_size)

            # Add cluster assignments
            if 'cluster_assignments' in output and output['cluster_assignments'] is not None:
                clusters = output['cluster_assignments']
                if hasattr(clusters, 'float'):
                    clusters = clusters.float().cpu().numpy()
                if duplicated:
                    clusters = clusters[:original_batch_size]
                all_predictions['cluster_assignments'].extend(clusters[:original_batch_size])
            else:
                # No clusters - extend with placeholder
                all_predictions['cluster_assignments'].extend([-1] * original_batch_size)

            # Add cluster confidence
            if 'cluster_confidence' in output and output['cluster_confidence'] is not None:
                conf = output['cluster_confidence']
                if hasattr(conf, 'float'):
                    conf = conf.float().cpu().numpy()
                if duplicated:
                    conf = conf[:original_batch_size]
                all_predictions['cluster_confidence'].extend(conf[:original_batch_size])
            else:
                # No confidence - extend with placeholder
                all_predictions['cluster_confidence'].extend([0.0] * original_batch_size)

        # Stack features
        if all_features:
            all_predictions['features'] = np.vstack(all_features)

        # Add domain features and quality metrics to predictions
        if domain_features_list:
            # Get all unique keys from domain features
            all_keys = set()
            for feat in domain_features_list:
                all_keys.update(feat.keys())

            for key in all_keys:
                all_predictions[f'domain_{key}'] = [feat.get(key, np.nan) for feat in domain_features_list]
            logger.info(f"Added {len(all_keys)} domain-specific features")

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
                # This should not happen now, but keep as safety
                logger.warning(f"Padding {key}: had {len(all_predictions[key])}, need {n_samples}")
                while len(all_predictions[key]) < n_samples:
                    if key == 'predictions':
                        all_predictions[key].append(-1)
                    elif key == 'probabilities':
                        num_classes = self.config.num_classes if self.config.num_classes else 2
                        all_predictions[key].append([0.0] * num_classes)
                    elif key == 'cluster_assignments':
                        all_predictions[key].append(-1)
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

            # CRITICAL FIX: Sort feature columns deterministically
            # Features are always indexed from 0 to n-1, but we need to ensure
            # the same image always produces the same feature values at the same index
            for i in range(features.shape[1]):
                data[f'feature_{i}'] = features[:, i]

        # Domain-specific features (prefix with domain_ or quality_)
        # CRITICAL FIX: Sort domain feature keys alphabetically for consistency
        domain_keys = [k for k in predictions.keys() if (k.startswith('domain_') or k.startswith('quality_')) and predictions[k] is not None]
        for key in sorted(domain_keys):  # Sort alphabetically for deterministic order
            values = predictions[key]
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
                # Add per-class probabilities (sorted by class index for consistency)
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

        # CRITICAL FIX: Sort all column keys deterministically before creating DataFrame
        # This ensures the same columns always appear in the same order
        sorted_keys = sorted(data.keys())
        sorted_data = {key: data[key] for key in sorted_keys}

        # Create DataFrame with sorted columns
        df = pd.DataFrame(sorted_data)

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

        logger.info(f"CSV columns ({len(sorted_data.keys())}): {', '.join(list(sorted_data.keys())[:10])}{'...' if len(sorted_data) > 10 else ''}")

    def _extract_domain_features(self, image_path: str, image_array: np.ndarray) -> Dict[str, float]:
        """Extract domain-specific features from image"""
        domain = self.config.get('dataset', {}).get('domain', 'general')

        if domain == 'astronomy':
            if not hasattr(self, '_astronomy_extractor'):
                self._astronomy_extractor = AstronomyFeatureExtractor(self.config)
            return self._astronomy_extractor.extract_features(image_array)

        elif domain == 'medical':
            if not hasattr(self, '_medical_extractor'):
                self._medical_extractor = MedicalFeatureExtractor()
            return self._medical_extractor.extract_features(image_array)

        elif domain == 'agriculture':
            if not hasattr(self, '_agriculture_extractor'):
                self._agriculture_extractor = AgricultureFeatureExtractor()
            return self._agriculture_extractor.extract_features(image_array)

        return {}

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
    # UPDATE CDBNNApplication to use ModelFactory
    # =============================================================================

    def _create_model(self) -> BaseAutoencoder:
        """Create model using ModelFactory"""
        return ModelFactory.create_model(self.config)


# =============================================================================
# UPDATED DOMAIN-AWARE CDBNN (Replace the existing class)
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

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply domain-specific preprocessing"""
        if self.domain_processor:
            return self.domain_processor.preprocess(image)
        return image

    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """Apply domain-specific preprocessing to batch"""
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
        """Extract domain-specific features"""
        if self.domain_processor:
            return self.domain_processor.extract_features(image)
        return {}

    def extract_domain_features_batch(self, images: np.ndarray) -> List[Dict[str, float]]:
        """Extract domain-specific features for batch"""
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
        """Get domain-specific quality metrics"""
        if self.domain_processor:
            return self.domain_processor.get_quality_metrics(image)
        return {}

    def get_domain_quality_metrics_batch(self, images: np.ndarray) -> List[Dict[str, float]]:
        """Get domain-specific quality metrics for batch"""
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
    """Complete Surveillance/CCTV processor with all features"""

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
        """Enhance low-light surveillance images"""
        if len(image.shape) == 3:
            # Convert to LAB
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab = lab.astype(np.float32) / 255.0

            # Enhance L channel
            l_channel = lab[:, :, 0]

            # Adaptive gamma correction
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
    """Complete Microscopy imaging processor with all features"""

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
    """Complete Industrial inspection processor with all features"""

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
# MODEL FACTORY - Creates appropriate model based on domain
# =============================================================================

class ModelFactory:
    """Factory for creating appropriate model based on configuration"""

    @staticmethod
    def create_model(config: GlobalConfig) -> nn.Module:
        """Create model with domain-specific enhancements"""
        input_shape = (config.in_channels, config.input_size[0], config.input_size[1])
        feature_dims = config.feature_dims
        compressed_dims = getattr(config, 'compressed_dims', min(64, max(8, feature_dims // 4)))
        config.compressed_dims = compressed_dims

        # Get domain from config
        domain = getattr(config, 'domain', 'general')
        image_type = getattr(config, 'image_type', 'general')

        # Create domain-specific model if configured
        if domain == 'astronomy' or image_type == 'astronomical':
            logger.info("Creating Astronomical Structure Preserving Autoencoder")
            return AstronomicalStructurePreservingAutoencoder(input_shape, feature_dims, config)

        elif domain == 'medical' or image_type == 'medical':
            logger.info("Creating Medical Structure Preserving Autoencoder")
            return MedicalStructurePreservingAutoencoder(input_shape, feature_dims, config)

        elif domain == 'agriculture' or image_type == 'agricultural':
            logger.info("Creating Agricultural Pattern Autoencoder")
            return AgriculturalPatternAutoencoder(input_shape, feature_dims, config)

        # Default to base autoencoder
        logger.info(f"Creating base autoencoder with {feature_dims}D → {compressed_dims}D features")
        return BaseAutoencoder(config)


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

class AstronomicalStructurePreservingAutoencoder(BaseAutoencoder):
    """Autoencoder specialized for astronomical imaging features"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: GlobalConfig):
        # Store config for later use
        self._domain_config = config
        super().__init__(config)

        # Get enhancement configurations
        self.enhancement_config = {
            'structure_preservation': True,
            'detail_preservation': True,
            'star_detection': True,
            'galaxy_features': True
        }

        # Initial channel transformation layer
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

        # Galaxy feature enhancement
        self.galaxy_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(size, size, kernel_size=3, padding=d, dilation=d),
                nn.GroupNorm(min(32, size), size),
                nn.LeakyReLU(0.2)
            ) for size, d in zip(self.encoder_channels, [1, 2, 4])
        ])

        self._cached_features = {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced encoding with astronomical feature preservation"""
        features = {}

        # Apply dataset-wide normalization if available
        if self.dataset_statistics is not None and self.dataset_statistics.is_calculated:
            x = self.dataset_statistics.normalize(x)

        # Initial channel transformation
        x = self.initial_transform(x)

        if self.enhancement_config.get('detail_preservation', True):
            # Multi-scale detail extraction
            detail_features = [module(x) for module in self.detail_preserving]
            features['details'] = sum(detail_features) / len(detail_features)
            x = x + 0.1 * features['details']

        if self.enhancement_config.get('star_detection', True):
            # Star detection
            features['stars'] = self.star_detector(x)
            x = x * (1 + 0.1 * features['stars'])

        # Regular encoding path with galaxy enhancement
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.enhancement_config.get('galaxy_features', True):
                if idx < len(self.galaxy_enhancer):
                    galaxy_features = self.galaxy_enhancer[idx](x)
                    x = x + 0.1 * galaxy_features

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

        # Add preserved features if available
        if hasattr(self, '_cached_features') and self._cached_features:
            features = self._cached_features
            if self.enhancement_config.get('detail_preservation', True):
                if 'details' in features:
                    x = x + 0.1 * features['details']

            if self.enhancement_config.get('star_detection', True):
                if 'stars' in features:
                    x = x * (1 + 0.1 * features['stars'])

            # Clear cached features
            self._cached_features = {}

        # Final channel transformation back to input channels
        x = nn.Conv2d(self.encoder_channels[0], self.in_channels, kernel_size=1).to(x.device)(x)

        return x

class MedicalStructurePreservingAutoencoder(BaseAutoencoder):
    """Autoencoder specialized for medical imaging features"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: GlobalConfig):
        super().__init__(config)

        # Get enhancement configurations
        self.enhancement_config = {
            'tissue_boundary': True,
            'lesion_detection': True,
            'contrast_enhancement': True
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

class AgriculturalPatternAutoencoder(BaseAutoencoder):
    """Autoencoder specialized for agricultural imaging features"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: GlobalConfig):
        super().__init__(config)

        # Get enhancement configurations
        self.enhancement_config = {
            'texture_analysis': True,
            'damage_detection': True,
            'color_anomaly': True
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
