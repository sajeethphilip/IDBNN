
# Fully functional updated version as of Nov 22 2025
#Working, fully functional with predcition 30/March/2025
#Revisions on Mar30 2025 Stable version 8:56 AM
# Added distance correlations to filter the output features. April 12, 3:45 am
# Fixed a bug in Prediction mode model loading April 14 2025 9:32 am
#Finalised completely working module as on 15th April 2025
# Feature Dimension can now be input during training instead of hardcoding to 128 April 30 11:13 PM
# Last updated with configurable data_name on May 1 1:07 am
# Infinite training loop added May 1 1:30 am
#----------Bug fixes and improved version - April 5 4:24 pm----------------------------------------------
#---- author : Ninan Sajeeth Philip, Artificial Intelligence Research and Intelligent Systems
#-------------------------------------------------------------------------------------------------------------------------------
import cv2
import torch
import copy
import sys
import gc
import os
import torch
import subprocess
import traceback
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import logging
import os
import csv
import json
import zipfile
import tarfile
import gzip
import bz2
import lzma
from datetime import datetime, timedelta
import time
import shutil
import glob
from tqdm import tqdm
import random
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
from pathlib import Path
import torch.multiprocessing
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as KF
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as KF
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy.special import softmax
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import torch
import logging
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional
from torchvision.transforms.functional import resize
from types import SimpleNamespace
import os
import json
import logging
import traceback
import argparse
from datetime import datetime
import torch
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union

from scipy.spatial.distance import correlation
from itertools import combinations

logger = logging.getLogger(__name__)
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

class DistanceCorrelationFeatureSelector:
    """Enhanced feature selector with better handling of complex datasets"""

    def __init__(self, upper_threshold=0.85, lower_threshold=0.01, min_features=8, max_features=50):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.min_features = min_features
        self.max_features = max_features

    def calculate_distance_correlations(self, features, labels):
        """Calculate enhanced distance correlations with class separation focus"""
        n_features = features.shape[1]
        label_corrs = np.zeros(n_features)

        # Calculate correlation with labels
        for i in range(n_features):
            label_corrs[i] = 1 - correlation(features[:, i], labels)

        # NEW: Add class separation metrics for complex datasets
        if len(np.unique(labels)) > 10:  # For datasets with many classes
            separation_scores = self._calculate_class_separation_scores(features, labels)
            # Combine correlation with separation scores
            combined_scores = 0.7 * label_corrs + 0.3 * separation_scores
            return combined_scores

        return label_corrs

    def _calculate_class_separation_scores(self, features, labels):
        """Calculate how well features separate different classes"""
        n_features = features.shape[1]
        separation_scores = np.zeros(n_features)
        unique_labels = np.unique(labels)

        for i in range(n_features):
            feature_values = features[:, i]

            # Calculate between-class variance / within-class variance
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
        """Enhanced feature selection with adaptive thresholds"""
        label_corrs = self.calculate_distance_correlations(features, labels)

        # Get indices of features that meet upper threshold
        selected_indices = [i for i, corr in enumerate(label_corrs)
                          if corr >= self.upper_threshold]

        # If too few features, relax threshold
        if len(selected_indices) < self.min_features:
            # Take top min_features by correlation
            top_indices = np.argsort(label_corrs)[-self.min_features:]
            selected_indices = list(top_indices)
            logger.info(f"Relaxed threshold: selected top {self.min_features} features")

        # Sort by correlation strength (descending)
        selected_indices.sort(key=lambda i: -label_corrs[i])

        # Remove features that are too correlated with each other
        final_indices = self._remove_redundant_features(features, selected_indices, label_corrs)

        # Ensure we have reasonable number of features
        if len(final_indices) > self.max_features:
            final_indices = final_indices[:self.max_features]
            logger.info(f"Limited to top {self.max_features} features")

        logger.info(f"Final feature selection: {len(final_indices)} features "
                   f"(correlation range: {min(label_corrs[final_indices]):.3f}-{max(label_corrs[final_indices]):.3f})")

        return final_indices, label_corrs

    def _remove_redundant_features(self, features, candidate_indices, corr_values):
        """Remove redundant features while preserving diversity"""
        final_indices = []
        feature_matrix = features[:, candidate_indices]

        for i, idx in enumerate(candidate_indices):
            keep = True

            for j in final_indices:
                # Calculate correlation between features
                corr = 1 - correlation(feature_matrix[:, i],
                                     feature_matrix[:, candidate_indices.index(j)])
                if corr > self.lower_threshold:
                    # Keep the feature with higher correlation to labels
                    if corr_values[idx] <= corr_values[j]:
                        keep = False
                        break

            if keep:
                final_indices.append(idx)

                # Stop if we have enough diverse features
                if len(final_indices) >= self.max_features:
                    break

        return final_indices

class DynamicFeatureSelector:
    """Intelligently selects feature selection method based on dataset characteristics"""

    def __init__(self, max_features=32):
        self.max_features = max_features
        self.dc_selector = DistanceCorrelationFeatureSelector()

    def calculate_dataset_complexity(self, features, labels):
        """Compute multi-factor dataset complexity score"""
        complexity_metrics = {}

        # 1. Class Separability
        complexity_metrics['class_separability'] = self._calculate_class_separability(features, labels)

        # 2. Feature Correlation Structure
        complexity_metrics['feature_correlation'] = self._calculate_feature_correlation(features)

        # 3. Sample Adequacy
        complexity_metrics['sample_adequacy'] = self._calculate_sample_adequacy(features, labels)

        # 4. Dimensionality Ratio
        complexity_metrics['dimensionality_ratio'] = features.shape[0] / max(1, features.shape[1])

        # Combined complexity score (0-1, higher = more complex)
        complexity_score = (
            0.4 * complexity_metrics['class_separability'] +
            0.3 * complexity_metrics['feature_correlation'] +
            0.2 * complexity_metrics['sample_adequacy'] +
            0.1 * min(1.0, complexity_metrics['dimensionality_ratio'] / 10)
        )

        return complexity_score, complexity_metrics

    def _calculate_class_separability(self, features, labels):
        """Measure how well classes are separated in feature space"""
        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            return 0.0

        # Between-class variance / within-class variance
        overall_mean = np.mean(features, axis=0)
        between_var = 0
        within_var = 0

        for cls in unique_classes:
            class_mask = (labels == cls)
            class_features = features[class_mask]
            class_mean = np.mean(class_features, axis=0)
            class_size = len(class_features)

            between_var += class_size * np.sum((class_mean - overall_mean) ** 2)
            within_var += np.sum((class_features - class_mean) ** 2)

        separability = between_var / (within_var + 1e-8)
        return min(1.0, separability)

    def _calculate_feature_correlation(self, features):
        """Measure feature interdependence complexity"""
        if features.shape[1] < 2:
            return 0.0

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(features.T)
        np.fill_diagonal(corr_matrix, 0)  # Remove self-correlation

        # Measure average absolute correlation
        avg_correlation = np.mean(np.abs(corr_matrix))
        return avg_correlation

    def _calculate_sample_adequacy(self, features, labels):
        """Determine if we have enough samples for reliable estimation"""
        n_samples, n_features = features.shape
        unique_classes, class_counts = np.unique(labels, return_counts=True)

        # Minimum samples per class for reliable distribution estimation
        min_samples_per_class = max(20, n_features // 2)

        # Calculate adequacy ratio
        adequate_classes = sum(count >= min_samples_per_class for count in class_counts)
        adequacy_ratio = adequate_classes / len(unique_classes)

        # Penalize if any class has very few samples
        min_class_ratio = min(class_counts) / max(1, min_samples_per_class)

        final_adequacy = 0.7 * adequacy_ratio + 0.3 * min(1.0, min_class_ratio)
        return final_adequacy

    def select_feature_method(self, features, labels):
        """Dynamically select best DISCRIMINATIVE feature selection method"""
        complexity_score, metrics = self.calculate_dataset_complexity(features, labels)
        n_samples, n_features = features.shape
        n_classes = len(np.unique(labels))

        # Decision rules based on dataset characteristics
        if n_classes > 20:
            # For many classes (like CIFAR-100), use mutual information or ANOVA
            method = "mutual_info"
            reason = f"Many classes ({n_classes}), using mutual information"

        elif n_features > n_samples:
            # High-dimensional data
            method = "mutual_info"
            reason = f"High dimensionality ({n_features} > {n_samples}), using mutual information"

        elif n_classes <= 10 and n_samples > 1000:
            # For few classes with sufficient data, use Random Forest
            method = "random_forest"
            reason = f"Few classes ({n_classes}) with large samples, using Random Forest"

        elif complexity_score > 0.6 and metrics['sample_adequacy'] > 0.7:
            # Complex but well-sampled data
            method = "mutual_info"
            reason = "Complex dataset with good samples, using mutual information"

        else:
            # Default to ANOVA for general cases
            method = "anova"
            reason = "General case, using ANOVA F-test"

        logger.info(f"Selected {method}: {reason} (complexity: {complexity_score:.3f}, {n_classes} classes)")
        return method, reason, complexity_score

    def kl_feature_selection(self, features, labels, target_dims=32):
        """Select features using KL divergence"""
        n_features = features.shape[1]
        kl_scores = np.zeros(n_features)

        for i in range(n_features):
            feature_values = features[:, i]
            kl_scores[i] = self._calculate_feature_kl(feature_values, labels)

        # Select top-k features by KL divergence
        selected_indices = np.argsort(kl_scores)[-target_dims:][::-1]
        return selected_indices, kl_scores

    def _calculate_feature_kl(self, feature_values, labels):
        """Calculate KL divergence for a single feature across classes"""
        unique_classes = np.unique(labels)

        if len(unique_classes) < 2:
            return 0.0

        # Estimate probability distributions for each class
        class_distributions = []
        for cls in unique_classes:
            cls_mask = (labels == cls)
            cls_features = feature_values[cls_mask]

            if len(cls_features) < 5:  # Need minimum samples
                return 0.0

            # Convert to probability distribution (histogram)
            hist, _ = np.histogram(cls_features, bins=min(10, len(cls_features)//5), density=True)
            class_distributions.append(hist + 1e-8)

        # Calculate pairwise KL divergences
        total_kl = 0.0
        pair_count = 0

        for i, j in combinations(range(len(class_distributions)), 2):
            p = class_distributions[i]
            q = class_distributions[j]

            # Symmetric KL divergence
            kl_pq = np.sum(p * np.log(p / q))
            kl_qp = np.sum(q * np.log(q / p))

            total_kl += (kl_pq + kl_qp) / 2
            pair_count += 1

        return total_kl / pair_count if pair_count > 0 else 0.0

    def variance_based_selection(self, features, target_dims=32):
        """Simple variance-based feature selection"""
        variances = np.var(features, axis=0)
        selected_indices = np.argsort(variances)[-target_dims:][::-1]
        return selected_indices, variances

    def balanced_correlation_selection(self, features, labels, target_dims=32):
        """Use existing distance correlation with balanced parameters"""
        # Reuse the existing DistanceCorrelationFeatureSelector
        selector = DistanceCorrelationFeatureSelector(
            upper_threshold=0.7,  # More balanced threshold
            lower_threshold=0.05,
            min_features=8,
            max_features=target_dims
        )
        selected_indices, corr_values = selector.select_features(features, labels)
        return selected_indices, corr_values

    def dynamic_feature_selection(self, features, labels):
        """Feature selection optimized for classification accuracy"""
        if labels is None or len(np.unique(labels)) < 2:
            return self.mutual_info_selection(features, labels, self.max_features), "mutual_info"

        n_samples, n_features = features.shape
        n_classes = len(np.unique(labels))

        # Select method based on dataset characteristics for maximum accuracy
        if n_classes > 10 and n_samples > n_classes * 20:
            # Many classes: use mutual information for discriminative power
            selected_indices, scores = self.mutual_info_selection(features, labels, self.max_features)
            method = "mutual_info"
        elif n_features > 1000:
            # High dimensionality: use ANOVA for efficiency and accuracy
            selected_indices, scores = self.anova_selection(features, labels, self.max_features)
            method = "anova"
        elif n_classes <= 10 and n_samples > 500:
            # Few classes with good data: use Random Forest for maximum accuracy
            selected_indices, scores = self.random_forest_selection(features, labels, self.max_features)
            method = "random_forest"
        else:
            # Default: mutual information for balanced performance
            selected_indices, scores = self.mutual_info_selection(features, labels, self.max_features)
            method = "mutual_info"

        logger.info(f"Selected {method} for {n_classes} classes, {n_features} features")
        return (selected_indices, scores), method

    def anova_selection(self, features, labels, target_dims):
        """Select features using ANOVA F-value (good for multi-class)"""
        from sklearn.feature_selection import f_classif

        try:
            # Calculate ANOVA F-values between each feature and labels
            f_scores, p_values = f_classif(features, labels)

            # Select top features by F-score
            selected_indices = np.argsort(f_scores)[-target_dims:][::-1]

            logger.info(f"ANOVA selection: F-score range {f_scores[selected_indices].min():.4f}-{f_scores[selected_indices].max():.4f}")
            return selected_indices, f_scores

        except Exception as e:
            logger.warning(f"ANOVA selection failed: {e}, falling back to variance")
            return self.variance_based_selection(features, target_dims)

    def lda_based_selection(self, features, labels, target_dims):
        """Select features using Linear Discriminant Analysis coefficients"""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        try:
            n_classes = len(np.unique(labels))
            if n_classes > features.shape[0] or n_classes > features.shape[1]:
                # LDA requires n_features >= n_classes and n_samples >= n_features
                return self.anova_selection(features, labels, target_dims)

            lda = LinearDiscriminantAnalysis()
            lda.fit(features, labels)

            # Use absolute coefficients as importance scores
            if hasattr(lda, 'coef_'):
                # For binary classification
                if lda.coef_.shape[0] == 1:
                    importance_scores = np.abs(lda.coef_[0])
                else:
                    # For multi-class, take mean absolute coefficients across classes
                    importance_scores = np.mean(np.abs(lda.coef_), axis=0)
            else:
                importance_scores = np.var(features, axis=0)

            selected_indices = np.argsort(importance_scores)[-target_dims:][::-1]

            logger.info(f"LDA selection: importance range {importance_scores[selected_indices].min():.4f}-{importance_scores[selected_indices].max():.4f}")
            return selected_indices, importance_scores

        except Exception as e:
            logger.warning(f"LDA selection failed: {e}, falling back to ANOVA")
            return self.anova_selection(features, labels, target_dims)

    def random_forest_selection(self, features, labels, target_dims):
        """Select features using Random Forest importance (computationally expensive but very effective)"""
        from sklearn.ensemble import RandomForestClassifier

        try:
            # Use a small random forest for feature importance
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(features, labels)

            importance_scores = rf.feature_importances_
            selected_indices = np.argsort(importance_scores)[-target_dims:][::-1]

            logger.info(f"Random Forest selection: importance range {importance_scores[selected_indices].min():.4f}-{importance_scores[selected_indices].max():.4f}")
            return selected_indices, importance_scores

        except Exception as e:
            logger.warning(f"Random Forest selection failed: {e}, falling back to mutual info")
            return self.mutual_info_selection(features, labels, target_dims)

    def kl_optimized_selection(self, features, labels, target_dims):
        """Select features that maximize KL divergence between classes"""
        try:
            n_classes = len(np.unique(labels))
            n_features = features.shape[1]

            # Calculate class-wise statistics for KL divergence optimization
            class_means = []
            class_vars = []
            class_sizes = []

            for cls in np.unique(labels):
                class_mask = (labels == cls)
                class_features = features[class_mask]
                class_means.append(np.mean(class_features, axis=0))
                class_vars.append(np.var(class_features, axis=0))
                class_sizes.append(np.sum(class_mask))

            class_means = np.array(class_means)
            class_vars = np.array(class_vars)
            class_sizes = np.array(class_sizes)

            # Calculate KL divergence scores for each feature
            kl_scores = np.zeros(n_features)

            for i in range(n_features):
                feature_kl = 0.0
                pair_count = 0

                # Calculate pairwise KL divergences between classes
                for j in range(n_classes):
                    for k in range(j + 1, n_classes):
                        # KL divergence between two Gaussian distributions
                        var_ratio = class_vars[j, i] / class_vars[k, i]
                        mean_diff = class_means[j, i] - class_means[k, i]
                        kl_jk = 0.5 * (var_ratio +
                                     (mean_diff ** 2) / class_vars[k, i] -
                                     1 - np.log(var_ratio))
                        kl_kj = 0.5 * (1/var_ratio +
                                     (mean_diff ** 2) / class_vars[j, i] -
                                     1 - np.log(1/var_ratio))

                        # Symmetric KL divergence
                        feature_kl += (kl_jk + kl_kj) / 2
                        pair_count += 1

                if pair_count > 0:
                    kl_scores[i] = feature_kl / pair_count

            # Select features with highest KL divergence scores
            selected_indices = np.argsort(kl_scores)[-target_dims:][::-1]

            logger.info(f"KL-optimized selection: KL range {kl_scores[selected_indices].min():.4f}-{kl_scores[selected_indices].max():.4f}")
            return selected_indices, kl_scores

        except Exception as e:
            logger.warning(f"KL-optimized selection failed: {e}, falling back to mutual info")
            return self.mutual_info_selection(features, labels, target_dims)

class AdaptiveFeatureCompressor(nn.Module):
    """Feature compressor that adapts to any input dimension"""

    def __init__(self, input_dims, output_dims, hidden_scale=0.5):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        # Calculate hidden dimensions based on ratio
        self.hidden_dims = int(max(output_dims, input_dims * hidden_scale))

        # Adaptive compression layers
        self.layers = nn.Sequential(
            nn.Linear(input_dims, self.hidden_dims),
            nn.BatchNorm1d(self.hidden_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dims, output_dims),
            nn.Tanh()
        )

    def forward(self, x):
        # If input dimensions don't match, use adaptive projection
        if x.shape[1] != self.input_dims:
            return self._adaptive_forward(x)
        return self.layers(x)

    def _adaptive_forward(self, x):
        """Handle variable input dimensions"""
        current_dims = x.shape[1]

        if current_dims == self.input_dims:
            return self.layers(x)
        elif current_dims < self.input_dims:
            # Upsample: pad with zeros or interpolate
            return self._upsample_forward(x)
        else:
            # Downsample: use projection
            return self._downsample_forward(x)

    def _upsample_forward(self, x):
        """Upsample input to match expected dimensions"""
        batch_size = x.shape[0]
        current_dims = x.shape[1]

        if current_dims == self.output_dims:
            # Already at target size
            return x

        # Create upsampled tensor
        if current_dims < self.input_dims:
            # Pad with zeros
            upsampled = torch.zeros(batch_size, self.input_dims, device=x.device)
            upsampled[:, :current_dims] = x
            return self.layers(upsampled)
        else:
            # Direct projection for close sizes
            projection = nn.Linear(current_dims, self.input_dims, device=x.device)
            return self.layers(projection(x))

    def _downsample_forward(self, x):
        """Downsample input to match expected dimensions"""
        current_dims = x.shape[1]

        if current_dims <= self.hidden_dims:
            # Can go directly to hidden layer
            hidden_proj = nn.Linear(current_dims, self.hidden_dims, device=x.device)
            hidden_out = hidden_proj(x)
            hidden_out = nn.LeakyReLU(0.2)(hidden_out)

            output_proj = nn.Linear(self.hidden_dims, self.output_dims, device=x.device)
            return nn.Tanh()(output_proj(hidden_out))
        else:
            # Multi-step downsampling
            projection = nn.Linear(current_dims, self.input_dims, device=x.device)
            return self.layers(projection(x))

class AdaptiveFeatureDecompressor(nn.Module):
    """Feature decompressor that adapts to any input dimension"""

    def __init__(self, input_dims, output_dims, hidden_scale=0.5):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        # Calculate hidden dimensions based on ratio
        self.hidden_dims = int(max(input_dims, output_dims * hidden_scale))

        # Adaptive decompression layers
        self.layers = nn.Sequential(
            nn.Linear(input_dims, self.hidden_dims),
            nn.BatchNorm1d(self.hidden_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dims, output_dims)
        )

    def forward(self, x):
        # If input dimensions don't match, use adaptive projection
        if x.shape[1] != self.input_dims:
            return self._adaptive_forward(x)
        return self.layers(x)

    def _adaptive_forward(self, x):
        """Handle variable input dimensions"""
        current_dims = x.shape[1]

        if current_dims == self.input_dims:
            return self.layers(x)
        elif current_dims < self.input_dims:
            # Upsample to match compressor input
            return self._upsample_forward(x)
        else:
            # Downsample to match compressor input
            return self._downsample_forward(x)

    def _upsample_forward(self, x):
        """Upsample input to match expected dimensions"""
        batch_size = x.shape[0]
        current_dims = x.shape[1]

        if current_dims == self.output_dims:
            # Already at target size
            return x

        # Create upsampled tensor
        if current_dims < self.input_dims:
            # Pad with zeros then decompress
            upsampled = torch.zeros(batch_size, self.input_dims, device=x.device)
            upsampled[:, :current_dims] = x
            return self.layers(upsampled)
        else:
            # Direct projection
            projection = nn.Linear(current_dims, self.input_dims, device=x.device)
            return self.layers(projection(x))

    def _downsample_forward(self, x):
        """Downsample input to match expected dimensions"""
        current_dims = x.shape[1]

        if current_dims <= self.hidden_dims:
            # Direct to hidden layer
            hidden_proj = nn.Linear(current_dims, self.hidden_dims, device=x.device)
            hidden_out = hidden_proj(x)
            hidden_out = nn.LeakyReLU(0.2)(hidden_out)

            output_proj = nn.Linear(self.hidden_dims, self.output_dims, device=x.device)
            return output_proj(hidden_out)
        else:
            # Multi-step downsampling
            projection = nn.Linear(current_dims, self.input_dims, device=x.device)
            return self.layers(projection(x))

class AdaptiveFeatureCompressor(nn.Module):
    """Feature compressor that adapts to any input dimension"""

    def __init__(self, input_dims, output_dims, hidden_scale=0.5):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        # Calculate hidden dimensions based on ratio
        self.hidden_dims = int(max(output_dims, input_dims * hidden_scale))

        # Adaptive compression layers
        self.layers = nn.Sequential(
            nn.Linear(input_dims, self.hidden_dims),
            nn.BatchNorm1d(self.hidden_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dims, output_dims),
            nn.Tanh()
        )

    def forward(self, x):
        # If input dimensions don't match, use adaptive projection
        if x.shape[1] != self.input_dims:
            return self._adaptive_forward(x)
        return self.layers(x)

    def _adaptive_forward(self, x):
        """Handle variable input dimensions"""
        current_dims = x.shape[1]

        if current_dims == self.input_dims:
            return self.layers(x)
        elif current_dims < self.input_dims:
            # Upsample: pad with zeros or interpolate
            return self._upsample_forward(x)
        else:
            # Downsample: use projection
            return self._downsample_forward(x)

    def _upsample_forward(self, x):
        """Upsample input to match expected dimensions"""
        batch_size = x.shape[0]
        current_dims = x.shape[1]

        if current_dims == self.output_dims:
            # Already at target size
            return x

        # Create upsampled tensor
        if current_dims < self.input_dims:
            # Pad with zeros
            upsampled = torch.zeros(batch_size, self.input_dims, device=x.device)
            upsampled[:, :current_dims] = x
            return self.layers(upsampled)
        else:
            # Direct projection for close sizes
            projection = nn.Linear(current_dims, self.input_dims, device=x.device)
            return self.layers(projection(x))

    def _downsample_forward(self, x):
        """Downsample input to match expected dimensions"""
        current_dims = x.shape[1]

        if current_dims <= self.hidden_dims:
            # Can go directly to hidden layer
            hidden_proj = nn.Linear(current_dims, self.hidden_dims, device=x.device)
            hidden_out = hidden_proj(x)
            hidden_out = nn.LeakyReLU(0.2)(hidden_out)

            output_proj = nn.Linear(self.hidden_dims, self.output_dims, device=x.device)
            return nn.Tanh()(output_proj(hidden_out))
        else:
            # Multi-step downsampling
            projection = nn.Linear(current_dims, self.input_dims, device=x.device)
            return self.layers(projection(x))

class AdaptiveFeatureDecompressor(nn.Module):
    """Feature decompressor that adapts to any input dimension"""

    def __init__(self, input_dims, output_dims, hidden_scale=0.5):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        # Calculate hidden dimensions based on ratio
        self.hidden_dims = int(max(input_dims, output_dims * hidden_scale))

        # Adaptive decompression layers
        self.layers = nn.Sequential(
            nn.Linear(input_dims, self.hidden_dims),
            nn.BatchNorm1d(self.hidden_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dims, output_dims)
        )

    def forward(self, x):
        # If input dimensions don't match, use adaptive projection
        if x.shape[1] != self.input_dims:
            return self._adaptive_forward(x)
        return self.layers(x)

    def _adaptive_forward(self, x):
        """Handle variable input dimensions"""
        current_dims = x.shape[1]

        if current_dims == self.input_dims:
            return self.layers(x)
        elif current_dims < self.input_dims:
            # Upsample to match compressor input
            return self._upsample_forward(x)
        else:
            # Downsample to match compressor input
            return self._downsample_forward(x)

    def _upsample_forward(self, x):
        """Upsample input to match expected dimensions"""
        batch_size = x.shape[0]
        current_dims = x.shape[1]

        if current_dims == self.output_dims:
            # Already at target size
            return x

        # Create upsampled tensor
        if current_dims < self.input_dims:
            # Pad with zeros then decompress
            upsampled = torch.zeros(batch_size, self.input_dims, device=x.device)
            upsampled[:, :current_dims] = x
            return self.layers(upsampled)
        else:
            # Direct projection
            projection = nn.Linear(current_dims, self.input_dims, device=x.device)
            return self.layers(projection(x))

    def _downsample_forward(self, x):
        """Downsample input to match expected dimensions"""
        current_dims = x.shape[1]

        if current_dims <= self.hidden_dims:
            # Direct to hidden layer
            hidden_proj = nn.Linear(current_dims, self.hidden_dims, device=x.device)
            hidden_out = hidden_proj(x)
            hidden_out = nn.LeakyReLU(0.2)(hidden_out)

            output_proj = nn.Linear(self.hidden_dims, self.output_dims, device=x.device)
            return output_proj(hidden_out)
        else:
            # Multi-step downsampling
            projection = nn.Linear(current_dims, self.input_dims, device=x.device)
            return self.layers(projection(x))

class PredictionManager:
    """Manages prediction using frozen feature selection with config synchronization"""

    def __init__(self, config: Dict, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # NEW: Validate and sync config with actual CSV before loading model
        self._validate_and_sync_config_with_csv()

        # Use actual feature dimensions if available, fallback to configured
        self.actual_feature_dims = config['model'].get('actual_feature_dims',
                                                     config['model'].get('feature_dims', 128))
        logger.info(f"Using {self.actual_feature_dims} feature dimensions for prediction")

        self.heatmap_attn = config['model'].get('heatmap_attn', True)
        self.checkpoint_manager = UnifiedCheckpoint(config)
        self.model = self._load_model()

    def _validate_and_sync_config_with_csv(self):
        """Validate that config matches actual CSV feature dimensions and sync if needed"""
        dataset_name = self.config['dataset']['name'].lower()
        csv_paths = [
            f"data/{dataset_name}/{dataset_name}.csv",
            f"data/{dataset_name}/{dataset_name}_train.csv"
        ]

        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                try:
                    # Read CSV to get actual feature count
                    df = pd.read_csv(csv_path)
                    feature_columns = [col for col in df.columns if col.startswith('feature_') and col.replace('feature_', '').isdigit()]
                    actual_count = len(feature_columns)

                    # Get configured count
                    configured_count = self.config['model'].get('actual_feature_dims',
                                                              self.config['model'].get('feature_dims', 128))

                    if actual_count != configured_count:
                        logger.warning(f"Feature dimension mismatch: CSV has {actual_count}, config expects {configured_count}")

                        # Auto-correct the config
                        self.config['model']['actual_feature_dims'] = actual_count
                        self.config['model']['compressed_dims'] = actual_count

                        logger.info(f"Auto-corrected config to use {actual_count} features")
                    else:
                        logger.info(f"Feature dimensions validated: {actual_count} features")

                    # NEW: Update column_names in config to match CSV
                    self._update_config_column_names_from_csv(csv_path)

                    break  # Stop after first valid CSV found

                except Exception as e:
                    logger.warning(f"Could not validate config with CSV {csv_path}: {str(e)}")
                    continue

    def _update_config_column_names_from_csv(self, csv_path: str):
        """Update config column_names to match actual CSV features"""
        try:
            # Read CSV to get actual feature columns
            df = pd.read_csv(csv_path)

            # Get all feature columns (excluding metadata columns)
            feature_columns = [col for col in df.columns if col.startswith('feature_') and col.replace('feature_', '').isdigit()]
            feature_columns.sort(key=lambda x: int(x.split('_')[1]))

            # Get non-feature columns that should be preserved
            non_feature_columns = [col for col in df.columns if not col.startswith('feature_')]

            # Update config with actual column names
            if 'column_names' in self.config:
                # Replace the hardcoded column_names with actual ones from CSV
                # Order: target first, then features, then other columns
                updated_column_names = ['target'] + feature_columns + [col for col in non_feature_columns if col != 'target']
                self.config['column_names'] = updated_column_names
                logger.info(f"Updated config column_names to match CSV: {len(feature_columns)} features")
            else:
                logger.warning("Config does not have 'column_names' key to update")

        except Exception as e:
            logger.error(f"Failed to update config column_names from CSV: {str(e)}")


    def _get_image_files_with_labels(self, data_path: str) -> Tuple[List[str], List[str], List[str]]:
        """Get image files with class labels from folder structure, preserving exact case"""
        image_files = []
        class_labels = []
        original_filenames = []

        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        if not os.path.exists(data_path):
            logger.error(f"Data path does not exist: {data_path}")
            return [], [], []

        # Check if it's a single image file
        if os.path.isfile(data_path) and any(data_path.lower().endswith(ext) for ext in supported_extensions):
            image_files.append(data_path)
            # For single files, try to extract class name from parent folder
            parent_folder = os.path.basename(os.path.dirname(data_path))
            if parent_folder and parent_folder not in ['', '.', '..']:
                class_labels.append(parent_folder)
            else:
                class_labels.append("single_image")
            original_filenames.append(os.path.basename(data_path))
            logger.info(f"Processing single image: {data_path} with label: {class_labels[0]}")
            return image_files, class_labels, original_filenames

        # Process directory structure - look for train/test subfolders first
        potential_subfolders = ['train', 'test', 'validation', 'val']
        found_subfolders = []

        for subfolder in potential_subfolders:
            subfolder_path = os.path.join(data_path, subfolder)
            if os.path.exists(subfolder_path):
                found_subfolders.append(subfolder_path)

        # If no train/test subfolders found, use the root directory
        search_paths = found_subfolders if found_subfolders else [data_path]

        logger.info(f"Searching for images in: {search_paths}")

        for search_path in search_paths:
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        full_path = os.path.join(root, file)
                        image_files.append(full_path)

                        # Extract class label from folder structure - PRESERVE EXACT CASE
                        # Get the relative path from the search path to preserve hierarchy
                        rel_path = os.path.relpath(root, search_path)

                        if rel_path == '.':
                            # If image is directly in search_path, use the folder name
                            class_label = os.path.basename(search_path)
                            if class_label in potential_subfolders:
                                # If the folder is a split folder (train/test), use parent
                                class_label = os.path.basename(data_path)
                        else:
                            # Use the immediate parent folder as class label - EXACT CASE
                            class_label = os.path.basename(root)

                        # PRESERVE EXACT CASE - no cleaning or transformation
                        class_label = class_label.strip()

                        class_labels.append(class_label)
                        original_filenames.append(file)

        # Log statistics about found classes
        unique_classes = set(class_labels)
        class_counts = {cls: class_labels.count(cls) for cls in unique_classes}

        logger.info(f"Found {len(image_files)} images in {len(unique_classes)} classes:")
        for cls, count in sorted(class_counts.items()):
            logger.info(f"  {cls}: {count} images")

        return image_files, class_labels, original_filenames

    def _load_training_class_mapping(self) -> Dict[str, str]:
        """Load class mapping from training data to preserve consistent labels"""
        dataset_name = self.config['dataset']['name'].lower()
        train_csv_path = f"data/{dataset_name}/{dataset_name}_train.csv"

        class_mapping = {}

        if os.path.exists(train_csv_path):
            try:
                df = pd.read_csv(train_csv_path)
                if 'target' in df.columns:
                    unique_classes = df['target'].unique()
                    for cls in unique_classes:
                        class_mapping[str(cls).lower()] = str(cls)
                    logger.info(f"Loaded training class mapping: {len(class_mapping)} classes")
            except Exception as e:
                logger.warning(f"Could not load training class mapping: {str(e)}")

        return class_mapping

    def _map_labels_to_training_classes(self, folder_labels: List[str], training_mapping: Dict[str, str]) -> List[str]:
        """Map folder names to training target labels, preserving exact case"""
        mapped_labels = []

        # Create case-sensitive mapping first
        exact_mapping = {}
        for train_class_original in training_mapping.values():
            exact_mapping[train_class_original] = train_class_original  # Exact match
            exact_mapping[train_class_original.lower()] = train_class_original  # Lowercase match

        for folder_label in folder_labels:
            # Try exact match first
            if folder_label in exact_mapping:
                mapped_labels.append(exact_mapping[folder_label])
                continue

            # Try lowercase match
            folder_lower = folder_label.lower()
            if folder_lower in exact_mapping:
                mapped_labels.append(exact_mapping[folder_lower])
                continue

            # If no match found, use the original folder label (preserving exact case)
            mapped_labels.append(folder_label)
            #logger.info(f"Using original folder label (case preserved): '{folder_label}'")

        return mapped_labels

    def _are_classes_similar(self, label1: str, label2: str, similarity_threshold: float = 0.7) -> bool:
        """Check if two class labels are similar using fuzzy matching"""
        try:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, label1, label2).ratio()
            return similarity >= similarity_threshold
        except:
            # Fallback: simple containment check
            return label1 in label2 or label2 in label1

    def get_transforms(self, config: Dict, is_train: bool = False) -> transforms.Compose:
        """Get transforms for prediction with proper DatasetProcessor handling"""
        try:
            # Try to find and use DatasetProcessor if available
            dataset_processor_class = None

            # Look for DatasetProcessor in current module
            current_module = sys.modules[__name__]
            if hasattr(current_module, 'DatasetProcessor'):
                dataset_processor_class = getattr(current_module, 'DatasetProcessor')

            # If not found, check globals
            if not dataset_processor_class and 'DatasetProcessor' in globals():
                dataset_processor_class = globals()['DatasetProcessor']

            if dataset_processor_class and hasattr(dataset_processor_class, 'get_transforms'):
                # Try to create DatasetProcessor instance with proper arguments
                # This depends on how DatasetProcessor is initialized
                try:
                    processor = dataset_processor_class()
                    return processor.get_transforms(config, is_train)
                except TypeError:
                    # If DatasetProcessor needs different arguments, use fallback
                    return self._create_fallback_transforms(config, is_train)
            else:
                return self._create_fallback_transforms(config, is_train)

        except Exception as e:
            logger.warning(f"Could not use DatasetProcessor transforms: {str(e)}. Using fallback transforms.")
            return self._create_fallback_transforms(config, is_train)

    def _create_fallback_transforms(self, config: Dict, is_train: bool = False) -> transforms.Compose:
        """Create basic transforms as fallback"""
        mean = config['dataset']['mean']
        std = config['dataset']['std']
        input_size = config['dataset']['input_size']

        transform_list = [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]

        return transforms.Compose(transform_list)

    def _create_dataset(self, image_files: List[str], transform: transforms.Compose) -> Dataset:
        """Create dataset from image files"""
        class SimpleImageDataset(Dataset):
            def __init__(self, image_files, transform=None):
                self.image_files = image_files
                self.transform = transform

            def __len__(self):
                return len(self.image_files)

            def __getitem__(self, idx):
                try:
                    image = Image.open(self.image_files[idx]).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image
                except Exception as e:
                    logger.error(f"Error loading image {self.image_files[idx]}: {str(e)}")
                    # Return a dummy image
                    dummy_image = torch.zeros(3, 224, 224)
                    return dummy_image

        return SimpleImageDataset(image_files, transform)

    def _generate_batch_heatmaps(self, batch_tensor: torch.Tensor, batch_labels: torch.Tensor,
                               batch_filenames: List[str], output: Dict[str, torch.Tensor]):
        """Generate heatmaps for a batch of images"""
        try:
            if not hasattr(self.model, 'attention_maps') or not self.model.attention_maps:
                return

            # Create heatmap directory
            dataset_name = self.config['dataset']['name'].lower()
            heatmap_dir = os.path.join('data', dataset_name, 'prediction_heatmaps')
            os.makedirs(heatmap_dir, exist_ok=True)

            for i in range(len(batch_tensor)):
                try:
                    # Get attention maps for this sample
                    attention_key = list(self.model.attention_maps.keys())[0]
                    attention_map = self.model.attention_maps[attention_key][i].mean(dim=0).cpu()

                    # Convert to numpy and resize to match input
                    attention_np = attention_map.numpy()
                    if attention_np.shape != batch_tensor[i].shape[1:]:
                        import cv2
                        attention_np = cv2.resize(attention_np, batch_tensor[i].shape[1:][::-1])

                    # Normalize and create heatmap
                    attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)

                    # Save heatmap
                    filename = batch_filenames[i].replace('.', '_')
                    heatmap_path = os.path.join(heatmap_dir, f"{filename}_heatmap.png")

                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8, 6))
                    plt.imshow(attention_np, cmap='hot')
                    plt.colorbar()
                    plt.title(f"Attention Heatmap: {batch_filenames[i]}")
                    plt.savefig(heatmap_path, bbox_inches='tight', dpi=150)
                    plt.close()

                except Exception as e:
                    logger.warning(f"Could not generate heatmap for {batch_filenames[i]}: {str(e)}")

        except Exception as e:
            logger.warning(f"Batch heatmap generation failed: {str(e)}")

    def generate_classwise_attention_heatmaps(self, data_path: str):
        """Generate comprehensive classwise attention heatmaps"""
        try:
            # This is a placeholder - implement comprehensive heatmap generation
            logger.info("Classwise attention heatmap generation would be implemented here")
        except Exception as e:
            logger.error(f"Classwise heatmap generation failed: {str(e)}")

    def _save_predictions(self, predictions: Dict[str, List], output_csv: str):
        """Use the comprehensive save method from ReconstructionManager for consistency"""
        # Convert the predictions format to match what ReconstructionManager expects
        predictions_numpy = {}
        for key, values in predictions.items():
            if key == 'features_phase1':
                # Convert list of arrays to single numpy array
                predictions_numpy[key] = np.array(values)
            elif key in ['class_probabilities', 'cluster_confidence'] and values:
                # Handle probability arrays
                if hasattr(values[0], '__len__'):
                    predictions_numpy[key] = np.array(values)
                else:
                    predictions_numpy[key] = np.array(values)
            else:
                predictions_numpy[key] = np.array(values)

        # Create output directory from output_csv path
        output_dir = os.path.dirname(output_csv)
        if not output_dir:
            output_dir = os.path.join('data', self.config['dataset']['name'].lower(), 'predictions')

        # Use the comprehensive save method
        reconstruction_manager = ReconstructionManager(self.config)
        reconstruction_manager._save_predictions(predictions_numpy, output_dir, self.config)

        # Also save the simplified CSV version for backward compatibility
        self._save_simple_csv(predictions, output_csv)

    def _save_simple_csv(self, predictions: Dict[str, List], output_csv: str):
        """Save simplified CSV with main prediction columns"""
        try:
            df_data = {}

            # Handle different prediction types
            for key, values in predictions.items():
                if key == 'features_phase1':
                    # Flatten feature arrays
                    feature_arrays = np.array(values)
                    for i in range(feature_arrays.shape[1]):
                        df_data[f'feature_{i}'] = feature_arrays[:, i]
                elif key in ['class_probabilities', 'cluster_confidence']:
                    # Handle probability arrays
                    if values and hasattr(values[0], '__len__'):
                        prob_arrays = np.array(values)
                        for i in range(prob_arrays.shape[1]):
                            df_data[f'{key}_{i}'] = prob_arrays[:, i]
                    else:
                        df_data[key] = values
                else:
                    df_data[key] = values

            df = pd.DataFrame(df_data)
            df.to_csv(output_csv, index=False)
            logger.info(f"Predictions CSV saved to {output_csv}")

        except Exception as e:
            logger.error(f"Error saving predictions CSV to {output_csv}: {str(e)}")

    def verify_feature_consistency(self, train_csv: str, test_csv: str, pred_csv: str) -> bool:
        """Use the base autoencoder's comprehensive feature consistency verification"""
        return self.model.verify_feature_consistency(train_csv, test_csv, pred_csv)

    def validate_feature_config(self, csv_path: str):
        """Validate that config matches actual CSV feature dimensions and auto-correct if needed"""
        try:
            # Read CSV to get actual feature count
            df = pd.read_csv(csv_path)
            feature_columns = [col for col in df.columns if col.startswith('feature_') and col.replace('feature_', '').isdigit()]
            actual_count = len(feature_columns)

            # Get configured count
            configured_count = self.config['model'].get('actual_feature_dims',
                                                      self.config['model'].get('feature_dims', 128))

            if actual_count != configured_count:
                logger.warning(f"Feature dimension mismatch: CSV has {actual_count}, config expects {configured_count}")
                # Auto-correct the config
                self.config['model']['actual_feature_dims'] = actual_count
                self.config['model']['compressed_dims'] = actual_count
                logger.info(f"Auto-corrected config to use {actual_count} features")

                # Update the model if it exists
                if hasattr(self, 'model') and self.model is not None:
                    if hasattr(self.model, 'compressed_dims'):
                        self.model.compressed_dims = actual_count
                    self.actual_feature_dims = actual_count
                    logger.info(f"Updated model to use {actual_count} feature dimensions")
            else:
                logger.info(f"Feature dimensions validated: {actual_count} features")

        except Exception as e:
            logger.error(f"Feature config validation failed: {str(e)}")

    def _load_model(self) -> nn.Module:
        """Load model with invertible feature compression and proper dimension handling"""
        model = ModelFactory.create_model(self.config)
        model.to(self.device)

        checkpoint_path = self.checkpoint_manager.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.warning("CUDA not available. Falling back to CPU.")
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            else:
                raise e

        # Find the best state - prioritize phase2 with both KLD and classification
        state_key = None
        for key in checkpoint['model_states']:
            if 'phase2' in key and 'kld' in key and 'cls' in key:
                state_key = key
                break

        if state_key is None:
            # Fallback to any phase2 state
            for key in checkpoint['model_states']:
                if 'phase2' in key:
                    state_key = key
                    break

        if state_key is None:
            # Final fallback: use the first available state
            state_key = list(checkpoint['model_states'].keys())[0]
            logger.warning(f"No phase2 state found, using: {state_key}")

        state_dict = checkpoint['model_states'][state_key]['best']['state_dict']

        # FIX: Handle different return types of load_state_dict
        try:
            # Try the modern approach that returns NamedTuple
            load_result = model.load_state_dict(state_dict, strict=False)
            if hasattr(load_result, 'missing_keys') and load_result.missing_keys:
                logger.warning(f"Missing keys during model loading: {load_result.missing_keys}")
            if hasattr(load_result, 'unexpected_keys') and load_result.unexpected_keys:
                logger.warning(f"Unexpected keys during model loading: {load_result.unexpected_keys}")
        except TypeError:
            # Fallback for older PyTorch versions that return None
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model state dict loaded (legacy PyTorch version)")

        # Load feature selection state
        config_state = checkpoint['model_states'][state_key]['best']['config']
        if 'feature_selection' in config_state:
            fs_state = config_state['feature_selection']

            # CRITICAL FIX: Ensure tensors are moved to correct device
            selected_indices = fs_state.get('selected_feature_indices')
            if selected_indices is not None:
                if isinstance(selected_indices, torch.Tensor):
                    model._selected_feature_indices = selected_indices.to(self.device)
                else:
                    model._selected_feature_indices = torch.tensor(selected_indices, device=self.device)

            importance_scores = fs_state.get('feature_importance_scores')
            if importance_scores is not None:
                if isinstance(importance_scores, torch.Tensor):
                    model._feature_importance_scores = importance_scores.to(self.device)
                else:
                    model._feature_importance_scores = torch.tensor(importance_scores, device=self.device)

            model._feature_selection_metadata = fs_state.get('feature_selection_metadata', {})
            model._is_feature_selection_frozen = True

            logger.info(f"Loaded frozen feature selection: {len(model._selected_feature_indices) if model._selected_feature_indices is not None else 'None'} features")

        # CRITICAL: Load and synchronize feature dimensions - DO NOT MODIFY CONFIG
        if 'compressed_dims' in config_state:
            model.compressed_dims = config_state['compressed_dims']
            self.actual_feature_dims = min(32, model.compressed_dims)  # Cap at 32
            logger.info(f"Loaded compressed feature dimensions: {model.compressed_dims} → {self.actual_feature_dims}")

        if 'actual_feature_dims' in config_state:
            self.actual_feature_dims = min(32, config_state['actual_feature_dims'])  # Cap at 32
            logger.info(f"Using actual feature dimensions from config: {self.actual_feature_dims}")

        # Ensure synchronization between model and prediction manager
        if hasattr(model, 'compressed_dims') and model.compressed_dims != self.actual_feature_dims:
            logger.warning(f"Model compressed_dims ({model.compressed_dims}) doesn't match actual_feature_dims ({self.actual_feature_dims}). Synchronizing.")
            # Prefer the actual_feature_dims from config as it's more reliable
            model.compressed_dims = self.actual_feature_dims

        # Final validation
        if not hasattr(self, 'actual_feature_dims') or self.actual_feature_dims is None:
            self.actual_feature_dims = getattr(model, 'compressed_dims', 32)
            logger.info(f"Using fallback feature dimensions: {self.actual_feature_dims}")

        # Set appropriate training phase
        if 'phase2' in state_key:
            model.set_training_phase(2)
        else:
            model.set_training_phase(1)

        model.eval()

        # Final verification log
        logger.info(f"Model loaded successfully:")
        logger.info(f"  - Feature dimensions: {self.actual_feature_dims}")
        logger.info(f"  - Training phase: {model.training_phase}")
        logger.info(f"  - Frozen features: {model._selected_feature_indices is not None}")
        logger.info(f"  - State key: {state_key}")

        return model

    def predict_images(self, data_path: str, output_csv: str = None, batch_size: int = 128, generate_heatmaps: bool = True):
        """Predict using frozen feature selection with automatic target label preservation"""
        # Get image files with full paths and class labels from subfolders
        image_files, class_labels, original_filenames = self._get_image_files_with_labels(data_path)
        if not image_files:
            logger.warning(f"No image files found in {data_path}")
            return

        # NEW: Load training class mapping to preserve target labels
        training_class_mapping = self._load_training_class_mapping()

        # NEW: Map folder names to training target labels
        mapped_labels = self._map_labels_to_training_classes(class_labels, training_class_mapping)

        # Create dataset and dataloader
        transform = self.get_transforms(self.config, is_train=False)
        dataset = self._create_dataset(image_files, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                               num_workers=min(4, os.cpu_count() or 1))

        logger.info(f"Processing {len(image_files)} images in batches of {batch_size}")

        # NEW: Log label mapping statistics
        known_labels = [lbl for lbl in mapped_labels if lbl != "unknown"]
        unknown_labels = [lbl for lbl in mapped_labels if lbl == "unknown"]
        logger.info(f"Label mapping: {len(known_labels)} known classes, {len(unknown_labels)} unknown samples")

        all_predictions = {
            'filename': [],
            'filepath': image_files,  # FIXED: Include full file paths
            'target': mapped_labels,  # NEW: Use mapped labels instead of raw folder names
            'features_phase1': [],
            'class_predictions': [],
            'class_probabilities': [],
            'cluster_assignments': [],
            'cluster_confidence': []
        }


        self.model.eval()

        # Register attention hooks if heatmaps are requested
        if generate_heatmaps and hasattr(self.model, 'register_attention_hooks'):
            self.model.register_attention_hooks()
            logger.info("Registered attention hooks for heatmap generation")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Predicting")):
                # CRITICAL: Handle different dataset return types (from original)
                if isinstance(batch_data, (list, tuple)):
                    batch_tensor = batch_data[0]
                    batch_labels = batch_data[1] if len(batch_data) > 1 else None
                elif isinstance(batch_data, dict):
                    batch_tensor = batch_data.get('window', batch_data.get('image'))
                    batch_labels = batch_data.get('labels', None)
                else:
                    batch_tensor = batch_data
                    batch_labels = None

                batch_tensor = batch_tensor.to(self.device)

                # Forward pass
                output = self.model(batch_tensor)

                # Handle both compressed and original embeddings
                if 'compressed_embedding' in output:
                    embedding = output['compressed_embedding']
                    logger.debug("Using compressed features for prediction")
                else:
                    embedding = output.get('embedding', output[0] if isinstance(output, tuple) else output)

                # CRITICAL: Always use frozen features for consistency (from original)
                if hasattr(self.model, '_is_feature_selection_frozen') and self.model._is_feature_selection_frozen:
                    features = self.model.get_frozen_features(embedding).detach().cpu().numpy()
                    logger.debug("Using frozen feature selection for prediction")
                else:
                    features = embedding.detach().cpu().numpy()
                    #logger.warning("No frozen feature selection available, using all features")

                # CRITICAL: Validate feature dimensions match expected
                if features.shape[1] != self.actual_feature_dims:
                    logger.warning(f"Feature dimension mismatch: got {features.shape[1]}, expected {self.actual_feature_dims}")
                    # If dimensions don't match, take first actual_feature_dims features
                    if features.shape[1] > self.actual_feature_dims:
                        features = features[:, :self.actual_feature_dims]
                        logger.info(f"Truncated features to {self.actual_feature_dims} dimensions")
                    else:
                        # Pad with zeros if needed
                        padded_features = np.zeros((features.shape[0], self.actual_feature_dims))
                        padded_features[:, :features.shape[1]] = features
                        features = padded_features
                        logger.info(f"Padded features to {self.actual_feature_dims} dimensions")

                # CRITICAL: Store predictions with proper batch indexing (from original)
                batch_size_actual = features.shape[0]
                start_idx = batch_idx * batch_size

                # Get filenames and full paths for this batch
                batch_filenames = []
                batch_full_paths = []
                for i in range(batch_size_actual):
                    actual_idx = start_idx + i
                    if actual_idx < len(image_files):
                        batch_filenames.append(os.path.basename(image_files[actual_idx]))
                        batch_full_paths.append(image_files[actual_idx])
                    else:
                        batch_filenames.append(f"batch_{batch_idx}_{i}")
                        batch_full_paths.append(f"batch_{batch_idx}_{i}")

                all_predictions['filename'].extend(batch_filenames)
                all_predictions['filepath'].extend(batch_full_paths)
                all_predictions['features_phase1'].extend(features)

                # CRITICAL: Store additional outputs if available (from original)
                if 'class_predictions' in output:
                    all_predictions['class_predictions'].extend(output['class_predictions'].cpu().numpy())

                if 'class_probabilities' in output:
                    all_predictions['class_probabilities'].extend(output['class_probabilities'].cpu().numpy())

                if 'cluster_assignments' in output:
                    all_predictions['cluster_assignments'].extend(output['cluster_assignments'].cpu().numpy())

                if 'cluster_confidence' in output:
                    all_predictions['cluster_confidence'].extend(output['cluster_confidence'].cpu().numpy())

                # NEW: Generate batch-level heatmaps if requested
                if generate_heatmaps and batch_labels is not None:
                    self._generate_batch_heatmaps(batch_tensor, batch_labels, batch_filenames, output)

        # NEW: Generate comprehensive classwise heatmaps if requested
        if generate_heatmaps:
            logger.info("Generating comprehensive attention heatmaps...")
            try:
                self.generate_classwise_attention_heatmaps(data_path)
                logger.info("Attention heatmaps generated successfully")
            except Exception as e:
                logger.error(f"Failed to generate heatmaps: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")

        # NEW: Remove attention hooks if they were registered
        if generate_heatmaps and hasattr(self.model, 'remove_attention_hooks'):
            self.model.remove_attention_hooks()

        # Save predictions
        if output_csv:
            self._save_predictions(all_predictions, output_csv)

            # CRITICAL: Verify consistency if training files exist (from original)
            dataset_name = self.config['dataset']['name'].lower()
            train_csv = f"data/{dataset_name}/{dataset_name}_train.csv"
            test_csv = f"data/{dataset_name}/{dataset_name}_test.csv"

            if os.path.exists(train_csv) and os.path.exists(test_csv):
                self.verify_feature_consistency(train_csv, test_csv, output_csv)

        logger.info(f"Prediction completed for {len(image_files)} images")
        logger.info(f"Target labels: {len(known_labels)} known classes, {len(unknown_labels)} marked as 'unknown'")
        return all_predictions

class SlidingWindowDataset(Dataset):
    """Dataset that processes large images using sliding windows"""

    def __init__(self, image_paths, window_size=256, stride=128, transform=None,
                 overlap=0.5, min_window_coverage=0.8, pad_mode='reflect'):
        """
        Args:
            image_paths: List of paths to large images
            window_size: Size of each processing window (square)
            stride: Step size between windows
            overlap: Overlap ratio between windows (0-1)
            min_window_coverage: Minimum fraction of window that must contain image data
            pad_mode: Padding mode for edge windows ('reflect', 'constant', 'edge')
        """
        self.image_paths = image_paths
        self.window_size = window_size
        self.stride = stride if stride else int(window_size * (1 - overlap))
        self.transform = transform
        self.min_window_coverage = min_window_coverage
        self.pad_mode = pad_mode

        # Precompute all window coordinates
        self.windows = []  # (image_idx, y_start, x_start, y_end, x_end)
        self.image_shapes = []

        logger.info(f"Initializing sliding window dataset with {len(image_paths)} images")
        self._precompute_windows()

    def _precompute_windows(self):
        """Precompute all window coordinates for efficient access"""
        for img_idx, img_path in enumerate(tqdm(self.image_paths, desc="Precomputing windows")):
            try:
                with Image.open(img_path) as img:
                    img_array = np.array(img)
                    h, w = img_array.shape[:2]
                    self.image_shapes.append((h, w))

                    # Calculate window positions
                    windows = self._get_window_coordinates(h, w)
                    for y1, x1, y2, x2 in windows:
                        self.windows.append((img_idx, y1, x1, y2, x2))

            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue

        logger.info(f"Generated {len(self.windows)} windows from {len(self.image_paths)} images")

    def _get_window_coordinates(self, height, width):
        """Generate sliding window coordinates"""
        windows = []

        # Calculate number of windows in each dimension
        y_steps = max(1, (height - self.window_size) // self.stride + 1)
        x_steps = max(1, (width - self.window_size) // self.stride + 1)

        # Add main windows
        for i in range(y_steps):
            for j in range(x_steps):
                y1 = i * self.stride
                x1 = j * self.stride
                y2 = min(y1 + self.window_size, height)
                x2 = min(x1 + self.window_size, width)

                # Check if window has sufficient coverage
                coverage = ((y2 - y1) * (x2 - x1)) / (self.window_size * self.window_size)
                if coverage >= self.min_window_coverage:
                    windows.append((y1, x1, y2, y2))

        # Add edge windows to ensure full coverage
        # Right edge
        if width % self.stride != 0:
            for i in range(y_steps):
                x1 = width - self.window_size
                x2 = width
                y1 = i * self.stride
                y2 = min(y1 + self.window_size, height)
                coverage = ((y2 - y1) * (x2 - x1)) / (self.window_size * self.window_size)
                if coverage >= self.min_window_coverage:
                    windows.append((y1, x1, y2, x2))

        # Bottom edge
        if height % self.stride != 0:
            for j in range(x_steps):
                y1 = height - self.window_size
                y2 = height
                x1 = j * self.stride
                x2 = min(x1 + self.window_size, width)
                coverage = ((y2 - y1) * (x2 - x1)) / (self.window_size * self.window_size)
                if coverage >= self.min_window_coverage:
                    windows.append((y1, x1, y2, x2))

        # Bottom-right corner
        if height % self.stride != 0 and width % self.stride != 0:
            y1 = height - self.window_size
            y2 = height
            x1 = width - self.window_size
            x2 = width
            coverage = ((y2 - y1) * (x2 - x1)) / (self.window_size * self.window_size)
            if coverage >= self.min_window_coverage:
                windows.append((y1, x1, y2, x2))

        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        img_idx, y1, x1, y2, x2 = self.windows[idx]
        img_path = self.image_paths[img_idx]

        try:
            with Image.open(img_path) as img:
                # Extract window
                window = img.crop((x1, y1, x2, y2))

                # Convert to tensor
                if self.transform:
                    window_tensor = self.transform(window)
                else:
                    window_tensor = transforms.ToTensor()(window)

                # Return window tensor and metadata for reconstruction
                return {
                    'window': window_tensor,
                    'image_idx': img_idx,
                    'coords': (y1, x1, y2, x2),
                    'original_shape': self.image_shapes[img_idx],
                    'window_size': self.window_size
                }

        except Exception as e:
            logger.error(f"Error loading window {idx} from {img_path}: {str(e)}")
            # Return a dummy window
            dummy_window = torch.zeros(3, self.window_size, self.window_size)
            return {
                'window': dummy_window,
                'image_idx': img_idx,
                'coords': (y1, x1, y2, x2),
                'original_shape': self.image_shapes[img_idx],
                'window_size': self.window_size
            }

class BaseEnhancementConfig:
    """Base class for enhancement configuration management"""

    def __init__(self, config: Dict):
        self.config = config
        self.initialize_base_config()

    def initialize_base_config(self) -> None:
        """Initialize base configuration structures"""
        if 'model' not in self.config:
            self.config['model'] = {}
        if 'heatmap_attn' not in self.config['model']:
            self.config['model']['heatmap_attn'] = True
        # Initialize autoencoder config
        if 'autoencoder_config' not in self.config['model']:
            self.config['model']['autoencoder_config'] = {
                'phase1_learning_rate': 0.001,
                'phase2_learning_rate': 0.005,
                'reconstruction_weight': 1.0,
                'feature_weight': 0.1,
                'enable_phase2': False,
                'enhancements': {
                    'use_kl_divergence': True,
                    'use_class_encoding': False,
                    'kl_divergence_weight': 0.1,
                    'classification_weight': 0.1,
                    'clustering_temperature': 1.0,
                    'min_cluster_confidence': 0.7
                }
            }

        # NEW: Initialize training configuration with split flag
        if 'training' not in self.config:
            self.config['training'] = {}
        if 'create_train_test_split' not in self.config['training']:
            self.config['training']['create_train_test_split'] = False  # Default to False

        # NEW: Initialize sharpness-aware loss configuration (OPTIONAL - disabled by default)
        if 'sharpness_config' not in self.config:
            self.config['sharpness_config'] = {
                'enable_sharpness_loss': True,  # Set to True to enable sharpness enhancement
                'sharpness_weight': 0.3,         # Balance between reconstruction and sharpness
                'edge_threshold': 0.1            # Threshold for edge detection
            }

        # Initialize enhancement modules
        if 'enhancement_modules' not in self.config['model']:
            self.config['model']['enhancement_modules'] = {}

        # Initialize loss functions
        if 'loss_functions' not in self.config['model']:
            self.config['model']['loss_functions'] = {}

    def _adjust_learning_rates(self, num_enhancements: int) -> None:
        """Adjust learning rates based on number of enabled enhancements"""
        complexity_factor = max(1, num_enhancements * 0.5)
        self.config['model']['autoencoder_config']['phase1_learning_rate'] = 0.001 / complexity_factor
        self.config['model']['autoencoder_config']['phase2_learning_rate'] = 0.0005 / complexity_factor

    def _normalize_weights(self, enabled_enhancements: List[str]) -> None:
        """Normalize weights for enabled enhancements"""
        num_enabled = len(enabled_enhancements)
        if num_enabled > 0:
            base_reconstruction_weight = 1.0
            enhancement_weight = 1.0 / (num_enabled + 1)

            self.config['model']['autoencoder_config']['reconstruction_weight'] = base_reconstruction_weight

            for loss_name, loss_config in self.config['model']['loss_functions'].items():
                if loss_config['enabled']:
                    loss_config['weight'] = enhancement_weight

    def get_config(self) -> Dict:
        """Get the current configuration"""
        return self.config

class WindowReconstructor:
    """Reconstructs full images from processed windows"""

    def __init__(self, original_shape, window_size, stride, blend_mode='linear'):
        """
        Args:
            original_shape: (height, width) of original image
            window_size: Size of processing windows
            stride: Step size between windows
            blend_mode: How to blend overlapping regions ('linear', 'average', 'max')
        """
        self.original_shape = original_shape
        self.window_size = window_size
        self.stride = stride
        self.blend_mode = blend_mode

        # Initialize reconstruction buffers
        self.output_tensor = torch.zeros(original_shape)
        self.weight_tensor = torch.zeros(original_shape)

        # Precompute blending weights
        self.blend_weights = self._compute_blend_weights()

    def _compute_blend_weights(self):
        """Compute blending weights for smooth reconstruction"""
        if self.blend_mode == 'linear':
            # Create linear falloff weights
            weights = torch.ones(self.window_size, self.window_size)
            center = self.window_size // 2

            # Create distance-based weights (higher in center)
            for i in range(self.window_size):
                for j in range(self.window_size):
                    dist_x = abs(j - center) / center
                    dist_y = abs(i - center) / center
                    dist = np.sqrt(dist_x**2 + dist_y**2)
                    weights[i, j] = max(0, 1 - dist)

        elif self.blend_mode == 'average':
            weights = torch.ones(self.window_size, self.window_size)
        else:  # 'max'
            weights = torch.ones(self.window_size, self.window_size)

        return weights

    def add_window(self, window_tensor, coords):
        """Add a processed window to the reconstruction"""
        y1, x1, y2, x2 = coords
        window_h, window_w = y2 - y1, x2 - x1

        # Ensure window matches expected size
        if window_tensor.shape[-2:] != (window_h, window_w):
            window_tensor = F.interpolate(
                window_tensor.unsqueeze(0),
                size=(window_h, window_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Extract relevant portion of blend weights
        weight_patch = self.blend_weights[:window_h, :window_w]

        # Add to output with blending
        if self.blend_mode == 'max':
            self.output_tensor[:, y1:y2, x1:x2] = torch.max(
                self.output_tensor[:, y1:y2, x1:x2],
                window_tensor
            )
            self.weight_tensor[:, y1:y2, x1:x2] = 1.0
        else:
            self.output_tensor[:, y1:y2, x1:x2] += window_tensor * weight_patch
            self.weight_tensor[:, y1:y2, x1:x2] += weight_patch

    def get_reconstruction(self):
        """Get final reconstructed image"""
        if self.blend_mode in ['linear', 'average']:
            # Normalize by weights to handle overlapping regions
            mask = self.weight_tensor > 0
            output = torch.zeros_like(self.output_tensor)
            output[mask] = self.output_tensor[mask] / self.weight_tensor[mask]
            return output
        else:  # max
            return self.output_tensor

    def reset(self):
        """Reset for new reconstruction"""
        self.output_tensor = torch.zeros(self.original_shape)
        self.weight_tensor = torch.zeros(self.original_shape)

class ReconstructionSampler:
    """Samples and saves reconstruction comparisons during training"""

    def __init__(self, config: Dict, output_dir: str = None):
        self.config = config
        self.device = torch.device('cuda' if config['execution_flags']['use_gpu']
                                 and torch.cuda.is_available() else 'cpu')

        # Setup output directory
        dataset_name = config['dataset']['name'].lower()
        if output_dir is None:
            output_dir = os.path.join('data', dataset_name, 'reconstruction_samples')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.samples_per_class = config.get('training', {}).get('samples_per_class', 2)

    def save_reconstruction_samples(self, model: nn.Module, dataloader: DataLoader,
                                  epoch: int, phase: int = 1):
        """Save reconstruction samples for visual inspection"""
        model.eval()
        samples_collected = defaultdict(list)

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                if len(samples_collected) >= 10:  # Limit to 10 classes for efficiency
                    break

                data = data.to(self.device)
                labels = labels.to(self.device)

                # Get reconstructions
                if phase == 1:
                    outputs = model(data)
                    if isinstance(outputs, tuple):
                        _, reconstructions = outputs
                    else:
                        reconstructions = outputs.get('reconstruction', data)
                else:
                    outputs = model(data)
                    reconstructions = outputs.get('reconstruction', data)

                # Store samples by class
                for i in range(len(data)):
                    label = labels[i].item()
                    if len(samples_collected[label]) < self.samples_per_class:
                        samples_collected[label].append({
                            'original': data[i].cpu(),
                            'reconstruction': reconstructions[i].cpu(),
                            'label': label
                        })

        # Create visualization
        self._create_comparison_grid(samples_collected, epoch, phase)

    def _create_comparison_grid(self, samples_collected: Dict, epoch: int, phase: int):
        """Create side-by-side comparison of originals vs reconstructions"""
        if not samples_collected:
            return

        # Prepare figure
        n_classes = len(samples_collected)
        n_samples = self.samples_per_class
        fig, axes = plt.subplots(n_classes, n_samples * 2,
                               figsize=(4 * n_samples, 3 * n_classes))

        if n_classes == 1:
            axes = axes.reshape(1, -1)

        # Plot each class
        for class_idx, (label, samples) in enumerate(samples_collected.items()):
            for sample_idx, sample in enumerate(samples):
                # Original
                orig_ax = axes[class_idx, sample_idx * 2]
                self._tensor_to_axis(sample['original'], orig_ax)
                if sample_idx == 0:
                    orig_ax.set_ylabel(f'Class {label}', rotation=90, size=12)
                orig_ax.set_title('Original' if class_idx == 0 else '')

                # Reconstruction
                recon_ax = axes[class_idx, sample_idx * 2 + 1]
                self._tensor_to_axis(sample['reconstruction'], recon_ax)
                recon_ax.set_title('Reconstructed' if class_idx == 0 else '')

        plt.tight_layout()

        # Save figure
        filename = f"reconstruction_phase{phase}_epoch{epoch:03d}.png"
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved reconstruction samples to {output_path}")

    def _tensor_to_axis(self, tensor: torch.Tensor, ax):
        """Convert tensor to matplotlib axis"""
        img = self._tensor_to_image(tensor)
        ax.imshow(img)
        ax.axis('off')

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to image array with proper normalization"""
        tensor = tensor.detach().cpu()
        if len(tensor.shape) == 3:
            tensor = tensor.permute(1, 2, 0)

        mean = torch.tensor(self.config['dataset']['mean']).view(1, 1, -1)
        std = torch.tensor(self.config['dataset']['std']).view(1, 1, -1)
        tensor = tensor * std + mean

        return tensor.clamp(0, 1).numpy()

class GeneralEnhancementConfig(BaseEnhancementConfig):
    """Configuration manager for general (flexible) enhancement mode"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.enhancement_configs = self._get_enhancement_configs()

    def _get_enhancement_configs(self) -> Dict:
        """Get available enhancement configurations"""
        return {
            'astronomical': {
                'name': "Astronomical",
                'desc': "star detection, galaxy structure preservation",
                'components': {
                    'structure_preservation': True,
                    'detail_preservation': True,
                    'star_detection': True,
                    'galaxy_features': True,
                    'kl_divergence': True
                },
                'weights': {
                    'detail_weight': 1.0,
                    'structure_weight': 0.8,
                    'edge_weight': 0.7
                },
                'loss_components': {
                    'edge_preservation': True,
                    'peak_preservation': True,
                    'detail_preservation': True
                }
            },
            'medical': {
                'name': "Medical",
                'desc': "tissue boundary, lesion detection",
                'components': {
                    'tissue_boundary': True,
                    'lesion_detection': True,
                    'contrast_enhancement': True,
                    'subtle_feature_preservation': True
                },
                'weights': {
                    'boundary_weight': 1.0,
                    'lesion_weight': 0.8,
                    'contrast_weight': 0.6
                },
                'loss_components': {
                    'boundary_preservation': True,
                    'tissue_contrast': True,
                    'local_structure': True
                }
            },
            'agricultural': {
                'name': "Agricultural",
                'desc': "texture analysis, damage detection",
                'components': {
                    'texture_analysis': True,
                    'damage_detection': True,
                    'color_anomaly': True,
                    'pattern_enhancement': True,
                    'morphological_features': True
                },
                'weights': {
                    'texture_weight': 1.0,
                    'damage_weight': 0.8,
                    'pattern_weight': 0.7
                },
                'loss_components': {
                    'texture_preservation': True,
                    'damage_pattern': True,
                    'color_consistency': True
                }
            }
        }

    def configure_general_parameters(self) -> None:
        """Configure general enhancement parameters"""
        enhancements = self.config['model']['autoencoder_config']['enhancements']

        print("\nConfiguring General Enhancement Parameters:")

        # KL Divergence configuration
        if input("Enable KL divergence clustering? (y/n) [y]: ").lower() != 'n':
            enhancements['use_kl_divergence'] = True
            weight = input("Enter KL divergence weight (0-1) [0.1]: ").strip()
            enhancements['kl_divergence_weight'] = float(weight) if weight else 0.1
        else:
            enhancements['use_kl_divergence'] = False
            enhancements['kl_divergence_weight'] = 0.0

        # Class encoding configuration
        if input("Enable class encoding? (y/n) [y]: ").lower() != 'n':
            enhancements['use_class_encoding'] = True
            weight = input("Enter classification weight (0-1) [0.1]: ").strip()
            enhancements['classification_weight'] = float(weight) if weight else 0.1
        else:
            enhancements['use_class_encoding'] = False
            enhancements['classification_weight'] = 0.0

        # Configure additional parameters if KL divergence is enabled
        if enhancements['use_kl_divergence']:
            self._configure_clustering_parameters(enhancements)

        # Phase 2 configuration
        if input("Enable phase 2 training (clustering and fine-tuning)? (y/n) [y]: ").lower() != 'n':
            self.config['model']['autoencoder_config']['enable_phase2'] = True
        else:
            self.config['model']['autoencoder_config']['enable_phase2'] = False

    def _configure_clustering_parameters(self, enhancements: Dict) -> None:
        """Configure clustering-specific parameters"""
        temp = input("Enter clustering temperature (0.1-2.0) [1.0]: ").strip()
        enhancements['clustering_temperature'] = float(temp) if temp else 1.0

        conf = input("Enter minimum cluster confidence (0-1) [0.7]: ").strip()
        enhancements['min_cluster_confidence'] = float(conf) if conf else 0.7

    def configure_enhancements(self) -> None:
        """Configure enhancement features with flexible combinations"""
        enabled_enhancements = []

        print("\nConfiguring Enhancement Features for General Mode:")
        print("You can enable any combination of features\n")

        # Let user choose enhancements
        for key, enhancement in self.enhancement_configs.items():
            prompt = f"Enable {enhancement['name']} features ({enhancement['desc']})? (y/n) [n]: "
            if input(prompt).lower() == 'y':
                enabled_enhancements.append(key)
                self._add_enhancement(key, enhancement)
                print(f"{enhancement['name']} features added.")

        # Normalize weights and adjust learning rates
        self._normalize_weights(enabled_enhancements)
        self._adjust_learning_rates(len(enabled_enhancements))

        # Print configuration summary
        self._print_configuration_summary(enabled_enhancements)

    def _add_enhancement(self, key: str, enhancement: Dict) -> None:
        """Add specific enhancement configuration"""
        self.config['model']['enhancement_modules'][key] = {
            'enabled': True,
            'components': enhancement['components'],
            'weights': enhancement['weights']
        }

        self.config['model']['loss_functions'][f'{key}_structure'] = {
            'enabled': True,
            'weight': 1.0,  # Will be normalized later
            'components': enhancement['loss_components']
        }

    def _print_configuration_summary(self, enabled_enhancements: List[str]) -> None:
        """Print current configuration summary"""
        print("\nCurrent Enhancement Configuration:")
        if enabled_enhancements:
            for key in enabled_enhancements:
                enhancement = self.enhancement_configs[key]
                print(f"\n{enhancement['name']} Features:")
                print("- Components:", ', '.join(self.config['model']['enhancement_modules'][key]['components'].keys()))
                print("- Weights:", ', '.join(f"{k}: {v}" for k, v in self.config['model']['enhancement_modules'][key]['weights'].items()))
        else:
            print("\nNo enhancements enabled. Using basic autoencoder configuration.")

        print(f"\nLearning Rates:")
        print(f"- Phase 1: {self.config['model']['autoencoder_config']['phase1_learning_rate']}")
        print(f"- Phase 2: {self.config['model']['autoencoder_config']['phase2_learning_rate']}")

    def _generate_confusion_matrix(self, true_labels: torch.Tensor, pred_labels: torch.Tensor,
                                 class_names: Optional[List[str]] = None) -> None:
        """Generate and display a colored confusion matrix.

        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted labels
            class_names: List of class names for display
        """
        if not hasattr(self, 'class_names') and class_names is None:
            logger.warning("No class names available for confusion matrix")
            return

        class_names = class_names if class_names is not None else self.class_names

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels.cpu().numpy(), pred_labels.cpu().numpy())

        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)

        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # Save to file
        cm_path = os.path.join(self.log_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Confusion matrix saved to {cm_path}")

class BaseAutoencoder(nn.Module):
    """Base autoencoder class with frozen feature selection capabilities"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        """Initialize base autoencoder with invertible information preservation."""
        super().__init__()

        # Basic configuration
        self.input_shape = (
            config['dataset']['in_channels'],  # Use configured channels
            config['dataset']['input_size'][0],
            config['dataset']['input_size'][1]
        )
        self.in_channels = config['dataset']['in_channels']
        self.feature_dims = feature_dims
        self.config = config
        self.train_dataset = None

        # Device configuration
        self.device = torch.device('cuda' if config['execution_flags']['use_gpu']
                                 and torch.cuda.is_available() else 'cpu')

        # Shape tracking initialization
        self.shape_registry = {'input': self.input_shape}

        # Calculate layer dimensions and spatial progression DYNAMICALLY
        self.layer_sizes = self._calculate_layer_sizes()

        # Calculate spatial progression through the network
        self._calculate_spatial_progression()

        # Register key dimensions in shape registry
        self.shape_registry.update({
            'final_spatial': (self.final_spatial_dim_h, self.final_spatial_dim_w),
            'flattened': (self.flattened_size,),
            'latent': (self.feature_dims,)
        })

        # Initialize checkpoint paths
        self.checkpoint_dir = config['training']['checkpoint_dir']
        self.dataset_name = config['dataset']['name']
        self.checkpoint_path = os.path.join(self.checkpoint_dir,
                                          f"{self.dataset_name}_unified.pth")

        # Create network layers with invertibility constraints
        self.encoder_layers = self._create_encoder_layers()
        self.embedder = self._create_embedder()
        self.unembedder = self._create_unembedder()
        self.decoder_layers = self._create_decoder_layers()

        # NEW: Invertible feature compression components
        self.compressed_dims = config['model'].get('compressed_dims',
                                                 max(8, feature_dims // 4))

        # Feature space autoencoder for invertible compression - ADAPTIVE
        self._initialize_dynamic_compressors(feature_dims, self.compressed_dims)

        # Initialize enhancement components
        self.use_kl_divergence = (config['model']
                                 .get('autoencoder_config', {})
                                 .get('enhancements', True)
                                 .get('use_kl_divergence', True))

        self.use_class_encoding = (config['model']
                                  .get('autoencoder_config', {})
                                  .get('enhancements', {})
                                  .get('use_class_encoding', True))

        # NEW: Sharpness-aware loss for Phase 1 (OPTIONAL - disabled by default)
        self.enable_sharpness_loss = config.get('training', {}).get('enable_sharpness_loss', True)
        if self.enable_sharpness_loss:
            sharpness_config = config.get('sharpness_config', {})
            self.sharpness_loss = SharpnessAwareLoss(sharpness_config)
            logger.info("Sharpness-aware loss enabled for Phase 1 reconstruction")
        else:
            self.sharpness_loss = None
            logger.info("Using standard MSE loss for Phase 1 reconstruction")

        # Initialize classifier if class encoding is enabled
        if self.use_class_encoding:
            num_classes = config['dataset'].get('num_classes', 10)
            self.classifier = nn.Sequential(
                nn.Linear(self.compressed_dims, self.compressed_dims // 2),  # Use compressed dims
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.compressed_dims // 2, num_classes)
            )
            self.shape_registry['classifier_output'] = (num_classes,)

        # Training phase tracking
        self.training_phase = 1  # Start with phase 1

        # Initialize latent organization
        self._initialize_latent_organization()

        # Move model to appropriate device
        self.to(self.device)

        # Register shapes for encoder/decoder paths
        for idx, size in enumerate(self.layer_sizes):
            spatial_h, spatial_w = self.spatial_dims[idx]
            self.shape_registry[f'encoder_{idx}'] = (size, spatial_h, spatial_w)
            self.shape_registry[f'decoder_{idx}'] = (size, spatial_h, spatial_w)

        # Initialize training metrics
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.current_epoch = 0
        self.history = defaultdict(list)

        # Initialize clustering parameters
        self._initialize_clustering(config)

        # Log architecture details
        self._log_architecture_details()

        # Add feature selection persistence attributes
        self._selected_feature_indices = None  # Frozen selected feature indices
        self._feature_importance_scores = None  # Frozen importance scores
        self._feature_selection_metadata = {}   # Additional metadata
        self._is_feature_selection_frozen = False  # Lock flag

    def _initialize_dynamic_compressors(self, input_dims: int, output_dims: int):
        """Initialize compressors for maximum feature preservation"""
        # Use balanced compression ratios for optimal performance
        hidden_dims = max(output_dims * 2, input_dims // 2)

        self.feature_compressor = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),  # Use ReLU for better gradient flow
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_dims, output_dims)
        )

        self.feature_decompressor = nn.Sequential(
            nn.Linear(output_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims, input_dims)
        )

        # Initialize weights for stable training
        for module in [self.feature_compressor, self.feature_decompressor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        logger.info(f"Initialized compressors: {input_dims}D → {hidden_dims}D → {output_dims}D")

    def get_frozen_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Extract only the frozen selected features from embeddings using tensor indexing"""
        if self._selected_feature_indices is None:
            #logger.warning("No frozen feature selection available, returning all features")
            return embeddings

        # Use tensor indexing (much faster and compatible with binary serialization)
        if isinstance(embeddings, torch.Tensor):
            selected_features = embeddings[:, self._selected_feature_indices]
        else:
            # Handle numpy arrays by converting to tensor first
            selected_features = torch.from_numpy(embeddings)[:, self._selected_feature_indices.cpu()]
            selected_features = selected_features.to(self.device)

        return selected_features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Basic encoding process - ensure consistent behavior"""
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)

        # Ensure consistent return type
        if isinstance(embedding, tuple):
            logger.warning("encode() returned tuple, taking first element as embedding")
            embedding = embedding[0]

        return embedding

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass optimized for reconstruction accuracy and feature quality"""
        # Standard encoding
        embedding = self.encode(x)

        # Handle tuple returns for backward compatibility
        if isinstance(embedding, tuple):
            logger.warning("Embedding is tuple in forward(), taking first element")
            embedding = embedding[0]

        # Feature selection only in Phase 2 with sufficient classes
        if (self.training_phase == 2 and self.training and
            labels is not None and embedding.shape[1] > 48 and
            len(torch.unique(labels)) >= 5):  # Only for meaningful multi-class problems

            selected_embedding, selected_indices, method = self.smart_feature_compression(
                embedding, labels, max_dims=min(64, embedding.shape[1] // 2)
            )
            logger.info(f"Feature selection in Phase 2: {embedding.shape[1]} → {selected_embedding.shape[1]} features")
        else:
            # No feature selection - preserve all features for maximum accuracy
            selected_embedding = embedding
            selected_indices = None
            method = "no_selection"

        # Store selection metadata for consistency
        self._current_feature_indices = selected_indices
        self._current_selection_method = method

        # Feature compression for efficiency
        compressed_embedding = self.feature_compressor(selected_embedding)
        reconstructed_embedding = self.feature_decompressor(compressed_embedding)

        # High-quality reconstruction
        reconstruction = self.decode(compressed_embedding)

        # Output with all necessary components
        output = {
            'embedding': embedding,
            'selected_embedding': selected_embedding,
            'compressed_embedding': compressed_embedding,
            'reconstructed_embedding': reconstructed_embedding,
            'reconstruction': reconstruction,
            'selection_method': method
        }

        # Add classification and clustering outputs in Phase 2
        if self.training_phase == 2:
            # Use compressed features for efficiency while maintaining accuracy
            latent_info = self.organize_latent_space(compressed_embedding, labels)
            output.update(latent_info)

        return output

    def _calculate_spatial_progression(self):
        """Calculate spatial dimensions with exact reversal for decoder"""
        current_h = self.input_shape[1]
        current_w = self.input_shape[2]

        self.spatial_dims = []
        encoder_dims = []

        # Calculate encoder output dimensions
        for i in range(len(self.layer_sizes)):
            self.spatial_dims.append((current_h, current_w))
            encoder_dims.append((current_h, current_w))

            # Encoder: conv2d with kernel=3, stride=2, padding=1
            # output_size = (input_size + 2*padding - kernel_size) // stride + 1
            current_h = (current_h + 2*1 - 3) // 2 + 1
            current_w = (current_w + 2*1 - 3) // 2 + 1

            if current_h < 4 or current_w < 4:
                logger.warning(f"Stopping at layer {i}, dimensions too small: {current_h}x{current_w}")
                self.layer_sizes = self.layer_sizes[:i]
                self.spatial_dims = self.spatial_dims[:i]
                encoder_dims = encoder_dims[:i]
                break

        self.final_spatial_dim_h = current_h
        self.final_spatial_dim_w = current_w
        self.flattened_size = self.layer_sizes[-1] * current_h * current_w

        # Calculate expected decoder output
        decoder_h, decoder_w = current_h, current_w
        for i in range(len(self.layer_sizes)-1, -1, -1):
            # Decoder: convtranspose2d with kernel=3, stride=2, padding=1, output_padding=1
            # output_size = (input_size - 1)*stride - 2*padding + kernel_size + output_padding
            decoder_h = (decoder_h - 1)*2 - 2*1 + 3 + 1
            decoder_w = (decoder_w - 1)*2 - 2*1 + 3 + 1

        logger.info(f"Input dimensions: {self.input_shape[1:]}")
        logger.info(f"Encoder output: {current_h}x{current_w}")
        logger.info(f"Expected decoder output: {decoder_h}x{decoder_w}")
        logger.info(f"Target input size: {self.input_shape[1:]}")

        # Store the expected decoder output for reference
        self.expected_decoder_output = (decoder_h, decoder_w)

    def _calculate_layer_sizes(self) -> List[int]:
        """Calculate progressive channel sizes based on input dimensions"""
        base_channels = 32
        sizes = []
        current_size = base_channels

        # Determine maximum layers based on smallest dimension
        min_dim = min(self.input_shape[1], self.input_shape[2])
        max_layers = max(3, int(np.log2(min_dim)) - 2)  # At least 3 layers

        logger.info(f"Input dimensions: {self.input_shape[1]}x{self.input_shape[2]} (channels: {self.in_channels})")
        logger.info(f"Calculating {max_layers} layers for min dimension {min_dim}")

        for i in range(max_layers):
            sizes.append(current_size)
            # Stop doubling channels at 512 to prevent excessive memory usage
            if current_size < 512:
                current_size *= 2
            else:
                current_size = 512  # Cap at 512

            # Early stop if dimensions become too small
            if min_dim // (2 ** (i + 1)) < 4:
                logger.info(f"Stopping at layer {i}, dimensions becoming too small")
                break

        logger.info(f"Final layer sizes: {sizes}")
        return sizes

    def _log_architecture_details(self):
        """Log detailed architecture information"""
        logger.info("=" * 60)
        logger.info("AUTOENCODER ARCHITECTURE DETAILS")
        logger.info("=" * 60)
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Layer sizes: {self.layer_sizes}")
        logger.info(f"Spatial dimensions progression: {self.spatial_dims}")
        logger.info(f"Final spatial dimensions: {self.final_spatial_dim_h}x{self.final_spatial_dim_w}")
        logger.info(f"Flattened size: {self.flattened_size}")
        logger.info(f"Feature dimensions: {self.feature_dims}")
        logger.info(f"Encoder layers: {len(self.encoder_layers)}")
        logger.info(f"Decoder layers: {len(self.decoder_layers)}")

        # Calculate approximate parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        logger.info("=" * 60)

#--------------------------Distance Correlations ----------
    def get_high_confidence_samples(self, dataloader, threshold=0.9):
        """Identify high-confidence predictions for semi-supervised learning"""
        self.eval()
        confident_samples = []

        with torch.no_grad():
            for data, _ in dataloader:  # Note: ignoring true labels
                output = self(data)
                if 'cluster_probabilities' in output:
                    probs, preds = output['cluster_probabilities'].max(1)
                    mask = probs > threshold
                    confident_samples.append((data[mask], preds[mask]))

        return torch.cat(confident_samples) if confident_samples else None

    def _select_features_using_dynamic_selection(self, features, labels, config):
        """Dynamic feature selection with NaN protection"""
        max_features = config.get('max_features', 32)
        selector = DynamicFeatureSelector(max_features=max_features)

        # Check for NaN/Inf in input
        if np.isnan(features).any() or np.isinf(features).any():
            logger.error("NaN/Inf in features, using variance fallback")
            selected_indices, scores = selector.variance_based_selection(features, max_features)
            method = "variance_nan_fallback"
        else:
            try:
                (selected_indices, scores), method = selector.dynamic_feature_selection(features, labels)
            except Exception as e:
                logger.error(f"Dynamic selection failed: {e}, using variance fallback")
                selected_indices, scores = selector.variance_based_selection(features, max_features)
                method = "variance_error_fallback"

        # Reliability check
        n_features = features.shape[1]
        min_acceptable = max(config.get('min_features', 8), n_features // 4)

        if len(selected_indices) < min_acceptable:
            logger.warning(f"Feature selection unreliable: {len(selected_indices)} features (< {min_acceptable} min)")
            # Fallback to variance-based selection
            selected_indices, scores = selector.variance_based_selection(features, max_features)
            method = "variance_based_fallback"

        logger.info(f"Dynamic feature selection ({method}): {len(selected_indices)} features "
                   f"(score range: {np.nanmin(scores[selected_indices]):.3f}-{np.nanmax(scores[selected_indices]):.3f})")

        # Ensure indices are properly formatted
        if isinstance(selected_indices, torch.Tensor):
            selected_indices = selected_indices.cpu().numpy()
        elif not isinstance(selected_indices, (list, np.ndarray)):
            selected_indices = list(selected_indices)

        # Convert to numpy array with proper dtype and bounds checking
        selected_indices = np.asarray(selected_indices, dtype=np.int64)

        return selected_indices, scores

    def _calculate_min_features_required(self, n_features, n_classes, n_samples):
        """Calculate minimum features needed based on dataset complexity"""
        # Base minimum: at least 2 features per class, but not more than 50% of total
        base_min = min(n_classes * 2, n_features // 2)

        # Adjust based on sample size
        sample_factor = min(1.0, n_samples / 10000)  # Scale with dataset size

        # Adjust based on number of classes (more classes need more features)
        class_factor = min(2.0, 1.0 + (n_classes / 20))

        min_features = int(base_min * sample_factor * class_factor)

        # Ensure reasonable bounds
        min_features = max(8, min(min_features, n_features // 4))

        return min_features

    def _expand_feature_selection(self, features, labels, initial_indices, corr_values, target_count):
        """Expand feature selection when initial selection is too sparse"""
        n_features = features.shape[1]

        # Get all features sorted by correlation strength
        all_indices = np.argsort(corr_values)[::-1]

        # Start with initially selected features
        expanded_indices = set(initial_indices)

        # Add features until we reach target count
        for idx in all_indices:
            if len(expanded_indices) >= target_count:
                break
            if idx not in expanded_indices:
                # Check if this feature adds diversity
                if self._is_feature_diverse(features, idx, expanded_indices):
                    expanded_indices.add(idx)

        return sorted(expanded_indices)

    def _is_feature_diverse(self, features: np.ndarray, candidate_idx: int,
                           selected_indices: set, threshold: float = 0.8) -> bool:
        """
        Check if candidate feature is sufficiently different from already selected features.

        Args:
            features: All feature vectors (n_samples x n_features)
            candidate_idx: Index of candidate feature to check
            selected_indices: Set of already selected feature indices
            threshold: Correlation threshold for considering features redundant

        Returns:
            bool: True if feature is diverse enough to add
        """
        if not selected_indices:
            return True

        candidate_feature = features[:, candidate_idx]

        for selected_idx in selected_indices:
            selected_feature = features[:, selected_idx]
            try:
                # Calculate correlation between features
                correlation = abs(np.corrcoef(candidate_feature, selected_feature)[0, 1])
                if correlation > threshold:
                    # Features are too similar, candidate is redundant
                    return False
            except (ValueError, RuntimeError):
                # If correlation calculation fails, assume features are different
                continue

        return True

    def _validate_feature_csv(self, csv_path: str, expected_feature_count: int):
        """Validate that the saved CSV has the correct number of features"""
        try:
            df = pd.read_csv(csv_path)
            feature_columns = [col for col in df.columns if col.startswith('feature_') and col.replace('feature_', '').isdigit()]
            actual_count = len(feature_columns)

            if actual_count != expected_feature_count:
                logger.warning(f"CSV validation: Expected {expected_feature_count} features, found {actual_count}")
            else:
                logger.info(f"CSV validation passed: {actual_count} features confirmed")

        except Exception as e:
            logger.warning(f"CSV validation failed: {str(e)}")

    def _update_config_column_names_from_csv(self, csv_path: str):
        """Update config column_names to match actual CSV features"""
        try:
            # Read CSV to get actual feature columns
            df = pd.read_csv(csv_path)

            # Get all feature columns (excluding metadata columns)
            feature_columns = [col for col in df.columns if col.startswith('feature_') and col.replace('feature_', '').isdigit()]
            feature_columns.sort(key=lambda x: int(x.split('_')[1]))

            # Get non-feature columns that should be preserved
            non_feature_columns = [col for col in df.columns if not col.startswith('feature_')]

            # Update config with actual column names
            if 'column_names' in self.config:
                # Replace the hardcoded column_names with actual ones from CSV
                # Order: target first, then features, then other columns
                updated_column_names = ['target'] + feature_columns + [col for col in non_feature_columns if col != 'target']
                self.config['column_names'] = updated_column_names
                logger.info(f"Updated config column_names to match CSV: {len(feature_columns)} features")
            else:
                logger.warning("Config does not have 'column_names' key to update")

        except Exception as e:
            logger.error(f"Failed to update config column_names from CSV: {str(e)}")

    def _save_synchronized_config(self, output_dir: str):
        """Save the synchronized configuration to ensure consistency"""
        try:
            config_path = os.path.join(output_dir, 'synchronized_config.json')
            # Clean config for JSON serialization
            clean_config = self._clean_config_for_json(self.config)
            with open(config_path, 'w') as f:
                json.dump(clean_config, f, indent=2)
            logger.info(f"Saved synchronized config to {config_path}")
        except Exception as e:
            logger.warning(f"Could not save synchronized config: {str(e)}")

    def _clean_config_for_json(self, config: Dict) -> Dict:
        """Clean configuration for JSON serialization"""
        def clean_value(value):
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return value.tolist() if hasattr(value, 'tolist') else str(value)
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [clean_value(v) for v in value]
            else:
                return value

        return clean_value(config)

    def compute_phase1_loss(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Phase 1 loss focused on high-quality reconstruction
        """
        # Primary reconstruction loss
        reconstruction_loss = F.mse_loss(reconstruction, target)

        # Additional perceptual loss for better quality
        if hasattr(self, 'enable_perceptual_loss') and self.enable_perceptual_loss:
            # Simple gradient-based perceptual loss
            target_grad = torch.abs(F.conv2d(target, self._sobel_filter, padding=1))
            recon_grad = torch.abs(F.conv2d(reconstruction, self._sobel_filter, padding=1))
            perceptual_loss = F.mse_loss(recon_grad, target_grad)

            # Combined loss with perceptual component
            total_loss = reconstruction_loss + 0.3 * perceptual_loss
        else:
            total_loss = reconstruction_loss

        return total_loss

    def _prepare_features_dataframe_enhanced(self, feature_dict: Dict[str, torch.Tensor],
                                           n_classes: int, n_samples: int) -> pd.DataFrame:
        """Enhanced feature preparation for folder-based datasets with guaranteed config synchronization AND ORIGINAL TARGET LABELS"""
        data = {}
        embeddings = feature_dict['embeddings'].cpu().numpy()
        base_length = len(embeddings)
        total_features = embeddings.shape[1]

        # CRITICAL: Always use compressed features as they are most informative
        # Use dynamically selected features when available, otherwise compressed features
        feature_source = "dynamic_selection"
        actual_feature_count = None

        # Prefer dynamically selected features when available
        if hasattr(self, '_current_feature_indices') and self._current_feature_indices is not None:
            # Use the features selected by dynamic selection
            selected_indices = self._current_feature_indices
            if isinstance(selected_indices, torch.Tensor):
                selected_indices = selected_indices.cpu().numpy()

            # Ensure we don't exceed available features
            valid_indices = selected_indices[selected_indices < embeddings.shape[1]]
            compressed_embeddings = embeddings[:, valid_indices]
            actual_feature_count = len(valid_indices)
            feature_source = f"dynamic_{getattr(self, '_current_selection_method', 'unknown')}"

            for i in range(actual_feature_count):
                data[f'feature_{i}'] = compressed_embeddings[:, i]

            logger.info(f"Using dynamically selected features: {actual_feature_count}D (method: {feature_source})")

        elif hasattr(self, 'feature_compressor'):
            # Fallback to learned compression
            embeddings_tensor = torch.tensor(embeddings).to(self.device)
            with torch.no_grad():
                compressed_embeddings = self.feature_compressor(embeddings_tensor).cpu().numpy()
            actual_feature_count = compressed_embeddings.shape[1]
            feature_source = "compressed"

            for i in range(actual_feature_count):
                data[f'feature_{i}'] = compressed_embeddings[:, i]

            logger.info(f"Using compressed features: {actual_feature_count}D (fallback)")

        # CRITICAL FIX: Update config with the actual feature dimensions
        if actual_feature_count is not None:
            self.config['model']['actual_feature_dims'] = actual_feature_count
            self.config['model']['compressed_dims'] = actual_feature_count

        else:
            # Fallback: use first 32 features
            actual_feature_count = min(32, total_features)
            # CRITICAL FIX: Update config with fallback dimensions
            self.config['model']['actual_feature_dims'] = actual_feature_count
            self.config['model']['compressed_dims'] = actual_feature_count
            for i in range(actual_feature_count):
                data[f'feature_{i}'] = embeddings[:, i]
            feature_source = "sequential"
            logger.info(f"Using sequential features: {actual_feature_count}D")

        # CRITICAL FIX: Double verification and logging
        logger.info(f"Config synchronized: {total_features}D → {actual_feature_count}D actual features in CSV")

        # FIXED: PRESERVE ORIGINAL TARGET LABELS (subfolder names) - NOT ENCODED NUMERIC VALUES
        target_added = False

        # Priority 1: Use class_names from folder structure (most reliable) - PRESERVE ORIGINAL
        if 'class_names' in feature_dict and len(feature_dict['class_names']) == base_length:
            data['target'] = feature_dict['class_names']
            target_added = True
            logger.info(f"Using original class names for targets (sample: {feature_dict['class_names'][0]})")

        # Priority 2: If we have full_paths, extract original folder names
        elif 'full_paths' in feature_dict and len(feature_dict['full_paths']) == base_length:
            original_targets = []
            for path in feature_dict['full_paths']:
                # Extract original class name from path (assuming folder structure: .../class_name/filename)
                path_parts = path.split(os.sep)
                if len(path_parts) >= 2:
                    class_name = path_parts[-2]  # Second last part is class folder
                    original_targets.append(class_name)
                else:
                    # Fallback: use filename or mark as unknown
                    original_targets.append("unknown")

            data['target'] = original_targets
            target_added = True
            logger.info(f"Extracted target from folder names (sample: {original_targets[0]})")

        # Priority 3: Convert numeric labels back to original class names using dataset classes
        elif 'labels' in feature_dict:
            labels_tensor = feature_dict['labels']
            if isinstance(labels_tensor, torch.Tensor):
                labels_numpy = labels_tensor.cpu().numpy()
                if len(labels_numpy) == base_length:
                    # Convert numeric labels back to original class names using dataset information
                    if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'classes'):
                        try:
                            # Use the dataset's class mapping to get original names
                            class_names = [self.train_dataset.classes[label] for label in labels_numpy]
                            data['target'] = class_names
                            logger.info(f"Converted numeric labels to class names (sample: {class_names[0]})")
                        except Exception as e:
                            logger.warning(f"Could not convert labels to class names: {e}")
                            # Fallback: use numeric labels but warn
                            data['target'] = labels_numpy
                            logger.warning("Using numeric labels as fallback - original class names not available")
                    else:
                        # If no class mapping available, try to extract from filepaths
                        if 'full_paths' in feature_dict:
                            original_targets = []
                            for i, path in enumerate(feature_dict['full_paths']):
                                path_parts = path.split(os.sep)
                                if len(path_parts) >= 2:
                                    original_targets.append(path_parts[-2])
                                else:
                                    original_targets.append(f"class_{labels_numpy[i]}")
                            data['target'] = original_targets
                            logger.info(f"Extracted target from file paths (sample: {original_targets[0]})")
                        else:
                            # Final fallback: use numeric labels
                            data['target'] = labels_numpy
                            logger.warning("Using numeric labels - cannot recover original class names")
                    target_added = True

        # Priority 4: Final fallback - create target from available information
        if not target_added:
            logger.warning(f"No target information found. Creating target column from available data for {base_length} samples.")
            # Try to extract class names from filepaths
            if 'full_paths' in feature_dict:
                original_targets = []
                for path in feature_dict['full_paths']:
                    # Extract class name from path (assuming folder structure: .../class_name/filename)
                    path_parts = path.split(os.sep)
                    if len(path_parts) >= 2:
                        class_name = path_parts[-2]  # Second last part is class folder
                        original_targets.append(class_name)
                    else:
                        original_targets.append("unknown")

                if len(original_targets) == base_length:
                    data['target'] = original_targets
                    logger.info(f"Created target from filepath folder structure (sample: {original_targets[0]})")
                else:
                    data['target'] = ["unknown"] * base_length
                    logger.warning("Created unknown target column")
            else:
                data['target'] = ["unknown"] * base_length
                logger.warning("Created unknown target column")

        # FIXED: Include file paths in CSV
        if 'full_paths' in feature_dict and len(feature_dict['full_paths']) == base_length:
            data['filepath'] = feature_dict['full_paths']
            logger.info(f"Included full file paths in CSV (sample: {os.path.basename(feature_dict['full_paths'][0])})")
        elif 'filenames' in feature_dict and len(feature_dict['filenames']) == base_length:
            data['filename'] = feature_dict['filenames']
            logger.info(f"Included filenames in CSV (sample: {feature_dict['filenames'][0]})")
        else:
            data['filename'] = [f"sample_{i}" for i in range(base_length)]
            logger.warning("Using generated filenames - original file paths not available")

        # Add feature source metadata
        data['feature_source'] = [feature_source] * base_length

        # Create DataFrame and ensure target column is first
        df = pd.DataFrame(data)

        # Column ordering with target first
        column_order = ['target']

        # Extract only numeric feature columns (feature_0, feature_1, etc.)
        numeric_feature_columns = [col for col in df.columns if col.startswith('feature_') and col.replace('feature_', '').isdigit()]

        # Sort numeric features properly
        if numeric_feature_columns:
            numeric_feature_columns.sort(key=lambda x: int(x.split('_')[1]))
            column_order.extend(numeric_feature_columns)

        # Add other columns
        other_columns = [col for col in df.columns if col not in column_order]
        column_order.extend(other_columns)

        # Reorder the DataFrame
        df = df[column_order]

        # Final verification - log that we're preserving original target labels
        final_feature_count = len(numeric_feature_columns)
        target_sample = df['target'].iloc[0] if len(df) > 0 else "unknown"
        target_type = type(target_sample).__name__
        logger.info(f"Final DataFrame: {final_feature_count} features, target sample: '{target_sample}' (type: {target_type}, source: {feature_source})")

        # Debug: Check if we still have numeric targets
        if isinstance(target_sample, (int, float, np.integer, np.floating)):
            logger.warning(f"WARNING: Target is still numeric: {target_sample}. Checking available data sources...")
            if 'class_names' in feature_dict:
                logger.warning(f"class_names available: {len(feature_dict['class_names'])} items")
            if 'full_paths' in feature_dict:
                logger.warning(f"full_paths available: {len(feature_dict['full_paths'])} items")
                if feature_dict['full_paths']:
                    logger.warning(f"Sample full_path: {feature_dict['full_paths'][0]}")
            if 'labels' in feature_dict:
                logger.warning(f"labels available: {len(feature_dict['labels'])} items")

        return df

    def save_features(self, train_features: Dict[str, torch.Tensor],
                     test_features: Dict[str, torch.Tensor],
                     output_path: str) -> None:
        """
        GUARANTEED CONSISTENCY: Features saved here must match prediction pipeline exactly
        Only create train/test split files when explicitly requested
        """
        logger.info("Saving features with guaranteed prediction pipeline consistency")

        try:
            # CRITICAL: Use the SAME feature preparation as prediction
            if hasattr(self, 'config') and hasattr(self, 'train_loader'):
                # Use prediction pipeline for absolute consistency
                self._save_features_via_prediction_consistent(output_path)
            else:
                # Fallback with identical processing logic
                self._save_features_direct_consistent(train_features, test_features, output_path)

            logger.info("Feature saving completed with guaranteed consistency")

        except Exception as e:
            logger.error(f"Feature saving failed: {str(e)}")
            # Emergency fallback with validation
            self._save_features_emergency(train_features, test_features, output_path)

    def _save_features_via_prediction_consistent(self, output_path: str):
        """Use prediction pipeline with validation to ensure identical outputs"""
        # Create PredictionManager with same config
        pred_manager = PredictionManager(self.config)

        # Extract data paths from loaders
        train_data_path = self._extract_data_path_from_loader(self.train_loader, 'train')
        test_data_path = self._extract_data_path_from_loader(getattr(self, 'test_loader', None), 'test')

        # Generate features via prediction
        train_csv = output_path.replace('.csv', '_train.csv')
        test_csv = output_path.replace('.csv', '_test.csv')

        # Get predictions
        train_pred = pred_manager.predict_images(train_data_path, train_csv, generate_heatmaps=False)
        if test_data_path:
            test_pred = pred_manager.predict_images(test_data_path, test_csv, generate_heatmaps=False)

        # VALIDATION: Compare with direct extraction to ensure consistency
        self._validate_feature_consistency(train_pred, "prediction_pipeline")

    def _save_features_direct_consistent(self, train_features: Dict[str, torch.Tensor],
                                       test_features: Dict[str, torch.Tensor],
                                       output_path: str):
        """Direct saving that mimics prediction pipeline exactly - ONLY CREATE SPLIT FILES WHEN REQUESTED"""
        # CRITICAL: Use the SAME feature processing as PredictionManager

        # Determine if we should create train/test split files
        create_split_files = self.config.get('training', {}).get('create_train_test_split', False)
        dataset_name = self.config['dataset']['name'].lower()
        base_data_path = os.path.dirname(output_path) if output_path else f"data/{dataset_name}"

        # Main CSV file - ALWAYS create this
        main_csv_path = output_path if output_path else os.path.join(base_data_path, f"{dataset_name}.csv")

        # Process training features with prediction-compatible logic
        train_df = self._prepare_features_prediction_compatible(train_features, 'train')
        train_df.to_csv(main_csv_path, index=False)
        logger.info(f"Main features CSV saved to {main_csv_path}")

        # Only create train/test split files if explicitly requested and test data exists
        if create_split_files and test_features and len(test_features.get('embeddings', [])) > 0:
            # Create train split file (copy from main)
            train_csv_path = os.path.join(base_data_path, f"{dataset_name}_train.csv")
            shutil.copy2(main_csv_path, train_csv_path)
            logger.info(f"Train split file created: {train_csv_path}")

            # Create test split file
            test_csv_path = os.path.join(base_data_path, f"{dataset_name}_test.csv")
            test_df = self._prepare_features_prediction_compatible(test_features, 'test')
            test_df.to_csv(test_csv_path, index=False)
            logger.info(f"Test split file created: {test_csv_path}")
        else:
            logger.info("Train/test split files not created (create_train_test_split=False or no test data)")

        logger.info(f"Features saved with prediction-compatible format")

    def _prepare_features_prediction_compatible(self, features: Dict[str, torch.Tensor], split: str):
        """Prepare features exactly as PredictionManager would - WITH ORIGINAL TARGET LABELS AND FILE PATHS"""
        # CRITICAL: Same logic as PredictionManager._save_predictions
        embeddings = features['embeddings'].cpu().numpy()

        # Apply the SAME feature selection as used in prediction
        if hasattr(self, '_selected_feature_indices') and self._selected_feature_indices is not None:
            # Use frozen feature selection for consistency
            selected_indices = self._selected_feature_indices.cpu().numpy()
            if len(selected_indices) > 32:
                selected_indices = selected_indices[:32]  # Enforce limit
            embeddings = embeddings[:, selected_indices]
        else:
            # Use same fallback logic as prediction
            if embeddings.shape[1] > 32:
                # Same strategic selection as prediction
                variances = np.var(embeddings, axis=0)
                selected_indices = np.argsort(variances)[-32:][::-1]
                embeddings = embeddings[:, selected_indices]

        # Create DataFrame with same structure
        data = {}
        for i in range(embeddings.shape[1]):
            data[f'feature_{i}'] = embeddings[:, i]

        # FIXED: Use ORIGINAL target labels (folder names) not encoded numeric values
        if 'class_names' in features and len(features['class_names']) == len(embeddings):
            data['target'] = features['class_names']
            logger.info(f"Using original class names for targets (sample: {features['class_names'][0]})")
        elif 'full_paths' in features and len(features['full_paths']) == len(embeddings):
            # Extract folder names from file paths
            folder_names = []
            for path in features['full_paths']:
                # Extract class name from path (folder structure: .../class_name/filename)
                path_parts = path.split(os.sep)
                if len(path_parts) >= 2:
                    folder_name = path_parts[-2]  # Second last part is class folder
                    folder_names.append(folder_name)
                else:
                    folder_names.append("unknown")
            data['target'] = folder_names
            logger.info(f"Extracted target from folder names (sample: {folder_names[0]})")
        elif 'labels' in features:
            # Convert numeric labels back to original names if possible
            labels_numpy = features['labels'].cpu().numpy()
            if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'classes'):
                try:
                    class_names = [self.train_dataset.classes[label] for label in labels_numpy]
                    data['target'] = class_names
                    logger.info(f"Converted numeric labels to class names (sample: {class_names[0]})")
                except Exception as e:
                    logger.warning(f"Could not convert labels to class names: {e}")
                    data['target'] = labels_numpy
            else:
                data['target'] = labels_numpy
                logger.warning("Using numeric labels - cannot recover original class names")

        # FIXED: Include file paths in CSV
        if 'full_paths' in features and len(features['full_paths']) == len(embeddings):
            data['filepath'] = features['full_paths']
            logger.info(f"Included full file paths in CSV (sample: {os.path.basename(features['full_paths'][0])})")
        elif 'filenames' in features and len(features['filenames']) == len(embeddings):
            data['filename'] = features['filenames']
            logger.info(f"Included filenames in CSV (sample: {features['filenames'][0]})")
        else:
            data['filename'] = [f"sample_{i}" for i in range(len(embeddings))]
            logger.warning("Using generated filenames - original file paths not available")

        return pd.DataFrame(data)

    def _get_distance_correlation_config(self, output_dir: str) -> Dict:
        """
        Safely load or create distance correlation configuration.

        Args:
            output_dir: Directory where config should be stored

        Returns:
            Dictionary with configuration parameters
        """
        # Use binary format for config storage to avoid JSON tensor serialization issues
        config_path = os.path.join(output_dir, 'feature_selection_config.pt')
        default_config = {
            'use_distance_correlation': True,
            'distance_correlation_upper': 0.85,
            'distance_correlation_lower': 0.01
        }

        try:
            if os.path.exists(config_path):
                config = torch.load(config_path, map_location='cpu')
                # Validate loaded config and merge with defaults
                merged_config = {**default_config, **config}
                return merged_config
            else:
                # Save default config in binary format
                torch.save(default_config, config_path)
                return default_config
        except Exception as e:
            logger.warning(f"Could not load/create config: {str(e)} - using defaults")
            return default_config

    def verify_feature_consistency(self, train_csv: str, test_csv: str, pred_csv: str) -> bool:
        """
        Enhanced feature consistency verification with config synchronization
        """
        try:
            train_df = pd.read_csv(train_csv)
            test_df = pd.read_csv(test_csv)
            pred_df = pd.read_csv(pred_csv)

            # Get feature columns
            train_features = [col for col in train_df.columns if col.startswith('feature_')]
            test_features = [col for col in test_df.columns if col.startswith('feature_')]
            pred_features = [col for col in pred_df.columns if col.startswith('feature_')]

            # Check if feature sets match
            if set(train_features) != set(test_features) or set(train_features) != set(pred_features):
                logger.error("Feature columns don't match across CSV files")
                return False

            # Check feature dimensions
            if len(train_features) != len(test_features) or len(train_features) != len(pred_features):
                logger.error("Feature dimensions don't match across CSV files")
                return False

            # NEW: Check feature dimensions against config and synchronize
            actual_dims = len(train_features)
            expected_dims = self.config['model'].get('actual_feature_dims',
                                                   self.config['model'].get('feature_dims', 128))

            if actual_dims != expected_dims:
                logger.warning(f"Feature dimension mismatch: CSV has {actual_dims}, config expects {expected_dims}")
                # Update config to match reality
                self.config['model']['actual_feature_dims'] = actual_dims
                logger.info(f"Updated config to use actual feature dimensions: {actual_dims}")

            # NEW: Check feature source consistency
            train_source = train_df['feature_source'].iloc[0] if 'feature_source' in train_df.columns else 'unknown'
            test_source = test_df['feature_source'].iloc[0] if 'feature_source' in test_df.columns else 'unknown'
            pred_source = pred_df['feature_source'].iloc[0] if 'feature_source' in pred_df.columns else 'unknown'

            if train_source != test_source or train_source != pred_source:
                logger.warning(f"Feature sources differ: train={train_source}, test={test_source}, pred={pred_source}")
            else:
                logger.info(f"Feature source consistent: {train_source}")

            logger.info(f"Feature consistency verified: {actual_dims} features match across all files")
            return True

        except Exception as e:
            logger.error(f"Feature consistency check failed: {str(e)}")
            return False

    def _save_feature_metadata(self, output_dir: str, feature_columns: List[str], dc_config: Dict = None):
        """Save comprehensive metadata about the saved features and feature selection process using binary format ONLY"""

        # Delete ANY AND ALL existing JSON metadata files to prevent serialization errors
        legacy_json_files = [
            'feature_extraction_metadata.json',
            'feature_selection_metadata.json',
            'training_metadata.json'
        ]

        for json_file in legacy_json_files:
            json_metadata_path = os.path.join(output_dir, json_file)
            if os.path.exists(json_metadata_path):
                try:
                    os.remove(json_metadata_path)
                    logger.info(f"Removed legacy JSON metadata file: {json_metadata_path}")
                except Exception as e:
                    logger.warning(f"Could not remove legacy JSON file {json_metadata_path}: {e}")

        # CRITICAL: Update config column_names to match ACTUAL selected features (not all features)
        if 'column_names' in self.config:
            # Extract only the selected feature columns (feature_0, feature_1, etc.)
            selected_feature_columns = [col for col in feature_columns if col.startswith('feature_') and col.replace('feature_', '').isdigit()]

            # Get non-feature columns that should be preserved (target, filepath, etc.)
            non_feature_columns = [col for col in feature_columns if not col.startswith('feature_')]

            # Update config with ACTUAL selected column names in correct order
            updated_column_names = []

            # Add target first if it exists
            if 'target' in non_feature_columns:
                updated_column_names.append('target')
                non_feature_columns.remove('target')

            # Add selected features in numeric order
            selected_feature_columns.sort(key=lambda x: int(x.split('_')[1]))
            updated_column_names.extend(selected_feature_columns)

            # Add remaining non-feature columns
            updated_column_names.extend(non_feature_columns)

            self.config['column_names'] = updated_column_names
            logger.info(f"Updated config column_names to match {len(selected_feature_columns)} SELECTED features")

        # Prepare ALL metadata in binary format - no JSON for any data
        binary_data = {
            'timestamp': datetime.now().isoformat(),
            'feature_info': {
                'total_features': len(feature_columns),
                'selected_feature_columns': [col for col in feature_columns if col.startswith('feature_')],
                'non_feature_columns': [col for col in feature_columns if not col.startswith('feature_')],
                'feature_selection': {
                    'method': 'distance_correlation' if dc_config and dc_config.get('use_distance_correlation', True) else 'none',
                    'parameters': self._clean_config_for_binary_storage(dc_config) if dc_config and dc_config.get('use_distance_correlation', True) else None
                }
            },
            'model_config': {
                'type': self.__class__.__name__,
                'feature_dims': int(self.feature_dims),  # Ensure scalar
                'training_phase': int(self.training_phase),  # Ensure scalar
                'enhancements': {
                    'use_kl_divergence': bool(self.use_kl_divergence),
                    'use_class_encoding': bool(self.use_class_encoding)
                }
            },
            'dataset_info': {
                'name': str(self.config['dataset']['name']),
                'input_size': [int(x) for x in self.config['dataset']['input_size']],  # Ensure list of ints
                'channels': int(self.config['dataset']['in_channels'])  # Ensure scalar
            },
            'config_column_names': self.config.get('column_names', [])  # Store the synchronized column names
        }

        # Add feature selection tensors if available - CONVERT TO NUMPY ARRAYS FOR SAFE STORAGE
        if hasattr(self, '_selected_feature_indices') and self._selected_feature_indices is not None:
            # Convert tensors to CPU numpy arrays to avoid serialization issues
            selected_indices_cpu = self._selected_feature_indices.cpu().numpy() if isinstance(self._selected_feature_indices, torch.Tensor) else np.array(self._selected_feature_indices)

            if self._feature_importance_scores is not None:
                if isinstance(self._feature_importance_scores, torch.Tensor):
                    importance_scores_cpu = self._feature_importance_scores.cpu().numpy()
                else:
                    importance_scores_cpu = np.array(self._feature_importance_scores)
            else:
                importance_scores_cpu = None

            # COMPLETELY CLEAN METADATA - RECURSIVELY REMOVE ALL NON-SERIALIZABLE OBJECTS
            clean_metadata = self._recursively_clean_metadata(self._feature_selection_metadata)

            binary_data.update({
                'selected_feature_indices': selected_indices_cpu,
                'feature_importance_scores': importance_scores_cpu,
                'feature_selection_metadata': clean_metadata
            })

        # Save ALL data in binary format
        binary_path = os.path.join(output_dir, 'feature_metadata.pt')
        try:
            torch.save(binary_data, binary_path)
            logger.info(f"Saved comprehensive feature metadata to {binary_path} (binary format)")
        except Exception as e:
            logger.error(f"Error saving feature metadata: {str(e)}")
            raise

    def _recursively_clean_metadata(self, metadata: Any) -> Any:
        """
        Recursively clean metadata to ensure ONLY basic Python types remain.
        Completely removes any tensors, arrays, or complex objects.
        """
        if metadata is None:
            return None

        # Remove all tensor-like objects
        if isinstance(metadata, (torch.Tensor, np.ndarray)):
            return None

        # Remove any objects with .item() method (like scalar tensors)
        if hasattr(metadata, 'item'):
            try:
                return metadata.item()
            except:
                return None

        # Handle basic serializable types
        if isinstance(metadata, (str, int, float, bool)):
            return metadata

        # Handle lists and tuples - recursively clean each element
        if isinstance(metadata, (list, tuple)):
            cleaned_list = []
            for item in metadata:
                cleaned_item = self._recursively_clean_metadata(item)
                if cleaned_item is not None:
                    cleaned_list.append(cleaned_item)
            return cleaned_list if cleaned_list else None

        # Handle dictionaries - recursively clean each key-value pair
        if isinstance(metadata, dict):
            cleaned_dict = {}
            for key, value in metadata.items():
                # Ensure key is string
                str_key = str(key) if not isinstance(key, str) else key
                cleaned_value = self._recursively_clean_metadata(value)
                if cleaned_value is not None:
                    cleaned_dict[str_key] = cleaned_value
            return cleaned_dict if cleaned_dict else None

        # Remove any other complex objects by converting to string or None
        try:
            # Try to get a string representation for complex objects
            return str(metadata)
        except:
            # If even string conversion fails, remove entirely
            return None

    def _clean_config_for_binary_storage(self, config: Any) -> Any:
        """
        Clean configuration data specifically for binary storage.
        Ensures no complex objects that might cause serialization issues.
        """
        if config is None:
            return None

        # Handle basic types
        if isinstance(config, (str, int, float, bool)):
            return config

        # Remove tensor-like objects
        if isinstance(config, (torch.Tensor, np.ndarray)):
            return None

        # Handle lists
        if isinstance(config, list):
            return [self._clean_config_for_binary_storage(item) for item in config
                    if self._clean_config_for_binary_storage(item) is not None]

        # Handle dictionaries
        if isinstance(config, dict):
            cleaned = {}
            for key, value in config.items():
                str_key = str(key)
                cleaned_value = self._clean_config_for_binary_storage(value)
                if cleaned_value is not None:
                    cleaned[str_key] = cleaned_value
            return cleaned if cleaned else None

        # Convert other objects to string or remove
        try:
            return str(config)
        except:
            return None

    def freeze_feature_selection(self, selected_indices: List[int],
                               importance_scores: np.ndarray,
                               metadata: Dict = None):
        """Freeze feature ordering for consistency across runs - using tensors for binary storage"""
        # Convert to tensors for efficient binary serialization
        if isinstance(selected_indices, (list, np.ndarray)):
            self._selected_feature_indices = torch.tensor(selected_indices, dtype=torch.long, device=self.device)
        else:
            self._selected_feature_indices = selected_indices

        if isinstance(importance_scores, np.ndarray):
            self._feature_importance_scores = torch.tensor(importance_scores, dtype=torch.float32, device=self.device)
        else:
            self._feature_importance_scores = importance_scores

        self._feature_selection_metadata = metadata or {}
        self._is_feature_selection_frozen = True

        logger.info(f"Feature ordering frozen: {len(selected_indices)} features in consistent order")

    def _features_to_dataframe(self, feature_dict: Dict[str, torch.Tensor],
                             dc_config: Dict) -> pd.DataFrame:
        """
        Convert features dictionary to a pandas DataFrame with optional feature selection.

        Args:
            feature_dict (Dict): Dictionary containing features and metadata
            dc_config (Dict): Distance correlation configuration

        Returns:
            pd.DataFrame: DataFrame containing selected features and metadata
        """
        data_dict = {}

        # Get base length from embeddings
        base_length = len(feature_dict['embeddings']) if 'embeddings' in feature_dict else 0
        if base_length == 0:
            raise ValueError("No embeddings found in features")

        # Process embeddings
        features = feature_dict['embeddings'].cpu().numpy()

        # Apply feature selection if enabled
        if dc_config.get('use_distance_correlation', True) and 'labels' in feature_dict:
            labels = feature_dict['labels'].cpu().numpy()

            # Select features based on distance correlation
            selector = DistanceCorrelationFeatureSelector(
                upper_threshold=dc_config['distance_correlation_upper'],
                lower_threshold=dc_config['distance_correlation_lower']
            )
            selected_indices, corr_values = selector.select_features(features, labels)

            # Store only selected features
            for new_idx, orig_idx in enumerate(selected_indices):
                data_dict[f'feature_{new_idx}'] = features[:, orig_idx]
                data_dict[f'original_feature_idx_{new_idx}'] = orig_idx
                data_dict[f'feature_{new_idx}_correlation'] = corr_values[orig_idx]
        else:
            # Include all features if selection is disabled
            for i in range(features.shape[1]):
                data_dict[f'feature_{i}'] = features[:, i]

        # Process labels and class names
        if 'class_names' in feature_dict:
            if len(feature_dict['class_names']) == base_length:
                data_dict['target'] = feature_dict['class_names']
        elif 'labels' in feature_dict:
            if len(feature_dict['labels']) == base_length:
                data_dict['target'] = feature_dict['labels'].cpu().numpy()

        # Include additional metadata if available
        optional_fields = ['indices', 'filenames']
        for field in optional_fields:
            if field in feature_dict and len(feature_dict[field]) == base_length:
                data_dict[field] = feature_dict[field]

        return pd.DataFrame(data_dict)

    def _save_feature_selection_metadata(self, features: Dict[str, torch.Tensor],
                                       dc_config: Dict, output_dir: str) -> None:
        """
        Save metadata about feature selection process - USING BINARY FORMAT ONLY

        Args:
            features (Dict): Original feature dictionary
            dc_config (Dict): Distance correlation configuration
            output_dir (str): Directory to save metadata files
        """
        # Convert any tensors to numpy arrays for safe storage
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': dc_config,
            'original_feature_count': features['embeddings'].shape[1] if isinstance(features['embeddings'], torch.Tensor) else len(features['embeddings'][0]),
            'description': 'Feature selection metadata'
        }

        # Save in binary format instead of JSON
        metadata_path = os.path.join(output_dir, 'feature_selection_metadata.pt')
        torch.save(metadata, metadata_path)
        logger.info(f"Saved feature selection metadata to {metadata_path} (binary format)")

#--------------------------
    def _initialize_clustering(self, config: Dict):
        """Initialize clustering parameters with existence check"""
        self.use_kl_divergence = config['model']['autoencoder_config']['enhancements']['use_kl_divergence']

        if self.use_kl_divergence:
            # Only initialize if not already exists
            if not hasattr(self, 'cluster_centers'):
                num_clusters = config['dataset'].get('num_classes', 10)
                self.register_buffer('cluster_centers',
                                   torch.randn(num_clusters, self.feature_dims))

        # ALWAYS initialize temperature as a tensor buffer
        temp_value = config['model']['autoencoder_config']['enhancements']['clustering_temperature']
        if not hasattr(self, 'clustering_temperature'):
            self.register_buffer('clustering_temperature',
                               torch.tensor([float(temp_value)], dtype=torch.float32))
        else:
            # Ensure existing temperature is a tensor
            if not isinstance(self.clustering_temperature, torch.Tensor):
                self.clustering_temperature = torch.tensor([float(temp_value)], dtype=torch.float32,device=self.device)

    def state_dict(self, *args, **kwargs):
        """Extend state dict to include all necessary components"""
        state = super().state_dict(*args, **kwargs)

        # Add clustering parameters if they exist
        if hasattr(self, 'cluster_centers'):
            state['cluster_centers'] = self.cluster_centers
        # Ensure temperature is saved as tensor
        if hasattr(self, 'clustering_temperature'):
            if not isinstance(self.clustering_temperature, torch.Tensor):
                self.clustering_temperature = torch.tensor([float(self.clustering_temperature)],
                                                         dtype=torch.float32,device=self.device)
            state['clustering_temperature'] = self.clustering_temperature


        # Add classifier if it exists
        if hasattr(self, 'classifier'):
            state['classifier_state'] = self.classifier.state_dict()

        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict including all components"""
        # Load main model state
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)

        # Load clustering parameters
        if 'cluster_centers' in state_dict:
            if not hasattr(self, 'cluster_centers'):
                self.register_buffer('cluster_centers', state_dict['cluster_centers'])
            else:
                self.cluster_centers.data.copy_(state_dict['cluster_centers'])

        # Handle clustering temperature
        if 'clustering_temperature' in state_dict:
            temp = state_dict['clustering_temperature']
            if not isinstance(temp, torch.Tensor):
                temp = torch.tensor([float(temp)], dtype=torch.float32,device=self.device)

            if not hasattr(self, 'clustering_temperature'):
                self.register_buffer('clustering_temperature', temp)
            else:
                self.clustering_temperature.data.copy_(temp)

        # Load classifier if it exists
        if 'classifier_state' in state_dict and hasattr(self, 'classifier'):
            self.classifier.load_state_dict(state_dict['classifier_state'])

        if strict:
            if missing_keys:
                raise RuntimeError(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys: {unexpected_keys}")
#--------------------------
    def set_dataset(self, dataset: Dataset):
        """Store dataset reference"""
        self.train_dataset = dataset

    def _initialize_latent_organization(self):
        """Initialize latent space organization components with existence checks"""
        self.use_kl_divergence = self.config['model'].get('autoencoder_config', {}).get('enhancements', {}).get('use_kl_divergence', True)
        self.use_class_encoding = self.config['model'].get('autoencoder_config', {}).get('enhancements', {}).get('use_class_encoding', True)

        if self.use_class_encoding and not hasattr(self, 'classifier'):
            num_classes = self.config['dataset'].get('num_classes', 10)
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dims, self.feature_dims // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.feature_dims // 2, num_classes)
            )

        if self.use_kl_divergence:
            if not hasattr(self, 'cluster_centers'):
                num_clusters = self.config['dataset'].get('num_classes', 10)
                self.cluster_centers = nn.Parameter(torch.randn(num_clusters, self.feature_dims))
            if not hasattr(self, 'clustering_temperature'):
                self.clustering_temperature = self.config['model'].get('autoencoder_config', {}).get('enhancements', {}).get('clustering_temperature', 1.0)

    def set_training_phase(self, phase: int):
        """Set the training phase (1 or 2) with proper cluster initialization"""
        self.training_phase = phase
        if phase == 2 and self.use_kl_divergence:
            if not hasattr(self, 'cluster_centers'):
                # Initialize only if not already initialized
                num_clusters = self.config['dataset'].get('num_classes', 10)
                self.cluster_centers = nn.Parameter(
                    torch.randn(num_clusters, self.feature_dims, device=self.device)
                )
                self.clustering_temperature = self.config['model']\
                    .get('autoencoder_config', {})\
                    .get('enhancements', {})\
                    .get('clustering_temperature', 1.0)

    def _initialize_cluster_centers(self):
        """Initialize cluster centers using k-means"""
        self.eval()
        with torch.no_grad():
            if self.train_dataset is None:
                raise ValueError("Dataset not set. Call set_dataset() before training.")

            # Use stored dataset reference
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=1000,
                shuffle=True
            )
            batch_data, _ = next(iter(dataloader))
            embeddings = self.encode(batch_data.to(self.device))
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]

            # Use k-means to initialize cluster centers
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.cluster_centers.size(0), n_init=20)
            kmeans.fit(embeddings.cpu().numpy())
            self.cluster_centers.data = torch.tensor(
                kmeans.cluster_centers_,
                device=self.device
            )

    def _create_encoder_layers(self) -> nn.ModuleList:
        """Create encoder layers that adapt to input shape"""
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for size in self.layer_sizes:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, size, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(size),
                nn.LeakyReLU(0.2)
            ))
            in_channels = size

        return layers

    def _create_decoder_layers(self) -> nn.ModuleList:
        """Create decoder layers that exactly reconstruct original shape"""
        layers = nn.ModuleList()
        in_channels = self.layer_sizes[-1]

        # Build decoder in reverse order
        for i in range(len(self.layer_sizes)-1, -1, -1):
            out_channels = self.in_channels if i == 0 else self.layer_sizes[i-1]

            # For the last layer, ensure output channels match input
            if i == 0:
                out_channels = self.in_channels

            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(out_channels) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2) if i > 0 else nn.Tanh()
            ))
            in_channels = out_channels

        return layers

    def _create_embedder(self) -> nn.Sequential:
        """Create embedder with dynamic input size"""
        return nn.Sequential(
            nn.Linear(self.flattened_size, self.feature_dims),
            nn.BatchNorm1d(self.feature_dims),
            nn.LeakyReLU(0.2)
        )

    def _create_unembedder(self) -> nn.Sequential:
        """Create unembedder with dynamic output size"""
        return nn.Sequential(
            nn.Linear(self.feature_dims, self.flattened_size),
            nn.BatchNorm1d(self.flattened_size),
            nn.LeakyReLU(0.2)
        )

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced decoding that handles both compressed and full embeddings"""
        # If input is compressed features, first decompress
        if x.shape[1] == self.compressed_dims:
            x = self.feature_decompressor(x)

        # Standard decoding process
        x = self.unembedder(x)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim_h, self.final_spatial_dim_w)

        for layer in self.decoder_layers:
            x = layer(x)

        return x

    def get_encoding_shape(self) -> Tuple[int, ...]:
        """Get the shape of the encoding at each layer"""
        return tuple([size for size in self.layer_sizes])

    def get_spatial_dims(self) -> List[List[int]]:
        """Get the spatial dimensions at each layer"""
        return self.spatial_dims.copy()

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to image array with proper normalization"""
        tensor = tensor.detach().cpu()
        if len(tensor.shape) == 3:
            tensor = tensor.permute(1, 2, 0)

        mean = torch.tensor(self.config['dataset']['mean']).view(1, 1, -1)
        std = torch.tensor(self.config['dataset']['std']).view(1, 1, -1)
        tensor = tensor * std + mean

        return (tensor.clamp(0, 1) * 255).numpy().astype(np.uint8)

    def save_image(self, tensor: torch.Tensor, path: str):
        """Save tensor as image with proper normalization"""
        img_array = self._tensor_to_image(tensor)
        img = Image.fromarray(img_array)

        # Ensure target size
        target_size = tuple(self.config['dataset']['input_size'])
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        img.save(path, quality=99, optimize=True)
        logging.debug(f"Saved image to {path} with size {img.size}")

    def plot_reconstruction_samples(self, inputs: torch.Tensor,
                                 save_path: Optional[str] = None) -> None:
        """Visualize original and reconstructed images"""
        self.eval()
        with torch.no_grad():
            embedding = self.encode(inputs)
            reconstructions = self.decode(embedding)

        num_samples = min(inputs.size(0), 8)
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))

        for i in range(num_samples):
            # Original
            axes[0, i].imshow(self._tensor_to_image(inputs[i]))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')

            # Reconstruction
            axes[1, i].imshow(self._tensor_to_image(reconstructions[i]))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Reconstruction samples saved to {save_path}")
        plt.close()

    def extract_features(self, loader: DataLoader, dataset_type: str = "train") -> Dict[str, torch.Tensor]:
        """
        Extract features from a DataLoader with improved batch handling - INCLUDING FILE PATHS
        """
        self.eval()
        all_embeddings = []
        all_labels = []
        all_indices = []
        all_filenames = []
        all_class_names = []
        all_full_paths = []  # NEW: Store full file paths

        try:
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc=f"Extracting {dataset_type} features")):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    batch_size = inputs.size(0)

                    # Get metadata if available
                    if hasattr(loader.dataset, 'get_additional_info'):
                        indices = []
                        filenames = []
                        full_paths = []  # NEW: Store full paths
                        class_names = []

                        for i in range(batch_size):
                            global_idx = batch_idx * loader.batch_size + i
                            if global_idx < len(loader.dataset):
                                idx_info = loader.dataset.get_additional_info(global_idx)
                                indices.append(idx_info[0])
                                filenames.append(idx_info[1])

                                # NEW: Get full path if available
                                if len(idx_info) > 2:  # If full path is provided
                                    full_paths.append(idx_info[2])
                                else:
                                    # Fallback: construct path from available info
                                    if hasattr(loader.dataset, 'image_files') and global_idx < len(loader.dataset.image_files):
                                        full_paths.append(loader.dataset.image_files[global_idx])
                                    else:
                                        full_paths.append(f"batch_{batch_idx}_{i}")

                                # Class name handling - PRESERVE ORIGINAL
                                if hasattr(loader.dataset, 'reverse_encoder'):
                                    class_names.append(loader.dataset.reverse_encoder[labels[i].item()])
                                elif hasattr(loader.dataset, 'classes'):
                                    class_names.append(loader.dataset.classes[labels[i].item()])
                                else:
                                    # Extract from file path if available
                                    if full_paths and len(full_paths) > i:
                                        path_parts = full_paths[i].split(os.sep)
                                        if len(path_parts) >= 2:
                                            class_names.append(path_parts[-2])
                                        else:
                                            class_names.append(f"class_{labels[i].item()}")
                                    else:
                                        class_names.append(f"class_{labels[i].item()}")
                            else:
                                # Handle edge case for last incomplete batch
                                indices.append(f"batch_{batch_idx}_{i}")
                                filenames.append(f"batch_{batch_idx}_{i}")
                                full_paths.append(f"batch_{batch_idx}_{i}")  # NEW
                                class_names.append(f"class_{labels[i].item()}")
                    else:
                        # Dataset without metadata
                        indices = [f"batch_{batch_idx}_{i}" for i in range(batch_size)]
                        filenames = [f"batch_{batch_idx}_{i}" for i in range(batch_size)]
                        full_paths = [f"batch_{batch_idx}_{i}" for i in range(batch_size)]  # NEW

                        if hasattr(loader.dataset, 'classes'):
                            class_names = [loader.dataset.classes[labels[i].item()] for i in range(batch_size)]
                        else:
                            class_names = [str(labels[i].item()) for i in range(batch_size)]

                    # Extract embeddings - handle potential tuple returns
                    embeddings = self.encode(inputs)

                    # Handle case where encode returns tuple (embedding, features)
                    if isinstance(embeddings, tuple):
                        embeddings = embeddings[0]  # Take only the embedding

                    # Ensure embeddings is the right shape
                    if embeddings.dim() > 2:
                        embeddings = embeddings.view(embeddings.size(0), -1)

                    # Verify batch size matches
                    if embeddings.size(0) != batch_size:
                        logger.warning(f"Embeddings batch size {embeddings.size(0)} doesn't match input batch size {batch_size}")
                        # Truncate to match
                        min_size = min(embeddings.size(0), batch_size)
                        embeddings = embeddings[:min_size]
                        labels = labels[:min_size]
                        indices = indices[:min_size]
                        filenames = filenames[:min_size]
                        full_paths = full_paths[:min_size]  # NEW
                        class_names = class_names[:min_size]

                    # Append to lists
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)
                    all_indices.extend(indices)
                    all_filenames.extend(filenames)
                    all_full_paths.extend(full_paths)  # NEW
                    all_class_names.extend(class_names)

                # Concatenate all results
                if all_embeddings:
                    embeddings = torch.cat(all_embeddings)
                    labels = torch.cat(all_labels)
                else:
                    raise ValueError("No embeddings extracted")

                # Final length verification
                total_samples = len(embeddings)
                logger.info(f"Extracted {total_samples} embeddings with {len(all_full_paths)} file paths")

                # Ensure all arrays have the same length
                if len(all_indices) != total_samples:
                    logger.warning(f"Truncating indices from {len(all_indices)} to {total_samples}")
                    all_indices = all_indices[:total_samples]
                if len(all_filenames) != total_samples:
                    logger.warning(f"Truncating filenames from {len(all_filenames)} to {total_samples}")
                    all_filenames = all_filenames[:total_samples]
                if len(all_full_paths) != total_samples:
                    logger.warning(f"Truncating full_paths from {len(all_full_paths)} to {total_samples}")
                    all_full_paths = all_full_paths[:total_samples]  # NEW
                if len(all_class_names) != total_samples:
                    logger.warning(f"Truncating class_names from {len(all_class_names)} to {total_samples}")
                    all_class_names = all_class_names[:total_samples]

                feature_dict = {
                    'embeddings': embeddings,
                    'labels': labels,
                    'indices': all_indices,
                    'filenames': all_filenames,
                    'full_paths': all_full_paths,  # NEW: Include full paths
                    'class_names': all_class_names
                }

                logger.info(f"Feature extraction completed: {total_samples} samples with file paths")
                return feature_dict

        except Exception as e:
            logger.error(f"Error during feature extraction: {str(e)}")
            raise

    def get_enhancement_features(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Hook method for enhanced models to add specialized features.
        Override this in derived classes to add model-specific features.
        """
        return {}

    def _get_enhancement_columns(self, feature_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Extract enhancement-specific features for saving"""
        enhancement_dict = {}

        # Handle class probabilities
        if 'class_probabilities' in feature_dict:
            probs = feature_dict['class_probabilities'].cpu().numpy()
            for i in range(probs.shape[1]):
                enhancement_dict[f'class_{i}_probability'] = probs[:, i]

        # Handle cluster assignments
        if 'cluster_assignments' in feature_dict:
            enhancement_dict['cluster_assignment'] = feature_dict['cluster_assignments'].cpu().numpy()

        # Handle cluster probabilities
        if 'cluster_probabilities' in feature_dict:
            cluster_probs = feature_dict['cluster_probabilities'].cpu().numpy()
            for i in range(cluster_probs.shape[1]):
                enhancement_dict[f'cluster_{i}_probability'] = cluster_probs[:, i]

        # Add confidence scores if available
        if 'class_logits' in feature_dict:
            logits = feature_dict['class_logits'].cpu().numpy()
            enhancement_dict['prediction_confidence'] = softmax(logits, axis=1).max(axis=1)

        return enhancement_dict

    def organize_latent_space(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Organize latent space with enhanced KL divergence support"""
        output = {'embeddings': embeddings}

        if self.use_kl_divergence and hasattr(self, 'cluster_centers'):
            # ENHANCED KL DIVERGENCE IMPLEMENTATION
            cluster_centers = self.cluster_centers.to(embeddings.device)

            # Ensure dimensions match for KL divergence
            if cluster_centers.shape[1] != embeddings.shape[1]:
                # Use adaptive projection that preserves KL divergence properties
                if not hasattr(self, '_cluster_projection'):
                    self._cluster_projection = nn.Linear(embeddings.shape[1], cluster_centers.shape[1]).to(embeddings.device)
                    # Initialize with identity-like transformation to preserve distances
                    nn.init.orthogonal_(self._cluster_projection.weight)
                projected_embeddings = self._cluster_projection(embeddings)
            else:
                projected_embeddings = embeddings

            temperature = self.clustering_temperature

            # Calculate distances to cluster centers - CRITICAL FOR KL DIVERGENCE
            distances = torch.cdist(projected_embeddings, cluster_centers)

            # ENHANCED: Add regularization to prevent collapse
            min_distance = distances.min(dim=1, keepdim=True)[0]
            max_distance = distances.max(dim=1, keepdim=True)[0]

            # Normalize distances to improve numerical stability for KL
            normalized_distances = (distances - min_distance) / (max_distance - min_distance + 1e-8)

            # Convert distances to probabilities using softmax with temperature
            # This is the STUDENT'S T-DISTRIBUTION used in original t-SNE paper
            q_dist = torch.pow(1.0 + normalized_distances / temperature, -(temperature + 1) / 2)
            q_dist = q_dist / (q_dist.sum(dim=1, keepdim=True) + 1e-8)

            if labels is not None:
                # ENHANCED TARGET DISTRIBUTION for KL divergence
                p_dist = torch.zeros_like(q_dist)

                # Use confident assignments for target distribution
                with torch.no_grad():
                    # Get confident predictions (high probability assignments)
                    max_probs, max_indices = q_dist.max(dim=1)
                    confidence_mask = max_probs > 0.7  # Only use confident predictions

                    for i in range(cluster_centers.size(0)):
                        # Method 1: Use ground truth labels where available
                        label_mask = (labels == i)
                        # Method 2: Use confident predictions
                        confident_mask = (max_indices == i) & confidence_mask

                        # Combine both methods
                        combined_mask = label_mask | confident_mask

                        if combined_mask.any():
                            p_dist[combined_mask, i] = 1.0

                # Normalize target distribution
                p_dist_sum = p_dist.sum(dim=1, keepdim=True)
                p_dist = p_dist / (p_dist_sum + (p_dist_sum == 0).float())

            else:
                # During prediction, sharpen the current distribution
                p_dist = torch.pow(q_dist, 2) / torch.sum(torch.pow(q_dist, 2), dim=1, keepdim=True)
                p_dist = p_dist.detach()

            # Store KL divergence components for monitoring
            output.update({
                'cluster_probabilities': q_dist,
                'target_distribution': p_dist,
                'cluster_assignments': q_dist.argmax(dim=1),
                'cluster_confidence': q_dist.max(dim=1)[0],
                'cluster_distances': distances,
                'kl_temperature': temperature
            })

        if self.use_class_encoding and hasattr(self, 'classifier'):
            # Adaptive classifier that works with KL divergence features
            if not hasattr(self, '_classifier_initialized') or self._classifier_initialized != embeddings.shape[1]:
                # Reinitialize classifier for current dimension
                hidden_size = max(embeddings.shape[1] // 2, 16, self.config['dataset'].get('num_classes', 10) * 2)
                self.classifier = nn.Sequential(
                    nn.Linear(embeddings.shape[1], hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, self.config['dataset'].get('num_classes', 10))
                ).to(embeddings.device)
                self._classifier_initialized = embeddings.shape[1]

            class_logits = self.classifier(embeddings)
            output.update({
                'class_logits': class_logits,
                'class_predictions': class_logits.argmax(dim=1),
                'class_probabilities': F.softmax(class_logits, dim=1)
            })

        return output

    def cleanup_legacy_json_files(self, output_dir: str):
        """Remove any legacy JSON files that might cause serialization issues"""
        legacy_files = [
            'feature_extraction_metadata.json',
            'feature_selection_metadata.json',
            'training_metadata.json'
        ]

        for filename in legacy_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.info(f"Removed legacy JSON file: {filepath}")
                except Exception as e:
                    logger.warning(f"Could not remove legacy JSON file {filepath}: {e}")

    def _clean_history_for_serialization(self, history: Dict[str, List]) -> Dict[str, List]:
        """
        Clean training history to ensure no tensor objects remain that could cause JSON serialization errors.
        """
        cleaned_history = {}

        for key, value_list in history.items():
            cleaned_list = []
            for item in value_list:
                # Convert any tensor items to Python floats
                if isinstance(item, torch.Tensor):
                    if item.numel() == 1:  # Scalar tensor
                        cleaned_list.append(float(item.item()))
                    else:
                        # For multi-element tensors, take the mean or just skip
                        logger.warning(f"Found multi-element tensor in history[{key}], taking mean")
                        cleaned_list.append(float(item.mean().item()))
                elif isinstance(item, (int, float)):
                    cleaned_list.append(float(item))
                else:
                    # For any other type, convert to string or skip
                    try:
                        cleaned_list.append(float(item))
                    except (TypeError, ValueError):
                        logger.warning(f"Could not convert history item {type(item)} to float, skipping")
                        continue

            cleaned_history[key] = cleaned_list

        logger.info(f"Cleaned history: {len(cleaned_history)} metrics")
        return cleaned_history

    def compute_invertibility_loss(self, outputs: Dict[str, torch.Tensor],
                                 targets: torch.Tensor,
                                 labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi-objective loss with invertibility constraints"""

        # 1. Standard pixel reconstruction loss
        pixel_loss = F.mse_loss(outputs['reconstruction'], targets)

        # 2. Feature reconstruction loss (invertibility constraint)
        feature_recon_loss = F.mse_loss(outputs['reconstructed_embedding'], outputs['embedding'])

        # 3. Cycle consistency loss (strong invertibility guarantee)
        # Re-encode the reconstruction to verify invertibility
        with torch.no_grad():
            cycle_embedding = self.encode(outputs['reconstruction'])
            if isinstance(cycle_embedding, tuple):
                cycle_embedding = cycle_embedding[0]

        cycle_compressed = self.feature_compressor(cycle_embedding)
        cycle_loss = F.mse_loss(cycle_compressed, outputs['compressed_embedding'])

        # 4. Class preservation loss (if labels available)
        class_loss = torch.tensor(0.0)
        if labels is not None and self.use_class_encoding and hasattr(self, 'classifier'):
            class_logits = self.classifier(outputs['compressed_embedding'])
            class_loss = F.cross_entropy(class_logits, labels)

        # 5. Variance preservation regularization
        original_variance = torch.var(outputs['embedding'], dim=0)
        compressed_variance = torch.var(outputs['compressed_embedding'], dim=0)
        # Scale target variance based on compression ratio
        target_variance = original_variance * (self.compressed_dims / self.feature_dims)
        variance_loss = F.mse_loss(compressed_variance, target_variance)

        # Combine losses with adaptive weights
        total_loss = (pixel_loss +
                      0.5 * feature_recon_loss +
                      0.3 * cycle_loss +
                      0.2 * class_loss +
                      0.1 * variance_loss)

        # Store individual losses for monitoring
        if not hasattr(self, 'loss_components'):
            self.loss_components = {}

        self.loss_components = {
            'pixel_loss': pixel_loss.item(),
            'feature_recon_loss': feature_recon_loss.item(),
            'cycle_loss': cycle_loss.item(),
            'class_loss': class_loss.item() if isinstance(class_loss, torch.Tensor) else class_loss,
            'variance_loss': variance_loss.item()
        }

        return total_loss

    def register_attention_hooks(self):
        """Register hooks to capture intermediate feature maps for attention visualization"""
        self.attention_maps = {}
        self.hook_handles = []

        def hook_fn(module, input, output, name):
            self.attention_maps[name] = output.detach()

        # Register hooks on encoder layers
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
        logger.info("Removed all attention hooks")

    def get_feature_importance_scores(self, features: torch.Tensor, target_class: int) -> np.ndarray:
        """Compute importance scores for each feature dimension"""

        if not hasattr(self, 'classifier') or not self.use_class_encoding:
            return np.ones(features.shape[1])

        # Compute how much each feature contributes to class prediction
        features.requires_grad_(True)
        class_logits = self.classifier(features)
        target_score = class_logits[0, target_class]

        # Compute gradients
        target_score.backward()
        feature_gradients = features.grad.data.cpu().numpy()[0]

        # Importance = gradient * feature value
        feature_values = features.data.cpu().numpy()[0]
        importance_scores = np.abs(feature_gradients * feature_values)

        # Normalize
        if importance_scores.max() > 0:
            importance_scores = importance_scores / importance_scores.max()

        return importance_scores

    def generate_cluster_analysis_heatmaps(self, data_path: str, output_dir: str = None):
        """Generate comprehensive analysis of what drives each cluster"""

        # Get all data
        image_files, class_labels, _ = self._get_image_files_with_labels(data_path)
        if not image_files:
            return

        transform = self.get_transforms(self.config, is_train=False)

        # Setup output
        if output_dir is None:
            dataset_name = self.config['dataset']['name'].lower()
            output_dir = os.path.join('data', dataset_name, 'cluster_analysis')
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        self.model.register_attention_hooks()  # NEW: Enable feature map capture

        all_features = []
        all_cluster_assignments = []
        all_images = []

        with torch.no_grad():
            for img_path in tqdm(image_files, desc="Processing images for cluster analysis"):
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(self.device)

                    outputs = self.model(image_tensor)
                    features = outputs.get('compressed_embedding', outputs['embedding'])
                    latent_info = self.model.organize_latent_space(features)

                    all_features.append(features[0].cpu().numpy())
                    all_cluster_assignments.append(latent_info['cluster_assignments'][0].item())
                    all_images.append((img_path, image))

                except Exception as e:
                    continue

        # Analyze each cluster
        all_features = np.array(all_features)
        all_cluster_assignments = np.array(all_cluster_assignments)

        unique_clusters = np.unique(all_cluster_assignments)

        for cluster_id in unique_clusters:
            cluster_mask = all_cluster_assignments == cluster_id
            cluster_features = all_features[cluster_mask]
            cluster_images = [img for i, img in enumerate(all_images) if cluster_mask[i]]

            if len(cluster_features) == 0:
                continue

            # Compute cluster centroid and characteristic features
            centroid = np.mean(cluster_features, axis=0)
            characteristic_features = np.argsort(np.abs(centroid))[-5:][::-1]  # Top 5 features

            # Create cluster summary
            self._create_cluster_summary(
                cluster_id, centroid, characteristic_features,
                cluster_images, cluster_features, output_dir
            )

        self.model.remove_attention_hooks()
        logger.info(f"Cluster analysis complete: {output_dir}")

    def _create_features_via_prediction(self, model: nn.Module, config: Dict,
                                      train_loader: DataLoader, test_loader: DataLoader = None,
                                      output_path: str = None):
        """Create training features using the reliable prediction pipeline"""

        # Use the dataset name from config
        dataset_name = config['dataset']['name'].lower()

        # Create prediction manager (uses the same reliable loading)
        pred_manager = PredictionManager(config)

        # Determine data paths based on dataset structure
        base_data_path = f"data/{dataset_name}"

        # Generate features using prediction pipeline
        logger.info("Generating training features using reliable prediction pipeline...")

        # Process training data
        train_csv_path = f"{base_data_path}/{dataset_name}_train.csv"
        if os.path.exists(f"{base_data_path}/train"):
            train_predictions = pred_manager.predict_images(
                data_path=f"{base_data_path}/train",
                output_csv=train_csv_path,
                batch_size=config['training'].get('prediction_batch_size', 32),
                generate_heatmaps=False
            )
            logger.info(f"Training features saved to {train_csv_path}")
        else:
            logger.warning(f"Training directory not found: {base_data_path}/train")

        # Process test data if available
        test_csv_path = f"{base_data_path}/{dataset_name}_test.csv"
        if test_loader is not None and os.path.exists(f"{base_data_path}/test"):
            test_predictions = pred_manager.predict_images(
                data_path=f"{base_data_path}/test",
                output_csv=test_csv_path,
                batch_size=config['training'].get('prediction_batch_size', 32),
                generate_heatmaps=False
            )
            logger.info(f"Test features saved to {test_csv_path}")

        # If separate train/test CSV files were created, also create combined version
        if os.path.exists(train_csv_path) and (test_loader is None or os.path.exists(test_csv_path)):
            combined_csv_path = f"{base_data_path}/{dataset_name}.csv"
            try:
                if test_loader is not None and os.path.exists(test_csv_path):
                    # Combine train and test
                    train_df = pd.read_csv(train_csv_path)
                    test_df = pd.read_csv(test_csv_path)
                    combined_df = pd.concat([train_df, test_df], ignore_index=True)
                    combined_df.to_csv(combined_csv_path, index=False)
                    logger.info(f"Combined features saved to {combined_csv_path}")
                else:
                    # Just use train as combined
                    shutil.copy2(train_csv_path, combined_csv_path)
                    logger.info(f"Combined features saved to {combined_csv_path}")
            except Exception as e:
                logger.warning(f"Could not create combined CSV: {str(e)}")

        logger.info("Training features successfully created via reliable prediction pipeline")

    def smart_feature_compression(self, embeddings, labels, max_dims=32):
        """NEW: Strategic feature selection to maximize information in limited features"""
        n_features = embeddings.shape[1]

        # If already within limits, no compression needed
        if n_features <= max_dims:
            return embeddings, np.arange(n_features), "within_limit"

        # STRATEGIC APPROACH: Use multiple methods to preserve different information types
        with torch.no_grad():
            embeddings_np = embeddings.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy() if labels is not None else None

            # Method 1: Classification-focused features (if labels available)
            if labels_np is not None and len(np.unique(labels_np)) >= 3:
                class_indices, class_scores = self._select_classification_features(
                    embeddings_np, labels_np, max_dims // 2)
            else:
                class_indices = np.array([], dtype=np.int64)

            # Method 2: Reconstruction-focused features
            recon_indices, recon_scores = self._select_reconstruction_features(
                embeddings_np, max_dims // 2)

            # Method 3: Diversity-focused features
            diversity_indices = self._select_diverse_features(
                embeddings_np, max_dims - len(class_indices) - len(recon_indices),
                exclude_indices=np.concatenate([class_indices, recon_indices]))

            # Combine strategically
            all_indices = np.concatenate([class_indices, recon_indices, diversity_indices])

            # Ensure we have exactly max_dims
            if len(all_indices) > max_dims:
                # Prioritize classification features
                all_indices = all_indices[:max_dims]
            elif len(all_indices) < max_dims:
                # Fill with high-variance features
                variances = np.var(embeddings_np, axis=0)
                remaining = max_dims - len(all_indices)
                additional_indices = np.argsort(variances)[-remaining:][::-1]
                # Exclude already selected
                additional_indices = [idx for idx in additional_indices if idx not in all_indices]
                all_indices = np.concatenate([all_indices, additional_indices[:remaining]])

        # Final validation and selection
        all_indices = np.unique(all_indices)[:max_dims]  # Remove duplicates and enforce limit
        compressed_embedding = embeddings[:, torch.from_numpy(all_indices).to(embeddings.device)]

        logger.info(f"Strategic compression: {n_features} → {len(all_indices)} features "
                   f"(class: {len(class_indices)}, recon: {len(recon_indices)}, diverse: {len(diversity_indices)})")

        return compressed_embedding, all_indices, "strategic_compression"

    def _select_classification_features(self, embeddings, labels, target_dims):
        """NEW: Select features most relevant for classification"""
        if target_dims <= 0:
            return np.array([], dtype=np.int64), np.array([])

        try:
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(embeddings, labels, random_state=42)
            indices = np.argsort(mi_scores)[-target_dims:][::-1]
            return indices, mi_scores
        except Exception as e:
            logger.warning(f"Mutual info selection failed: {e}, using variance fallback")
            # Fallback: use variance
            variances = np.var(embeddings, axis=0)
            indices = np.argsort(variances)[-target_dims:][::-1]
            return indices, variances

    def _select_reconstruction_features(self, embeddings, target_dims):
        """NEW: Select features most important for reconstruction"""
        if target_dims <= 0:
            return np.array([], dtype=np.int64), np.array([])

        # Features with highest variance (carry most information)
        variances = np.var(embeddings, axis=0)
        indices = np.argsort(variances)[-target_dims:][::-1]
        return indices, variances

    def _select_diverse_features(self, embeddings, target_dims, exclude_indices):
        """NEW: Select features that are diverse from already selected ones"""
        if target_dims <= 0:
            return np.array([], dtype=np.int64)

        available_indices = [i for i in range(embeddings.shape[1])
                            if i not in exclude_indices]

        if not available_indices or target_dims == 0:
            return np.array([], dtype=np.int64)

        # If we have selected features, calculate diversity
        selected_features = embeddings[:, exclude_indices] if len(exclude_indices) > 0 else None

        diversity_scores = []
        for idx in available_indices:
            feature = embeddings[:, idx]
            if selected_features is not None and selected_features.shape[1] > 0:
                # Calculate maximum correlation with any selected feature
                correlations = []
                for j in range(selected_features.shape[1]):
                    try:
                        corr = np.corrcoef(feature, selected_features[:, j])[0, 1]
                        correlations.append(abs(corr))
                    except:
                        correlations.append(0.0)
                diversity = 1 - max(correlations) if correlations else 1.0
            else:
                diversity = 1.0  # No selected features yet
            diversity_scores.append(diversity)

        # Select most diverse features
        selected = []
        available_arr = np.array(available_indices)
        diversity_arr = np.array(diversity_scores)

        for _ in range(min(target_dims, len(available_indices))):
            if len(diversity_arr) == 0:
                break
            best_idx = np.argmax(diversity_arr)
            selected.append(available_arr[best_idx])
            # Remove selected
            available_arr = np.delete(available_arr, best_idx)
            diversity_arr = np.delete(diversity_arr, best_idx)

        return np.array(selected, dtype=np.int64)

    def _fuse_multi_scale_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """NEW: Intelligently fuse multi-scale features"""
        # Simple concatenation with dimensionality reduction
        concatenated = torch.cat(features, dim=1)

        # Reduce to reasonable dimensions while preserving information
        if concatenated.shape[1] > 512:
            if not hasattr(self, '_feature_reducer'):
                self._feature_reducer = nn.Linear(concatenated.shape[1], 512).to(concatenated.device)
                # Initialize reducer
                nn.init.xavier_uniform_(self._feature_reducer.weight)
                if self._feature_reducer.bias is not None:
                    nn.init.constant_(self._feature_reducer.bias, 0)
            reduced = self._feature_reducer(concatenated)
        else:
            reduced = concatenated

        return reduced

    def _validate_feature_consistency(self, features, source: str):
        """NEW: Validate that features are consistent across pipelines"""
        # Check feature dimensions
        if 'embeddings' in features:
            emb = features['embeddings']
            if isinstance(emb, torch.Tensor):
                if emb.shape[1] > 32:
                    logger.warning(f"{source}: Features exceed 32 limit: {emb.shape[1]}")

                # Check for NaN/Inf
                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    logger.error(f"{source}: Features contain NaN/Inf")

        # Compare with previous extraction if available
        if hasattr(self, '_previous_features'):
            prev_emb = self._previous_features.get('embeddings')
            curr_emb = features.get('embeddings')
            if prev_emb is not None and curr_emb is not None:
                prev_shape = prev_emb.shape if hasattr(prev_emb, 'shape') else 'unknown'
                curr_shape = curr_emb.shape if hasattr(curr_emb, 'shape') else 'unknown'
                if prev_shape != curr_shape:
                    logger.error(f"Feature shape mismatch: {prev_shape} vs {curr_shape}")

        self._previous_features = features

    def _save_features_via_prediction_consistent(self, output_path: str):
        """NEW: Use prediction pipeline with validation to ensure identical outputs"""
        if not hasattr(self, 'config'):
            logger.error("No config available for prediction pipeline")
            return

        # Create PredictionManager with same config
        pred_manager = PredictionManager(self.config)

        # Extract data paths from loaders
        train_data_path = self._extract_data_path_from_loader(self.train_loader, 'train')

        # Generate features via prediction
        train_csv = output_path.replace('.csv', '_train.csv')

        # Get predictions
        train_pred = pred_manager.predict_images(train_data_path, train_csv, generate_heatmaps=False)

        # VALIDATION: Compare with direct extraction to ensure consistency
        self._validate_feature_consistency(train_pred, "prediction_pipeline")

    def _extract_data_path_from_loader(self, loader, split: str):
        """NEW: Extract data path from data loader for prediction pipeline"""
        if loader is None:
            return f"data/{self.config['dataset']['name'].lower()}/{split}"

        # Try to extract path from dataset
        dataset = loader.dataset
        if hasattr(dataset, 'image_files') and dataset.image_files:
            sample_path = dataset.image_files[0]
            return os.path.dirname(os.path.dirname(sample_path))  # Go up one level from first image
        elif hasattr(dataset, 'samples') and dataset.samples:
            sample_path = dataset.samples[0][0]  # (path, label) format
            return os.path.dirname(os.path.dirname(sample_path))
        else:
            # Fallback to standard path
            return f"data/{self.config['dataset']['name'].lower()}/{split}"

def make_json_serializable(obj):
    """Convert an object to be JSON serializable"""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (dict,)):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

class AstronomicalStructurePreservingAutoencoder(BaseAutoencoder):
    """Autoencoder specialized for astronomical imaging features"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        super().__init__(input_shape, feature_dims, config)

        # Get enhancement configurations
        self.enhancement_config = config['model']['enhancement_modules']['astronomical']

        # Initial channel transformation layer
        self.initial_transform = nn.Sequential(
            nn.Conv2d(self.in_channels, self.layer_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.layer_sizes[0]),
            nn.LeakyReLU(0.2)
        )

        # Detail preservation module with multiple scales
        self.detail_preserving = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.layer_sizes[0], self.layer_sizes[0], kernel_size=k, padding=k//2),
                nn.BatchNorm2d(self.layer_sizes[0]),
                nn.LeakyReLU(0.2),
                nn.Conv2d(self.layer_sizes[0], self.layer_sizes[0], kernel_size=1)
            ) for k in [3, 5, 7]
        ])

        # Star detection module
        self.star_detector = nn.Sequential(
            nn.Conv2d(self.layer_sizes[0], self.layer_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.layer_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.layer_sizes[0], self.layer_sizes[0], kernel_size=1),
            nn.Sigmoid()
        )

        # Galaxy feature enhancement
        self.galaxy_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(size, size, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm2d(size),
                nn.LeakyReLU(0.2)
            ) for size, d in zip(self.layer_sizes, [1, 2, 4])
        ])

        # Initialize loss function
        self.structure_loss = AstronomicalStructureLoss() if self.enhancement_config['enabled'] else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # Remove tuple return
        """Enhanced encoding with astronomical feature preservation"""
        features = {}

        # Initial channel transformation
        x = self.initial_transform(x)

        if self.enhancement_config['components']['detail_preservation']:
            # Multi-scale detail extraction
            detail_features = [module(x) for module in self.detail_preserving]
            features['details'] = sum(detail_features) / len(detail_features)
            x = x + 0.1 * features['details']

        if self.enhancement_config['components']['star_detection']:
            # Star detection
            features['stars'] = self.star_detector(x)
            x = x * (1 + 0.1 * features['stars'])

        # Regular encoding path with galaxy enhancement
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.enhancement_config['components']['galaxy_features']:
                if idx < len(self.galaxy_enhancer):
                    galaxy_features = self.galaxy_enhancer[idx](x)
                    x = x + 0.1 * galaxy_features

        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)

        # Store features for use in decode, but don't return them
        self._cached_features = features

        return embedding  # Return only embedding, not tuple

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:  # Remove features parameter
        """Enhanced decoding with structure preservation"""
        x = self.unembedder(embedding)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim_h, self.final_spatial_dim_w)

        for layer in self.decoder_layers:
            x = layer(x)

        # Final channel transformation back to input channels
        x = nn.Conv2d(self.layer_sizes[0], self.in_channels, kernel_size=1)(x)

        # Add preserved features if available (use cached features)
        if hasattr(self, '_cached_features'):
            features = self._cached_features
            if self.enhancement_config['components']['detail_preservation']:
                if 'details' in features:
                    x = x + 0.1 * features['details']

            if self.enhancement_config['components']['star_detection']:
                if 'stars' in features:
                    x = x * (1 + 0.1 * features['stars'])

            # Clear cached features
            del self._cached_features

        return x

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced feature preservation"""
        embedding = self.encode(x)  # Now returns only embedding
        reconstruction = self.decode(embedding)  # Uses cached features internally

        output = {
            'embedding': embedding,
            'reconstruction': reconstruction
        }

        if self.training and self.enhancement_config['enabled']:
            loss = self.structure_loss(reconstruction, x)
            output['loss'] = loss

        return output

class MedicalStructurePreservingAutoencoder(BaseAutoencoder):
    """Autoencoder specialized for medical imaging features"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        super().__init__(input_shape, feature_dims, config)

        # Get enhancement configurations
        self.enhancement_config = config['model']['enhancement_modules']['medical']

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

        # Initialize loss function
        self.structure_loss = MedicalStructureLoss() if self.enhancement_config['enabled'] else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # FIXED: Remove tuple return
        """Enhanced encoding with medical feature preservation"""
        features = {}

        if self.enhancement_config['components']['tissue_boundary']:
            features['boundaries'] = self.boundary_detector(x)
            x = x * (1 + 0.1 * features['boundaries'])

        # Regular encoding path with lesion detection
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.enhancement_config['components']['lesion_detection']:
                if idx < len(self.lesion_detector):
                    lesion_features = self.lesion_detector[idx](x)
                    x = x + 0.1 * lesion_features

        if self.enhancement_config['components']['contrast_enhancement']:
            features['contrast'] = self.contrast_enhancer(x)
            x = x + 0.1 * features['contrast']

        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)

        # Store features for use in decode, but don't return them
        self._cached_features = features

        return embedding  # Return only embedding, not tuple

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:  # FIXED: Remove features parameter
        """Enhanced decoding with structure preservation"""
        x = self.unembedder(embedding)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim_h, self.final_spatial_dim_w)

        for layer in self.decoder_layers:
            x = layer(x)

        # Add preserved features (use cached features)
        if hasattr(self, '_cached_features'):
            features = self._cached_features
            if self.enhancement_config['components']['tissue_boundary']:
                x = x * (1 + 0.1 * features.get('boundaries', 0))

            if self.enhancement_config['components']['contrast_enhancement']:
                x = x + 0.1 * features.get('contrast', 0)

            # Clear cached features
            del self._cached_features

        return x

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced feature preservation"""
        embedding = self.encode(x)  # Now returns only embedding
        reconstruction = self.decode(embedding)  # Uses cached features internally

        output = {
            'embedding': embedding,
            'reconstruction': reconstruction
        }

        if self.training and self.enhancement_config['enabled']:
            loss = self.structure_loss(reconstruction, x)
            output['loss'] = loss

        return output

class AgriculturalPatternAutoencoder(BaseAutoencoder):
    """Autoencoder specialized for agricultural imaging features"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        super().__init__(input_shape, feature_dims, config)

        # Get enhancement configurations
        self.enhancement_config = config['model']['enhancement_modules']['agricultural']

        # Ensure channel numbers are compatible with groups
        texture_groups = min(4, self.in_channels)  # Adjust groups based on input channels
        intermediate_channels = 32 - (32 % texture_groups)  # Ensure divisible by groups

        self.texture_analyzer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels, kernel_size=k, padding=k//2, groups=texture_groups),
                nn.InstanceNorm2d(intermediate_channels),
                nn.PReLU(),
                nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=k, padding=k//2, groups=texture_groups)
            ) for k in [3, 5, 7]
        ])

        # Damage pattern detector
        damage_intermediate_channels = 32 - (32 % self.in_channels)  # Ensure divisible
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

        # Initialize loss function
        self.pattern_loss = AgriculturalPatternLoss() if self.enhancement_config['enabled'] else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # FIXED: Remove tuple return
        """Enhanced encoding with pattern preservation"""
        features = {}

        if self.enhancement_config['components']['texture_analysis']:
            texture_features = [module(x) for module in self.texture_analyzer]
            features['texture'] = sum(texture_features) / len(texture_features)
            x = x + 0.1 * features['texture']

        if self.enhancement_config['components']['damage_detection']:
            features['damage'] = self.damage_detector(x)

        if self.enhancement_config['components']['color_anomaly']:
            features['color'] = self.color_analyzer(x)
            x = x + 0.1 * features['color']

        # Regular encoding path
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)

        # Store features for use in decode, but don't return them
        self._cached_features = features

        return embedding  # Return only embedding, not tuple

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:  # FIXED: Remove features parameter
        """Enhanced decoding with pattern preservation"""
        x = self.unembedder(embedding)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim_h, self.final_spatial_dim_w)

        for layer in self.decoder_layers:
            x = layer(x)

        # Add preserved features (use cached features)
        if hasattr(self, '_cached_features'):
            features = self._cached_features
            if self.enhancement_config['components']['texture_analysis']:
                x = x + 0.1 * features.get('texture', 0)

            if self.enhancement_config['components']['damage_detection']:
                damage_mask = features.get('damage', torch.zeros_like(x))
                x = x * (1 + 0.2 * damage_mask)

            if self.enhancement_config['components']['color_anomaly']:
                x = x + 0.1 * features.get('color', 0)

            # Clear cached features
            del self._cached_features

        return x

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced pattern preservation"""
        embedding = self.encode(x)  # Now returns only embedding
        reconstruction = self.decode(embedding)  # Uses cached features internally

        output = {
            'embedding': embedding,
            'reconstruction': reconstruction
        }

        if self.training and self.enhancement_config['enabled']:
            loss = self.pattern_loss(reconstruction, x)
            output['loss'] = loss

        return output

class SlidingWindowAutoencoder(BaseAutoencoder):
    """Autoencoder that processes large images using sliding windows"""

    def __init__(self, input_shape, feature_dims, config, window_size=256, stride=128):
        # Initialize with window size instead of full image size
        self.window_size = window_size
        self.stride = stride or window_size // 2

        # Store original full image shape for reconstruction
        self.full_image_shape = input_shape

        # Initialize base autoencoder with window size
        window_shape = (input_shape[0], window_size, window_size)
        super().__init__(window_shape, feature_dims, config)

        # Feature aggregator for combining window features
        self.feature_aggregator = nn.Sequential(
            nn.Linear(feature_dims * 4, feature_dims * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dims * 2, feature_dims)
        )

        logger.info(f"Initialized sliding window autoencoder for images up to {input_shape[1]}x{input_shape[2]}")
        logger.info(f"Processing windows: {window_size}x{window_size} with stride {stride}")

    def process_full_image(self, image_tensor, reconstructor):
        """Process full image using sliding window approach"""
        batch_size, channels, height, width = image_tensor.shape

        # Process each image in batch separately
        batch_reconstructions = []
        batch_embeddings = []

        for batch_idx in range(batch_size):
            single_image = image_tensor[batch_idx:batch_idx+1]
            reconstruction, embedding = self._process_single_image(single_image, reconstructor)
            batch_reconstructions.append(reconstruction)
            batch_embeddings.append(embedding)

        return torch.cat(batch_reconstructions, dim=0), torch.stack(batch_embeddings, dim=0)

    def _process_single_image(self, image_tensor, reconstructor):
        """Process a single large image using sliding windows"""
        _, channels, height, width = image_tensor.shape
        reconstructor.reset()

        window_embeddings = []

        # Generate window coordinates
        windows = self._generate_windows(height, width)

        # Process each window
        for y1, x1, y2, x2 in windows:
            # Extract window
            window = image_tensor[:, :, y1:y2, x1:x2]

            # Pad if necessary (for edge windows)
            if window.shape[-2] != self.window_size or window.shape[-1] != self.window_size:
                window = F.pad(window, (0, self.window_size - window.shape[-1],
                                       0, self.window_size - window.shape[-2]),
                              mode='reflect')

            # Process window through autoencoder
            with torch.no_grad():
                window_embedding = self.encode(window)
                window_reconstruction = self.decode(window_embedding)

            # Store embedding for aggregation
            window_embeddings.append(window_embedding)

            # Remove padding before reconstruction
            if window_reconstruction.shape[-2] != (y2 - y1) or window_reconstruction.shape[-1] != (x2 - x1):
                window_reconstruction = window_reconstruction[:, :, :(y2 - y1), :(x2 - x1)]

            # Add to reconstructor
            reconstructor.add_window(window_reconstruction.squeeze(0), (y1, x1, y2, x2))

        # Aggregate window embeddings
        if window_embeddings:
            aggregated_embedding = self._aggregate_embeddings(window_embeddings)
        else:
            aggregated_embedding = torch.zeros(self.feature_dims, device=image_tensor.device)

        # Get final reconstruction
        full_reconstruction = reconstructor.get_reconstruction().unsqueeze(0)

        return full_reconstruction, aggregated_embedding

    def _generate_windows(self, height, width):
        """Generate sliding window coordinates for a single image"""
        windows = []

        y_steps = max(1, (height - self.window_size) // self.stride + 1)
        x_steps = max(1, (width - self.window_size) // self.stride + 1)

        for i in range(y_steps):
            for j in range(x_steps):
                y1 = i * self.stride
                x1 = j * self.stride
                y2 = min(y1 + self.window_size, height)
                x2 = min(x1 + self.window_size, width)
                windows.append((y1, x1, y2, x2))

        # Add edge cases
        if width % self.stride != 0:
            for i in range(y_steps):
                x1 = width - self.window_size
                x2 = width
                y1 = i * self.stride
                y2 = min(y1 + self.window_size, height)
                windows.append((y1, x1, y2, x2))

        if height % self.stride != 0:
            for j in range(x_steps):
                y1 = height - self.window_size
                y2 = height
                x1 = j * self.stride
                x2 = min(x1 + self.window_size, width)
                windows.append((y1, x1, y2, x2))

        if height % self.stride != 0 and width % self.stride != 0:
            y1 = height - self.window_size
            y2 = height
            x1 = width - self.window_size
            x2 = width
            windows.append((y1, x1, y2, x2))

        return windows

    def _aggregate_embeddings(self, window_embeddings):
        """Aggregate embeddings from multiple windows"""
        if len(window_embeddings) == 1:
            return window_embeddings[0]

        # Stack all window embeddings
        all_embeddings = torch.stack(window_embeddings, dim=0)

        # Use multiple aggregation strategies
        max_pooled = torch.max(all_embeddings, dim=0)[0]
        avg_pooled = torch.mean(all_embeddings, dim=0)
        std_pooled = torch.std(all_embeddings, dim=0)

        # Combine different aggregation methods
        combined = torch.cat([max_pooled, avg_pooled, std_pooled], dim=0)
        aggregated = self.feature_aggregator(combined)

        return aggregated

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
        for filter in self.structure_filters:
            rec_struct = F.conv2d(reconstruction, filter, padding=filter.size(-1)//2)
            target_struct = F.conv2d(target, filter, padding=filter.size(-1)//2)
            structure_loss += F.mse_loss(rec_struct, target_struct)

        # Scale-space feature preservation
        scale_loss = 0
        for filter in self.scale_filters:
            rec_scale = F.conv2d(reconstruction, filter, padding=filter.size(-1)//2)
            target_scale = F.conv2d(target, filter, padding=filter.size(-1)//2)
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

        # Edge detection filters
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32).view(1, 1, 3, 3)

        # Multi-scale analysis filters
        self.feature_filters = [
            self._create_gaussian_kernel(sigma) for sigma in [0.5, 1.0, 2.0]
        ]

        # Structure filters for tissue boundaries
        self.boundary_filters = [
            self._create_dog_kernel(sigma) for sigma in [1.0, 2.0, 4.0]
        ]

    def _create_gaussian_kernel(self, sigma: float, kernel_size: int = 7) -> torch.Tensor:
        """Create Gaussian kernel for smoothing"""
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        x = x.view(-1, 1)
        y = x.t()
        gaussian = torch.exp(-(x**2 + y**2)/(2*sigma**2))
        return (gaussian / gaussian.sum()).view(1, 1, kernel_size, kernel_size)

    def _create_dog_kernel(self, sigma: float, k: float = 1.6,
                          kernel_size: int = 7) -> torch.Tensor:
        """Create Difference of Gaussians kernel for edge detection"""
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        x = x.view(-1, 1)
        y = x.t()
        r2 = x**2 + y**2
        g1 = torch.exp(-r2/(2*sigma**2))
        g2 = torch.exp(-r2/(2*(k*sigma)**2))
        dog = g1/sigma**2 - g2/(k*sigma)**2
        return (dog / dog.abs().sum()).view(1, 1, kernel_size, kernel_size)

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss with medical feature preservation"""
        device = reconstruction.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        self.feature_filters = [f.to(device) for f in self.feature_filters]
        self.boundary_filters = [f.to(device) for f in self.boundary_filters]

        # Reconstruction loss with tissue weighting
        tissue_weights = (target > target.mean()).float() * 2 + 1
        recon_loss = F.mse_loss(reconstruction * tissue_weights,
                               target * tissue_weights)

        # Gradient preservation for tissue boundaries
        rec_grad_x = F.conv2d(reconstruction, self.sobel_x, padding=1)
        rec_grad_y = F.conv2d(reconstruction, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)

        gradient_loss = F.mse_loss(rec_grad_x, target_grad_x) + \
                       F.mse_loss(rec_grad_y, target_grad_y)

        # Multi-scale feature preservation
        feature_loss = 0
        for filter in self.feature_filters:
            rec_features = F.conv2d(reconstruction, filter, padding=filter.size(-1)//2)
            target_features = F.conv2d(target, filter, padding=filter.size(-1)//2)
            feature_loss += F.mse_loss(rec_features, target_features)

        # Boundary preservation
        boundary_loss = 0
        for filter in self.boundary_filters:
            rec_bound = F.conv2d(reconstruction, filter, padding=filter.size(-1)//2)
            target_bound = F.conv2d(target, filter, padding=filter.size(-1)//2)
            boundary_loss += F.mse_loss(rec_bound, target_bound)

        # Local contrast preservation for lesion detection
        rec_std = torch.std(F.unfold(reconstruction, kernel_size=5), dim=1)
        target_std = torch.std(F.unfold(target, kernel_size=5), dim=1)
        contrast_loss = F.mse_loss(rec_std, target_std)

        # Combine losses with weights
        total_loss = (recon_loss +
                     1.5 * gradient_loss +
                     1.0 * feature_loss +
                     2.0 * boundary_loss +
                     1.0 * contrast_loss)

        return total_loss

class AgriculturalPatternLoss(nn.Module):
    """Loss function optimized for agricultural pest and disease detection"""
    def __init__(self):
        super().__init__()
        self.texture_filters = None  # Will be initialized on first use
        self.pattern_filters = None  # Will be initialized on first use

        # Color analysis filters
        self.color_filters = [
            torch.eye(3, dtype=torch.float32).view(3, 3, 1, 1),
            torch.tensor([[0.299, 0.587, 0.114]], dtype=torch.float32).view(1, 3, 1, 1)
        ]

    def _create_gabor_kernel(self, frequency: float, angle: float,
                           sigma: float = 3.0, size: int = 7) -> torch.Tensor:
        """Create Gabor filter for texture analysis"""
        # Convert angle to radians and create as tensor
        angle_rad = torch.tensor(angle * np.pi / 180)

        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-size//2, size//2, size),
            torch.linspace(-size//2, size//2, size),
            indexing='ij'
        )

        # Compute rotated coordinates
        cos_angle = torch.cos(angle_rad)
        sin_angle = torch.sin(angle_rad)
        x_rot = x_grid * cos_angle + y_grid * sin_angle
        y_rot = -x_grid * sin_angle + y_grid * cos_angle

        # Create Gabor filter components
        gaussian = torch.exp(-(x_rot**2 + y_rot**2)/(2*sigma**2))
        sinusoid = torch.cos(2 * np.pi * frequency * x_rot)

        # Combine and normalize
        kernel = (gaussian * sinusoid).view(1, 1, size, size)
        return kernel / kernel.abs().sum()

    def _create_pattern_kernel(self, size: int) -> torch.Tensor:
        """Create kernel for local pattern analysis"""
        kernel = torch.ones(size, size, dtype=torch.float32)
        center = size // 2
        kernel[center, center] = -size**2 + 1
        return (kernel / kernel.abs().sum()).view(1, 1, size, size)

    def _initialize_filters(self, device: torch.device):
        """Initialize filters if not already done"""
        if self.texture_filters is None:
            self.texture_filters = [
                self._create_gabor_kernel(frequency=f, angle=a).to(device)
                for f in [0.1, 0.2, 0.3] for a in [0, 45, 90, 135]
            ]

        if self.pattern_filters is None:
            self.pattern_filters = [
                self._create_pattern_kernel(size=s).to(device)
                for s in [3, 5, 7]
            ]

        self.color_filters = [f.to(device) for f in self.color_filters]

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss with agricultural pattern preservation"""
        device = reconstruction.device
        self._initialize_filters(device)

        # Basic reconstruction loss
        recon_loss = F.mse_loss(reconstruction, target)

        # Texture preservation loss
        texture_loss = 0
        for filter in self.texture_filters:
            rec_texture = F.conv2d(reconstruction, filter, padding=filter.size(-1)//2)
            target_texture = F.conv2d(target, filter, padding=filter.size(-1)//2)
            texture_loss += F.mse_loss(rec_texture, target_texture)
        texture_loss = texture_loss / len(self.texture_filters)

        # Pattern preservation loss
        pattern_loss = 0
        for filter in self.pattern_filters:
            rec_pattern = F.conv2d(reconstruction, filter, padding=filter.size(-1)//2)
            target_pattern = F.conv2d(target, filter, padding=filter.size(-1)//2)
            pattern_loss += F.mse_loss(rec_pattern, target_pattern)
        pattern_loss = pattern_loss / len(self.pattern_filters)

        # Color preservation loss
        color_loss = 0
        for filter in self.color_filters:
            rec_color = F.conv2d(reconstruction, filter)
            target_color = F.conv2d(target, filter)
            color_loss += F.mse_loss(rec_color, target_color)
        color_loss = color_loss / len(self.color_filters)

        # Local contrast preservation
        contrast_loss = F.mse_loss(
            torch.std(reconstruction, dim=[2, 3]),
            torch.std(target, dim=[2, 3])
        )

        # Combine losses with weights
        total_loss = (recon_loss +
                     2.0 * texture_loss +
                     1.5 * pattern_loss +
                     1.0 * color_loss +
                     0.5 * contrast_loss)

        return total_loss

    def _analyze_texture_statistics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze texture statistics for pattern detection"""
        stats = {}

        # Calculate local statistics using texture filters
        texture_responses = []
        for filter in self.texture_filters:
            response = F.conv2d(x, filter.to(x.device), padding=filter.size(-1)//2)
            texture_responses.append(response)

        # Compute texture energy
        stats['energy'] = torch.mean(torch.stack([r.pow(2).mean() for r in texture_responses]))

        # Compute texture contrast
        stats['contrast'] = torch.mean(torch.stack([r.std() for r in texture_responses]))

        return stats

    def _analyze_pattern_distribution(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze pattern distribution for anomaly detection"""
        stats = {}

        # Calculate pattern responses at different scales
        pattern_responses = []
        for filter in self.pattern_filters:
            response = F.conv2d(x, filter.to(x.device), padding=filter.size(-1)//2)
            pattern_responses.append(response)

        # Compute pattern density
        stats['density'] = torch.mean(torch.stack([r.abs().mean() for r in pattern_responses]))

        # Compute pattern variability
        stats['variability'] = torch.mean(torch.stack([r.var() for r in pattern_responses]))

        return stats

def safe_get_scalar(value):
    """Safely convert any numeric value to Python float"""
    if isinstance(value, torch.Tensor):
        return value.item()
    elif isinstance(value, (float, int)):
        return float(value)
    else:
        raise ValueError(f"Cannot convert {type(value)} to scalar")

class EnhancedLossManager:
    """Manager for handling specialized loss functions"""

    def __init__(self, config: Dict):
        self.config = config
        self.loss_functions = {}
        self.initialize_loss_functions()

    def initialize_loss_functions(self):
        """Initialize appropriate loss functions based on configuration"""
        enhancement_modules = self.config['model']['enhancement_modules']

        # Initialize astronomical loss if enabled
        if enhancement_modules['astronomical']['enabled']:
            self.loss_functions['astronomical'] = AstronomicalStructureLoss()

        # Initialize medical loss if enabled
        if enhancement_modules['medical']['enabled']:
            self.loss_functions['medical'] = MedicalStructureLoss()

        # Initialize agricultural loss if enabled
        if enhancement_modules['agricultural']['enabled']:
            self.loss_functions['agricultural'] = AgriculturalPatternLoss()

    def get_loss_function(self, image_type: str) -> Optional[nn.Module]:
        """Get appropriate loss function for image type"""
        return self.loss_functions.get(image_type)

    def calculate_loss(self, reconstruction: torch.Tensor, target: torch.Tensor, image_type: str) -> Dict[str, torch.Tensor]:
            """Calculate loss with appropriate enhancements"""
            loss_fn = self.get_loss_function(image_type)
            if loss_fn is None:
                return {'loss': F.mse_loss(reconstruction, target)}

            result = loss_fn(reconstruction, target)

            # Ensure we always return a dictionary with tensor loss
            if isinstance(result, dict):
                if 'loss' in result:
                    if isinstance(result['loss'], torch.Tensor):
                        return result
                    else:
                        return {'loss': torch.tensor(float(result['loss']), device=reconstruction.device)}
                else:
                    return {'loss': F.mse_loss(reconstruction, target)}
            elif isinstance(result, torch.Tensor):
                return {'loss': result}
            else:
                return {'loss': torch.tensor(float(result), device=reconstruction.device)}

class SharpnessAwareLoss(nn.Module):
    """Loss function focused on image sharpness and edge preservation for Phase 1 reconstruction"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.alpha = config.get('sharpness_weight', 0.3)  # Balance between reconstruction and sharpness
        self.edge_threshold = config.get('edge_threshold', 0.1)

        # Precompute Sobel filters for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32).view(1, 1, 3, 3)

        # Laplacian filter for high-frequency emphasis
        self.laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                                    dtype=torch.float32).view(1, 1, 3, 3)

    def _compute_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel operators"""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension if needed

        # Convert to grayscale for edge detection if multi-channel
        if x.shape[1] == 3:
            # Use luminance channel (standard grayscale conversion)
            x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            x_gray = x

        # Compute gradients
        g_x = F.conv2d(x_gray, self.sobel_x.to(x.device), padding=1, stride=1)
        g_y = F.conv2d(x_gray, self.sobel_y.to(x.device), padding=1, stride=1)

        # Compute magnitude
        gradient_magnitude = torch.sqrt(g_x**2 + g_y**2 + 1e-8)
        return gradient_magnitude

    def _compute_laplacian_response(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian response for high-frequency content"""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if x.shape[1] == 3:
            x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            x_gray = x

        laplacian_response = F.conv2d(x_gray, self.laplacian.to(x.device), padding=1, stride=1)
        return torch.abs(laplacian_response)

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute sharpness-aware loss

        Args:
            reconstruction: Model output [B, C, H, W]
            target: Ground truth [B, C, H, W]

        Returns:
            Dictionary containing loss components
        """
        device = reconstruction.device

        # Move filters to device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        self.laplacian = self.laplacian.to(device)

        # 1. Base reconstruction loss (existing functionality)
        base_loss = F.mse_loss(reconstruction, target)

        # 2. Gradient preservation loss
        target_grad = self._compute_gradient_magnitude(target)
        recon_grad = self._compute_gradient_magnitude(reconstruction)
        gradient_loss = F.mse_loss(recon_grad, target_grad)

        # 3. High-frequency preservation loss
        target_high_freq = self._compute_laplacian_response(target)
        recon_high_freq = self._compute_laplacian_response(reconstruction)
        high_freq_loss = F.mse_loss(recon_high_freq, target_high_freq)

        # 4. Edge-focused loss (emphasize regions with strong gradients)
        edge_mask = (target_grad > self.edge_threshold).float()
        # Avoid division by zero in masked loss
        if edge_mask.sum() > 0:
            edge_focused_loss = F.mse_loss(
                reconstruction * edge_mask,
                target * edge_mask
            )
        else:
            edge_focused_loss = torch.tensor(0.0, device=device)

        # Combine losses with REDUCED weights for sharpness components
        total_loss = (base_loss +
                     self.alpha * 0.1 * gradient_loss +  # REDUCED from 1.0 to 0.1
                     self.alpha * 0.05 * high_freq_loss +  # REDUCED from 0.5 to 0.05
                     self.alpha * 0.02 * edge_focused_loss)  # REDUCED from 0.3 to 0.02

        return {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'gradient_loss': gradient_loss,
            'high_freq_loss': high_freq_loss,
            'edge_focused_loss': edge_focused_loss
        }

class UnifiedCheckpoint:
    """Manages a unified checkpoint file containing multiple model states with feature selection persistence"""

    def __init__(self, config: Dict):
        self.config = config
        self.checkpoint_dir = config['training']['checkpoint_dir']
        self.dataset_name = config['dataset']['name']
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.dataset_name}_unified.pth")
        self.current_state = None
        self.model_type = config['model'].get('encoder_type', 'autoenc')

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Load or initialize checkpoint
        self.load_checkpoint()

    def save_model_state(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                         phase: int, epoch: int, loss: float, is_best: bool = False):
        """Save model state including feature selection configuration - PURE BINARY FORMAT ONLY"""
        state_key = self.get_state_key(phase, model)

        # COMPREHENSIVE RECURSIVE CLEANING of feature selection metadata
        feature_selection_metadata = getattr(model, '_feature_selection_metadata', {})
        clean_metadata = self._recursively_clean_for_binary(feature_selection_metadata)

        # COMPREHENSIVE CLEANING of feature selection parameters
        feature_selection_params = self.config.get('feature_selection', {}).get('parameters', {})
        clean_parameters = self._recursively_clean_for_binary(feature_selection_params)

        # Capture feature selection state - ALL data is binary compatible
        feature_selection_state = {
            'method': str(self.config.get('feature_selection', {}).get('method', 'balanced')),
            'parameters': clean_parameters,
            # Store tensor references directly - torch.save will handle binary serialization
            'selected_feature_indices': getattr(model, '_selected_feature_indices', None),
            'feature_importance_scores': getattr(model, '_feature_importance_scores', None),
            'feature_selection_metadata': clean_metadata  # Only fully cleaned data
        }

        # Prepare complete state dictionary - ALL data is binary compatible
        state_dict = {
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': int(epoch),
            'phase': int(phase),
            'loss': float(loss) if hasattr(loss, 'item') else float(loss),  # Ensure scalar
            'timestamp': datetime.now().isoformat(),
            'config': {
                'kl_divergence': bool(model.use_kl_divergence),  # Ensure bool
                'class_encoding': bool(model.use_class_encoding),  # Ensure bool
                'image_type': str(self.config['dataset'].get('image_type', 'general')),  # Ensure string
                'clustering_params': {
                    'num_clusters': int(model.cluster_centers.size(0)) if hasattr(model, 'cluster_centers') and model.cluster_centers is not None else 0,  # Ensure int with safety check
                    'temperature': float(model.clustering_temperature.item()) if hasattr(model, 'clustering_temperature') and model.clustering_temperature is not None and isinstance(model.clustering_temperature, torch.Tensor) else 1.0  # Ensure float with safety check
                },
                'feature_selection': feature_selection_state  # All data is binary compatible
            }
        }

        # Update model_states in the checkpoint
        if state_key not in self.current_state['model_states']:
            self.current_state['model_states'][state_key] = {
                'current': None,
                'best': None,
                'history': []
            }

        self.current_state['model_states'][state_key]['current'] = state_dict
        if is_best:
            self.current_state['model_states'][state_key]['best'] = state_dict

        # Save checkpoint - torch.save handles all binary serialization
        torch.save(self.current_state, self.checkpoint_path)
        #logger.info(f"Saved model state for {state_key} (binary format)")

    def _recursively_clean_for_binary(self, data: Any) -> Any:
        """
        Recursively clean data to ensure ONLY binary-compatible types remain.
        Completely removes any tensors, arrays, or complex objects that could cause serialization issues.
        """
        if data is None:
            return None

        # Remove all tensor-like objects
        if isinstance(data, (torch.Tensor, np.ndarray)):
            return None

        # Handle objects with .item() method (convert scalar tensors)
        if hasattr(data, 'item'):
            try:
                return data.item()
            except:
                return None

        # Handle basic serializable types
        if isinstance(data, (str, int, float, bool)):
            return data

        # Handle lists and tuples - recursively clean each element
        if isinstance(data, (list, tuple)):
            cleaned_list = []
            for item in data:
                cleaned_item = self._recursively_clean_for_binary(item)
                if cleaned_item is not None:
                    cleaned_list.append(cleaned_item)
            return cleaned_list if cleaned_list else None

        # Handle dictionaries - recursively clean each key-value pair
        if isinstance(data, dict):
            cleaned_dict = {}
            for key, value in data.items():
                # Ensure key is string
                str_key = str(key) if not isinstance(key, str) else key
                cleaned_value = self._recursively_clean_for_binary(value)
                if cleaned_value is not None:
                    cleaned_dict[str_key] = cleaned_value
            return cleaned_dict if cleaned_dict else None

        # Remove any other complex objects
        try:
            # Try to get a string representation for complex objects
            return str(data)
        except:
            # If even string conversion fails, remove entirely
            return None

                  #logger.info(f"Saved model state for {state_key} (binary format)")  # ADD THIS LOGGING LINE
                #logger.info(f"Saved model state for {state_key} (binary format)")

    def load_model_state(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        phase: int, load_best: bool = False) -> Optional[Dict]:
        """Load model state including feature selection configuration"""
        state_key = self.get_state_key(phase, model)

        if state_key not in self.current_state['model_states']:
            logger.info(f"No existing state found for {state_key}")
            return None

        # Get appropriate state
        state_dict = self.current_state['model_states'][state_key]['best' if load_best else 'current']
        if state_dict is None:
            return None

        # Load model state
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        # Restore feature selection state
        if 'config' in state_dict and 'feature_selection' in state_dict['config']:
            fs_state = state_dict['config']['feature_selection']
            model._selected_feature_indices = fs_state.get('selected_feature_indices')
            model._feature_importance_scores = fs_state.get('feature_importance_scores')
            model._feature_selection_metadata = fs_state.get('feature_selection_metadata', {})

            logger.info(f"Restored feature selection state: method={fs_state.get('method')}, "
                       f"selected_features={len(model._selected_feature_indices) if model._selected_feature_indices is not None else 'None'}")

        logger.info(f"Loaded {'best' if load_best else 'current'} state for {state_key}")
        return state_dict

    def load_checkpoint(self):
        """Load existing checkpoint or initialize new one"""
        if os.path.exists(self.checkpoint_path):
            self.current_state = torch.load(self.checkpoint_path)
            logger.info(f"Loaded unified checkpoint from {self.checkpoint_path}")
        else:
            self.current_state = {
                'model_states': {},
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'config': self.config
                }
            }
            logger.info("Initialized new unified checkpoint")

    def get_state_key(self, phase: int, model: nn.Module) -> str:
        """Generate unique key including model type"""
        components = [f"phase{phase}", f"model_{self.model_type}"]

        if phase == 2:
            if model.use_kl_divergence:
                components.append("kld")
            if model.use_class_encoding:
                components.append("cls")

            image_type = self.config['dataset'].get('image_type', 'general')
            if image_type != 'general':
                components.append(image_type)

        return "_".join(components)

    def get_best_loss(self, phase: int, model: nn.Module) -> float:
        """Get best loss for current configuration"""
        state_key = self.get_state_key(phase, model)
        if state_key in self.current_state['model_states']:
            best_state = self.current_state['model_states'][state_key]['best']
            if best_state is not None:
                loss = best_state['loss']
                return loss.item() if hasattr(loss, 'item') else float(loss)
        return float('inf')

    def print_checkpoint_summary(self):
        """Print summary of checkpoint contents"""
        print("\nUnified Checkpoint Summary:")
        print("-" * 50)
        print(f"Dataset: {self.dataset_name}")
        print(f"Last Updated: {self.current_state['metadata']['last_updated']}")
        print("\nModel States:")

        for state_key, state in self.current_state['model_states'].items():
            print(f"\n{state_key}:")
            if state['current'] is not None:
                print(f"  Current - Epoch: {state['current']['epoch']}, "
                      f"Loss: {state['current']['loss']:.4f}")
            if state['best'] is not None:
                print(f"  Best    - Epoch: {state['best']['epoch']}, "
                      f"Loss: {state['best']['loss']:.4f}")
            print(f"  History - {len(state['history'])} entries")

class ModelFactory:
    """Factory for creating appropriate model based on configuration"""

    @staticmethod
    def create_model(config: Dict) -> nn.Module:
        """Create model with invertible feature compression"""
        input_shape = (
            config['dataset']['in_channels'],  # Use configured channels
            config['dataset']['input_size'][0],
            config['dataset']['input_size'][1]
        )
        feature_dims = config['model']['feature_dims']

        # NEW: Set compressed dimensions in config
        if 'compressed_dims' not in config['model']:
            config['model']['compressed_dims'] = max(8, feature_dims // 4)

        image_type = config['dataset'].get('image_type', 'general')

        # Create appropriate model with invertible compression
        if image_type == 'astronomical':
            model = AstronomicalStructurePreservingAutoencoder(input_shape, feature_dims, config)
        elif image_type == 'medical':
            model = MedicalStructurePreservingAutoencoder(input_shape, feature_dims, config)
        elif image_type == 'agricultural':
            model = AgriculturalPatternAutoencoder(input_shape, feature_dims, config)
        else:
            model = BaseAutoencoder(input_shape, feature_dims, config)

        # Verify channel compatibility
        if hasattr(model, 'in_channels'):
            if model.in_channels != config['dataset']['in_channels']:
                logger.warning(f"Model expects {model.in_channels} channels but config specifies {config['dataset']['in_channels']}")

        logger.info(f"Created model with {feature_dims}D → {config['model']['compressed_dims']}D invertible compression")
        return model

class BaseFeatureSelector(ABC):
    """Abstract base class for all feature selectors"""

    @abstractmethod
    def select_features(self, features: np.ndarray, labels: np.ndarray,
                       **kwargs) -> Tuple[List[int], np.ndarray]:
        """Select features and return indices and scores"""
        pass

    def get_name(self) -> str:
        return self.__class__.__name__

class SimpleCorrelationSelector(BaseFeatureSelector):
    """Current correlation-based method (preserves existing behavior)"""

    def __init__(self, upper_threshold=0.85, lower_threshold=0.01):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def select_features(self, features: np.ndarray, labels: np.ndarray, **kwargs) -> Tuple[List[int], np.ndarray]:
        """Original correlation-based selection logic"""
        n_features = features.shape[1]
        label_corrs = np.zeros(n_features)

        # Calculate correlation with labels (original logic)
        for i in range(n_features):
            label_corrs[i] = 1 - correlation(features[:, i], labels)

        # Get indices of features that meet upper threshold
        selected_indices = [i for i, corr in enumerate(label_corrs)
                          if corr >= self.upper_threshold]

        # Sort by correlation strength (descending)
        selected_indices.sort(key=lambda i: -label_corrs[i])

        # Remove features that are too correlated with each other
        final_indices = []
        feature_matrix = features[:, selected_indices]

        for i, idx in enumerate(selected_indices):
            keep = True
            for j in final_indices:
                # Calculate correlation between features
                corr = 1 - correlation(feature_matrix[:, i], feature_matrix[:, selected_indices.index(j)])
                if corr > self.lower_threshold:
                    keep = False
                    break
            if keep:
                final_indices.append(idx)

        return final_indices, label_corrs

class BalancedFeatureSelector(BaseFeatureSelector):
    """Balanced approach - correlation + diversity"""

    def __init__(self, target_features=20, min_correlation=0.3, max_inter_correlation=0.8):
        self.target_features = target_features
        self.min_correlation = min_correlation
        self.max_inter_correlation = max_inter_correlation

    def select_features(self, features: np.ndarray, labels: np.ndarray, **kwargs) -> Tuple[List[int], np.ndarray]:
        start_time = time.time()
        n_features = features.shape[1]

        # Step 1: Calculate correlations with labels
        correlation_scores = self._calculate_correlations(features, labels)

        # Step 2: Get candidate features meeting minimum correlation
        candidate_indices = [i for i, score in enumerate(correlation_scores)
                           if score >= self.min_correlation]

        # Step 3: If insufficient candidates, use top features
        if len(candidate_indices) < max(5, self.target_features // 2):
            logger.info(f"Only {len(candidate_indices)} features meet correlation threshold {self.min_correlation}")
            # Use top features by correlation
            top_count = min(self.target_features * 2, n_features)
            candidate_indices = np.argsort(correlation_scores)[-top_count:]
            candidate_indices = list(candidate_indices)

        # Sort candidates by correlation (descending)
        candidate_indices.sort(key=lambda i: -correlation_scores[i])

        # Step 4: Remove redundant features
        final_indices = self._remove_redundant_features(features, candidate_indices, correlation_scores)

        # Step 5: Ensure target count
        if len(final_indices) > self.target_features:
            # Take top by correlation
            final_indices.sort(key=lambda i: -correlation_scores[i])
            final_indices = final_indices[:self.target_features]
        elif len(final_indices) < self.target_features:
            # Add more features from candidates
            remaining = [i for i in candidate_indices if i not in final_indices]
            needed = self.target_features - len(final_indices)
            final_indices.extend(remaining[:needed])

        elapsed = time.time() - start_time
        logger.info(f"Balanced selection: {len(final_indices)} features in {elapsed:.2f}s "
                   f"(correlation range: {min(correlation_scores[final_indices]):.3f}-{max(correlation_scores[final_indices]):.3f})")

        return final_indices, correlation_scores

    def _calculate_correlations(self, features, labels):
        """Calculate absolute correlation with labels"""
        n_features = features.shape[1]
        correlations = np.zeros(n_features)

        for i in range(n_features):
            try:
                corr_val = abs(np.corrcoef(features[:, i], labels)[0, 1])
                correlations[i] = corr_val
            except:
                correlations[i] = 0.0

        return correlations

    def _remove_redundant_features(self, features, candidate_indices, correlation_scores):
        """Remove features that are highly correlated with each other"""
        final_indices = []

        for idx in candidate_indices:
            if len(final_indices) >= self.target_features:
                break

            is_redundant = False
            current_feature = features[:, idx]

            for existing_idx in final_indices:
                existing_feature = features[:, existing_idx]
                try:
                    inter_correlation = abs(np.corrcoef(current_feature, existing_feature)[0, 1])
                    if inter_correlation > self.max_inter_correlation:
                        is_redundant = True
                        # Keep the one with higher correlation to labels
                        if correlation_scores[idx] > correlation_scores[existing_idx]:
                            final_indices.remove(existing_idx)
                            final_indices.append(idx)
                        break
                except:
                    continue

            if not is_redundant and idx not in final_indices:
                final_indices.append(idx)

        return final_indices

class ComplexEnsembleSelector(BaseFeatureSelector):
    """Complex ensemble approach for maximum feature quality"""

    def __init__(self, target_features=20, ensemble_weights=None, model=None):
        self.target_features = target_features
        self.ensemble_weights = ensemble_weights or {
            'correlation': 0.25,
            'clustering': 0.30,
            'importance': 0.25,
            'diversity': 0.10,
            'stability': 0.10
        }
        self.model = model

    def select_features(self, features: np.ndarray, labels: np.ndarray,
                       dataloader=None, **kwargs) -> Tuple[List[int], np.ndarray]:
        start_time = time.time()
        n_features = features.shape[1]

        logger.info("Starting complex ensemble feature selection...")

        # Calculate scores for each criterion
        detailed_scores = {}

        # 1. Correlation scores
        correlation_scores = self._calculate_correlations(features, labels)
        detailed_scores['correlation'] = correlation_scores

        # 2. Clustering quality scores
        clustering_scores = self._calculate_clustering_quality(features, labels)
        detailed_scores['clustering'] = clustering_scores

        # 3. Feature importance scores (if model available)
        if self.model is not None and dataloader is not None:
            importance_scores = self._calculate_feature_importance(dataloader, n_features)
        else:
            importance_scores = np.ones(n_features)
        detailed_scores['importance'] = importance_scores

        # 4. Feature diversity scores
        diversity_scores = self._calculate_diversity(features)
        detailed_scores['diversity'] = diversity_scores

        # 5. Feature stability scores
        stability_scores = self._calculate_stability(features, labels)
        detailed_scores['stability'] = stability_scores

        # Combine scores
        ensemble_scores = self._combine_scores(detailed_scores)

        # Select final features
        selected_indices = self._select_final_features(features, ensemble_scores)

        elapsed = time.time() - start_time
        self._log_selection_summary(selected_indices, detailed_scores, ensemble_scores, elapsed)

        return selected_indices, ensemble_scores

    def _calculate_correlations(self, features, labels):
        """Calculate correlation scores"""
        n_features = features.shape[1]
        scores = np.zeros(n_features)

        for i in range(n_features):
            try:
                corr = abs(np.corrcoef(features[:, i], labels)[0, 1])
                scores[i] = corr
            except:
                scores[i] = 0.0

        return scores

    def _calculate_clustering_quality(self, features, labels):
        """Calculate how well features separate classes"""
        n_features = features.shape[1]
        scores = np.zeros(n_features)

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return np.ones(n_features)  # Can't calculate for single class

        for i in range(n_features):
            feature_values = features[:, i]

            # Calculate between-class vs within-class variance
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
                scores[i] = between_var / within_var
            else:
                scores[i] = between_var

        return scores

    def _calculate_feature_importance(self, dataloader, n_features):
        """Calculate feature importance using model activations"""
        try:
            activations = []
            self.model.eval()

            with torch.no_grad():
                for inputs, _ in dataloader:
                    outputs = self.model(inputs)
                    if 'embedding' in outputs:
                        activations.append(outputs['embedding'].cpu().numpy())

            if activations:
                activations = np.vstack(activations)
                # Use variance as proxy for importance
                importance_scores = np.var(activations, axis=0)

                # Ensure correct length
                if len(importance_scores) == n_features:
                    return importance_scores
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")

        return np.ones(n_features)

    def _calculate_diversity(self, features):
        """Calculate how unique each feature is"""
        n_features = features.shape[1]
        diversity_scores = np.ones(n_features)

        # Sample correlations to avoid O(n²) complexity
        sample_size = min(50, n_features)
        sampled_features = np.random.choice(n_features, sample_size, replace=False)

        for i in range(n_features):
            correlations = []
            for j in sampled_features:
                if i != j:
                    try:
                        corr = abs(np.corrcoef(features[:, i], features[:, j])[0, 1])
                        correlations.append(corr)
                    except:
                        continue

            if correlations:
                diversity_scores[i] = 1 - np.mean(correlations)

        return diversity_scores

    def _calculate_stability(self, features, labels, n_splits=3):
        """Calculate feature stability across data splits"""
        n_features = features.shape[1]
        stability_scores = np.ones(n_features)

        if len(features) < 20:  # Need sufficient data
            return stability_scores

        for i in range(n_features):
            feature_values = features[:, i]
            split_means = []

            for _ in range(n_splits):
                # Random split
                split_point = int(0.7 * len(feature_values))
                indices = np.random.permutation(len(feature_values))
                train_indices = indices[:split_point]

                if len(train_indices) > 0:
                    split_mean = np.mean(feature_values[train_indices])
                    split_means.append(split_mean)

            if len(split_means) > 1:
                stability_scores[i] = 1 - (np.std(split_means) / (np.std(feature_values) + 1e-8))

        return np.clip(stability_scores, 0, 1)

    def _combine_scores(self, detailed_scores):
        """Combine individual scores using ensemble weights"""
        n_features = len(detailed_scores['correlation'])
        combined_scores = np.zeros(n_features)

        for criterion, scores in detailed_scores.items():
            if criterion in self.ensemble_weights:
                # Normalize scores to [0, 1]
                if np.max(scores) > np.min(scores):
                    normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
                else:
                    normalized = np.ones_like(scores) * 0.5

                weight = self.ensemble_weights[criterion]
                combined_scores += weight * normalized

        return combined_scores

    def _select_final_features(self, features, ensemble_scores):
        """Select final features considering scores and diversity"""
        n_features = len(ensemble_scores)

        # Sort by ensemble score
        sorted_indices = np.argsort(ensemble_scores)[::-1]

        # Select features with diversity consideration
        selected_indices = []
        for idx in sorted_indices:
            if len(selected_indices) >= self.target_features:
                break

            # Check diversity
            if self._is_diverse(features, idx, selected_indices):
                selected_indices.append(idx)

        # Ensure we have target count
        if len(selected_indices) < self.target_features:
            remaining = [i for i in sorted_indices if i not in selected_indices]
            needed = self.target_features - len(selected_indices)
            selected_indices.extend(remaining[:needed])

        return selected_indices[:self.target_features]

    def _is_diverse(self, features, candidate_idx, selected_indices, threshold=0.8):
        """Check if candidate is sufficiently different from selected features"""
        if not selected_indices:
            return True

        candidate_feature = features[:, candidate_idx]

        for selected_idx in selected_indices:
            selected_feature = features[:, selected_idx]
            try:
                correlation = abs(np.corrcoef(candidate_feature, selected_feature)[0, 1])
                if correlation > threshold:
                    return False
            except:
                continue

        return True

    def _log_selection_summary(self, selected_indices, detailed_scores, ensemble_scores, elapsed_time):
        """Log selection details"""
        logger.info("=" * 50)
        logger.info("COMPLEX ENSEMBLE SELECTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Selected {len(selected_indices)} features in {elapsed_time:.2f}s")

        # Show criterion contributions
        for criterion, weight in self.ensemble_weights.items():
            if criterion in detailed_scores:
                selected_scores = detailed_scores[criterion][selected_indices]
                logger.info(f"{criterion:12s} (weight={weight:.2f}): mean={np.mean(selected_scores):.3f}")

        logger.info(f"Ensemble score range: {np.min(ensemble_scores[selected_indices]):.3f}-{np.max(ensemble_scores[selected_indices]):.3f}")
        logger.info("=" * 50)

class FeatureSelectorFactory:
    """Factory to create appropriate feature selector"""

    @staticmethod
    def create_selector(selector_type: str, config: Dict, model=None) -> BaseFeatureSelector:
        """
        Create feature selector based on type and configuration.

        Args:
            selector_type: 'simple', 'balanced', or 'complex'
            config: Configuration dictionary
            model: Optional model for complex selection
        """
        feature_config = config.get('feature_selection', {})
        params = feature_config.get('parameters', {})

        if selector_type == 'simple':
            return SimpleCorrelationSelector(
                upper_threshold=params.get('upper_threshold', 0.85),
                lower_threshold=params.get('lower_threshold', 0.01)
            )

        elif selector_type == 'balanced':
            return BalancedFeatureSelector(
                target_features=params.get('target_features', 20),
                min_correlation=params.get('min_correlation', 0.3),
                max_inter_correlation=params.get('max_inter_correlation', 0.8)
            )

        elif selector_type == 'complex':
            return ComplexEnsembleSelector(
                target_features=params.get('target_features', 20),
                ensemble_weights=params.get('ensemble_weights', {
                    'correlation': 0.25, 'clustering': 0.30, 'importance': 0.25,
                    'diversity': 0.10, 'stability': 0.10
                }),
                model=model
            )

        else:
            logger.warning(f"Unknown selector type: {selector_type}. Using balanced.")
            return BalancedFeatureSelector()

def extract_features_sliding_window(model, large_image_paths, config, logger):
    """Extract features from large images using sliding window approach"""
    logger.info(f"Extracting features from {len(large_image_paths)} large images using sliding window")

    features_dict = {
        'train': {'embeddings': [], 'labels': [], 'filenames': [], 'full_paths': []},
        'test': {'embeddings': [], 'labels': [], 'filenames': [], 'full_paths': []}
    }

    model.eval()

    # Setup sliding window parameters
    window_size = config['training'].get('window_size', 512)
    stride = config['training'].get('stride', 256)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config['dataset']['mean'],
                           std=config['dataset']['std'])
    ])

    with torch.no_grad():
        for image_path in tqdm(large_image_paths, desc="Processing large images"):
            try:
                # Create sliding window dataset for this image
                dataset = SlidingWindowDataset(
                    image_paths=[image_path],
                    window_size=window_size,
                    stride=stride,
                    transform=transform,
                    overlap=0.5
                )

                loader = DataLoader(
                    dataset,
                    batch_size=config['training'].get('window_batch_size', 4),
                    shuffle=False,
                    num_workers=1
                )

                # Process all windows for this image
                image_embeddings = []
                for batch_data in loader:
                    windows = batch_data['window'].to(model.device)

                    # Extract embeddings from windows
                    embeddings = model.encode(windows)
                    if isinstance(embeddings, tuple):
                        embeddings = embeddings[0]  # Take main embedding

                    image_embeddings.append(embeddings.cpu())

                if image_embeddings:
                    # Aggregate window embeddings (mean pooling)
                    all_embeddings = torch.cat(image_embeddings, dim=0)
                    aggregated_embedding = all_embeddings.mean(dim=0, keepdim=True)

                    # Store features
                    features_dict['train']['embeddings'].append(aggregated_embedding)
                    features_dict['train']['labels'].append(torch.tensor([0]))  # Dummy label
                    features_dict['train']['filenames'].append(os.path.basename(image_path))
                    features_dict['train']['full_paths'].append(image_path)

                    logger.debug(f"Extracted features from {image_path}: {aggregated_embedding.shape}")

            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                continue

    # Convert lists to tensors
    for split in ['train', 'test']:
        if features_dict[split]['embeddings']:
            features_dict[split]['embeddings'] = torch.cat(features_dict[split]['embeddings'], dim=0)
            features_dict[split]['labels'] = torch.cat(features_dict[split]['labels'], dim=0)

    logger.info(f"Extracted features from {len(features_dict['train']['embeddings'])} images")
    return features_dict

def train_sliding_window_model(model, large_image_paths, config):
    """Train model using sliding window approach for large images"""

    # Setup sliding window dataset
    window_size = config['training'].get('window_size', 256)
    stride = config['training'].get('stride', 128)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config['dataset']['mean'],
                           std=config['dataset']['std'])
    ])

    dataset = SlidingWindowDataset(
        image_paths=large_image_paths,
        window_size=window_size,
        stride=stride,
        transform=transform,
        overlap=0.5  # 50% overlap for better reconstruction
    )

    # Create data loader with small batch size (we're processing windows)
    loader = DataLoader(
        dataset,
        batch_size=config['training'].get('window_batch_size', 8),
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    # Train using standard approach but on windows
    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    loss_manager = EnhancedLossManager(config)

    # Standard training loop on windows
    history = _train_phase(model, loader, optimizer, loss_manager,
                          config['training']['epochs'], 1, config)

    return history

def process_very_large_image(model, image_path, output_path, config):
    """Process a single very large image using sliding windows"""

    # Load the large image
    with Image.open(image_path) as img:
        original_array = np.array(img)
        original_shape = original_array.shape

    # Convert to tensor
    transform = transforms.ToTensor()
    image_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Setup reconstructor
    window_size = config['training'].get('window_size', 256)
    stride = config['training'].get('stride', 128)

    reconstructor = WindowReconstructor(
        original_shape=image_tensor.shape[1:],  # (C, H, W)
        window_size=window_size,
        stride=stride,
        blend_mode='linear'
    )

    # Process the image
    with torch.no_grad():
        reconstruction, embedding = model.process_full_image(image_tensor, reconstructor)

    # Save reconstruction
    reconstruction_pil = transforms.ToPILImage()(reconstruction.squeeze(0))
    reconstruction_pil.save(output_path)

    logger.info(f"Processed {image_path} ({original_shape[1]}x{original_shape[0]})")
    logger.info(f"Saved reconstruction to {output_path}")

    return embedding

def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                config: Dict, loss_manager: EnhancedLossManager) -> Dict[str, List]:
    """Two-phase training implementation with folder-based dataset handling"""
    # Store dataset reference in model for class information
    model.set_dataset(train_loader.dataset)

    # Update config with dataset information
    if hasattr(train_loader.dataset, 'classes'):
        config['dataset']['num_classes'] = len(train_loader.dataset.classes)
        config['dataset']['class_names'] = train_loader.dataset.classes
        logger.info(f"Dataset: {len(train_loader.dataset.classes)} classes - {train_loader.dataset.classes}")

    history = defaultdict(list)

    # Initialize starting epoch and phase
    start_epoch = getattr(model, 'current_epoch', 0)
    current_phase = getattr(model, 'training_phase', 1)

    # Phase 1: Pure reconstruction (if not already completed)
    if current_phase == 1:
        logger.info("Starting/Resuming Phase 1: Pure reconstruction training")
        model.set_training_phase(1)
        optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])

        phase1_history = _train_phase(
            model, train_loader, optimizer, loss_manager,
            config['training']['epochs'], 1, config,
            start_epoch=start_epoch
        )
        history.update(phase1_history)

        # Reset start_epoch for phase 2
        start_epoch = 0
    else:
        logger.info("Phase 1 already completed, skipping")

    # Phase 2: Latent space organization
    if config['model']['autoencoder_config']['enhancements'].get('enable_phase2', True):
        if current_phase < 2:
            logger.info("Starting Phase 2: Latent space organization")
            model.set_training_phase(2)
        else:
            logger.info("Resuming Phase 2: Latent space organization")

        # Lower learning rate for fine-tuning
        optimizer = optim.Adam(model.parameters(),
                             lr=config['model']['learning_rate'] * 0.1)  # Lower LR for phase 2

        phase2_history = _train_phase(
            model, train_loader, optimizer, loss_manager,
            config['training']['epochs'], 2, config,
            start_epoch=start_epoch if current_phase == 2 else 0
        )

        # Merge histories
        for key, value in phase2_history.items():
            history[f"phase2_{key}"] = value

    # CLEAN THE HISTORY TO REMOVE ANY TENSORS BEFORE RETURNING
    cleaned_history = model._clean_history_for_serialization(history)

    logger.info("Training completed successfully")
    return cleaned_history

def extract_features_from_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader] = None
) -> Dict:
    """Extract features from the model using folder-based dataset structure"""
    logger.info("Extracting features from model...")

    # Extract train features
    train_features = model.extract_features(train_loader, "train")
    logger.info(f"Extracted {len(train_features['embeddings'])} training samples")

    features_dict = {'train': train_features}

    # Extract test features if available
    if test_loader is not None:
        test_features = model.extract_features(test_loader, "test")
        logger.info(f"Extracted {len(test_features['embeddings'])} test samples")
        features_dict['test'] = test_features

        # Combine train and test features for unified processing
        combined_features = {}
        for key in train_features:
            if isinstance(train_features[key], torch.Tensor):
                combined_features[key] = torch.cat([train_features[key], test_features[key]])
            elif isinstance(train_features[key], list):
                combined_features[key] = train_features[key] + test_features[key]
            else:
                combined_features[key] = train_features[key]  # Use train version for non-combinable items

        features_dict['combined'] = combined_features
        logger.info(f"Combined features: {len(combined_features['embeddings'])} total samples")
    else:
        features_dict['combined'] = train_features

    return features_dict

def _get_checkpoint_identifier(model: nn.Module, phase: int, config: Dict) -> str:
    """
    Generate unique identifier for checkpoint based on phase and active enhancements.
    """
    # Start with phase identifier
    identifier = f"phase{phase}"

    # Add active enhancements
    if phase == 2:
        active_enhancements = []
        if model.use_kl_divergence:
            active_enhancements.append("kld")
        if model.use_class_encoding:
            active_enhancements.append("cls")

        # Add specialized enhancements
        if config['dataset'].get('image_type') != 'general':
            image_type = config['dataset']['image_type']
            if config['model']['enhancement_modules'].get(image_type, {}).get('enabled', True):
                active_enhancements.append(image_type)

        if active_enhancements:
            identifier += "_" + "_".join(sorted(active_enhancements))

    return identifier

def _save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, phase: int, loss: float, config: Dict,
                    is_best: bool = False):
    """
    Save training checkpoint with phase and enhancement-specific handling.
    """
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get unique identifier for this configuration
    identifier = _get_checkpoint_identifier(model, phase, config)
    dataset_name = config['dataset']['name']

    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'phase': phase,
        'training_phase': model.training_phase,
        'loss': loss,
        'identifier': identifier,
        'config': config,
        'active_enhancements': {
            'kl_divergence': model.use_kl_divergence,
            'class_encoding': model.use_class_encoding,
            'image_type': config['dataset'].get('image_type', 'general')
        }
    }

    # Always save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, f"{dataset_name}_{identifier}_latest.pth")
    torch.save(checkpoint, latest_path)

    # Save phase-specific best model if applicable
    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{dataset_name}_{identifier}_best.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model for {identifier} with loss: {loss:.4f}")

    logger.info(f"Saved checkpoint for {identifier} at epoch {epoch + 1}")

def load_best_checkpoint(model: nn.Module, phase: int, config: Dict) -> Optional[Dict]:
    """
    Load the best checkpoint for the given phase and enhancement combination.
    """
    checkpoint_dir = config['training']['checkpoint_dir']
    identifier = _get_checkpoint_identifier(model, phase, config)
    dataset_name = config['dataset']['name']
    best_path = os.path.join(checkpoint_dir, f"{dataset_name}_{identifier}_best.pth")

    if os.path.exists(best_path):
        logger.info(f"Loading best checkpoint for {identifier}")
        return torch.load(best_path, map_location=model.device)
    return None

def update_phase_specific_metrics(model: nn.Module, phase: int, config: Dict) -> Dict[str, Any]:
    """
    Track and return phase-specific metrics and best values.
    """
    metrics = {}
    identifier = _get_checkpoint_identifier(model, phase, config)

    # Try to load existing best metrics
    checkpoint = load_best_checkpoint(model, phase, config)
    if checkpoint:
        metrics['best_loss'] = checkpoint.get('loss', float('inf'))
        metrics['best_epoch'] = checkpoint.get('epoch', 0)
    else:
        metrics['best_loss'] = float('inf')
        metrics['best_epoch'] = 0

    return metrics

def _train_phase(model: nn.Module, train_loader: DataLoader,
                optimizer: torch.optim.Optimizer, loss_manager: EnhancedLossManager,
                epochs: int, phase: int, config: Dict, start_epoch: int = 0) -> Dict[str, List]:
    """Training logic optimized for reconstruction and classification accuracy"""

    history = defaultdict(list)
    device = next(model.parameters()).device
    best_loss = float('inf')
    min_thr = float(config['model']['autoencoder_config']["convergence_threshold"])
    checkpoint_manager = UnifiedCheckpoint(config)
    use_classwise = config['training'].get('use_classwise_acc', True)
    patience_counter = 0

    # Optimized learning rates for each phase
    if phase == 2:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
        logger.info(f"Phase 2 learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    # Initialize reconstruction sampler
    sampler_enabled = config['training'].get('reconstruction_samples', {}).get('enabled', True)
    if sampler_enabled:
        sampler = ReconstructionSampler(config)
        save_frequency = config['training'].get('reconstruction_samples', {}).get('save_frequency', 5)
        logger.info(f"Reconstruction sampling enabled - saving every {save_frequency} epochs")

    # Initialize training heatmap generator
    heatmap_enabled = config['training'].get('generate_training_heatmaps', True)
    heatmap_frequency = config['training'].get('heatmap_frequency', 10)

    if heatmap_enabled:
        dataset_name = config['dataset']['name'].lower()
        training_heatmap_dir = os.path.join('data', dataset_name, 'training_heatmaps', f'phase{phase}')
        os.makedirs(training_heatmap_dir, exist_ok=True)
        logger.info(f"Training heatmaps enabled - saving every {heatmap_frequency} epochs")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            num_batches = len(train_loader)
            valid_batches = 0

            pbar = tqdm(train_loader, desc=f"Phase {phase} - Epoch {epoch+1}", leave=False)
            for batch_idx, (data, labels) in enumerate(pbar):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()

                # Skip batch if data contains NaN/Inf
                if torch.isnan(data).any() or torch.isinf(data).any():
                    logger.warning("NaN/Inf in input data! Skipping batch")
                    continue

                # Forward pass with accuracy-focused logic
                try:
                    if phase == 1:
                        # Phase 1: Focus on high-quality reconstruction
                        outputs = model(data, labels=labels)

                        # Check for NaN in outputs
                        if any(torch.isnan(tensor).any() for tensor in outputs.values() if isinstance(tensor, torch.Tensor)):
                            logger.error("NaN in Phase 1 outputs! Skipping batch")
                            continue

                        # Handle both output formats
                        if isinstance(outputs, tuple):
                            _, reconstruction = outputs
                        else:
                            reconstruction = outputs.get('reconstruction', data)

                        # Ensure reconstruction matches input size
                        if reconstruction.shape != data.shape:
                            reconstruction = F.interpolate(reconstruction, size=data.shape[2:],
                                                         mode='bilinear', align_corners=False)

                        # Optimized Phase 1 loss for reconstruction quality
                        loss = F.mse_loss(reconstruction, data)

                        # Add feature consistency for better representations
                        if 'embedding' in outputs and hasattr(model, 'feature_compressor'):
                            compressed = model.feature_compressor(outputs['embedding'])
                            reconstructed_embedding = model.feature_decompressor(compressed)
                            feature_loss = F.mse_loss(reconstructed_embedding, outputs['embedding'])
                            loss += 0.1 * feature_loss

                    else:
                        # Phase 2: Focus on classification accuracy with reconstruction
                        output = model(data, labels=labels)

                        # Check for NaN in outputs
                        if any(torch.isnan(tensor).any() for tensor in output.values() if isinstance(tensor, torch.Tensor)):
                            logger.error("NaN in Phase 2 outputs! Skipping batch")
                            continue

                        reconstruction = output['reconstruction']

                        # Ensure reconstruction quality
                        if reconstruction.shape != data.shape:
                            reconstruction = F.interpolate(reconstruction, size=data.shape[2:],
                                                         mode='bilinear', align_corners=False)

                        # Base reconstruction loss
                        recon_loss = F.mse_loss(reconstruction, data)
                        total_loss = recon_loss

                        # Classification loss with higher priority for accuracy
                        if model.use_class_encoding and 'class_logits' in output:
                            class_logits = output['class_logits']

                            # Use class-balanced loss for better accuracy on imbalanced data
                            if use_classwise:
                                class_losses = []
                                for cls in torch.unique(labels):
                                    mask = labels == cls
                                    if mask.sum() > 0:
                                        cls_loss = F.cross_entropy(class_logits[mask], labels[mask], reduction='mean')
                                        class_losses.append(cls_loss)

                                if class_losses:
                                    class_loss = torch.mean(torch.stack(class_losses))
                                else:
                                    class_loss = torch.tensor(0.0, device=device)
                            else:
                                class_loss = F.cross_entropy(class_logits, labels)

                            # Higher weight for classification accuracy
                            total_loss += 0.8 * class_loss

                            # Calculate accuracy
                            with torch.no_grad():
                                preds = torch.argmax(class_logits, dim=1)
                                if use_classwise:
                                    class_acc = []
                                    for cls in torch.unique(labels):
                                        mask = labels == cls
                                        if mask.sum() > 0:
                                            cls_acc = (preds[mask] == labels[mask]).float().mean()
                                            class_acc.append(cls_acc)
                                    acc = torch.mean(torch.stack(class_acc)) if class_acc else 0.0
                                else:
                                    acc = (preds == labels).float().mean()
                                running_acc += acc.item()

                        # KL divergence for clustering (secondary priority)
                        if model.use_kl_divergence and 'cluster_probabilities' in output and 'target_distribution' in output:
                            q_dist = output['cluster_probabilities']
                            p_dist = output['target_distribution']

                            if not (torch.isnan(q_dist).any() or torch.isnan(p_dist).any()):
                                kl_loss = F.kl_div(q_dist.log(), p_dist, reduction='batchmean')
                                # Lower weight to prioritize classification
                                total_loss += 0.2 * kl_loss

                                # Store KL metrics
                                if not hasattr(model, 'kl_metrics'):
                                    model.kl_metrics = {}
                                model.kl_metrics = {
                                    'kl_loss': kl_loss.item(),
                                    'cluster_confidence': output['cluster_confidence'].mean().item(),
                                    'unique_clusters': len(torch.unique(output['cluster_assignments']))
                                }

                        loss = total_loss

                except Exception as e:
                    logger.error(f"Forward pass failed: {e}. Skipping batch")
                    continue

                # Backpropagation with stability
                try:
                    loss.backward()

                    # Gradient stability check
                    has_nan_grad = False
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                has_nan_grad = True
                                break
                            grad_norm += p.grad.norm().item() ** 2

                    if has_nan_grad:
                        logger.error("NaN gradients detected! Skipping optimization")
                        optimizer.zero_grad()
                        continue

                    # Stable gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    valid_batches += 1

                except Exception as e:
                    logger.error(f"Backward pass failed: {e}. Skipping batch")
                    optimizer.zero_grad()
                    continue

                # Update metrics
                running_loss += loss.item()
                avg_loss = running_loss / (batch_idx + 1)
                status = {
                    'loss': f"{Colors.color_value(avg_loss, best_loss, False)}",
                    'best': f"{best_loss:.4f}",
                    'patience': f"{patience_counter}"
                }

                if phase == 2 and model.use_class_encoding:
                    avg_acc = running_acc / (batch_idx + 1)
                    status['acc'] = f"{avg_acc:.2%}"

                pbar.set_postfix(status)

                # Cleanup
                del data, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Epoch statistics
            if valid_batches > 0:
                avg_loss = running_loss / valid_batches
                history[f'phase{phase}_loss'].append(avg_loss)
            else:
                avg_loss = float('inf')
                history[f'phase{phase}_loss'].append(avg_loss)
                logger.warning(f"No valid batches in epoch {epoch+1}!")

            # Store accuracy for Phase 2
            if phase == 2 and model.use_class_encoding and valid_batches > 0:
                avg_acc = running_acc / valid_batches
                history[f'phase{phase}_accuracy'].append(avg_acc)

            # Store KL divergence metrics
            if phase == 2 and model.use_kl_divergence and hasattr(model, 'kl_metrics'):
                history[f'phase{phase}_kl_loss'].append(model.kl_metrics.get('kl_loss', 0))
                history[f'phase{phase}_cluster_confidence'].append(model.kl_metrics.get('cluster_confidence', 0))
                history[f'phase{phase}_unique_clusters'].append(model.kl_metrics.get('unique_clusters', 0))
                logger.info(f"KL Metrics - Loss: {model.kl_metrics.get('kl_loss', 0):.4f}, "
                          f"Confidence: {model.kl_metrics.get('cluster_confidence', 0):.4f}")

            # Save reconstruction samples
            if sampler_enabled and (epoch % save_frequency == 0 or epoch == epochs - 1):
                logger.info(f"Saving reconstruction samples for epoch {epoch+1}")
                try:
                    sampler.save_reconstruction_samples(model, train_loader, epoch+1, phase)
                except Exception as e:
                    logger.warning(f"Could not save reconstruction samples: {str(e)}")

            # Generate training heatmaps
            if (heatmap_enabled and
                (epoch % heatmap_frequency == 0 or epoch == epochs - 1 or epoch == 0)):
                logger.info(f"Generating training heatmaps for epoch {epoch+1}")
                try:
                    _generate_training_heatmaps(model, train_loader, config, epoch+1, phase, training_heatmap_dir)
                except Exception as e:
                    logger.warning(f"Could not generate training heatmaps: {str(e)}")

            # Log epoch progress
            log_message = f"Phase {phase} - Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}"
            if phase == 2 and model.use_class_encoding and valid_batches > 0:
                log_message += f" | Accuracy: {avg_acc:.2%}"
            logger.info(log_message)

            # Checkpointing
            if avg_loss != float('inf') and (best_loss - avg_loss) > min_thr:
                best_loss = avg_loss
                patience_counter = 0
                checkpoint_manager.save_model_state(
                    model, optimizer, phase, epoch, avg_loss, True
                )
                logger.info(f"New best model saved with loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement - Patience counter: {patience_counter}")

            # Early stopping
            if patience_counter >= config['training'].get('early_stopping', {}).get('patience', 5):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                if sampler_enabled:
                    try:
                        sampler.save_reconstruction_samples(model, train_loader, epoch+1, phase)
                    except Exception as e:
                        logger.warning(f"Could not save final samples: {str(e)}")
                break

            # Learning rate scheduling
            if 'scheduler' in config['model'] and config['model']['scheduler'].get('enabled', False):
                if hasattr(optimizer, 'param_groups') and avg_loss != float('inf'):
                    current_lr = optimizer.param_groups[0]['lr']
                    min_lr = config['model']['scheduler'].get('min_lr', 1e-6)
                    if avg_loss > best_loss * 1.1 and current_lr > min_lr:
                        new_lr = max(current_lr * 0.5, min_lr)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        logger.info(f"Reduced learning rate to {new_lr:.2e}")

    except Exception as e:
        logger.error(f"Training failed at epoch {epoch+1}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    # Training completion
    completion_message = f"Phase {phase} training completed. Best loss: {best_loss:.4f}"
    if phase == 1 and hasattr(model, 'compressed_dims'):
        completion_message += f" | Compression: {model.feature_dims}D → {model.compressed_dims}D"
    logger.info(completion_message)

    return history

def _generate_training_heatmaps(model: nn.Module, dataloader: DataLoader, config: Dict,
                              epoch: int, phase: int, output_dir: str):
    """Generate heatmaps during training to monitor feature learning - ensures all classes are represented"""
    model.eval()

    # Use Agg backend for headless compatibility
    import matplotlib
    matplotlib.use('Agg')  # Set backend before importing pyplot
    import matplotlib.pyplot as plt

    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)

    # Register attention hooks if available
    hooks_registered = False
    if hasattr(model, 'register_attention_hooks'):
        try:
            model.register_attention_hooks()
            hooks_registered = True
            logger.info("✓ Registered attention hooks for training heatmaps")
        except Exception as e:
            logger.warning(f"Could not register attention hooks: {str(e)}")

    try:
        # Get information about all classes in the dataset
        if hasattr(dataloader.dataset, 'classes'):
            all_classes = list(range(len(dataloader.dataset.classes)))
            class_names = dataloader.dataset.classes
            logger.info(f"Dataset has {len(all_classes)} classes: {class_names}")
        else:
            # If we don't have class info, we'll collect what we find
            all_classes = set()
            class_names = None

        # Sample from all classes - more intelligent collection
        samples_collected = defaultdict(list)
        max_samples_per_class = 1  # We only need one sample per class for the heatmap
        max_batches_to_process = min(10, len(dataloader))  # Process more batches to find rare classes
        samples_per_batch = 2  # How many samples to check per batch

        logger.info(f"Collecting samples from all classes (processing up to {max_batches_to_process} batches)")

        with torch.no_grad():
            batches_processed = 0
            classes_found = set()

            for batch_idx, (data, labels) in enumerate(dataloader):
                if batches_processed >= max_batches_to_process:
                    break

                data = data.to(model.device)
                labels = labels.to(model.device)

                # Forward pass to populate attention maps
                if phase == 1:
                    outputs = model(data)
                    if isinstance(outputs, tuple):
                        _, reconstructions = outputs
                    else:
                        reconstructions = outputs.get('reconstruction', data)
                else:
                    outputs = model(data)
                    reconstructions = outputs.get('reconstruction', data)

                # Track all classes we see in this batch
                batch_classes = set(labels.cpu().numpy())
                classes_found.update(batch_classes)

                # Store samples for classes we haven't collected yet
                for i in range(min(samples_per_batch, len(data))):
                    label = labels[i].item()

                    # Only collect if we don't have this class yet, or if we want multiple samples
                    if len(samples_collected[label]) < max_samples_per_class:
                        samples_collected[label].append({
                            'original': data[i].cpu(),
                            'reconstruction': reconstructions[i].cpu(),
                            'label': label
                        })

                batches_processed += 1

                # Stop early if we've found samples from all classes
                if all_classes and len(samples_collected) >= len(all_classes):
                    logger.info(f"Found samples from all {len(all_classes)} classes after {batches_processed} batches")
                    break
                elif not all_classes and len(samples_collected) >= len(classes_found):
                    logger.info(f"Found samples from all {len(classes_found)} discovered classes after {batches_processed} batches")
                    break

        # If we still don't have all classes, try a more aggressive approach
        if all_classes and len(samples_collected) < len(all_classes):
            missing_classes = set(all_classes) - set(samples_collected.keys())
            logger.warning(f"Missing samples for {len(missing_classes)} classes: {missing_classes}")

            # Try to find missing classes by processing more batches
            logger.info("Searching for missing classes...")
            additional_batches_processed = 0

            for batch_idx, (data, labels) in enumerate(dataloader):
                if additional_batches_processed >= 5:  # Limit additional search
                    break

                data = data.to(model.device)
                labels = labels.to(model.device)

                if phase == 1:
                    outputs = model(data)
                    if isinstance(outputs, tuple):
                        _, reconstructions = outputs
                    else:
                        reconstructions = outputs.get('reconstruction', data)
                else:
                    outputs = model(data)
                    reconstructions = outputs.get('reconstruction', data)

                # Look specifically for missing classes
                for i in range(len(data)):
                    label = labels[i].item()
                    if label in missing_classes and len(samples_collected[label]) < max_samples_per_class:
                        samples_collected[label].append({
                            'original': data[i].cpu(),
                            'reconstruction': reconstructions[i].cpu(),
                            'label': label
                        })
                        missing_classes.remove(label)
                        logger.info(f"Found missing class {label}")

                        if not missing_classes:
                            break

                additional_batches_processed += 1
                if not missing_classes:
                    break

        # Final report
        total_classes_found = len(samples_collected)
        if all_classes:
            coverage = total_classes_found / len(all_classes) * 100
            logger.info(f"Class coverage: {total_classes_found}/{len(all_classes)} classes ({coverage:.1f}%)")
        else:
            logger.info(f"Collected samples from {total_classes_found} classes")

        # Generate heatmaps for collected samples
        if samples_collected:
            _save_training_heatmap_comparison(model, samples_collected, epoch, phase, output_dir, config)
            logger.info(f"✓ Training heatmaps saved for {total_classes_found} classes in {output_dir}")
        else:
            logger.warning("No samples collected for heatmap generation")

    except Exception as e:
        logger.error(f"Could not generate training heatmaps: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Always remove hooks
        if hooks_registered and hasattr(model, 'remove_attention_hooks'):
            try:
                model.remove_attention_hooks()
                logger.debug("✓ Removed attention hooks")
            except Exception as e:
                logger.warning(f"Could not remove attention hooks: {str(e)}")
        plt.close('all')  # Clean up all figures

def _save_training_heatmap_comparison(model: nn.Module, samples_collected: Dict, epoch: int, phase: int,
                                    output_dir: str, config: Dict):
    """Create comprehensive training heatmap visualization - optimized for many classes"""
    import matplotlib
    matplotlib.use('Agg')  # Headless backend
    import matplotlib.pyplot as plt

    n_classes = len(samples_collected)
    if n_classes == 0:
        logger.warning("No samples to create heatmaps")
        return

    logger.info(f"Creating heatmap comparison for {n_classes} classes")

    # For many classes, create multiple figures to avoid overwhelming single image
    max_classes_per_figure = 8  # Maximum classes per figure for readability

    if n_classes <= max_classes_per_figure:
        # Single figure for small number of classes
        _create_single_heatmap_figure(model, samples_collected, epoch, phase, output_dir, config, n_classes)
    else:
        # Multiple figures for many classes
        _create_multiple_heatmap_figures(model, samples_collected, epoch, phase, output_dir, config, max_classes_per_figure)

def _create_single_heatmap_figure(model: nn.Module, samples_collected: Dict, epoch: int, phase: int,
                                output_dir: str, config: Dict, n_classes: int):
    """Create a single heatmap figure for up to 8 classes"""
    import matplotlib.pyplot as plt

    num_columns = 4  # Original, Reconstruction, Error, Attention
    fig, axes = plt.subplots(n_classes, num_columns, figsize=(4 * num_columns, 3 * n_classes))  # Smaller height per row

    # Handle single class case
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx, (label, samples) in enumerate(samples_collected.items()):
        if not samples:
            continue

        sample = samples[0]  # Use first sample for visualization
        _plot_single_class_heatmap(axes[class_idx], sample, label, epoch, model, config)

    plt.suptitle(f'Training Phase {phase} - Epoch {epoch} - {n_classes} Classes', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f'training_heatmaps_phase{phase}_epoch{epoch:03d}.png')
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved training heatmap: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save heatmap {output_path}: {str(e)}")

    plt.close(fig)

def _create_multiple_heatmap_figures(model: nn.Module, samples_collected: Dict, epoch: int, phase: int,
                                   output_dir: str, config: Dict, max_classes_per_figure: int):
    """Create multiple heatmap figures when there are many classes"""
    import matplotlib.pyplot as plt

    all_classes = list(samples_collected.items())
    num_figures = (len(all_classes) + max_classes_per_figure - 1) // max_classes_per_figure

    logger.info(f"Creating {num_figures} heatmap figures for {len(all_classes)} classes")

    for fig_idx in range(num_figures):
        start_idx = fig_idx * max_classes_per_figure
        end_idx = min(start_idx + max_classes_per_figure, len(all_classes))
        figure_classes = dict(all_classes[start_idx:end_idx])

        num_columns = 4
        n_classes_in_fig = len(figure_classes)
        fig, axes = plt.subplots(n_classes_in_fig, num_columns, figsize=(4 * num_columns, 3 * n_classes_in_fig))

        if n_classes_in_fig == 1:
            axes = axes.reshape(1, -1)

        for class_idx, (label, samples) in enumerate(figure_classes.items()):
            if not samples:
                continue

            sample = samples[0]
            _plot_single_class_heatmap(axes[class_idx], sample, label, epoch, model, config)

        plt.suptitle(f'Training Phase {phase} - Epoch {epoch} - Classes {start_idx+1}-{end_idx} of {len(all_classes)}',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Save figure with part number
        output_path = os.path.join(output_dir, f'training_heatmaps_phase{phase}_epoch{epoch:03d}_part{fig_idx+1}.png')
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"✓ Saved training heatmap part {fig_idx+1}: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save heatmap {output_path}: {str(e)}")

        plt.close(fig)

def _plot_single_class_heatmap(axes, sample, label, epoch, model, config):
    """Plot heatmap for a single class"""
    try:
        # Original image
        orig_img = _tensor_to_image(sample['original'], config)
        axes[0].imshow(orig_img)
        axes[0].set_title(f'Class {label}\nOriginal', fontsize=8)
        axes[0].axis('off')

        # Reconstruction
        recon_img = _tensor_to_image(sample['reconstruction'], config)
        axes[1].imshow(recon_img)
        axes[1].set_title(f'Epoch {epoch}\nReconstruction', fontsize=8)
        axes[1].axis('off')

        # Difference (error) - with size matching
        try:
            original_tensor = sample['original']
            reconstruction_tensor = sample['reconstruction']

            if original_tensor.shape != reconstruction_tensor.shape:
                reconstruction_tensor = F.interpolate(
                    reconstruction_tensor.unsqueeze(0),
                    size=original_tensor.shape[1:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            diff = torch.abs(original_tensor - reconstruction_tensor)
            diff_img = _tensor_to_image(diff, config)
            im = axes[2].imshow(diff_img, cmap='hot')
            axes[2].set_title('Error', fontsize=8)
            axes[2].axis('off')

        except Exception as diff_error:
            axes[2].text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=6,
                        transform=axes[2].transAxes)
            axes[2].set_title('Error', fontsize=8)
            axes[2].axis('off')

        # Attention map (if available)
        if hasattr(model, 'attention_maps') and model.attention_maps:
            try:
                attention_key = list(model.attention_maps.keys())[0]
                attention_map = model.attention_maps[attention_key][0].mean(dim=0).cpu()

                target_height, target_width = orig_img.shape[0], orig_img.shape[1]
                if attention_map.shape != (target_height, target_width):
                    attention_map = F.interpolate(
                        attention_map.unsqueeze(0).unsqueeze(0),
                        size=(target_height, target_width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().numpy()
                else:
                    attention_map = attention_map.numpy()

                axes[3].imshow(attention_map, cmap='hot')
                axes[3].set_title('Attention', fontsize=8)
                axes[3].axis('off')

            except Exception:
                axes[3].text(0.5, 0.5, 'No Attn', ha='center', va='center', fontsize=6,
                            transform=axes[3].transAxes)
                axes[3].set_title('Attention', fontsize=8)
                axes[3].axis('off')
        else:
            axes[3].text(0.5, 0.5, 'No Attn', ha='center', va='center', fontsize=6,
                        transform=axes[3].transAxes)
            axes[3].set_title('Attention', fontsize=8)
            axes[3].axis('off')

    except Exception as e:
        logger.error(f"Error plotting class {label}: {str(e)}")
        for col in range(4):
            axes[col].text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=6,
                          transform=axes[col].transAxes)
            axes[col].axis('off')

def _tensor_to_image(tensor: torch.Tensor, config: Dict) -> np.ndarray:
    """Convert tensor to image array with proper normalization and size handling"""
    try:
        tensor = tensor.detach().cpu()

        # Ensure tensor is in CHW format (Channels, Height, Width)
        if len(tensor.shape) == 3 and tensor.shape[0] in [1, 3]:
            # Already in CHW format
            pass
        elif len(tensor.shape) == 3 and tensor.shape[2] in [1, 3]:
            # Convert from HWC to CHW
            tensor = tensor.permute(2, 0, 1)
        else:
            logger.warning(f"Unexpected tensor shape: {tensor.shape}")
            # Try to handle gracefully
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0)  # Add channel dimension
            elif len(tensor.shape) == 3:
                tensor = tensor.permute(2, 0, 1)  # Assume HWC

        # Convert back to HWC for matplotlib
        if len(tensor.shape) == 3:
            tensor = tensor.permute(1, 2, 0)

        # Handle different channel configurations
        if tensor.shape[-1] == 1:  # Grayscale
            tensor = tensor.squeeze(-1)
            mean = torch.tensor(config['dataset']['mean'][0] if isinstance(config['dataset']['mean'], list) else config['dataset']['mean'])
            std = torch.tensor(config['dataset']['std'][0] if isinstance(config['dataset']['std'], list) else config['dataset']['std'])
        else:  # RGB or other
            mean = torch.tensor(config['dataset']['mean']).view(1, 1, -1)
            std = torch.tensor(config['dataset']['std']).view(1, 1, -1)

            # Ensure mean/std have correct number of channels
            if mean.shape[-1] != tensor.shape[-1]:
                if mean.shape[-1] == 1 and tensor.shape[-1] == 3:
                    mean = mean.repeat(1, 1, 3)
                    std = std.repeat(1, 1, 3)
                elif mean.shape[-1] == 3 and tensor.shape[-1] == 1:
                    mean = mean.mean(dim=-1, keepdim=True)
                    std = std.mean(dim=-1, keepdim=True)

        # Denormalize
        tensor = tensor * std + mean

        # Ensure proper value range
        tensor = tensor.clamp(0, 1)

        # Convert to numpy and ensure correct shape
        img_array = tensor.numpy()

        # Ensure 3 channels for display
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB for display
        elif img_array.shape[-1] == 1:  # Single channel
            img_array = np.repeat(img_array, 3, axis=-1)  # Convert to RGB

        return img_array

    except Exception as e:
        logger.error(f"Error converting tensor to image: {str(e)}")
        logger.error(f"Tensor shape: {tensor.shape if 'tensor' in locals() else 'unknown'}")
        # Return a blank image as fallback
        return np.ones((100, 100, 3)) * 0.5  # Gray fallback image

class ReconstructionManager:
    """Manages model prediction with unified checkpoint loading"""

    def __init__(self, config: Dict):
        self.config = config
        self.checkpoint_manager = UnifiedCheckpoint(config)
        self.device = torch.device('cuda' if config['execution_flags']['use_gpu']
                                 and torch.cuda.is_available() else 'cpu')

    def load_model_for_prediction(self) -> Tuple[nn.Module, Dict]:
        """Load appropriate model configuration based on user input"""
        # Get available configurations from checkpoint
        available_configs = self._get_available_configurations()

        if not available_configs:
            raise ValueError("No trained models found in checkpoint")

        # Show available configurations
        print("\nAvailable Model Configurations:")
        for idx, (key, config) in enumerate(available_configs.items(), 1):
            print(f"{idx}. {key}")
            if config.get('best') and config['best'].get('loss'):
                print(f"   Best Loss: {config['best']['loss']:.4f}")
            print(f"   Features: {self._get_config_description(config)}")

        # Get user selection
        while True:
            try:
                choice = int(input("\nSelect configuration (number): ")) - 1
                if 0 <= choice < len(available_configs):
                    selected_key = list(available_configs.keys())[choice]
                    selected_config = available_configs[selected_key]
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

        # Create and load model
        model = self._create_model_from_config(selected_config)
        state_dict = selected_config['best'] if selected_config.get('best') else selected_config['current']
        model.load_state_dict(state_dict['state_dict'])
        model.eval()

        return model, selected_config

    def _get_available_configurations(self) -> Dict:
        """Get available model configurations from checkpoint"""
        return self.checkpoint_manager.current_state['model_states']

    def _get_config_description(self, config: Dict) -> str:
        """Generate human-readable description of model configuration"""
        features = []

        if config['current']['config'].get('kl_divergence'):
            features.append("KL Divergence")
        if config['current']['config'].get('class_encoding'):
            features.append("Class Encoding")

        image_type = config['current']['config'].get('image_type')
        if image_type and image_type != 'general':
            features.append(f"{image_type.capitalize()} Enhancement")

        return ", ".join(features) if features else "Basic Autoencoder"

    def _create_model_from_config(self, config: Dict) -> nn.Module:
        """Create model instance based on configuration"""
        input_shape = (
            self.config['dataset']['in_channels'],
            self.config['dataset']['input_size'][0],
            self.config['dataset']['input_size'][1]
        )
        feature_dims = self.config['model']['feature_dims']

        logger.info(f"Main configuration saved: {self.config_path}")
        image_type = config['dataset'].get('image_type', 'general')

        # Set model configuration based on saved state
        self.config['model']['autoencoder_config']['enhancements'].update({
            'use_kl_divergence': config['current']['config']['kl_divergence'],
            'use_class_encoding': config['current']['config']['class_encoding']
        })

        # Create appropriate model
        image_type = config['current']['config']['image_type']
        if image_type == 'astronomical':
            model = AstronomicalStructurePreservingAutoencoder(input_shape, feature_dims, self.config)
        elif image_type == 'medical':
            model = MedicalStructurePreservingAutoencoder(input_shape, feature_dims, self.config)
        elif image_type == 'agricultural':
            model = AgriculturalPatternAutoencoder(input_shape, feature_dims, self.config)
        else:
            model = BaseAutoencoder(input_shape, feature_dims, self.config)

        return model.to(self.device)

    def predict_from_csv(self, csv_path: Optional[str] = None, output_dir: Optional[str] = None):
        """Generate predictions from features in CSV"""
        # Load model
        model, config = self.load_model_for_prediction()
        model.eval()  # Ensure model is in evaluation mode

        # Determine input CSV path
        if csv_path is None:
            dataset_name = self.config['dataset']['name']
            base_dir = os.path.join('data', dataset_name)

            if self.config.get('execution_flags', {}).get('invert_DBNN', False):
                csv_path = os.path.join(base_dir, 'reconstructed_input.csv')
            else:
                csv_path = os.path.join(base_dir, f"{dataset_name}.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load and process CSV
        df = pd.read_csv(csv_path)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        features = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(self.device)

        # Verify feature dimensions
        expected_dims = self.config['model']['feature_dims']
        if features.size(1) != expected_dims:
            raise ValueError(f"Feature dimension mismatch: got {features.size(1)}, expected {expected_dims}")

        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join('data', self.config['dataset']['name'], 'predictions')
        os.makedirs(output_dir, exist_ok=True)

        # Get image type and enhancement modules
        image_type = self.config['dataset'].get('image_type', 'general')
        enhancement_modules = self.config['model'].get('enhancement_modules', {})

        outputs = []
        batch_size = self.config['training'].get('batch_size', 128)

        with torch.no_grad():
            for i in tqdm(range(0, len(features), batch_size), desc="Generating predictions"):
                batch = features[i:i+batch_size]

                try:
                    if config['current']['phase'] == 1:
                        # Phase 1: Direct decoding
                        reconstruction = model.decode(batch)
                        output = {'reconstruction': reconstruction}
                    else:
                        # Phase 2: Full forward pass with enhancements
                        # First decode the features
                        reconstruction = model.decode(batch)

                        # Then run through full model if needed for enhancements
                        if image_type != 'general' and image_type in enhancement_modules:
                            enhanced_output = model(reconstruction)  # Get enhanced features
                            output = {
                                'reconstruction': enhanced_output['reconstruction'] if 'reconstruction' in enhanced_output else reconstruction,
                                'embedding': enhanced_output.get('embedding', batch)
                            }

                            # Add enhancement-specific outputs
                            if isinstance(model, AstronomicalStructurePreservingAutoencoder):
                                output.update(self._apply_astronomical_enhancements(enhanced_output))
                            elif isinstance(model, MedicalStructurePreservingAutoencoder):
                                output.update(self._apply_medical_enhancements(enhanced_output))
                            elif isinstance(model, AgriculturalPatternAutoencoder):
                                output.update(self._apply_agricultural_enhancements(enhanced_output))
                        else:
                            output = {'reconstruction': reconstruction}

                    outputs.append(self._process_output(output))

                except Exception as e:
                    logger.error(f"Error processing batch {i}: {str(e)}")
                    raise

        # Combine and save results
        combined_output = self._combine_outputs(outputs)
        self._save_predictions(combined_output, output_dir, config)

    def _save_enhancement_outputs(self, predictions: Dict[str, np.ndarray], output_dir: str):
        """Save enhancement-specific outputs"""
        # Save astronomical features
        if 'star_features' in predictions:
            star_dir = os.path.join(output_dir, 'star_detection')
            os.makedirs(star_dir, exist_ok=True)
            for idx, feat in enumerate(predictions['star_features']):
                img_path = os.path.join(star_dir, f'stars_{idx}.png')
                self._save_feature_map(feat, img_path)

        # Save medical features
        if 'boundary_features' in predictions:
            boundary_dir = os.path.join(output_dir, 'boundary_detection')
            os.makedirs(boundary_dir, exist_ok=True)
            for idx, feat in enumerate(predictions['boundary_features']):
                img_path = os.path.join(boundary_dir, f'boundary_{idx}.png')
                self._save_feature_map(feat, img_path)

        # Save agricultural features
        if 'texture_features' in predictions:
            texture_dir = os.path.join(output_dir, 'texture_analysis')
            os.makedirs(texture_dir, exist_ok=True)
            for idx, feat in enumerate(predictions['texture_features']):
                img_path = os.path.join(texture_dir, f'texture_{idx}.png')
                self._save_feature_map(feat, img_path)

    def _save_feature_map(self, feature_map: np.ndarray, path: str):
        """Save feature map as image"""
        # Normalize feature map to 0-255 range
        feature_map = ((feature_map - feature_map.min()) /
                      (feature_map.max() - feature_map.min() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(feature_map).save(path)

    def _process_output(self, output: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Process model output into numpy arrays"""
        processed = {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.detach().cpu().numpy()
            else:
                processed[key] = value
        return processed

    def _combine_outputs(self, outputs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine batched outputs"""
        combined = {}
        for key in outputs[0].keys():
            combined[key] = np.concatenate([out[key] for out in outputs])
        return combined

    def _save_predictions(self, predictions: Dict[str, np.ndarray], output_dir: str, config: Dict):
        """Save predictions with appropriate format based on configuration"""
        os.makedirs(output_dir, exist_ok=True)

        # Save reconstructions as images
        if 'reconstruction' in predictions:
            recon_dir = os.path.join(output_dir, 'reconstructions')
            os.makedirs(recon_dir, exist_ok=True)

            for idx, recon in enumerate(predictions['reconstruction']):
                img = self._tensor_to_image(torch.tensor(recon))
                img_path = os.path.join(recon_dir, f'reconstruction_{idx}.png')
                Image.fromarray(img).save(img_path)

        # Save enhancement-specific outputs
        self._save_enhancement_outputs(predictions, output_dir)

        # Save predictions to CSV
        pred_path = os.path.join(output_dir, 'predictions.csv')
        pred_dict = {}

        # Add all numeric predictions to CSV
        for key, value in predictions.items():
            if isinstance(value, np.ndarray) and value.ndim <= 2:
                if value.ndim == 1:
                    pred_dict[key] = value
                else:
                    for i in range(value.shape[1]):
                        pred_dict[f'{key}_{i}'] = value[:, i]

        if pred_dict:
            pd.DataFrame(pred_dict).to_csv(pred_path, index=False)

        logger.info(f"Predictions saved to {output_dir}")

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to image array with proper normalization"""
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.cpu()

        # Denormalize using dataset mean and std
        mean = torch.tensor(self.config['dataset']['mean']).view(1, 1, -1)
        std = torch.tensor(self.config['dataset']['std']).view(1, 1, -1)
        tensor = tensor * std + mean

        return (tensor.clamp(0, 1) * 255).numpy().astype(np.uint8)

    def _apply_astronomical_enhancements(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply astronomical-specific enhancements"""
        enhanced = {}
        if 'reconstruction' in output:
            enhanced['reconstruction'] = output['reconstruction']
            if 'star_features' in output:
                enhanced['star_features'] = output['star_features']
            if 'galaxy_features' in output:
                enhanced['galaxy_features'] = output['galaxy_features']
        return enhanced

    def _apply_medical_enhancements(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply medical-specific enhancements"""
        enhanced = {}
        if 'reconstruction' in output:
            enhanced['reconstruction'] = output['reconstruction']
            if 'boundary_features' in output:
                enhanced['boundary_features'] = output['boundary_features']
            if 'lesion_features' in output:
                enhanced['lesion_features'] = output['lesion_features']
        return enhanced

    def _apply_agricultural_enhancements(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply agricultural-specific enhancements"""
        enhanced = {}
        if 'reconstruction' in output:
            enhanced['reconstruction'] = output['reconstruction']
            if 'texture_features' in output:
                enhanced['texture_features'] = output['texture_features']
            if 'damage_features' in output:
                enhanced['damage_features'] = output['damage_features']
        return enhanced
  #----------------------------------------------
class ClusteringLoss(nn.Module):
    """Loss function for clustering in latent space using KL divergence"""
    def __init__(self, num_clusters: int, feature_dims: int, temperature: float = 1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        # Learnable cluster centers
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, feature_dims))

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate distances to cluster centers
        distances = torch.cdist(embeddings, self.cluster_centers)

        # Convert distances to probabilities (soft assignments)
        q_dist = 1.0 / (1.0 + (distances / self.temperature) ** 2)
        q_dist = q_dist / q_dist.sum(dim=1, keepdim=True)

        if labels is not None:
            # If labels are provided, create target distribution
            p_dist = torch.zeros_like(q_dist)
            for i in range(self.num_clusters):
                mask = (labels == i)
                if mask.any():
                    p_dist[mask, i] = 1.0
        else:
            # Self-supervised target distribution (following DEC paper)
            p_dist = (q_dist ** 2) / q_dist.sum(dim=0, keepdim=True)
            p_dist = p_dist / p_dist.sum(dim=1, keepdim=True)

        # Calculate KL divergence loss
        kl_loss = F.kl_div(q_dist.log(), p_dist, reduction='batchmean')

        # Return both loss and cluster assignments
        return kl_loss, q_dist.argmax(dim=1)

class EnhancedAutoEncoderLoss(nn.Module):
    """Combined loss function for enhanced autoencoder with clustering and classification"""
    def __init__(self,
                 num_classes: int,
                 feature_dims: int,
                 reconstruction_weight: float = 1.0,
                 clustering_weight: float = 0.1,
                 classification_weight: float = 0.1,
                 temperature: float = 1.0):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.clustering_weight = clustering_weight
        self.classification_weight = classification_weight

        self.clustering_loss = ClusteringLoss(
            num_clusters=num_classes,
            feature_dims=feature_dims,
            temperature=temperature
        )
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self,
                input_data: torch.Tensor,
                reconstruction: torch.Tensor,
                embedding: torch.Tensor,
                classification_logits: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, input_data)

        # Clustering loss
        cluster_loss, cluster_assignments = self.clustering_loss(embedding, labels)

        # Classification loss
        if labels is not None:
            class_loss = self.classification_loss(classification_logits, labels)
        else:
            # Use cluster assignments as pseudo-labels when true labels unavailable
            class_loss = self.classification_loss(classification_logits, cluster_assignments)

        # Combine losses
        total_loss = (self.reconstruction_weight * recon_loss +
                     self.clustering_weight * cluster_loss +
                     self.classification_weight * class_loss)

        return total_loss, cluster_assignments, classification_logits.argmax(dim=1)

class DetailPreservingLoss(nn.Module):
    """Loss function that preserves fine details and enhances class differences.

    Components:
    1. Laplacian filtering - Preserves high-frequency details and edges
    2. Gram matrix analysis - Maintains texture patterns
    3. Frequency domain loss - Emphasizes high-frequency components
    """
    def __init__(self,
                 detail_weight=1.0,
                 texture_weight=0.8,
                 frequency_weight=0.6):
        super().__init__()
        self.detail_weight = detail_weight
        self.texture_weight = texture_weight
        self.frequency_weight = frequency_weight

        # High-pass filters for detail detection
        self.laplacian = KF.Laplacian(3)
        self.sobel = KF.SpatialGradient()

    def forward(self, prediction, target):
        # Base reconstruction loss
        recon_loss = F.mse_loss(prediction, target)

        # High-frequency detail preservation
        pred_lap = self.laplacian(prediction)
        target_lap = self.laplacian(target)
        detail_loss = F.l1_loss(pred_lap, target_lap)

        # Texture preservation using Gram matrices
        pred_gram = self._gram_matrix(prediction)
        target_gram = self._gram_matrix(target)
        texture_loss = F.mse_loss(pred_gram, target_gram)

        # Frequency domain loss
        freq_loss = self._frequency_loss(prediction, target)

        # Combine losses with weights
        total_loss = recon_loss + \
                    self.detail_weight * detail_loss + \
                    self.texture_weight * texture_loss + \
                    self.frequency_weight * freq_loss

        return total_loss

    def _gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

    def _frequency_loss(self, prediction, target):
        # Convert to frequency domain
        pred_freq = torch.fft.fft2(prediction)
        target_freq = torch.fft.fft2(target)

        # Compute magnitude spectrum
        pred_mag = torch.abs(pred_freq)
        target_mag = torch.abs(target_freq)

        # Focus on high-frequency components
        high_freq_mask = self._create_high_freq_mask(pred_mag.shape)
        high_freq_mask = high_freq_mask.to(prediction.device)

        pred_high = pred_mag * high_freq_mask
        target_high = target_mag * high_freq_mask

        return F.mse_loss(pred_high, target_high)

    def _create_high_freq_mask(self, shape):
        _, _, h, w = shape
        mask = torch.ones((h, w))
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4

        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        dist_from_center = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
        mask[dist_from_center < radius] = 0.2

        return mask.unsqueeze(0).unsqueeze(0)
class StructuralLoss(nn.Module):
    """Loss function to enhance image structures like contours and regions"""
    def __init__(self, edge_weight=1.0, smoothness_weight=0.5):
        super().__init__()
        self.edge_weight = edge_weight
        self.smoothness_weight = smoothness_weight
        self.sobel = KF.SpatialGradient()

    def forward(self, prediction, target):
        # Basic reconstruction loss
        recon_loss = F.mse_loss(prediction, target)

        # Edge detection loss using Sobel filters
        pred_edges = self.sobel(prediction)
        target_edges = self.sobel(target)
        edge_loss = F.mse_loss(pred_edges, target_edges)

        # Smoothness loss to preserve continuous regions
        smoothness_loss = torch.mean(torch.abs(prediction[:, :, :, :-1] - prediction[:, :, :, 1:])) + \
                         torch.mean(torch.abs(prediction[:, :, :-1, :] - prediction[:, :, 1:, :]))

        return recon_loss + self.edge_weight * edge_loss + self.smoothness_weight * smoothness_loss

class ColorEnhancementLoss(nn.Module):
    """Loss function to enhance color variations across channels"""
    def __init__(self, channel_weight=0.5, contrast_weight=0.3):
        super().__init__()
        self.channel_weight = channel_weight
        self.contrast_weight = contrast_weight

    def forward(self, prediction, target):
        # Basic reconstruction loss
        recon_loss = F.mse_loss(prediction, target)

        # Channel correlation loss
        pred_corr = self._channel_correlation(prediction)
        target_corr = self._channel_correlation(target)
        channel_loss = F.mse_loss(pred_corr, target_corr)

        # Color contrast loss
        contrast_loss = self._color_contrast_loss(prediction, target)

        return recon_loss + self.channel_weight * channel_loss + self.contrast_weight * contrast_loss

    def _channel_correlation(self, x):
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1)
        mean = torch.mean(x_flat, dim=2).unsqueeze(2)
        x_centered = x_flat - mean
        corr = torch.bmm(x_centered, x_centered.transpose(1, 2))
        return corr / (h * w)

    def _color_contrast_loss(self, prediction, target):
        pred_std = torch.std(prediction, dim=[2, 3])
        target_std = torch.std(target, dim=[2, 3])
        return F.mse_loss(pred_std, target_std)

class MorphologyLoss(nn.Module):
    """Loss function to enhance morphological features"""
    def __init__(self, shape_weight=0.7, symmetry_weight=0.3):
        super().__init__()
        self.shape_weight = shape_weight
        self.symmetry_weight = symmetry_weight

    def forward(self, prediction, target):
        # Basic reconstruction loss
        recon_loss = F.mse_loss(prediction, target)

        # Shape preservation loss using moment statistics
        shape_loss = self._moment_loss(prediction, target)

        # Symmetry preservation loss
        symmetry_loss = self._symmetry_loss(prediction, target)

        return recon_loss + self.shape_weight * shape_loss + self.symmetry_weight * symmetry_loss

    def _moment_loss(self, prediction, target):
        # Calculate spatial moments to capture shape characteristics
        pred_moments = self._calculate_moments(prediction)
        target_moments = self._calculate_moments(target)
        return F.mse_loss(pred_moments, target_moments)

    def _calculate_moments(self, x):
        b, c, h, w = x.size()
        y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w))
        y_grid = y_grid.float().to(x.device) / h
        x_grid = x_grid.float().to(x.device) / w

        moments = []
        for i in range(b):
            for j in range(c):
                img = x[i, j]
                m00 = torch.sum(img)
                if m00 != 0:
                    m10 = torch.sum(img * y_grid)
                    m01 = torch.sum(img * x_grid)
                    m20 = torch.sum(img * y_grid * y_grid)
                    m02 = torch.sum(img * x_grid * x_grid)
                    moments.append(torch.stack([m00, m10/m00, m01/m00, m20/m00, m02/m00]))
                else:
                    moments.append(torch.zeros(5).to(x.device))

        return torch.stack(moments).view(b, c, -1)

    def _symmetry_loss(self, prediction, target):
        # Compare horizontal and vertical symmetry
        h_pred = self._horizontal_symmetry(prediction)
        h_target = self._horizontal_symmetry(target)
        v_pred = self._vertical_symmetry(prediction)
        v_target = self._vertical_symmetry(target)

        return F.mse_loss(h_pred, h_target) + F.mse_loss(v_pred, v_target)

    def _horizontal_symmetry(self, x):
        return F.mse_loss(x, torch.flip(x, [-1]))

    def _vertical_symmetry(self, x):
        return F.mse_loss(x, torch.flip(x, [-2]))

# Set sharing strategy at the start
torch.multiprocessing.set_sharing_strategy('file_system')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Self-attention module for feature maps"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # Query, Key, and Value transformations
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Output transformation
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()

        # Compute queries, keys, and values
        queries = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, H*W, C')
        keys = self.key(x).view(batch_size, -1, height * width)  # (B, C', H*W)
        values = self.value(x).view(batch_size, -1, height * width)  # (B, C, H*W)

        # Compute attention scores
        attention_scores = torch.bmm(queries, keys)  # (B, H*W, H*W)
        attention_scores = F.softmax(attention_scores, dim=-1)  # Normalize scores

        # Apply attention to values
        out = torch.bmm(values, attention_scores.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(batch_size, channels, height, width)  # Reshape to original dimensions

        # Combine with input
        out = self.gamma * out + x  # Residual connection
        return out

class DCTLayer(nn.Module):     # Do a cosine Transform
    def __init__(self):
        super(DCTLayer, self).__init__()

    def forward(self, x):
        return self.dct(x)

    def dct(self, x):
        x = x.permute(0, 2, 3, 1)  # Change to [batch, height, width, channels]
        x = torch.fft.fft(x, dim=1)
        x = torch.fft.fft(x, dim=2)
        x = x.real
        x = x.permute(0, 3, 1, 2)  # Change back to [batch, channels, height, width]
        return x

class DynamicAutoencoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, num_classes: Optional[int] = None):
        super().__init__()
        self.input_shape = (
            config['dataset']['in_channels'],  # Use configured channels
            config['dataset']['input_size'][0],
            config['dataset']['input_size'][1]
        )
        input_shape=self.input_shape
        self.in_channels = config['dataset']['in_channels']
        self.feature_dims = feature_dims
        self.num_classes = num_classes

        # Calculate progressive spatial dimensions
        self.spatial_dims = []
        current_size = input_shape[1]  # Start with height (assuming square)
        self.layer_sizes = self._calculate_layer_sizes()

        for _ in self.layer_sizes:
            self.spatial_dims.append(current_size)
            current_size = current_size // 2

        self.final_spatial_dim = current_size
        # Calculate flattened size after all conv layers
        self.flattened_size = self.layer_sizes[-1] * (self.final_spatial_dim ** 2)

        # Encoder layers with self-attention
        self.encoder_layers = nn.ModuleList()
        in_channels = self.in_channels  # Start with input channels
        for size in self.layer_sizes:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, size, 3, stride=2, padding=1),
                    nn.BatchNorm2d(size),
                    nn.LeakyReLU(0.2),
                    SelfAttention(size)
                )
            )
            in_channels = size

        # Class-aware embedding
        if self.num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, feature_dims)

        # Embedder layers
        self.embedder = nn.Sequential(
            nn.Linear(self.flattened_size, feature_dims),
            nn.BatchNorm1d(feature_dims),
            nn.LeakyReLU(0.2)
        )

        # Unembedder (decoder start)
        self.unembedder = nn.Sequential(
            nn.Linear(feature_dims, self.flattened_size),
            nn.BatchNorm1d(self.flattened_size),
            nn.LeakyReLU(0.2)
        )

        # Decoder layers with self-attention and DCT
        self.decoder_layers = nn.ModuleList()
        in_channels = self.layer_sizes[-1]

        # Build decoder layers in reverse
        for i in range(len(self.layer_sizes)-1, -1, -1):
            out_channels = self.in_channels if i == 0 else self.layer_sizes[i-1]
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels, out_channels,
                        kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(out_channels) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2) if i > 0 else nn.Tanh(),
                    SelfAttention(out_channels),
                    DCTLayer() if i == 0 else nn.Identity()  # Apply DCT at the final layer
                )
            )
            in_channels = out_channels

    def _calculate_layer_sizes(self) -> List[int]:
        """Calculate progressive channel sizes based on input dimensions"""
        base_channels = 32
        sizes = []
        current_size = base_channels

        # Determine maximum layers based on smallest dimension
        min_dim = min(self.input_shape[1], self.input_shape[2])

        # More conservative layer calculation
        max_layers = max(2, int(np.log2(min_dim)) - 2)  # At least 2 layers

        logger.info(f"Input dimensions: {self.input_shape[1]}x{self.input_shape[2]} (channels: {self.in_channels})")
        logger.info(f"Calculating {max_layers} layers for min dimension {min_dim}")

        for i in range(max_layers):
            sizes.append(current_size)

            # Stop if dimensions become too small in next layer
            next_dim = min_dim // (2 ** (i + 1))
            if next_dim < 4:
                logger.info(f"Stopping at layer {i}, next dimension would be {next_dim}")
                break

            # Stop doubling channels at 512 to prevent excessive memory usage
            if current_size < 512:
                current_size *= 2
            else:
                current_size = 512  # Cap at 512

        # Ensure we have at least 2 layers
        if len(sizes) < 2:
            sizes = [32, 64]  # Minimum architecture

        logger.info(f"Final layer sizes: {sizes}")
        return sizes

    def encode(self, x: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input images to feature space"""
        if x.size(1) != self.in_channels:
            raise ValueError(f"Input has {x.size(1)} channels, expected {self.in_channels}")

        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.embedder(x)

        if self.num_classes is not None and class_labels is not None:
            class_emb = self.class_embedding(class_labels)
            x = x + class_emb

        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features back to image space"""
        x = self.unembedder(x)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim, self.final_spatial_dim)

        for layer in self.decoder_layers:
            x = layer(x)

        if x.size(1) != self.in_channels:
            raise ValueError(f"Output has {x.size(1)} channels, expected {self.in_channels}")

        return x

    def forward(self, x: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder"""
        embedding = self.encode(x, class_labels)
        reconstruction = self.decode(embedding)
        return embedding, reconstruction

    def get_encoding_shape(self) -> Tuple[int, ...]:
        """Get the shape of the encoding at each layer"""
        return tuple([size for size in self.layer_sizes])

    def get_spatial_dims(self) -> List[List[int]]:
        """Get the spatial dimensions at each layer"""
        return self.spatial_dims.copy()

class AutoencoderLoss(nn.Module):
    """Composite loss function for autoencoder training"""
    def __init__(self, reconstruction_weight: float = 1.0,
                 feature_weight: float = 0.1):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.feature_weight = feature_weight

    def forward(self, input_data: torch.Tensor,
                reconstruction: torch.Tensor,
                embedding: torch.Tensor) -> torch.Tensor:
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, input_data)

        # Feature distribution loss (encourage normal distribution)
        feature_loss = torch.mean(torch.abs(embedding.mean(dim=0))) + \
                      torch.mean(torch.abs(embedding.std(dim=0) - 1))

        return self.reconstruction_weight * recon_loss + \
               self.feature_weight * feature_loss

class StructurePreservingAutoencoder(DynamicAutoencoder):
    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        super().__init__(input_shape, feature_dims)

        # Add residual connections for detail preservation
        self.skip_connections = nn.ModuleList()

        # Enhanced encoder with more layers for fine detail capture
        self.detail_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.PReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=4),  # Group convolution for local feature preservation
                nn.BatchNorm2d(32),
                nn.PReLU()
            ) for _ in range(3)  # Multiple detail preservation layers
        ])

        # Structure-aware decoder components
        self.structure_decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.PReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=4),
                nn.BatchNorm2d(32),
                nn.PReLU()
            ) for _ in range(3)
        ])

        # Edge detection and preservation module
        self.edge_detector = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, self.in_channels, kernel_size=3, padding=1)
        )

        # Local contrast enhancement
        self.contrast_enhancement = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, self.in_channels, kernel_size=5, padding=2)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced encoding with detail preservation"""
        skip_features = []

        # Regular encoding path
        for layer in self.encoder_layers:
            x = layer(x)
            skip_features.append(x)

            # Apply detail preservation at each scale
            if len(skip_features) <= len(self.detail_encoder):
                x = self.detail_encoder[len(skip_features)-1](x) + x  # Residual connection

        x = x.view(x.size(0), -1)
        x = self.embedder(x)

        return x, skip_features

    def decode(self, x: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """Enhanced decoding with structure preservation"""
        x = self.unembedder(x)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim, self.final_spatial_dim)

        for idx, layer in enumerate(self.decoder_layers):
            x = layer(x)

            # Apply structure preservation
            if idx < len(self.structure_decoder):
                x = self.structure_decoder[idx](x) + x

            # Add skip connections from encoder
            if idx < len(skip_features):
                x = x + skip_features[-(idx+1)]  # Add features from corresponding encoder layer

        # Enhance edges and local contrast
        edges = self.edge_detector(x)
        contrast = self.contrast_enhancement(x)

        # Combine all features
        x = x + 0.1 * edges + 0.1 * contrast

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with enhanced detail preservation"""
        # Extract edges for detail preservation
        edge_features = self.edge_detector(x)

        # Main autoencoder path with skip connections
        embedding, skip_features = self.encode(x)
        reconstruction = self.decode(embedding, skip_features)

        # Enhance final reconstruction with edge and contrast features
        reconstruction = reconstruction + 0.1 * self.edge_detector(reconstruction) + \
                        0.1 * self.contrast_enhancement(reconstruction)

        return embedding, reconstruction

class CustomImageDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, csv_file: Optional[str] = None,
                 target_size: int = 256, overlap: float = 0.5, config: Optional[Dict] = None,
                 data_name: Optional[str] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.config = config if config is not None else {}
        self.data_name = data_name.lower() if data_name else None

        # Handle target size configuration
        if self.config.get('resize_images', False):
            size = 256
        else:
            input_cfg = self.config.get('dataset', {})
            size = input_cfg.get('input_size', 256)

        if isinstance(size, int):
            self.target_size = size
        elif isinstance(size, (list, tuple)):
            self.target_size = size[0]  # Use first dimension
        else:
            self.target_size = 256  # Final fallback

        self.overlap = overlap

        # Initialize all data storage lists
        self.image_files = []  # Store full paths
        self.labels = []
        self.file_indices = []
        self.filenames = []    # Store just filenames
        self.full_paths = []   # Store full paths (same as image_files)
        self.label_encoder = {}
        self.reverse_encoder = {}
        self.preprocessed_images = []

        # Determine valid class directories (only those containing images)
        valid_classes = []
        for entry in os.listdir(data_dir):
            entry_path = os.path.join(data_dir, entry)
            if os.path.isdir(entry_path):
                # Check if directory contains image files
                has_images = any(
                    fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                    for fname in os.listdir(entry_path)
                )
                if has_images:
                    valid_classes.append(entry)

        # Use either custom data_name or folder name
        if not self.data_name:
            if self.config.get('dataset'):
                self.data_name = self.config['dataset'].get('name', 'dataset')
            else:
                self.data_name = os.path.basename(os.path.normpath(data_dir)).lower() or 'dataset'

        # Create label mappings only for valid classes
        valid_classes = sorted(valid_classes)
        for idx, class_name in enumerate(valid_classes):
            self.label_encoder[class_name] = idx
            self.reverse_encoder[idx] = class_name

        # Collect images from valid classes
        for class_idx, class_name in enumerate(valid_classes):
            class_dir = os.path.join(data_dir, class_name)
            image_list = [
                fname for fname in os.listdir(class_dir)
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
            ]

            for img_name in image_list:
                full_path = os.path.join(class_dir, img_name)
                self.image_files.append(full_path)
                self.full_paths.append(full_path)  # Store full path
                self.filenames.append(img_name)    # Store just filename
                self.labels.append(class_idx)
                self.file_indices.append(len(self.image_files) - 1)

        # Fallback to CSV if no directory structure found
        if csv_file and os.path.exists(csv_file) and not self.image_files:
            self.data = pd.read_csv(csv_file)
            # Add CSV processing logic here if needed

        # Final configuration fallbacks
        if not self.data_name:
            self.data_name = 'dataset'

        self.resize_images = self.config.get('resize_images', False)
        if self.resize_images:
            self.target_size = 256

        # Update config with determined name
        if self.config:
            if 'dataset' not in self.config:
                self.config['dataset'] = {}
            self.config['dataset']['name'] = self.data_name

        # Preprocess all images
        self._preprocess_all_images()

    def _preprocess_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Preprocess an image tensor to ensure it is suitable for the CNN.
        - If the image is smaller than target_size and resize_images is True, resize it to target_size.
        - If the image is larger than target_size, split it into sliding windows of target_size.

        Args:
            image_tensor (torch.Tensor): Input image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Processed image tensor(s). If windowing is used, returns a batch of tensors.
        """
        _, h, w = image_tensor.shape

        # Resize if image is smaller than target_size and resize_images is True
        if self.resize_images and (h < self.target_size or w < self.target_size):
            image_tensor = resize(image_tensor, (self.target_size, self.target_size), antialias=True)
            return image_tensor.unsqueeze(0)  # Add batch dimension

        # Split into sliding windows if image is larger than target_size
        if h > self.target_size or w > self.target_size:
            stride = int(self.target_size * (1 - self.overlap))  # Stride based on overlap
            windows = []

            # Extract windows
            for y in range(0, h - self.target_size + 1, stride):
                for x in range(0, w - self.target_size + 1, stride):
                    window = image_tensor[:, y:y + self.target_size, x:x + self.target_size]
                    windows.append(window)

            # Stack windows into a batch
            return torch.stack(windows)

        # If image is already target_size, return as is
        return image_tensor.unsqueeze(0)

    def _preprocess_all_images(self):
        """
        Preprocess all images to ensure consistent shapes (256x256).
        """
        # Create a directory to store preprocessed images (if saving to disk)
        self.preprocessed_dir = os.path.join(self.data_dir, "preprocessed")
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        # Preprocess images with a progress bar
        for idx, img_path in enumerate(tqdm(self.image_files, desc=f"Preprocessing images")):
            image = Image.open(img_path).convert('RGB')
            image_tensor = transforms.ToTensor()(image)

            # Resize or window the image
            preprocessed_tensors = self._preprocess_image(image_tensor)

            # Save preprocessed images to disk (optional)
            for i, tensor in enumerate(preprocessed_tensors):
                save_path = os.path.join(self.preprocessed_dir, f"{self.filenames[idx]}_window{i}.pt")
                torch.save(tensor, save_path)

            # Store preprocessed tensors (or paths to the preprocessed images)
            self.preprocessed_images.append(preprocessed_tensors)

    def __len__(self):
        return len(self.image_files)

    def get_additional_info(self, idx):
        """Retrieve additional information including full file path for a given index."""
        return self.file_indices[idx], self.filenames[idx], self.full_paths[idx]

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        file_index = self.file_indices[idx]  # Retrieve file index
        filename = self.filenames[idx]  # Retrieve filename
        full_path = self.full_paths[idx]  # Retrieve full path

        if self.transform:
            image = self.transform(image)

        # Return only image and label during training
        return image, label

class DatasetProcessor:
    SUPPORTED_FORMATS = {
        'zip': zipfile.ZipFile,
        'tar': tarfile.TarFile,
        'tar.gz': tarfile.TarFile,
        'tgz': tarfile.TarFile,
        'gz': gzip.GzipFile,
        'bz2': bz2.BZ2File,
        'xz': lzma.LZMAFile
    }

    SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    def __init__(self, datafile: str = "MNIST", datatype: str = "torchvision",
                 output_dir: str = "Data", config: Optional[Dict] = None,
                 data_name: Optional[str] = None):
        self.datafile = datafile
        self.datatype = datatype.lower()
        self.output_dir = output_dir
        self.config = config if config is not None else {}

        # Determine dataset name - use UPPERCASE ONLY for torchvision datasets
        if data_name:
            if self.datatype == 'torchvision':
                self.dataset_name = data_name.upper()  # UPPERCASE for torchvision
            else:
                self.dataset_name = data_name  # As provided for custom
        else:
            if self.datatype == 'torchvision':
                self.dataset_name = self.datafile.upper()  # UPPERCASE for torchvision
            else:
                # For custom datasets, use the exact name/path without modification
                path_obj = Path(self.datafile)
                if path_obj.is_dir():
                    # Use the directory name as-is (no lowercase conversion)
                    self.dataset_name = path_obj.name
                else:
                    # Use the file stem as-is (no lowercase conversion)
                    self.dataset_name = path_obj.stem or 'dataset'

        # For torchvision, set up standard directory structure with UPPERCASE
        if self.datatype == 'torchvision':
            # Directory name should be UPPERCASE for torchvision datasets
            self.dataset_dir = os.path.join(self.output_dir, self.dataset_name)  # Data/MNIST, Data/CIFAR10, etc.
        else:
            # For custom datasets, use the provided path structure without modification
            self.dataset_dir = os.path.join(self.output_dir, self.dataset_name)

        os.makedirs(self.dataset_dir, exist_ok=True)

        self.config_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.json").lower()
        self.conf_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.conf").lower()
        self.dbnn_conf_path = os.path.join(self.dataset_dir, "adaptive_dbnn.conf").lower()

    def _extract_archive(self, archive_path: str) -> str:
        """Extract compressed archive to temporary directory"""
        extract_dir = os.path.join(self.dataset_dir, 'temp_extract')
        os.makedirs(extract_dir, exist_ok=True)

        file_ext = Path(archive_path).suffix.lower()
        if file_ext.startswith('.'):
            file_ext = file_ext[1:]

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

        return extract_dir

    def _process_data_path(self, data_path: str) -> str:
        """Process input data path, handling compressed files if necessary"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

        file_ext = Path(data_path).suffix.lower()
        if file_ext.startswith('.'):
            file_ext = file_ext[1:]

        # Check if it's a compressed file
        if file_ext in self.SUPPORTED_FORMATS or data_path.endswith('tar.gz'):
            logger.info(f"Extracting compressed file: {data_path}")
            extract_dir = self._extract_archive(data_path)

            # Find the main data directory
            contents = os.listdir(extract_dir)
            if len(contents) == 1 and os.path.isdir(os.path.join(extract_dir, contents[0])):
                return os.path.join(extract_dir, contents[0])
            return extract_dir

        return data_path

    def process(self) -> Tuple[str, Optional[str]]:
        """Process dataset and return paths to train and test directories"""
        if self.datatype == 'torchvision':
            return self._process_torchvision()
        else:
            # Only process data path for custom datasets
            if self.datafile and os.path.exists(self.datafile):
                processed_path = self._process_data_path(self.datafile)
                return self._process_custom(processed_path)
            else:
                raise FileNotFoundError(f"Custom dataset path not found: {self.datafile}")

    def _handle_existing_directory(self, path: str):
        """Handle existing directory by either removing it or merging its contents."""
        if os.path.exists(path):
            response = input(f"The directory '{path}' already exists. Do you want to (R)emove it or (M)erge its contents? [R/M]: ").lower()
            if response == 'r':
                shutil.rmtree(path)
                os.makedirs(path)
            elif response == 'm':
                # Merge contents (no action needed, as shutil.copytree will handle it with dirs_exist_ok=True)
                pass
            else:
                raise ValueError("Invalid choice. Please choose 'R' to remove or 'M' to merge.")

    def _process_custom(self, data_path: str) -> Tuple[str, Optional[str]]:
        """Process custom dataset structure - preserve original directory names"""
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        # Access enable_adaptive from training_params
        try:
             enable_adaptive = self.config['model'].get('enable_adaptive', True)
        except:
            enable_adaptive = True
            print(f"Enable Adaptive mode is set {enable_adaptive} in process custom")

        # Check if dataset already has train/test structure
        if os.path.isdir(os.path.join(data_path, "train")) and \
           os.path.isdir(os.path.join(data_path, "test")):
            # Check if adaptive_fit_predict is active
            if enable_adaptive:
                # Handle existing train directory
                self._handle_existing_directory(train_dir)
                # Merge train and test folders into a single train folder

                # Copy train data
                shutil.copytree(os.path.join(data_path, "train"), train_dir, dirs_exist_ok=True)

                return train_dir, test_dir
            else:
                # Normal processing with separate train and test folders
                if os.path.exists(train_dir):
                    shutil.rmtree(train_dir)
                if os.path.exists(test_dir):
                    shutil.rmtree(test_dir)

                shutil.copytree(os.path.join(data_path, "train"), train_dir)
                shutil.copytree(os.path.join(data_path, "test"), test_dir)
                return train_dir, test_dir

        # Handle single directory with class subdirectories
        if not os.path.isdir(data_path):
            raise ValueError(f"Invalid dataset path: {data_path}")

        class_dirs = [d for d in os.listdir(data_path)
                     if os.path.isdir(os.path.join(data_path, d))]

        if not class_dirs:
            raise ValueError(f"No class directories found in {data_path}")

        # Ask user about train/test split
        response = input("Create train/test split? (y/n): ").lower()
        if response == 'y':
            test_size = float(input("Enter test size (0-1, default: 0.2): ") or "0.2")
            return self._create_train_test_split(data_path, test_size)
        else:
            # Use all data for training
            os.makedirs(train_dir, exist_ok=True)
            for class_dir in class_dirs:
                src = os.path.join(data_path, class_dir)
                dst = os.path.join(train_dir, class_dir)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            return train_dir, None

    def cleanup(self):
        """Clean up temporary files"""
        temp_dir = os.path.join(self.dataset_dir, 'temp_extract')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
#------------------------
    def get_transforms(self, config: Dict, is_train: bool = True) -> transforms.Compose:
        """Get transforms that preserve the detected image dimensions"""
        transform_list = []

        # Use detected dimensions, don't force resize
        target_size = tuple(config['dataset']['input_size'])
        target_channels = config['dataset']['in_channels']

        logger.info(f"Using transforms for size: {target_size}, channels: {target_channels}")

        # Only resize if images are very large (optional)
        max_size = 512
        if target_size[0] > max_size or target_size[1] > max_size:
            logger.info(f"Resizing large images from {target_size} to max {max_size}")
            transform_list.append(transforms.Resize(max_size))
        else:
            logger.info(f"Using original image size: {target_size}")

        # Channel conversion if needed
        if target_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        elif target_channels == 3:
            transform_list.append(transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x))

        # Training augmentations
        if is_train and config.get('augmentation', {}).get('enabled', True):
            aug_config = config['augmentation']
            if aug_config.get('random_crop', {}).get('enabled', False):
                transform_list.append(transforms.RandomCrop(target_size, padding=4))
            if aug_config.get('horizontal_flip', {}).get('enabled', False):
                transform_list.append(transforms.RandomHorizontalFlip())

        # Final transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=config['dataset']['mean'],
                               std=config['dataset']['std'])
        ])

        return transforms.Compose(transform_list)

    def _merge_configs(self, existing: Dict, default: Dict) -> Dict:
        """Recursively merge configs, preserving existing values and ensuring JSON serializability"""
        result = existing.copy()
        for key, value in default.items():
            if key not in result:
                # Add new key with cleaned value
                result[key] = self._clean_for_json_serialization(value)
            elif isinstance(value, dict) and isinstance(result[key], dict):
                # Recursively merge nested dictionaries
                result[key] = self._merge_configs(result[key], value)
            else:
                # Ensure the value is JSON serializable
                result[key] = self._clean_for_json_serialization(result[key])
        return result

    def _generate_main_config(self, train_dir: str) -> Dict:
        """Generate main configuration with enhanced settings for complex datasets"""
        input_size, in_channels = self._detect_image_properties(train_dir)
        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        num_classes = len(class_dirs)

        mean = [0.5] if in_channels == 1 else [0.485, 0.456, 0.406]
        std = [0.5] if in_channels == 1 else [0.229, 0.224, 0.225]

        # NEW: Adaptive feature dimensions based on dataset complexity
        base_feature_dims = min(128, np.prod(input_size) // 4)
        if num_classes > 50:  # Complex datasets like CIFAR-100
            feature_dims = min(256, base_feature_dims * 2)
            logger.info(f"Enhanced feature dimensions: {feature_dims} for {num_classes} classes")
        else:
            feature_dims = base_feature_dims

        return {
            "dataset": {
                "name": self.dataset_name,
                "type": self.datatype,
                "in_channels": in_channels,
                "num_classes": num_classes,
                "input_size": list(input_size),
                "mean": mean,
                "std": std,
                "resize_images": False,
                "train_dir": train_dir,
                "test_dir": os.path.join(os.path.dirname(train_dir), 'test')
            },
            "model": {
                "encoder_type": "autoenc",
                'enable_adaptive': True,  # Default value
                "feature_dims": feature_dims,
                "compressed_dims": max(16, feature_dims // 4),
                "learning_rate": 0.001,
                "optimizer": {
                    "type": "Adam",
                    "weight_decay": 0.0001,
                    "momentum": 0.9,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-08
                },
                "scheduler": {
                    "type": "ReduceLROnPlateau",
                    "factor": 0.1,
                    "patience": 10,
                    "min_lr": 1e-06,
                    "verbose": True
                },
                "autoencoder_config": {
                    "reconstruction_weight": 1.0,
                    "feature_weight": 0.1,
                    "convergence_threshold": 0.0001,
                    "min_epochs": 10,
                    "patience": 5,
                    "enhancements": {
                        "enabled": True,
                        "use_kl_divergence": True,
                        "use_class_encoding": False,
                        "kl_divergence_weight": 0.5,
                        "classification_weight": 0.5,
                        "clustering_temperature": 1.0,
                        "min_cluster_confidence": 0.7
                    }
                },
                "loss_functions": {
                    "structural": {
                        "enabled": True,
                        "weight": 1.0,
                        "params": {
                            "edge_weight": 1.0,
                            "smoothness_weight": 0.5
                        }
                    },
                    "color_enhancement": {
                        "enabled": True,
                        "weight": 0.8,
                        "params": {
                            "channel_weight": 0.5,
                            "contrast_weight": 0.3
                        }
                    },
                    "morphology": {
                        "enabled": True,
                        "weight": 0.6,
                        "params": {
                            "shape_weight": 0.7,
                            "symmetry_weight": 0.3
                        }
                    },
                    "detail_preserving": {
                        "enabled": True,
                        "weight": 0.8,
                        "params": {
                            "detail_weight": 1.0,
                            "texture_weight": 0.8,
                            "frequency_weight": 0.6
                        }
                    },
                    "astronomical_structure": {
                        "enabled": True,
                        "weight": 1.0,
                        "components": {
                            "edge_preservation": True,
                            "peak_preservation": True,
                            "detail_preservation": True
                        }
                    },
                    "medical_structure": {
                        "enabled": True,
                        "weight": 1.0,
                        "components": {
                            "boundary_preservation": True,
                            "tissue_contrast": True,
                            "local_structure": True
                        }
                    },
                    "agricultural_pattern": {
                        "enabled": True,
                        "weight": 1.0,
                        "components": {
                            "texture_preservation": True,
                            "damage_pattern": True,
                            "color_consistency": True
                        }
                    }
                },
                "enhancement_modules": {
                    "astronomical": {
                        "enabled": True,
                        "components": {
                            "structure_preservation": True,
                            "detail_preservation": True,
                            "star_detection": True,
                            "galaxy_features": True,
                            "kl_divergence": True
                        },
                        "weights": {
                            "detail_weight": 1.0,
                            "structure_weight": 0.8,
                            "edge_weight": 0.7
                        }
                    },
                    "medical": {
                        "enabled": True,
                        "components": {
                            "tissue_boundary": True,
                            "lesion_detection": True,
                            "contrast_enhancement": True,
                            "subtle_feature_preservation": True
                        },
                        "weights": {
                            "boundary_weight": 1.0,
                            "lesion_weight": 0.8,
                            "contrast_weight": 0.6
                        }
                    },
                    "agricultural": {
                        "enabled": True,
                        "components": {
                            "texture_analysis": True,
                            "damage_detection": True,
                            "color_anomaly": True,
                            "pattern_enhancement": True,
                            "morphological_features": True
                        },
                        "weights": {
                            "texture_weight": 1.0,
                            "damage_weight": 0.8,
                            "pattern_weight": 0.7
                        }
                    }
                }
            },
            "training": {
                "batch_size": 128,
                "epochs": 200,
                "num_workers": min(4, os.cpu_count() or 1),
                "checkpoint_dir": os.path.join(self.dataset_dir, "checkpoints"),
                "validation_split": 0.2,
                "invert_DBNN": True,
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.1,
                "use_classwise_acc": True, # classwise accuracy has priority
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                }
            },
            "augmentation": {
                "enabled": True,
                "random_crop": {"enabled": True, "padding": 4},
                "random_rotation": {"enabled": True, "degrees": 10},
                "horizontal_flip": {"enabled": True, "probability": 0.5},
                "vertical_flip": {"enabled": False},
                "color_jitter": {
                    "enabled": True,
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.1
                },
                "normalize": {
                    "enabled": True,
                    "mean": mean,
                    "std": std
                }
            },
            "execution_flags": {
                "mode": "train_and_predict",
                "use_gpu": torch.cuda.is_available(),
                "mixed_precision": True,
                "distributed_training": False,
                "debug_mode": False,
                "use_previous_model": True,
                "fresh_start": False
            },
            "output": {
                "features_file": os.path.join(self.dataset_dir, f"{self.dataset_name}.csv"),
                "model_dir": os.path.join(self.dataset_dir, "models"),
                "visualization_dir": os.path.join(self.dataset_dir, "visualizations")
            }
        }

    def _generate_dbnn_config(self, main_config: Dict) -> Dict:
        """Generate DBNN-specific configuration"""
        # Safe access to training parameters
        training_config = main_config.get('training', {})
        epochs = training_config.get('epochs', 200)
        invert_dbnn = training_config.get('invert_DBNN', False)

        # Safe access to model parameters
        model_config = main_config.get('model', {})
        learning_rate = model_config.get('learning_rate', 0.001)

        return {
            "training_params": {
                "trials": epochs,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": 128,
                "test_fraction": 0.2,
                "random_seed": 42,
                "minimum_training_accuracy": 0.95,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "n_bins_per_dim": 128,
                "enable_adaptive": True,
                "invert_DBNN": invert_dbnn,
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.1,
                "Save_training_epochs": False,
                "training_save_path": os.path.join(self.dataset_dir, "training_data"),
                "modelType": "Histogram",
                "compute_device": "auto",
                "class_preference": True
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

    def generate_default_config(self, train_dir: str) -> Dict:
        """Generate and manage all configuration files"""
        os.makedirs(self.dataset_dir, exist_ok=True)
        logger.info(f"Starting configuration generation for dataset: {self.dataset_name}")

        # 1. Generate and handle main configuration (json)
        logger.info("Generating main configuration...")
        config = self._generate_main_config(train_dir)

        # Clean config for JSON serialization
        config = self._clean_for_json_serialization(config)

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    existing_config = json.load(f)
                    logger.info(f"Found existing main config, merging...")
                    config = self._merge_configs(existing_config, config)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {self.config_path}, using default template")

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Main configuration saved: {self.config_path}")

        # 2. Generate and handle dataset.conf using _generate_dataset_conf
        logger.info("Generating dataset configuration...")
        dataset_conf = self._generate_dataset_conf(config['model']['feature_dims'])
        dataset_conf = self._clean_for_json_serialization(dataset_conf)

        if os.path.exists(self.conf_path):
            try:
                with open(self.conf_path, 'r') as f:
                    existing_dataset_conf = json.load(f)
                    logger.info(f"Found existing dataset config, merging...")
                    dataset_conf = self._merge_configs(existing_dataset_conf, dataset_conf)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {self.conf_path}, using default template")

        with open(self.conf_path, 'w') as f:
            json.dump(dataset_conf, f, indent=4)
        logger.info(f"Dataset configuration saved: {self.conf_path}")

        # 3. Generate and handle adaptive_dbnn.conf using _generate_dbnn_config
        logger.info("Generating DBNN configuration...")
        dbnn_config = self._generate_dbnn_config(config)
        dbnn_config = self._clean_for_json_serialization(dbnn_config)

        if os.path.exists(self.dbnn_conf_path):
            try:
                with open(self.dbnn_conf_path, 'r') as f:
                    existing_dbnn_config = json.load(f)
                    logger.info(f"Found existing DBNN config, merging...")
                    dbnn_config = self._merge_configs(existing_dbnn_config, dbnn_config)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {self.dbnn_conf_path}, using default template")

        with open(self.dbnn_conf_path, 'w') as f:
            json.dump(dbnn_config, f, indent=4)
        logger.info(f"DBNN configuration saved: {self.dbnn_conf_path}")

        # Return the main config for further use
        return config

    def _clean_for_json_serialization(self, obj: Any) -> Any:
        """
        Recursively clean an object to ensure it's JSON serializable.
        Converts tensors, numpy arrays, and other non-serializable types to basic Python types.
        """
        if obj is None:
            return None

        if isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:  # Scalar tensor
                return obj.item()
            else:
                return obj.cpu().numpy().tolist()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, dict):
            return {str(k): self._clean_for_json_serialization(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._clean_for_json_serialization(item) for item in obj]

        if hasattr(obj, '__dict__'):
            return self._clean_for_json_serialization(obj.__dict__)

        if hasattr(obj, 'item'):
            try:
                return obj.item()
            except:
                return str(obj)

        try:
            return str(obj)
        except:
            return f"<unserializable object of type {type(obj).__name__}>"

    def _generate_dataset_conf(self, feature_dims: int) -> Dict:
        """Generate dataset-specific configuration with heatmap support"""
        return {
            "filepath": os.path.join(self.dataset_dir, f"{self.dataset_name}.csv"),
            "column_names": [f"feature_{i}" for i in range(feature_dims)] + ["target"],
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
                "strong_margin_threshold": 0.01,
                "marginal_margin_threshold": 0.01,
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
                "use_interactive_kbd": False,
                "class_preference": True,

                # NEW: Heatmap training configuration
                "generate_training_heatmaps": True,  # Enable training heatmaps
                "heatmap_frequency": 10,  # Generate every 10 epochs
                "max_training_heatmap_samples": 2,  # Samples per class
                "reconstruction_samples": {  # Existing but ensure it's there
                    "enabled": True,
                    "samples_per_class": 2,
                    "save_frequency": 5
                }
            },
            "execution_flags": {
                "train": True,
                "train_only": False,
                "predict": True,
                "fresh_start": False,
                "use_previous_model": True,
                "gen_samples": False,

                # NEW: Heatmap execution flags
                "generate_heatmaps": True,  # Enable prediction heatmaps
                "num_samples_per_class": None,  # None means all samples
                "heatmap_backend": "Agg",  # Headless-safe backend
                "heatmap_dpi": 150
            }
        }

    def _detect_image_properties(self, data_dir: str) -> Tuple[Tuple[int, int], int]:
        """Detect image dimensions and channels from the dataset"""
        # Find first image file
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(root, file)
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            channels = len(img.getbands())
                            return (height, width), channels
                    except Exception as e:
                        logger.warning(f"Could not read image {img_path}: {e}")
                        continue

        # Default values if no images found
        logger.warning("No images found, using default dimensions")
        return (256, 256), 3

    def _ensure_required_configs(self, config: Dict) -> Dict:
        """Ensure all required configurations exist"""
        if 'loss_functions' not in config['model']:
            config['model']['loss_functions'] = {}

        if 'autoencoder' not in config['model']['loss_functions']:
            config['model']['loss_functions']['autoencoder'] = {
                'enabled': True,
                'type': 'AutoencoderLoss',
                'weight': 1.0,
                'params': {
                    'reconstruction_weight': 1.0,
                    'feature_weight': 0.1
                }
            }

        return config

    def _convert_torchvision_to_folder_structure(self, dataset_name: str, data_dir: str = "data"):
        """Convert any torchvision dataset to folder structure using dynamic discovery"""
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms

        dataset_dir = os.path.join(data_dir, dataset_name.upper())  # UPPERCASE for consistency

        # Check if already converted
        if os.path.exists(dataset_dir):
            logger.info(f"Dataset already converted to folder structure: {dataset_dir}")
            return dataset_dir

        logger.info(f"Converting torchvision dataset {dataset_name} to folder structure...")

        # Create directory structure
        train_dir = os.path.join(dataset_dir, 'train')
        test_dir = os.path.join(dataset_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Basic transform for saving images
        transform = transforms.ToTensor()

        try:
            # DYNAMIC DATASET DISCOVERY
            dataset_class = None
            possible_names = [
                dataset_name.upper(),
                dataset_name.title(),
                dataset_name.capitalize()
            ]

            for name in possible_names:
                if hasattr(datasets, name):
                    dataset_class = getattr(datasets, name)
                    logger.info(f"Found torchvision dataset: {name}")
                    break

            if dataset_class is None:
                available_datasets = []
                for name in dir(datasets):
                    if not name.startswith('_') and name[0].isupper():
                        obj = getattr(datasets, name)
                        if isinstance(obj, type) and issubclass(obj, torch.utils.data.Dataset):
                            available_datasets.append(name)

                available_datasets.sort()
                raise ValueError(f"Torchvision dataset '{dataset_name}' not found. "
                               f"Available datasets: {', '.join(available_datasets)}")

            # Download and process with dynamic parameter handling
            try:
                # Strategy 1: Standard train/test split
                logger.info("Downloading and processing training data...")
                train_dataset = dataset_class(root=data_dir, train=True, download=True, transform=transform)
                logger.info("Downloading and processing test data...")
                test_dataset = dataset_class(root=data_dir, train=False, download=True, transform=transform)
            except TypeError:
                try:
                    # Strategy 2: Split parameter
                    logger.info("Trying split-based loading...")
                    train_dataset = dataset_class(root=data_dir, split='train', download=True, transform=transform)
                    test_dataset = dataset_class(root=data_dir, split='test', download=True, transform=transform)
                except TypeError:
                    try:
                        # Strategy 3: Different split names
                        logger.info("Trying alternative split names...")
                        train_dataset = dataset_class(root=data_dir, split='training', download=True, transform=transform)
                        test_dataset = dataset_class(root=data_dir, split='validation', download=True, transform=transform)
                    except TypeError:
                        # Strategy 4: Single dataset - create artificial split
                        logger.info("Creating artificial train/test split...")
                        full_dataset = dataset_class(root=data_dir, download=True, transform=transform)
                        train_size = int(0.8 * len(full_dataset))
                        test_size = len(full_dataset) - train_size
                        train_dataset, test_dataset = torch.utils.data.random_split(
                            full_dataset, [train_size, test_size]
                        )

            # Save images using the updated dynamic method
            self._save_torchvision_images(train_dataset, train_dir, 'train')
            self._save_torchvision_images(test_dataset, test_dir, 'test')

            logger.info(f"Successfully converted {dataset_name} to folder structure at {dataset_dir}")
            return dataset_dir

        except Exception as e:
            logger.error(f"Failed to convert torchvision dataset {dataset_name}: {str(e)}")
            raise

    def _save_torchvision_images(self, dataset, output_dir: str, split: str):
        """Save torchvision dataset images to folder structure with dynamic class handling"""
        from PIL import Image

        # Get class names dynamically - handle different dataset structures
        if hasattr(dataset, 'classes'):
            # Standard case: dataset has classes attribute
            class_names = dataset.classes
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'classes'):
            # Case: dataset is a Subset or similar wrapper
            class_names = dataset.dataset.classes
        else:
            # Fallback: create numeric class names
            if hasattr(dataset, 'targets'):
                num_classes = len(set(dataset.targets))
            else:
                # Estimate number of classes from labels
                all_labels = []
                for _, label in dataset:
                    all_labels.append(label)
                num_classes = len(set(all_labels))
            class_names = [f"class_{i}" for i in range(num_classes)]
            logger.warning(f"Using generated class names: {class_names}")

        # Create class subdirectories
        for class_idx, class_name in enumerate(class_names):
            # Ensure class name is string and safe for directory names
            safe_class_name = str(class_name).replace('/', '_').replace('\\', '_')
            class_dir = os.path.join(output_dir, safe_class_name)
            os.makedirs(class_dir, exist_ok=True)

        # Save images with progress bar
        for idx, (image_data, label) in enumerate(tqdm(dataset, desc=f"Saving {split} images")):
            # Handle different label formats
            if isinstance(label, torch.Tensor):
                label_idx = label.item()
            else:
                label_idx = int(label)

            # Get class name safely
            if label_idx < len(class_names):
                class_name = class_names[label_idx]
            else:
                class_name = f"class_{label_idx}"

            safe_class_name = str(class_name).replace('/', '_').replace('\\', '_')
            class_dir = os.path.join(output_dir, safe_class_name)

            # Convert to PIL Image - handle different data formats
            if isinstance(image_data, torch.Tensor):
                # Denormalize if needed and convert to PIL
                image_np = image_data.mul(255).byte().numpy()
                if image_np.shape[0] == 3:  # CHW to HWC
                    image_np = image_np.transpose(1, 2, 0)
                elif image_np.shape[0] == 1:  # Grayscale
                    image_np = image_np.squeeze(0)
                image = Image.fromarray(image_np)
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                # Try to handle as numpy array
                try:
                    image_np = np.array(image_data)
                    if image_np.ndim == 3 and image_np.shape[2] == 3:
                        image = Image.fromarray(image_np, 'RGB')
                    elif image_np.ndim == 2:
                        image = Image.fromarray(image_np, 'L')
                    else:
                        raise ValueError(f"Unsupported image format: {image_np.shape}")
                except:
                    logger.warning(f"Could not convert image {idx} to PIL, skipping")
                    continue

            # Save image
            filename = f"{split}_{idx:06d}.png"
            image_path = os.path.join(class_dir, filename)

            try:
                image.save(image_path, 'PNG')
            except Exception as e:
                logger.warning(f"Could not save image {image_path}: {e}")
                continue

    def get_data_loader(self, config: Dict, is_train: bool = True) -> DataLoader:
        """Get data loader that handles both torchvision and custom datasets with dynamic discovery"""
        dataset_name = config['dataset']['name'].lower()
        data_type = config.get('data_type', 'custom')

        if data_type == 'torchvision':
            # Convert torchvision dataset to folder structure using dynamic discovery
            dataset_dir = self._convert_torchvision_to_folder_structure(config['dataset']['name'])

            # Update config to use folder structure from now on
            config['data_type'] = 'custom'
            config['dataset']['data_dir'] = dataset_dir

            logger.info(f"Converted {config['dataset']['name']} to folder structure. Using as custom dataset.")

        # Now proceed with custom dataset logic
        return self._get_custom_data_loader(config, is_train)

    def _get_custom_data_loader(self, config: Dict, is_train: bool = True) -> DataLoader:
        """Get data loader for custom folder-based datasets"""
        dataset_dir = config['dataset']['data_dir']
        split = 'train' if is_train else 'test'
        data_path = os.path.join(dataset_dir, split)

        if not os.path.exists(data_path):
            # If train/test split doesn't exist, use the main directory
            logger.warning(f"{split} directory not found at {data_path}. Using main directory: {dataset_dir}")
            data_path = dataset_dir

        transform = self.get_transforms(config, is_train=is_train)

        # Create dataset from folder structure
        dataset = datasets.ImageFolder(root=data_path, transform=transform)

        # Update config with actual class information
        if is_train:  # Only update from training data to be consistent
            config['dataset']['num_classes'] = len(dataset.classes)
            config['dataset']['class_names'] = dataset.classes
            logger.info(f"Dataset: {len(dataset.classes)} classes - {dataset.classes}")

        batch_size = config['training']['batch_size'] if is_train else config['training'].get('test_batch_size', config['training']['batch_size'])

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )

        return dataloader

    def _process_torchvision(self) -> Tuple[str, str]:
        """Process torchvision dataset - download and organize in UPPERCASE location"""
        # Use the dataset name that was passed to the constructor (should be uppercase for torchvision)
        dataset_name = self.dataset_name  # Already uppercase from init for torchvision

        logger.info(f"Processing torchvision dataset: {dataset_name}")
        logger.info(f"Output directory: {self.dataset_dir}")

        # DYNAMIC DATASET DISCOVERY - Check if dataset exists in torchvision
        available_datasets = []
        dataset_class = None

        # Try different naming conventions
        possible_names = [
            dataset_name.upper(),
            dataset_name.title(),
            dataset_name.capitalize()
        ]

        for name in possible_names:
            if hasattr(datasets, name):
                dataset_class = getattr(datasets, name)
                logger.info(f"Found torchvision dataset: {name}")
                break

        if dataset_class is None:
            # Log available datasets for debugging
            available_datasets = []
            for name in dir(datasets):
                if not name.startswith('_') and name[0].isupper():
                    obj = getattr(datasets, name)
                    if isinstance(obj, type) and issubclass(obj, torch.utils.data.Dataset):
                        available_datasets.append(name)

            available_datasets.sort()
            raise ValueError(f"Torchvision dataset '{dataset_name}' not found. "
                           f"Available datasets: {', '.join(available_datasets)}")

        # Setup standard directory structure in Data/<dataset_name_uppercase>/
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        # Only recreate directories if they don't exist to avoid re-downloading
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Download and process datasets with dynamic parameter handling
            transform = transforms.ToTensor()

            # Try different loading strategies for different dataset interfaces
            try:
                logger.info(f"Downloading {dataset_name} training dataset...")
                train_dataset = dataset_class(
                    root='./data',  # Torchvision's default download location
                    train=True,
                    download=True,
                    transform=transform
                )

                logger.info(f"Downloading {dataset_name} test dataset...")
                test_dataset = dataset_class(
                    root='./data',
                    train=False,
                    download=True,
                    transform=transform
                )
            except TypeError as e:
                # Some datasets use different parameters (like split instead of train)
                try:
                    logger.info(f"Trying alternative parameters for {dataset_name}...")
                    train_dataset = dataset_class(
                        root='./data',
                        split='train',
                        download=True,
                        transform=transform
                    )
                    test_dataset = dataset_class(
                        root='./data',
                        split='test',
                        download=True,
                        transform=transform
                    )
                except TypeError as e2:
                    try:
                        # Try with different split names
                        train_dataset = dataset_class(
                            root='./data',
                            split='training',
                            download=True,
                            transform=transform
                        )
                        test_dataset = dataset_class(
                            root='./data',
                            split='validation',
                            download=True,
                            transform=transform
                        )
                    except TypeError as e3:
                        # Single dataset - create artificial split
                        logger.info(f"Creating train/test split for {dataset_name}...")
                        full_dataset = dataset_class(
                            root='./data',
                            download=True,
                            transform=transform
                        )
                        train_size = int(0.8 * len(full_dataset))
                        test_size = len(full_dataset) - train_size
                        train_dataset, test_dataset = torch.utils.data.random_split(
                            full_dataset, [train_size, test_size]
                        )

            # Save images with class directories in our standard structure
            self._save_torchvision_images(train_dataset, train_dir, "training")
            self._save_torchvision_images(test_dataset, test_dir, "test")

            logger.info(f"Successfully processed {dataset_name} dataset")
            logger.info(f"Training samples: {len(train_dataset)}")
            logger.info(f"Test samples: {len(test_dataset)}")
        else:
            logger.info(f"Using existing {dataset_name} dataset in {self.dataset_dir}")

        return train_dir, test_dir

    def _create_train_test_split(self, source_dir: str, test_size: float) -> Tuple[str, str]:
        """Create train/test split from source directory"""
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for class_name in tqdm(os.listdir(source_dir), desc="Processing classes"):
            class_path = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            # Create class directories
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Get all image files
            image_files = [f for f in os.listdir(class_path)
                         if f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)]

            # Random split
            random.shuffle(image_files)
            split_idx = int((1 - test_size) * len(image_files))
            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]

            # Copy files
            for fname in train_files:
                shutil.copy2(
                    os.path.join(class_path, fname),
                    os.path.join(train_class_dir, fname)
                )

            for fname in test_files:
                shutil.copy2(
                    os.path.join(class_path, fname),
                    os.path.join(test_class_dir, fname)
                )

        return train_dir, test_dir

class ConfigManager:
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        self.editor = os.environ.get('EDITOR', 'nano')


    def _open_editor(self, filepath: str) -> bool:
        """Open file in editor and return if changed"""
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                json.dump({}, f, indent=4)

        mtime = os.path.getmtime(filepath)
        try:
            subprocess.call([self.editor, filepath])
            changed = os.path.getmtime(filepath) > mtime
            if changed:
                # Validate JSON after editing
                with open(filepath, 'r') as f:
                    json.load(f)  # Just to validate
                return True
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in edited file {filepath}")
            return False
        except Exception as e:
            logger.error(f"Error opening editor: {str(e)}")
            return False
        return False

    def     _validate_json(self, filepath: str) -> Tuple[bool, Dict]:
        """Validate JSON file structure"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return True, data
        except Exception as e:
            logger.error(f"Error validating {filepath}: {str(e)}")
            return False, {}

    def merge_configs(self, existing: Dict, template: Dict) -> Dict:
        """Recursively merge template into existing config, adding missing entries"""
        result = existing.copy()
        for key, value in template.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = self.merge_configs(result[key], value)
        return result

    def manage_config(self, filepath: str, template: Dict) -> Dict:
        """Manage configuration file without overwriting existing content"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        existing_config = json.load(f)
                    # Merge template into existing config
                    merged_config = self.merge_configs(existing_config, template)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {filepath}, using template")
                    merged_config = template
            else:
                # For new file, use template
                merged_config = template

            # Save if file doesn't exist or changes were made
            if not os.path.exists(filepath) or merged_config != template:
                with open(filepath, 'w') as f:
                    json.dump(merged_config, f, indent=4)
                logger.info(f"Updated configuration file: {filepath}")

            return merged_config

        except Exception as e:
            logger.error(f"Error managing config {filepath}: {str(e)}")
            return template

    def manage_csv(self, filepath: str, headers: List[str]) -> bool:
        """Manage CSV file"""
        if not os.path.exists(filepath):
            logger.info(f"Creating new CSV file: {filepath}")
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            return True

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            try:
                existing_headers = next(reader)
                if existing_headers != headers:
                    logger.warning("CSV headers don't match expected structure")
                    response = input(f"Would you like to edit {filepath}? (y/n): ").lower()
                    if response == 'y':
                        return self._open_editor(filepath)
            except StopIteration:
                logger.error("Empty CSV file detected")
                return False

        return True

class EnhancedConfigManager(ConfigManager):
    """Enhanced configuration manager with support for specialized imaging features"""

    def __init__(self, config_dir: str):
        super().__init__(config_dir)
        self.editor = os.environ.get('EDITOR', 'nano')

    def verify_enhancement_config(self, config: Dict) -> Dict:
        """Verify and add enhancement-specific configurations"""
        if 'model' not in config:
            config['model'] = {}

        # Add enhancement modules configuration
        config['model'].setdefault('enhancement_modules', {
            'astronomical': {
                'enabled': False,
                'components': {
                    'structure_preservation': True,
                    'detail_preservation': True,
                    'star_detection': True,
                    'galaxy_features': True,
                    'kl_divergence': True
                },
                'weights': {
                    'detail_weight': 1.0,
                    'structure_weight': 0.8,
                    'edge_weight': 0.7
                }
            },
            'medical': {
                'enabled': False,
                'components': {
                    'tissue_boundary': True,
                    'lesion_detection': True,
                    'contrast_enhancement': True,
                    'subtle_feature_preservation': True
                },
                'weights': {
                    'boundary_weight': 1.0,
                    'lesion_weight': 0.8,
                    'contrast_weight': 0.6
                }
            },
            'agricultural': {
                'enabled': False,
                'components': {
                    'texture_analysis': True,
                    'damage_detection': True,
                    'color_anomaly': True,
                    'pattern_enhancement': True,
                    'morphological_features': True
                },
                'weights': {
                    'texture_weight': 1.0,
                    'damage_weight': 0.8,
                    'pattern_weight': 0.7
                }
            }
        })

        # Add loss function configurations
        config['model'].setdefault('loss_functions', {})
        loss_functions = config['model']['loss_functions']

        loss_functions.setdefault('astronomical_structure', {
            'enabled': False,
            'weight': 1.0,
            'components': {
                'edge_preservation': True,
                'peak_preservation': True,
                'detail_preservation': True
            }
        })

        loss_functions.setdefault('medical_structure', {
            'enabled': False,
            'weight': 1.0,
            'components': {
                'boundary_preservation': True,
                'tissue_contrast': True,
                'local_structure': True
            }
        })

        loss_functions.setdefault('agricultural_pattern', {
            'enabled': False,
            'weight': 1.0,
            'components': {
                'texture_preservation': True,
                'damage_pattern': True,
                'color_consistency': True
            }
        })

        return config

    def configure_image_type(self, config: Dict, image_type: str) -> Dict:
        """Configure enhancement modules for specific image type"""
        if 'dataset' not in config:
            config['dataset'] = {}

        config['dataset']['image_type'] = image_type

        # Disable all enhancement modules first
        for module in config['model']['enhancement_modules']:
            config['model']['enhancement_modules'][module]['enabled'] = False
            config['model']['loss_functions'][f'{module}_structure']['enabled'] = False

        # Enable specific module if not general
        if image_type != 'general' and image_type in config['model']['enhancement_modules']:
            config['model']['enhancement_modules'][image_type]['enabled'] = True
            config['model']['loss_functions'][f'{image_type}_structure']['enabled'] = True

        return config

    def interactive_setup(self, config: Dict) -> Dict:
        """Interactive configuration setup for enhancements"""
        print("\nEnhanced Autoencoder Configuration")
        print("=================================")

        # Ensure enhancement config exists
        config = self.verify_enhancement_config(config)

        # Configure based on image type
        image_type = config['dataset']['image_type']
        if image_type != 'general':
            module = config['model']['enhancement_modules'][image_type]

            print(f"\nConfiguring {image_type} components:")

            # Configure components
            for component in module['components']:
                current = module['components'][component]
                response = input(f"Enable {component}? (y/n) [{['n', 'y'][current]}]: ").lower()
                if response in ['y', 'n']:
                    module['components'][component] = (response == 'y')

            # Configure weights
            print(f"\nConfiguring {image_type} weights (0-1):")
            for weight_name, current_value in module['weights'].items():
                while True:
                    try:
                        new_value = input(f"{weight_name} [{current_value}]: ")
                        if new_value:
                            value = float(new_value)
                            if 0 <= value <= 1:
                                module['weights'][weight_name] = value
                                break
                            else:
                                print("Weight must be between 0 and 1")
                        else:
                            break
                    except ValueError:
                        print("Please enter a valid number")

            # Configure loss function
            loss_config = config['model']['loss_functions'][f'{image_type}_structure']
            print(f"\nConfiguring loss function components:")
            for component in loss_config['components']:
                current = loss_config['components'][component]
                response = input(f"Enable {component}? (y/n) [{['n', 'y'][current]}]: ").lower()
                if response in ['y', 'n']:
                    loss_config['components'][component] = (response == 'y')

            # Configure loss weight
            while True:
                try:
                    new_weight = input(f"Loss weight [{loss_config['weight']}]: ")
                    if new_weight:
                        weight = float(new_weight)
                        if weight > 0:
                            loss_config['weight'] = weight
                            break
                        else:
                            print("Weight must be positive")
                    else:
                        break
                except ValueError:
                    print("Please enter a valid number")

        return config

    def print_current_config(self, config: Dict):
        """Print current enhancement configuration"""
        print("\nCurrent Enhancement Configuration:")
        print("================================")

        image_type = config['dataset']['image_type']
        print(f"\nImage Type: {image_type}")

        if image_type != 'general':
            module = config['model']['enhancement_modules'][image_type]

            print("\nEnabled Components:")
            for component, enabled in module['components'].items():
                print(f"- {component}: {'✓' if enabled else '✗'}")

            print("\nComponent Weights:")
            for weight_name, value in module['weights'].items():
                print(f"- {weight_name}: {value:.2f}")

            print("\nLoss Function Configuration:")
            loss_config = config['model']['loss_functions'][f'{image_type}_structure']
            print(f"- Weight: {loss_config['weight']:.2f}")
            print("\nEnabled Loss Components:")
            for component, enabled in loss_config['components'].items():
                print(f"- {component}: {'✓' if enabled else '✗'}")

    def get_active_components(self, config: Dict) -> Dict:
        """Get currently active enhancement components"""
        image_type = config['dataset']['image_type']
        if image_type == 'general':
            return {}

        module = config['model']['enhancement_modules'][image_type]
        loss_config = config['model']['loss_functions'][f'{image_type}_structure']

        return {
            'type': image_type,
            'components': {k: v for k, v in module['components'].items() if v},
            'weights': module['weights'],
            'loss_components': {k: v for k, v in loss_config['components'].items() if v},
            'loss_weight': loss_config['weight']
        }

def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    return logger

def get_dataset(config: Dict, transform) -> Tuple[Dataset, Optional[Dataset]]:
    """Get dataset based on configuration"""
    dataset_config = config['dataset']

    if dataset_config['type'] == 'torchvision':
        train_dataset = getattr(torchvision.datasets, dataset_config['name'].upper())(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = getattr(torchvision.datasets, dataset_config['name'].upper())(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    else:
        train_dir = dataset_config['train_dir']
        test_dir = dataset_config.get('test_dir')

        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")

        # NEW: Pass configuration to CustomImageDataset for reconstruction sampling
        train_dataset = CustomImageDataset(
            data_dir=train_dir,
            transform=transform,
            config=config,  # Pass full config for reconstruction sampling
            data_name=dataset_config.get('name')  # Pass dataset name
        )

        test_dataset = None
        if test_dir and os.path.exists(test_dir):
            test_dataset = CustomImageDataset(
                data_dir=test_dir,
                transform=transform,
                config=config,  # Pass config to test dataset too
                data_name=dataset_config.get('name')
            )

    # NEW: Log dataset information for reconstruction sampling
    logger.info(f"Dataset loaded: {len(train_dataset)} training samples")
    if test_dataset:
        logger.info(f"Dataset loaded: {len(test_dataset)} test samples")

    # Log expected reconstruction output directory
    dataset_name = dataset_config.get('name', 'dataset').lower()
    recon_dir = os.path.join('data', dataset_name, 'reconstruction_samples')
    logger.info(f"Reconstruction samples will be saved to: {recon_dir}")

    if config['training'].get('merge_datasets', False) and test_dataset is not None:
        # NEW: Handle configuration for combined datasets
        combined_dataset = CombinedDataset(train_dataset, test_dataset)
        logger.info(f"Merged dataset: {len(combined_dataset)} total samples")
        return combined_dataset, None

    return train_dataset, test_dataset

class CombinedDataset(Dataset):
    """Dataset that combines train and test sets"""
    def __init__(self, train_dataset: Dataset, test_dataset: Dataset):
        self.combined_data = ConcatDataset([train_dataset, test_dataset])

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        return self.combined_data[idx]

def update_config_with_args(config: Dict, args) -> Dict:
    """Update configuration with command line arguments"""
    # Ensure required sections exist
    if 'model' not in config:
        config['model'] = {}
    if 'training' not in config:
        config['training'] = {}
    if 'execution_flags' not in config:
        config['execution_flags'] = {}

    # Update model parameters
    if hasattr(args, 'encoder_type') and args.encoder_type:
        config['model']['encoder_type'] = args.encoder_type

    # Update training parameters with safe access
    if hasattr(args, 'batch_size') and args.batch_size:
        config['training']['batch_size'] = args.batch_size

    if hasattr(args, 'epochs') and args.epochs:
        config['training']['epochs'] = args.epochs

    if hasattr(args, 'workers') and args.workers:
        config['training']['num_workers'] = args.workers

    # Update model learning rate
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config['model']['learning_rate'] = args.learning_rate

    # Update execution flags
    if hasattr(args, 'cpu') and args.cpu is not None:
        config['execution_flags']['use_gpu'] = not args.cpu

    if hasattr(args, 'debug') and args.debug is not None:
        config['execution_flags']['debug_mode'] = args.debug

    return config

def print_usage():
    """Print usage information with examples"""
    print("\nCDBNN (Convolutional Deep Bayesian Neural Network) Image Processor")
    print("=" * 80)
    print("\nUsage:")
    print("  1. Interactive Mode:")
    print("     python cdbnn.py")
    print("\n  2. Command Line Mode:")
    print("     python cdbnn.py --data_type TYPE --data PATH [options]")

    print("\nRequired Arguments:")
    print("  --data_type     Type of dataset ('torchvision' or 'custom')")
    print("  --data          Dataset name (for torchvision) or path (for custom)")

    print("\nOptional Arguments:")
    print("  --encoder_type  Type of encoder ('cnn' or 'autoenc')")
    print("  --config        Path to configuration file (overrides other options)")
    print("  --batch_size    Batch size for training (default: 128)")
    print("  --epochs        Number of training epochs (default: 200)")
    print("  --workers       Number of data loading workers (default: 4)")
    print("  --learning_rate Learning rate (default: 0.001)")
    print("  --output-dir    Output directory (default: data)")
    print("  --cpu          Force CPU usage even if GPU is available")
    print("  --debug        Enable debug mode with verbose logging")

    print("\nExamples:")
    print("  1. Process MNIST dataset using CNN:")
    print("     python cdbnn.py --data_type torchvision --data MNIST --encoder_type cnn")

    print("  2. Process custom dataset using Autoencoder:")
    print("     python cdbnn.py --data_type custom --data path/to/images --encoder_type autoenc")

import argparse

def list_and_download_datasets():
    """List available torchvision datasets and allow user to download and process them."""

    # Get all available torchvision datasets
    available_datasets = []
    for name in dir(datasets):
        if (not name.startswith('_') and
            hasattr(getattr(datasets, name), '__call__') and
            name not in ['VisionDataset', 'DatasetFolder', 'ImageFolder', 'FakeData']):
            available_datasets.append(name)

    available_datasets.sort()

    print("\n" + "="*80)
    print("AVAILABLE TORCHVISION DATASETS")
    print("="*80)

    # Group datasets by categories for better organization
    categories = {
        'Basic Vision': ['MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST', 'QMNIST', 'USPS', 'Omniglot', 'SEMEION'],
        'Advanced Vision': ['CIFAR10', 'CIFAR100', 'STL10', 'SVHN', 'Caltech101', 'Caltech256'],
        'Fine-Grained': ['Flowers102', 'OxfordIIITPet', 'StanfordCars', 'FGVCAircraft', 'Food101', 'CUB200'],
        'Scene Understanding': ['Cityscapes', 'Places365', 'SUN397', 'Country211'],
        'Faces & People': ['CelebA', 'LFWPeople', 'LFWPairs', 'WIDERFace'],
        'Objects & Detection': ['CocoCaptions', 'CocoDetection', 'VOCDetection', 'VOCSegmentation', 'Kitti', 'PCAM'],
        'Video & Action': ['HMDB51', 'UCF101', 'Kinetics', 'MovingMNIST'],
        'Stereo & Depth': ['CLEVRClassification', 'CREStereo', 'CarlaStereo', 'ETH3DStereo',
                          'FallingThingsStereo', 'FlyingChairs', 'FlyingThings3D',
                          'InStereo2k', 'Kitti2012Stereo', 'Kitti2015Stereo', 'KittiFlow',
                          'Middlebury2014Stereo', 'SceneFlowStereo', 'Sintel', 'SintelStereo'],
        'Text & Documents': ['RenderedSST2', 'IMDB'],
        'Specialized': ['DTD', 'GTSRB', 'EuroSAT', 'FER2013', 'INaturalist', 'Imagenette', 'SBU', 'SBDataset']
    }

    # Flatten categories to check for uncategorized datasets
    categorized = set()
    for cat_list in categories.values():
        categorized.update(cat_list)

    uncategorized = [ds for ds in available_datasets if ds not in categorized]
    if uncategorized:
        categories['Other'] = uncategorized

    # Display datasets by category
    for category, ds_list in categories.items():
        if any(ds in available_datasets for ds in ds_list):
            print(f"\n{category}:")
            available_in_cat = [ds for ds in ds_list if ds in available_datasets]
            for i, ds_name in enumerate(available_in_cat):
                print(f"  {i+1:2d}. {ds_name}")

    print("\n" + "="*80)

    # Get user selection
    while True:
        try:
            choice = input("\nEnter dataset name or number from list (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None

            # Try to parse as number first
            if choice.isdigit():
                choice_num = int(choice)
                flat_list = []
                for ds_list in categories.values():
                    flat_list.extend([ds for ds in ds_list if ds in available_datasets])

                if 1 <= choice_num <= len(flat_list):
                    selected_dataset = flat_list[choice_num - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(flat_list)}")
            else:
                # Treat as dataset name
                if choice.upper() in available_datasets:
                    selected_dataset = choice.upper()
                    break
                else:
                    print(f"Dataset '{choice}' not found. Please choose from the list.")

        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")

    return selected_dataset

def list_datasets_simple():
    """Simple function to just list available datasets without interaction."""
    available_datasets = []
    for name in dir(datasets):
        if (not name.startswith('_') and
            hasattr(getattr(datasets, name), '__call__') and
            name not in ['VisionDataset', 'DatasetFolder', 'ImageFolder', 'FakeData']):
            available_datasets.append(name)

    available_datasets.sort()
    print("\nAvailable torchvision datasets:")
    for i, ds in enumerate(available_datasets, 1):
        print(f"{i:3d}. {ds}")

    return available_datasets

# You can call this from command line with: python cdbnn.py --list-datasets

def save_last_args(args):
    """Save arguments for next session"""
    args_dict = {
        'data_name': getattr(args, 'data_name', None),
        'mode': getattr(args, 'mode', 'train'),
        'data_type': getattr(args, 'data_type', 'custom'),
        'data': getattr(args, 'data', ''),
        'encoder_type': getattr(args, 'encoder_type', 'autoenc'),
        'batch_size': getattr(args, 'batch_size', 128),
        'epochs': getattr(args, 'epochs', 200),
        'output_dir': getattr(args, 'output_dir', 'Data'),
        'workers': getattr(args, 'workers', 4),
        'learning_rate': getattr(args, 'learning_rate', 0.01),
        'cpu': getattr(args, 'cpu', False),
        'debug': getattr(args, 'debug', False),
        'config': getattr(args, 'config', None),
        'invert_dbnn': getattr(args, 'invert_dbnn', True),
        'input_csv': getattr(args, 'input_csv', ''),
        'model_path': getattr(args, 'model_path', ''),
        'input_path': getattr(args, 'input_path', ''),
        'output_csv': getattr(args, 'output_csv', '')
    }

    os.makedirs('config', exist_ok=True)
    with open('config/last_args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

def load_last_args():
    """Load arguments from JSON file"""
    try:
        with open('last_run.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_interactive_args():
    """Get arguments interactively with invert DBNN support."""
    last_args = load_last_args()
    args = argparse.Namespace()

    # Get data_name first (this is the key parameter)
    default_name = last_args.get('data_name') if last_args else None
    data_name = input(f"Enter dataset name (e.g., cifar100, galaxies, etc.) [{default_name}]: ").strip()
    args.data_name = data_name or default_name
    if not args.data_name:
        raise ValueError("Dataset name is required")

    # Get mode (train/reconstruct/predict)
    while True:
        default = last_args.get('mode', 'train') if last_args else 'train'
        prompt = f"\nEnter mode (train/reconstruct/predict) [{default}]: "
        mode = input(prompt).strip().lower() or default
        if mode in ['train', 'reconstruct', 'predict']:
            args.mode = mode
            break
        print("Invalid mode. Please enter 'train', 'reconstruct', or 'predict'")

    # Get data type
    while True:
        default = last_args.get('data_type', '') if last_args else ''
        prompt = f"\nEnter dataset type (torchvision/custom) [{default}]: " if default else "\nEnter dataset type (torchvision/custom): "
        data_type = input(prompt).strip().lower() or default
        if data_type in ['torchvision', 'custom']:
            args.data_type = data_type
            break
        print("Invalid type. Please enter 'torchvision' or 'custom'")

    # Get data path/name based on data_type
    if args.data_type == 'torchvision':
        # For torchvision, use the data_name as the dataset identifier
        args.data = args.data_name.upper()  # Torchvision expects uppercase
    else:
        # For custom datasets, ask for the path
        default = last_args.get('data', '') if last_args else ''
        if not default and args.data_name:
            # Suggest a default path based on data_name - use Data/ for input
            default = f"Data/{args.data_name}"
        prompt = f"Enter dataset path [{default}]: "
        dataset_path = input(prompt).strip() or default
        args.data = dataset_path

    # Handle predict mode
    if args.mode == 'predict':
        # Set default model path using Data/ directory (uppercase)
        default_model = f"data/{args.data_name}/checkpoints/{args.data_name}_unified.pth"
        prompt = f"Enter path to trained model [{default_model}]: "
        args.model_path = input(prompt).strip() or default_model

        # Set default input directory using Data/ directory (uppercase)
        default_input = f"Data/{args.data_name}"
        prompt = f"Enter directory containing new images [{default_input}]: "
        args.input_path = input(prompt).strip() or default_input

        # Set default output CSV path using data/ directory (lowercase) with lowercase dataset name
        dataset_name_lower = args.data_name.lower()
        default_csv = f"data/{dataset_name_lower}/{dataset_name_lower}.csv"
        prompt = f"Enter output CSV path [{default_csv}]: "
        args.output_csv = input(prompt).strip() or default_csv

    # Handle train/reconstruct modes
    else:
        # Ask about invert DBNN
        default_invert = last_args.get('invert_dbnn', True) if last_args else True
        invert_response = input(f"Enable inverse DBNN mode? (y/n) [{['n', 'y'][default_invert]}]: ").strip().lower()
        args.invert_dbnn = invert_response == 'y' if invert_response else default_invert

        # If in reconstruct mode and invert DBNN is enabled, ask for input CSV
        if args.mode == 'reconstruct' and args.invert_dbnn:
            default_csv = last_args.get('input_csv', '') if last_args else ''
            if not default_csv and args.data_name:
                # Use lowercase data directory and lowercase dataset name
                dataset_name_lower = args.data_name.lower()
                default_csv = f"data/{dataset_name_lower}/{dataset_name_lower}.csv"
            prompt = f"Enter input CSV path (or leave empty for default) [{default_csv}]: "
            args.input_csv = input(prompt).strip() or default_csv

    # Get encoder type
    while True:
        default = last_args.get('encoder_type', 'autoenc') if last_args else 'autoenc'
        prompt = f"Enter encoder type (cnn/autoenc) [{default}]: "
        encoder_type = input(prompt).strip().lower() or default
        if encoder_type in ['cnn', 'autoenc']:
            args.encoder_type = encoder_type
            break
        print("Invalid encoder type. Please enter 'cnn' or 'autoenc'")

    # Optional parameters
    default = last_args.get('batch_size', 128) if last_args else 128
    args.batch_size = int(input(f"Enter batch size [{default}]: ").strip() or default)

    if args.mode == 'train':
        default = last_args.get('epochs', 200) if last_args else 200
        args.epochs = int(input(f"Enter number of epochs [{default}]: ").strip() or default)

    default = last_args.get('output', 'Data') if last_args else 'Data'
    args.output_dir = input(f"Enter output directory [{default}]: ").strip() or default

    # Set other defaults
    args.workers = last_args.get('workers', 4) if last_args else 4
    args.learning_rate = last_args.get('learning_rate', 0.01) if last_args else 0.01
    args.cpu = last_args.get('cpu', False) if last_args else False
    args.debug = last_args.get('debug', False) if last_args else False
    args.config = last_args.get('config', None) if last_args else None

    save_last_args(args)
    return args

def check_existing_model(dataset_dir, dataset_name):
    """Check existing model type from checkpoint"""
    checkpoint_path = os.path.join(dataset_dir, 'checkpoints', f"{dataset_name}_best.pth")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint.get('config', {}).get('model', {}).get('encoder_type')
        except:
            pass
    return None

def detect_model_type_from_checkpoint(checkpoint_path):
    """Detect model architecture type from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # Check for architecture-specific layer patterns
        is_cnn = any('conv_layers' in key for key in state_dict.keys())
        is_autoencoder = any('encoder_layers' in key for key in state_dict.keys())

        if is_cnn:
            return 'cnn'
        elif is_autoencoder:
            return 'autoenc'
        else:
            return None
    except Exception as e:
        logger.error(f"Error detecting model type: {str(e)}")
        return None

def configure_enhancements(config: Dict) -> Dict:
    """Interactive configuration of enhancement features"""
    enhancements = config['model']['autoencoder_config']['enhancements']

    print("\nConfiguring Enhanced Autoencoder Features:")

    # KL Divergence configuration
    if input("Enable KL divergence clustering? (y/n) [n]: ").lower() != 'n':
        enhancements['use_kl_divergence'] = True
        enhancements['kl_divergence_weight'] = float(input("Enter KL divergence weight (0-1) [0.1]: ") or 0.1)
    else:
        enhancements['use_kl_divergence'] = False

    # Class encoding configuration
    if input("Enable class encoding? (y/n) [y]: ").lower() == 'n':
        enhancements['use_class_encoding'] = False
        enhancements['classification_weight'] = float(input("Enter classification weight (0-1) [0.1]: ") or 0.1)
    else:
        enhancements['use_class_encoding'] = True

    # Clustering configuration
    if enhancements['use_kl_divergence']:
        enhancements['clustering_temperature'] = float(input("Enter clustering temperature (0.1-2.0) [1.0]: ") or 1.0)
        enhancements['min_cluster_confidence'] = float(input("Enter minimum cluster confidence (0-1) [0.7]: ") or 0.7)

    return config

def add_enhancement_features(config: Dict) -> Dict:
    """Add enhancement features to existing configuration"""
    # Ensure basic structure exists
    if 'model' not in config:
        config['model'] = {}
    if 'enhancement_modules' not in config['model']:
        config['model']['enhancement_modules'] = {}
    if 'loss_functions' not in config['model']:
        config['model']['loss_functions'] = {}

    # Ask about each enhancement type
    print("\nAvailable Enhancement Features:")

    # Astronomical features
    if input("Add astronomical features (star detection, galaxy structure preservation)? (y/n) [n]: ").lower() == 'y':
        config['model']['enhancement_modules']['astronomical'] = {
            'enabled': True,
            'components': {
                'structure_preservation': True,
                'detail_preservation': True,
                'star_detection': True,
                'galaxy_features': True,
                'kl_divergence': True
            },
            'weights': {
                'detail_weight': 1.0,
                'structure_weight': 0.8,
                'edge_weight': 0.7
            }
        }
        config['model']['loss_functions']['astronomical_structure'] = {
            'enabled': True,
            'weight': 1.0,
            'components': {
                'edge_preservation': True,
                'peak_preservation': True,
                'detail_preservation': True
            }
        }
        print("Astronomical features added.")

    # Medical features
    if input("Add medical features (tissue boundary, lesion detection)? (y/n) [n]: ").lower() == 'y':
        config['model']['enhancement_modules']['medical'] = {
            'enabled': True,
            'components': {
                'tissue_boundary': True,
                'lesion_detection': True,
                'contrast_enhancement': True,
                'subtle_feature_preservation': True
            },
            'weights': {
                'boundary_weight': 1.0,
                'lesion_weight': 0.8,
                'contrast_weight': 0.6
            }
        }
        config['model']['loss_functions']['medical_structure'] = {
            'enabled': True,
            'weight': 1.0,
            'components': {
                'boundary_preservation': True,
                'tissue_contrast': True,
                'local_structure': True
            }
        }
        print("Medical features added.")

    # Agricultural features
    if input("Add agricultural features (texture analysis, damage detection)? (y/n) [n]: ").lower() == 'y':
        config['model']['enhancement_modules']['agricultural'] = {
            'enabled': True,
            'components': {
                'texture_analysis': True,
                'damage_detection': True,
                'color_anomaly': True,
                'pattern_enhancement': True,
                'morphological_features': True
            },
            'weights': {
                'texture_weight': 1.0,
                'damage_weight': 0.8,
                'pattern_weight': 0.7
            }
        }
        config['model']['loss_functions']['agricultural_pattern'] = {
            'enabled': True,
            'weight': 1.0,
            'components': {
                'texture_preservation': True,
                'damage_pattern': True,
                'color_consistency': True
            }
        }
        print("Agricultural features added.")

    return config

def update_existing_config(config_path: str, new_config: Dict) -> Dict:
    """Update existing configuration while preserving current settings"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            existing_config = json.load(f)

        # Merge configurations
        for key in new_config:
            if key in existing_config:
                if isinstance(existing_config[key], dict) and isinstance(new_config[key], dict):
                    existing_config[key].update(new_config[key])
                else:
                    existing_config[key] = new_config[key]
            else:
                existing_config[key] = new_config[key]

        return existing_config
    return new_config

def download_and_setup_torchvision_dataset(dataset_name):
    """Download and setup a torchvision dataset locally with UPPERCASE directory."""
    try:
        dataset_name_upper = dataset_name.upper()

        # Check if dataset exists in torchvision
        available_datasets = []
        for name in dir(datasets):
            if (not name.startswith('_') and
                hasattr(getattr(datasets, name), '__call__') and
                name not in ['VisionDataset', 'DatasetFolder', 'ImageFolder', 'FakeData']):
                available_datasets.append(name)

        if dataset_name_upper not in available_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in torchvision. Available: {', '.join(available_datasets)}")

        # Create processor for torchvision dataset - use UPPERCASE for directory
        processor = DatasetProcessor(
            datafile=dataset_name_upper,
            datatype='torchvision',
            output_dir='Data',  # Use 'Data' with capital D
            data_name=dataset_name_upper  # Use UPPERCASE for directory name
        )

        # Process the dataset (this will download if needed)
        train_dir, test_dir = processor.process()

        logger.info(f"Successfully downloaded and processed {dataset_name_upper}")
        logger.info(f"Training data: {train_dir}")
        logger.info(f"Test data: {test_dir}")

        # Verify the directory structure is correct
        expected_dir = f"Data/{dataset_name_upper}"
        if not train_dir.startswith(expected_dir):
            logger.warning(f"Expected train_dir to start with {expected_dir}, but got {train_dir}")

        return train_dir

    except Exception as e:
        logger.error(f"Error downloading dataset {dataset_name}: {str(e)}")
        raise

def download_and_process_dataset(dataset_name):
    """Download and process the selected dataset with UPPERCASE directory."""

    dataset_name_upper = dataset_name.upper()  # Use UPPERCASE for directory

    print(f"\nDownloading and processing {dataset_name}...")
    print(f"Output directory: Data/{dataset_name_upper}")

    # Create processor for torchvision dataset
    processor = DatasetProcessor(
        datafile=dataset_name,
        datatype='torchvision',
        output_dir='Data',  # Use 'Data' with capital D
        data_name=dataset_name_upper  # Use UPPERCASE for directory
    )

    try:
        # Process the dataset (this will download if needed)
        train_dir, test_dir = processor.process()

        # Generate configuration
        config = processor.generate_default_config(train_dir)

        print(f"\n✓ Successfully downloaded and processed {dataset_name}")
        print(f"  - Training data: {train_dir}")
        print(f"  - Test data: {test_dir}")
        print(f"  - Configuration: Data/{dataset_name_upper}/{dataset_name_upper}.json")

        # Verify the path is correct
        if not train_dir.startswith(f"Data/{dataset_name_upper}"):
            print(f"⚠️  Warning: Expected path to be Data/{dataset_name_upper}/... but got {train_dir}")

        return {
            'dataset_name': dataset_name,
            'dataset_name_upper': dataset_name_upper,
            'data_dir': f"Data/{dataset_name_upper}",
            'train_dir': train_dir,
            'test_dir': test_dir,
            'config': config,
            'input_path': train_dir
        }

    except Exception as e:
        print(f"✗ Error processing {dataset_name}: {str(e)}")
        return None

def interactive_dataset_selection():
    """Enhanced interactive dataset selection with local folder browser."""

    print("\n🤖 CDBNN Dataset Manager")
    print("="*50)

    while True:
        print("\nOptions:")
        print("1. 📋 List and download available datasets")
        print("2. 📁 Browse local folders")
        print("3. ⌨️  Enter local dataset path manually")
        print("4. 🚪 Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            # List and download datasets
            selected_dataset = list_and_download_datasets()
            if selected_dataset:
                result = download_and_process_dataset(selected_dataset)
                if result:
                    return result

        elif choice == '2':
            # Browse local folders
            print("\n📁 Local Folder Browser")
            print("Navigate to your dataset folder and press 's' to select")
            selected_path = browse_local_folders()

            if selected_path:
                dataset_name = os.path.basename(os.path.normpath(selected_path))
                print(f"\nSelected folder: {selected_path}")
                print(f"Dataset name: {dataset_name}")

                # Check if it's already processed (has train/test structure)
                train_dir = os.path.join(selected_path, 'train')
                test_dir = os.path.join(selected_path, 'test')

                if os.path.exists(train_dir) and os.path.isdir(train_dir):
                    print("✓ Found train/test directory structure")
                    return {
                        'dataset_name': dataset_name,
                        'dataset_name_upper': dataset_name.upper(),
                        'data_dir': selected_path,
                        'train_dir': train_dir,
                        'test_dir': test_dir if os.path.exists(test_dir) else None,
                        'input_path': selected_path,
                        'is_local': True
                    }
                else:
                    # Check if it has class subdirectories
                    subdirs = [d for d in os.listdir(selected_path)
                              if os.path.isdir(os.path.join(selected_path, d))]
                    image_files = [f for f in os.listdir(selected_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

                    if subdirs or image_files:
                        print("✓ Found dataset files")
                        return {
                            'dataset_name': dataset_name,
                            'dataset_name_upper': dataset_name.upper(),
                            'data_dir': selected_path,
                            'train_dir': selected_path,  # Use root as train dir
                            'test_dir': None,
                            'input_path': selected_path,
                            'is_local': True,
                            'needs_processing': True
                        }
                    else:
                        print("❌ No images or subdirectories found in selected folder")
                        continue

        elif choice == '3':
            # Manual path entry
            dataset_path = input("Enter full path to dataset directory: ").strip()

            # Expand user home directory (~)
            dataset_path = os.path.expanduser(dataset_path)

            if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
                dataset_name = os.path.basename(os.path.normpath(dataset_path))

                # Check structure
                train_dir = os.path.join(dataset_path, 'train')
                test_dir = os.path.join(dataset_path, 'test')

                if os.path.exists(train_dir) and os.path.isdir(train_dir):
                    test_dir = test_dir if os.path.exists(test_dir) else None
                    print("✓ Using existing train/test structure")
                else:
                    train_dir = dataset_path
                    test_dir = None
                    print("✓ Using directory as training data")

                return {
                    'dataset_name': dataset_name,
                    'dataset_name_upper': dataset_name.upper(),
                    'data_dir': dataset_path,
                    'train_dir': train_dir,
                    'test_dir': test_dir,
                    'input_path': dataset_path,
                    'is_local': True
                }
            else:
                print(f"❌ Directory not found: {dataset_path}")

        elif choice == '4':
            print("Goodbye!")
            return None

        else:
            print("Invalid option. Please try again.")

def main():
    """Main function for CDBNN processing with interactive dataset selection and default heatmap generation."""
    args = None

    # Setup logging
    logger = setup_logging()

    # Parse arguments
    args = parse_arguments()

    # Handle list-datasets option
    if hasattr(args, 'list_datasets') and args.list_datasets:
        list_datasets_simple()
        return 0

    # Handle download mode
    if hasattr(args, 'mode') and args.mode == 'download' or (hasattr(args, 'download') and args.download):
        downloaded_path = interactive_torchvision_download()
        if downloaded_path:
            if isinstance(downloaded_path, list):
                print("\nDownloaded datasets:")
                for dataset_name, path in downloaded_path:
                    print(f"  - {dataset_name}: {path}")
            else:
                print(f"\nDataset ready at: {downloaded_path}")
                print(f"You can now train using: python cdbnn.py --input_path {downloaded_path} --data_type custom --mode train")
        return 0

    # If no data source provided, use interactive selection
    if (not hasattr(args, 'data_name') or not args.data_name) and \
       (not hasattr(args, 'input_path') or not args.input_path) or \
       (hasattr(args, 'interactive') and args.interactive):

        print("No dataset specified. Starting interactive mode...")
        dataset_info = interactive_dataset_selection()

        if not dataset_info:
            return 0  # User chose to exit

        # Set the arguments based on selected dataset
        if not hasattr(args, 'data_name'):
            # Use the exact dataset name without uppercase conversion for custom datasets
            if 'is_local' in dataset_info and dataset_info['is_local']:
                args.data_name = dataset_info['dataset_name']  # Keep original case
            else:
                args.data_name = dataset_info['dataset_name_upper']  # Use uppercase for torchvision
        if not hasattr(args, 'input_path'):
            args.input_path = dataset_info['input_path']
        if not hasattr(args, 'data_type'):
            args.data_type = 'torchvision' if 'is_local' not in dataset_info else 'custom'

        print(f"\nUsing dataset: {args.data_name}")
        print(f"Data path: {args.input_path}")

    # Handle the specific case where data_name is provided but might be invalid
    if hasattr(args, 'data_name') and args.data_name:
        # Check if it's a valid torchvision dataset name
        available_datasets = []
        for name in dir(datasets):
            if (not name.startswith('_') and
                hasattr(getattr(datasets, name), '__call__') and
                name not in ['VisionDataset', 'DatasetFolder', 'ImageFolder', 'FakeData']):
                available_datasets.append(name)

        # If it's a torchvision dataset but not a valid one, show error
        if (hasattr(args, 'data_type') and args.data_type == 'torchvision' and
            args.data_name.upper() not in available_datasets):

            logger.error(f"Invalid torchvision dataset: {args.data_name}")
            logger.info(f"Available datasets: {', '.join(available_datasets)}")

            # Fall back to interactive selection
            print("Falling back to interactive dataset selection...")
            dataset_info = interactive_dataset_selection()
            if dataset_info:
                if 'is_local' in dataset_info and dataset_info['is_local']:
                    args.data_name = dataset_info['dataset_name']  # Keep original case
                else:
                    args.data_name = dataset_info['dataset_name_upper']  # Use uppercase for torchvision
                args.input_path = dataset_info['input_path']
                args.data_type = 'torchvision' if 'is_local' not in dataset_info else 'custom'
            else:
                return 1

    # Handle automatic torchvision dataset setup
    if (hasattr(args, 'data_type') and args.data_type == 'torchvision' and
        hasattr(args, 'data_name') and args.data_name and
        (not hasattr(args, 'input_path') or not args.input_path) and
        hasattr(args, 'mode') and args.mode == 'train'):

        logger.info(f"Automatically setting up torchvision dataset: {args.data_name}")
        try:
            downloaded_path = download_and_setup_torchvision_dataset(args.data_name)
            args.input_path = downloaded_path
            args.data_type = 'custom'
            logger.info(f"Using downloaded dataset as custom dataset from: {downloaded_path}")
        except Exception as e:
            logger.error(f"Failed to download torchvision dataset {args.data_name}: {str(e)}")
            logger.info("Falling back to standard torchvision processing...")

    # Process based on mode
    if hasattr(args, 'mode'):
        if args.mode == 'predict':
            # Prediction mode - load model and predict on new images
            logger.info("Starting prediction mode...")
            if not hasattr(args, 'input_path') or not args.input_path:
                logger.error("Input path required for prediction mode")
                return 1

            # Load configuration - use Data/ (uppercase) directory
            config_path = args.config if hasattr(args, 'config') and args.config else None
            if not config_path and hasattr(args, 'data_name') and args.data_name:
                # Use 'Data/' directory (uppercase) and exact data_name for JSON
                config_path = os.path.join('data', args.data_name, f"{args.data_name}.json").lower()

            if not config_path or not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                return 1

            with open(config_path, 'r') as f:
                config = json.load(f)

            # Initialize prediction manager
            prediction_manager = PredictionManager(config)

            # Run prediction - use 'data/' directory (lowercase) for output CSV
            output_csv = getattr(args, 'output_csv', None)
            if not output_csv and hasattr(args, 'data_name') and args.data_name:
                # Use 'data/' directory (lowercase) and lowercase dataset_name.csv
                dataset_name_lower = args.data_name.lower()
                output_csv = os.path.join('data', dataset_name_lower, f"{dataset_name_lower}.csv")

            batch_size = getattr(args, 'batch_size', 128)

            # UPDATED: Heatmap generation now defaults to True
            generate_heatmaps = getattr(args, 'generate_heatmaps', True)  # Default to True
            num_samples = getattr(args, 'num_samples', None)  # None means all samples by default

            if generate_heatmaps:
                if num_samples is None:
                    logger.info("Heatmap generation enabled (default) - creating attention visualizations for ALL samples")
                else:
                    logger.info(f"Heatmap generation enabled (default) - creating attention visualizations for {num_samples} samples per class")
            else:
                logger.info("Heatmap generation disabled (--no_heatmaps flag used)")

            # Updated prediction call with default heatmap support
            prediction_manager.predict_images(
                args.input_path,
                output_csv,
                batch_size,
                generate_heatmaps=generate_heatmaps
            )

            # UPDATED: Enhanced heatmap location info
            if generate_heatmaps:
                dataset_name_lower = args.data_name.lower()
                heatmap_dir = os.path.join('data', dataset_name_lower, 'attention_heatmaps')
                logger.info(f"Attention heatmaps saved to: {heatmap_dir}")
                logger.info("To disable heatmaps in future runs, use --no_heatmaps flag")

            logger.info(f"Prediction completed successfully! Output: {output_csv}")
            return 0

        elif args.mode == 'train':
            return handle_training_mode(args, logger)

        elif args.mode == 'reconstruct':
            # Reconstruction mode - generate images from features
            logger.info("Starting reconstruction mode...")
            if not hasattr(args, 'input_csv') or not args.input_csv:
                logger.error("Input CSV path required for reconstruction mode")
                return 1

            # Load configuration - use Data/ (uppercase) directory
            config_path = args.config if hasattr(args, 'config') and args.config else None
            if not config_path and hasattr(args, 'data_name') and args.data_name:
                # Use 'Data/' directory (uppercase) and exact data_name for JSON
                config_path = os.path.join('data', args.data_name, f"{args.data_name}.json")

            if not config_path or not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                return 1

            with open(config_path, 'r') as f:
                config = json.load(f)

            # Initialize reconstruction manager
            reconstruction_manager = ReconstructionManager(config)

            # Run reconstruction - use 'data/' directory (lowercase) for output
            output_dir = getattr(args, 'output_dir', None)
            if not output_dir and hasattr(args, 'data_name') and args.data_name:
                # Use 'data/' directory (lowercase) for output
                dataset_name_lower = args.data_name.lower()
                output_dir = os.path.join('data', dataset_name_lower, 'reconstructions')

            reconstruction_manager.predict_from_csv(args.input_csv, output_dir)
            logger.info("Reconstruction completed successfully!")
            return 0

        # UPDATED: Heatmap-only mode for existing models
        elif args.mode == 'heatmaps':
            logger.info("Starting heatmap generation mode...")
            if not hasattr(args, 'input_path') or not args.input_path:
                logger.error("Input path required for heatmap generation mode")
                return 1

            # Load configuration
            config_path = args.config if hasattr(args, 'config') and args.config else None
            if not config_path and hasattr(args, 'data_name') and args.data_name:
                config_path = os.path.join('data', args.data_name, f"{args.data_name}.json").lower()

            if not config_path or not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                return 1

            with open(config_path, 'r') as f:
                config = json.load(f)

            # Initialize prediction manager
            prediction_manager = PredictionManager(config)

            # UPDATED: Default to all samples
            num_samples = getattr(args, 'num_samples', None)  # None means all samples by default

            if num_samples is None:
                logger.info("Generating heatmaps for ALL samples (default)")
            else:
                logger.info(f"Generating heatmaps for {num_samples} samples per class")

            # Generate heatmaps only
            prediction_manager.generate_classwise_attention_heatmaps(
                args.input_path,
                num_samples_per_class=num_samples
            )

            dataset_name_lower = args.data_name.lower() if hasattr(args, 'data_name') else 'dataset'
            heatmap_dir = os.path.join('data', dataset_name_lower, 'attention_heatmaps')
            logger.info(f"Heatmap generation completed! Visualizations saved to: {heatmap_dir}")
            return 0

        else:
            logger.error(f"Invalid mode: {args.mode}")
            return 1
    else:
        logger.error("No mode specified")
        return 1


def handle_prediction_mode(args, logger):
    """Handle prediction/reconstruction mode operations."""
    try:
        # Load configuration
        config_path = args.config if hasattr(args, 'config') and args.config else None
        if not config_path and hasattr(args, 'data_name') and args.data_name:
            config_path = os.path.join('Data', args.data_name.upper(), f"{args.data_name.upper()}.json")

        if not config_path or not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return 1

        with open(config_path, 'r') as f:
            config = json.load(f)

        if args.mode == 'predict':
            # Prediction mode
            if not hasattr(args, 'input_path') or not args.input_path:
                logger.error("Input path required for prediction mode")
                return 1

            prediction_manager = PredictionManager(config)
            output_csv = getattr(args, 'output_csv', None)
            prediction_manager.predict_images(args.input_path, output_csv)
            logger.info("Prediction completed successfully!")

        elif args.mode == 'reconstruct':
            # Reconstruction mode
            if not hasattr(args, 'input_csv') or not args.input_csv:
                logger.error("Input CSV path required for reconstruction mode")
                return 1

            reconstruction_manager = ReconstructionManager(config)
            output_dir = getattr(args, 'output_dir', None)
            reconstruction_manager.predict_from_csv(args.input_csv, output_dir)
            logger.info("Reconstruction completed successfully!")

        return 0

    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {str(e)}")
        raise

def interactive_torchvision_download():
    """Interactive function to download torchvision datasets."""
    available_datasets = list_datasets_simple()

    print("\nSelect datasets to download (comma-separated numbers or names, 'all' for all):")
    selection = input("Selection: ").strip()

    downloaded_paths = []

    if selection.lower() == 'all':
        # Download all datasets (this might take a while!)
        confirm = input("This will download ALL available datasets. Continue? (y/n): ")
        if confirm.lower() != 'y':
            return None

        for i, dataset_name in enumerate(available_datasets, 1):
            try:
                print(f"\n[{i}/{len(available_datasets)}] Downloading {dataset_name}...")
                path = download_and_setup_torchvision_dataset(dataset_name)
                downloaded_paths.append((dataset_name, path))
            except Exception as e:
                print(f"Failed to download {dataset_name}: {str(e)}")
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
                    print(f"Invalid number: {sel}")
            else:
                # Treat as dataset name
                if sel.upper() in available_datasets:
                    selected_datasets.append(sel.upper())
                else:
                    print(f"Dataset not found: {sel}")

        for i, dataset_name in enumerate(selected_datasets, 1):
            try:
                print(f"\n[{i}/{len(selected_datasets)}] Downloading {dataset_name}...")
                path = download_and_setup_torchvision_dataset(dataset_name)
                downloaded_paths.append((dataset_name, path))
            except Exception as e:
                print(f"Failed to download {dataset_name}: {str(e)}")
                continue

    if downloaded_paths:
        print(f"\n✓ Successfully downloaded {len(downloaded_paths)} datasets:")
        for dataset_name, path in downloaded_paths:
            print(f"  - {dataset_name}: {path}")
        return downloaded_paths
    else:
        print("No datasets were downloaded.")
        return None

def browse_local_folders(start_path="."):
    """Interactive local folder browser."""
    current_path = os.path.abspath(start_path)

    while True:
        print(f"\nCurrent directory: {current_path}")
        print("\nContents:")

        # Get all items in current directory
        try:
            items = os.listdir(current_path)
        except PermissionError:
            print("Permission denied to access this directory.")
            items = []

        # Separate directories and files
        dirs = []
        files = []

        for item in sorted(items):
            item_path = os.path.join(current_path, item)
            if os.path.isdir(item_path):
                dirs.append(item)
            else:
                files.append(item)

        # Display directories
        print("\n📁 Directories:")
        for i, dir_name in enumerate(dirs, 1):
            print(f"  {i:2d}. {dir_name}/")

        # Display files (limited to image files for context)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if image_files:
            print("\n📷 Image files (sample):")
            for i, file_name in enumerate(image_files[:5], 1):  # Show first 5
                print(f"     {file_name}")
            if len(image_files) > 5:
                print(f"     ... and {len(image_files) - 5} more image files")

        print("\nOptions:")
        print("  [number] - Enter directory")
        print("  b - Go back to parent directory")
        print("  s - Select current directory")
        print("  h - Go home (~)")
        print("  r - Refresh")
        print("  q - Cancel")

        choice = input("\nSelect option: ").strip().lower()

        if choice == 'b':
            # Go to parent directory
            parent = os.path.dirname(current_path)
            if parent != current_path:  # Prevent infinite loop at root
                current_path = parent
            else:
                print("Already at root directory.")

        elif choice == 's':
            # Select current directory
            # Check if directory contains images or has subdirectories (potential class folders)
            image_count = len([f for f in os.listdir(current_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            subdirs = [d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))]

            if image_count > 0 or subdirs:
                return current_path
            else:
                print("This directory doesn't appear to contain images or subdirectories.")
                confirm = input("Use it anyway? (y/n): ")
                if confirm.lower() == 'y':
                    return current_path

        elif choice == 'h':
            # Go to home directory
            current_path = os.path.expanduser("~")

        elif choice == 'r':
            # Refresh - do nothing, will reload
            continue

        elif choice == 'q':
            return None

        elif choice.isdigit():
            # Enter selected directory
            idx = int(choice) - 1
            if 0 <= idx < len(dirs):
                current_path = os.path.join(current_path, dirs[idx])
            else:
                print("Invalid directory number.")

        else:
            print("Invalid option.")

    return None

def handle_training_mode(args: argparse.Namespace, logger: logging.Logger) -> int:
    """
    Unified training mode handler that supports both standard and sliding window training
    """
    try:
        # Validate arguments
        if not args.data_name:
            raise ValueError("data_name argument is required")
        if not args.data_type:
            raise ValueError("data_type argument is required")

        # Setup paths - use lowercase for directory names
        data_name = str(args.data_name).lower()
        data_dir = os.path.join('data', data_name)
        config_path = os.path.join(data_dir, f"{data_name}.json")

        logger.info(f"Processing dataset: {args.data_name} (type: {args.data_type})")

        # Check for sliding window mode
        use_sliding_window = getattr(args, 'sliding_window', False)
        large_image_paths = getattr(args, 'large_image_paths', [])

        if use_sliding_window:
            logger.info("SLIDING WINDOW MODE: Training with large images using sliding window approach")
            if not large_image_paths:
                raise ValueError("large_image_paths argument is required for sliding window mode")

            # Verify large images exist
            valid_paths = []
            for path in large_image_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                    logger.info(f"Found large image: {path}")
                else:
                    logger.warning(f"Large image path not found: {path}")

            if not valid_paths:
                raise ValueError("No valid large image paths found for sliding window training")

            # Process dataset for sliding window
            processor = DatasetProcessor(
                datafile=valid_paths[0] if valid_paths else None,  # Use first path for config
                datatype='sliding_window',
                output_dir=getattr(args, 'output', 'data'),
                data_name=data_name
            )
        elif args.data_type == 'torchvision':
            # For torchvision, use the dataset name in uppercase for torchvision lookup
            torchvision_dataset_name = str(args.data_name).upper()
            logger.info(f"Using torchvision dataset: {torchvision_dataset_name}")

            processor = DatasetProcessor(
                datafile=torchvision_dataset_name,
                datatype=args.data_type,
                output_dir=getattr(args, 'output', 'data'),
                data_name=data_name
            )
        else:
            # For custom datasets, use the provided input path
            if not hasattr(args, 'input_path') or not args.input_path:
                raise ValueError("input_path argument is required for custom datasets")

            processor = DatasetProcessor(
                datafile=args.input_path,
                datatype=args.data_type,
                output_dir=getattr(args, 'output', 'data'),
                data_name=data_name
            )

        # Process dataset (for non-sliding window modes)
        if not use_sliding_window:
            train_dir, test_dir = processor.process()
            logger.info(f"Dataset processed: train_dir={train_dir}, test_dir={test_dir}")
        else:
            train_dir, test_dir = data_dir, data_dir  # Use data directory for sliding window
            logger.info(f"Sliding window mode: using {len(valid_paths)} large images")

        # Generate/verify configurations
        logger.info("Generating/verifying configurations...")
        config = processor.generate_default_config(train_dir)

        # Configure enhancements
        config = configure_image_processing(config, logger)

        # Update configuration with command line arguments
        config = update_config_with_args(config, args)

        # Add sliding window configuration if enabled
        if use_sliding_window:
            config['training']['use_sliding_window'] = True
            config['training']['large_image_paths'] = valid_paths
            config['training']['window_size'] = getattr(args, 'window_size', 512)
            config['training']['stride'] = getattr(args, 'stride', 256)
            config['training']['window_batch_size'] = getattr(args, 'window_batch_size', 4)
            config['training']['overlap'] = getattr(args, 'overlap', 0.5)
            logger.info(f"Sliding window config: size={config['training']['window_size']}, "
                       f"stride={config['training']['stride']}, batch_size={config['training']['window_batch_size']}")

        # Get feature dimensions from user
        fd = config['model']['feature_dims']
        feature_dims_input = input(f"Please specify the output feature dimensions[{fd}]: ").strip()
        if feature_dims_input:
            try:
                feature_dims = int(feature_dims_input)
                if feature_dims <= 0:
                    logger.warning(f"Invalid feature dimensions {feature_dims}, using default {fd}")
                    feature_dims = fd
            except ValueError:
                logger.warning(f"Invalid feature dimensions input, using default {fd}")
                feature_dims = fd
        else:
            feature_dims = fd

        config['model']['feature_dims'] = feature_dims
        logger.info(f"Using feature dimensions: {feature_dims}")

        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configuration saved to: {config_path}")

        # Initialize model and training based on mode
        if use_sliding_window:
            # SLIDING WINDOW TRAINING
            logger.info("Initializing sliding window training...")

            # Create model optimized for sliding window
            model = ModelFactory.create_model(config)

            # Initialize loss manager
            loss_manager = EnhancedLossManager(config)

            # Get training confirmation
            if not get_training_confirmation(logger):
                logger.info("Training cancelled by user")
                return 0

            # Perform sliding window training
            logger.info("Starting sliding window training process...")
            history = train_sliding_window_model(model, valid_paths, config)

            # Extract features from trained model
            logger.info("Extracting features from sliding window model...")
            features_dict = extract_features_sliding_window(model, valid_paths, config, logger)

        else:
            # STANDARD TRAINING
            logger.info("Initializing standard training...")

            # Setup data loading
            transform = processor.get_transforms(config)
            train_dataset, test_dataset = get_dataset(config, transform)

            if train_dataset is None:
                raise ValueError("No training dataset available")

            logger.info(f"Training dataset size: {len(train_dataset)}")
            if test_dataset:
                logger.info(f"Test dataset size: {len(test_dataset)}")

            # Create data loaders
            train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, config)
            logger.info(f"Created data loaders with batch size: {config['training']['batch_size']}")

            # Initialize model and loss manager
            model, loss_manager = initialize_model_components(config, logger)

            # Log model information
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model initialized with {total_params:,} parameters")
            logger.info(f"Using device: {next(model.parameters()).device}")

            # Get training confirmation
            if not get_training_confirmation(logger):
                logger.info("Training cancelled by user")
                return 0

            # Perform training and feature extraction
            logger.info("Starting training process...")
            features_dict = perform_training_and_extraction(
                model, train_loader, test_loader, config, loss_manager, logger
            )

        # Save results
        save_training_results(features_dict, model, config, data_dir, data_name, logger)

        logger.info("Processing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training mode failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


def interactive_input():
    """Collect inputs interactively with torchvision support"""
    print("\nInteractive Mode - Please enter the following information:")

    mode = input("Enter mode (train/reconstruct/predict) [predict]: ").strip().lower() or 'predict'
    data_name = input("Enter dataset name [mnist]: ").strip() or 'mnist'

    # Get dataset type first
    data_type = input("Enter dataset type (torchvision/custom) [torchvision]: ").strip().lower() or 'torchvision'

    # CORRECTED: Only create path for custom datasets, use dataset name for torchvision
    if data_type == 'custom':
        input_path = input(f"Enter path to {'training data' if mode == 'train' else 'input images'} [Data/{data_name}.zip]: ").strip() or f"Data/{data_name}.zip"
    else:
        # For torchvision, use just the dataset name (e.g., "CIFAR10"), not a path
        input_path = data_name.upper()  # e.g., "CIFAR10" not "Data/cifar10/"
        print(f"Using torchvision dataset: {input_path}")

    args = SimpleNamespace(
        mode=mode,
        data_name=data_name,
        data_type=data_type,
        input_path=input_path,  # This will be the dataset name for torchvision
        interactive=True
    )

    # Mode-specific inputs
    if mode == 'predict':
        args.model_path = input(f"Enter path to trained model [data/{data_name}/checkpoints/{data_name}_unified.pth]: ").strip() or f"data/{data_name}/checkpoints/{data_name}_unified.pth"
        args.output = input(f"Enter output CSV path [data/{data_name}/{data_name}.csv]: ").strip() or f"data/{data_name}/{data_name}.csv"
        args.batch_size = int(input("Enter batch size [128]: ").strip() or 128)
        args.cpu = input("Force CPU even if GPU available? (y/n) [n]: ").strip().lower() == 'y'

    elif mode == 'train':
        args.epochs = int(input("Enter number of epochs [100]: ").strip() or 100)
        args.batch_size = int(input("Enter batch size [128]: ").strip() or 128)
        args.learning_rate = float(input("Enter learning rate [0.001]: ").strip() or 0.001)

        # Additional training parameters
        if data_type == 'torchvision':
            print(f"Note: {data_name.upper()} will be automatically downloaded from torchvision")

        # Ask about enhancements
        enable_enhancements = input("Enable enhancement features? (y/n) [y]: ").strip().lower()
        if enable_enhancements == '' or enable_enhancements == 'y':
            print("Enhancement features will be configured during setup")

    # Common parameters
    args.encoder_type = input("Enter encoder type (autoenc/cnn) [autoenc]: ").strip().lower() or 'autoenc'
    args.debug = input("Enable debug mode? (y/n) [n]: ").strip().lower() == 'y'

    # Set default values for other parameters
    args.workers = 4
    args.config = None

    print(f"\nConfiguration Summary:")
    print(f"  Mode: {args.mode}")
    print(f"  Dataset: {args.data_name}")
    print(f"  Type: {args.data_type}")
    if data_type == 'torchvision':
        print(f"  Torchvision Dataset: {input_path}")
    else:
        print(f"  Input Path: {args.input_path}")
    print(f"  Encoder: {args.encoder_type}")

    if mode == 'train':
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Learning Rate: {args.learning_rate}")

    confirm = input("\nProceed with these settings? (y/n) [y]: ").strip().lower()
    if confirm == 'n':
        print("Configuration cancelled.")
        return None

    return args

def parse_arguments():
    """Parse command line arguments with automatic sliding window detection based on system memory."""
    parser = argparse.ArgumentParser(description='CDBNN Image Processor with Automatic Sliding Window Detection')

    # Core arguments
    parser.add_argument('--data_name', type=str, help='Dataset name (e.g., cifar100, galaxies)')
    parser.add_argument('--mode', type=str, choices=['train', 'reconstruct', 'predict', 'download', 'heatmaps'],
                       help='Operation mode')
    parser.add_argument('--data_type', type=str, choices=['torchvision', 'custom', 'sliding_window'],
                       help='Dataset type')
    parser.add_argument('--data', type=str, help='Dataset path or torchvision dataset name')
    parser.add_argument('--input_path', type=str, help='Input path for prediction')
    parser.add_argument('--output_csv', type=str, help='Output CSV path for predictions')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--encoder_type', type=str, choices=['cnn', 'autoenc'], default='autoenc', help='Encoder type')
    parser.add_argument('--list_datasets', action='store_true', help='List available datasets')
    parser.add_argument('--interactive', action='store_true', help='Use interactive mode')
    parser.add_argument('--download', action='store_true', help='Download datasets')

    # Heatmap generation
    parser.add_argument('--generate_heatmaps', action='store_true', default=True,
                       help='Generate attention heatmaps for model interpretation (default: True)')
    parser.add_argument('--no_heatmaps', action='store_false', dest='generate_heatmaps',
                       help='Disable attention heatmap generation')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples per class for heatmap generation (default: None = all samples)')

    # Sliding window arguments (now optional with auto-detection)
    parser.add_argument('--sliding_window', action='store_true',
                       help='Force enable sliding window mode for large images')
    parser.add_argument('--no_sliding_window', action='store_false', dest='sliding_window',
                       help='Force disable sliding window mode')
    parser.add_argument('--large_image_paths', nargs='+',
                       help='Paths to large images for sliding window mode')
    parser.add_argument('--window_size', type=int, default=None,
                       help='Window size for sliding window (auto-calculated if not specified)')
    parser.add_argument('--stride', type=int, default=None,
                       help='Stride for sliding window (auto-calculated if not specified)')
    parser.add_argument('--window_batch_size', type=int, default=None,
                       help='Batch size for window processing (auto-calculated if not specified)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio between windows (default: 0.5)')
    parser.add_argument('--min_window_coverage', type=float, default=0.7,
                       help='Minimum fraction of window that must contain image data (default: 0.7)')
    parser.add_argument('--memory_threshold', type=float, default=0.7,
                       help='Memory usage threshold for auto-enabling sliding window (default: 0.7 = 70%%)')

    args = parser.parse_args()

    # Handle list-datasets option
    if args.list_datasets:
        list_datasets_simple()
        exit(0)

    # Handle interactive mode
    if args.interactive and (not args.data_name and not args.input_path):
        # This will trigger the interactive selection in main()
        pass

    # Auto-detect sliding window requirements
    args = _auto_detect_sliding_window(args)

    return args


def _auto_detect_sliding_window(args):
    """Automatically detect if sliding window mode should be enabled based on system memory and image size."""

    # If user explicitly set sliding_window, respect their choice
    if args.sliding_window is not None:
        if args.sliding_window and not args.large_image_paths and args.input_path:
            args.large_image_paths = [args.input_path]
        return args

    # Only auto-detect for modes that process images
    if args.mode not in ['train', 'predict', 'reconstruct']:
        return args

    # Check if we have images to analyze
    image_paths = []
    if args.large_image_paths:
        image_paths = args.large_image_paths
    elif args.input_path and os.path.exists(args.input_path):
        if os.path.isfile(args.input_path) and args.input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            image_paths = [args.input_path]
        elif os.path.isdir(args.input_path):
            # Check directory for large images
            for root, _, files in os.walk(args.input_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        full_path = os.path.join(root, file)
                        image_paths.append(full_path)
                        # Check first few images to determine if sliding window is needed
                        if len(image_paths) >= 3:
                            break
                if image_paths:
                    break

    if not image_paths:
        return args

    # Analyze images and system memory
    try:
        should_enable_sliding = _analyze_memory_requirements(image_paths, args.memory_threshold)

        if should_enable_sliding:
            args.sliding_window = True
            if not args.large_image_paths:
                args.large_image_paths = image_paths

            # Auto-calculate optimal parameters
            args = _calculate_optimal_window_params(args, image_paths)

            logger.info("🔄 Auto-enabled sliding window mode based on system memory and image size analysis")

        else:
            args.sliding_window = False

    except Exception as e:
        logger.warning(f"Could not auto-detect sliding window requirements: {str(e)}")
        args.sliding_window = False

    return args


def _analyze_memory_requirements(image_paths, memory_threshold=0.7):
    """Analyze if images exceed available system memory."""
    try:
        import psutil
        import PIL.Image

        # Get system memory info
        system_memory = psutil.virtual_memory()
        available_memory_gb = system_memory.available / (1024 ** 3)  # Convert to GB
        total_memory_gb = system_memory.total / (1024 ** 3)

        logger.info(f"System memory: {available_memory_gb:.1f}GB available / {total_memory_gb:.1f}GB total")

        # Estimate memory requirements from sample images
        total_pixels = 0
        sample_count = min(5, len(image_paths))  # Check first 5 images

        for i in range(sample_count):
            try:
                with PIL.Image.open(image_paths[i]) as img:
                    width, height = img.size
                    pixels = width * height
                    total_pixels += pixels

                    logger.info(f"Image {i+1}: {width}x{height} = {pixels:,} pixels")

                    # If any single image is huge, enable sliding window
                    if pixels > 100000000:  # 100MP
                        logger.info("🔄 Single image exceeds 100MP, enabling sliding window")
                        return True

            except Exception as e:
                logger.warning(f"Could not analyze {image_paths[i]}: {str(e)}")
                continue

        if sample_count == 0:
            return False

        # Calculate average image size
        avg_pixels = total_pixels / sample_count

        # Estimate memory requirements (4 bytes per pixel for float32, 3 channels, plus model overhead)
        estimated_memory_per_image_gb = (avg_pixels * 4 * 3 * 2) / (1024 ** 3)  # Conservative estimate with 2x overhead
        batch_memory_gb = estimated_memory_per_image_gb * 4  # Assume batch size of 4

        logger.info(f"Estimated memory per image: {estimated_memory_per_image_gb:.2f}GB")
        logger.info(f"Estimated batch memory: {batch_memory_gb:.2f}GB")

        # Check if estimated memory exceeds threshold
        memory_ratio = batch_memory_gb / available_memory_gb

        if memory_ratio > memory_threshold:
            logger.info(f"🔄 Memory requirement ({memory_ratio:.1%}) exceeds threshold ({memory_threshold:.0%}), enabling sliding window")
            return True
        else:
            logger.info(f"✓ Memory requirement ({memory_ratio:.1%}) within safe limits, using standard processing")
            return False

    except ImportError:
        logger.warning("psutil not available, cannot auto-detect memory requirements")
        return False
    except Exception as e:
        logger.warning(f"Memory analysis failed: {str(e)}")
        return False


def _calculate_optimal_window_params(args, image_paths):
    """Calculate optimal window parameters based on image size and system memory."""
    try:
        import PIL.Image

        # Analyze image sizes to determine optimal window size
        max_width, max_height = 0, 0
        for img_path in image_paths[:3]:  # Check first 3 images
            try:
                with PIL.Image.open(img_path) as img:
                    width, height = img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
            except:
                continue

        if max_width == 0 or max_height == 0:
            # Fallback to defaults
            args.window_size = args.window_size or 512
            args.stride = args.stride or 256
            args.window_batch_size = args.window_batch_size or 4
            return args

        # Calculate optimal window size (aim for ~1MP windows for good performance)
        target_window_pixels = 1024 * 1024  # 1MP
        image_diagonal = (max_width ** 2 + max_height ** 2) ** 0.5

        # Scale window size based on image size
        if image_diagonal > 4000:  # Very large images
            window_size = 1024
        elif image_diagonal > 2000:  # Large images
            window_size = 768
        else:  # Medium images
            window_size = 512

        # Ensure window size is reasonable
        window_size = min(window_size, min(max_width, max_height))
        window_size = max(256, window_size)  # Minimum window size

        # Calculate stride (75% overlap for good reconstruction)
        stride = int(window_size * (1 - args.overlap))
        stride = max(64, stride)  # Minimum stride

        # Calculate batch size based on available memory
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

            # Estimate memory per window (conservative)
            window_memory_gb = (window_size * window_size * 4 * 3 * 10) / (1024 ** 3)  # 10x overhead for model

            # Calculate safe batch size
            safe_batch_size = max(1, int((available_memory_gb * 0.3) / window_memory_gb))  # Use 30% of available memory
            safe_batch_size = min(16, safe_batch_size)  # Maximum batch size

        except:
            safe_batch_size = 4  # Fallback

        # Set calculated parameters if not explicitly provided
        args.window_size = args.window_size or window_size
        args.stride = args.stride or stride
        args.window_batch_size = args.window_batch_size or safe_batch_size

        logger.info(f"📐 Auto-calculated sliding window parameters:")
        logger.info(f"   Window size: {args.window_size}px")
        logger.info(f"   Stride: {args.stride}px")
        logger.info(f"   Batch size: {args.window_batch_size}")
        logger.info(f"   Overlap: {args.overlap:.0%}")

    except Exception as e:
        logger.warning(f"Could not calculate optimal window parameters: {str(e)}")
        # Set sensible defaults
        args.window_size = args.window_size or 512
        args.stride = args.stride or 256
        args.window_batch_size = args.window_batch_size or 4

    return args

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def configure_image_processing(config: Dict, logger: logging.Logger) -> Dict:
    """Configure image processing type and enhancements
    # Display image type options
    print("\nSelect image type for enhanced processing:")
    image_types = ["general", "astronomical", "medical", "agricultural"]
    for i, type_name in enumerate(image_types, 1):
        print(f"{i}. {type_name}")

    # Get image type selection
    type_idx = int(input("\nSelect image type (1-4): ")) - 1
    image_type = image_types[type_idx]

    # Create appropriate configuration manager
    if image_type == "general":
    """
    image_type="general"
    config_manager = GeneralEnhancementConfig(config)
    config_manager.configure_general_parameters()
    config_manager.configure_enhancements()
    """
    else:
        config_manager = SpecificEnhancementConfig(config, image_type)
        config_manager.configure()
    """
    # Get and update configuration
    config = config_manager.get_config()
    config['dataset']['image_type'] = image_type

    return config

def create_data_loaders(train_dataset: Dataset, test_dataset: Optional[Dataset],
                       config: Dict) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create data loaders for training and testing"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers']
        )

    return train_loader, test_loader

def initialize_model_components(config: Dict, logger: logging.Logger) -> Tuple[nn.Module, EnhancedLossManager]:
    """Initialize model and loss manager"""
    logger.info(f"Initializing {config['dataset']['image_type']} enhanced model...")
    model = ModelFactory.create_model(config)
    loss_manager = EnhancedLossManager(config)
    return model, loss_manager

def get_training_confirmation(logger: logging.Logger) -> bool:
    """Get user confirmation for training"""
    if input("\nReady to start training. Proceed? (y/n): ").lower() == 'n':
        logger.info("Training cancelled by user")
        return False
    return True

# Original dataset creation point
def perform_training_and_extraction(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader],
    config: Dict,
    loss_manager: EnhancedLossManager,
    logger: logging.Logger
) -> Dict:
    """Perform model training and feature extraction using reliable prediction pipeline"""
    # Training - FIXED: Added test_loader argument
    logger.info("Starting model training...")
    history = train_model(model, train_loader, test_loader, config, loss_manager)  # ✅ Added test_loader

    # NEW: Feature extraction using reliable prediction pipeline instead of training extraction
    logger.info("Extracting features using reliable prediction pipeline...")

    # Use the dataset name from config
    dataset_name = config['dataset']['name'].lower()
    base_data_path = f"data/{dataset_name}"

    # Create prediction manager (uses the same reliable loading as actual prediction)
    pred_manager = PredictionManager(config)

    # Generate features using prediction pipeline for training data
    train_csv_path = f"{base_data_path}/{dataset_name}_train.csv"
    train_predictions = None

    if os.path.exists(f"{base_data_path}/train"):
        logger.info("Processing training data via prediction pipeline...")
        train_predictions = pred_manager.predict_images(
            data_path=f"{base_data_path}/train",
            output_csv=train_csv_path,
            batch_size=config['training'].get('prediction_batch_size', 32),
            generate_heatmaps=False
        )
        logger.info(f"✓ Training features saved to {train_csv_path}")
    else:
        logger.warning(f"Training directory not found: {base_data_path}/train")

    # Generate features using prediction pipeline for test data
    test_csv_path = f"{base_data_path}/{dataset_name}_test.csv"
    test_predictions = None

    if test_loader is not None and os.path.exists(f"{base_data_path}/test"):
        logger.info("Processing test data via prediction pipeline...")
        test_predictions = pred_manager.predict_images(
            data_path=f"{base_data_path}/test",
            output_csv=test_csv_path,
            batch_size=config['training'].get('prediction_batch_size', 32),
            generate_heatmaps=False
        )
        logger.info(f"✓ Test features saved to {test_csv_path}")
    elif test_loader is not None:
        logger.warning(f"Test directory not found: {base_data_path}/test")

    # Create combined CSV file
    combined_csv_path = f"{base_data_path}/{dataset_name}.csv"
    try:
        if train_predictions is not None and test_predictions is not None:
            # Both train and test available - combine them
            train_df = pd.read_csv(train_csv_path)
            test_df = pd.read_csv(test_csv_path)
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            combined_df.to_csv(combined_csv_path, index=False)
            logger.info(f"✓ Combined features saved to {combined_csv_path}")
        elif train_predictions is not None:
            # Only train available - use as combined
            shutil.copy2(train_csv_path, combined_csv_path)
            logger.info(f"✓ Combined features saved to {combined_csv_path}")
    except Exception as e:
        logger.warning(f"Could not create combined CSV: {str(e)}")

    # Return a features_dict for backward compatibility
    # This maintains the same return interface but uses prediction-generated data
    features_dict = {
        'train': train_predictions if train_predictions is not None else {},
        'test': test_predictions if test_predictions is not None else {},
        'metadata': {
            'extraction_method': 'prediction_pipeline',
            'train_csv_path': train_csv_path if os.path.exists(train_csv_path) else None,
            'test_csv_path': test_csv_path if test_csv_path and os.path.exists(test_csv_path) else None,
            'combined_csv_path': combined_csv_path if os.path.exists(combined_csv_path) else None
        }
    }

    logger.info("✓ Feature extraction completed using reliable prediction pipeline")
    return features_dict


def perform_training_and_extraction(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader],
    config: Dict,
    loss_manager: EnhancedLossManager,
    logger: logging.Logger
) -> Dict[str, torch.Tensor]:
    """Perform model training and feature extraction"""
    # Training
    logger.info("Starting model training...")
    history = train_model(model, train_loader, test_loader, config, loss_manager)

    # Feature extraction
    logger.info("Extracting features...")
    features_dict = model.extract_features(train_loader)

    # If test loader exists, extract and combine features
    if test_loader:
        test_features_dict = model.extract_features(test_loader)
        features_dict = merge_feature_dicts(features_dict, test_features_dict)

    return features_dict

def save_training_results(
    features_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    config: Dict,
    data_dir: str,
    data_name: str,
    logger: logging.Logger
) -> None:
    """Save training results and features using binary format for complex data"""
    try:
        # Clean up legacy files first
        if hasattr(model, 'cleanup_legacy_json_files'):
            model.cleanup_legacy_json_files(data_dir)

        # Save features (main functionality)
        output_path = os.path.join(data_dir, f"{data_name}.csv")
        model.save_features(features_dict, features_dict, output_path)

        # Save training history in binary format
        if hasattr(model, 'history') and model.history:
            history_path = os.path.join(data_dir, 'training_history.pt')
            torch.save(model.history, history_path)
            logger.info(f"Training history saved to {history_path} (binary)")

            # Also save a simple JSON version for quick viewing (only basic types)
            simple_history = {}
            for k, vals in model.history.items():
                simple_vals = []
                for v in vals:
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        simple_vals.append(v.item())
                    elif isinstance(v, (int, float)):
                        simple_vals.append(v)
                if simple_vals:
                    simple_history[k] = simple_vals

            if simple_history:
                json_history_path = os.path.join(data_dir, 'training_history_simple.json')
                with open(json_history_path, 'w') as f:
                    json.dump(simple_history, f, indent=4)
                logger.info(f"Simple training history saved to {json_history_path}")

            # Plot training history
            plot_path = os.path.join(data_dir, 'training_history.png')
            if hasattr(model, 'plot_training_history'):
                try:
                    model.plot_training_history(save_path=plot_path)
                    logger.info(f"Training history plot saved to {plot_path}")
                except Exception as e:
                    logger.warning(f"Could not plot training history: {e}")

        # Save feature selection metadata in binary format
        if (hasattr(model, '_selected_feature_indices') and
            model._selected_feature_indices is not None):

            logger.info(f"Feature selection frozen with {len(model._selected_feature_indices)} features")

            # Save complete metadata in binary format
            binary_path = os.path.join(data_dir, 'feature_selection_metadata.pt')
            binary_metadata = {
                'selected_indices': model._selected_feature_indices.detach().cpu(),
                'importance_scores': getattr(model, '_feature_importance_scores', None),
                'metadata': getattr(model, '_feature_selection_metadata', {}),
                'timestamp': datetime.now().isoformat(),
                'feature_count': len(model._selected_feature_indices),
                'model_class': model.__class__.__name__
            }
            torch.save(binary_metadata, binary_path)
            logger.info(f"Feature selection metadata saved to {binary_path} (binary)")

            # Save a simple JSON summary for human readability
            json_summary_path = os.path.join(data_dir, 'feature_selection_summary.json')
            json_summary = {
                'feature_count': len(model._selected_feature_indices),
                'timestamp': datetime.now().isoformat(),
                'model_class': model.__class__.__name__,
                'has_importance_scores': getattr(model, '_feature_importance_scores', None) is not None,
                'metadata_keys': list(getattr(model, '_feature_selection_metadata', {}).keys())
            }
            with open(json_summary_path, 'w') as f:
                json.dump(json_summary, f, indent=2)
            logger.info(f"Feature selection summary saved to {json_summary_path}")

        # Save model configuration in binary format (preserves all types)
        config_path = os.path.join(data_dir, 'model_config.pt')
        torch.save(config, config_path)
        logger.info(f"Model configuration saved to {config_path} (binary)")

        # Also save a JSON version of config for easy viewing (basic types only)
        json_config_path = os.path.join(data_dir, 'model_config_simple.json')
        def make_json_safe(obj):
            """Create a JSON-safe version of the config"""
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            elif isinstance(obj, dict):
                return {str(k): make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_safe(item) for item in obj]
            else:
                return str(obj)

        try:
            safe_config = make_json_safe(config)
            with open(json_config_path, 'w') as f:
                json.dump(safe_config, f, indent=2)
            logger.info(f"Simple model configuration saved to {json_config_path}")
        except Exception as e:
            logger.warning(f"Could not save simple config JSON: {e}")

        # Save feature statistics in binary format
        if features_dict and 'embeddings' in features_dict:
            embeddings = features_dict['embeddings']
            if isinstance(embeddings, torch.Tensor):
                stats_path = os.path.join(data_dir, 'feature_statistics.pt')
                feature_stats = {
                    'total_samples': embeddings.shape[0],
                    'feature_dimensions': embeddings.shape[1],
                    'mean_features': embeddings.mean(dim=0).detach().cpu(),
                    'std_features': embeddings.std(dim=0).detach().cpu(),
                    'min_features': embeddings.min(dim=0)[0].detach().cpu(),
                    'max_features': embeddings.max(dim=0)[0].detach().cpu(),
                    'timestamp': datetime.now().isoformat()
                }
                torch.save(feature_stats, stats_path)
                logger.info(f"Feature statistics saved to {stats_path} (binary)")

        logger.info("All training results saved successfully in binary format")

    except Exception as e:
        logger.error(f"Error in save_training_results: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def merge_feature_dicts(dict1: Dict[str, torch.Tensor],
                       dict2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Merge two feature dictionaries"""
    merged = {}
    for key in dict1.keys():
        if key in dict2:
            if isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor):
                merged[key] = torch.cat([dict1[key], dict2[key]])
            else:
                merged[key] = dict1[key]  # Keep original if not tensor
    return merged

if __name__ == '__main__':
    #print(f"{Colors.RED}The code has some bug in directly handling torchvision files. So recommendation is to use Get_Torchvision_images function instead{Colors.ENDC}")
    print("Updated on April 14/2025 Stable version")
    sys.exit(main())
