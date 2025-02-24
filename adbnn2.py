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
#--------------------------------------------------DBNN Code  Starts----------------------------------------------
import torch
import json
import numpy as np
from itertools import combinations
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

def save_label_encoder(label_encoder, dataset_name):
    """
    Save the LabelEncoder object to the data folder.

    Args:
        label_encoder: The LabelEncoder object
        dataset_name: Name of the dataset (e.g., 'mnist')
    """
    encoder_path = os.path.join("data", dataset_name, f"{dataset_name}_label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"LabelEncoder saved to {encoder_path}")

def load_label_encoder(dataset_name):
    """
    Load the LabelEncoder object from the data folder.

    Args:
        dataset_name: Name of the dataset (e.g., 'mnist')

    Returns:
        label_encoder: The LabelEncoder object
    """
    encoder_path = os.path.join("data", dataset_name, f"{dataset_name}_label_encoder.pkl")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"LabelEncoder file not found: {encoder_path}")

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder

def load_config(dataset_name):
    """
    Load configuration from <dataset_name>.conf file.

    Args:
        dataset_name: Name of the dataset (e.g., 'mnist')

    Returns:
        config: Dictionary containing configuration parameters
    """
    config_path = os.path.join("data", dataset_name, f"{dataset_name}.conf")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Validate column_names and target_column
    column_names = config.get("column_names", [])
    target_column = config.get("target_column", "target")

    if not column_names:
        raise ValueError("column_names must be specified in the configuration file")

    if target_column not in column_names:
        raise ValueError(f"Target column '{target_column}' not found in column_names")

    return config

import torch
import numpy as np
from itertools import combinations
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import json
import os
import pickle

class DBNN:
    def __init__(self, config, device='cuda'):
        """
        Initialize the DBNN model.

        Args:
            config: Dictionary containing configuration parameters.
            device: Device to use ('cuda' or 'cpu').
        """
        self.config = config
        self.device = device

        # Extract parameters from config
        self.n_bins_per_dim = config.get("training_params", {}).get("n_bins_per_dim", 20)
        self.invert_DBNN = config.get("training_params", {}).get("invert_DBNN", False)
        self.lr = config.get("training_params", {}).get("learning_rate", 0.1)
        self.max_epochs = config.get("training_params", {}).get("epochs", 1000)
        self.batch_size = config.get("training_params", {}).get("batch_size", 32)
        self.initial_samples_per_class = config.get("training_params", {}).get("initial_samples_per_class", 5)
        self.margin = config.get("active_learning", {}).get("marginal_margin_threshold", 0.1)

        # Initialize uniform priors (weights)
        self.W = None  # Will be initialized after n_classes is known

        # Initialize pairwise likelihood parameters
        self.likelihood_params = {
            'bin_edges': [],  # Bin edges for each feature pair
            'bin_probs': []   # Bin probabilities for each feature pair and class
        }

        # Initialize label encoder
        self.label_encoder = LabelEncoder()

        # Initialize inverse DBNN (will be set after n_features is known)
        self.inverse_model = None

    def initialize_from_data(self, X, y):
        """
        Initialize model parameters based on the dataset.

        Args:
            X: Input features (n_samples, n_features).
            y: Target labels (n_samples,).
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        self.n_classes = len(self.label_encoder.classes_)  # Number of classes
        self.n_features = X.shape[1]  # Number of features

        # Initialize uniform priors (weights)
        self.W = torch.ones(self.n_classes, device=self.device) / self.n_classes

        # Initialize inverse DBNN if enabled
        if self.invert_DBNN:
            self.inverse_model = InvertibleDBNN(
                forward_model=self,
                feature_dims=self.n_features,
                reconstruction_weight=self.config.get("training_params", {}).get("reconstruction_weight", 0.5),
                feedback_strength=self.config.get("training_params", {}).get("feedback_strength", 0.3),
                debug=False
            )

        return y_encoded

    def compute_pairwise_likelihood(self, X, y):
        """
        Compute pairwise likelihood parameters from training data.

        Args:
            X: Input features (n_samples, n_features).
            y: Target labels (n_samples,).
        """
        n_samples, n_features = X.shape
        feature_pairs = list(combinations(range(n_features), 2))  # All unique feature pairs

        # Store feature pairs for later use
        self.likelihood_params['feature_pairs'] = feature_pairs


        for pair in feature_pairs:
            # Extract data for the current feature pair
            pair_data = X[:, pair]  # Shape: (n_samples, 2)


            # Ensure pair_data is 2-dimensional
            if pair_data.dim() == 1:
                pair_data = pair_data.unsqueeze(1)  # Convert to 2D if 1D


            # Compute bin edges and probabilities
            bin_edges = self._compute_bin_edges(pair_data)
            bin_probs = self._compute_bin_probs(pair_data, y, bin_edges)

            # Store likelihood parameters
            self.likelihood_params['bin_edges'].append(bin_edges)
            self.likelihood_params['bin_probs'].append(bin_probs)

    def compute_posterior(self, X):
        """
        Compute posterior probabilities for each class.

        Args:
            X: Input features (n_samples, n_features).

        Returns:
            posteriors: Posterior probabilities (n_samples, n_classes).
        """
        n_samples = X.shape[0]
        log_posteriors = torch.zeros(n_samples, self.n_classes, device=self.device)


        # Compute joint likelihood approximation using pairwise likelihoods
        for pair_idx, (bin_edges, bin_probs) in enumerate(zip(
            self.likelihood_params['bin_edges'],
            self.likelihood_params['bin_probs']
        )):
            # Extract data for the current feature pair
            pair_data = X[:, self.likelihood_params['feature_pairs'][pair_idx]]  # Correct extraction


            # Ensure pair_data is 2-dimensional
            if pair_data.dim() == 1:
                pair_data = pair_data.unsqueeze(1)


            # Compute bin indices
            bin_indices = self._compute_bin_indices(pair_data, bin_edges)


            # Add log likelihood for each class
            for c in range(self.n_classes):
                log_posteriors[:, c] += torch.log(bin_probs[c][bin_indices[:, 0], bin_indices[:, 1]] + 1e-10)

        # Add log prior
        log_posteriors += torch.log(self.W)

        # Normalize to get posterior probabilities
        posteriors = torch.softmax(log_posteriors, dim=1)
        return posteriors

    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Adaptive training loop using parameters from the configuration file.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training labels (n_samples,).
            X_test: Test features (optional).
            y_test: Test labels (optional).
        """
        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.long, device=self.device)

        # Step 1: Select initial divergent examples for each class
        initial_indices = self._select_divergent_examples(X_train, y_train, self.initial_samples_per_class)
        X_initial = X_train[initial_indices]
        y_initial = y_train[initial_indices]

        # Step 2: Train on initial subset
        self._train_on_subset(X_initial, y_initial)

        # Step 3: Adaptive learning loop
        remaining_indices = torch.tensor([i for i in range(len(X_train)) if i not in initial_indices], device=self.device)
        while len(remaining_indices) > 0:
            # Predict on remaining data
            X_remaining = X_train[remaining_indices]
            y_remaining = y_train[remaining_indices]
            posteriors = self.compute_posterior(X_remaining)
            pred_classes = torch.argmax(posteriors, dim=1)

            # Identify misclassified examples
            misclassified = pred_classes != y_remaining
            if not misclassified.any():
                break

            # Select most informative misclassified examples
            new_indices = self._select_informative_examples(X_remaining, y_remaining, posteriors, pred_classes, self.margin)
            if len(new_indices) == 0:
                break

            # Add new examples to training set
            X_initial = torch.cat([X_initial, X_remaining[new_indices]])
            y_initial = torch.cat([y_initial, y_remaining[new_indices]])

            # Train on updated subset
            self._train_on_subset(X_initial, y_initial)

            # Update remaining indices
            remaining_indices = remaining_indices[~misclassified]

    def _train_on_subset(self, X, y):
        """
        Train on a subset of data.

        Args:
            X: Input features (n_samples, n_features).
            y: Target labels (n_samples,).
        """
        # Compute pairwise likelihood parameters
        self.compute_pairwise_likelihood(X, y)

        # Create DataLoader for efficient batching
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in tqdm(range(self.max_epochs)):
            for X_batch, y_batch in train_loader:
                # Compute posteriors
                posteriors = self.compute_posterior(X_batch)

                # Get predicted classes
                pred_classes = torch.argmax(posteriors, dim=1)

                # Identify misclassified examples
                misclassified = pred_classes != y_batch
                if not misclassified.any():
                    continue

                # Update weights for misclassified examples
                for idx in torch.where(misclassified)[0]:
                    true_class = y_batch[idx]
                    pred_class = pred_classes[idx]

                    P1 = posteriors[idx, true_class]
                    P2 = posteriors[idx, pred_class]

                    # Update prior (weight)
                    self.W[true_class] += self.lr * (1 - P1 / P2)

            # Normalize weights to maintain uniform prior
            self.W = self.W / self.W.sum()

    def _select_divergent_examples(self, X, y, samples_per_class):
        """
        Select divergent examples for each class.

        Args:
            X: Input features (n_samples, n_features).
            y: Target labels (n_samples,).
            samples_per_class: Number of divergent examples to select per class.

        Returns:
            indices: Indices of selected examples.
        """
        indices = []
        for c in range(self.n_classes):
            class_indices = torch.where(y == c)[0]
            class_data = X[class_indices]

            # Compute pairwise distances
            distances = torch.cdist(class_data, class_data)
            mean_distances = distances.mean(dim=1)

            # Select top samples with maximum divergence
            top_indices = torch.topk(mean_distances, samples_per_class).indices
            indices.extend(class_indices[top_indices].tolist())

        return torch.tensor(indices, device=self.device)

    def _select_informative_examples(self, X, y, posteriors, pred_classes, margin):
        """
        Select informative misclassified examples.

        Args:
            X: Input features (n_samples, n_features).
            y: Target labels (n_samples,).
            posteriors: Posterior probabilities (n_samples, n_classes).
            pred_classes: Predicted classes (n_samples,).
            margin: Margin for selecting informative examples.

        Returns:
            indices: Indices of selected examples.
        """
        misclassified = torch.where(pred_classes != y)[0]
        if len(misclassified) == 0:
            return torch.tensor([], device=self.device)

        # Compute posterior probabilities for misclassified examples
        true_probs = posteriors[misclassified, y[misclassified]]
        pred_probs = posteriors[misclassified, pred_classes[misclassified]]

        # Select examples with maximum posterior probability for the wrong class
        max_pred_probs = pred_probs.max()
        informative_mask = (pred_probs >= max_pred_probs - margin)
        return misclassified[informative_mask]

    def _compute_bin_edges(self, pair_data):
        """
        Compute bin edges for a feature pair.

        Args:
            pair_data: Input data for a feature pair (n_samples, 2).

        Returns:
            bin_edges: Bin edges for the feature pair (2, n_bins_per_dim + 1).
        """
        min_val = pair_data.min(dim=0).values
        max_val = pair_data.max(dim=0).values
        padding = (max_val - min_val) * 0.01
        bin_edges = torch.stack([
            torch.linspace(min_val[i] - padding[i], max_val[i] + padding[i], self.n_bins_per_dim + 1)
            for i in range(2)
        ]).to(self.device)
        return bin_edges.contiguous()

    def _compute_bin_probs(self, pair_data, y, bin_edges):
        """
        Compute bin probabilities for each class.

        Args:
            pair_data: Input data for a feature pair (n_samples, 2).
            y: Target labels (n_samples,).
            bin_edges: Bin edges for the feature pair (2, n_bins_per_dim + 1).

        Returns:
            bin_probs: Bin probabilities for each class (n_classes, n_bins_per_dim, n_bins_per_dim).
        """
        bin_probs = torch.zeros(self.n_classes, self.n_bins_per_dim, self.n_bins_per_dim, device=self.device)

        for c in range(self.n_classes):
            class_data = pair_data[y == c]

            # Ensure class_data is 2-dimensional
            if class_data.dim() == 1:
                class_data = class_data.unsqueeze(1)

            # Convert bin_edges to a tuple of tensors
            bins_tuple = tuple(bin_edges[i] for i in range(bin_edges.shape[0]))

            # Compute histogram using histogramdd
            bin_counts = torch.histogramdd(class_data, bins=bins_tuple)[0]

            # Apply Laplace smoothing
            bin_probs[c] = (bin_counts + 1) / (class_data.shape[0] + self.n_bins_per_dim ** 2)


        return bin_probs

    def _compute_bin_indices(self, pair_data, bin_edges):
        """
        Compute bin indices for a feature pair.

        Args:
            pair_data: Input data for a feature pair (n_samples, 2).
            bin_edges: Bin edges for the feature pair (2, n_bins_per_dim + 1).

        Returns:
            bin_indices: Bin indices for the feature pair (n_samples, 2).
        """
        # Debug: Verify pair_data shape
        print(f"pair_data shape in _compute_bin_indices: {pair_data.shape}")  # Debug

        # Ensure pair_data is 2-dimensional
        if pair_data.dim() == 1:
            pair_data = pair_data.unsqueeze(1)

        # Debug: Verify pair_data shape after reshaping
        print(f"pair_data shape after reshaping in _compute_bin_indices: {pair_data.shape}")  # Debug

        # Ensure pair_data is contiguous
        pair_data = pair_data.contiguous()

        # Move tensors to CPU for torch.bucketize
        pair_data_cpu = pair_data.cpu()
        bin_edges_cpu = bin_edges.cpu()

        # Compute bin indices for each feature in the pair
        bin_indices = torch.stack([
            torch.bucketize(pair_data_cpu[:, i], bin_edges_cpu[i].contiguous()) - 1
            for i in range(2)
        ]).t()  # Transpose to get shape [n_samples, 2]

        # Move bin_indices back to the original device
        bin_indices = bin_indices.to(pair_data.device)

        # Debug: Verify bin_indices shape
        print(f"bin_indices shape: {bin_indices.shape}")  # Debug

        return bin_indices

class InvertibleDBNN(torch.nn.Module):
    def __init__(self, forward_model, feature_dims, reconstruction_weight=0.5, feedback_strength=0.3, debug=False):
        """
        Initialize the InvertibleDBNN model.

        Args:
            forward_model: The forward DBNN model.
            feature_dims: Number of input feature dimensions.
            reconstruction_weight: Weight for reconstruction loss (0-1).
            feedback_strength: Strength of reconstruction feedback (0-1).
            debug: Enable debug logging.
        """
        super(InvertibleDBNN, self).__init__()
        self.forward_model = forward_model
        self.device = forward_model.device
        self.feature_dims = feature_dims
        self.reconstruction_weight = reconstruction_weight
        self.feedback_strength = feedback_strength

        # Initialize model components
        self.n_classes = len(self.forward_model.label_encoder.classes_)
        self.inverse_likelihood_params = None
        self.inverse_feature_pairs = None

        # Feature scaling parameters as buffers
        self.register_buffer('min_vals', None)
        self.register_buffer('max_vals', None)
        self.register_buffer('scale_factors', None)

        # Initialize all components
        self._initialize_inverse_components()

    def _initialize_inverse_components(self):
        """
        Initialize inverse model parameters with proper buffer handling.
        """
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

    def reconstruct_features(self, class_probs):
        """
        Reconstruct input features from class probabilities.

        Args:
            class_probs: Class probabilities (n_samples, n_classes).

        Returns:
            reconstructed_features: Reconstructed features (n_samples, feature_dims).
        """
        with torch.no_grad():
            # Ensure class_probs is 2D
            if class_probs.dim() == 1:
                class_probs = class_probs.unsqueeze(1)

            # Compute inverse posterior
            reconstructed_features = self._compute_inverse_posterior(class_probs)

            # Unscale features if scaling parameters are available
            if hasattr(self, 'min_vals') and self.min_vals is not None:
                reconstructed_features = self._unscale_features(reconstructed_features)

            return reconstructed_features

    def _compute_inverse_posterior(self, class_probs):
        """
        Compute inverse posterior to reconstruct features.

        Args:
            class_probs: Class probabilities (n_samples, n_classes).

        Returns:
            reconstructed_features: Reconstructed features (n_samples, feature_dims).
        """
        batch_size = class_probs.shape[0]
        reconstructed_features = torch.zeros(
            (batch_size, self.feature_dims),
            device=self.device,
            dtype=class_probs.dtype
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

    def _compute_reconstruction_loss(self, original_features, reconstructed_features, reduction='mean'):
        """
        Compute reconstruction loss between original and reconstructed features.

        Args:
            original_features: Original input features (n_samples, feature_dims).
            reconstructed_features: Reconstructed features (n_samples, feature_dims).
            reduction: Reduction method for the loss ('mean' or 'sum').

        Returns:
            loss: Reconstruction loss.
        """
        # Scale features if necessary
        if hasattr(self, 'min_vals') and self.min_vals is not None:
            original_features = self._scale_features(original_features)
            reconstructed_features = self._scale_features(reconstructed_features)

        # Compute mean squared error (MSE) loss
        mse_loss = torch.mean((original_features - reconstructed_features) ** 2, dim=1)

        if reduction == 'mean':
            return mse_loss.mean()
        elif reduction == 'sum':
            return mse_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction method: {reduction}")

    def _scale_features(self, features):
        """
        Scale features to [0, 1] range.

        Args:
            features: Input features (n_samples, feature_dims).

        Returns:
            scaled_features: Scaled features (n_samples, feature_dims).
        """
        return (features - self.min_vals) / self.scale_factors

    def _unscale_features(self, scaled_features):
        """
        Convert scaled features back to original range.

        Args:
            scaled_features: Scaled features (n_samples, feature_dims).

        Returns:
            unscaled_features: Unscaled features (n_samples, feature_dims).
        """
        return (scaled_features * self.scale_factors) + self.min_vals

    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train the inverse DBNN model.

        Args:
            X_train: Training features (n_samples, feature_dims).
            y_train: Training labels (n_samples,).
            X_test: Test features (optional).
            y_test: Test labels (optional).
        """
        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.long, device=self.device)

        # Compute feature scaling parameters
        self._compute_feature_scaling(X_train)

        # Create DataLoader for efficient batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.forward_model.batch_size, shuffle=True)

        # Initialize optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.forward_model.lr)

        # Training loop
        for epoch in tqdm(range(self.forward_model.max_epochs)):
            for X_batch, y_batch in train_loader:
                # Forward pass: Compute class probabilities
                posteriors = self.forward_model.compute_posterior(X_batch)

                # Reconstruct features
                reconstructed_features = self.reconstruct_features(posteriors)

                # Compute reconstruction loss
                loss = self._compute_reconstruction_loss(X_batch, reconstructed_features)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print loss for monitoring
            print(f"Epoch {epoch + 1}/{self.forward_model.max_epochs}, Loss: {loss.item()}")

    def _compute_feature_scaling(self, features):
        """
        Compute feature scaling parameters for consistent reconstruction.

        Args:
            features: Input features (n_samples, feature_dims).
        """
        with torch.no_grad():
            self.min_vals = features.min(dim=0)[0]
            self.max_vals = features.max(dim=0)[0]
            self.scale_factors = self.max_vals - self.min_vals
            self.scale_factors[self.scale_factors == 0] = 1.0  # Avoid division by zero


#---------------------------------------------------DBNN Code  Ends----------------------------------------------
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
                "minimum_training_accuracy": 0.95,  # Added default value
                "enable_vectorized": False,  # Default to classic training
                "vectorization_warning_acknowledged": False  # Track if user has acknowledged
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
#----------------------------------------------------------------------------------------------------------
import os
import json
import pandas as pd
import torch
#from dbnn import DBNN  # Assuming the DBNN class is in a file named dbnn.py

def load_config(dataset_name):
    """
    Load configuration from <dataname>.conf file.
    """
    config_path = os.path.join("data", dataset_name, f"{dataset_name}.conf")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_dataset(dataset_name, config):
    """
    Load dataset from <dataname>.csv file.
    """
    data_path = os.path.join("data", dataset_name, f"{dataset_name}.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Read CSV file
    df = pd.read_csv(data_path, sep=config.get("separator", ","))

    # Ensure target column exists
    target_column = config.get("target_column", "target")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    return df, target_column

def main(dataset_name):
    """
    Main function to train and predict on a dataset using the DBNN model.

    Args:
        dataset_name: Name of the dataset (e.g., 'mnist').
    """
    # Load configuration
    config = load_config(dataset_name)

    # Load dataset
    data_path = os.path.join("data", dataset_name, f"{dataset_name}.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Read the dataset
    df = pd.read_csv(data_path, sep=config.get("separator", ","))

    # Filter columns based on configuration
    column_names = config.get("column_names", [])
    target_column = config.get("target_column", "target")

    # Ensure target_column is in column_names
    if target_column not in column_names:
        raise ValueError(f"Target column '{target_column}' not found in column_names")

    # Filter the dataset to include only the specified columns
    df = df[column_names]


    # Separate features and target
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values


    # Initialize DBNN model
    dbnn = DBNN(config, device="cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model parameters based on the dataset and encode labels
    y_encoded = dbnn.initialize_from_data(X, y)

    # Save the LabelEncoder for later use
    save_label_encoder(dbnn.label_encoder, dataset_name)

    # Train the model using encoded labels
    dbnn.train(X, y_encoded)

    # Predict on the entire dataset
    posteriors = dbnn.compute_posterior(torch.tensor(X, dtype=torch.float32, device=dbnn.device))
    pred_classes = torch.argmax(posteriors, dim=1).cpu().numpy()

    # Decode predictions to original labels
    pred_labels = dbnn.label_encoder.inverse_transform(pred_classes)

    # Save predictions
    df["predicted_class"] = pred_labels
    output_path = os.path.join("data", dataset_name, f"{dataset_name}_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and predict using DBNN.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset (subfolder in data/)")
    args = parser.parse_args()

    # Run main function
    main(args.dataset_name)
