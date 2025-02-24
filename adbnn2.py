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
import os, re
import json
from itertools import combinations
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import normaltest
import pickle
import configparser
import traceback
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import torch.amp

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
def compute_feature_ranges(X, dataset_name):
    """
    Compute and save the min and max values for each feature in the dataset.

    Args:
        X: Input features (n_samples, n_features).
        dataset_name: Name of the dataset (e.g., 'mnist').

    Returns:
        min_vals: Min values for each feature (n_features,).
        max_vals: Max values for each feature (n_features,).
    """
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # Save min and max values to a .pkl file
    ranges_path = os.path.join("data", dataset_name, f"{dataset_name}_feature_ranges.pkl")
    with open(ranges_path, "wb") as f:
        pickle.dump({"min_vals": min_vals, "max_vals": max_vals}, f)

    print(f"Feature ranges saved to {ranges_path}")
    return min_vals, max_vals

def load_feature_ranges(dataset_name):
    """
    Load the min and max values for each feature from a .pkl file.

    Args:
        dataset_name: Name of the dataset (e.g., 'mnist').

    Returns:
        min_vals: Min values for each feature (n_features,).
        max_vals: Max values for each feature (n_features,).
    """
    ranges_path = os.path.join("data", dataset_name, f"{dataset_name}_feature_ranges.pkl")
    if not os.path.exists(ranges_path):
        raise FileNotFoundError(f"Feature ranges file not found: {ranges_path}")

    with open(ranges_path, "rb") as f:
        ranges = pickle.load(f)

    return ranges["min_vals"], ranges["max_vals"]

def scale_features(X, min_vals, max_vals):
    """
    Scale features to the range [0, 1] using min and max values.

    Args:
        X: Input features (n_samples, n_features).
        min_vals: Min values for each feature (n_features,).
        max_vals: Max values for each feature (n_features,).

    Returns:
        X_scaled: Scaled features (n_samples, n_features).
    """
    # Avoid division by zero
    scale_factors = max_vals - min_vals
    scale_factors[scale_factors == 0] = 1.0

    # Scale features
    X_scaled = (X - min_vals) / scale_factors
    return X_scaled



class DBNN:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.n_bins_per_dim = config.get("training_params", {}).get("n_bins_per_dim", 21)
        self.invert_DBNN = config.get("training_params", {}).get("invert_DBNN", True)
        self.lr = config.get("training_params", {}).get("learning_rate", 0.1)
        self.max_epochs = config.get("training_params", {}).get("epochs", 1000)
        self.batch_size = config.get("training_params", {}).get("batch_size", 32)
        self.initial_samples_per_class = config.get("training_params", {}).get("initial_samples_per_class", 5)
        self.margin = config.get("active_learning", {}).get("marginal_margin_threshold", 0.1)
        self.W = None
        self.likelihood_params = {'bin_edges': [], 'bin_probs': []}
        self.label_encoder = LabelEncoder()
        self.inverse_model = None
        self.min_vals = None
        self.max_vals = None

    def initialize_from_data(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)
        self.n_classes = len(self.label_encoder.classes_)
        self.n_features = X.shape[1]

        # Compute and save feature ranges
        self.min_vals, self.max_vals = compute_feature_ranges(X, self.config["dataset_name"])

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

    def scale_features(self, X):
        """
        Scale features to the range [0, 1] using saved min and max values.

        Args:
            X: Input features (n_samples, n_features).

        Returns:
            X_scaled: Scaled features (n_samples, n_features).
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Feature ranges (min_vals and max_vals) must be initialized before scaling features.")

        return scale_features(X, self.min_vals, self.max_vals)

    def compute_pairwise_likelihood(self, X, y):
        n_samples, n_features = X.shape
        feature_pairs = list(combinations(range(n_features), 2))
        self.likelihood_params['feature_pairs'] = feature_pairs
        for pair in feature_pairs:
            pair_data = X[:, pair]
            if pair_data.dim() == 1:
                pair_data = pair_data.unsqueeze(1)
            bin_edges = self._compute_bin_edges(pair_data)
            bin_probs = self._compute_bin_probs(pair_data, y, bin_edges)
            self.likelihood_params['bin_edges'].append(bin_edges)
            self.likelihood_params['bin_probs'].append(bin_probs)

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
        padding = (max_val - min_val) * 0.01  # Add 1% padding to avoid edge cases
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

            # Move tensors to CPU for torch.histogramdd
            class_data_cpu = class_data.cpu()
            bin_edges_cpu = bin_edges.cpu()

            # Convert bin_edges to a tuple of tensors
            bins_tuple = tuple(bin_edges_cpu[i] for i in range(bin_edges_cpu.shape[0]))

            # Compute histogram using histogramdd on CPU
            bin_counts = torch.histogramdd(class_data_cpu, bins=bins_tuple)[0]

            # Move bin_counts back to the original device
            bin_counts = bin_counts.to(self.device)

            # Apply Laplace smoothing
            bin_probs[c] = (bin_counts + 1) / (class_data.shape[0] + self.n_bins_per_dim ** 2)

        # Debug: Print bin_probs
        print(f"bin_probs shape: {bin_probs.shape}")
        print(f"bin_probs: {bin_probs}")

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
        # Ensure pair_data is 2-dimensional and contiguous
        if pair_data.dim() == 1:
            pair_data = pair_data.unsqueeze(1)
        pair_data = pair_data.contiguous()

        # Move tensors to CPU for torch.bucketize
        pair_data_cpu = pair_data.cpu()
        bin_edges_cpu = bin_edges.cpu()

        # Compute bin indices for each feature in the pair
        bin_indices = torch.stack([
            torch.bucketize(pair_data_cpu[:, i], bin_edges_cpu[i].contiguous()) - 1
            for i in range(2)
        ]).t()  # Transpose to get shape [n_samples, 2]

        # Clamp bin_indices to valid range [0, n_bins_per_dim - 1]
        bin_indices = torch.clamp(bin_indices, 0, self.n_bins_per_dim - 1)

        # Debug: Print bin_indices
        print(f"bin_indices after clamping: {bin_indices}")

        # Move bin_indices back to the original device
        bin_indices = bin_indices.to(pair_data.device)

        return bin_indices

    def compute_posterior(self, X):
        """
        Compute posterior probabilities for each class given the input features.

        Args:
            X: Input features (n_samples, n_features).

        Returns:
            posteriors: Posterior probabilities (n_samples, n_classes).
        """
        n_samples = X.shape[0]
        posteriors = torch.zeros((n_samples, self.n_classes), device=self.device)

        # Compute the log-posterior for each class
        for c in range(self.n_classes):
            log_posterior = torch.log(self.W[c])  # Log of prior

            # Accumulate the log-likelihood from each feature pair
            for i, pair in enumerate(self.likelihood_params['feature_pairs']):
                pair_data = X[:, pair]
                bin_indices = self._compute_bin_indices(pair_data, self.likelihood_params['bin_edges'][i])
                bin_probs = self.likelihood_params['bin_probs'][i][c]

                # Debug: Print shapes and values
                print(f"bin_indices shape: {bin_indices.shape}")
                print(f"bin_probs shape: {bin_probs.shape}")
                print(f"bin_indices: {bin_indices}")
                print(f"bin_probs: {bin_probs}")

                # Ensure bin_indices are within valid range
                valid_indices = (bin_indices[:, 0] >= 0) & (bin_indices[:, 0] < self.n_bins_per_dim) & \
                                (bin_indices[:, 1] >= 0) & (bin_indices[:, 1] < self.n_bins_per_dim)

                # Use valid indices to compute log-likelihood
                if valid_indices.any():
                    log_posterior[valid_indices] += torch.log(bin_probs[bin_indices[valid_indices, 0], bin_indices[valid_indices, 1]])

            # Store the log-posterior for this class
            posteriors[:, c] = log_posterior

        # Convert log-posteriors to probabilities using softmax
        posteriors = torch.softmax(posteriors, dim=1)
        return posteriors

    def train(self, X_train, y_train, X_test=None, y_test=None):
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.long, device=self.device)
        initial_indices = self._select_divergent_examples(X_train, y_train, self.initial_samples_per_class)
        X_initial = X_train[initial_indices]
        y_initial = y_train[initial_indices]
        self._train_on_subset(X_initial, y_initial)
        remaining_indices = torch.tensor([i for i in range(len(X_train)) if i not in initial_indices], device=self.device)
        while len(remaining_indices) > 0:
            X_remaining = X_train[remaining_indices]
            y_remaining = y_train[remaining_indices]
            posteriors = self.compute_posterior(X_remaining)
            pred_classes = torch.argmax(posteriors, dim=1)
            misclassified = pred_classes != y_remaining
            if not misclassified.any():
                break
            new_indices = self._select_informative_examples(X_remaining, y_remaining, posteriors, pred_classes, self.margin)
            if len(new_indices) == 0:
                break
            X_initial = torch.cat([X_initial, X_remaining[new_indices]])
            y_initial = torch.cat([y_initial, y_remaining[new_indices]])
            self._train_on_subset(X_initial, y_initial)
            remaining_indices = remaining_indices[~misclassified]

    def _train_on_subset(self, X, y):
        self.compute_pairwise_likelihood(X, y)
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in tqdm(range(self.max_epochs)):
            for X_batch, y_batch in train_loader:
                posteriors = self.compute_posterior(X_batch)
                pred_classes = torch.argmax(posteriors, dim=1)
                misclassified = pred_classes != y_batch
                if not misclassified.any():
                    continue
                for idx in torch.where(misclassified)[0]:
                    true_class = y_batch[idx]
                    pred_class = pred_classes[idx]
                    P1 = posteriors[idx, true_class]
                    P2 = posteriors[idx, pred_class]
                    self.W[true_class] += self.lr * (1 - P1 / P2)
                self.W = self.W / self.W.sum()

    def _select_divergent_examples(self, X, y, samples_per_class):
        indices = []
        for c in range(self.n_classes):
            class_indices = torch.where(y == c)[0]
            class_data = X[class_indices]
            distances = torch.cdist(class_data, class_data)
            mean_distances = distances.mean(dim=1)
            top_indices = torch.topk(mean_distances, samples_per_class).indices
            indices.extend(class_indices[top_indices].tolist())
        return torch.tensor(indices, device=self.device)

    def _select_informative_examples(self, X, y, posteriors, pred_classes, margin):
        misclassified = torch.where(pred_classes != y)[0]
        if len(misclassified) == 0:
            return torch.tensor([], device=self.device)
        true_probs = posteriors[misclassified, y[misclassified]]
        pred_probs = posteriors[misclassified, pred_classes[misclassified]]
        max_pred_probs = pred_probs.max()
        informative_mask = (pred_probs >= max_pred_probs - margin)
        return misclassified[informative_mask]

class InvertibleDBNN(torch.nn.Module):
    def __init__(self, forward_model, feature_dims, reconstruction_weight=0.5, feedback_strength=0.3, debug=False):
        super(InvertibleDBNN, self).__init__()
        self.forward_model = forward_model
        self.device = forward_model.device
        self.feature_dims = feature_dims
        self.reconstruction_weight = reconstruction_weight
        self.feedback_strength = feedback_strength
        self.n_classes = len(self.forward_model.label_encoder.classes_)
        self.inverse_likelihood_params = None
        self.inverse_feature_pairs = None
        self.register_buffer('min_vals', None)
        self.register_buffer('max_vals', None)
        self.register_buffer('scale_factors', None)
        self._initialize_inverse_components()

    def _initialize_inverse_components(self):
        class_indices = torch.arange(self.n_classes, device=self.device)
        feature_indices = torch.arange(self.feature_dims, device=self.device)
        feature_pairs = torch.cartesian_prod(class_indices, feature_indices)
        if not hasattr(self, 'inverse_feature_pairs'):
            self.register_buffer('inverse_feature_pairs', feature_pairs)
        else:
            self.inverse_feature_pairs = feature_pairs
        n_pairs = len(feature_pairs)
        self.weight_linear = torch.nn.Parameter(
            torch.empty((n_pairs, self.feature_dims), device=self.device),
            requires_grad=True
        )
        self.weight_nonlinear = torch.nn.Parameter(
            torch.empty((n_pairs, self.feature_dims), device=self.device),
            requires_grad=True
        )
        self.bias_linear = torch.nn.Parameter(
            torch.zeros(self.feature_dims, device=self.device),
            requires_grad=True
        )
        self.bias_nonlinear = torch.nn.Parameter(
            torch.zeros(self.feature_dims, device=self.device),
            requires_grad=True
        )
        self.layer_norm = torch.nn.LayerNorm(self.feature_dims).to(self.device)
        self.feature_attention = torch.nn.Parameter(
            torch.ones(self.feature_dims, device=self.device),
            requires_grad=True
        )

    def reconstruct_features(self, class_probs):
        with torch.no_grad():
            if class_probs.dim() == 1:
                class_probs = class_probs.unsqueeze(1)
            reconstructed_features = self._compute_inverse_posterior(class_probs)
            if hasattr(self, 'min_vals') and self.min_vals is not None:
                reconstructed_features = self._unscale_features(reconstructed_features)
            return reconstructed_features

    def _compute_inverse_posterior(self, class_probs):
        batch_size = class_probs.shape[0]
        reconstructed_features = torch.zeros(
            (batch_size, self.feature_dims),
            device=self.device,
            dtype=class_probs.dtype
        )
        attention_weights = torch.softmax(self.feature_attention, dim=0)
        linear_features = torch.zeros_like(reconstructed_features)
        nonlinear_features = torch.zeros_like(reconstructed_features)
        for feat_idx in range(self.feature_dims):
            relevant_pairs = torch.where(self.inverse_feature_pairs[:, 1] == feat_idx)[0]
            class_contributions = class_probs[:, self.inverse_feature_pairs[relevant_pairs, 0]]
            linear_weights = self.weight_linear[relevant_pairs, feat_idx]
            linear_features[:, feat_idx] = torch.mm(
                class_contributions,
                linear_weights.unsqueeze(1)
            ).squeeze()
            nonlinear_weights = self.weight_nonlinear[relevant_pairs, feat_idx]
            nonlinear_features[:, feat_idx] = torch.tanh(torch.mm(
                class_contributions,
                nonlinear_weights.unsqueeze(1)
            ).squeeze())
        reconstructed_features = (
            attention_weights * linear_features +
            (1 - attention_weights) * nonlinear_features
        )
        reconstructed_features += self.bias_linear + self.bias_nonlinear
        reconstructed_features = self.layer_norm(reconstructed_features)
        return reconstructed_features

    def _compute_reconstruction_loss(self, original_features, reconstructed_features, reduction='mean'):
        if hasattr(self, 'min_vals') and self.min_vals is not None:
            original_features = self._scale_features(original_features)
            reconstructed_features = self._scale_features(reconstructed_features)
        mse_loss = torch.mean((original_features - reconstructed_features) ** 2, dim=1)
        if reduction == 'mean':
            return mse_loss.mean()
        elif reduction == 'sum':
            return mse_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction method: {reduction}")

    def _scale_features(self, features):
        return (features - self.min_vals) / self.scale_factors

    def _unscale_features(self, scaled_features):
        return (scaled_features * self.scale_factors) + self.min_vals

    def train(self, X_train, y_train, X_test=None, y_test=None):
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.long, device=self.device)
        self._compute_feature_scaling(X_train)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.forward_model.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.forward_model.lr)
        for epoch in tqdm(range(self.forward_model.max_epochs)):
            for X_batch, y_batch in train_loader:
                posteriors = self.forward_model.compute_posterior(X_batch)
                reconstructed_features = self.reconstruct_features(posteriors)
                loss = self._compute_reconstruction_loss(X_batch, reconstructed_features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/{self.forward_model.max_epochs}, Loss: {loss.item()}")

    def _compute_feature_scaling(self, features):
        with torch.no_grad():
            self.min_vals = features.min(dim=0)[0]
            self.max_vals = features.max(dim=0)[0]
            self.scale_factors = self.max_vals - self.min_vals
            self.scale_factors[self.scale_factors == 0] = 1.0

def save_label_encoder(label_encoder, dataset_name):
    encoder_path = os.path.join("data", dataset_name, f"{dataset_name}_label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"LabelEncoder saved to {encoder_path}")

def load_label_encoder(dataset_name):
    encoder_path = os.path.join("data", dataset_name, f"{dataset_name}_label_encoder.pkl")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"LabelEncoder file not found: {encoder_path}")
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder

def load_config(dataset_name):
    config_path = os.path.join("data", dataset_name, f"{dataset_name}.conf")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_dataset(dataset_name, config):
    data_path = os.path.join("data", dataset_name, f"{dataset_name}.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    df = pd.read_csv(data_path, sep=config.get("separator", ","))
    target_column = config.get("target_column", "target")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    return df, target_column

def main(dataset_name):
    config = load_config(dataset_name)
    data_path = os.path.join("data", dataset_name, f"{dataset_name}.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Read the dataset
    df = pd.read_csv(data_path, sep=config.get("separator", ","))
    column_names = config.get("column_names", [])
    target_column = config.get("target_column", "target")

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

    # Scale features using saved min and max values
    X_scaled = dbnn.scale_features(X)

    # Save the LabelEncoder for later use
    save_label_encoder(dbnn.label_encoder, dataset_name)

    # Train the model using scaled features
    dbnn.train(X_scaled, y_encoded)

    # Predict on the entire dataset
    posteriors = dbnn.compute_posterior(torch.tensor(X_scaled, dtype=torch.float32, device=dbnn.device))
    pred_classes = torch.argmax(posteriors, dim=1).cpu().numpy()

    # Decode predictions to original labels
    pred_labels = dbnn.label_encoder.inverse_transform(pred_classes)

    # Save predictions
    df["predicted_class"] = pred_labels
    output_path = os.path.join("data", dataset_name, f"{dataset_name}_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict using DBNN.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset (subfolder in data/)")
    args = parser.parse_args()
    main(args.dataset_name)
