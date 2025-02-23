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
from Astro_Utils import AstronomicalStructurePreservingAutoencoder,AstronomicalStructureLoss
from basic_utils import BaseAutoencoder,BaseEnhancementConfig,GeneralEnhancementConfig,EnhancedLossManager,ModelFactory
from Medical_Utils import  MedicalStructurePreservingAutoencoder





# Update the training loop to handle the new feature dictionary format
def train_model(model: nn.Module, train_loader: DataLoader,
                config: Dict, loss_manager: EnhancedLossManager) -> Dict[str, List]:
    """Two-phase training implementation with checkpoint handling"""
    # Store dataset reference in model
    model.set_dataset(train_loader.dataset)

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
                             lr=config['model']['learning_rate'])

        phase2_history = _train_phase(
            model, train_loader, optimizer, loss_manager,
            config['training']['epochs'], 2, config,
            start_epoch=start_epoch if current_phase == 2 else 0
        )

        # Merge histories
        for key, value in phase2_history.items():
            history[f"phase2_{key}"] = value

    return history


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
            if config['model']['enhancement_modules'].get(image_type, {}).get('enabled', False):
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
    """Training logic for each phase with enhanced checkpoint handling"""
    history = defaultdict(list)
    device = next(model.parameters()).device

    # Get phase-specific metrics
    # Initialize unified checkpoint
    checkpoint_manager = UnifiedCheckpoint(config)

    # Load best loss from checkpoint
    best_loss = checkpoint_manager.get_best_loss(phase, model)
    patience_counter = 0

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            num_batches = len(train_loader)  # Get total number of batches

            # Training loop
            pbar = tqdm(train_loader, desc=f"Phase {phase} - Epoch {epoch+1}")
            for batch_idx, (data, labels) in enumerate(pbar):
                try:
                    # Move data to correct device
                    if isinstance(data, (list, tuple)):
                        data = data[0]
                    data = data.to(device)
                    labels = labels.to(device)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass based on phase
                    if phase == 1:
                        # Phase 1: Only reconstruction
                        embeddings = model.encode(data)
                        if isinstance(embeddings, tuple):
                            embeddings = embeddings[0]
                        reconstruction = model.decode(embeddings)
                        loss = F.mse_loss(reconstruction, data)
                    else:
                        # Phase 2: Include clustering and classification
                        output = model(data)
                        if isinstance(output, dict):
                            reconstruction = output['reconstruction']
                            embedding = output['embedding']
                        else:
                            embedding, reconstruction = output

                        # Calculate base loss
                        loss = loss_manager.calculate_loss(
                            reconstruction, data,
                            config['dataset'].get('image_type', 'general')
                        )['loss']

                        # Add KL divergence loss if enabled
                        if model.use_kl_divergence:
                            latent_info = model.organize_latent_space(embedding, labels)
                            kl_weight = config['model']['autoencoder_config']['enhancements']['kl_divergence_weight']
                            if isinstance(latent_info, dict) and 'cluster_probabilities' in latent_info:
                                kl_loss = F.kl_div(
                                    latent_info['cluster_probabilities'].log(),
                                    latent_info['target_distribution'],
                                    reduction='batchmean'
                                )
                                loss += kl_weight * kl_loss

                        # Add classification loss if enabled
                        if model.use_class_encoding and hasattr(model, 'classifier'):
                            class_weight = config['model']['autoencoder_config']['enhancements']['classification_weight']
                            class_logits = model.classifier(embedding)
                            class_loss = F.cross_entropy(class_logits, labels)
                            loss += class_weight * class_loss

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    # Update running loss - handle possible NaN or inf
                    current_loss = loss.item()
                    if not (np.isnan(current_loss) or np.isinf(current_loss)):
                        running_loss += current_loss

                    # Calculate current average loss safely
                    current_avg_loss = running_loss / (batch_idx + 1)  # Add 1 to avoid division by zero

                    # Update progress bar with safe values
                    pbar.set_postfix({
                        'loss': f'{current_avg_loss:.4f}',
                        'best': f'{best_loss:.4f}'
                    })

                    # Memory cleanup
                    del data, loss
                    if phase == 2:
                        del output
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue

            # Safely calculate epoch average loss
            if num_batches > 0:
                avg_loss = running_loss / num_batches
            else:
                avg_loss = float('inf')
                logger.warning("No valid batches in epoch!")

            # Record history
            history[f'phase{phase}_loss'].append(avg_loss)

            # Save checkpoint and check for best model
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            checkpoint_manager.save_model_state(
                model=model,
                optimizer=optimizer,
                phase=phase,
                epoch=epoch,
                loss=avg_loss,
                is_best=is_best
            )
            # Early stopping check
            patience = config['training'].get('early_stopping', {}).get('patience', 5)
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered for phase {phase} after {epoch + 1} epochs")
                break

            logger.info(f'Phase {phase} - Epoch {epoch+1}: Loss = {avg_loss:.4f}, Best = {best_loss:.4f}')

            # Clean up at end of epoch
            pbar.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in training phase {phase}: {str(e)}")
        raise

    return history

class PredictionManager:
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
        batch_size = self.config['training'].get('batch_size', 32)

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

class EnhancedEncoder(nn.Module):
    """Enhanced encoder with classification head"""
    def __init__(self, base_encoder: nn.Module, num_classes: int, feature_dims: int):
        super().__init__()
        self.base_encoder = base_encoder
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims, feature_dims // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dims // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.base_encoder(x)
        class_logits = self.classifier(embedding)
        return embedding, class_logits




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

class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction models"""
    def __init__(self, config: Dict, device: str = None):
        """Initialize base feature extractor"""
        self.config = self.verify_config(config)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if self.config['execution_flags']['use_gpu']
                                     and torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize common parameters
        self.feature_dims = self.config['model']['feature_dims']
        self.learning_rate = self.config['model'].get('learning_rate', 0.001)

        # Initialize training metrics
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.history = defaultdict(list)
        self.training_log = []
        self.training_start_time = time.time()

        # Setup logging directory
        self.log_dir = os.path.join('Traininglog', self.config['dataset']['name'])
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize model
        self.feature_extractor = self._create_model()

        # Load checkpoint or initialize optimizer
        if not self.config['execution_flags'].get('fresh_start', False):
            self._load_from_checkpoint()

        # Initialize optimizer if not created during checkpoint loading
        if not hasattr(self, 'optimizer'):
            self.optimizer = self._initialize_optimizer()
            logger.info(f"Initialized {self.optimizer.__class__.__name__} optimizer")

        # Initialize scheduler
        self.scheduler = None
        if self.config['model'].get('scheduler'):
            self.scheduler = self._initialize_scheduler()
            if self.scheduler:
                logger.info(f"Initialized {self.scheduler.__class__.__name__} scheduler")

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create and return the feature extraction model"""
        pass

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer based on configuration"""
        optimizer_config = self.config['model'].get('optimizer', {})

        # Set base parameters
        optimizer_params = {
            'lr': self.learning_rate,
            'weight_decay': optimizer_config.get('weight_decay', 1e-4)
        }

        # Configure optimizer-specific parameters
        optimizer_type = optimizer_config.get('type', 'Adam')
        if optimizer_type == 'SGD':
            optimizer_params['momentum'] = optimizer_config.get('momentum', 0.9)
            optimizer_params['nesterov'] = optimizer_config.get('nesterov', False)
        elif optimizer_type == 'Adam':
            optimizer_params['betas'] = (
                optimizer_config.get('beta1', 0.9),
                optimizer_config.get('beta2', 0.999)
            )
            optimizer_params['eps'] = optimizer_config.get('epsilon', 1e-8)

        # Get optimizer class
        try:
            optimizer_class = getattr(optim, optimizer_type)
        except AttributeError:
            logger.warning(f"Optimizer {optimizer_type} not found, using Adam")
            optimizer_class = optim.Adam
            optimizer_type = 'Adam'

        # Create and return optimizer
        optimizer = optimizer_class(
            self.feature_extractor.parameters(),
            **optimizer_params
        )

        logger.info(f"Initialized {optimizer_type} optimizer with parameters: {optimizer_params}")
        return optimizer

    def _initialize_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler if specified in config"""
        scheduler_config = self.config['model'].get('scheduler', {})
        if not scheduler_config:
            return None

        scheduler_type = scheduler_config.get('type')
        if not scheduler_type:
            return None

        try:
            if scheduler_type == 'StepLR':
                return optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 7),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'ReduceLROnPlateau':
                return optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=scheduler_config.get('factor', 0.1),
                    patience=scheduler_config.get('patience', 10),
                    verbose=True
                )
            elif scheduler_type == 'CosineAnnealingLR':
                return optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get('T_max', self.config['training']['epochs']),
                    eta_min=scheduler_config.get('eta_min', 0)
                )
        except Exception as e:
            logger.warning(f"Failed to initialize scheduler: {str(e)}")
            return None

        return None

    def verify_config(self, config: Dict) -> Dict:
        """Verify and fill in missing configuration values with complete options"""
        if 'dataset' not in config:
            raise ValueError("Configuration must contain 'dataset' section")

        # Ensure all required sections exist
        required_sections = ['dataset', 'model', 'training', 'execution_flags',
                            'likelihood_config', 'active_learning']
        for section in required_sections:
            if section not in config:
                config[section] = {}

        # Dataset configuration
        dataset = config['dataset']
        dataset.setdefault('name', 'custom_dataset')
        dataset.setdefault('type', 'custom')
        dataset.setdefault('in_channels', 3)
        dataset.setdefault('input_size', [224, 224])
        dataset.setdefault('mean', [0.485, 0.456, 0.406])
        dataset.setdefault('std', [0.229, 0.224, 0.225])

        # Model configuration
        model = config.setdefault('model', {})
        model.setdefault('feature_dims', 128)
        model.setdefault('learning_rate', 0.001)
        model.setdefault('encoder_type', 'cnn')
        model.setdefault('modelType', 'Histogram')

        # Add autoencoder enhancement configuration
        autoencoder_config = model.setdefault('autoencoder_config', {})
        autoencoder_config.setdefault('enhancements', {
            'use_kl_divergence': True,
            'use_class_encoding': True,
            'kl_divergence_weight': 0.1,
            'classification_weight': 0.1,
            'clustering_temperature': 1.0,
            'min_cluster_confidence': 0.7,
            'reconstruction_weight': 1.0,
            'feature_weight': 0.1
        })

        # Complete optimizer configuration
        optimizer_config = model.setdefault('optimizer', {})
        optimizer_config.update({
            'type': 'Adam',
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        })

        # Complete scheduler configuration
        scheduler_config = model.setdefault('scheduler', {})
        scheduler_config.update({
            'type': 'ReduceLROnPlateau',
            'factor': 0.1,
            'patience': 10,
            'verbose': True,
            'min_lr': 1e-6
        })

        # Loss functions configuration
        loss_functions = model.setdefault('loss_functions', {})

        # Enhanced autoencoder loss
        loss_functions.setdefault('enhanced_autoencoder', {
            'enabled': True,
            'type': 'EnhancedAutoEncoderLoss',
            'weight': 1.0,
            'params': {
                'reconstruction_weight': 1.0,
                'clustering_weight': 0.1,
                'classification_weight': 0.1,
                'feature_weight': 0.1
            }
        })

        # Detail preserving loss
        loss_functions.setdefault('detail_preserving', {
            'enabled': True,
            'type': 'DetailPreservingLoss',
            'weight': 0.8,
            'params': {
                'detail_weight': 1.0,
                'texture_weight': 0.8,
                'frequency_weight': 0.6
            }
        })

        # Structural loss
        loss_functions.setdefault('structural', {
            'enabled': True,
            'type': 'StructuralLoss',
            'weight': 0.7,
            'params': {
                'edge_weight': 1.0,
                'smoothness_weight': 0.5
            }
        })

        # Color enhancement loss
        loss_functions.setdefault('color_enhancement', {
            'enabled': True,
            'type': 'ColorEnhancementLoss',
            'weight': 0.5,
            'params': {
                'channel_weight': 0.5,
                'contrast_weight': 0.3
            }
        })

        # Morphology loss
        loss_functions.setdefault('morphology', {
            'enabled': True,
            'type': 'MorphologyLoss',
            'weight': 0.3,
            'params': {
                'shape_weight': 0.7,
                'symmetry_weight': 0.3
            }
        })

        # Original autoencoder loss
        loss_functions.setdefault('autoencoder', {
            'enabled': False,
            'type': 'AutoencoderLoss',
            'weight': 1.0,
            'params': {
                'reconstruction_weight': 1.0,
                'feature_weight': 0.1
            }
        })

        # Training configuration
        training = config.setdefault('training', {})
        training.update({
            'batch_size': 32,
            'epochs': 20,
            'num_workers': min(4, os.cpu_count() or 1),
            'checkpoint_dir': os.path.join('Model', 'checkpoints'),
            'trials': 100,
            'test_fraction': 0.2,
            'random_seed': 42,
            'minimum_training_accuracy': 0.95,
            'cardinality_threshold': 0.9,
            'cardinality_tolerance': 4,
            'n_bins_per_dim': 20,
            'enable_adaptive': True,
            'invert_DBNN': False,
            'reconstruction_weight': 0.5,
            'feedback_strength': 0.3,
            'inverse_learning_rate': 0.1,
            'Save_training_epochs': False,
            'training_save_path': 'training_data',
            'early_stopping': {
                'enabled': True,
                'patience': 5,
                'min_delta': 0.001
            },
            'validation_split': 0.2
        })

        # Execution flags
        exec_flags = config.setdefault('execution_flags', {})
        exec_flags.update({
            'mode': 'train_and_predict',
            'use_gpu': torch.cuda.is_available(),
            'mixed_precision': True,
            'distributed_training': False,
            'debug_mode': False,
            'use_previous_model': True,
            'fresh_start': False,
            'train': True,
            'train_only': False,
            'predict': True
        })

        # Likelihood configuration
        likelihood = config.setdefault('likelihood_config', {})
        likelihood.update({
            'feature_group_size': 2,
            'max_combinations': 1000,
            'bin_sizes': [20]
        })

        # Active learning configuration
        active = config.setdefault('active_learning', {})
        active.update({
            'tolerance': 1.0,
            'cardinality_threshold_percentile': 95,
            'strong_margin_threshold': 0.3,
            'marginal_margin_threshold': 0.1,
            'min_divergence': 0.1
        })

        # Output configuration
        output = config.setdefault('output', {})
        output.update({
            'image_dir': 'output/images',
            'mode': 'train',
            'csv_dir': os.path.join('data', config['dataset']['name']),
            'input_csv': None,
            'class_info': {
                'include_given_class': True,
                'include_predicted_class': True,
                'include_cluster_probabilities': True,
                'confidence_threshold': 0.5
            },
            'visualization': {
                'enabled': True,
                'save_reconstructions': True,
                'save_feature_distributions': True,
                'save_latent_space': True,
                'max_samples': 1000
            }
        })

        return config

    @abstractmethod
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch"""
        pass

    @abstractmethod
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        pass

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """Train the model"""
        early_stopping = self.config['training'].get('early_stopping', {})
        patience = early_stopping.get('patience', 5)
        min_delta = early_stopping.get('min_delta', 0.001)
        max_epochs = self.config['training']['epochs']

        patience_counter = 0
        best_val_metric = float('inf')

        try:
            for epoch in range(self.current_epoch, max_epochs):
                self.current_epoch = epoch

                # Training
                train_loss, train_acc = self._train_epoch(train_loader)

                # Create summary for this epoch
                epoch_dir = os.path.join('data', self.config['dataset']['name'],
                                       'training_decoder_output', f'epoch_{epoch}')
                if os.path.exists(epoch_dir):
                    self.create_training_summary(epoch_dir)


                # Validation
                val_loss, val_acc = None, None
                if val_loader:
                    val_loss, val_acc = self._validate(val_loader)
                    current_metric = val_loss
                else:
                    current_metric = train_loss

                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()

                # Log metrics
                self.log_training_metrics(epoch, train_loss, train_acc, val_loss, val_acc,
                                       train_loader, val_loader)

                # Save checkpoint
                self._save_checkpoint(is_best=False)

                # Check for improvement
                if current_metric < best_val_metric - min_delta:
                    best_val_metric = current_metric
                    patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return self.history

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def log_training_metrics(self, epoch: int, train_loss: float, train_acc: float,
                           test_loss: Optional[float] = None, test_acc: Optional[float] = None,
                           train_loader: Optional[DataLoader] = None,
                           test_loader: Optional[DataLoader] = None):
        """Log training metrics"""
        elapsed_time = time.time() - self.training_start_time

        metrics = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_time': elapsed_time,
            'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_samples': len(train_loader.dataset) if train_loader else None,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_samples': len(test_loader.dataset) if test_loader else None
        }

        self.training_log.append(metrics)

        log_df = pd.DataFrame(self.training_log)
        log_path = os.path.join(self.log_dir, 'training_metrics.csv')
        log_df.to_csv(log_path, index=False)

        logger.info(f"Epoch {epoch + 1}: "
                   f"Train Loss {train_loss:.4f}, Acc {train_acc:.2f}%" +
                   (f", Test Loss {test_loss:.4f}, Acc {test_acc:.2f}%"
                    if test_loss is not None else ""))

    @abstractmethod
    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from data loader"""
        pass



class AutoEncoderFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config: Dict, device: str = None):
        super().__init__(config, device)
        self.output_image_dir = os.path.join('data', config['dataset']['name'],
                                            'output', 'images',
                                            Path(config['dataset']['name']).stem)
        os.makedirs(self.output_image_dir, exist_ok=True)

    def _create_model(self) -> nn.Module:
        """Create autoencoder model"""
        input_shape = (self.config['dataset']['in_channels'],
                      *self.config['dataset']['input_size'])
        return DynamicAutoencoder(
            input_shape=input_shape,
            feature_dims=self.feature_dims
        ).to(self.device)

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch with reconstruction visualization"""
        self.feature_extractor.train()
        running_loss = 0.0
        reconstruction_accuracy = 0.0

        # Create output directory for training reconstructions
        output_dir = os.path.join('data', self.config['dataset']['name'],
                                'training_decoder_output', f'epoch_{self.current_epoch}')
        os.makedirs(output_dir, exist_ok=True)

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}',
                   unit='batch', leave=False)

        for batch_idx, (inputs, _) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device)

                # Log input shape and channels
                logger.debug(f"Input tensor shape: {inputs.shape}, channels: {inputs.size(1)}")

                self.optimizer.zero_grad()
                embedding, reconstruction = self.feature_extractor(inputs)

                # Verify reconstruction shape matches input
                if reconstruction.shape != inputs.shape:
                    raise ValueError(f"Reconstruction shape {reconstruction.shape} "
                                  f"doesn't match input shape {inputs.shape}")

                # Calculate loss
                loss = self._calculate_loss(inputs, reconstruction, embedding)
                loss.backward()
                self.optimizer.step()

                # Update metrics
                running_loss += loss.item()
                reconstruction_accuracy += 1.0 - F.mse_loss(reconstruction, inputs).item()

                # Save reconstructions periodically
                if batch_idx % 50 == 0:
                    self._save_training_batch(inputs, reconstruction, batch_idx, output_dir)

                # Update progress bar
                batch_loss = running_loss / (batch_idx + 1)
                batch_acc = (reconstruction_accuracy / (batch_idx + 1)) * 100
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'recon_acc': f'{batch_acc:.2f}%'
                })

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                raise

        pbar.close()
        return (running_loss / len(train_loader),
                (reconstruction_accuracy / len(train_loader)) * 100)

    def _calculate_loss(self, inputs: torch.Tensor, reconstruction: torch.Tensor,
                      embedding: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss for autoencoder"""
        ae_config = self.config['model']['autoencoder_config']
        return AutoencoderLoss(
            reconstruction_weight=ae_config['reconstruction_weight'],
            feature_weight=ae_config['feature_weight']
        )(inputs, reconstruction, embedding)

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.feature_extractor.eval()
        running_loss = 0.0
        reconstruction_accuracy = 0.0

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(self.device)
                embedding, reconstruction = self.feature_extractor(inputs)

                loss = self._calculate_loss(inputs, reconstruction, embedding)
                running_loss += loss.item()
                reconstruction_accuracy += 1.0 - F.mse_loss(reconstruction, inputs).item()

                del inputs, embedding, reconstruction, loss

        return (running_loss / len(val_loader),
                (reconstruction_accuracy / len(val_loader)) * 100)



    def _save_training_batch(self, inputs: torch.Tensor, reconstructions: torch.Tensor,
                           batch_idx: int, output_dir: str):
        """Save training batch images with proper error handling"""
        with torch.no_grad():
            for i in range(min(5, inputs.size(0))):
                try:
                    orig_input = inputs[i]
                    recon = reconstructions[i]

                    # Verify channel consistency
                    expected_channels = self.config['dataset']['in_channels']
                    if orig_input.size(0) != expected_channels or recon.size(0) != expected_channels:
                        raise ValueError(f"Channel mismatch: input={orig_input.size(0)}, "
                                      f"recon={recon.size(0)}, expected={expected_channels}")

                    # Save images
                    orig_path = os.path.join(output_dir, f'batch_{batch_idx}_sample_{i}_original.png')
                    recon_path = os.path.join(output_dir, f'batch_{batch_idx}_sample_{i}_reconstruction.png')

                    self.save_training_image(orig_input, orig_path)
                    self.save_training_image(recon, recon_path)

                except Exception as e:
                    logger.error(f"Error saving training sample {i} from batch {batch_idx}: {str(e)}")

    def save_training_image(self, tensor: torch.Tensor, path: str):
        """Save training image with robust channel handling"""
        try:
            tensor = tensor.detach().cpu()
            expected_channels = self.config['dataset']['in_channels']

            # Ensure we're working with the right shape [C, H, W]
            if len(tensor.shape) == 4:
                tensor = tensor.squeeze(0)

            if tensor.shape[0] != expected_channels:
                raise ValueError(f"Expected {expected_channels} channels, got {tensor.shape[0]}")

            # Move to [H, W, C] for image saving
            tensor = tensor.permute(1, 2, 0)

            # Get normalization parameters
            mean = torch.tensor(self.config['dataset']['mean'], dtype=tensor.dtype)
            std = torch.tensor(self.config['dataset']['std'], dtype=tensor.dtype)

            # Ensure mean/std match channel count
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)

            # Denormalize
            tensor = tensor * std + mean
            tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)

            # Handle single-channel case
            if tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)

            # Save image
            img = Image.fromarray(tensor.numpy())
            img.save(path)

        except Exception as e:
            logger.error(f"Error saving training image: {str(e)}")
            logger.error(f"Tensor shape at error: {tensor.shape if 'tensor' in locals() else 'unknown'}")
            raise

    def predict_from_csv(self, csv_path: str):
        """Generate reconstructions from feature vectors in CSV"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        features = torch.tensor(df[feature_cols].values, dtype=torch.float32)

        # Verify feature dimensions
        expected_dims = self.config['model']['feature_dims']
        if features.size(1) != expected_dims:
            raise ValueError(f"Feature dimension mismatch: got {features.size(1)}, expected {expected_dims}")

        self.feature_extractor.eval()
        output_dir = self.config['output']['image_dir']
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for idx, feature_vec in enumerate(tqdm(features, desc="Generating reconstructions")):
                try:
                    # Ensure proper shape and device placement
                    feature_vec = feature_vec.to(self.device).unsqueeze(0)

                    # Generate reconstruction
                    reconstruction = self.feature_extractor.decode(feature_vec)

                    # Verify channel count
                    if reconstruction.size(1) != self.config['dataset']['in_channels']:
                        raise ValueError(f"Reconstruction channel mismatch: got {reconstruction.size(1)}, "
                                      f"expected {self.config['dataset']['in_channels']}")

                    # Save reconstructed image
                    img_path = os.path.join('data', self.config['dataset']['name'],output_dir, f"reconstruction_{idx}.png")
                    self.save_reconstructed_image(reconstruction[0], img_path)

                except Exception as e:
                    logger.error(f"Error processing feature vector {idx}: {str(e)}")

    def save_reconstructed_image(self, tensor: torch.Tensor, path: str):
        """Save reconstructed tensor as image with proper normalization"""
        try:
            tensor = tensor.detach().cpu()

            # Verify channel count
            if tensor.size(0) != self.config['dataset']['in_channels']:
                raise ValueError(f"Expected {self.config['dataset']['in_channels']} channels, got {tensor.size(0)}")

            # Move to [H, W, C] for image saving
            tensor = tensor.permute(1, 2, 0)

            # Get normalization parameters
            mean = torch.tensor(self.config['dataset']['mean'], dtype=tensor.dtype)
            std = torch.tensor(self.config['dataset']['std'], dtype=tensor.dtype)

            # Reshape for broadcasting
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)

            # Denormalize
            tensor = tensor * std + mean
            tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)

            # Handle single-channel case
            if tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)

            # Save image
            img = Image.fromarray(tensor.numpy())
            img.save(path)

        except Exception as e:
            logger.error(f"Error saving reconstructed image: {str(e)}")
            logger.error(f"Tensor shape at error: {tensor.shape if 'tensor' in locals() else 'unknown'}")
            raise

    def plot_reconstruction_samples(self, loader: DataLoader, num_samples: int = 8,
                                 save_path: Optional[str] = None):
        """Visualize original and reconstructed images"""
        self.feature_extractor.eval()

        # Get samples
        images, _ = next(iter(loader))
        images = images[:num_samples].to(self.device)

        with torch.no_grad():
            _, reconstructions = self.feature_extractor(images)

        # Plot results
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))

        for i in range(num_samples):
            # Original
            axes[0, i].imshow(self._tensor_to_image(images[i]))
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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Reconstruction samples saved to {save_path}")
        plt.close()

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to image array with proper normalization"""
        tensor = tensor.cpu()

        # Move to [H, W, C]
        if len(tensor.shape) == 3:
            tensor = tensor.permute(1, 2, 0)

        # Denormalize
        mean = torch.tensor(self.config['dataset']['mean']).view(1, 1, -1)
        std = torch.tensor(self.config['dataset']['std']).view(1, 1, -1)
        tensor = tensor * std + mean

        # Convert to uint8
        return (tensor.clamp(0, 1) * 255).numpy().astype(np.uint8)

    def plot_latent_space(self, dataloader: DataLoader, num_samples: int = 1000,
                         save_path: Optional[str] = None):
        """Plot 2D visualization of latent space"""
        if self.feature_dims < 2:
            logger.warning("Latent space dimension too small for visualization")
            return

        self.feature_extractor.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                if len(embeddings) * inputs.size(0) >= num_samples:
                    break

                inputs = inputs.to(self.device)
                embedding = self.feature_extractor.encode(inputs)
                embeddings.append(embedding.cpu())
                labels.extend(targets.tolist())

        embeddings = torch.cat(embeddings, dim=0)[:num_samples]
        labels = labels[:num_samples]

        # Use PCA for visualization if dimensions > 2
        if self.feature_dims > 2:
            from sklearn.decomposition import PCA
            embeddings = PCA(n_components=2).fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('Latent Space Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Latent space visualization saved to {save_path}")
        plt.close()

    def _load_from_checkpoint(self):
        """Load model and training state from checkpoint"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Try to find latest checkpoint
        checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Load model state
                self.feature_extractor.load_state_dict(checkpoint['state_dict'])

                # Initialize and load optimizer
                self.optimizer = self._initialize_optimizer()
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Optimizer state loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {str(e)}")

                # Load training state
                self.current_epoch = checkpoint.get('epoch', 0)
                self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))

                # Load history
                if 'history' in checkpoint:
                    self.history = defaultdict(list, checkpoint['history'])

                logger.info("Checkpoint loaded successfully")

            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                self.optimizer = self._initialize_optimizer()
        else:
            logger.info("No checkpoint found, starting from scratch")
            self.optimizer = self._initialize_optimizer()

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        if not os.path.exists(checkpoint_dir):
            return None

        dataset_name = self.config['dataset']['name']

        # Check for best model first
        best_path = os.path.join(checkpoint_dir, f"{dataset_name}_best.pth")
        if os.path.exists(best_path):
            return best_path

        # Check for latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            return checkpoint_path

        return None

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'state_dict': self.feature_extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'history': dict(self.history),
            'config': self.config
        }

        # Save latest checkpoint
        dataset_name = self.config['dataset']['name']
        filename = f"{dataset_name}_{'best' if is_best else 'checkpoint'}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved {'best' if is_best else 'latest'} checkpoint to {checkpoint_path}")

    def create_training_summary(self, epoch_dir: str):
        """Create HTML summary of training reconstructions"""
        summary_path = os.path.join(epoch_dir, 'summary.html')

        html_content = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '<style>',
            '.image-pair { display: inline-block; margin: 10px; text-align: center; }',
            '.image-pair img { width: 128px; height: 128px; margin: 5px; }',
            '</style>',
            '</head>',
            '<body>',
            f'<h1>Training Reconstructions - Epoch {self.current_epoch + 1}</h1>'
        ]

        # Find all image pairs
        original_images = sorted(glob.glob(os.path.join(epoch_dir, '*_original.png')))

        for orig_path in original_images:
            recon_path = orig_path.replace('_original.png', '_reconstruction.png')
            if os.path.exists(recon_path):
                base_name = os.path.basename(orig_path)
                pair_id = base_name.split('_original')[0]

                html_content.extend([
                    '<div class="image-pair">',
                    f'<p>{pair_id}</p>',
                    f'<img src="{os.path.basename(orig_path)}" alt="Original">',
                    f'<img src="{os.path.basename(recon_path)}" alt="Reconstruction">',
                    '</div>'
                ])

        html_content.extend(['</body>', '</html>'])

        with open(summary_path, 'w') as f:
            f.write('\n'.join(html_content))

        logger.info(f"Created training summary: {summary_path}")

    def _verify_config(self):
        """Verify configuration has all required fields"""
        required_fields = {
            'dataset': ['in_channels', 'input_size', 'mean', 'std'],
            'model': ['feature_dims', 'learning_rate', 'autoencoder_config'],
            'training': ['batch_size', 'epochs', 'checkpoint_dir']
        }

        for section, fields in required_fields.items():
            if section not in self.config:
                raise ValueError(f"Missing config section: {section}")
            for field in fields:
                if field not in self.config[section]:
                    raise ValueError(f"Missing config field: {section}.{field}")

    def log_training_metrics(self, epoch: int, train_loss: float, train_acc: float,
                            test_loss: Optional[float] = None, test_acc: Optional[float] = None,
                            train_loader: Optional[DataLoader] = None,
                            test_loader: Optional[DataLoader] = None):
        """Log training metrics"""
        elapsed_time = time.time() - self.training_start_time

        metrics = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_time': elapsed_time,
            'elapsed_time_formatted': str(timedelta(seconds=int(elapsed_time))),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_samples': len(train_loader.dataset) if train_loader else None,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_samples': len(test_loader.dataset) if test_loader else None
        }

        self.training_log.append(metrics)

        log_df = pd.DataFrame(self.training_log)
        log_path = os.path.join(self.log_dir, 'training_metrics.csv')
        log_df.to_csv(log_path, index=False)

        logger.info(f"Epoch {epoch + 1}: "
                   f"Train Loss {train_loss:.4f}, Acc {train_acc:.2f}%" +
                   (f", Test Loss {test_loss:.4f}, Acc {test_acc:.2f}%"
                    if test_loss is not None else ""))

    def _initialize_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler if specified in config"""
        scheduler_config = self.config['model'].get('scheduler', {})
        if not scheduler_config:
            return None

        scheduler_type = scheduler_config.get('type')
        if not scheduler_type:
            return None

        try:
            if scheduler_type == 'StepLR':
                return optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 7),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'ReduceLROnPlateau':
                return optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=scheduler_config.get('factor', 0.1),
                    patience=scheduler_config.get('patience', 10),
                    verbose=True
                )
            elif scheduler_type == 'CosineAnnealingLR':
                return optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get('T_max', self.config['training']['epochs']),
                    eta_min=scheduler_config.get('eta_min', 0)
                )
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {str(e)}")
            return None

        return None

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer based on configuration"""
        optimizer_config = self.config['model'].get('optimizer', {})

        # Set base parameters
        optimizer_params = {
            'lr': self.learning_rate,
            'weight_decay': optimizer_config.get('weight_decay', 1e-4)
        }

        # Configure optimizer-specific parameters
        optimizer_type = optimizer_config.get('type', 'Adam')
        if optimizer_type == 'SGD':
            optimizer_params['momentum'] = optimizer_config.get('momentum', 0.9)
            optimizer_params['nesterov'] = optimizer_config.get('nesterov', False)
        elif optimizer_type == 'Adam':
            optimizer_params['betas'] = (
                optimizer_config.get('beta1', 0.9),
                optimizer_config.get('beta2', 0.999)
            )
            optimizer_params['eps'] = optimizer_config.get('epsilon', 1e-8)

        # Get optimizer class
        try:
            optimizer_class = getattr(optim, optimizer_type)
        except AttributeError:
            logger.warning(f"Optimizer {optimizer_type} not found, using Adam")
            optimizer_class = optim.Adam
            optimizer_type = 'Adam'

        # Create and return optimizer
        optimizer = optimizer_class(
            self.feature_extractor.parameters(),
            **optimizer_params
        )

        logger.info(f"Initialized {optimizer_type} optimizer with parameters: {optimizer_params}")
        return optimizer

    def plot_feature_distribution(self, loader: DataLoader, save_path: Optional[str] = None):
        """Plot distribution of extracted features"""
        features, _ = self.extract_features(loader)
        features = features.numpy()

        plt.figure(figsize=(12, 6))
        plt.hist(features.flatten(), bins=50, density=True)
        plt.title('Feature Distribution')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Feature distribution plot saved to {save_path}")
        plt.close()

    def generate_reconstructions(self):
        """Generate reconstructed images based on config mode"""
        invert_dbnn = self.config.get('execution_flags', {}).get('invert_DBNN', False)
        dataset_name = self.config['dataset']['name']
        base_dir = os.path.join('data', dataset_name)

        # Determine input file
        if invert_dbnn:
            input_file = os.path.join(base_dir, 'reconstructed_input.csv')
        else:
            input_file = os.path.join(base_dir, f"{dataset_name}.csv")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Read embeddings
        df = pd.read_csv(input_file)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        embeddings = torch.tensor(df[feature_cols].values, dtype=torch.float32)

        # Generate reconstructions
        self.feature_extractor.eval()
        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i+batch_size].to(self.device)
                reconstructions = self.feature_extractor.decode(batch)

                # Save reconstructed images
                for j, reconstruction in enumerate(reconstructions):
                    idx = i + j
                    filename = f"reconstruction_{idx}.png"
                    self.save_reconstructed_image(filename, reconstruction)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """Train the feature extractor"""
        early_stopping = self.config['training'].get('early_stopping', {})
        patience = early_stopping.get('patience', 5)
        min_delta = early_stopping.get('min_delta', 0.001)
        max_epochs = self.config['training']['epochs']

        patience_counter = 0
        best_val_metric = float('inf')

        if not hasattr(self, 'training_start_time'):
            self.training_start_time = time.time()

        try:
            for epoch in range(self.current_epoch, max_epochs):
                self.current_epoch = epoch

                # Training
                train_loss, train_acc = self._train_epoch(train_loader)

                # Create summary for this epoch
                epoch_dir = os.path.join('data', self.config['dataset']['name'],
                                       'training_decoder_output', f'epoch_{epoch}')
                if os.path.exists(epoch_dir):
                    self.create_training_summary(epoch_dir)

                # Validation
                val_loss, val_acc = None, None
                if val_loader:
                    val_loss, val_acc = self._validate(val_loader)
                    current_metric = val_loss
                else:
                    current_metric = train_loss

                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()

                # Log metrics
                self.log_training_metrics(epoch, train_loss, train_acc, val_loss, val_acc,
                                       train_loader, val_loader)

                # Save checkpoint
                self._save_checkpoint(is_best=False)

                # Check for improvement
                if current_metric < best_val_metric - min_delta:
                    best_val_metric = current_metric
                    patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return self.history

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def visualize_reconstructions(self, dataloader: DataLoader, num_samples: int = 8,
                                save_path: Optional[str] = None):
        """Plot grid of original and reconstructed images"""
        self.feature_extractor.eval()

        # Get samples
        original_images = []
        reconstructed_images = []

        with torch.no_grad():
            for images, _ in dataloader:
                if len(original_images) >= num_samples:
                    break

                batch_images = images.to(self.device)
                _, reconstructions = self.feature_extractor(batch_images)

                original_images.extend(images.cpu())
                reconstructed_images.extend(reconstructions.cpu())

        # Select required number of samples
        original_images = original_images[:num_samples]
        reconstructed_images = reconstructed_images[:num_samples]

        # Create plot
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))

        for i in range(num_samples):
            # Plot original
            axes[0, i].imshow(self._tensor_to_image(original_images[i]))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')

            # Plot reconstruction
            axes[1, i].imshow(self._tensor_to_image(reconstructed_images[i]))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Reconstruction visualization saved to {save_path}")
        plt.close()

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training metrics history with enhanced metrics"""
        if not self.history:
            logger.warning("No training history available to plot")
            return

        plt.figure(figsize=(15, 5))

        # Plot loss history
        plt.subplot(1, 2, 1)
        if 'train_loss' in self.history:
            plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot enhancement-specific metrics if available
        plt.subplot(1, 2, 2)
        metrics = [k for k in self.history.keys()
                  if k not in ['train_loss', 'val_loss', 'train_acc', 'val_acc']]

        if metrics:
            for metric in metrics[:3]:  # Plot up to 3 additional metrics
                values = [float(v) if isinstance(v, torch.Tensor) else v
                         for v in self.history[metric]]
                plt.plot(values, label=metric.replace('_', ' ').title())
            plt.title('Enhancement Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
        else:
            # If no enhancement metrics, plot accuracy
            if 'train_acc' in self.history:
                plt.plot(self.history['train_acc'], label='Train Acc')
            if 'val_acc' in self.history:
                plt.plot(self.history['val_acc'], label='Val Acc')
            plt.title('Accuracy History')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        plt.close()
    def get_reconstruction(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Get reconstruction for input tensor"""
        self.feature_extractor.eval()
        with torch.no_grad():
            embedding, reconstruction = self.feature_extractor(input_tensor)
            return reconstruction

    def get_reconstruction_error(self, input_tensor: torch.Tensor) -> float:
        """Calculate reconstruction error for input tensor"""
        reconstruction = self.get_reconstruction(input_tensor)
        return F.mse_loss(reconstruction, input_tensor).item()

    def get_feature_shape(self) -> Tuple[int, ...]:
        """Get shape of extracted features"""
        return (self.feature_dims,)

    def save_model(self, path: str):
        """Save model to path"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.feature_extractor.state_dict(),
            'config': self.config,
            'feature_dims': self.feature_dims
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from path"""
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.feature_extractor.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

class EnhancedFeatureExtractor(AutoEncoderFeatureExtractor):
    """Enhanced feature extractor with clustering and classification capabilities"""

    def _create_model(self) -> nn.Module:
        """Create enhanced autoencoder model"""
        input_shape = (self.config['dataset']['in_channels'],
                      *self.config['dataset']['input_size'])
        num_classes = self.config['dataset'].get('num_classes', 10)  # Default to 10 if not specified

        return EnhancedDynamicAutoencoder(
            input_shape=input_shape,
            feature_dims=self.feature_dims,
            num_classes=num_classes,
            config=self.config
        ).to(self.device)


    def save_features(self, features_dict: Dict[str, torch.Tensor], output_path: str):
        """Enhanced feature saving with class information"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Get output configuration
            output_config = self.config['output']['class_info']
            confidence_threshold = output_config['confidence_threshold']

            # Prepare feature dictionary
            feature_dict = {
                f'feature_{i}': features_dict['features'][:, i].numpy()
                for i in range(features_dict['features'].shape[1])
            }

            # Add class information based on configuration
            if output_config['include_given_class']:
                feature_dict['given_class'] = features_dict['given_labels'].numpy()

            if output_config['include_predicted_class'] and 'predicted_labels' in features_dict:
                predictions = features_dict['predicted_labels'].numpy()
                if 'cluster_probabilities' in features_dict:
                    probabilities = features_dict['cluster_probabilities'].numpy()
                    max_probs = probabilities.max(axis=1)

                    # Only include confident predictions
                    confident_mask = max_probs >= confidence_threshold
                    feature_dict['predicted_class'] = np.where(
                        confident_mask,
                        predictions,
                        -1  # Use -1 for low confidence predictions
                    )
                    feature_dict['prediction_confidence'] = max_probs
                else:
                    feature_dict['predicted_class'] = predictions

            if output_config['include_cluster_probabilities'] and 'cluster_probabilities' in features_dict:
                probs = features_dict['cluster_probabilities'].numpy()
                for i in range(probs.shape[1]):
                    feature_dict[f'class_{i}_probability'] = probs[:, i]

            # Create DataFrame and save
            df = pd.DataFrame(feature_dict)
            df.to_csv(output_path, index=False)

            # Save metadata
            metadata = {
                'feature_dims': features_dict['features'].shape[1],
                'num_samples': len(df),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': {
                    'confidence_threshold': confidence_threshold,
                    'included_fields': list(feature_dict.keys())
                }
            }

            metadata_path = os.path.join(
                os.path.dirname(output_path),
                'feature_extraction_metadata.json'
            )
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            logger.info(f"Features and metadata saved to {os.path.dirname(output_path)}")

        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise

    def analyze_predictions(self) -> Dict:
        """Analyze prediction quality and clustering effectiveness"""
        if not hasattr(self, 'latest_features'):
            logger.warning("No features available for analysis")
            return {}

        features_dict = self.latest_features
        metrics = {
            'total_samples': len(features_dict['given_labels']),
            'feature_dims': features_dict['features'].shape[1]
        }

        if 'predicted_labels' in features_dict:
            given_labels = features_dict['given_labels'].numpy()
            predicted_labels = features_dict['predicted_labels'].numpy()

            # Calculate accuracy for samples with known labels
            valid_mask = given_labels != -1
            if valid_mask.any():
                metrics['accuracy'] = float(
                    (predicted_labels[valid_mask] == given_labels[valid_mask]).mean()
                )

                # Per-class accuracy
                unique_classes = np.unique(given_labels[valid_mask])
                metrics['per_class_accuracy'] = {
                    f'class_{int(class_idx)}': float(
                        (predicted_labels[given_labels == class_idx] == class_idx).mean()
                    )
                    for class_idx in unique_classes
                }

        if 'cluster_probabilities' in features_dict:
            probs = features_dict['cluster_probabilities'].numpy()
            metrics.update({
                'average_confidence': float(probs.max(axis=1).mean()),
                'high_confidence_ratio': float((probs.max(axis=1) >= 0.9).mean())
            })

        return metrics


class FeatureExtractorCNN(nn.Module):
    """CNN-based feature extractor model"""
    def __init__(self, in_channels: int = 3, feature_dims: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, feature_dims)
        self.batch_norm = nn.BatchNorm1d(feature_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.batch_norm(x)
        return x

class DynamicAutoencoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int):
        super().__init__()
        self.input_shape = input_shape  # e.g., (3, 32, 32) for CIFAR
        self.in_channels = input_shape[0]  # Store input channels explicitly
        self.feature_dims = feature_dims

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

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        in_channels = self.in_channels  # Start with input channels
        for size in self.layer_sizes:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, size, 3, stride=2, padding=1),
                    nn.BatchNorm2d(size),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = size

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

        # Decoder layers with careful channel tracking
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
                    nn.LeakyReLU(0.2) if i > 0 else nn.Tanh()
                )
            )
            in_channels = out_channels

    def _calculate_layer_sizes(self) -> List[int]:
        """Calculate progressive channel sizes for encoder/decoder"""
        # Start with input channels
        base_channels = 32  # Reduced from 64 to handle smaller images
        sizes = []
        current_size = base_channels

        # Calculate maximum number of downsampling layers based on smallest spatial dimension
        min_dim = min(self.input_shape[1:])
        max_layers = int(np.log2(min_dim)) - 2  # Ensure we don't reduce too much

        for _ in range(max_layers):
            sizes.append(current_size)
            if current_size < 256:  # Reduced from 512 to handle smaller images
                current_size *= 2

        logger.info(f"Layer sizes: {sizes}")
        return sizes

    def _calculate_flattened_size(self) -> int:
        """Calculate size of flattened feature maps before linear layer"""
        reduction_factor = 2 ** (len(self.layer_sizes) - 1)
        reduced_dims = [dim // reduction_factor for dim in self.spatial_dims]
        return self.layer_sizes[-1] * np.prod(reduced_dims)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images to feature space"""
        # Verify input channels
        if x.size(1) != self.in_channels:
            raise ValueError(f"Input has {x.size(1)} channels, expected {self.in_channels}")

        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return self.embedder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features back to image space"""
        x = self.unembedder(x)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim, self.final_spatial_dim)

        for layer in self.decoder_layers:
            x = layer(x)

        # Verify output shape
        if x.size(1) != self.in_channels:
            raise ValueError(f"Output has {x.size(1)} channels, expected {self.in_channels}")

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder"""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return embedding, reconstruction

    def get_encoding_shape(self) -> Tuple[int, ...]:
        """Get the shape of the encoding at each layer"""
        return tuple([size for size in self.layer_sizes])

    def get_spatial_dims(self) -> List[List[int]]:
        """Get the spatial dimensions at each layer"""
        return self.spatial_dims.copy()

class EnhancedDynamicAutoencoder(DynamicAutoencoder):
    """Enhanced autoencoder with clustering and classification capabilities"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, num_classes: int, config: Dict):
        # Initialize parent class first
        super().__init__(input_shape, feature_dims)

        self.num_classes = num_classes

        # Get enhancement configuration
        self.enhancement_config = config['model']['autoencoder_config']['enhancements']

        # Add classification head if class encoding is enabled
        if self.enhancement_config['use_class_encoding']:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dims, feature_dims // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(feature_dims // 2, num_classes)
            )
        else:
            self.classifier = None

        # Initialize enhanced loss function
        loss_params = config['model']['loss_functions']['enhanced_autoencoder']['params']
        self.loss_fn = EnhancedAutoEncoderLoss(
            num_classes=num_classes,
            feature_dims=feature_dims,
            reconstruction_weight=loss_params['reconstruction_weight'],
            clustering_weight=loss_params['clustering_weight'] if self.enhancement_config['use_kl_divergence'] else 0.0,
            classification_weight=loss_params['classification_weight'] if self.enhancement_config['use_class_encoding'] else 0.0,
            temperature=self.enhancement_config['clustering_temperature']
        )

    def _calculate_layer_sizes(self) -> List[int]:
        """Calculate progressive channel sizes for encoder/decoder"""
        # Maintain original calculation logic
        base_channels = 32
        sizes = []
        current_size = base_channels

        min_dim = min(self.input_shape[1:])
        max_layers = int(np.log2(min_dim)) - 2

        for _ in range(max_layers):
            sizes.append(current_size)
            if current_size < 256:
                current_size *= 2

        logger.info(f"Layer sizes: {sizes}")
        return sizes

    def _calculate_flattened_size(self) -> int:
        """Calculate size of flattened feature maps before linear layer"""
        reduction_factor = 2 ** (len(self.layer_sizes) - 1)
        reduced_dims = [dim // reduction_factor for dim in self.spatial_dims]
        return self.layer_sizes[-1] * np.prod(reduced_dims)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Enhanced encode method returning both embedding and class logits"""
        # Verify input channels
        if x.size(1) != self.in_channels:
            raise ValueError(f"Input has {x.size(1)} channels, expected {self.in_channels}")

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)

        # Add classification if enabled
        if self.classifier is not None:
            class_logits = self.classifier(embedding)
            return embedding, class_logits
        return embedding, None

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode features back to image space"""
        x = self.unembedder(embedding)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim, self.final_spatial_dim)

        for layer in self.decoder_layers:
            x = layer(x)

        # Verify output shape
        if x.size(1) != self.in_channels:
            raise ValueError(f"Output has {x.size(1)} channels, expected {self.in_channels}")

        return x

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass returning all relevant information"""
        # Get embeddings and class predictions
        embedding, class_logits = self.encode(x)

        # Get reconstruction
        reconstruction = self.decode(embedding)

        # Prepare output dictionary
        output = {
            'reconstruction': reconstruction,
            'embedding': embedding
        }

        # Add classification-related outputs if enabled
        if class_logits is not None:
            output['class_logits'] = class_logits
            output['class_predictions'] = class_logits.argmax(dim=1)

        # Calculate loss if in training mode
        if self.training:
            loss, cluster_assignments, class_predictions = self.loss_fn(
                x, reconstruction, embedding,
                class_logits if class_logits is not None else torch.zeros((x.size(0), self.num_classes), device=x.device),
                labels
            )
            output.update({
                'loss': loss,
                'cluster_assignments': cluster_assignments,
                'class_predictions': class_predictions
            })

        return output

    def get_feature_info(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract comprehensive feature information"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

            # Add softmax probabilities if classification is enabled
            if 'class_logits' in output:
                output['class_probabilities'] = F.softmax(output['class_logits'], dim=1)

            return output

    def get_encoding_shape(self) -> Tuple[int, ...]:
        """Get the shape of the encoding at each layer"""
        return tuple([size for size in self.layer_sizes])

    def get_spatial_dims(self) -> List[List[int]]:
        """Get the spatial dimensions at each layer"""
        return self.spatial_dims.copy()

    def plot_reconstruction_samples(self, inputs: torch.Tensor,
                                 save_path: Optional[str] = None) -> None:
        """Visualize original and reconstructed images"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            reconstructions = outputs['reconstruction']

        # Create visualization grid
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
            logger.info(f"Reconstruction samples saved to {save_path}")
        plt.close()

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to image array with proper normalization"""
        tensor = tensor.cpu()
        if len(tensor.shape) == 3:
            tensor = tensor.permute(1, 2, 0)

        # Denormalize
        mean = torch.tensor(self.mean).view(1, 1, -1)
        std = torch.tensor(self.std).view(1, 1, -1)
        tensor = tensor * std + mean

        return (tensor.clamp(0, 1) * 255).numpy().astype(np.uint8)

    def visualize_latent_space(self, embeddings: torch.Tensor,
                             labels: Optional[torch.Tensor] = None,
                             save_path: Optional[str] = None) -> None:
        """Visualize the latent space using PCA or t-SNE"""
        self.eval()
        with torch.no_grad():
            embeddings_np = embeddings.cpu().numpy()

            # Reduce dimensions for visualization
            if embeddings_np.shape[1] > 2:
                from sklearn.decomposition import PCA
                embeddings_2d = PCA(n_components=2).fit_transform(embeddings_np)
            else:
                embeddings_2d = embeddings_np

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                c=labels.cpu().numpy() if labels is not None else None,
                                cmap='tab10')

            if labels is not None:
                plt.colorbar(scatter)
            plt.title('Latent Space Visualization')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')

            if save_path:
                plt.savefig(save_path)
                logger.info(f"Latent space visualization saved to {save_path}")
            plt.close()

    def predict_from_features(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate predictions from feature vectors"""
        self.eval()
        with torch.no_grad():
            reconstruction = self.decode(features)
            output = {'reconstruction': reconstruction}

            if self.classifier is not None:
                class_logits = self.classifier(features)
                output.update({
                    'class_logits': class_logits,
                    'class_predictions': class_logits.argmax(dim=1),
                    'class_probabilities': F.softmax(class_logits, dim=1)
                })

            return output
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


class CNNFeatureExtractor(BaseFeatureExtractor):
    """CNN-based feature extractor implementation"""

    def _create_model(self) -> nn.Module:
        """Create CNN model"""
        return FeatureExtractorCNN(
            in_channels=self.config['dataset']['in_channels'],
            feature_dims=self.feature_dims
        ).to(self.device)

    def _load_from_checkpoint(self):
        """Load model from checkpoint"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Try to find latest checkpoint
        checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Load model state
                self.feature_extractor.load_state_dict(checkpoint['state_dict'])

                # Initialize and load optimizer
                self.optimizer = self._initialize_optimizer()
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Optimizer state loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {str(e)}")

                # Load training state
                self.current_epoch = checkpoint.get('epoch', 0)
                self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))

                # Load history
                if 'history' in checkpoint:
                    self.history = defaultdict(list, checkpoint['history'])

                logger.info("Checkpoint loaded successfully")

            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                self.optimizer = self._initialize_optimizer()
        else:
            logger.info("No checkpoint found, starting from scratch")
            self.optimizer = self._initialize_optimizer()

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file"""
        dataset_name = self.config['dataset']['name']
        checkpoint_dir = os.path.join('data', dataset_name, 'checkpoints')

        if not os.path.exists(checkpoint_dir):
            return None

        # Check for best model first
        best_path = os.path.join(checkpoint_dir, f"{dataset_name}_best.pth")
        if os.path.exists(best_path):
            return best_path

        # Check for latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            return checkpoint_path

        return None

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'state_dict': self.feature_extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'history': dict(self.history),
            'config': self.config
        }

        # Save latest checkpoint
        dataset_name = self.config['dataset']['name']
        filename = f"{dataset_name}_{'best' if is_best else 'checkpoint'}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved {'best' if is_best else 'latest'} checkpoint to {checkpoint_path}")

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch"""
        self.feature_extractor.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}',
                   unit='batch', leave=False)

        try:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.feature_extractor(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                batch_loss = running_loss / (batch_idx + 1)
                batch_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{batch_acc:.2f}%'
                })

                # Cleanup
                del inputs, outputs, loss
                if batch_idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            raise

        pbar.close()
        return running_loss / len(train_loader), 100. * correct / total

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.feature_extractor.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.feature_extractor(inputs)
                loss = F.cross_entropy(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Cleanup
                del inputs, outputs, loss

        return running_loss / len(val_loader), 100. * correct / total

    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from data"""
        self.feature_extractor.eval()
        features = []
        labels = []

        try:
            with torch.no_grad():
                for inputs, targets in tqdm(loader, desc="Extracting features"):
                    inputs = inputs.to(self.device)
                    outputs = self.feature_extractor(inputs)
                    features.append(outputs.cpu())
                    labels.append(targets)

                    # Cleanup
                    del inputs, outputs
                    if len(features) % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            return torch.cat(features), torch.cat(labels)

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def get_feature_shape(self) -> Tuple[int, ...]:
        """Get shape of extracted features"""
        return (self.feature_dims,)

    def plot_feature_distribution(self, loader: DataLoader, save_path: Optional[str] = None):
        """Plot distribution of extracted features"""
        features, _ = self.extract_features(loader)
        features = features.numpy()

        plt.figure(figsize=(12, 6))
        plt.hist(features.flatten(), bins=50, density=True)
        plt.title('Feature Distribution')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Feature distribution plot saved to {save_path}")
        plt.close()



class FeatureExtractorFactory:
    """Factory class for creating feature extractors"""

    @staticmethod
    def create(config: Dict, device: Optional[str] = None) -> BaseFeatureExtractor:
        """
        Create appropriate feature extractor based on configuration.

        Args:
            config: Configuration dictionary
            device: Optional device specification

        Returns:
            Instance of appropriate feature extractor
        """
        encoder_type = config['model'].get('encoder_type', 'cnn').lower()

        if encoder_type == 'cnn':
            return CNNFeatureExtractor(config, device)
        elif encoder_type == 'autoenc':
            return AutoEncoderFeatureExtractor(config, device)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

class EnhancedAutoEncoderFeatureExtractor(AutoEncoderFeatureExtractor):
    def predict_from_csv(self, csv_path: str):
        """Generate reconstructions from feature vectors with optimal scaling"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        features = torch.tensor(df[feature_cols].values, dtype=torch.float32)

        # Verify feature dimensions
        expected_dims = self.config['model']['feature_dims']
        if features.size(1) != expected_dims:
            raise ValueError(f"Feature dimension mismatch: got {features.size(1)}, expected {expected_dims}")

        # Get target dimensions from config
        target_size = tuple(self.config['dataset']['input_size'])
        target_channels = self.config['dataset']['in_channels']
        logger.info(f"Target image size: {target_size}, channels: {target_channels}")

        self.feature_extractor.eval()
        output_dir = self.config['output']['image_dir']
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for idx, feature_vec in enumerate(tqdm(features, desc="Generating reconstructions")):
                try:
                    # Generate reconstruction
                    feature_vec = feature_vec.to(self.device).unsqueeze(0)
                    reconstruction = self.feature_extractor.decode(feature_vec)

                    # Handle channel mismatch if any
                    if reconstruction.size(1) != target_channels:
                        if target_channels == 1 and reconstruction.size(1) == 3:
                            # Use proper RGB to grayscale conversion
                            reconstruction = 0.299 * reconstruction[:, 0:1] + \
                                          0.587 * reconstruction[:, 1:2] + \
                                          0.114 * reconstruction[:, 2:3]
                        elif target_channels == 3 and reconstruction.size(1) == 1:
                            reconstruction = reconstruction.repeat(1, 3, 1, 1)

                    # Apply optimal scaling using interpolate
                    current_size = (reconstruction.size(2), reconstruction.size(3))
                    if current_size != target_size:
                        # Choose interpolation mode based on scaling factor
                        scale_factor = (target_size[0] / current_size[0],
                                      target_size[1] / current_size[1])

                        # Use bicubic for upscaling and area for downscaling
                        mode = 'bicubic' if min(scale_factor) > 1 else 'area'

                        reconstruction = F.interpolate(
                            reconstruction,
                            size=target_size,
                            mode=mode,
                            align_corners=False if mode == 'bicubic' else None
                        )

                    # Save reconstructed image
                    img_path = os.path.join(output_dir, f"reconstruction_{idx}.png")
                    self.save_reconstructed_image(reconstruction[0], img_path)

                except Exception as e:
                    logger.error(f"Error processing feature vector {idx}: {str(e)}")

    def save_reconstructed_image(self, tensor: torch.Tensor, path: str):
        """Save reconstructed tensor as image with optimal quality preservation"""
        try:
            tensor = tensor.detach().cpu()

            # Verify channel count
            if tensor.size(0) != self.config['dataset']['in_channels']:
                raise ValueError(f"Expected {self.config['dataset']['in_channels']} channels, got {tensor.size(0)}")

            # Move to [H, W, C] for image saving
            tensor = tensor.permute(1, 2, 0)

            # Get normalization parameters from config
            mean = torch.tensor(self.config['dataset']['mean'], dtype=tensor.dtype)
            std = torch.tensor(self.config['dataset']['std'], dtype=tensor.dtype)

            # Reshape for broadcasting
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)

            # Denormalize
            tensor = tensor * std + mean
            tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)

            # Handle single-channel case
            if tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)

            # Convert to PIL Image
            img = Image.fromarray(tensor.numpy())

            # Verify if any final resizing is needed
            target_size = tuple(self.config['dataset']['input_size'])
            if img.size != target_size:
                # Use LANCZOS for upscaling and BICUBIC for downscaling
                if img.size[0] < target_size[0] or img.size[1] < target_size[1]:
                    resample = Image.Resampling.LANCZOS
                else:
                    resample = Image.Resampling.BICUBIC

                img = img.resize(target_size, resample=resample)

            # Save with maximum quality
            img.save(path, quality=95, optimize=True)
            logger.debug(f"Saved image to {path} with size {img.size}")

        except Exception as e:
            logger.error(f"Error saving reconstructed image: {str(e)}")
            logger.error(f"Tensor shape at error: {tensor.shape if 'tensor' in locals() else 'unknown'}")
            raise



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


def get_feature_extractor(config: Dict, device: Optional[str] = None) -> BaseFeatureExtractor:
    """Get appropriate feature extractor with enhanced image handling"""
    encoder_type = config['model'].get('encoder_type', 'cnn').lower()

    if encoder_type == 'cnn':
        return CNNFeatureExtractor(config, device)
    elif encoder_type == 'autoenc':
        # Always use enhanced version for autoencoder
        return EnhancedAutoEncoderFeatureExtractor(config, device)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

class CustomImageDataset(Dataset):
    """Custom dataset for loading images from directory structure"""
    def __init__(self, data_dir: str, transform=None, csv_file: Optional[str] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.label_encoder = {}
        self.reverse_encoder = {}

        if csv_file and os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file)
        else:
            self.image_files = []
            self.labels = []
            unique_labels = sorted(os.listdir(data_dir))

            for idx, label in enumerate(unique_labels):
                self.label_encoder[label] = idx
                self.reverse_encoder[idx] = label

            # Save label encodings
            encoding_file = os.path.join(data_dir, 'label_encodings.json')
            with open(encoding_file, 'w') as f:
                json.dump({
                    'label_to_id': self.label_encoder,
                    'id_to_label': self.reverse_encoder
                }, f, indent=4)

            for class_name in unique_labels:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            self.image_files.append(os.path.join(class_dir, img_name))
                            self.labels.append(self.label_encoder[class_name])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

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
                 output_dir: str = "data"):
        self.datafile = datafile
        self.datatype = datatype.lower()
        self.output_dir = output_dir

        if self.datatype == 'torchvision':
            self.dataset_name = self.datafile.lower()
        else:
            self.dataset_name = Path(self.datafile).stem.lower()

        self.dataset_dir = os.path.join(output_dir, self.dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.config_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.json")
        self.conf_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.conf")
        self.dbnn_conf_path = os.path.join(self.dataset_dir, "adaptive_dbnn.conf")

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
            # Process the data path first
            processed_path = self._process_data_path(self.datafile)
            return self._process_custom(processed_path)

    def _process_custom(self, data_path: str) -> Tuple[str, Optional[str]]:
        """Process custom dataset structure"""
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        # Check if dataset already has train/test structure
        if os.path.isdir(os.path.join(data_path, "train")) and \
           os.path.isdir(os.path.join(data_path, "test")):
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
        """Get transforms based on configuration"""
        transform_list = []

        # Handle resolution and channel conversion first
        target_size = tuple(config['dataset']['input_size'])
        target_channels = config['dataset']['in_channels']

        # Resolution adjustment
        transform_list.append(transforms.Resize(target_size))

        # Channel conversion
        if target_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        # Training augmentations
        if is_train and config.get('augmentation', {}).get('enabled', True):
            aug_config = config['augmentation']
            if aug_config.get('random_crop', {}).get('enabled', False):
                transform_list.append(transforms.RandomCrop(target_size, padding=4))
            if aug_config.get('horizontal_flip', {}).get('enabled', False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug_config.get('color_jitter', {}).get('enabled', False):
                transform_list.append(transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ))

        # Final transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=config['dataset']['mean'],
                               std=config['dataset']['std'])
        ])

        return transforms.Compose(transform_list)


    def _generate_main_config(self, train_dir: str) -> Dict:
        """Generate main configuration with all necessary parameters"""
        input_size, in_channels = self._detect_image_properties(train_dir)
        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        num_classes = len(class_dirs)

        mean = [0.5] if in_channels == 1 else [0.485, 0.456, 0.406]
        std = [0.5] if in_channels == 1 else [0.229, 0.224, 0.225]
        feature_dims = min(128, np.prod(input_size) // 4)

        return {
            "dataset": {
                "name": self.dataset_name,
                "type": self.datatype,
                "in_channels": in_channels,
                "num_classes": num_classes,
                "input_size": list(input_size),
                "mean": mean,
                "std": std,
                "train_dir": train_dir,
                "test_dir": os.path.join(os.path.dirname(train_dir), 'test')
            },
             "model": {
                "encoder_type": "autoenc",
                "feature_dims": 128,
                "learning_rate": 0.01,
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
                    "convergence_threshold": 0.001,
                    "min_epochs": 10,
                    "patience": 5,
                    "enhancements": {
                        "enabled": True,
                        "use_kl_divergence": True,
                        "use_class_encoding": True,
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
                "batch_size": 32,
                "epochs": 20,
                "num_workers": min(4, os.cpu_count() or 1),
                "checkpoint_dir": os.path.join(self.dataset_dir, "checkpoints"),
                "validation_split": 0.2,
                "invert_DBNN": True,
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.1,
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

    def _generate_dataset_conf(self, feature_dims: int) -> Dict:
        """Generate dataset-specific configuration"""
        return {
            "file_path": os.path.join(self.dataset_dir, f"{self.dataset_name}.csv"),
            "column_names": [f"feature_{i}" for i in range(feature_dims)] + ["target"],
            "separator": ",",
            "has_header": True,
            "target_column": "target",
            "modelType": "Histogram",
            "feature_group_size": 2,
            "max_combinations": 1000,
            "bin_sizes": [21],
            "active_learning": {
                "tolerance": 1.0,
                "cardinality_threshold_percentile": 95,
                "strong_margin_threshold": 0.3,
                "marginal_margin_threshold": 0.1,
                "min_divergence": 0.1
            },
            "training_params": {
                "trials": 100,
                "epochs": 1000,
                "learning_rate": 0.1,
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
                "inverse_learning_rate": 0.1,
                "Save_training_epochs": True,
                "training_save_path": "training_data",
                "enable_vectorized": False,
                "vectorization_warning_acknowledged": False,
                "compute_device": "auto",
                "use_interactive_kbd": False
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

    def _generate_dbnn_config(self, main_config: Dict) -> Dict:
        """Generate DBNN-specific configuration"""
        return {
            "training_params": {
                "trials": main_config['training']['epochs'],
                "epochs": main_config['training']['epochs'],
                "learning_rate": main_config['model']['learning_rate'],
                "test_fraction": 0.2,
                "random_seed": 42,
                "minimum_training_accuracy": 0.95,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "n_bins_per_dim": 20,
                "enable_adaptive": True,
                "invert_DBNN": main_config['training'].get('invert_DBNN', False),
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.1,
                "Save_training_epochs": False,
                "training_save_path": os.path.join(self.dataset_dir, "training_data"),
                "modelType": "Histogram",
                "compute_device": "auto"
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

    def _merge_configs(self, existing: Dict, default: Dict) -> Dict:
        """Recursively merge configs, preserving existing values"""
        result = existing.copy()
        for key, value in default.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
        return result

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


    def _detect_image_properties(self, folder_path: str) -> Tuple[Tuple[int, int], int]:
        """Detect actual image properties but use config values if specified"""
        # Load existing config if available
        config_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'dataset' in config:
                    dataset_config = config['dataset']
                    if all(key in dataset_config for key in ['input_size', 'in_channels']):
                        logger.info("Using image properties from config file")
                        return (tuple(dataset_config['input_size']),
                                dataset_config['in_channels'])

        # Fall back to detection from files
        size_counts = defaultdict(int)
        channel_counts = defaultdict(int)
        samples_checked = 0

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS):
                    try:
                        with Image.open(os.path.join(root, file)) as img:
                            tensor = transforms.ToTensor()(img)
                            height, width = tensor.shape[1], tensor.shape[2]
                            channels = tensor.shape[0]

                            size_counts[(width, height)] += 1
                            channel_counts[channels] += 1
                            samples_checked += 1

                            if samples_checked >= 50:
                                break
                    except Exception as e:
                        logger.warning(f"Could not process image {file}: {str(e)}")
                        continue

            if samples_checked >= 50:
                break

        if not size_counts:
            raise ValueError(f"No valid images found in {folder_path}")

        input_size = max(size_counts.items(), key=lambda x: x[1])[0]
        in_channels = max(channel_counts.items(), key=lambda x: x[1])[0]

        return input_size, in_channels


    def _process_torchvision(self) -> Tuple[str, str]:
        """Process torchvision dataset"""
        dataset_name = self.datafile.upper()
        if not hasattr(datasets, dataset_name):
            raise ValueError(f"Torchvision dataset {dataset_name} not found")

        # Setup paths in dataset-specific directory
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Download and process datasets
        transform = transforms.ToTensor()

        train_dataset = getattr(datasets, dataset_name)(
            root=self.output_dir,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = getattr(datasets, dataset_name)(
            root=self.output_dir,
            train=False,
            download=True,
            transform=transform
        )

        # Save images with class directories
        def save_dataset_images(dataset, output_dir, split_name):
            logger.info(f"Processing {split_name} split...")

            class_to_idx = getattr(dataset, 'class_to_idx', None)
            if class_to_idx:
                idx_to_class = {v: k for k, v in class_to_idx.items()}

            with tqdm(total=len(dataset), desc=f"Saving {split_name} images") as pbar:
                for idx, (img, label) in enumerate(dataset):
                    class_name = idx_to_class[label] if class_to_idx else str(label)
                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)

                    if isinstance(img, torch.Tensor):
                        img = transforms.ToPILImage()(img)

                    img_path = os.path.join(class_dir, f"{idx}.png")
                    img.save(img_path)
                    pbar.update(1)

        save_dataset_images(train_dataset, train_dir, "training")
        save_dataset_images(test_dataset, test_dir, "test")

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

    def _detect_image_properties(self, folder_path: str) -> Tuple[Tuple[int, int], int]:
        """Detect image size and channels from dataset"""
        img_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        size_counts = defaultdict(int)
        channel_counts = defaultdict(int)

        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in img_formats):
                    try:
                        with Image.open(os.path.join(root, file)) as img:
                            tensor = transforms.ToTensor()(img)
                            height, width = tensor.shape[1], tensor.shape[2]
                            channels = tensor.shape[0]

                            size_counts[(width, height)] += 1
                            channel_counts[channels] += 1
                    except Exception as e:
                        logger.warning(f"Could not read image {file}: {str(e)}")
                        continue

            if sum(size_counts.values()) >= 50:
                break

        if not size_counts:
            raise ValueError(f"No valid images found in {folder_path}")

        input_size = max(size_counts, key=size_counts.get)
        in_channels = max(channel_counts, key=channel_counts.get)

        return input_size, in_channels

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

        train_dataset = CustomImageDataset(
            data_dir=train_dir,
            transform=transform
        )

        test_dataset = None
        if test_dir and os.path.exists(test_dir):
            test_dataset = CustomImageDataset(
                data_dir=test_dir,
                transform=transform
            )

    if config['training'].get('merge_datasets', False) and test_dataset is not None:
        return CombinedDataset(train_dataset, test_dataset), None

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
    if hasattr(args, 'encoder_type'):
        config['model']['encoder_type'] = args.encoder_type
    if hasattr(args, 'batch_size'):
        config['training']['batch_size'] = args.batch_size
    if hasattr(args, 'epochs'):
        config['training']['epochs'] = args.epochs
    if hasattr(args, 'workers'):
        config['training']['num_workers'] = args.workers
    if hasattr(args, 'learning_rate'):
        config['model']['learning_rate'] = args.learning_rate
    if hasattr(args, 'cpu'):
        config['execution_flags']['use_gpu'] = not args.cpu
    if hasattr(args, 'debug'):
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
    print("  --batch_size    Batch size for training (default: 32)")
    print("  --epochs        Number of training epochs (default: 20)")
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

def parse_arguments():
    if len(sys.argv) == 1:
        return get_interactive_args()

    parser = argparse.ArgumentParser(description='CDBNN Feature Extractor')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--data', type=str, help='dataset name/path')
    parser.add_argument('--data_type', type=str, choices=['torchvision', 'custom'], default='custom')
    parser.add_argument('--encoder_type', type=str, choices=['cnn', 'autoenc'], default='cnn')
    parser.add_argument('--config', type=str, help='path to configuration file')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--output-dir', type=str, default='data', help='output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--cpu', action='store_true', help='force CPU usage')
    parser.add_argument('--invert-dbnn', action='store_true', help='enable inverse DBNN mode')
    parser.add_argument('--input-csv', type=str, help='input CSV for prediction or inverse DBNN')

    return parser.parse_args()


def save_last_args(args):
    """Save arguments to JSON file"""
    args_dict = vars(args)
    with open('last_run.json', 'w') as f:
        json.dump(args_dict, f, indent=4)

def load_last_args():
    """Load arguments from JSON file"""
    try:
        with open('last_run.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_interactive_args():
    """Get arguments interactively with invert DBNN support"""
    last_args = load_last_args()
    args = argparse.Namespace()
    args.mode = input("\nEnter mode (train/predict) [train]: ").strip().lower() or 'train'

    # Get data type
    while True:
        default = last_args.get('data_type', '') if last_args else ''
        prompt = f"\nEnter dataset type (torchvision/custom) [{default}]: " if default else "\nEnter dataset type (torchvision/custom): "
        data_type = input(prompt).strip().lower() or default
        if data_type in ['torchvision', 'custom']:
            args.data_type = data_type
            break
        print("Invalid type. Please enter 'torchvision' or 'custom'")

    # Get data path/name
    default = last_args.get('data', '') if last_args else ''
    prompt = f"Enter dataset name/path [{default}]: " if default else "Enter dataset name/path: "
    args.data = input(prompt).strip() or default

    # Ask about invert DBNN
    default_invert = last_args.get('invert_dbnn', True) if last_args else True
    invert_response = input(f"Enable inverse DBNN mode? (y/n) [{['n', 'y'][default_invert]}]: ").strip().lower()
    args.invert_dbnn = invert_response == 'y' if invert_response else default_invert

    # If in predict mode and invert DBNN is enabled, ask for input CSV
    if args.mode == 'predict' and args.invert_dbnn:
        default_csv = last_args.get(f'input_csv', '') if last_args else ''
        prompt = f"Enter input CSV path (or leave empty for default) [{default_csv}]: "
        args.input_csv = input(prompt).strip() or default_csv

    # Get encoder type
    while True:
        default = last_args.get('encoder_type', 'cnn') if last_args else 'autoenc'
        prompt = f"Enter encoder type (cnn/autoenc) [{default}]: "
        encoder_type = input(prompt).strip().lower() or default
        if encoder_type in ['cnn', 'autoenc']:
            args.encoder_type = encoder_type
            break
        print("Invalid encoder type. Please enter 'cnn' or 'autoenc'")

    # Optional parameters
    default = last_args.get('batch_size', 32) if last_args else 32
    args.batch_size = int(input(f"Enter batch size [{default}]: ").strip() or default)

    default = last_args.get('epochs', 20) if last_args else 20
    args.epochs = int(input(f"Enter number of epochs [{default}]: ").strip() or default)

    default = last_args.get('output_dir', 'data') if last_args else 'data'
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
    if input("Enable KL divergence clustering? (y/n) [y]: ").lower() != 'n':
        enhancements['use_kl_divergence'] = True
        enhancements['kl_divergence_weight'] = float(input("Enter KL divergence weight (0-1) [0.1]: ") or 0.1)
    else:
        enhancements['use_kl_divergence'] = False

    # Class encoding configuration
    if input("Enable class encoding? (y/n) [y]: ").lower() != 'n':
        enhancements['use_class_encoding'] = True
        enhancements['classification_weight'] = float(input("Enter classification weight (0-1) [0.1]: ") or 0.1)
    else:
        enhancements['use_class_encoding'] = False

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
def main():
    """Main function for CDBNN processing with enhancement configurations"""
    args = None
    try:
        # Setup logging and parse arguments
        logger = setup_logging()
        args = parse_arguments()

        # Process based on mode
        if args.mode == 'train':
            return handle_training_mode(args, logger)
        elif args.mode == 'predict':
            return handle_prediction_mode(args, logger)
        else:
            logger.error(f"Invalid mode: {args.mode}")
            return 1

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        if args and args.debug:
            traceback.print_exc()
        return 1

def handle_training_mode(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle training mode operations"""
    try:
        # Setup paths
        data_name = os.path.splitext(os.path.basename(args.data))[0]
        data_dir = os.path.join('data', data_name)
        config_path = os.path.join(data_dir, f"{data_name}.json")

        # Process dataset
        processor = DatasetProcessor(args.data, args.data_type, getattr(args, 'output_dir', 'data'))
        train_dir, test_dir = processor.process()
        logger.info(f"Dataset processed: train_dir={train_dir}, test_dir={test_dir}")

        # Generate/verify configurations
        logger.info("Generating/verifying configurations...")
        config = processor.generate_default_config(train_dir)

        # Configure enhancements
        config = configure_image_processing(config, logger)

        # Update configuration with command line arguments
        config = update_config_with_args(config, args)

        # Setup data loading
        transform = processor.get_transforms(config)
        train_dataset, test_dataset = get_dataset(config, transform)

        if train_dataset is None:
            raise ValueError("No training dataset available")

        # Create data loaders
        train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, config)

        # Initialize model and loss manager
        model, loss_manager = initialize_model_components(config, logger)

        # Get training confirmation
        if not get_training_confirmation(logger):
            return 0

        # Perform training and feature extraction
        features_dict = perform_training_and_extraction(
            model, train_loader, test_loader, config, loss_manager, logger
        )

        # Save results
        save_training_results(features_dict, model, config, data_dir, data_name, logger)

        logger.info("Processing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error in training mode: {str(e)}")
        raise

def handle_prediction_mode(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle prediction mode operations"""
    try:
        # Setup paths
        data_name = os.path.splitext(os.path.basename(args.data))[0]
        data_dir = os.path.join('data', data_name)

        # Load configuration
        config_path = os.path.join(data_dir, f"{data_name}.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Determine input CSV
        if args.invert_dbnn:
            input_csv = args.input_csv if args.input_csv else os.path.join(data_dir, 'reconstructed_input.csv')
        else:
            input_csv = args.input_csv if args.input_csv else os.path.join(data_dir, f"{data_name}.csv")

        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        # Setup output directory
        output_dir = os.path.join(data_dir, 'predictions', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir, exist_ok=True)

        # Initialize prediction manager and generate predictions
        predictor = PredictionManager(config)
        predictor.predict_from_csv(input_csv, output_dir)

        logger.info("Predictions completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error in prediction mode: {str(e)}")
        if args.debug:
            traceback.print_exc()
        return 1

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
    if input("\nReady to start training. Proceed? (y/n): ").lower() != 'y':
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
    """Perform model training and feature extraction"""
    # Training
    logger.info("Starting model training...")
    # HERE: train_loader is passed but train_dataset isn't stored anywhere
    history = train_model(model, train_loader, config, loss_manager)

    # Feature extraction
    logger.info("Extracting features...")
    # HERE: We need train_dataset but it's not accessible
    features_dict = extract_features_from_model(model, train_loader, test_loader)
    return features_dict

def extract_features_from_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader]
) -> Dict:
    """Extract features from the model"""
    if isinstance(model, (AstronomicalStructurePreservingAutoencoder,
                         MedicalStructurePreservingAutoencoder,
                         AgriculturalPatternAutoencoder)):
        features_dict = model.extract_features_with_class_info(train_loader)

        if test_loader:
            test_features = model.extract_features_with_class_info(test_loader)
            for key in features_dict:
                if isinstance(features_dict[key], torch.Tensor):
                    features_dict[key] = torch.cat([features_dict[key], test_features[key]])
    else:
        train_features, train_labels = model.extract_features(train_loader)
        if test_loader:
            test_features, test_labels = model.extract_features(test_loader)
            features = torch.cat([train_features, test_features])
            labels = torch.cat([train_labels, test_labels])
        else:
            features = train_features
            labels = train_labels
        features_dict = {'features': features, 'labels': labels}

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
    history = train_model(model, train_loader, config, loss_manager)

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
    """Save training results and features"""
    # Save features
    output_path = os.path.join(data_dir, f"{data_name}.csv")
    model.save_features(features_dict, output_path)

    # Save training history if available
    if hasattr(model, 'history') and model.history:
        history_path = os.path.join(data_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert tensor values to float for JSON serialization
            serializable_history = {
                k: [float(v) if isinstance(v, torch.Tensor) else v
                   for v in vals]
                for k, vals in model.history.items()
            }
            json.dump(serializable_history, f, indent=4)

        # Plot training history
        plot_path = os.path.join(data_dir, 'training_history.png')
        if isinstance(model, BaseAutoencoder):
            model.plot_training_history(save_path=plot_path)

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
    sys.exit(main())
