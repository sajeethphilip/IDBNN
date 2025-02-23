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
from basic_utils import BaseAutoencoder,BaseEnhancementConfig,GeneralEnhancementConfig,EnhancedLossManager,ModelFactory,DatasetProcessor
from Medical_Utils import  MedicalStructurePreservingAutoencoder



# Set sharing strategy at the start
torch.multiprocessing.set_sharing_strategy('file_system')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
