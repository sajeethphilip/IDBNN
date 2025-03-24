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
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import json
import torch
import logging
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from torchvision.transforms.functional import resize
from collections import defaultdict

logger = logging.getLogger(__name__)


class PredictionManager:
    """Manages the prediction phase for both CNN and Autoencoder models."""

    def __init__(self, config: Dict, model_path: str, output_dir: str):
        """
        Initialize the PredictionManager.

        Args:
            config (Dict): Configuration dictionary containing dataset and model parameters.
            model_path (str): Path to the trained model checkpoint.
            output_dir (str): Directory where label encoders and other outputs will be saved.
        """
        self.config = config
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = torch.device('cuda' if config['execution_flags']['use_gpu'] and torch.cuda.is_available() else 'cpu')

        # Load label encoders
        self.label_encoder, self.reverse_encoder = self._load_or_generate_label_encoders()

        # Load model with proper enhancement handling
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> torch.nn.Module:
        """Load model with proper enhancement initialization"""
        # Create base model
        if self.config['model']['encoder_type'] == 'cnn':
            model = FeatureExtractorCNN(
                in_channels=self.config['dataset']['in_channels'],
                feature_dims=self.config['model']['feature_dims'],
                config=self.config
            )
        elif self.config['model']['encoder_type'] == 'autoenc':
            model = EnhancedAutoEncoderFeatureExtractor(self.config)
        else:
            raise ValueError(f"Unsupported encoder type: {self.config['model']['encoder_type']}")

        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Handle both unified and enhanced checkpoint formats
        if 'model_states' in checkpoint:  # Enhanced model checkpoint
            state_key = self._get_enhanced_state_key(checkpoint['model_states'])
            if state_key not in checkpoint['model_states']:
                available_keys = ", ".join(checkpoint['model_states'].keys())
                raise ValueError(f"State key {state_key} not found. Available keys: {available_keys}")
            model.load_state_dict(checkpoint['model_states'][state_key]['current']['state_dict'])
        elif 'state_dict' in checkpoint:  # Standard checkpoint
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError("Invalid checkpoint format - missing both 'model_states' and 'state_dict'")

        # Initialize enhancements if needed
        if self.config['model']['autoencoder_config']['enhancements']['use_kl_divergence']:
            if hasattr(model, '_initialize_cluster_centers'):
                model._initialize_cluster_centers()
            elif hasattr(model, 'cluster_centers'):
                num_classes = self.config['dataset'].get('num_classes', 10)
                model.cluster_centers = nn.Parameter(
                    torch.randn(num_classes, self.config['model']['feature_dims']))

        model.to(self.device)
        return model

    def _get_enhanced_state_key(self, model_states: Dict) -> str:
        """Generate state key considering all enhancement combinations"""
        components = []

        # Phase detection
        phase = 2 if self.config['model']['autoencoder_config']['enhancements']['enable_phase2'] else 1
        components.append(f'phase{phase}')

        # KL divergence
        if phase == 2 and self.config['model']['autoencoder_config']['enhancements']['use_kl_divergence']:
            components.append('kld')

        # Class encoding
        if phase == 2 and self.config['model']['autoencoder_config']['enhancements']['use_class_encoding']:
            components.append('cls')

        # Image type enhancements
        img_type = self.config['dataset'].get('image_type', 'general')
        if img_type != 'general':
            components.append(img_type)

        # For CNN models, use a simplified key
        if self.config['model']['encoder_type'] == 'cnn':
            return 'cnn_' + '_'.join(components)

        return '_'.join(components)

    def _get_state_key(self, model_states: Dict) -> str:
        """Autoencoder-specific state key generation"""
        if self.config['model']['encoder_type'] == 'cnn':
            raise ValueError("CNN models don't use state keys")

        phase = 2 if self.config['model']['autoencoder_config']['enhancements']['enable_phase2'] else 1
        components = [f"phase{phase}"]

        if phase == 2:
            if self.config['model']['autoencoder_config']['enhancements']['use_kl_divergence']:
                components.append("kld")
                if self.config['model']['autoencoder_config']['enhancements']['use_class_encoding']:
                    components.append("cls")

                image_type = self.config['dataset'].get('image_type', 'general')
                if image_type != 'general':
                    components.append(image_type)

            return "_".join(components)

    def _load_or_generate_label_encoders(self) -> Tuple[Dict, Dict]:
        """Load label encoders if they exist, otherwise generate them from the input directory."""
        label_encoder_path = os.path.join(self.output_dir, "label_encoder.json")
        reverse_encoder_path = os.path.join(self.output_dir, "reverse_encoder.json")

        if os.path.exists(label_encoder_path) and os.path.exists(reverse_encoder_path):
            # Load existing label encoders
            with open(label_encoder_path, 'r') as f:
                label_encoder = json.load(f)
            with open(reverse_encoder_path, 'r') as f:
                reverse_encoder = json.load(f)
        else:
            # Generate new label encoders from the input directory
            input_dir = self.config.get('dataset', {}).get('train_dir', None)
            if not input_dir or not os.path.exists(input_dir):
                raise FileNotFoundError(
                    f"Training directory not found: {input_dir}. "
                    "Please provide a valid training directory to generate label encoders."
                )

            # Scan the training directory for class folders
            class_dirs = [d for d in os.listdir(input_dir)
                          if os.path.isdir(os.path.join(input_dir, d))]
            if not class_dirs:
                raise ValueError(f"No class directories found in {input_dir}.")

            # Generate label encoders
            label_encoder = {}
            reverse_encoder = {}
            for idx, class_name in enumerate(sorted(class_dirs)):
                label_encoder[class_name] = idx
                reverse_encoder[idx] = class_name

            # Save the label encoders
            os.makedirs(self.output_dir, exist_ok=True)
            with open(label_encoder_path, 'w') as f:
                json.dump(label_encoder, f, indent=4)
            with open(reverse_encoder_path, 'w') as f:
                json.dump(reverse_encoder, f, indent=4)

        return label_encoder, reverse_encoder

    def predict_from_folder(self, folder_path: str, output_csv_path: str) -> None:
        """Predict features with full enhancement support"""
        dataset = CustomImageDataset(folder_path, transform=self._get_transforms())
        dataloader = DataLoader(dataset, batch_size=self.config['training']['batch_size'], shuffle=False)

        # Initialize storage for all possible outputs
        all_features = []
        all_labels = []
        all_filenames = []
        all_class_names = []
        cluster_data = []
        enhancement_features = defaultdict(list)
        # Initialize progress bar
        pbar = tqdm(dataloader,
                    desc="Predicting features",
                    unit="batch",
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
        try:
            features_dict = self.model.extract_features(dataloader)
            self._save_prediction_results(
                all_features=features_dict['features'].numpy(),
                all_labels=features_dict['labels'].tolist(),
                all_filenames=features_dict['filenames'],
                all_class_names=features_dict['class_names'],
                enhancement_features={
                    k: [v] for k, v in features_dict.items()
                    if k not in ['features', 'labels', 'filenames', 'class_names']
                },
                output_csv_path=output_csv_path
            )
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise
        with torch.no_grad():
            for batch_idx, (images, labels, file_indices, filenames) in enumerate(pbar):
                images = images.to(self.device)
                outputs = self.model(images)

                # Handle different model output formats
                if isinstance(outputs, dict):  # Enhanced autoencoder
                    features = outputs['embedding']

                    # Store enhancement outputs
                    for key in ['cluster_probabilities', 'class_probabilities', 'cluster_assignments']:
                        if key in outputs:
                            enhancement_features[key].append(outputs[key].cpu())

                elif isinstance(outputs, tuple):  # Basic autoencoder
                    features = outputs[0]
                else:  # CNN output
                    features = outputs

                # Store core features
                all_features.append(features.cpu())
                all_labels.extend(labels.tolist())
                all_filenames.extend(filenames)
                all_class_names.extend([self.reverse_encoder[label.item()] for label in labels])

        # Process and save all data
        self._save_prediction_results(
            all_features=torch.cat(all_features, dim=0).numpy(),
            all_labels=all_labels,
            all_filenames=all_filenames,
            all_class_names=all_class_names,
            enhancement_features=enhancement_features,
            output_csv_path=output_csv_path
        )

    def _save_prediction_results(self, all_features, all_labels, all_filenames,
                               all_class_names, enhancement_features, output_csv_path):
        """Save all prediction data with enhancement support"""
        feature_columns = [f'feature_{i}' for i in range(all_features.shape[1])]
        data_dict = {col: all_features[:, i] for i, col in enumerate(feature_columns)}

        # Core metadata
        data_dict.update({
            'target': all_labels,
            'filename': all_filenames,
            'class_name': all_class_names
        })

        # Add enhancement data if available
        for key, values in enhancement_features.items():
            if key == 'cluster_probabilities':
                for i in range(values[0].shape[1]):
                    data_dict[f'cluster_{i}_prob'] = torch.cat(values, dim=0)[:, i].numpy()
            elif key == 'class_probabilities':
                for i in range(values[0].shape[1]):
                    data_dict[f'class_{i}_prob'] = torch.cat(values, dim=0)[:, i].numpy()
            elif key == 'cluster_assignments':
                data_dict['cluster_assignment'] = torch.cat(values, dim=0).numpy()

        # Save to CSV
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        pd.DataFrame(data_dict).to_csv(output_csv_path, index=False)

        # Save enhancement metadata
        if enhancement_features:
            meta_path = os.path.join(os.path.dirname(output_csv_path), 'enhancement_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'enhancements_used': list(enhancement_features.keys()),
                    'config': self.config['model']['autoencoder_config']['enhancements']
                }, f, indent=4)


    def _get_transforms(self) -> transforms.Compose:
        """Ensure consistent 256x256 input size"""
        transform_list = [
            transforms.Resize(256),  # First resize to 256x256
            transforms.CenterCrop(256),  # Ensure exact dimensions
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['dataset']['mean'],
                std=self.config['dataset']['std']
            )
        ]

        if self.config['dataset']['in_channels'] == 1:
            transform_list.insert(0, transforms.Grayscale(num_output_channels=1))

        return transforms.Compose(transform_list)

class BaseEnhancementConfig:
    """Base class for enhancement configuration management"""

    def __init__(self, config: Dict):
        self.config = config
        self.initialize_base_config()

    def initialize_base_config(self) -> None:
        """Initialize base configuration structures"""
        if 'model' not in self.config:
            self.config['model'] = {}

        # Initialize autoencoder config
        if 'autoencoder_config' not in self.config['model']:
            self.config['model']['autoencoder_config'] = {
                'phase1_learning_rate': 0.001,
                'phase2_learning_rate': 0.005,
                'reconstruction_weight': 1.0,
                'feature_weight': 0.1,
                'enable_phase2': True,
                'enhancements': {
                    'use_kl_divergence': True,
                    'use_class_encoding': True,
                    'kl_divergence_weight': 0.1,
                    'classification_weight': 0.1,
                    'clustering_temperature': 1.0,
                    'min_cluster_confidence': 0.7
                }
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


class BaseAutoencoder(nn.Module):
    """Base autoencoder class with all foundational methods"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        """Initialize base autoencoder with shape management and all core components.

        Args:
            input_shape: Tuple of (channels, height, width)
            feature_dims: Dimension of latent space features
            config: Configuration dictionary
        """
        super().__init__()

        # Basic configuration
        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.feature_dims = feature_dims
        self.config = config
        self.train_dataset = None

        # Device configuration
        self.device = torch.device('cuda' if config['execution_flags']['use_gpu']
                                 and torch.cuda.is_available() else 'cpu')

        # Shape tracking initialization
        self.shape_registry = {'input': input_shape}
        self.spatial_dims = []
        current_size = input_shape[1]

        # Calculate layer dimensions
        self.layer_sizes = self._calculate_layer_sizes()

        # Track progressive spatial dimensions
        for _ in self.layer_sizes:
            self.spatial_dims.append(current_size)
            current_size = current_size // 2

        # Final dimensions
        self.final_spatial_dim = current_size
        self.flattened_size = self.layer_sizes[-1] * (self.final_spatial_dim ** 2)

        # Register key dimensions in shape registry
        self.shape_registry.update({
            'final_spatial': (self.final_spatial_dim, self.final_spatial_dim),
            'flattened': (self.flattened_size,),
            'latent': (self.feature_dims,)
        })

        # Initialize checkpoint paths
        self.checkpoint_dir = config['training']['checkpoint_dir']
        self.dataset_name = config['dataset']['name']
        self.checkpoint_path = os.path.join(self.checkpoint_dir,
                                          f"{self.dataset_name}_unified.pth")

        # Create network layers
        self.encoder_layers = self._create_encoder_layers()
        self.embedder = self._create_embedder()
        self.unembedder = self._create_unembedder()
        self.decoder_layers = self._create_decoder_layers()

        # Initialize enhancement components
        self.use_kl_divergence = (config['model']
                                 .get('autoencoder_config', {})
                                 .get('enhancements', {})
                                 .get('use_kl_divergence', True))

        self.use_class_encoding = (config['model']
                                  .get('autoencoder_config', {})
                                  .get('enhancements', {})
                                  .get('use_class_encoding', True))

        # Initialize classifier if class encoding is enabled
        if self.use_class_encoding:
            num_classes = config['dataset'].get('num_classes', 10)
            self.classifier = nn.Sequential(
                nn.Linear(feature_dims, feature_dims // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(feature_dims // 2, num_classes)
            )
            self.shape_registry['classifier_output'] = (num_classes,)

        # Initialize clustering if KL divergence is enabled
        if self.use_kl_divergence:
            num_clusters = config['dataset'].get('num_classes', 10)
            self.cluster_centers = nn.Parameter(
                torch.randn(num_clusters, feature_dims)
            )
            self.clustering_temperature = (config['model']
                                         .get('autoencoder_config', {})
                                         .get('enhancements', {})
                                         .get('clustering_temperature', 1.0))
            self.shape_registry['cluster_centers'] = (num_clusters, feature_dims)

        # Training phase tracking
        self.training_phase = 1  # Start with phase 1

        # Initialize latent organization
        self._initialize_latent_organization()

        # Move model to appropriate device
        self.to(self.device)

        # Register shapes for encoder/decoder paths
        for idx, size in enumerate(self.layer_sizes):
            spatial_dim = self.spatial_dims[idx]
            self.shape_registry[f'encoder_{idx}'] = (size, spatial_dim, spatial_dim)
            self.shape_registry[f'decoder_{idx}'] = (size, spatial_dim, spatial_dim)

        # Initialize training metrics
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.current_epoch = 0
        self.history = defaultdict(list)
#--------------------------


#--------------------------
    def set_dataset(self, dataset: Dataset):
        """Store dataset reference"""
        self.train_dataset = dataset

    def _initialize_latent_organization(self):
        """Initialize latent space organization components"""
        self.use_kl_divergence = self.config['model'].get('autoencoder_config', {}).get('enhancements', {}).get('use_kl_divergence', True)
        self.use_class_encoding = self.config['model'].get('autoencoder_config', {}).get('enhancements', {}).get('use_class_encoding', True)

        if self.use_class_encoding:
            num_classes = self.config['dataset'].get('num_classes', 10)
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dims, self.feature_dims // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.feature_dims // 2, num_classes)
            )

        if self.use_kl_divergence:
            num_clusters = self.config['dataset'].get('num_classes', 10)
            self.cluster_centers = nn.Parameter(torch.randn(num_clusters, self.feature_dims))
            self.clustering_temperature = self.config['model'].get('autoencoder_config', {}).get('enhancements', {}).get('clustering_temperature', 1.0)

    def set_training_phase(self, phase: int):
        """Set the training phase (1 or 2)"""
        self.training_phase = phase
        if phase == 2:
            # Initialize cluster centers if in phase 2
            if self.use_kl_divergence:
                # ERROR HERE: Trying to access config['dataset']['train_dataset']
                self._initialize_cluster_centers()

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

    def _calculate_layer_sizes(self) -> List[int]:
        """Calculate progressive channel sizes for encoder/decoder"""
        base_channels = 32
        sizes = []
        current_size = base_channels

        min_dim = min(self.input_shape[1:])
        max_layers = int(np.log2(min_dim)) - 2

        for _ in range(max_layers):
            sizes.append(current_size)
            if current_size < 256:
                current_size *= 2

        logging.info(f"Layer sizes: {sizes}")
        return sizes

    def _create_encoder_layers(self) -> nn.ModuleList:
        """Create encoder layers"""
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

    def _create_embedder(self) -> nn.Sequential:
        """Create embedder layers"""
        return nn.Sequential(
            nn.Linear(self.flattened_size, self.feature_dims),
            nn.BatchNorm1d(self.feature_dims),
            nn.LeakyReLU(0.2)
        )

    def _create_unembedder(self) -> nn.Sequential:
        """Create unembedder layers"""
        return nn.Sequential(
            nn.Linear(self.feature_dims, self.flattened_size),
            nn.BatchNorm1d(self.flattened_size),
            nn.LeakyReLU(0.2)
        )

    def _create_decoder_layers(self) -> nn.ModuleList:
        """Create decoder layers"""
        layers = nn.ModuleList()
        in_channels = self.layer_sizes[-1]

        for i in range(len(self.layer_sizes)-1, -1, -1):
            out_channels = self.in_channels if i == 0 else self.layer_sizes[i-1]
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Basic encoding process"""
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return self.embedder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Basic decoding process"""
        x = self.unembedder(x)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim, self.final_spatial_dim)

        for layer in self.decoder_layers:
            x = layer(x)

        return x

    def forward(self, x: torch.Tensor) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with flexible output format"""
        embedding = self.encode(x)
        if isinstance(embedding, tuple):
            embedding = embedding[0]
        reconstruction = self.decode(embedding)

        if self.training_phase == 2:
            # Return dictionary format in phase 2
            output = {
                'embedding': embedding,
                'reconstruction': reconstruction
            }

            if self.use_class_encoding and hasattr(self, 'classifier'):
                class_logits = self.classifier(embedding)
                output['class_logits'] = class_logits
                output['class_predictions'] = class_logits.argmax(dim=1)

            if self.use_kl_divergence:
                latent_info = self.organize_latent_space(embedding)
                output.update(latent_info)

            return output
        else:
            # Return tuple format in phase 1
            return embedding, reconstruction

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

        img.save(path, quality=95, optimize=True)
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

    def extract_features(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        self.eval()
        all_embeddings = []
        all_labels = []
        all_indices = []  # Store file indices
        all_filenames = []  # Store filenames
        all_class_names = []  # Store actual class names

        try:
            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc="Extracting features"):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Get additional information (file_index and filename)
                    indices = [loader.dataset.get_additional_info(idx)[0] for idx in range(len(inputs))]
                    filenames = [loader.dataset.get_additional_info(idx)[1] for idx in range(len(inputs))]
                    # Get actual class names using reverse_encoder
                    class_names = [loader.dataset.reverse_encoder[label.item()] for label in labels]

                    embeddings = self.encode(inputs)
                    if isinstance(embeddings, tuple):
                        embeddings = embeddings[0]

                    all_embeddings.append(embeddings)
                    all_labels.append(labels)
                    all_indices.extend(indices)  # Append file indices
                    all_filenames.extend(filenames)  # Append filenames
                    all_class_names.extend(class_names)  # Append actual class names

                # Concatenate all results
                embeddings = torch.cat(all_embeddings)
                labels = torch.cat(all_labels)

                feature_dict = {
                    'embeddings': embeddings,
                    'labels': labels,
                    'indices': all_indices,  # Include indices in the feature dictionary
                    'filenames': all_filenames,  # Include filenames in the feature dictionary
                    'class_names': all_class_names  # Include actual class names
                }

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

    def save_features(self, feature_dict: Dict[str, torch.Tensor], output_path: str):
        """
        Universal feature saving method for all autoencoder variants.

        Args:
            feature_dict: Dictionary containing features and related information
            output_path: Path to save the CSV file
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Determine which features to save
            feature_columns = []
            data_dict = {}

            # Process embeddings
            if 'embeddings' in feature_dict:
                embeddings = feature_dict['embeddings'].cpu().numpy()
                for i in range(embeddings.shape[1]):
                    col_name = f'feature_{i}'
                    feature_columns.append(col_name)
                    data_dict[col_name] = embeddings[:, i]

            # Process labels/targets
            if 'labels' in feature_dict:
                data_dict['target'] = feature_dict['labels'].cpu().numpy()
                feature_columns.append('target')

            # Process file indices
            if 'indices' in feature_dict:
                data_dict['file_index'] = feature_dict['indices']
                feature_columns.append('file_index')

            # Process filenames (if available)
            if 'filenames' in feature_dict:
                data_dict['filename'] = feature_dict['filenames']  # Already a list, no need for .cpu()
                feature_columns.append('filename')

            # Process actual class names
            if 'class_names' in feature_dict:
                data_dict['class_name'] = feature_dict['class_names']  # Already a list, no need for .cpu()
                feature_columns.append('class_name')

            # Process enhancement features if present
            enhancement_features = self._get_enhancement_columns(feature_dict)
            data_dict.update(enhancement_features)
            feature_columns.extend(enhancement_features.keys())

            # Save in chunks to manage memory
            chunk_size = 1000
            total_samples = len(next(iter(data_dict.values())))

            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)

                # Create chunk dictionary
                chunk_dict = {
                    col: data_dict[col][start_idx:end_idx]
                    for col in feature_columns
                }

                # Save chunk to CSV
                df = pd.DataFrame(chunk_dict)
                mode = 'w' if start_idx == 0 else 'a'
                header = start_idx == 0
                df.to_csv(output_path, mode=mode, index=False, header=header)

                # Clean up
                del df, chunk_dict
                gc.collect()

            # Save metadata
            self._save_feature_metadata(output_path, feature_columns)

            logger.info(f"Features saved to {output_path}")
            logger.info(f"Total features saved: {len(feature_columns)}")

        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise

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

    def _save_feature_metadata(self, output_path: str, feature_columns: List[str]):
        """Save metadata about the saved features"""
        metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_features': len(feature_columns),
            'feature_columns': feature_columns,
            'model_config': {
                'type': self.__class__.__name__,
                'feature_dims': self.feature_dims,
                'training_phase': self.training_phase,
                'enhancements': {
                    'use_kl_divergence': self.use_kl_divergence,
                    'use_class_encoding': self.use_class_encoding
                }
            }
        }

        metadata_path = os.path.join(
            os.path.dirname(output_path),
            'feature_extraction_metadata.json'
        )

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def organize_latent_space(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Organize latent space using KL divergence and class labels"""
        output = {'embeddings': embeddings}  # Keep on same device as input

        if self.use_kl_divergence:
            # Ensure cluster centers are on same device
            cluster_centers = self.cluster_centers.to(embeddings.device)

            # Calculate distances to cluster centers
            distances = torch.cdist(embeddings, cluster_centers)

            # Convert distances to probabilities (soft assignments)
            q_dist = 1.0 / (1.0 + (distances / self.clustering_temperature) ** 2)
            q_dist = q_dist / q_dist.sum(dim=1, keepdim=True)

            if labels is not None:
                # Create target distribution if labels are provided
                p_dist = torch.zeros_like(q_dist)
                for i in range(self.cluster_centers.size(0)):
                    mask = (labels == i)
                    if mask.any():
                        p_dist[mask, i] = 1.0
            else:
                # Self-supervised target distribution
                p_dist = (q_dist ** 2) / q_dist.sum(dim=0, keepdim=True)
                p_dist = p_dist / p_dist.sum(dim=1, keepdim=True)

            output.update({
                'cluster_probabilities': q_dist,
                'target_distribution': p_dist,
                'cluster_assignments': q_dist.argmax(dim=1)
            })

        if self.use_class_encoding and hasattr(self, 'classifier'):
            # Move classifier to same device if needed
            self.classifier = self.classifier.to(embeddings.device)
            class_logits = self.classifier(embeddings)
            output.update({
                'class_logits': class_logits,
                'class_predictions': class_logits.argmax(dim=1),
                'class_probabilities': F.softmax(class_logits, dim=1)
            })

        return output


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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        return embedding, features

    def decode(self, embedding: torch.Tensor, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enhanced decoding with structure preservation"""
        x = self.unembedder(embedding)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim, self.final_spatial_dim)

        for layer in self.decoder_layers:
            x = layer(x)

        # Final channel transformation back to input channels
        x = nn.Conv2d(self.layer_sizes[0], self.in_channels, kernel_size=1)(x)

        # Add preserved features if available
        if self.enhancement_config['components']['detail_preservation']:
            if 'details' in features:
                x = x + 0.1 * features['details']

        if self.enhancement_config['components']['star_detection']:
            if 'stars' in features:
                x = x * (1 + 0.1 * features['stars'])

        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced feature preservation"""
        embedding, features = self.encode(x)
        reconstruction = self.decode(embedding, features)

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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        return embedding, features

    def decode(self, embedding: torch.Tensor, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enhanced decoding with structure preservation"""
        x = self.unembedder(embedding)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim, self.final_spatial_dim)

        for layer in self.decoder_layers:
            x = layer(x)

        # Add preserved features
        if self.enhancement_config['components']['tissue_boundary']:
            x = x * (1 + 0.1 * features.get('boundaries', 0))

        if self.enhancement_config['components']['contrast_enhancement']:
            x = x + 0.1 * features.get('contrast', 0)

        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced feature preservation"""
        embedding, features = self.encode(x)
        reconstruction = self.decode(embedding, features)

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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        return embedding, features

    def decode(self, embedding: torch.Tensor, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enhanced decoding with pattern preservation"""
        x = self.unembedder(embedding)
        x = x.view(x.size(0), self.layer_sizes[-1],
                  self.final_spatial_dim, self.final_spatial_dim)

        for layer in self.decoder_layers:
            x = layer(x)

        # Add preserved features
        if self.enhancement_config['components']['texture_analysis']:
            x = x + 0.1 * features.get('texture', 0)

        if self.enhancement_config['components']['damage_detection']:
            damage_mask = features.get('damage', torch.zeros_like(x))
            x = x * (1 + 0.2 * damage_mask)

        if self.enhancement_config['components']['color_anomaly']:
            x = x + 0.1 * features.get('color', 0)

        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced pattern preservation"""
        embedding, features = self.encode(x)
        reconstruction = self.decode(embedding, features)

        output = {
            'embedding': embedding,
            'reconstruction': reconstruction
        }

        if self.training and self.enhancement_config['enabled']:
            loss = self.pattern_loss(reconstruction, x)
            output['loss'] = loss

        return output

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

    def calculate_loss(self, reconstruction: torch.Tensor, target: torch.Tensor,
                      image_type: str) -> Dict[str, torch.Tensor]:
        """Calculate loss with appropriate enhancements"""
        loss_fn = self.get_loss_function(image_type)
        if loss_fn is None:
            return {'loss': F.mse_loss(reconstruction, target)}

        loss = loss_fn(reconstruction, target)

        # Get additional statistics if available
        stats = {}
        if isinstance(loss_fn, AgriculturalPatternLoss):
            texture_stats = loss_fn._analyze_texture_statistics(reconstruction)
            pattern_stats = loss_fn._analyze_pattern_distribution(reconstruction)
            stats.update({
                'texture_stats': texture_stats,
                'pattern_stats': pattern_stats
            })

        return {
            'loss': loss,
            'stats': stats
        }


class UnifiedCheckpoint:
    """Manages a unified checkpoint file containing multiple model states"""

    def __init__(self, config: Dict):
        self.config = config
        self.checkpoint_dir = config['training']['checkpoint_dir']
        self.dataset_name = config['dataset']['name']
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.dataset_name}_unified.pth")
        self.current_state = None

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Load or initialize checkpoint
        self.load_checkpoint()

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
        """Generate unique key for current model state"""
        components = [f"phase{phase}"]

        if phase == 2:
            if model.use_kl_divergence:
                components.append("kld")
            if model.use_class_encoding:
                components.append("cls")

            image_type = self.config['dataset'].get('image_type', 'general')
            if image_type != 'general':
                components.append(image_type)

        return "_".join(components)

    def save_model_state(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                         phase: int, epoch: int, loss: float, is_best: bool = False):
        """Save current model state to unified checkpoint."""
        state_key = self.get_state_key(phase, model)

        # Prepare state dictionary
        state_dict = {
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'phase': phase,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'kl_divergence': model.use_kl_divergence,
                'class_encoding': model.use_class_encoding,
                'image_type': self.config['dataset'].get('image_type', 'general')
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

        # Save checkpoint
        torch.save(self.current_state, self.checkpoint_path)
        logger.info(f"Saved state {state_key} to unified checkpoint")

    def load_model_state(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        phase: int, load_best: bool = False) -> Optional[Dict]:
        """Load model state from unified checkpoint"""
        state_key = self.get_state_key(phase, model)

        if state_key not in self.current_state['model_states']:
            logger.info(f"No existing state found for {state_key}")
            return None

        # Get appropriate state
        state_dict = self.current_state['model_states'][state_key]['best' if load_best else 'current']
        if state_dict is None:
            return None

        # Load state
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        logger.info(f"Loaded {'best' if load_best else 'current'} state for {state_key}")
        return state_dict

    def get_best_loss(self, phase: int, model: nn.Module) -> float:
        """Get best loss for current configuration"""
        state_key = self.get_state_key(phase, model)
        if state_key in self.current_state['model_states']:
            best_state = self.current_state['model_states'][state_key]['best']
            if best_state is not None:
                return best_state['loss']
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
        """Factory function with proper enhancement initialization"""
        input_shape = (
            config['dataset']['in_channels'],
            config['dataset']['input_size'][0],
            config['dataset']['input_size'][1]
        )
        feature_dims = config['model']['feature_dims']

        if config['model']['encoder_type'] == 'cnn':
            model = FeatureExtractorCNN(
                in_channels=input_shape[0],
                feature_dims=feature_dims,
                config=config
            )
        else:
            model = EnhancedAutoEncoderFeatureExtractor(config)

        # Initialize enhancements
        enhancements = config['model']['autoencoder_config']['enhancements']
        if enhancements['use_kl_divergence']:
            if hasattr(model, '_initialize_cluster_centers'):
                model._initialize_cluster_centers()
            elif not hasattr(model, 'cluster_centers'):
                num_classes = config['dataset'].get('num_classes', 10)
                model.cluster_centers = nn.Parameter(
                    torch.randn(num_classes, feature_dims)
                )

        return model.to(torch.device('cuda' if config['execution_flags']['use_gpu'] else 'cpu'))

# Update the training loop to handle the new feature dictionary format
def train_model(model: nn.Module, train_loader: DataLoader, config: Dict) -> Dict[str, List]:
    """Training function with proper enhancement handling and accurate metrics"""
    # Configuration setup with validation
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    autoencoder_config = model_config.get('autoencoder_config', {})
    enhancements = autoencoder_config.get('enhancements', {})
    enhancement_modules = model_config.get('enhancement_modules', {})

    # Defaults with validation
    defaults = {
        'learning_rate': 0.001,
        'epochs': 20,
        'enable_phase2': False,
        'reconstruction_weight': 1.0,
        'classification_weight': 0.1,
        'kl_divergence_weight': 0.1,
        'phase2_epoch_ratio': 0.5,
        'feature_loss_weight': 0.01
    }

    # Apply config with validation
    learning_rate = model_config.get('learning_rate', defaults['learning_rate'])
    total_epochs = training_config.get('epochs', defaults['epochs'])
    enable_phase2 = enhancements.get('enable_phase2', defaults['enable_phase2'])
    phase2_epoch = int(total_epochs * enhancements.get('phase2_epoch_ratio', defaults['phase2_epoch_ratio'])) if enable_phase2 else total_epochs

    # Initialize training components
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = defaultdict(list)
    current_phase = 1
    best_accuracy = 0.0

    # Main training loop
    with tqdm(range(total_epochs), desc="Training Progress", unit="epoch") as epoch_pbar:
        for epoch in epoch_pbar:
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            # Phase transition handling
            if enable_phase2 and epoch == phase2_epoch and current_phase == 1:
                current_phase = 2
                model.set_training_phase(2)
                # Update learning rate for phase 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * 0.5  # Phase 2 LR

                if hasattr(model, '_initialize_cluster_centers'):
                    model._initialize_cluster_centers()
                history['phase_transition'] = epoch
                tqdm.write(f"\nTransitioned to Phase 2 at epoch {epoch+1}, LR={learning_rate*0.5}")

            # Batch processing
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False) as batch_pbar:
                for inputs, labels in batch_pbar:
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs)
                    if not isinstance(outputs, dict):
                        raise ValueError(f"Model outputs must be dictionary, got {type(outputs)}")

                    loss_components = {}
                    total_loss = torch.tensor(0.0, device=model.device)
                    batch_correct = 0
                    batch_total = labels.size(0)

                    # UNIVERSAL LOSS COMPONENTS
                    if 'features' in outputs:
                        feat_loss = torch.norm(outputs['features'], p=2)
                        loss_components['feat'] = feat_loss.item()
                        total_loss += defaults['feature_loss_weight'] * feat_loss

                    # ENHANCEMENT LOSSES (only if enabled in config)
                    enhancement_losses = {
                        'star_features': ('astronomical', 0.1),
                        'galaxy_features': ('astronomical', 0.1),
                        'tissue_boundaries': ('medical', 0.1),
                        'texture_features': ('agricultural', 0.1)
                    }

                    for enh_key, (module_name, weight) in enhancement_losses.items():
                        if enh_key in outputs and enhancement_modules.get(module_name, {}).get('enabled', False):
                            enh_loss = torch.norm(outputs[enh_key], p=2)
                            loss_components[enh_key] = enh_loss.item()
                            total_loss += weight * enh_loss

                    # PHASE-SPECIFIC LOSSES
                    if current_phase == 1 and 'reconstruction' in outputs:
                        recon_loss = F.mse_loss(outputs['reconstruction'], inputs)
                        loss_components['recon'] = recon_loss.item()
                        total_loss += enhancements.get('reconstruction_weight', defaults['reconstruction_weight']) * recon_loss

                    elif current_phase == 2 and 'cluster_probabilities' in outputs:
                        target_dist = outputs.get('target_distribution', outputs['cluster_probabilities'].detach())
                        kl_loss = F.kl_div(
                            outputs['cluster_probabilities'].log(),
                            target_dist,
                            reduction='batchmean'
                        )
                        loss_components['kl'] = kl_loss.item()
                        total_loss += enhancements.get('kl_divergence_weight', defaults['kl_divergence_weight']) * kl_loss

                    # CLASSIFICATION LOSS (if enabled)
                    if 'class_logits' in outputs and enhancements.get('use_class_encoding', False):
                        cls_loss = F.cross_entropy(outputs['class_logits'], labels)
                        loss_components['cls'] = cls_loss.item()
                        total_loss += enhancements.get('classification_weight', defaults['classification_weight']) * cls_loss
                        _, predicted = outputs['class_logits'].max(1)
                        batch_correct = predicted.eq(labels).sum().item()
                        epoch_correct += batch_correct
                        epoch_total += batch_total

                    # Verify active loss components
                    if not loss_components:
                        raise ValueError(f"No active loss components. Model outputs: {list(outputs.keys())}")

                    # Backward pass
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.item()

                    # Update batch progress
                    batch_pbar.set_postfix({
                        'loss': f"{total_loss.item():.4f}",
                        **{k: f"{v:.4f}" for k, v in loss_components.items()},
                        'acc': f"{(batch_correct/batch_total*100):.2f}%" if batch_total > 0 else "N/A"
                    })

            # Epoch statistics
            epoch_loss /= len(train_loader)
            history['loss'].append(epoch_loss)

            if epoch_total > 0:
                epoch_acc = epoch_correct / epoch_total
                history['accuracy'].append(epoch_acc)
                if epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc

            # Update epoch progress
            epoch_pbar.set_postfix({
                'loss': f"{epoch_loss:.4f}",
                'acc': f"{(history['accuracy'][-1]*100):.2f}%" if 'accuracy' in history else "N/A",
                'phase': current_phase,
                'best_acc': f"{best_accuracy*100:.2f}%"
            })

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
        self.feature_extractor = self._create_model()
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


        # Load checkpoint or initialize optimizer
        if not self.config['execution_flags'].get('fresh_start', False):
            self._load_from_checkpoint()



    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create and return the feature extraction model"""
        pass

    def load_state_dict(self, state_dict: Dict):
        """Load model state from a state dictionary."""
        self.feature_extractor.load_state_dict(state_dict)
        self.feature_extractor.eval()

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'state_dict': self.feature_extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
            'history': dict(self.history),
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved {'best' if is_best else 'latest'} checkpoint to {path}")


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
            'batch_size': 128,
            'epochs': 20,
            'num_workers': min(4, os.cpu_count() or 1),
            'checkpoint_dir': os.path.join('Model', 'checkpoints'),
            'trials': 100,
            'test_fraction': 0.2,
            'random_seed': 42,
            'minimum_training_accuracy': 0.95,
            'cardinality_threshold': 0.9,
            'cardinality_tolerance': 4,
            'n_bins_per_dim': 128,
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
            'predict': False,
            'reconstruct':True
        })

        # Likelihood configuration
        likelihood = config.setdefault('likelihood_config', {})
        likelihood.update({
            'feature_group_size': 2,
            'max_combinations': 1000,
            'bin_sizes': [128]
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


    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from the dataset using the autoencoder.

        Args:
            loader (DataLoader): DataLoader for the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Extracted features and corresponding labels.
        """
        self.feature_extractor.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Extracting features"):
                inputs = inputs.to(self.device)
                embeddings, _ = self.feature_extractor(inputs)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)

        return torch.cat(all_embeddings), torch.cat(all_labels)

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

        # Initialize tqdm progress bar
        pbar = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {self.current_epoch + 1}",
                    unit="batch",
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

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
            batch_size = 128
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

class FeatureExtractorCNN(nn.Module):
    """Enhanced 7-layer CNN with support for multiple enhancement types"""

    def __init__(self, in_channels: int, feature_dims: int, config: Dict):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.dropout_prob = 0.5
        self.device = torch.device('cuda' if config['execution_flags']['use_gpu']
                                 and torch.cuda.is_available() else 'cpu')

        # Original 7-layer architecture
        self.conv_layers = nn.Sequential(
            # Layer 1: 256x256 -> 128x128
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_prob),

            # Layer 2: 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_prob),

            # Layer 3: 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_prob),

            # Layer 4: 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_prob),

            # Layer 5: 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_prob),

            # Layer 6: 8x8 -> 4x4
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_prob),

            # Layer 7: 4x4 -> 1x1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(512, feature_dims)
        self.batch_norm = nn.BatchNorm1d(feature_dims)

        # Initialize all enhancements
        self._init_enhancements(config)
        self._init_clustering(config)

    def _init_enhancements(self, config: Dict):
        """Initialize all enhancement modules based on config"""
        self.enhancement_modules = nn.ModuleDict()
        enh_config = config['model']['enhancement_modules']

        # Astronomical enhancements
        if enh_config.get('astronomical', {}).get('enabled', False):
            self.enhancement_modules['astronomical'] = nn.ModuleDict({
                'star_detection': nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(512, 1, kernel_size=1),
                    nn.Sigmoid()
                ),
                'galaxy_structure': nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                )
            })

        # Medical enhancements
        if enh_config.get('medical', {}).get('enabled', False):
            self.enhancement_modules['medical'] = nn.ModuleDict({
                'tissue_boundary': nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.PReLU(),
                    nn.Conv2d(512, 1, kernel_size=1),
                    nn.Sigmoid()
                ),
                'lesion_detection': nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.PReLU()
                )
            })

        # Agricultural enhancements
        if enh_config.get('agricultural', {}).get('enabled', False):
            self.enhancement_modules['agricultural'] = nn.ModuleDict({
                'texture_analysis': nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=4),
                    nn.InstanceNorm2d(512),
                    nn.PReLU()
                ),
                'damage_detection': nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.PReLU(),
                    nn.Conv2d(512, 1, kernel_size=1),
                    nn.Sigmoid()
                )
            })

        # Replace forward if any enhancements are enabled
        if len(self.enhancement_modules) > 0:
            self.original_forward = self.forward
            self.forward = self.enhanced_forward

    def _init_clustering(self, config: Dict):
        """Initialize clustering parameters from config"""
        self.use_kl_divergence = config['model']['autoencoder_config']['enhancements']['use_kl_divergence']
        if self.use_kl_divergence:
            num_classes = config['dataset'].get('num_classes', 10)
            self.cluster_centers = nn.Parameter(torch.randn(num_classes, self.feature_dims))
            self.clustering_temperature = config['model']['autoencoder_config']['enhancements']['clustering_temperature']

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict]:
        """Standard forward pass without enhancements"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        features = self.batch_norm(features) if features.size(0) > 1 else features

        if self.use_kl_divergence:
            cluster_outputs = self.organize_latent_space(features)
            return {**cluster_outputs, 'features': features}

        return features

    def enhanced_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with all enabled enhancements"""
        base_output = self.original_forward(x)
        intermediate_features = self.conv_layers[:-2](x)  # Get features before final layers

        enhancements = {}

        # Apply all enabled enhancements
        if 'astronomical' in self.enhancement_modules:
            enhancements.update({
                'star_features': self.enhancement_modules['astronomical']['star_detection'](intermediate_features),
                'galaxy_features': self.enhancement_modules['astronomical']['galaxy_structure'](intermediate_features)
            })

        if 'medical' in self.enhancement_modules:
            enhancements.update({
                'tissue_boundaries': self.enhancement_modules['medical']['tissue_boundary'](intermediate_features),
                'lesion_features': self.enhancement_modules['medical']['lesion_detection'](intermediate_features)
            })

        if 'agricultural' in self.enhancement_modules:
            enhancements.update({
                'texture_features': self.enhancement_modules['agricultural']['texture_analysis'](intermediate_features),
                'damage_features': self.enhancement_modules['agricultural']['damage_detection'](intermediate_features)
            })

        # Combine outputs
        if isinstance(base_output, dict):
            return {**base_output, **enhancements}
        return {'features': base_output, **enhancements}

    def organize_latent_space(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        """KL divergence clustering"""
        device = embeddings.device
        cluster_centers = self.cluster_centers.to(device)

        distances = torch.cdist(embeddings, cluster_centers)
        q_dist = 1.0 / (1.0 + (distances / self.clustering_temperature) ** 2)
        q_dist = q_dist / q_dist.sum(dim=1, keepdim=True)

        if labels is not None and self.training:
            p_dist = torch.zeros_like(q_dist)
            for i in range(cluster_centers.size(0)):
                mask = (labels == i)
                if mask.any():
                    p_dist[mask, i] = 1.0
        else:
            p_dist = (q_dist ** 2) / q_dist.sum(dim=0, keepdim=True)
            p_dist = p_dist / p_dist.sum(dim=1, keepdim=True)

        return {
            'cluster_probabilities': q_dist,
            'target_distribution': p_dist,
            'cluster_assignments': q_dist.argmax(dim=1)
        }

    def extract_features(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Robust feature extraction with exact metadata synchronization"""
        self.eval()
        all_features = []
        all_labels = []
        all_filenames = []
        all_class_names = []
        enhancement_features = defaultdict(list)

        # Get exact dataset size (accounts for partial batches)
        total_samples = len(loader.dataset)
        actual_batch_size = loader.batch_size

        pbar = tqdm(loader, desc="Extracting features", unit="batch",
                   bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(pbar):
                # Get ACTUAL number of samples in this batch (last batch may be smaller)
                actual_samples_in_batch = len(labels)
                inputs = inputs.to(self.device)

                # Forward pass
                outputs = self(inputs)

                # Handle output formats
                if isinstance(outputs, dict):
                    features = outputs.get('features', outputs)
                    # Store enhancements
                    for key, value in outputs.items():
                        if key not in ['features']:
                            enhancement_features[key].append(value.cpu())
                else:
                    features = outputs

                # Store features and labels
                all_features.append(features.cpu())
                batch_labels = labels.tolist()
                all_labels.extend(batch_labels)

                # METADATA COLLECTION WITH PROPER INDEXING
                if hasattr(loader.dataset, 'get_additional_info'):
                    start_idx = batch_idx * actual_batch_size
                    end_idx = min(start_idx + actual_samples_in_batch, total_samples)

                    # Get metadata only for present samples
                    batch_metadata = []
                    for i in range(start_idx, end_idx):
                        try:
                            meta = loader.dataset.get_additional_info(i)
                            batch_metadata.append(meta)
                        except IndexError:
                            logger.warning(f"Metadata missing for index {i}")
                            continue

                    if batch_metadata:
                        indices, filenames = zip(*batch_metadata)
                        all_filenames.extend(filenames)

                # Class names handling
                if hasattr(loader.dataset, 'reverse_encoder'):
                    try:
                        all_class_names.extend([
                            loader.dataset.reverse_encoder[label.item()]
                            for label in labels
                        ])
                    except KeyError as e:
                        logger.error(f"Missing class mapping for label {e}")
                        raise ValueError(f"Class mapping missing for one or more labels")

                pbar.set_postfix({
                    'batch': f"{batch_idx + 1}/{len(loader)}",
                    'features': f"{features.shape[-1]}D"
                })

        # VALIDATION WITH TRUE SAMPLE COUNT
        if len(all_labels) != total_samples:
            logger.error(f"CRITICAL: Label count mismatch. Expected {total_samples}, got {len(all_labels)}")
            logger.error(f"Possible data loader issue. Checking batch sizes...")
            logger.error(f"Configured batch size: {actual_batch_size}")
            logger.error(f"Total batches: {len(loader)}")
            logger.error(f"Calculated samples: {len(loader)*actual_batch_size} vs dataset: {total_samples}")
            raise ValueError(f"Label count mismatch. Expected {total_samples}, got {len(all_labels)}")

        # Prepare final output with exact synchronization
        feature_dict = {
            'features': torch.cat(all_features),
            'labels': torch.tensor(all_labels),
        }

        # Only include metadata if we have ALL samples accounted for
        if len(all_filenames) == total_samples:
            feature_dict['filenames'] = all_filenames
        else:
            logger.warning(f"Filename count mismatch. Expected {total_samples}, got {len(all_filenames)}")

        if len(all_class_names) == total_samples:
            feature_dict['class_names'] = all_class_names
        else:
            logger.warning(f"Class name count mismatch. Expected {total_samples}, got {len(all_class_names)}")

        # Enhancement features
        for key, values in enhancement_features.items():
            if values and len(values) == len(all_features):
                feature_dict[key] = torch.cat(values, dim=0)
            else:
                logger.warning(f"Enhancement feature '{key}' count mismatch")

        return feature_dict

class FeatureExtractorCNN_old(nn.Module):
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
        self.input_shape = input_shape  # e.g., (3, 32, 32) for CIFAR
        self.in_channels = input_shape[0]  # Store input channels explicitly
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
        """Calculate progressive channel sizes for encoder/decoder"""
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


class CNNFeatureExtractor(BaseFeatureExtractor):
    """CNN-based feature extractor implementation"""

    def _create_model(self) -> nn.Module:
        """Create CNN model"""
        return FeatureExtractorCNN(
            in_channels=self.config['dataset']['in_channels'],
            feature_dims=self.feature_dims,
            config=self.config
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

                # Debug: Ensure model is initialized
                if not hasattr(self, 'feature_extractor') or self.feature_extractor is None:
                    logger.error("Model (feature_extractor) is not initialized. Initializing now...")
                    self.feature_extractor = self._create_model()

                # Determine the appropriate device for loading
                if torch.cuda.is_available():
                    map_location = self.device  # Use GPU if available
                else:
                    map_location = torch.device('cpu')  # Fallback to CPU

                # Load checkpoint with weights_only=True for security
                checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

                # Debug: Verify checkpoint contents
                if 'state_dict' not in checkpoint:
                    logger.error("Checkpoint does not contain 'state_dict'. Cannot load model.")
                    raise KeyError("Checkpoint missing 'state_dict'")

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

        pbar = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Train Epoch {self.current_epoch + 1}",
                    unit="batch",
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

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
    def __init__(self, data_dir: str, transform=None, csv_file: Optional[str] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.file_indices = []  # Store file indices
        self.filenames = []  # Initialize filenames list
        self.label_encoder = {}
        self.reverse_encoder = {}

        if csv_file and os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file)
        else:
            unique_labels = sorted(os.listdir(data_dir))
            for idx, label in enumerate(unique_labels):
                self.label_encoder[label] = idx
                self.reverse_encoder[idx] = label

            for class_name in unique_labels:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            self.image_files.append(os.path.join(class_dir, img_name))
                            self.labels.append(self.label_encoder[class_name])
                            self.file_indices.append(len(self.image_files) - 1)  # Assign unique index
                            self.filenames.append(img_name)  # Populate filenames list

    def __len__(self):
        return len(self.image_files)

    def get_additional_info(self, idx):
        """Retrieve additional information (file_index and filename) for a given index."""
        return self.file_indices[idx], self.filenames[idx]

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        file_index = self.file_indices[idx]  # Retrieve file index
        filename = self.filenames[idx]  # Retrieve filename

        if self.transform:
            image = self.transform(image)
            image = transforms.functional.resize(image, (256, 256),antialias=True)

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
        transform_list = [
                    transforms.Resize(256),  # First resize to ensure minimum size
                    transforms.CenterCrop(256),  # Then crop to exact size
                ]

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
                "batch_size": 128,
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
            "max_combinations": 10000,
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
                "learning_rate": 0.001,
                "batch_size":128,
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
                "batch_size":128,
                "test_fraction": 0.2,
                "random_seed": 42,
                "minimum_training_accuracy": 0.95,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "n_bins_per_dim": 128,
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
    print("  --batch_size    Batch size for training (default: 128)")
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
    parser.add_argument('--mode', choices=['train', 'reconstruct','predict'], default='train')
    parser.add_argument('--data', type=str, help='dataset name/path')
    parser.add_argument('--data_type', type=str, choices=['torchvision', 'custom'], default='custom')
    parser.add_argument('--encoder_type', type=str, choices=['cnn', 'autoenc'], default='cnn')
    parser.add_argument('--config', type=str, help='path to configuration file')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--output-dir', type=str, default='data', help='output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
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
    """Get arguments interactively with invert DBNN support."""
    last_args = load_last_args()
    args = argparse.Namespace()

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

    # Get data path/name
    default = last_args.get('data', '') if last_args else ''
    prompt = f"Enter dataset name/path [{default}]: " if default else "Enter dataset name/path: "
    args.data = input(prompt).strip() or default

    # Handle predict mode
    if args.mode == 'predict':
        # Set default model path
        dataset_name = Path(args.data).stem if args.data else 'dataset'
        default_model = (f"data/{dataset_name}/checkpoints/{dataset_name}_unified.pth")
        prompt = f"Enter path to trained model [{default_model}]: "
        args.model_path = input(prompt).strip() or default_model

        # Set default input directory
        default_input = args.data if args.data else ''
        prompt = f"Enter directory containing new images [{default_input}]: "
        args.input_dir = input(prompt).strip() or default_input

        # Set default output CSV path
        default_csv = os.path.join('data', dataset_name, f"{dataset_name}_predictions.csv")
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
    default = last_args.get('batch_size', 128) if last_args else 128
    args.batch_size = int(input(f"Enter batch size [{default}]: ").strip() or default)

    if args.mode == 'train':
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
        if args.mode == 'predict':
            # Load the config
            config_path = os.path.join(args.output_dir, args.data, f"{args.data}.json")
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Initialize the PredictionManager
            predictor = PredictionManager(
                config=config,
                model_path=args.model_path,
                output_dir=args.output_dir
            )

            # Perform predictions
            logger.info("Starting prediction process...")
            predictor.predict_from_folder(
                folder_path=args.input_dir,
                output_csv_path=args.output_csv
            )
            logger.info(f"Predictions saved to {args.output_csv}")

        elif args.mode == 'train':
            return handle_training_mode(args, logger)
        elif args.mode == 'reconstruct':
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
        print("Features Extracted")
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
    if input("\nReady to start training. Proceed? (y/n): ").lower() == 'n':
        logger.info("Training cancelled by user")
        return False
    return True



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
    history = train_model(model, train_loader, config)

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
