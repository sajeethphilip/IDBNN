import torch.nn as nn
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
from abc import ABC, abstractmethod
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import logging
import csv
import json
import zipfile
import tarfile
import gzip
import bz2
import lzma
import os
import time
import shutil
import glob
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from PIL import Image
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
        """
        Universal feature extraction method for all autoencoder variants.
        Handles both basic and enhanced feature extraction with proper device management.
        """
        self.eval()
        all_embeddings = []
        all_labels = []

        try:
            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc="Extracting features"):
                    # Move data to correct device
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Get embeddings
                    if self.training_phase == 2 and hasattr(self, 'forward'):
                        # Use full forward pass in phase 2 to get all enhancement features
                        outputs = self(inputs)
                        if isinstance(outputs, dict):
                            embeddings = outputs['embedding']
                        else:
                            embeddings = outputs[0]
                    else:
                        # Basic embedding extraction
                        embeddings = self.encode(inputs)
                        if isinstance(embeddings, tuple):
                            embeddings = embeddings[0]

                    # Store results (keeping on device for now)
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)

                # Concatenate all results while still on device
                embeddings = torch.cat(all_embeddings)
                labels = torch.cat(all_labels)

                # Initialize base feature dictionary
                feature_dict = {
                    'embeddings': embeddings,
                    'labels': labels
                }

                # Add enhancement features if in phase 2
                if self.training_phase == 2:
                    # Add clustering information if enabled
                    if self.use_kl_divergence:
                        cluster_info = self.organize_latent_space(embeddings, labels)
                        feature_dict.update(cluster_info)

                    # Add classification information if enabled
                    if self.use_class_encoding and hasattr(self, 'classifier'):
                        class_logits = self.classifier(embeddings)
                        feature_dict.update({
                            'class_logits': class_logits,
                            'class_predictions': class_logits.argmax(dim=1),
                            'class_probabilities': F.softmax(class_logits, dim=1)
                        })

                    # Add specialized features for enhanced models
                    if hasattr(self, 'get_enhancement_features'):
                        enhancement_features = self.get_enhancement_features(embeddings)
                        feature_dict.update(enhancement_features)

                # Move all tensors to CPU for final output
                for key in feature_dict:
                    if isinstance(feature_dict[key], torch.Tensor):
                        feature_dict[key] = feature_dict[key].cpu()

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
        """Save current model state to unified checkpoint"""
        state_key = self.get_state_key(phase, model)

        # Get existing state or create new one
        if state_key not in self.current_state['model_states']:
            self.current_state['model_states'][state_key] = {
                'current': None,
                'best': None,
                'history': []
            }

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

        # Update current state
        self.current_state['model_states'][state_key]['current'] = state_dict

        # Update best state if applicable
        if is_best:
            self.current_state['model_states'][state_key]['best'] = state_dict

        # Add to history (keeping last 5 states)
        self.current_state['model_states'][state_key]['history'].append({
            'epoch': epoch,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        })
        self.current_state['model_states'][state_key]['history'] = \
            self.current_state['model_states'][state_key]['history'][-5:]

        # Update metadata
        self.current_state['metadata']['last_updated'] = datetime.now().isoformat()

        # Save checkpoint
        torch.save(self.current_state, self.checkpoint_path)
        logger.info(f"Saved state {state_key} to unified checkpoint")

        if is_best:
            logger.info(f"New best model for {state_key} with loss: {loss:.4f}")

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
        """Create appropriate model based on configuration"""

        # Create input shape tuple properly
        input_shape = (
            config['dataset']['in_channels'],
            config['dataset']['input_size'][0],
            config['dataset']['input_size'][1]
        )
        feature_dims = config['model']['feature_dims']


        # Determine device
        device = torch.device('cuda' if config['execution_flags']['use_gpu']
                            and torch.cuda.is_available() else 'cpu')


        # Get enabled enhancements
        enhancements = []
        if 'enhancement_modules' in config['model']:
            for module_type, module_config in config['model']['enhancement_modules'].items():
                if module_config.get('enabled', False):
                    enhancements.append(module_type)

        if enhancements:
            logger.info(f"Creating model with enhancements: {', '.join(enhancements)}")

        # Create appropriate model based on image type and enhancements
        image_type = config['dataset'].get('image_type', 'general')
        model = None

        try:
            if image_type == 'astronomical':
                model = AstronomicalStructurePreservingAutoencoder(input_shape, feature_dims, config)
            elif image_type == 'medical':
                model = MedicalStructurePreservingAutoencoder(input_shape, feature_dims, config)
            elif image_type == 'agricultural':
                model = AgriculturalPatternAutoencoder(input_shape, feature_dims, config)
            else:
                # For 'general' type, use enhanced base autoencoder if enhancements are enabled
                if enhancements:
                    model = BaseAutoencoder(input_shape, feature_dims, config)
                else:
                    model = BaseAutoencoder(input_shape, feature_dims, config)

            return model.to(device)

        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

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
