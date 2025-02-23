import torch.nn as nn
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np

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

class SpecificEnhancementConfig(BaseEnhancementConfig):
    """Configuration manager for specific enhancement types"""

    def __init__(self, config: Dict, enhancement_type: str):
        super().__init__(config)
        self.enhancement_type = enhancement_type

    def configure(self) -> None:
        """Configure specific enhancement type"""
        print(f"\nConfiguring {self.enhancement_type.title()} Enhancement:")

        # Enable specific enhancement
        self.config['model']['enhancement_modules'][self.enhancement_type] = {
            'enabled': True
        }

        # Configure general parameters
        self.configure_general_parameters()

        # Adjust learning rates
        self._adjust_learning_rates(1)  # Single enhancement type

        print(f"\n{self.enhancement_type.title()} enhancement configured successfully.")

    def configure_general_parameters(self) -> None:
        """Configure general parameters for specific enhancement type"""
        enhancements = self.config['model']['autoencoder_config']['enhancements']

        # Configure basic parameters
        enhancements['use_kl_divergence'] = True
        enhancements['use_class_encoding'] = True
        enhancements['kl_divergence_weight'] = 0.1
        enhancements['classification_weight'] = 0.1
        enhancements['clustering_temperature'] = 1.0
        enhancements['min_cluster_confidence'] = 0.7

        # Enable phase 2 by default for specific enhancements
        self.config['model']['autoencoder_config']['enable_phase2'] = True


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


class DualDeviceManager:
    def __init__(self, primary_device='cuda', backup_device='cpu'):
        self.primary_device = torch.device(primary_device if torch.cuda.is_available() else 'cpu')
        self.backup_device = torch.device(backup_device)
        self.buffer_size = 1000  # Adjustable buffer size
        self.data_buffer = {}

    def load_data(self, data, key):
        """Load data with backup on CPU"""
        # Store backup on CPU
        self.data_buffer[key] = data.to(self.backup_device)
        # Return compute version on primary device
        return data.to(self.primary_device)

    def retrieve_backup(self, key):
        """Retrieve backup data and move to primary device"""
        if key in self.data_buffer:
            return self.data_buffer[key].to(self.primary_device)
        return None

    def clear_buffer(self):
        """Clear the backup buffer"""
        self.data_buffer.clear()
        torch.cuda.empty_cache()

class DualDeviceDataLoader:
    def __init__(self, dataloader, device_manager):
        self.dataloader = dataloader
        self.device_manager = device_manager

    def __iter__(self):
        for batch_idx, (data, target) in enumerate(self.dataloader):
            # Load to both devices
            data_gpu = self.device_manager.load_data(data, f'batch_{batch_idx}')
            target_gpu = self.device_manager.load_data(target, f'target_{batch_idx}')

            yield data_gpu, target_gpu

            # Clear old batches periodically
            if batch_idx % 10 == 0:
                self.device_manager.clear_buffer()

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
