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

