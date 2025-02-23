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
