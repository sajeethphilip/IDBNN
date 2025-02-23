import torch.nn as nn
import torch
from basic_utils import BaseAutoencoder
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

