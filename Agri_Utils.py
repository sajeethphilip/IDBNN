import torch.nn as nn
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
from basic_utils import BaseAutoencoder

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
