import torch.nn as nn
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
from basic_utils import BaseAutoencoder

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
