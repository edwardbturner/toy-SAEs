import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, n_samples=1000, signal_dim=10, noise_level=0.1, sparsity=0.3, curvature_scale=1.0, seed=42):
        """
        Create a synthetic dataset with controllable noise, sparsity, and curvature.

        Args:
            n_samples: Number of samples in the dataset
            signal_dim: Dimension of the underlying signal
            noise_level: Standard deviation of the Gaussian noise
            sparsity: Fraction of non-zero coefficients (between 0 and 1)
            curvature_scale: Scale of the Gaussian curvature components
            seed: Random seed for reproducibility
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

        # Generate underlying signal (random orthogonal basis)
        signal_basis = torch.randn(signal_dim, signal_dim, device=device)
        signal_basis, _ = torch.linalg.qr(signal_basis)  # Make it orthogonal

        # Generate sparse coefficients
        coefficients = torch.zeros(n_samples, signal_dim, device=device)
        n_nonzero = int(signal_dim * sparsity)

        # Generate all non-zero indices at once
        nonzero_indices = torch.stack([torch.randperm(signal_dim, device=device)[:n_nonzero] for _ in range(n_samples)])

        # Generate all non-zero values at once
        nonzero_values = torch.randn(n_samples, n_nonzero, device=device)

        # Efficiently assign non-zero values using scatter_
        coefficients.scatter_(1, nonzero_indices, nonzero_values)

        # Generate Gaussian curvature components
        # Create a 1D grid of points for the Gaussian
        x = torch.linspace(-3, 3, signal_dim, device=device)

        # Generate random centers and scales for the Gaussians
        centers = torch.rand(n_samples, 1, device=device) * 6 - 3
        scales = torch.rand(n_samples, 1, device=device) * 2 + 0.5

        # Compute Gaussian curvature for each sample
        curvature = torch.zeros(n_samples, signal_dim, device=device)
        for i in range(n_samples):
            # Compute 1D Gaussian
            gaussian = torch.exp(-((x - centers[i]) ** 2) / (2 * scales[i] ** 2))
            curvature[i] = gaussian * curvature_scale

        # Combine sparse and curvature components
        self.clean_data = coefficients @ signal_basis + curvature

        # Add noise
        self.noise = torch.randn(n_samples, signal_dim, device=device) * noise_level
        self.data = self.clean_data + self.noise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.clean_data[idx]

    def get_clean_data(self):
        return self.clean_data

    def get_noisy_data(self):
        return self.data
