import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(
        self,
        n_samples=1000,
        signal_dim=10,
        noise_level=0.1,
        sparsity=0.3,
        curvature_scale=1.0,
        seed=42,
        space_type="hyperbolic",  # Options: "hyperbolic", "spherical", "euclidean"
    ):
        """
        Create a synthetic dataset with controllable noise, sparsity, and non-Euclidean geometry.

        Args:
            n_samples: Number of samples in the dataset
            signal_dim: Dimension of the underlying signal
            noise_level: Standard deviation of the Gaussian noise
            sparsity: Fraction of non-zero coefficients (between 0 and 1)
            curvature_scale: Scale of the curvature (negative for hyperbolic, positive for spherical)
            seed: Random seed for reproducibility
            space_type: Type of geometric space ("hyperbolic", "spherical", "euclidean")
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

        # Generate underlying signal (random orthogonal basis)
        signal_basis = torch.randn(signal_dim, signal_dim, device=device)
        signal_basis, _ = torch.linalg.qr(signal_basis)  # Make it orthogonal

        # Generate sparse coefficients
        coefficients = torch.zeros(n_samples, signal_dim, device=device)
        n_nonzero = int(signal_dim * sparsity)

        nonzero_indices = torch.stack([torch.randperm(signal_dim, device=device)[:n_nonzero] for _ in range(n_samples)])
        nonzero_values = torch.randn(n_samples, n_nonzero, device=device)
        coefficients.scatter_(1, nonzero_indices, nonzero_values)

        if space_type == "hyperbolic":
            # Generate points in hyperbolic space (Poincaré ball model)
            # First generate points in Euclidean space
            euclidean_points = torch.randn(n_samples, signal_dim, device=device)
            euclidean_norm = torch.norm(euclidean_points, dim=1, keepdim=True)

            # Project to hyperbolic space using Poincaré ball model
            # The curvature_scale controls the "radius" of the hyperbolic space
            hyperbolic_points = euclidean_points / (1 + torch.sqrt(1 + curvature_scale * euclidean_norm**2))

            # Combine with sparse components
            self.clean_data = coefficients @ signal_basis + hyperbolic_points

        elif space_type == "spherical":
            # Generate points on a sphere
            spherical_points = torch.randn(n_samples, signal_dim, device=device)
            spherical_points = spherical_points / torch.norm(spherical_points, dim=1, keepdim=True)

            # Scale by curvature (radius of the sphere)
            spherical_points = spherical_points * curvature_scale

            # Combine with sparse components
            self.clean_data = coefficients @ signal_basis + spherical_points

        else:  # euclidean
            # Original Euclidean space generation
            self.clean_data = coefficients @ signal_basis

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
