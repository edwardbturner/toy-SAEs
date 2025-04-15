import random

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_synthetic_data(
    seed: int,
    n_samples: int,
    activation_size: int,
    signal_to_noise_ratio: float,  # Signal-to-noise ratio: higher means cleaner signals
    superposition_multiplier: float,  # Controls # signals: n_signals = activation_size * superposition_multiplier
    non_euclidean: float,  # 0: Euclidean; 1: fully warped
    non_orthogonal: float,  # 0: fully orthogonal signals; 1: as generated (non-orthogonal)
    hierarchical: float,  # 0: independent signals; 1: signals grouped in clusters
) -> torch.Tensor:
    """
    Generate synthetic data with control over several characteristics.

    Args:
        seed: Random seed for reproducibility.
        n_samples: Number of samples to generate.
        activation_size: Size of each activation vector.
        signal_to_noise_ratio: Signal-to-noise ratio. Higher values mean cleaner signals.
        superposition_multiplier: Controls number of signals: n_signals = activation_size * superposition_multiplier
        non_euclidean: Fraction (0 to 1) controlling the degree of non-linear warping.
        non_orthogonal: Fraction (0 to 1) controlling the deviation from an orthogonal basis.
                        0 means the signals are forced to be fully orthogonal (if possible),
                        1 uses the generated signals as-is.
        hierarchical: Fraction (0 to 1) controlling the degree of hierarchical structure.
                        0 yields independent signals; 1 forces signals to come from a few clusters.

    Returns:
        A data tensor of shape (n_samples, activation_size)
    """
    # Validate basic parameters
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if activation_size <= 0:
        raise ValueError("activation_size must be positive")
    if signal_to_noise_ratio <= 0:
        raise ValueError("signal_to_noise_ratio must be positive")
    if superposition_multiplier <= 0:
        raise ValueError("superposition_multiplier must be positive")

    for param_name, param_value in [
        ("non_euclidean", non_euclidean),
        ("non_orthogonal", non_orthogonal),
        ("hierarchical", hierarchical),
    ]:
        if not (0.0 <= param_value <= 1.0):
            raise ValueError(f"{param_name} must be between 0 and 1")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Calculate number of signals based on superposition_multiplier
    n_signals = max(1, int(activation_size * superposition_multiplier))

    # Step 1: Generate base random signals and normalize
    random_signals = torch.randn(n_signals, activation_size)
    random_signals = random_signals / random_signals.norm(dim=1, keepdim=True)

    # Step 2: Introduce hierarchical structure if requested
    if hierarchical > 0:
        n_clusters = max(1, int(n_signals * hierarchical))
        # Generate cluster centers (normalized)
        cluster_centers = torch.randn(n_clusters, activation_size)
        cluster_centers = cluster_centers / cluster_centers.norm(dim=1, keepdim=True)
        hierarchical_signals = []
        # For each signal, assign a cluster and add a small noise offset
        for _ in range(n_signals):
            cluster_idx = random.randint(0, n_clusters - 1)
            # You can adjust noise_scale to control how tightly each signal adheres to its cluster center
            noise_scale = 0.1
            sig = cluster_centers[cluster_idx] + torch.randn(activation_size) * noise_scale
            norm = sig.norm()
            sig = sig if norm == 0 else sig / norm
            hierarchical_signals.append(sig)
        stacked_signals = torch.stack(hierarchical_signals)
        # Interpolate between independent signals and the hierarchical version
        signals = (1 - hierarchical) * random_signals + hierarchical * stacked_signals
        signals = signals / signals.norm(dim=1, keepdim=True)
    else:
        signals = random_signals

    # Step 3: Control non-orthogonality.
    # If non_orthogonal < 1, we interpolate signals with an orthogonalized version.
    if non_orthogonal < 1.0:
        if n_signals <= activation_size:
            # Compute a QR decomposition (orthogonalization)
            q, _ = torch.linalg.qr(signals)
            orthogonal_signals = q
        else:
            # If you have more signals than the dimension, full orthogonalization is impossible.
            orthogonal_signals = signals
        # Interpolate between the original and the orthogonal signals.
        signals = (non_orthogonal * signals) + ((1 - non_orthogonal) * orthogonal_signals)
        signals = signals / signals.norm(dim=1, keepdim=True)

    # Step 4: Create coefficient matrix and compute signal component
    coeffs = torch.randn(n_samples, n_signals)
    signal_component = torch.matmul(coeffs, signals)  # Signals are already normalized

    # Step 5: Add noise component with appropriate SNR
    noise_component = torch.randn(n_samples, activation_size)
    # Scale noise to achieve desired SNR
    noise_scale = 1.0 / signal_to_noise_ratio
    data = signal_component + noise_component * noise_scale

    # Step 6: Apply a non-Euclidean transform if requested.
    # Here we warp the data by mixing in a sine nonlinearity.
    if non_euclidean > 0:
        data = (1 - non_euclidean) * data + non_euclidean * torch.sin(data)

    return data


class SyntheticDataset(Dataset):
    def __init__(
        self,
        n_samples=1000,
        activation_size=10,
        signal_to_noise_ratio=10.0,
        superposition_multiplier=1.0,
        non_euclidean=0.0,
        non_orthogonal=0.0,
        hierarchical=0.0,
        seed=42,
    ):
        """
        Create a synthetic dataset with controllable characteristics.

        Args:
            n_samples: Number of samples in the dataset
            activation_size: Size of each activation vector
            signal_to_noise_ratio: Signal-to-noise ratio (higher means cleaner signals)
            superposition_multiplier: Controls # signals (n_signals = activation_size * superposition_multiplier)
            non_euclidean: Fraction (0 to 1) controlling degree of non-linear warping
            non_orthogonal: Fraction (0 to 1) controlling deviation from orthogonal basis
            hierarchical: Fraction (0 to 1) controlling degree of hierarchical structure
            seed: Random seed for reproducibility
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate the data
        self.data = generate_synthetic_data(
            seed=seed,
            n_samples=n_samples,
            activation_size=activation_size,
            signal_to_noise_ratio=signal_to_noise_ratio,
            superposition_multiplier=superposition_multiplier,
            non_euclidean=non_euclidean,
            non_orthogonal=non_orthogonal,
            hierarchical=hierarchical,
        ).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # For now, returning the same data as both input and target

    def get_clean_data(self):
        return self.data

    def get_noisy_data(self):
        return self.data
