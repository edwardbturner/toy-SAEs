from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass
class ExperimentConfig:
    """Configuration for the sparse autoencoder experiment."""

    # Dataset parameters
    base_signal_dim: int = 10
    hidden_dim_ratios: list[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0, 4.0])
    n_train_samples: int = 1000
    n_test_samples: int = 200
    noise_level: float = 1.0
    signal_sparsity: float = 0.3  # Fraction of non-zero coefficients in the signal
    space_type: Literal["hyperbolic", "spherical", "euclidean"] = "hyperbolic"  # Type of geometric space

    # Model parameters
    sparsity_weight: float = 0.01  # Weight for L1 regularization
    n_epochs: int = 100
    lr: float = 0.001
    batch_size: int = 32
    early_stopping_patience: int = 10

    # Experiment parameters
    output_dir: str = "images"
    curvature_scales: Optional[list[float]] = field(
        default_factory=lambda: None
    )  # Will be set to np.linspace(0.1, 2.0, 20) if None

    # Plotting parameters
    x_axis: Literal["noise", "dimension_ratio", "curvature", "space_type"] = "noise"
    plot_series: Literal["noise", "dimension_ratio", "curvature", "space_type"] = "space_type"
    include_sparse: bool = True
    metrics_to_plot: list[str] = field(
        default_factory=lambda: [
            "reconstruction_loss",
            "l1_loss",
            "sparsity",
            "activation_magnitude",
            "cosine_similarity",
            "activation_similarity",
        ]
    )
    metric_labels: dict[str, str] = field(
        default_factory=lambda: {
            "reconstruction_loss": "Reconstruction Loss (MSE)",
            "l1_loss": "L1 Loss",
            "sparsity": "Feature Sparsity",
            "activation_magnitude": "Activation Magnitude",
            "cosine_similarity": "Cosine Similarity",
            "activation_similarity": "Activation Similarity",
        }
    )

    def __post_init__(self):
        """Initialize derived parameters and validate configuration."""
        if self.curvature_scales is None:
            import numpy as np

            self.curvature_scales = np.linspace(0.1, 2.0, 20).tolist()

        # Validate that x_axis and plot_series are different
        if self.x_axis == self.plot_series:
            raise ValueError("x_axis and plot_series must be different")

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

    def get_fixed_params(self) -> dict[str, Any]:
        """Get the fixed parameter values for the current configuration."""
        fixed_params = {
            "noise": self.noise_level,
            "dimension_ratio": self.hidden_dim_ratios[0],
            "curvature": self.curvature_scales[0] if self.curvature_scales else 1.0,
            "space_type": self.space_type,
        }
        # Remove the varying parameters
        del fixed_params[self.x_axis]
        del fixed_params[self.plot_series]
        return fixed_params

    def get_varying_params(self) -> dict[str, list[Any]]:
        """Get the varying parameter values for the current configuration."""
        return {
            "noise": [self.noise_level],
            "dimension_ratio": self.hidden_dim_ratios,
            "curvature": self.curvature_scales if self.curvature_scales else [1.0],
            "space_type": ["hyperbolic", "spherical", "euclidean"],
        }


# Default configuration
default_config = ExperimentConfig()
