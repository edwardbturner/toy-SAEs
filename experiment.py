import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from config import ExperimentConfig, default_config
from dataset import SyntheticDataset
from sae import SparseAutoencoder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(
    x: torch.Tensor, x_reconstructed: torch.Tensor, h: torch.Tensor, clean_x: torch.Tensor
) -> dict[str, float]:
    """
    Compute various metrics for the SAE.

    Args:
        x: Input tensor
        x_reconstructed: Reconstructed tensor
        h: Hidden layer activations
        clean_x: Clean input tensor (without noise)

    Returns:
        Dictionary containing various metrics
    """
    try:
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(x_reconstructed, x).item()

        # L1 sparsity loss
        l1_loss = torch.mean(torch.abs(h)).item()

        # Feature sparsity (percentage of features that are non-zero)
        sparsity = (h > 0).float().mean().item()

        # Average feature activation magnitude
        activation_magnitude = torch.mean(h).item()

        # Cosine similarity between clean and reconstructed signals
        clean_np = clean_x.cpu().numpy()
        reconstructed_np = x_reconstructed.cpu().numpy()
        cos_sim = np.mean([cosine_similarity([clean_np[i]], [reconstructed_np[i]])[0][0] for i in range(len(clean_np))])

        # Activation similarity (L2 norm ratio)
        clean_norm = torch.norm(clean_x, dim=1).mean().item()
        reconstructed_norm = torch.norm(x_reconstructed, dim=1).mean().item()
        activation_similarity = reconstructed_norm / clean_norm

        return {
            "reconstruction_loss": reconstruction_loss,
            "l1_loss": l1_loss,
            "sparsity": sparsity,
            "activation_magnitude": activation_magnitude,
            "cosine_similarity": cos_sim,
            "activation_similarity": activation_similarity,
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise


def train_sae(
    train_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    hidden_dim: int,
    sparsity_weight: float = 0.01,
    n_epochs: int = 100,
    lr: float = 0.001,
    early_stopping_patience: int = 10,
) -> dict[str, float]:
    """
    Train a sparse autoencoder and return the test metrics.

    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        sparsity_weight: Weight for L1 regularization
        n_epochs: Number of training epochs
        lr: Learning rate
        early_stopping_patience: Number of epochs to wait before early stopping

    Returns:
        Dictionary containing test metrics
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        model = SparseAutoencoder(input_dim, hidden_dim, sparsity_weight).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in tqdm(range(n_epochs), desc="Training"):
            train_loss = model.train_epoch(train_loader, optimizer, device)

            # Evaluate on test set
            test_metrics = model.evaluate(test_loader, device)
            current_loss = test_metrics["reconstruction_loss"]

            # Learning rate scheduling
            scheduler.step(current_loss)

            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss:.4f}")

        # Load best model state if available
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model.evaluate(test_loader, device)

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


def run_experiment(config: ExperimentConfig = default_config) -> None:
    """
    Run the complete experiment with the given configuration.

    Args:
        config: Experiment configuration. Uses default_config if not specified.
    """
    try:
        # Initialize results storage
        results: dict[bool, dict[Any, dict[str, list[float]]]] = {}
        for is_sparse in [False, True]:
            results[is_sparse] = {}
            for series_val in config.get_varying_params()[config.plot_series]:
                results[is_sparse][series_val] = {metric: [] for metric in config.metrics_to_plot}

        # Get fixed parameters
        fixed_params = config.get_fixed_params()

        # Run experiments
        for is_sparse in [False, True]:
            logger.info(f"\nRunning experiments with {'sparse' if is_sparse else 'non-sparse'} signals")

            for series_val in tqdm(config.get_varying_params()[config.plot_series], desc="Series values"):
                # Create experiment config with current parameters
                current_config = ExperimentConfig(
                    **{
                        **config.__dict__,
                        **fixed_params,
                        config.plot_series: series_val,
                    }
                )

                # Create datasets
                train_dataset = SyntheticDataset(
                    current_config.n_train_samples,
                    current_config.base_signal_dim,
                    current_config.noise_level,
                    sparsity=current_config.signal_sparsity if is_sparse else 1.0,
                    curvature_scale=current_config.curvature_scales[0] if current_config.curvature_scales else 1.0,
                    space_type=current_config.space_type,
                )
                test_dataset = SyntheticDataset(
                    current_config.n_test_samples,
                    current_config.base_signal_dim,
                    current_config.noise_level,
                    sparsity=current_config.signal_sparsity if is_sparse else 1.0,
                    curvature_scale=current_config.curvature_scales[0] if current_config.curvature_scales else 1.0,
                    space_type=current_config.space_type,
                    seed=43,
                )

                train_loader = DataLoader(
                    train_dataset, batch_size=current_config.batch_size, shuffle=True, pin_memory=False, num_workers=0
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=current_config.batch_size, pin_memory=False, num_workers=0
                )

                # Train and evaluate
                metrics = train_sae(
                    train_loader,
                    test_loader,
                    current_config.base_signal_dim,
                    int(current_config.base_signal_dim * current_config.hidden_dim_ratios[0]),
                    current_config.sparsity_weight,
                    current_config.n_epochs,
                    current_config.lr,
                    current_config.early_stopping_patience,
                )

                # Store results
                for metric in config.metrics_to_plot:
                    results[is_sparse][series_val][metric].append(metrics[metric])

        # Plot results
        plot_results(results, config)

    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        raise


def plot_results(results: dict[bool, dict[Any, dict[str, list[float]]]], config: ExperimentConfig) -> None:
    """
    Plot and save the experiment results.

    Args:
        results: Dictionary containing experiment results
        config: Experiment configuration
    """
    try:
        # Get fixed and varying parameters
        fixed_params = config.get_fixed_params()
        varying_params = config.get_varying_params()

        # Get x-axis values and labels
        x_values = varying_params[config.x_axis]
        x_labels = {
            "noise": "Noise Level",
            "dimension_ratio": "Hidden Dimension / Signal Dimension Ratio",
            "curvature": "Curvature Scale",
            "space_type": "Space Type",
        }

        # Get series values and labels
        series_values = varying_params[config.plot_series]
        series_labels = {
            "noise": lambda v: f"Noise Level = {v}",
            "dimension_ratio": lambda v: f"Dim Ratio = {v}",
            "curvature": lambda v: f"Curvature = {v}",
            "space_type": lambda v: f"Space = {v}",
        }

        # Create a plot for each metric
        for metric in config.metrics_to_plot:
            plt.figure(figsize=(10, 6))

            # Plot each series
            for series_val in series_values:
                # Get the results for this series value
                series_results = results[False][series_val]

                # Plot non-sparse case
                (line,) = plt.plot(
                    x_values,
                    series_results[metric],
                    "-",
                    label=series_labels[config.plot_series](series_val),
                )

                # Plot sparse case if configured
                if config.include_sparse:
                    sparse_results = results[True][series_val]

                    plt.plot(
                        x_values,
                        sparse_results[metric],
                        "--",
                        color=line.get_color(),
                        label=f"{series_labels[config.plot_series](series_val)} (sparse)",
                    )

            # Add fixed parameters to title
            fixed_params_str = ", ".join(f"{k} = {v}" for k, v in fixed_params.items())
            plt.xlabel(x_labels[config.x_axis])
            plt.ylabel(f"Test {config.metric_labels[metric]}")
            plt.title(f"{config.metric_labels[metric]} vs {x_labels[config.x_axis]}\nFixed: {fixed_params_str}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(config.output_dir) / f"{config.x_axis}_{metric}.png")
            plt.close()

    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")
        raise


def run_noise_experiment(config: ExperimentConfig = default_config) -> None:
    """
    Run experiment varying noise levels for hyperbolic space.

    Args:
        config: Experiment configuration. Uses hyperbolic_noise_config if not specified.
    """
    try:
        # Define noise levels to test
        noise_levels = np.linspace(0.0, 2.0, 20)  # From no noise to high noise

        # Initialize results storage
        results: dict[str, list[float]] = {
            "reconstruction_loss": [],
            "l1_loss": [],
            "sparsity": [],
            "activation_magnitude": [],
            "cosine_similarity": [],
            "activation_similarity": [],
        }

        # Run experiments for each noise level
        for noise_level in tqdm(noise_levels, desc="Noise levels"):
            # Update config with current noise level
            current_config = ExperimentConfig(**{**config.__dict__, "noise_level": float(noise_level)})

            # Create datasets
            train_dataset = SyntheticDataset(
                current_config.n_train_samples,
                current_config.base_signal_dim,
                current_config.noise_level,
                current_config.signal_sparsity,
                current_config.curvature_scales[0] if current_config.curvature_scales is not None else -1.0,
                space_type=current_config.space_type,
            )
            test_dataset = SyntheticDataset(
                current_config.n_test_samples,
                current_config.base_signal_dim,
                current_config.noise_level,
                current_config.signal_sparsity,
                current_config.curvature_scales[0] if current_config.curvature_scales is not None else -1.0,
                space_type=current_config.space_type,
                seed=43,
            )

            train_loader = DataLoader(
                train_dataset, batch_size=current_config.batch_size, shuffle=True, pin_memory=False, num_workers=0
            )
            test_loader = DataLoader(
                test_dataset, batch_size=current_config.batch_size, pin_memory=False, num_workers=0
            )

            # Train and evaluate
            metrics = train_sae(
                train_loader,
                test_loader,
                current_config.base_signal_dim,
                current_config.base_signal_dim,  # hidden_dim = input_dim
                current_config.sparsity_weight,
                current_config.n_epochs,
                current_config.lr,
                current_config.early_stopping_patience,
            )

            # Store results
            for metric, value in metrics.items():
                results[metric].append(value)

        # Plot results
        plot_noise_results(results, noise_levels, config)

    except Exception as e:
        logger.error(f"Error running noise experiment: {str(e)}")
        raise


def plot_noise_results(results: dict[str, list[float]], noise_levels: np.ndarray, config: ExperimentConfig) -> None:
    """
    Plot and save the noise experiment results.

    Args:
        results: Dictionary containing experiment results
        noise_levels: Array of noise levels tested
        config: Experiment configuration
    """
    try:
        # Create a plot for each metric
        for metric in config.metrics_to_plot:
            plt.figure(figsize=(10, 6))
            plt.plot(noise_levels, results[metric], "-o")
            plt.xlabel("Noise Level")
            plt.ylabel(f"Test {config.metric_labels[metric]}")
            plt.title(f"{config.metric_labels[metric]} vs Noise Level")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(Path(config.output_dir) / f"noise_{metric}.png")
            plt.close()

    except Exception as e:
        logger.error(f"Error plotting noise results: {str(e)}")
        raise


if __name__ == "__main__":
    run_noise_experiment()
