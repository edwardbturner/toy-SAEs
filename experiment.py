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


def plot_results(results: dict[Any, dict[str, list[float]]], output_dir: str) -> None:
    """
    Plot the experiment results.

    Args:
        results: Dictionary containing experiment results
        output_dir: Directory to save output plots
    """
    try:
        # Define metrics to plot
        metrics = [
            "reconstruction_loss",
            "l1_loss",
            "sparsity",
            "activation_magnitude",
            "cosine_similarity",
            "activation_similarity",
        ]
        metric_labels = {
            "reconstruction_loss": "Reconstruction Loss",
            "l1_loss": "L1 Loss",
            "sparsity": "Sparsity",
            "activation_magnitude": "Activation Magnitude",
            "cosine_similarity": "Cosine Similarity",
            "activation_similarity": "Activation Similarity",
        }

        # Plot each metric
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            x_values = list(results.keys())
            y_values = [np.mean(results[x][metric]) for x in x_values]
            y_errors = [np.std(results[x][metric]) for x in x_values]

            plt.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                capsize=5,
            )

            plt.xlabel("Signal-to-Noise Ratio")
            plt.ylabel(metric_labels[metric])
            plt.title(f"{metric_labels[metric]} vs Signal-to-Noise Ratio")
            plt.grid(True)

            # Save the plot
            plt.savefig(Path(output_dir) / f"{metric}.png")
            plt.close()

    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")
        raise


def run_experiment(
    # Dataset parameters
    base_signal_dim: int = 10,
    n_train_samples: int = 1000,
    n_test_samples: int = 200,
    signal_to_noise_ratio: float = 10.0,  # Higher means cleaner signals
    superposition_multiplier: float = 1.0,  # Controls number of signals
    non_euclidean: float = 0.0,  # 0: Euclidean; 1: fully warped
    non_orthogonal: float = 0.0,  # 0: fully orthogonal; 1: as generated
    hierarchical: float = 0.0,  # 0: independent; 1: clustered
    # Model parameters
    hidden_dim_ratio: float = 1.0,  # Ratio of hidden dimension to input dimension
    sparsity_weight: float = 0.01,  # Weight for L1 regularization
    n_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    early_stopping_patience: int = 10,
    # Experiment parameters
    output_dir: str = "images",
) -> None:
    """
    Run the complete experiment with the given parameters.

    Args:
        base_signal_dim: Dimension of the base signal
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
        signal_to_noise_ratio: Signal-to-noise ratio (higher means cleaner signals)
        superposition_multiplier: Controls number of signals
        non_euclidean: Degree of non-linear warping (0: Euclidean; 1: fully warped)
        non_orthogonal: Degree of non-orthogonality (0: fully orthogonal; 1: as generated)
        hierarchical: Degree of hierarchical structure (0: independent; 1: clustered)
        hidden_dim_ratio: Ratio of hidden dimension to input dimension
        sparsity_weight: Weight for L1 regularization
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        early_stopping_patience: Number of epochs to wait before early stopping
        output_dir: Directory to save output plots
    """
    try:
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Initialize results storage
        results = {
            signal_to_noise_ratio: {
                "reconstruction_loss": [],
                "l1_loss": [],
                "sparsity": [],
                "activation_magnitude": [],
                "cosine_similarity": [],
                "activation_similarity": [],
            }
        }

        # Create datasets
        train_dataset = SyntheticDataset(
            n_samples=n_train_samples,
            activation_size=base_signal_dim,
            signal_to_noise_ratio=signal_to_noise_ratio,
            superposition_multiplier=superposition_multiplier,
            non_euclidean=non_euclidean,
            non_orthogonal=non_orthogonal,
            hierarchical=hierarchical,
        )
        test_dataset = SyntheticDataset(
            n_samples=n_test_samples,
            activation_size=base_signal_dim,
            signal_to_noise_ratio=signal_to_noise_ratio,
            superposition_multiplier=superposition_multiplier,
            non_euclidean=non_euclidean,
            non_orthogonal=non_orthogonal,
            hierarchical=hierarchical,
            seed=43,  # Different seed for test set
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=False, num_workers=0)

        # Train and evaluate
        metrics = train_sae(
            train_loader,
            test_loader,
            base_signal_dim,
            int(base_signal_dim * hidden_dim_ratio),
            sparsity_weight,
            n_epochs,
            lr,
            early_stopping_patience,
        )

        # Store results
        for metric in metrics:
            results[signal_to_noise_ratio][metric].append(metrics[metric])

        # Plot results
        plot_results(results, output_dir)

    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        raise


if __name__ == "__main__":
    run_experiment(
        # Dataset parameters
        base_signal_dim=10,
        n_train_samples=1000,
        n_test_samples=200,
        signal_to_noise_ratio=10.0,
        superposition_multiplier=1.0,
        non_euclidean=0.0,
        non_orthogonal=0.0,
        hierarchical=0.0,
        # Model parameters
        hidden_dim_ratio=1.0,
        sparsity_weight=0.01,
        n_epochs=100,
        lr=0.001,
        batch_size=32,
        early_stopping_patience=10,
        # Experiment parameters
        output_dir="images",
    )
