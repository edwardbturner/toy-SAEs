import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from dataset import SyntheticDataset
from sae import BaseAutoencoder, BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE, VanillaSAE

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
    sparsity_weight: float,
    n_epochs: int,
    lr: float,
    early_stopping_patience: int,
    sae_type: str,
    cfg: dict,
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
        sae_type: Type of SAE to train ("vanilla", "batch_topk", or "matryoshka")
        cfg: Configuration dictionary for the SAE

    Returns:
        Dictionary containing test metrics
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Create appropriate SAE model
        sae_cfg = {
            "act_size": input_dim,
            "dict_size": hidden_dim,
            "l1_coeff": sparsity_weight,
            "device": device,
            **cfg,
        }

        model: BaseAutoencoder
        if sae_type == "vanilla":
            model = VanillaSAE(sae_cfg)
        elif sae_type == "batch_topk":
            model = BatchTopKSAE(sae_cfg)
        elif sae_type == "matryoshka":
            model = GlobalBatchTopKMatryoshkaSAE(sae_cfg)
        else:
            raise ValueError(f"Unknown SAE type: {sae_type}")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in tqdm(range(n_epochs), desc=f"Training {sae_type} SAE"):
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


def plot_results(results: dict[str, dict[float, dict[str, list[float]]]], output_dir: str, x_axis_param: str) -> None:
    """
    Plot the experiment results.

    Args:
        results: Dictionary containing experiment results for each SAE type
        output_dir: Directory to save output plots
        x_axis_param: Name of parameter varied on x-axis
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

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

        # Define colors for each SAE type
        colors = {
            "vanilla": "blue",
            "batch_topk": "red",
            "matryoshka": "green",
        }

        # Create readable x-axis parameter name
        x_axis_label = " ".join(x_axis_param.split("_")).title()

        # Create a single figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        # Plot each metric in its own subplot
        for i, metric in enumerate(metrics):
            for sae_type, sae_results in results.items():
                x_values = list(sae_results.keys())
                y_values = [np.mean(sae_results[x][metric]) for x in x_values]

                axes[i].plot(
                    x_values,
                    y_values,
                    "o--",  # Use circles with dashed lines
                    markersize=8,  # Size of the circles
                    color=colors[sae_type],
                    label=sae_type.replace("_", " ").title(),
                )

            axes[i].set_xlabel(x_axis_label)
            axes[i].set_ylabel(metric_labels[metric])
            axes[i].set_title(f"{metric_labels[metric]} vs {x_axis_label}")
            axes[i].grid(True)
            axes[i].legend()

            # Use log scale for x-axis if values span multiple orders of magnitude
            if min(x_values) > 0 and max(x_values) / min(x_values) > 10:
                axes[i].set_xscale("log")

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"all_metrics_{x_axis_param}.png")
        plt.close()

    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")
        raise


def run_experiment(
    # Dataset parameters
    activation_size: int,
    dictionary_size: int,  # Size of the dictionary (hidden dimension)
    n_train_samples: int,
    n_test_samples: int,
    signal_to_noise_ratio: float,
    superposition_multiplier: float,
    non_euclidean: float,
    non_orthogonal: float,
    hierarchical: float,
    # Model parameters
    sparsity_weight: float,
    n_epochs: int,
    lr: float,
    batch_size: int,
    early_stopping_patience: int,
    # Experiment parameters
    output_dir: str,
    x_axis_param: str,
    x_axis_values: list[float],
    cfg: dict,
) -> None:
    """
    Run the complete experiment with the given parameters.

    Args:
        activation_size: Size of each activation vector
        dictionary_size: Size of the dictionary (hidden dimension)
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
        signal_to_noise_ratio: Signal-to-noise ratio (higher means cleaner signals)
        superposition_multiplier: Controls number of signals
        non_euclidean: Degree of non-linear warping (0: Euclidean; 1: fully warped)
        non_orthogonal: Degree of non-orthogonality (0: fully orthogonal; 1: as generated)
        hierarchical: Degree of hierarchical structure (0: independent; 1: clustered)
        sparsity_weight: Weight for L1 regularization
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        early_stopping_patience: Number of epochs to wait before early stopping
        output_dir: Directory to save output plots
        x_axis_param: Parameter to vary on x-axis
        x_axis_values: Values to use for x-axis parameter
        cfg: Configuration dictionary for the SAE
    """
    try:
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Initialize results storage for each SAE type
        results: dict[str, dict[float, dict[str, list[float]]]] = {
            "vanilla": {
                x_val: {
                    metric: []
                    for metric in [
                        "reconstruction_loss",
                        "l1_loss",
                        "sparsity",
                        "activation_magnitude",
                        "cosine_similarity",
                        "activation_similarity",
                    ]
                }
                for x_val in x_axis_values
            },
            "batch_topk": {
                x_val: {
                    metric: []
                    for metric in [
                        "reconstruction_loss",
                        "l1_loss",
                        "sparsity",
                        "activation_magnitude",
                        "cosine_similarity",
                        "activation_similarity",
                    ]
                }
                for x_val in x_axis_values
            },
            "matryoshka": {
                x_val: {
                    metric: []
                    for metric in [
                        "reconstruction_loss",
                        "l1_loss",
                        "sparsity",
                        "activation_magnitude",
                        "cosine_similarity",
                        "activation_similarity",
                    ]
                }
                for x_val in x_axis_values
            },
        }

        # Run experiment for each x-axis value
        for x_val in x_axis_values:
            # Create parameter dictionary with current x-axis value
            current_params = {
                "activation_size": activation_size,
                "signal_to_noise_ratio": signal_to_noise_ratio,
                "superposition_multiplier": superposition_multiplier,
                "non_euclidean": non_euclidean,
                "non_orthogonal": non_orthogonal,
                "hierarchical": hierarchical,
            }
            current_params[x_axis_param] = x_val

            # Create datasets
            train_dataset = SyntheticDataset(
                n_samples=n_train_samples,
                activation_size=current_params["activation_size"],
                signal_to_noise_ratio=current_params["signal_to_noise_ratio"],
                superposition_multiplier=current_params["superposition_multiplier"],
                non_euclidean=current_params["non_euclidean"],
                non_orthogonal=current_params["non_orthogonal"],
                hierarchical=current_params["hierarchical"],
            )
            test_dataset = SyntheticDataset(
                n_samples=n_test_samples,
                activation_size=current_params["activation_size"],
                signal_to_noise_ratio=current_params["signal_to_noise_ratio"],
                superposition_multiplier=current_params["superposition_multiplier"],
                non_euclidean=current_params["non_euclidean"],
                non_orthogonal=current_params["non_orthogonal"],
                hierarchical=current_params["hierarchical"],
                seed=43,  # Different seed for test set
            )

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=False, num_workers=0)

            # Train and evaluate each SAE type
            for sae_type in ["vanilla", "batch_topk", "matryoshka"]:
                metrics = train_sae(
                    train_loader,
                    test_loader,
                    int(current_params["activation_size"]),
                    dictionary_size,  # Use dictionary_size instead of activation_size
                    sparsity_weight,
                    n_epochs,
                    lr,
                    early_stopping_patience,
                    sae_type,
                    cfg,
                )

                # Store results
                for metric in metrics:
                    results[sae_type][x_val][metric].append(metrics[metric])

        # Plot results
        plot_results(results, output_dir, x_axis_param)

    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        raise


if __name__ == "__main__":
    # Dataset parameters
    activation_size = 128  # mimiking d_model
    dictionary_size = 512  # SAE dictionary size
    n_train_samples = 10000
    n_test_samples = 2000
    signal_to_noise_ratio = 10.0  # Higher means cleaner signals
    superposition_multiplier = 1.0  # Controls number of signals
    non_euclidean = 0.0  # 0: Euclidean; 1: fully warped
    non_orthogonal = 0.0  # 0: fully orthogonal; 1: as generated
    hierarchical = 0.0  # 0: independent; 1: clustered

    # Model parameters
    sparsity_weight = 0.01  # Weight for L1 regularization
    n_epochs = 200
    lr = 0.001
    batch_size = 32
    early_stopping_patience = 10

    # Experiment parameters
    output_dir = "images"
    x_axis_param = "non_euclidean"  # Parameter to vary on x-axis

    if x_axis_param == "signal_to_noise_ratio":
        x_axis_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    elif x_axis_param == "superposition_multiplier":
        x_axis_values = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    elif x_axis_param == "non_euclidean":
        x_axis_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    elif x_axis_param == "non_orthogonal":
        x_axis_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    elif x_axis_param == "hierarchical":
        x_axis_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        raise ValueError(f"Unknown x-axis parameter: {x_axis_param}")

    # SAE configuration parameters
    cfg = {
        "dtype": torch.float32,
        "seed": 42,
        "input_unit_norm": False,
        "n_batches_to_dead": 10,
        "top_k": 10,
        "top_k_aux": 5,
        "aux_penalty": 0.1,
        "group_sizes": [
            dictionary_size // 16,
            dictionary_size // 16,
            dictionary_size // 8,
            dictionary_size // 4,
            dictionary_size // 2,
        ],  # For matryoshka SAE, should sum to dictionary_size
    }

    run_experiment(
        activation_size=activation_size,
        dictionary_size=dictionary_size,
        n_train_samples=n_train_samples,
        n_test_samples=n_test_samples,
        signal_to_noise_ratio=signal_to_noise_ratio,
        superposition_multiplier=superposition_multiplier,
        non_euclidean=non_euclidean,
        non_orthogonal=non_orthogonal,
        hierarchical=hierarchical,
        sparsity_weight=sparsity_weight,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        output_dir=output_dir,
        x_axis_param=x_axis_param,
        x_axis_values=x_axis_values,
        cfg=cfg,
    )
