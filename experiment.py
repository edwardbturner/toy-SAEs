import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from torch.utils.data import DataLoader

from dataset import SyntheticDataset
from sae import SparseAutoencoder


def compute_metrics(x, x_reconstructed, h, clean_x):
    """Compute various metrics for the SAE."""
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


def train_sae(train_loader, test_loader, input_dim, hidden_dim, sparsity_weight=0.01, n_epochs=100, lr=0.001):
    """Train a sparse autoencoder and return the test metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseAutoencoder(input_dim, hidden_dim, sparsity_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(n_epochs):
        train_loss = model.train_epoch(train_loader, optimizer, device)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss:.4f}")

    # Evaluate on test set
    return model.evaluate(test_loader, device)


def run_experiment():
    # Create images directory if it doesn't exist
    os.makedirs("images2", exist_ok=True)

    # Experiment parameters
    base_signal_dim = 10
    hidden_dim_ratios = [0.5, 0.75, 1.0, 1.5, 2.0, 4.0]  # hidden_dim / signal_dim ratios
    n_train_samples = 1000
    n_test_samples = 200
    noise_level = 1.0  # Fixed noise level
    sparsity_weight = 0.01
    n_epochs = 100
    lr = 0.001
    batch_size = 32
    signal_sparsity = 0.3  # Fraction of non-zero coefficients in the signal
    curvature_scales = np.linspace(0.1, 2.0, 20)  # Vary curvature scale

    # Store results for both sparse and non-sparse cases
    results = {}
    for is_sparse in [False, True]:
        results[is_sparse] = {}
        for ratio in hidden_dim_ratios:
            results[is_sparse][ratio] = {
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

    for is_sparse in [False, True]:
        print(f"\nRunning experiments with {'sparse' if is_sparse else 'non-sparse'} signals")
        for ratio in hidden_dim_ratios:
            signal_dim = base_signal_dim
            hidden_dim = int(signal_dim * ratio)
            print(f"  hidden_dim/signal_dim ratio: {ratio}")
            print(f"  signal_dim: {signal_dim}, hidden_dim: {hidden_dim}")

            for curvature_scale in curvature_scales:
                print(f"    Curvature scale: {curvature_scale:.2f}")

                # Create datasets
                train_dataset = SyntheticDataset(
                    n_train_samples,
                    signal_dim,
                    noise_level,
                    sparsity=signal_sparsity if is_sparse else 1.0,
                    curvature_scale=curvature_scale,
                )
                test_dataset = SyntheticDataset(
                    n_test_samples,
                    signal_dim,
                    noise_level,
                    sparsity=signal_sparsity if is_sparse else 1.0,
                    curvature_scale=curvature_scale,
                    seed=43,
                )

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2
                )
                test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=2)

                # Train and evaluate
                metrics = train_sae(train_loader, test_loader, signal_dim, hidden_dim, sparsity_weight, n_epochs, lr)

                # Store results
                for metric, value in metrics.items():
                    results[is_sparse][ratio][metric].append(value)

    # Plot results
    metrics_to_plot = [
        ("reconstruction_loss", "Reconstruction Loss (MSE)"),
        ("l1_loss", "L1 Loss"),
        ("sparsity", "Feature Sparsity"),
        ("activation_magnitude", "Activation Magnitude"),
        ("cosine_similarity", "Cosine Similarity"),
        ("activation_similarity", "Activation Similarity"),
    ]

    # Create a figure for each metric
    for metric, title in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        for ratio in hidden_dim_ratios:
            # Plot non-sparse case
            (line,) = plt.plot(
                curvature_scales, results[False][ratio][metric], "-", label=f"hidden_dim/signal_dim = {ratio}"
            )
            # Plot sparse case with dashed lines using the same color
            plt.plot(
                curvature_scales,
                results[True][ratio][metric],
                "--",
                color=line.get_color(),
                label=f"hidden_dim/signal_dim = {ratio} (sparse)",
            )

        plt.xlabel("Curvature Scale")
        plt.ylabel(f"Test {title}")
        plt.title(f"{title} vs Curvature Scale (Noise Level = {noise_level})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"images2/curvature_generalization_{metric}.png")
        plt.close()

    # Create a summary plot with all metrics for each ratio
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    axes = axes.flatten()

    for ax, ratio in zip(axes, hidden_dim_ratios):
        for metric, title in metrics_to_plot:
            # Plot non-sparse case
            (line,) = ax.plot(curvature_scales, results[False][ratio][metric], "-", label=f"Test {title}")
            # Plot sparse case with dashed lines using the same color
            ax.plot(
                curvature_scales,
                results[True][ratio][metric],
                "--",
                color=line.get_color(),
                label=f"Test {title} (sparse)",
            )

        ax.set_xlabel("Curvature Scale")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"All Metrics (hidden_dim/signal_dim = {ratio}, Noise Level = {noise_level})")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("images2/curvature_generalization_summary.png")
    plt.close()


if __name__ == "__main__":
    run_experiment()
