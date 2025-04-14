import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_weight=0.01):
        """
        Sparse Autoencoder with L1 regularization on the hidden layer.

        Args:
            input_dim: Dimension of input data
            hidden_dim: Dimension of hidden layer (number of features)
            sparsity_weight: Weight of the L1 regularization term
        """
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_weight = sparsity_weight

        # Initialize weights for better training
        nn.init.kaiming_normal_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_normal_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        # Apply ReLU activation
        h = F.relu(h)
        # Decoder
        x_reconstructed = self.decoder(h)
        return x_reconstructed, h

    def loss(self, x, x_reconstructed, h):
        """Compute all losses in a single forward pass."""
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction="mean")

        # L1 regularization on hidden layer
        l1_loss = torch.mean(torch.abs(h))

        # Total loss
        total_loss = reconstruction_loss + self.sparsity_weight * l1_loss

        return total_loss, reconstruction_loss, l1_loss

    @torch.no_grad()
    def compute_metrics(self, x, x_reconstructed, h, clean_x):
        """Compute all metrics in a single forward pass."""
        # Reconstruction metrics
        reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction="mean").item()
        cos_sim = F.cosine_similarity(x_reconstructed, clean_x, dim=1).mean().item()

        # Activation metrics
        l1_loss = torch.mean(torch.abs(h)).item()
        sparsity = (h > 0).float().mean().item()
        activation_magnitude = torch.mean(h).item()

        # Magnitude preservation
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

    def train_epoch(self, train_loader, optimizer, device):
        """Train for one epoch."""
        self.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_reconstructed, h = self(x)
            loss, _, _ = self.loss(x, x_reconstructed, h)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, test_loader, device):
        """Evaluate the model on test data."""
        self.eval()
        all_metrics = []
        for x, clean_x in test_loader:
            x = x.to(device)
            clean_x = clean_x.to(device)
            x_reconstructed, h = self(x)
            metrics = self.compute_metrics(x, x_reconstructed, h, clean_x)
            all_metrics.append(metrics)

        # Average metrics across batches
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            avg_metrics[metric] = sum(m[metric] for m in all_metrics) / len(all_metrics)
        return avg_metrics
