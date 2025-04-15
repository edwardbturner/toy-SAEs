# toy-SAEs
Experiments of Sparse Autoencoders (SAEs) on Toy Datasets

## Overview
This project implements and experiments with Sparse Autoencoders (SAEs) on synthetic datasets generated from different geometric spaces (Euclidean, Hyperbolic, and Spherical). The goal is to study how SAEs perform on data with different geometric properties.

## Features
- Synthetic dataset generation in different geometric spaces
- Configurable SAE architecture
- Training with early stopping and model checkpointing
- Comprehensive metrics tracking
- Visualization of results

## Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy 1.21+
- Matplotlib 3.5+
- scikit-learn 1.0+

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run experiments with different configurations:
```bash
python experiment.py
```

## Project Structure
- `config.py`: Configuration management
- `dataset.py`: Dataset generation and management
- `sae.py`: Sparse Autoencoder implementation
- `experiment.py`: Experiment execution and visualization

## License
MIT License
