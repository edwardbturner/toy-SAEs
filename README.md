# toy-SAEs
Experiments of Sparse Autoencoders (SAEs) on Toy Datasets. Note this essentially went no where even with a lot of hyper-parameter playing.

Discussing it in a presentation I found the below from Arthur Conmy rather insightful and would flag it to anyone before using this/doing toy datasets in general:

*We tried toy data as in https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition for a month at GDM in 2023 and it was not useful IMO
*Just training SAEs on tiny real language models like GELU-1L seems more helpful (e.g. I think Sen came up with Gated SAEs and JUmpRelu saes by iterating on Gelu1l)



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
