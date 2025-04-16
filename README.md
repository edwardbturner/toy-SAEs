# toy-SAEs

Experiments with Sparse Autoencoders (SAEs) on toy datasets.  
> **Note**: These experiments didn’t lead anywhere meaningful, even after a lot of hyperparameter tuning.

---

While presenting some work around this, I found the following insight from **Arthur Conmy** worth flagging for anyone considering using toy datasets:

> *"We tried toy data as in [this LessWrong post](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition) for a month at GDM in 2023 and it was not useful IMO. Just training SAEs on tiny real language models like GELU-1L seems more helpful (e.g. I think Sen came up with Gated SAEs and JUmpRelu SAEs by iterating on GELU-1L)."*

---

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
```bash
git clone https://github.com/edwardbturner/toy-SAEs.git
cd toy-SAEs
pip install -r requirements.txt
