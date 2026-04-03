# MNIST Diffusion Transformer from Scratch

A minimal implementation of a diffusion-style generative model using a Transformer-based architecture (DiT-style) trained on the MNIST dataset.

This project builds a vector field model from scratch and demonstrates both deterministic (ODE) and stochastic (SDE) sampling for image generation.

---

## Features

- Patch-based image tokenization
- Transformer architecture for vector field prediction
- Time conditioning via sinusoidal embeddings
- Adaptive normalization (time-dependent modulation)
- Multi-head self-attention
- Diffusion-style training objective
- Generation using:
  - ODE (deterministic)
  - SDE (stochastic)

---

## Repository Structure

    .
    ├── model.py        # Transformer-based vector field model (DiT-style)
    ├── train.py        # Training loop and loss computation
    ├── sample.py       # Image generation (ODE and SDE)
    ├── data.py         # MNIST loading and batching
    ├── config.py       # Hyperparameters and device setup
    ├── README.md

---

## Dataset

This project uses the **MNIST dataset** (handwritten digits, 28×28 grayscale images), automatically downloaded via `torchvision`.

---

## Setup

## Setup and Usage

### Clone the Repository

    git clone https://github.com/your-username/mnist-dit-from-scratch.git
    cd mnist-dit-from-scratch

---

### Create Environment and Install Dependencies

    python -m venv .venv
    source .venv/bin/activate
    pip install torch torchvision matplotlib

---

### Train the Model

    python train.py

- Trains on MNIST  
- Saves weights as `model.pth`  
- Plots training loss  

---

### Generate Samples

    python sample.py

- Loads `model.pth`  
- Generates images via ODE and SDE  

---

### Notes

- Run `train.py` before `sample.py`  
- By default, trains on digit **0** only  

To train on all digits, set:

```python
train_digit = None
```
---

## Train the Model

    python train.py

This trains a vector field model using a diffusion-style objective.

**Note:**  
For simplicity, training is currently performed on a single digit class (`0`).

You can modify this behavior in `train.py`:

```python
image, label = get_batch('train', train_digit)