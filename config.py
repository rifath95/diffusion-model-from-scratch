import torch

# Device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
# Hyperparameters

## Model
d_hidden = 256
n_layers = 4
n_heads = 16
n_patch = 2
assert 28 % n_patch == 0, 'H not divisible n_patch'
h = 28//n_patch
assert d_hidden % n_heads == 0, 'Hidden dimension not divisible by number of heads'
d_head = d_hidden // n_heads

## Training
batch_size = 128
lr = 1e-3
betas = (0.9, 0.95)
weight_decay = 1e-4

n_steps = 1000
train_digit = 0

## Sampling
delta_t = 0.02
sde_diffusion_coeff = 0.1







