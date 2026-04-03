
import os
import matplotlib.pyplot as plt
import torch

from config import *
from model import VectorField

# Give friendly message if model.pth is not there yet
if not os.path.exists("model.pth"):
    print("model.pth not found.")
    print("Please run `python train.py` first. That trains the model and creates model.pth.")
    print("Then run `python sample.py` to generate images using the trained weights.")
    raise SystemExit(1)

# Initialize the model with the trained parameters
model = VectorField().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Generating with ODE
def generate_with_ODE():
    x = torch.randn(1,28,28, device=device) # pure noise at time=0
    time = torch.tensor([[0.0]], device=device)   # [1,1]
    with torch.no_grad():
        for _ in range(int(1/delta_t)):
            x    += model(x,time) * delta_t
            time += delta_t
    return x

# Generate with SDE
def generate_with_SDE():
    x = torch.randn(1,28,28, device=device) # pure noise at time=0
    time = torch.tensor([[0.0]], device=device)   # [1,1]
    with torch.no_grad():
        for _ in range(int(1/delta_t)):
            diffusion = sde_diffusion_coeff * (1-time) # this diffusion term can be an arbitrary function of time
            epsilon = torch.randn(1,28,28, device=device)
            x    += ((1 + ((diffusion**2) * time/(2*(1-time)))) * model(x,time) + ((diffusion**2)/(2*(1-time))) * x) * delta_t + diffusion * (delta_t**0.5) * epsilon
            time += delta_t
    return x

# Generate images
img_ODE = generate_with_ODE()
img_SDE = generate_with_SDE()

# Plot
plt.figure()
plt.imshow(img_ODE.detach().to('cpu').squeeze(), cmap="grey")
plt.axis("off")
plt.title("Generated with ODE")

plt.figure()
plt.imshow(img_SDE.detach().to('cpu').squeeze(), cmap="grey")
plt.axis("off")
plt.title("Generated with SDE")

plt.show()