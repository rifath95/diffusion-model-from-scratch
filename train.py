import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import *
from data import get_batch
from model import VectorField

# Initialize model
model = VectorField()
model = model.to(device)
parameter_size = sum(p.numel() for p in model.parameters())
print(f'{parameter_size} parameters, device {device}')

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    betas=betas,
    weight_decay=weight_decay
)

# Training with a single digit for simplicity

losses = []

for step in range(n_steps):
    time = torch.rand(batch_size,1, device=device)   # [B,1]
    t = time.view(batch_size,1,1)     # [B,1,1]
    
    image, label = get_batch('train', train_digit)   # [B,28,28] , [B] 
    noise = torch.randn(batch_size,28,28, device=device)
    
    noisy_image = t * image + (1-t) * noise   # [B,28,28]
    
    vector_field = model(noisy_image,time)
    
    loss = F.mse_loss(vector_field, image - noise)
    losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % (n_steps//10) == 0:
        print(f'step {step}, loss {loss.item()}')


# Save the trained model parameters
torch.save(model.state_dict(), "model.pth")
print("Saved trained weights to model.pth")

# Plot loss
plt.figure()
plt.plot(losses)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()