import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *


def patchify(image): # image.shape = [B,28,28]
    B = image.size(0)
    x = image.view(B,h,n_patch,h,n_patch)  # [B,h,n_patch,h,n_patch]
    x = x.transpose(2,3)  # [B,h,h,n_patch,n_patch]
    patchified_img = x.reshape(B,h,h,n_patch**2)
    return patchified_img   # [B,h,h,n_patch**2]

def unpatchify(patchified_img): # [B,h,h,n_patch**2]
    B = patchified_img.size(0)
    x = patchified_img.view(B,h,h,n_patch,n_patch)
    x = x.transpose(2,3)   # [B,h,n_patch,h,n_patch]
    unpatcified_img = x.reshape(B,28,28)
    return unpatcified_img

def TimeEmb(time): # time.shape = [B,1]
    assert d_hidden % 2 == 0, 'must have even hidden dimension to do time embedding'
    m = d_hidden//2
    omega_min = 1.0
    omega_max = 10000.0
    k = torch.arange(m, device=time.device, dtype=time.dtype)  # [m]
    omega = omega_min * ((omega_max/omega_min)**(k/(m-1)))  # [m]
    omega_t = 2 * torch.pi * omega * time  # [m] * [B,1] -> [1,m] * [B,1] -> [B(rep),m] * [B,m(rep)] = [B,m]
    time_emb = ((2/d_hidden)**0.5) * torch.cat((torch.cos(omega_t), torch.sin(omega_t)), dim=-1)
    return time_emb  # [B,d_hidden]

class patch_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding  = nn.Linear(n_patch**2, d_hidden, bias=False)
        self.horizontal = nn.Embedding(h, d_hidden)
        self.vertical = nn.Embedding(h,d_hidden)
    def forward(self,patched_images):  # patched_images.shape = [B,h,h,n_patch**2], time.shape = [B,1]
        B = patched_images.size(0)
        token_embedding = self.embedding(patched_images)  # [B,h,h,d_hidden]
        h_emb = self.horizontal(torch.arange(h, device=patched_images.device)).view(1,h,d_hidden)  # [1,h,d_hidden]
        v_emb = self.vertical(torch.arange(h, device=patched_images.device)).view(h,1,d_hidden)    # [h,1,d_hidden]
        token_embedding = token_embedding + h_emb + v_emb     # [B,h,h,d_hidden] + [1,h,d_hidden] + [h,1,d_hidden]  --broadcast-> [B,h,h,d_hidden] + [B(rep),h (rep),h,d_hidden] + [B(rep),h, h (rep),d_hidden]
        token_embedding = token_embedding.view(B,h**2, d_hidden)  # [B,N = h**2, d_hidden]
        return token_embedding  # [B,N,d_hidden]

class AdaptiveNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale  = nn.Linear(d_hidden, d_hidden)
        self.offset = nn.Linear(d_hidden, d_hidden)
    def forward(self, hidden_layer, time_emb):  # [B,N,d_hidden] , [B,d_hidden]
        B,N,d = hidden_layer.shape
        scale  = self.scale(time_emb).view(B,1,d_hidden)   # [B,1,d_hidden]
        offset = self.offset(time_emb).view(B,1,d_hidden)  # [B,1,d_hidden]
        out = (1 + scale) * hidden_layer + offset   # [B,1,d_hidden] * [B,N,d_hidden] + [B,1,d_hidden] --> [B,N(rep),d_hidden] * [B,N,d_hidden] + [B,N(rep),d_hidden] = [B,N,d_hidden] 
        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(d_hidden, 3 * d_hidden)
        self.proj = nn.Linear(d_hidden,d_hidden)
    def forward(self,x):   # x.shape = [B,N,d_hidden]
        B,N,C = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)  #  [B,N,d_hidden]
        q = q.view(B,N,n_heads,d_head).transpose(1,2)  #  [B,n_heads,N,d_head]
        k = k.view(B,N,n_heads,d_head).transpose(1,2)  #  [B,n_heads,N,d_head]
        v = v.view(B,N,n_heads,d_head).transpose(1,2)  #  [B,n_heads,N,d_head]

        wei = (q @ k.transpose(-1,-2)) * (d_head**-0.5)   # [B,n_heads,N,d_head] @ [B,n_heads,d_head,N] = [B,n_heads,N,N]
        wei = F.softmax(wei, dim=-1)

        attn = (wei @ v).transpose(1,2)   # [B,N,n_heads,d_head]
        attn = attn.reshape(B,N,d_hidden)
        attn = self.proj(attn)
        return attn

class Feedforward(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(d_hidden, 4*d_hidden)
        self.gelu = nn.GELU()
        self.down = nn.Linear(4*d_hidden, d_hidden)
    def forward(self,x):
        x = self.up(x)
        x = self.gelu(x)
        x = self.down(x)
        return x

class DiTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attn_ln = nn.LayerNorm(d_hidden)
        self.pre_attn_AdN = AdaptiveNormalization()
        self.attention = MultiHeadedAttention()
        self.pre_mlp_ln = nn.LayerNorm(d_hidden)
        self.pre_mlp_AdN = AdaptiveNormalization()
        self.ffwd = Feedforward()
    def forward(self,tup):
        x,time_emb = tup
        x = x + self.attention(self.pre_attn_AdN(self.pre_attn_ln(x),time_emb))
        x = x + self.ffwd(self.pre_mlp_AdN(self.pre_mlp_ln(x),time_emb))
        tup = (x, time_emb)
        return tup

# VectorField Neural Network Model Architecture

class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = patch_embedding()
        self.blocks = nn.Sequential(*[DiTBlock() for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_hidden)
        self.final_AdN = AdaptiveNormalization()
        self.unembed = nn.Linear(d_hidden,n_patch**2)

    def forward(self,image,time):  # image.shape = [B,28,28], time.shape = [B,1]
        B = image.size(0)
        patchified_image = patchify(image)    # [B,h,h,n_patch**2]
        x = self.embedding(patchified_image)  # [B,N,d_hidden]
        time_emb = TimeEmb(time)   # [B,d_hidden]
        tup = (x,time_emb)
        tup = self.blocks(tup)
        x, time_emb = tup
        x = self.final_ln(x)  # [B,N,d_hidden]
        x = self.final_AdN(x,time_emb)
        x = self.unembed(x)   # [B,N,n_patch**2]

        x = x.view(B,h,h,n_patch**2)
        vectorfield = unpatchify(x)        
        return vectorfield