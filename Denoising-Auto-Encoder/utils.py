import torch
import torch.optim as optim
import torch.nn as nn

noise_factor = 0.5


def add_noise(x):
    return x + noise_factor * torch.randn_like(x)

