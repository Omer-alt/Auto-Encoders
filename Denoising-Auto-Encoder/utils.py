import torch
import torch.optim as optim
import torch.nn as nn

# noise_factor = 0.8


# def add_gaussian_noise(x):
#     return x + noise_factor * torch.randn_like(x)

# import torch

# def add_salt_and_pepper_noise(x, noise_level=0.05):

#     mask = torch.rand_like(x)
#     salt = (mask < noise_level / 2).float()
#     pepper = (mask > 1 - noise_level / 2).float()
#     return x * (1 - salt) * (1 - pepper) + salt * 1.0 + pepper * 0.0


# import torch

def add_random_dropout_noise(x, dropout_prob=0.01):

    mask = torch.rand_like(x)
    dropout_mask = (mask < dropout_prob).float()
    return x * dropout_mask
