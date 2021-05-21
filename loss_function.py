import torch
from torch.distributions.von_mises import VonMises

# Compute the loss function
def compute_loss(mus, ks, ys):
    # print('mus', mus)
    # print('sizes', mus.shape, ks.shape, ys.shape)
    assert mus.shape == ks.shape == ys.shape
    return -VonMises(mus, ks).log_prob(ys).mean() - 1.0738


# Compute the angle in radians between (mu_x, mu_y) and (0, 1)
def compute_mu_angle(mu_x, mu_y):
    # atan2 takes y-coord as first parameter and x-coord as second parameter
    return torch.atan2(mu_y, mu_x)