import torch
from torch.distributions.von_mises import VonMises
import math
import torch.nn as nn

bce = nn.BCELoss()

# Compute the loss function
def compute_loss(mus, ks, ys):
    # print('mus', mus)
    # print('sizes', mus.shape, ks.shape, ys.shape)
    assert mus.shape == ks.shape == ys.shape
    return -VonMises(mus, ks).log_prob(ys).mean() # - 1.0738

def compute_loss_regression(mus, ys):
    phis = torch.abs(mus - ys)
    phis = torch.remainder(phis, math.pi * 2) # this shouldn't happen though
    distance = 0
    for phi in phis:
        distance += (math.pi * 2) - phi if phi > math.pi else phi
    return distance / mus.shape[0]

def compute_loss_regression2(mu_x, mu_y, ys):
    y_x = torch.cos(ys)
    y_y = torch.sin(ys)
    i = mu_x * y_x + mu_y * y_y
    if torch.isnan(i).any():
        print(mu_x)
        print(mu_y)
        print(ys)
        raise Exception('1', i)
    # i = torch.clamp(i, min=-1, max=1)
    if (i < -1).any() or (i > 1).any():
        print('mu_x', mu_x)
        print('y_x', y_x)
        print('mu_y', mu_y)
        print('y_y', y_y)
        
        raise Exception('out of bounds', i)

    acos = torch.acos(i)
    if torch.isnan(acos).any():
        raise Exception('2', acos, i)
    res = torch.mean(acos)
    #for i in range(acos.shape[0]):
        # -0.94677734375 -0.3212890625 0.9432709813117981 0.332023948431015 nan
        #print(float(mu_x[i]), float(mu_y[i]), float(y_x[i]), float(y_y[i]), float(acos[i]))
    return res

# Compute the angle in radians between (mu_x, mu_y) and (0, 1)
def compute_mu_angle(mu_x, mu_y):
    # atan2 takes y-coord as first parameter and x-coord as second parameter
    return torch.atan2(mu_y, mu_x)