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

def compute_loss_regression(mu_x, mu_y, ys):
    # convert mu to angle
    mus = torch.atan2(mu_y, mu_x) # angle with (1,0) -> between -pi and pi
    for i in range(mus.shape[0]):
        val = mus[i]
        if val < 0:
            val = 2 * math.pi + val
        mus[i] = val

    if (mus < 0).any() or (mus > 2*math.pi).any():
        raise Exception("mus out of bounds 0 to 2pi ", mus)

    if (ys < 0).any() or (ys > 2*math.pi).any():
        raise Exception("ys out of bounds 0 to 2pi ", ys)

    phis = torch.abs(mus - ys) # diff between 0 and 2pi
    if (phis < 0).any() or (phis > 2*math.pi).any():
        raise Exception("phis out of bounds ", phis)

    # phis = torch.remainder(phis, math.pi) # this shouldn't happen though
    distance = 0
    for phi in phis:
        # if phi < -math.pi or phi > math.pi:
        #    raise Exception("Phi out of bounds", phi)
        # distance += (math.pi * 2) - phi if phi > math.pi else phi
        # distance += phi % (math.pi * 2) - math.pi

        # between 0 and 2pi
        if phi <= math.pi:
            distance += phi
        else: # if phi > math.pi
            distance += ((2 * math.pi) - phi)

    return distance / mus.shape[0]
    # """

def compute_loss_regression2(mu_x, mu_y, ys):
    y_x = torch.cos(ys)
    y_y = torch.sin(ys)

    # if (y_x < -1).any() or (y_x > 1).any() or (y_y < -1).any() or (y_y > 1).any():
    #    raise Exception('actual value was not within bounds...')

    i = mu_x * y_x + mu_y * y_y
    if torch.isnan(i).any():
        print(mu_x)
        print(mu_y)
        print(ys)
        raise Exception('1', i)

    """
    # i = torch.clamp(i, min=-1, max=1)
    if (i < -1).any() or (i > 1).any():
        print('mu_x', mu_x)
        print('y_x', y_x)
        print('mu_y', mu_y)
        print('y_y', y_y)
        
        raise Exception('out of bounds', i)
    """

    # acos = torch.acos(i)
    # if torch.isnan(acos).any():
    #     raise Exception('2', acos, i)
    # print('angles', acos)

    # i is between -1 and 1
    # -1 is angle of pi, while 1 is angle of 0 (optimal)
    # thus we want our answer to be as close to 1 as possible
    i *= -1
    # now we simply want our i to be as low as possible, which works per definition of minimizing loss
    i += 1
    # just as a nice preview we want to minimize towards 0 instead of -1

    # print('angles', i)

    res = torch.mean(i)
    #for i in range(acos.shape[0]):
        # -0.94677734375 -0.3212890625 0.9432709813117981 0.332023948431015 nan
        #print(float(mu_x[i]), float(mu_y[i]), float(y_x[i]), float(y_y[i]), float(acos[i]))
    return res

# Compute the angle in radians between (mu_x, mu_y) and (0, 1)
def compute_mu_angle(mu_x, mu_y):
    # atan2 takes y-coord as first parameter and x-coord as second parameter
    return torch.atan2(mu_y, mu_x)