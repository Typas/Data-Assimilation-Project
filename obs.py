#!/usr/bin/env python3
"""
Generate the observation from nature run
apply NaN to non-obs points
Save:
  y_o.txt
"""
import numpy as np
from scipy.integrate import ode
import lorenz96
from settings import *

sd = 0.1
err = np.random.randn(int(Tmax/dT)+1, N//2)

def obs_mask(e):
    ((i, j), x) = e
    if j % 2 == 0:
        return np.nan
    else:
        return x * (1 + err[i, j//2]*sd)

y_nature = np.genfromtxt('x_t.txt')
y_o = np.array(list(map(obs_mask, np.ndenumerate(y_nature))))

np.savetxt('y_o.txt', y_o)
