#!/usr/bin/env python3

import numpy as np
from settings import *
import matplotlib.pyplot as plt

x_b_std = np.genfromtxt('data/x_b_none.txt')
x_b_test = np.genfromtxt('data/x_b_none_test.txt')
delta = abs(x_b_std[1:, :] - x_b_test[1:, :])
plt.figure()
plt.plot(delta)
plt.show()
