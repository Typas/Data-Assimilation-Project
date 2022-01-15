#!/usr/bin/env python3

import numpy as np
from settings import *
import matplotlib.pyplot as plt

y_o_save = np.genfromtxt('data/y_o_sync.txt')
x_t_save = np.genfromtxt('data/x_t_sync.txt')
d1 = y_o_save[:, 1] - x_t_save[:, 0]
d2 = y_o_save[:, 3] - x_t_save[:, 2]
plt.figure()
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)
ax1.scatter(d1, d2)
print(d1)
print(d2)
ax2.hist(d1)
ax3.hist(d2)
plt.show()
