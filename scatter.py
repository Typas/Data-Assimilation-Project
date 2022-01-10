#!/usr/bin/env python3

import numpy as np
from settings import *
import matplotlib.pyplot as plt

y_o_save = np.genfromtxt('y_o.txt')
plt.figure()
plt.scatter(y_o_save[:, 0], y_o_save[:, 1])
plt.show()
