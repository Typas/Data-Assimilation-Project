#!/usr/bin/env python3
import numpy as np
B = np.reshape(np.genfromtxt("data/b_nmc.txt"), (40, 40))
import matplotlib.pyplot as plt
plt.contourf(B)
plt.colorbar()
plt.show()
