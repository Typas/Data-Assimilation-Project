#!/usr/bin/env python3
import numpy as np

x_t = np.genfromtxt('x_t.txt')

(x_t_time, x_t_data) = np.hsplit(x_t, [1])
x_t_time = np.ndarray.flatten(x_t_time)

print(np.shape(x_t_data))
print(x_t_time[-1])
print(x_t_time[20])
print(x_t_time)
