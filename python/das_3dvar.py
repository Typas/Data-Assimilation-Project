#!/usr/bin/env python3

"""
The data assimilation system
Load:
  x_a_init.txt
  y_o.txt
Save:
  x_b_3d.txt
  x_a_3d.txt
"""
import numpy as np
from numpy.core.fromnumeric import shape
from scipy.integrate import ode
from scipy.optimize import minimize
from numpy.linalg import inv
from numpy.matlib import identity
import lorenz96
from settings import *

# load initial condition
x_a_init = np.genfromtxt('x_a_init.txt')
#x_a_init = np.genfromtxt('x_t.txt')[0] + 1.e-4  # using nature run value plus a small error (for test purpose)

# load observations
y_o_save = np.genfromtxt('y_o.txt')

(y_o_time, y_o_data) = np.hsplit(y_o_save, [1])
y_o_time = np.ndarray.flatten(y_o_time)
y_o_save = np.zeros((nT+1, N))
y_o_length = nT+1
for j in range(N):
    k = 0
    for i in range(y_o_length):
        if not np.isnan(y_o_data[i, j]):
            y_o_save[k, j] = y_o_data[i, j]
            k += 1

# initial x_b: no values at the initial time (assign NaN)
x_b_save = np.full((1,N), np.nan, dtype='f8')

# initial x_a: from x_a_init
x_a_save = np.array([x_a_init])

# observation operator
H = identity(N, dtype='f8')
for i in range(N):
    if i % 2 == 1:
        H[i, i] = 0.0

# initial background error covariance
mu = np.mean(y_o_save[0], 0)
e = np.reshape(y_o_save[0] - mu, (N, 1))
co = np.cov(e@e.T)
R = np.diag(np.diagonal(co))
B = R

tt = 1
while tt <= nT:
    tts = tt - 1
    Ts = tts * dT  # forecast start time
    Ta = tt  * dT  # forecast end time (DA analysis time)
    print('Cycle =', tt, ', Ts =', round(Ts, 10), ', Ta =', round(Ta, 10))

    #--------------
    # forecast step
    #--------------

    solver = ode(lorenz96.f).set_integrator('dopri5')
    solver.set_initial_value(x_a_save[tts], Ts).set_f_params(F)
    solver.integrate(Ta)
    x_b_save = np.vstack((x_b_save, np.reshape([solver.y], (N,))))

    #--------------
    # analysis step
    #--------------

    # background
    x_b = x_b_save[tt].transpose()

    # observation
    y_o = y_o_save[tt].transpose()

    # innovation
    # observation covariance operator
    # assume analysis as truth
    mu = np.mean(y_o_save[tts], 0)
    e = np.reshape(y_o_save[tts] - mu, (N, 1))
    co = np.cov(e@e.T)
    R = np.diag(np.diagonal(co))
    # cost function
    def J(x):
        J1x = (x_b-x)
        J1 = J1x.T.dot(inv(B)).dot(J1x)
        J2y = y_o-H.dot(x)
        J2 = J2y.dot(inv(R)).dot(J2y.T)
        return (0.5*J1+0.5*J2).item()

    # analysis scheme
    J_min = minimize(J, x_b, method='CG')
    x_a = J_min.x

    x_a_save = np.vstack((x_a_save, x_a))

    # # NMC method
    # alpha = 0.25
    # t1 = 0.2
    # t2 = 0.4

    # solver = ode(lorenz96.f).set_integrator('dopri5')
    # solver.set_initial_value(x_a_save[tts], Ta).set_f_params(F)
    # solver.integrate(Ta+t1)
    # x_b_1 = np.reshape(np.array(solver.y), (N, 1))

    # solver = ode(lorenz96.f).set_integrator('dopri5')
    # solver.set_initial_value(x_a_save[tts], Ta).set_f_params(F)
    # solver.integrate(Ta+t2)
    # x_b_2 = np.reshape(np.array(solver.y), (N, 1))

    # dxb = x_b_2 - x_b_1
    # B = alpha * dxb.dot(dxb.T)

    tt += 1

# save background and analysis data
np.savetxt('x_b_3d.txt', x_b_save)
np.savetxt('x_a_3d.txt', x_a_save)