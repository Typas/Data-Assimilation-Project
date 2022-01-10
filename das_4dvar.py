#!/usr/bin/env python3

"""
The data assimilation system
Load:
  x_a_init.txt
  y_o.txt
Save:
  x_b_4d.txt
  x_a_4d.txt
"""
import numpy as np
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
y_o_save = y_o_data

# initial x_b: no values at the initial time (assign NaN)
x_b_save = np.full((1,N), np.nan, dtype='f8')

# initial x_a: from x_a_init
x_a_save = np.array([x_a_init])

# observation operator
H = identity(N, dtype='f8')
for i in range(N):
    if i % 2 == 1:
        H[i, i] = 0.0
        pass
    pass

# initial background error covariance
mu = np.mean(y_o_save[0], 0)
e = np.reshape(y_o_save[0] - mu, (N, 1))
co = np.cov(e@e.T)
R = np.diag(np.diagonal(co))
B_init = R

tt = 1
to = 0
toe = len(y_o_time)
while tt <= nT:
    dU = dT / 10.0
    tts = tt - 1
    tos = to       # observation start time index
    Ts = tts * dT  # forecast start time
    Ta = tt  * dT  # forecast end time (DA analysis time)
    # assimilation window from Ts to Ta
    for i in range(tos, toe):
        to = i
        if abs(y_o_time[i] - Ta) < 1e-5:
            break
        pass
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
    y_o = np.nan_to_num(y_o_save[to].transpose())

    # innovation

    # analysis scheme (no assimilation in this example)
    # observation covariance operator
    # assume analysis as truth
    R = []
    for j in range(tos, to):
        y_o = np.nan_to_num(y_o_save[j].transpose())
        mu = np.mean(y_o, 0)
        e = np.reshape(y_o - mu, (N, 1))
        co = np.cov(e@e.T)
        Rj = np.diag(np.diagonal(co))
        if np.linalg.det(Rj) == 0.0:
            raise ValueError('det = 0 when j = {}'.format(j))
        R.append(Rj)
        pass

    # TLM
    def m(t, M, Fx):
        return Fx.dot(M)

    def Jac(x):
        jac = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    jac[i][j] = -1
                elif j == (i+1)%N:
                    jac[i][j] = x[(i+N-1)%N]
                elif j == (i+N-1)%N:
                    jac[i][j] = x[(i+1)%N] - x[(i+N-2)%N]
                elif j == (i+N-2)%N:
                    jac[i][j] = -x[(i+N-1)%N]
                pass
            pass
        return jac

    M = []
    x0 = x_a_save[tts]
    Us = Ts
    for j in range(tos, to):
        solver = ode(lorenz96.f).set_integrator('dopri5')
        solver.set_initial_value(x0, Us+dU*(j-tos)).set_f_params(F)
        solver.integrate(Us+dU*(j+1-tos))
        x = solver.y
        I = identity(N)
        solver = ode(m).set_integrator('dopri5')
        solver.set_initial_value(I, Us).set_f_params(Jac(x))
        solver.integrate(Us+dU*(j+1-tos))
        Mj = solver.y
        M.append(Mj)
        x0 = x
        pass

    # cost function
    def J(x_init):
        dx = x_b-x_init
        J1 = 0.5*dx.T.dot(inv(B_init)).dot(dx).item()
        J2 = 0.0
        for i in range(tos, to):
            dy = y_o_save[i] - H.dot(M[i-tos].dot(x_init))
            J2 += 0.5 * dy.dot(inv(R[i-tos])).dot(dy.T).item()
            pass
        return J1 + J2

    J_min = minimize(J, x_b, method='CG', tol=1e-6)
    x_a = J_min.x
    x_a_save = np.vstack([x_a_save, x_a])
    tt += 1
    pass

# save background and analysis data
np.savetxt('x_b_4d.txt', x_b_save)
np.savetxt('x_a_4d.txt', x_a_save)
