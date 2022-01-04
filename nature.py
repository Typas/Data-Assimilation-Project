"""
Generate the nature run
Save:
  x_t.txt
"""
import numpy as np
from scipy.integrate import ode
import lorenz96
from settings import *

# settings of spin-up
sigma_x0 = 0.1  # size of initial perturpation
Tspinup = 100.  # initial spin-up time

# spin-up from a random initail value
x_t_0 = sigma_x0 * np.random.randn(N)

solver = ode(lorenz96.f).set_integrator('dopri5', nsteps=10000)
solver.set_initial_value(x_t_0, 0.).set_f_params(F)
solver.integrate(Tspinup)
x_t_save = np.array([solver.y], dtype='f8')
x_t_time = np.array([0.0], dtype='f8')

# setting of finer nature
dT = dT/10
nT = nT*10

# create nature
solver = ode(lorenz96.f).set_integrator('dopri5')
solver.set_initial_value(x_t_save[0], 0.).set_f_params(F)

tt = 1
while solver.successful() and tt <= nT:
    solver.integrate(solver.t + dT)
    step_y = np.array([solver.y], dtype='f8')
    step_t = np.array([solver.t], dtype='f8')
    x_t_save = np.vstack((x_t_save, step_y))
    x_t_time = np.vstack((x_t_time, step_t))
    # print('timestep =', tt, round(solver.t, 10))
    tt += 1

x_t_save = np.hstack((x_t_time, x_t_save))
# save data
np.savetxt('x_t.txt', x_t_save)
