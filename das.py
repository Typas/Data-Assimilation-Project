"""
The data assimilation system (no assimilation example)
Load:
  x_a_init.txt
Save:
  x_b.txt
  x_a.txt
"""
import numpy as np
from scipy.integrate import ode
import lorenz96
from settings import *

# load initial condition
x_a_init = np.genfromtxt('x_a_init.txt')
#x_a_init = np.genfromtxt('x_t.txt')[0] + 1.e-4  # using nature run value plus a small error (for test purpose)

# load observations
# y_o_save = ......

# initial x_b: no values at the initial time (assign NaN)
x_b_save = np.full((1,N), np.nan, dtype='f8')

# initial x_a: from x_a_init
x_a_save = np.array([x_a_init])

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
    x_b_save = np.vstack([x_b_save, [solver.y]])

    #--------------
    # analysis step
    #--------------

    # background
    x_b = x_b_save[tt].transpose()

    # observation
    #y_o = ......

    # innovation
    #y_b = np.dot(H, x_b)
    #d = y_o - y_b

    # analysis scheme (no assimilation in this example)
    x_a = x_b

    x_a_save = np.vstack([x_a_save, x_a.transpose()])
    tt += 1

# save background and analysis data
np.savetxt('x_b.txt', x_b_save)
np.savetxt('x_a.txt', x_a_save)
