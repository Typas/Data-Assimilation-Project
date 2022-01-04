"""
Plot the data assimilation results
Read:
  x_t.txt
  x_b.txt
  x_a.txt
"""
import numpy as np
from numpy.core.fromnumeric import shape
from settings import *
import matplotlib.pyplot as plt

# load data
x_t_save = np.genfromtxt('x_t.txt')
# x_b_save = np.genfromtxt('x_b.txt')
x_a_save = np.genfromtxt('x_a_none.txt')
x_a_oi = np.genfromtxt('x_a_oi.txt')
x_a_3d = np.genfromtxt('x_a_3d.txt')
x_a_4d = np.genfromtxt('x_a_4d.txt')

def RMSE(x_a, x_t):
    return np.sqrt(np.mean((x_a-x_t)**2))

# # Plot time series of a single grid point
# pt = 3
# plt.figure()
# plt.plot(np.arange(nT+1) * dT, x_t_save[:,pt-1], 'k+--', label=r'$x^t_{' + str(pt) + '}$')
# plt.plot(np.arange(nT+1) * dT, x_a_save[:,pt-1], 'co-' , label=r'$NoDA x^a_{' + str(pt) + '}$')
# plt.plot(np.arange(nT+1) * dT, x_a_oi[:,pt-1], 'go-' , label=r'$OI x^a_{' + str(pt) + '}$')
# plt.plot(np.arange(nT+1) * dT, x_a_3d[:,pt-1], 'bo-' , label=r'$3DVar x^a_{' + str(pt) + '}$')
# # plt.plot(np.arange(nT+1) * dT, x_a_4d[:,pt-1], 'ro-' , label=r'$4DVar x^a_{' + str(pt) + '}$')
# plt.xlabel(r'$t$', size=18)
# plt.ylabel(r'$x$', size=18)
# plt.title(r'Time series of $x_{' + str(pt) + '}$', size=20)
# plt.legend(loc='upper right', numpoints=1, prop={'size':18})
# plt.savefig('timeseries.png', dpi=200)
# plt.show()
# plt.clf()

# Plot time series of RMSE
T = len(x_t_save)
rmse_no = []
rmse_oi = []
rmse_3d = []
rmse_4d = []
for i in range(T):
    rmse_no.append(RMSE(x_a_save[i], x_t_save[i]))
    rmse_oi.append(RMSE(x_a_oi[i], x_t_save[i]))
    rmse_3d.append(RMSE(x_a_3d[i], x_t_save[i]))
    rmse_4d.append(RMSE(x_a_4d[i], x_t_save[i]))
# plt.figure()
plt.plot(np.arange(nT+1) * dT, rmse_no, 'c', label='NoDA - RMSE')
plt.plot(np.arange(nT+1) * dT, rmse_oi, 'g', label='OI - RMSE')
plt.plot(np.arange(nT+1) * dT, rmse_3d, 'b' , label='3DVar - RMSE')
# plt.plot(np.arange(nT+1) * dT, rmse_4d, 'r' , label='4DVar - RMSE')
plt.xlabel(r'$t$', size=18)
# plt.ylabel(r'$x$', size=18)
plt.title('RMS errors in analyses')
plt.legend(loc='upper right', numpoints=1, prop={'size':18})
plt.savefig('RMSE.png', dpi=300)
plt.show()
