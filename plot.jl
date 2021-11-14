"""
Plot the data assimilation results
Read:
  x_t.txt
  x_b.txt
  x_a.txt
"""

using Plots
include("settings.jl")
"""python
# load data
x_t_save = np.genfromtxt('x_t.txt')
#x_b_save = np.genfromtxt('x_b.txt')
x_a_save = np.genfromtxt('x_a.txt')

# Plot time series of a single grid point
pt = 3
plt.figure()
plt.plot(np.arange(nT+1) * dT, x_t_save[:,pt-1], 'k+--', label=r'$x^t_{' + str(pt) + '}$')
#plt.plot(np.arange(nT+1) * dT, x_b_save[:,pt-1], 'go-' , label=r'$x^b_{' + str(pt) + '}$')
plt.plot(np.arange(nT+1) * dT, x_a_save[:,pt-1], 'bo-' , label=r'$x^a_{' + str(pt) + '}$')
plt.xlabel(r'$t$', size=18)
plt.ylabel(r'$x$', size=18)
plt.title(r'Time series of $x_{' + str(pt) + '}$', size=20)
plt.legend(loc='upper right', numpoints=1, prop={'size':18})
plt.savefig('timeseries.png', dpi=200)
plt.show()
"""
