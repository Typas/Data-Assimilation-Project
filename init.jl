"""
Create initial condition for DA experiment
Save:
  x_a_init.txt
"""

using DifferentialEquations: ode
include("settings.jl")
include("lorenz96.jl")

# settings of spin-up
const σ_x0 = 0.1       # size of initial perturpation
const T_spinup = 100.0 # initial spin-up time

# spin-up from a random initial value
x_a_0 = σ_x0 * randn(N)

"""python
solver = ode(lorenz96.f).set_integrator('dopri5', nsteps=10000)
solver.set_initial_value(x_a_0, 0.).set_f_params(F)
solver.integrate(Tspinup)
x_a_init = np.array(solver.y, dtype='f8')

# save the initial condition for DA experiment
np.savetxt('x_a_init.txt', x_a_init)
"""
