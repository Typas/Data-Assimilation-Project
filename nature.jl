"""
Generate the nature run
Save:
    x_t.txt
"""
using DifferentialEquations: ode
include("settings.jl")
include("lorenz96.jl")

# settings of spin-up
const σ_x0 = 0.1       # size of initial perturpation
const T_spinup = 100.0 # initial spin-up time

# spin-up from a random initial value
x_t_0 = σ_x0 * randn(N)

"""python
solver = ode(lorenz96.f).set_integrator('dopri5', nsteps=10000)
solver.set_initial_value(x_t_0, 0.).set_f_params(F)
solver.integrate(Tspinup)
x_t_save = np.array([solver.y], dtype='f8')

# create nature
solver = ode(lorenz96.f).set_integrator('dopri5')
solver.set_initial_value(x_t_save[0], 0.).set_f_params(F)

tt = 1
while solver.successful() and tt <= nT:
    solver.integrate(solver.t + dT)
    x_t_save = np.vstack([x_t_save, [solver.y]])
#    print('timestep =', tt, round(solver.t, 10))
    tt += 1

# save data
np.savetxt('x_t.txt', x_t_save)
"""
