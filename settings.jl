"""
General settings
"""

# parameters
const N = 40                      # number of grid point
const F = 8.0                     # forcing term
const T_max = 10.0                # time length of the experiment
const dT = 0.05                   # forecast-analysis cycle length
const nT = convert(Int, T_max/dT) # number of cycles
