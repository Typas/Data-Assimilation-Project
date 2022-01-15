"""
The data assimilation system (no assimilation example)
Load:
    x_a_init.txt
Save:
    x_b_none.txt
    x_a_none.txt
"""

include("setting.jl");
using DelimitedFiles;
using OrdinaryDiffEq;

# load initial condition
x_a_init = readdlm("data/x_a_init.txt");

# initial x_b: no values at the initial time (assign NaN)
x_b_save = Matrix{Float64}(undef, nT + 1, N);
x_b_save[1, :] .= NaN;

# initial x_a: from x_a_init
x_a_save = similar(x_b_save);
x_a_save[1, :] = x_a_init';

for tt = 1:nT
    tts = tt - 1
    tti = tt + 1
    Ts = tts * dT
    Ta = tt * dT
    println("Cycle = ", tt, ", Ts = ", round(Ts; digits = 3), ", Ta = ", round(Ta; digits = 3))

    #--------------
    # forecast step
    #--------------
    prob = ODEProblem(lorenz96, x_a_save[tt, :], (Ts, Ta))
    sol = solve(prob, Tsit5())
    x_b_save[tti, :] = sol[end]

    #--------------
    # analysis step
    #--------------
    # background
    x_b = sol[end]

    # analysis scheme (no assimilation in this example)
    x_a = x_b

    x_a_save[tti, :] = x_a
end # end tt loop

open("data/x_a_none.txt", "w") do xa
    writedlm(xa, x_a_save)
end

open("data/x_b_none.txt", "w") do xb
    writedlm(xb, x_b_save)
end
