"""
The data assimilation system (Optimal Interpolation)
Load:
    x_a_init.txt
    y_o.txt
Save:
    x_b_oi.txt
    x_a_oi.txt
"""

include("setting.jl");
using DelimitedFiles;
using LinearAlgebra;
using OrdinaryDiffEq;

# load initial condition
x_a_init = readdlm("data/x_a_init.txt");

# load observations
y_o_save = readdlm("data/y_o.txt");
y_o_time = y_o_save[:, 1];
y_o_data = y_o_save[:, 2:end];

# merge into sync
y_o_save = fill(NaN, (nT + 1, N))
for j = 1:N
    ke = 0
    for i = 1:nT+1
        # find nearest time
        ks = ke + 1
        Ta = (i - 1) * dT
        Ts = Ta - dT / 2
        Te = Ta + dT / 2
        ke = ks
        for k = ks:length(y_o_time)
            if y_o_time[k] > Te
                break
            end
            ke = k
        end
        min_dt = dT
        for k = ks:ke
            dt = abs(y_o_time[k] - Ta)
            if !isnan(y_o_data[k, j]) && dt < min_dt
                y_o_save[i, j] = y_o_data[k, j]
                min_dt = dt
            end
        end
    end
end

# initial x_b: no values at the initial time (assign NaN)
x_b_save = Matrix{Float64}(undef, nT + 1, N);
x_b_save[1, :] .= NaN;

# initial x_a: from x_a_init
x_a_save = similar(x_b_save);
x_a_save[1, :] = x_a_init';

# initial background error variance
σ = 0.2
R = Diagonal(repeat([σ^2], N))
B = R
B = readdlm("data/b_nmc.txt")

# observation operator
function observation_operators(y_o::Matrix{Float64})::Vector{Matrix{Float64}}
    len = size(y_o, 1)
    hs = Vector{Matrix{Float64}}()
    for i = 1:len
        row = map(x -> isnan(x) ? 0.0 : 1.0, y_o[i, :])
        push!(hs, Diagonal(row))
    end
    hs
end
H = observation_operators(y_o_save)

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

    # observation
    y_o = replace(y -> isnan(y) ? 0.0 : y, y_o_save[tti, :])

    # innovation
    d = y_o - H[tti] * x_b

    # analysis scheme
    K = B * H[tti]' / (H[tti] * B * H[tti]' + R)
    x_a = x_b + K * d

    x_a_save[tti, :] = x_a
end # end tt loop

open("data/x_a_oi.txt", "w") do xa
    writedlm(xa, x_a_save)
end

open("data/x_b_oi.txt", "w") do xb
    writedlm(xb, x_b_save)
end
