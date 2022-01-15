"""
The data assimilation system (4D Var)
Load:
    x_a_init.txt
    y_o.txt
Save:
    x_b_4d.txt
    x_a_4d.txt
"""

include("setting.jl");
using DelimitedFiles;
using LinearAlgebra;
using OrdinaryDiffEq;
using Optim, LineSearches;

# load initial condition
x_a_init = readdlm("data/x_a_init.txt");

# load observations
y_o_save = readdlm("data/y_o.txt");
y_o_time = y_o_save[:, 1];
y_o_data = y_o_save[:, 2:end];

# initial x_a: from x_a_init
x_a_save = fill(NaN, nT + 1, N)
x_a_save[1, :] = x_a_init';

# initial x_b: no values at the initial time (assign NaN)
x_b_save = fill(NaN, nT + 1, N)

# initial background error variance
σ = 0.2
R = convert(Matrix, Diagonal(repeat([σ^2], N)));
B_init = R
B_init = readdlm("data/b_nmc.txt");

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

H = observation_operators(y_o_data)

# cost function
function J(
    x✶::Vector{Float64},
    x_b::Vector{Float64},
    B✶::Matrix{Float64},
    ts::Vector{Float64},
    y_o::Matrix{Float64},
    H::Vector{Matrix{Float64}},
    R::Vector{Matrix{Float64}})::Float64
    @assert length(H) == length(R)
    @assert length(R) == size(y_o, 1)
    n = length(R)
    xs, _ = traj(x✶, ts)
    db = x_b - x✶
    J1 = db' * inv(B✶) * db
    @assert all(!, isnan.(J1))
    J2 = 0.0
    for i = 1:n
        @inbounds dy = replace(y -> isnan(y) ? 0.0 : y, y_o[i, :]) - H[i] * xs[i]
        @inbounds J2 += dy' * inv(R[i]) * dy
        @assert all(!, isnan.(J2)) "j2 nan in loop $i"
    end
    0.5 * J1 + 0.5 * J2
end

# gradient of cost function, s[:] notation is mandatory
function J_grad!(
    s::Vector{Float64},
    x✶::Vector{Float64},
    x_b::Vector{Float64},
    B✶::Matrix{Float64},
    ts::Vector{Float64},
    y_o::Matrix{Float64},
    H::Vector{Matrix{Float64}},
    R::Vector{Matrix{Float64}})
    n = length(R)
    xs, M = traj(x✶, ts)
    Jg1 = inv(B✶) * (x✶ - x_b)
    @assert all(!, isnan.(Jg1))

    Jg2 = zeros(size(Jg1))
    for i = 1:n
        @inbounds dgy = replace(y -> isnan(y) ? 0.0 : y, y_o[i, :]) - H[i] * xs[i]
        @inbounds Jg2 += transpose(M[i]) * transpose(H[i]) * inv(R[i]) * dgy
        @assert all(!, isnan.(Jg2))
    end
    @inbounds s[:] = Jg1 - Jg2
end

# jacobian of lorenz96
function jac_lorenz(x::Vector{Float64})::Matrix{Float64}
    # jac[i, j] = -1 when j = i
    # jac[i, j] = x[i-1] when j = i+1
    # jac[i, j] = x[i+1] - x[i-2] when j = i-1
    # jac[i, j] = -x[i-1] when j = i-2
    len = length(x)
    lenm = len - 1
    jac = zeros(len, len)

    for i = 1:len
        ip1 = (i + 1 + lenm) % len + 1
        im1 = (i - 1 + lenm) % len + 1
        im2 = (i - 2 + lenm) % len + 1
        @inbounds jac[i, i] = -1
        @inbounds jac[i, ip1] = x[im1]
        @inbounds jac[i, im1] = x[ip1] - x[im2]
        @inbounds jac[i, im2] = -x[im1]
    end
    jac
end

function traj(x✶::Vector{Float64}, times::Vector{Float64})::Tuple{Vector{Vector},Vector{Matrix{Float64}}}
    # some missing time step
    # assumption of constant dU might fail
    nw = length(times) - 1
    nt = length(times)
    # this assertion should not exist
    @assert nw != 0
    ms = Vector{Matrix}(undef, nt)
    @inbounds ms[1] = Matrix(1.0I, N, N)
    xs = Vector{Vector}(undef, nt)
    @inbounds xs[1] = x✶
    dU = diff(times)
    fine_step = 100
    xt = x✶

    for i = 1:nw
        @inbounds ms[i+1] = ms[i]
        if dU[i] > 1e-12
            @inbounds ddU = dU[i] / fine_step
            for _ = 1:fine_step
                L = I + ddU * jac_lorenz(xt)
                xt += ddU * lorenz96(xt, Missing, Missing)
                @inbounds ms[i+1] = L * ms[i+1]
            end
        end
        @inbounds xs[i+1] = xt
    end

    xs[2:end], ms[2:end]
end

tbias = 1.0 * dT
ibias = convert(Integer, cld(tbias, dT))
ua = 1
ue = length(y_o_time)

for tt = 1:nT-ibias
    tts = tt - 1
    tti = tt + 1
    Ts = tts * dT
    Ta = tt * dT
    Wa = Ta + tbias
    # assimilation window from Ts to Ta
    global us = ua + 1
    for i = us:ue
        global ua = i
        if abs(y_o_time[i] - Wa) < dT / 100.0
            break
        elseif Wa < y_o_time[i]
            break
        end
    end
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
    x_b_time = Ta

    # observation
    y_o = y_o_data[us:ua, :]
    h_o = H[us:ua]
    r_o = [repeat(R, 1, 1, ua - us + 1)[:, :, i] for i in 1:(ua-us+1)]

    # analysis scheme (4dvar)
    traj_time = [x_b_time; y_o_time[us:ua]]

    opt = optimize(x -> J(x, x_b, B_init, traj_time, y_o, h_o, r_o),
        (s, x) -> J_grad!(s, x, x_b, B_init, traj_time, y_o, h_o, r_o),
        x_b,
        method = ConjugateGradient(),
        g_tol = 1e-12,
        iterations = 100
    )
    # @show opt
    @assert Optim.converged(opt)

    x_a = opt.minimizer


    x_a_save[tti, :] = x_a
end # end tt loop

open("data/x_a_4d.txt", "w") do xa
    writedlm(xa, x_a_save)
end

open("data/x_b_4d.txt", "w") do xb
    writedlm(xb, x_b_save)
end
