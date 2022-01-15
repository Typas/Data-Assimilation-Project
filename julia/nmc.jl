include("setting.jl");
using DelimitedFiles;
@time using OrdinaryDiffEq;

x_a_oi = readdlm("data/x_a_oi.txt");
cycles = size(x_a_oi, 1);

ti_start = 4+1;
t1 = 4.0*dT
t2 = 8.0*dT;
α = 2.5e-1 # rescaling factor

B = ones(Float64, N, N)

for d = 0:20
    coeffs = Vector();

    for i in ti_start:cycles
        # integrate time period t1, end in t
        prob = ODEProblem(lorenz96, x_a_oi[i, :], (0.0, t1));
        sol = solve(prob, Tsit5());
        xf_t1 = sol[end];

        # integrate time period t2, end in t
        prob = ODEProblem(lorenz96, x_a_oi[i-4, :], (0.0, t2));
        sol = solve(prob, Tsit5());
        xf_t2 = sol[end];

        δf = xf_t2 - xf_t1;
        pb = δf * δf';
        for j in 1:N
            pos = (j+d+40-1)%40+1;
            neg = (j-d+40-1)%40+1;
            push!(coeffs, pb[j, pos]);
            push!(coeffs, pb[j, neg]);
        end
    end

    filter!(x->!isnan(x), coeffs);
    E = sum(coeffs) / length(coeffs);
    for i in 1:N
        pos = (i+d+40-1)%40+1;
        neg = (i-d+40-1)%40+1;
        global B[i, pos] = α * E
        global B[i, neg] = α * E
    end
end

open("data/b_nmc.txt", "w") do nmc
    writedlm(nmc, B)
end
