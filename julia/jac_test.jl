include("setting.jl")

# jacobian of lorenz96
function jac_lorenz(x::Vector{Float64})::Matrix{Float64}
    # jac[i, j] = -1 when j = i
    # jac[i, j] = x[i-1] when j = i+1
    # jac[i, j] = x[i+1] - x[i-2] when j = i-1
    # jac[i, j] = -x[i-1] when j = i-2
    len = length(x)
    lenm = len - 1;
    jac = zeros(len, len)

    for i = 1:len
        ip1 = (i+1+lenm)%len+1
        im1 = (i-1+lenm)%len+1
        im2 = (i-2+lenm)%len+1
        @inbounds jac[i, i] = -1
        @inbounds jac[i, ip1] = x[im1]
        @inbounds jac[i, im1] = x[ip1] - x[im2]
        @inbounds jac[i, im2] = -x[im1]
    end
    jac
end

a = repeat([1., 3, 6, 4, 5], 3)
da = repeat([.2, .2, .2, .2, .2], 3)
b = a + da
fa = lorenz96(a, Missing, Missing)
fb = lorenz96(b, Missing, Missing)
jfa = jac_lorenz(a) * da
display(jac_lorenz(a))
println()
@show fa + jfa - fb
@assert all(abs.(fa+jfa-fb) .< 1e-6)
