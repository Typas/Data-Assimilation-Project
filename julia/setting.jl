const N = 40
const F = 8.0

const Tmax = 10.0
const dT = 0.05
const nT = convert(Int32, Tmax/dT)

function lorenz96(x::Vector{Float64}, _, _)
    (circshift(x, -1) - circshift(x, 2)) .* circshift(x, 1) - x .+ F
end
