module Shaymin

using Distributions

function normal(x::Float64, μ::Float64, σ::Float64)
    return rand(Normal(μ, σ))
end

end # module Shaymin
