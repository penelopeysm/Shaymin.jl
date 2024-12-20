using Test
using StableRNGs
using Random
using Turing
using Distributions

Random.seed!(468)

# Define a new normal distribution for tracking
struct MyNorm <: ContinuousUnivariateDistribution end
function Distributions.rand(rng::AbstractRNG, ::MyNorm)
    res = randn(rng)
    @info "MyNorm with rng: $rng gave result: $res"
    return res
end
Distributions.logpdf(::MyNorm, x) = logpdf(Normal(), x)

@testset verbose = true "Shaymin.jl" begin
    @testset "using global seed" begin
        x1 = randn(3)
        @info x1
    end

    @testset "stablerng" begin
        x2 = randn(StableRNG(468), 3)
        @info x2
    end

    @testset "ess"  begin
        @model function f()
            x ~ MyNorm()
        end
        Random.seed!(468)
        alg = PG(15)
        chain = sample(StableRNG(468), f(), alg, 10; progress=false)
        @show mean(chain[:x])
    end
end
