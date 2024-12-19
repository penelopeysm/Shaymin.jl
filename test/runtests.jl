using Test
using StableRNGs
using Random
using Turing

Random.seed!(468)

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
        @model function f(y)
            a ~ Normal(0, 1)
            y ~ Normal(a, 1)
        end

        Random.seed!(468)
        alg = Gibbs(CSMC(15, :a))
        chain = sample(StableRNG(468), f(1.5), alg, 50; progress=false)
        @show mean(chain[:a])
        # @show mean(chain[:b])
    end
end
