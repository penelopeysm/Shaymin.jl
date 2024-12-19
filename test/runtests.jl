using Test
using StableRNGs
using Random
using Turing

Random.seed!(468)

@testset verbose = true "Shaymin.jl" begin
    @testset "randn" begin
        x1 = Random.randn(10)
        @info x1
        @test length(x1) == 10
    end

    @testset "stablerng" begin
        x2 = rand(StableRNG(468), 10)
        @info x2
        @test length(x2) == 10
    end

    @testset "ess"  begin
        @model function f(y)
            a ~ Normal(0, 1)
            b ~ Normal(a, 1)
            y ~ Normal(b, 1)
        end

        Random.seed!(468)
        alg = Gibbs(ESS(:a), ESS(:b))
        chain = sample(StableRNG(468), f(1.5), alg, 50; progress=false)
        @show mean(chain[:a])
        @show mean(chain[:b])
    end
end
