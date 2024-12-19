using Test
using StableRNGs
using Random
using DynamicPPL
using Distributions
using AdvancedPS
using Random123

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

    @testset "random123" begin
        x3 = randn(Random123.Philox2x(1), 3)
        @info x3
    end

    # @testset "ess"  begin
    #     @model function f(y)
    #         a ~ Normal(0, 1)
    #         y ~ Normal(a, 1)
    #     end
    #     Random.seed!(468)
    #     alg = AdvancedPS.PG(15)
    #     chain = sample(StableRNG(468), f(1.5), alg, 50; progress=false)
    #     @show chain
    #     # @show mean(chain[:b])
    # end
end
