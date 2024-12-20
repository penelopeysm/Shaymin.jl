using Test
using StableRNGs
using Random
using AdvancedPS
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
        alg = PG(15)
        chain = sample(StableRNG(468), f(1.5), alg, 50; progress=false)
        @show mean(chain[:a])
    end

    @testset "advancedps" begin
        Random.seed!(468)
        x3 = randn(AdvancedPS.TracedRNG(), 3)
        @info x3
    end
end
