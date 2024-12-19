using Test
using StableRNGs
using Random
using AdvancedPS

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

    @testset "advancedps" begin
        x3 = randn(AdvancedPS.TracedRNG(), 3)
        @info x3
    end
end
