using Test
using StableRNGs
using Random

Random.seed!(468)

@testset verbose = true "Shaymin.jl" begin
    @testset "randn" begin
        x1 = Random.randn(10)
        @info x1
        @test length(x1) == 10
    end

    @testset "stablerng" begin
        x2 = rand(StableRNG(1), 10)
        @info x2
        @test length(x2) == 10
    end
end
