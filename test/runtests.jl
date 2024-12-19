using Test
using StableRNGs

@testset verbose = true "Shaymin.jl" begin
    x = rand(StableRNG(1), 10)
    @info x
    @test length(x) == 10
end
