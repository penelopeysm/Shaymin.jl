using Test
using Random
using Turing
using AdvancedPS
using Statistics: mean, var
using StableRNGs

# patch as per PR
function AdvancedPS.split(key::Integer, n::Integer = 1)
    T = typeof(key)
    inner_rng = Random.MersenneTwister(key)
    return rand(inner_rng, T, n)
end

@testset verbose = true "Shaymin.jl" begin
    @testset "pg reproducibility" begin
        @model f() = x ~ Normal()
        chain = sample(StableRNG(468), f(), PG(15), 500; progress = false)

        @show mean(chain[:x])
        @show var(chain[:x])
    end
end
