using AbstractMCMC
using Distributions
using DynamicPPL
using LinearAlgebra
using Random
using Test
using Turing

Random.seed!(100)

include("models.jl")

samplers = (
    "SampleFromUniform" => (SampleFromUniform(), 1000),
    "NUTS" => (NUTS(), 100)
)

@testset verbose = true "DynamicPPL.jl" begin
    @testset verbose = true "$(model.f)" for model in TEST_MODELS
        @testset "$(name) @ $(N) iters" for (name, (spl, N)) in samplers
            chain_init = sample(model, spl, N; progress=false)
        end
    end
end
