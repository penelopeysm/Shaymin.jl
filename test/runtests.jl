using Test
using StableRNGs
using Random
using Turing
using Distributions
using RandomNumbers
using Pkg
using AdvancedPS
using Random123

function AdvancedPS.gen_seed(rng::Random.AbstractRNG, ::AdvancedPS.TracedRNG{<:Integer}, sampler::Random.Sampler)
    x = Random.rand(rng, sampler)
    @show "1", rng, x
    return x
end
function AdvancedPS.seed_from_rng!(
    pc::AdvancedPS.ParticleContainer{T,<:AdvancedPS.TracedRNG{R,N,<:Random123.AbstractR123{I}}},
    rng::Random.AbstractRNG,
    ref::Union{AdvancedPS.Particle,Nothing}=nothing,
) where {T,R,N,I}
    n = length(pc.vals)
    nseeds = isnothing(ref) ? n : n - 1
    @show I

    sampler = Random.Sampler(rng, I)
    @show sampler
    for i in 1:nseeds
        subrng = pc.vals[i].rng
        Random.seed!(subrng, gen_seed(rng, subrng, sampler))
    end
    Random.seed!(pc.rng, gen_seed(rng, pc.rng, sampler))

    return pc
end

# Pkg.develop(path="/Users/pyong/ppl/aps")
# Pkg.add(path="https://github.com/TuringLang/AdvancedPS.jl.git", rev="dc902432")

Random.seed!(468)

# Define a new normal distribution for tracking
struct MyNorm <: ContinuousUnivariateDistribution end
function Distributions.rand(rng::Random.AbstractRNG, ::MyNorm)
    res = randn(rng)
    @info "MyNorm with rng: $rng gave result: $res"
    return res
end
Distributions.logpdf(::MyNorm, x) = logpdf(Normal(), x)

@testset verbose = true "Shaymin.jl" begin
    # @testset "using global seed" begin
    #     x1 = randn(3)
    #     @info x1
    # end
    #
    # @testset "stablerng" begin
    #     x2 = randn(StableRNG(468), 3)
    #     @info x2
    # end
    #
    # @testset "randomnumbers" begin
    #     Random.seed!(468)
    #     s = RandomNumbers.gen_seed(UInt64)
    #     @info s
    # end
    #
    # @testset "randomnumbers again" begin
    #     Random.seed!(468)
    #     s = RandomNumbers.gen_seed(UInt64)
    #     @info s
    # end
    #
    # @testset "randomdevice" begin
    #     Random.seed!(468)
    #     rng = RandomDevice()
    #     x3 = randn(rng, 3)
    #     @info x3
    # end
    #
    # @testset "randomdevice again" begin
    #     Random.seed!(468)
    #     rng = RandomDevice()
    #     x3 = randn(rng, 3)
    #     @info x3
    # end

    @testset "Sampler" begin
        @show Random.Sampler(StableRNG(468), UInt64)
    end
    
    # @testset "tracedrng no seed" begin
    #     rng = AdvancedPS.TracedRNG()
    #     @show rng
    # end
    #
    # @testset "tracedrng no seed 2" begin
    #     rng = AdvancedPS.TracedRNG()
    #     @show rng
    # end
    #
    # @testset "tracedrng seed" begin
    #     Random.seed!(468)
    #     rng = AdvancedPS.TracedRNG()
    #     @show rng
    # end
    #
    # @testset "tracedrng seed 2" begin
    #     Random.seed!(468)
    #     rng = AdvancedPS.TracedRNG()
    #     @show rng
    # end

    # reproducibly different
    @testset "pg"  begin
        @model function f()
            x ~ MyNorm()
        end
        Random.seed!(468)
        alg = PG(15)
        chain = sample(StableRNG(468), f(), alg, 10; progress=false)
        @show mean(chain[:x])
    end
end
