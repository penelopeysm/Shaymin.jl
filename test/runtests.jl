using Test
using StableRNGs
using DynamicPPL
using AbstractMCMC
using Random
using Turing
using Distributions
using RandomNumbers
using Pkg
using AdvancedPS
using Random123

function AdvancedPS.update_keys!(pc::AdvancedPS.ParticleContainer, ref::Union{AdvancedPS.Particle,Nothing}=nothing)
    # Update keys to new particle ids
    println("update_keys! begin")
    @show [v.rng for v in pc.vals]
    nparticles = length(pc)
    n = ref === nothing ? nparticles : nparticles - 1
    for i in 1:n
        pi = pc.vals[i]
        k = my_split(AdvancedPS.state(pi.rng.rng))
        @show k
        Random.seed!(pi.rng, k[1])
    end
    println("update_keys! end")
    @show [v.rng for v in pc.vals]
    return nothing
end

function my_split(key::Integer, n::Integer=1)
    T = typeof(key) # Make sure the type of `key` is consistent on W32 and W64 systems.
    @show T
    @show typeof(UInt(1))
    retval = T[hash(key, i) for i in UInt(1):UInt(n)]
    @show retval
    return retval
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

    # @testset "Sampler" begin
    #     @show Random.Sampler(StableRNG(468), UInt64)
    # end
    
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
    
    @testset "hash" begin
        T = UInt64
        @show hash(T(468), hash(T(1)))
        @show hash(T(468), hash(T(2)))
        @show hash(T(468), hash(T(3)))
    end

    # @testset "pg"  begin
    #     @model function f()
    #         x ~ MyNorm()
    #     end
    #     Random.seed!(468)
    #     alg = PG(15)
    #     chain = sample(StableRNG(468), f(), alg, 10; progress=false)
    #     @show mean(chain[:x])
    # end
end
