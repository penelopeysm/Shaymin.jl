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
        Random.seed!(subrng, AdvancedPS.gen_seed(rng, subrng, sampler))
    end
    Random.seed!(pc.rng, AdvancedPS.gen_seed(rng, pc.rng, sampler))
    println("1")
    @show [v.rng for v in pc.vals]
    return pc
end

function DynamicPPL.initialstep(
    rng::Random.AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::DynamicPPL.Sampler{<:PG},
    vi::DynamicPPL.AbstractVarInfo;
    kwargs...,
)
    # Reset the VarInfo before new sweep
    DynamicPPL.reset_num_produce!(vi)
    DynamicPPL.set_retained_vns_del_by_spl!(vi, spl)
    DynamicPPL.resetlogp!!(vi)

    # Create a new set of particles
    num_particles = spl.alg.nparticles
    particles = AdvancedPS.ParticleContainer(
        [AdvancedPS.Trace(model, spl, vi, AdvancedPS.TracedRNG()) for _ in 1:num_particles],
        AdvancedPS.TracedRNG(),
        rng,
    )
    println("2")
    @show [v.rng for v in particles.vals]

    # Perform a particle sweep.
    logevidence = AdvancedPS.sweep!(rng, particles, spl.alg.resampler, spl)
    @show logevidence
    println("3")
    @show [v.rng for v in particles.vals]

    # Pick a particle to be retained.
    Ws = AdvancedPS.getweights(particles)
    indx = AdvancedPS.randcat(rng, Ws)
    @show indx
    reference = particles.vals[indx]

    # Compute the first transition.
    _vi = reference.model.f.varinfo
    transition = Turing.Inference.PGTransition(model, _vi, logevidence)
    state = Turing.Inference.PGState(_vi, reference.rng)

    @show transition
    @show state

    return transition, state
end

function AdvancedPS.resample_propagate!(
    ::Random.AbstractRNG,
    pc::AdvancedPS.ParticleContainer,
    sampler::T,
    randcat=AdvancedPS.DEFAULT_RESAMPLER,
    ref::Union{AdvancedPS.Particle,Nothing}=nothing;
    weights=AdvancedPS.getweights(pc),
) where {T<:AbstractMCMC.AbstractSampler}
    # sample ancestor indices
    n = length(pc)
    nresamples = ref === nothing ? n : n - 1
    indx = randcat(pc.rng, weights, nresamples)
    println("resample_propagate!")
    @show indx

    # count number of children for each particle
    num_children = zeros(Int, n)
    @inbounds for i in indx
        num_children[i] += 1
    end

    # fork particles
    particles = collect(pc)
    children = similar(particles)
    j = 0
    @inbounds for i in 1:n
        ni = num_children[i]
        if ni > 0
            # fork first child
            pi = particles[i]
            isref = pi === ref
            p = isref ? AdvancedPS.fork(pi, isref) : pi

            key = isref ? AdvancedPS.safe_get_refseed(ref.rng) : AdvancedPS.state(p.rng.rng) # Pick up the alternative rng stream if using the reference particle
            nsplits = isref ? ni + 1 : ni # We need one more seed to refresh the alternative rng stream
            seeds = split(key, nsplits)
            isref && AdvancedPS.safe_set_refseed!(ref.rng, seeds[end]) # Refresh the alternative rng stream

            Random.seed!(p.rng, seeds[1])

            children[j += 1] = p
            # fork additional children
            for k in 2:ni
                part = AdvancedPS.fork(p, isref)
                Random.seed!(part.rng, seeds[k])
                children[j += 1] = part
            end
        end
    end

    if ref !== nothing
        # Insert the retained particle. This is based on the replaying trick for efficiency
        # reasons. If we implement PG using task copying, we need to store Nx * T particles!
        AdvancedPS.update_ref!(ref, pc, sampler)
        @inbounds children[n] = ref
    end

    # replace particles and log weights in the container with new particles and weights
    pc.vals = children
    AdvancedPS.reset_logweights!(pc)

    return pc
end

function AdvancedPS.resample_propagate!(
    rng::Random.AbstractRNG,
    pc::AdvancedPS.ParticleContainer,
    sampler::T,
    resampler::AdvancedPS.ResampleWithESSThreshold,
    ref::Union{AdvancedPS.Particle,Nothing}=nothing;
    weights=AdvancedPS.getweights(pc),
) where {T<:AbstractMCMC.AbstractSampler}
    # Compute the effective sample size ``1 / ∑ wᵢ²`` with normalized weights ``wᵢ``
    ess = inv(sum(abs2, weights))

    if ess ≤ resampler.threshold * length(pc)
        AdvancedPS.resample_propagate!(rng, pc, sampler, resampler.resampler, ref; weights=weights)
    else
        println("update_keys!")
        AdvancedPS.update_keys!(pc, ref)
    end

    return pc
end

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
        @show hash(468, 1)
        @show hash(468, 2)
        @show hash(468, 3)
    end

    # reproducibly different
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
