using Test
using StableRNGs
using Random
using Turing

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

    @testset "ess"  begin
        @model function MoGtest(D)
            mu1 ~ Normal(1, 1)
            mu2 ~ Normal(4, 1)
            z1 ~ Categorical(2)
            if z1 == 1
                D[1] ~ Normal(mu1, 1)
            else
                D[1] ~ Normal(mu2, 1)
            end
            z2 ~ Categorical(2)
            if z2 == 1
                D[2] ~ Normal(mu1, 1)
            else
                D[2] ~ Normal(mu2, 1)
            end
            z3 ~ Categorical(2)
            if z3 == 1
                D[3] ~ Normal(mu1, 1)
            else
                D[3] ~ Normal(mu2, 1)
            end
            z4 ~ Categorical(2)
            if z4 == 1
                D[4] ~ Normal(mu1, 1)
            else
                D[4] ~ Normal(mu2, 1)
            end
            return z1, z2, z3, z4, mu1, mu2
        end
        MoGtest_default = MoGtest([1.0 1.0 4.0 4.0])

        # alg = Gibbs(
        #     (@varname(z1), @varname(z2), @varname(z3), @varname(z4)) => CSMC(15),
        #     @varname(mu1) => ESS(),
        #     @varname(mu2) => ESS(),
        # )
        Random.seed!(468)
        alg = Gibbs(CSMC(15, :z1, :z2, :z3, :z4), ESS(:mu1), ESS(:mu2))
        chain = sample(StableRNG(468), MoGtest_default, alg, 50)
        @show mean(chain[:z1])
        @show mean(chain[:z2])
        @show mean(chain[:z3])
        @show mean(chain[:z4])
    end
end
