#############################
# Non-submodel for comparison
#############################

@model function demo_assume_observe_literal()
    # univariate `assume` and literal `observe`
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))

    return (; s=s, m=m, x=[1.5, 2.0], logp=getlogp(__varinfo__))
end

#############################
# submodel in assume
#############################

@model function _prior_dot_assume(::Type{TV}=Vector{Float64}) where {TV}
    s = TV(undef, 2)
    s .~ InverseGamma(2, 3)
    m = TV(undef, 2)
    m .~ Normal.(0, sqrt.(s))
    return s, m
end

@model function demo_assume_submodel_observe_index_literal_old()
    @submodel s, m = _prior_dot_assume()
    1.5 ~ Normal(m[1], sqrt(s[1]))
    2.0 ~ Normal(m[2], sqrt(s[2]))
    return (; s=s, m=m, x=[1.5, 2.0], logp=getlogp(__varinfo__))
end

# @model function demo_assume_submodel_observe_index_literal_new()
#     priors ~ to_submodel(_prior_dot_assume())
#     s, m = priors
#     1.5 ~ Normal(m[1], sqrt(s[1]))
#     2.0 ~ Normal(m[2], sqrt(s[2]))
#     return (; s=s, m=m, x=[1.5, 2.0], logp=getlogp(__varinfo__))
# end

#############################
# submodel in observe
#############################

@model function _likelihood_mltivariate_observe(s, m, x)
    return x ~ MvNormal(m, Diagonal(s))
end

@model function demo_dot_assume_observe_submodel_old(
    x=[1.5, 2.0], ::Type{TV}=Vector{Float64}
) where {TV}
    s = TV(undef, length(x))
    s .~ InverseGamma(2, 3)
    m = TV(undef, length(x))
    m .~ Normal.(0, sqrt.(s))
    # Submodel likelihood
    @submodel _likelihood_mltivariate_observe(s, m, x)
    return (; s=s, m=m, x=x, logp=getlogp(__varinfo__))
end

# @model function demo_dot_assume_observe_submodel_new(
#     x=[1.5, 2.0], ::Type{TV}=Vector{Float64}
# ) where {TV}
#     s = TV(undef, length(x))
#     s .~ InverseGamma(2, 3)
#     m = TV(undef, length(x))
#     m .~ Normal.(0, sqrt.(s))
#     # Submodel likelihood
#     _ignore ~ to_submodel(_likelihood_mltivariate_observe(s, m, x))
#     return (; s=s, m=m, x=x, logp=getlogp(__varinfo__))
# end

TEST_MODELS = [
    demo_assume_observe_literal(),
    demo_assume_submodel_observe_index_literal_old(),
    # demo_assume_submodel_observe_index_literal_new(),
    demo_dot_assume_observe_submodel_old(),
    # demo_dot_assume_observe_submodel_new(),
]
