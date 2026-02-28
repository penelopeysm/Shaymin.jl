using Test
using Distributions

# Minimal inline of VectorBijectors product distribution code to reproduce segfault on Windows.

function wladj end

# -- common.jl --
abstract type ScalarToScalarBijector end
struct TypedIdentity <: ScalarToScalarBijector end
(::TypedIdentity)(x) = x
wladj(::TypedIdentity, x::T) where {T<:Number} = (x, zero(T))
wladj(::TypedIdentity, x::AbstractArray{T}) where {T<:Number} = (x, zero(T))

# -- univariate.jl --
struct VectWrap{B<:ScalarToScalarBijector}
    bijector::B
end
(w::VectWrap)(x) = [w.bijector(x)]
function wladj(w::VectWrap, x::Number)
    y, ladj = wladj(w.bijector, x)
    return ([y], ladj)
end

struct OnlyWrap{B<:ScalarToScalarBijector}
    bijector::B
end
(w::OnlyWrap)(x) = w.bijector(x[])
function wladj(w::OnlyWrap, x::AbstractVector)
    return wladj(w.bijector, x[])
end

to_vec(::UnivariateDistribution) = VectWrap(TypedIdentity())
from_vec(::UnivariateDistribution) = OnlyWrap(TypedIdentity())
to_linked_vec(::UnivariateDistribution) = VectWrap(TypedIdentity())
from_linked_vec(::UnivariateDistribution) = OnlyWrap(TypedIdentity())
vec_length(::UnivariateDistribution) = 1
linked_vec_length(::UnivariateDistribution) = 1

# -- product.jl --
_fzero(::Type{T}) where {T<:Number} = zero(T)
_fzero(@nospecialize(T)) = 0.0

struct ProductVecTransform{TTrf,Trng,D}
    transforms::TTrf
    ranges::Trng
    base_size::D
end

struct ProductVecInvTransform{TTrf,Trng,D}
    transforms::TTrf
    ranges::Trng
    base_size::D
end

@generated function wladj(
    t::ProductVecTransform{<:NTuple{P,Any},<:NTuple{P,Any},<:NTuple{N,Int}},
    x::AbstractArray{T},
) where {P,N,T}
    exprs = []
    push!(exprs, :(total_length = sum(length, t.ranges)))
    push!(exprs, :(logjac = _fzero(T)))
    push!(exprs, :(y = Vector{T}(undef, total_length)))
    colons = fill(:, N)
    y_syms = Symbol.(:y, 1:P)
    logjac_syms = Symbol.(:lj, 1:P)
    for (i, (y_sym, lj_sym)) in enumerate(zip(y_syms, logjac_syms))
        if N == 0
            push!(
                exprs,
                :(($y_sym, $lj_sym) = wladj(t.transforms[$i], x[$i])),
            )
        else
            push!(
                exprs,
                :(
                    ($y_sym, $lj_sym) = wladj(
                        t.transforms[$i], view(x, $colons..., $i)
                    )
                ),
            )
        end
        push!(exprs, :(y[t.ranges[$i]] .= $y_sym))
        push!(exprs, :(logjac += $lj_sym))
    end
    push!(exprs, :(return (y, logjac)))
    return Expr(:block, exprs...)
end

#! format: off
@generated function (t::ProductVecTransform{<:NTuple{P,Any},<:NTuple{P,Any},<:NTuple{N,Int}})(
    x::AbstractArray{T}
) where {P,N,T}
#! format: on
    exprs = []
    push!(exprs, :(total_length = sum(length, t.ranges)))
    push!(exprs, :(y = Vector{T}(undef, total_length)))
    colons = fill(:, N)
    for i in 1:P
        if N == 0
            push!(exprs, :(y[t.ranges[$i]] = t.transforms[$i](x[$i])))
        else
            push!(exprs, :(y[t.ranges[$i]] .= t.transforms[$i](view(x, $colons..., $i))))
        end
    end
    push!(exprs, :(return y))
    return Expr(:block, exprs...)
end

@generated function wladj(
    t::ProductVecInvTransform{<:NTuple{P,Any},<:NTuple{P,Any},<:NTuple{N,Int}},
    y::AbstractVector{T},
) where {P,N,T}
    exprs = []
    push!(exprs, :(x = Array{T}(undef, t.base_size..., P)))
    push!(exprs, :(logjac = _fzero(T)))
    colons = fill(:, N)
    x_syms = Symbol.(:x, 1:P)
    lj_syms = Symbol.(:lj, 1:P)
    for (i, (x_sym, lj_sym)) in enumerate(zip(x_syms, lj_syms))
        push!(
            exprs,
            :(
                ($x_sym, $lj_sym) = wladj(
                    t.transforms[$i], view(y, t.ranges[$i])
                )
            ),
        )
        push!(exprs, :(x[$colons..., $i] = $x_sym))
        push!(exprs, :(logjac += $lj_sym))
    end
    push!(exprs, :(return (x, logjac)))
    return Expr(:block, exprs...)
end

#! format: off
@generated function (t::ProductVecInvTransform{<:NTuple{P,Any},<:NTuple{P,Any},<:NTuple{N,Int}})(
    y::AbstractVector{T}
) where {P,N,T}
#! format: on
    exprs = []
    push!(exprs, :(x = Array{T}(undef, t.base_size..., P)))
    colons = fill(:, N)
    for i in 1:P
        push!(exprs, :(x[$colons..., $i] = t.transforms[$i](view(y, t.ranges[$i]))))
    end
    push!(exprs, :(return x))
    return Expr(:block, exprs...)
end

@generated function _make_transform(
    dists::NTuple{NDists,Distribution}, indiv_transform_fn, length_fn, struct_type
) where {NDists}
    exprs = []
    trfms = Expr(:tuple)
    for i in 1:NDists
        push!(trfms.args, :(indiv_transform_fn(dists[$i])))
    end
    push!(exprs, :(trfms = $trfms))
    push!(exprs, :(ranges = ()))
    push!(exprs, :(offset = 1))
    for i in 1:NDists
        push!(exprs, :(this_length = length_fn(dists[$i])))
        push!(exprs, :(ranges = (ranges..., offset:(offset + this_length - 1))))
        push!(exprs, :(offset += this_length))
    end
    push!(exprs, :(return struct_type(trfms, ranges, size(dists[1]))))
    return Expr(:block, exprs...)
end

function _to_vec(d::ProductDistribution)
    return _make_transform(d.dists, to_vec, vec_length, ProductVecTransform)
end
function _from_vec(d::ProductDistribution)
    return _make_transform(d.dists, from_vec, vec_length, ProductVecInvTransform)
end
function _to_linked_vec(d::ProductDistribution)
    return _make_transform(d.dists, to_linked_vec, linked_vec_length, ProductVecTransform)
end
function _from_linked_vec(d::ProductDistribution)
    return _make_transform(
        d.dists, from_linked_vec, linked_vec_length, ProductVecInvTransform
    )
end

# -- test --
d = product_distribution(Normal())

@testset "Product distribution segfault reproducer" begin
    x = rand(d)

    ffwd = _to_vec(d)
    frvs = _from_vec(d)
    @test frvs(ffwd(x)) ≈ x

    ffwd_l = _to_linked_vec(d)
    frvs_l = _from_linked_vec(d)
    @test frvs_l(ffwd_l(x)) ≈ x

    y = randn(1)
    x2 = frvs_l(y)
    @test ffwd_l(x2) ≈ y

    y, logjac = wladj(ffwd_l, x)
    @test y ≈ ffwd_l(x)
end
