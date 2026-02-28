import DifferentiationInterface as DI
import Enzyme as E

# Minimal reproducer for Enzyme + @generated function segfault on Windows + Julia 1.12

abstract type ScalarToScalarBijector end
struct TypedIdentity <: ScalarToScalarBijector end
(::TypedIdentity)(x) = x

struct VectWrap{B<:ScalarToScalarBijector}
    bijector::B
end
(w::VectWrap)(x) = [w.bijector(x)]

struct OnlyWrap{B<:ScalarToScalarBijector}
    bijector::B
end
(w::OnlyWrap)(x) = w.bijector(x[])

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

@generated function (t::ProductVecTransform{<:NTuple{P,Any},<:NTuple{P,Any},<:NTuple{N,Int}})(
    x::AbstractArray{T}
) where {P,N,T}
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

# Manually construct what _make_transform(product_distribution(Normal())) would produce
ffwd = ProductVecTransform((VectWrap(TypedIdentity()),), (1:1,), ())
frvs = ProductVecInvTransform((OnlyWrap(TypedIdentity()),), (1:1,), ())

adtype = DI.AutoEnzyme(; mode=E.Forward, function_annotation=E.Const)

xvec = randn(1)
DI.jacobian(ffwd ∘ frvs, adtype, xvec)
