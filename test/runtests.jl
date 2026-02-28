import DifferentiationInterface as DI
import Enzyme as E

# Minimal reproducer for Enzyme + @generated function segfault on Windows + Julia 1.12

vec_wrap(x) = [x]
only_wrap(x) = x[]

struct ProductVecTransform{TTrf}
    transforms::TTrf
end

struct ProductVecInvTransform{TTrf}
    transforms::TTrf
end

@generated function (t::ProductVecTransform{<:NTuple{P,Any}})(
    x::AbstractArray{T}
) where {P,T}
    exprs = []
    push!(exprs, :(y = Vector{T}(undef, P)))
    for i in 1:P
        push!(exprs, :(y[1:1] = t.transforms[$i](x[$i])))
    end
    push!(exprs, :(return y))
    return Expr(:block, exprs...)
end

@generated function (t::ProductVecInvTransform{<:NTuple{P,Any}})(
    y::AbstractVector{T}
) where {P,T}
    exprs = []
    push!(exprs, :(x = Vector{T}(undef, P)))
    for i in 1:P
        push!(exprs, :(x[$i] = t.transforms[$i](view(y, 1:1))))
    end
    push!(exprs, :(return x))
    return Expr(:block, exprs...)
end

ffwd = ProductVecTransform((vec_wrap,))
frvs = ProductVecInvTransform((only_wrap,))

adtype = DI.AutoEnzyme(; mode=E.Forward, function_annotation=E.Const)

xvec = randn(1)
DI.jacobian(ffwd ∘ frvs, adtype, xvec)

@info "Done"
