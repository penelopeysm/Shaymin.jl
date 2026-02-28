import DifferentiationInterface as DI
import Enzyme as E

# Minimal reproducer for Enzyme + @generated function segfault on Windows + Julia 1.12

vec_wrap(x) = [x]
only_wrap(x) = x[]

struct ProductVecTransform{TTrf,Trng}
    transforms::TTrf
    ranges::Trng
end

struct ProductVecInvTransform{TTrf,Trng}
    transforms::TTrf
    ranges::Trng
end

function (t::ProductVecTransform)(x::AbstractArray{T}) where {T}
    total_length = sum(length, t.ranges)
    y = Vector{T}(undef, total_length)
    y[t.ranges[1]] = t.transforms[1](x[1])
    return y
end
function (t::ProductVecInvTransform)(y::AbstractVector{T}) where {T}
    x = Vector{T}(undef, 1)
    x[1] = t.transforms[1](view(y, t.ranges[1]))
    return x
end

# @generated function (t::ProductVecTransform{<:NTuple{P,Any},<:NTuple{P,Any}})(
#     x::AbstractArray{T}
# ) where {P,T}
#     exprs = []
#     push!(exprs, :(total_length = sum(length, t.ranges)))
#     push!(exprs, :(y = Vector{T}(undef, total_length)))
#     for i in 1:P
#         push!(exprs, :(y[t.ranges[$i]] = t.transforms[$i](x[$i])))
#     end
#     push!(exprs, :(return y))
#     return Expr(:block, exprs...)
# end
#
# @generated function (t::ProductVecInvTransform{<:NTuple{P,Any},<:NTuple{P,Any}})(
#     y::AbstractVector{T}
# ) where {P,T}
#     exprs = []
#     push!(exprs, :(x = Vector{T}(undef, P)))
#     for i in 1:P
#         push!(exprs, :(x[$i] = t.transforms[$i](view(y, t.ranges[$i]))))
#     end
#     push!(exprs, :(return x))
#     return Expr(:block, exprs...)
# end

ffwd = ProductVecTransform((vec_wrap,), (1:1,))
frvs = ProductVecInvTransform((only_wrap,), (1:1,))

adtype = DI.AutoEnzyme(; mode=E.Forward, function_annotation=E.Const)

xvec = randn(1)
DI.jacobian(ffwd ∘ frvs, adtype, xvec)

@info "Done"
