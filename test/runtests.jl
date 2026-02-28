import DifferentiationInterface as DI
import Enzyme as E

# Minimal reproducer for Enzyme + @generated function segfault on Windows + Julia 1.12

vec_wrap(x) = [x]
only_wrap(x) = x[]

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

# What the @generated functions produce for P=1, N=0
function (t::ProductVecTransform)(x::AbstractArray{T}) where {T}
    y = Vector{T}(undef, sum(length, t.ranges))
    y[t.ranges[1]] = t.transforms[1](x[1])
    return y
end

function (t::ProductVecInvTransform)(y::AbstractVector{T}) where {T}
    x = Array{T}(undef, t.base_size..., 1)
    x[1] = t.transforms[1](view(y, t.ranges[1]))
    return x
end

ffwd = ProductVecTransform((vec_wrap,), (1:1,), ())
frvs = ProductVecInvTransform((only_wrap,), (1:1,), ())

adtype = DI.AutoEnzyme(; mode=E.Forward, function_annotation=E.Const)

xvec = randn(1)
DI.jacobian(ffwd ∘ frvs, adtype, xvec)
