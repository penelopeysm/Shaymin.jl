import DifferentiationInterface as DI
import Enzyme as E

# Minimal reproducer for Enzyme + @generated function segfault on Windows + Julia 1.12

vec_wrap(x) = [x]
only_wrap(x) = x[]

struct Fwd{F}
    f::F
end
function (t::Fwd)(x::AbstractVector{T}) where {T}
    y = Vector{T}(undef, 1)
    y[1] = t.f(x[1])
    return y
end

struct Rvs{F}
    f::F
end
function (t::Rvs)(y::AbstractVector{T}) where {T}
    x = Vector{T}(undef, 1)
    x[1] = t.f(view(y, 1:1))
    return x
end

ffwd = Fwd(vec_wrap)
frvs = Rvs(only_wrap)

adtype = DI.AutoEnzyme(; mode=E.Forward, function_annotation=E.Const)

xvec = randn(1)
DI.jacobian(ffwd ∘ frvs, adtype, xvec)
