import DifferentiationInterface as DI
import Enzyme as E

# Minimal reproducer for Enzyme + @generated function segfault on Windows + Julia 1.12

vec_wrap(x) = [x]
only_wrap(x) = x[]

struct Fwd{F,Trng}
    f::F
    ranges::Trng
end
function (t::Fwd)(x::AbstractVector{T}) where {T}
    y = Vector{T}(undef, 1)
    y[t.ranges[1]] = t.f(x[1])
    return y
end

struct Rvs{F,Trng}
    f::F
    ranges::Trng
end
function (t::Rvs)(y::AbstractVector{T}) where {T}
    x = Vector{T}(undef, 1)
    x[1] = t.f(view(y, t.ranges[1]))
    return x
end

ffwd = Fwd(vec_wrap, (1:1,))
frvs = Rvs(only_wrap, (1:1,))

adtype = DI.AutoEnzyme(; mode=E.Forward, function_annotation=E.Const)

xvec = randn(1)
DI.jacobian(ffwd ∘ frvs, adtype, xvec)

@info "Done"
