import Enzyme as E

# Minimal reproducer for Enzyme + callable struct with UnitRange field segfault on Windows + Julia 1.12

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
    y[t.ranges[1]] = [x[1]]
    return y
end
function (t::ProductVecInvTransform)(y::AbstractVector{T}) where {T}
    x = Vector{T}(undef, 1)
    x[1] = view(y, t.ranges[1])[]
    return x
end

ffwd = ProductVecTransform((vec_wrap,), (1:1,))
frvs = ProductVecInvTransform((only_wrap,), (1:1,))

f = ffwd ∘ frvs
xvec = randn(1)
E.jacobian(E.Forward, f, xvec)

@info "Done"
