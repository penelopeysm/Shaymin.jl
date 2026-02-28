import Enzyme as E

# Minimal reproducer for Enzyme + callable struct with UnitRange field segfault on Windows + Julia 1.12

vec_wrap(x) = [x]
only_wrap(x) = x[]

struct ProductVecTransform{T1,T2,Trng}
    fwd_transforms::T1
    rvs_transforms::T2
    ranges::Trng
end

function (t::ProductVecTransform)(x::AbstractArray{T}) where {T}
    total_length = sum(length, t.ranges)
    y = Vector{T}(undef, total_length)
    y[t.ranges[1]] = t.fwd_transforms[1](x[1])
    x = Vector{T}(undef, 1)
    x[1] = t.rvs_transforms[1](view(y, t.ranges[1]))
    return x
end

f = ProductVecTransform((vec_wrap,), (only_wrap,), (1:1,))
xvec = randn(1)
E.jacobian(E.Forward, f, xvec)

@info "Done"
