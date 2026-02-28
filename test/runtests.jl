using Enzyme: Forward, Reverse, jacobian

struct ProductVecTransform{Trng}
    ranges::Trng
end

struct ProductVecInvTransform{Trng}
    ranges::Trng
end

function (t::ProductVecTransform)(x::AbstractArray{T}) where {T}
    y = Vector{T}(undef, 1)
    y[t.ranges[1]] = [x[1]]
    return y
end
function (t::ProductVecInvTransform)(y::AbstractVector{T}) where {T}
    x = Vector{T}(undef, 1)
    x[1] = view(y, t.ranges[1])[]
    return x
end

ffwd = ProductVecTransform((1:1,))
frvs = ProductVecInvTransform((1:1,))

f = ffwd ∘ frvs
xvec = randn(1)
jacobian(Forward, f, xvec)

@info "Done"
