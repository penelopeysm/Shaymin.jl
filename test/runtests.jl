using Enzyme: Forward, Reverse, jacobian

struct ProductVecTransform{Trng}
    range::Trng
end

struct ProductVecInvTransform{Trng}
    range::Trng
end

function (t::ProductVecTransform)(x::AbstractArray{T}) where {T}
    total_length = length(t.range)
    y = Vector{T}(undef, total_length)
    y[t.range] = [x[1]]
    return y
end
function (t::ProductVecInvTransform)(y::AbstractVector{T}) where {T}
    x = Vector{T}(undef, 1)
    x[1] = view(y, t.range)[]
    return x
end

ffwd = ProductVecTransform(1:1)
frvs = ProductVecInvTransform(1:1)

f = ffwd ∘ frvs
xvec = randn(1)
jacobian(Forward, f, xvec)

@info "Done"
