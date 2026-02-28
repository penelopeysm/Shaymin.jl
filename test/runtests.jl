using Enzyme: Forward, Reverse, jacobian

struct Foo1{Trng}
    ranges::Trng
end
struct Foo2{Trng}
    ranges::Trng
end

function (t::Foo1)(x::AbstractArray{T}) where {T}
    total_length = sum(length, t.ranges)
    y = Vector{T}(undef, total_length)
    y[t.ranges[1]] = [x[1]]
    return y
end
function (t::Foo2)(y::AbstractVector{T}) where {T}
    return y
    # x = Vector{T}(undef, 1)
    # x[1] = view(y, t.ranges[1])[]
    # return x
end

f = Foo1((1:1,)) ∘ Foo2((1:1,))
x = randn(1)
jacobian(Forward, f, x)
