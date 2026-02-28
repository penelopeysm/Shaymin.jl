using InteractiveUtils: versioninfo
versioninfo()

using Enzyme: Forward, Reverse, jacobian
Enzyme.Compiler.VERBOSE_ERRORS[] = true

struct F{S}
    ranges::S
end

function (t::F)(x::AbstractArray{T}) where {T}
    total_length = sum(length, t.ranges)
    y = Vector{T}(undef, total_length)
    y[1:1] = [x[1]]
    return y
end

# must be composed with identity for it to fail
f = F((1:1,)) ∘ identity
x = randn(1)
jacobian(Forward, f, x)
