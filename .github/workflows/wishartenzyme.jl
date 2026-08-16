using Enzyme

function invpd(yvec::AbstractVector{T}) where {T<:Real}
    X = zeros(T, 2, 2)
    return X * X'
end
xv = randn(3)
f(x) = sum(invpd(x))

@info f(xv)
@info gradient(Reverse, Const(f), xv)
