using Enzyme

function invpd(yvec::AbstractVector{T}) where {T<:Real}
    X = zeros(T, 2, 2)
    X[1, 1] = yvec[1]
    return X * X'
end
xv = randn(1)
f(x) = sum(invpd(x))

@info f(xv)
@info gradient(Reverse, Const(f), xv)
