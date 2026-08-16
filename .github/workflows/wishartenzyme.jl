using Enzyme

function invpd(yvec::AbstractVector{T}) where {T<:Real}
    X = zeros(T, 2, 2)
    X[1, 1] = yvec[1]
    X[1, 2] = yvec[2]
    X[2, 1] = yvec[3]
    return X * X'
end
xv = randn(3)
f(x) = sum(invpd(x))

@info f(xv)
@info gradient(Reverse, Const(f), xv)
