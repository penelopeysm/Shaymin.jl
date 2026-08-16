using Enzyme

function invpd(d::Int, yvec::AbstractVector{T}) where {T<:Real}
    X = zeros(T, d, d)
    idx = 1
    for i in 1:d
        for j in 1:i
            X[i, j] = yvec[idx]
            idx += 1
        end
    end
    return X
end
xv = randn(3)
f(x) = sum(invpd(2, x))

@info f(xv)
@info gradient(Reverse, Const(f), xv)
