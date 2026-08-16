using Enzyme

function invpd(yvec::AbstractVector{T}) where {T<:Real}
    X = zeros(T, 2, 2)
    idx = 1
    for i in 1:2
        for j in 1:i
            X[i, j] = yvec[idx]
            idx += 1
        end
    end
    return X
end
xv = randn(3)
f(x) = sum(invpd(x))

@info f(xv)
@info gradient(Reverse, Const(f), xv)
