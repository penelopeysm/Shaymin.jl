using Enzyme

function f(y)
    X = zeros(2, 2)
    X[1, 1] = y[1]
    return sum(X * X')
end
yv = randn(1)
@info f(yv)

@info gradient(Reverse, Const(f), yv)
