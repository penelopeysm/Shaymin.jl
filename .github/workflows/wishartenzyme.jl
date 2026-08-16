using Enzyme

function f(y)
    X = zeros(1, 1)
    X[1, 1] = y[1]
    return sum(X * X')
end
y = [0.5]
@info f(y)
@info gradient(Reverse, Const(f), y)
