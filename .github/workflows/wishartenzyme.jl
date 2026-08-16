# using DynamicPPL, Distributions, LogDensityProblems, ADTypes, Enzyme
# @model f() = x ~ Wishart(7, [1.0 0.5; 0.5 1.0])
# adtype = AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const)
# DynamicPPL.TestUtils.AD.run_ad(f(), adtype; test=false, benchmark=true)

using Distributions, Enzyme

struct InvPD
    original_size::Int
end
function (ip::InvPD)(yvec::AbstractVector{T}) where {T<:Real}
    d = ip.original_size
    X = zeros(T, d, d)
    idx = 1
    z = zero(T)
    weight = d + 1
    for i in 1:d
        for j in 1:i
            if i == j
                X[i, j] = exp(yvec[idx])
                z += weight * yvec[idx]
                weight -= 1
            else
                X[i, j] = yvec[idx]
            end
            idx += 1
        end
    end
    # logjac = z + (d * oftype(z, logtwo))
    return X * X' # , logjac
end

const d = Wishart(7, [1.0 0.5; 0.5 1.0])
xv = randn(3)
const invl = InvPD(2)
f(x) = logpdf(d, invl(x))

@info f(xv)
@info gradient(Reverse, Const(f), xv)
