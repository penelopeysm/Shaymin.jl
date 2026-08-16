# using DynamicPPL, Distributions, LogDensityProblems, ADTypes, Enzyme
# @model f() = x ~ Wishart(7, [1.0 0.5; 0.5 1.0])
# adtype = AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const)
# DynamicPPL.TestUtils.AD.run_ad(f(), adtype; test=false, benchmark=true)

using Distributions, Enzyme, Bijectors

const d = Wishart(7, [1.0 0.5; 0.5 1.0])
x = rand(d)
xv = Bijectors.VectorBijectors.to_linked_vec(d)(x)
const invl = Bijectors.VectorBijectors.from_linked_vec(d)
f(x) = logpdf(d, invl(x))

@info f(xv)
@info gradient(Reverse, Const(f), xv)
