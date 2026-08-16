# using DynamicPPL, Distributions, LogDensityProblems, ADTypes, Enzyme
# @model f() = x ~ Wishart(7, [1.0 0.5; 0.5 1.0])
# adtype = AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const)
# DynamicPPL.TestUtils.AD.run_ad(f(), adtype; test=false, benchmark=true)

using Distributions, Enzyme

const d = Wishart(7, [1.0 0.5; 0.5 1.0])
x = rand(d)
f(x) = logpdf(d, x)

@info f(x)
@info gradient(Reverse, Const(f), x)
