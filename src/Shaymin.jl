module Shaymin

export shay

using Distributions

"""
    shay()

Generate a very cute number from a normal distribution.

```jldoctest
julia> using Shaymin, Random

julia> Random.seed!(492)
TaskLocalRNG()

julia> shay()
-0.7193893056115281
```
"""
function shay()
    return rand(Normal(0, 1))
end

end # module Shaymin
