module Shaymin

export shay

"""
    shay()

Generate a very cute number from a normal distribution.

```jldoctest
julia> using Shaymin, Random

julia> Random.seed!(492)  # Shaymin's Pokedex number
TaskLocalRNG()

julia> shay()
-0.7193893056115281
```
"""
function shay()
    return randn()
end

end # module Shaymin
