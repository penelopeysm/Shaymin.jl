module Shaymin

export shay

"""
    shay()
    shay(n::Number)

Generate a very cute number from a normal distribution.

```jldoctest
julia> using Shaymin, Random

julia> Random.seed!(492)  # Shaymin's Pokedex number
TaskLocalRNG()

julia> shay()
-0.7193893056115281
```
"""
# Normal
function shay()
    return randn()
end

# Multiply by n
function shay(n)
        return randn() * n
end

end # module Shaymin
