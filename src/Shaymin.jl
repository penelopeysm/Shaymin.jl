module Shaymin

export shay

using Distributions

"""
    shay()

    Generate a very cute number from a normal distribution.
"""
function shay()
    return rand(Normal(0, 1))
end

end # module Shaymin
