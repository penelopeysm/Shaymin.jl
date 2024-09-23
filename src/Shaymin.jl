module Shaymin

export normal

using Distributions

function normal()
    return rand(Normal(0, 2))
end

end # module Shaymin
