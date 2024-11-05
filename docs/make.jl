using Documenter, Shaymin

makedocs(
    sitename = "Hedgehog",
    format = Documenter.HTML(),
)

deploydocs(
    repo = "github.com/penelopeysm/Shaymin.jl.git",
)
