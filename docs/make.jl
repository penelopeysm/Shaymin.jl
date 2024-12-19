using Documenter, Shaymin

makedocs(
    sitename = "Hedgehog",
    modules = [Shaymin],
    format = Documenter.HTML(),
    doctest = true,
)
