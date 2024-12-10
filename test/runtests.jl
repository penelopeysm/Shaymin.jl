using Documenter
using Random
using Shaymin
using Test

@testset "doctests" begin
    DocMeta.setdocmeta!(
        Shaymin,
        :DocTestSetup,
        :(using Shaymin, Random);
        recursive=true,
    )

    doctest(Shaymin)
end
