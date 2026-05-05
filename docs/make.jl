using Pkg
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using RegisterPenalty

DocMeta.setdocmeta!(
    RegisterPenalty,
    :DocTestSetup,
    :(using RegisterPenalty, RegisterCore);
    recursive = true,
)

makedocs(
    modules = [RegisterPenalty],
    sitename = "RegisterPenalty.jl",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/HolyLab/RegisterPenalty.jl.git",
    devbranch = "master",
)
