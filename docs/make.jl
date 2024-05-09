using KAN
using Documenter

DocMeta.setdocmeta!(KAN, :DocTestSetup, :(using KAN); recursive=true)

makedocs(;
    modules=[KAN],
    authors="rishabh",
    sitename="KAN.jl",
    format=Documenter.HTML(;
        canonical="https://rbSparky.github.io/KAN.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/rbSparky/KAN.jl",
    devbranch="main",
)
