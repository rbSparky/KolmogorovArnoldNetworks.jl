# KolmogorovArnoldNetworks.jl
# This is in very early stages
## If you are interested in becoming a maintainer, please open a PR!

KolmogorovArnoldNetworks.jl is a Julia library providing implementations of [Kolmogorov-Arnold neural networks](https://arxiv.org/abs/2404.19756). Implementation inspired by [Efficient KAN](https://github.com/Blealtan/efficient-kan).

Please feel free to contribute!

## TODO: Fix example.jl
Current error:
```
KANLinear forward - input x shape: (784, 128)
b_splines - x shape: (784, 128), grid shape: (784, 12)
b_splines - bases shape: (784, 128, 8)
KANLinear forward - base_output shape: (128, 128), spline_output shape: (128, 128)
KANLinear forward - base_output shape: (128, 128), spline_output shape: (128, 128)
KANLinear forward - input x shape: (128, 128)
b_splines - x shape: (128, 128), grid shape: (128, 12)
b_splines - bases shape: (128, 128, 8)
KANLinear forward - base_output shape: (64, 128), spline_output shape: (64, 128)
KANLinear forward - base_output shape: (64, 128), spline_output shape: (64, 128)
KANLinear forward - input x shape: (64, 128)
b_splines - x shape: (64, 128), grid shape: (64, 12)
b_splines - bases shape: (64, 128, 8)
KANLinear forward - base_output shape: (10, 128), spline_output shape: (10, 128)
KANLinear forward - base_output shape: (10, 128), spline_output shape: (10, 128)
KAN forward - final output shape: (10, 128)
ERROR: DomainError with Loss is NaN on data item 1, stopping training:
```

## Installation (soon)

You can install KolmogorovArnoldNetworks.jl using the Julia package manager. From the Julia REPL, type `]` to enter the package manager mode, then run:

```julia
pkg> add KolmogorovArnoldNetworks
```
