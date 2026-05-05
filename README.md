# RegisterPenalty.jl

[![CI](https://github.com/HolyLab/RegisterPenalty.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/RegisterPenalty.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/HolyLab/RegisterPenalty.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/RegisterPenalty.jl)
[![docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/RegisterPenalty.jl/stable)
[![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/RegisterPenalty.jl/dev)

RegisterPenalty computes the total optimization penalty for image registration,
combining a **data term** (mismatch between fixed and moving images) and a
**regularization term** (penalty for deformations that deviate too far from
well-behaved transformations). It is part of the
[HolyLab](https://github.com/HolyLab) image-registration ecosystem.

## Installation

This package is registered in the
[HolyLabRegistry](https://github.com/HolyLab/HolyLabRegistry). Add the registry
once, then install as usual:

```julia
using Pkg
pkg"registry add https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterPenalty")
```

## Overview

A typical registration loop looks like this:

1. Compute mismatch arrays between the fixed and (warped) moving images using
   `RegisterMismatch`.
2. Call [`interpolate_mm!`](https://HolyLab.github.io/RegisterPenalty.jl/stable/api/#RegisterPenalty.interpolate_mm!) to prepare the mismatch data for sub-pixel interpolation.
3. Create an [`AffinePenalty`](https://HolyLab.github.io/RegisterPenalty.jl/stable/api/#RegisterPenalty.AffinePenalty) that encodes the regularization geometry.
4. Call [`penalty!`](https://HolyLab.github.io/RegisterPenalty.jl/stable/api/#RegisterPenalty.penalty!) inside an optimizer to get the total penalty and its gradient.

## Quick example

```julia
using RegisterPenalty, RegisterCore, RegisterDeformation

# 1. Prepare mismatch data (produced by RegisterMismatch in practice)
mms = [MismatchArray(Float64, (11, 11)) for i in 1:3, j in 1:3]
# ... fill mms with mismatch data ...
mmis = interpolate_mm!(mms)          # quadratic B-spline interpolant (default)

# 2. Set up regularization: penalizes deviations from affine transformations
#    nodes are the deformation grid coordinates, λ controls penalty strength
nodes = (range(1.0, 512.0, length=3), range(1.0, 512.0, length=3))
ap = AffinePenalty(nodes, 0.1)       # AffinePenalty{Float64,2}

# 3. Evaluate penalty + gradient at a given deformation ϕ
g = similar(ϕ.u)
fill!(g, zero(eltype(g)))
val = penalty!(g, ϕ, identity, ap, mmis)
# val: scalar penalty; g: gradient with respect to ϕ.u
```

For temporal sequences (multiple time points), pass a `Vector` of deformations
and a temporal roughness coefficient `λt`:

```julia
val = penalty!(gs, ϕs, identity, ap, λt, mmis)
```

## Key types and functions

| Symbol | Description |
|--------|-------------|
| `AffinePenalty` | Regularizer that penalizes non-affine deformations |
| `DeformationPenalty` | Abstract supertype; subclass to add custom regularization |
| `interpolate_mm!` | Prepare mismatch arrays for sub-pixel evaluation |
| `penalty!` | Compute total (data + regularization) penalty and gradient |

See the [documentation](https://HolyLab.github.io/RegisterPenalty.jl/stable) for
the full API reference.
