# RegisterPenalty.jl

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
2. Call [`interpolate_mm!`](@ref) to prepare the mismatch data for sub-pixel
   interpolation.
3. Create an [`AffinePenalty`](@ref) that encodes the regularization geometry.
4. Call [`penalty!`](@ref) inside an optimizer to get the total penalty and its
   gradient.

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

## Incremental registration

When registration is done in stages — computing a coarse deformation `ϕ_old`
first and then refining with `ϕ` — pass `ϕ_old` as the third argument to
`penalty!`. The regularization is then evaluated on the composed deformation
`ϕ_old(ϕ)`, while the data term uses the *residual* mismatch (computed after
applying `ϕ_old`).

```julia
# ϕ_old is an interpolating deformation from the previous stage
val = penalty!(g, ϕ, ϕ_old, ap, mmis)
```

## Custom regularization

Subclass [`DeformationPenalty`](@ref) and define a `penalty!(g, dp, ϕ_c)`
method to implement your own regularization:

```julia
struct MyPenalty{T,N} <: DeformationPenalty{T,N}
    λ::T
end

function RegisterPenalty.penalty!(g, dp::MyPenalty, ϕ_c::AbstractDeformation)
    # compute penalty value, write gradient into g, return scalar
end
```

## Module docstring

```@docs
RegisterPenalty
```
