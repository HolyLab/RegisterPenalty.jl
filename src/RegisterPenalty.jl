# Penalty function for registration of a single image

module RegisterPenalty

using CachedInterpolations: CachedInterpolations, CachedInterpolation
using CenterIndexedArrays: CenterIndexedArrays, CenterIndexedArray
using Interpolations:
    Interpolations,
    AbstractInterpolation,
    BSpline,
    Gridded,
    InPlace,
    Linear,
    OnCell,
    Quadratic,
    Degree,
    InterpolationType
using LinearAlgebra: LinearAlgebra, qr
using StaticArrays: StaticArrays, SVector
using Base.Cartesian
using RegisterCore: RegisterCore, ColonFun, MismatchArray, NumDenom, maxshift
using RegisterDeformation:
    RegisterDeformation,
    AbstractDeformation,
    GridDeformation,
    compose,
    interpolate!,
    vecgradient!,
    vecindex,
    convert_from_fixed, # internal API, used intentionally
    convert_to_fixed     # internal API, used intentionally

export AffinePenalty, DeformationPenalty, penalty!, interpolate_mm!


"""
This module computes the total registration penalty, combining both
"data" (the mismatch between `fixed` and `moving` images) and
"regularization" (a penalty applied to deformations that do not fit
some pre-conceived notion of "goodness").

The main exported types/functions are:

- [`AffinePenalty`](@ref): regularization that penalizes deviations from an affine transformation
- [`DeformationPenalty`](@ref): abstract supertype; subtype to implement custom regularization
- [`penalty!`](@ref): compute the total, data, or regularization penalty
- [`interpolate_mm!`](@ref): prepare mismatch arrays for interpolation
"""
RegisterPenalty


"""
    DeformationPenalty{T, N}

Abstract supertype for regularization penalties on `N`-dimensional deformation
fields with element type `T`.

Subtypes implement regularization by defining a `penalty!` method:

    penalty!(g, dp::MyPenalty, ϕ_c::AbstractDeformation) -> scalar

where `g` is pre-allocated gradient storage (written in-place) and `ϕ_c` is
the (possibly composed) deformation being penalized. The built-in subtype is
[`AffinePenalty`](@ref).

`Base.eltype` and `Base.ndims` are defined for all subtypes and
return `T` and `N` respectively.
"""
abstract type DeformationPenalty{T, N} end
Base.eltype(::Type{DeformationPenalty{T, N}}) where {T, N} = T
Base.eltype(::Type{DP}) where {DP <: DeformationPenalty} = eltype(supertype(DP))
Base.eltype(dp::DeformationPenalty) = eltype(typeof(dp))
Base.ndims(::Type{DeformationPenalty{T, N}}) where {T, N} = N
Base.ndims(::Type{DP}) where {DP <: DeformationPenalty} = ndims(supertype(DP))
Base.ndims(dp::DeformationPenalty) = ndims(typeof(dp))

"""
    p = penalty!(g, ϕ, ϕ_old, dp::DeformationPenalty, mmis)
    p = penalty!(g, ϕ, ϕ_old, dp::DeformationPenalty, mmis, keep)

Compute the total penalty (regularization + data) for deformation `ϕ`, mismatch
data `mmis`, and an optional "base" deformation `ϕ_old`. Returns a scalar of
the same element type as `ϕ`.

When `ϕ_old` is not `identity`, the regularization penalty is evaluated on the
composed deformation `ϕ_c = ϕ_old(ϕ)`, while the data penalty uses `ϕ` and
`mmis` (which should express the *residual* mismatch after applying `ϕ_old`).
This supports an incremental registration workflow:

- Compute initial deformation `ϕ_0` that partially aligns `fixed` and `moving`
- Warp `moving` with `ϕ_0`
- Compute residual mismatch between `fixed` and the warped image
- Optimize `ϕ_1` to correct the residual; final deformation is `ϕ_0(ϕ_1)`

This method resolves to:

    val =  penalty!(g, dp, ϕ_c, g_c)   # regularization; see [`penalty!(g, dp, ϕ_c)`](@ref)
    val += penalty!(g, ϕ, mmis, keep)  # data;           see [`penalty!(g, ϕ, mmis)`](@ref)

`g_c` is the Jacobian of `ϕ_c` with respect to `ϕ.u`. When `ϕ_old == identity`,
no composition is needed and `ϕ` is used directly.

`ϕ_old` must be interpolating if not `identity`; `ϕ` must not be interpolating.

`g` may be a flat `Vector{T}` or an array of `SVector{N,T}` with the same shape
as `ϕ.u`. It is **written** by the regularization term and **incremented** by
the data term; initialize it with zeros before calling.
"""
function penalty!(g, ϕ, ϕ_old, dp::DeformationPenalty, mmis::AbstractArray, keep = trues(size(mmis)))
    T = eltype(ϕ)
    val = zero(T)
    # Volume penalty
    if ϕ_old == identity
        val = penalty!(g, dp, ϕ)
    else
        ϕ_c, g_c = compose(ϕ_old, ϕ)
        val = penalty!(g, dp, ϕ_c, g_c)
    end
    if !isfinite(val)
        return val
    end
    # Data penalty
    val += penalty!(g, ϕ, mmis, keep)
    return convert(T, val)
end

# Allow it to be called without StaticArrays
function penalty!(g::Array{T}, ϕ, ϕ_old, dp::DeformationPenalty, mmis::AbstractArray, keep = trues(size(mmis))) where {T <: Number}
    gf = convert_to_fixed(g, (ndims(dp), size(ϕ.u)...))
    return penalty!(gf, ϕ, ϕ_old, dp, mmis, keep)
end


"""
    p = penalty!(gs, ϕs, ϕs_old, dp, λt, mmis)
    p = penalty!(gs, ϕs, ϕs_old, dp, λt, mmis, keeps)

Evaluate the total penalty for a temporal sequence of deformations `ϕs` and
the corresponding mismatch data `mmis`. `λt` is the temporal roughness penalty
coefficient. Returns a scalar of the same element type as the deformations.

This is the temporal-sequence analogue of the single-frame
[`penalty!(g, ϕ, ϕ_old, dp, mmis)`](@ref): it calls the single-frame penalty
for each time point and adds the temporal-roughness penalty
`penalty!(gs, λt, ϕs)`.
"""
function penalty!(gs, ϕs::AbstractVector{D}, ϕs_old, dp::DeformationPenalty, λt::Number, mmis::AbstractArray, keep = trues(size(mmis))) where {D <: AbstractDeformation}
    ntimes = length(ϕs)
    size(mmis)[end] == ntimes || throw(DimensionMismatch("Number of deformations $ntimes does not agree with mismatch data of size $(size(mmis))"))
    s = _penalty!(gs, ϕs, ϕs_old, dp, mmis, keep, 1)
    for i in 2:ntimes
        isfinite(s) || break
        s += _penalty!(gs, ϕs, ϕs_old, dp, mmis, keep, i)
    end
    return s + penalty!(gs, λt, ϕs)
end


function penalty!(gs::Array{T}, ϕs::AbstractVector{D}, ϕs_old, dp::DeformationPenalty, λt::Number, mmis::AbstractArray, keep = trues(size(mmis))) where {T <: Number, D <: AbstractDeformation}
    gf = convert_to_fixed(gs, (ndims(dp), size(first(ϕs).u)..., length(ϕs)))
    return penalty!(gf, ϕs, ϕs_old, dp, λt, mmis, keep)
end

function _penalty!(gs, ϕs, ϕs_old, dp::DeformationPenalty{T, N}, mmis, keeps, i) where {T, N}
    colons = ntuple(ColonFun, Val(N))
    indexes = (colons..., i)
    #mmi = view(mmis, indexes...)  # making these views runs afoul of inference limits
    mmi = mmis[indexes...]
    #keep = view(keeps, indexes...)
    keep = keeps[indexes...]
    calc_gradient = gs != nothing && !isempty(gs)
    g = calc_gradient ? view(gs, indexes...) : nothing
    return if isa(ϕs_old, AbstractVector)
        penalty!(g, ϕs[i], ϕs_old[i], dp, mmi, keep)
    else
        penalty!(g, ϕs[i], ϕs_old, dp, mmi, keep)
    end
end


################
# Data penalty #
################

const CenteredInterpolant{T, N, A <: AbstractInterpolation} = Union{MismatchArray{T, N, A}, CachedInterpolation{T, N}}

"""
    p = penalty!(g, ϕ, mmis)
    p = penalty!(g, ϕ, mmis, keep)
    p = penalty!(g, u, mmis)
    p = penalty!(g, u, mmis, keep)

Compute the data penalty — the total mismatch between `fixed` and `moving`
given deformation `ϕ` (or displacement array `u`). Returns a scalar of the
same element type as the mismatch data.

`mmis` is an array-of-`CenterIndexedArray`s prepared by [`interpolate_mm!`](@ref).
`keep` is an optional boolean array the same size as `mmis`; blocks with
`keep[i] == false` are skipped (treated as zero mismatch).

The penalty is a globally-normalized ratio:

        pnum_1 + pnum_2 + ... + pnum_n
   p = --------------------------------
        pden_1 + pden_2 + ... + pden_n

where each `pnum_i / pden_i` is the mismatch at aperture `i` evaluated at
`ϕ.u[i]` (or `u[i]`).

`g` is pre-allocated gradient storage (same shape as `ϕ.u` or `u`), and may be
`nothing` or empty to skip gradient computation. This function **adds** to `g`;
initialize it with zeros or call the regularization penalty first.
"""
function penalty!(g, ϕ::AbstractDeformation, mmis::AbstractArray{M}, keep = trues(size(mmis))) where {M <: CenteredInterpolant}
    return penalty!(g, ϕ.u, mmis, keep)
end

function penalty!(g::Array{Tg}, ϕ::AbstractDeformation, mmis::AbstractArray{M}, keep = trues(size(mmis))) where {Tg <: Number, M <: CenteredInterpolant}
    gf = convert_to_fixed(g, (ndims(ϕ), size(ϕ.u)...))
    return penalty!(gf, ϕ, mmis, keep)
end

function penalty!(g, u::AbstractArray{SVector{Dim, Tu}}, mmis::AbstractArray{M}, keep = trues(size(mmis))) where {Tu, Dim, M <: CenteredInterpolant}
    # This "outer" function just handles the chain rule for computing the
    # total penalty and gradient. The "real" work is done by penalty_nd!.
    nblocks = length(mmis)
    length(u) == nblocks || error("u should have length $nblocks, but length(u) = $(length(u))")
    calc_gradient = g != nothing && !isempty(g)
    if calc_gradient
        if length(g) != length(u)
            error("length(g) = $(length(g)) but length(u) = $(length(u))")
        end
    end
    if calc_gradient
        gnd = similar(u, SVector{Dim, NumDenom{Tu}})
        nd = penalty_nd!(gnd, u, mmis, keep)
        N, D = nd.num, nd.denom
        invD = 1 / D
        NinvD2 = N * invD * invD
        for i in 1:length(g)
            g[i] += _wsum(gnd[i], invD, -NinvD2)
        end
        return N * invD
    else
        nd = penalty_nd!(g, u, mmis, keep)
        N, D = nd.num, nd.denom
        return N / D
    end
end

# Computes pnum_i and pden_i and their gradients
function penalty_nd!(gnd, u::AbstractArray, mmis, keep)
    N = ndims(u)
    T = eltype(eltype(u))
    calc_grad = gnd != nothing && !isempty(gnd)
    mxs = maxshift(first(mmis))
    nd = NumDenom{T}(0, 0)
    nanT = convert(T, NaN)
    local gtmp
    if calc_grad
        gtmp = Vector{NumDenom{T}}(undef, N)
    end
    for iblock in 1:length(mmis)
        mmi = mmis[iblock]
        if !keep[iblock]
            if calc_grad
                gnd[iblock] = SVector(ntuple(_ -> NumDenom{T}(0, 0), Val(N)))
            end
            continue
        end
        # Check bounds
        dx = u[iblock]
        if !checkbounds_shift(dx, mxs)
            return NumDenom{T}(nanT, nanT)
        end
        # Evaluate the value
        nd += vecindex(mmi, dx)
        # Evaluate the gradient
        if calc_grad
            vecgradient!(gtmp, mmi, dx)
            gnd[iblock] = gtmp
        end
    end
    return nd
end

penalty_nd!(gnd, u::AbstractInterpolation, mmis, keep) = error("ϕ must not be interpolating")

@generated function checkbounds_shift(dx::SVector{N}, mxs) where {N}
    return quote
        @nexprs $N d -> (
            if abs(dx[d]) >= mxs[d] - 0.5
                return false
            end
        )
        true
    end
end

@generated function _wsum(x::SVector{N}, cnum, cdenom) where {N}
    args = [:(cnum * x[$d].num + cdenom * x[$d].denom) for d in 1:N]
    return quote
        SVector($(args...))
    end
end

##########################
# Regularization penalty #
##########################

### Affine-residual penalty
"""
    AffinePenalty(nodes::NTuple{N, <:AbstractVector}, λ)
    AffinePenalty(nodes::AbstractVector{<:AbstractVector}, λ)
    AffinePenalty(nodes::AbstractMatrix, λ)

Construct an affine-residual regularization penalty for use with [`penalty!`](@ref).
Returns an `AffinePenalty{T,N}` where `T` is the element type inferred from `nodes`
and `N` is the spatial dimensionality.

The penalty measures how much a deformation's displacement field `u` deviates from
any affine transformation:

    penalty = (λ/n) * ∑_i ‖u_i - a_i‖²

where `{a_i}` is the least-squares affine fit to `{u_i}` and `n = length(u)`.

For regular grids, pass `nodes` as an `N`-tuple of coordinate vectors (one per
spatial dimension) or as a vector of such vectors. For irregular point clouds,
pass an `N×npoints` matrix whose columns are the node coordinates.

# Examples

```jldoctest
julia> nodes = (range(-1.0, 1.0, length=5), range(-1.0, 1.0, length=5));

julia> ap = AffinePenalty(nodes, 0.1);

julia> ndims(ap), eltype(ap)
(2, Float64)
```
"""
mutable struct AffinePenalty{T, N} <: DeformationPenalty{T, N}
    const F::Matrix{T}   # geometry data for the affine-residual penalty
    λ::T           # regularization coefficient

    AffinePenalty{T, N}(F::Matrix{T}, λ::T, _) where {T, N} = new{T, N}(F, λ)

    function AffinePenalty{T, N}(nodes::NTuple{N}, λ) where {T, N}
        gridsize = map(length, nodes)
        C = Matrix{Float64}(undef, prod(gridsize), N + 1)
        i = 0
        for I in CartesianIndices(gridsize)
            C[i += 1, N + 1] = 1
            for j in 1:N
                C[i, j] = nodes[j][I[j]]  # I[j]
            end
        end
        F, _ = qr(C)
        return new{T, N}(Array(F), λ)
    end

    function AffinePenalty{T, N}(nodes::AbstractMatrix, λ) where {T, N}
        C = hcat(nodes', ones(eltype(nodes), size(nodes, 2)))
        F, _ = qr(C)
        return new{T, N}(F, λ)
    end
end

AffinePenalty(nodes::NTuple{N, <:AbstractVector{T}}, λ) where {T, N} = AffinePenalty{T, N}(nodes, λ)
AffinePenalty(nodes::AbstractVector{<:AbstractVector{T}}, λ) where {T} = AffinePenalty{T, length(nodes)}((nodes...,), λ)
AffinePenalty(nodes::AbstractMatrix{T}, λ) where {T} = AffinePenalty{T, size(nodes, 1)}(nodes, λ)

Base.convert(::Type{AffinePenalty{T, N}}, ap::AffinePenalty) where {T, N} = AffinePenalty{T, N}(convert(Matrix{T}, ap.F), convert(T, ap.λ), 0)

"""
    p = penalty!(g, dp::DeformationPenalty, ϕ_c)
    p = penalty!(g, dp::DeformationPenalty, ϕ_c, g_c)

Compute the regularization penalty for a (possibly composed) deformation `ϕ_c`.
Returns a non-negative scalar of the same element type as `ϕ_c`.

The `_c` suffix indicates "composed": use the two-argument form when
`ϕ_c = ϕ_old(ϕ)` and `g_c` is the Jacobian of that composition with respect to
`ϕ.u`. When `ϕ` is not derived by composition, pass it directly as `ϕ_c` and
omit `g_c`.

`dp` determines the type of penalty; the built-in implementation is
[`AffinePenalty`](@ref). Custom regularization can be added by subtyping
[`DeformationPenalty`](@ref) and defining a new `penalty!` method.

If `g` is non-`nothing` and non-empty, the gradient of the penalty is
**written** into `g` (unlike the data penalty, which adds to `g`). When `g_c`
is supplied, `g` is adjusted by the chain rule so that it is the gradient with
respect to `ϕ.u` rather than `ϕ_c.u`.
"""
function penalty!(g, dp::AffinePenalty, ϕ_c::AbstractDeformation{T, N}) where {T, N}
    return penalty!(g, dp, ϕ_c.u)
end

function penalty!(g, dp::AffinePenalty, u::AbstractArray{SVector{N, T}, N}) where {T, N}
    F, λ = dp.F, dp.λ
    if λ == 0
        if g != nothing && !isempty(g)
            fill!(g, zero(eltype(g)))
        end
        return λ * one(eltype(F)) * one(T)
    end
    n = length(u)
    U = convert_from_fixed(u, (n,))
    A = (U * F) * F'   # projection onto an affine transformation
    dU = U - A
    λ /= n
    if g != nothing && !isempty(g)
        λ2 = 2λ
        du = convert_to_fixed(SVector{N, T}, dU, (n,))
        for j in 1:n
            g[j] = λ2 * du[j]
        end
    end
    return λ * sum(abs2, dU)
end

function penalty!(g, dp::AffinePenalty, ϕ_c, g_c)
    ret = penalty!(g, dp, ϕ_c)
    if g != nothing
        for i in 1:length(g)
            g[i] = g_c[i]' * g[i]
        end
    end
    return ret
end

###
### Temporal penalty
###
"""
    p = penalty!(g, λt, ϕs)

Compute the temporal-roughness penalty

    (λt/2) * ∑_i ‖ϕ_{i+1} - ϕ_i‖²

for a vector `ϕs` of `GridDeformation`s. Returns a non-negative scalar of the
same element type as the deformations.

`g`, if non-`nothing` and non-empty, must be a flat `Vector` whose length equals
`length(ϕs) * length(first(ϕs).u)`. On return, `g` holds the gradient of the
penalty with respect to all displacement vectors.
"""
function penalty!(g, λt::Real, ϕs::Vector{D}) where {D <: GridDeformation}
    if g == nothing || isempty(g)
        return penalty(λt, ϕs)
    end
    ngrid = length(first(ϕs).u)
    if length(g) != length(ϕs) * ngrid
        gsize = (size(first(ϕs).u)..., length(ϕs))
        throw(DimensionMismatch("g's length $(length(g)) inconsistent with $gsize"))
    end
    local s = zero(eltype(D))
    λt2 = λt / 2
    for i in 1:(length(ϕs) - 1)
        ϕ = ϕs[i]
        ϕp = ϕs[i + 1]
        goffset = ngrid * (i - 1)
        for k in 1:ngrid
            du = ϕp.u[k] - ϕ.u[k]
            dv = λt * du
            g[goffset + k] -= dv
            g[goffset + ngrid + k] += dv
            s += λt2 * sum(abs2, du)
        end
    end
    return s
end

function penalty!(g::Array{T}, λt::Real, ϕs::Vector{D}) where {T <: Number, D <: GridDeformation}
    N = ndims(first(ϕs))
    sz = size(first(ϕs).u)
    gf = convert_to_fixed(g, (N, sz..., length(ϕs)))
    return penalty!(gf, λt, ϕs)
end

function penalty(λt::Real, ϕs::Vector{D}) where {D <: GridDeformation}
    s = zero(eltype(D))
    ngrid = length(first(ϕs).u)
    λt2 = λt / 2
    for i in 1:(length(ϕs) - 1)
        ϕ = ϕs[i]
        ϕp = ϕs[i + 1]
        for k in 1:ngrid
            du = ϕp.u[k] - ϕ.u[k]
            s += λt2 * sum(abs2, du)
        end
    end
    return s
end

function vec2ϕs(x::Array{T}, gridsize::NTuple{N, Int}, n, nodes) where {T, N}
    xr = convert_to_fixed(SVector{N, T}, x, (gridsize..., n))
    colons = ntuple(d -> Colon(), N)::NTuple{N, Colon}
    return [GridDeformation(view(xr, colons..., i), nodes) for i in 1:n]
end

"""
    mmi  = interpolate_mm!(mm)
    mmi  = interpolate_mm!(mm, Quadratic)
    mmi  = interpolate_mm!(mm, Linear)
    mmi  = interpolate_mm!(mm, itype)
    mmis = interpolate_mm!(mms)
    mmis = interpolate_mm!(mms, Quadratic)
    mmis = interpolate_mm!(mms, itype)

Prepare a `MismatchArray` (as returned by RegisterMismatch) for sub-pixel
interpolation. Returns a `CenterIndexedArray{NumDenom{T},N}` wrapping a
B-spline or gridded interpolant, ready for use with [`penalty!`](@ref).

The `order` argument is a `Degree` subtype from Interpolations.jl: `Quadratic`
(default) or `Linear`. With `Quadratic`, the array data are overwritten
in-place with B-spline prefilter coefficients. For fine-grained control over
the boundary condition or interpolation scheme, pass an `InterpolationType`
directly:

    mmi = interpolate_mm!(mm, BSpline(Quadratic(InPlace(OnCell()))))

The array-of-`MismatchArray`s form processes each element and returns a
`Vector{CenterIndexedArray{NumDenom{T},N}}`.

# Examples

```jldoctest
julia> using RegisterCore

julia> mm = MismatchArray(Float64, (11, 11));

julia> mmi = interpolate_mm!(mm);

julia> eltype(mmi), ndims(mmi)
(NumDenom{Float64}, 2)

julia> mms = [MismatchArray(Float64, (5, 5)) for _ in 1:4];

julia> mmis = interpolate_mm!(mms);

julia> length(mmis), eltype(first(mmis))
(4, NumDenom{Float64})
```
"""
function interpolate_mm!(mms::AbstractArray{T}, itype::InterpolationType) where {T <: MismatchArray}
    f = x -> CenterIndexedArray(interpolate!(x.data, itype))
    return map(f, mms)
end

function interpolate_mm!(mm::MismatchArray, itype::InterpolationType)
    return CenterIndexedArray(interpolate!(mm.data, itype))
end

interpolate_mm!(arg, ::Type{order}) where {order <: Degree} =
    interpolate_mm!(arg, itptype(order))

interpolate_mm!(arg) = interpolate_mm!(arg, Quadratic)

itptype(::Type{Quadratic}) = BSpline(Quadratic(InPlace(OnCell())))
itptype(::Type{Linear}) = Gridded(Linear())

@generated function Interpolations.gradient!(g::AbstractVector, A::CenterIndexedArray{T, N}, i::Number...) where {T, N}
    length(i) == N || error("Must use $N indexes")
    args = [:(i[$d] + A.halfsize[$d] + 1) for  d in 1:N]
    meta = Expr(:meta, :inline)
    return :($meta; Interpolations.gradient!(g, A.data, $(args...)))
end

end  # module
