module AbstractPPLMooncakeExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators:
    Evaluators, Prepared, VectorEvaluator, NamedTupleEvaluator, _ad_output_arity
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake: Mooncake

const _MooncakeAD = Union{AutoMooncake,AutoMooncakeForward}

# `NoTangent` stops Mooncake from deriving a `Tangent{...}` for the evaluator
# wrapper's fields on each backward pass. Load-bearing for both:
#   * `NamedTupleEvaluator` — passed directly to Mooncake on the NamedTuple
#                             gradient path.
#   * `VectorEvaluator`     — wrapped by the order=2 path (Mooncake's Hessian
#                             API accepts only `AbstractVector` arguments, so
#                             context is closed over via `VectorEvaluator{false}`).
Mooncake.tangent_type(::Type{<:VectorEvaluator}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:NamedTupleEvaluator}) = Mooncake.NoTangent

# Type parameters:
#
#   * `A::Symbol` — `:scalar` / `:vector` for order=1 (output arity), `:hessian`
#                   for order=2. Drives every dispatch decision below.
#   * `Target`    — `Nothing` for order=1 (Mooncake's gradient/Jacobian API
#                   takes `f` and `context` as separate args). For `:hessian`,
#                   a `VectorEvaluator{false}` that closes over `f` and
#                   `context`, since Mooncake's Hessian API accepts only
#                   `AbstractVector` arguments. The evaluator's `NoTangent`
#                   tangent type prevents differentiation of its fields.
#   * `C`         — the underlying Mooncake cache, or `Nothing` for the
#                   empty-input shortcut.
#   * `G`         — gradient cache populated only at order=2 so the order=1
#                   `value_and_gradient!!` entry on a Hessian prep skips the
#                   Hessian work. `Nothing` for every order=1 path.
struct MooncakeCache{A,Target,C,G}
    target::Target
    cache::C
    gradient_cache::G
    function MooncakeCache{A}(
        target::Target, cache::C, gradient_cache::G=nothing
    ) where {A,Target,C,G}
        return new{A,Target,C,G}(target, cache, gradient_cache)
    end
end
MooncakeCache{A}(cache) where {A} = MooncakeCache{A}(nothing, cache)

_mooncake_config(adtype) = adtype.config === nothing ? Mooncake.Config() : adtype.config

# Mooncake exposes separate `prepare_*_cache` entries per AD mode; the call
# shape (callable + active arg + extra args) is the same. Used by the
# NamedTuple path, the order=1 scalar branch, and the order=2 gradient prep.
function _mooncake_gradient_cache(::AutoMooncake, f, x, args...; config)
    return Mooncake.prepare_gradient_cache(f, x, args...; config)
end
function _mooncake_gradient_cache(::AutoMooncakeForward, f, x, args...; config)
    return Mooncake.prepare_derivative_cache(f, x, args...; config)
end

function AbstractPPL.prepare(
    adtype::_MooncakeAD, problem, values::NamedTuple; check_dims::Bool=true
)
    evaluator = AbstractPPL.prepare(problem, values; check_dims)::NamedTupleEvaluator
    config = _mooncake_config(adtype)
    cache = _mooncake_gradient_cache(adtype, evaluator, values; config)
    return Prepared(adtype, evaluator, cache)
end

"""
    prepare(adtype::AutoMooncake, problem, x; check_dims=true, context::Tuple=(), order=1)
    prepare(adtype::AutoMooncakeForward, problem, x; check_dims=true, context::Tuple=(), order=1)

Prepare a Mooncake gradient, Jacobian, or Hessian evaluator for a dense vector
input. `order=1` (default) picks gradient/Jacobian by output arity;
`order=2` builds Hessian machinery (`value_gradient_and_hessian!!`) and
requires a scalar-valued problem.

Non-`DenseVector` inputs (views, strided slices) are rejected: Mooncake
assumes a contiguous primal and otherwise returns shape-incorrect tangents
on reverse mode and crashes on forward/Jacobian paths.

`context` follows the base `prepare` contract — the prepared evaluator
computes `problem(x, context...)` with AD differentiating only `x`. One
Mooncake-specific restriction for `order=1`: vector-valued problems require
`context=()`. `order=2` accepts any `context`.

Empty input (`length(x) == 0`) is supported with any `context`; Mooncake
builds no tape for zero-length `x`, so the prepared evaluator's AD entry
short-circuits without invoking Mooncake.
"""
function AbstractPPL.prepare(
    adtype::_MooncakeAD,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
    context::Tuple=(),
    order::Int=1,
)
    Evaluators._validate_ad_order(order)
    x isa DenseVector || throw(
        ArgumentError(
            "AutoMooncake / AutoMooncakeForward require a dense vector input " *
            "(e.g. `Vector{<:Real}`); got $(typeof(x)). Wrap non-dense inputs " *
            "(views, strided slices) with `collect` before calling `prepare`.",
        ),
    )
    evaluator = AbstractPPL.prepare(problem, x; check_dims, context)::VectorEvaluator
    arity = _ad_output_arity(evaluator(x))
    config = _mooncake_config(adtype)
    if order == 2
        arity === :scalar || Evaluators._throw_hessian_needs_scalar()
        # `{false}` skips the per-call shape check — `_check_ad_input` on the
        # AD entry already validates `x`. `dim` is unused for `{false}`.
        target = VectorEvaluator{false}(evaluator.f, 0, evaluator.context)
        length(x) == 0 && return Prepared(
            adtype, evaluator, MooncakeCache{:hessian}(target, nothing), Val(2)
        )
        hess_cache = Mooncake.prepare_hessian_cache(target, x; config)
        # Order=1 gradient cache so `value_and_gradient!!` on the same prep
        # skips the Hessian work. Mooncake's `value_and_gradient!!` runs on
        # `evaluator.f` with context-as-extra-args, distinct from the wrapped
        # `target` used by the Hessian API.
        grad_cache = _mooncake_gradient_cache(
            adtype, evaluator.f, x, evaluator.context...; config
        )
        return Prepared(
            adtype,
            evaluator,
            MooncakeCache{:hessian}(target, hess_cache, grad_cache),
            Val(2),
        )
    end
    if !isempty(evaluator.context) && arity !== :scalar
        throw(
            ArgumentError(
                "Non-empty `context` is only supported for scalar-valued problems."
            ),
        )
    end
    # Mooncake builds no tape for length-zero `x`; tag with `Nothing` so the
    # empty-input methods below shortcut without invoking Mooncake. Empty `x`
    # with non-empty context also routes here — the hot-path shortcut just
    # calls `p.evaluator(x)` which already does `f([], context...)`.
    length(x) == 0 && return Prepared(adtype, evaluator, MooncakeCache{arity}(nothing))
    # Compile the tape on the evaluator's `f` and `context` (not the raw
    # `problem` / `context` kwargs): a downstream override of structural
    # `prepare` may return a `VectorEvaluator` whose `.f`/`.context` differ
    # from the caller-supplied values, and the hot path reads them off the
    # evaluator. The reverse-mode vector branch is the only one that can't
    # share `_mooncake_gradient_cache` — it needs `prepare_pullback_cache`.
    cache = if arity !== :scalar && adtype isa AutoMooncake
        Mooncake.prepare_pullback_cache(evaluator.f, x; config)
    else
        _mooncake_gradient_cache(adtype, evaluator.f, x, evaluator.context...; config)
    end
    return Prepared(adtype, evaluator, MooncakeCache{arity}(cache))
end

# Input-shape validation is delegated to the AD backend: Mooncake catches
# top-level NamedTuple-type mismatches, and the inner
# `NamedTupleEvaluator{CheckInput}` callable catches nested-array size
# mismatches (gated by `check_dims`). Running `_assert_namedtuple_shape`
# again here would duplicate the second check on every AD call.
# (`∂f` is `NoTangent` thanks to the `tangent_type` overload above.)
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:_MooncakeAD,<:NamedTupleEvaluator}, values::NamedTuple
)
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, values)
    return (val, grad)
end

# Empty-input shortcut. `MooncakeCache{:scalar,Nothing,Nothing}` is strictly
# more specific than `MooncakeCache{:scalar}`, so dispatch unambiguously selects
# this method over the general scalar-gradient hot path below for zero-length
# `x`.
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:scalar,Nothing,Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[])
end

# Scalar-gradient hot path. Reverse mode (`Mooncake.Cache`) needs
# `args_to_zero` to mark `x` as the lone active input (`false` on `f`,
# `true` on `x`, `false` on each context value); forward mode
# (`ForwardCache`) derives activity from its seeded argument and rejects
# the kwarg. The `adtype isa AutoMooncake` branch is compile-folded
# since the concrete type lives in `Prepared`'s type parameters. Empty
# `context` collapses the splat and reduces `args_to_zero` to
# `(false, true)`. `tangents[2]` is the `x`-gradient; trailing entries
# (one per context value) are inactive and discarded.
@inline function _mooncake_value_and_gradient(adtype, gcache, e::VectorEvaluator, x)
    val, tangents = if adtype isa AutoMooncake
        Mooncake.value_and_gradient!!(
            gcache,
            e.f,
            x,
            e.context...;
            args_to_zero=(false, true, map(_ -> false, e.context)...),
        )
    else
        Mooncake.value_and_gradient!!(gcache, e.f, x, e.context...)
    end
    return (val, tangents[2])
end

@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:scalar}},
    x::AbstractVector{<:Real},
)
    Evaluators._check_ad_input(p.evaluator, x)
    return _mooncake_value_and_gradient(p.adtype, p.cache.cache, p.evaluator, x)
end

# Arity-mismatch errors as dedicated methods so dispatch on
# `MooncakeCache{:scalar}` vs `{:vector}` vs `{:hessian}` resolves at compile
# time instead of a runtime check on the cache contents.
@inline function AbstractPPL.value_and_gradient!!(
    ::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:vector}},
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_gradient_needs_scalar()
end

# Empty-input shortcut for order=2 preps: same `Nothing` specificity trick
# as the scalar case. `gradient_cache` is `Nothing` only on the empty-x prep.
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{
        <:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:hessian,<:Any,Nothing,Nothing}
    },
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[])
end

# Order=2 prep also satisfies the order=1 gradient contract via the dedicated
# gradient cache built at prep time — skips the O(n²) Hessian work.
@inline function AbstractPPL.value_and_gradient!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:hessian}},
    x::AbstractVector{<:Real},
)
    Evaluators._check_ad_input(p.evaluator, x)
    return _mooncake_value_and_gradient(p.adtype, p.cache.gradient_cache, p.evaluator, x)
end

@inline function AbstractPPL.value_and_jacobian!!(
    ::Prepared{
        <:_MooncakeAD,
        <:VectorEvaluator,
        <:Union{MooncakeCache{:scalar},MooncakeCache{:hessian}},
    },
    ::AbstractVector{<:Real},
)
    return Evaluators._throw_jacobian_needs_vector()
end

# Empty-input jacobian shortcut. Same `Nothing` specificity trick as the
# scalar case above; skips Mooncake entirely.
@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:vector,Nothing,Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    val = p.evaluator(x)
    return (val, similar(x, length(val), 0))
end

@inline function AbstractPPL.value_and_jacobian!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:vector}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    # Vector arity rejects non-empty `context` at prepare time, so the tape
    # is compiled on `problem(x)` and there is no splat or `args_to_zero` to
    # propagate. Mooncake's `value_and_jacobian!!` returns `(val, jac)`
    # directly with `x` as the only active argument.
    return Mooncake.value_and_jacobian!!(p.cache.cache, p.evaluator.f, x)
end

# Order=1 prep rejected for Hessian. `MooncakeCache{:hessian}` has dedicated
# methods below that are strictly more specific, so this catch-all only fires
# for `:scalar` / `:vector`.
@inline function AbstractPPL.value_gradient_and_hessian!!(
    ::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache}, ::AbstractVector{<:Real}
)
    return Evaluators._throw_hessian_needs_order_2_prep()
end

# Empty-input shortcut — Mooncake builds no tape for length-zero `x`.
@inline function AbstractPPL.value_gradient_and_hessian!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:hessian,<:Any,Nothing}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    return (p.evaluator(x), T[], similar(x, 0, 0))
end

@inline function AbstractPPL.value_gradient_and_hessian!!(
    p::Prepared{<:_MooncakeAD,<:VectorEvaluator,<:MooncakeCache{:hessian}},
    x::AbstractVector{T},
) where {T<:Real}
    Evaluators._check_ad_input(p.evaluator, x)
    # Mooncake's `value_gradient_and_hessian!!` currently allocates fresh
    # gradient and Hessian arrays per call despite the `!!` name (unlike its
    # `value_and_gradient!!`, which does alias `HVPCache` storage). The
    # AbstractPPL `!!` contract permits aliasing rather than requiring it, so
    # this is conformant; once Mooncake's `HVPCache` is updated to reuse
    # output buffers, the returned arrays here will alias automatically with
    # no extension change needed.
    return Mooncake.value_gradient_and_hessian!!(p.cache.cache, p.cache.target, x)
end

end # module
