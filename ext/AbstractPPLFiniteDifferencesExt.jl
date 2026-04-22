module AbstractPPLFiniteDifferencesExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using AbstractPPL.ADProblems: _assert_namedtuple_shape
using AbstractPPL.Utils: flatten_to!!, unflatten_to!!
using ADTypes: AutoFiniteDifferences
using FiniteDifferences: FiniteDifferences

const DEFAULT_TEST_FDM = FiniteDifferences.central_fdm(5, 1)

function _test_autograd_ref(p, x::AbstractVector{<:AbstractFloat}, fdm=DEFAULT_TEST_FDM)
    return (p.evaluator(x), FiniteDifferences.grad(fdm, p.evaluator, x)[1])
end

function _test_autograd_ref(p, values::NamedTuple, fdm=DEFAULT_TEST_FDM)
    _assert_namedtuple_shape(p.evaluator, values)
    x = flatten_to!!(nothing, values)
    f = x -> p.evaluator(unflatten_to!!(p.evaluator.inputspec, x))
    val = p.evaluator(values)
    grad = FiniteDifferences.grad(fdm, f, x)[1]
    return (val, unflatten_to!!(p.evaluator.inputspec, grad))
end

function _assert_test_autograd_matches(
    val_ad, grad_ad, val_fd, grad_fd; atol=1e-5, rtol=1e-5
)
    isapprox(val_ad, val_fd; atol=atol, rtol=rtol) || throw(
        ArgumentError(
            "Value mismatch against finite differences: got $val_ad, expected $val_fd."
        ),
    )
    isapprox(grad_ad, grad_fd; atol=atol, rtol=rtol) || throw(
        ArgumentError(
            "Gradient mismatch against finite differences with atol=$atol and rtol=$rtol.",
        ),
    )
    return nothing
end

function AbstractPPL.test_autograd(
    prepared, x::AbstractVector; atol=1e-5, rtol=1e-5, fdm=DEFAULT_TEST_FDM
)
    val_ad, grad_ad = AbstractPPL.value_and_gradient(prepared, x)
    val_fd, grad_fd = _test_autograd_ref(prepared, x, fdm)
    return _assert_test_autograd_matches(val_ad, grad_ad, val_fd, grad_fd; atol, rtol)
end

function AbstractPPL.test_autograd(
    prepared, values::NamedTuple; atol=1e-5, rtol=1e-5, fdm=DEFAULT_TEST_FDM
)
    val_ad, grad_ad = AbstractPPL.value_and_gradient(prepared, values)
    val_fd, grad_fd = _test_autograd_ref(prepared, values, fdm)
    return _assert_test_autograd_matches(val_ad, grad_ad, val_fd, grad_fd; atol, rtol)
end

struct FDPrepared{E,F,M}
    evaluator::E
    f::F
    fdm::M
end

AbstractPPL.capabilities(::Type{<:FDPrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::FDPrepared) = AbstractPPL.dimension(p.evaluator)

(p::FDPrepared)(x) = p.evaluator(x)

function AbstractPPL.prepare(
    adtype::AutoFiniteDifferences, problem, values::NamedTuple; check_dims::Bool=true
)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator{check_dims}(
        AbstractPPL.prepare(problem, values), values
    )
    f = x -> evaluator(unflatten_to!!(values, x))
    return FDPrepared(evaluator, f, adtype.fdm)
end

function AbstractPPL.prepare(
    adtype::AutoFiniteDifferences,
    problem,
    x::AbstractVector{<:AbstractFloat};
    check_dims::Bool=true,
)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(
        AbstractPPL.prepare(problem, x), length(x)
    )
    return FDPrepared(evaluator, evaluator, adtype.fdm)
end

function AbstractPPL.value_and_gradient(
    p::FDPrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator}, values::NamedTuple
)
    _assert_namedtuple_shape(p.evaluator, values)
    x = flatten_to!!(nothing, values)
    val = p.evaluator(values)
    grad = FiniteDifferences.grad(p.fdm, p.f, x)[1]
    return (val, unflatten_to!!(p.evaluator.inputspec, grad))
end

function AbstractPPL.value_and_gradient(
    p::FDPrepared{<:AbstractPPL.ADProblems.VectorEvaluator},
    x::AbstractVector{<:AbstractFloat},
)
    val = p.evaluator(x)
    grad = FiniteDifferences.grad(p.fdm, p.f, x)[1]
    return (val, grad)
end

end # module
