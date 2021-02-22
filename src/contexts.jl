abstract type AbstractContext


"""
    struct JointContext <: AbstractContext end

The `JointContext` is used to evaluate a probabilistic program as a joint distribution over data
and parameters.
"""
struct JointContext <: AbstractContext end


"""
    struct PriorContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `PriorContext` is used to evaluate the prior distribution of a conditioned probabilistic model 
(i.e., the unconditional distribution of the parameters `vars`).
"""
struct PriorContext{Tvars} <: AbstractContext
    vars::Tvars
end
PriorContext() = PriorContext(nothing)


"""
    struct LikelihoodContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `LikelihoodContext` is used to evaluate the likelihood distribution of a conditioned
probabilistic model (i.e., the conditional distribution of the data given the the parameters).

If `vars` is `nothing`, the parameter values inside the `VarInfo` will be used by
default.
"""
struct LikelihoodContext{Tvars} <: AbstractContext
    vars::Tvars
end
LikelihoodContext() = LikelihoodContext(nothing)


"""
    struct MiniBatchContext{Tctx, T} <: AbstractContext
        ctx::Tctx
        scale::T
    end

The `MiniBatchContext` wraps another context to modify the joint log-density as `log(prior) + scale
* log(likelihood of a batch)` when running the model, where `scale` is typically the
number of data points divided by the batch size.  (Since this affects only the likelihood, it may be
applied to joint and likelihood contexts).

This is useful in batch-based stochastic gradient descent algorithms which optimize `log(prior) +
log(likelihood of all the data points)` in the expectation.
"""
struct MiniBatchContext{Tctx, T} <: AbstractContext
    ctx::Tctx
    scale::T
end
function MiniBatchContext(ctx = JointContext(); batch_size, npoints)
    return MiniBatchContext(ctx, npoints / batch_size)
end
