"""
    AbstractModel

Common base type for models expressed as probabilistic programs.
"""
abstract type AbstractModel end


"""
    logprior(model, varinfo)

Return the log prior probability of variables `varinfo` for the probabilistic `model`.

See also [`logjoint`](@ref) and [`loglikelihood`](@ref).
"""
function logprior end


"""
    logjoint(model, varinfo)

Return the log joint probability of variables `varinfo` for the probabilistic `model`.

See [`logjoint`](@ref) and [`loglikelihood`](@ref).
"""
function logjoint end


"""
    condition(model, prob_expr)

## Examples

```
julia> condition(m, @P X | (; Y = y))
```
"""
#function query end
