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
