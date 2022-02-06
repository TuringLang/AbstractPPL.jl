using AbstractPPL
using SparseArrays

## Example taken from Mamba
line = Dict{Symbol, Any}(
  :x => [1, 2, 3, 4, 5],
  :y => [1, 3, 3, 3, 5]
)

line[:xmat] = [ones(5) line[:x]]

modelwithargs = (
    β = (zeros(2), (), () -> MvNormal(2, sqrt(1000)), :Stochastic),
    xmat = (line[:xmat], (), () -> line[:xmat], :Logical), 
    s2 = (0.0, (), () -> InverseGamma(2.0,3.0), :Stochastic), 
    μ = (zeros(5), (:xmat, :β), (xmat, β) -> xmat * β, :Logical), 
    y = (zeros(5), (:μ, :s2), (μ, s2) -> MvNormal(μ, sqrt(s2)), :Stochastic)
)

modelwithoutargs = (
    β = (zeros(2), () -> MvNormal(2, sqrt(1000)), :Stochastic),
    xmat = (line[:xmat], () -> line[:xmat], :Logical), 
    s2 = (0.0, () -> InverseGamma(2.0,3.0), :Stochastic), 
    μ = (zeros(5), (xmat, β) -> xmat * β, :Logical), 
    y = (zeros(5), (μ, s2) -> MvNormal(μ, sqrt(s2)), :Stochastic)
)

m1 = Model(; zip(keys(modelwithoutargs), values(modelwithoutargs))...) # uses Model(; kwargs...) constructor
m2 = Model(modelwithargs)  # uses Model(nt::NamedTuple) constructor

@test typeof(m1) == Model
@test typeof(m2) == Model

for (i, j) in zip(keys(m1), keys(m2))
    @test i == j
    v1 = values(m1[i])
    v2 = values(m2[j])
    @test v1[[1, 2, 4]] == v2[[1, 2, 4]]
end

A = sparse([0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 1 1 0 0; 1 0 0 1 0])
@test dag(m1) == A == dag(m2)

@test length(m1) == length(modelwithargs) == 5
@test eltype(m1) == valtype(m1)

# test keys are VarNames
for key in keys(m1)
    @test typeof(key) <: VarName
end

# test Model constructor for model with single parent node
@test typeof(
        Model(
        μ = (1.0, () -> 3, :Logical), 
        y = (1.0, (μ) -> MvNormal(μ, sqrt(1)), :Stochastic)
        )
    ) == Model

# test ErrorException for parent node not being found
@test_throws ErrorException Model((
    μ = (zeros(5), (:β,), () -> 3, :Logical), 
    y = (zeros(5), (:μ,), (μ) -> MvNormal(μ, sqrt(1)), :Stochastic)
))