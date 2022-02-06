using AbstractPPL
using SparseArrays

## Example taken from Mamba
line = Dict{Symbol, Any}(
  :x => [1, 2, 3, 4, 5],
  :y => [1, 3, 3, 3, 5]
)
line[:xmat] = [ones(5) line[:x]]

# just making it a NamedTuple so that the values can be tested later. Constructor should be used as Model(;kwargs...).
model = (
    β = (zeros(2), () -> MvNormal(2, sqrt(1000)), :Stochastic),
    xmat = (line[:xmat], () -> line[:xmat], :Logical), 
    s2 = (0.0, () -> InverseGamma(2.0,3.0), :Stochastic), 
    μ = (zeros(5), (xmat, β) -> xmat * β, :Logical), 
    y = (zeros(5), (μ, s2) -> MvNormal(μ, sqrt(s2)), :Stochastic)
)

# construct the model!
m = Model(; zip(keys(model), values(model))...) # uses Model(; kwargs...) constructor

# test the type of the model is correct
@test typeof(m) <: GraphInfo <: AbstractModelTrace
@test typeof(m) == GraphInfo{(:s2, :xmat, :β, :μ, :y)}

# test the dag is correct
A = sparse([0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 1 1 0 0; 1 0 0 1 0])
@test dag(m) == A

@test length(m) == 5
@test eltype(m) == valtype(m)

# check the values from the NamedTuple match the values in the fields of GraphInfo
vals = AbstractPPL.getvals(model)
for (i, field) in enumerate([:value, :eval, :kind])
    @test eval( :(values(m.$field) == vals[$i]) )
end

# test the right inputs have been inferred 
@test m.input == (s2 = (), xmat = (), β = (), μ = (:xmat, :β), y = (:μ, :s2))

# test keys are VarNames
for key in keys(m)
    @test typeof(key) <: VarName
end


# test Model constructor for model with single parent node
@test typeof(
        Model(μ = (1.0, () -> 3, :Logical), y = (1.0, (μ) -> MvNormal(μ, sqrt(1)), :Stochastic))
    ) == GraphInfo{(:μ, :y)}

# test ErrorException for parent node not found
@test_throws ErrorException Model( μ = (1.0, (β) -> 3, :Logical), y = (1.0, (μ) -> MvNormal(μ, sqrt(1)), :Stochastic))