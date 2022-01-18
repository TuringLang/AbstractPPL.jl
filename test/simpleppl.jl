using AbstractPPL
using SparseArrays

## Example
line = Dict{Symbol, Any}(
  :x => [1, 2, 3, 4, 5],
  :y => [1, 3, 3, 3, 5]
)

line[:xmat] = [ones(5) line[:x]]

model = (
    β = (zeros(2), (), () -> MvNormal(2, sqrt(1000)), :Stochastic),
    xmat = (line[:xmat], (), () -> line[:xmat], :Logical), 
    s2 = (0.0, (), () -> InverseGamma(2.0,3.0), :Stochastic), 
    μ = (zeros(5), (:xmat, :β), (xmat, β) -> xmat * β, :Logical), 
    y = (zeros(5), (:μ, :s2), (μ, s2) -> MvNormal(μ, sqrt(s2)), :Stochastic)
)

# add :Data type to support condition/decondition
# condition(model::Model, params::NamedTuple)

m = Model(model)
@test typeof(m) == Model

dag = sparse([0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 1 1 0 0; 1 0 0 1 0])
@test m.DAG.A == dag

for key in keys(model)
    @test values(m[VarName{key}()]) == model[key]
end
