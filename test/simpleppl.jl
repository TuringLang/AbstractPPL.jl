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

# m1 = Model(
#     #β = (zeros(2), (), () -> MvNormal(2, sqrt(1000)), :Stochastic),
#     μ = (zeros(5), (), () -> 3, :Logical), 
#     y = (zeros(5), (:μ), (μ) -> MvNormal(μ, sqrt(1)), :Stochastic)
# )

m = Model(; zip(keys(model), values(model))...) # uses Model(; kwargs...) constructor
m1 = Model(model)  # uses Model(nt::NamedTuple) constructor

@test typeof(m) == Model
@test typeof(m1) == Model

dag = sparse([0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 1 1 0 0; 1 0 0 1 0])
@test m.DAG.A == dag

@test length(m) == length(model) == 5
@test eltype(m) == valtype(m)

ks = keys(model)[[3, 2, 1, 4, 5]] # reorder model keys to match topologically ordered Model

for (key, vs) in zip(ks, values(m))
    @test m[VarName{key}()] == vs
    @test values(m[VarName{key}()]) == model[key] == values(vs)
end

for key in keys(m)
    @test typeof(key) <: VarName
end

for (i, node) in enumerate(m)
    @test values(node) == values(model[ks[i]])
end