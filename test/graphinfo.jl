using AbstractPPL
import AbstractPPL.GraphPPL:GraphInfo, Model, get_dag, set_node_value!, 
                            get_node_value, get_sorted_vertices, get_node_eval,
                            get_nodekind, get_node_input
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
@test typeof(m) <: Model
sorted_vertices = get_sorted_vertices(m)
@test typeof(m) == Model{Tuple(sorted_vertices)}
@test typeof(m.g) <: GraphInfo <: AbstractModelTrace
@test typeof(m.g) == GraphInfo{Tuple(sorted_vertices)}

# test the dag is correct
A = sparse([0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 1 1 0 0; 1 0 0 1 0])
@test get_dag(m) == A

@test length(m) == 5
@test eltype(m) == valtype(m)

# check the values from the NamedTuple match the values in the fields of GraphInfo
vals, evals, kinds = AbstractPPL.GraphPPL.getvals(model[Tuple(sorted_vertices)])
inputs = (s2 = (), xmat = (), β = (), μ = (:xmat, :β), y = (:μ, :s2))

for (i, vn) in enumerate(keys(m))
    @test vn isa VarName
    @test get_node_value(m, vn) == vals[i]
    @test get_node_eval(m, vn) == evals[i]
    @test get_nodekind(m, vn) == kinds[i]
    @test get_node_input(m, vn) == inputs[i]
end

for node in m 
    @test typeof(node) <: NamedTuple{fieldnames(GraphInfo)[1:4]}
end

# test Model constructor for model with single parent node
single_parent_m = Model(μ = (1.0, () -> 3, :Logical), y = (1.0, (μ) -> MvNormal(μ, sqrt(1)), :Stochastic))
@test typeof(single_parent_m) == Model{(:μ, :y)}
@test typeof(single_parent_m.g) == GraphInfo{(:μ, :y)}

# test setindex

@test_throws AssertionError set_node_value!(m, @varname(s2), [0.0])
@test_throws AssertionError set_node_value!(m, @varname(s2), (1.0,))
set_node_value!(m, @varname(s2), 1.0)
@test get_node_value(m, @varname s2) == 1.0

# test ErrorException for parent node not found
@test_throws ErrorException Model( μ = (1.0, (β) -> 3, :Logical), y = (1.0, (μ) -> MvNormal(μ, sqrt(1)), :Stochastic))

# test AssertionError thrown for kwargs with the wrong order of inputs
@test_throws AssertionError Model( μ = ((β) -> 3, 1.0, :Logical), y = (1.0, (μ) -> MvNormal(μ, sqrt(1)), :Stochastic))