using AbstractPPL

using Distributions
## Example
test = (
    a = ([0,1], (), () -> MvNormal(zeros(2), 1), :Stochastic), 
    b = (0, (), () -> 42, :Logical), 
    c = (0, (:a, :b), (a, b) -> MvNormal(a, 2sqrt(b)), :Stochastic) # consider a :Data type/kind? 
)
# Make example GLM
# add :Data type to support condition/decondition

# test = (
#     a1 = ([0,1], (), 1), :HyperParameter), 
#     a = ([], (), (a1) -> MvNormal(zeros(2), a1), :Stochastic), 
#     b = (0, (), () -> 42, :Logical), 
#     c = (0, (:a, :b), (a, b) -> MvNormal(a, 2sqrt(b)), :Stochastic) # consider a :Data type/kind? 
# )

m = Model(test)

m.Data.value == (a = [0, 1], b = 0, c = 0)
m.Data.input == (a = (), b = (), c = (:a, :b))
m.Data.kind == (a = :Stochastic, b = :Logical, c = :Stochastic)

m[@varname(a)]