using AbstractPPL
import Base.getindex
using SparseArrays
using Setfield
using Setfield: PropertyLens, get

"""
    ModelState(value::NamedTuple{T}, input::NamedTuple{T}, eval::NamedTuple{T}, kind::NamedTuple{T})

Record the state of the model as a struct of NamedTuples, all
sharing the same key values, namely, those of the model parameters.
`value` should store the initial/current value of the parameters.
`input` stores a tuple of inputs for a given node. `eval` are the
anonymous functions associated with each node. These might typically
be either deterministic values or some distribution, but could an 
arbitrary julia program. `kind` is a tuple of symbols indicating
whether the node is a logical or stochastic node.
"""
struct ModelState{T}
    value::NamedTuple{T}
    input::NamedTuple{T}
    eval::NamedTuple{T}
    kind::NamedTuple{T}
end

"""
    DAG(inputs)

Struct containing the adjacency matrix for a particular model and 
the topologically ordered vertex list.
"""
struct DAG
    A::SparseMatrixCSC
    sorted_vertices::Vector{Symbol}
end

function DAG(inputs) 
    input_names = keys(inputs)
    A = adjacency_matrix(inputs) 
    sorted_vertices = topological_sort_by_dfs(A)
    sorted_A = permute(A, collect(1:length(inputs)), sorted_vertices)
    DAG(sorted_A, collect(input_names[sorted_vertices]))
end

"""
    Model(nt::NamedTuple{T})

Model type and constructor that stores the `ModelState` and
`DAG` of the instantiated model. The constructor takes as an input
a named tuple of nodes and their value, input, eval and kind. 

# Examples
```jl-doctest
julia> Model(
               s2 = (0.0, (), () -> InverseGamma(2.0,3.0), :Stochastic), 
               μ = (1.0, (), () -> 1.0, :Logical), 
               y = (0.0, (:μ, :s2), (μ, s2) -> MvNormal(μ, sqrt(s2)), :Stochastic)
           )
Nodes: 
μ = (value = 1.0, input = (), eval = var"#241#244"(), kind = :Logical)
s2 = (value = 0.0, input = (), eval = var"#240#243"(), kind = :Stochastic)
y = (value = 0.0, input = (:μ, :s2), eval = var"#242#245"(), kind = :Stochastic)
DAG: 
3×3 SparseArrays.SparseMatrixCSC{Bool, Int64} with 2 stored entries:
⋅  ⋅  ⋅
⋅  ⋅  ⋅
1  1  ⋅
```
"""
struct Model
    ModelState::ModelState
    DAG::DAG

    Model(value, input, eval, kind) = new(ModelState(value, input, eval, kind), DAG(input))
end

@generated function Model(nt::NamedTuple{T}) where T
    values = [:(nt[$i][$j]) for i in 1:length(T), j in 1:4]
    m = [:(NamedTuple{T}(($(values[:,i]...), ))) for i in 1:4]
    return :(Model(($(m...),)...))
end

function Model(;kwargs...)
    Model(values(kwargs))
end

function Base.show(io::IO, m::Model)
    print(io, "Nodes: \n")
    for node in nodes(m)
        print(io, "$node = ", m[VarName{node}()], "\n")
    end
    print(io, "DAG: \n") 
    display(m.DAG.A)
end


"""
    adjacency_matrix(inputs)

For a NamedTuple{T} with vertices `T` paired with tuples of input nodes,
`adjacency_matrix` constructs the adjacency matrix using the order 
of variables given by `T`. 

# Examples
```jl-doctest
julia> inputs = (a = (), b = (), c = (:a, :b))
(a = (), b = (), c = (:a, :b))

julia> AbstractPPL.adjacency_matrix(inputs)
3×3 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
  ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅ 
 1.0  1.0   ⋅
``` 
"""
function adjacency_matrix(inputs)
    N = length(inputs)
    nodes = keys(inputs)
    A = spzeros(Bool, N, N)
    for (row_n, node) in enumerate(nodes)
        v_inputs = inputs[node]
        setinput!(A, row_n, nodes, v_inputs)
    end
    A
end

function setinput!(A::SparseMatrixCSC{Bool, Int64}, row_n, nodes, v_inputs::Symbol)
    ind = findfirst(==(v_inputs), nodes)
    A[row_n, ind] = true
end

function setinput!(A::SparseMatrixCSC{Bool, Int64}, row_n, nodes, v_inputs)
    for inp in v_inputs
        ind = findfirst(==(inp), nodes)
        A[row_n, ind] = true
    end
    A
end

adjacency_matrix(m::Model) = adjacency_matrix(m.ModelState.input)

function outneighbors(A::SparseMatrixCSC, u::T) where T <: Int
    #adapted from Graph.jl
    inds, _ = findnz(A[:, u])
    inds
end

function topological_sort_by_dfs(A)
    # lifted from Graphs.jl
    # Depth first search implementation optimized from http://www.cs.nott.ac.uk/~psznza/G5BADS03/graphs2.pdf
    n_verts = size(A)[1]
    vcolor = zeros(UInt8, n_verts)
    verts = Vector{Int64}()
    for v in 1:n_verts
        vcolor[v] != 0 && continue
        S = Vector{Int64}([v])
        vcolor[v] = 1
        while !isempty(S)
            u = S[end]
            w = 0
            for n in outneighbors(A, u)
                if vcolor[n] == 1
                    error("The input graph contains at least one loop.") # TODO 0.7 should we use a different error?
                elseif vcolor[n] == 0
                    w = n
                    break
                end
            end
            if w != 0
                vcolor[w] = 1
                push!(S, w)
            else
                vcolor[u] = 2
                push!(verts, u)
                pop!(S)
            end
        end
    end
    return reverse(verts)
end

"""
    Base.getindex(m::Model, vn::VarName{p})

Index a Model with a `VarName{p}` lens. Retrieves the `value``, `input`,
`eval` and `kind` for node `p`.

# Examples

```jl-doctest 
# add a model 

julia> m[@varname y]
(value = 0.0, input = (:μ, :s2), eval = var"#35#38"(), kind = :Stochastic)

```
"""
@generated function Base.getindex(m::ModelState, vn::VarName{p}) where {p}
    fns = fieldnames(ModelState)
    name_lens = Setfield.PropertyLens{p}()
    field_lenses = [Setfield.PropertyLens{f}() for f in fns]
    values = [:(get(m, Setfield.compose($l, $name_lens, getlens(vn)))) for l in field_lenses]
    return :(NamedTuple{$(fns)}(($(values...),)))
end

function Base.getindex(m::Model, vn::VarName)
    getindex(m.ModelState, vn)
end

function Base.iterate(m::Model, state=1)
    state > length(nodes(m)) ? nothing : (m[VarName{m.DAG.sorted_vertices[state]}()], state+1)
end

Base.eltype(m::Model) = NamedTuple{fieldnames(ModelState)}
Base.IteratorEltype(m::Model) = HasEltype()

Base.keys(m::Model) = (VarName{n}() for n in m.DAG.sorted_vertices)
Base.values(m::Model) = Base.Generator(identity, m)
Base.length(m::Model) = length(nodes(m))
Base.keytype(m::Model) = eltype(keys(m))
Base.valtype(m::Model) = eltype(m)

"""
    dag(m::Model)

Returns the adjacency matrix of the model as a SparseArray.
"""
dag(m::Model) = m.DAG.A

"""
    nodes(m::Model)

Returns a `Vector{Symbol}` containing the sorted vertices 
of the DAG. 
"""
nodes(m::Model) = m.DAG.sorted_vertices

# # General eval function
# function evalf(f::Function, m::Model)
#     nodes = m.DAG.sorted_vertex_list
#     symlist = keys(m.ModelState.input)
#     vals = (;)
#     for (i, n) in enumerate(nodes)
#         node = symlist[n]
#         input_nodes = m.ModelState.input[node]
#         if m.ModelState.kind[node] == :Stochastic
#             if length(input_nodes) == 0
#                 vals = merge(vals, [node=>f(m.ModelState.eval[node]())])                
#             elseif length(input_nodes) > 0 
#                 inputs = [vals[n] for n in input_nodes]
#                 vals = merge(vals, [node=>f(m.ModelState.eval[node](inputs...))])
#             end
#         else
#             vals = merge(vals, [node=>m.ModelState.eval[node]()])
#         end
#     end
#     vals
# end