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
    DAG(inputs::NamedTuple{T})

Struct containing the adjacency matrix for a particular model and 
the topologically ordered vertex list.
"""
struct DAG
    A::SparseMatrixCSC
    sorted_vertex_list::Vector{Int64}
    
    function DAG(in::NamedTuple) 
        A = adjacency_matrix(in) 
        v_list = topological_sort_by_dfs(A)
        new(A, v_list)
    end
end

"""
    Model(nt::NamedTuple{T})

Model type and constructor that stores the `ModelState` and
`DAG` of the instantiated model. The constructor takes as an input
a named tuple of nodes and their value, input, eval and kind. 

# Examples

julia> nt = (
               s2 = (0.0, (), () -> InverseGamma(2.0,3.0), :Stochastic), 
               μ = (1.0, (), () -> 1.0, :Logical), 
               y = (0.0, (:μ, :s2), (μ, s2) -> MvNormal(μ, sqrt(s2)), :Stochastic)
           )
(s2 = (0.0, (), var"#33#36"(), :Stochastic), μ = (1.0, (), var"#34#37"(), :Logical), y = (0.0, (:μ, :s2), var"#35#38"(), :Stochastic))

julia> Model(nt)
Model(AbstractPPL.ModelState{(:s2, :μ, :y)}((s2 = 0.0, μ = 1.0, y = 0.0), (s2 = (), μ = (), y = (:μ, :s2)), (s2 = var"#33#36"(), μ = var"#34#37"(), y = var"#35#38"()), (s2 = :Stochastic, μ = :Logical, y = :Stochastic)), AbstractPPL.DAG(
  ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅ 
 1.0  1.0   ⋅ , [2, 1, 3]))
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

"""
    adjacency_matrix(inputs::NamedTuple)

For a NamedTuple{T} with edges `T` paired with tuples of input nodes,
`adjacency_matrix` constructs the adjacency matrix using the order 
of variables given by `T`. 

# Examples

julia> inputs = (a = (), b = (), c = (:a, :b))
(a = (), b = (), c = (:a, :b))

julia> AbstractPPL.adjacency_matrix(inputs)
3×3 SparseMatrixCSC{Float64, Int64} with 2 stored entries:
  ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅ 
 1.0  1.0   ⋅ 
"""
function adjacency_matrix(inputs::NamedTuple)
    N = length(inputs)
    nodes = keys(inputs)
    A = spzeros(N,N)
    for (i, node) in enumerate(nodes)
        v_inputs = inputs[node] # try as vector 
       if v_inputs != () 
            for inp in v_inputs
                ind = findall(x -> x == inp, nodes)
                A[i, ind[1]] = 1
            end
        end
    end
    A
end

adjacency_matrix(m::Model) = adjacency_matrix(m.ModelState.input)

function outneighbors(A::SparseMatrixCSC, u::T) where T <: Int
    #adapted from Graph.jl
    a = Array(A[:,u])
    findall(x->x==1, a)
end

function topological_sort_by_dfs(A)
    # lifted from Graphs.jl
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

# Examples

julia> m[@varname y]
(value = 0.0, input = (:μ, :s2), eval = var"#35#38"(), kind = :Stochastic)

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

# add docs 
# complete compat with abstractPPL api

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