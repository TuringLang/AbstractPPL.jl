using AbstractPPL
import Base.getindex
using SparseArrays
using Setfield
using Setfield: PropertyLens, get

"""
    GraphInfo

Record the state of the model as a struct of NamedTuples, all
sharing the same key values, namely, those of the model parameters.
`value` should store the initial/current value of the parameters.
`input` stores a tuple of inputs for a given node. `eval` are the
anonymous functions associated with each node. These might typically
be either deterministic values or some distribution, but could an 
arbitrary julia program. `kind` is a tuple of symbols indicating
whether the node is a logical or stochastic node. Additionally, the 
adjacency matrix and topologically ordered vertex list and stored.

GraphInfo is instantiated using the `Model` constctor. 
"""

struct GraphInfo{T} <: AbstractModelTrace
    value::NamedTuple{T}
    input::NamedTuple{T}
    eval::NamedTuple{T}
    kind::NamedTuple{T}
    A::SparseMatrixCSC
    sorted_vertices::Vector{Symbol}
end

"""
    Model(;kwargs...)

`Model` type constructor that takes in named arguments for 
nodes and returns a `Model`. Nodes are pairs of variable names
and tuples containing default value, an eval function 
and node type. The inputs of each node are inferred from 
their anonymous functions. The returned object has a type 
GraphInfo{(sorted_vertices...)}.

# Examples
```jl-doctest
julia> using AbstractPPL

julia> Model(
               s2 = (0.0, () -> InverseGamma(2.0,3.0), :Stochastic), 
               μ = (1.0, () -> 1.0, :Logical), 
               y = (0.0, (μ, s2) -> MvNormal(μ, sqrt(s2)), :Stochastic)
           )
Nodes: 
μ = (value = 1.0, input = (), eval = var"#6#9"(), kind = :Logical)
s2 = (value = 0.0, input = (), eval = var"#5#8"(), kind = :Stochastic)
y = (value = 0.0, input = (:μ, :s2), eval = var"#7#10"(), kind = :Stochastic)
```
"""

struct Model{T}
    g::GraphInfo{T}
end

function Model(;kwargs...)
    vals = getvals(NamedTuple(kwargs))
    args = [argnames(f) for f in vals[2]]
    A, sorted_vertices = DAG(NamedTuple{keys(kwargs)}(args))    
    modelinputs = NamedTuple{Tuple(sorted_vertices)}.([vals[1], Tuple.(args), vals[2], vals[3]])
    Model(GraphInfo(modelinputs..., A, sorted_vertices))
end


"""
    DAG(inputs)

Function taking in a NamedTuple containing the inputs to each node 
and returns the implied adjacency matrix and topologically ordered 
vertex list.
"""
function DAG(inputs)
    input_names = Symbol[keys(inputs)...]
    println(inputs)
    A = adjacency_matrix(inputs) 
    sorted_vertices = topological_sort_by_dfs(A)
    sorted_A = permute(A, collect(1:length(inputs)), sorted_vertices)
    sorted_A, input_names[sorted_vertices]
end

"""
    getvals(nt::NamedTuple{T})

Takes in the arguments to Model(;kwargs...) as a NamedTuple and 
reorders into a tuple of tuples each containing either of value, 
input, eval and kind, as required by the GraphInfo type. 
"""
@generated function getvals(nt::NamedTuple{T}) where T
    values = [:(nt[$i][$j]) for i in 1:length(T), j in 1:3]
    m = [:($(values[:,i]...), ) for i in 1:3]
    return :($(m...),)
end

"""
    argnames(f::Function)

Returns a Vector{Symbol} of the inputs to an anonymous function `f`.
"""
argnames(f::Function) = Base.method_argnames(first(methods(f)))[2:end]

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
function adjacency_matrix(inputs::NamedTuple{nodes}) where {nodes}
    N = length(inputs)
    col_inds = NamedTuple{nodes}(ntuple(identity, N))
    A = spzeros(Bool, N, N)
    for (row, node) in enumerate(nodes)
        for input in inputs[node]
            if input ∉ nodes
                error("Parent node of $(input) not found in node set: $(nodes)")
            end
            col = col_inds[input]
            A[row, col] = true
        end
    end
    return A
end

function outneighbors(A::SparseMatrixCSC, u::T) where T <: Int
    #adapted from Graph.jl https://github.com/JuliaGraphs/Graphs.jl/blob/06669054ed470bcfe4b2ad90ed974f2e65c84bb6/src/interface.jl#L302
    inds, _ = findnz(A[:, u])
    inds
end

function topological_sort_by_dfs(A)
    # lifted from Graphs.jl https://github.com/JuliaGraphs/Graphs.jl/blob/06669054ed470bcfe4b2ad90ed974f2e65c84bb6/src/traversals/dfs.jl#L44
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
julia> using AbstractPPL

julia> m = Model( s2 = (0.0, () -> InverseGamma(2.0,3.0), :Stochastic), 
                   μ = (1.0, () -> 1.0, :Logical), 
                   y = (0.0, (μ, s2) -> MvNormal(μ, sqrt(s2)), :Stochastic))
(s2 = Symbol[], μ = Symbol[], y = [:μ, :s2])
Nodes: 
μ = (value = 0.0, input = (), eval = var"#43#46"(), kind = :Stochastic)
s2 = (value = 1.0, input = (), eval = var"#44#47"(), kind = :Logical)
y = (value = 0.0, input = (:μ, :s2), eval = var"#45#48"(), kind = :Stochastic)


julia> m[@varname y]
(value = 0.0, input = (:μ, :s2), eval = var"#45#48"(), kind = :Stochastic)
```
"""
@generated function Base.getindex(g::GraphInfo, vn::VarName{p}) where {p}
    fns = fieldnames(GraphInfo)[1:4]
    name_lens = Setfield.PropertyLens{p}()
    field_lenses = [Setfield.PropertyLens{f}() for f in fns]
    values = [:(get(g, Setfield.compose($l, $name_lens, getlens(vn)))) for l in field_lenses]
    return :(NamedTuple{$(fns)}(($(values...),)))
end

function Base.getindex(m::Model, vn::VarName)
    return m.g[vn]
end

function Base.show(io::IO, m::Model)
    print(io, "Nodes: \n")
    for node in nodes(m)
        print(io, "$node = ", m[VarName{node}()], "\n")
    end
end


function Base.iterate(m::Model, state=1)
    state > length(nodes(m)) ? nothing : (m[VarName{m.g.sorted_vertices[state]}()], state+1)
end

Base.eltype(m::Model) = NamedTuple{fieldnames(GraphInfo)[1:4]}
Base.IteratorEltype(m::Model) = HasEltype()

Base.keys(m::Model) = (VarName{n}() for n in m.g.sorted_vertices)
Base.values(m::Model) = Base.Generator(identity, m)
Base.length(m::Model) = length(nodes(m))
Base.keytype(m::Model) = eltype(keys(m))
Base.valtype(m::Model) = eltype(m)


"""
    dag(m::Model)

Returns the adjacency matrix of the model as a SparseArray.
"""
dag(m::Model) = m.g.A

"""
    nodes(m::Model)

Returns a `Vector{Symbol}` containing the sorted vertices 
of the DAG. 
"""
nodes(m::Model) = m.g.sorted_vertices