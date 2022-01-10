using AbstractPPL
import Base.getindex
using SparseArrays
using Setfield
using Setfield: PropertyLens, get

struct Data{T}
    value::NamedTuple{T}
    input::NamedTuple{T}
    eval::NamedTuple{T}
    kind::NamedTuple{T}
end

struct DAG
    A::SparseMatrixCSC
    sorted_vertex_list::Vector{Int64}
    
    function DAG(in::NamedTuple) 
        A = adjacency_matrix(in) 
        v_list = topological_sort_by_dfs(A)
        new(A, v_list)
    end
end

struct Model
    Data::Data
    DAG::DAG

    Model(value, input, eval, kind) = new(Data(value, input, eval, kind), DAG(input))
end

function Model(nt::T) where T <: NamedTuple
    ks = keys(nt)
    vals = [nt[k][i] for k in ks, i in 1:4] # create matrix of tuple values
    m = [(; zip(ks, vals[:,i])...) for i in 1:4] # zip each colum of vals with keys
    Model(m...)
end # look into generated functions

@generated function Base.getindex(m::Data, vn::VarName{p}) where {p}
        fns = fieldnames(Data)
        name_lens = Setfield.PropertyLens{p}()
        field_lenses = [Setfield.PropertyLens{f}() for f in fns]
        values = [:(get(m, Setfield.compose($l, $name_lens, getlens(vn)))) for l in field_lenses]
        return :(NamedTuple{$(fns)}(($(values...),)))

end

t = eval(:(NamedTuple{(:a, )}(1.0)))

## Functions needed for type constructors
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

adjacency_matrix(m::Model) = adjacency_matrix(m.Data.input)


function outneighbors(A::SparseMatrixCSC, u::T) where T <: Int
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


## indexing functions using Setfield
@generated function Base.getindex(m::Data, vn::VarName{p}) where {p}
    fns = fieldnames(Data)
    name_lens = Setfield.PropertyLens{p}()
    field_lenses = [Setfield.PropertyLens{f}() for f in fns]
    values = [:(get(m, Setfield.compose($l, $name_lens, getlens(vn)))) for l in field_lenses]
    return :(NamedTuple{$(fns)}(($(values...),)))
end

function Base.getindex(m::Model, vn::VarName)
    getindex(m.Data, vn)
end

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

ks = keys(test)
vals = [test[k][i] for k in ks, i in 1:4] # create matrix of tuple values
m = [(; zip(ks, vals[:,i])...) for i in 1:4] # zip each colum of vals with keys

@code_warntype Model(m...)
@code_warntype Model(test)

m = Model(test)

m.Data.value == (a = [0, 1], b = 0, c = 0)
m.Data.input == (a = (), b = (), c = (:a, :b))
m.Data.kind == (a = :Stochastic, b = :Logical, c = :Stochastic)

m[@varname(a)]

# # General eval function
# function evalf(f::Function, m::Model)
#     nodes = m.DAG.sorted_vertex_list
#     symlist = keys(m.Data.input)
#     vals = (;)
#     for (i, n) in enumerate(nodes)
#         node = symlist[n]
#         input_nodes = m.Data.input[node]
#         if m.Data.kind[node] == :Stochastic
#             if length(input_nodes) == 0
#                 vals = merge(vals, [node=>f(m.Data.eval[node]())])                
#             elseif length(input_nodes) > 0 
#                 inputs = [vals[n] for n in input_nodes]
#                 vals = merge(vals, [node=>f(m.Data.eval[node](inputs...))])
#             end
#         else
#             vals = merge(vals, [node=>m.Data.eval[node]()])
#         end
#     end
#     vals
# end
