using AbstractPPL
using Distributions # just added for testing
import Base.getindex
using SparseArrays
using Setfield
using Setfield: PropertyLens, get

mutable struct Model
    value
    input
    eval
    kind
    dag

    Model(value, input, eval, kind) = new(value, input, eval, kind, DAG(input))
end

mutable struct DAG
    A::SparseMatrixCSC
    sorted_vertex_list::Vector{Int64}
    
    function DAG(in::NamedTuple) 
        A = adjacency_matrix(in) 
        v_list = topological_sort_by_dfs(A)
        new(A, v_list)
    end
end

function adjacency_matrix(inputs::NamedTuple)
    N = length(inputs)
    nodes = keys(inputs)
    A = spzeros(N,N)
    for (i, node) in enumerate(nodes)
        if inputs[node] != ()
            inputs = inputs[node]
            for inp in inputs
                ind = findall(x -> x == inp, nodes)[1]
                A[i, ind] = 1
            end
        end
    end
    A
end

adjacency_matrix(m::Model) = adjacency_matrix(m.input)

function Model(nt::T) where T <: NamedTuple
    ks = keys(nt)
    vals = [nt[k][i] for k in ks, i in 1:4] # create matrix of tuple values
    m = [(; zip(ks, vals[:,i])...) for i in 1:4] # zip each colum of vals with keys
    Model(m[1], m[2], m[3], m[4])
end

function Base.getindex(m::Model, vn::VarName)
    fns = fieldnames(typeof(m))[1:end-1]
    vals = [get_property(m, i, vn) for i in fns]
    (;zip(fns, vals)...)
end

function get_property(m::Model, field::Symbol, vn::VarName)
    get(m, PropertyLens{field}() ∘ PropertyLens{getsym(vn)}())
end


function outneighbors(A::SparseMatrixCSC, u::T) where T <: Int
    a = Array(A[:,u])
    findall(x->x==1, a)
end

function topological_sort_by_dfs(A)
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

function evalf(m::Model, f::Function)
    nodes = m.dag.sorted_vertex_list
    symlist = keys(m.input)
    vals = (;)
    for (i, n) in enumerate(nodes)
        node = symlist[n]
        input_nodes = m.input[node]
        if m.kind[node] == :Stochastic
            if length(input_nodes) == 0
                vals = merge(vals, [node=>f(m.eval[node]())])                
            elseif length(input_nodes) > 0 
                inputs = [vals[n] for n in input_nodes]
                vals = merge(vals, [node=>f(m.eval[node](inputs...))])
            end
        else
            vals = merge(vals, [node=>m.eval[node]()])
        end
    end
    vals
end

test = (
    a = ([0,1], (), () -> MvNormal(zeros(2), 1), :Stochastic), # should this explictly call rand(Normal())?
    b = (0, (), () -> 42, :Logical), 
    c = (0, (:a, :b), (a, b) -> MvNormal(a, 2sqrt(b)), :Stochastic)
)

m = Model(test)

m.value == (a = [0, 1], b = 0, c = 0)
m.input == (a = (), b = (), c = (:a, :b))
#m.eval == (a = () -> D(), b = () -> 42, c = (a, b) -> Normal(a, 2sqrt(b)))
m.kind == (a = :Stochastic, b = :Logical, c = :Stochastic)


m[@varname(a)]

evalf(m, rand)

#m[@varname(a)] == (value = [0, 1], input = (), eval = ..., kind = Stochastic)
# @varname(a)::VarName{:a, IdentityLens}

# @generated function gi(m::NamedTuple{fns}, vn::VarName{p}) where {fns, p}
#     name_lens = Setfield.PropertyLens{p}()
#     field_lenses = [Setfield.PropertyLens{f}() for f in fns]
#     values = [:(get(m, Setfield.compose($l, $name_lens, getlens(vn)))) for l in field_lenses]
#     return :(NamedTuple{fns}(($(values...),)))
# end

# gi(m.value, @varname(a))