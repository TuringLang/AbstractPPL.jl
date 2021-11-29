# Adding interface for DAG representation using parts of SimplePPL/Mamba.jl
using Graphs
using Distributions


ElementOrVector{T} = Union{T,Vector{T}}

#################### Variate Types ####################

abstract type ScalarVariate <: Real end
abstract type ArrayVariate{N} <: DenseArray{Float64,N} end

const AbstractVariate = Union{ScalarVariate,ArrayVariate}
const VectorVariate = ArrayVariate{1}
const MatrixVariate = ArrayVariate{2}


#################### Distribution Types ####################

const DistributionStruct =
    Union{Distribution,Array{UnivariateDistribution},Array{MultivariateDistribution}}

#################### Null Distribution ####################

struct NullUnivariateDistribution <: UnivariateDistribution{ValueSupport} end


#################### Dependent Types ####################

mutable struct ScalarLogical <: ScalarVariate
    value::Float64
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
end

mutable struct ArrayLogical{N} <: ArrayVariate{N}
    value::Array{Float64,N}
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
end

mutable struct ScalarStochastic <: ScalarVariate
    value::Float64
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
    distr::UnivariateDistribution
end

mutable struct ArrayStochastic{N} <: ArrayVariate{N}
    value::Array{Float64,N}
    symbol::Symbol
    monitor::Vector{Int}
    eval::Function
    sources::Vector{Symbol}
    targets::Vector{Symbol}
    distr::DistributionStruct
end

const AbstractLogical = Union{ScalarLogical,ArrayLogical}
const AbstractStochastic = Union{ScalarStochastic,ArrayStochastic}
const AbstractDependent = Union{AbstractLogical,AbstractStochastic}

#################### Variate ####################

#################### Conversions ####################

Base.convert(::Type{Bool}, v::ScalarVariate) = convert(Bool, v.value)
Base.convert(::Type{T}, v::ScalarVariate) where {T<:Integer} = convert(T, v.value)
Base.convert(::Type{T}, v::ScalarVariate) where {T<:AbstractFloat} = convert(T, v.value)
Base.AbstractFloat(v::ScalarVariate) = convert(Float64, v)
Base.Float64(v::ScalarVariate) = convert(Float64, v)

Base.convert(::Type{Matrix}, v::MatrixVariate) = v.value
Base.convert(::Type{Vector}, v::VectorVariate) = v.value
Base.convert(
    ::Union{Type{Array{T}},Type{Array{T,N}}},
    v::ArrayVariate{N},
) where {T<:Real,N} = convert(Array{T,N}, v.value)

Base.unsafe_convert(::Type{Ptr{Float64}}, v::ArrayVariate) = pointer(v.value)


macro promote_scalarvariate(V)
    quote
        Base.promote_rule(::Type{$(esc(V))}, ::Type{T}) where {T<:Real} = Float64
    end
end


#################### Base Functions ####################

Base.size(v::AbstractVariate) = size(v.value)

Base.stride(v::ArrayVariate, k::Int) = stride(v.value, k)


#################### Indexing ####################

Base.getindex(v::ScalarVariate, ind::Int) = v.value[ind]

Base.getindex(v::ScalarVariate, inds::Union{StepRange{Int,Int},Vector{Int}}) =
    Float64[v[i] for i in inds]

Base.getindex(v::ArrayVariate, inds::Int...) = getindex(v.value, inds...)


Base.setindex!(v::ScalarVariate, x::Real, ind::Int) = (v.value = x[ind])

function Base.setindex!(
    v::ScalarVariate,
    x::Vector{T},
    inds::Union{StepRange{Int,Int},Vector{Int}},
) where {T<:Real}
    nx = length(x)
    ninds = length(inds)
    nx == ninds ||
        throw(DimensionMismatch("tried to assign $nx elements to $ninds destinations"))

    for i = 1:nx
        v[inds[i]] = x[i]
    end
end

Base.setindex!(v::ArrayVariate, x, inds::Int...) = setindex!(v.value, x, inds...)


#################### I/O ####################

function Base.show(io::IO, v::AbstractVariate)
    print(io, "Object of type \"$(summary(v))\"\n")
    show(io, v.value)
end

#################### Auxiliary Functions ####################

function names(v::ScalarVariate, prefix)
    AbstractString[string(prefix)]
end

function names(v::ArrayVariate, prefix)
    offset = ndims(v) > 1 ? 1 : 2
    values = similar(v.value, AbstractString)
    for i = 1:length(v)
        s = string(ind2sub(size(v), i))
        values[i] = string(prefix, "[", s[2:(end-offset)], "]")
    end
    values
end


#################### Mathematical Operators ####################

const BinaryScalarMethods = [
    :(Base.:+),
    :(Base.:-),
    :(Base.:*),
    :(Base.:/),
    :(Base.:\),
    :(Base.:^),
    :(Base.:(==)),
    :(Base.:(!=)),
    :(Base.:<),
    :(Base.:(<=)),
    :(Base.:>),
    :(Base.:(>=)),
    :(Base.cld),
    :(Base.div),
    :(Base.divrem),
    :(Base.fld),
    :(Base.mod),
    :(Base.rem),
]

for op in BinaryScalarMethods
    @eval ($op)(x::ScalarVariate, y::ScalarVariate) = ($op)(x.value, y.value)
end

const RoundScalarMethods = [:(Base.ceil), :(Base.floor), :(Base.round), :(Base.trunc)]

for op in RoundScalarMethods
    @eval ($op)(x::ScalarVariate) = ($op)(x.value)
    @eval ($op)(::Type{T}, x::ScalarVariate) where {T} = ($op)(T, x.value)
end

const UnaryScalarMethods = [
    :(Base.:+),
    :(Base.:-),
    :(Base.abs),
    :(Base.isfinite),
    :(Base.isinf),
    :(Base.isinteger),
    :(Base.isnan),
    :(Base.mod2pi),
    :(Base.one),
    :(Base.sign),
    :(Base.zero),
]

for op in UnaryScalarMethods
    @eval ($op)(x::ScalarVariate) = ($op)(x.value)
end

  
#################### Model Types ####################

struct ModelGraph
    graph::DiGraph
    keys::Vector{Symbol}
end

mutable struct Model
    nodes::Dict{Symbol,Any}
    hasinputs::Bool
    hasinits::Bool
end

#################### Dependent ####################

const depfxargs = [(:model, Model)]


#################### Base Methods ####################

function Base.show(io::IO, d::AbstractDependent)
    msg = string(
        ifelse(isempty(d.monitor), "An un", "A "),
        "monitored node of type \"",
        summary(d),
        "\"\n",
    )
    print(io, msg)
    show(io, d.value)
end

function showall(io::IO, d::AbstractDependent)
    show(io, d)
    print(io, "\nFunction:\n")
    show(io, "text/plain", first(code_typed(d.eval)))
    print(io, "\n\nSource Nodes:\n")
    show(io, d.sources)
    print(io, "\n\nTarget Nodes:\n")
    show(io, d.targets)
end

dims(d::AbstractDependent) = size(d)

function names(d::AbstractDependent)
    names(d, d.symbol)
end

function setmonitor!(d::AbstractDependent, monitor::Bool)
    value = monitor ? Int[0] : Int[]
    setmonitor!(d, value)
end

function setmonitor!(d::AbstractDependent, monitor::Vector{Int})
    values = monitor
    n = length(unlist(d))
    if n > 0 && !isempty(monitor)
        if monitor[1] == 0
            values = collect(1:n)
        elseif minimum(monitor) < 1 || maximum(monitor) > n
            throw(BoundsError())
        end
    end
    d.monitor = values
    d
end


#################### Distribution Fallbacks ####################

unlist(d::AbstractDependent, transform::Bool = false) = unlist(d, d.value, transform)

unlist(d::AbstractDependent, x::Real, transform::Bool = false) = [x]

unlist(d::AbstractDependent, x::AbstractArray, transform::Bool = false) = vec(x)

relist(d::AbstractDependent, x::AbstractArray, transform::Bool = false) =
    relistlength(d, x, transform)[1]

logpdf(d::AbstractDependent, transform::Bool = false) = 0.0

logpdf(d::AbstractDependent, x, transform::Bool = false) = 0.0


#################### Logical ####################

@promote_scalarvariate ScalarLogical


#################### Constructors ####################

function Logical(f::Function, monitor::Union{Bool,Vector{Int}} = true)
    value = Float64(NaN)
    fx, src = modelfxsrc(depfxargs, f)
    l = ScalarLogical(value, :nothing, Int[], fx, src, Symbol[])
    setmonitor!(l, monitor)
end

Logical(f::Function, d::Integer, args...) = Logical(d, f, args...)

function Logical(d::Integer, f::Function, monitor::Union{Bool,Vector{Int}} = true)
    value = Array{Float64}(undef, fill(0, d)...)
    fx, src = modelfxsrc(depfxargs, f)
    l = ArrayLogical(value, :nothing, Int[], fx, src, Symbol[])
    setmonitor!(l, monitor)
end

ScalarLogical(x::T) where {T<:Real} = x


#################### Updating ####################

function setinits!(l::AbstractLogical, m::Model, ::Any = nothing)
    l.value = l.eval(m)
    setmonitor!(l, l.monitor)
end

#################### Stochastic ####################

#################### Base Methods ####################

@promote_scalarvariate ScalarStochastic

function showall(io::IO, s::AbstractStochastic)
    show(io, s)
    print(io, "\n\nDistribution:\n")
    show(io, s.distr)
    print(io, "\nFunction:\n")
    show(io, "text/plain", first(code_typed(s.eval)))
    print(io, "\n\nSource Nodes:\n")
    show(io, s.sources)
    print(io, "\n\nTarget Nodes:\n")
    show(io, s.targets)
end


#################### Constructors ####################

function Stochastic(f::Function, monitor::Union{Bool,Vector{Int}} = true)
    value = Float64(NaN)
    fx, src = modelfxsrc(depfxargs, f)
    s = ScalarStochastic(
        value,
        :nothing,
        Int[],
        fx,
        src,
        Symbol[],
        NullUnivariateDistribution(),
    )
    setmonitor!(s, monitor)
end

Stochastic(f::Function, d::Integer, args...) = Stochastic(d, f, args...)

function Stochastic(d::Integer, f::Function, monitor::Union{Bool,Vector{Int}} = true)
    value = Array{Float64}(undef, fill(0, d)...)
    fx, src = modelfxsrc(depfxargs, f)
    s = ArrayStochastic(
        value,
        :nothing,
        Int[],
        fx,
        src,
        Symbol[],
        NullUnivariateDistribution(),
    )
    setmonitor!(s, monitor)
end

ScalarStochastic(x::T) where {T<:Real} = x


#################### Updating ####################

function setinits!(s::ScalarStochastic, m::Model, x::Real)
    s.value = convert(Float64, x)
    s.distr = s.eval(m)
    setmonitor!(s, s.monitor)
end

function setinits!(s::ArrayStochastic, m::Model, x::DenseArray)
    s.value = convert(typeof(s.value), copy(x))
    s.distr = s.eval(m)
    if !isa(s.distr, UnivariateDistribution) && dims(s) != dims(s.distr)
        throw(DimensionMismatch("incompatible distribution for stochastic node"))
    end
    setmonitor!(s, s.monitor)
end

function setinits!(s::AbstractStochastic, m::Model, x)
    throw(ArgumentError("incompatible initial value for node : $(s.symbol)"))
end

#################### Model Graph ####################

function ModelGraph(m::Model)
    allkeys = keys(m, :all)
    g = DiGraph(length(allkeys))
    lookup = Dict(allkeys[i] => i for i = 1:length(allkeys))
    for key in keys(m)
        node = m[key]
        if isa(node, AbstractDependent)
            for src in node.sources
                add_edge!(g, lookup[src], lookup[key])
            end
        end
    end
    ModelGraph(g, allkeys)
end


graph(m::Model) = ModelGraph(m)

function any_stochastic(dag::ModelGraph, v::Int, m::Model)
    found = false
    for t in outneighbors(dag.graph, v)
        tkey = dag.keys[t]
        if isa(m[tkey], AbstractStochastic) || any_stochastic(dag, t, m)
            found = true
            break
        end
    end
    found
end

function gettargets(dag::ModelGraph, v::Int, terminalkeys::Vector{Symbol})
    values = Symbol[]
    for t in outneighbors(dag.graph, v)
        tkey = dag.keys[t]
        push!(values, tkey)
        if !(tkey in terminalkeys)
            values = union(values, gettargets(dag, t, terminalkeys))
        end
    end
    values
end

function tsort(m::Model)
    dag = ModelGraph(m)
    dag.keys[topological_sort_by_dfs(dag.graph)]
end

#################### Model Initialization ####################

function setinits!(m::Model, inits::Dict{Symbol})
    m.hasinputs || throw(ArgumentError("inputs must be set before inits"))
    m.iter = 0
    for key in keys(m, :dependent)
        node = m[key]
        if isa(node, AbstractStochastic)
            haskey(inits, key) ||
                throw(ArgumentError("missing initial value for node : $key"))
            setinits!(node, m, inits[key])
        else
            setinits!(node, m)
        end
    end
    m.hasinits = true
    m
end

function setinits!(m::Model, inits::Vector{V} where {V<:Dict{Symbol}})
    n = length(inits)
    m.states = Array{ModelState}(undef, n)
    for i = n:-1:1
        setinits!(m, inits[i])
        m.states[i] = ModelState(unlist(m), deepcopy(gettune(m)))
    end
    m
end

function setinputs!(m::Model, inputs::Dict{Symbol})
    for key in keys(m, :input)
        haskey(inputs, key) || throw(ArgumentError("missing inputs for node : $key"))
        isa(inputs[key], AbstractDependent) &&
            throw(ArgumentError("inputs cannot be Dependent types"))
        m.nodes[key] = deepcopy(inputs[key])
    end
    m.hasinputs = true
    m
end


#################### Core Model Functionality ####################

#################### Constructors ####################

function Model(; nodes..., )
    nodedict = Dict{Symbol,Any}()
    for (key, value) in nodes
        isa(value, AbstractDependent) ||
            throw(ArgumentError("nodes are not all Dependent types"))
        node = deepcopy(value)
        node.symbol = key
        nodedict[key] = node
    end
    m = Model(nodedict, false, false)
    dag = ModelGraph(m)
    dependentkeys = keys(m, :dependent)
    terminalkeys = keys(m, :stochastic)
    for v in vertices(dag.graph)
        vkey = dag.keys[v]
        if vkey in dependentkeys
            m[vkey].targets = intersect(dependentkeys, gettargets(dag, v, terminalkeys))
        end
    end
    m
end

#################### Indexing ####################

Base.getindex(m::Model, nodekey::Symbol) = m.nodes[nodekey]


function Base.setindex!(m::Model, value, nodekey::Symbol)
    node = m[nodekey]
    if isa(node, AbstractDependent)
        node.value = value
    else
        m.nodes[nodekey] = convert(typeof(node), value)
    end
end

function Base.setindex!(m::Model, values::Dict, nodekeys::Vector{Symbol})
    for key in nodekeys
        m[key] = values[key]
    end
end

function Base.setindex!(m::Model, value, nodekeys::Vector{Symbol})
    length(nodekeys) == 1 || throw(BoundsError())
    m[first(nodekeys)] = value
end


Base.keys(m::Model) = collect(keys(m.nodes))

function Base.keys(m::Model, ntype::Symbol, at...)
    ntype == :block ? keys_block(m, at...) :
    ntype == :all ? keys_all(m) :
    ntype == :assigned ? keys_assigned(m) :
    ntype == :dependent ? keys_dependent(m) :
    ntype == :independent ? keys_independent(m) :
    ntype == :input ? keys_independent(m) :
    ntype == :logical ? keys_logical(m) :
    ntype == :monitor ? keys_monitor(m) :
    ntype == :output ? keys_output(m) :
    ntype == :source ? keys_source(m, at...) :
    ntype == :stochastic ? keys_stochastic(m) :
    ntype == :target ? keys_target(m, at...) :
    throw(ArgumentError("unsupported node type $ntype"))
end

function keys_all(m::Model)
    values = Symbol[]
    for key in keys(m)
        node = m[key]
        if isa(node, AbstractDependent)
            push!(values, key)
            append!(values, node.sources)
        end
    end
    unique(values)
end

function keys_assigned(m::Model)
    if m.hasinits
        values = keys(m)
    else
        values = Symbol[]
        for key in keys(m)
            if !isa(m[key], AbstractDependent)
                push!(values, key)
            end
        end
    end
    values
end

function keys_dependent(m::Model)
    values = Symbol[]
    for key in keys(m)
        if isa(m[key], AbstractDependent)
            push!(values, key)
        end
    end
    intersect(tsort(m), values)
end

function keys_independent(m::Model)
    deps = Symbol[]
    for key in keys(m)
        if isa(m[key], AbstractDependent)
            push!(deps, key)
        end
    end
    setdiff(keys(m, :all), deps)
end

function keys_logical(m::Model)
    values = Symbol[]
    for key in keys(m)
        if isa(m[key], AbstractLogical)
            push!(values, key)
        end
    end
    values
end

function keys_monitor(m::Model)
    values = Symbol[]
    for key in keys(m)
        node = m[key]
        if isa(node, AbstractDependent) && !isempty(node.monitor)
            push!(values, key)
        end
    end
    values
end

function keys_output(m::Model)
    values = Symbol[]
    dag = ModelGraph(m)
    for v in vertices(dag.graph)
        vkey = dag.keys[v]
        if isa(m[vkey], AbstractStochastic) && !any_stochastic(dag, v, m)
            push!(values, vkey)
        end
    end
    values
end

keys_source(m::Model, nodekey::Symbol) = m[nodekey].sources

function keys_source(m::Model, nodekeys::Vector{Symbol})
    values = Symbol[]
    for key in nodekeys
        append!(values, m[key].sources)
    end
    unique(values)
end

function keys_stochastic(m::Model)
    values = Symbol[]
    for key in keys(m)
        if isa(m[key], AbstractStochastic)
            push!(values, key)
        end
    end
    values
end

keys_target(m::Model, nodekey::Symbol) = m[nodekey].targets

function keys_target(m::Model, nodekeys::Vector{Symbol})
    values = Symbol[]
    for key in nodekeys
        append!(values, m[key].targets)
    end
    intersect(keys(m, :dependent), values)
end


#################### Display ####################

function Base.show(io::IO, m::Model)
    showf(io, m, Base.show)
end

function showall(io::IO, m::Model)
    showf(io, m, Base.showall)
end

function showf(io::IO, m::Model, f::Function)
    print(io, "Object of type \"$(summary(m))\"\n")
    width = displaysize()[2] - 1
    for node in keys(m)
        print(io, string("-"^width, "\n", node, ":\n"))
        f(io, m[node])
        println(io)
    end
end

#################### Model Expression Operators ####################

function modelfx(literalargs::Vector{Tuple{Symbol, DataType}}, f::Function)
    modelfxsrc(literalargs, f)[1]
  end
  
  function modelfxsrc(literalargs::Vector{Tuple{Symbol, DataType}}, f::Function)
    args = Expr(:tuple, map(arg -> Expr(:(::), arg[1], arg[2]), literalargs)...)
    expr, src = modelexprsrc(f, literalargs)
    fx = Core.eval(Main, Expr(:function, args, expr))
    (fx, src)
  end
  
  
  function modelexprsrc(f::Function, literalargs::Vector{Tuple{Symbol, DataType}})
    m = first(methods(f).ms)
    argnames = Base.method_argnames(m)
    fkeys = Symbol[argnames[2:end]...]
    ftypes = DataType[m.sig.parameters[2:end]...]
    n = length(fkeys)
  
    literalinds = Int[]
    for (key, T) in literalargs
      i = findfirst(fkey -> fkey == key, fkeys)
      if i != nothing && ftypes[i] == T
        push!(literalinds, i)
      end
    end
    nodeinds = setdiff(1:n, literalinds)
  
    all(T -> T == Any, ftypes[nodeinds]) ||
      throw(ArgumentError("model node arguments are not all of type Any"))
  
    modelargs = Array{Any}(undef, n)
    for i in nodeinds
      modelargs[i] = Expr(:ref, :model, QuoteNode(fkeys[i]))
    end
    for i in literalinds
      modelargs[i] = fkeys[i]
    end
    expr = Expr(:block, Expr(:(=), :f, f), Expr(:call, :f, modelargs...))
  
    (expr, fkeys[nodeinds])
  end

#################### DistributionStruct ####################

#################### Base Methods ####################

dims(d::DistributionStruct) = size(d)

function dims(D::Array{MultivariateDistribution})
  size(D)..., mapreduce(length, max, D)
end
