### Serialisation to JSON / string

using JSON: JSON

# String constants for each index type that we support serialisation /
# deserialisation of
const _BASE_INTEGER_TYPE = "Base.Integer"
const _BASE_VECTOR_TYPE = "Base.Vector"
const _BASE_UNITRANGE_TYPE = "Base.UnitRange"
const _BASE_STEPRANGE_TYPE = "Base.StepRange"
const _BASE_ONETO_TYPE = "Base.OneTo"
const _BASE_COLON_TYPE = "Base.Colon"
const _CONCRETIZED_SLICE_TYPE = "AbstractPPL.ConcretizedSlice"
const _BASE_TUPLE_TYPE = "Base.Tuple"

"""
    index_to_dict(::Integer)
    index_to_dict(::AbstractVector{Int})
    index_to_dict(::UnitRange)
    index_to_dict(::StepRange)
    index_to_dict(::Colon)
    index_to_dict(::ConcretizedSlice{T, Base.OneTo{I}}) where {T, I}
    index_to_dict(::Tuple)

Convert an index `i` to a dictionary representation.
"""
index_to_dict(i::Integer) = Dict("type" => _BASE_INTEGER_TYPE, "value" => i)
index_to_dict(v::Vector{Int}) = Dict("type" => _BASE_VECTOR_TYPE, "values" => v)
function index_to_dict(r::UnitRange)
    return Dict("type" => _BASE_UNITRANGE_TYPE, "start" => r.start, "stop" => r.stop)
end
function index_to_dict(r::StepRange)
    return Dict(
        "type" => _BASE_STEPRANGE_TYPE,
        "start" => r.start,
        "stop" => r.stop,
        "step" => r.step,
    )
end
function index_to_dict(r::Base.OneTo{I}) where {I}
    return Dict("type" => _BASE_ONETO_TYPE, "stop" => r.stop)
end
index_to_dict(::Colon) = Dict("type" => _BASE_COLON_TYPE)
function index_to_dict(s::ConcretizedSlice{T,R}) where {T,R}
    return Dict("type" => _CONCRETIZED_SLICE_TYPE, "range" => index_to_dict(s.range))
end
function index_to_dict(t::Tuple)
    return Dict("type" => _BASE_TUPLE_TYPE, "values" => map(index_to_dict, t))
end

"""
    dict_to_index(dict)
    dict_to_index(symbol_val, dict)

Convert a dictionary representation of an index `dict` to an index.

Users can extend the functionality of `dict_to_index` (and hence `VarName`
de/serialisation) by extending this method along with [`index_to_dict`](@ref).
Specifically, suppose you have a custom index type `MyIndexType` and you want
to be able to de/serialise a `VarName` containing this index type. You should
then implement the following two methods:

1. `AbstractPPL.index_to_dict(i::MyModule.MyIndexType)` should return a
   dictionary representation of the index `i`. This dictionary must contain the
   key `"type"`, and the corresponding value must be a string that uniquely
   identifies the index type. Generally, it makes sense to use the name of the
   type (perhaps prefixed with module qualifiers) as this value to avoid
   clashes. The remainder of the dictionary can have any structure you like.

2. Suppose the value of `index_to_dict(i)["type"]` is `"MyModule.MyIndexType"`.
   You should then implement the corresponding method
   `AbstractPPL.dict_to_index(::Val{Symbol("MyModule.MyIndexType")}, dict)`,
   which should take the dictionary representation as the second argument and
   return the original `MyIndexType` object.

To see an example of this in action, you can look in the the AbstractPPL test
suite, which contains a test for serialising OffsetArrays.
"""
function dict_to_index(dict)
    t = dict["type"]
    if t == _BASE_INTEGER_TYPE
        return dict["value"]
    elseif t == _BASE_VECTOR_TYPE
        return collect(Int, dict["values"])
    elseif t == _BASE_UNITRANGE_TYPE
        return dict["start"]:dict["stop"]
    elseif t == _BASE_STEPRANGE_TYPE
        return dict["start"]:dict["step"]:dict["stop"]
    elseif t == _BASE_ONETO_TYPE
        return Base.OneTo(dict["stop"])
    elseif t == _BASE_COLON_TYPE
        return Colon()
    elseif t == _CONCRETIZED_SLICE_TYPE
        return ConcretizedSlice(Base.Slice(dict_to_index(dict["range"])))
    elseif t == _BASE_TUPLE_TYPE
        return tuple(map(dict_to_index, dict["values"])...)
    else
        # Will error if the method is not defined, but this hook allows users
        # to extend this function
        return dict_to_index(Val(Symbol(t)), dict)
    end
end

optic_to_dict(::typeof(identity)) = Dict("type" => "identity")
function optic_to_dict(::PropertyLens{sym}) where {sym}
    return Dict("type" => "property", "field" => String(sym))
end
optic_to_dict(i::IndexLens) = Dict("type" => "index", "indices" => index_to_dict(i.indices))
function optic_to_dict(c::ComposedFunction)
    return Dict(
        "type" => "composed",
        "outer" => optic_to_dict(c.outer),
        "inner" => optic_to_dict(c.inner),
    )
end

function dict_to_optic(dict)
    if dict["type"] == "identity"
        return identity
    elseif dict["type"] == "index"
        return IndexLens(dict_to_index(dict["indices"]))
    elseif dict["type"] == "property"
        return PropertyLens{Symbol(dict["field"])}()
    elseif dict["type"] == "composed"
        return dict_to_optic(dict["outer"]) ∘ dict_to_optic(dict["inner"])
    else
        error("Unknown optic type: $(dict["type"])")
    end
end

function varname_to_dict(vn::VarName)
    return Dict("sym" => getsym(vn), "optic" => optic_to_dict(getoptic(vn)))
end

function dict_to_varname(dict::Dict{<:AbstractString,Any})
    return VarName{Symbol(dict["sym"])}(dict_to_optic(dict["optic"]))
end

"""
    varname_to_string(vn::VarName)

Convert a `VarName` as a string, via an intermediate dictionary. This differs
from `string(vn)` in that concretised slices are faithfully represented (rather
than being pretty-printed as colons).

For `VarName`s which index into an array, this function will only work if the
indices can be serialised. This is true for all standard Julia index types, but
if you are using custom index types, you will need to implement the
`index_to_dict` and `dict_to_index` methods for those types. See the
documentation of [`dict_to_index`](@ref) for instructions on how to do this.

```jldoctest
julia> varname_to_string(@varname(x))
"{\\"optic\\":{\\"type\\":\\"identity\\"},\\"sym\\":\\"x\\"}"

julia> varname_to_string(@varname(x.a))
"{\\"optic\\":{\\"field\\":\\"a\\",\\"type\\":\\"property\\"},\\"sym\\":\\"x\\"}"

julia> y = ones(2); varname_to_string(@varname(y[:]))
"{\\"optic\\":{\\"indices\\":{\\"values\\":[{\\"type\\":\\"Base.Colon\\"}],\\"type\\":\\"Base.Tuple\\"},\\"type\\":\\"index\\"},\\"sym\\":\\"y\\"}"

julia> y = ones(2); varname_to_string(@varname(y[:], true))
"{\\"optic\\":{\\"indices\\":{\\"values\\":[{\\"range\\":{\\"stop\\":2,\\"type\\":\\"Base.OneTo\\"},\\"type\\":\\"AbstractPPL.ConcretizedSlice\\"}],\\"type\\":\\"Base.Tuple\\"},\\"type\\":\\"index\\"},\\"sym\\":\\"y\\"}"
```
"""
varname_to_string(vn::VarName) = JSON.json(varname_to_dict(vn))

"""
    string_to_varname(str::AbstractString)

Convert a string representation of a `VarName` back to a `VarName`. The string
should have been generated by `varname_to_string`.
"""
string_to_varname(str::AbstractString) = dict_to_varname(JSON.parse(str))
