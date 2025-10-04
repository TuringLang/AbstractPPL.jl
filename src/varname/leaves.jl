using LinearAlgebra: LinearAlgebra

"""
    varname_leaves(vn::VarName, val)

Return an iterator over all varnames that are represented by `vn` on `val`.

# Examples
```jldoctest
julia> using AbstractPPL: varname_leaves

julia> foreach(println, varname_leaves(@varname(x), rand(2)))
x[1]
x[2]

julia> foreach(println, varname_leaves(@varname(x[1:2]), rand(2)))
x[1:2][1]
x[1:2][2]

julia> x = (y = 1, z = [[2.0], [3.0]]);

julia> foreach(println, varname_leaves(@varname(x), x))
x.y
x.z[1][1]
x.z[2][1]
```
"""
varname_leaves(vn::VarName, ::Real) = [vn]
function varname_leaves(vn::VarName, val::AbstractArray{<:Union{Real,Missing}})
    return (
        VarName{getsym(vn)}(Accessors.IndexLens(Tuple(I)) ∘ getoptic(vn)) for
        I in CartesianIndices(val)
    )
end
function varname_leaves(vn::VarName, val::AbstractArray)
    return Iterators.flatten(
        varname_leaves(
            VarName{getsym(vn)}(Accessors.IndexLens(Tuple(I)) ∘ getoptic(vn)), val[I]
        ) for I in CartesianIndices(val)
    )
end
function varname_leaves(vn::VarName, val::NamedTuple)
    iter = Iterators.map(keys(val)) do k
        optic = Accessors.PropertyLens{k}()
        varname_leaves(VarName{getsym(vn)}(optic ∘ getoptic(vn)), optic(val))
    end
    return Iterators.flatten(iter)
end

"""
    varname_and_value_leaves(vn::VarName, val)

Return an iterator over all varname-value pairs that are represented by `vn` on `val`.

# Examples
```jldoctest varname-and-value-leaves
julia> using AbstractPPL: varname_and_value_leaves

julia> foreach(println, varname_and_value_leaves(@varname(x), 1:2))
(x[1], 1)
(x[2], 2)

julia> foreach(println, varname_and_value_leaves(@varname(x[1:2]), 1:2))
(x[1:2][1], 1)
(x[1:2][2], 2)

julia> x = (y = 1, z = [[2.0], [3.0]]);

julia> foreach(println, varname_and_value_leaves(@varname(x), x))
(x.y, 1)
(x.z[1][1], 2.0)
(x.z[2][1], 3.0)
```

There is also some special handling for certain types:

```jldoctest varname-and-value-leaves
julia> using LinearAlgebra

julia> x = reshape(1:4, 2, 2);

julia> # `LowerTriangular`
       foreach(println, varname_and_value_leaves(@varname(x), LowerTriangular(x)))
(x[1, 1], 1)
(x[2, 1], 2)
(x[2, 2], 4)

julia> # `UpperTriangular`
       foreach(println, varname_and_value_leaves(@varname(x), UpperTriangular(x)))
(x[1, 1], 1)
(x[1, 2], 3)
(x[2, 2], 4)

julia> # `Cholesky` with lower-triangular
       foreach(println, varname_and_value_leaves(@varname(x), Cholesky([1.0 0.0; 0.0 1.0], 'L', 0)))
(x.L[1, 1], 1.0)
(x.L[2, 1], 0.0)
(x.L[2, 2], 1.0)

julia> # `Cholesky` with upper-triangular
       foreach(println, varname_and_value_leaves(@varname(x), Cholesky([1.0 0.0; 0.0 1.0], 'U', 0)))
(x.U[1, 1], 1.0)
(x.U[1, 2], 0.0)
(x.U[2, 2], 1.0)
```
"""
function varname_and_value_leaves(vn::VarName, x)
    return Iterators.map(value, Iterators.flatten(varname_and_value_leaves_inner(vn, x)))
end

"""
    varname_and_value_leaves(container)

Return an iterator over all varname-value pairs that are represented by `container`.

This is the same as [`varname_and_value_leaves(vn::VarName, x)`](@ref) but over a container
containing multiple varnames.

See also: [`varname_and_value_leaves(vn::VarName, x)`](@ref).

# Examples
```jldoctest varname-and-value-leaves-container
julia> using AbstractPPL: varname_and_value_leaves

julia> using OrderedCollections: OrderedDict

julia> # With an `AbstractDict` (we use `OrderedDict` here
       # to ensure consistent ordering in doctests)
       dict = OrderedDict(@varname(y) => 1, @varname(z) => [[2.0], [3.0]]);

julia> foreach(println, varname_and_value_leaves(dict))
(y, 1)
(z[1][1], 2.0)
(z[2][1], 3.0)

julia> # With a `NamedTuple`
       nt = (y = 1, z = [[2.0], [3.0]]);

julia> foreach(println, varname_and_value_leaves(nt))
(y, 1)
(z[1][1], 2.0)
(z[2][1], 3.0)
```
"""
function varname_and_value_leaves(container::AbstractDict)
    return Iterators.flatten(varname_and_value_leaves(k, v) for (k, v) in container)
end
function varname_and_value_leaves(container::NamedTuple)
    return Iterators.flatten(
        varname_and_value_leaves(VarName{k}(), v) for (k, v) in pairs(container)
    )
end

"""
    Leaf{T}

A container that represents the leaf of a nested structure, implementing
`iterate` to return itself.

This is particularly useful in conjunction with `Iterators.flatten` to
prevent flattening of nested structures.
"""
struct Leaf{T}
    value::T
end

Leaf(xs...) = Leaf(xs)

# Allow us to treat `Leaf` as an iterator containing a single element.
# Something like an `[x]` would also be an iterator with a single element,
# but when we call `flatten` on this, it would also iterate over `x`,
# unflattening that too. By making `Leaf` a single-element iterator, which
# returns itself, we can call `iterate` on this as many times as we like
# without causing any change. The result is that `Iterators.flatten`
# will _not_ unflatten `Leaf`s.
# Note that this is similar to how `Base.iterate` is implemented for `Real`::
#
#    julia> iterate(1)
#    (1, nothing)
#
# One immediate example where this becomes in our scenario is that we might
# have `missing` values in our data, which does _not_ have an `iterate`
# implemented. Calling `Iterators.flatten` on this would cause an error.
Base.iterate(leaf::Leaf) = leaf, nothing
Base.iterate(::Leaf, _) = nothing

# Convenience.
value(leaf::Leaf) = leaf.value

# Leaf-types.
varname_and_value_leaves_inner(vn::VarName, x::Real) = [Leaf(vn, x)]
function varname_and_value_leaves_inner(
    vn::VarName, val::AbstractArray{<:Union{Real,Missing}}
)
    return (
        Leaf(
            VarName{getsym(vn)}(Accessors.IndexLens(Tuple(I)) ∘ AbstractPPL.getoptic(vn)),
            val[I],
        ) for I in CartesianIndices(val)
    )
end
# Containers.
function varname_and_value_leaves_inner(vn::VarName, val::AbstractArray)
    return Iterators.flatten(
        varname_and_value_leaves_inner(
            VarName{getsym(vn)}(Accessors.IndexLens(Tuple(I)) ∘ AbstractPPL.getoptic(vn)),
            val[I],
        ) for I in CartesianIndices(val)
    )
end
function varname_and_value_leaves_inner(vn::VarName, val::NamedTuple)
    iter = Iterators.map(keys(val)) do k
        optic = Accessors.PropertyLens{k}()
        varname_and_value_leaves_inner(
            VarName{getsym(vn)}(optic ∘ getoptic(vn)), optic(val)
        )
    end

    return Iterators.flatten(iter)
end
# Special types.
function varname_and_value_leaves_inner(vn::VarName, x::LinearAlgebra.Cholesky)
    # TODO: Or do we use `PDMat` here?
    return if x.uplo == 'L'
        varname_and_value_leaves_inner(Accessors.PropertyLens{:L}() ∘ vn, x.L)
    else
        varname_and_value_leaves_inner(Accessors.PropertyLens{:U}() ∘ vn, x.U)
    end
end
function varname_and_value_leaves_inner(vn::VarName, x::LinearAlgebra.LowerTriangular)
    return (
        Leaf(VarName{getsym(vn)}(Accessors.IndexLens(Tuple(I)) ∘ getoptic(vn)), x[I])
        # Iteration over the lower-triangular indices.
        for I in CartesianIndices(x) if I[1] >= I[2]
    )
end
function varname_and_value_leaves_inner(vn::VarName, x::LinearAlgebra.UpperTriangular)
    return (
        Leaf(VarName{getsym(vn)}(Accessors.IndexLens(Tuple(I)) ∘ getoptic(vn)), x[I])
        # Iteration over the upper-triangular indices.
        for I in CartesianIndices(x) if I[1] <= I[2]
    )
end
