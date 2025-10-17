## 0.13.3

Bumped compatibility for JSON.jl to include v1.

## 0.13.2

Implemented `varname_leaves` for `LinearAlgebra.Cholesky`.

## 0.13.1

Moved the functions `varname_leaves` and `varname_and_value_leaves` to AbstractPPL.
They are now part of the public API of AbstractPPL.

## 0.13.0

Minimum compatibility has been bumped to Julia 1.10.

Added the new functions `hasvalue(container::T, ::VarName[, ::Distribution])` and `getvalue(container::T, ::VarName[, ::Distribution])`, where `T` is either `NamedTuple` or `AbstractDict{<:VarName}`.

These functions check whether a given `VarName` has a value in the given `NamedTuple` or `AbstractDict`, and return the value if it exists.

The optional `Distribution` argument allows one to reconstruct a full value from its component indices.
For example, if `container` has `x[1]` and `x[2]`, then `hasvalue(container, @varname(x), dist)` will return true if `size(dist) == (2,)` (for example, `MvNormal(zeros(2), I)`).
In this case plain `hasvalue(container, @varname(x))` would return `false`, since we can not know whether the vector-valued variable `x` has all of its elements specified in `container` (there might be an `x[3]` missing).

These functions (without the `Distribution` argument) were previously in DynamicPPL.jl (albeit unexported).

## 0.12.0

### VarName constructors

Removed the constructors `VarName(vn, optic)` (this wasn't deprecated, but was dangerous as it would silently discard the existing optic in `vn`), and `VarName(vn, ::Tuple)` (which was deprecated).

Usage of `VarName(vn, optic)` can be directly replaced with `VarName{getsym(vn)}(optic)`.

### Optic normalisation

In the inner constructor of a VarName, its optic is now normalised to ensure that the associativity of ComposedFunction is always the same, and that compositions with identity are removed.
This helps to prevent subtle bugs where VarNames with semantically equal optics are not considered equal.

## 0.11.0

Added the `prefix(vn::VarName, vn_prefix::VarName)` and `unprefix(vn::VarName, vn_prefix::VarName)` functions.
