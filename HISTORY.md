## 0.12.1

Minimum compatibility has been bumped to Julia 1.10.

Added the new functions `hasvalue(container::T, ::VarName[, ::Distribution])` and `getvalue(container::T, ::VarName[, ::Distribution])`, where `T` is either `NamedTuple` or `AbstractDict{<:VarName}`.

These functions check whether a given `VarName` has a value in the given `NamedTuple` or `AbstractDict`, and return the value if it exists.

The optional `Distribution` argument allows one to reconstruct a full value from its component indices.
For example, if `container` has `x[1]` and `x[2]`, then `hasvalue(container, @varname(x), dist)` will return true if `size(dist) == (2,)` (for example, `MvNormal(zeros(2), I)`).

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
