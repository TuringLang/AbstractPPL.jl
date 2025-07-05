## 0.12.0

### VarName constructors

Removed the constructors `VarName(vn, optic)` (this wasn't deprecated, but was dangerous as it would silently discard the existing optic in `vn`), and `VarName(vn, ::Tuple)` (which was deprecated).

Usage of `VarName(vn, optic)` can be directly replaced with `VarName{getsym(vn)}(optic)`.

### Optic normalisation

In the inner constructor of a VarName, its optic is now normalised to ensure that the associativity of ComposedFunction is always the same, and that compositions with identity are removed.
This helps to prevent subtle bugs where VarNames with semantically equal optics are not considered equal.

## 0.11.0

Added the `prefix(vn::VarName, vn_prefix::VarName)` and `unprefix(vn::VarName, vn_prefix::VarName)` functions.
