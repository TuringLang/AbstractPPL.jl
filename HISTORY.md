## 0.14.0

This release overhauls the `VarName` type.
Much of the external API for traversing and manipulating `VarName`s has been preserved, but there are significant changes:

**Internal representation**

The `optic` field of VarName now uses our hand-rolled optic types, which are subtypes of `AbstractPPL.AbstractOptic`.
Previously these were optics from Accessors.jl.

This change was made for two reasons: firstly, it is easier to provide custom behaviour for VarNames as we avoid running into possible type piracy issues, and secondly, the linked-list data structure used in `AbstractOptic` is easier to work with than Accessors.jl, which used `Base.ComposedFunction` to represent optic compositions and required a lot of care to avoid issues with associativity and identity optics.

To construct an optic, the easiest way is to use the `@opticof` macro, which superficially behaves similarly to `Accessors.@optic` (for example, you can write `@opticof _[1].y.z`), but also supports automatic concretization by passing a second parameter (just like `@varname`).

**Concretization**

VarNames using 'dynamic' indices, i.e., `begin` and `end`, are now instantiated in a 'dynamic' form, meaning that these indices are unresolved.
These indices need to be resolved, or concretized, against the actual container.
For example, `@varname(x[end])` is dynamic, but when concretized against `x = randn(3)`, this becomes `@varname(x[3])`.
This can be done using `concretize(varname, x)`.

The idea of concretization is not new to AbstractPPL.
However, there are some differences:

  - Colons are no longer concretized: they *always* remain as Colons, even after calling `concretize`.
  - Previously, AbstractPPL would refuse to allow you to construct unconcretized versions of `begin` and `end`. This is no longer the case; you can now create such VarNames in their unconcretized forms.
    This is useful, for example, when indexing into a chain that contains `x` as a variable-length vector. This change allows you to write `chain[@varname(x[end])]` without having AbstractPPL throw an error.

**Interface**

The `vsym` function (and `@vsym`) has been removed; you should use `getsym(vn)` instead.

The `Base.get` and `Base.set!` methods for VarNames have been removed (these were responsible for method ambiguities).

VarNames cannot be composed with optics now (you need to compose the optics yourself).

The `inspace` function has been removed (it used to be relevant for Turing's old Gibbs sampler; but now it no longer serves any use).

## 0.13.6

Fix a missing qualifier in AbstractPPLDistributionsExt.

## 0.13.5

Implemented a generic `varname_leaves` and `varname_and_value_leaves` for other unsupported types.

## 0.13.4

Added missing methods for `subsumes(::IndexLens, ::PropertyLens)` and vice versa.

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
