# VarNames and optics

## VarNames: an overview

One of the most important parts of AbstractPPL.jl is the `VarName` type, which is used throughout the TuringLang ecosystem to represent names of random variables.

Fundamentally, a `VarName` comprises a symbol (which represents the name of the variable itself) and an optic (which tells us which part of the variable we might be interested in).
For example, `x.a[1]` means the first element of the field `a` of the variable `x`.
Here, `x` is the symbol, and `.a[1]` is the optic.

VarNames can be created using the `@varname` macro:

```@example vn
using AbstractPPL

vn = @varname(x.a[1])
```

```@docs
VarName
@varname
varname
```

You can obtain the components of a `VarName` using the `getsym` and `getoptic` functions:

```@example vn
getsym(vn), getoptic(vn)
```

```@docs
getsym
getoptic
```

## Dynamic indices

VarNames may contain 'dynamic' indices, that is, indices whose meaning is not known until they are resolved against a specific value.
For example, `x[end]` refers to the last element of `x`; but we don't know what that means until we know what `x` is.

Specifically, `begin` and `end` symbols in indices are treated as dynamic indices.
This is also true for any expression that contains `begin` or `end`, such as `end-1` or `1:3:end`.

Dynamic indices are represented using an internal type, `AbstractPPL.DynamicIndex`.

```@example vn
vn_dyn = @varname(x[1:2:end])
```

You can detect whether a VarName contains dynamic indices using the `is_dynamic` function:

```@example vn
is_dynamic(vn_dyn)
```

```@docs
is_dynamic
```

These dynamic indices can be resolved, or _concretized_, by passing a specific value to the `concretize` function:

```@example vn
x = randn(5)
vn_conc = concretize(vn_dyn, x)
```

```@docs
concretize
```

## Optics

The optics used in AbstractPPL.jl are represented as a linked list.
For example, the optic `.a[1]` is a `Property` optic that contains an `Index` optic as its child.
That means that the 'elements' of the linked list can be read from left-to-right:

```
Property{:a} -> Index{1} -> Iden
```

All optic linked lists are terminated with an `Iden` optic, which represents the identity function.

```@example vn
optic = getoptic(@varname x.a[1])
dump(optic)
```

```@docs
AbstractOptic
Property
Index
Iden
```

Instead of calling `getoptic(@varname(...))`, you can directly use the [`@opticof`](@ref) macro to create optics:

```@example vn
optic = @opticof(_.a[1])
```

```@docs
@opticof
```

## Getting and setting

Optics are callable structs, and when passed a value will extract the relevant part of that
value.

```@example vn
data = (a=[10, 20, 30], b="hello")
optic = @opticof(_.a[2])
optic(data)
```

You can set values using `Accessors.set` (which AbstractPPL re-exports).
Note, though, that this will not mutate the original value.
Furthermore, you cannot use the handy macros like `Accessors.@set`, since those will use the
optics from Accessors.jl.

```@example vn
new_data = set(data, optic, 99)
new_data, data
```

If you want to try to mutate values, you can wrap an optic using `with_mutation`.

```@example vn
optic_mut = with_mutation(optic)
set(data, optic_mut, 99)
data
```

```@docs
with_mutation
```

## Composing and decomposing optics

If you have two optics, you can compose them using the `∘` operator:

```@example vn
optic1 = @opticof(_.a)
optic2 = @opticof(_[1])
composed = optic2 ∘ optic1
```

Notice the order of composition here, which can be counterintuitive: `optic2 ∘ optic1` means "first apply `optic1`, then apply `optic2`", and thus this represents the optic `.a[1]` (not `.[1].a`).

```@docs
Base.:∘(::AbstractOptic, ::AbstractOptic)
```

`Base.cat(optics...)` is also provided, which composes optics in a more intuitive sense (indeed, if you think of an optic as a linked list, this can be thought of as concatenating the lists).
The following is equivalent to the previous example:

```@example vn
composed2 = Base.cat(optic1, optic2)
```

```@docs
Base.cat(::AbstractOptic...)
```

Several functions are provided to decompose optics, which all stem from their linked-list structure.
Their names directly mirror Haskell's functions for decomposing lists, but are prefixed with `o`:

```@docs
ohead
otail
oinit
olast
```

For example, `ohead` returns the first element of the optic linked list, and `otail` returns the rest of the list after removing the head:

```@example vn
optic = @opticof(_.a[1].b[2])
ohead(optic), otail(optic)
```

Convesely, `oinit` returns the optic linked list without its last element, and `olast` returns the last element:

```@example vn
oinit(optic), olast(optic)
```

If the optic only has a single element, then `oinit` and `otail` return `Iden`, while `ohead` and `olast` return the optic itself:

```@example vn
optic_single = @opticof(_.a)
oinit(optic_single), olast(optic_single), ohead(optic_single), otail(optic_single)
```

## Converting VarNames to optics and back

Sometimes it is useful to treat a VarName's top level symbol as if it were part of the optic.
For example, when indexing into a NamedTuple `nt`, we might want to treat the entire VarName `x.a[1]` as an optic that can be applied to a NamedTuple: i.e., we want to access the `nt.x` field rather than the variable `x` itself.
This can be achieved with:

```@docs
varname_to_optic
optic_to_varname
```

## Subsumption

Sometimes, we want to check whether one VarName 'subsumes' another; that is, whether a VarName refers to a part of another VarName.
This is done using the [`subsumes`](@ref) function:

```@example vn
vn1 = @varname(x.a)
vn2 = @varname(x.a[1])
subsumes(vn1, vn2)
```

```@docs
subsumes
```

## Prefixing and unprefixing

Composing two optics can be done using the `∘` operator, as shown above.
But what if we want to compose two `VarName`s?
This is used, for example, in DynamicPPL's submodel functionality.

```@docs
prefix
unprefix
```

## VarName leaves

The following functions are used to extract the 'leaves' of a VarName, that is, the atomic components of a VarName that do not have any further substructure.
For example, for a vector variable `x`, the leaves would be `x[1]`, `x[2]`, etc.

```@docs
varname_leaves
varname_and_value_leaves
```

## Reading from a container with a VarName (or optic)

```@docs
canview
hasvalue
getvalue
```

## Serializing VarNames

```@docs
index_to_dict
dict_to_index
varname_to_string
string_to_varname
```
