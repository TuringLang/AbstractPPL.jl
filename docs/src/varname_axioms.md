# `VarName`-based axioms

What follows is mostly an attempt to formalize subsumption.

First, remember that in Turing.jl we can always work with _concretized_ `VarName`s: `begin`/`end`,
`:`, and boolean indexing are all turned into some form of concrete cartesian or array indexing
(assuming [this suggestion](https://github.com/TuringLang/AbstractPPL.jl/issues/35) being
implemented).  This makes all index comparisons static.

Now, `VarName`s have a compositional structure: they can be built by composing a root variable with
more and more lenses (`VarName{v}()` starts off with an `IdentityLens`):

```julia
julia> vn = VarName{:x}() ∘ Setfield.IndexLens((1:10, 1) ∘ Setfield.IndexLens((2,)))
x[1:10,1][2]
```

(_Note that the composition function, `∘`, is really in wrong order; but this is a heritage of
Setfield.jl._)

By "subsumption", we mean the notion of a `VarName` expressing a more nested path than another one:

```julia
subsumes(@varname(x.a), @varname(x.a[1]))
@varname(x.a) ⊒ @varname(x.a[1]) # \sqsupseteq
@varname(x.a) ⋢ @varname(x.a[1]) # \nsqsubseteq
```

Thus, we have the following axioms for `VarName`s ("variables" are `VarName{n}()`):

 1. `x ⊑ x` for all variables `x`
 2. `x ≍ y` for `x ≠ y` (i.e., distinct variables are incomparable; `x ⋢ y` and `y ⋢ x`) (`≍` is `\asymp`)
 3. `x ∘ ℓ ⊑ x` for all variables `x` and lenses `ℓ`
 4. `x ∘ ℓ₁ ⊑ x ∘ ℓ₂ ⇔ ℓ₁ ⊑ ℓ₂`

For the last axiom to work, we also have to define subsumption of individual, non-composed lenses:

 1. `PropertyLens(a) == PropertyLens(b) ⇔ a == b`, for all symbols `a`, `b`
 2. `FunctionLens(f) == FunctionLens(g) ⇔ f == g` (under extensional equality; I'm only mentioning
    this in case we ever generalize to Bijector-ed variables like `@varname(log(x))`)
 3. `IndexLens(ι₁) ⊑ IndexLens(ι₂)` if the index tuple `ι₂` covers all indices in `ι₁`; for example,
    `_[1, 2:10] ⊑ _[1:10, 1:20]`.  (_This is a bit fuzzy and not all corner cases have been
    considered yet!_)
 4. `IdentityLens() == IdentityLens()`
 5. `ℓ₁ ≍ ℓ₂`, otherwise

Together, this should make `VarName`s under subsumption a reflexive poset.

The fundamental requirement for `VarInfo`s is then:

```
vi[x ∘ ℓ] == get(vi[x], ℓ)
```

So we always want the following to work, automatically:

```julia
vi = insert!!(vi, vn, x)
vi[vn] == x
```

(the trivial case), and

```julia
x = set!!(x, ℓ₁, a)
x = set!!(x, ℓ₂, b)
vi = insert!!(vi, vn, x)
vi[vn ∘ ℓ₁] == a
vi[vn ∘ ℓ₂] == b
```

since `vn` subsumes both `vn ∘ ℓ₁` and `vn ∘ ℓ₂`.

Whether the opposite case is supported may depend on the implementation.  The most complicated part
is "unification":

```julia
vi = insert!!(vi, vn ∘ ℓ₁, a)
vi = insert!!(vi, vn ∘ ℓ₂, b)
get(vi[vn], ℓ₁) == a
get(vi[vn], ℓ₂) == b
```

where `vn ∘ ℓ₁` and `vn ∘ ℓ₂` need to be recognized as "children" of a common parent `vn`.