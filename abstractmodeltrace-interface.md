# `AbstractModelType` interface proposal

## Background

### Why do we do this?

As I have said before: 

> There are many aspects that make VarInfo a very complex data structure.

Currently, there is an insane amount of complexity and implementation details in `varinfo.jl`, which
has been rewritten multiple times with different concerns in mind – most times to improve concrete
needs of Turing.jl, such as type stability, or requirements of specific samplers.

This unfortunately makes `VarInfo` extremely opaque: it is hard to refactor without breaking
anything (nobody really dares touching it), and a lot of knowledge about Turing.jl/DynamicPPL.jl
internals is needed in order to judge the effects of changes.

### Design choices

Recently, @torfjelde [has shown](https://github.com/TuringLang/DynamicPPL.jl/pull/267/files) that a
much simpler implementation is feasible – basically, just a wrapped `NamedTuple` with a minimal
interface.

The purpose of this proposal is twofold: first, to think about what a sufficient interface for
`AbstractModelTrace`, the abstract supertype of `VarInfo`, should be, to allow multiple specialized
variants and refactor the existing ones (typed/untyped and simple).  Second, to view the problem as
the design of an abstract data type: the specification of construction and modification mechanisms
for a dictionary-like structure.

Related previous discussions:

- [Discussion about `VarName`](https://github.com/TuringLang/AbstractPPL.jl/discussions/7)
- [`AbstractVarInfo` representation](https://github.com/TuringLang/AbstractPPL.jl/discussions/5)

Additionally (but closely related), the second part tries to formalize the “subsumption” mechanism
of `VarName`s, and its interaction with using `VarName`s as keys/indices.

Our discussions take place in what is a bit of a fuzzy zone between the part that is really
“abstract”, and meant for the wider purpuse of AbstractPPL.jl – the implementation of probabilistic
programming systems in general – and our concrete needs within DPPL.  I hope to always stay abstract
and reusable; and there are already a couple of candidates for APPL clients other than DPPL, which
will hopefully keep us focused: simulation based calibration, SimplePPL (a BUGS-like frontend), and
ParetoSmoothing.jl.

### What is going to change?

- For the end user of Turing.jl: nothing.  You usually don’t use `VarInfo`, or the raw evaluator
interface, anyways.  (Although if the newer data structures are more user-friendly, they might occur
in more places in the future?)
- For people having a look into code using `VarInfo`, or starting to hack on Turing.jl/DPPL.jl: a
huge reduction in cognitive complexity.  `VarInfo` implementations should be readable on their own,
and the implemented functions layed out somewhere.  Its usages should look like for any other nice,
normal data structure.
- For core DPPL.jl implementors: same as the previous, plus: a standard against which to improve and
test `VarInfo`, and a clearly defined design space for new data structures.
- For AbstractPPL.jl clients/PPL implementors: an interface to program against (as with the rest of
APPL), and an existing set of well-specified, flexible trace data types with different
characteristics.

And in terms of implementation work in DPPL.jl: once the interface is fixed (or even during fixing
it), varinfo.jl will undergo a heavy refactoring – which should make it _simpler_! (No three
different getter functions with slightly different semantics, etc…).


## Dictionary interface

The basic idea is for all `VarInfo`s to behave like ordered dictionaries with `VarName` keys – all
common operations should just work.  There are two things that make them more special, though:

1. “Fancy indexing”: since `VarName`s are structured themselves, the `VarInfo` should be have a bit
   like a trie, in the sense that all prefixes of stored keys should be retrievable.  Also,
   subsumption of `VarName`s should be respected (see end of this document):

    ```julia
    vi[@varname(x.a)] = [1,2,3]
    vi[@varname(x.b)] = [4,5,6]
    vi[@varname(x.a[2])] == 2
    vi[@varname(x)] == (; a = [1,2,3], b = [4,5,6])
    ```
    
    Generalizations that go beyond simple cases (those that you can imagine by storing individual
    `setfield!`s in a tree) need not be implemented in the beginning; e.g.,

    ```julia
    vi[@varname(x[1])] = 1
    vi[@varname(x[2])] = 2
    keys(vi) == [x[1], x[2]]
    
    vi[@varname(x)] = [1,2]
    keys(vi) == [x]
    ```
    
2. (_This has to be discussed further._)  Information other than the sampled values, such as flags,
   metadata, pointwise likelihoods, etc., can in principle be stored in multiple of these “`VarInfo`
   dicts” with parallel structure.  For efficiency, it is thinkable to devise a design such that
   multiple fields can be stored under the same indexing structure.

    ```julia
    vi[@varname(x[1])] == 1
    vi[@varname(x[1])].meta["bla"] == false
    ```
    
    or something in that direction.
    
    (This is logically equivalent to a dictionary with named tuple values.  Maybe we can do what
    [`DictTable`](https://github.com/JuliaData/TypedTables.jl/blob/main/src/DictTable.jl) does?)
    
    The old `order` field, indicating at which position in the evaluator function a variable has
    been added (essentially a counter of insertions) can actually be left out completely, since the
    dictionary is specified to be ordered by insertion.
    
    The important question here is: should the “joint data structure” behave like a dictionary of
    `NamedTuple`s (`eltype(vi) == @NamedTuple{value::T, ℓ::Float64, meta}`), or like a struct of
    dicts with shared keys (`eltype(vi.value) <: T`, `eltype(vi.ℓ) <: Float64`, …)?
    
The required dictionary functions are about the following:

- Pure functions: 
  - `iterate`, yielding pairs of `VarName` and the stored value
  - `IteratorEltype == HasEltype()`, `IteratorSize = HasLength()`
  - `keys`, `values`, `pairs`, `length` consistent with `iterate`
  - `eltype`, `keytype`, `valuetype`
  - `get`, `getindex`, `haskey` for indexing by `VarName`
  - `merge` to join two `VarInfo`s
- Mutating functions:
  - `insert!!`, `set!!`
  - `merge!!` to add and join elements (TODO: think about `merge`)
  - `setindex!!`
  - `empty!!`, `delete!!`, `unset!!` (_Are these really used anywhere? Not having them makes persistent
    implementations much easier!_)
    
I believe that adopting the interface of
[Dictionaries.jl](https://github.com/andyferris/Dictionaries.jl), not `Base.AbstractDict`, would be
ideal, since their approach make key sharing and certain operations naturally easy (particularly
“broadcast-style”, i.e., transformations on the values, but not the keys).

Other `Base` functions, like `enumerate`, should follow from the above.

`length` might appear weird – but it should definitely be consistent with the iterator.
  
It would be really cool if `merge` supported the combination of distinct types of implementations,
e.g., a dynamic and a tuple-based part.

To support both mutable and immutable/persistent implementations, let’s require consistent
BangBang.jl style mutators throughout.


## Transformations/Bijectors

Transformations should ideally be handled explicitely and from outside: automatically by the
compiler macro, or at the places required by samplers.

Implementation-wise, they can probably be expressed as folds?

```julia
map(v -> link(v.dist, v.value), vi)
```


## Linearization

There are multiple possible approaches to handle this:

1. As a special case of conversion: `Vector(vi)`
2. `copy!(vals_array, vi)`.
3. As a fold: `mapreduce(v -> vec(v.value), append!, vi, init=Float64[])`

Also here, I think that the best implementation would be through a fold.  Variants (1) or (2) might
additionally be provided as syntactic sugar.


---

# `VarName`-based axioms

What follows is mostly an attempt to formalize subsumption.

First, remember that in Turing.jl we can always work with _concretized_ `VarName`s: `begin`/`end`,
`:`, and boolean indexing are all turned into some form of concrete cartesian or array indexing
(assuming [this suggestion](https://github.com/TuringLang/AbstractPPL.jl/issues/35) being
implemented).  This makes all index comparisons static.

Now, `VarName`s have a compositional structure: they can be built by composing a root variable with
more and more lenses (`VarName{v}()` starts off with an `IdentityLens`):

```julia
julia> vn = VarName{:x}() ∘ Setfield.IndexLens((1:10, 1) ∘ Setfield.IndexLens((2, )))
x[1:10,1][2]
```

(_Note that the composition function, `∘`, is really in wrong order; but this is a heritage of
Setfield.jl._)

By “subsumption”, we mean the notion of a `VarName` expressing a more nested path than another one:

```julia
subsumes(@varname(x.a), @varname(x.a[1]))
@varname(x.a) ⊒ @varname(x.a[1]) # \sqsupseteq
@varname(x.a) ⋢ @varname(x.a[1]) # \nsqsubseteq
```

Thus, we have the following axioms for `VarName`s (“variables” are `VarName{n}()`):

1. `x ⊑ x` for all variables `x`
2. `x ≍ y` for `x ≠ y` (i.e., distinct variables are incomparable; `x ⋢ y` and `y ⋢ x`) (`≍` is `\asymp`)
3. `x ∘ ℓ ⊑ x` for all variables `x` and lenses `ℓ`
4. `x ∘ ℓ₁ ⊑ x ∘ ℓ₂ ⇔ ℓ₁ ⊑ ℓ₂`

For the last axiom to work, we also have to define subsumption of individual, non-composed lenses:

1. `PropertyLens(a) == PropertyLens(b) ⇔ a == b`, for all symbols `a`, `b`
2. `FunctionLens(f) == FunctionLens(g) ⇔ f == g` (under extensional equality; I’m only mentioning
   this in case we ever generalize to Bijector-ed variables like `@varname(log(x))`)
3. `IndexLens(ι₁) ⊑ IndexLens(ι₂)` if the index tuple `ι₂` covers all indices in `ι₁`; for example,
   `_[1, 2:10] ⊑ _[1:10, 1:20]`.  (_This is a bit fuzzy and not all corner cases have been
   considered yet!_)
4. `IdentityLens() == IdentityLens()`
4. `ℓ₁ ≍ ℓ₂`, otherwise

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
is “unification”:

```julia
vi = insert!!(vi, vn ∘ ℓ₁, a)
vi = insert!!(vi, vn ∘ ℓ₂, b)
get(vi[vn], ℓ₁) == a
get(vi[vn], ℓ₂) == b
```

where `vn ∘ ℓ₁` and `vn ∘ ℓ₂` need to be recognized as “children” of a common parent `vn`.
