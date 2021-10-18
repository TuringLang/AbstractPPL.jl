# VarInfo interface proposal

The basic idea is for all `VarInfos` to behave like dictionaries with `VarName` keys – all common
operations should just work.  There are two things that make them more special, though:

1. “Fancy indexing”: since `VarName`s are structured themselves, the VarInfo should be have a bit
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
    
2. (This is to be discussed.)  Information other than the sampled values, such as flags, orders,
   variable likelihoods, etc., can in principle be stored in multiple of these “VarInfo dicts” with
   parallel structure.  For efficiency, it is thinkable to devise a design such that multiple fields
   can be stored under the same indexing structure.

    ```julia
    vi[@varname(x[1])] == 1
    vi[@varname(x[1])].meta["bla"] == false
    ```
    
    or something in that direction.
    
    (This is logically equivalent to a dictionary with named tuple values.  Maybe we can do what
    [`DictTable`](https://github.com/JuliaData/TypedTables.jl/blob/main/src/DictTable.jl) does?)
    
    The old `order` field, indicating at which position in the evaluator function a variable has
    been added (essentially a count of `push!`es) could be left out completely if the dictionary is
    specified to be ordered by insertion.
    
The required dictionary functions are about these:

- Pure functions: 
  - `iterate`, yielding pairs of `VarName` and the stored value
  - `IteratorEltype == HasEltype()`, `IteratorSize = HasLength()`
  - `keys`, `values`, `pairs`, `length` consistent with `iterate`
  - `eltype`, `keytype`, `valuetype`
  - `get`, `getindex`, `haskey` for indexing by `VarName`
  - `merge` to join two VarInfos
- Mutating functions:
  - `push!!`, `merge!!` to add and join elements (TODO: think about `merge`)
  - `setindex!!`
  - `empty!!`, `delete!!` (_Are these really used anywhere? Not having them makes persistent
    implementations much easier!_)

Other functions, like `enumerate`, should follow from these.

`length` gets weird, though – but it should definitely be consistent with the iterator.  For this
and other purposes, there should be a separate leaf iterator.
  
It would be really cool if `merge` supported the combination of distinct types of implementations,
e.g., a dynamic and a tuple-based part.

To support both mutable and immutable/persistent implementations, let’s require consistent BangBang
style mutators throughout.
  
## Transformations/Bijectors

Transformations should ideally be handled explicitely and from outside. 

Also via fold?

```julia
mapreduce((k, v) -> k => biject(v), push!, vi, init=VarInfo())
```

## Linearization

There are multiple possible approaches to handle this:

1. As a special case of conversion: `Vector(vi)`
2. `copy!(vals_array, vi)`.
3. As a catamorphism: `mapreduce(flatten, append!, vi, init=Float64[])`

---

# `VarName`-based axioms

What follows is mostly an attempt to formalize subsumption.

First, remember that in Turing.jl we can always work with _concretized_ `VarName`s: `begin`/`end`,
`:`, and boolean indexing are all turned into some form of concrete cartesian or array indexing
(assuming [this suggestion](https://github.com/TuringLang/AbstractPPL.jl/issues/35) being
implemented).  This makes all index comparisons static.

Now, `VarName`s have a compositional structure: they can be built by composing a root variable with
more and more lenses (`VarName{v}()` starts of with an `IdentityLens`):

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
vi[x ∘ ℓ] == get(vi[x], ℓ)`
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


