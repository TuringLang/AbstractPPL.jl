# VarNamedTuple

`VarNamedTuple` is a lossless, `VarName`-keyed value container. It preserves nested
property and array structure while allowing probabilistic programming packages to exchange
values without depending on one frontend's model representation.

Use [`@vnt`](@ref) for literal construction and a `VarName` to access a value. Functional
updates are available through BangBang's `setindex!!` interface.

```@example varnamedtuple
using AbstractPPL

values = @vnt begin
    location := 1.0
    scale := 2.0
end
values[@varname(location)]
```

## API

```@docs
VarNamedTuple
AbstractPPL.@vnt
subset
map_pairs!!
map_values!!
apply!!
densify!!
skeleton
NoTemplate
SkipTemplate
AbstractPPL.templated_setindex!!
```
