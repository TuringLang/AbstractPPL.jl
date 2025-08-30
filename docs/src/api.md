# API

## VarNames

```@docs
VarName
getsym
getoptic
inspace
subsumes
subsumedby
vsym
@varname
@vsym
```

## VarName prefixing and unprefixing

```@docs
prefix
unprefix
```

## Extracting values corresponding to a VarName

```@docs
hasvalue
getvalue
```

## Splitting VarNames up into components

```@docs
varname_leaves
varname_and_value_leaves
```

## VarName serialisation

```@docs
index_to_dict
dict_to_index
varname_to_string
string_to_varname
```

## Abstract model functions

```@docs
AbstractProbabilisticProgram
condition
decondition
fix
unfix
logdensityof
AbstractContext
evaluate!!
```

## Abstract traces

```@docs
AbstractModelTrace
```
