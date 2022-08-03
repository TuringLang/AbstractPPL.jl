module AbstractPPL

# VarName
export VarName, getsym, getlens, inspace, subsumes, varname, vsym, @varname, @vsym


# Abstract model functions
export AbstractProbabilisticProgram, condition, decondition, logdensityof, densityof, AbstractContext, evaluate!!

# Abstract traces
export AbstractModelTrace


include("varname.jl")
include("abstractmodeltrace.jl")
include("abstractprobprog.jl")
include("evaluate.jl")
include("deprecations.jl")

# GraphInfo
module GraphPPL
    include("graphinfo.jl")    
end

end # module
