# Run a named extension test in its own isolated Julia environment.
#
# Usage (from the repo root):
#   LABEL=ext/logdensityproblems julia test/run_extras.jl

const VALID_LABELS = ("ext/logdensityproblems",)

label = get(ENV, "LABEL", nothing)
label === nothing && error("Set LABEL to one of: $(join(VALID_LABELS, ", "))")
label in VALID_LABELS ||
    error("Unknown LABEL=$label. Valid options: $(join(VALID_LABELS, ", "))")

include(joinpath(@__DIR__, label, "main.jl"))
