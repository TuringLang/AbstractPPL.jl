# Run a named extension test in its own isolated Julia environment.
#
# Usage (from the repo root):
#   LABEL=ext/differentiationinterface julia test/run_extras.jl

const VALID_LABELS = ("ext/differentiationinterface",)

label = get(ENV, "LABEL", nothing)
label in VALID_LABELS ||
    error("Set LABEL to one of: $(join(VALID_LABELS, ", ")) (got `$label`).")

include(joinpath(@__DIR__, label, "main.jl"))
