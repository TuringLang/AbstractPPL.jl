# Run a named extension test in its own isolated Julia environment.
#
# Usage (from the repo root):
#   LABEL=logdensityproblems julia test/run_extras.jl

const TEST_SUBDIRS = (logdensityproblems="ext",)
const VALID_LABELS = string.(keys(TEST_SUBDIRS))

label = get(ENV, "LABEL", nothing)
label === nothing && error("Set LABEL to one of: $(join(VALID_LABELS, ", "))")
label in VALID_LABELS ||
    error("Unknown LABEL=$label. Valid options: $(join(VALID_LABELS, ", "))")

subdir = TEST_SUBDIRS[Symbol(label)]
include(joinpath(@__DIR__, subdir, label, label * ".jl"))
