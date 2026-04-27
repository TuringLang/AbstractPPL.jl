# Run a named AD-extension test in its own isolated Julia environment.
#
# Usage (from the repo root):
#   LABEL=enzyme                    julia test/run_ext_tests.jl
#   LABEL=differentiation_interface julia test/run_ext_tests.jl
#   LABEL=logdensityproblems        julia test/run_ext_tests.jl
#   LABEL=reversediff               julia test/run_ext_tests.jl

const TEST_SUBDIRS = (
    enzyme="integration",
    differentiation_interface="ext",
    logdensityproblems="ext",
    reversediff="integration",
)
const VALID_LABELS = string.(keys(TEST_SUBDIRS))

label = get(ENV, "LABEL", nothing)
label === nothing && error("Set LABEL to one of: $(join(VALID_LABELS, ", "))")
label in VALID_LABELS ||
    error("Unknown LABEL=$label. Valid options: $(join(VALID_LABELS, ", "))")

subdir = TEST_SUBDIRS[Symbol(label)]
include(joinpath(@__DIR__, subdir, label, label * ".jl"))
