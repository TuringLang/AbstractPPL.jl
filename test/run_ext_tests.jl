# Run a named AD-extension test in its own isolated Julia environment.
#
# Usage (from the repo root):
#   LABEL=forward_diff              julia test/run_ext_tests.jl
#   LABEL=mooncake                  julia test/run_ext_tests.jl
#   LABEL=enzyme                    julia test/run_ext_tests.jl
#   LABEL=differentiation_interface julia test/run_ext_tests.jl
#   LABEL=finite_differences        julia test/run_ext_tests.jl
#   LABEL=logdensityproblems        julia test/run_ext_tests.jl

const TEST_SUBDIRS = (
    forward_diff="ext",
    mooncake="ext",
    enzyme="integration",
    differentiation_interface="ext",
    finite_differences="ext",
    logdensityproblems="ext",
)
const VALID_LABELS = string.(keys(TEST_SUBDIRS))

label = get(ENV, "LABEL", nothing)
label === nothing && error("Set LABEL to one of: $(join(VALID_LABELS, ", "))")
label in VALID_LABELS ||
    error("Unknown LABEL=$label. Valid options: $(join(VALID_LABELS, ", "))")

subdir = TEST_SUBDIRS[Symbol(label)]
include(joinpath(@__DIR__, subdir, label, label * ".jl"))
