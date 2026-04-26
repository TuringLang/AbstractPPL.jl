# Run a named AD-extension test in its own isolated Julia environment.
#
# Usage (from the repo root):
#   LABEL=forward_diff              julia test/run_ext_tests.jl
#   LABEL=mooncake                  julia test/run_ext_tests.jl
#   LABEL=enzyme                    julia test/run_ext_tests.jl
#   LABEL=differentiation_interface julia test/run_ext_tests.jl
#   LABEL=logdensityproblems        julia test/run_ext_tests.jl

const VALID_LABELS = [
    "forward_diff",
    "mooncake",
    "enzyme",
    "differentiation_interface",
    "finite_differences",
    "logdensityproblems",
]

label = get(ENV, "LABEL", nothing)
label === nothing && error("Set LABEL to one of: $(join(VALID_LABELS, ", "))")
label in VALID_LABELS ||
    error("Unknown LABEL=$label. Valid options: $(join(VALID_LABELS, ", "))")

include(joinpath(@__DIR__, "ext", label, label * ".jl"))
