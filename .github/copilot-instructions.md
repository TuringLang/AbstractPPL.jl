# AbstractPPL.jl

AbstractPPL.jl is a Julia package that provides abstract interfaces and APIs for probabilistic programming languages. It defines core abstractions like `VarName` for variable names, `AbstractProbabilisticProgram` for probabilistic models, and utilities for managing traces and variable values.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

- Bootstrap, build, and test the repository:
  - Julia is pre-installed (version 1.11.6)
  - Navigate to the repository: `cd /home/runner/work/AbstractPPL.jl/AbstractPPL.jl`
  - **Package instantiation**: `julia --project=. -e "using Pkg; Pkg.instantiate()"` -- takes 2-3 minutes on first run due to dependency downloads. NEVER CANCEL. Set timeout to 300+ seconds.
  - **Main tests**: `julia --project=. -e "using Pkg; Pkg.test()"` -- takes 3 minutes including precompilation. NEVER CANCEL. Set timeout to 300+ seconds.
  - **Tests only (skip precompilation)**: `julia --project=test -e "using Pkg; Pkg.develop(path=\".\"); ENV[\"GROUP\"] = \"Tests\"; include(\"test/runtests.jl\")"` -- takes 30 seconds. Set timeout to 60+ seconds.
  - **Doctests only**: `julia --project=test -e "ENV[\"GROUP\"] = \"Doctests\"; include(\"test/runtests.jl\")"` -- takes 15 seconds. Set timeout to 60+ seconds.

- Build documentation:
  - Navigate to docs directory: `cd docs`  
  - **Documentation build**: `julia --project=. -e "using Pkg; Pkg.instantiate(); include(\"make.jl\")"` -- takes 1-2 minutes. NEVER CANCEL. Set timeout to 180+ seconds.
  - Documentation artifacts are created in `docs/build/`

## Validation

- **ALWAYS run the full test suite** before making any changes to understand baseline: `julia --project=. -e "using Pkg; Pkg.test()"`
- **ALWAYS run formatting check** before committing: `julia --project=. -e "using Pkg; Pkg.add(\"JuliaFormatter\"); using JuliaFormatter; format(\".\", verbose=true)"` -- takes 1 minute. Set timeout to 120+ seconds.
- **Code formatting** is enforced via CI. Check `.JuliaFormatter.toml` for style configuration.
- **Manual validation scenarios**: Always test core functionality after making changes:
  ```julia
  julia --project=. -e "
  using AbstractPPL
  # Test VarName creation and introspection
  vn = @varname x[1,2]
  println(\"VarName: \", vn, \", Symbol: \", getsym(vn))
  # Test hasvalue/getvalue with dictionaries
  vals = Dict(@varname(x) => [1.0, 2.0, 3.0])
  println(\"Has x: \", hasvalue(vals, @varname(x)))
  println(\"Value: \", getvalue(vals, @varname(x)))
  # Test VarName composition
  vn2 = @varname y.a[1]
  println(\"Composed VarName: \", vn2)
  "
  ```
  Expected output should show VarName objects, successful value retrieval, and proper optic handling.
- Test different GROUP environments: `Tests` for main tests, `Doctests` for documentation tests, or `All` for both
- **Integration tests** exist for downstream packages like DynamicPPL.jl - these are run automatically in CI

## Common Tasks

The following commands are essential for working with this codebase:

### Package Development Commands
```bash
# Install dependencies (NEVER CANCEL - 2-3 minutes)
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Run full test suite (NEVER CANCEL - 3 minutes)  
julia --project=. -e "using Pkg; Pkg.test()"

# Run only main tests (30 seconds)
julia --project=test -e "using Pkg; Pkg.develop(path=\".\"); ENV[\"GROUP\"] = \"Tests\"; include(\"test/runtests.jl\")"

# Run only doctests (15 seconds)
julia --project=test -e "ENV[\"GROUP\"] = \"Doctests\"; include(\"test/runtests.jl\")"

# Format code (1 minute)
julia --project=. -e "using Pkg; Pkg.add(\"JuliaFormatter\"); using JuliaFormatter; format(\".\", verbose=true)"

# Build documentation (NEVER CANCEL - 1-2 minutes)
cd docs && julia --project=. -e "using Pkg; Pkg.instantiate(); include(\"make.jl\")"
```

### Repository Structure
```
.
├── .github/workflows/     # CI/CD workflows (CI, formatting, docs, integration tests)
├── .JuliaFormatter.toml   # Code formatting configuration
├── docs/                  # Documentation source and build
├── ext/                   # Package extensions (DistributionsExt)
├── src/                   # Main package source code
├── test/                  # Test files and test dependencies
├── Project.toml           # Main package dependencies
└── README.md             # Package documentation
```

### Key Source Files
- `src/AbstractPPL.jl` - Main module and exports
- `src/varname.jl` - VarName implementation and utilities  
- `src/abstractprobprog.jl` - AbstractProbabilisticProgram interface
- `src/hasvalue.jl` - Functions for checking/extracting values from containers
- `src/abstractmodeltrace.jl` - Abstract trace interface
- `ext/AbstractPPLDistributionsExt.jl` - Extensions for working with Distributions.jl

### Key Test Files  
- `test/runtests.jl` - Main test runner (supports GROUP environment variable)
- `test/varname.jl` - Tests for VarName functionality
- `test/hasvalue.jl` - Tests for hasvalue/getvalue functions
- `test/abstractprobprog.jl` - Tests for AbstractProbabilisticProgram interface

## Timing Expectations

**CRITICAL**: Set appropriate timeouts for all commands. DO NOT use default timeouts.

- **Package instantiation**: 2-3 minutes (downloads dependencies) - Set timeout to 300+ seconds
- **Full test suite**: 3 minutes (includes precompilation) - Set timeout to 300+ seconds  
- **Main tests only**: 30 seconds - Set timeout to 60+ seconds
- **Doctests only**: 15 seconds - Set timeout to 60+ seconds
- **Code formatting**: 1 minute - Set timeout to 120+ seconds
- **Documentation build**: 1-2 minutes - Set timeout to 180+ seconds

**NEVER CANCEL** any of these operations. Julia package operations can appear to hang but are often downloading dependencies or precompiling.

## Troubleshooting

- **Network issues**: Julia package downloads may fail due to network restrictions. This is expected in CI environments.
- **Precompilation warnings**: Various deprecation warnings during precompilation are normal and expected.
- **Git warnings**: Documentation build may show git remote warnings - these are harmless.
- **Test environment**: Some operations require using `--project=test` to access test dependencies like Documenter.jl.
- **JuliaFormatter**: Must be added fresh each time with `Pkg.add("JuliaFormatter")` as it's not a permanent dependency.
- **Package resolution warnings**: "could not download" warnings from pkg.julialang.org are expected in restricted environments.
- **Manifest.toml changes**: Installing JuliaFormatter temporarily modifies Project.toml - always run `git checkout -- Project.toml` after formatting.

## Performance Notes

- **First run**: Package instantiation and first test run take longer due to compilation and downloads
- **Subsequent runs**: Tests run faster (~25 seconds) when packages are already compiled
- **Parallel testing**: The test suite runs tests in parallel, which is why it completes quickly
- **Memory usage**: Julia precompilation can use significant memory - this is normal

## CI/CD Information

The repository uses GitHub Actions with these key workflows:
- **CI.yml**: Runs tests on multiple Julia versions and platforms
- **Format.yml**: Enforces code formatting with JuliaFormatter
- **Docs.yml**: Builds and deploys documentation
- **DocTests.yml**: Runs doctests separately  
- **IntegrationTest.yml**: Tests against downstream packages (DynamicPPL.jl)

Always run `julia --project=. -e "using JuliaFormatter; format(\".\")"` before committing to pass the Format.yml check.

## Julia Package Development Concepts

For developers new to Julia packages:
- **Project.toml**: Defines package dependencies and metadata (like package.json or requirements.txt)
- **Manifest.toml**: Lock file for exact dependency versions (automatically generated)
- **--project=.**: Tells Julia to use the current directory's Project.toml environment
- **--project=test**: Uses the test/ subdirectory environment with test-specific dependencies
- **Package extensions**: Code in ext/ that's loaded only when optional dependencies are available (like DistributionsExt)
- **using Pkg**: Julia's built-in package manager (like npm, pip, etc.)
- **Precompilation**: Julia compiles packages on first use for faster subsequent loading