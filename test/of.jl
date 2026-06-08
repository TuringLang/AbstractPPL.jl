using Test

using AbstractPPL
using AbstractPPL:
    OfType,
    OfInt,
    OfReal,
    OfArray,
    OfNamedTuple,
    OfConstantWrapper,
    SymbolicRef,
    SymbolicExpr,
    get_names,
    get_types,
    flatten,
    unflatten,
    has_symbolic_dims,
    get_unresolved_symbols,
    get_dims,
    get_element_type,
    get_ndims,
    get_lower,
    get_upper

using Random: MersenneTwister

@testset "Basic type creation" begin
    @testset "Simple type creation" begin
        @test of(Int) == OfInt{Nothing,Nothing}
        @test of(Int, 0, 10) == OfInt{0,10}
        @test of(Real) == OfReal{Float64,Nothing,Nothing}
        @test of(Real, 0.0, 1.0) == OfReal{Float64,0.0,1.0}

        @test of(Array, 5) == OfArray{Float64,1,Tuple{5}}
        @test of(Array, 3, 4) == OfArray{Float64,2,Tuple{3,4}}
        @test of(Array, Int, 2, 2) == OfArray{Int,2,Tuple{2,2}}
    end

    @testset "Symbolic bounds" begin
        T1 = of(Real, :lower, :upper)
        @test T1 == OfReal{Float64,SymbolicRef{:lower},SymbolicRef{:upper}}

        T2 = of(Int, 0, :max)
        @test T2 == OfInt{0,SymbolicRef{:max}}

        T3 = of(Real, :min, :max; constant=true)
        @test T3 == OfConstantWrapper{OfReal{Float64,SymbolicRef{:min},SymbolicRef{:max}}}
    end

    @testset "Explicit float types" begin
        @test of(Float64) == OfReal{Float64,Nothing,Nothing}
        @test of(Float64, 0.0, 1.0) == OfReal{Float64,0.0,1.0}
        @test of(Float64; constant=true) ==
            OfConstantWrapper{OfReal{Float64,Nothing,Nothing}}

        @test of(Float32) == OfReal{Float32,Nothing,Nothing}
        @test of(Float32, -1.0f0, 1.0f0) == OfReal{Float32,-1.0f0,1.0f0}
        @test of(Float32; constant=true) ==
            OfConstantWrapper{OfReal{Float32,Nothing,Nothing}}

        @test of(Real) == OfReal{Float64,Nothing,Nothing}

        @test rand(of(Float64)) isa Float64
        @test rand(of(Float32)) isa Float32
        @test rand(of(Real)) isa Float64

        @test zero(of(Float64)) isa Float64
        @test zero(of(Float32)) isa Float32
        @test zero(of(Real)) isa Float64

        val64 = rand(of(Float64, 0.0, 1.0))
        @test val64 isa Float64
        @test 0.0 <= val64 <= 1.0

        val32 = rand(of(Float32, -1.0f0, 1.0f0))
        @test val32 isa Float32
        @test -1.0f0 <= val32 <= 1.0f0
    end
end

@testset "@of macro tests" begin
    @testset "Basic constant syntax" begin
        T1 = of(Int; constant=true)
        @test T1 == OfConstantWrapper{OfInt{Nothing,Nothing}}
        @test string(T1) == "of(Int; constant=true)"

        T2 = of(Real; constant=true)
        @test T2 == OfConstantWrapper{OfReal{Float64,Nothing,Nothing}}
        @test string(T2) == "of(Real; constant=true)"

        T3 = of(Int)
        @test T3 == OfInt{Nothing,Nothing}

        T4 = of(Real, 0, 10)
        @test T4 == OfReal{Float64,0,10}

        @test_throws ErrorException of(Array, 10; constant=true)
        @test_throws ErrorException of(Array, Float64, 5, 5; constant=true)
    end

    @testset "Simple @of macro" begin
        T = @of(mu = of(Real), sigma = of(Real, 0, nothing), data = of(Array, 10))

        @test T <: OfNamedTuple
        names = get_names(T)
        @test names == (:mu, :sigma, :data)

        types = get_types(T)
        @test types.parameters[1] == OfReal{Float64,Nothing,Nothing}
        @test types.parameters[2] == OfReal{Float64,0,Nothing}
        @test types.parameters[3] == OfArray{Float64,1,Tuple{10}}
    end

    @testset "@of with constants and references" begin
        T = @of(
            rows = of(Int; constant=true),
            cols = of(Int; constant=true),
            data = of(Array, rows, cols)
        )

        @test T <: OfNamedTuple
        names = get_names(T)
        @test names == (:rows, :cols, :data)

        types = get_types(T)
        @test types.parameters[1] == OfConstantWrapper{OfInt{Nothing,Nothing}}
        @test types.parameters[2] == OfConstantWrapper{OfInt{Nothing,Nothing}}
        @test types.parameters[3] == OfArray{Float64,2,Tuple{:rows,:cols}}
    end

    @testset "@of with expressions" begin
        # Arithmetic expressions in dimensions are supported and encode as SymbolicExpr.
        T = @of(n = of(Int; constant=true), data = of(Array, n + 1, 2 * n))
        @test T <: OfNamedTuple
        CT = of(T; n=5)
        @test get_dims(get_types(CT).parameters[1]) == (6, 10)
    end

    @testset "@of with float types" begin
        T = @of(
            f64_val = of(Float64),
            f32_val = of(Float32, 0.0f0, 1.0f0),
            real_val = of(Real; constant=true),
            f64_array = of(Array, Float64, 3),
            f32_array = of(Array, Float32, 2, 2)
        )

        types = get_types(T)
        @test types.parameters[1] == OfReal{Float64,Nothing,Nothing}
        @test types.parameters[2] == OfReal{Float32,0.0f0,1.0f0}
        @test types.parameters[3] == OfConstantWrapper{OfReal{Float64,Nothing,Nothing}}
        @test types.parameters[4] == OfArray{Float64,1,Tuple{3}}
        @test types.parameters[5] == OfArray{Float32,2,Tuple{2,2}}

        instance = T(; real_val=5.0)
        @test instance.f64_val isa Float64
        @test instance.f32_val isa Float32
        @test instance.f64_array isa Vector{Float64}
        @test instance.f32_array isa Matrix{Float32}
    end

    @testset "Concrete instance creation" begin
        MatrixType = @of(
            rows = of(Int; constant=true),
            cols = of(Int; constant=true),
            data = of(Array, rows, cols)
        )

        instance = MatrixType(; rows=3, cols=4)

        @test instance isa NamedTuple
        @test keys(instance) == (:data,)
        @test instance.data isa Matrix{Float64}
        @test size(instance.data) == (3, 4)
        @test all(instance.data .== 0.0)

        test_data = rand(3, 4)
        instance2 = MatrixType(; rows=3, cols=4, data=test_data)
        @test instance2.data ≈ test_data
    end

    @testset "rand and zero with constants" begin
        T = @of(n = of(Int; constant=true), data = of(Array, n))

        CT = of(T; n=5)
        val = rand(CT)
        @test haskey(val, :data)
        @test size(val.data) == (5,)

        CT2 = of(T; n=3)
        val = zero(CT2)
        @test haskey(val, :data)
        @test size(val.data) == (3,)
        @test all(val.data .== 0.0)
    end

    @testset "flatten/unflatten preserves array types" begin
        T = @of(
            rows = of(Int; constant=true),
            cols = of(Int; constant=true),
            data = of(Array, rows, cols)
        )
        ConcreteType = of(T; rows=2, cols=3)

        original = (data=rand(Float64, 2, 3),)

        flat = flatten(ConcreteType, original)
        reconstructed = unflatten(ConcreteType, flat)

        @test typeof(reconstructed.data) == typeof(original.data)
        @test typeof(reconstructed.data) == Matrix{Float64}
        @test reconstructed.data ≈ original.data
    end

    @testset "flatten/unflatten with concrete types" begin
        T = @of(
            rows = of(Int; constant=true),
            cols = of(Int; constant=true),
            scale = of(Real, 0.1, 10.0),
            data = of(Array, rows, cols)
        )

        instance = T(; rows=3, cols=2)
        @test instance isa NamedTuple
        @test keys(instance) == (:scale, :data)
        @test size(instance.data) == (3, 2)

        CT = of(T; rows=3, cols=2)

        flat = flatten(CT, instance)
        @test length(flat) == 7  # 1 scale + 6 data elements

        reconstructed = unflatten(CT, flat)
        @test reconstructed.scale ≈ instance.scale
        @test reconstructed.data ≈ instance.data

        instance2 = (scale=2.5, data=rand(3, 2))
        flat2 = flatten(CT, instance2)
        reconstructed2 = unflatten(CT, flat2)
        @test reconstructed2.scale ≈ 2.5
        @test reconstructed2.data ≈ instance2.data
    end
end

@testset "Symbolic bounds tests" begin
    @testset "Symbolic references in @of macro" begin
        T = @of(min = of(Real, 0, 10), max = of(Real, 20, 30), value = of(Real, min, max))

        types = get_types(T)
        @test types.parameters[3] == OfReal{Float64,SymbolicRef{:min},SymbolicRef{:max}}
    end

    @testset "Symbolic bounds in named tuples" begin
        T = @of(
            lower_bound = of(Real, 0, nothing),
            upper_bound = of(Real, lower_bound, nothing),
            param = of(Real, lower_bound, upper_bound),
        )

        types = get_types(T)
        @test types.parameters[1] == OfReal{Float64,0,Nothing}
        @test types.parameters[2] == OfReal{Float64,SymbolicRef{:lower_bound},Nothing}
        @test types.parameters[3] ==
            OfReal{Float64,SymbolicRef{:lower_bound},SymbolicRef{:upper_bound}}
    end

    @testset "@of macro with symbolic bounds" begin
        Schema = @of(
            min_val = of(Real, 0, 10),
            max_val = of(Real, min_val, 100),
            param = of(Real, min_val, max_val)
        )

        types = get_types(Schema)
        @test types.parameters[1] == OfReal{Float64,0,10}
        @test types.parameters[2] == OfReal{Float64,SymbolicRef{:min_val},100}
        @test types.parameters[3] ==
            OfReal{Float64,SymbolicRef{:min_val},SymbolicRef{:max_val}}
    end

    @testset "Symbolic bounds with constants" begin
        Schema = @of(
            lower = of(Real, 0, nothing; constant=true),
            upper = of(Real, lower, nothing; constant=true),
            x = of(Real, lower, upper)
        )

        types = get_types(Schema)
        @test types.parameters[1] == OfConstantWrapper{OfReal{Float64,0,Nothing}}
        @test types.parameters[2] ==
            OfConstantWrapper{OfReal{Float64,SymbolicRef{:lower},Nothing}}
        @test types.parameters[3] == OfReal{Float64,SymbolicRef{:lower},SymbolicRef{:upper}}
    end

    @testset "Concrete instance creation with symbolic resolution" begin
        Schema = @of(
            min_bound = of(Real; constant=true),
            max_bound = of(Real; constant=true),
            value = of(Real, min_bound, max_bound)
        )

        instance = Schema(; min_bound=0.0, max_bound=1.0)

        @test instance isa NamedTuple
        @test keys(instance) == (:value,)
        @test instance.value == 0.0

        instance2 = Schema(; min_bound=0.0, max_bound=1.0, value=0.5)
        @test instance2.value == 0.5
    end

    @testset "Validation with symbolic bounds" begin
        # A symbolic bound is resolved from the constant during construction, then the
        # provided value is validated against the resolved bound.
        Schema = @of(
            threshold = of(Real, 0, nothing; constant=true), value = of(Real, 0, threshold)
        )

        instance = Schema(; threshold=10.0, value=5.0)
        @test keys(instance) == (:value,)
        @test instance.value == 5.0

        # value above the resolved upper bound (threshold) must throw
        @test_throws ErrorException Schema(; threshold=10.0, value=15.0)
        # value below the lower bound must throw
        @test_throws ErrorException Schema(; threshold=10.0, value=-1.0)
    end
end

@testset "Constant Elimination After Concretization" begin
    @testset "Basic elimination" begin
        T = @of(
            n = of(Int; constant=true), m = of(Int; constant=true), data = of(Array, n, m)
        )

        CT = of(T; n=3, m=4)

        @test get_names(CT) == (:data,)

        types = get_types(CT)
        @test types.parameters[1] == of(Array, 3, 4)

        instance = rand(CT)
        @test instance isa NamedTuple
        @test !haskey(instance, :n)
        @test !haskey(instance, :m)
        @test haskey(instance, :data)
        @test size(instance.data) == (3, 4)

        @test length(CT) == 12  # 3×4 array
    end

    @testset "Constants with bounds" begin
        T = @of(
            lower = of(Int, 1, 10; constant=true),
            upper = of(Int, 50, 100; constant=true),
            value = of(Real, lower, upper)
        )

        CT = of(T; lower=5, upper=75)

        types = get_types(CT)
        @test get_names(CT) == (:value,)
        @test types.parameters[1] == of(Real, 5, 75)
    end

    @testset "Partial concretization" begin
        T = @of(
            a = of(Int; constant=true), b = of(Int; constant=true), data = of(Array, a, b)
        )

        CT = of(T; a=10)

        names = get_names(CT)
        types = get_types(CT)
        # 'a' is eliminated, 'b' still wrapped as constant, data uses concrete 'a'
        @test :a ∉ names
        @test :b ∈ names
        @test :data ∈ names

        b_idx = findfirst(==(Symbol("b")), names)
        @test types.parameters[b_idx] <: OfConstantWrapper

        data_idx = findfirst(==(Symbol("data")), names)
        @test types.parameters[data_idx] == of(Array, 10, :b)
    end

    @testset "Nested structures" begin
        InnerT = @of(size = of(Int; constant=true), values = of(Array, size))

        OuterT = @of(n = of(Int; constant=true), inner = InnerT)

        CT = of(OuterT; n=5, size=3)

        outer_names = get_names(CT)
        @test :n ∉ outer_names
        @test :inner ∈ outer_names

        outer_types = get_types(CT)
        inner_type = outer_types.parameters[1]

        inner_names = get_names(inner_type)
        @test :size ∉ inner_names
        @test :values ∈ inner_names

        inner_types = get_types(inner_type)
        @test inner_types.parameters[1] == of(Array, 3)
    end

    @testset "Symbolic dimension checking" begin
        T = @of(const_field = of(Int; constant=true), regular_field = of(Real, 0, 1))

        @test has_symbolic_dims(T) == true

        CT = of(T; const_field=42)
        @test has_symbolic_dims(CT) == false

        @test get_names(CT) == (:regular_field,)
    end
end

@testset "Multi-hop constant dependencies" begin
    @testset "Chain dependencies" begin
        T = @of(
            a = of(Int, 1, 10; constant=true),
            b = of(Int, a, 20; constant=true),
            c = of(Int, b, 30; constant=true),
            data = of(Array, c, c)
        )

        CT = of(T; a=5, b=10, c=15)

        @test get_names(CT) == (:data,)
        types = get_types(CT)
        @test types.parameters[1] == of(Array, 15, 15)
    end

    @testset "Expression dependencies" begin
        T = @of(
            base = of(Int, 2, 5; constant=true),
            width = of(Int, base, base * 2; constant=true),
            height = of(Int, base, base * 3; constant=true),
            volume = of(Array, width, height, base)
        )

        CT = of(T; base=3, width=5, height=7)

        @test get_names(CT) == (:volume,)
        types = get_types(CT)
        @test types.parameters[1] == of(Array, 5, 7, 3)
    end
end

@testset "Type operations" begin
    @testset "length calculation" begin
        @test length(of(Int)) == 1
        @test length(of(Real)) == 1
        @test length(of(Array, 5)) == 5
        @test length(of(Array, 3, 4)) == 12
        @test length(of(Array, Int, 2, 3, 4)) == 24

        T = @of(a = of(Int), b = of(Real), c = of(Array, 3))
        @test length(T) == 5  # 1 + 1 + 3

        T2 = @of(n = of(Int; constant=true), data = of(Array, n))
        CT = of(T2; n=10)
        @test length(CT) == 10  # only data field remains (n is eliminated)
    end

    @testset "rand generation" begin
        @test rand(of(Int, 1, 10)) isa Int
        @test rand(of(Real, 0.0, 1.0)) isa Float64
        arr = rand(of(Array, 5, 3))
        @test arr isa Matrix{Float64}
        @test size(arr) == (5, 3)

        T = @of(x = of(Real), y = of(Int, 0, 100), z = of(Array, 2, 2))
        instance = rand(T)
        @test instance isa NamedTuple
        @test haskey(instance, :x) && instance.x isa Float64
        @test haskey(instance, :y) && instance.y isa Int
        @test haskey(instance, :z) && instance.z isa Matrix{Float64}
    end

    @testset "zero generation" begin
        @test zero(of(Int)) == 0
        @test zero(of(Real)) == 0.0
        arr = zero(of(Array, 3, 2))
        @test arr isa Matrix{Float64}
        @test all(arr .== 0.0)

        T = @of(a = of(Int), b = of(Real), c = of(Array, 2))
        instance = zero(T)
        @test instance.a == 0
        @test instance.b == 0.0
        @test all(instance.c .== 0.0)
    end

    @testset "flatten/unflatten" begin
        T = @of(
            int_val = of(Int),
            real_val = of(Real),
            vec = of(Array, 3),
            mat = of(Array, 2, 2)
        )

        original = (int_val=42, real_val=3.14, vec=[1.0, 2.0, 3.0], mat=[4.0 5.0; 6.0 7.0])
        flat = flatten(T, original)
        reconstructed = unflatten(T, flat)

        @test reconstructed.int_val == original.int_val
        @test reconstructed.real_val ≈ original.real_val
        @test reconstructed.vec ≈ original.vec
        @test reconstructed.mat ≈ original.mat

        @test length(flat) == length(T)
    end

    @testset "flatten promotes mixed element types" begin
        # A flat vector has a single element type: the promotion of the declared leaf types.
        # Mixing Float32 and Float64 fields therefore yields a Float64 vector, and unflatten
        # reconstructs every float leaf at that promoted precision (the declared type is a
        # floor, not a forced down-conversion — see the AD round-trip test below).
        T = @of(
            f64 = of(Float64, 0.0, 1.0),
            f32 = of(Float32, -1.0f0, 1.0f0),
            f64_vec = of(Array, Float64, 3),
            f32_mat = of(Array, Float32, 2, 2)
        )

        original = (
            f64=0.5, f32=0.25f0, f64_vec=[0.1, 0.2, 0.3], f32_mat=Float32[0.1 0.2; 0.3 0.4]
        )

        flat = flatten(T, original)
        @test flat isa Vector{Float64}  # concrete promoted eltype, not Vector{Real}

        reconstructed = unflatten(T, flat)
        @test reconstructed.f64 isa Float64
        @test reconstructed.f32 isa Float64       # promoted to the flat vector's eltype
        @test reconstructed.f64_vec isa Vector{Float64}
        @test reconstructed.f32_mat isa Matrix{Float64}

        @test reconstructed.f64 ≈ original.f64
        @test reconstructed.f32 ≈ original.f32
        @test reconstructed.f64_vec ≈ original.f64_vec
        @test reconstructed.f32_mat ≈ original.f32_mat
    end

    @testset "flatten preserves a uniform element type" begin
        # When every leaf shares an element type, flatten/unflatten round-trip it exactly.
        T32 = @of(a = of(Float32), v = of(Array, Float32, 2))
        flat32 = flatten(T32, (a=0.5f0, v=Float32[1, 2]))
        @test flat32 isa Vector{Float32}
        r32 = unflatten(T32, flat32)
        @test r32.a isa Float32
        @test r32.v isa Vector{Float32}

        # A pure-integer structure stays integer-typed (no gratuitous Float64 coercion).
        Tint = @of(i = of(Int), w = of(Array, Int, 2))
        flatint = flatten(Tint, (i=3, w=[4, 5]))
        @test flatint isa Vector{Int}
        @test flatint == [3, 4, 5]
    end
end

@testset "Array type specifications" begin
    @testset "Different element types" begin
        T1 = of(Array, Int, 5)
        @test get_element_type(T1) == Int
        @test get_ndims(T1) == 1
        @test get_dims(T1) == (5,)

        T2 = of(Array, Bool, 3, 3)
        @test get_element_type(T2) == Bool
        @test get_ndims(T2) == 2
        @test get_dims(T2) == (3, 3)

        T3 = of(Array, 10)
        @test get_element_type(T3) == Float64
    end

    @testset "Symbolic dimensions in arrays" begin
        T = @of(
            rows = of(Int; constant=true),
            cols = of(Int; constant=true),
            matrix = of(Array, rows, cols),
            tensor = of(Array, rows, cols, 3)
        )

        types = get_types(T)
        mat_type = types.parameters[3]
        @test get_dims(mat_type) == (:rows, :cols)

        tensor_type = types.parameters[4]
        @test get_dims(tensor_type) == (:rows, :cols, 3)

        CT = of(T; rows=2, cols=4)
        # rows and cols are eliminated, only matrix and tensor remain
        @test get_names(CT) == (:matrix, :tensor)
        ct_types = get_types(CT)
        @test get_dims(ct_types.parameters[1]) == (2, 4)
        @test get_dims(ct_types.parameters[2]) == (2, 4, 3)
    end
end

@testset "Constructor with default_value" begin
    @testset "Basic default_value usage" begin
        T = @of(
            rows = of(Int; constant=true),
            cols = of(Int; constant=true),
            scale = of(Real, 0.1, 10.0),
            data = of(Array, rows, cols)
        )

        # No positional arg: each leaf defaults to zero() (or its lower bound).
        instance1 = T(; rows=3, cols=2)
        @test instance1.scale == 0.1
        @test all(instance1.data .== 0.0)

        instance2 = T(1.5; rows=3, cols=2)
        @test instance2.scale == 1.5
        @test all(instance2.data .== 1.5)

        instance3 = T(missing; rows=3, cols=2)
        @test instance3.scale === missing
        @test all(instance3.data .=== missing)

        instance4 = T(2.0; rows=3, cols=2, scale=5.0)
        @test instance4.scale == 5.0
        @test all(instance4.data .== 2.0)
    end

    @testset "Default value validation" begin
        T = @of(n = of(Int; constant=true), bounded = of(Real, 0, 10), data = of(Array, n))

        # Valid default_value within bounds
        instance = T(5.0; n=3)
        @test instance.bounded == 5.0
        @test all(instance.data .== 5.0)

        # Invalid default_value outside bounds should throw
        @test_throws ErrorException T(15.0; n=3)  # 15.0 > upper bound 10
        @test_throws ErrorException T(-5.0; n=3)  # -5.0 < lower bound 0
    end

    @testset "Different types with default_value" begin
        T = @of(
            size = of(Int; constant=true),
            int_val = of(Int, 1, 100),
            real_val = of(Real),
            vec = of(Array, size),
            mat = of(Array, size, size)
        )

        instance1 = T(42; size=2)
        @test instance1.int_val == 42
        @test instance1.real_val == 42.0
        @test all(instance1.vec .== 42.0)
        @test all(instance1.mat .== 42.0)

        instance2 = T(3.14; size=2)
        @test instance2.int_val == 3  # Should round to Int
        @test instance2.real_val ≈ 3.14
        @test all(instance2.vec .≈ 3.14)
        @test all(instance2.mat .≈ 3.14)
    end

    @testset "Nested structures with default_value" begin
        # For nested structures, we need a simpler example
        # The inner structure's constants should be handled at the outer level
        OuterT = @of(n = of(Int; constant=true), scale = of(Real), vec = of(Array, n))

        instance = OuterT(7.0; n=5)
        @test instance.scale == 7.0
        @test instance.vec isa Vector{Float64}
        @test length(instance.vec) == 5
        @test all(instance.vec .== 7.0)
    end

    @testset "Type stability of default_value" begin
        T = @of(n = of(Int; constant=true), data = of(Array, n))

        # The two constructor methods should be type-stable
        CT = of(T; n=5)

        # Method 1: no positional argument
        @inferred NamedTuple{(:data,),Tuple{Vector{Float64}}} T(; n=5)

        # Method 2: with positional default_value
        @inferred NamedTuple{(:data,),Tuple{Vector{Float64}}} T(1.0; n=5)
    end
end

@testset "Edge cases and error handling" begin
    @testset "Invalid bounds" begin
        # Test that invalid bounds are caught during validation
        T = @of(value = of(Real, 0, 10))
        @test_throws ErrorException T(value=15.0)  # value > upper bound
        @test_throws ErrorException T(value=-5.0)  # value < lower bound
    end

    @testset "Missing required constants" begin
        T = @of(n = of(Int; constant=true), data = of(Array, n))

        # Should throw when trying to use without providing constant
        @test_throws ErrorException rand(T)
        @test_throws ErrorException zero(T)

        # Should throw when trying to create instance without providing constant
        @test_throws ErrorException T()
        @test_throws ErrorException T(data=rand(5))
    end

    @testset "Type display" begin
        @test string(of(Int)) == "of(Int)"
        @test string(of(Int, 0, 10)) == "of(Int, 0, 10)"
        @test string(of(Real, 0.0, nothing)) == "of(Float64, 0.0, nothing)"
        @test string(of(Float64, 0.0, nothing)) == "of(Float64, 0.0, nothing)"
        @test string(of(Float32, 0.0f0, nothing)) == "of(Float32, 0.0, nothing)"
        @test string(of(Array, 5)) == "of(Array, 5)"
        @test string(of(Array, Float32, 3, 3)) == "of(Array, Float32, 3, 3)"

        # Test constant wrapper display
        @test string(of(Int; constant=true)) == "of(Int; constant=true)"
        @test string(of(Real, 0, 1; constant=true)) == "of(Real, 0, 1; constant=true)"
        @test string(of(Float64; constant=true)) == "of(Real; constant=true)"
        @test string(of(Float32; constant=true)) == "of(Float32; constant=true)"

        # Test that types without bounds don't show "nothing"
        T = @of(rows = of(Int), cols = of(Int), data = of(Array, 3, 4))
        str = string(T)
        @test occursin("rows = of(Int)", str)
        @test occursin("cols = of(Int)", str)
        @test !occursin("nothing", str)
    end

    @testset "Type inference from values" begin
        @test of(1.0) == of(Float64)
        @test of(1.0f0) == of(Float32)
        @test of(1) == of(Int)
        @test of(1//2) == of(Float64)  # Rationals default to Float64
        @test of(big(1.0)) == of(BigFloat)

        # Test arrays
        @test of([1.0, 2.0, 3.0]) == of(Array, Float64, 3)
        @test of(Float32[1.0, 2.0]) == of(Array, Float32, 2)
        @test of([1 2; 3 4]) == of(Array, Int, 2, 2)
    end
end

@testset "Show method for NamedTuple" begin
    T = @of(x = of(Real), y = of(Int, 0, 10))
    str = string(T)
    @test occursin("@of(", str)
    @test occursin("x=of(Float64)", str)
    @test occursin("y=of(Int, 0, 10)", str)

    # Test multiline display with many fields
    T2 = @of(a = of(Real), b = of(Int), c = of(Array, 3, 4), d = of(Float32, 0.0, 1.0))
    str2 = string(T2)
    @test occursin("@of(", str2)

    # Test with constants
    T3 = @of(n = of(Int; constant=true), data = of(Array, n, 2))
    str3 = string(T3)
    @test occursin("of(Int; constant=true)", str3)
    @test occursin("of(Array, n, 2)", str3)
end

@testset "Type concretization" begin
    T = @of(n = of(Int; constant=true), data = of(Array, n))

    ConcreteT = of(T; n=5)
    @test ConcreteT <: OfNamedTuple
    @test get_names(ConcreteT) == (:data,)
    types = get_types(ConcreteT)
    @test get_dims(types.parameters[1]) == (5,)

    # Test expression dimensions
    T2 = @of(
        n = of(Int; constant=true),
        original = of(Array, n, n),
        padded = of(Array, n + 1, n + 1),
        doubled = of(Array, 2 * n, n)
    )

    ConcreteT2 = of(T2; n=10)
    names = get_names(ConcreteT2)
    @test names == (:original, :padded, :doubled)

    types2 = get_types(ConcreteT2)
    @test get_dims(types2.parameters[1]) == (10, 10)
    @test get_dims(types2.parameters[2]) == (11, 11)
    @test get_dims(types2.parameters[3]) == (20, 10)

    # Test bounded types with symbolic references
    T3 = @of(
        lower = of(Real; constant=true),
        upper = of(Real; constant=true),
        param = of(Real, lower, upper)
    )

    ConcreteT3 = of(T3; lower=0.0, upper=1.0)
    types3 = get_types(ConcreteT3)
    @test get_lower(types3.parameters[1]) == 0.0
    @test get_upper(types3.parameters[1]) == 1.0
end

@testset "Expression processing in @of macro" begin
    T = @of(
        n = of(Int; constant=true),
        data1 = of(Array, n + 1),
        data2 = of(Array, n * 2),
        data3 = of(Array, n - 1),
        data4 = of(Array, n / 2)
    )

    ConcreteT = of(T; n=10)
    types = get_types(ConcreteT)
    @test get_dims(types.parameters[1]) == (11,)
    @test get_dims(types.parameters[2]) == (20,)
    @test get_dims(types.parameters[3]) == (9,)
    @test get_dims(types.parameters[4]) == (5,)

    # Test nested expressions
    T2 = @of(
        a = of(Int; constant=true),
        b = of(Int; constant=true),
        data = of(Array, (a + b) * 2)
    )

    ConcreteT2 = of(T2; a=10, b=4)
    types2 = get_types(ConcreteT2)
    @test get_dims(types2.parameters[1]) == (28,)  # (10+4)*2

    # Test division that requires integer result
    T3 = @of(n = of(Int; constant=true), data = of(Array, n / 3))

    # Should error when n=10 since 10/3 is not an integer
    @test_throws ErrorException of(T3; n=10)

    # But should work when n=9
    ConcreteT3 = of(T3; n=9)
    types3 = get_types(ConcreteT3)
    @test get_dims(types3.parameters[1]) == (3,)
end

# A submodule that does ONLY `using AbstractPPL`, with no access to internal names. This is
# the real downstream scope; the testsets above import `SymbolicExpr` and so cannot catch a
# macro that emits an unqualified reference to it.
module DownstreamScope
using AbstractPPL
using Test

@testset "@of expands in a using-only scope" begin
    # Plain symbolic dimensions (resolved at runtime, no injected type name).
    Tsym = @of(n = of(Int; constant=true), data = of(Array, n, 2))
    @test of(Tsym; n=4) isa Type

    # Arithmetic dimensions inject `SymbolicExpr`, which must be emitted fully qualified so
    # it resolves even though it is only `public`, not exported.
    Texpr = @of(
        n = of(Int; constant=true),
        a = of(Array, n + 1),
        b = of(Array, 2 * n, n),
        c = of(Array, (n + 1) * 2)
    )
    CT = of(Texpr; n=5)
    @test size(rand(CT).a) == (6,)
    @test size(rand(CT).b) == (10, 5)
    @test size(rand(CT).c) == (12,)

    # Symbolic bounds likewise resolve in a using-only scope.
    Tbound = @of(lo = of(Real; constant=true), x = of(Real, lo, nothing))
    @test of(Tbound; lo=0.0) isa Type
end
end # module DownstreamScope

@testset "show is safe for non-concrete types" begin
    # Rendering a free-typevar `of`-type (method signatures, stacktraces, Documenter) must not
    # touch the static params; previously this threw UndefVarError, fatal mid-backtrace.
    @test sprint(show, Tuple{Type{OfArray{T,N,D}}} where {T,N,D}) isa String
    @test sprint(show, Tuple{Type{OfReal{T,L,U}}} where {T,L,U}) isa String
    @test sprint(show, Tuple{Type{OfInt{L,U}}} where {L,U}) isa String
    @test sprint(show, Tuple{Type{OfNamedTuple{Names,Types}}} where {Names,Types}) isa
        String
    # Listing the methods of a function with `of`-typed signatures must not crash.
    @test sprint(show, methods(rand)) isa String

    # Concrete types still print in the pretty `of(...)` form.
    @test string(of(Array, 2, 3)) == "of(Array, 2, 3)"
    @test string(of(Array, Int)) == "of(Array, Int64)"  # 0-dim: no trailing comma
end

@testset "flatten/unflatten numeric contract" begin
    @testset "concrete, promoted element type" begin
        @test flatten(of(Int), 3) isa Vector{Int}
        @test flatten(of(Float64), 1.5) isa Vector{Float64}
        T = @of(i = of(Int), x = of(Real))
        @test flatten(T, (i=2, x=1.5)) isa Vector{Float64}  # promote(Int, Float64)
    end

    @testset "AD/wide eltypes flow through unflatten" begin
        # BigFloat stands in for ForwardDiff.Dual: any `<:Real` wider than the declared type
        # must survive unflatten without being coerced back to Float64.
        T = @of(x = of(Real), data = of(Array, 2, 2))
        flat = BigFloat[big"1.0", big"2.0", big"3.0", big"4.0", big"5.0"]
        r = unflatten(T, flat)
        @test r.x isa BigFloat
        @test r.data isa Matrix{BigFloat}
        @test r.data == BigFloat[2 4; 3 5]
    end

    @testset "declared type is a precision floor" begin
        # Narrow input widens up to the declared float type.
        @test unflatten(of(Array, 2, 2), [1, 3, 2, 4]) isa Matrix{Float64}
        @test unflatten(of(Float64), Float32[0.5]) isa Float64
        # Integer leaves round to Int regardless of input eltype.
        @test unflatten(of(Int), [2.0]) === 2
    end

    @testset "type stability on the sampler path" begin
        T = @of(
            x = of(Real),
            n = of(Int),
            data = of(Array, 2, 2),
            inner = @of(a = of(Real), v = of(Array, 3))
        )
        v = collect(1.0:10.0)
        nt = @inferred unflatten(T, v)
        @inferred flatten(T, nt)
        @inferred length(T)
        @inferred size(T)
        @test flatten(T, unflatten(T, v)) == v
    end

    @testset "length / count errors" begin
        T = @of(x = of(Real), data = of(Array, 2, 2))
        @test length(T) == 5
        @test_throws ErrorException unflatten(T, [1.0, 2.0])           # too few
        @test_throws ErrorException unflatten(T, collect(1.0:6.0))     # too many
    end
end

@testset "rand threads an explicit RNG" begin
    T = @of(
        x = of(Real, 0, 1), n = of(Int, 1, 10), v = of(Array, 3), inner = @of(a = of(Real))
    )
    @test rand(MersenneTwister(42), T) == rand(MersenneTwister(42), T)
    @test rand(MersenneTwister(1), of(Float64, 0, 1)) ==
        rand(MersenneTwister(1), of(Float64, 0, 1))
    @test rand(MersenneTwister(1), of(Array, 2, 2)) ==
        rand(MersenneTwister(1), of(Array, 2, 2))
    # The RNG-accepting method is what downstream samplers dispatch on.
    @test hasmethod(rand, Tuple{MersenneTwister,Type{OfReal{Float64,Nothing,Nothing}}})
    @test (@inferred rand(MersenneTwister(1), T)) isa NamedTuple
end

@testset "symbolic bounds are detected" begin
    # has_symbolic_dims / get_unresolved_symbols must see symbolic *bounds*, not just dims,
    # so zero/rand/flatten fail with a clean message instead of a raw MethodError.
    T = @of(lo = of(Real), x = of(Real, lo, nothing))
    @test has_symbolic_dims(T)
    @test :lo in get_unresolved_symbols(T)
    @test_throws ErrorException zero(T)
    @test_throws ErrorException rand(T)
    @test_throws ErrorException flatten(T, (lo=0.0, x=1.0))

    # Expression bounds too.
    T2 = @of(base = of(Int; constant=true), x = of(Int, base, base * 2))
    @test has_symbolic_dims(T2)
    @test :base in get_unresolved_symbols(T2)
end

@testset "top-level constants are rejected, never silent" begin
    # A bare OfConstantWrapper is not flattenable; both ops must throw, not return `nothing`.
    @test has_symbolic_dims(of(Int; constant=true))
    @test_throws ErrorException flatten(of(Int; constant=true), 5)
    @test_throws ErrorException unflatten(of(Int; constant=true), Float64[])
    @test_throws ErrorException unflatten(of(Int; constant=true), missing)
end

@testset "symbolic division guards the result" begin
    T = @of(n = of(Int; constant=true), data = of(Array, n / 2))
    @test get_dims(get_types(of(T; n=10)).parameters[1]) == (5,)
    # A non-integer quotient raises the dedicated error, not a raw InexactError.
    err = try
        of(T; n=11)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("not an integer", err.msg)
end

@testset "leaf types are specifications, not instantiable" begin
    @test_throws ErrorException OfReal{Float64,Nothing,Nothing}()
    @test_throws ErrorException OfInt{Nothing,Nothing}()
    @test_throws ErrorException OfArray{Float64,1,Tuple{3}}()
    @test_throws ErrorException OfConstantWrapper{OfInt{Nothing,Nothing}}()
end

@testset "eval_symbolic_expr operations and errors" begin
    eval_expr = AbstractPPL.eval_symbolic_expr
    @test eval_expr((:+, (:*, :n, 2), 1), (n=3,)) == 7   # nested
    @test eval_expr((:-, :n), (n=3,)) == -3              # unary minus
    @test eval_expr((:-, :n, 1), (n=3,)) == 2            # binary minus
    @test eval_expr((:*, :n, 2), (n=3,)) == 6
    @test eval_expr((:/, :n, 2), (n=6,)) == 3            # exact division
    @test_throws ErrorException eval_expr((:+,), (n=3,))            # too few elements
    @test_throws ErrorException eval_expr((:^, :n, 2), (n=3,))      # unsupported op
    @test_throws ErrorException eval_expr((:+, :m, 1), (n=3,))      # unknown symbol
    @test_throws ErrorException eval_expr((:/, :n, 2), (n=5,))      # non-integer quotient
    @test_throws ErrorException eval_expr((:/, :n, 2, 1), (n=6,))   # division needs 2 args
end

@testset "rand covers single-sided and unbounded branches" begin
    rng = MersenneTwister(0)
    @test rand(rng, of(Real, 0.0, nothing)) >= 0.0   # lower-only: shifted exponential
    @test rand(rng, of(Real, nothing, 1.0)) <= 1.0   # upper-only
    @test rand(rng, of(Int, 5, nothing)) >= 5        # lower-only
    @test rand(rng, of(Int, nothing, 5)) <= 5        # upper-only
    @test rand(rng, of(Int)) isa Int                 # unbounded
end

@testset "zero respects the far bound" begin
    @test zero(of(Real, nothing, -1.0)) == -1.0   # upper < 0
    @test zero(of(Real, 2.0, 5.0)) == 2.0         # lower > 0
    @test zero(of(Int, 5, 10)) == 5               # lower > 0
    @test zero(of(Int, nothing, -3)) == -3        # upper < 0
end

@testset "value validation rejects mismatches" begin
    @test_throws ErrorException (@of(x=of(Real)))(; x="not real")
    @test_throws ErrorException (@of(i=of(Int)))(; i=3.5)   # non-integer Real
    @test_throws ErrorException (@of(i=of(Int)))(; i="x")   # non-Real
    @test (@of(i=of(Int)))(; i=3.0).i === 3                  # whole-number Real converts
    @test_throws ErrorException flatten(@of(x=of(Real), y=of(Real)), (x=1.0,))  # missing field
    @test_throws ErrorException of(@of(n=of(Int; constant=true)); n=3)          # all fields constant
    @test_throws ErrorException (@of(n=of(Int; constant=true), d=of(Array, n)))()  # missing constant
    @test_throws ErrorException AbstractPPL._create_with_default(
        OfConstantWrapper{OfInt{Nothing,Nothing}}, 0
    )
end

@testset "default_value initialises non-constant fields" begin
    T = @of(n=of(Int; constant=true), x=of(Real), arr=of(Array, n))
    inst = T(missing; n=2)
    @test inst.x === missing
    @test all(ismissing, inst.arr)
end

@testset "symbol collection across dims and bounds" begin
    T = @of(
        n = of(Int; constant=true),
        s = of(Real; constant=true),
        lo = of(Real, 0.0, 1.0),
        bnd = of(Real, n, nothing),   # symbolic bound
        a = of(Array, n, 2 * n),      # symbolic + arithmetic dims
    )
    syms = get_unresolved_symbols(T)
    @test :n in syms
    @test :s in syms
    @test has_symbolic_dims(T)
    @test !has_symbolic_dims(@of(x=of(Real), y=of(Array, 2, 2)))
    @test get_unresolved_symbols(of(Int; constant=true)) isa Vector{Symbol}
end

@testset "symbolic bounds resolve under concretization" begin
    TR = @of(n=of(Int; constant=true), r=of(Real, n, nothing))
    @test get_lower(get_types(of(TR; n=2)).parameters[1]) == 2
    TI = @of(m=of(Int; constant=true), k=of(Int, m, nothing))
    @test get_lower(get_types(of(TI; m=4)).parameters[1]) == 4
    TE = @of(n=of(Int; constant=true), e=of(Real, n + 1, nothing))
    @test get_lower(get_types(of(TE; n=5)).parameters[1]) == 6
    @test has_symbolic_dims(of(TR))   # nothing resolved -> reference is retained
end

@testset "unflatten with missing builds a missing-filled structure" begin
    P = @of(x=of(Real), a=of(Array, 2, 2))
    um = unflatten(P, missing)
    @test um.x === missing
    @test all(ismissing, um.a)
end

@testset "expression formatting for display" begin
    fmt = AbstractPPL.expr_tuple_to_string
    @test fmt((:*, (:+, :n, 1), 2)) == "(n + 1) * 2"
    @test fmt((:/, 2, (:+, :n, 1))) == "2 / (n + 1)"
    @test fmt((:+, :n, 1)) == "n + 1"
    @test fmt((:+,)) isa String          # too short
    @test fmt((:+, 1, 2, 3)) isa String  # not a binary op
    @test fmt((:%, :n, 2)) isa String    # unsupported op
end

@testset "show covers colored, symbolic, and fallback paths" begin
    # Non-color array with symbolic and arithmetic dimensions.
    s = sprint(show, @of(n=of(Int; constant=true), a=of(Array, n, n + 1)))
    @test occursin("n", s)
    @test occursin("n + 1", s)

    # A colored render of a rich spec exercises the highlighting branches.
    Tc = @of(
        n = of(Int; constant=true),         # constant Int
        s = of(Real; constant=true),        # constant Real
        bc = of(Int, 1, 9; constant=true),  # bounded constant
        lo = of(Real, 0.0, 1.0),
        bnd = of(Real, n, nothing),         # symbolic bound referencing a constant
        a = of(Array, n, 2 * n),            # symbolic + arithmetic dims
    )
    cs = sprint(show, Tc; context=(:color => true))
    @test occursin("@of(", cs)
    @test occursin("constant=true", cs)

    # Colored bounded scalars and constants.
    @test occursin("Int", sprint(show, of(Int, 0, 10); context=(:color => true)))
    @test occursin(
        "constant", sprint(show, of(Int, 0, 10; constant=true); context=(:color => true))
    )
    @test occursin(
        "constant", sprint(show, of(Int; constant=true); context=(:color => true))
    )
    @test occursin(
        "constant", sprint(show, of(Real; constant=true); context=(:color => true))
    )

    # Constant-wrapper show fallbacks for shapes the public API never produces.
    @test occursin(
        "OfConstantWrapper", sprint(show, OfConstantWrapper{OfArray{Float64,1,Tuple{3}}})
    )
    @test occursin(
        "OfConstantWrapper",
        sprint(
            show,
            OfConstantWrapper{OfNamedTuple{(:x,),Tuple{OfReal{Float64,Nothing,Nothing}}}},
        ),
    )

    # Colored symbolic bounds: constant-wrapped (both ends) and a symbolic-expression bound.
    Tb = @of(
        n = of(Int; constant=true),
        cc = of(Real, n, n; constant=true),  # constant field, both bounds symbolic
        be = of(Real, n + 1, nothing),       # symbolic-expression bound
    )
    @test occursin("of(", sprint(show, Tb; context=(:color => true)))
    # A symbolic bound whose symbol is not a known constant still renders.
    @test occursin("foo", sprint(show, of(Real, :foo, nothing); context=(:color => true)))
end

@testset "type inference from NamedTuple of values" begin
    T = of((a=1, b=2.0, c=[1.0, 2.0]))
    @test T <: OfNamedTuple
    @test get_names(T) == (:a, :b, :c)
    @test get_types(T).parameters[1] == OfInt{Nothing,Nothing}
    @test get_types(T).parameters[2] == OfReal{Float64,Nothing,Nothing}
end

@testset "missing default fills integer fields too" begin
    T = @of(n=of(Int; constant=true), i=of(Int), arr=of(Array, n))
    inst = T(missing; n=2)
    @test inst.i === missing
    @test all(ismissing, inst.arr)
end

@testset "@of rejects malformed field specifications" begin
    # `@of` errors during macro expansion for non-`field = spec` arguments.
    @test (
        try
            macroexpand(@__MODULE__, :(@of(of(Real))))
            false
        catch
            true
        end
    )
    # A non-symbol field name is rejected.
    @test (
        try
            macroexpand(@__MODULE__, :(@of(a.b = of(Real))))
            false
        catch
            true
        end
    )
end

@testset "nested arithmetic dimensions collect inner symbols" begin
    T = @of(n=of(Int; constant=true), m=of(Array, (n + 1) * 2))
    @test :n in get_unresolved_symbols(T)
    @test get_dims(get_types(of(T; n=3)).parameters[1]) == (8,)
end
