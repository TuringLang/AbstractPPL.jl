@testset "deprecations.jl" begin
    @test (@test_deprecated VarName(:x)) == VarName{:x}()
    @test (@test_deprecated VarName(:x, ((1,), (:, 2)))) == VarName{:x}(((1,), (:, 2)))
end
