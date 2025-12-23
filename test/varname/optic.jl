module OpticTests

using Test
using AbstractPPL

@testset verbose = true "varname/optic.jl" begin
    # Note that much of the functionality in optic.jl is tested by varname.jl (for example,
    # pretty-printing VarNames essentially boils down to pretty-printing optics). So, this
    # file focuses on tests that are specific to optics.

    @testset "composition" begin
        @testset "with identity" begin
            i = AbstractPPL.Iden()
            o = getoptic(@varname(x.a.b))
            @test i ∘ i == i
            @test i ∘ o == o
            @test o ∘ i == o
        end

        o1 = getoptic(@varname(x.a.b))
        o2 = getoptic(@varname(x[1][2]))
        @test o1 ∘ o2 == getoptic(@varname(x[1][2].a.b))
        @test o2 ∘ o1 == getoptic(@varname(x.a.b[1][2]))
    end

    @testset "decomposition" begin end
end

end # module
