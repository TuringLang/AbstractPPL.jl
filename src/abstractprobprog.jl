using AbstractMCMC


"""
    AbstractProbabilisticProgram

Common base type for models expressed as probabilistic programs.
"""
abstract type AbstractProbabilisticProgram <: AbstractMCMC.AbstractModel end
