""" 
    AbstractContext

Common base type for evaluation contexts.
"""
abstract type AbstractContext end

""" 
evaluate!!

General API for model operations, e.g. prior evaluation, log density, log joint etc.
"""
function evaluate!! end
