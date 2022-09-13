using Test
using BenchmarkTools

using FiniteDiff
using LinearAlgebra
using Random
using SparseArrays

using DojoLight

@testset "dynamics"             verbose=true begin include("dynamics.jl") end
