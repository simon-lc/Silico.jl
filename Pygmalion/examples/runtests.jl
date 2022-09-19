using Test
using BenchmarkTools

using LinearAlgebra
using Random
using SparseArrays
using FiniteDiff

using Mehrotra

@testset "block sparse"             verbose=true begin include("block_sparse.jl") end
@testset "random ncp"               verbose=true begin include("random_ncp.jl") end
@testset "contact ncp"              verbose=true begin include("contact_ncp.jl") end
@testset "differentiability"        verbose=true begin include("differentiability.jl") end
@testset "linear solver"            verbose=true begin include("linear_solver.jl") end
@testset "compressed solver"        verbose=true begin include("compressed_solver.jl") end
@testset "relaxed complementarity"  verbose=true begin include("relaxed_complementarity.jl") end
@testset "decoupling"               verbose=true begin include("decoupling.jl") end
@testset "finite difference"        verbose=true begin include("finite_difference.jl") end
@testset "residual"                 verbose=true begin include("residual.jl") end
@testset "consistency"              verbose=true begin include("consistency.jl") end
