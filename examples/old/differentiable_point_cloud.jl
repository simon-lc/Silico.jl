using GLVisualizer
using GeometryBasics
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions
using Optim
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using ForwardDiff


include("../cvxnet/loss.jl")
# include("../cvxnet/point_cloud.jl")
include("../cvxnet/softmax.jl")


Xlims = [-2, 2]
Ylims = [-2, 2]
S = 100

X = range(Xlims..., length=S)
Y = range(Ylims..., length=S)
V = zeros(S,S)

A0 = [
    +1.0 -0.4;
    +0.0 +1.0;
    -1.0 -0.4;
    +0.0 -1.0;
    ]
for i = 1:4
    A0[i,:] ./= norm(A0[i,:])
end
b0 = 0.5*[
    +1.5,
    +1,
    +1.5,
    +0,
    ];
o0 = [0, 0.0]

Af = [0 1.0]
bf = [0.0]
of = [0, 0.0]

plt = plot_polytope(A0, b0, 1e2, xlims=Xlims, ylims=Ylims, S=100)
display(plt)

e = [0.0, 3.0]
v = [0.0, -1.0]
δ = 4e+0

# julia_intersection(e, v, A0, b0, o0)
# julia_intersection(e, v, Af, bf, of)
# julia_intersection(e, v, [A0, Af], [b0, bf], [o0, of], δ)
β = range(-0.2π, -0.8π, length=300)
θ, polytope_dimensions = pack_halfspaces([A0, Af], [b0, bf], [o0, of])
P = julia_point_cloud(e, β, θ, polytope_dimensions, δ)
julia_point_cloud!(P, e, β, θ, polytope_dimensions, δ)
ForwardDiff.jacobian(θ -> julia_point_cloud(e, β, θ, polytope_dimensions, δ), θ)
ForwardDiff.gradient(θ -> julia_loss([P], [e], [β], δ, θ .+ 0.001, polytope_dimensions), θ)


scatter!(plt, P[1,:], P[2,:])
julia_loss([P], [e], [β], δ, θ, polytope_dimensions)


# @benchmark P = julia_point_cloud(e, β, θ, polytope_dimensions, δ)
# @benchmark julia_point_cloud!(P, e, β, θ, polytope_dimensions, δ)
# @benchmark ForwardDiff.jacobian(θ -> julia_point_cloud(e, β, θ, polytope_dimensions, δ), θ)
