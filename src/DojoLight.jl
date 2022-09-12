module DojoLight

using BenchmarkTools
using CUDA
using Clustering
using DirectTrajectoryOptimization
using FileIO
using Flux
using ForwardDiff
using GLPK
using GeometryBasics
using Ipopt
using LinearAlgebra
using MeshCat
using Meshing
using Nonconvex
using NonconvexIpopt
using NonconvexPercival
using Optim
using Plots
using Polyhedra
using Quaternions
using RobotVisualizer
using Mehrotra

include("rotate.jl")
include("quaternion.jl")
include("node.jl")
include("shape.jl")

include("dynamics/body.jl")
include("dynamics/poly_halfspace.jl")
include("dynamics/poly_poly.jl")
include("dynamics/poly_sphere.jl")
include("dynamics/sphere_halfspace.jl")
include("dynamics/sphere_sphere.jl")

include("quasistatic_dynamics/robot.jl")
include("quasistatic_dynamics/object.jl")

include("mechanism.jl")
include("contact_frame.jl")
include("storage.jl")
include("simulate.jl")
include("dynamics.jl")
include("continuation.jl")

include("visuals.jl")
# include("visuals_opengl.jl")

include("../environment/bundle_collision.jl")
include("../environment/bundle_drop.jl")
include("../environment/learned_bundle.jl")
include("../environment/polytope_drop.jl")
include("../environment/quasistatic_manipulation.jl")
include("../environment/quasistatic_sphere_box.jl")
include("../environment/quasistatic_sphere_bundle.jl")
include("../environment/quasistatic_sphere_drop.jl")
include("../environment/sphere_bundle.jl")
include("../environment/sphere_collision.jl")
include("../environment/sphere_drop.jl")

end


# using Pkg
# Pkg.add(path="/home/simon/.julia/dev/Mehrotra.jl")
