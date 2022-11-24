module DojoLight

using GeometryBasics
using LinearAlgebra
using Mehrotra
using MeshCat
using Printf
using Quaternions
using RobotVisualizer
using StaticArrays
using Colors


include("rotate.jl")
include("quaternion.jl")
include("node.jl")
include("shape.jl")

include("dynamics/body.jl")
include("dynamics/robot.jl")
include("dynamics/object.jl")

include("contact/3d_methods.jl")
include("contact/contact.jl")
include("contact/poly_halfspace.jl")
include("contact/poly_poly.jl")
include("contact/poly_sphere.jl")
include("contact/sphere_halfspace.jl")
include("contact/sphere_sphere.jl")

include("contact/bilevel/collision_detection.jl")
include("contact/bilevel/bilevel_contact.jl")
include("contact/bilevel/bilevel_methods.jl")

include("mechanism.jl")
include("contact_frame.jl")
include("storage.jl")
include("simulate.jl")
include("dynamics.jl")
include("continuation.jl")
include("visuals.jl")
include("utils.jl")
# include("visuals_opengl.jl")

include("../environment/3d_polytope_collision.jl")
include("../environment/3d_polytope_drop.jl")
include("../environment/3d_sphere_collision.jl")
include("../environment/3d_sphere_drop.jl")
include("../environment/bilevel_polytope_collision.jl")
include("../environment/bundle_collision.jl")
include("../environment/bundle_drop.jl")
include("../environment/capsule_collision.jl")
include("../environment/diverse_collision.jl")
include("../environment/padded_polytope_drop.jl")
include("../environment/polytope_collision.jl")
include("../environment/polytope_drop.jl")
include("../environment/polytope_grasper.jl")
include("../environment/polytope_insertion.jl")
include("../environment/quasistatic_manipulation.jl")
include("../environment/quasistatic_sphere_box.jl")
include("../environment/quasistatic_sphere_bundle.jl")
include("../environment/quasistatic_sphere_drop.jl")
include("../environment/sphere_bundle.jl")
include("../environment/sphere_collision.jl")
include("../environment/sphere_drop.jl")

export
    Mechanism
end


# using Pkg
# Pkg.add(path="/home/simon/.julia/dev/DirectTrajectoryOptimization.jl")
# Pkg.add(path="/home/simon/.julia/dev/Mehrotra.jl")
# Pkg.add(url="https://github.com/simon-lc/Mehrotra.jl")
