using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions
using Plots

vis = Visualizer()
open(vis)
render(vis)

include("../src/Silico.jl")

include("../environment/polytope_bundle.jl")
include("../environment/polytope_drop.jl")


################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1);


mech = get_sphere_bundle(;
# mech = get_polytope_drop(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.2,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Options(
        verbose=true,
        complementarity_tolerance=1e-4,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=true,
        complementarity_correction=0.5,
        # complementarity_decoupling=true
        )
    );

# Main.@profiler solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2 = [+0.1,1.5,-0.25]
xc2 = [-0.0,0.5,-2.25]
vp15 = [-0,0,-0.0]
vc15 = [+0,0,+0.0]
z0 = [xp2; vp15; xc2; vc15]

u0 = zeros(6)
H0 = 100

@elapsed storage = simulate!(mech, z0, H0)
# Main.@profiler [solve!(mech.solver) for i=1:300]
# @benchmark $solve!($(mech.solver))

# 7.5/0.148
# 14.8/0.148


################################################################################
# visualization
################################################################################
set_floor!(vis)
set_light!(vis)
set_background!(vis)
visualize!(vis, mech, storage, build=true)


scatter(storage.iterations)
# plot!(hcat(storage.variables...)')
# RobotVisualizer.convert_frames_to_video_and_gif("sphere_polytope_drop")
