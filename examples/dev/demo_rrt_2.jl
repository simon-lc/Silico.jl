using Graphs
using GraphRecipes
using Plots
using Statistics
using Random
using BenchmarkTools

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -1*9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.5


mech = get_quasistatic_sphere_box(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    control_mode=:robot,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-5,
        compressed_search_direction=true,
        sparse_solver=false,
        differentiate=false,
        warm_start=false,
        # complementarity_decoupling=true
        )
    );


################################################################################
# test simulation
################################################################################
x0_box    = [+1.0, 0.4, -0.00]
x0_sphere = [-0.0, 0.4, -0.00]
z0 = [x0_box; x0_sphere]

u0 = [0; 0; 0; x0_sphere]
H0 = 140

ctrl = open_loop_controller([u0])
@elapsed storage = simulate!(mech, deepcopy(z0), H0, controller=ctrl)
# @benchmark Mehrotra.solve!(mech.solver)

visualize!(vis, mech, storage, build=true)

include("rrt_methods.jl")

################################################################################
# test RRT
################################################################################
x0_box    = [+0.60, 0.40, -0.00π]
x0_sphere = [+0.00, 0.40, -0.00π]
z0 = [x0_box; x0_sphere]

x1_box    = [+0.50, 0.40, -0.5π]
x1_sphere = [+1.50, 1.00, -0.00π]
z1 = [x1_box; x1_sphere]

set_mechanism!(vis, mech, z0, name=:start)
set_mechanism!(vis, mech, z1, name=:goal)


# build_mechanism!(vis, mech, name=:start, color=RGBA(1,0,0,0.3))
# build_mechanism!(vis, mech, name=:goal, color=RGBA(0,1,0,0.3))
# build_mechanism!(vis, mech, name=:subgoal, color=RGBA(1,1,1,1))
# build_mechanism!(vis, mech, name=:new, color=RGBA(0,0,0,1))
# build_mechanism!(vis, mech, name=:nearest, color=RGBA(1,0,0,1))

# settransform!(vis[:start], MeshCat.Translation(+0.8,0,0))
# settransform!(vis[:goal], MeshCat.Translation(+0.4,0,0))
# settransform!(vis[:subgoal], MeshCat.Translation(-1.2,0,0))
# settransform!(vis[:new], MeshCat.Translation(-0.8,0,0))
# settransform!(vis[:nearest], MeshCat.Translation(-0.4,0,0))

K0 = 1500
γ0 = 3e-3
ρ0 = 3e-3
ϵ0 = 3e-1
tree0, vertices0 = rrt_solve!(mech, z0, z1, K0; γ=γ0, ρ=ρ0, ϵ=ϵ0, goal_distance=0.05)

# names0 = [" $i " for i = 1:nv(tree0)]
# plt = GraphRecipes.graphplot(tree0, names=names0, curvature_scalar=0.01, linewidth=3, fontsize=10)

i_nearest0 = index_nearest(mech, vertices0, z1; γ=γ0, ρ=3e-3)
trace0 = get_trace(tree0, i_nearest0)

visualize!(vis, mech, vertices0[trace0])
plot(hcat(vertices0[trace0]...)')
scatter!(hcat(vertices0[trace0]...)')

# RobotVisualizer.convert_frames_to_video_and_gif("rrt_rotate_and_push_smooth")
