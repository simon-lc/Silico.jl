using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions
using Plots

vis = Visualizer()
open(vis)

include("../src/Silico.jl")

################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1);


mech = get_quasistatic_manipulation(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.3,
    finger_friction_coefficient=0.9,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Options(
        verbose=true,
        complementarity_tolerance=1e-4,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        # complementarity_decoupling=true
        )
    );

# solve!(mech.solver)
# Main.@profiler solve!(mech.solver)
################################################################################
# test simulation
################################################################################
x_object0  = [+0.00, +0.30, +0.25]
x_finger10 = [-0.50, +0.20, -0.00]
x_finger20 = [+0.60, +0.20, -0.00]
# z0 = [x_object; x_finger1]
z0 = [x_object0; x_finger10; x_finger20]


H0 = 140
# u0 = zeros(9)
x_object_goal  = [0.00, 0.00, 0.00]
x_finger1_goal = [-0.30, +1.40, -0.00]
x_finger2_goal = [+0.40, +1.40, -0.00]

U = []
for i = 1:H0
    α = i/H0
    u_object = [0,0,0]
    u_finger1 = α * x_finger1_goal + (1-α) * x_finger10
    u_finger2 = α * x_finger2_goal + (1-α) * x_finger20
    u = [
        u_object;
        u_finger1;
        u_finger2;
        ]
    push!(U, u)
end
# u0 = zeros(6)
# u0 = [0;0;0; +x_finger10; x_finger20]
ctrl = open_loop_controller([u0])
ctrl = open_loop_controller(U)

@elapsed storage = simulate!(mech, z0, H0, controller=ctrl)
# Main.@profiler [solve!(mech.solver) for i=1:300]
# @benchmark $solve!($(mech.solver))

scatter(storage.iterations)

################################################################################
# visualization
################################################################################
set_floor!(vis)
set_light!(vis)
set_background!(vis)
visualize!(vis, mech, storage, build=true)


scatter(storage.iterations)
# plot!(hcat(storage.variables...)')
RobotVisualizer.convert_frames_to_video_and_gif("hand_coded_lift_clean")
