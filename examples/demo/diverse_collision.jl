using Plots
using Statistics
using Random

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

green = RGBA(4/255,191/255,173/255,1.0)
turquoise = RGBA(2/255,115/255,115/255,1.0)
black = RGBA(0.2,0.2,0.25,1)

################################################################################
# define mechanism
################################################################################
timestep = 0.01;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.6

mech = get_diverse_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    minkowski_radius=0.3,
    union_radius=0.15,
    segment=4.0,
    A = [[0 1; +2.5 -1; -2.5 -1], [5 1; -5 1; 1 5; 1 -5.0]],
    b = [0.3 * ones(3), 0.3 * ones(4)],
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-5,
        compressed_search_direction=true,
        # compressed_search_direction=false,
        sparse_solver=false,
        warm_start=true,
        complementarity_backstep=1e-2,
        )
    );

################################################################################
# test simulation
################################################################################
x1 = [+0.0,1.5,-0.20]
v1 = [-0,0,-1.0]
x2 = [-1.8,3.5,+0.75]
v2 = [-0,0,-0.0]
z0 = [x1; v1; x2; v2]



H0 = 250
mech.contacts
Mehrotra.initialize_solver!(mech.solver)
@elapsed storage = simulate!(mech, z0, H0)

################################################################################
# visualization
################################################################################
vis, anim = visualize!(vis, mech, storage, name=:green, color=green)
vis, anim = visualize!(vis, mech, storage, name=:turquoise, color=turquoise, animation=anim)

scatter(storage.iterations)
# plot!(hcat(storage.variables...)')
set_floor!(vis, x=0.3)

set_camera!(vis, zoom=10.0, cam_pos=[-33, 0, 0])

################################################################################
# union and minkowski
################################################################################
A = [[0 1; +2.5 -1; -2.5 -1], [5 1; -5 1; 1 5; 1 -5.0]]
b = [0.3 * ones(3), 0.3 * ones(4)]
