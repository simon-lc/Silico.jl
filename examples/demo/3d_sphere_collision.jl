using Plots
using Statistics
using Random

################################################################################
# visualization
################################################################################
vis = Visualizer()
render(vis)
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# define mechanism
################################################################################
timestep = 0.02;
gravity = -1*9.81;
mass = 0.2;
inertia = 0.8 * Matrix(Diagonal(ones(3)));
friction_coefficient = 0.3

mech = get_3d_sphere_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    # method_type=:symbolic,
    method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-5,
        # compressed_search_direction=true,
        compressed_search_direction=false,
        sparse_solver=false,
        warm_start=true,
        complementarity_backstep=1e-2,
        )
    )

################################################################################
# test simulation
################################################################################
xp2 =  [+0.00, +0.20, +1.00, 1,0,0,0]
vp15 = [+0.00, -0.00, +0.00, 0,0,0]
xc2 =  [+0.00, +0.20, +2.00, 1,0,0,0]
vc15 = [+0.00, -0.00, +0.00, 0,0,0]
z0 = [xp2; vp15; xc2; vc15]

H0 = 100
@elapsed storage = simulate!(mech, deepcopy(z0), H0)

################################################################################
# visualizati
################################################################################
build_mechanism!(vis, mech)
set_mechanism!(vis, mech, storage, 1)

visualize!(vis, mech, storage, build=false)

scatter(storage.iterations)
