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

################################################################################
# define mechanism
################################################################################
timestep = 0.02;
gravity = -0.00;
mass = 1.0;
inertia = 0.1 * ones(1,1);
friction_coefficient = 0.2

mech = get_capsule_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    # method_type=:symbolic,
    method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-5,
        # compressed_search_direction=true,
        compressed_search_direction=false,
        sparse_solver=false,
        warm_start=true,
        complementarity_backstep=3e-3,
        )
    );

# solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2 = [+1.00, +1.00, -1.0]
xc2 = [-1.00, +1.00, -2.4]
vp15 = [-1.0, +0.0, -0.2]
vc15 = [+1.0, +0.0, +0.2]
z0 = [xp2; vp15; xc2; vc15]

H0 = 150
@elapsed storage = simulate!(mech, z0, H0)

################################################################################
# visualization
################################################################################
build_mechanism!(vis, mech)
set_mechanism!(vis, mech, storage, 10)

visualize!(vis, mech, storage, build=false)

# scatter(storage.iterations)
# plot!(hcat(storage.variables...)')
