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
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.4 * ones(1,1);
radius = 0.50

mech = get_padded_polytope_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    radius=radius,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-3,
        # compressed_search_direction=true,
        compressed_search_direction=false,
        sparse_solver=false,
        warm_start=true,
        complementarity_backstep=1e-1,
        )
    );

################################################################################
# test simulation
################################################################################
xp2 = [+0.0,1.5,-1.25]
vp15 = [-0,0,-6.0]
z0 = [xp2; vp15]

u0 = zeros(3)
H0 = 150

@elapsed storage = simulate!(mech, z0, H0)

################################################################################
# visualization
################################################################################
visualize!(vis, mech, storage, build=true, color=RGBA(0.2,0.2,0.25,1))

# scatter(storage.iterations)
# plot!(hcat(storage.variables...)')
