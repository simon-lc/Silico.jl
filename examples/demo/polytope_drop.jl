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
inertia = 0.2 * ones(1);

mech = get_polytope_drop(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    # method_type=:symbolic,
    method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-3,
        # compressed_search_direction=true,
        compressed_search_direction=false,
        sparse_solver=false,
        warm_start=false,
        )
    );

# solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2 = [+0.0,1.5,-0.001]
vp15 = [-0,0,-0.0]
z0 = [xp2; vp15]

u0 = zeros(3)
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
