using Plots
using Statistics
using Random

################################################################################
# visualization
################################################################################
vis = Visualizer()
render(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# define mechanism
################################################################################
timestep = 0.02;
gravity = -0*9.81;
mass = 0.2;
inertia = 0.8 * Matrix(Diagonal(ones(3)));
friction_coefficient = 0.1

mech = get_3d_sphere_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-4,
        # compressed_search_direction=true,
        compressed_search_direction=false,
        sparse_solver=false,
        warm_start=false,
        )
    )

################################################################################
# test simulation
################################################################################
xp2 =  [+0.00, +0.20, +1.00, 1,0,0,0]
vp15 = [+0.00, -0.50, +1.50, +10,+10,0]
z0 = [xp2; vp15]

H0 = 200
@elapsed storage = simulate!(mech, deepcopy(z0), H0)

################################################################################
# visualization
################################################################################
build_mechanism!(vis, mech)
set_mechanism!(vis, mech, storage, 1)

visualize!(vis, mech, storage, build=false)

scatter(storage.iterations)
# plot(hcat(storage.variables...)')
