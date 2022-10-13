vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# demo
################################################################################
A0 = [
    [
        +1.0 -0.0;
        -0.1 +1.0;
        -1.0 -0.3;
        +0.0 -1.0;
        ],
    [
        +1.0 +0.3;
        +0.0 +1.0;
        -1.0 +0.2;
        +0.0 -1.0;
        +0.4 -0.8;
        ],
    ]
b0 = [
        0.50*[+1.0, +1.0, +1.0, +1.0],
        0.35*[+1.0, +1.0, +1.0, +1.0, +1.0],
    ]

mech = get_polytope_collision(;
    timestep=0.05,
    gravity=1*-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=1.0,
    method_type=:symbolic,
    # method_type=:finite_difference,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-4,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=true,
        complementarity_correction=0.5,
        # complementarity_backstep=1e-2,
        )
    )
# Mehrotra.solve!(mech.solver)

################################################################################
# test simulation
################################################################################
xp2 = [+0.25, +0.80, -0.1]
xc2 = [-0.00, +2.00, -0.4]
vp15 = [-0.0, +0.0, -0.0]
vc15 = [+0.0, +0.0, +0.0]
z0 = [xp2; vp15; xc2; vc15]

u0 = zeros(6)
H0 = 100

@elapsed storage = simulate!(mech, z0, H0)

################################################################################
# visualization
################################################################################
visualize!(vis, mech, storage, build=true)

using Plots
storage.x
plot(hcat([storage.x[i][1] for i=1:H0]...)')
plot(hcat([storage.x[i][2] for i=1:H0]...)')
plot(hcat([storage.v[i][2] for i=1:H0]...)')
scatter(storage.iterations)
# plot!(hcat(storage.variables...)')
# RobotVisualizer.convert_frames_to_video_and_gif("sphere_polytope_drop")
