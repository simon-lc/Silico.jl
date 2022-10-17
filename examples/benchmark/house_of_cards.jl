using Plots

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# define mechanisms
################################################################################
l1 = 0.5
l2 = 0.025
A = [
    [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 +0.0;
        +0.0 +1.0;
        -1.0 +0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 +0.0;
        +0.0 +1.0;
        -1.0 +0.0;
        +0.0 -1.0;
        ],
    ]
b = [
        1.0*[l1, l2, l1, l2],
        1.0*[l1, l2, l1, l2],
        1.0*[l1, l2, l1, l2],
        1.0*[l1, l2, l1, l2],
        1.0*[2l1, l2, 2l1, l2],
    ]

timestep = 0.01
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.50

mech = get_polytope_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    A=A,
    b=b,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-3,
        compressed_search_direction=false,
        max_iterations=60,
        sparse_solver=true,
        warm_start=false,
        # complementarity_backstep=1e-1,
        )
    )

mech.solver.options.complementarity_tolerance = 1e-3
mech.solver.options.complementarity_correction = 0.5

################################################################################
# simulation
################################################################################
H = 10

################################################################################
# test no gravity
################################################################################
θ = π/2 - π/10
Δx = l1 * cos(θ) + l2 * sin(θ)
Δy = l1 * sin(θ) + l2 * cos(θ)
x12 = [-1Δx, +1Δy, +θ]
x22 = [+1Δx, +1Δy, -θ]
x32 = [2l1-Δx, +1Δy, +θ]
x42 = [2l1+Δx, +1Δy, -θ]
x52 = [l1, +4.2Δy, 0]

v115 = [-0.0, +0.0, -0.0]
v215 = [-0.0, +0.0, -0.0]
v315 = [+0.0, -0.0, +0.0]
v415 = [+0.0, -0.0, +0.0]
v515 = [+0.0, -0.0, +0.0]
z0 = [x12; v115; x22; v215; x32; v315; x42; v415; x52; v515]

set_gravity!(mech, gravity)
Mehrotra.initialize_solver!(mech.solver)
@elapsed storage = simulate!(mech, deepcopy(z0), H)
vis, anim = visualize!(vis, mech, storage, name=:single, color=RGBA(1,1,1,0.8))
scatter(storage.iterations, color=:red)


# RobotVisualizer.convert_frames_to_video_and_gif("jenga_contact_point")
