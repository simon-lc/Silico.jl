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
A0 = [
    [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    ]
b0 = [
        0.25*[+1.0, +1.0, +1.0, +1.0],
    ]

timestep = 0.02
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.50

mech = get_polytope_insertion(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-10,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=true,
        warm_start=true,
        # complementarity_backstep=1e-1,
        )
    )


################################################################################
# simulation
################################################################################
H = 200


################################################################################
# test no gravity
################################################################################
x12 = [+0.05, +2.40, +0.1]
x22 = [-0.05, +1.80, -0.1]
x32 = [-0.00, +1.00, -0.2]
x42 = [-0.00, +0.30, -0.2]
v115 = [-0.0, +0.0, -0.0]
v215 = [-0.0, +0.0, -0.0]
v315 = [+0.0, -0.0, +0.0]
v415 = [+0.0, -0.0, +0.0]
z0 = [x12; v115; x22; v215; x32; v315; x42; v415]

set_gravity!(mech, gravity)
Mehrotra.initialize_solver!(mech.solver)
@elapsed storage = simulate!(mech, deepcopy(z0), H)
vis, anim = visualize!(vis, mech, storage, name=:single, color=RGBA(1,1,1,0.8))
scatter(storage.iterations, color=:red)


# RobotVisualizer.convert_frames_to_video_and_gif("jenga_contact_point")
