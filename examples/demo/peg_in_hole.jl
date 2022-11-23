using Plots

################################################################################
# visualization
################################################################################
green = RGBA(4/255,191/255,173/255,1.0)
turquoise = RGBA(2/255,115/255,115/255,1.0)
gray = RGBA(0.6, 0.6, 0.6, 1.0)

vis = Visualizer()
open(vis)
set_floor!(vis, origin=[-0.05,0,0], x=0.10, color=gray)
set_light!(vis)
set_background!(vis)
set_camera!(vis, zoom=40.0, cam_pos=[65,0,0.0])


################################################################################
# define mechanisms
################################################################################
A0 = [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        +1.0 -0.2;
        -1.0 -0.2;
    ]
b0 = 0.249*[+1.0, +3.0, +1.0, +1.0, 0.9, 0.9]

timestep = 0.05
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.1

mech = get_polytope_insertion(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    # method_type=:symbolic,
    method_type=:finite_difference,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-5,
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
H = 40
x2 = [+0.00, +1.50, +0.25]
v15 = [-0.0, +0.0, -0.0]
z0 = [x2; v15]

set_gravity!(mech, gravity)
Mehrotra.initialize_solver!(mech.solver)
@elapsed storage = simulate!(mech, deepcopy(z0), H)
vis, anim = visualize!(vis, mech, storage, name=:single, color=green, env_color=turquoise)
scatter(storage.iterations, color=:red)

RobotVisualizer.convert_frames_to_video_and_gif("peg_in_hole_contact_dots")
