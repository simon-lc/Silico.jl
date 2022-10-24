################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis, origin=[-0.05, 0, 0], x=0.1, color=RGBA(0.8,0.8,0.8,1))
set_light!(vis, direction="Negative", ambient=0.60)
set_background!(vis)
set_camera!(vis, zoom=30.0, cam_pos=[50.0, 0, 0])

green = RGBA(4/255,191/255,173/255,1.0)
turquoise = RGBA(2/255,115/255,115/255,1.0)

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
    [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    ]
b0 = [
        0.25*[+1.0, +1.0, +1.0, +1.0],
        0.25*[+1.0, +1.0, +1.0, +1.0],
    ]

timestep = 0.10
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
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-10,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        warm_start=false,
        # complementarity_backstep=1e-1,
        )
    )



################################################################################
# simulation
################################################################################
H = 25

x12  = [+0.20, +0.25, -0.00]
v115 = [+0.00, +0.00, -0.00]
x22  = [+0.00, +0.75, -0.00]
v215 = [+0.00, +0.00, -0.00]
z0 = [x12; v115; x22; v215]

set_gravity!(mech, gravity)
Mehrotra.initialize_solver!(mech.solver)
@elapsed storage = simulate!(mech, deepcopy(z0), H)
vis, anim = visualize!(vis, mech, storage,
    name=:robot_green,
    color=green,
    show_contact=false)

vis, anim = visualize!(vis, mech, storage,
    name=:robot_turquoise,
    color=turquoise,
    show_contact=false, animation=anim)
