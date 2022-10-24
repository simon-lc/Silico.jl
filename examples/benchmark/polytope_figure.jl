################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis, origin=[-0.05, 0, 0], x=0.1, color=RGBA(0.8,0.8,0.8,1))
set_light!(vis, direction="Negative", ambient=0.60)
set_background!(vis)
set_camera!(vis, zoom=30.0, cam_pos=[60.0, 0, 0])

green = RGBA(4/255,191/255,173/255,1.0)
turquoise = RGBA(2/255,115/255,115/255,1.0)

################################################################################
# define mechanisms
################################################################################
A0 = [
    [
        +1.0 +0.2;
        -0.0 +1.0;
        -1.0 -0.3;
        +0.0 -1.0;
        +0.8 -0.8;
        ],
    ]
b0 = [
        0.40*[+1.0, +1.0, +1.0, +1.0, +1.0],
    ]

timestep = 0.02
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.20

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
        complementarity_tolerance=1e-4,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        warm_start=false,
        # warm_start=true,
        # complementarity_backstep=1e-1,
        )
    )

################################################################################
# simulation
################################################################################
H = 100

xp2  = [-2.00, +0.90, -0.00]
vp15 = [+4.00, +0.00, -3.00]
z0 = [xp2; vp15]

set_gravity!(mech, gravity)
Mehrotra.initialize_solver!(mech.solver)
@elapsed storage = simulate!(mech, deepcopy(z0), H)
vis, anim = visualize!(vis, mech, storage,
    name=:robot,
    color=green,
    show_contact=false)

for i = 1:5:H
    α = (0.5H + i)/(10H)
    faded_green = RGBA(green.r, green.g, green.b, α)
    build_mechanism!(vis[:trace], mech,
        show_contact=false,
        color=faded_green,
        name=Symbol(:robot_, i))
    set_mechanism!(vis[:trace], mech, storage, i,
        show_contact=false,
        name=Symbol(:robot_, i))
end
