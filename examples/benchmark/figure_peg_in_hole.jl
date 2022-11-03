################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis, origin=[-0.05, 0, 0], x=0.1, color=RGBA(0.8,0.8,0.8,1))
set_light!(vis, direction="Negative", ambient=0.60)
set_background!(vis)
set_camera!(vis, zoom=30.0, cam_pos=[100.0, 0, 0])

green = RGBA(4/255,191/255,173/255,1.0)
turquoise = RGBA(2/255,115/255,115/255,1.0)

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

timestep = 0.01
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.01

mech = get_polytope_insertion(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-5,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=true,
        warm_start=true,
        )
    )


################################################################################
# simulation
################################################################################
H = 250

x2 = [+0.00, +1.50, +0.48]
v15 = [-0.0, +0.0, -0.0]
z0 = [x2; v15]

set_gravity!(mech, gravity)
Mehrotra.initialize_solver!(mech.solver)
@elapsed storage = simulate!(mech, deepcopy(z0), H)
vis, anim = visualize!(vis, mech, storage,
    name=:robot,
    color=green,
    show_contact=false)
build_mechanism!(vis, mech,
    env_color=turquoise,
    color=green,
    show_contact=false)
set_mechanism!(vis, mech, storage, H,
    show_contact=false,
    name=:robot)


for i = 1:10:H
    α = (0.5H + i)/(10H)
    faded_green = RGBA(green.r, green.g, green.b, α)
    build_mechanism!(vis[:trace], mech,
        show_contact=false,
        env_color=turquoise,
        color=faded_green,
        name=Symbol(:robot_, i))
    set_mechanism!(vis[:trace], mech, storage, i,
        show_contact=false,
        name=Symbol(:robot_, i))
end

# convert_frames_to_video_and_gif("peg_in_hole_no_friction")
