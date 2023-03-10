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
friction_coefficient = 0.03

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
        complementarity_tolerance=1e-9,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=true,
        warm_start=false,
        )
    )


################################################################################
# simulation
################################################################################
H = 500

# x2 = [+0.00, +1.50, +0.48]
x2 = [+0.00, +1.50, +0.30]
v15 = [-0.0, +0.0, -0.0]
z0 = [x2; v15]

set_gravity!(mech, gravity)
Mehrotra.initialize_solver!(mech.solver)
@elapsed storage = simulate!(mech, deepcopy(z0), H)
vis, anim = visualize!(vis, mech, storage,
    name=:robot,
    color=green,
    show_contact=true)
settransform!(vis[:robot][:contacts], MeshCat.Translation(0.05,0,0.0))
build_mechanism!(vis, mech,
    env_color=turquoise,
    color=green,
    show_contact=false)
set_mechanism!(vis, mech, storage, H,
    show_contact=true,
    name=:robot)


# show trace
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
# convert_frames_to_video_and_gif("peg_in_hole_contact")


# Measure the forces on both sides in comparison with the force applied by the floor.
contact_floor = mech.contacts[1]
contact_left = mech.contacts[2]
contact_right = mech.contacts[3]
x = mech.variables

# c, γ, ψ, β, λp, sγ, sψ, sβ, sp =
F_floor = unpack_variables(x[contact_floor.index.variables], contact_floor)[2][1]
# c, α, βp, βc, γ, ψ, β, λα, λp, λc, sγ, sψ, sβ, sα, sp, sc =
F_left = unpack_variables(x[contact_left.index.variables], contact_left)[5][1]
# c, α, βp, βc, γ, ψ, β, λα, λp, λc, sγ, sψ, sβ, sα, sp, sc =
F_right = unpack_variables(x[contact_right.index.variables], contact_right)[5][1]

F_floor / F_right
F_floor / F_left
