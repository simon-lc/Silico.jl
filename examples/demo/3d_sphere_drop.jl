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
mass = 0.2;
inertia = 0.8 * Matrix(Diagonal(ones(3)));
inertia = Matrix(Diagonal([0.1, 0.5, 0.9]));
friction_coefficient = 0.5

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
        residual_tolerance=1e-5,
        compressed_search_direction=true,
        # compressed_search_direction=false,
        sparse_solver=false,
        warm_start=true,
        complementarity_backstep=1e-2,
        )
    )

################################################################################
# test simulation
################################################################################
xp2 =  [+0.00, +0.00, +1.50, 1,0,0,0]
vp15 = [+0.00, -0.00, +0.00, +0.0,+5.0,+0.0]
vp15 = [+0.00, -0.00, +0.00, +0.0,+0.0,+0.0]

norm(50*ones(3) * timestep)

z0 = [xp2; vp15]

H0 = 100
@elapsed storage = simulate!(mech, deepcopy(z0), H0)

################################################################################
# visualization
################################################################################
build_mechanism!(vis, mech)
set_mechanism!(vis, mech, storage, 1)

visualize!(vis, mech, storage, color=RGBA(1,1,1,1))

scatter(storage.iterations .== 30)
scatter(storage.iterations)
plot!(hcat([storage.v[i][1][4:6] for i=1:H0]...)')
# plot(hcat(storage.variables...)')
# plot!(hcat([storage.v[i][1][4:6] for i=1:H0]...)')

# plot(mech.solver.data.residual.all)

# x0 = 0.5 * ones(3)
# x1 = 0.5 * ones(3)
# f(x) = sqrt(1 - x'*x) * Diagonal([1, 0.2, 1.0]) * x
# inert = 0.1*rand(3,3)
# inert = inert' * inert
# f(x) = sqrt(1 - x'*x) * inert * x - cross(x, inert * x)
# g(x) = sqrt(1 - x'*x) * inert * x + cross(x, inert * x)
#
# plot(hcat([f(x * ones(3)) for x ∈ 0.01:0.01:0.579]...)')
# plot!(hcat([g(x * ones(3)) for x ∈ 0.01:0.01:0.579]...)')
#
# f(0.57*ones(3))
setobject!(vis[:robot][:bodies][:pbody][:x], HyperRectangle(Vec(0,0,0), Vec(0.9, 0.1, 0.1)))
setobject!(vis[:robot][:bodies][:pbody][:y], HyperRectangle(Vec(0,0,0), Vec(0.1, 0.7, 0.1)))
setobject!(vis[:robot][:bodies][:pbody][:z], HyperRectangle(Vec(0,0,0), Vec(0.1, 0.1, 0.5)))

RobotVisualizer.convert_frames_to_video_and_gif("sphere_drop")








################################################################################
# visualization
################################################################################
α_out = 0.20
α_out = 0.75
α_in = 0.75
dark_gray = RGBA(43/255, 43/255, 43/255, 1.0)
orange = RGBA(255/255, 165/255, 0/255, α_in)
α_orange = RGBA(255/255, 165/255, 0/255, α_out)
green = RGBA(0/255, 255/255, 0/255, 1.0)
light_blue = RGBA(160/255, 160/255, 255/255, α_in)
α_light_blue = RGBA(160/255, 160/255, 255/255, α_out)
black = RGBA(0/255, 0/255, 0/255, 1.0)
white = RGBA(255/255, 255/255, 255/255, 1.0)

# vis = Visualizer()
# open(vis)
set_light!(vis, ambient=0.62, direction="Negative")
setprop!(vis["/Lights/FillLight/<object>"], "intensity", 1.11)
setprop!(vis["/Lights/PointLightNegativeX/<object>"], "intensity", 1.42)
set_floor!(vis, x=0, y=0, z=0, color=RGBA(0,0,0,0))
set_background!(vis, top_color=dark_gray, bottom_color=dark_gray)
set_camera!(vis, zoom=6, cam_pos=[10,0,0])

build_segment!(vis[:collision], color=white, segment_radius=0.05, name=:α)
build_segment!(vis[:collision], color=orange, segment_radius=0.05, name=:ϕ)

A1 = [
    +0 +0 +1;
    +0 +0 -1;
    +0 +1 +0;
    +0 -1 +0;
    +1 +0 +0;
    -1 +0 +0;
    +1 +1 +1;
    +1 +1 -1;
    +1 -1 +1;
    +1 -1 -1;
    -1 +1 +1;
    -1 +1 -1;
    -1 -1 +1;
    -1 -1 -1.0;
    ]
b1 = 0.45*[ones(6); 1.6ones(8)]
b2 = 0.45*[ones(6); 1.2ones(8)]
build_polytope!(vis[:collision][:p1][:in], A1, b1, name=:polytope_1, color=orange)
build_polytope!(vis[:collision][:p2][:in], A1, b2, name=:polytope_2, color=light_blue)
build_polytope!(vis[:collision][:p1][:out], A1, b1, name=:polytope_1, color=α_orange)
build_polytope!(vis[:collision][:p2][:out], A1, b2, name=:polytope_2, color=α_light_blue)




function set_collision!(vis::Visualizer, α)
    tα = α / 2.0
    α_start = [0,0,-0.75]
    α_goal = [0,0.75α,-0.75]
    set_segment!(vis, α_start, α_goal, name=:α)

    ϕ = α - 1
    ϕ_start = [0,0,-0.5]
    ϕ_goal = [0,0.75ϕ,-0.5]
    set_segment!(vis, ϕ_start, ϕ_goal, name=:ϕ)

    if α > 1.0
        scaling = MeshCat.LinearMap(α * I)
        settransform!(vis[:p1][:out], scaling)
        settransform!(vis[:p2][:out], scaling)
        scaling = MeshCat.LinearMap(I)
        settransform!(vis[:p1][:in], scaling)
        settransform!(vis[:p2][:in], scaling)
    else
        scaling = MeshCat.LinearMap(α * I)
        settransform!(vis[:p1][:in], scaling)
        settransform!(vis[:p2][:in], scaling)
        scaling = MeshCat.LinearMap(I)
        settransform!(vis[:p1][:out], scaling)
        settransform!(vis[:p2][:out], scaling)
    end
end

################################################################################
# scaling
################################################################################
p1 = [-0.75, 0.5]
p2 = [+0.75, 0.5]
q1 = [+0.5]
q2 = [-0.5]
RobotVisualizer.set_2d_polytope!(vis[:collision], p1, q1, name=:p1)
RobotVisualizer.set_2d_polytope!(vis[:collision], p2, q2, name=:p2)

anim = MeshCat.Animation(40)
A = [Vector(0.001:0.005:2); Vector(2:-0.0049:0.0)]
for (i,α) in enumerate(A)
    atframe(anim, i) do
        set_collision!(vis[:collision], α)
    end
end
setanimation!(vis, anim)


################################################################################
# separated
################################################################################
p1 = [-0.75, 0.5]
p2 = [+0.75, 0.5]
q1 = [+0.5]
q2 = [-0.5]
RobotVisualizer.set_2d_polytope!(vis[:collision], p1, q1, name=:p1)
RobotVisualizer.set_2d_polytope!(vis[:collision], p2, q2, name=:p2)

anim = MeshCat.Animation(40)
A = Vector(1.0:0.0025:1.64)
A = [A; 1.6436*ones(160)]
for (i,α) in enumerate(A)
    atframe(anim, i) do
        set_collision!(vis[:collision], α)
    end
end
setanimation!(vis, anim)

################################################################################
# overlapping
################################################################################
p1 = [-0.40, 0.5]
p2 = [+0.40, 0.5]
q1 = [+0.5]
q2 = [-0.5]
RobotVisualizer.set_2d_polytope!(vis[:collision], p1, q1, name=:p1)
RobotVisualizer.set_2d_polytope!(vis[:collision], p2, q2, name=:p2)

anim = MeshCat.Animation(40)
A = Vector(1.0:-0.0007:0.8813)
A = [A; 0.8813*ones(160)]
for (i,α) in enumerate(A)
    atframe(anim, i) do
        set_collision!(vis[:collision], α)
    end
end
setanimation!(vis, anim)

RobotVisualizer.convert_frames_to_video_and_gif("scaling_slow")
RobotVisualizer.convert_frames_to_video_and_gif("overlapping_slow")
RobotVisualizer.convert_frames_to_video_and_gif("separated_slow")
