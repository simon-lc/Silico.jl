using Graphs
using GraphRecipes
using Plots
using Statistics
using Random
using BenchmarkTools
using MeshCatMechanisms
using RigidBodyDynamics


include("rrt_methods.jl")
include("grasper_visuals.jl")

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)
set_camera!(vis, zoom=4.0, cam_pos=[5,0,0.0])

green = RGBA(4/255,191/255,173/255,1.0)
turquoise = RGBA(2/255,115/255,115/255,1.0)

################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -1*9.81;
mass = 0.5;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.5


mech = get_quasistatic_sphere_box(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    num_sphere=2,
    method_type=:symbolic,
    # method_type=:finite_difference,
    control_mode=:robot,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-5,
        compressed_search_direction=true,
        sparse_solver=false,
        differentiate=false,
        warm_start=false,
        # complementarity_decoupling=true
        )
    )


################################################################################
# test simulation
################################################################################
x0_box      = [+1.0, 0.4, +0.0π]
x0_sphere_1 = [-0.0, 0.4, -0.00]
x0_sphere_2 = [-0.0, 0.8, -0.00]
z0 = [x0_box; x0_sphere_1; x0_sphere_2]

u0 = [0; 0; 0; x0_sphere_1; x0_sphere_2]
H0 = 140

ctrl = open_loop_controller([u0])
@elapsed storage = simulate!(mech, deepcopy(z0), H0, controller=ctrl)
visualize!(vis, mech, storage, show_contact=false)


################################################################################
# test RRT
################################################################################
x0_box      = [+1.00, 0.4, -0.00]
x0_sphere_1 = [+0.25, 0.4, -0.00]
x0_sphere_2 = [+1.75, 0.4, -0.00]
z0 = [x0_box; x0_sphere_1; x0_sphere_2]

x1_box    = [+0.00, 0.40, -0.5π]
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
x1_box      = [+1.0, 0.95, -0.5π] # works well
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
x1_box      = [+1.5, 1.0, +0.60π] # works well
x1_box      = [+1.5, 1.0, -0.60π] # works well
# x1_box      = [+1.5, 1.5, -0.60π]
# x1_box      = [+0.5, 1.0, -0.60π]
# x1_box      = [+1.5, 1.0, -0.00π]
x1_sphere_1 = [-0.0, 0.4, -0.00]
x1_sphere_2 = [-2.0, 0.8, -0.00]
z1 = [x1_box; x1_sphere_1; x1_sphere_2]

set_mechanism!(vis, mech, z0, name=:start)
set_mechanism!(vis, mech, z1, name=:goal)


build_mechanism!(vis, mech, name=:start, show_contact=false, color=RGBA(1,0,0,0.3))
build_mechanism!(vis, mech, name=:goal, show_contact=false, color=RGBA(0,1,0,0.3))
build_mechanism!(vis, mech, name=:subgoal, show_contact=false, color=RGBA(1,1,1,1))
build_mechanism!(vis, mech, name=:new, show_contact=false, color=RGBA(0,0,0,1))
build_mechanism!(vis, mech, name=:nearest, show_contact=false, color=RGBA(1,0,0,1))

settransform!(vis[:start], MeshCat.Translation(+0.8,0,0))
settransform!(vis[:goal], MeshCat.Translation(+0.4,0,0))
settransform!(vis[:subgoal], MeshCat.Translation(-1.2,0,0))
settransform!(vis[:new], MeshCat.Translation(-0.8,0,0))
settransform!(vis[:nearest], MeshCat.Translation(-0.4,0,0))

K0 = 1000
γ0 = 3e-3
ρ0 = 3e-3
ϵ0 = 2e-1
qu_min = [0.00, 0.4, -1.0*π]
qu_max = [2.00, 2.0, +1.0*π]
qa_min = 0*[0, 0, 0]
qa_max = 0*[0, 0, 0]
q_min0 = [qu_min; qa_min; qa_min]
q_max0 = [qu_max; qa_max; qa_max]

function sample_method(q)
    antipodal = q[7:8] - q[4:5]
    antipodal ./= 1e-5 + norm(antipodal)
    antipodal .*= 0.15
    δqa = [q[1:2] - antipodal; 0; q[1:2] + antipodal;  0;] - q[4:9]
    return δqa
end

@elapsed tree0, vertices0, metrics0 = rrt_solve!(mech, z0, z1, K0;
    γ=γ0,
    ρ=ρ0,
    ϵ=ϵ0,
    q_min=q_min0,
    q_max=q_max0,
    proba_goal_sample=0.50,
    proba_contact_sample=0.10,
    goal_distance=0.05,
    sample_method=sample_method,
    seed=0)

# names0 = [" $i " for i = 1:nv(tree0)]
# plt = GraphRecipes.graphplot(tree0, names=names0, curvature_scalar=0.01, linewidth=3, fontsize=10)

i_nearest0 = index_nearest(mech, vertices0, metrics0, z1; γ=γ0, ρ=ρ0)
trace0 = get_trace(tree0, i_nearest0)

_, anim = visualize!(vis[:rrt], mech, vertices0[trace0][3:end])
plot(hcat(vertices0[trace0]...)')
scatter!(hcat(vertices0[trace0]...)')


################################################################################
# grasper vis
################################################################################
pose_trajectory = grasper_trajectory(vertices0[trace0], segment=[1.20, 0.80], radius=0.10)
_, anim = visualize!(vis[:rrt], pose_trajectory[3:end],
    animation=anim,
    segment=[1.20, 0.80],
    radius=0.10,
    color=RGBA(0.9, 0.9, 0.9, 1.0))
scale = 0.1
offset = -0.9
scaling = MeshCat.LinearMap(I * scale)
translation = MeshCat.Translation(0.0, offset, 0.0)
transformation = MeshCat.compose(translation, scaling)
settransform!(vis[:rrt], transformation)

################################################################################
# panda vis
################################################################################
urdf = joinpath(@__DIR__, "..", "deps", "panda_end_effector.urdf")
robot = parse_urdf(urdf)
mvis = MechanismVisualizer(robot, URDFVisuals(urdf), vis)
q0 = [-π/2,
    +0.60,
    +0.00,
    -1.80,
    +0.00,
    +2.40,
    +π/2,
    0.05,
    0.05]
set_configuration!(mvis, q0)

end_effector_trajectory = deepcopy(pose_trajectory)
end_effector_trajectory = [s[1:3] .* [scale, scale, 1.0] + [offset, 0, 0]
    for s in end_effector_trajectory[3:end]]
q_trajectory = panda_trajectory(end_effector_trajectory, mvis, initial_q=q0)
mvis, anim = visualize!(mvis, q_trajectory, animation=anim)

################################################################################
# floor
################################################################################
block_1 = MeshCat.HyperRectangle(Vec(-0.17,-0.17,-0.2), Vec(0.34,0.34,0.3))

block_2 = MeshCat.HyperRectangle(Vec(-0.10,offset,-0.2), Vec(0.20,0.20,0.3))
block_mat = MeshPhongMaterial(color=RGBA(0.3, 0.3, 0.3, 1.0))
setobject!(vis[:floor][:block1], block_1, block_mat)
setobject!(vis[:floor][:block2], block_2, block_mat)
set_floor!(vis, origin=[0,0,-0.10])

box_mat = MeshPhongMaterial(color=green)
box = MeshCat.HyperRectangle(Vec(-0.4,-0.4,-0.4), Vec(0.8,0.8,0.8))
setobject!(vis[:rrt][:robot][:bodies][:box], box, box_mat)

box_mat = MeshPhongMaterial(color=green, wireframe=true)
box = MeshCat.HyperRectangle(Vec(-0.4,-0.4,-0.4), Vec(0.8,0.8,0.8))
setobject!(vis[:rrt][:goal], box, box_mat)
settransform!(vis[:rrt][:goal], MeshCat.compose(
    MeshCat.Translation(SVector{3}(0,x1_box[1],x1_box[2])),
    MeshCat.LinearMap(rotationmatrix(RotX(x1_box[3]))),
    ))

setvisible!(vis[:desired_0], false)
setvisible!(vis[:desired_1], false)
setvisible!(vis[:desired_2], false)
setvisible!(vis[:world][:link1][:link2][:link3][:link4][:link5][:link6][:link7][:after_joint7][:point_0], false)
setvisible!(vis[:world][:link1][:link2][:link3][:link4][:link5][:link6][:link7][:after_joint7][:point_1], false)
setvisible!(vis[:world][:link1][:link2][:link3][:link4][:link5][:link6][:link7][:after_joint7][:point_2], false)
setvisible!(vis[:world][:link1][:link2][:link3][:link4][:link5][:link6][:link7][:finger1], false)
setvisible!(vis[:world][:link1][:link2][:link3][:link4][:link5][:link6][:link7][:finger2], false)
