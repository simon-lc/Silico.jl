using Graphs
using GraphRecipes
using Plots
using Statistics
using Random
using BenchmarkTools

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# demo
################################################################################
timestep = 0.05
gravity = -1*9.81
mass = 1.0
inertia = 0.8 * ones(1,1)
friction_coefficient = 0.5


mech = get_polytope_grasper(;
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
        compressed_search_direction=false,
        sparse_solver=false,
        differentiate=false,
        warm_start=false,
        complementarity_backstep=1e-2,
        )
    )
# mech.bodies[1].mass ./= 3
# update_parameters!(mech)
mech.bodies
mech.contacts

################################################################################
# test simulation
################################################################################
x0_box       = [0.0, 0.3, +0.0π]
x0_capsule_2 = [-0.5, 0.5, -0.5π]
x0_capsule_4 = [+0.5, 0.5, -0.5π]
z0 = [x0_box; x0_capsule_2; x0_capsule_4]

u0 = [x0_box; x0_capsule_2; x0_capsule_4]

x1_capsule_2 = [-0.25, 0.35, -0.50π]
x1_capsule_4 = [+0.25, 0.35, -0.50π]
u0 = [x0_box; x1_capsule_2; x1_capsule_4]
H0 = 140
U = [u0 + i/H0 * [0, 0, 0, 0, 1, 0, 0, 1, 0] for i = 1:H0]

ctrl = open_loop_controller([u0])
ctrl = open_loop_controller(U)
@elapsed storage = simulate!(mech, deepcopy(z0), H0, controller=ctrl)

visualize!(vis, mech, storage, show_contact=false)

include("rrt_methods.jl")





################################################################################
# test RRT
################################################################################
x0_box       = [+0.00, 0.25, -0.00π]
x0_capsule_2 = [-0.30, 0.30, -0.50π]
x0_capsule_4 = [+0.30, 0.30, -0.50π]
z0 = [x0_box; x0_capsule_2; x0_capsule_4]

x1_box       = [+0.20, 0.25, -0.00π] # works
x1_capsule_2 = [-0.10, 0.30, -0.50π]
x1_capsule_4 = [+0.50, 0.30, -0.50π]
z1 = [x1_box; x1_capsule_2; x1_capsule_4]


x0_box       = [+0.00, 0.25, -0.00π]
x0_capsule_2 = [-0.47, 0.40, -0.25π]
x0_capsule_4 = [+0.47, 0.40, -0.75π]
z0 = [x0_box; x0_capsule_2; x0_capsule_4]

x1_box       = [+0.20, 0.32, -0.10π]
x1_capsule_2 = [-0.10, 0.30, -0.50π]
x1_capsule_4 = [+0.50, 0.30, -0.50π]
z1 = [x1_box; x1_capsule_2; x1_capsule_4]


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

mech.dimensions
mech.bodies
mech.contacts

K0 = 1000
γ0 = 3e-3
ρ0 = 3e-3
ϵ0 = 1e-1
# ϵ0 = 0.5e-1
qu_min = [-0.10, 0.25, -0.0*π]
qu_max = [+0.40, 0.35, +0.0*π]
qa_min = 0*[0, 0, 0]
qa_max = 0*[0, 0, 0]
q_min0 = [qu_min; qa_min; qa_min]
q_max0 = [qu_max; qa_max; qa_max]

function sample_method(q)
    antipodal = q[7:8] - q[4:5]
    antipodal ./= 1e-5 + norm(antipodal)
    antipodal .*= 0.30
    δqa = [q[1:2] - antipodal; -0.25π; q[1:2] + antipodal;  -0.75π;] - q[4:9]
    # δqa = [q[1:3] - q[4:6]; q[1:3] - q[7:9]]
    return δqa
end

tree0, vertices0, metrics0 = rrt_solve!(mech, z0, z1, K0;
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

visualize!(vis, mech, vertices0[trace0])
plot(hcat(vertices0[trace0]...)')
scatter!(hcat(vertices0[trace0]...)')

# RobotVisualizer.convert_frames_to_video_and_gif("rrt_box_lift")
mech.solver.options.residual_tolerance
mech.solver.options.complementarity_tolerance



function max_to_min(z_max; l=0.5)
    c2 = z_max[1:2]
    c4 = z_max[4:5]
    θ2 = z_max[3]
    θ4 = z_max[6]
    v2 = [cos(θ2), sin(θ2)]
    v4 = [cos(θ4), sin(θ4)]
    p1 = c2 - l/2 * v2
    p3 = c4 - l/2 * v4
    if norm(p3 - p1) >= 2l
        p0 = (p1 + p3) / 2
        Δ = p3 - p1
        α0 = atan(Δ[2], Δ[1]) - π/2
        α1 = π/2
        α2 = π + θ2 - atan(Δ[2], Δ[1])
        α3 = - θ4 + atan(Δ[2], Δ[1])
    else
        Δ = p3 - p1
        α0 = atan(Δ[2], Δ[1]) - π/2

        p = (p1 + p3) / 2
        h = sqrt(l^2 - norm(p - p1)^2)
        p0 = p + h * [cos(α0), sin(α0)]

        α1 = acos(h/l)
        α2 = α0 + α1 + θ2
        α3 = -(α0 - α1 + θ4)
    end
    z_min = [p0, α0, α1, α2, α3]
    return z_min
end

function min_to_max(z_min; l=0.5)
    p0 = z_min[1:2]
    α0 = z_min[3]
    α1 = z_min[4]
    α2 = z_min[5]
    α3 = z_min[6]

    θ1 = α0 - α1
    θ2 = α0 - α1 + α2
    c2 = p0 + l * [cos(θ1), sin(θ1)] + l/2 * [cos(θ2), sin(θ2)]

    θ3 = α0 + α1
    θ4 = α0 + α1 - α3
    c4 = p0 + l * [cos(θ3), sin(θ3)] + l/2 * [cos(θ4), sin(θ4)]

    z_max = [c2; θ2; c4; θ4]
    return z_max
end

function clamp_min(z; l=0.5,
        z_high=[+10.0,+10.0,+2.00π,+0.00π,+0.50π,+0.40π,+0.45π,+0.45π],
        z_low= [-10.0,-10.0,-2.00π,-1.00π,+0.00π,+0.20π,+0.00π,+0.00π],
        )
    z̄ = clamp.(z, z_min, z_max)
    return z̄
end
