using Polyhedra
using MeshCat
using RobotVisualizer
using Graphs
# using StaticArrays
# using Quaternions
using Plots
using GraphRecipes

vis = Visualizer()
open(vis)

include("../src/DojoLight.jl")

################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -1*9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.5


mech = get_quasistatic_sphere_box2(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    control_mode=:robot,
    options=Options(
        # verbose=true,
        verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        # complementarity_decoupling=true
        )
    );

# mech_object = get_quasistatic_sphere_box(;
#     timestep=1.0, #5e-2,#timestep,
#     gravity=0.00000000000000000000,#gravity,
#     mass=mass,
#     inertia=inertia,
#     friction_coefficient=friction_coefficient,
#     method_type=:symbolic,
#     # method_type=:finite_difference,
#     control_mode=:object,
#     options=Options(
#         # verbose=true,
#         verbose=false,
#         complementarity_tolerance=1e-4,
#         compressed_search_direction=true,
#         max_iterations=30,
#         sparse_solver=true,
#         differentiate=false,
#         warm_start=false,
#         complementarity_correction=0.5,
#         # complementarity_decoupling=true
#         )
#     );

# solve!(mech.solver)
# Main.@profiler solve!(mech.solver)
################################################################################
# test simulation
################################################################################
x0_box = [+1.0, 0.4, -0.00]
x0_sphere = [-0.0, 0.2, -0.00]
z0 = [x0_box; x0_sphere]

x1_box = [+2.0, 0.4, -0.00]
x1_sphere = [-0.0, 0.2, -0.00]
z1 = [x1_box; x1_sphere]

# u0 = zeros(6)
# u0 = [0; 0; 0; 0; 0.2; 0]
u0 = [0; 0; 0; x0_sphere]
H0 = 140

ctrl = open_loop_controller([u0])
@elapsed storage = simulate!(mech, z0, H0, controller=ctrl)
# Main.@profiler [solve!(mech.solver) for i=1:300]
# @benchmark $solve!($(mech.solver))

scatter(storage.iterations)

################################################################################
# visualization
################################################################################
set_floor!(vis)
set_light!(vis)
set_background!(vis)
visualize!(vis, mech, storage, build=true)


################################################################################
# RRT
################################################################################
function mahalanobis_metric(mechanism::Mechanism1170, q̄, q; γ=1e-5, ρ=3e-4)
    # set the complementarity_tolerance to ρ
    old_complementarity_tolerance = mechanism.solver.options.complementarity_tolerance
    mechanism.solver.options.complementarity_tolerance = ρ
    mechanism.solver.central_paths.tolerance_central_path .*= ρ / old_complementarity_tolerance

    nq = mechanism.dimensions.state
    nu = mechanism.dimensions.input
    # q̄a = get_actuated_state()
    ū = [zeros(3); q̄[4:6]] # [box; sphere]
    B = zeros(nq, nu)
    μ = zeros(nq)

    quasistatic_dynamics_jacobian_input(B, mechanism, q̄, ū, nothing)
    # this is doubling the computation, we need to remove this or leverage the solver's internal logic
    dynamics(μ, mechanism, q̄, ū, nothing)

    uBa = B[1:3,4:6] # only the unactuated states, only the control of the actuated bodies
    Σγ = uBa * uBa' + γ*I
    qu = q[1:3]
    μu = μ[1:3]
    d = 0.5 * (qu - μu)' * (Σγ \ (qu - μu))
    # plt = spy(Σγ, markersize=40)
    # display(plt)

    # reset the complementarity_tolerance
    mechanism.solver.options.complementarity_tolerance = old_complementarity_tolerance
    mechanism.solver.central_paths.tolerance_central_path .*= old_complementarity_tolerance / ρ
    return Σγ, μu, uBa
end

function mahalanobis_evaluation(q, Σγ, μu)
    qu = q[1:3]
    d = 0.5 * (qu - μu)' * (Σγ \ (qu - μu))
    return d
end

function mahalanobis_evaluation(mechanism::Mechanism1170, q̄, q; γ=1e-5, ρ=3e-4)
    Σγ, μu, _ = mahalanobis_metric(mechanism, q̄, q; γ=γ, ρ=ρ)
    return mahalanobis_evaluation(q, Σγ, μu)
end

function index_nearest(mechanism::Mechanism1170, vertices, q_subgoal; γ=1e-5, ρ=3e-4)
    i_nearest = 0
    min_distance = +Inf

    for (i, q̄) in enumerate(vertices)
        distance = mahalanobis_evaluation(mechanism, q̄, q_subgoal; γ=γ, ρ=ρ)
        # @show distance
        if distance <= min_distance
            min_distance = distance
            i_nearest = i
        end
    end
    @show min_distance
    return i_nearest
end

function extend(mechanism::Mechanism1170, q_nearest, q_subgoal; γ=1e-5, ρ=3e-4, ϵ=3e-1)
    nq = mechanism.dimensions.state

    # compute the smooth dynamics approximation
    Σγ, μu, uBa = mahalanobis_metric(mechanism, q_nearest, q_subgoal; γ=γ, ρ=ρ)

    # find control that best matches desired step
    qu_subgoal = q_subgoal[1:3]
    # @show round.(qu_subgoal, digits=5)
    # @show round.(μu, digits=5)
    # @show round.(uBa, digits=5)
    # δu = uBa \ -(μu - qu_subgoal)
    δu = -(uBa'*uBa + 1e-2*I) \ (uBa'*(μu - qu_subgoal))
    # plt = spy(uBa, markersize=40)
    # display(plt)
    # @show norm(uBa * δu + μu - qu_subgoal)
    # @show round.(δu, digits=5)
    δu = δu ./ (1e-5 + norm(δu))
    # @show round.(δu, digits=5)
    # @show round.(δu, digits=5)

    # compute the control
    qa_nearest = q_nearest[4:6]
    u = [zeros(3); qa_nearest + ϵ * δu]
    # @show round.(u, digits=5)
    # compute the new vertex
    q_new = zeros(nq)

    # @show round.(q_nearest, digits=5)
    dynamics(q_new, mechanism, q_nearest, u, nothing)
    return q_new
end

function sample_subgoal(mechanism::Mechanism1170)
    nq = mechanism.dimensions.state

    qu_min = [0.60, 0.4, -1.0*π]
    qu_max = [0.90, 0.7, +1.0*π]
    qa_min = [0.00, 0.1, -1.0*π]
    qa_max = [0.30, 0.9, +1.0*π]
    q_min = [qu_min; qa_min]
    q_max = [qu_max; qa_max]
    if rand() > 0.50
        q_candidate = deepcopy(z1)
    else
        α = rand(nq)
        q_candidate = α .* q_min + (1 .- α) .* q_max
    end

    # # project onto feasible space
    # q_subgoal = feasibility_projection(mechanism, q_candidate)
    # return q_subgoal
end

# TODO this is not perfect we need a different dynamics to perfectly project
# we need ignore gravity, masses etc
function feasibility_projection(mechanism::Mechanism1170, q)
    nq = mechanism.dimensions.state
    nu = mechanism.dimensions.state

    qa = q[4:6]
    # u = [zeros(3); qa]
    u = zeros(6)
    q_projected = zeros(nq)
    dynamics(q_projected, mechanism, q, u, nothing)
    return q_projected
end


# x0_box    = [+0.60, 0.40, -0.00]
# x0_sphere = [-0.00, 0.20, -0.00]
# z0 = [x0_box; x0_sphere]
#
# x1_box    = [+0.61, 0.40, -0.00]
# x1_sphere = [-0.00, 0.20, -0.00]
# z1 = [x1_box; x1_sphere]
#
# x2_box    = [+0.61, 0.40, -0.00]
# x2_sphere = [+0.01, 0.20, -0.00]
# z2 = [x2_box; x2_sphere]
#
# x3_box    = [+0.60, 0.40, -0.00]
# x3_sphere = [+0.01, 0.20, -0.00]
# z3 = [x3_box; x3_sphere]
#
# set_mechanism!(vis, mech, z0)
# set_mechanism!(vis, mech, z1)
# set_mechanism!(vis, mech, z2)
# set_mechanism!(vis, mech, z3)
# mahalanobis_metric(mech, z0, z0, ρ=1e+1)
# mahalanobis_metric(mech, z0, z0, ρ=1e-0)
# mahalanobis_metric(mech, z0, z0, ρ=1e-1)
# mahalanobis_metric(mech, z0, z0, ρ=1e-2)
# mahalanobis_metric(mech, z0, z0, ρ=1e-3)
# mahalanobis_metric(mech, z0, z0, ρ=1e-4)
# mahalanobis_metric(mech, z0, z0, ρ=1e-5)
# mahalanobis_metric(mech, z0, z0, ρ=1e-6)
# mahalanobis_metric(mech, z0, z0, ρ=1e-7)
# mahalanobis_metric(mech, z0, z0, ρ=1e-8)
# mahalanobis_evaluation(mech, z0, z0)
# mahalanobis_evaluation(mech, z0, z1)
# mahalanobis_evaluation(mech, z0, z2)
# mahalanobis_evaluation(mech, z0, z3)
#
# mahalanobis_evaluation(mech, z0, z0)
# mahalanobis_evaluation(mech, z1, z0)
# mahalanobis_evaluation(mech, z2, z0)
# mahalanobis_evaluation(mech, z3, z0)
#
# index_nearest(mech, [z0, z1, z2, z3], z0)
# x0_box    = [+0.60, 1.40, -0.00]
# x0_sphere = [+0.00, 1.40, -0.00]
# z0 = [x0_box; x0_sphere]
#
# x1_box    = [+0.70, 1.40, -0.00]
# x1_sphere = [+0.10, 1.40, -0.00]
# z1 = [x1_box; x1_sphere]
#
# z05 = extend(mech, z0, z1; γ=1e-5, ρ=3e-4, ϵ=3e-1)
# set_mechanism!(vis, mech, z0)
# set_mechanism!(vis, mech, z05)
# set_mechanism!(vis, mech, z1)
#
# x0_box    = [+0.60, 1.40, -0.50]
# x0_sphere = [+0.00, 1.40, -0.50]
# z0 = [x0_box; x0_sphere]
#
# x1_box    = [+0.60, 1.40, -0.00]
# x1_sphere = [+0.10, 1.40, -0.00]
# z1 = [x1_box; x1_sphere]
#
# z0π = feasibility_projection(mech, z0)
# z1π = feasibility_projection(mech, z1)
# set_mechanism!(vis, mech, z0)
# set_mechanism!(vis, mech, z0π)
# set_mechanism!(vis, mech, z1)
# set_mechanism!(vis, mech, z1π)
#
#
# set_mechanism!(vis, mech, sample_subgoal(mech))
# set_mechanism!(vis, mech, sample_subgoal(mech_object))
#
# mech.bodies


function rrt_solve!(mechanism::Mechanism1170, q_init, q_goal, K::Int; γ=1e-5, ρ=3e-4, ϵ=3e-1, goal_distance=1.0)
    tree = SimpleDiGraph(1)
    vertices = [q_init]

    for i = 1:K
        q_subgoal = sample_subgoal(mechanism)
        set_mechanism!(vis, mech, q_subgoal, name=:subgoal)


        i_nearest = index_nearest(mechanism, vertices, q_subgoal, γ=γ, ρ=ρ)
        # @show i_nearest
        q_nearest = vertices[i_nearest]

        q_new = extend(mechanism, q_nearest, q_subgoal)
        extend(mechanism, q_nearest, q_subgoal; γ=γ, ρ=ρ, ϵ=ϵ)
        # update tree
        push!(vertices, q_new)
        add_vertex!(tree)
        add_edge!(tree, i_nearest, nv(tree))

        set_mechanism!(vis, mech, q_nearest, name=:nearest)
        set_mechanism!(vis, mech, q_new, name=:new)
        sleep(0.02)
        distance = mahalanobis_evaluation(mechanism, q_new, q_goal; γ=1e-5, ρ=3e-4)
        # @show distance
        if distance <= goal_distance
            return tree, vertices
        end
    end
    return tree, vertices
end

x0_box    = [+0.60, 0.40, -0.00π]
x0_sphere = [+0.00, 0.40, -0.00π]
z0 = [x0_box; x0_sphere]

x1_box    = [+0.97, 0.50, -0.10π]
x1_sphere = [+0.30, 0.40, -0.00π]
# x1_box    = [+0.98, 0.57, -0.25π]
# x1_sphere = [+0.23, 0.65, -0.00π]
z1 = [x1_box; x1_sphere]

set_mechanism!(vis, mech, z0, name=:start)
set_mechanism!(vis, mech, z1, name=:goal)


build_mechanism!(vis, mech, name=:start, color=RGBA(1,1,1,0.3))
build_mechanism!(vis, mech, name=:goal, color=RGBA(1,1,1,0.3))
build_mechanism!(vis, mech, name=:subgoal, color=RGBA(1,1,1,1))
build_mechanism!(vis, mech, name=:new, color=RGBA(0,0,0,1))
build_mechanism!(vis, mech, name=:nearest, color=RGBA(1,0,0,1))

settransform!(vis[:start], MeshCat.Translation(+0.8,0,0))
settransform!(vis[:goal], MeshCat.Translation(+0.4,0,0))
settransform!(vis[:subgoal], MeshCat.Translation(-1.2,0,0))
settransform!(vis[:new], MeshCat.Translation(-0.8,0,0))
settransform!(vis[:nearest], MeshCat.Translation(-0.4,0,0))

K0 = 80
γ0 = 3e-3
ρ0 = 3e-3
tree0, vertices0 = rrt_solve!(mech, z0, z1, K0; γ=γ0, ρ=ρ0, ϵ=8e-1, goal_distance=0.05)
tree0
vertices0

names = [" $i " for i = 1:nv(tree0)]
plt = GraphRecipes.graphplot(tree0, names=names, curvature_scalar=0.01, linewidth=3, fontsize=10)

i_nearest0 = index_nearest(mech, vertices0, z1; γ=γ0, ρ=3e-3)
trace0 = [i_nearest0]
while true
    parents = inneighbors(tree0, trace0[1])
    if length(parents) == 1
        pushfirst!(trace0, parents[1])
    elseif length(parents) == 0
        break
    else
        error("many parents???")
    end
end


trace0
visualize!(vis, mech, vertices0[1:6])
visualize!(vis, mech, vertices0[trace0])


plot(hcat(vertices0[trace0]...)')
scatter!(hcat(vertices0[trace0]...)')


RobotVisualizer.convert_frames_to_video_and_gif("rrt_tilted_box")


# Σγ, μu, uBa = mahalanobis_metric(mech, z0, z1; γ=1e-5, ρ=3e-4)
# μ = zeros(6)
# dynamics(μ, mech, z0, [0;0;0; z0[4:6]], nothing)
# μ
#
# qu_subgoal = z1[1:3]
# qu_subgoal - μu
# uBa
# δu = uBa \ -(μu - qu_subgoal)
# δu = (uBa'*uBa + 1e-1*I) \ (uBa'*(μu - qu_subgoal))
# uBa * δu
#
#
#
