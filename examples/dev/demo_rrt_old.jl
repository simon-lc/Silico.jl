using Graphs
using GraphRecipes
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
# demo
################################################################################
timestep = 0.05;
gravity = -1*9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.5


mech = get_quasistatic_sphere_box(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    control_mode=:robot,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-5,
        compressed_search_direction=true,
        sparse_solver=true,
        differentiate=false,
        warm_start=false,
        # complementarity_decoupling=true
        )
    );


################################################################################
# test simulation
################################################################################
x0_box = [+1.0, 0.4, -0.00]
x0_sphere = [-0.0, 0.4, -0.00]
z0 = [x0_box; x0_sphere]

x1_box = [+2.0, 0.4, -0.00]
x1_sphere = [-0.0, 0.2, -0.00]
z1 = [x1_box; x1_sphere]

u0 = [0; 0; 0; x0_sphere]
H0 = 140

ctrl = open_loop_controller([u0])
@elapsed storage = simulate!(mech, deepcopy(z0), H0, controller=ctrl)
scatter(storage.iterations)

visualize!(vis, mech, storage, build=true)


################################################################################
# RRT
################################################################################
function mahalanobis_metric(mechanism::Mechanism, q̄; γ=1e-5, ρ=3e-4)
    # set the complementarity_tolerance to ρ
    old_complementarity_tolerance = mechanism.solver.options.complementarity_tolerance
    mechanism.solver.options.complementarity_tolerance = ρ
    mechanism.solver.central_paths.tolerance_central_path .*= ρ / old_complementarity_tolerance

    nq = mechanism.dimensions.state
    nu = mechanism.dimensions.input
    ū = [zeros(3); q̄[4:6]] # [box; sphere]
    B = zeros(nq, nu)
    μ = zeros(nq)

    quasistatic_dynamics_jacobian_input(B, mechanism, q̄, ū)
    # this is doubling the computation, we need to remove this or leverage the solver's internal logic
    dynamics(μ, mechanism, q̄, ū)

    uBa = B[1:3,4:6] # only the unactuated states, only the control of the actuated bodies
    Σγ = uBa * uBa' + γ*I
    μu = μ[1:3]

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

function index_nearest(mechanism::Mechanism, vertices, metrics, q_subgoal; γ=1e-5, ρ=3e-4)
    i_nearest = 0
    min_distance = +Inf

    for (i, q̄) in enumerate(vertices)
        Σγ, μu, uBa = metrics[i]
        distance = mahalanobis_evaluation(q_subgoal, Σγ, μu)
        if distance <= min_distance
            min_distance = distance
            i_nearest = i
        end
    end
    return i_nearest
end

function extend(mechanism::Mechanism, q_nearest, metric, q_subgoal; ϵ=3e-1)
    nq = mechanism.dimensions.state

    # compute the smooth dynamics approximation
    Σγ, μu, uBa = metric

    # find control that best matches desired step
    qu_subgoal = q_subgoal[1:3]
    δu = -(uBa'*uBa + 1e-2*I) \ (uBa'*(μu - qu_subgoal))
    δu = δu ./ (1e-5 + norm(δu))

    # compute the control
    qa_nearest = q_nearest[4:6]
    u = [zeros(3); qa_nearest + ϵ * δu]

    # compute the new vertex
    q_new = zeros(nq)
    dynamics(q_new, mechanism, q_nearest, u)
    return q_new
end

function sample_subgoal(mechanism::Mechanism)
    nq = mechanism.dimensions.state

    qu_min = [0.00, 0.4, -1.0*π]
    qu_max = [2.00, 0.7, +1.0*π]
    qa_min = [0.00, 0.1, -0.0*π]
    qa_max = [2.00, 1.0, +0.0*π]
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
function feasibility_projection(mechanism::Mechanism, q)
    nq = mechanism.dimensions.state
    nu = mechanism.dimensions.state

    qa = q[4:6]
    # u = [zeros(3); qa]
    u = zeros(6)
    q_projected = zeros(nq)
    dynamics(q_projected, mechanism, q, u)
    return q_projected
end

function contact_sample(mechanism::Mechanism, q_nearest)
    q_contact = deepcopy(q_nearest)
    # q_contact[4:6] .= 0.5 * q_contact[4:6] + 0.5 * q_contact[1:3]
    q_contact[4:6] .= 0.05 * q_contact[4:6] + 0.95 * q_contact[1:3]
    qa_contact = q_contact[4:6]
    u = [zeros(3); qa_contact]
    q_new = zeros(6)
    dynamics(q_new, mechanism, q_nearest, u)
    return q_new
end

function rrt_solve!(mechanism::Mechanism, q_init, q_goal, K::Int; γ=1e-5, ρ=3e-4, ϵ=3e-1, goal_distance=1.0)
    tree = SimpleDiGraph(1)
    vertices = [q_init]
    metrics = [mahalanobis_metric(mechanism, q_init; γ=γ, ρ=ρ)]


    for i = 1:K
        q_subgoal = sample_subgoal(mechanism)
        set_mechanism!(vis, mech, q_subgoal, name=:subgoal)


        i_nearest = index_nearest(mechanism, vertices, metrics, q_subgoal, γ=γ, ρ=ρ)
        q_nearest = vertices[i_nearest]
        metric_nearest = metrics[i_nearest]

        q_new = zeros(6)
        if rand() > 0.10
            q_new = extend(mechanism, q_nearest, metric_nearest, q_subgoal, ϵ=ϵ)
        else
            q_new = contact_sample(mechanism, q_nearest)
        end
        metric_new = mahalanobis_metric(mechanism, q_new, γ=γ, ρ=ρ)



        # update tree
        push!(metrics, metric_new)
        push!(vertices, q_new)
        add_vertex!(tree)
        add_edge!(tree, i_nearest, nv(tree))

        set_mechanism!(vis, mech, q_nearest, name=:nearest)
        set_mechanism!(vis, mech, q_new, name=:new)

        # strict_metric_new = mahalanobis_metric(mechanism, q_new, γ=1e-5, ρ=3e-4)
        distance = mahalanobis_evaluation(q_goal, metric_new[1], metric_new[2])
        # distance = mahalanobis_evaluation(q_goal, strict_metric_new[1], strict_metric_new[2])
        if distance <= goal_distance
            return tree, vertices
        end
    end
    return tree, vertices
end

x0_box    = [+0.60, 0.40, -0.00π]
x0_sphere = [+0.00, 0.40, -0.00π]
z0 = [x0_box; x0_sphere]

x1_box    = [+0.00, 0.40, -0.5π]
x1_sphere = [+1.50, 1.00, -0.00π]
z1 = [x1_box; x1_sphere]

set_mechanism!(vis, mech, z0, name=:start)
set_mechanism!(vis, mech, z1, name=:goal)


build_mechanism!(vis, mech, name=:start, color=RGBA(1,0,0,0.3))
build_mechanism!(vis, mech, name=:goal, color=RGBA(0,1,0,0.3))
build_mechanism!(vis, mech, name=:subgoal, color=RGBA(1,1,1,1))
build_mechanism!(vis, mech, name=:new, color=RGBA(0,0,0,1))
build_mechanism!(vis, mech, name=:nearest, color=RGBA(1,0,0,1))

settransform!(vis[:start], MeshCat.Translation(+0.8,0,0))
settransform!(vis[:goal], MeshCat.Translation(+0.4,0,0))
settransform!(vis[:subgoal], MeshCat.Translation(-1.2,0,0))
settransform!(vis[:new], MeshCat.Translation(-0.8,0,0))
settransform!(vis[:nearest], MeshCat.Translation(-0.4,0,0))

K0 = 500
γ0 = 3e-3
ρ0 = 3e-3
ϵ0 = 3e-1
tree0, vertices0 = rrt_solve!(mech, z0, z1, K0; γ=γ0, ρ=ρ0, ϵ=ϵ0, goal_distance=0.05)
tree0
vertices0

names0 = [" $i " for i = 1:nv(tree0)]
# plt = GraphRecipes.graphplot(tree0, names=names0, curvature_scalar=0.01, linewidth=3, fontsize=10)

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

visualize!(vis, mech, vertices0[trace0])
plot(hcat(vertices0[trace0]...)')
scatter!(hcat(vertices0[trace0]...)')

# RobotVisualizer.convert_frames_to_video_and_gif("rrt_rotate_and_long_push")
