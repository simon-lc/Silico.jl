################################################################################
# RRT
################################################################################
function actuated_indices(mechanism)
    actuated_idx = Vector{Int}()
    off = 0
    for body in mechanism.bodies
        pose_dim = pose_dimension(body)
        if typeof(body) <: QuasistaticObject
        elseif typeof(body) <: QuasistaticRobot
            push!(actuated_idx, off+1:off+pose_dim...)
        else
            error("wrong body type")
        end
        off += pose_dim
    end
    return actuated_idx
end

function unactuated_indices(mechanism)
    unactuated_idx = Vector{Int}()
    off = 0
    for body in mechanism.bodies
        pose_dim = pose_dimension(body)
        if typeof(body) <: QuasistaticObject
            push!(unactuated_idx, off+1:off+pose_dim...)
        elseif typeof(body) <: QuasistaticRobot
        else
            error("wrong body type")
        end
        off += pose_dim
    end
    return unactuated_idx
end

function mahalanobis_metric(mechanism::Mechanism, q̄; γ=1e-5, ρ=3e-4)
    # set the complementarity_tolerance to ρ
    old_complementarity_tolerance = mechanism.solver.options.complementarity_tolerance
    mechanism.solver.options.complementarity_tolerance = ρ
    mechanism.solver.central_paths.tolerance_central_path .*= ρ / old_complementarity_tolerance

    nq = mechanism.dimensions.state
    nu = mechanism.dimensions.input
    actuated_idx = actuated_indices(mechanism)
    unactuated_idx = unactuated_indices(mechanism)
    ū = zeros(nu)
    ū[actuated_idx] .= q̄[actuated_idx]
    B = zeros(nq, nu)
    μ = zeros(nq)

    quasistatic_dynamics_jacobian_input(B, mechanism, q̄, ū)
    # this is doubling the computation, we need to remove this or leverage the solver's internal logic
    # dynamics(μ, mechanism, q̄, ū)
    get_next_state!(μ, mechanism)


    # uBa = B[1:3,4:6] # only the unactuated states, only the control of the actuated bodies
    uBa = B[unactuated_idx, actuated_idx] # only the unactuated states, only the control of the actuated bodies
    Σγ = uBa * uBa' + γ*I
    Σγ_inv = inv(Σγ)
    μu = μ[unactuated_idx]

    # reset the complementarity_tolerance
    mechanism.solver.options.complementarity_tolerance = old_complementarity_tolerance
    mechanism.solver.central_paths.tolerance_central_path .*= old_complementarity_tolerance / ρ
    return Σγ_inv, μu, uBa
end

function mahalanobis_evaluation(qu::Vector{T}, Σγ_inv::Matrix{T}, μu::Vector{T}) where T
    d = 0.5 * (qu - μu)' * Σγ_inv * (qu - μu)
    return d
end

function index_nearest(mechanism::Mechanism, vertices, metrics, q_subgoal; γ=1e-5, ρ=3e-4)
    unactuated_idx = unactuated_indices(mechanism)
    i_nearest = 0
    min_distance = +Inf

    for (i, q̄) in enumerate(vertices)
        Σγ_inv, μu, uBa = metrics[i]
        qu_subgoal = q_subgoal[unactuated_idx]
        distance = mahalanobis_evaluation(qu_subgoal, Σγ_inv, μu)
        if distance <= min_distance
            min_distance = distance
            i_nearest = i
        end
    end
    return i_nearest
end

function extend(mechanism::Mechanism, q_nearest, metric, q_subgoal; ϵ=3e-1)
    nq = mechanism.dimensions.state
    nu = mechanism.dimensions.input
    actuated_idx = actuated_indices(mechanism)
    unactuated_idx = unactuated_indices(mechanism)

    # compute the smooth dynamics approximation
    Σγ_inv, μu, uBa = metric

    # find control that best matches desired step
    # qu_subgoal = q_subgoal[1:3]
    qu_subgoal = q_subgoal[unactuated_idx]
    δu = -(uBa'*uBa + 1e-2*I) \ (uBa'*(μu - qu_subgoal))
    # δu = -(uBa'*uBa + 1e-2*I) \ (uBa'*(μu - qu_subgoal))
    δu = δu ./ (1e-5 + norm(δu))

    # compute the control
    qa_nearest = q_nearest[actuated_idx]
    # u = [zeros(3); qa_nearest + ϵ * δu]
    u = zeros(nu)
    u[actuated_idx] .=  qa_nearest + ϵ * δu

    # compute the new vertex
    q_new = zeros(nq)
    dynamics(q_new, mechanism, q_nearest, u)
    return q_new
end

function sample_subgoal(mechanism::Mechanism, q_goal;
        q_min=zeros(mechanism.dimensions.state),
        q_max=zeros(mechanism.dimensions.state),
        proba_goal_sample=0.50,
        )
    nq = mechanism.dimensions.state

    if rand() > proba_goal_sample
        q_candidate = deepcopy(q_goal)
    else
        α = rand(nq)
        q_candidate = α .* q_min + (1 .- α) .* q_max
    end
    # # project onto feasible space
    # q_subgoal = feasibility_projection(mechanism, q_candidate)
    # return q_subgoal
end


# # TODO this is not perfect we need a different dynamics to perfectly project
# # we need ignore gravity, masses etc
# function feasibility_projection(mechanism::Mechanism, q)
#     nq = mechanism.dimensions.state
#     nu = mechanism.dimensions.state
#
#     qa = q[4:6]
#     u = [zeros(3); qa]
#     # u = zeros(6)
#     q_projected = zeros(nq)
#     dynamics(q_projected, mechanism, q, u)
#     return q_projected
# end

function contact_sample(mechanism::Mechanism, q_nearest, sample_method::Function)
    nq = mechanism.dimensions.state
    nu = mechanism.dimensions.input
    actuated_idx = actuated_indices(mechanism)
    unactuated_idx = unactuated_indices(mechanism)

    q_contact = deepcopy(q_nearest)
    δqa = sample_method(q_contact)

    u = zeros(nu)
    u[actuated_idx] .= q_contact[actuated_idx] + δqa
    q_new = zeros(nq)
    dynamics(q_new, mechanism, q_nearest, u)
    return q_new
end

function rrt_solve!(mechanism::Mechanism, q_init, q_goal, K::Int;
        γ=1e-5,
        ρ=3e-4,
        ϵ=3e-1,
        q_min=zeros(mechanism.dimensions.state),
        q_max=zeros(mechanism.dimensions.state),
        proba_goal_sample=0.50,
        proba_contact_sample=0.10,
        goal_distance=1.0,
        sample_method=q->zeros(length(actuated_indices(mechanism))),
        seed=0)

    nq = mechanism.dimensions.state
    unactuated_idx = unactuated_indices(mechanism)
    Random.seed!(seed)
    tree = SimpleDiGraph(1)
    vertices = [q_init]
    metrics = [mahalanobis_metric(mechanism, q_init; γ=γ, ρ=ρ)]


    for i = 1:K
        q_subgoal = sample_subgoal(mechanism, q_goal,
            q_min=q_min,
            q_max=q_max,
            proba_goal_sample=proba_goal_sample)
        set_mechanism!(vis, mech, q_subgoal, name=:subgoal)

        i_nearest = index_nearest(mechanism, vertices, metrics, q_subgoal, γ=γ, ρ=ρ)
        q_nearest = vertices[i_nearest]
        metric_nearest = metrics[i_nearest]

        q_new = zeros(nq)
        if rand() > proba_contact_sample
            q_new = extend(mechanism, q_nearest, metric_nearest, q_subgoal, ϵ=ϵ)
        else
            q_new = contact_sample(mechanism, q_nearest, sample_method)
        end
        metric_new = mahalanobis_metric(mechanism, q_new, γ=γ, ρ=ρ)

        # update tree
        push!(metrics, metric_new)
        push!(vertices, q_new)
        add_vertex!(tree)
        add_edge!(tree, i_nearest, nv(tree))

        set_mechanism!(vis, mech, q_nearest, name=:nearest)
        set_mechanism!(vis, mech, q_new, name=:new)

        # qu_goal = q_goal[1:3]
        qu_goal = q_goal[unactuated_idx]
        distance = mahalanobis_evaluation(qu_goal, metric_new[1], metric_new[2])
        (distance <= goal_distance) && break
    end
    return tree, vertices, metrics
end

function get_trace(tree, idx)
    trace = [idx]
    while true
        parents = inneighbors(tree, trace[1])
        if length(parents) == 1
            pushfirst!(trace, parents[1])
        elseif length(parents) == 0
            break
        else
            error("many parents???")
        end
    end
    return trace
end
