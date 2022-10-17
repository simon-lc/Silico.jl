mutable struct Performance1130{T}
    violation::T
    iterations::T
    solve_failed::Bool
    solve_errored::Bool
end

function performance_evaluation(mechanism::Mechanism,
        timestep, complementarity_tolerance, initial_conditions,
        H; vis=Visualizer())

    max_iterations = mechanism.solver.options.max_iterations
    errored = false
    storage = nothing

    set_timestep!(mechanism, timestep)
    mechanism.solver.options.complementarity_tolerance = complementarity_tolerance
    mechanism.solver.options.residual_tolerance = complementarity_tolerance / 10

    Mehrotra.initialize_solver!(mechanism.solver)
    try
        storage = simulate!(mechanism, copy(initial_conditions), H)
        visualize!(vis, mechanism, storage)
    catch e
        println("Solver errored")
        errored = true
    end

    # iterations
    iterations = mean(storage.iterations)

    # failure
    failed = any(storage.iterations .== max_iterations)

    # violation
    violation = contact_violation(mechanism, storage)
    return Performance1130(violation, iterations, failed, errored)
end

function contact_violation(mechanism::Mechanism, storage::TraceStorage{T,H}) where {T,H}
    violation = 0.0

    for contact in mechanism.contacts
        parent_shape = contact.parent_shape
        child_shape = contact.child_shape
        detector = CollisionDetector(parent_shape, child_shape)
        if (typeof(contact)<:EnvContact2D) || (typeof(contact)<:EnvBilevelContact2D40)
            idx = find_body_index(mechanism.bodies, contact.parent_name)
            for t = 1:H
                ϕ = contact_data(storage.x[t][idx], zeros(3), detector)[1]
                violation = max(violation, max(-ϕ[1], 0.0))
            end
        end
    end
    return violation
end
