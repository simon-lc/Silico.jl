function dynamics(z1, mechanism::Mechanism, z, u, w)
    solver = mechanism.solver
    solver.options.differentiate = false

    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(solver)

    get_next_state!(z1, mechanism)
    return nothing
end

function dynamics_jacobian_state(dz, mechanism::Mechanism{T,D,NB}, z, u, w) where {T,D,NB}
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]

    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(solver)

    # idx_parameters = solver.indices.parameter_keywords[:state]
    idx_parameters_state = mechanism.indices.parameter_state
    idx_solution_state = mechanism.indices.solution_state
    idx_velocity = vcat([6(i-1) .+ (4:6) for i=1:NB]...)
    idx_pose = vcat([6(i-1) .+ (1:3) for i=1:NB]...)
    dz[idx_pose,:] .= timestep * solver.data.solution_sensitivity[idx_solution_state, idx_parameters_state]
    dz[idx_pose,idx_pose] .+= I(length(idx_pose))
    dz[idx_velocity,:] .= solver.data.solution_sensitivity[idx_solution_state, idx_parameters_state]
    return nothing
end

function dynamics_jacobian_input(du, mechanism::Mechanism{T,D,NB}, z, u, w) where {T,D,NB}
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]

    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(solver)

    # idx_parameters = solver.indices.parameter_keywords[:input]
    idx_parameters = mechanism.indices.input
    idx_solution = mechanism.indices.solution_state
    idx_velocity = vcat([6(i-1) .+ (4:6) for i=1:NB]...)
    idx_pose = vcat([6(i-1) .+ (1:3) for i=1:NB]...)
    du[idx_pose,:] .= timestep * solver.data.solution_sensitivity[idx_solution, idx_parameters]
    du[idx_velocity,:] .= solver.data.solution_sensitivity[idx_solution, idx_parameters]
    return nothing
end

# function dynamics_jacobian_parameters(dθ, mechanism::Mechanism{T,D,NB}, z, u, parameters_idx, w) where {T,D,NB}
function dynamics_jacobian_parameters(dθ, mechanism::Mechanism{T,D,NB}, z, u, w) where {T,D,NB}
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]

    # mechanism.solver.parameters[parameters_idx] .= θ
    # update_nodes!(mechanism)
    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(solver)

    idx_parameters_state = mechanism.indices.parameter_state
    idx_solution_state = mechanism.indices.solution_state
    idx_velocity = vcat([6(i-1) .+ (4:6) for i=1:NB]...)
    idx_pose = vcat([6(i-1) .+ (1:3) for i=1:NB]...)
    # dθ[idx_pose,:] .= timestep * solver.data.solution_sensitivity[idx_solution_state, parameters_idx]
    dθ[idx_pose,:] .= timestep * solver.data.solution_sensitivity[idx_solution_state, :]
    dθ[idx_pose, idx_parameters_state[idx_pose]] .+= I(length(idx_pose))
    # dθ[idx_velocity,:] .= solver.data.solution_sensitivity[idx_solution_state, parameters_idx]
    dθ[idx_velocity,:] .= solver.data.solution_sensitivity[idx_solution_state, :]
    return nothing
end

function quasistatic_dynamics_jacobian_state(dz, mechanism::Mechanism{T,D,NB}, z, u, w) where {T,D,NB}
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]

    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(solver)

    # idx_parameters = solver.indices.parameter_keywords[:state]
    idx_parameters_state = mechanism.indices.parameter_state
    idx_solution_state = mechanism.indices.solution_state
    dz .= timestep * solver.data.solution_sensitivity[idx_solution_state, idx_parameters_state]
    dz .+= Diagonal(ones(size(dz,1)))
    return nothing
end

function quasistatic_dynamics_jacobian_input(du, mechanism::Mechanism{T,D,NB}, z, u, w) where {T,D,NB}
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]

    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(solver)

    # idx_parameters = solver.indices.parameter_keywords[:input]
    idx_parameters = mechanism.indices.input
    idx_solution = mechanism.indices.solution_state
    du .= timestep * solver.data.solution_sensitivity[idx_solution, idx_parameters]
    return nothing
end
