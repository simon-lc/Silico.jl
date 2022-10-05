function set_state_control_parameters!(mechanism::Mechanism, z, u; w=Vector(), idx_parameters=0:-1)
    # set mechanism.solver.parameters and updates the modes accordingly
    mechanism.solver.parameters[idx_parameters] .= w
    update_nodes!(mechanism)
    # set the parameters in mechanism.bodies and mechanism.contacts
    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    # update the mechanism.solver.parameters to be consistent with the parameters in mechanism.bodies and mechanism.contacts
    update_parameters!(mechanism)
    return nothing
end

function explicit_dynamics(mechanism::Mechanism, z, u;
        w=Vector(), idx_parameters=0:-1)

    z1 = zeros(mechanism.dimensions.state)
    dynamics(z1, mechanism, z, u; w=w, idx_parameters=idx_parameters)
    return z1
end

function dynamics(z1, mechanism::Mechanism, z, u;
        w=Vector(), idx_parameters=0:-1)

    set_state_control_parameters!(mechanism, z, u; w=w, idx_parameters=idx_parameters)
    solver = mechanism.solver
    solver.options.differentiate = false
    Mehrotra.solve!(solver)

    # extract result
    get_next_state!(z1, mechanism)
    return nothing
end

function dynamics_jacobian_state(dz, mechanism::Mechanism{T,D,NB}, z, u;
        w=Vector(), idx_parameters=0:-1) where {T,D,NB}

    set_state_control_parameters!(mechanism, z, u; w=w, idx_parameters=idx_parameters)
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]
    Mehrotra.solve!(solver)

    # extract result
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


function dynamics_jacobian_input(du, mechanism::Mechanism{T,D,NB}, z, u;
        w=Vector(), idx_parameters=0:-1) where {T,D,NB}

    set_state_control_parameters!(mechanism, z, u; w=w, idx_parameters=idx_parameters)
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]
    Mehrotra.solve!(solver)

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
function dynamics_jacobian_parameters(dw, mechanism::Mechanism{T,D,NB}, z, u;
        w=Vector(), idx_parameters=0:-1) where {T,D,NB}

    set_state_control_parameters!(mechanism, z, u; w=w, idx_parameters=idx_parameters)
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]
    Mehrotra.solve!(solver)

    idx_parameters_state = mechanism.indices.parameter_state
    idx_solution_state = mechanism.indices.solution_state
    idx_velocity = vcat([6(i-1) .+ (4:6) for i=1:NB]...)
    idx_pose = vcat([6(i-1) .+ (1:3) for i=1:NB]...)

    dw[idx_pose,:] .= timestep * solver.data.solution_sensitivity[idx_solution_state, idx_parameters]
    for i in idx_pose
        if idx_parameters_state[i] ∈ idx_parameters
            dw[i, idx_parameters_state[i]] .+= 1.0
        end
    end
    dw[idx_velocity,:] .= solver.data.solution_sensitivity[idx_solution_state, idx_parameters]
    return nothing
end


function dynamics_jacobian_state_parameters(dz, dw, mechanism::Mechanism{T,D,NB}, z, u;
        w=Vector(), idx_parameters=0:-1) where {T,D,NB}

    set_state_control_parameters!(mechanism, z, u; w=w, idx_parameters=idx_parameters)
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]
    Mehrotra.solve!(solver)

    # extract result
    # idx_parameters = solver.indices.parameter_keywords[:state]
    idx_parameters_state = mechanism.indices.parameter_state
    idx_solution_state = mechanism.indices.solution_state
    idx_velocity = vcat([6(i-1) .+ (4:6) for i=1:NB]...)
    idx_pose = vcat([6(i-1) .+ (1:3) for i=1:NB]...)

    dz[idx_pose,:] .= timestep * solver.data.solution_sensitivity[idx_solution_state, idx_parameters_state]
    dz[idx_pose,idx_pose] .+= I(length(idx_pose))
    dz[idx_velocity,:] .= solver.data.solution_sensitivity[idx_solution_state, idx_parameters_state]

    dw[idx_pose,:] .= timestep * solver.data.solution_sensitivity[idx_solution_state, idx_parameters]
    for i in idx_pose
        if idx_parameters_state[i] ∈ idx_parameters
            dw[i, idx_parameters_state[i]] .+= 1.0
        end
    end
    dw[idx_velocity,:] .= solver.data.solution_sensitivity[idx_solution_state, idx_parameters]
    return nothing
end

function quasistatic_dynamics_jacobian_state(dz, mechanism::Mechanism{T,D,NB}, z, u;
        w=Vector(), idx_parameters=0:-1) where {T,D,NB}

    set_state_control_parameters!(mechanism, z, u; w=w, idx_parameters=idx_parameters)
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]
    DojoLight.solve!(solver)

    idx_parameters_state = mechanism.indices.parameter_state
    idx_solution_state = mechanism.indices.solution_state
    dz .= timestep * solver.data.solution_sensitivity[idx_solution_state, idx_parameters_state]
    dz .+= Diagonal(ones(size(dz,1)))
    return nothing
end

function quasistatic_dynamics_jacobian_input(du, mechanism::Mechanism{T,D,NB}, z, u;
        w=Vector(), idx_parameters=0:-1) where {T,D,NB}

    set_state_control_parameters!(mechanism, z, u; w=w, idx_parameters=idx_parameters)
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]
    DojoLight.solve!(solver)

    idx_parameters = mechanism.indices.input
    idx_solution = mechanism.indices.solution_state
    du .= timestep * solver.data.solution_sensitivity[idx_solution, idx_parameters]
    return nothing
end
