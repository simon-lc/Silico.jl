function update_parameters!(mechanism::Mechanism)
    bodies = mechanism.bodies
    contacts = mechanism.contacts
    solver = mechanism.solver

    off = 0
    for node in [bodies; contacts]
        θ = get_parameters(node)
        nθ = parameter_dimension(node)
        solver.parameters[off .+ (1:nθ)] .= θ; off += nθ
    end
    # update the consistency logic
    solver.consistency.solved .= false
    Mehrotra.set_bool!(solver.consistency.differentiated, false)
    return nothing
end

function update_nodes!(mechanism::Mechanism)
    bodies = mechanism.bodies
    contacts = mechanism.contacts
    solver = mechanism.solver

    off = 0
    for node in [bodies; contacts]
        nθ = parameter_dimension(node)
        θ = solver.parameters[off .+ (1:nθ)]; off += nθ
        set_parameters!(node, θ)
    end
    return nothing
end

function set_input!(mechanism::Mechanism, u)
    off = 0
    for body in mechanism.bodies
        nu = length(body.input)
        body.input .= u[off .+ (1:nu)]; off += nu
    end
    return nothing
end

function get_input(mechanism::Mechanism{T,D,NB}) where {T,D,NB}
    off = 0
    nu = sum(input_dimension.(mechanism.bodies))
    u = zeros(nu)
    for body in mechanism.bodies
        ni = length(body.input)
        u[off .+ (1:ni)] .= body.input; off += ni
    end
    return u
end

function set_current_state!(mechanism::Mechanism, z)
    off = 0

    for body in mechanism.bodies
        nx = state_dimension(body)
        set_current_state!(body, view(z, off .+ (1:nx))); off += nx
    end
    return nothing
end

function get_current_state(mechanism::Mechanism{T,D,NB}) where {T,D,NB}
    nz = sum(state_dimension.(mechanism.bodies))

    off = 0
    z = zeros(nz)
    for body in mechanism.bodies
        zi = get_current_state(body)
        ni = state_dimension(body)
        z[off .+ (1:ni)] .= zi; off += ni
    end
    return z
end

function get_next_state(mechanism::Mechanism{T,D,NB}) where {T,D,NB}
    nz = sum(state_dimension.(mechanism.bodies))
    z = zeros(nz)
    get_next_state!(z, mechanism)
    return z
end

function get_next_state(variables, body::Body)
    z = zeros(state_dimension(body))
    get_next_state!(z, variables, body)
    return z
end

function get_next_pose(variables, body::AbstractBody)
    z = zeros(state_dimension(body))
    get_next_state!(z, variables, body)
    return z[1:pose_dimension(body)]
end

function get_next_state!(z, mechanism::Mechanism{T,D,NB}) where {T,D,NB}
    variables = mechanism.solver.solution.all

    off = 0
    for body in mechanism.bodies
        nz = state_dimension(body)
        get_next_state!(view(z, off .+ (1:nz)), variables, body); off += nz
    end
    return nothing
end

function step!(mechanism::Mechanism, z0, u)
    set_current_state!(mechanism, z0)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(mechanism.solver)
    update_nodes!(mechanism)
    z1 = get_next_state(mechanism)
    return z1
end

function step!(mechanism::Mechanism, z0; controller::Function=m->nothing)
    set_current_state!(mechanism, z0)
    controller(mechanism) # sets the control inputs u
    update_parameters!(mechanism)
    Mehrotra.solve!(mechanism.solver)
    update_nodes!(mechanism)
    z1 = get_next_state(mechanism)
    return z1
end

function simulate!(mechanism::Mechanism{T}, z0, H::Int;
        controller::Function=(m,i)->nothing) where T

    storage = TraceStorage(mechanism.dimensions, H, T)
    z = copy(z0)
    for i = 1:H
        z .= step!(mechanism, z, controller=m -> controller(m,i))
        record!(storage, mechanism, i)
    end
    return storage
end

function open_loop_controller(u::Vector)
    H = length(u)
    function ctrl(mechanism, i)
        set_input!(mechanism, u[min(H,i)])
        update_parameters!(mechanism)
        return nothing
    end
    return ctrl
end
