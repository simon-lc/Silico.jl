################################################################################
# body
################################################################################
struct QuasistaticObject{T,D} <: Body{T}
    name::Symbol
    index::NodeIndices
    pose::Vector{T}
    input::Vector{T}
    gravity::Vector{T}
    timestep::Vector{T}
    mass::Vector{T}
    inertia::Matrix{T}
    shapes::Vector
end

function QuasistaticObject(timestep::T, mass, inertia::Matrix,
        shapes::Vector;
        gravity=-9.81,
        name::Symbol=:body,
        index::NodeIndices=NodeIndices()) where T

    D = 2
    return QuasistaticObject{T,D}(
        name,
        index,
        zeros(D+1),
        zeros(D+1),
        [gravity],
        [timestep],
        [mass],
        inertia,
        shapes,
    )
end

primal_dimension(body::QuasistaticObject{T,D}) where {T,D} = 3
cone_dimension(body::QuasistaticObject{T,D}) where {T,D} = 0

function parameter_dimension(body::QuasistaticObject{T,D}) where {T,D}
    @assert D == 2
    nq = 3 # configuration
    nu = 3 # input
    n_gravity = 1 # mass
    n_timestep = 1 # mass
    n_mass = 1 # mass
    n_inertia = 1 # inertia
    nθ = nq + nu + n_gravity + n_timestep + n_mass + n_inertia
    return nθ
end

function unpack_variables(x::Vector, body::QuasistaticObject{T}) where T
    return x
end

function get_parameters(body::QuasistaticObject{T,D}) where {T,D}
    @assert D == 2
    pose = body.pose
    input = body.input

    gravity = body.gravity
    timestep = body.timestep
    mass = body.mass
    inertia = body.inertia
    θ = [pose; input; gravity; timestep; mass; inertia[1]]
    return θ
end

function set_parameters!(body::QuasistaticObject{T,D}, θ) where {T,D}
    pose, input, timestep, gravity, mass, inertia = unpack_parameters(θ, body)
    body.pose .= pose
    body.input .= input

    body.gravity .= gravity
    body.timestep .= timestep
    body.mass .= mass
    body.inertia .= inertia
    return nothing
end

function unpack_parameters(θ::Vector, body::QuasistaticObject{T,D}) where {T,D}
    @assert D == 2
    off = 0
    pose = θ[off .+ (1:D+1)]; off += D+1
    input = θ[off .+ (1:D+1)]; off += D+1

    gravity = θ[off .+ (1:1)]; off += 1
    timestep = θ[off .+ (1:1)]; off += 1
    mass = θ[off .+ (1:1)]; off += 1
    inertia = θ[off .+ 1] * ones(1,1); off += 1
    return pose, input, timestep, gravity, mass, inertia
end
parameter_state_indices(body::QuasistaticObject) = Vector(1:3)
parameter_input_indices(body::QuasistaticObject) = Vector(4:6)

function unpack_pose_timestep(θ::Vector, body::QuasistaticObject{T,D}) where {T,D}
    pose, input, timestep, gravity, mass, inertia = unpack_parameters(θ, body)
    return pose, timestep
end

function residual!(e, x, θ, body::QuasistaticObject)
    index = body.index
    # variables = primals = velocity
    v25 = unpack_variables(x[index.variables], body)
    # parameters
    p2, u, timestep, gravity, mass, inertia = unpack_parameters(θ[index.parameters], body)
    # integrator
    p3 = p2 + timestep[1] * v25

    # mass matrix
    M = Diagonal([mass[1]; mass[1]; inertia[1]])
    # dynamics
    optimality = M * v25 - timestep[1] * [0; mass .* gravity; 0] - u * timestep[1];
    e[index.optimality] .+= optimality
    return nothing
end

function get_current_state(body::QuasistaticObject{T}) where T
    nz = length(body.pose)

    off = 0
    z = zeros(T,nz)
    z[off .+ (1:nz)] .= body.pose; off += nz
    return z
end

function set_current_state!(body::QuasistaticObject, z)
    body.pose .= z
    return nothing
end

function get_next_state!(z, variables, body::QuasistaticObject{T}) where T
    p2 = body.pose
    timestep = body.timestep
    v25 = unpack_variables(variables[body.index.variables], body)

    nx = length(p2)
    z .= p2 + timestep[1] .* v25
    return nothing
end

state_dimension(body::QuasistaticObject) = 3
input_dimension(body::QuasistaticObject) = 3
