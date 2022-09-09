################################################################################
# body
################################################################################
struct Body1170{T,D} <: Body{T}
    name::Symbol
    index::NodeIndices1170
    pose::Vector{T}
    velocity::Vector{T}
    input::Vector{T}
    gravity::Vector{T}
    timestep::Vector{T}
    mass::Vector{T}
    inertia::Matrix{T}
    shapes::Vector
end

function Body1170(timestep::T, mass, inertia::Matrix,
        shapes::Vector;
        gravity=-9.81,
        name::Symbol=:body,
        index::NodeIndices1170=NodeIndices1170()) where T

    D = 2
    return Body1170{T,D}(
        name,
        index,
        zeros(D+1),
        zeros(D+1),
        zeros(D+1),
        [gravity],
        [timestep],
        [mass],
        inertia,
        shapes,
    )
end

primal_dimension(body::Body1170{T,D}) where {T,D} = 3
cone_dimension(body::Body1170{T,D}) where {T,D} = 0

function parameter_dimension(body::Body1170{T,D}) where {T,D}
    @assert D == 2
    nq = 3 # configuration
    nv = 3 # velocity
    nu = 3 # input
    n_gravity = 1 # mass
    n_timestep = 1 # mass
    n_mass = 1 # mass
    n_inertia = 1 # inertia
    nθ = nq + nv + nu + n_gravity + n_timestep + n_mass + n_inertia
    return nθ
end

function unpack_variables(x::Vector, body::Body1170{T}) where T
    return x
end

function get_parameters(body::Body1170{T,D}) where {T,D}
    @assert D == 2
    pose = body.pose
    velocity = body.velocity
    input = body.input

    gravity = body.gravity
    timestep = body.timestep
    mass = body.mass
    inertia = body.inertia
    θ = [pose; velocity; input; gravity; timestep; mass; inertia[1]]
    return θ
end

function set_parameters!(body::Body1170{T,D}, θ) where {T,D}
    pose, velocity, input, timestep, gravity, mass, inertia = unpack_parameters(θ, body)
    body.pose .= pose
    body.velocity .= velocity
    body.input .= input

    body.gravity .= gravity
    body.timestep .= timestep
    body.mass .= mass
    body.inertia .= inertia
    return nothing
end

function unpack_parameters(θ::Vector, body::Body1170{T,D}) where {T,D}
    @assert D == 2
    off = 0
    pose = θ[off .+ (1:D+1)]; off += D+1
    velocity = θ[off .+ (1:D+1)]; off += D+1
    input = θ[off .+ (1:D+1)]; off += D+1

    gravity = θ[off .+ (1:1)]; off += 1
    timestep = θ[off .+ (1:1)]; off += 1
    mass = θ[off .+ (1:1)]; off += 1
    inertia = θ[off .+ 1] * ones(1,1); off += 1
    return pose, velocity, input, timestep, gravity, mass, inertia
end
parameter_state_indices(body::Body1170) = Vector(1:6)
parameter_input_indices(body::Body1170) = Vector(7:9)

function unpack_pose_timestep(θ::Vector, body::Body1170{T,D}) where {T,D}
    pose, velocity, input, timestep, gravity, mass, inertia = unpack_parameters(θ, body)
    return pose, timestep
end

function find_body(bodies::AbstractVector{<:Body1170}, name::Symbol)
    idx = findfirst(x -> x == name, getfield.(bodies, :name))
    return bodies[idx]
end

function residual!(e, x, θ, body::Body1170)
    index = body.index
    # variables = primals = velocity
    v25 = unpack_variables(x[index.variables], body)
    # parameters
    p2, v15, u, timestep, gravity, mass, inertia = unpack_parameters(θ[index.parameters], body)
    # integrator
    p1 = p2 - timestep[1] * v15
    p3 = p2 + timestep[1] * v25

    # mass matrix
    M = Diagonal([mass[1]; mass[1]; inertia[1]])
    # dynamics
    optimality = M * (p3 - 2*p2 + p1)/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - u * timestep[1];
    e[index.optimality] .+= optimality
    return nothing
end

function get_current_state(body::Body1170{T}) where T
    nx = length(body.pose)
    nv = length(body.velocity)

    off = 0
    z = zeros(T,nx+nv)
    z[off .+ (1:nx)] .= body.pose; off += nx
    z[off .+ (1:nv)] .= body.velocity; off += nv
    return z
end

function set_current_state!(body::Body1170, z)
    nx = length(body.pose)
    nv = length(body.velocity)

    off = 0
    body.pose .= z[off .+ (1:nx)]; off += nx
    body.velocity .= z[off .+ (1:nv)]; off += nv
    return nothing
end

function get_next_state!(z, variables, body::Body1170{T}) where T
    p2 = body.pose
    timestep = body.timestep
    v25 = unpack_variables(variables[body.index.variables], body)

    nx = length(p2)
    nv = length(v25)
    off = 0
    z[off .+ (1:nx)] .= p2 + timestep[1] .* v25; off += nx
    z[off .+ (1:nv)] .= v25; off += nv
    return nothing
end

state_dimension(body::Body1170) = 3 + 3
input_dimension(body::Body1170) = 3
