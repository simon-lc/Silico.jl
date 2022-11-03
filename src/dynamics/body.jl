################################################################################
# body
################################################################################
struct Body{T,D} <: AbstractBody{T,D}
    name::Symbol
    index::NodeIndices
    pose::Vector{T}
    velocity::Vector{T}
    input::Vector{T}
    gravity::Vector{T}
    timestep::Vector{T}
    mass::Vector{T}
    inertia::Matrix{T}
    shapes::Vector
end

function Body(timestep::T, mass, inertia::Matrix,
        shapes::Vector;
        gravity=-9.81,
        name::Symbol=:body,
        index::NodeIndices=NodeIndices(),
        D=2) where T

    pose = zeros(3)
    velocity = zeros(3)
    input = zeros(3)
    if D == 3
        pose = zeros(7)
        velocity = zeros(6)
        input = zeros(6)
    end
    return Body{T,D}(
        name,
        index,
        pose,
        velocity,
        input,
        [gravity],
        [timestep],
        [mass],
        inertia,
        shapes,
    )
end

primal_dimension(body::Body{T,D}) where {T,D} = (D == 2) ? 3 : 6
cone_dimension(body::Body{T,D}) where {T,D} = 0

function parameter_dimension(body::Body{T,D}) where {T,D}
    nq = (D==2) ? 3 : 7 # configuration
    nv = (D==2) ? 3 : 6 # velocity
    nu = (D==2) ? 3 : 6 # input
    n_gravity = 1 # mass
    n_timestep = 1 # mass
    n_mass = 1 # mass
    n_inertia = (D==2) ? 1 : 3 # inertia
    nθ = nq + nv + nu + n_gravity + n_timestep + n_mass + n_inertia
    return nθ
end

unpack_variables(x::Vector, body::Body{T,2}) where T = x
function unpack_variables(x::Vector, body::Body{T,3}) where T
    v25 = x[1:3]
    ϕ25 = x[4:6]
    return v25, ϕ25
end

function get_parameters(body::Body{T,D}) where {T,D}
    pose = body.pose
    velocity = body.velocity
    input = body.input

    gravity = body.gravity
    timestep = body.timestep
    mass = body.mass
    inertia = body.inertia
    inertia_vec = (D==2) ? inertia[1] : diag(inertia)
    θ = [pose; velocity; input; gravity; timestep; mass; inertia_vec]
    return θ
end

function set_parameters!(body::Body{T,D}, θ) where {T,D}
    pose, velocity, input, timestep, gravity, mass, inertia = unpack_parameters(θ, body)
    body.pose .= pose
    body.velocity .= velocity
    body.input .= input

    body.gravity .= gravity
    body.timestep .= timestep
    body.mass .= mass
    if D == 2
        body.inertia .= inertia
    else
        body.inertia[diagind(3,3)] .= diag(inertia)
    end
    return nothing
end

function unpack_parameters(θ::Vector, body::Body{T,2}) where T
    off = 0
    pose = θ[off .+ (1:3)]; off += 3
    velocity = θ[off .+ (1:3)]; off += 3
    input = θ[off .+ (1:3)]; off += 3

    gravity = θ[off .+ (1:1)]; off += 1
    timestep = θ[off .+ (1:1)]; off += 1
    mass = θ[off .+ (1:1)]; off += 1
    inertia = θ[off .+ 1] * ones(1,1); off += 1
    return pose, velocity, input, timestep, gravity, mass, inertia
end

function unpack_parameters(θ::Vector, body::Body{T,3}) where T
    off = 0
    pose = θ[off .+ (1:7)]; off += 7
    velocity = θ[off .+ (1:6)]; off += 6
    input = θ[off .+ (1:6)]; off += 6

    gravity = θ[off .+ (1:1)]; off += 1
    timestep = θ[off .+ (1:1)]; off += 1
    mass = θ[off .+ (1:1)]; off += 1
    inertia = Diagonal(θ[off .+ (1:3)]); off += 3
    return pose, velocity, input, timestep, gravity, mass, inertia
end

parameter_state_indices(body::Body{T,2}) where T = Vector(1:6)
parameter_input_indices(body::Body{T,2}) where T = Vector(7:9)
parameter_state_indices(body::Body{T,3}) where T = Vector(1:13)
parameter_input_indices(body::Body{T,3}) where T = Vector(14:19)

function unpack_pose_timestep(θ::Vector, body::Body{T,D}) where {T,D}
    pose, velocity, input, timestep, gravity, mass, inertia = unpack_parameters(θ, body)
    return pose, timestep
end

function find_body(bodies::AbstractVector{<:Body}, name::Symbol)
    idx = findfirst(x -> x == name, getfield.(bodies, :name))
    return bodies[idx]
end

function residual!(e, x, θ, body::Body{T,2}) where T
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

function residual!(e, x, θ, body::Body{T,3}) where T
    index = body.index
    # variables = primals = velocity
    v25, ϕ25 = unpack_variables(x[index.variables], body)
    # parameters
    p2, vϕ15, u, timestep, gravity, mass, inertia = unpack_parameters(θ[index.parameters], body)
    x2 = p2[1:3]
    q2 = p2[4:7]
    v15 = vϕ15[1:3]
    ϕ15 = vϕ15[4:6]
    # integrator
    Δt = timestep[1]
    x1 = x2 - Δt * v15
    x3 = x2 + Δt * v25
    Δϕ15 = Δt * ϕ15
    Δϕ25 = Δt * ϕ25

    # dynamics
    linear_optimality = mass[1] * (x3 - 2*x2 + x1)/Δt - Δt * [0; 0; mass .* gravity] - u[1:3] * Δt;
    angular_optimality =
        + sqrt(1 - min(0.5,Δϕ25'*Δϕ25)) * inertia * Δϕ25 + cross(Δϕ25, inertia * Δϕ25) +
        - sqrt(1 - Δϕ15'*Δϕ15) * inertia * Δϕ15 + cross(Δϕ15, inertia * Δϕ15) +
        - Δt^2 * u[4:6] / 2

    e[index.optimality] .+= [linear_optimality; angular_optimality]
    return nothing
end

function get_current_state(body::Body{T}) where T
    np = pose_dimension(body)
    nv = velocity_dimension(body)

    off = 0
    z = zeros(T,np+nv)
    z[off .+ (1:np)] .= body.pose; off += np
    z[off .+ (1:nv)] .= body.velocity; off += nv
    return z
end

function set_current_state!(body::Body, z)
    np = pose_dimension(body)
    nv = velocity_dimension(body)

    off = 0
    body.pose .= z[off .+ (1:np)]; off += np
    body.velocity .= z[off .+ (1:nv)]; off += nv
    return nothing
end

function get_next_state!(z, variables, body::Body{T,2}) where T
    p2 = body.pose
    timestep = body.timestep
    v25 = unpack_variables(variables[body.index.variables], body)

    np = pose_dimension(body)
    nv = velocity_dimension(body)
    off = 0
    z[off .+ (1:np)] .= p2 + timestep[1] .* v25; off += np
    z[off .+ (1:nv)] .= v25; off += nv
    return nothing
end

function get_next_state!(z, variables, body::Body{T,3}) where T
    x2 = body.pose[1:3]
    q2 = body.pose[4:7]
    timestep = body.timestep
    v25, ϕ25 = unpack_variables(variables[body.index.variables], body)
    ϕ15 = body.velocity[4:6]
    Δt = timestep[1]
    Δϕ15 = Δt * ϕ15
    Δϕ25 = Δt * ϕ25
    inertia = body.inertia

    nv = velocity_dimension(body)
    off = 0
    z[off .+ (1:3)] .= x2 + timestep[1] .* v25; off += 3
    z[off .+ (1:4)] .= quaternion_increment(q2, timestep[1] .* ϕ25); off += 4
    z[off .+ (1:nv)] .= [v25; ϕ25]; off += nv
    return nothing
end

state_dimension(body::Body) = pose_dimension(body) + velocity_dimension(body)
pose_dimension(body::Body{T,2}) where T = 3
velocity_dimension(body::Body{T,2}) where T = 3
input_dimension(body::Body{T,2}) where T = 3

pose_dimension(body::Body{T,3}) where T = 7
velocity_dimension(body::Body{T,3}) where T = 6
input_dimension(body::Body{T,3}) where T = 6
