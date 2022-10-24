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

function Body3D(timestep::T, mass, inertia::Matrix,
        shapes::Vector;
        gravity=-9.81,
        name::Symbol=:body,
        index::NodeIndices=NodeIndices()) where T

    D = 3
    return Body{T,D}(
        name,
        index,
        zeros(7),
        zeros(6),
        zeros(6),
        [gravity],
        [timestep],
        [mass],
        inertia,
        shapes,
    )
end

primal_dimension(body::Body{T,D}) where {T,D} = (D == 2) ? 3 : 6
cone_dimension(body::Body{T,D}) where {T,D} = 0


timestep = 0.1
mass = 1.0
inertia = Matrix(Diagonal([1,1,1.0]))
shapes = [SphereShape(0.2)]
body = Body3D(timestep, mass, inertia, shapes)


function parameter_dimension(body::Body{T,3}) where {T}
    nq = 7 # configuration
    nv = 6 # velocity
    nu = 6 # input
    n_gravity = 1 # mass
    n_timestep = 1 # mass
    n_mass = 1 # mass
    n_inertia = 3 # inertia
    nθ = nq + nv + nu + n_gravity + n_timestep + n_mass + n_inertia
    return nθ
end

function unpack_variables(x::Vector, body::Body{T,3}) where T
    v25 = x[1:3]
    ϕ25 = x[4:6]
    return v25, ϕ25
end

function get_parameters(body::Body{T,3}) where T
    pose = body.pose
    velocity = body.velocity
    input = body.input

    gravity = body.gravity
    timestep = body.timestep
    mass = body.mass
    inertia = body.inertia
    θ = [pose; velocity; input; gravity; timestep; mass; diag(inertia)]
    return θ
end

function set_parameters!(body::Body{T,3}, θ) where T
    pose, velocity, input, timestep, gravity, mass, inertia = unpack_parameters(θ, body)
    body.pose .= pose
    body.velocity .= velocity
    body.input .= input

    body.gravity .= gravity
    body.timestep .= timestep
    body.mass .= mass
    body.inertia[diagind(3,3)] .= inertia
    return nothing
end

function unpack_parameters(θ::Vector, body::Body{T,3}) where T
    off = 0
    pose = θ[off .+ (1:7)]; off += 7
    velocity = θ[off .+ (1:6)]; off += 6
    input = θ[off .+ (1:6)]; off += 6

    gravity = θ[off .+ (1:1)]; off += 1
    timestep = θ[off .+ (1:1)]; off += 1
    mass = θ[off .+ (1:1)]; off += 1
    inertia = Matrix(Diagonal(θ[off .+ (1:3)])); off += 3
    return pose, velocity, input, timestep, gravity, mass, inertia
end
parameter_state_indices(body::Body{T,3}) where T = Vector(1:13)
parameter_input_indices(body::Body{T,3}) where T = Vector(14:19)

function residual!(e, x, θ, body::Body)
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
    linear_optimality = mass[1] * (x3 - 2*x2 + x1)/Δt - Δt * [0; mass .* gravity; 0] - u[1:3] * Δt;
    angular_optimality =
        + sqrt(1 - Δϕ25'*Δϕ25) * inertia * Δϕ25 + cross(Δϕ25, inertia * Δϕ25) +
        - sqrt(1 - Δϕ15'*Δϕ15) * inertia * Δϕ15 + cross(Δϕ15, inertia * Δϕ15) +
        - Δt^2 * u[4:6] / 2

    e[index.optimality] .+= [linear_optimality; angular_optimality]
    return nothing
end

function get_next_state!(z, variables, body::Body{T}) where T
    x2 = body.pose[1:3]
    q2 = body.pose[4:7]
    timestep = body.timestep
    v25, ϕ25 = unpack_variables(variables[body.index.variables], body)

    nv = length(v25) + length(ϕ25)
    off = 0
    z[off .+ (1:3)] .= x2 + timestep[1] .* v25; off += 3
    z[off .+ (1:4)] .= quaternion_increment(q2, timestep[1] .* ϕ25); off += 4

    z[off .+ (1:nv)] .= [v25; ϕ25]; off += nv
    return nothing
end

state_dimension(body::Body{T,3}) where T = 7 + 6
input_dimension(body::Body{T,3}) where T = 6
