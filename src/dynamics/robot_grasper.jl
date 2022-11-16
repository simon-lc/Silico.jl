################################################################################
# body
################################################################################
struct RobotGrasper{T,D} <: AbstractBody{T,D}
    name::Symbol
    index::NodeIndices
    pose::Vector{T}
    input::Vector{T}
    gravity::Vector{T}
    timestep::Vector{T}
    mass::Vector{T}
    inertia::Matrix{T}
    stiffness::Vector{T}
    shapes::Vector
end

function RobotGrasper(timestep::T, mass, inertia::Matrix,
        shapes::Vector;
        gravity=-9.81,
        stiffness=1e2*ones(6),
        name::Symbol=:body,
        index::NodeIndices=NodeIndices()) where T

    D = 2
    return RobotGrasper{T,D}(
        name,
        index,
        zeros(6),
        zeros(6),
        [gravity],
        [timestep],
        [mass],
        inertia,
        stiffness,
        shapes,
    )
end

primal_dimension(body::RobotGrasper{T,D}) where {T,D} = 6
cone_dimension(body::RobotGrasper{T,D}) where {T,D} = 0

function parameter_dimension(body::RobotGrasper{T,D}) where {T,D}
    @assert D == 2
    nq = 6 # configuration
    nu = 6 # input
    n_gravity = 1 # mass
    n_timestep = 1 # mass
    n_mass = 1 # mass
    n_inertia = 1 # inertia
    n_stiffness = 6
    nθ = nq + nu + n_gravity + n_timestep + n_mass + n_inertia + n_stiffness
    return nθ
end

function unpack_variables(x::Vector, body::RobotGrasper{T}) where T
    return x
end

function get_parameters(body::RobotGrasper{T,D}) where {T,D}
    @assert D == 2
    pose = body.pose
    input = body.input

    gravity = body.gravity
    timestep = body.timestep
    mass = body.mass
    inertia = body.inertia
    stiffness = body.stiffness
    θ = [pose; input; gravity; timestep; mass; inertia[1]; stiffness]
    return θ
end

function set_parameters!(body::RobotGrasper{T,D}, θ) where {T,D}
    pose, input, timestep, gravity, mass, inertia, stiffness = unpack_parameters(θ, body)
    body.pose .= pose
    body.input .= input

    body.gravity .= gravity
    body.timestep .= timestep
    body.mass .= mass
    body.inertia .= inertia
    body.stiffness .= stiffness
    return nothing
end

function unpack_parameters(θ::Vector, body::RobotGrasper{T,D}) where {T,D}
    @assert D == 2
    off = 0
    pose = θ[off .+ (1:6)]; off += 6
    input = θ[off .+ (1:6)]; off += 6

    gravity = θ[off .+ (1:1)]; off += 1
    timestep = θ[off .+ (1:1)]; off += 1
    mass = θ[off .+ (1:1)]; off += 1
    inertia = θ[off .+ 1] * ones(1,1); off += 1
    stiffness = θ[off .+ (1:6)]; off += 6
    return pose, input, timestep, gravity, mass, inertia, stiffness
end
parameter_state_indices(body::RobotGrasper) = Vector(1:6)
parameter_input_indices(body::RobotGrasper) = Vector(7:12)

function unpack_pose_timestep(θ::Vector, body::RobotGrasper{T,D}) where {T,D}
    pose, input, timestep, gravity, mass, inertia, stiffness = unpack_parameters(θ, body)
    return pose, timestep
end

function residual!(e, x, θ, body::RobotGrasper)
    index = body.index
    # variables = primals = velocity
    v25 = unpack_variables(x[index.variables], body)
    # parameters
    p2, u, timestep, gravity, mass, inertia, stiffness = unpack_parameters(θ[index.parameters], body)
    # integrator
    p3 = p2 + timestep[1] * v25

    # mass matrix
    K = Diagonal(stiffness)
    # dynamics
    optimality = timestep[1] * K * (p3 - u) - timestep[1] * [0; mass .* gravity; 0; 0; 0; 0];
    e[index.optimality] .+= optimality
    return nothing
end

function get_current_state(body::RobotGrasper{T}) where T
    nz = length(body.pose)

    off = 0
    z = zeros(T,nz)
    z[off .+ (1:nz)] .= body.pose; off += nz
    return z
end

function set_current_state!(body::RobotGrasper, z)
    body.pose .= z
    return nothing
end

function get_next_state!(z, variables, body::RobotGrasper{T}) where T
    p2 = body.pose
    timestep = body.timestep
    v25 = unpack_variables(variables[body.index.variables], body)

    nx = length(p2)
    z .= p2 + timestep[1] .* v25
    return nothing
end

state_dimension(body::RobotGrasper) = pose_dimension(body) + velocity_dimension(body)
pose_dimension(body::RobotGrasper) = 6
velocity_dimension(body::RobotGrasper) = 0
input_dimension(body::RobotGrasper) = 6

function set_body!(vis::Visualizer, body::RobotGrasper{T,D}, pose; name=body.name) where {T,D}
    segment = body.shapes[1].segment[1]
    for i = 1:4
        xi = capsule_pose(pose, segment; capsule_id=i)
        settransform!(vis[:bodies][name][Symbol(i)], MeshCat.compose(
            MeshCat.Translation(SVector{3}(0, xi[1:2]...)),
            MeshCat.LinearMap(rotationmatrix(RotX(xi[3]))),
            )
        )
    end
    return nothing
end

function capsule_pose(pose, segment; capsule_id::Int=1)
    x = pose[1:2]
    θ = pose[3]
    α1 = pose[4] # ∈ [+0.20π, +0.40π]
    α2 = pose[5] # ∈ [+0.00π, +0.45π]
    α3 = pose[6] # ∈ [+0.00π, +0.45π]
    p = zeros(3)
    if capsule_id == 1
        β1 = θ - α1
        x1 = x + segment/2 * [cos(β1), sin(β1)]
        p = [x1; β1]
    elseif capsule_id == 2
        β1 = θ - α1
        β2 = θ - α1 + α2
        x2 = x + segment * [cos(β1), sin(β1)] + segment/2 * [cos(β2), sin(β2)]
        p = [x2; β2]
    elseif capsule_id == 3
        β3 = θ + α1
        x3 = x + segment/2 * [cos(β3), sin(β3)]
        p = [x3; β3]
    elseif capsule_id == 4
        β3 = θ + α1
        β4 = θ + α1 - α3
        x4 = x + segment * [cos(β3), sin(β3)] + segment/2 * [cos(β4), sin(β4)]
        p = [x4; β4]
    end
    return p
end
