################################################################################
# contact
################################################################################

struct CollisionDetection{T,D,NP,NC} <: Node{T}
    name::Symbol
    parent_pose::Vector{T}
    child_pose::Vector{T}
    parent_shape::Shape{T}
    child_shape::Shape{T}
end

function CollisionDetection(parent_shape::Shape{T}, child_shape::Shape{T};
        name::Symbol=:detection,
        ) where {T}

    index = NodeIndices()

    D = 2
    Np = constraint_dimension(parent_shape)
    Nc = constraint_dimension(child_shape)
    return CollisionDetection{T,D,Np,Nc}(
        name,
        zeros(3),
        zeros(3),
        parent_shape,
        child_shape,
        )
end

primal_dimension(detection::CollisionDetection{T,D}) where {T,D} = D + 1 + # c, α
    primal_dimension(detection.parent_shape) +
    primal_dimension(detection.child_shape)

cone_dimension(detection::CollisionDetection) = 1 + # γ, ψ, β, λα
    cone_dimension(detection.parent_shape) +
    cone_dimension(detection.child_shape)

parameter_dimension(detection::CollisionDetection) = 3 + 3 + # parent_pose, child_pose
    parameter_dimension(detection.parent_shape) +
    parameter_dimension(detection.child_shape)

function unpack_variables(x::Vector, detection::CollisionDetection{T,D,NP,NC}) where {T,D,NP,NC}
    nβp = primal_dimension(detection.parent_shape)
    nβc = primal_dimension(detection.child_shape)
    nλp = cone_dimension(detection.parent_shape)
    nλc = cone_dimension(detection.child_shape)

    off = 0
    c = x[off .+ (1:2)]; off += 2
    α = x[off .+ (1:1)]; off += 1
    βp = x[off .+ (1:nβp)]; off += nβp
    βc = x[off .+ (1:nβc)]; off += nβc

    λα = x[off .+ (1:1)]; off += 1
    λp = x[off .+ (1:nλp)]; off += nλp
    λc = x[off .+ (1:nλc)]; off += nλc

    sα = x[off .+ (1:1)]; off += 1
    sp = x[off .+ (1:nλp)]; off += nλp
    sc = x[off .+ (1:nλc)]; off += nλc
    return c, α, βp, βc, λα, λp, λc, sα, sp, sc
end

function get_parameters(detection::CollisionDetection{T,D}) where {T,D}
    θ = [
        detection.parent_pose;
        detection.child_pose;
        get_parameters(detection.parent_shape);
        get_parameters(detection.child_shape);
        ]
    return θ
end

function set_parameters!(detection::CollisionDetection{T,D,NP,NC}, θ) where {T,D,NP,NC}
    parent_pose, child_pose, parent_parameters, child_parameters = unpack_parameters(θ, detection)
    detection.parent_pose .= parent_pose
    detection.child_pose .= child_pose
    set_parameters!(detection.parent_shape, parent_parameters)
    set_parameters!(detection.child_shape, child_parameters)
    return nothing
end

function unpack_parameters(θ::Vector, detection::CollisionDetection{T,D,NP,NC}) where {T,D,NP,NC}
    @assert D == 2
    np = parameter_dimension(detection.parent_shape)
    nc = parameter_dimension(detection.child_shape)

    off = 0
    parent_pose = θ[off .+ (1:3)]; off += 3
    child_pose = θ[off .+ (1:3)]; off += 3
    parent_parameters = θ[off .+ (1:np)]; off += np
    child_parameters = θ[off .+ (1:nc)]; off += nc
    return parent_pose, child_pose, parent_parameters, child_parameters
end

function detection_residual(primals, duals, slacks, parameters, detection::CollisionDetection)
    # unpack parameters
    pp3, pc3, parent_parameters, child_parameters =
        unpack_parameters(parameters, detection)

    # unpack variables
    c, α, βp, βc, λα, λp, λc, sα, sp, sc =
        unpack_variables([primals; duals; slacks], detection)

    # contact position in the world frame
    contact_w = c + (pp3 + pc3)[1:2] / 2
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

    # constraints
    shape_p = detection.parent_shape
    shape_c = detection.child_shape
    gp = constraint(shape_p, contact_p, α, βp) # positive
    gc = constraint(shape_c, contact_c, α, βc) # positive
    ∇α_gp = constraint_jacobian_α(shape_p, contact_p, α, βp)
    ∇α_gc = constraint_jacobian_α(shape_c, contact_c, α, βc)
    ∇p_gp = constraint_jacobian_p(shape_p, contact_p, α, βp)
    ∇p_gc = constraint_jacobian_p(shape_c, contact_c, α, βc)
    ∇β_gp = constraint_jacobian_β(shape_p, contact_p, α, βp)
    ∇β_gc = constraint_jacobian_β(shape_c, contact_c, α, βc)
    ∇o_gp = constraint_jacobian_o(shape_p, contact_p, α, βp)
    ∇o_gc = constraint_jacobian_o(shape_c, contact_c, α, βc)

    # contact normal and tangent in the world frame
    normal_pw = -x_2d_rotation(pp3[3:3]) * ∇o_gp' * λp
    normal_cw = +x_2d_rotation(pc3[3:3]) * ∇o_gc' * λc
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw
    tangent_cw = R * normal_cw

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    wRc = [tangent_cw normal_cw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # contact equality
    optimality = [
        x_2d_rotation(pp3[3:3]) * ∇p_gp' * λp + x_2d_rotation(pc3[3:3]) * ∇p_gc' * λc;
        1 .- ∇α_gp' * λp - ∇α_gc' * λc;
        ∇β_gp' * λp;
        ∇β_gc' * λc;
    ]

    slackness = [
        sα - α;
        sp - gp;
        sc - gc;
    ]

    residual = [optimality; slackness]
    return residual
end

struct CollisionDetector{T,D,NP,NC} <: Node{T}
    detection::CollisionDetection{T,D,NP,NC}
    solver::Mehrotra.Solver{T}
end

function CollisionDetector(parent_shape::Shape{T}, child_shape::Shape{T};
        name::Symbol=:detection,
        complementarity_tolerance=1e-4,
        ) where {T}

    detection = CollisionDetection(parent_shape, child_shape; name=name)

    local_residual(primals, duals, slacks, parameters) =
        detection_residual(primals, duals, slacks, parameters, detection)
    num_primals = primal_dimension(detection)
    num_cone = cone_dimension(detection)
    parameters = get_parameters(detection)

    solver = Mehrotra.Solver(local_residual, num_primals, num_cone,
        parameters=parameters,
        method_type=:symbolic,
        options=Mehrotra.Options(
            max_iterations=20,
            verbose=false,
            complementarity_tolerance=complementarity_tolerance,
            compressed_search_direction=false,
            sparse_solver=false,
            warm_start=false,
            )
        )

    return CollisionDetector(detection, solver)
end

function contact_data(parent_pose, child_pose, detector::CollisionDetector{T,D,NC,NP}) where {T,D,NC,NP}
    detection = detector.detection
    solver = detector.solver
    solver.options.differentiate = true

    detection.parent_pose .= parent_pose
    detection.child_pose .= child_pose
    θ = get_parameters(detection)
    solver.parameters .= θ

    Mehrotra.solve!(solver)
    c, α, βp, βc, λα, λp, λc, sα, sp, sc = unpack_variables(solver.solution.all, detection)

    ϕ = α .- 1
    contact_w = c + (parent_pose + child_pose)[1:2] / 2

    normal_pw = +x_2d_rotation(parent_pose[3:3]) * solver.data.solution_sensitivity[2,1:2]
    normal_cw = -x_2d_rotation(child_pose[3:3]) * solver.data.solution_sensitivity[2,4:5]
    return ϕ, contact_w, normal_pw, normal_cw
end

A0 = [
    [
        +1.0 -0.0;
        -0.1 +1.0;
        -1.0 -0.3;
        +0.0 -1.0;
        ],
    [
        +1.0 +0.3;
        +0.0 +1.0;
        -1.0 +0.2;
        +0.0 -1.0;
        +0.4 -0.8;
        ],
    ]
b0 = [
        0.50*[+1.0, +1.0, +1.0, +1.0],
        0.35*[+1.0, +1.0, +1.0, +1.0, +1.0],
    ]

timestep = 0.05
mass = 1.0
inertia = [1.0;;]
parent_shapes = [PolytopeShape(A0[1], b0[1])]
child_shapes = [PolytopeShape(A0[2], b0[2])]
gravity = -9.81

parent_body = Body(timestep, mass, inertia, parent_shapes,
    gravity=+gravity, name=:parent_body)

child_body = Body(timestep, mass, inertia, child_shapes,
    gravity=+gravity, name=:child_body)

# detection = CollisionDetection(parent_body.shapes[1], child_body.shapes[1])
#
# num_primals = primal_dimension(detection)
# num_cone = cone_dimension(detection)
#
# primals = zeros(num_primals)
# duals = ones(num_cone)
# slacks = ones(num_cone)
# parameters = get_parameters(detection)
# detection_residual(primals, duals, slacks, parameters, detection)
#
# local_residual(primals, duals, slacks, parameters) =
#     detection_residual(primals, duals, slacks, parameters, detection)
#
# solver = Mehrotra.Solver(local_residual, num_primals, num_cone,
#     parameters=parameters,
#     method_type=:symbolic,
#     options=Mehrotra.Options(
#         max_iterations=20,
#         verbose=false,
#         complementarity_tolerance=1e-4,
#         compressed_search_direction=false,
#         sparse_solver=false,
#         warm_start=false,
#         )
#     )
#
# Mehrotra.solve!(solver)

detector = CollisionDetector(parent_body.shapes[1], child_body.shapes[1])

parent_pose = 3rand(3)
child_pose = rand(3)
contact_data(parent_pose, child_pose, detector)
