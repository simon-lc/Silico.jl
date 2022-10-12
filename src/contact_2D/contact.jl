################################################################################
# contact
################################################################################
abstract type Contact2D1{T,D,NP,NC} <: Node{T} end

struct Contact2D1150{T,D,NP,NC} <: Contact2D1{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    child_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T}
    child_shape::Shape{T}
    friction_coefficient::Vector{T}
end

struct EnvContact2D1150{T,D,NP,NC} <: Contact2D1{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T}
    child_shape::Shape{T}
    friction_coefficient::Vector{T}
end

function Contact2D1150(parent_body::AbstractBody{T}, child_body::AbstractBody{T};
        parent_shape_id::Int=1,
        child_shape_id::Int=1,
        name::Symbol=:contact,
        friction_coefficient=0.2) where {T}

    parent_name = parent_body.name
    child_name = child_body.name
    parent_shape = deepcopy(parent_body.shapes[parent_shape_id])
    child_shape = deepcopy(child_body.shapes[child_shape_id])

    index = NodeIndices()

    D = 2
    Np = constraint_dimension(parent_shape)
    Nc = constraint_dimension(child_shape)
    return Contact2D1150{T,D,Np,Nc}(
        name,
        parent_name,
        child_name,
        index,
        parent_shape,
        child_shape,
        [friction_coefficient],
        )
end

function EnvContact2D1150(parent_body::AbstractBody{T}, child_shape::Shape{T};
        parent_shape_id::Int=1,
        name::Symbol=:contact,
        friction_coefficient=0.2) where {T}

    parent_name = parent_body.name
    parent_shape = deepcopy(parent_body.shapes[parent_shape_id])
    child_shape = deepcopy(child_shape)

    index = NodeIndices()

    D = 2
    Np = constraint_dimension(parent_shape)
    Nc = constraint_dimension(child_shape)
    return EnvContact2D1150{T,D,Np,Nc}(
        name,
        parent_name,
        index,
        parent_shape,
        child_shape,
        [friction_coefficient],
        )
end

primal_dimension(contact::Contact2D1{T,D}) where {T,D} = D + 1 + # c, α
    primal_dimension(contact.parent_shape) +
    primal_dimension(contact.child_shape)

cone_dimension(contact::Contact2D1) = 1 + 1 + 2 + 1 + # γ, ψ, β, λα
    cone_dimension(contact.parent_shape) +
    cone_dimension(contact.child_shape)

parameter_dimension(contact::Contact2D1) = 1 +  # friction_coefficient
    parameter_dimension(contact.parent_shape) +
    parameter_dimension(contact.child_shape)

function unpack_variables(x::Vector, contact::Contact2D1{T,D,NP,NC}) where {T,D,NP,NC}
    nβp = primal_dimension(contact.parent_shape)
    nβc = primal_dimension(contact.child_shape)
    nλp = cone_dimension(contact.parent_shape)
    nλc = cone_dimension(contact.child_shape)

    off = 0
    c = x[off .+ (1:2)]; off += 2
    α = x[off .+ (1:1)]; off += 1
    βp = x[off .+ (1:nβp)]; off += nβp
    βc = x[off .+ (1:nβc)]; off += nβc

    γ = x[off .+ (1:1)]; off += 1
    ψ = x[off .+ (1:1)]; off += 1
    β = x[off .+ (1:2)]; off += 2
    λα = x[off .+ (1:1)]; off += 1
    λp = x[off .+ (1:nλp)]; off += nλp
    λc = x[off .+ (1:nλc)]; off += nλc

    sγ = x[off .+ (1:1)]; off += 1
    sψ = x[off .+ (1:1)]; off += 1
    sβ = x[off .+ (1:2)]; off += 2
    sα = x[off .+ (1:1)]; off += 1
    sp = x[off .+ (1:nλp)]; off += nλp
    sc = x[off .+ (1:nλc)]; off += nλc
    return c, α, βp, βc, γ, ψ, β, λα, λp, λc, sγ, sψ, sβ, sα, sp, sc
end

function get_parameters(contact::Contact2D1{T,D}) where {T,D}
    θ = [
        contact.friction_coefficient;
        get_parameters(contact.parent_shape);
        get_parameters(contact.child_shape);
        ]
    return θ
end

function set_parameters!(contact::Contact2D1{T,D,NP,NC}, θ) where {T,D,NP,NC}
    friction_coefficient, parent_parameters, child_parameters = unpack_parameters(θ, contact)
    contact.friction_coefficient .= friction_coefficient
    set_parameters!(contact.parent_shape, parent_parameters)
    set_parameters!(contact.child_shape, child_parameters)
    return nothing
end

function unpack_parameters(θ::Vector, contact::Contact2D1{T,D,NP,NC}) where {T,D,NP,NC}
    @assert D == 2
    np = parameter_dimension(contact.parent_shape)
    nc = parameter_dimension(contact.child_shape)

    off = 0
    friction_coefficient = θ[off .+ (1:1)]; off += 1
    parent_parameters = θ[off .+ (1:np)]; off += np
    child_parameters = θ[off .+ (1:nc)]; off += nc
    return friction_coefficient, parent_parameters, child_parameters
end

function residual!(e, x, θ, contact::Contact2D1150{T,D,NP,NC},
        pbody::AbstractBody, cbody::AbstractBody) where {T,D,NP,NC}

    # unpack parameters
    friction_coefficient, parent_parameters, child_parameters =
        unpack_parameters(θ[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(θ[cbody.index.parameters], cbody)

    # unpack variables
    c, α, βp, βc, γ, ψ, β, λα, λp, λc, sγ, sψ, sβ, sα, sp, sc =
        unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    vc25 = unpack_variables(x[cbody.index.variables], cbody)
    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25

    #signed distance function
    ϕ = α[1] - 1.0

    # contact position in the world frame
    contact_w = c + (pp3 + pc3)[1:2] / 2
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

    # constraints
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
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
    normal_cw = +x_2d_rotation(pc3[3:3]) * ∇o_gp' * λc
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw
    tangent_cw = R * normal_cw

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    wRc = [tangent_cw normal_cw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # force at the contact point in the contact frame
    f = [β[1] - β[2]; γ]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    f_cw = -wRc * f # child
    # torques at the centers of masses in world frame
    τ_pw = (skew([contact_w - pp3[1:2]; 0]) * [f_pw; 0])[3:3]
    τ_cw = (skew([contact_w - pc3[1:2]; 0]) * [f_cw; 0])[3:3]
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; τ_pw]
    wrench_c = [f_cw; τ_cw]

    # tangential velocities at the contact point
    tanvel_p = vp25[1:2] + (skew([pp3[1:2] - contact_w; 0]) * [zeros(2); vp25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel_c = vc25[1:2] + (skew([pc3[1:2] - contact_w; 0]) * [zeros(2); vc25[3]])[1:2]
    tanvel_c = tanvel_c' * tangent_cw
    tanvel = tanvel_p - tanvel_c

    # contact equality
    optimality = [
        x_2d_rotation(pp3[3:3]) * ∇p_gp' * λp + x_2d_rotation(pc3[3:3]) * ∇p_gc' * λc;
        1 .- ∇α_gp' * λp - ∇α_gc' * λc;
        ∇β_gp' * λp;
        ∇β_gc' * λc;
    ]

    slackness = [
        sγ - [ϕ];
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(2));
        sα - α;
        sp - gp;
        sc - gc;
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.optimality] .+= optimality
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    e[cbody.index.optimality] .-= wrench_c
    return nothing
end

function residual!(e, x, θ, contact::EnvContact2D1150{T,D,NP},
        pbody::AbstractBody) where {T,D,NP}

    # unpack parameters
    friction_coefficient, parent_parameters, child_parameters =
        unpack_parameters(θ[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)

    # unpack variables
    c, α, βp, βc, γ, ψ, β, λα, λp, λc, sγ, sψ, sβ, sα, sp, sc =
        unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    pp3 = pp2 + timestep_p[1] * vp25

    #signed distance function
    ϕ = α[1] - 1.0

    # contact position in the world frame
    contact_w = c + pp3[1:2]
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = contact_w

    # constraints
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
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
    normal_pw ./= norm(normal_pw) + 1e-4
    # normal_pw = -x_2d_rotation(pp3[3:3]) * ∇o_gc' * λc
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # force at the contact point in the contact frame
    f = [β[1] - β[2]; γ]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    # torques at the centers of masses in world frame
    τ_pw = (skew([contact_w - pp3[1:2]; 0]) * [f_pw; 0])[3:3]
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; τ_pw]

    # tangential velocities at the contact point
    tanvel_p = vp25[1:2] + (skew([pp3[1:2] - contact_w; 0]) * [zeros(2); vp25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel = tanvel_p

    optimality = [
        x_2d_rotation(pp3[3:3]) * ∇p_gp' * λp + ∇p_gc' * λc;
        1 .- ∇α_gp' * λp - ∇α_gc' * λc;
        ∇β_gp' * λp;
        ∇β_gc' * λc;
    ]

    slackness = [
        sγ - [ϕ];
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(2));
        sα - α;
        sp - gp;
        sc - gc;
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.optimality] .+= optimality
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    return nothing
end

function residual!(e, x, θ, contact::Contact2D1150, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    cbody = find_body(bodies, contact.child_name)
    residual!(e, x, θ, contact, pbody, cbody)
    return nothing
end

function residual!(e, x, θ, contact::EnvContact2D1150, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    # cbody = find_body(bodies, contact.child_name)
    residual!(e, x, θ, contact, pbody)
    return nothing
end

# for visualization
function contact_frame(contact::Contact2D1150, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    c, α, βp, βc, γ, ψ, β, λα, λp, λc, sγ, sψ, sβ, sα, sp, sc =
        unpack_variables(variables[contact.index.variables], contact)
    vp25 = unpack_variables(variables[pbody.index.variables], pbody)
    vc25 = unpack_variables(variables[cbody.index.variables], cbody)

    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(parameters[cbody.index.parameters], cbody)

    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25
    # contact position in the world frame
    contact_w = c + (pp3 + pc3)[1:2] / 2
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])

    # constraints
    shape_p = contact.parent_shape
    ∇p_gp = constraint_jacobian_p(shape_p, contact_p, α, βp)
    normal = x_2d_rotation(pp3[3:3]) * ∇p_gp' * λp
    R = [0 1; -1 0]
    tangent = R * normal

    return contact_w, normal, tangent
end

# for visualization
function contact_frame(contact::EnvContact2D1150, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    c, α, βp, βc, γ, ψ, β, λα, λp, λc, sγ, sψ, sβ, sα, sp, sc =
        unpack_variables(variables[contact.index.variables], contact)
    vp25 = unpack_variables(variables[pbody.index.variables], pbody)

    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)

    pp3 = pp2 + timestep_p[1] * vp25
    # contact position in the world frame
    contact_w = c + pp3[1:2]
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])

    # constraints
    shape_p = contact.parent_shape
    ∇p_gp = constraint_jacobian_p(shape_p, contact_p, α, βp)
    normal = x_2d_rotation(pp3[3:3]) * ∇p_gp' * λp
    R = [0 1; -1 0]
    tangent = R * normal

    return contact_w, normal, tangent
end

# timestep = 0.01
# mass = 1.0
# inertia = [1.0;;]
# Ap = [
#     1.0  0.0;
#     0.0  1.0;
#     -1.0  0.0;
#     0.0 -1.0;
#     ]
# bp = 0.2*[
#     +1,
#     +1,
#     +1,
#     1,
#     ];
# Ac = [
#      1.0  0.0;
#      0.0  1.0;
#     -1.0  0.0;
#      0.0 -1.0;
#     ]
# bc = 0.2*[
#     1,
#     1,
#     1,
#     1,
#     ];
#
# parent_shape = PolytopeShape(Ap, bp)
# child_shape = PolytopeShape(Ac, bc)
# parent_shapes = [parent_shape]
# child_shapes = [child_shape]
# parent_body = Body(timestep, mass, inertia, parent_shapes)
# child_body = Body(timestep, mass, inertia, child_shapes)
# contact = Contact2D1150(parent_body, child_body)
#
# num_primals = primal_dimension(contact)
# num_cone = cone_dimension(contact)
# num_params = parameter_dimension(contact)
#
# x = rand(num_primals + 2 * num_cone)
# unpack_variables(x, contact)
#
# get_parameters(contact)
#
# θ = rand(num_params)
# set_parameters!(contact, θ)
#
# unpack_parameters(θ, contact)
#
# nodes = [parent_body, child_body, contact]
# bodies = [parent_body, child_body]
# indexing!(nodes)
# num_primals = sum(primal_dimension.(nodes))
# num_cone = sum(cone_dimension.(nodes))
# num_equality = num_primals + num_cone
#
# primals = zeros(num_primals)
# duals = 0.1*ones(num_cone)
# slacks = 0.1*ones(num_cone)
#
# x = [primals; duals; slacks]
# e = zeros(eltype(x), num_equality)
# θ = vcat(get_parameters.(nodes)...)
#
# # body
# residual!(e, x, θ, parent_body)
# residual!(e, x, θ, child_body)
#
# # contact
# residual!(e, x, θ, contact, bodies)



function get_polytope_collision(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    A=[
    [
        +1.0 -0.0;
        +0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 +0.0;
        +0.0 +1.0;
        -1.0 +0.0;
        +0.0 -1.0;
        ],
    ],
    b=[
        0.5*[+1.0, +1.0, +1.0, +1.0],
        0.5*[+1.0, +1.0, +1.0, +1.0],
    ],
    method_type::Symbol=:finite_difference,
    options=Mehrotra.Options(
        # verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )
    )

    N = length(A)
    # nodes
    shapes = [PolytopeShape(A[i], b[i]) for i=1:N]
    floor_shape = HalfspaceShape([0.0, 1.0])

    bodies = [Body(timestep, mass, inertia, shapes[i:i],
        gravity=+gravity, name=Symbol(:body_, i)) for i=1:N]

    body_contacts = vcat([
        [Contact2D1150(bodies[i], bodies[j],
            friction_coefficient=friction_coefficient,
            name=Symbol(:contact_, i, :_, j)) for i=1:j-1]
        for j=1:N]...)
    env_contacts = [
        EnvContact2D1150(bodies[i], floor_shape,
            friction_coefficient=friction_coefficient,
            name=Symbol(:env_contact_, i)) for i=1:N]

    contacts = [body_contacts; env_contacts]
    indexing!([bodies; contacts])

    local_mechanism_residual(primals, duals, slacks, parameters) =
        mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)

    mechanism = Mechanism(
        local_mechanism_residual,
        bodies,
        contacts,
        options=options,
        method_type=method_type)

    Mehrotra.initialize_solver!(mechanism.solver)
    return mechanism
end

mech = get_polytope_collision()




# vis = Visualizer()
# open(vis)


################################################################################
# demo
################################################################################
A0 = [
    [
        +1.0 -0.0;
        +0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 +0.0;
        +0.0 +1.0;
        -1.0 +0.0;
        +0.0 -1.0;
        ],
    ]
b0 = [
        0.5*[+1.0, +1.0, +1.0, +1.0],
        0.5*[+1.0, +1.0, +1.0, +1.0],
    ]

mech = get_polytope_collision(;
    timestep=0.05,
    gravity=1*-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=2.0,
    method_type=:symbolic,
    # method_type=:finite_difference,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-4,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        )
    )
# Mehrotra.solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2 = [+2.0, +1.0, -2.0]
xc2 = [-0.0, +1.0, -3.0]
vp15 = [-1.0, +0.0, -0.0]
vc15 = [+0.0, +0.0, +10.0]
z0 = [xp2; vp15; xc2; vc15]

u0 = zeros(6)
H0 = 100

@elapsed storage = simulate!(mech, z0, H0)

################################################################################
# visualization
################################################################################
set_floor!(vis)
set_light!(vis)
set_background!(vis)
visualize!(vis, mech, storage, build=true)

using Plots
storage.x
plot(hcat([storage.x[i][1] for i=1:H0]...)')
plot(hcat([storage.x[i][2] for i=1:H0]...)')
scatter(storage.iterations)
# plot!(hcat(storage.variables...)')
# RobotVisualizer.convert_frames_to_video_and_gif("sphere_polytope_drop")
