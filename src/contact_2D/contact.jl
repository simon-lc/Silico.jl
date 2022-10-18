################################################################################
# contact
################################################################################
abstract type Contact{T,D,NP,NC} <: Node{T} end

struct Contact2D{T,D,NP,NC} <: Contact{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    child_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T}
    child_shape::Shape{T}
    friction_coefficient::Vector{T}
end

struct EnvContact2D{T,D,NP,NC} <: Contact{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T}
    child_shape::Shape{T}
    friction_coefficient::Vector{T}
end

function Contact2D(parent_body::AbstractBody{T}, child_body::AbstractBody{T};
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
    return Contact2D{T,D,Np,Nc}(
        name,
        parent_name,
        child_name,
        index,
        parent_shape,
        child_shape,
        [friction_coefficient],
        )
end

function EnvContact2D(parent_body::AbstractBody{T}, child_shape::Shape{T};
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
    return EnvContact2D{T,D,Np,Nc}(
        name,
        parent_name,
        index,
        parent_shape,
        child_shape,
        [friction_coefficient],
        )
end

primal_dimension(contact::Contact{T,D}) where {T,D} = D + 1 + # c, α
    primal_dimension(contact.parent_shape) +
    primal_dimension(contact.child_shape)

cone_dimension(contact::Contact) = 1 + 1 + 2 + 1 + # γ, ψ, β, λα
    cone_dimension(contact.parent_shape) +
    cone_dimension(contact.child_shape)

parameter_dimension(contact::Contact) = 1 + # friction_coefficient
    parameter_dimension(contact.parent_shape) +
    parameter_dimension(contact.child_shape)

function unpack_variables(x::Vector, contact::Contact{T,D,NP,NC}) where {T,D,NP,NC}
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

function get_parameters(contact::Contact{T,D}) where {T,D}
    θ = [
        contact.friction_coefficient;
        get_parameters(contact.parent_shape);
        get_parameters(contact.child_shape);
        ]
    return θ
end

function set_parameters!(contact::Contact{T,D,NP,NC}, θ) where {T,D,NP,NC}
    friction_coefficient, parent_parameters, child_parameters = unpack_parameters(θ, contact)
    contact.friction_coefficient .= friction_coefficient
    set_parameters!(contact.parent_shape, parent_parameters)
    set_parameters!(contact.child_shape, child_parameters)
    return nothing
end

function unpack_parameters(θ::Vector, contact::Contact{T,D,NP,NC}) where {T,D,NP,NC}
    @assert D == 2
    np = parameter_dimension(contact.parent_shape)
    nc = parameter_dimension(contact.child_shape)

    off = 0
    friction_coefficient = θ[off .+ (1:1)]; off += 1
    parent_parameters = θ[off .+ (1:np)]; off += np
    child_parameters = θ[off .+ (1:nc)]; off += nc
    return friction_coefficient, parent_parameters, child_parameters
end

function residual!(e, x, θ, contact::Contact2D{T,D,NP,NC},
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
    normal_cw = +x_2d_rotation(pc3[3:3]) * ∇o_gc' * λc
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

function residual!(e, x, θ, contact::EnvContact2D{T,D,NP},
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

function residual!(e, x, θ, contact::Contact, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    cbody = find_body(bodies, contact.child_name)
    residual!(e, x, θ, contact, pbody, cbody)
    return nothing
end

function residual!(e, x, θ, contact::EnvContact2D, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    residual!(e, x, θ, contact, pbody)
    return nothing
end
