################################################################################
# contact
################################################################################
struct PolyPoly{T,D,NP,NC} <: Node{T}
    name::Symbol
    parent_name::Symbol
    child_name::Symbol
    index::NodeIndices
    friction_coefficient::Vector{T}
    A_parent_collider::Matrix{T} #polytope
    b_parent_collider::Vector{T} #polytope
    A_child_collider::Matrix{T} #polytope
    b_child_collider::Vector{T} #polytope
end

function PolyPoly(parent_body::AbstractBody{T}, child_body::AbstractBody{T};
        parent_shape_id::Int=1,
        child_shape_id::Int=1,
        name::Symbol=:contact,
        friction_coefficient=0.2) where {T}

    parent_name = parent_body.name
    child_name = child_body.name
    Ap = copy(parent_body.shapes[parent_shape_id].A)
    bp = copy(parent_body.shapes[parent_shape_id].b)
    Ac = copy(child_body.shapes[child_shape_id].A)
    bc = copy(child_body.shapes[child_shape_id].b)

    return PolyPoly(parent_name, child_name, friction_coefficient, Ap, bp, Ac, bc;
        name=name)
end

function PolyPoly(
        parent_name::Symbol,
        child_name::Symbol,
        friction_coefficient,
        Ap::Matrix{T},
        bp::Vector{T},
        Ac::Matrix{T},
        bc::Vector{T};
        name::Symbol=:contact) where {T}

    d = size(Ap, 2)
    np = size(Ap, 1)
    nc = size(Ac, 1)
    index = NodeIndices()
    return PolyPoly{T,d,np,nc}(
        name,
        parent_name,
        child_name,
        index,
        [friction_coefficient],
        Ap,
        bp,
        Ac,
        bc,
    )
end

primal_dimension(contact::PolyPoly{T,D}) where {T,D} = D + 1 # x, ϕ
cone_dimension(contact::PolyPoly{T,D,NP,NC}) where {T,D,NP,NC} = 1 + 1 + 2 + NP + NC # γ ψ β λp, λc

function parameter_dimension(contact::PolyPoly{T,D}) where {T,D}
    nAp = length(contact.A_parent_collider)
    nbp = length(contact.b_parent_collider)
    nAc = length(contact.A_child_collider)
    nbc = length(contact.b_child_collider)
    nθ = 1 + nAp + nbp + nAc + nbc
    return nθ
end

function unpack_variables(x::Vector, contact::PolyPoly{T,D,NP,NC}) where {T,D,NP,NC}
    num_cone = cone_dimension(contact)
    off = 0
    c = x[off .+ (1:2)]; off += 2
    ϕ = x[off .+ (1:1)]; off += 1

    γ = x[off .+ (1:1)]; off += 1
    ψ = x[off .+ (1:1)]; off += 1
    β = x[off .+ (1:2)]; off += 2
    λp = x[off .+ (1:NP)]; off += NP
    λc = x[off .+ (1:NC)]; off += NC

    sγ = x[off .+ (1:1)]; off += 1
    sψ = x[off .+ (1:1)]; off += 1
    sβ = x[off .+ (1:2)]; off += 2
    sp = x[off .+ (1:NP)]; off += NP
    sc = x[off .+ (1:NC)]; off += NC
    return c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc
end

function get_parameters(contact::PolyPoly{T,D}) where {T,D}
    θ = [
        contact.friction_coefficient;
        vec(contact.A_parent_collider); contact.b_parent_collider;
        vec(contact.A_child_collider); contact.b_child_collider;
        ]
    return θ
end

function set_parameters!(contact::PolyPoly{T,D,NP,NC}, θ) where {T,D,NP,NC}
    friction_coefficient, A_parent_collider, b_parent_collider, A_child_collider, b_child_collider =
        unpack_parameters(θ, contact)
    contact.friction_coefficient .= friction_coefficient
    contact.A_parent_collider .= A_parent_collider
    contact.b_parent_collider .= b_parent_collider
    contact.A_child_collider .= A_child_collider
    contact.b_child_collider .= b_child_collider
    return nothing
end

function unpack_parameters(θ::Vector, contact::PolyPoly{T,D,NP,NC}) where {T,D,NP,NC}
    @assert D == 2
    off = 0
    friction_coefficient = θ[off .+ (1:1)]; off += 1
    A_parent_collider = reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    b_parent_collider = θ[off .+ (1:NP)]; off += NP
    A_child_collider = reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    b_child_collider = θ[off .+ (1:NC)]; off += NC
    return friction_coefficient, A_parent_collider, b_parent_collider, A_child_collider, b_child_collider
end

function residual!(e, x, θ, contact::PolyPoly{T,D,NP,NC},
        pbody::AbstractBody, cbody::AbstractBody) where {T,D,NP,NC}

    # unpack parameters
    friction_coefficient, Ap, bp, Ac, bc = unpack_parameters(θ[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(θ[cbody.index.parameters], cbody)

    # unpack variables
    c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc = unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    vc25 = unpack_variables(x[cbody.index.variables], cbody)
    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25

    # contact position in the world frame
    contact_w = c + (pp3 + pc3)[1:2] / 2
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

    # contact normal and tangent in the world frame
    normal_pw = -x_2d_rotation(pp3[3:3]) * Ap' * λp
    normal_cw = +x_2d_rotation(pc3[3:3]) * Ac' * λc
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
        x_2d_rotation(pp3[3:3]) * Ap' * λp + x_2d_rotation(pc3[3:3]) * Ac' * λc;
        1 - sum(λp) - sum(λc)
    ]
    slackness = [
        sγ - ϕ;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(2));
        sp - (- Ap * contact_p + bp + ϕ .* ones(NP));
        sc - (- Ac * contact_c + bc + ϕ .* ones(NC));
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.optimality] .+= optimality
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    e[cbody.index.optimality] .-= wrench_c
    return nothing
end

function residual!(e, x, θ, contact::PolyPoly, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    cbody = find_body(bodies, contact.child_name)
    residual!(e, x, θ, contact, pbody, cbody)
    return nothing
end


#
#
# options=Mehrotra.Options(
#         verbose=false,
#         complementarity_tolerance=1e-4,
#         # compressed_search_direction=true,
#         max_iterations=30,
#         sparse_solver=false,
#         warm_start=true,
#         )
#
# bodies, contacts = get_polytope_drop(;
#     timestep=0.05,
#     gravity=-9.81,
#     mass=1.0,
#     inertia=0.2 * ones(1,1),
#     friction_coefficient=0.9,
#     options=options,
#     )
#
# nodes = [bodies; contacts]
# num_variables = sum(variable_dimension.(nodes))
# num_primals = sum(primal_dimension.(nodes))
# num_cone = sum(cone_dimension.(nodes))
# num_parameters = sum(parameter_dimension.(nodes))
# num_equality = num_primals + num_cone
#
# dim = Dimensions(num_primals, num_cone, num_parameters);
# idx = Indices(num_primals, num_cone, num_parameters);
#
# contact = contacts[1]
# pbody = find_body(bodies, contact.parent_name)
# cbody = find_body(bodies, contact.child_name)
#
# x = Symbolics.variables(:x, 1:dim.variables);
# θ = Symbolics.variables(:θ, 1:dim.parameters);
# e = Symbolics.variables(:e, 1:dim.equality);
# f = Symbolics.variables(:e, 1:dim.equality);
# f .= e;
#
# # residual!(f, x, θ, pbody);
# residual!(f, x, θ, contacts[1], pbody, cbody);
# # residual!(f, x, θ, contacts[2], pbody);
#
# f;
# # fx = Matrix(Symbolics.sparsejacobian(f, x))
# # fθ = Matrix(Symbolics.sparsejacobian(f, θ))
# fx = Symbolics.sparsejacobian(f, x).nzval
# # fθ = Symbolics.sparsejacobian(f, θ).nzval
# for entry in Symbolics.sparsejacobian(f, x).nzval
#     @show entry
# end
#
#
# a = 1
