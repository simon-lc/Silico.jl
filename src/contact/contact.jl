################################################################################
# contact
################################################################################
abstract type Contact{T,D,NP,NC} <: Node{T} end
space_dimension(contact::Contact{T,D}) where {T,D} = D

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

primal_dimension(contact::Contact{T,D}) where {T,D} = D + 1 + # c, ??
    primal_dimension(contact.parent_shape) +
    primal_dimension(contact.child_shape)

cone_dimension(contact::Contact) = 1 + 1 + 2 + 1 + # ??, ??, ??, ????
    cone_dimension(contact.parent_shape) +
    cone_dimension(contact.child_shape)

parameter_dimension(contact::Contact) = 1 + # friction_coefficient
    parameter_dimension(contact.parent_shape) +
    parameter_dimension(contact.child_shape)

function unpack_variables(x::Vector, contact::Contact{T,D,NP,NC}) where {T,D,NP,NC}
    n??p = primal_dimension(contact.parent_shape)
    n??c = primal_dimension(contact.child_shape)
    n??p = cone_dimension(contact.parent_shape)
    n??c = cone_dimension(contact.child_shape)

    off = 0
    c = x[off .+ (1:2)]; off += 2
    ?? = x[off .+ (1:1)]; off += 1
    ??p = x[off .+ (1:n??p)]; off += n??p
    ??c = x[off .+ (1:n??c)]; off += n??c

    ?? = x[off .+ (1:1)]; off += 1
    ?? = x[off .+ (1:1)]; off += 1
    ?? = x[off .+ (1:2)]; off += 2
    ???? = x[off .+ (1:1)]; off += 1
    ??p = x[off .+ (1:n??p)]; off += n??p
    ??c = x[off .+ (1:n??c)]; off += n??c

    s?? = x[off .+ (1:1)]; off += 1
    s?? = x[off .+ (1:1)]; off += 1
    s?? = x[off .+ (1:2)]; off += 2
    s?? = x[off .+ (1:1)]; off += 1
    sp = x[off .+ (1:n??p)]; off += n??p
    sc = x[off .+ (1:n??c)]; off += n??c
    return c, ??, ??p, ??c, ??, ??, ??, ????, ??p, ??c, s??, s??, s??, s??, sp, sc
end

function get_parameters(contact::Contact{T,D}) where {T,D}
    ?? = [
        contact.friction_coefficient;
        get_parameters(contact.parent_shape);
        get_parameters(contact.child_shape);
        ]
    return ??
end

function set_parameters!(contact::Contact{T,D,NP,NC}, ??) where {T,D,NP,NC}
    friction_coefficient, parent_parameters, child_parameters = unpack_parameters(??, contact)
    contact.friction_coefficient .= friction_coefficient
    set_parameters!(contact.parent_shape, parent_parameters)
    set_parameters!(contact.child_shape, child_parameters)
    return nothing
end

function unpack_parameters(??::Vector, contact::Contact{T,D,NP,NC}) where {T,D,NP,NC}
    np = parameter_dimension(contact.parent_shape)
    nc = parameter_dimension(contact.child_shape)

    off = 0
    friction_coefficient = ??[off .+ (1:1)]; off += 1
    parent_parameters = ??[off .+ (1:np)]; off += np
    child_parameters = ??[off .+ (1:nc)]; off += nc
    return friction_coefficient, parent_parameters, child_parameters
end

function residual!(e, x, ??, contact::Contact2D{T,D,NP,NC},
        pbody::AbstractBody, cbody::AbstractBody) where {T,D,NP,NC}

    # unpack parameters
    friction_coefficient, parent_parameters, child_parameters =
        unpack_parameters(??[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(??[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(??[cbody.index.parameters], cbody)

    # unpack variables
    c, ??, ??p, ??c, ??, ??, ??, ????, ??p, ??c, s??, s??, s??, s??, sp, sc =
        unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    vc25 = unpack_variables(x[cbody.index.variables], cbody)
    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25

    #signed distance function
    ?? = ??[1] - 1.0

    # contact position in the world frame
    contact_w = c + (pp3 + pc3)[1:2] / 2
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

    # constraints
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
    gp = constraint(shape_p, contact_p, ??, ??p) # positive
    gc = constraint(shape_c, contact_c, ??, ??c) # positive
    ?????_gp = constraint_jacobian_??(shape_p, contact_p, ??, ??p)
    ?????_gc = constraint_jacobian_??(shape_c, contact_c, ??, ??c)
    ???p_gp = constraint_jacobian_p(shape_p, contact_p, ??, ??p)
    ???p_gc = constraint_jacobian_p(shape_c, contact_c, ??, ??c)
    ?????_gp = constraint_jacobian_??(shape_p, contact_p, ??, ??p)
    ?????_gc = constraint_jacobian_??(shape_c, contact_c, ??, ??c)
    ???o_gp = constraint_jacobian_o(shape_p, contact_p, ??, ??p)
    ???o_gc = constraint_jacobian_o(shape_c, contact_c, ??, ??c)

    # contact normal and tangent in the world frame
    normal_pw = -x_2d_rotation(pp3[3:3]) * ???o_gp' * ??p
    normal_cw = +x_2d_rotation(pc3[3:3]) * ???o_gc' * ??c
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw
    tangent_cw = R * normal_cw

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    wRc = [tangent_cw normal_cw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # force at the contact point in the contact frame
    f = [??[1] - ??[2]; ??]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    f_cw = -wRc * f # child
    # torques at the centers of masses in world frame
    ??_pw = (skew([contact_w - pp3[1:2]; 0]) * [f_pw; 0])[3:3]
    ??_cw = (skew([contact_w - pc3[1:2]; 0]) * [f_cw; 0])[3:3]
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; ??_pw]
    wrench_c = [f_cw; ??_cw]

    # tangential velocities at the contact point
    tanvel_p = vp25[1:2] + (skew([pp3[1:2] - contact_w; 0]) * [zeros(2); vp25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel_c = vc25[1:2] + (skew([pc3[1:2] - contact_w; 0]) * [zeros(2); vc25[3]])[1:2]
    tanvel_c = tanvel_c' * tangent_cw
    tanvel = tanvel_p - tanvel_c

    # contact equality
    optimality = [
        x_2d_rotation(pp3[3:3]) * ???p_gp' * ??p + x_2d_rotation(pc3[3:3]) * ???p_gc' * ??c;
        1 .- ?????_gp' * ??p - ?????_gc' * ??c;
        ?????_gp' * ??p;
        ?????_gc' * ??c;
    ]

    slackness = [
        s?? - [??];
        s?? - (friction_coefficient[1] * ?? - [sum(??)]);
        s?? - ([+tanvel; -tanvel] + ??[1]*ones(2));
        s?? - ??;
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

function residual!(e, x, ??, contact::EnvContact2D{T,D,NP},
        pbody::AbstractBody) where {T,D,NP}

    # unpack parameters
    friction_coefficient, parent_parameters, child_parameters =
        unpack_parameters(??[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(??[pbody.index.parameters], pbody)

    # unpack variables
    c, ??, ??p, ??c, ??, ??, ??, ????, ??p, ??c, s??, s??, s??, s??, sp, sc =
        unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    pp3 = pp2 + timestep_p[1] * vp25

    #signed distance function
    ?? = ??[1] - 1.0

    # contact position in the world frame
    contact_w = c + pp3[1:2]
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = contact_w

    # constraints
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
    gp = constraint(shape_p, contact_p, ??, ??p) # positive
    gc = constraint(shape_c, contact_c, ??, ??c) # positive
    ?????_gp = constraint_jacobian_??(shape_p, contact_p, ??, ??p)
    ?????_gc = constraint_jacobian_??(shape_c, contact_c, ??, ??c)
    ???p_gp = constraint_jacobian_p(shape_p, contact_p, ??, ??p)
    ???p_gc = constraint_jacobian_p(shape_c, contact_c, ??, ??c)
    ?????_gp = constraint_jacobian_??(shape_p, contact_p, ??, ??p)
    ?????_gc = constraint_jacobian_??(shape_c, contact_c, ??, ??c)
    ???o_gp = constraint_jacobian_o(shape_p, contact_p, ??, ??p)
    ???o_gc = constraint_jacobian_o(shape_c, contact_c, ??, ??c)

    # contact normal and tangent in the world frame
    normal_pw = -x_2d_rotation(pp3[3:3]) * ???o_gp' * ??p
    # normal_pw = -x_2d_rotation(pp3[3:3]) * ???o_gc' * ??c
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # force at the contact point in the contact frame
    f = [??[1] - ??[2]; ??]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    # torques at the centers of masses in world frame
    ??_pw = (skew([contact_w - pp3[1:2]; 0]) * [f_pw; 0])[3:3]
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; ??_pw]

    # tangential velocities at the contact point
    tanvel_p = vp25[1:2] + (skew([pp3[1:2] - contact_w; 0]) * [zeros(2); vp25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel = tanvel_p

    optimality = [
        x_2d_rotation(pp3[3:3]) * ???p_gp' * ??p + ???p_gc' * ??c;
        1 .- ?????_gp' * ??p - ?????_gc' * ??c;
        ?????_gp' * ??p;
        ?????_gc' * ??c;
    ]

    slackness = [
        s?? - [??];
        s?? - (friction_coefficient[1] * ?? - [sum(??)]);
        s?? - ([+tanvel; -tanvel] + ??[1]*ones(2));
        s?? - ??;
        sp - gp;
        sc - gc;
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.optimality] .+= optimality
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    return nothing
end

function residual!(e, x, ??, contact::Contact, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    cbody = find_body(bodies, contact.child_name)
    residual!(e, x, ??, contact, pbody, cbody)
    return nothing
end

function residual!(e, x, ??, contact::EnvContact2D, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    residual!(e, x, ??, contact, pbody)
    return nothing
end
