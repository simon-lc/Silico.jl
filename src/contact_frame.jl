function contact_normal(contact::PolyHalfSpace{T,D}, mechanism::Mechanism) where {T,D}
    pbody = find_body(mechanism.bodies, contact.parent_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    pp3 = get_next_pose(variables, pbody)

    # unpack variables
    c, γ, ψ, β, λp, sγ, sψ, sβ, sp =
        unpack_variables(variables[contact.index.variables], contact)

    # unpack parameters
    friction_coefficient, Ap, bop, normalc, offc =
        split_parameters(parameters[contact.index.parameters], contact)

    if D == 2
        contact_point = c + pp3[1:2]
        normal = -x_2d_rotation(pp3[3:3]) * Ap' * λp
    else
        contact_point = c + pp3[1:3]
        normal = normalc
    end

    return contact_point, normal
end

function contact_normal(contact::PolyPoly{T,D}, mechanism::Mechanism) where {T,D}
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    pp3 = get_next_pose(variables, pbody)
    pc3 = get_next_pose(variables, cbody)

    # unpack variables
    c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc =
        unpack_variables(variables[contact.index.variables], contact)

    # unpack parameters
    friction_coefficient, Ap, bop, Ac, boc =
        split_parameters(parameters[contact.index.parameters], contact)

    if D == 2
        contact_point = c + (pp3 + pc3)[1:2] ./ 2
        normal = -x_2d_rotation(pp3[3:3]) * Ap' * λp
    else
        # contact_point = nothing
        # normal = nothing
    end

    return contact_point, normal
end

function contact_normal(contact::PolySphere{T,D}, mechanism::Mechanism) where {T,D}
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    pp3 = get_next_pose(variables, pbody)
    pc3 = get_next_pose(variables, cbody)

    # unpack variables
    c, γ, ψ, β, λp, sγ, sψ, sβ, sp =
        unpack_variables(variables[contact.index.variables], contact)

    # unpack parameters
    friction_coefficient, Ap, bop, radc, offc =
        split_parameters(parameters[contact.index.parameters], contact)

    if D == 2
        contact_point = c + pp3[1:2]
        normal = pc3[1:2] + offc - contact_point
    else
        # contact_point = nothing
        # normal = nothing
    end

    return contact_point, normal
end

function contact_normal(contact::SphereHalfSpace{T,D}, mechanism::Mechanism) where {T,D}
    pbody = find_body(mechanism.bodies, contact.parent_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    pp3 = get_next_pose(variables, pbody)

    # unpack parameters
    friction_coefficient, radp, offp, normalc, offc =
        split_parameters(parameters[contact.index.parameters], contact)

    if D == 2
        # analytical contact position in the world frame
        contact_point = pp3[1:2] + offp - radp[1] .* normalc
        # contact normal and tangent in the world frame
        normal = normalc
    else
        contact_point = pp3[1:3] - radp[1] .* normalc # we are missing offpw
        normal = normalc
    end

    return contact_point, normal
end

function contact_normal(contact::SphereSphere{T,D}, mechanism::Mechanism) where {T,D}
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    pp3 = get_next_pose(variables, pbody)
    pc3 = get_next_pose(variables, cbody)

    # unpack parameters
    friction_coefficient, radp, offp, radc, offc =
        unpack_parameters(parameters[contact.index.parameters], contact)

    if D == 2
        # contact normal and tangent in the world frame
        normal = (pp3[1:2] + offp - pc3[1:2] - offc)
        n = normal / (1e-6 + norm(normal))
        # analytical contact position in the world frame
        contact_point = 0.5 * (pp3[1:2] + offp + radp[1] * n + pc3[1:2] + offc - radc[1] * n)
    else
        normal = (xp3 - xc3)
        normal ./= norm(xp3 - xc3) + 1e-10
        contact_point = 0.5 * (xp3 - radp[1] .* normal) + 0.5 * (xc3 + radc[1] .* normal)
    end

    return contact_point, normal
end

function contact_normal(contact::Contact2D, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    pp3 = get_next_pose(variables, pbody)
    pc3 = get_next_pose(variables, cbody)

    c, α, βp, βc, γ, ψ, β, λα, λp, λc, sγ, sψ, sβ, sα, sp, sc =
        unpack_variables(variables[contact.index.variables], contact)

    # contact position in the world frame
    contact_w = c + (pp3 + pc3)[1:2] / 2
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])

    # constraints
    shape_p = contact.parent_shape
    ∇p_gp = constraint_jacobian_p(shape_p, contact_p, α, βp)
    normal = x_2d_rotation(pp3[3:3]) * ∇p_gp' * λp

    return contact_w, normal
end

function contact_normal(contact::EnvContact2D, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    pp3 = get_next_pose(variables, pbody)

    c, α, βp, βc, γ, ψ, β, λα, λp, λc, sγ, sψ, sβ, sα, sp, sc =
        unpack_variables(variables[contact.index.variables], contact)

    # contact position in the world frame
    contact_w = c + pp3[1:2]
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])

    # constraints
    shape_p = contact.parent_shape
    ∇p_gp = constraint_jacobian_p(shape_p, contact_p, α, βp)
    normal = x_2d_rotation(pp3[3:3]) * ∇p_gp' * λp

    return contact_w, normal
end

function contact_normal(contact::BilevelContact2D, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    pp3 = get_next_pose(variables, pbody)
    pc3 = get_next_pose(variables, cbody)

    ϕ, contact_w, normal_pw, normal_cw = contact_data(pp3, pc3, contact.detector)
    normal = normal_pw

    return contact_w, normal
end

function contact_normal(contact::EnvBilevelContact2D, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    pp3 = get_next_pose(variables, pbody)

    ϕ, contact_w, normal_pw, normal_cw = contact_data(pp3, zeros(3), contact.detector)
    normal = normal_pw

    return contact_w, normal
end

function contact_frame_2D(contact, mechanism::Mechanism)
    contact_point, normal = contact_normal(contact, mechanism)

    R = [0 1; -1 0]
    tangent_x = R * normal
    tangent_y = [0, 0]
    return contact_point, normal, tangent_x, tangent_y
end

function contact_frame_3D(contact, mechanism::Mechanism)
    contact_point, normal = contact_normal(contact, mechanism)

    tangent_candidate = [1, 0, 0] # need to use a vector orthogonal to the normal at timestep 2
    tangent_x, tangent_y, wRp = tangential_plane(normal, tangent_candidate=tangent_candidate)
    return contact_point, normal, tangent_x, tangent_y
end

contact_frame(contact::Contact2D, mechanism::Mechanism) = contact_frame_2D(contact, mechanism)
contact_frame(contact::EnvContact2D, mechanism::Mechanism) = contact_frame_2D(contact, mechanism)
contact_frame(contact::BilevelContact2D, mechanism::Mechanism) = contact_frame_2D(contact, mechanism)
contact_frame(contact::EnvBilevelContact2D, mechanism::Mechanism) = contact_frame_2D(contact, mechanism)

contact_frame(contact::PolyHalfSpace{T,D}, m::Mechanism) where {T,D} = (D==2) ? contact_frame_2D(contact, m) : contact_frame_3D(contact, m)
contact_frame(contact::PolyPoly{T,D}, m::Mechanism) where {T,D} = (D==2) ? contact_frame_2D(contact, m) : contact_frame_3D(contact, m)
contact_frame(contact::PolySphere{T,D}, m::Mechanism) where {T,D} = (D==2) ? contact_frame_2D(contact, m) : contact_frame_3D(contact, m)
contact_frame(contact::SphereHalfSpace{T,D}, m::Mechanism) where {T,D} = (D==2) ? contact_frame_2D(contact, m) : contact_frame_3D(contact, m)
contact_frame(contact::SphereSphere{T,D}, m::Mechanism) where {T,D} = (D==2) ? contact_frame_2D(contact, m) : contact_frame_3D(contact, m)
