# for visualization
function contact_frame(contact::PolyHalfSpace, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    c, γ, ψ, β, λp, sγ, sψ, sβ, sp =
        unpack_variables(variables[contact.index.variables], contact)
    # unpack parameters
    friction_coefficient, parent_parameters, child_parameters =
        unpack_parameters(parameters[contact.index.parameters], contact)
    shape_p = contact.parent_shape
    Ap, bp, op = unpack_parameters(shape_p, parent_parameters)

    vp25 = unpack_variables(variables[pbody.index.variables], pbody)
    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)

    pp3 = pp2 + timestep_p[1] * vp25
    contact_point = c + pp3[1:2]
    normal = -x_2d_rotation(pp3[3:3]) * Ap' * λp
    R = [0 1; -1 0]
    tangent = R * normal

    return contact_point, normal, tangent
end

# for visualization
function contact_frame(contact::PolyPoly, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc =
        unpack_variables(variables[contact.index.variables], contact)

    # unpack parameters
    friction_coefficient, parent_parameters, child_parameters =
        unpack_parameters(parameters[contact.index.parameters], contact)
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
    Ap, bp, op = unpack_parameters(shape_p, parent_parameters)
    Ac, bc, oc = unpack_parameters(shape_c, child_parameters)

    vp25 = unpack_variables(variables[pbody.index.variables], pbody)
    vc25 = unpack_variables(variables[cbody.index.variables], cbody)
    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(parameters[cbody.index.parameters], cbody)

    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25
    contact_point = c + (pp3 + pc3)[1:2] ./ 2
    normal = -x_2d_rotation(pp3[3:3]) * Ap' * λp
    R = [0 1; -1 0]
    tangent = R * normal

    return contact_point, normal, tangent
end

# for visualization
function contact_frame(contact::PolySphere, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    c, γ, ψ, β, λp, sγ, sψ, sβ, sp =
        unpack_variables(variables[contact.index.variables], contact)

    # unpack parameters
    friction_coefficient, parent_parameters, child_parameters =
        unpack_parameters(parameters[contact.index.parameters], contact)
    shape_c = contact.child_shape
    radc, offc = unpack_parameters(shape_c, child_parameters)

    vp25 = unpack_variables(variables[pbody.index.variables], pbody)
    vc25 = unpack_variables(variables[cbody.index.variables], cbody)
    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(parameters[cbody.index.parameters], cbody)

    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25
    contact_point = c + pp3[1:2]
    normal = pc3[1:2] + offc - contact_point
    R = [0 1; -1 0]
    tangent = R * normal

    return contact_point, normal, tangent
end

# for visualization
function contact_frame(contact::SphereHalfSpace, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    # unpack parameters
    friction_coefficient, parent_parameters, child_parameters =
        unpack_parameters(parameters[contact.index.parameters], contact)
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
    radp, offp = unpack_parameters(shape_p, parent_parameters)
    normalc, offc = unpack_parameters(shape_c, child_parameters)
    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)

    # unpack variables
    vp25 = unpack_variables(variables[pbody.index.variables], pbody)
    pp3 = pp2 + timestep_p[1] * vp25

    # analytical contact position in the world frame
    contact_point = pp3[1:2] + offp - radp[1] .* normalc

    # contact normal and tangent in the world frame
    normal = normalc
    R = [0 1; -1 0]
    tangent = R * normalc

    return contact_point, normal, tangent
end

# for visualization
function contact_frame(contact::SphereSphere, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    # unpack parameters
    friction_coefficient, parent_parameters, child_parameters =
        unpack_parameters(parameters[contact.index.parameters], contact)
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
    radp, offp = unpack_parameters(shape_p, parent_parameters)
    radc, offc = unpack_parameters(shape_c, child_parameters)

    vp25 = unpack_variables(variables[pbody.index.variables], pbody)
    vc25 = unpack_variables(variables[cbody.index.variables], cbody)
    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(parameters[cbody.index.parameters], cbody)

    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25
    # contact normal and tangent in the world frame
    normal = (pp3[1:2] + offp - pc3[1:2] - offc)
    R = [0 1; -1 0]
    tangent = R * normal
    n = normal / (1e-6 + norm(normal))
    # contact position in the world frame
    contact_point = 0.5 * (pp3[1:2] + offp + radp[1] * n + pc3[1:2] + offc - radc[1] * n)

    return contact_point, normal, tangent
end

# for visualization
function contact_frame(contact::Contact2D, mechanism::Mechanism)
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
function contact_frame(contact::EnvContact2D, mechanism::Mechanism)
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

# for visualization
function contact_frame(contact::BilevelContact2D40, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    γ, ψ, β, sγ, sψ, sβ =
        unpack_variables(variables[contact.index.variables], contact)
    vp25 = unpack_variables(variables[pbody.index.variables], pbody)
    vc25 = unpack_variables(variables[cbody.index.variables], cbody)

    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(parameters[cbody.index.parameters], cbody)

    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25

    ϕ, contact_w, normal_pw, normal_cw = contact_data(pp3, pc3, contact.detector)
    normal = normal_pw
    R = [0 1; -1 0]
    tangent = R * normal

    return contact_w, normal, tangent
end

# for visualization
function contact_frame(contact::EnvBilevelContact2D40, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    γ, ψ, β, sγ, sψ, sβ =
        unpack_variables(variables[contact.index.variables], contact)
    vp25 = unpack_variables(variables[pbody.index.variables], pbody)

    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)

    pp3 = pp2 + timestep_p[1] * vp25

    ϕ, contact_w, normal_pw, normal_cw = contact_data(pp3, zeros(3), contact.detector)
    normal = normal_pw
    R = [0 1; -1 0]
    tangent = R * normal

    return contact_w, normal, tangent
end
