# for visualization
function contact_frame(contact::PolyPoly, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc =
        unpack_variables(variables[contact.index.variables], contact)
    friction_coefficient, Ap, bp, Ac, bc =
        unpack_parameters(parameters[contact.index.parameters], contact)
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
function contact_frame(contact::PolyHalfSpace, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc =
        unpack_variables(variables[contact.index.variables], contact)
    friction_coefficient, Ap, bp, Ac, bc =
        unpack_parameters(parameters[contact.index.parameters], contact)
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
function contact_frame(contact::SphereHalfSpace, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    # unpack parameters
    friction_coefficient, parent_radius, Ac, bc =
        unpack_parameters(parameters[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)

    # unpack variables
    vp25 = unpack_variables(variables[pbody.index.variables], pbody)
    pp3 = pp2 + timestep_p[1] * vp25

    # analytical contact position in the world frame
    contact_point = pp3[1:2] - parent_radius[1] * Ac[1,:] # assumes the child is fized, other need a rotation here
    # analytical signed distance function
    ϕ = [contact_point' * Ac[1,:]] - bc
    # contact_p is expressed in pbody's frame

    # contact normal and tangent in the world frame
    normal = Ac[1,:]
    R = [0 1; -1 0]
    tangent = R * normal

    return contact_point, normal, tangent
end

# for visualization
function contact_frame(contact::SphereSphere, mechanism::Mechanism)
    pbody = find_body(mechanism.bodies, contact.parent_name)
    cbody = find_body(mechanism.bodies, contact.child_name)

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters

    friction_coefficient, radp, offp, radc, offc =
        unpack_parameters(parameters[contact.index.parameters], contact)
    vp25 = unpack_variables(variables[pbody.index.variables], pbody)
    vc25 = unpack_variables(variables[cbody.index.variables], cbody)
    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(parameters[cbody.index.parameters], cbody)

    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25
    # contact normal and tangent in the world frame
    normal = (pp3 - pc3)[1:2]
    R = [0 1; -1 0]
    tangent = R * normal
    n = normal / (1e-6 + norm(normal))
    # contact position in the world frame
    contact_point = 0.5 * (pp3[1:2] + radp[1] * n + pc3[1:2] - radc[1] * n)

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
    friction_coefficient, Ap, bp, radc, offc =
        unpack_parameters(parameters[contact.index.parameters], contact)
    vp25 = unpack_variables(variables[pbody.index.variables], pbody)
    vc25 = unpack_variables(variables[cbody.index.variables], cbody)
    pp2, timestep_p = unpack_pose_timestep(parameters[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(parameters[cbody.index.parameters], cbody)

    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25
    contact_point = c + pp3[1:2]
    normal = pc3[1:2] - contact_point
    R = [0 1; -1 0]
    tangent = R * normal

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
