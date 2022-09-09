# for visualization
function contact_frame(contact::PolyPoly1170, mechanism::Mechanism1170)
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
function contact_frame(contact::PolyHalfSpace1170, mechanism::Mechanism1170)
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
function contact_frame(contact::SphereHalfSpace1170, mechanism::Mechanism1170)
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
function contact_frame(contact::SphereSphere1170, mechanism::Mechanism1170)
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
function contact_frame(contact::PolySphere1170, mechanism::Mechanism1170)
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
