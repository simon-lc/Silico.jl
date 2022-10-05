function get_sphere_collision(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
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

    sphere_radius = 0.2
    Af = [0.0  +1.0]
    bf = [0.0]

    # nodes
    shapes = [SphereShape(sphere_radius)]
    bodies = [
        Body(timestep, mass, inertia, shapes, gravity=+gravity, name=:pbody1),
        Body(timestep, mass, inertia, shapes, gravity=+gravity, name=:pbody2),
        ]
    contacts = [
        SphereHalfSpace(bodies[1], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:halfspace_p1),
        SphereHalfSpace(bodies[2], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:halfspace_p2),
        SphereSphere(bodies[1], bodies[2],
            friction_coefficient=friction_coefficient,
            name=:sphere_sphere),
        ]
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
