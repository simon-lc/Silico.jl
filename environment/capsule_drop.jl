function get_capsule_drop(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    radius = 0.10,
    segment = 1.0,
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

    # nodes
    shapes = [CapsuleShape(radius, segment),
        SphereShape(radius, [+segment/2, 0]),
        SphereShape(radius, [-segment/2, 0]),]
    floor_shape = HalfspaceShape([0.0, 1.0])

    bodies = [Body(timestep, mass, inertia, shapes,
        gravity=+gravity, name=Symbol(:body_, 1))]

    contacts = [
        SphereHalfSpace(bodies[1], floor_shape,
            parent_shape_id=2,
            friction_coefficient=friction_coefficient,
            name=:contact_1),
        SphereHalfSpace(bodies[1], floor_shape,
            parent_shape_id=3,
            friction_coefficient=friction_coefficient,
            name=:contact_2),
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
