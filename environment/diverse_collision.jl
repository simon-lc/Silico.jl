function get_diverse_collision(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    method_type::Symbol=:finite_difference,
    minkowski_radius=0.1,
    union_radius=0.1,
    segment=1.0,
    A = [[1 0; -1 0; 0 1; 0 -1.0], [1 0; 1 1; -1 1.0]],
    b = [0.5 * ones(4), 0.5 * ones(3)],
    options=Mehrotra.Options(
        # verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )
    )

    normalize_A!.(A)

    # nodes

    Acap = [1 0; -1 0; 0 1; 0 -1]
    bcap = [segment/2, segment/2, 1e-3, 1e-3]
    union_shapes = [
        PolytopeShape(A[1], b[1]),
        PaddedPolytopeShape110(union_radius, Acap, bcap),
        SphereShape(union_radius, [+segment/2, 0]),
        SphereShape(union_radius, [-segment/2, 0]),
        ]
    minkowski_shapes = [PaddedPolytopeShape110(minkowski_radius, A[2], b[2])]
    bodies = [
        Body(timestep, mass, inertia, union_shapes, gravity=+gravity, name=:union),
        Body(timestep, mass, inertia, minkowski_shapes, gravity=+gravity, name=:minkowski),
        ]
    normal = [0.0, 1.0]
    position_offset = [0.0, 0.0]
    floor_shape = HalfspaceShape(normal, position_offset)
    contacts = [
        Contact2D(bodies[1], bodies[2];
            name=:contact_1,
            parent_shape_id=1,
            friction_coefficient=friction_coefficient),
        Contact2D(bodies[1], bodies[2];
            name=:contact_2,
            parent_shape_id=2,
            friction_coefficient=friction_coefficient),
        EnvContact2D(bodies[1], floor_shape;
            name=:floor_1_1,
            parent_shape_id=1,
            friction_coefficient=friction_coefficient),
        SphereHalfSpace(bodies[1], floor_shape;
            name=:floor_1_3,
            parent_shape_id=3,
            friction_coefficient=friction_coefficient),
        SphereHalfSpace(bodies[1], floor_shape;
            name=:floor_1_4,
            parent_shape_id=4,
            friction_coefficient=friction_coefficient),
        EnvContact2D(bodies[2], floor_shape;
            name=:floor_2,
            friction_coefficient=friction_coefficient),
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
