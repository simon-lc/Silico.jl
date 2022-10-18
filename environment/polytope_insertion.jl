function get_polytope_insertion(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    method_type::Symbol=:finite_difference,
    A = [
        +1.0 +0.2;
        +0.1 +1.0;
        -1.0 +0.3;
        +0.0 -1.0;
        ],
    b = 0.5*[
        +1,
        +1,
        +1,
        +1,
        ],
    options=Mehrotra.Options(
        # verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )
    )

    DojoLight.normalize_A!(A)

    A1 = [
        +1.0 +0.0;
        +0.0 +1.0;
        -1.0 +0.0;
        +0.0 -1.0;
        ]
    b1 = 0.25*[
        +1,
        +2,
        +1,
        +2,
        ]
    o_left = [-0.5, 0.5]
    o_right = [+0.5, 0.5]
    # nodes
    shapes = [PolytopeShape(A, b)]
    bodies = [
        Body(timestep, mass, inertia, shapes, gravity=+gravity, name=:pbody),
        ]
    shape_left = PolytopeShape(A1, b1, o_left)
    shape_right = PolytopeShape(A1, b1, o_right)
    normal = [0.0, 1.0]
    position_offset = [0.0, 0.0]
    floor_shape = HalfspaceShape(normal, position_offset)
    contacts = [
        PolyHalfSpace(bodies[1], floor_shape,
            friction_coefficient=friction_coefficient,
            name=:floor),
        EnvContact2D(bodies[1], shape_left;
                name=:env_left,
                friction_coefficient=friction_coefficient),
        EnvContact2D(bodies[1], shape_right;
                name=:env_right,
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
