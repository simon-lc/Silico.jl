function get_3d_sphere_collision(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    method_type::Symbol=:finite_difference,
    options=Mehrotra.Options(
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )
    )

    # nodes
    shapes = [SphereShape(0.3,zeros(3)), SphereShape(0.3,zeros(3))]
    bodies = [
        Body(timestep, mass, inertia, shapes[1:1], gravity=+gravity, name=:pbody, D=3),
        Body(timestep, mass, inertia, shapes[2:2], gravity=+gravity, name=:cbody, D=3),
        ]
    normal = [0.0, 0.0, 1.0]
    floor_shape = HalfspaceShape(normal)
    contacts = [
        SphereSphere(bodies[1], bodies[2];
            name=:sphere_sphere,
            friction_coefficient=friction_coefficient),
        SphereHalfSpace(bodies[1], floor_shape;
            name=:floor_p,
            friction_coefficient=friction_coefficient),
        SphereHalfSpace(bodies[2], floor_shape;
            name=:floor_c,
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
