function get_quasistatic_sphere_box(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    num_sphere=1,
    A=[[1 0; 0 1; -1 0; 0 -1]],
    b=[0.4*ones(4)],
    method_type::Symbol=:finite_difference,
    control_mode::Symbol=:robot, # :robot or :object
    options=Mehrotra.Options(
        # verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )
    )

    child_radius = 0.1

    # nodes
    N = length(b)
    body_shapes = [PolytopeShape(A[i], b[i]) for i=1:N]
    manipulator_shapes = [SphereShape(child_radius)]

    box = QuasistaticObject(timestep, mass, inertia, body_shapes, gravity=+gravity, name=:box)
    sphere_1 = (control_mode == :robot) ?
        QuasistaticRobot(timestep, mass, inertia, manipulator_shapes, gravity=0.0*gravity, name=:sphere_1) :
        QuasistaticObject(timestep, mass, inertia, manipulator_shapes, gravity=gravity, name=:sphere_1)
    sphere_2 = (control_mode == :robot) ?
        QuasistaticRobot(timestep, mass, inertia, manipulator_shapes, gravity=0.0*gravity, name=:sphere_2) :
        QuasistaticObject(timestep, mass, inertia, manipulator_shapes, gravity=gravity, name=:sphere_2)
    floor_shape = HalfspaceShape([0.0, 1.0])

    if num_sphere == 1
        bodies = [box, sphere_1]
        sphere_contacts = [PolySphere(bodies[1], bodies[2],
            parent_shape_id=i,
            friction_coefficient=friction_coefficient,
            name=Symbol(:box_sphere_, i)) for i=1:N]
        floor_contacts = [PolyHalfSpace(bodies[1], floor_shape,
            parent_shape_id=i,
            friction_coefficient=friction_coefficient/3,
            name=Symbol(:box_floor_, i)) for i=1:N]
        contacts = [SphereHalfSpace(bodies[2], floor_shape,
                friction_coefficient=friction_coefficient/3,
                name=:halfspace_sphere); sphere_contacts; floor_contacts]
    else
        bodies = [box, sphere_1, sphere_2]
        sphere_1_contacts = [PolySphere(bodies[1], bodies[2],
            parent_shape_id=i,
            friction_coefficient=friction_coefficient,
            name=Symbol(:box_sphere_1_, i)) for i=1:N]
        sphere_2_contacts = [PolySphere(bodies[1], bodies[3],
            parent_shape_id=i,
            friction_coefficient=friction_coefficient,
            name=Symbol(:box_sphere_2_, i)) for i=1:N]
        floor_contacts = [PolyHalfSpace(bodies[1], floor_shape,
            parent_shape_id=i,
            friction_coefficient=friction_coefficient/3,
            name=Symbol(:box_floor_, i)) for i=1:N]
        contacts = [
            SphereHalfSpace(bodies[2], floor_shape,
                friction_coefficient=friction_coefficient/3,
                name=:halfspace_sphere_1);
            SphereHalfSpace(bodies[3], floor_shape,
                friction_coefficient=friction_coefficient/3,
                name=:halfspace_sphere_2);
            sphere_1_contacts;
            sphere_2_contacts;
            floor_contacts;
            ]
    end

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
