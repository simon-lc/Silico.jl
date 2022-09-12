function visualize!(vis::Visualizer, context::CvxContext1320, measurements::Vector{<:Measurement};
        animation::MeshCat.Animation=MeshCat.Animation(Int(floor(1/context.mechanism.bodies[1].timestep[1]))),
        name::Symbol=:context)

    mechanism = context.mechanism
    cameras = context.cameras

    # build point cloud
    num_points = size.(measurements[1].d, 2)
    build_point_cloud!(vis[name], num_points)
    eye_positions = [c.eye_position for c in cameras]

    # animate point cloud
    for (i, measurement) in enumerate(measurements)
        MeshCat.atframe(animation, i) do
            set_2d_point_cloud!(vis[name], eye_positions, measurement.d)
        end
    end
    # set animation
    MeshCat.setanimation!(vis, animation)

    # add robot visualization
    _, animation = visualize!(vis[name], mechanism, [m.z for m in measurements], animation=animation)
    return vis, animation
end

function visualize_solve!(vis::Visualizer, context::CvxContext1320, prior, solution, trace;
        framerate::Int=10,
        name::Symbol=:solve,
        animation::MeshCat.Animation=MeshCat.Animation(framerate))

    T = length(trace)
    nz = context.mechanism.dimensions.state
    mechanism = context.mechanism
    cameras = context.cameras

    # build iterates
    for i = 1:T
        build_iterate!(vis[name], context, prior, trace[i];
                name=Symbol(i), color=RGBA(1,1,1,0.4))
    end

    # build ground truth
    build_iterate!(vis[name], context, prior, solution;
            name=:solution, color=RGBA(1,0,0,0.4))
    settransform!(vis[name][:solution], MeshCat.Translation(-0.6,0,0))

    for i = 1:T
        atframe(animation, i) do
            for ii = 1:T
                setvisible!(vis[name][Symbol(ii)], ii == i)
            end
        end
    end
    MeshCat.setanimation!(vis, animation)
    return vis, animation
end

function build_iterate!(vis::Visualizer, context::CvxContext1320, prior, iterate;
        name::Symbol=:iterate,
        color=RGBA(1,1,1,0.4),
        color_prior=RGBA(0.3,0.3,0.3,0.4),
        color_predicted=RGBA(0.3,0.3,1,0.4),
        )

    subvis = vis[name]
    cameras = context.cameras
    mechanism = context.mechanism
    nz = mechanism.dimensions.state

    state_prior = prior[1:nz]
    params_prior = prior[nz+1:end]
    state = iterate[1:nz]
    params = iterate[nz+1:end]
    measurement = measurement_model(context, state, params)
    state_predicted, _ = process_model(context, state_prior, params)

    set_parameters!(context, params_prior)
    build_mechanism!(subvis, mechanism, name=:prior, show_contact=false,
        color=color_prior, center_of_mass_color=RGBA(0,0,0,1.0))
    set_mechanism!(subvis, mechanism, state_prior, name=:prior)
    settransform!(vis[name][:prior], MeshCat.Translation(-0.2,0,0))

    set_parameters!(context, params)
    build_mechanism!(subvis, mechanism, name=:predicted, show_contact=false,
        color=color_predicted, center_of_mass_color=RGBA(0,0,0,1.0))
    set_mechanism!(subvis, mechanism, state_predicted, name=:predicted)
    settransform!(vis[name][:predicted], MeshCat.Translation(-0.1,0,0))

    set_parameters!(context, params)
    build_mechanism!(subvis, mechanism, name=:mechanism, show_contact=false,
        color=color, center_of_mass_color=RGBA(0,0,0,1.0))
    set_mechanism!(subvis, mechanism, state, name=:mechanism)
    settransform!(vis[name][:mechanism], MeshCat.Translation(-0.0,0,0))



    for (j,camera) in enumerate(cameras)
        num_points = length(camera.camera_rays)
        build_point_cloud!(subvis[Symbol(:camera, j)], [num_points], name=:point_cloud, color=color)
        set_2d_point_cloud!(subvis[Symbol(:camera, j)], [camera.eye_position], measurement.d, name=:point_cloud)
    end

    return nothing
end
