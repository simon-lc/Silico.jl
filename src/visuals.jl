######################################################################
# shape
######################################################################
function build_shape!(vis::Visualizer, shape::PolytopeShape{T,Ng,D};
        collider_color=RGBA(0.2, 0.2, 0.2, 0.8),
        ) where {T,Ng,D}

    A = shape.A
    b = shape.b
    o = shape.o
    if D == 2
        build_2d_polytope!(vis, A, b + A * o, color=collider_color)
    else
        build_polytope!(vis, A, b + A * o, color=collider_color)
    end
    return nothing
end

function build_shape!(vis::Visualizer, shape::SphereShape{T,Ng,D};
        collider_color=RGBA(0.2, 0.2, 0.2, 0.8),
        ) where {T,Ng,D}

    δ = 5e-3 * [0, 0, 1]
    offset = [zeros(3-D); shape.position_offset]
    radius = shape.radius[1]
    setobject!(vis,
        HyperSphere(GeometryBasics.Point(offset...), radius),
        MeshPhongMaterial(color=collider_color));
    setobject!(vis[:fake],
        HyperSphere(GeometryBasics.Point((offset .+ δ)...), radius),
        MeshPhongMaterial(color=RGBA(0.8, 0.8, 0.8, 0.8)));
    return nothing
end

function build_contact_shape!(vis::Visualizer, shape::Shape; collider_color=nothing)
end

function build_contact_shape!(vis::Visualizer, shape::PolytopeShape;
        collider_color=RGBA(0.5,0.5,0.5,0.0))
    build_shape!(vis, shape; collider_color=collider_color)
    return nothing
end

######################################################################
# body
######################################################################
function build_body!(vis::Visualizer, body::AbstractBody;
        name::Symbol=body.name,
        collider_color=RGBA(0.2, 0.2, 0.2, 0.8),
        center_of_mass_color=RGBA(1, 1, 1, 1.0),
        center_of_mass_radius=0.025,
        ) where T

    # shapes
    for (i,shape) in enumerate(body.shapes)
        build_shape!(vis[:bodies][name][Symbol(i)], shape, collider_color=collider_color)
    end

    # center of mass
    setobject!(vis[:bodies][name][:com],
        HyperSphere(GeometryBasics.Point(0,0,0.), center_of_mass_radius),
        MeshPhongMaterial(color=center_of_mass_color));
    return nothing
end

function set_body!(vis::Visualizer, body::AbstractBody{T,D}, pose; name=body.name) where {T,D}
    x = [0; pose[1:2]]
    q = RotX(pose[3])
    if D == 3
        x = pose[1:3]
        q = Quaternion(pose[4:7]...)
    end
    settransform!(vis[:bodies][name], MeshCat.compose(
        MeshCat.Translation(SVector{3}(x)),
        MeshCat.LinearMap(rotationmatrix(q)),
        )
    )
    return nothing
end

rotationmatrix(Quaternion(1, 0.0, 0.1, 0.0))

######################################################################
# mechanism
######################################################################
function build_mechanism!(vis::Visualizer, mechanism::Mechanism;
        show_contact::Bool=true,
        color=RGBA(1, 1, 1, 0.8),
        center_of_mass_color=RGBA(1,1,1,1.0),
        env_color=RGBA(0.5, 0.5, 0.5, 0.8),
        name::Symbol=:robot)

    for body in mechanism.bodies
        build_body!(vis[name], body, collider_color=color, center_of_mass_color=center_of_mass_color)
    end
    if show_contact
        for contact in mechanism.contacts
            build_frame!(vis[name][:contacts], dimension=space_dimension(contact), name=contact.name)
        end
    end
    for contact in mechanism.contacts
        build_contact_shape!(vis[contact.name], contact.child_shape; collider_color=env_color)
    end
    return nothing
end

function set_mechanism!(vis::Visualizer, mechanism::Mechanism, storage::TraceStorage,
        i::Int; show_contact::Bool=true, name::Symbol=:robot)

    for (j,body) in enumerate(mechanism.bodies)
        set_body!(vis[name], body, storage.x[i][j])
    end
    if show_contact
        for (j, contact) in enumerate(mechanism.contacts)
            ii = max(1,i-1) # needed otherwise the contact frame is a one step ahead of the bodies.
            origin = storage.contact_point[ii][j]
            normal = storage.normal[ii][j]
            tangent_x = storage.tangent_x[ii][j]
            tangent_y = storage.tangent_y[ii][j]
            set_frame!(vis[name][:contacts], origin, normal, tangent_x, tangent_y, name=contact.name)
        end
    end
    return nothing
end

function set_mechanism!(vis::Visualizer, mechanism::Mechanism, z; name::Symbol=:robot)
    off = 0
    for (j,body) in enumerate(mechanism.bodies)
        nz = state_dimension(body)
        set_body!(vis[name], body, z[off .+ (1:nz)]); off += nz
    end
    return nothing
end

function visualize!(vis::Visualizer, mechanism::Mechanism, storage::TraceStorage{T,H};
        build::Bool=true,
        show_contact::Bool=true,
        color=RGBA(0.2, 0.2, 0.2, 0.8),
        name::Symbol=:robot,
        animation=MeshCat.Animation(Int(floor(1/mechanism.bodies[1].timestep[1])))) where {T,H}

    build && build_mechanism!(vis, mechanism, show_contact=show_contact, color=color, name=name)
    for i = 1:H
        atframe(animation, i) do
            set_mechanism!(vis, mechanism, storage, i, show_contact=show_contact, name=name)
        end
    end
    MeshCat.setanimation!(vis, animation)
    return vis, animation
end

function visualize!(vis::Visualizer, mechanism::Mechanism, z;
        build::Bool=true,
        name::Symbol=:robot,
        animation=MeshCat.Animation(Int(floor(1/mechanism.bodies[1].timestep[1])))) where {T}

    H = length(z)
    build && build_mechanism!(vis, mechanism, show_contact=false, name=name)
    for i = 1:H
        atframe(animation, i) do
            set_mechanism!(vis, mechanism, z[i], name=name)
        end
    end
    MeshCat.setanimation!(vis, animation)
    return vis, animation
end


function set_frame!(vis::Visualizer, origin, normal, tangent_x, tangent_y; name::Symbol=:contact)
    dimension = length(origin)
    if dimension == 2
        origin = [0; origin]
        tangent_x = [1; tangent_x]
        tangent_y = [0; tangent_y]
        normal = [0; normal]
    end
    settransform!(vis[name][:origin],
        MeshCat.Translation(MeshCat.SVector{3}(origin...)))
    set_segment!(vis[name], origin, origin+normal; name=:normal)
    set_segment!(vis[name], origin, origin+tangent_x; name=:tangent_x)
    set_segment!(vis[name], origin, origin+tangent_y; name=:tangent_y)
    return nothing
end

function build_frame!(vis::Visualizer;
    dimension::Int=3,
    name::Symbol=:contact,
    origin_color=RGBA(0.2, 0.2, 0.2, 0.8),
    normal_axis_color=RGBA(0, 1, 0, 0.8),
    tangent_axis_color=RGBA(1, 0, 0, 0.8),
    origin_radius=0.025,
    ) where T

    # axes
    if dimension == 3
        build_segment!(vis[name];
            color=tangent_axis_color,
            segment_radius=origin_radius/2,
            name=:tangent_x)
    end
    build_segment!(vis[name];
        color=tangent_axis_color,
        segment_radius=origin_radius/2,
        name=:tangent_y)

    build_segment!(vis[name];
        color=normal_axis_color,
        segment_radius=origin_radius/2,
        name=:normal)

    # origin
    setobject!(vis[name][:origin],
        HyperSphere(GeometryBasics.Point(0,0,0.), origin_radius),
        MeshPhongMaterial(color=origin_color));
    return nothing
end
