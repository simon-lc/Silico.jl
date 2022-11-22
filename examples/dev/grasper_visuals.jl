function capsule_pose(pose, segment; capsule_id::Int=1)
    x = pose[1:2]
    α0 = pose[3]
    α1 = pose[4] # ∈ [+0.20π, +0.40π]
    α2 = pose[5] # ∈ [+0.00π, +0.45π]
    α3 = pose[6] # ∈ [+0.00π, +0.45π]
    p = zeros(3)
    if capsule_id == 1
        β1 = α0 - α1
        x1 = x - segment[1]/2 * [cos(α0+π/2), sin(α0+π/2)] + segment[2]/2 * [cos(β1), sin(β1)]
        p = [x1; β1]
    elseif capsule_id == 2
        β1 = α0 - α1
        β2 = α0 - α1 + α2
        x2 = x - segment[1]/2 * [cos(α0+π/2), sin(α0+π/2)] + segment[2] * [cos(β1), sin(β1)] + segment[3]/2 * [cos(β2), sin(β2)]
        p = [x2; β2]
    elseif capsule_id == 3
        β3 = α0 + α1
        x3 = x + segment[1]/2 * [cos(α0+π/2), sin(α0+π/2)] + segment[2]/2 * [cos(β3), sin(β3)]
        p = [x3; β3]
    elseif capsule_id == 4
        β3 = α0 + α1
        β4 = α0 + α1 - α3
        x4 = x + segment[1]/2 * [cos(α0+π/2), sin(α0+π/2)] + segment[2] * [cos(β3), sin(β3)] + segment[3]/2 * [cos(β4), sin(β4)]
        p = [x4; β4]
    end
    return p
end

function set_grasper!(vis::Visualizer, pose;
        name=:grasper,
        segment=[1.00, 1.00, 1.00])
    for k = 1:9
        for i = 1:4
            xi = capsule_pose(pose, segment; capsule_id=i)
            settransform!(vis[name][Symbol(k)][Symbol(:capsule_,i)], MeshCat.compose(
                MeshCat.Translation(SVector{3}(0, xi[1:2]...)),
                MeshCat.LinearMap(rotationmatrix(RotX(xi[3]))),
                )
            )
        end
    end
    return nothing
end

function pose_cost(pose, state, pose_ref;
        segment=[1.00, 1.00, 1.00],
        radius=0.10,
        A=[[1 0; 0 1; -1 0; 0 -1]],
        b=[0.4*ones(4)],
        )

    box = state[1:3]
    sphere_2 = state[4:6]
    sphere_4 = state[7:9]

    α0 = pose[3]
    α1 = pose[4]
    α2 = pose[5]
    α3 = pose[6]
    c0l = pose[1:2] - segment[1]/2 * [cos(α0+π/2), sin(α0+π/2)]
    c0r = pose[1:2] + segment[1]/2 * [cos(α0+π/2), sin(α0+π/2)]

    θ1 = α0 - α1
    c1 = c0l + segment[2] * [cos(θ1), sin(θ1)]

    θ1 = α0 - α1
    θ2 = α0 - α1 + α2
    c2 = c1 + segment[3] * [cos(θ2), sin(θ2)]

    θ3 = α0 + α1
    c3 = c0r + segment[2] * [cos(θ3), sin(θ3)]

    θ3 = α0 + α1
    θ4 = α0 + α1 - α3
    c4 = c3 + segment[3] * [cos(θ4), sin(θ4)]

    l = 0.0
    l += 1e0 * 0.5 * (c2 - sphere_2[1:2])' * (c2 - sphere_2[1:2])
    l += 1e0 * 0.5 * (c4 - sphere_4[1:2])' * (c4 - sphere_4[1:2])
    l += 1e-2 * 0.5 * (pose - pose_ref)' * (pose - pose_ref)
    l += 1e-2 * 0.5 * (pose[4] - 0.20π)^2
    l += 1e0* min(0, c1[2] - radius)^2
    l += 1e0* min(0, c2[2] - radius)^2
    l += 1e0* min(0, c3[2] - radius)^2
    l += 1e0* min(0, c4[2] - radius)^2

    for i = 1:length(b)
        for α ∈ 0:0.1:1
            xw = (α * c0l + (1 - α) * c1)
            xb = x_2d_rotation(box[3:3])' * (xw - box[1:2])
            l += 1e-1 * 0.5 * min(0, maximum(A[i] * xb - b[i]) - radius)^2
            xw = (α * c1 + (1 - α) * c2)
            xb = x_2d_rotation(box[3:3])' * (xw - box[1:2])
            l += 1e-1 * 0.5 * min(0, maximum(A[i] * xb - b[i]) - radius)^2
            xw = (α * c0r + (1 - α) * c3)
            xb = x_2d_rotation(box[3:3])' * (xw - box[1:2])
            l += 1e-1 * 0.5 * min(0, maximum(A[i] * xb - b[i]) - radius)^2
            xw = (α * c3 + (1 - α) * c4)
            xb = x_2d_rotation(box[3:3])' * (xw - box[1:2])
            l += 1e-1 * 0.5 * min(0, maximum(A[i] * xb - b[i]) - radius)^2
        end
    end
    return l
end

function clamp_pose(pose;
        pose_high=[+10.0,+10.0,+5.00π,+0.50π,+0.75π,+0.75π],
        pose_low= [-10.0,-10.0,-5.00π,-0.10π,+0.00π,+0.00π],
        )
    return clamp.(pose, pose_low, pose_high)
end

function grasper_pose(pose_ref, maximal_state;
        segment=[1.00, 1.00, 1.00],
        radius=0.10,
        A=[[1 0; 0 1; -1 0; 0 -1]],
        b=[0.4*ones(4)],
        )

    pose = deepcopy(pose_ref)
    l_prev = 0
    cost(pose) = pose_cost(pose, maximal_state, pose_ref,
        segment=segment,
        radius=radius,
        A=A,
        b=b)

    for i = 1:20
        l = cost(pose)
        ((l < 1e-4) || (abs(l_prev - l) < 1e-8)) && break
        (l < 1e-4) && break
        @show l
        H = Mehrotra.FiniteDiff.finite_difference_hessian(
            pose -> cost(pose), pose)
        g = Mehrotra.FiniteDiff.finite_difference_gradient(
            pose -> cost(pose), pose)
        Δ = - (H + 1e-6 * I) \ g

        α = 1.0
        for j = 1:20
            candidate_l = cost(clamp_pose(pose + α * Δ))
            (candidate_l <= l) && break
            α *= 0.5
        end
        pose = clamp_pose(pose + α * Δ)
        l_prev = l
    end
    return pose
end

function build_grasper!(vis::Visualizer;
        name=:grasper,
        segment=[1.00, 1.00, 1.00],
        radius=0.10,
        A=[[1 0; 0 1; -1 0; 0 -1]],
        b=[0.4*ones(4)],
        color=RGBA(0,0,0,1),
        )

    for k = 1:9
        for i = 1:4
            j = 2 + (i == 2 || i == 4)
            shape = CapsuleShape(radius, segment[j])
            build_shape!(vis[name][Symbol(k)][Symbol(:capsule_,i)], shape, collider_color=color)
            settransform!(vis[name][Symbol(k)], MeshCat.Translation((k-4)*0.005, 0, 0))
        end
    end
    return nothing
end

function grasper_trajectory(state_trajectory;
        segment=[1.00, 1.00, 1.00],
        radius=0.10,
        A=[[1 0; 0 1; -1 0; 0 -1]],
        b=[0.4*ones(4)],
        initial_pose=[0, 1, -π/2, 0.8, 1.0, 1.0])

    pose = initial_pose
    pose_trajectory = []
    for i = 1:length(state_trajectory)
        @show i
        state = deepcopy(state_trajectory[i])
        new_pose = grasper_pose(pose, state,
            segment=segment,
            radius=radius,
            A=A,
            b=b)
        push!(pose_trajectory, new_pose)
        pose = new_pose
    end
    return pose_trajectory
end

function visualize!(vis::Visualizer, pose_trajectory;
        segment=[1.00, 1.00, 1.00],
        radius=0.10,
        A=[1 0; 0 1; -1 0; 0 -1],
        b=0.4*ones(4),
        build=true,
        color=RGBA(0.2,0.2,0.2,1),
        name::Symbol=:grasper,
        animation=MeshCat.Animation(Int(floor(1/0.05)))
        )

    build && build_grasper!(vis;
            name=name,
            segment=segment,
            radius=radius,
            A=A,
            b=b,
            color=color,
            )

    for i = 1:length(pose_trajectory)
        atframe(animation, i) do
            set_grasper!(vis, pose_trajectory[i], name=name, segment=segment)
        end
    end
    MeshCat.setanimation!(vis, animation)
    return vis, animation
end
