################################################################################
# visualize convex_bundle
################################################################################
function build_2d_convex_bundle!(vis::Visualizer, θ, polytope_dimensions;
        name::Symbol=:convex_bundle, color=RGBA(1,1,0,0.4))

    np = length(polytope_dimensions)
    A, b, o = unpack_halfspaces(θ, polytope_dimensions)

    for i = 1:np
		setobject!(
			vis[name][Symbol(i)][:center],
			HyperSphere(MeshCat.Point(0,o[i]...), 0.025),
			MeshPhongMaterial(color=RGBA(color.r, color.g, color.b, 1.0)))
        build_2d_polytope!(vis[name], A[i], b[i] + A[i]*o[i], name=Symbol(i), color=color)
    end
    return nothing
end

################################################################################
# visualize point cloud
################################################################################
function build_point_cloud!(vis::Visualizer, num_points::Vector{Int};
        color=RGBA(0.1,0.1,0.1,1), name::Symbol=:point_cloud, names=Symbol.(1:length(num_points)))

    for (i,n) in enumerate(num_points)
        build_point_cloud!(vis[name], n, name=names[i], color=color)
    end
    return nothing
end

function build_point_cloud!(vis::Visualizer, num_points::Int;
        name::Symbol=:point_cloud, fov_length=0.2,
        color=RGBA(0.1,0.1,0.1,1))

    for i = 1:num_points
        setobject!(
            vis[name][:eyeposition],
            HyperSphere(MeshCat.Point(0,0,0.0), 0.05),
            MeshPhongMaterial(color=color))
		setobject!(
			vis[name][:points][Symbol(i)],
			HyperSphere(MeshCat.Point(0,0,0.0), 0.025),
			MeshPhongMaterial(color=color))

        if i == 1 || i == num_points
            fov_name = Symbol(:fov_, i)
            build_segment!(vis[name]; color=color, name=fov_name)
        end
    end
    return nothing
end

function set_2d_point_cloud!(vis::Visualizer, e::Vector{<:Vector}, P::Vector{<:Matrix};
        name::Symbol=:point_cloud)

    ee = [[0; ei] for ei  in e]
    Pe = [[zeros(size(Pi,2))'; Pi] for Pi in P]
    set_point_cloud!(vis, ee, Pe, name=name)
    return nothing
end

function set_point_cloud!(vis::Visualizer, e::Vector{<:Vector}, P::Vector{<:Matrix};
        name::Symbol=:point_cloud)

    ne = length(e)
    for i = 1:ne
        set_point_cloud!(vis[name], e[i], P[i], name=Symbol(i))
    end
    return nothing
end

function set_point_cloud!(vis::Visualizer, e::Vector, P::Matrix;
        name::Symbol=:point_cloud, fov_length=0.2)

	num_points = size(P, 2)
    for i = 1:num_points
        settransform!(vis[name][:eyeposition], MeshCat.Translation(e...))
		settransform!(vis[name][:points][Symbol(i)], MeshCat.Translation(P[:,i]...))

        if i == 1 || i == num_points
            v = (P[:,i] - e)
            v = v ./ norm(v) * fov_length
            fov_name = Symbol(:fov_, i)
            set_segment!(vis[name], e, e + v; name=fov_name)
        end
    end
    return nothing
end


################################################################################
# plotting
################################################################################
function plot_halfspace(plt, a, b; show_plot::Bool=true)
    R = [0 1; -1 0]
    v = R * a[1,:]
    x0 = a \ b
    xl = x0 - 100*v
    xu = x0 + 100*v
    plt = plot!(plt, [xl[1], xu[1]], [xl[2], xu[2]], linewidth=5.0, color=:white)
    show_plot && display(plt)
    return plt
end

function plot_polytope(A, b, δ;
        xlims=(-1,1),
        ylims=(-1,1),
        S::Int=25,
        size=(400,400),
        plot_heatmap=true,
        plot_halfspaces=true,
        countour_color=:black,
        levels=[0.0],
        plt=plot())

    plot!(plt,
        aspectratio=1.0,
        xlims=xlims,
        ylims=ylims,
        xlabel="x", ylabel="y",
        size=size)

    X = range(xlims..., length=S)
    Y = range(ylims..., length=S)
    V = zeros(S,S)

    for i = 1:S
        for j = 1:S
            p = [X[i], Y[j]]
            V[j,i] = sdf(p, A, b, δ)[1]
        end
    end

    if plot_heatmap
        plt = heatmap!(plt,
        X, Y, V,
        )
    end

    if plot_halfspaces
        for i = 1:length(b)
            plt = plot_halfspace(plt, A[i:i,:], b[i:i], show_plot=false)
        end
    end
    plt = contour(plt, X,Y,V, levels=levels, color=countour_color, linewidth=2.0, legend=false)
    return plt
end

function plot_heatmap(;
        xlims=(-1,1),
        ylims=(-1,1),
        S::Int=25,
        size=(400,400),
        plot_heatmap=true,
        plot_halfspaces=true,
        countour_color=:black,
        f=x->x,
        levels=[0.0],
        plt=plot())

    plot!(plt,
        aspectratio=1.0,
        xlims=xlims,
        ylims=ylims,
        xlabel="x", ylabel="y",
        size=size)

    X = range(xlims..., length=S)
    Y = range(ylims..., length=S)
    V = zeros(S,S)

    for i = 1:S
        for j = 1:S
            p = [X[i], Y[j]]
            V[j,i] = f(p)
        end
    end

    if plot_heatmap
        plt = heatmap!(plt,
        X, Y, V,
        )
    end

    plt = contour(plt, X,Y,V, levels=levels, color=countour_color, linewidth=2.0, legend=false)
    return plt
end


function visualize_kmeans!(vis::Visualizer, θ, polytope_dimensions, d_object, kmres)
	colors = [
	    RGBA(1,0,0,1),
	    RGBA(0,1,0,1),
	    RGBA(0,0,1,1),
	    RGBA(0,1,1,1),
	    RGBA(1,1,0,1),
	    RGBA(1,0,1,1),
	    RGBA(0.5,0.5,0.5,1),
	];

	# display k-mean result
	for i = 1:size(d_object, 2)
	    ik = kmres.assignments[i]
	    setobject!(
	        vis[:cluster][Symbol(ik)][Symbol(i)],
	        HyperSphere(MeshCat.Point(0,0,0.0), 0.035),
	        MeshPhongMaterial(color=colors[ik%7+1]))
	    settransform!(vis[:cluster][Symbol(ik)][Symbol(i)], MeshCat.Translation(0.2, d_object[:,i]...))
	end
	for i = 1:np
	    setobject!(
	        vis[:cluster][Symbol(i)][:center],
	        HyperRectangle(MeshCat.Vec(-0.05,-0.05,-0.05), MeshCat.Vec(0.1,0.1,0.1)),
	        MeshPhongMaterial(color=colors[i%7+1]))
	    settransform!(vis[:cluster][Symbol(i)][:center], MeshCat.Translation(0.2, kmres.centers[:,i]...))
	end
	build_2d_convex_bundle!(vis, θ, polytope_dimensions, name=:initial, color=RGBA(1,1,1,0.4))
	return nothing
end

function visualize_polytope_iterates!(vis::Visualizer, θ, polytope_dimensions;
		max_iterations::Int=length(θ),
		color=RGBA(1,1,0,0.4),
		animation=MeshCat.Animation(10))
	T = length(θ)

	for i = 1:max_iterations
		im = min(i, length(θ))
		build_2d_convex_bundle!(vis[:iterates], θ[im], polytope_dimensions, name=Symbol(i), color=color)
	end

	for i = 1:max_iterations
		atframe(animation, i) do
	        for ii = 1:max_iterations
	            setvisible!(vis[:iterates][Symbol(ii)], ii == i)
	        end
	    end
	end
	settransform!(vis[:iterates], MeshCat.Translation(0.05,0,0.0))
	setanimation!(vis, animation)
	return vis, animation
end

function visualize_iterates!(vis::Visualizer, θ, polytope_dimensions, e, β, ρ;
		point_cloud::Bool=false,
		max_iterations::Int=length(θ),
		color=RGBA(1,1,0,0.4),
		animation=MeshCat.Animation(10))
	nβ = length(β)
	T = length(θ)

	for i = 1:max_iterations
		im = min(i, length(θ))
		build_2d_convex_bundle!(vis[:iterates], θ[im], polytope_dimensions, name=Symbol(i), color=color)
	end
	point_cloud && build_point_cloud!(vis[:iterates][:point_cloud], nβ;
		color=RGBA(0.8,0.1,0.1,1), name=Symbol(1))

	# for i = 1:max_iterations
	# 	atframe(animation, i) do
	# 		for ii = 1:max_iterations
	# 			setvisible!(vis[:iterates][Symbol(ii)], false)
	# 		end
	# 	end
	# end

	for i = 1:max_iterations
		atframe(animation, i) do
			if point_cloud
				im = min(i, length(θ))
				θ_f, polytope_dimensions_f = add_floor(θ[im], polytope_dimensions)
				d = trans_point_cloud(e, β, ρ, unpack_halfspaces(θ_f, polytope_dimensions_f)...)
				set_2d_point_cloud!(vis[:iterates], [e], [d]; name=:point_cloud)
			end
	        for ii = 1:max_iterations
	            setvisible!(vis[:iterates][Symbol(ii)], ii == i)
	        end
	    end
	end
	settransform!(vis[:iterates], MeshCat.Translation(0.05,0,0.0))
	setanimation!(vis, animation)
	return vis, animation
end


#
# S = 50
# xl = [-2, 2.0]
# yl = [-0, 2.0]
# X = range(xl..., length=S)
# Y = range(yl..., length=S)
# V = zeros(S,S)
# for (i, p1) in enumerate(X)
# 	for (j, p2) in enumerate(Y)
# 		p = [p1, p2]
# 		V[j,i] = sdf4(p, AAt, bbt, oot, 100.0)
# 		V[j,i] = sigmoid(-20*sdf(p, AAt[3], bbt[3], oot[3], 1000.0))
# 		# V[j,i] = sdf(p, AAt[1], bbt[1], oot[1], 1000.0)
# 		# V[j,i] = sdf(p, AAt[2], bbt[2], oot[2], 1000.0)
# 		# V[j,i] = sdf(p, AAt[3], bbt[3], oot[3], 1000.0)
# 		# V[j,i] = sdf(p, AAt[4], bbt[4], oot[4], 1000.0)
# 		# V[j,i] = sdf(p, AAt[5], bbt[5], oot[5], 1000.0)
# 		# V[j,i] = sdf(p, AAt[6], bbt[6], oot[6], 1000.0)
# 		# V[j,i] = sdf(p, AAt[7], bbt[7], oot[7], 1000.0)
# 	end
# end
#
# plt = heatmap(
# 	X, Y, V,
# 	aspectratio=1.0,
# 	xlims=xl,
# 	ylims=yl,
# 	xlabel="x", ylabel="y",
# 	)
# plt = contour(plt, X,Y,V, levels=[0.0], color=:white, linewidth=2.0)
#
# # plot_polytope(AAt[1], bbt[1] + AAt[1] * oot[1], 100.0, xlims=[-2,2], ylims=[0,2])
# # plot_polytope(AAt[2], bbt[2] + AAt[2] * oot[2], 100.0, xlims=[-2,2], ylims=[0,2])
# # plot_polytope(AAt[3], bbt[3] + AAt[3] * oot[3], 100.0, xlims=[-2,2], ylims=[0,2])
# # plot_polytope(AAt[4], bbt[4] + AAt[4] * oot[4], 100.0, xlims=[-2,2], ylims=[0,2])
# # plot_polytope(AAt[5], bbt[5] + AAt[5] * oot[5], 100.0, xlims=[-2,2], ylims=[0,2])
# # plot_polytope(AAt[6], bbt[6] + AAt[6] * oot[6], 100.0, xlims=[-2,2], ylims=[0,2])
# plot_polytope(AAt[7], bbt[7] + AAt[7] * oot[7], 100.0, xlims=[-2,2], ylims=[0,2])
#



# Apt, bpt, opt = unpack_halfspaces(θiter[end], polytope_dimensions)
# # inside sampling, overlap penalty
# for i = 1
# 	p = opt[i]
# 	for j = 1:length(bpt[i])
# 		p = opt[i] + 1.0 * Apt[i][j,:] .* bpt[i][j] / norm(Apt[i][j,:])^2
# 		setobject!(vis[:sampling][Symbol(i)][Symbol(j)], HyperSphere(MeshCat.Point(0, p...), 0.05),
# 			MeshPhongMaterial(color=RGBA(1,0,0,1)))
# 		sleep(0.1)
# 	end
# end
