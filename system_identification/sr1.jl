using Nonconvex
Nonconvex.@load Ipopt

function trust_region_model(n::Int; xinit=zeros(n), lb=-1e3, ub=1e3, verbose=false,
		ineq_scale=ones(n), max_cpu_time=0.5)

	function ineq_fct(x;)
		xs = ineq_scale .* x
		return [xs' * xs - 1]
	end

	model = Nonconvex.Model()
	addvar!(model, lb*ones(n), ub*ones(n), init=xinit, integer=falses(n))
	add_ineq_constraint!(model, ineq_fct)

	alg = IpoptAlg()
	print_level = verbose ? 5 : 0
	options = IpoptOptions(print_level=print_level, max_cpu_time=max_cpu_time)
	tr_model = (model, alg, xinit, options)
	return tr_model
end

function trust_region_solve!(tr_model::Tuple, g::AbstractVector, H::AbstractMatrix, Δ; ineq_scale=ones(n))
	model, alg, xinit, options = tr_model

	function obj_fct(x)
		x̂ = x .* ineq_scale
		return g' * x̂ + 0.5 * Δ * x̂' * H * x̂
	end

	set_objective!(model, obj_fct)

	result = Nonconvex.optimize(model, alg, xinit, options=options)
	return Δ .* result.minimizer .* ineq_scale
end

# n = 10
# g0 = 100 * ones(n)
# H0 = Matrix(Diagonal(ones(n)))
# Δ0 = 1e3
# tr_model = trust_region_model(n, max_cpu_time=0.30)
# norm(trust_region_solve!(tr_model, g0, H0, Δ0))

function sr1_solver!(xinit, loss, grad, projection, step_projection;
        max_iterations=20,
        reg_min=1e-4,
        reg_max=1e+0,
        reg_step=2.0,
        line_search_iterations=15,
        line_search_schedule=0.5,
        loss_tolerance=1e-4,
        grad_tolerance=1e-4,
		η=1e-6,
		r=0.5,
		Δx_scale=ones(length(xinit)),
		Δinit=1.0,
        Binit=Diagonal(ones(length(xinit))))

    n = length(xinit)
	tr_model = trust_region_model(n, ineq_scale=Δx_scale, max_cpu_time=0.25)

    # initialization
    stall = 0
    x = deepcopy(xinit)
    B = deepcopy(Binit)
	Δ = deepcopy(Δinit)
    g_next = zeros(n)
	g = grad(x)
    trace = [deepcopy(x)]
    reg = reg_max

    # SR1
    # https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf
    # Algorithm 6.1
    # Procedure 18.2
    for iterations = 1:max_iterations
        l = loss(x)
        ((stall >= 5) || (l < loss_tolerance) || (norm(g, Inf) < grad_tolerance)) && break

		Δx = trust_region_solve!(tr_model, g, B , Δ, ineq_scale=Δx_scale)
		Δx = step_projection(Δx)
		g_next = grad(x + Δx)

		y = g_next - g
		ared = l - loss(projection(x + Δx))
		pred = - (g'*Δx + 0.5 * Δx' * B * Δx)

		if ared / pred > η
			x = projection(x + Δx)
			g .= g_next
		end
		if ared / pred > 0.75
			if norm(Δx) <= 0.8 * Δ
				Δ = Δ
			else
				Δ = 2 * Δ
			end
		elseif 0.1 <= ared / pred <= 0.75
			Δ = Δ
		else
			Δ = 0.5 * Δ
		end

		if abs(Δx' * (y - B * Δx)) >= r * norm(Δx) * norm(y - B * Δx)
			@show "update"
			B = B + ((y - B * Δx) * (y - B * Δx)') / ((y - B * Δx)' * Δx)
		else
			B = B
		end

        # header
        if rem(iterations - 1, 10) == 0
            @printf "-------------------------------------------------------------------\n"
            @printf "iter   loss        step        |step|∞     |grad|∞     reg         \n"
            @printf "-------------------------------------------------------------------\n"
        end
        # iteration information
        @printf("%3d   %9.2e   %9.2e   %9.2e   %9.2e   %9.2e\n",
            iterations,
            l,
            Δ,
            norm(Δx, Inf),
            norm(g, Inf),
            reg,
            )
        push!(trace, deepcopy(x))
    end
    return x, trace
end

function sr1_solver!(xinit, loss, grad, projection, step_projection;
        max_iterations=20,
        reg_min=1e-4,
        reg_max=1e+0,
        reg_step=2.0,
        line_search_iterations=15,
        line_search_schedule=0.5,
        loss_tolerance=1e-4,
        grad_tolerance=1e-4,
		η=1e-6,
		r=0.5,
		Δx_scale=ones(length(xinit)),
		Δinit=1.0,
        Binit=Diagonal(ones(length(xinit))))

    n = length(xinit)
	tr_model = trust_region_model(n, ineq_scale=Δx_scale, max_cpu_time=0.25)

    # initialization
    stall = 0
    x = deepcopy(xinit)
    B = deepcopy(Binit)
	Δ = deepcopy(Δinit)
    g_next = zeros(n)
	g = grad(x)
    trace = [deepcopy(x)]
    reg = reg_max

	m   = zeros(size(x))
    v   = zeros(size(x))
    b1  = 0.9
    b2  = 0.999
    a   = 0.01
    eps = 1e-8
    t   = 0

    # SR1
    # https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf
    # Algorithm 6.1
    # Procedure 18.2
    for iterations = 1:max_iterations
        l = loss(x)
        ((stall >= 5) || (l < loss_tolerance) || (norm(g, Inf) < grad_tolerance)) && break

		t += 1
	    m = b1 .* m + (1 - b1) .* g
	    v = b2 .* v + (1 - b2) .* g .^ 2
	    mhat = m ./ (1 - b1^t)
	    vhat = v ./ (1 - b2^t)
		Δx = - a .* (mhat ./ (sqrt.(vhat) .+ eps))

		# Δx = trust_region_solve!(tr_model, g, B , Δ, ineq_scale=Δx_scale)
		Δx = trust_region_solve!(tr_model, mhat, B + Diagonal(sqrt.(vhat) .+ eps), Δ, ineq_scale=Δx_scale)
		Δx = step_projection(Δx)
		g_next = grad(x + Δx)

		y = g_next - g
		ared = l - loss(projection(x + Δx))
		pred = - (g'*Δx + 0.5 * Δx' * B * Δx)
		if ared / pred > η
			x = projection(x + Δx)
			g .= g_next
		end
		if ared / pred > 0.75
			if norm(Δx) <= 0.8 * Δ
				Δ = Δ
			else
				Δ = 2 * Δ
			end
		elseif 0.1 <= ared / pred <= 0.75
			Δ = Δ
		else
			Δ = 0.5 * Δ
		end

		if abs(Δx' * (y - B * Δx)) >= r * norm(Δx) * norm(y - B * Δx)
			@show "update"
			B = B + ((y - B * Δx) * (y - B * Δx)') / (1e-6 + (y - B * Δx)' * Δx)
		else
			B = B
		end

        # header
        if rem(iterations - 1, 10) == 0
            @printf "-------------------------------------------------------------------\n"
            @printf "iter   loss        step        |step|∞     |grad|∞     reg         \n"
            @printf "-------------------------------------------------------------------\n"
        end
        # iteration information
        @printf("%3d   %9.2e   %9.2e   %9.2e   %9.2e   %9.2e\n",
            iterations,
            l,
            Δ,
            norm(Δx, Inf),
            norm(g, Inf),
            reg,
            )
        push!(trace, deepcopy(x))
    end
    return x, trace
end
