function newton_solver!(xinit, loss, grad, hess, projection, clamping;
        max_iterations=20,
        reg_min=1e-4,
        reg_max=1e+0,
        reg_step=2.0,
        line_search_iterations=15,
        residual_tolerance=1e-4,
        D=Diagonal(ones(length(xinit))))

    stall = 0
    x = deepcopy(xinit)
    trace = [deepcopy(x)]
    reg = reg_max

    # newton's method
    for iterations = 1:max_iterations
        (stall >= 5) && break
        l = loss(x)
        (l < residual_tolerance) && break
        g = grad(x)
        H = hess(x)

        # reg = clamp(norm(g, Inf)/10, reg_min, reg_max)
        Δx = - (H + reg * D) \ g

        # linesearch
        α = 1.0
        for j = 1:line_search_iterations
            l_candidate = loss(projection(x + clamping(α * Δx)))
            if l_candidate <= l
                reg = clamp(reg/reg_step, reg_min, reg_max)
                stall = 0
                break
            end
            α /= 2
            if j == 10
                stall += 1
                α = 0.0
                reg = clamp(reg*exp(3.0*log(reg_step)), reg_min, reg_max)
            end
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
            mean(α),
            norm(clamping(α * Δx), Inf),
            norm(g, Inf),
            reg,
            )
        x = projection(x + clamping(α * Δx))
        push!(trace, deepcopy(x))
    end
    return x, trace
end
