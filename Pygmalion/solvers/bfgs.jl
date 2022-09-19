function bfgs_solver!(xinit, loss, grad, projection, step_projection;
        max_iterations=20,
        reg_min=1e-4,
        reg_max=1e+0,
        reg_step=2.0,
        line_search_iterations=15,
        line_search_schedule=0.5,
        loss_tolerance=1e-4,
        grad_tolerance=1e-4,
        # Hinit=Diagonal(ones(length(xinit))))
        Binit=Diagonal(ones(length(xinit))))

    n = length(xinit)
    # initialization
    stall = 0
    x = deepcopy(xinit)
    # H = deepcopy(Hinit)
    B = deepcopy(Binit)
    g_previous = zeros(n)
    trace = [deepcopy(x)]
    reg = reg_max
    l_minimum = loss(xinit)

    # BFGS
    # https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf
    # Algorithm 6.1
    # Procedure 18.2
    for iterations = 1:max_iterations
        (stall >= 5) && break
        l = loss(x)
        (l < loss_tolerance) && break
        g = grad(x)
        (norm(g, Inf) < grad_tolerance) && break
        y = g - g_previous

        # Δx = - (inv(H) + reg * I) \ g
        # @show isposdef(B)
        # Δx = - (B + reg * I) \ g
        Δx = - (B + reg * Binit) \ g
        # Δx = - B \ g

        # linesearch
        α = 1.0
        for j = 1:line_search_iterations
            l_candidate = loss(projection(x + step_projection(α * Δx)))
            if (l_candidate <= 1.00 * l_minimum)
                (l_candidate <= l_minimum) && (l_minimum = l_candidate)
                # (j > 6) && (reg = clamp(reg/reg_step, reg_min, reg_max))
                reg = clamp(reg/reg_step, reg_min, reg_max)
                stall = 0
                break
            end
            α *= line_search_schedule
            if j == line_search_iterations
                stall += 1
                α = 0.0
                reg = clamp(reg*exp(3.0*log(reg_step)), reg_min, reg_max)
            end
        end

        s = step_projection(α * Δx)

        # θ = 1.0
        # r = deepcopy(y)
        # for j = 1:100
        #     r = θ * y + (1 - θ) * B * s
        #     if r' * s > 0
        #         break
        #     end
        #     θ *= 0.8
        #     if j == 100
        #         θ = 0.0
        #         break
        #     end
        # end
        θ = 1.0
        r = θ * y + (1 - θ) * B * s
        B = B - B * s * s' * B / (1e-3 + s' * B * s) + r * r' / (1e-3 + s' * r)
        # H .= (I - ρr * s * r') * H * (I - ρr * r * s') + ρr * s * s'

        # θ = 1.0
        # if s' * y >= 0.2 * s' * B * s
        #     θ = 1.0
        # else
        #     θ = (0.8 * s' * B * s) / (s' * B * s - s' * y)
        # end
        # r = θ * y + (1 - θ) * B * s
        # B = B - B * s * s' * B / (s' * B * s) + r * r' / (s' * r)

        # ρ = 1 / (y' * s)
        # if 1 / ρ > 0
        #     H .= (I - ρ * s * y') * H * (I - ρ * y * s') + ρ * s * s'
        #     # D = inv(H) + 1/reg * Hinit
        #     # H = inv(D)
        # else
        #     @show iterations
        #     D = inv(H) + 1/reg * Hinit
        #     H = inv(D)
        #     nothing
        # end

        # Bs = (H \ s)
        # if s'* y >= 0.2 * s' * Bs
        #     θ = 1.0
        # else
        #     θ = (0.8 * s' * Bs) / (s' * Bs - s' * y)
        # end
        # r = θ * y + (1 - θ) * Bs
        # ρr = 1 / (r' * s)
        # H .= (I - ρr * s * r') * H * (I - ρr * r * s') + ρr * s * s'


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
            norm(step_projection(α * Δx), Inf),
            norm(g, Inf),
            reg,
            )
        x = projection(x + step_projection(α * Δx))
        push!(trace, deepcopy(x))
        g_previous .= g
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
        Binit=Diagonal(ones(length(xinit))))

    n = length(xinit)
    # initialization
    stall = 0
    x = deepcopy(xinit)
    B = deepcopy(Binit)
    g_previous = zeros(n)
    trace = [deepcopy(x)]
    reg = reg_max
    l_minimum = loss(xinit)

    # SR1
    # https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf
    # Algorithm 6.1
    # Procedure 18.2
    for iterations = 1:max_iterations
        (stall >= 5) && break
        l = loss(x)
        (l < loss_tolerance) && break
        g = grad(x)
        (norm(g, Inf) < grad_tolerance) && break
        y = g - g_previous

        # Δx = - (B + reg * Binit) \ g
        Δx = - B \ g

        # linesearch
        α = 1.0
        for j = 1:line_search_iterations
            l_candidate = loss(projection(x + step_projection(α * Δx)))
            if (l_candidate <= 1.00 * l_minimum)
                (l_candidate <= l_minimum) && (l_minimum = l_candidate)
                reg = clamp(reg/reg_step, reg_min, reg_max)
                stall = 0
                break
            end
            α *= line_search_schedule
            if j == line_search_iterations
                stall += 1
                α = 0.0
                reg = clamp(reg*exp(3.0*log(reg_step)), reg_min, reg_max)
            end
        end

        s = step_projection(α * Δx)
        if s' * (y - B * s) >= 1e-5 * norm(s) * norm(y - B * s)
            B = B + ((y - B * s) * (y - B * s)') / ((y - B * s)' * s)
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
            norm(step_projection(α * Δx), Inf),
            norm(g, Inf),
            reg,
            )
        x = projection(x + step_projection(α * Δx))
        push!(trace, deepcopy(x))
        g_previous .= g
    end
    return x, trace
end
