# This is a module implementing vanilla Adam (https://arxiv.org/abs/1412.6980).
export Adam, step!

# Struct containing all necessary info
mutable struct Adam
	x::AbstractArray{Float64}     # Parameter array
	g::AbstractArray{Float64}     # Parameter gradient
    s::AbstractArray{Float64}     # Parameter step
    loss::Function                # Loss function
    grad::Function                # Gradient function
    m::AbstractArray{Float64}     # First moment
    v::AbstractArray{Float64}     # Second moment
    b1::Float64                   # Exp. decay first moment
    b2::Float64                   # Exp. decay second moment
    a::Float64                    # Step size
    eps::Float64                  # Epsilon for stability
    t::Int                        # Time step (iteration)
end

# Outer constructor
function Adam(x::AbstractArray{Float64}, loss::Function, grad::Function)
	g   = zeros(size(x))
	s   = zeros(size(x))
    m   = zeros(size(x))
    v   = zeros(size(x))
    b1  = 0.9
    b2  = 0.999
    a   = 0.001
    eps = 1e-8
    t   = 0
    Adam(x, g, s, loss, grad, m, v, b1, b2, a, eps, t)
end

# Step function with optional keyword arguments for the data passed to grad()
function step!(opt::Adam, projection; data...)
    opt.t += 1
    opt.g = opt.grad(opt.x; data...)
    opt.m = opt.b1 .* opt.m + (1 - opt.b1) .* opt.g
    opt.v = opt.b2 .* opt.v + (1 - opt.b2) .* opt.g .^ 2
    mhat = opt.m ./ (1 - opt.b1^opt.t)
    vhat = opt.v ./ (1 - opt.b2^opt.t)
	opt.s = - opt.a .* (mhat ./ (sqrt.(vhat) .+ opt.eps))
    opt.x += opt.s
	opt.x .= projection(opt.x)
end

function adam_solve!(opt::Adam;
        max_iterations::Int=100,
        l_tolerance=0.1,
		projection=x->x,
        data...)
    iterates = [deepcopy(opt.x)]
    for iterations = 1:max_iterations
		l = opt.loss(opt.x; data...)
		(l <= l_tolerance) && break
        step!(opt, projection; data...)
        push!(iterates, deepcopy(opt.x))

		# header
        if rem(iterations - 1, 10) == 0
            @printf "-------------------------------------------------------------------\n"
			@printf "iter   loss        |step|∞     |grad|∞     |1st mmt|∞  |2nd mmt|∞  \n"
            @printf "-------------------------------------------------------------------\n"
        end
        # iteration information
		@printf("%3d   %9.2e   %9.2e   %9.2e   %9.2e   %9.2e\n",
            iterations,
            l,
			norm(opt.s, Inf),
			norm(opt.g, Inf),
			norm(opt.m, Inf),
            norm(opt.v, Inf),
            )
    end
    return deepcopy(opt.x), iterates
end
