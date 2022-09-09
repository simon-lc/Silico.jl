# This is a module implementing vanilla Adam (https://arxiv.org/abs/1412.6980).
export Adam, step!

# Struct containing all necessary info
mutable struct Adam
    x::AbstractArray{Float64}     # Parameter array
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
    m   = zeros(size(x))
    v   = zeros(size(x))
    b1  = 0.9
    b2  = 0.999
    a   = 0.001
    eps = 1e-8
    t   = 0
    Adam(x, loss, grad, m, v, b1, b2, a, eps, t)
end

# Step function with optional keyword arguments for the data passed to grad()
function step!(opt::Adam, projection; data...)
    opt.t += 1
    gt    = opt.grad(opt.x; data...)
    opt.m = opt.b1 .* opt.m + (1 - opt.b1) .* gt
    opt.v = opt.b2 .* opt.v + (1 - opt.b2) .* gt .^ 2
    mhat = opt.m ./ (1 - opt.b1^opt.t)
    vhat = opt.v ./ (1 - opt.b2^opt.t)
    opt.x -= opt.a .* (mhat ./ (sqrt.(vhat) .+ opt.eps))
	opt.x .= projection(opt.x)
end

function adam_solve!(opt::Adam;
        max_iterations::Int=100,
        l_tolerance=0.3,
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
            @printf "iter   loss        step        |step|∞     |grad|∞     reg         \n"
            @printf "-------------------------------------------------------------------\n"
        end
        # iteration information
		# @printf("%3d   %9.2e   %9.2e   %9.2e   %9.2e   %9.2e\n",
        @printf("%3d   %9.2e\n",
            iterations,
            l,
            # mean(α),
            # norm(clamping(α * Δx), Inf),
            # norm(g, Inf),
            # reg,
            )

    end
    return deepcopy(opt.x), iterates
end
#
#
# adam_opt = Adam(θinit, local_loss, local_grad)
# θsol0, θiter0 = adam_solve!(adam_opt)
