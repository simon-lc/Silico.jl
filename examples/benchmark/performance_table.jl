using Plots
using Statistics
using Random
using BenchmarkTools

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# define mechanism
################################################################################
timestep = 0.01;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.2;

mech = get_polytope_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    # method_type=:symbolic,
    method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=false,
        differentiate=false,
        residual_tolerance=0.5e-6,
        complementarity_tolerance=0.5e-6,
        compressed_search_direction=true,
        # compressed_search_direction=false,
        sparse_solver=false,
        warm_start=true,
        )
    );

################################################################################
# test simulation
################################################################################
xp2 = [+0.0,1.5,-0.001]
vp15 = [-0,0,-0.0]
z0 = [xp2; vp15]
H0 = 100000

function stable_ctrl(mechanism, i)
    s = get_current_state(mechanism.bodies[1])
    kr = 10.0
    kp = 2.0 * [1, 1, 0]
    kd = 0.5 * [1, 1, 1]
    u = kr * (rand(3) .- [0.4, 0.1, 0]) - kp .* s[1:3] - kd .* s[4:6]
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    return nothing
end

Random.seed!(0)
storage, solve_time = simulate!(mech, z0, H0, controller=stable_ctrl, violation=:absolute, timing=true)
solve_time_μs = 1e6 * solve_time / H0

mean(storage.cone_product_violations)
maximum(storage.cone_product_violations)
mean(storage.equality_violations)
maximum(storage.equality_violations)
mean(storage.iterations)
maximum(storage.iterations)

################################################################################
# visualization
################################################################################
build_mechanism!(vis, mech)
set_mechanism!(vis, mech, storage, 10)

# visualize!(vis, mech, storage, build=false)

scatter(storage.iterations)
plot!(hcat(storage.variables...)')
plot(storage.equality_violations)
plot(storage.cone_product_violations)












################################################################################
# define mechanisms
################################################################################
A0 = [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        +1.0 -0.2;
        -1.0 -0.2;
    ]
b0 = 0.23*[+1.0, +3.0, +1.0, +1.0, 0.9, 0.9]

mech = get_polytope_insertion(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    # method_type=:symbolic,
    method_type=:finite_difference,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=false,
        differentiate=false,
        residual_tolerance=0.5e-6,
        complementarity_tolerance=0.5e-6,
        compressed_search_direction=true,
        # compressed_search_direction=false,
        # complementarity_backstep=1e-1,
        sparse_solver=false,
        # sparse_solver=true,
        warm_start=true,
        )
    )

################################################################################
# simulation
################################################################################
H0 = 10000
x2 = [+0.00, +0.50, +0.05]
v15 = [-0.0, +0.0, -0.0]
z0 = [x2; v15]


function stable_ctrl(mechanism, i)
    s = get_current_state(mechanism.bodies[1])
    kr = 10.0
    kp = 2.0 * [1, 5, 1]
    kd = 0.5 * [1, 1, 1]
    u = kr * (rand(3) .- [0.5, 0.5, 0.5]) - kp .* (s[1:3] - [0,1.5,0]) - kd .* s[4:6]
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    return nothing
end

storage, solve_time = simulate!(mech, z0, H0, controller=stable_ctrl, violation=:absolute, timing=true)
solve_time_μs = 1e6 * solve_time / H0

mean(storage.cone_product_violations)
maximum(storage.cone_product_violations)
mean(storage.equality_violations)
maximum(storage.equality_violations)
mean(storage.iterations)
maximum(storage.iterations)


vis, anim = visualize!(vis, mech, storage)
scatter(storage.iterations, color=:red)





















################################################################################
# define mechanisms
################################################################################
A0 = [
    [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 +0.0;
        +0.0 +1.0;
        -1.0 +0.0;
        +0.0 -1.0;
        ],
    ]
b0 = [
        0.25*[+1.0, +1.0, +1.0, +1.0],
        0.25*[+1.0, +1.0, +1.0, +1.0],
        0.25*[+1.0, +1.0, +1.0, +1.0],
        0.25*[+1.0, +1.0, +1.0, +1.0],
    ]
friction_coefficient = 0.6

mech = get_polytope_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    # method_type=:symbolic,
    method_type=:finite_difference,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=false,
        differentiate=false,
        residual_tolerance=0.5e-6,
        complementarity_tolerance=0.5e-6,
        # compressed_search_direction=false,
        compressed_search_direction=true,
        # sparse_solver=false,
        sparse_solver=true,
        warm_start=true,
        )
    )


################################################################################
# simulation
################################################################################
N = 100
solve_times = zeros(N)
cone_product_violations_mean = zeros(N)
cone_product_violations_max = zeros(N)
equality_violations_mean = zeros(N)
equality_violations_max = zeros(N)
iterations_mean = zeros(N)
iterations_max = zeros(N)

Random.seed!(0)
for i = 1:N
    @show i
    x12 = [+0.05, +2.40, +0.1]
    x22 = [-0.05, +1.80, -0.1]
    x32 = [-0.00, +1.00, -0.2]
    x42 = [-0.00, +0.30, -0.2]
    v115 = 1.0 * (rand(3) .- 0.5)
    v215 = 1.0 * (rand(3) .- 0.5)
    v315 = 1.0 * (rand(3) .- 0.5)
    v415 = 1.0 * (rand(3) .- 0.5)
    z0 = [x12; v115; x22; v215; x32; v315; x42; v415]
    H0 = 1000

    storage, solve_time = simulate!(mech, z0, H0, violation=:absolute, timing=true)
    solve_times[i] = solve_time
    cone_product_violations_mean[i] = mean(storage.cone_product_violations)
    cone_product_violations_max[i] = maximum(storage.cone_product_violations)
    equality_violations_mean[i] = mean(storage.equality_violations)
    equality_violations_max[i] = maximum(storage.equality_violations)
    iterations_mean[i] = mean(storage.iterations)
    iterations_max[i] = maximum(storage.iterations)
end

solve_time_μs = 1e6 * mean(solve_time) / H0
mean(cone_product_violations_mean)
maximum(cone_product_violations_max)
mean(equality_violations_mean)
maximum(equality_violations_max)
mean(iterations_mean)
maximum(iterations_max)

x12 = [+0.05, +2.40, +0.1]
x22 = [-0.05, +1.80, -0.1]
x32 = [-0.00, +1.00, -0.2]
x42 = [-0.00, +0.30, -0.2]
v115 = 1.1 * (rand(3) .- 0.5)
v215 = 1.1 * (rand(3) .- 0.5)
v315 = 1.1 * (rand(3) .- 0.5)
v415 = 1.1 * (rand(3) .- 0.5)
z0 = [x12; v115; x22; v215; x32; v315; x42; v415]
H0 = 1000

storage, solve_time = simulate!(mech, z0, H0, violation=:absolute, timing=true)
vis, anim = visualize!(vis, mech, storage)
scatter(storage.iterations, color=:red)














mech = get_diverse_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    minkowski_radius=0.3,
    union_radius=0.15,
    segment=4.0,
    A = [[0 1; +2.5 -1; -2.5 -1], [5 1; -5 1; 1 5; 1 -5.0]],
    b = [0.3 * ones(3), 0.3 * ones(4)],
    # method_type=:symbolic,
    method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=false,
        differentiate=false,
        residual_tolerance=0.5e-6,
        complementarity_tolerance=0.5e-6,
        # compressed_search_direction=false,
        compressed_search_direction=true,
        # complementarity_backstep=1e-2,
        # sparse_solver=false,
        sparse_solver=true,
        warm_start=true,
        )
    );
mech.solver.solution.all

################################################################################
# test simulation
################################################################################
x1 = [+0.0,1.5,-0.20]
v1 = [-0,0,-1.0]
x2 = [-1.8,3.5,+0.75]
v2 = [-0,0,-0.0]
z0 = [x1; v1; x2; v2]
H0 = 1000

storage, solve_time = simulate!(mech, z0, H0, violation=:absolute, timing=true)
solve_time_μs = 1e6 * solve_time / H0
mean(storage.cone_product_violations)
maximum(storage.cone_product_violations)
mean(storage.equality_violations)
maximum(storage.equality_violations)
mean(storage.iterations)
maximum(storage.iterations)

################################################################################
# visualization
################################################################################
vis, anim = visualize!(vis, mech, storage)
scatter(storage.iterations)
















inertia = 0.8 * Matrix(Diagonal(ones(3)));
A=[
    +0 +0 +1;
    +0 +0 -1;
    +0 +1 +0;
    +0 -1 +0;
    +1 +0 +0;
    -1 +0 +0;
    +1 +1 +1;
    +1 +1 -1;
    +1 -1 +1;
    +1 -1 -1;
    -1 +1 +1;
    -1 +1 -1;
    -1 -1 +1;
    -1 -1 -1;
    ]
b=0.45*[ones(6); 1.5ones(8)]

mech = get_3d_polytope_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    A=A,
    b=b,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=false,
        differentiate=false,
        residual_tolerance=0.5e-6,
        complementarity_tolerance=0.5e-6,
        # compressed_search_direction=false,
        compressed_search_direction=true,
        # complementarity_backstep=1e-2,
        # sparse_solver=false,
        sparse_solver=true,
        warm_start=true,
        )
    )

################################################################################
# test simulation
################################################################################
qp2 = normalize([1,0,0,0.0])
qp2 = normalize([0,1,1,1.0])
xp2 =  [+0.00; +0.00; +1.00; qp2]
vp15 = [+2.00, -0.00, +0.00, +1.0,+1.0,+0.4]
z0 = [xp2; vp15]

H0 = 100

mech.solver.solution.primals[1:6] .= deepcopy(vp15)
mech.solver.solution.primals[7:9] .= zeros(3)

storage, solve_time = simulate!(mech, z0, H0, violation=:absolute, timing=true)
solve_time_μs = 1e6 * solve_time / H0
mean(storage.cone_product_violations)
maximum(storage.cone_product_violations)
mean(storage.equality_violations)
maximum(storage.equality_violations)
mean(storage.iterations)
maximum(storage.iterations)

################################################################################
# visualization
################################################################################
build_mechanism!(vis, mech)
set_mechanism!(vis, mech, storage, 1)

visualize!(vis, mech, storage, build=false)


scatter(mech.solver.data.residual.all)
scatter(mech.solver.data.residual.primals)
scatter(mech.solver.data.residual.duals)
scatter(mech.solver.data.residual.slacks)
scatter(storage.iterations)

mech.solver.dimensions.primals
