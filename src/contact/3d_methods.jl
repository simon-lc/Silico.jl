function quaternion_product(q, p)
    qw = q[1]
    qv = q[2:4]
    pw = p[1]
    pv = p[2:4]

    qp = [qw * pw - qv' * pv; qw .* pv + pw .* qv + cross(qv, pv)]
    return qp
end

function dagger(q)
    qd = [q[1]; -q[2:4]]
    return qd
end

function quaternion_difference(q0, q1)
    return quaternion_product(dagger(q0), q1)
end

function quaternion_ϕ(q0, q1)
    ϕ = quaternion_product(dagger(q0), q1)[2:4]
end

function quaternion_increment(q, ϕ)
    quaternion_product(q, [sqrt(1 - ϕ'*ϕ); ϕ])
end

function Lmat(q)
    [
        q[1] -q[2] -q[3] -q[4];
        q[2]  q[1] -q[4]  q[3];
        q[3]  q[4]  q[1] -q[2];
        q[4] -q[3]  q[2]  q[1];
    ]
end

function Rmat(q)
    [
        q[1] -q[2] -q[3] -q[4];
        q[2]  q[1]  q[4] -q[3];
        q[3] -q[4]  q[1]  q[2];
        q[4]  q[3] -q[2]  q[1];
    ]
end

# function Vmat()
#     [
#         0 1 0 0.0;
#         0 0 1 0.0;
#         0 0 0 1.0;
#     ]
# end

vector_rotate(q, x) = Vmat() * Lmat(q) * Rmat(q)' * Vmat()' * x

function tangential_plane(normal; tangent_candidate=[1,0,0.0])
    n = normal ./ (norm(normal) + 1e-10)
    tangent1 = tangent_candidate - tangent_candidate'*n * n
    tangent1 ./= norm(tangent1) + 1e-10
    tangent2 = cross(normal, tangent1)
    # rotation matrix from contact frame to world frame
    # n points towards the parent body, [t,y,n] forms an oriented vector basis
    wRb = [tangent1 tangent2 normal]
    return tangent1, tangent2, wRb
end

# q0 = normalize([1.0, 1, 0, 0])
# q1 = normalize([1.05, 1.0, 0, 0])
#
# quaternion_product(q0, q1)
# dagger(q0)
# quaternion_difference(q0, q1)
# ϕ01 = quaternion_ϕ(q0, q1)
# quaternion_increment(q0, ϕ01) - q1
