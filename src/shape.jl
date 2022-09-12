abstract type Shape{T} end

struct PolytopeShape{T}
    A::Matrix{T}
    b::Vector{T}
    o::Vector{T}
end

function PolytopeShape(A, b::Vector{T}, o=zeros(T,size(A,2))) where T
    return PolytopeShape{T}(A, b, o)
end

struct SphereShape{T}
    radius::Vector{T}
    position_offset::Vector{T}
end

function SphereShape(radius::T, position_offset=zeros(T,2)) where T
    return SphereShape{T}([radius], position_offset)
end
