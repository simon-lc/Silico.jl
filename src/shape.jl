abstract type Shape1170{T} end

struct PolytopeShape1170{T}
    A::Matrix{T}
    b::Vector{T}
    o::Vector{T}
end

function PolytopeShape1170(A, b::Vector{T}, o=zeros(T,size(A,2))) where T
    return PolytopeShape1170{T}(A, b, o)
end

struct SphereShape1170{T}
    radius::Vector{T}
    position_offset::Vector{T}
end

function SphereShape1170(radius::T, position_offset=zeros(T,2)) where T
    return SphereShape1170{T}([radius], position_offset)
end
