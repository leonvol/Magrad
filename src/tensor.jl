# Tensor type
# Holds all the needed information for general calculations 
# and info needed for backprop 
mutable struct Tensor{T,N} <: AbstractArray{T,N}

    data::Array{T,N}
    
    req_grad::Bool
    grad::Array{Float32, N}

    # Attributes to keep track of computation
    # Operation by which the Tensor was created
    op::Type
    # Other elements (Tensors, Arrays) whose computation created this Tensor
    prev::Tuple
    
    function Tensor(data::Array{T,N}, 
                    req_grad::Bool=true; # Tensors should keep track of computation by default
                    grad::Array{Float64,N}=zeros(size(data)),
                    op=Type{}, 
                    prev=()) where {T,N}
        new{T,N}(data, req_grad, grad, op, prev)
    end
end

# Tensor should behave like an AbstractArray with keeping track of computations and the ability to backpropagate
# implement AbstractArray interface
import Base: size, getindex, setindex!

Base.IndexStyle(::Tensor) = IndexLinear()

Base.size(tensor::Tensor) = size(tensor.data)

@inline function Base.getindex(tensor::Tensor, i::Int)
    getindex(tensor.data, i)
end

@inline function Base.setindex!(tensor::Tensor, v, i::Int)
    setindex!(tensor.data, v, i)
end

include("ops.jl")
include("grad.jl")
include("broadcast.jl")

function backward(t::Tensor, result::Bool=true)
    # End condition
    # No previous nodes, backward computation ends here
    if t.prev == ()
        return
    end
    
    # Implicit grad creation
    if result
        t.grad = ones(size(t))
    end

    calculategrad(t.op, t.prev[1], t.prev[2], t)
    
    # traverse computation graph in reverse order
    backward(t.prev[1], false)
    backward(t.prev[2], false)
end

# Gradient of this node should not/cannot be determined
function backward(t::Any, result::Bool) end