# Tensor type
# Holds all the needed information for general calculations 
# and info needed for backprop 
mutable struct Tensor{T,N} <: AbstractArray{T,N}

    data::Array{T,N}
    
    req_grad::Bool
    grad::Array{Float32, N}

    # Attributes to keep track of computation
    # Operation by which the Tensor was created
    op
    # Temporarily needed to allow for correct broadcasted operation autograd 
    # Will be removed with introduction of gradient tape
    # Holds information whether this Tensor was produced by a broadcasted op
    broadcasted::Bool

    # Other elements (Tensors, Arrays) whose computation created this Tensor
    prev::Tuple
    
    function Tensor(data::Array{T,N}, 
                    req_grad::Bool=true; # Tensors should keep track of computation by default
                    grad::Array{Float64,N}=zeros(size(data)),
                    op=Type{}, 
                    prev=(),
                    broadcasted=false) where {T,N}
        new{T,N}(data, req_grad, grad, op, broadcasted, prev)
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

# wrapper to handle broadcast grad calculation logic
function calculategrad_wrapper(t::Tensor)
    if t.broadcasted
        calculategrad_broadcast(t.op, t.prev[1], t.prev[2], t)
        return
    end
    calculategrad(t.op, t.prev[1], t.prev[2], t)
end


function backward(t::Tensor, result::Bool=true)
    # end codition
    # no previous nodes, backward computation ends here
    if t.prev == ()
        return
    end
    
    # implicit grad creation
    if result
        t.grad = ones(size(t))
    end

    calculategrad_wrapper(t)
    
    # traverse computation graph in reverse order
    backward(t.prev[1], false)
    backward(t.prev[2], false)
end

# dispatch on type whose gradient should not/cannot be determined
function backward(t::Any, result::Bool) end