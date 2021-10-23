include("util.jl") # data(), addgrad()

# Gradient calculation methods for non broadcasted ops
# Disptach on op type
function calculategrad(::typeof(+), 
                        t::Tensor, u::Union{Tensor,AbstractArray,Number}, res::Tensor)
    t.grad .+= res.grad
    addgrad(u, res.grad)
end

function calculategrad(::typeof(-),
                        t::Tensor, u::Union{Tensor,AbstractArray,Number}, res::Tensor)
    t.grad .+= res.grad
    addgrad(u, -res.grad)
end

function calculategrad(::typeof(*), 
                        t::Tensor, u::Union{Tensor,AbstractArray}, res::Tensor)
    t.grad .+= res.grad * data(u)'
    addgrad(u, t.data' * res.grad)
end

function calculategrad(::typeof(*), t::Tensor, n::Number, res::Tensor)
    t.grad .+= res.grad * n
end

function calculategrad(op::Any, ::Any, ::Any, ::Any)
    throw(error("Grad calculation for $op not possible"))
end

# Gradient calculation methods for broadcasted ops
# Dispatch on op type
function calculategrad_broadcast(::typeof(+), 
                        t::Tensor, u::Union{Tensor,AbstractArray,Number}, res::Tensor)
    t.grad .+= res.grad
    addgrad(u, res.grad)
end

function calculategrad_broadcast(::typeof(-),
                        t::Tensor, u::Union{Tensor,AbstractArray,Number}, res::Tensor)
    t.grad .+= res.grad
    addgrad(u, -res.grad)
end

function calculategrad_broadcast(::typeof(*), 
                                t::Tensor, u::Union{Tensor,AbstractArray}, res::Tensor)   
    t.grad .+= res.grad .* data(u)
    addgrad(u, res.grad .* t.data)
end
