include("util.jl") # data(), addgrad!()

# Gradient calculation methods for non broadcasted ops
# Disptach on op type
function calculategrad!(::Type{typeof(*)}, 
                        t::Tensor, u::Union{Tensor,AbstractArray}, res::Tensor)
    t.grad .+= res.grad * data(u)'
    addgrad!(u, t.data' * res.grad)
    return nothing
end


# Gradient calculation methods for broadcasted ops
# Dispatch on op type
import Base.Broadcast: BroadcastFunction

function calculategrad!(::Type{BroadcastFunction{typeof(+)}}, 
                        t::Tensor, u::Union{Tensor,AbstractArray,Number}, res::Tensor)
    t.grad .+= res.grad
    addgrad!(u, res.grad)
    return nothing
end

function calculategrad!(::Type{BroadcastFunction{typeof(-)}},
                        t::Tensor, u::Union{Tensor,AbstractArray,Number}, res::Tensor)
    t.grad .+= res.grad
    addgrad!(u, -res.grad)
    return nothing
end

function calculategrad!(::Type{BroadcastFunction{typeof(*)}}, 
                                t::Tensor, u::Union{Tensor,AbstractArray}, res::Tensor)   
    t.grad .+= res.grad .* data(u)
    addgrad!(u, res.grad .* t.data)
    return nothing
end
