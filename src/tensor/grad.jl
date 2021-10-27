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

# Generate methods to calculate gradients for every order of arguments 
for (T, S) in ((Tensor, Union{AbstractArray,Number}),
                (Union{AbstractArray,Number}, Tensor),
                (Tensor, Tensor)) # Needed to fix ambiguity
    @eval begin
        function calculategrad!(::Type{BroadcastFunction{typeof(+)}}, 
                                t::$T, u::$S, res::Tensor)
            addgrad!(t, res.grad)
            addgrad!(u, res.grad)
            return nothing
        end

        function calculategrad!(::Type{BroadcastFunction{typeof(-)}},
                                t::$T, u::$S, res::Tensor)
            addgrad!(t, res.grad)
            addgrad!(u, -res.grad)
            return nothing
        end

        function calculategrad!(::Type{BroadcastFunction{typeof(*)}}, 
                                t::$T, u::$S, res::Tensor)   
            addgrad!(t, res.grad .* data(u))
            addgrad!(u, res.grad .* data(t))
            return nothing
        end
    end
end
