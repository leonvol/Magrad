include("util.jl")

import Base.Broadcast: BroadcastFunction

# Function definition for every possible broadcast op involving a Tensor
# Needed to keep track of computation for backward pass
for (F, S) in ((Tensor, Tensor), (Tensor, Any), (Any, Tensor))
    @eval begin
        # Generic broadcast function for computations involving a Tensor
        function Broadcast.broadcasted(op::Any, x::$F, y::$S)
            Tensor(broadcast(op, data(x), data(y)), op=BroadcastFunction{typeof(op)}, prev=(x, y))
        end
    end
end