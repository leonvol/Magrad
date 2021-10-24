include("util.jl")

import Base: *

# No overloads for + and - methods
# Julia will automatically call the according per element op for addition/subtraction if dimensions are the same
# Otherwise a call to the dotted operation (.+) is needed which calls the broadcasted/per-element op  

# To fix ambiguity with *(AbstractArray, AbstractArray) method in LinearAlgebra
# Define specialized function for every AbstractArray subtype (and Tensor struct)
for S in (Tensor, AbstractArray, AbstractVector, AbstractMatrix)
@eval begin
        function *(t::Tensor, a::$S)
            Tensor(data(t) * data(a), op=typeof(*), prev=(t, a))
        end
    end
end
