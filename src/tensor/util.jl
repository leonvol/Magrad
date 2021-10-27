function addgrad!(t::Tensor, grad)
    t.grad .+= grad
end

function addgrad!(u::Union{AbstractArray,Number}, grad) end

# methods to get (underlying) data
@inline function data(t::Tensor)
    t.data
end

@inline function data(u::Union{AbstractArray,Number})
    u
end