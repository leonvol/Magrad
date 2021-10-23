import Base.Broadcast: Broadcasted, ArrayStyle

# Custom BroadcastStlye for Tensors based on ArrayStyle broadcasting
Base.BroadcastStyle(::Type{<:Tensor}) = Broadcast.ArrayStyle{Tensor}()

# Actually method called on allocation of the broadcast result
# Gradient calculation is currently only supported for the last broadcasted op
# of a potentially larger computation graph
function Base.similar(bc::Broadcasted{ArrayStyle{Tensor}}, ::Type{ElType}) where {ElType} 
    operation = bc.f
    previous_nodes = bc.args
    Tensor(similar(zeros(bc.axes)), op=operation, prev=previous_nodes, broadcasted=true)
end
