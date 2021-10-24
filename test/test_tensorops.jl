using Test

# Size of the Tensors used in the tests
size_ = (3, 3)

@testset "tensor ops" begin
        # Every op has to be tested for every possible following combination of datatypes
        # Tensor tensor
        # Tensor AbstractArray
        # AbstractArray Tensor
        # Tensor Number
        # Number Tensor
    @testset "ops" begin
        # Set does not test for Tensor Number as that is a per element op called by .+

        @testset "+ tensor op" begin
            ten = Tensor(zeros(size_))
            
            # Tensor Tensor
            a = ten + ten
            @test a == zeros(size_)

            arr = ones(size_)
            # Tensor AbstractArray
            d = arr + ten
            @test d == ones(size_)

            # AbstractArray Tensor 
            c = ten + arr
            @test c == ones(size_)
        end

        @testset "- tensor op" begin 
            ten = Tensor(ones(size_))
            
            # Tensor Tensor
            @test ten - ten == zeros(size_)

            arr = ones(size_)
            # Tensor AbstractArray
            @test arr + ten == 2ones(size_)

            # AbstractArray Tensor 
            @test ten + arr == 2ones(size_)
        end
        
        # matmul
        @testset "* tensor op" begin
            ten = Tensor(ones(size_))
            
            # Tensor Tensor
            @test ten * ten == 3ones(size_)

            arr = ones(size_)
            # Tensor AbstractArray
            @test arr * ten == 3ones(size_)

            # AbstractArray Tensor 
            @test ten * arr == 3ones(size_)
        end
    end

    # Tests broadcasting/per element calculation functionality
    @testset "broadcasted ops" begin 
        @testset "+ broadcasted tensor op" begin
            ten = Tensor(ones(size_)) 
            
            # tensor tensor
            # same size
            a = Tensor(ones(size_))
            res = ten .+ a

            @test res == [2 2 2; 2 2 2; 2 2 2]
            
            # smaller but compatible size
            b = Tensor(ones(size_[1], 1))
            res = ten .+ b
            @test res == [2 2 2; 2 2 2; 2 2 2]
            
            # to create bigger than original tensor
            # important to test broadcast allocation functionality
            c = Tensor(ones(size_[1], 1))
            d = Tensor(ones(1, size_[1]))
            res = d .+ c
            @test res == [2 2 2; 2 2 2; 2 2 2]
            
            # tensor array
            # same size
            e = ones(size_)
            res = ten .+ e
            @test res == [2 2 2; 2 2 2; 2 2 2]
            
            # with real need to broadcast
            f = ones(size_[1], 1)
            res = ten .+ e
            @test res == [2 2 2; 2 2 2; 2 2 2]

            # array tensor
            g = ones(size_)
            res = g .+ ten
            @test res == [2 2 2; 2 2 2; 2 2 2]
            
            # with real need to broadcast
            h = ones(size_[1], 1)
            res = h .+ ten
            @test res == [2 2 2; 2 2 2; 2 2 2]

            # tensor number
            @test ten .+ 1 == 2ones(size_)
            # number tensor
            @test 1 .+ ten == 2ones(size_)
        end

        @testset "- broadcasted tensor op" begin 
            ten = Tensor(ones(size_)) 
            
            # tensor tensor
            # same size
            a = Tensor(ones(size_))
            res = ten .- a

            @test res == zeros(size_)
            
            # smaller but compatible size
            b = Tensor(ones(size_[1], 1))
            res = ten .- b
            @test res == zeros(size_)
            
            # to create bigger than original tensor
            # important to test broadcast allocation functionality
            c = Tensor(ones(size_[1], 1))
            d = Tensor(ones(1, size_[1]))
            res = d .- c
            @test res == zeros(size_)
            
            # tensor array
            # same size
            e = ones(size_)
            res = ten .- e
            @test res == zeros(size_)
            
            # with real need to broadcast
            f = ones(size_[1], 1)
            res = ten .- e
            @test res == zeros(size_)

            # array tensor
            g = ones(size_)
            res = g .- ten
            @test res == zeros(size_)
            
            # with real need to broadcast
            h = ones(size_[1], 1)
            res = h .- ten
            @test res == zeros(size_)

            # tensor number
            @test ten .- 1 == zeros(size_)
            # number tensor
            @test 1 .- ten == zeros(size_)
        end

        @testset "* broadcasted tensor op" begin 
            ten = Tensor(ones(size_)) 
            
            # per element tensor tensor
            # same size
            a = Tensor(zeros(size_))
            res = ten .* a

            @test res == zeros(size_)
            
            # smaller but compatible size
            b = Tensor(zeros(size_[1], 1))
            res = ten .* b
            @test res == zeros(size_)
            
            # to create bigger than original tensor
            # important to test broadcast allocation functionality
            c = Tensor(zeros(size_[1], 1))
            d = Tensor(ones(1, size_[1]))
            res = d .* c
            @test res == zeros(size_)
            
            # tensor array
            # same size
            e = zeros(size_)
            res = ten .* e
            @test res == zeros(size_)
            
            # with real need to broadcast
            f = zeros(size_[1], 1)
            res = ten .* e
            @test res == zeros(size_)

            # array tensor
            g = zeros(size_)
            res = g .* ten
            @test res == zeros(size_)
            
            # with real need to broadcast
            h = zeros(size_[1], 1)
            res = h .* ten
            @test res == zeros(size_)

            # tensor number
            @test ten .* 3 == 3ones(size_)
            # number tensor
            @test 3 .* ten == 3ones(size_)
        end
    end
end

# Tests grad calculation functionality and correctness
# Very specific examples tested against PyTorch
@testset "backward" begin
    zero = zeros(3, 3)
    one = ones(3, 3)

    a = Tensor(one)
    b = Tensor(2one)
    c = Tensor(3one)
    d = Tensor(4one)
    res0 = a .* b
    res1 = c .* d
    res = (a .* b) .* (c .* d)

    backward(res)
    @test a.grad .- 24 == zero && b.grad .- 12 == zero && c.grad .- 8 == zero && d.grad .- 6 == zero

    
    a = Tensor(one)
    b = Tensor(zero)
    c = 3
    d = Tensor(zero)
    e = Tensor(one)
    f = 2
    g = Tensor(one)
    
    ab = a + b
    abc = ab .+ c
    de = d - e
    def_ = de .- f
    abcdef = abc .* def_
    abcdefg = abcdef * g
    
    backward(abcdefg)

    @test a.grad .+ 9 == zero && 
            b.grad .+ 9 == zero && 
            d.grad .- 12 == zero && 
            e.grad .+ 12 == zero && 
            g.grad .+ 36 == zero

    
    # bascially just matmul
    mat = [1 2 3; 4 5 6; 7 8 9]
    t0 = Tensor(mat)
    t1 = Tensor(mat)

    res = t0 * t1

    backward(res)
    #= 
    t0.grad 6.0  15.0  24.0
            6.0  15.0  24.0
            6.0  15.0  24.0
    t1.grad 12.0  12.0  12.0
            15.0  15.0  15.0
            18.0  18.0  18.0 
    =#
    @test t0.grad == [6 15 24; 6 15 24; 6 15 24] &&
            t1.grad == [12 12 12; 15 15 15; 18 18 18]

end