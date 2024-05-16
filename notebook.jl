### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 32fbc94f-710f-435f-91a8-13c07b9671f0
begin
	using LinearAlgebra
	using StaticArrays
	using SymPy
	using Plots
	using Images
	using PlutoUI
	using LazyGrids
	using BenchmarkTools
	using PartialFunctions
	using OffsetArrays
	using LatexPrint
	using ForwardDiff
	import ImageMagick
end

# ╔═╡ e8f7cad1-65a6-4814-9a06-5c90c87ec297
begin
	using AbstractTrees
	AbstractTrees.children(d::DataType) = subtypes(d)
	print_tree(Number)
end

# ╔═╡ 2289adfc-0bb4-4a07-9315-7ec870a0f0c6
log2(2) == log(ℯ) == log(16, 16) == asin(sin(1)) == 1

# ╔═╡ 54b587e9-566c-494a-839d-5f916911ba32
begin
	x, y, z = 1, 2, 3
	z = let z = 7, (x, y) = (y, z)  # closure
		x // y
	end
	x, y, z
end

# ╔═╡ eba461a3-2026-47c8-8a4a-78dbf8975a3c
let
	sq(x) = x ^ 2
	double(f) = x -> f(f(x))  # anonymous function
	map(double(sq), [3, "3"])
end

# ╔═╡ 132467b1-315f-40e1-b6fb-a56537d54f27
# begin
# 	# Task 2: Knapsack
# 	function knapsack(items::Vector{Pair{Int64, Int64}}, max_weight::Int64) :: Int64
# 		# your code here
# 		if length(items) == 0
# 			return 0
# 		else
# 			totval = knapsack(items[2:end], max_weight)  # exclude 1st item
# 			weight, value = first(items)
# 			if weight <= max_weight
# 				totval2 = value + knapsack(items[2:end], max_weight-weight)
# 				totval = max(totval, totval2)
# 			end
# 			return totval
# 		end
# 	end
# 	knapsack([3 => 4, 2 => 5, 4 => 8, 1 => 3, 3 => 2], 10)
# end

# ╔═╡ c7c67354-296f-4858-878d-81243ebfa32b
# begin
# 	Base.Pair(p::Bool, q::Bool) = !p || q  # operator overloading
# 	table = ["| p | q | p ⇒ q | ¬q ⇒ ¬p", "| --- | --- | --- | --- |"]
# 	for p in [true, false]
# 		for q in [true, false]
# 			push!(table, "| $p | $q | $(p => q) | $(!q => !p)")
# 			@assert (p => q) == (!q => !p)
# 		end
# 	end
# 	Markdown.parse(join(table, '\n'))
# end

# ╔═╡ 8cbd2f0f-cfd9-418a-9333-e49596c81c06
Base.adjoint(f::Function) = x -> ForwardDiff.derivative(f, x)

# ╔═╡ b700e215-3ba4-45e7-bea2-89f6f7ff73f7
# Task: implement multi-threaded version

# ╔═╡ 7e0fc6d8-b9af-4743-a634-bac7fdc6ff5d
# begin
# 	is_prime(n) = let m = Int(floor(sqrt(n)))
# 		all(gcd(n, k) == 1 for k ∈ 2:m)
# 	end
# 	P(n) = n^2 + n + 41
# 	findfirst(!is_prime ∘ P, 1:100)
# end

# ╔═╡ edf29ef0-272c-4067-8d6c-89aa8a217a3a
# function prime_sieve(n)
# 	mask = trues(n)
# 	mask[1] = false
# 	m = Int(floor(sqrt(n)))
# 	for i ∈ 2:m
# 		k = Int(n ÷ i)
# 		mask[2i:i:k*i] .= false
# 	end
# 	findall(mask)
# end

# ╔═╡ 03ca5540-a071-4139-9c6d-af42e036be6b
# prime_sieve(100)

# ╔═╡ 8e05724b-943c-43a2-8319-3815ed65e4ab
# @bind N Slider(10:20_000_000, show_value=true)

# ╔═╡ 1e8ebf68-08df-4426-9a14-5a2748023ce0
# let π(n) = length(prime_sieve(n))
# 	π(N), π(N) / (N / log(N))
# end

# ╔═╡ f98aefcb-58f4-4721-af1c-7171951362d9
# function plotJulia(c, xRng = [-1.25, 1.25], yRng = [-1.25, 1.25])
#     # Array of iteration counts
#     nx = 500
#     ny = 500
#     iterCounts = zeros(nx,ny)
#     # Locations 
#     xLocs = range(minimum(xRng),stop = maximum(xRng),length = nx)
#     yLocs = range(minimum(yRng),stop = maximum(yRng),length = ny)
#     # Loop over x,y locations iterating until maxIter reached, or |z|² > 4
#     maxIters = 200
#     for j = 1:nx
#         for k = 1:ny
#             z = xLocs[j] + yLocs[k]im
#             while iterCounts[j,k] < maxIters && norm(z) < 2
#                 z = z^2 + c
#                 iterCounts[j,k] += 1
#             end
#         end
#     end

#     # Display Julia set as a heatmap
#     heatmap(iterCounts', size=(1000, 800))
# end

# ╔═╡ 178afdd9-a049-4d91-bdd9-66cbde48fc7d
# @gif for θ=LinRange(0.88π, 0.89π, 200)
# 	plotJulia(0.6*(sin(θ)+im*cos(θ)))
# end fps=30

# ╔═╡ badf8efb-114b-459b-9de4-3cb6c2b3849a
# begin
# 	# define the Lorenz attractor
# 	Base.@kwdef mutable struct Lorenz
# 	    dt::Float64 = 0.02
# 	    σ::Float64 = 10
# 	    ρ::Float64 = 28
# 	    β::Float64 = 8/3
# 	    x::Float64 = 1
# 	    y::Float64 = 1
# 	    z::Float64 = 1
# 	end
	
# 	function step!(l::Lorenz)
# 	    dx = l.σ * (l.y - l.x)
# 	    dy = l.x * (l.ρ - l.z) - l.y
# 	    dz = l.x * l.y - l.β * l.z
# 	    l.x += l.dt * dx
# 	    l.y += l.dt * dy
# 	    l.z += l.dt * dz
# 	end
	
# 	attractor = Lorenz()
	
	
# 	# initialize a 3D plot with 1 empty series
# 	plt = plot3d(
# 	    1,
# 	    xlim = (-30, 30),
# 	    ylim = (-30, 30),
# 	    zlim = (0, 60),
# 	    title = "Lorenz Attractor",
# 	    legend = false,
# 	    marker = 2,
# 	)
	
# 	# build an animated gif by pushing new points to the plot, saving every 10th frame
# 	@gif for i=1:1500
# 	    step!(attractor)
# 	    push!(plt, attractor.x, attractor.y, attractor.z)
# 	end every 10
# end

# ╔═╡ 1806ebc4-ab49-4d29-821e-886784f3c2ed
[1 2 3
 5 6 4
 9 7 8]  # or [1 2 3; 5 6 4; 9 7 8]

# ╔═╡ 496aa019-b453-4dcb-9ebb-c01a1c5b14b3
zeros(3, 3)

# ╔═╡ d7bdb281-a363-476a-893b-4b7392f5c993
A = rand(Float64, (3, 4))

# ╔═╡ 2b7e66ad-731d-4074-867a-66c54fabfd71
let B = @show similar(A)
	fill!(B, 1)
end

# ╔═╡ 9c1c8163-71ba-4261-bda2-d39ff729606c
transpose(A)

# ╔═╡ bc704cc3-c5eb-4119-8ed8-85b596fcaa8e
[A[:, 3:4]; A[[1,3], 1:2:end]]  # concat vertically

# ╔═╡ 539c0dbe-9ec3-4ea4-80e7-7ee7fd9324d6
[sum(A .^ 2, dims=2) maximum(A, dims=2)]  # concat horizontally

# ╔═╡ 6d424f75-c63f-4f08-bee9-31b57756690c
diff(cumsum(A, dims=2), dims=2) ≈ A[:, 2:end]

# ╔═╡ d1a086a9-92d4-427e-a2d3-0946d39de9a2
let B = reshape(A, 2, 6)
	B[2, 3] = -999
	i = @show findfirst(A .== -999)
	C = @view B[1:2, 2:3]
	A[i] = -1
	C
end

# ╔═╡ b17c985c-3869-482b-8199-23b1500893fe
img = "https://images.fineartamerica.com/images-medium-large-5/1-earth-from-space-kevin-a-horganscience-photo-library.jpg" |> download |> load

# ╔═╡ f0edaa1c-cd6d-4722-9d3f-0f6328095c48
typeof(img)

# ╔═╡ 6b313419-288a-4f20-8453-e61fa463d225
SVD_results = [svd(f.(img)) for f in [red, green, blue]];

# ╔═╡ 4149ce52-2e86-4c6b-92ca-d66fd9b8348d
@bind K Slider(1:60, show_value=true, default=30)

# ╔═╡ 023ccc34-42aa-4bbe-a127-3fdc48eec87a
subtypes(Integer), supertypes(Integer)

# ╔═╡ ee0e9437-428f-420c-8733-7b0a00a94670
# Task: solve the integer partition problem with @memoize

# ╔═╡ c85e44f0-600a-4491-ba95-bf5b38d83578
begin
	"""`D` dimensional `F` (real/complex) vector space of vector type `T`."""
	abstract type VectorSpace{T, D, F<:Number} end

	basis(s::VectorSpace) = s.basis
	basis(s::VectorSpace, i::Integer) = basis(s)[i]
	ndim(s::VectorSpace{<:Any,D,<:Number}) where {D} = D

	struct Vect{D, F, V<:VectorSpace{<:Any,D,F}}
		coefs::SVector{D,<:F}
		space::V

		function Vect(x::AbstractVector, s::VectorSpace{T,D,F}) where {T,D,F}
			new{D,F,typeof(s)}(SVector{D,F}(x), s)
		end
	end

	Base.size(v::Vect) = size(v.coefs)
	Base.length(v::Vect) = length(v.coefs)
	value(v::Vect) = sum(v.coefs .* basis(v.space))

	abstract type InnerProdSpace{T, D, F} <: VectorSpace{T, D, F} end

	struct EuclideanSpace{D, F} <: InnerProdSpace{SVector{D,F}, D, F}
		basis::SMatrix{D,D,F}

		function EuclideanSpace{D,F}(basis=I) where {D,F}
			new{D,F}(SMatrix{D,D,F}(basis))
		end
		
		function EuclideanSpace(s::InnerProdSpace{T,D,F}) where {T,D,F}
			new{D,F}([u ⋅ v for u in basis(s), v in basis(s)])
		end
	end

	basis(s::EuclideanSpace) = eachcol(s.basis)
	value(v::Vect{<:Any,<:Number,<:EuclideanSpace}) = v.space.basis * v.coefs

	const ℝ{N} = EuclideanSpace{N, Float64}
	const ℂ{N} = EuclideanSpace{N, ComplexF64}

	function space_type(x::AbstractArray{T,N}) where {T<:Number,N}
		T <: Real ? ℝ{size(x, 1)} : ℂ{size(x, 1)}
	end

	Vect(coefs::AbstractVector; basis=I) = Vect(coefs, space_type(coefs)(basis))
	vecspace(basis::AbstractMatrix) = space_type(basis)(basis)

	function Base.getindex(s::VectorSpace, x::T...) where {T}
		Vect(T <: AbstractVector && length(x) == 1 ? x[1] : collect(x), s)
	end
	# abstract type LinMap{VectorSpace, VectorSpace} end

	# struct Operator{N,F} <: LinMap{EuclideanSpace{N,F}, EuclideanSpace{N,F}}
	# 	mat::SMatrix{N,N,F}

	# 	Operator(s::EuclideanSpace{N,F}, t::EuclideanSpace{N,F}) where {N,F} =
	# 		new{N,F}(t.basis * s.basis ^ -1)
	
	# 	(L::Operator)(s::T) where {T<:EuclideanSpace} = T(L.mat * s.basis)
	# end

	# matrix(L::LinMap) = L.mat

	# abstract type Functional{V<:VectorSpace, F<:Field} end
	
	function Base.:+(u::Vect{D,F,V}, v::Vect{D,F,V}) where {D,F,V}
		if u.space === v.space
			Vect(u.coefs .+ v.coefs, u.space)
		else
			u + u.space \ v  # project v into the space of u
		end
	end

	Base.:-(v::Vect) = Vect(-v.coefs, v.space)
	Base.:*(a::Number, v::Vect) = Vect(a.*v.coefs, v.space)
	Base.:*(x::Vect, y::Number) = y * x
	Base.:-(x, y) = x + (-y)
	Base.:/(x, y::Number) = (1/y) * x

	# Base.:*(x::Field, L::T) where {T<:LinMap} = T(x * matrix(L))
	# Base.:+(F::T, G::T) where {T<:LinMap} = T(matrix(F) + matrix(G))

	function Base.:\(s::InnerProdSpace{T,D,F}, x::T) where {T,D,F}
		B = orthogonalize(s)  # make an orthonormal basis
		b = [x ⋅ e for e in B]
		A = [b ⋅ e for e in B, b in basis(s)]
		Vect(A \ b, s)
	end

	Base.:\(s::InnerProdSpace, v::Vect) = s \ value(v)
	Base.:\(s::EuclideanSpace, v::Vector) = Vect(s.basis \ v, s)

	⋅(u::Vect, v::Vect) = sum(a * (u ⋅ x) for (a, x) in zip(v.coefs, basis(v.space)))
	⋅(u::Vect, x) = sum(a * (x ⋅ y) for (a, y) in zip(u.coefs, basis(u.space)))
	⋅(x, u::Vect) = u ⋅ x
	⋅(x::AbstractVector, y::AbstractVector) = sum(a * b for (a, b) in zip(x, y))

	norm(v::Union{AbstractVector, Vect}) = sqrt(v ⋅ v)
	proj(u, v) = (u ⋅ v) / (v ⋅ v) * v

	function orthogonalize(s::V) where {T,D,F,V<:InnerProdSpace{T,D,F}}
		new_basis = [basis(s, 1)]
		for i in 2:ndim(s)
			u = basis(s, i)
			push!(new_basis, u - sum(proj(u, v) for v in new_basis))
		end
		new_basis ./ norm.(new_basis)
	end
end

# ╔═╡ 6db134a0-0962-11ef-310f-99600cd50dfb
[3 + 4, 3 * 4, 3 / 4, 3 \ 4, 3 ÷ 4, 3 ^ 4]

# ╔═╡ 6d3c390d-7e2c-401f-b313-31975874e657
let
	expi(x) = exp(im * x)
	z = @show expi(π)
	z ≈ -1
end

# ╔═╡ a94f9691-3906-43cd-bf0e-55003ecbb1fc
let
	fact(n) = n < 0 ? error("negative input") : n < 2 ? big(1) : n * fact(n-1)
	# use big() to prevent overflow
	fact.(1:30)
end

# ╔═╡ a4e6194b-5cfd-4968-8d22-6d7ed4e15523
begin
	sum_numbers(s::String) = sum(length(m) > 0 ? parse(Int, m) : 0
								 for m in split(s, r"[^\d]+"))
	fruits = "5 apples, 12 oranges, 4 bananas"
	println(sum_numbers(fruits), " fruits in total!")
end

# ╔═╡ dddf76d2-dd8e-4c17-9cc0-256e20db0abe
sin'(0), cos'(π/2)

# ╔═╡ db5d41fc-ea93-4abc-9efa-b232ef7f37e2
function estimate_pi_mc(n=100_000_000)
	count(1:n) do _
		rand()^2 + rand()^2 < 1
	end / n * 4
end

# ╔═╡ fe402f42-0d4a-46fe-ad6d-57eb578d2cf4
begin
	function newton_method_iter(f, x, max_iter=10000, tol=1e-8)
		Channel() do ch
			for _ in 1:max_iter
				dx = f(x) / f'(x)
				x -= dx
				push!(ch, x)
				if abs(dx) < tol
					return x
				end
			end
			@warn "root not found after $max_iter iterations"
		end
	end
	Base.last(c::Channel) = foldl((_, x) -> x, c)
	@time last(newton_method_iter(sin, 2))
end

# ╔═╡ 9a379335-9430-4fa2-9b84-15e192ace090
@time estimate_pi_mc(400_000_000)

# ╔═╡ 697828f5-2e3b-4c25-86e0-cc97cfdb8432
@time let N = 400_000_000, k = Threads.nthreads()
	sum(fetch.(Threads.@spawn estimate_pi_mc(N÷k) for _ in 1:k)) / k
end

# ╔═╡ 175d7472-4a23-4715-a39e-4a434cac46b1
@time let N = 100_000_000
	# generator (laze evaluation)
	fracs = ((i ÷ 2 * 2) / ((i-1) ÷ 2 * 2 + 1) for i in 2:N)
	# fracs = [(i ÷ 2 * 2) / ((i-1) ÷ 2 * 2 + 1) for i in 2:N]
	2 * prod(fracs)
end

# ╔═╡ 37396c37-0291-4ce6-92c6-defb547f94f5
@time foldr((f, x) -> f(x), Iterators.repeated(x -> sin(x) + x, 10), init=1)

# ╔═╡ cd635cc3-c4fc-4da7-9f75-960a2aa118f4
begin
	myfun(x) = sin(3x + 4cos(2x))
	let x = range(0, 2π, 1000)
		plot(x, myfun.(x), label="f(x)", title="My Function Plot")
		plot!(x, myfun'.(x), label="f'(x)")
	end
end

# ╔═╡ cc982bfb-8147-4e69-be5c-c75c053e8e33
let f = myfun, x = range(-2, 2, 1000)
	y = f.(x)
	x0 = 0
	# x0 = -1
	@gif for x1 in newton_method_iter(f, x0, 30)
		plot(x, y, label="f(x)", title="Newton's method")
		plot!([x0, x1], [f(x0), 0], label="x₀", markershape=:circle)
		plot!([x1, x1], [0, f(x1)], label=nothing, linestyle=:dash)
		annotate!(x0, f(x0), ("x = $(Float16(x0))", 8, :top))
		x0 = x1
	end fps=2
end

# ╔═╡ a29da296-cbde-4004-8022-ef5888509b38
[i + j*im for i in 1:3, j in 1:3]

# ╔═╡ 95803efa-1141-42dd-8bcb-3778841d423f
size(A), size(A, 1)

# ╔═╡ f31efc51-cf0c-4093-af7a-b4e3a2e84b2f
M = A * A'  # adjoint

# ╔═╡ 91d47e62-3207-48ec-a255-c1f1ece56d5b
M ^ 2

# ╔═╡ d1d6309d-ccf8-4799-b09b-8405d3f8c302
exp(M)

# ╔═╡ 0d33e716-36c0-4a74-b30a-7ee19aeb26d3
rank(M), tr(M), det(M), diag(M)

# ╔═╡ 68617bc1-8d21-4df7-8e1b-6751059e732e
# Task: implement mean
function mean(A::Array, dims::Integer...)
	if length(dims) == 0
		return sum(A) / length(A)
	end
	for i in sort(collect(dims), rev=true)
		A = sum(A, dims=i) ./ size(A, i)
	end
	return A
end

# ╔═╡ 9e1cdf98-5b5d-4b46-a779-560b6f552d3c
let b = [3, 2, 1]
	x = @show M \ b  # inv(M) * b
	M * x
end

# ╔═╡ a0ef34ff-bb06-43e8-acd4-ee228a6edc68
let eig = eigen(M)
	@show eig.values
	@show eig.vectors
	λ, V = eig
	M * V ≈ λ' .* V
end

# ╔═╡ 672ee9e6-1928-4eab-b2d9-b34179cf99fc
begin
	data = map(SVD_results) do (U, Σ, V)
		U_K = U[:, 1:K]
		Σ_K = Diagonal(Σ[1:K])
		V_K = V[:, 1:K]
		U_K * Σ_K * V_K'
	end
	hcat(img, RGB.(data...))
end

# ╔═╡ 8515d621-6221-4711-a486-0b9c3d3f6be4
function transform_image(img::AbstractMatrix{<:RGB}, basis::Matrix{<:Real})
	M, N = size(img)
	A = OffsetMatrix(img, -M÷2, -N÷2)
	I, J = convert.(UnitRange, axes(A))
	sx, sy = svd(basis).S
	P = Iterators.product((I.start*sy):(I.stop*sy), (J.start*sx):(J.stop*sx))
	Idx = Int32.(floor.(inv(basis) * reshape(collect(Iterators.flatten(P)), 2, :)))
	blank = RGB(0, 0, 0)
	color(i, j) = i in I && j in J ? A[i,j] : blank
	C = color.(eachrow(Idx)...)
	reshape(C, length.(P.iterators)...)
end

# ╔═╡ b0a5af7c-235f-47c1-b3c8-a3f635ef53f1
begin
	mutable struct BST  # Binary Search Tree
		key; val
		left; right

		function BST(items::Pair...; root=true)
			if length(items) == 0
				root ? new(nothing, nothing, nothing, nothing) : nothing
			else
				items = collect(items) |> sort!
				i = length(items) ÷ 2 + 1
				l = BST(items[1:i-1]..., root=false)
				r = BST(items[i+1:end]..., root=false)
				new(items[i]..., l, r)
			end
		end
	end

	AbstractTrees.children(t::BST) =
		t.left == t.right == nothing ? [] : [t.left, t.right]

	Base.show(io::IO, t::BST) = 
		print_tree((io, t) -> print(io, t == nothing ? t : t.key => t.val), io, t)
end

# ╔═╡ c92826c1-61f0-44e6-b333-c60deae80f68
t = BST(1=>2, 2=>3, 0=>1, 5=>6, 4=>5)

# ╔═╡ d7f3b4a4-99db-4a6d-a744-1b9e5a3f9360
begin
	function fields(x::T) where {T}
		Dict(k => getfield(x, k) for k in fieldnames(T))
	end

	fields(t)
end

# ╔═╡ 0a58b720-ea82-48e2-aa5c-ab0e9c530b41
begin
	t[3] = 4
	t[4] = 99
	t[6] = 7
	t
end

# ╔═╡ f2fce8ac-7800-4153-b0fa-21168300984e
begin
	function Base.getindex(t::BST, key)
		if t.key == nothing  # empty tree
			throw(KeyError("key $key not found"))
		end
		if t.key == key
			return t.val
		end

		c = key < t.key ? t.left : t.right

		if c == nothing
			throw(KeyError("key $key not found"))
		else
			return c[key]
		end
	end

	function Base.setindex!(t::BST, val, key)
		if t.key == nothing  # empty tree
			t.key = key
		end
		if t.key == key
			return t.val = val
		end

		s = key < t.key ? :left : :right
		c = getfield(t, s)

		if c == nothing
			return setfield!(t, s, BST(key => val))
		else
			return c[key] = val
		end
	end

	# function get!(f, t::BST{K,V}, key) :: V where {K,V}
	# 	if t.key == nothing  # empty tree
	# 		t.key = key
	# 		t.val = f()
	# 	end

	# 	if key == t.key
	# 		return t.val
	# 	end

	# 	d = key < t.key ? :left : :right
	# 	c = getfield(t, d)  # child node

	# 	if c == nothing
	# 		c = BST{K,V}(key => f())
	# 		setfield!(t, d, c)
	# 		return c.val
	# 	else
	# 		return get!(f, c, key)
	# 	end
	# end
end

# ╔═╡ 0e8fcc58-1f93-48ef-9c74-f5abdcdecbd3
md"""
# Basic Calculation
"""

# ╔═╡ ffdb33f9-0bd8-4795-a4b3-aee5c5620fb7
md"# Variables and Functions"

# ╔═╡ 55d77b19-e600-4ce8-9dc1-8d2458c99da2
md"Estimate π using Monte Carlo"

# ╔═╡ e47e0b0a-377a-451e-93f8-d4430706ef19
md"""Estimate π using

$$π = \frac{2\cdot2\cdot4\cdot4\cdot6\cdot6\ldots}
		   {1\cdot3\cdot3\cdot5\cdot5\cdot7\ldots}$$
"""

# ╔═╡ 7f0583b7-8d5c-4660-ad47-5f64aa6f57bb
md"""Estimate π using Newton's method:

``\pi`` is a root of ``sin(x)``.
"""

# ╔═╡ 36305f0a-a02d-4ba0-8a4b-12a645a6cf39
md"Estimate π using fixed point iteration:

``f(x), f(f(x)), \ldots`` converges to a fixed point ``x_0`` of ``f`` , i.e. ``f(x_0) = x_0``. 

``\pi`` is a fixed point of ``sin(x) + x``, since ``sin(\pi) + \pi = \pi``."

# ╔═╡ 551603af-23fc-49b6-a3c0-f15c199452dc
md"""
# Linear Algebra
"""

# ╔═╡ aae002c8-7160-40f2-a01b-dc1192dbd6d2
@bind T PlutoUI.combine() do Child
	θ = Child("θ", Slider(0:5:360, show_value=true))
	ϕ = Child("ϕ", Slider(0:5:360, show_value=true))
	x = Child("x", Slider(0.1:0.02:1, show_value=true, default=0.5))
	y = Child("y", Slider(0.1:0.02:1, show_value=true, default=0.5))
	md"""
	1. rotation: $θ
	1. horizontal scale: $x
	1. vertical scale: $y
	1. rotation: $ϕ
	"""
end

# ╔═╡ 8a565363-d3bb-423d-ab43-642f781925e4
begin
	rotate(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
	scaley(a) = [a 0; 0 1]
	scalex(a) = [1 0; 0 a]

	trans = rotate(-T.ϕ * π/180) * scalex(T.x) * scaley(T.y) * rotate(T.θ * π/180)
	transform_image(img, trans) #|> size
end

# ╔═╡ a1eaec8f-3527-43f6-b3e6-360c64b156e5
md"""# Type System"""

# ╔═╡ fb7b8d5e-0a75-4139-a1fd-e8f5f3b7db98
begin
	using MacroTools: combinedef, splitarg, splitdef
	
	macro memoize(expr)
		# parse the function definition into a Dict
	    def_dict = splitdef(expr)
		fname = def_dict[:name]  # name of the function
		args = map(splitarg, def_dict[:args])  # parse function arguments
	    vars = Tuple(name for (name, typ, is_splat, default) in args)

		# rename the original function definition
	    def_dict_unmemoized = copy(def_dict)
	    f = def_dict_unmemoized[:name] = Symbol("#", fname, "_orig")

		# initialize the cache and define it as a global const variable
	    cache = BST()
	    fcachename = Symbol("#", fname, "_cache")
		mod = __module__
		fcache = isdefined(mod, fcachename) ?
             getfield(mod, fcachename) :
             Core.eval(mod, :(const $fcachename = $cache))
	
		# modify the function definition to make it use caching
		def_dict[:body] = quote
			try
				$fcache[$(vars...)]
			catch e
				if e isa KeyError
					$fcache[$(vars...)] = $f($(vars...))
				else
					throw(e)
				end
			end
	    end

		# return the modified Julia expression
	    esc(quote
	        $(combinedef(def_dict_unmemoized))  # define the original function
	        $(combinedef(def_dict))  # define the cached function
	    end)
	end
end

# ╔═╡ 00ffff53-e77e-46e3-be19-672cb1b87716
let
	@memoize mfib(n) = n < 2 ? big(n) : mfib(n-2) + mfib(n-1)
	fib(n) = n < 2 ? big(n) : fib(n-2) + fib(n-1)
	@time @show mfib(32)
	@time @show fib(32)
end;

# ╔═╡ 6fda8490-fdab-43a1-84c7-217d7bcf12d2
begin
	u = Vect([-3, 4])
	v = Vect([2, -1], basis=[3 2; 1 0])
	2u + v
end

# ╔═╡ 5f07854d-0d4c-4de6-8d4a-4b319f372bc4
# let L = Operator(u.space, v.space)
# 	@assert L(u.space) == (v.space \ u).coefs
# end

# ╔═╡ 0b6a3d60-a8b5-4439-bd29-046a8a4b488e
let
	m = hcat(orthogonalize(v.space)...)
	w = Vect([3, -4], basis=m)
	@show norm(w)
end

# ╔═╡ e96c765f-5f5f-41db-8191-1a0d444bcde2
# begin
# 	struct Func{L, H}
# 		exp::Sym
# 		var::Sym
# 	end

# 	struct FuncSpace{D, F, L, H} <: InnerProdSpace{Sym, D, F}
# 		basis::SVector{D,Sym}
# 		var::Sym

# 		function FuncSpace(basis::Vector{<:Sym}, (lo, hi), field=Real)
# 			(var,) = free_symbols(basis)
# 			new{length(basis),field,lo,hi}(basis, var)
# 		end
	
# 		function FuncSpace(basis::Vector{Func{L,H}}, field=Real) where {L,H}
# 			(var,) = Set(f.var for f in basis)
# 			new{length(basis),field,L,H}([f.exp for f in basis], var)
# 		end
# 	end

# 	basis(s::FuncSpace{D,F,L,H}) where {D,F,L,H} = Func{L,H}.(s.basis, s.var)
# 	value(v::Vect{D,F,<:FuncSpace}) where {D,F} = sum(v.coefs .* basis(v.space)).exp
# 	(v::Vect{D,F,FuncSpace})(x) where {D,F} = value(v)(x)

# 	function PolySpace(D, range=(0, 1), field=Real, var=:x)
# 		basis = symbols(var, real=true) .^ (0:(D-1))
# 		FuncSpace(basis, range, field)
# 	end

# 	function FourierSpace(D, range=(0, 2π), field=Real, var=:x)
# 		a, b = range
# 		T = b - a
# 		t = symbols(var, real=true) - a
# 		basis = [exp(-im*2π*k*t/T) for k = 0:(D-1)]
# 		basis = field <: Real ? real.(basis) : basis
# 		FuncSpace(basis, range, field)
# 	end

# 	⋅(f::Sym, g::Func{L,H}) where {L,H} = let h = g.exp
# 		integrate(real(f * conj(h)), (g.var, L, H))
# 	end
# 	⋅(f::T, g::T) where {L,H,T<:Func{L,H}} = f ⋅ g.exp

# 	Base.:\(s::FuncSpace, v::Function) = s \ v(symbols(:x, real=true))
# 	Base.:*(a, f::T) where {T<:Func} = T(a * f.exp, f.var)
# 	Base.:+(f::T, g::T) where {T<:Func} = T(f.exp + g.exp, f.var)
# 	Base.:-(f::T) where {T<:Func} = T(-f.exp, f.var)
# end

# ╔═╡ 331ceecb-53e4-41f9-b8de-2aece6d1eea8
# begin
# 	p1 = PolySpace(5, (-1, 1))
# 	v1 = p1[1,1,1,1,1]
# 	s1 = FuncSpace(gram_schmidt(v1.space))
# 	u1 = s1 \ v1
# end

# ╔═╡ 25319673-99ff-436d-bf5d-34c540f4faf1
# begin
# 	p2 = FourierSpace(5, (-1, 1))
# 	v2 = p2[1:5]
# 	s2 = FuncSpace(gram_schmidt(v2.space))
# 	u2 = s2 \ v2
# end

# ╔═╡ 752d42d4-bc56-4b0d-a457-c4dbf32b27f8
function compare_plots(f, g, xvalues, title="", d=0.001)
	rng = xvalues[1]:d:xvalues[end]
	p = plot(f, rng, label="function", legend = :outertopright, title=title)
	plot!(g, rng, label="polynomial")
	scatter!(xvalues, zeros(length(xvalues)), label=false, color=:grey, markersize=1)
	return p
end

# ╔═╡ 2a5009df-5e84-4b50-b819-c19333eb4a8b
let
	g = value(PolySpace(8, (-2, 2)) \ myfun)
	println(g)
	compare_plots(f, g, -2.2:0.1:2.2)
end

# ╔═╡ a116db18-a4c5-4661-aa0b-5098114d201e
# let
# 	g = value(FourierSpace(4, (-2, 2)) \ myfun)
# 	println(g)
# 	compare_plots(f, g, -2.2:0.1:2.2)
# end

# ╔═╡ 4abe4232-ca33-4d07-b20a-de2f0dc68591
typeof(Differential(Sym(:x), 2))

# ╔═╡ 2eded938-c85c-4bf4-95aa-8d8b02c6cbbb
begin
	struct DiffEqSystem{F, D} <: VectorSpace{Vector{Differential}, D, F}
		coefs::Matrix{Sym}
		var::Sym

		function DiffEqSystem{F}(coefs::Matrix{<:Sym}) where {F}
			new{F,size(coefs,1)}(coefs, free_symbols(vec(coefs))...)
		end
	end
	
	basis(s::DiffEqSystem{F,D}) where {F,D} = Differential.(s.var, 0:(D+1))
end

# ╔═╡ 23123586-648e-4187-829f-aba7f4114bd5
# let
# 	@syms x
# 	eqs = DiffEqSystem{Complex}([x 1 x 3; x+1 x 3 x])
# 	basis(eqs)
# end

# ╔═╡ 4671e428-7d3e-4ea4-9480-fd7bae3d28f6
md"# Data Science"

# ╔═╡ 74f3e3b6-20ee-4aa4-ac24-701cedb8196e
# begin
# 	using ForwardDiff
# 	affine(X, W, b) = X * W .+ b
# 	ndims = [6, 8, 8, 1]
# 	params = (W=rand(6, 4), b=rand(4)')
# 	let X = rand(10, 6), y = rand(0:1, 6)
# 		ForwardDiff.gradient(W -> sum(affine(X, W, params.b)), params.W)
# 	# 	p = forward(X, layers)
# 	# 	p = exp.(-p) ./ (1 .+ exp.(-p))
# 	# 	loss = sum(p)
# 	end
# end

# ╔═╡ 9e4e4f8d-1d34-4c35-b814-b8d6708bf2ab
macro show_all(block)
	if block.head == Symbol("let")
		lines = block.args[2].args
		foreach(enumerate(lines)) do (i, ex)
			if isa(ex, Union{Symbol, Expr})
				lines[i] = :(@show $ex)
			end
		end
	end
	block
end

# ╔═╡ 4532b560-a6b2-4364-a548-ffb5d25b0d5c
@show_all let a = -6:3:6, b = [1, 2, 3], c = Iterators.cycle([0, 1])
	a .* 2
	reverse(a)
	a[1:3:end]
	max(a...)  # maximum(a)
	abs.(a) |> sum
	count(iseven, a)
	[0.1x for x in a if 0 <= x < 6]
	b'  # adjoint(b)
	size(b')
	vcat(a[end:-2:1], b)
	zip(b, c) |> collect
	map(+, a, b, c)
	foldl(-, b)  # 1 - 2 - 3 (starting from the left)
end;

# ╔═╡ 90808947-4a59-4903-8912-5e43839f91e0
@show_all let s = "Hello world!"
	reverse(s)
	join(split(s, 'l'), '1')
	"$(s[1:5]) Julia " * "😃" ^ 3
	prod(filter(islowercase, s))
	replace(s, r"([aeiou])" => s"\1\1")  # regex
end;

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractTrees = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
ImageMagick = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LatexPrint = "d2208f48-c256-5759-9089-c25ed2a93924"
LazyGrids = "7031d0ef-c40d-4431-b2f8-61a8d2f650db"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MacroTools = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[compat]
AbstractTrees = "~0.4.5"
BenchmarkTools = "~1.5.0"
ForwardDiff = "~0.10.36"
ImageMagick = "~1.3.1"
Images = "~0.26.1"
LatexPrint = "~1.1.0"
LazyGrids = "~0.5.0"
MacroTools = "~0.5.13"
OffsetArrays = "~1.14.0"
PartialFunctions = "~1.2.0"
Plots = "~1.40.4"
PlutoUI = "~0.7.59"
StaticArrays = "~1.9.3"
SymPy = "~2.0.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.3"
manifest_format = "2.0"
project_hash = "58f97e1420ee7e36d71a6d377028c9f2093aa653"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "133a240faec6e074e07c31ee75619c90544179cf"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.10.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "601f7e7b3d36f18790e2caf83a882d88e9b71ff1"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a4c43f59baa34011e303e76f5c8c91bf58415aaf"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "575cd02e080939a33b6df6c5853d14924c08e35b"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.23.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonEq]]
git-tree-sha1 = "6b0f0354b8eb954cdba708fb262ef00ee7274468"
uuid = "3709ef60-1bee-4518-9f2f-acd86f176c50"
version = "0.2.1"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "6cbbd4d241d7e6579ab354737f4dd95ca43946e1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "51cab8e982c5b598eea9c8ceaced4b58d9dd37c9"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "f9d7112bfff8a19a3a4ea4e03a8e6a91fe8456bf"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8e2d86e06ceb4580110d9e716be26658effc5bfd"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "da121cbdc95b065da07fbb93638367737969693f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.8+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "359a1ba2e320790ddbe4ee8b4d54a305c0ea2aff"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.0+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "3863330da5466410782f2bffc64f3d505a6a8334"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.10.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "2c3ec1f90bb4a8f7beafb0cffea8a4c3f4e636ab"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.6"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HistogramThresholding]]
deps = ["ImageBase", "LinearAlgebra", "MappedArrays"]
git-tree-sha1 = "7194dfbb2f8d945abdaf68fa9480a965d6661e69"
uuid = "2c695a8d-9458-5d45-9878-1b8a99cf7853"
version = "0.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageBinarization]]
deps = ["HistogramThresholding", "ImageCore", "LinearAlgebra", "Polynomials", "Reexport", "Statistics"]
git-tree-sha1 = "f5356e7203c4a9954962e3757c08033f2efe578a"
uuid = "cbc4b850-ae4b-5111-9e64-df94c024a13d"
version = "0.3.0"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageCorners]]
deps = ["ImageCore", "ImageFiltering", "PrecompileTools", "StaticArrays", "StatsBase"]
git-tree-sha1 = "24c52de051293745a9bad7d73497708954562b79"
uuid = "89d5987c-236e-4e32-acd0-25bd6bd87b70"
version = "0.1.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "432ae2b430a18c58eb7eca9ef8d0f2db90bc749c"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.8"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "8e2eae13d144d545ef829324f1f0a5a4fe4340f3"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.3.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "8d2e786fd090199a91ecbf4a66d03aedd0fb24d4"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.11+4"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "3ff0ca203501c3eedde3c6fa7fd76b703c336b5f"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.2"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e0884bdf01bbbb111aea77c348368a86fb4b5ab6"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.1"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "12fdd617c7fe25dc4a6cc804d657cc4b2230302b"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.1"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be50fe8df3acbffa0274a744f1a99d29c45a57f4"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "PrecompileTools", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "5ea6acdd53a51d897672edb694e3cc2912f3f8a7"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.46"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3336abae9a713d2210bb57ab484b1e065edd7d23"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "70c5da094887fd2cae843b8db33920bac4b6f07d"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LatexPrint]]
deps = ["Requires"]
git-tree-sha1 = "67019e47920747446c28c9ff4eadef0306c14031"
uuid = "d2208f48-c256-5759-9089-c25ed2a93924"
version = "1.1.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "e0b5cd21dc1b44ec6e64f351976f961e6f31d6c4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.3"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "62edfee3211981241b57ff1cedf4d74d79519277"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyGrids]]
deps = ["Statistics"]
git-tree-sha1 = "f43d10fea7e448a60e92976bbd8bfbca7a6e5d09"
uuid = "7031d0ef-c40d-4431-b2f8-61a8d2f650db"
version = "0.5.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4b683b19157282f50bfd5dcaa2efe5295814ea22"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27fd5cc10be85658cacfe11bb81bee216af13eda"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg"]
git-tree-sha1 = "110897e7db2d6836be22c18bffd9422218ee6284"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.12.0+0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "8f6786d8b2b3248d79db3ad359ce95382d5a6df8"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.170"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "80b2833b56d466b3858d565adcd16a4a05f2089b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.1.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded64ff6d4fdd1cb68dfcbb818c69e144a5b2e4c"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.16"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "e64b4f5ea6b7389f6f046d13d4896a8f9c1ba71e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "76374b6e7f632c130e78100b166e5a48464256f8"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.4.0+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PartialFunctions]]
deps = ["MacroTools"]
git-tree-sha1 = "47b49a4dbc23b76682205c646252c0f9e1eb75af"
uuid = "570af359-4316-4cb7-8c74-252c00c2016b"
version = "1.2.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "442e1e7ac27dd5ff8825c3fa62fbd1e86397974b"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.4"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "3aa2bb4982e575acd7583f01531f241af077b163"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.13"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "763a8ceb07833dd51bb9e3bbca372de32c0605ad"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.0"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "994cc27cdacca10e68feb291673ec3a76aa2fae9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "2a0a5d8569f481ff8840e3b7c84bbf188db6a3fe"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.7.0"
weakdeps = ["RecipesBase"]

    [deps.Rotations.extensions]
    RotationsRecipesBaseExt = "RecipesBase"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "3aac6d68c5e57449f5b9b865c9ba50ac2970c4cf"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.42"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "d2fdac9ff3906e27f7a618d47b676941baa6c80c"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.10"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "5d66818a39bb04bf328e92bc933ec5b4ee88e436"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.5.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.SymPy]]
deps = ["CommonEq", "CommonSolve", "LinearAlgebra", "PyCall", "SpecialFunctions", "SymPyCore"]
git-tree-sha1 = "8d727c118eb31ffad73cce569b7bb29eef5fb9ad"
uuid = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
version = "2.0.1"

[[deps.SymPyCore]]
deps = ["CommonEq", "CommonSolve", "Latexify", "LinearAlgebra", "Markdown", "RecipesBase", "SpecialFunctions"]
git-tree-sha1 = "d2e8b52c18ad76cc8827eb134b9ba4bb7699ec59"
uuid = "458b697b-88f0-4a86-b56b-78b75cfb3531"
version = "0.1.18"

    [deps.SymPyCore.extensions]
    SymPyCoreSymbolicUtilsExt = "SymbolicUtils"

    [deps.SymPyCore.weakdeps]
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "71509f04d045ec714c4748c785a59045c3736349"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.7"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "6129a4faf6242e7c3581116fbe3270f3ab17c90d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.67"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "532e22cf7be8462035d092ff21fada7527e2c488"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.6+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e678132f07ddb5bfa46857f0d7620fb9be675d3b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═32fbc94f-710f-435f-91a8-13c07b9671f0
# ╟─0e8fcc58-1f93-48ef-9c74-f5abdcdecbd3
# ╠═6db134a0-0962-11ef-310f-99600cd50dfb
# ╠═2289adfc-0bb4-4a07-9315-7ec870a0f0c6
# ╟─ffdb33f9-0bd8-4795-a4b3-aee5c5620fb7
# ╠═6d3c390d-7e2c-401f-b313-31975874e657
# ╠═54b587e9-566c-494a-839d-5f916911ba32
# ╠═a94f9691-3906-43cd-bf0e-55003ecbb1fc
# ╠═eba461a3-2026-47c8-8a4a-78dbf8975a3c
# ╠═4532b560-a6b2-4364-a548-ffb5d25b0d5c
# ╠═90808947-4a59-4903-8912-5e43839f91e0
# ╠═a4e6194b-5cfd-4968-8d22-6d7ed4e15523
# ╠═132467b1-315f-40e1-b6fb-a56537d54f27
# ╠═c7c67354-296f-4858-878d-81243ebfa32b
# ╠═8cbd2f0f-cfd9-418a-9333-e49596c81c06
# ╠═dddf76d2-dd8e-4c17-9cc0-256e20db0abe
# ╟─55d77b19-e600-4ce8-9dc1-8d2458c99da2
# ╠═db5d41fc-ea93-4abc-9efa-b232ef7f37e2
# ╠═9a379335-9430-4fa2-9b84-15e192ace090
# ╠═697828f5-2e3b-4c25-86e0-cc97cfdb8432
# ╟─e47e0b0a-377a-451e-93f8-d4430706ef19
# ╠═175d7472-4a23-4715-a39e-4a434cac46b1
# ╠═b700e215-3ba4-45e7-bea2-89f6f7ff73f7
# ╟─7f0583b7-8d5c-4660-ad47-5f64aa6f57bb
# ╠═fe402f42-0d4a-46fe-ad6d-57eb578d2cf4
# ╟─36305f0a-a02d-4ba0-8a4b-12a645a6cf39
# ╠═37396c37-0291-4ce6-92c6-defb547f94f5
# ╠═cd635cc3-c4fc-4da7-9f75-960a2aa118f4
# ╠═cc982bfb-8147-4e69-be5c-c75c053e8e33
# ╠═7e0fc6d8-b9af-4743-a634-bac7fdc6ff5d
# ╠═edf29ef0-272c-4067-8d6c-89aa8a217a3a
# ╠═03ca5540-a071-4139-9c6d-af42e036be6b
# ╠═8e05724b-943c-43a2-8319-3815ed65e4ab
# ╠═1e8ebf68-08df-4426-9a14-5a2748023ce0
# ╠═f98aefcb-58f4-4721-af1c-7171951362d9
# ╠═178afdd9-a049-4d91-bdd9-66cbde48fc7d
# ╠═badf8efb-114b-459b-9de4-3cb6c2b3849a
# ╟─551603af-23fc-49b6-a3c0-f15c199452dc
# ╠═1806ebc4-ab49-4d29-821e-886784f3c2ed
# ╠═496aa019-b453-4dcb-9ebb-c01a1c5b14b3
# ╠═a29da296-cbde-4004-8022-ef5888509b38
# ╠═d7bdb281-a363-476a-893b-4b7392f5c993
# ╠═95803efa-1141-42dd-8bcb-3778841d423f
# ╠═2b7e66ad-731d-4074-867a-66c54fabfd71
# ╠═9c1c8163-71ba-4261-bda2-d39ff729606c
# ╠═bc704cc3-c5eb-4119-8ed8-85b596fcaa8e
# ╠═539c0dbe-9ec3-4ea4-80e7-7ee7fd9324d6
# ╠═6d424f75-c63f-4f08-bee9-31b57756690c
# ╠═d1a086a9-92d4-427e-a2d3-0946d39de9a2
# ╠═f31efc51-cf0c-4093-af7a-b4e3a2e84b2f
# ╠═91d47e62-3207-48ec-a255-c1f1ece56d5b
# ╠═d1d6309d-ccf8-4799-b09b-8405d3f8c302
# ╠═68617bc1-8d21-4df7-8e1b-6751059e732e
# ╠═9e1cdf98-5b5d-4b46-a779-560b6f552d3c
# ╠═0d33e716-36c0-4a74-b30a-7ee19aeb26d3
# ╠═a0ef34ff-bb06-43e8-acd4-ee228a6edc68
# ╠═b17c985c-3869-482b-8199-23b1500893fe
# ╠═f0edaa1c-cd6d-4722-9d3f-0f6328095c48
# ╠═6b313419-288a-4f20-8453-e61fa463d225
# ╟─4149ce52-2e86-4c6b-92ca-d66fd9b8348d
# ╠═672ee9e6-1928-4eab-b2d9-b34179cf99fc
# ╠═8515d621-6221-4711-a486-0b9c3d3f6be4
# ╟─aae002c8-7160-40f2-a01b-dc1192dbd6d2
# ╠═8a565363-d3bb-423d-ab43-642f781925e4
# ╟─a1eaec8f-3527-43f6-b3e6-360c64b156e5
# ╠═023ccc34-42aa-4bbe-a127-3fdc48eec87a
# ╠═e8f7cad1-65a6-4814-9a06-5c90c87ec297
# ╠═b0a5af7c-235f-47c1-b3c8-a3f635ef53f1
# ╠═c92826c1-61f0-44e6-b333-c60deae80f68
# ╠═d7f3b4a4-99db-4a6d-a744-1b9e5a3f9360
# ╠═f2fce8ac-7800-4153-b0fa-21168300984e
# ╠═0a58b720-ea82-48e2-aa5c-ab0e9c530b41
# ╟─fb7b8d5e-0a75-4139-a1fd-e8f5f3b7db98
# ╠═00ffff53-e77e-46e3-be19-672cb1b87716
# ╠═ee0e9437-428f-420c-8733-7b0a00a94670
# ╠═c85e44f0-600a-4491-ba95-bf5b38d83578
# ╠═6fda8490-fdab-43a1-84c7-217d7bcf12d2
# ╠═5f07854d-0d4c-4de6-8d4a-4b319f372bc4
# ╠═0b6a3d60-a8b5-4439-bd29-046a8a4b488e
# ╠═e96c765f-5f5f-41db-8191-1a0d444bcde2
# ╠═331ceecb-53e4-41f9-b8de-2aece6d1eea8
# ╠═25319673-99ff-436d-bf5d-34c540f4faf1
# ╠═752d42d4-bc56-4b0d-a457-c4dbf32b27f8
# ╠═2a5009df-5e84-4b50-b819-c19333eb4a8b
# ╠═a116db18-a4c5-4661-aa0b-5098114d201e
# ╠═4abe4232-ca33-4d07-b20a-de2f0dc68591
# ╠═2eded938-c85c-4bf4-95aa-8d8b02c6cbbb
# ╠═23123586-648e-4187-829f-aba7f4114bd5
# ╟─4671e428-7d3e-4ea4-9480-fd7bae3d28f6
# ╠═74f3e3b6-20ee-4aa4-ac24-701cedb8196e
# ╟─9e4e4f8d-1d34-4c35-b814-b8d6708bf2ab
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002