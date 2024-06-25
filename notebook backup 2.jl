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
	using Symbolics
	using BenchmarkTools
	using PartialFunctions
	using OffsetArrays
	using ForwardDiff
	using OrdinaryDiffEq
	using Statistics
	using Markdown
end

# ╔═╡ c147a9be-8ce8-4df8-a1f1-1fc6c05a73cc
begin
	using AbstractTrees
	AbstractTrees.children(d::DataType) = subtypes(d)
	print_tree(Number)
end

# ╔═╡ 810898ac-546f-4832-9922-8eb61f154325
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
	    cache = Dict()
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

# ╔═╡ dd861e69-2c98-49e1-9970-f5e898048d04
begin
	using KernelAbstractions

	@kernel function estimate_pi_kernel(a, n)
		i = @index(Global)
		k = prod(@groupsize())
		@inbounds a[i] = prod((i ÷ 2 * 2) / ((i-1) ÷ 2 * 2 + 1) for i in 1+i:k:n)
	end

	device = CPU()  # GPU()

	@time let N = 300_000_000, K = 64
		A = ones(K)
		run! = estimate_pi_kernel(device, K)
		run!(A, N, ndrange=size(A))
		synchronize(device)
		2 * prod(A)
	end
end

# ╔═╡ cd2ebd41-45d3-4dd0-aba9-3f42070663b4
begin
	using CSV, DataFrames, NaNStatistics, DifferentialEquations
	import PlotlyJS, ModelingToolkit
	import ModelingToolkit: Differential, ODESystem
	plotlyjs()
end

# ╔═╡ 135b50a4-66c1-49c5-9865-63d4311ba694
Complex{Float64}

# ╔═╡ 65b87a6f-1edd-4b48-9878-90dc112881e6
subtypes(Integer), supertypes(Integer)

# ╔═╡ 0858a0ec-bbf0-4d04-8755-f88838272822
@which 2im

# ╔═╡ 369536ba-ffdc-49fc-9acf-1572300714be
methods(!)

# ╔═╡ a1f21125-0a64-45cc-833a-801d0539892b
names(Statistics)

# ╔═╡ 6ff63d76-7fd6-47fd-9c0e-8a72967a77a2
[1 2 3
 5 6 4
 9 7 8]  # or [1 2 3; 5 6 4; 9 7 8]

# ╔═╡ 9fd0358b-47e1-4e19-b446-3f8c91769d1c
zeros(3, 3)

# ╔═╡ 836b97b4-5350-4d85-bf60-17e52d6f9ef0
let B = @show similar(A)
	fill!(B, 3)
end

# ╔═╡ b618ae29-ed57-48fc-8102-1767a51a5b7a
factorial(5)

# ╔═╡ b6b5be30-5443-4709-9a24-9608b17f20f2
factorial(32)

# ╔═╡ eebaac1a-1b99-44e5-b9c0-1a3ed1ea8c0b
let
	function factorial(n)
		if !(n isa Integer)
			throw(ValueError("input is not an integer"))
		elseif n < 0
			throw(ValueError("input cannot be negative"))
		else
			prod(big(1):big(n))
		end
	end
	@time factorial.(0:32)
end

# ╔═╡ 8ab3b64e-e1e4-4cdf-9385-9573ad4d5863
begin
	⟹(p::Bool, q::Bool) = !p | q  # \implies
	⟺(p::Bool, q::Bool) = (p ⟹ q) & (q ⟹ p)  # \iff
	bools = Set([true, false])
	# equivalence of contrapositive statements
	all((p ⟹ q) ⟺ (!q ⟹ !p) for p ∈ bools, q ∈ bools)
	# see https://github.com/JuliaLang/julia/blob/master/src/julia-parser.scm for the symbols that can be defined as infix binary operators
end

# ╔═╡ 321c78d1-d175-43de-8069-cdf7cedaf20c
@time fib(32)

# ╔═╡ 76b59e6b-b1b9-49e3-8fea-fc8a15238c99
begin
	@memoize mfib(n) = n < 2 ? big(n) : mfib(n-2) + mfib(n-1)
	@time @show fib.(1:32)
	@time @show mfib.(1:32)
end

# ╔═╡ 1d7dce85-6652-4743-9ecc-46145a00bc70
macroexpand(@__MODULE__, :(@memoize mfib(n) = n < 2 ? big(n) : mfib(n-2) + mfib(n-1)))

# ╔═╡ 4585ea88-51e7-42c6-a014-e8097db09695
function fastfib(n)
    z = BigInt()
    ccall((:__gmpz_fib_ui, :libgmp), Cvoid, (Ref{BigInt}, Culong), z, n)
    return z
end

# ╔═╡ 11834ef0-f9a4-49a5-92ff-4ffe2c217912
@time fastfib(32)

# ╔═╡ 9a379335-9430-4fa2-9b84-15e192ace090
@time estimate_pi_mc(300_000_000)

# ╔═╡ 34d06f6b-5e8e-4823-98e7-e3707feb528d
let task = Threads.@spawn estimate_pi_mc()
	@show task
	fetch(task)
end

# ╔═╡ 697828f5-2e3b-4c25-86e0-cc97cfdb8432
@time let N = 300_000_000, k = @show Threads.nthreads()
	mean(fetch.(Threads.@spawn estimate_pi_mc(N÷k) for _ in 1:k))
end

# ╔═╡ 57f8919a-aadd-4e3b-9a24-cdca7870caa9
@time let N = 300_000_000
	# generator (laze evaluation)
	fracs = ((i ÷ 2 * 2) / ((i-1) ÷ 2 * 2 + 1) for i in 2:N)
	# fracs = [(i ÷ 2 * 2) / ((i-1) ÷ 2 * 2 + 1) for i in 2:N]
	2 * prod(fracs)
end

# ╔═╡ b700e215-3ba4-45e7-bea2-89f6f7ff73f7
# Task: implement multi-threaded version

# ╔═╡ 37396c37-0291-4ce6-92c6-defb547f94f5
@time nfold(x -> sin(x) + x, 5)(1)

# ╔═╡ 4cd08b52-0848-458a-8b92-9909015d8edd
img = let url = "https://images.fineartamerica.com/images-medium-large-5/1-earth-from-space-kevin-a-horganscience-photo-library.jpg"
	load(@show download(url))
end

# ╔═╡ f0edaa1c-cd6d-4722-9d3f-0f6328095c48
typeof(img)

# ╔═╡ 6b313419-288a-4f20-8453-e61fa463d225
SVD_results = [svd(f.(img)) for f in [red, green, blue]];

# ╔═╡ 4149ce52-2e86-4c6b-92ca-d66fd9b8348d
@bind K Slider(1:60, show_value=true, default=30)

# ╔═╡ 818de785-c18d-4b35-9a16-4453b7c9ade7
let kernel = centered([1 2 -1; 2 0 -2; -1 -2 1])
	imfilter(load("/tmp/my_earth.png"), kernel)
end

# ╔═╡ 5d477221-6c4a-409a-98aa-4373b7e05bce
run(`rm /tmp/my_earth.png`)

# ╔═╡ c8c3b967-ddb1-406b-8393-3fee26054d93
@code_llvm Normal()(1)

# ╔═╡ c85e44f0-600a-4491-ba95-bf5b38d83578
begin
	"""`D` dimensional `F` (real/complex) vector space of vector type `T`."""
	abstract type VectorSpace{T, D, F<:Number} end

	basis(s::VectorSpace) = s.basis
	basis(s::VectorSpace, i::Integer) = basis(s)[i]

	struct Vect{D, F, V<:VectorSpace{<:Any,D,F}}
		coefs::SVector{D,<:F}
		space::V

		function Vect(x::AbstractVector, s::VectorSpace{T,D,F}) where {T,D,F}
			new{D,F,typeof(s)}(SVector{D,F}(x), s)
		end
	end

	Base.length(v::Vect) = length(v.coefs)
	vec(v::Vect) = sum(c * b for (b, c) in zip(basis(v.space), v.coefs))

	abstract type InnerProdSpace{T, D, F} <: VectorSpace{T, D, F} end

	struct EuclideanSpace{D, F} <: InnerProdSpace{SVector{D,F}, D, F}
		basis::SMatrix{D,D,F}

		function EuclideanSpace{D,F}(basis=I) where {D,F}
			new{D,F}(SMatrix{D,D,F}(basis))
		end
	end

	basis(s::EuclideanSpace) = eachcol(s.basis)

	function space_type(x::AbstractArray{T,N}) where {T<:Number,N}
		EuclideanSpace{size(x, 1), T <: Real ? Float64 : ComplexF64}
	end

	const ℝ{N} = EuclideanSpace{N, Float64}
	const ℂ{N} = EuclideanSpace{N, ComplexF64}

	Vect(coefs::AbstractVector; basis=I) = let
		T = space_type(basis isa AbstractArray ? basis : coefs)
		Vect(coefs, T(basis))
	end
	
	VectorSpace(basis::AbstractMatrix) = space_type(basis)(basis)

	Base.in(x::Vect, V::Type{<:VectorSpace}) = x.space isa V

	function Base.getindex(s::VectorSpace, x::T...) where {T}
		Vect(T <: AbstractVector && length(x) == 1 ? x[1] : collect(x), s)
	end

	function Base.:+(u::Vect{D,<:Number,V}, v::Vect{D,<:Number,V}) where {D,V}
		if u.space == v.space
			Vect(u.coefs .+ v.coefs, u.space)
		else
			vec(u) + vec(v)
		end
	end
	
	Base.:-(v::Vect) = Vect(-v.coefs, v.space)
	Base.:*(a::Number, v::Vect) = Vect(a.*v.coefs, v.space)
	Base.:*(x::Vect, y::Number) = y * x
	Base.:-(x, y) = x + (-y)
	Base.:/(x, y::Number) = (1/y) * x
	
	proj(v::Vect, s::VectorSpace) = proj(vec(v), s)

	function proj(x::T, s::VectorSpace{T,D,F}) where {T,D,F}
		B = basis(s)
		if !(x isa AbstractVector) && s isa InnerProdSpace
			x = [x ⋅ v for v in B]
			B = [u ⋅ v for u in B, v in B]
		end
		Vect(B \ x, s)
	end
end

# ╔═╡ bd367484-2cd2-43b4-9ecf-61bff5511614
function Statistics.mean(A::Array, dims::Integer...)
	if length(dims) == 0
		return sum(A) / length(A)
	end
	for i in sort(collect(dims), rev=true)
		A = sum(A, dims=i) ./ size(A, i)
	end
	return A
end

# ╔═╡ 7a2baa7c-af89-478f-99ee-f5e9fca9f871
begin
	Base.adjoint(f::Function) = x -> ForwardDiff.derivative(f, x)
	sin'(0), cos'(π/2)
end

# ╔═╡ e96c765f-5f5f-41db-8191-1a0d444bcde2
begin
	struct Func{L, H}
		exp::Sym
		var::Sym
	end

	struct FuncSpace{D, F, L, H} <: InnerProdSpace{Sym, D, F}
		basis::SVector{D,Sym}
		var::Sym

		function FuncSpace(basis::Vector{<:Sym}, (lo, hi), field=Real)
			var = only(free_symbols(basis))
			new{length(basis),field,lo,hi}(basis, var)
		end
	
		function FuncSpace(basis::Vector{Func{L,H}}, field=Real) where {L,H}
			var = only(Set(f.var for f in basis))
			new{length(basis),field,L,H}([f.exp for f in basis], var)
		end
	end

	basis(s::FuncSpace{D,F,L,H}) where {D,F,L,H} = Func{L,H}.(s.basis, s.var)
	vec(v::Vect{D,F,<:FuncSpace}) where {D,F} = sum(v.coefs .* basis(v.space)).exp
	(v::Vect{D,F,FuncSpace})(x) where {D,F} = vec(v)(x)

	function PolySpace(deg, range=(0, 1), field=Real, var=:x)
		basis = symbols(var, real=true) .^ (0:deg)
		FuncSpace(basis, range, field)
	end

	function FourierSpace(deg, range=(0, 2π), field=Real, var=:x)
		a, b = range
		T = b - a
		t = symbols(var, real=true) - a
		basis = [exp(-im*PI*k*t/T) for k = 0:deg]
		basis = field <: Real ? real.(basis) : basis
		FuncSpace(basis, range, field)
	end

	Base.:*(a, f::T) where {T<:Func} = T(a * f.exp, f.var)
	Base.:+(f::T, g::T) where {T<:Func} = T(f.exp + g.exp, f.var)
	Base.:-(f::T) where {T<:Func} = T(-f.exp, f.var)

	proj(f::Func, s::FuncSpace) = proj(f.exp, s)
	proj(f::Function, s::FuncSpace) = proj(f(s.var), s)
end

# ╔═╡ f1f274f1-3392-4fff-8e4c-90a4259ef7f0
1 + 2 * 3

# ╔═╡ 6a624939-5e57-450c-b574-7927ffe23ec4
z = exp(im * π)

# ╔═╡ 09f83f22-e305-4d21-a984-10bdf3b458cd
z == -1, z ≈ -1  # tuple

# ╔═╡ 96f1a1a6-7a15-4f8d-81ed-2f6785c15aa8
angle(z)

# ╔═╡ 001d5569-c199-4758-875b-f1057466476d
M = [i + j*im for i in 1:3, j in 1:3]

# ╔═╡ f8e9f8c6-99e7-4919-957c-85608a012e83
M', transpose(M)

# ╔═╡ 7c75d3c1-7757-4afb-a3fc-980798869633
M ^ 2, exp(M)

# ╔═╡ fb605635-8083-4448-b20c-522a45717e3b
rank(M), tr(M), det(M), diag(M)

# ╔═╡ d49583e5-0e8b-44d7-9b4d-25d8e732446b
let b = [3, 2, 1]
	x = @show M \ b  # inv(M) * b
	M * x
end

# ╔═╡ 32342cf4-1149-4443-b257-3da55837d3ba
let eig = eigen(M)
	@show eig.values
	@show eig.vectors
	λ, V = eig
	M * V ≈ λ' .* V
end

# ╔═╡ 47bb0f8b-899d-485b-ba2e-3e2a7a99a1de
let factorial(n) = n < 2 ? big(1) : n * factorial(n-1)
	@show factorial(32)
	@time factorial.(0:32)
end

# ╔═╡ eba461a3-2026-47c8-8a4a-78dbf8975a3c
begin
	sq(x) = x ^ 2
	double(f) = x -> f(f(x))  # anonymous function
	@show map(double(sq), [3, "3"])
	triple(f) = f ∘ f ∘ f
	inc = Base.Fix1(+, 1)  # inc = x -> 1 + x
	@show triple(double)(inc)(0)  # applies inc for 2^3 times
	nfold(f, n) = foldr(∘, fill(f, n))
	nfold(triple, 3)(cos)
end

# ╔═╡ 4b2e7984-023e-49f2-83b3-3cc01819e239
fib(n) = n < 2 ? big(n) : fib(n-2) + fib(n-1)

# ╔═╡ c56a1f71-ecdb-41ef-a614-379b32a270db
let f(g) = n -> n < 2 ? n : g(n-1) + g(n-2)
	partial_fib(i) = nfold(f, i)(x -> NaN)
	for i in 1:8
		println(partial_fib(i).(1:8))
	end
	Y_fib = (x -> f(y -> x(x)(y)))(x -> f(y -> x(x)(y)))  # Y combinator
	Y_fib.(1:8)  # f(f(f(f(...))))
end

# ╔═╡ 768ea165-199f-4106-bcab-d476ebf7dea6
function prime_sieve(n)
	mask = trues(n)
	mask[1] = false
	m = Int(floor(sqrt(n)))
	for i in 2:m
		if mask[i]
			mask[2i:i:end] .= false
		end
	end
	findall(mask)
end

# ╔═╡ 78ed1373-b51c-43c2-9227-9d9785848d69
function factorize(n)
	n <= 0 && throw(DomainError("cannot factorize nonpositive integer"))
	factors = Dict()
	for p in prime_sieve(n)
		while n % p == 0
			n ÷= p
			factors[p] = get(factors, p, 0) + 1
		end
		if n <= 1
			break
		end
	end
	return factors
end

# ╔═╡ 9865e00a-0c32-419e-a8cc-0b4cb35b7031
lucas_lehmer(n, m) = n == 0 ? 4 : (lucas_lehmer(n-1, m)^2 - 2) % m

# ╔═╡ a306cd12-7218-4767-8650-7034dcb6e303
function perfect_numbers(N)
	primes = prime_sieve(N)
	mersennes = big(1) .<< primes .- 1
	[big(2)^(p-1) * m for (p, m) in zip(primes, mersennes)
	 if p < 3 || lucas_lehmer(p-2, m) == 0]
end

# ╔═╡ 9f457e6d-ce3e-41fa-b3ce-eb3c4993286e
perfect_numbers(100)

# ╔═╡ 11d02c6a-a28b-4b6e-9b2a-83ea20e01c28
function is_perfect_number(n)
	pfs, degs = zip(pairs(factorize(n))...)
	factors = []
	for ds in Iterators.product([0:d for d in degs]...)
		push!(factors, prod(p ^ d for (p, d) in zip(pfs, ds)))
	end
	sum(factors) == 2n
end

# ╔═╡ 64ee1549-b481-49a1-88d6-1961b9b82a91
[(n, is_perfect_number(n)) for n in perfect_numbers(15)]

# ╔═╡ fe402f42-0d4a-46fe-ad6d-57eb578d2cf4
begin
	function newton_method(f, x; max_iter=10000, tol=1e-12, buf=nothing)
		for _ in 1:max_iter
			dx = f(x) / f'(x)
			x -= dx
			if buf != nothing
				push!(buf, x)
			end
			if abs(dx) < tol
				return x
			end
		end
		@warn "root not found after $max_iter iterations"
	end
	@time newton_method(sin, 2)
end

# ╔═╡ cc982bfb-8147-4e69-be5c-c75c053e8e33
let f = sin, x = range(-1, 5, 1000)
	y = f.(x)
	x0 = 2
	xs = []
	newton_method(f, x0, max_iter=30, buf=xs)
	@gif for x1 in xs
		plot(x, y, label="f(x)", title="Newton's method")
		plot!([x0, x1], [f(x0), 0], label="tagent line", markershape=:circle)
		plot!([x1, x1], [0, f(x1)], label=nothing, linestyle=:dash)
		annotate!(x0, f(x0), ("x = $(Float16(x0))", 8, :top))
		x0 = x1
	end fps=2
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
	# A = OffsetMatrix(img, -M÷2, -N÷2)
	A = centered(img)
	I, J = convert.(UnitRange, axes(A))
	sx, sy = svd(basis).S
	P = Iterators.product((I.start*sy):(I.stop*sy), (J.start*sx):(J.stop*sx))
	Idx = Int32.(floor.(inv(basis) * reshape(collect(Iterators.flatten(P)), 2, :)))
	blank = RGB(0, 0, 0)
	color(i, j) = i in I && j in J ? A[i,j] : blank
	C = color.(eachrow(Idx)...)
	reshape(C, length.(P.iterators)...)
end

# ╔═╡ f057d9db-8c17-49a9-a910-c6c9553e9c93
struct Normal
	μ :: Float64  # try removing the type declarations
	σ :: Float64

	Normal(μ=0.0, σ=1.0) = new(μ, σ)

	(p::Normal)(x) = exp(-0.5((x-p.μ)/p.σ)^2) / (p.σ * √2π)
end

# ╔═╡ bbf11fd6-4a18-4f4e-9022-6801ec387df9
let p = Normal()
	@code_warntype p(1)
end

# ╔═╡ d3d0a9d6-4540-4d87-9254-489a9335a322
begin
	Base.rand(P::Normal, dims::Integer...) = randn(dims...) .* P.σ .+ P.μ
	Statistics.mean(P::Normal) = P.μ
	Statistics.std(P::Normal) = P.σ
	Statistics.var(P::Normal) = P.σ ^ 2
end

# ╔═╡ 07d671cb-179f-413a-be83-8d2a0628aa22
let θ = rand(), z = exp(im * θ)  # bind variables locally
	x, y = @show reim(z)
	x ^ 2 + y ^ 2 == abs(z) ^ 2
end

# ╔═╡ d4d25d4f-18e6-4a38-8589-3aa7eb47f745
A = rand(Float64, (3, 4))

# ╔═╡ 7888f53a-56d7-4b50-a79c-bd8e472a0938
size(A), size(A, 1)

# ╔═╡ 087e605d-6741-4b58-bdb3-c4d12fd82869
[A[:, 3:4]; A[[1,3], 1:2:end]]  # concat vertically

# ╔═╡ 0a68a4ac-f0e7-43fe-9e39-11dbf007bc7e
[sum(A .^ 2, dims=2) maximum(A, dims=2)]  # concat horizontally

# ╔═╡ 6c63086a-dc7a-485e-95d6-acc5ef8d700b
diff(cumsum(A, dims=2), dims=2) ≈ A[:, 2:end]

# ╔═╡ 053cddd8-48ef-4fba-a510-3ffcce200ea5
let B = reshape(A, 2, 6)
	B[2, 3] = -999
	i = @show findfirst(A .== -999)
	C = @view B[1:2, 2:3]
	A[i] = -1
	C
end

# ╔═╡ 3e2836cd-f94f-41e9-8151-62bfff3af303
mean(M), mean(M, 1), mean(M, 2), mean(M, 1, 2)

# ╔═╡ db5d41fc-ea93-4abc-9efa-b232ef7f37e2
function estimate_pi_mc(n=100_000_000)
	mean(1:n) do _
		rand()^2 + rand()^2 < 1
	end / n * 4
end

# ╔═╡ 175d7472-4a23-4715-a39e-4a434cac46b1
let N = 300_000_000, K = 120
	times = [Float64[] for _ in 1:Threads.nthreads()]
	@time let
		A = ones(K)
		@Threads.threads for i in 1:K
			t0 = time()
			A[i] = prod((i ÷ 2 * 2) / ((i-1) ÷ 2 * 2 + 1) for i in 1+i:K:N)
			push!(times[Threads.threadid()], time() - t0)
		end
		@show 2 * prod(A)
	end
	[f(ts) for ts in times, f in [length, mean]]
end

# ╔═╡ 0c593b1c-e74a-46ed-8f66-47d1456b3636
let p1 = Normal()
	p2 = Normal(-4.0, 0.7)
	@show mean(p1), var(p1), mean(p2), var(p2)
	xs = vcat(rand(p1, 2000), rand(p2, 2000))
	@show mean(xs)
	@show mean((xs .- mean(xs)) .^ 2)
	histogram(xs, label=false, normalize=true, nbin=80)
	let x = range(-10, 10, 1000)
		plot!(x, p1.(x), label="N$((p1.μ, p1.σ))")
		plot!(x, p2.(x), label="N$((p2.μ, p2.σ))")
	end
end

# ╔═╡ 40ae0bd6-0fe5-4537-9cee-9485a8b741f3
methodswith(Vect)

# ╔═╡ 248c304c-de63-4523-b334-e201ed5f07f2
begin
	u = Vect([3, 4])
	v = Vect([3, 4], basis=[1 2; 3 4])
	@show v ∈ ℝ{2}
	u + v
end

# ╔═╡ 2a5009df-5e84-4b50-b819-c19333eb4a8b
let
	f(x) = cos(2x^2)
	g = @show vec(proj(f, p1))
	h = @show vec(proj(f, p2))
	xs = -2.2:0.001:2.2
	plot(f, xs, label="function", legend = :outertopright)
	plot!(g, xs, label="polynomial")
	plot!(h, xs, label="fourier")
end

# ╔═╡ 3f880785-c0ef-447c-b5ee-db258fc9db16
"Linear recurrence of the form ``x_{n+D} = \\sum_{k=0}^{D-1} c_k x_{n+k}``"
struct LinearRecurrence{D} <: VectorSpace{Vector, D, Float64}
	coefs::SVector{D, <:Number}
	roots::SVector{D, <:Number}  # roots of characteristic polynomial

	function LinearRecurrence(coefs...)
		D = length(coefs)
		v = collect(coefs)
		m = vcat(v', I(D)[1:end-1, :])
		new{D}(v, eigvals(m))
	end
end

# ╔═╡ 6b847bf5-87df-4e87-8b1c-fb9da5da0d54
basis(lr::LinearRecurrence{D}) where {D} = hcat([r .^ (1:D) for r in lr.roots]...)

# ╔═╡ 8e4eb2c7-20d3-4aa0-8c09-34fdb8293a9c
begin
	import LinearAlgebra: dot  # to overload the '⋅' operator

	dot(A::StaticMatrix, x::StaticVector) = A * x
	dot(u::Vect, v::Vect) = 
		sum(a * (u ⋅ x) for (a, x) in zip(v.coefs, basis(v.space)))
	dot(u::Vect, x) = sum(a * (x ⋅ y) for (a, y) in zip(u.coefs, basis(u.space)))
	dot(f::Func, g::Func) = dot(f, g.exp)
	dot(f::Func{L,H}, g) where {L,H} =
		integrate(SymPy.simplify(real(f.exp * conj(g))), (f.var, L, H))
	dot(x, u::Union{Vect, Func}) = dot(u, x)

	norm(v) = sqrt(v ⋅ v)
	proj(u, v) = (u ⋅ v) / (v ⋅ v) * v

	function orthogonalize(s::V) where {T,D,F,V<:InnerProdSpace{T,D,F}}
		new_basis = [basis(s, 1)]
		for i in 2:D
			u = basis(s, i)
			push!(new_basis, u - sum(proj(u, v) for v in new_basis))
		end
		new_basis ./ norm.(new_basis)
	end
end

# ╔═╡ 0462c099-18ee-415d-b06a-7bb8eafa360f
let
	m = hcat(orthogonalize(v.space)...)
	w = Vect([3, -4], basis=@show m)
	norm(w)
end

# ╔═╡ 331ceecb-53e4-41f9-b8de-2aece6d1eea8
begin
	p1 = @show PolySpace(6, (-2, 2))
	v1 = p1[1:7]
	s1 = FuncSpace(orthogonalize(p1))
	u1 = proj(v1, s1)
	@show vec(u1).evalf()
	u1
end

# ╔═╡ 25319673-99ff-436d-bf5d-34c540f4faf1
begin
	p2 = @show FourierSpace(6, (-2, 2))
	v2 = p2[0, 1, -1, 2, -2, 3, -3]
	s2 = FuncSpace(orthogonalize(p2))
	u2 = proj(v2, s2)
	@show vec(u2).evalf()
	u2
end

# ╔═╡ 9510e0c5-c658-4358-871f-3c433a8f183d
begin
	Base.getindex(x::Vect{<:Any,<:Number,<:LinearRecurrence}, i::Integer) = 
		(x.space.roots .^ i) ⋅ x.coefs
	Base.getindex(x::Vect{<:Any,<:Number,<:LinearRecurrence}, i) = (j -> x[j]).(i)
end

# ╔═╡ 907a813d-945d-40b2-9262-cc2975b8a436
let R = LinearRecurrence(1, 1)  # x[n+2] = x[n+1] + x[n]
	@show basis(R)
	x = R[1, 1]
	@show x[1:5]
	fib = proj([1, 1], R)
	@show fib.coefs
	@time fib[1:32]
end

# ╔═╡ 2eded938-c85c-4bf4-95aa-8d8b02c6cbbb
"Solutions of homogeneous ODE of the form ``\\sum_{k=0}^Dc_k\\frac{d^k}{dt^k}x(t)=0``"
struct ODESolutions{D} <: VectorSpace{Vector, D, Float64}
	eq::Vector
	basis::Vector
	var::Sym

	function ODESolutions(coefs...; var=symbols(:t, real=true))
		v = collect(coefs)
		ord = length(v) - 1
		eq = dot(v, var .^ (0:ord))
		new{ord}(v, exp.(solve(eq, var) .* var), var)
	end
end

# ╔═╡ fb1ad4f2-5b02-4080-bb0f-94a1bf918562
function Base.getindex(ode::ODESolutions, inivals::Pair...)
	ts, xs = collect.(zip(inivals...))
	A = [subs(b, ode.var => t) for t in ts, b in basis(ode)]
	ode[A \ xs]
end

# ╔═╡ 0e8fcc58-1f93-48ef-9c74-f5abdcdecbd3
md"""
# Julia Fundamentals
"""

# ╔═╡ 6cfc8e81-4fe5-4555-870e-427167e68ab1
md"## Types and Fields"

# ╔═╡ f16eab5d-994b-4e94-9d51-acc52021ccd5
md"### Arrays"

# ╔═╡ 05aaecbc-e5b2-43ce-9bf3-e6aef1a3860f
md"## Functions"

# ╔═╡ be46feb4-0e24-4b35-b9ed-c07c776f5424
md"### Some logic"

# ╔═╡ 8fc20559-3ebe-478a-a96a-d0b2f9ca7e6a
md"### Higher order functions"

# ╔═╡ 64b53133-51e6-400e-9047-c2c18d1805f5
md"# Case Study: Fibonacci Sequence"

# ╔═╡ fb1c8a2f-babb-4b93-be81-a1fe843fbfa5
md"Recursion as fixed point of higher order function."

# ╔═╡ b14c1634-2b8b-4ad3-ad3a-2ea44c071471
md"A faster option: calling C function."

# ╔═╡ d66efb8a-03f5-4879-9ae4-5242d8288d67
md"# Case Study: Perfect Numbers"

# ╔═╡ 768bfdcc-11b6-4fd9-9b0b-42035ef89664
md"""
!!! danger "Task"
	Implement prime factorization
"""

# ╔═╡ 4121fd34-e7ae-4a04-a32f-69660e3b85ca
md"Euclid-Euler Theorem"

# ╔═╡ f646fd23-beeb-48d8-80d2-503442371d83
md"""
!!! danger "Task"
	Implement perfect number test
"""

# ╔═╡ 078dbda5-3eec-4c2b-9c74-1349511cdb82
md"# Case Study: Estimate π"

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

# ╔═╡ 89143850-3a09-4407-abf6-066087d180ef
md"""Estimate π using continued fraction ([source](https://en.wikipedia.org/wiki/Euler%27s_continued_fraction_formula#A_continued_fraction_for_%CF%80
)):

`` \pi = \frac{4}{1+\frac{1^2}{2+\frac{3^2}{2+\frac{5^2}{2+\ldots}}}} ``

"""

# ╔═╡ 521a5ff0-44b6-4065-8fb6-6dd3c1b1689a
md"""
!!! danger "Task"
	Estimate π using the formula above
"""

# ╔═╡ 551603af-23fc-49b6-a3c0-f15c199452dc
md"""
# Case Study: Image Transformation
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
	img2 = transform_image(img, trans)
end

# ╔═╡ af195cbb-058e-429f-a2a8-0635b827d54f
save("/tmp/my_earth.png", img2)

# ╔═╡ a1eaec8f-3527-43f6-b3e6-360c64b156e5
md"""# Case Study: Normal Distribution"""

# ╔═╡ dafb5973-8bf9-4436-b7e2-cd9fe29e1d2c
md"Try adding and removing the type declarations in `Normal` to see the difference of generated bitcodes."

# ╔═╡ b3ab790a-e748-4634-816a-ed7acd3f31d3
md"# Case Study: Vector Space"

# ╔═╡ cd139cca-0b06-40da-b2b7-f49e3b3bcd52
let ode = ODESolutions(1, 2, -3)  # x + 2x' - 3x'' = 0
	x = vec(ode[0=>2, 1=>1])
	D = SymPy.Differential(ode.var)
	@show x + 2D(x) - 3(D^2)(x)
	x(0).evalf(), x(1).evalf()
end

# ╔═╡ 113a69c4-ada0-4e64-b910-64434f191a7e
md"# Case Study: Energy Balance Model"

# ╔═╡ 479ddea4-9370-408e-8000-1bf02d873a02
begin
	# https://dataframes.juliadata.org/stable/man/comparisons/
	CO2_historical_path = download("https://www.epa.gov/system/files/other-files/2022-07/ghg-concentrations_fig-1.csv")
	# CO2_historical_path = "/home/tcai/Downloads/total-ghg-emissions.csv"

	offset = findfirst(startswith("Year"), readlines(CO2_historical_path))

	CO2_historical_data_raw = CSV.read(
		CO2_historical_path, DataFrame; 
		header=offset, 
		skipto=offset+2,
	)

	last(CO2_historical_data_raw, 220)
end

# ╔═╡ b484e82c-9960-43e2-b90b-f08dfd36214e
begin
	CO2_historical_data = subset(CO2_historical_data_raw, "Year" => y -> y .>= 1850)
	values = replace(Matrix(CO2_historical_data[:,2:end]), missing=>NaN)
	CO2_historical_data.CO2 = reshape(nanmean(values, dims=2), :)
	select!(CO2_historical_data, :Year, :CO2)
	first(CO2_historical_data, 5), last(CO2_historical_data, 5)
end

# ╔═╡ d5ef8924-6ad4-4aeb-9a62-387e1274de66
# Task 1: fit a polynomial to the Keeling curve
begin
	CO2(t) = features(t .- 1850) * CO2_params
	features(t) = hcat(ones(length(t)), t.^3)
	CO2_params = let
		t = CO2_historical_data[:, "Year"] .- 1850
		y = CO2_historical_data[:, "CO2"]
		X = features(t)
		p = X \ y
	end
end

# ╔═╡ 94038e3a-3d01-439c-a01c-2c7aacf16147
begin
	years = 1850:2030
	let df = CO2_historical_data
		plot(df[:, "Year"] , df[:, "CO2"], 
			 label="Global atmospheric CO₂ concentration")
		plot!(years, CO2(years), label="Fitted curve", legend=:bottomright)
	end
	title!("CO₂ observations and fit")
end

# ╔═╡ f2678b18-3819-421a-9b66-b698e132a1d9
begin
	@ModelingToolkit.parameters t α a S β γ C
	@ModelingToolkit.variables Y(t) RC(t)

	absorbed_solar_radiation = (1 - α) * S / 4
	outgoing_thermal_radiation = β - γ * Y
	greenhouse_effect = a * log(RC)

	D = Differential(t)
	eqs = [
		C * D(Y) ~ 
			absorbed_solar_radiation - 
			outgoing_thermal_radiation + 
			greenhouse_effect,
		RC ~ CO2(t+1850) / CO2(1850)
		# RC ~ 1 + (t/220)^3
	]
end

# ╔═╡ ce55e76c-f309-4f34-a8fc-2db2af7d030a
@mtkbuild sys = ODESystem(eqs, t)

# ╔═╡ 17b7c372-269c-44b8-8d7b-048ce10e64f6
begin
	ini = [Y => 14.0]  # initial condition
	ps = [  # parameters
		a => 5.0, 
		α => 0.3, 
		C => 51, 
		S => 1368, 
		β => 221.2, 
		γ => -1.3,
	]
	tspan = (0, 2024-1850)
	prob = ODEProblem(sys, ini, tspan, ps)
end

# ╔═╡ abdf8da8-be19-4428-9230-8bb46f8d03e5
begin
	temps = vcat(solve(prob).(30:180)...)
	plot(1880:2030, temps, lw=2, legend=:topleft,
		 label="Predicted Temperature from model")
	xlabel!("year")
	ylabel!("Temp °C")
end

# ╔═╡ cfd3c52f-8b2c-48ab-86f3-2b8835abddd0
begin
	T_url = "https://data.giss.nasa.gov/gistemp/graphs/graph_data/Global_Mean_Estimates_based_on_Land_and_Ocean_Data/graph.txt"
	s = read(download(T_url), String)
	io = replace(s, r" +" => " ") |> IOBuffer
	T_df = CSV.read(io, DataFrame, header=false, skipto=6);
	T_df = rename(T_df[:,1:2], :Column1=>:year, :Column2=>:temp)
	T_df.temp .+= 14.15
	T_df
end

# ╔═╡ ae59ec71-6362-4dd9-bfc7-74fb8c52f777
plot!(T_df[:, :year], T_df[:, :temp], 
	  color=:black, label="NASA Observations", legend=:topleft)

# ╔═╡ d28ba6a0-e1b1-43ca-8744-976da374c98d
md"""The reason why the predicted temperature is lower than the observation is probably that we have not taken into account other greenhouse gases and feedback factors such as water vapour."""

# ╔═╡ 6f9c7c9c-5992-49bb-8090-98a03203414a
# begin
# 	using DiffEqFlux
	
# 	function neural_ode(t, data_dim; saveat = t)
# 	  f = FastChain(FastDense(data_dim, 64, swish),
# 					FastDense(64, 32, swish),
# 					FastDense(32, data_dim))
	
# 	  node = NeuralODE(f, (minimum(t), maximum(t)), Tsit5(),
# 					   saveat = saveat, abstol = 1e-9,
# 					   reltol = 1e-9)
# 	end
# end

# ╔═╡ 37008e44-c0fc-42e9-851a-c4e4a3a522d5


# ╔═╡ 9e4e4f8d-1d34-4c35-b814-b8d6708bf2ab
macro show_all(block)
	if block.head == Symbol("let")
		lines = block.args[2].args
	elseif block.head == Symbol("block")
		lines = block.args
	else
		return block
	end
	foreach(enumerate(lines)) do (i, ex)
		if ex isa Union{Symbol, Expr, Number}
			lines[i] = :(@show $ex)
		end
	end
	return block
end

# ╔═╡ f722fd52-3f0c-4546-90c9-9b885c10a3a5
@show_all begin
	3 + 4; 3 * 4; 3 / 4; 3 ÷ 4; 4 % 3; 3 ^ 4;
end

# ╔═╡ 0a34b4fe-9ca8-42ad-a0f7-a1f29b234ff0
@show_all begin
	log2(4); log(ℯ); log10(1e4); log(4, 1024); sqrt(4); exp(4); cos(0); acos(0);
end

# ╔═╡ 6d3c390d-7e2c-401f-b313-31975874e657
@show_all begin
	typeof(z)
	z.re  # real(z)
	getfield(z, :im)  # imag(z)
	z isa Complex
	z isa Number
end;

# ╔═╡ 9da04702-752f-4bc7-a62f-eaa17d86fa4d
@show_all let T = Complex{Int64}
	T <: Complex
	T <: Number
	T <: Complex{<:Real}
	T <: Complex{Real}
	Dict(zip(fieldnames(T), fieldtypes(T)))
end

# ╔═╡ 2ba441e8-0d84-4bf4-b25a-cc17a64cb554
@show_all let a = -6:3:6, b = [1, 2, 3]
	length(a)
	reverse(a)
	a .* 2
	a[1:3:end]
	a[end:-2:1]
	maximum(a)  # max(a...)
	abs.(a)
	a .^ 2 |> sum  # sum of squares
	count(iseven, a)
	vcat(a, b)
	zip(a, b) |> collect
	map(-, a, b)
	push!(b, 4, 5)
	deleteat!(b, 1)
	accumulate(*, b)
	foldl(-, b)  # 2 - 3 - 4 - 5 (starting from the left)
	["$i: $n" for (i, n) in enumerate(b) if n % 2 != 0]
end

# ╔═╡ 672fb2ff-5782-4411-85d0-ca83506372c8
begin
	almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]))
	still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]))
	keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]))
	yays = [md"Fantastic!", md"Splendid!", md"Great!", md"Yay ❤", md"Great! 🎉", md"Well done!", md"Keep it up!", md"Good job!", md"Awesome!", md"You got the right answer!", md"Let's move on to the next section."]
	correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]))
end

# ╔═╡ 04ea591e-bd74-4926-82a6-d6a09f242a71
let
	result = all([
		factorize(66) == Dict(2=>1, 3=>1, 11=>1),
		factorize(1) == Dict(),
		factorize(2) == Dict(2=>1),
		factorize(9) == Dict(3=>2),
	])
	if ismissing(result)
		still_missing()
	elseif isnothing(result)
		keep_working(md"Did you forget to write `return`?")
	elseif result
		correct()
	else
		keep_working()
	end
end

# ╔═╡ Cell order:
# ╠═32fbc94f-710f-435f-91a8-13c07b9671f0
# ╟─0e8fcc58-1f93-48ef-9c74-f5abdcdecbd3
# ╠═f1f274f1-3392-4fff-8e4c-90a4259ef7f0
# ╠═f722fd52-3f0c-4546-90c9-9b885c10a3a5
# ╠═0a34b4fe-9ca8-42ad-a0f7-a1f29b234ff0
# ╠═6a624939-5e57-450c-b574-7927ffe23ec4
# ╠═09f83f22-e305-4d21-a984-10bdf3b458cd
# ╠═96f1a1a6-7a15-4f8d-81ed-2f6785c15aa8
# ╠═07d671cb-179f-413a-be83-8d2a0628aa22
# ╟─6cfc8e81-4fe5-4555-870e-427167e68ab1
# ╠═6d3c390d-7e2c-401f-b313-31975874e657
# ╠═135b50a4-66c1-49c5-9865-63d4311ba694
# ╠═9da04702-752f-4bc7-a62f-eaa17d86fa4d
# ╠═65b87a6f-1edd-4b48-9878-90dc112881e6
# ╠═c147a9be-8ce8-4df8-a1f1-1fc6c05a73cc
# ╠═0858a0ec-bbf0-4d04-8755-f88838272822
# ╠═369536ba-ffdc-49fc-9acf-1572300714be
# ╠═a1f21125-0a64-45cc-833a-801d0539892b
# ╟─f16eab5d-994b-4e94-9d51-acc52021ccd5
# ╠═2ba441e8-0d84-4bf4-b25a-cc17a64cb554
# ╠═6ff63d76-7fd6-47fd-9c0e-8a72967a77a2
# ╠═9fd0358b-47e1-4e19-b446-3f8c91769d1c
# ╠═d4d25d4f-18e6-4a38-8589-3aa7eb47f745
# ╠═7888f53a-56d7-4b50-a79c-bd8e472a0938
# ╠═836b97b4-5350-4d85-bf60-17e52d6f9ef0
# ╠═087e605d-6741-4b58-bdb3-c4d12fd82869
# ╠═0a68a4ac-f0e7-43fe-9e39-11dbf007bc7e
# ╠═6c63086a-dc7a-485e-95d6-acc5ef8d700b
# ╠═053cddd8-48ef-4fba-a510-3ffcce200ea5
# ╠═001d5569-c199-4758-875b-f1057466476d
# ╠═f8e9f8c6-99e7-4919-957c-85608a012e83
# ╠═7c75d3c1-7757-4afb-a3fc-980798869633
# ╠═bd367484-2cd2-43b4-9ecf-61bff5511614
# ╠═3e2836cd-f94f-41e9-8151-62bfff3af303
# ╠═d49583e5-0e8b-44d7-9b4d-25d8e732446b
# ╠═fb605635-8083-4448-b20c-522a45717e3b
# ╠═32342cf4-1149-4443-b257-3da55837d3ba
# ╟─05aaecbc-e5b2-43ce-9bf3-e6aef1a3860f
# ╠═b618ae29-ed57-48fc-8102-1767a51a5b7a
# ╠═b6b5be30-5443-4709-9a24-9608b17f20f2
# ╠═47bb0f8b-899d-485b-ba2e-3e2a7a99a1de
# ╠═eebaac1a-1b99-44e5-b9c0-1a3ed1ea8c0b
# ╟─be46feb4-0e24-4b35-b9ed-c07c776f5424
# ╠═8ab3b64e-e1e4-4cdf-9385-9573ad4d5863
# ╟─8fc20559-3ebe-478a-a96a-d0b2f9ca7e6a
# ╠═eba461a3-2026-47c8-8a4a-78dbf8975a3c
# ╟─64b53133-51e6-400e-9047-c2c18d1805f5
# ╠═4b2e7984-023e-49f2-83b3-3cc01819e239
# ╠═321c78d1-d175-43de-8069-cdf7cedaf20c
# ╟─fb1c8a2f-babb-4b93-be81-a1fe843fbfa5
# ╠═c56a1f71-ecdb-41ef-a614-379b32a270db
# ╟─810898ac-546f-4832-9922-8eb61f154325
# ╠═76b59e6b-b1b9-49e3-8fea-fc8a15238c99
# ╠═1d7dce85-6652-4743-9ecc-46145a00bc70
# ╟─b14c1634-2b8b-4ad3-ad3a-2ea44c071471
# ╠═4585ea88-51e7-42c6-a014-e8097db09695
# ╠═11834ef0-f9a4-49a5-92ff-4ffe2c217912
# ╟─d66efb8a-03f5-4879-9ae4-5242d8288d67
# ╠═768ea165-199f-4106-bcab-d476ebf7dea6
# ╟─768bfdcc-11b6-4fd9-9b0b-42035ef89664
# ╠═78ed1373-b51c-43c2-9227-9d9785848d69
# ╟─04ea591e-bd74-4926-82a6-d6a09f242a71
# ╠═9865e00a-0c32-419e-a8cc-0b4cb35b7031
# ╟─4121fd34-e7ae-4a04-a32f-69660e3b85ca
# ╠═a306cd12-7218-4767-8650-7034dcb6e303
# ╠═9f457e6d-ce3e-41fa-b3ce-eb3c4993286e
# ╟─f646fd23-beeb-48d8-80d2-503442371d83
# ╠═11d02c6a-a28b-4b6e-9b2a-83ea20e01c28
# ╠═64ee1549-b481-49a1-88d6-1961b9b82a91
# ╟─078dbda5-3eec-4c2b-9c74-1349511cdb82
# ╟─55d77b19-e600-4ce8-9dc1-8d2458c99da2
# ╠═db5d41fc-ea93-4abc-9efa-b232ef7f37e2
# ╠═9a379335-9430-4fa2-9b84-15e192ace090
# ╠═34d06f6b-5e8e-4823-98e7-e3707feb528d
# ╠═697828f5-2e3b-4c25-86e0-cc97cfdb8432
# ╟─e47e0b0a-377a-451e-93f8-d4430706ef19
# ╠═57f8919a-aadd-4e3b-9a24-cdca7870caa9
# ╠═b700e215-3ba4-45e7-bea2-89f6f7ff73f7
# ╠═175d7472-4a23-4715-a39e-4a434cac46b1
# ╠═dd861e69-2c98-49e1-9970-f5e898048d04
# ╟─7f0583b7-8d5c-4660-ad47-5f64aa6f57bb
# ╠═7a2baa7c-af89-478f-99ee-f5e9fca9f871
# ╠═fe402f42-0d4a-46fe-ad6d-57eb578d2cf4
# ╠═cc982bfb-8147-4e69-be5c-c75c053e8e33
# ╟─36305f0a-a02d-4ba0-8a4b-12a645a6cf39
# ╠═37396c37-0291-4ce6-92c6-defb547f94f5
# ╟─89143850-3a09-4407-abf6-066087d180ef
# ╟─521a5ff0-44b6-4065-8fb6-6dd3c1b1689a
# ╟─551603af-23fc-49b6-a3c0-f15c199452dc
# ╠═4cd08b52-0848-458a-8b92-9909015d8edd
# ╠═f0edaa1c-cd6d-4722-9d3f-0f6328095c48
# ╠═6b313419-288a-4f20-8453-e61fa463d225
# ╠═4149ce52-2e86-4c6b-92ca-d66fd9b8348d
# ╠═672ee9e6-1928-4eab-b2d9-b34179cf99fc
# ╠═8515d621-6221-4711-a486-0b9c3d3f6be4
# ╟─aae002c8-7160-40f2-a01b-dc1192dbd6d2
# ╠═8a565363-d3bb-423d-ab43-642f781925e4
# ╠═af195cbb-058e-429f-a2a8-0635b827d54f
# ╠═818de785-c18d-4b35-9a16-4453b7c9ade7
# ╠═5d477221-6c4a-409a-98aa-4373b7e05bce
# ╟─a1eaec8f-3527-43f6-b3e6-360c64b156e5
# ╠═f057d9db-8c17-49a9-a910-c6c9553e9c93
# ╟─dafb5973-8bf9-4436-b7e2-cd9fe29e1d2c
# ╠═bbf11fd6-4a18-4f4e-9022-6801ec387df9
# ╠═c8c3b967-ddb1-406b-8393-3fee26054d93
# ╠═d3d0a9d6-4540-4d87-9254-489a9335a322
# ╠═0c593b1c-e74a-46ed-8f66-47d1456b3636
# ╟─b3ab790a-e748-4634-816a-ed7acd3f31d3
# ╠═c85e44f0-600a-4491-ba95-bf5b38d83578
# ╠═e96c765f-5f5f-41db-8191-1a0d444bcde2
# ╠═8e4eb2c7-20d3-4aa0-8c09-34fdb8293a9c
# ╠═40ae0bd6-0fe5-4537-9cee-9485a8b741f3
# ╠═248c304c-de63-4523-b334-e201ed5f07f2
# ╠═0462c099-18ee-415d-b06a-7bb8eafa360f
# ╠═331ceecb-53e4-41f9-b8de-2aece6d1eea8
# ╠═25319673-99ff-436d-bf5d-34c540f4faf1
# ╠═2a5009df-5e84-4b50-b819-c19333eb4a8b
# ╠═3f880785-c0ef-447c-b5ee-db258fc9db16
# ╠═6b847bf5-87df-4e87-8b1c-fb9da5da0d54
# ╠═9510e0c5-c658-4358-871f-3c433a8f183d
# ╠═907a813d-945d-40b2-9262-cc2975b8a436
# ╠═2eded938-c85c-4bf4-95aa-8d8b02c6cbbb
# ╠═fb1ad4f2-5b02-4080-bb0f-94a1bf918562
# ╠═cd139cca-0b06-40da-b2b7-f49e3b3bcd52
# ╟─113a69c4-ada0-4e64-b910-64434f191a7e
# ╠═cd2ebd41-45d3-4dd0-aba9-3f42070663b4
# ╠═479ddea4-9370-408e-8000-1bf02d873a02
# ╠═b484e82c-9960-43e2-b90b-f08dfd36214e
# ╠═d5ef8924-6ad4-4aeb-9a62-387e1274de66
# ╠═94038e3a-3d01-439c-a01c-2c7aacf16147
# ╠═f2678b18-3819-421a-9b66-b698e132a1d9
# ╠═ce55e76c-f309-4f34-a8fc-2db2af7d030a
# ╠═abdf8da8-be19-4428-9230-8bb46f8d03e5
# ╠═17b7c372-269c-44b8-8d7b-048ce10e64f6
# ╠═cfd3c52f-8b2c-48ab-86f3-2b8835abddd0
# ╠═ae59ec71-6362-4dd9-bfc7-74fb8c52f777
# ╟─d28ba6a0-e1b1-43ca-8744-976da374c98d
# ╠═6f9c7c9c-5992-49bb-8090-98a03203414a
# ╠═37008e44-c0fc-42e9-851a-c4e4a3a522d5
# ╟─9e4e4f8d-1d34-4c35-b814-b8d6708bf2ab
# ╟─672fb2ff-5782-4411-85d0-ca83506372c8
