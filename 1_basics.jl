### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ a025f4ac-9f39-4d05-9e0f-c12b9145d7c6
begin
	using PlutoUI
	# import all names (e.g. TableOfContents) exported by "PlutoUI" into the global namespace
	TableOfContents(aside=true)
	
	include("utils.jl")  # load local source file
	using .Utilities  # "utils.jl" has a module called "Utilities"
end

# ╔═╡ a9561c08-2b07-4590-b901-d9cbd60355ee
begin
	using AbstractTrees
	AbstractTrees.children(d::DataType) = subtypes(d)
	print_tree(Number)
end

# ╔═╡ a4e1bde7-2de3-4df9-8dc3-f25aafac7dfd
begin
	using Statistics
	names(Statistics)
end

# ╔═╡ 0939489d-79d2-4c1a-9841-17d9ae448d94
md"# An Introduction to Julia Programming"

# ╔═╡ b4cb3a82-d740-4d02-b0f4-f18ec9500b4f
md"""
!!! tip "Tip"
	Open the **Live Docs** at the bottom right and click a symbol to read its documentation. 
"""

# ╔═╡ 4efa23f3-e705-469e-8e82-fb6d0e4589a3
md"## Basic Calculation"

# ╔═╡ 52ab5184-2f0f-11ef-3034-8fd6a5c8a2cb
(1 + 2) * 3 ^ 2

# ╔═╡ 0f63f358-310c-4475-a17b-6376ce26f903
@show log2(4) log(ℯ) log10(1e4) log(4, 1024) sqrt(4) exp(4);

# ╔═╡ a8d9989e-79df-42c5-aaf2-7212ad26a9da
md"The `log` function has a method `log(x)` that calculates the *natural logarithm* of `x` (base is ℯ), and another method `log(b, x)` that uses the given base `b`. This is an example of the `multiple dispatch` mechanism of Julia."

# ╔═╡ 3ae5a286-cc9d-4837-a6de-c79bad078df4
z = exp(im * π)

# ╔═╡ 662c94fb-a2b9-4970-86a8-5f952d118309
z == -1 ? print("Hurray!") : print("Why?")  # ternary operator

# ╔═╡ 86e687c7-4052-4d11-9ef9-7ac6b59cb8ae
md"It's a common problem of floating point arithmetic. [All positional (base-N) number systems share this problem with precision](https://stackoverflow.com/questions/588004/is-floating-point-math-broken?rq=1)."

# ╔═╡ cdac9eca-48a6-44dd-9926-a1e0959c2c31
(@show 0.1 + 0.2) == 0.3

# ╔═╡ 5c64daca-361a-4c3b-92e0-b179c834a63e
z ≈ -1 ? print("Hurray!") : print("Why?")  # \approx => ≈, equivalent to isapprox()

# ╔═╡ 79dd50f1-bd99-4384-b691-4bdb73096161
let θ = rand(), z = exp(im * θ)  # let...end binds variables locally
	# θ is a random float number in [0, 1)
	@show θ abs(z) angle(z) cos(θ) real(z) sin(θ) imag(z)
end

# ╔═╡ b541204e-3054-4504-b8f4-913209f19913
md"## Control Structures"

# ╔═╡ efe4fd6a-b130-4f95-a95c-b0473022ffe9
md"**Error Handling:**"

# ╔═╡ 76f1b9df-46e4-4920-b62d-f6e802f9a8ec
try  # try running this cell for multiple times to see different results
	let a = rand((1, true, false))  # a random item in this tuple
		@show a
		a && θ || error("a is false")  # && and || have short-circuit evaluation
	end
catch e
	@show e
	if e isa TypeError  # a == 1
		throw(ErrorException("1 is not Boolean!"))
	elseif e isa UndefVarError  # a == true
		println("$(e.var) is not defined!")
	else  # a == false
		rethrow()
	end
end

# ╔═╡ e68d2aa6-f69b-47ad-9319-44a91d678097
Tz = typeof(z)

# ╔═╡ 12e18c54-1d43-4f23-89e1-f578f3f34cb0
md"ComplexF64 is a *composite* type as it is a collection of named fields."

# ╔═╡ 536ddaff-814b-4dd9-bbac-27008527f43c
md"**For Loop:**"

# ╔═╡ 8af405f5-01c3-45e3-8451-3e3ac287466f
for s in fieldnames(Tz)
	println("z.", s, " = ", getfield(z, s))
end

# ╔═╡ a292b548-502b-455b-9ed8-15843b0930dc
fieldtypes(Tz)

# ╔═╡ 4ec99d1e-bbb8-4dde-9276-6532bf4eeb64
fieldnames(Float64)  # primitive type (no fields)

# ╔═╡ ccd1d5e8-88b6-40af-a850-e16deb9718e9
md"**While Loop:**"

# ╔═╡ 0c05213d-5390-40e0-8c92-676774067e28
let T = Float64
	while T != Any  # Any is union of all types
		T = supertype(T)
		@show T
		if z isa T  # whether z is of type T
			@assert Tz <: T  # whether Tz is a subtype of T
			# @assert is a macro that throws an AssertionError if the cond is false
			println("$Tz <: $T")  # string interpolation
		end
	end
end

# ╔═╡ 01e35e8d-cb99-45fb-8770-2e23f3ec7c7c
md"""
!!! danger "Task"
	Write a loop that for each field of `z`, prints whether its type is primitive or composite.
"""

# ╔═╡ 8d8c7053-1a23-485f-90c5-2db999f7581d
md"(Show this cell for a sample solution)"
# for T in fieldtypes(typeof(z))
# 	println(fieldtypes(T) == () ? "primitive" : "composite")
# end

# ╔═╡ b5b168db-b896-41bb-afeb-08e328d7b28e
md"## Function Definition"

# ╔═╡ 8bed8fcc-1ea4-414f-a188-910b74632085
# arguments before the ; are positional, after are keyword
# there can be defaults in both categories
# anything without a default must be assigned when the function is called
# ... before the ; accepts any number of positional arguments
# ... after the ; accepts any keyword arguments
# the names args and kwargs are conventional for these extra arguments
# the string above the function definiction is documentation

"My custon function."
function func(a, b=0, args...; c, d=1, kwargs...)
	@show a b args c d kwargs
	println()
	return a
end

# ╔═╡ 40c36c39-b3b7-4c12-a116-7c0ddb079085
begin
	func('a', 2, 3, 4, c=3, e=7)
	func(1, c=7)
end

# ╔═╡ 4a851df6-3894-42a9-9acd-eb25a56f5535
md"### Higher Order Functions"

# ╔═╡ 1396345b-8abf-48ac-8bfa-6c641a395c2c
begin
	double(f) = x -> f(f(x))  # a simpler way of function definition
	inc = x -> x + 1  # anonymous function
	@show double(inc)(0)
	
	triple(f) = f ∘ f ∘ f  # \circ -> ∘ (function composition)
	@show triple(inc)(0)
	
	nfold(f, n) = reduce(∘, Iterators.repeated(f, n))
	@show nfold(triple, 4)(inc)(0)  # applies `inc` for 3^4 times
end

# ╔═╡ 41f7af8e-28b2-4216-aac6-2827dda5e6db
md"Estimate π using fixed point iteration:"

# ╔═╡ ef02cbb9-11af-49e9-a996-f2c44c9c1191
1 |> nfold(x -> x + sin(x), 5)
# `a |> f` (pipe operator) is equivalent to `f(a)`

# ╔═╡ c7bff4de-88ca-4264-bf83-4a2f08728395
md"### Recursive Functions"

# ╔═╡ 82b29e45-827c-44b9-9225-17ae863c34bd
factorial(4)

# ╔═╡ 0fcdc4f2-f369-4d09-ac6c-c42d2a8172ce
factorial(30)

# ╔═╡ eb40f8e1-4e03-4099-bba0-29e0ea43ed79
fact_big(n::Integer) = n < 2 ? big(1) : n * fact_big(n - 1)

# ╔═╡ 161587eb-5676-4154-9dff-abe0868efc03
fact_big(30)

# ╔═╡ c827b4d5-67fe-48df-aad2-15af280b7050
md"See [Arbitrary Precision Arithmetic](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic) in the Julia documentation."

# ╔═╡ 360d4228-a59a-4915-bf23-dd5537274d78
md"""
!!! danger "Task"
	Implement the `fib(n)` function to calculate the `n`-th Fibonacci number.
"""

# ╔═╡ 94f505a0-c146-4ce6-8274-4d84edfd0abe
md"""
!!! hint
	Recall that `fib(0) = 0`, `fib(1) = 1`, and `fib(n) = fib(n-1) + fib(n-2)`.
"""

# ╔═╡ 0abdd55f-f7aa-4896-9f6d-f8c2ea638acf
function fib(n)
	missing  # replace `missing` with your answer
end

# ╔═╡ a338fd44-7ccb-4607-bdc7-01ada39f02b9
md"(Show this cell for a sample solution)"
# sum(n...) = reduce(+, ns)

# ╔═╡ 8116c816-ab72-4415-94bf-a66ad7f52d2d
function fib_c(n)
    z = BigInt()
    @ccall "libgmp".__gmpz_fib_ui(z::Ref{BigInt}, n::Culong)::Cvoid
	# Call the fib function in the C library libgmp
    return z
end

# ╔═╡ bd7bed63-714c-417c-822c-2c07419d59db
let
	result = all(fib(n) == fib_c(n) for n in 0:30)
	if ismissing(result)
		Utilities.still_missing()
	elseif result
		Utilities.correct()
	else
		Utilities.keep_working()
	end
end

# ╔═╡ 13104a6c-0eb7-42d7-961d-addc55f06588
md"## Type System"

# ╔═╡ 002bd083-00d2-4fd6-965f-9415d85f23f6
subtypes(Integer), supertypes(Integer)

# ╔═╡ e9f8aee3-aa16-446b-aeec-8d1aae6e7169
Union{Int, Integer}

# ╔═╡ 18aab5fb-7add-4ada-b42e-2bc62968d6bc
isabstracttype(Integer)

# ╔═╡ 0c4a6998-8863-404e-96c2-952df70839ab
isconcretetype(Int64)

# ╔═╡ 2e034e29-8755-43d5-b557-d247df23f50e
md"### Define Custom Types"

# ╔═╡ e3f7a77a-8c9e-4f15-af47-551fd959b2a6
abstract type Distribution end

# ╔═╡ f3b4eba4-5471-441e-b199-69fd07f528e2
md"A piece of Julia code is called 'type-stable' if all input and output variables have a concrete type, either by explicit declaration or by inference from the Julia compiler. Type-stable code will run much faster as the compiler can generate statically typed code and optimize it at compile-time."

# ╔═╡ 2e6521be-ff66-47a9-8c19-68216cb62f3d
md"We can see that the length of the LLVM bitcodes generated from a piece of type-stable Julia code is much shorter than its type-instable version. The following example will compare their performance."

# ╔═╡ 149a64ba-6d5b-4416-bc2d-8e1ae897c71d
function probability(P::Distribution, lo::Float64, hi::Float64; step=1e-6)
	step * sum(P(x) for x in lo:step:hi)
end

# ╔═╡ 7b6e1d43-c72c-4bd9-b493-838b05e845c4
md"## Collection Data Types"

# ╔═╡ 69283b2e-bd47-4c3c-890f-677b253183e7
v = [1, 2, 3, 4, 5]

# ╔═╡ d7186b34-117c-4a11-8907-91766a038425
v[1]  # index starts from 1 in Juila

# ╔═╡ 7434577e-3147-4128-8f58-81ef081dd10a
v[1:2], v[1:2:end], v[end-1:-2:1]  # unlike Python, indices cannot be omitted

# ╔═╡ a2c92fca-fbab-4396-b472-a53d7a858abe
typeof(v)

# ╔═╡ 0f3b3f22-89f3-491d-be29-57438d83f4cd
length(v)

# ╔═╡ 2c0b579b-302c-458e-bfb0-75ce768de5bd
v .* 2  # broadcasting

# ╔═╡ 28b55fda-da32-4b71-a18e-fabec0c7fb73
2v

# ╔═╡ b3321c01-db3d-42ed-9ea7-142e8773bc28
sqrt.(v)

# ╔═╡ 760ff5fd-689b-4afe-9336-cc480fb6b486
let r = 1:2:5
	@show (3v)[r]
	collect(r)  # convert to array
end

# ╔═╡ 4f62d53f-11bb-4e53-b759-d6f49eec5cd4
let a = Array{Float64}(undef, 2, 3)  # initialize a 2x3 Matrix of Float64s
	for i in 1:2, j in 1:3
		a[i, j] = i * j
	end
	a
end

# ╔═╡ 3cfce228-b634-4e31-b3f3-ddadb6c7a53d
Array{Int, 2}

# ╔═╡ 952db525-9d54-4b56-a09f-3014a9ca9293
[i * j for i in 1:2, j in 1:3]  # array comprehension

# ╔═╡ 6b3a83eb-e316-46b5-a097-233145ab1bcc
[1 2 3
 5 6 4
 9 7 8]  # or [1 2 3; 5 6 4; 9 7 8]

# ╔═╡ d02b8c20-6e43-435c-ba9f-870b1bb5fae9
zeros(3, 3)

# ╔═╡ b5eb64a4-6572-405f-bed4-7e483f6e50e5
rand(2, 2, 2)

# ╔═╡ 8bc03ce0-2fe3-45ca-9c1a-9bd2a98bc41e
A = rand(ComplexF64, (3, 2))

# ╔═╡ d1ca8fb0-580f-4625-aba3-dd18e054ee48
size(A), size(A, 1)

# ╔═╡ 9fc3a808-5a53-44e9-9f45-5939d9064c30
A[1:2:end, :]

# ╔═╡ 1603ceb6-e8a8-486e-8bff-c721b57ab2eb
reshape(A, :, 3)  # same as A.reshape(-1, 3) in Python

# ╔═╡ 8ea9ecaf-6d66-4e57-8606-e79fdc8415e5
[A; A]  # concat vertically (same as vcat(A, A))

# ╔═╡ 9bb81880-067c-4bde-a12f-c37eb4be2846
[A A]  # concat horizontally (same as hcat(A, A))

# ╔═╡ 12008adf-5162-484c-af6b-30b2d43f46b5
sum(abs2, A, dims=2)  # map abs2 to A first, then sum along the 2nd axis

# ╔═╡ 65f92119-b389-491c-b809-fab91636c53a
mean(A)

# ╔═╡ 9cc9456e-fdac-4f56-89c4-e3ddf8a5f0af
mean(A, dims=1)

# ╔═╡ 47aae1fe-5c76-4f47-ab94-d8c784c59c35
methods(mean)

# ╔═╡ 6b95a054-c3f7-4777-bbcd-ccbd12741234
@which mean(1:100)

# ╔═╡ 2db6eff5-e28a-4594-a13b-3ed429d47140
methodswith(StepRange)

# ╔═╡ 26f43214-3b99-4c99-9512-398a28f9ae0a
md"""
!!! danger "Task"
	Generate a 1000×2 random matrix of float numbers from the normal distribution ``N(0, 1)`` and assign it to `Q`.
"""

# ╔═╡ f942be94-a50f-4bd5-9987-ed0124531dd3
md"""
!!! hint
	Use the function `randn`.
"""

# ╔═╡ b226106d-6f21-4d72-951c-c4d9d01cbbcb
Q = missing  # replace missing with your answer

# ╔═╡ aa0c8fec-254b-4805-bf07-b1ce7266685c
begin
	md"(Show this cell for a sample solution)"
	# Q = randn(1000, 2)
end

# ╔═╡ 03c85588-2237-4d17-9755-bd3386f8e348
md"""
!!! danger "Task"
	Scale each column of Q by the values in the second row of `S`, and then shift each column of Q by the first row of `S`. These operations should be in-place.
"""

# ╔═╡ 24077fc9-4d06-4b80-91be-321a7bb0fe5c
S = [2.0 -1.0; 4.0 0.2]

# ╔═╡ 50cb4c19-1d76-4844-8bc7-bc564aa34ab8
begin
	md"(Show this cell for a sample solution)"
	# @. Q = S[2, :]' * Q + S[1, :]'  # @. converts all operators to a broacasting one
end

# ╔═╡ 8615c4ca-7e2b-49fb-bb0f-078347a7c56b
md"""
!!! danger "Task"
	Calculate the `mean` and `std` of each column of `Q` and concatenate them vertically. Compare the result with `S`.
"""

# ╔═╡ be7f3b8d-70e6-4ec3-a98f-07fbe17fb06a
begin
	md"(Show this cell for a sample solution)"
	# [mean(R, dims=1); std(R, dims=1)]
end

# ╔═╡ 66cae8d2-8e20-4b1e-9dae-e120eee4d944
md"## Linear Algebra"

# ╔═╡ 5af22ae0-effd-4589-bd1f-d375299b6848
M = rand(3, 3)

# ╔═╡ 5ee4f31b-ebae-4d8f-8ccc-6df671de6965
begin
	using LinearAlgebra
	rank(M), tr(M), det(M), diag(M)
end

# ╔═╡ 50c86554-ff09-4e4a-94e8-0f30b83e8655
@show 3+4 3*4 3/4 3÷4 4%3 3^4 3<4 3>=4 3==4 3!=4;
# @show is a macro that prints expressions and their evaluated values

# ╔═╡ 0f2aff9d-778b-4a08-9c33-c1866279c686
begin
	abstract type AbstractNormal <: Distribution end
	(p::AbstractNormal)(x) = exp(-0.5((x-p.μ)/p.σ)^2) / (p.σ * √(2π))  # \sqrt => √
end

# ╔═╡ 48e93319-299b-40b9-bbf9-09d18d683c9c
struct NormalUntyped <: AbstractNormal
	μ
	σ
end

# ╔═╡ fa1283d5-b3d5-46d4-a34c-4cddc32ab284
fieldtypes(NormalUntyped)

# ╔═╡ 322ea469-2961-46b0-a93c-20e2c8f94328
p1 = NormalUntyped(0, 1)

# ╔═╡ ec5238e4-f445-491c-bd14-8e1aba59049f
p1(0)

# ╔═╡ cfeb3928-cc2f-47a3-8a9b-e17eabd79a33
@code_warntype p1(0)

# ╔═╡ c6739f52-f87f-4bef-8c32-ce3ec4942342
@code_llvm p1(0)

# ╔═╡ d00e9d96-59c7-4bd6-9667-340505d5ed5f
@time probability(p1, -1.96, 1.96)

# ╔═╡ 035f9794-43ea-4e19-860c-a66fd0ea1a14
struct Normal <: AbstractNormal
	μ :: Float64
	σ :: Float64
end

# ╔═╡ 57f30a3c-7d28-4819-958a-bf1859d6947c
p2 = Normal(0, 1)

# ╔═╡ 024aa7d5-a569-4639-851f-b7d491855202
@code_warntype p2(0)

# ╔═╡ f640df71-ae15-4b67-a30e-c806ea532a19
@code_llvm p2(0)

# ╔═╡ 8e8a900f-1d6c-4d65-afda-b03e64f3c9c8
@time probability(p2, -1.96, 1.96)

# ╔═╡ 76d2cfde-bdd8-4e45-83dd-92d3c651691f
struct NormalParametric{T} <: AbstractNormal
	μ :: T
	σ :: T
end

# ╔═╡ 1e36bd1d-cb83-4e48-a5dc-f88bf04636ca
p3 = NormalParametric(0f0, 5f-1)  # float32 version of 8e-1

# ╔═╡ b088c77f-9732-4c63-88f9-9bcd911e461c
@code_warntype p3(0)

# ╔═╡ af5fffbd-baf5-46e4-b285-3a98a5d01e55
@time probability(p3, -1.96, 1.96)

# ╔═╡ 8efda77f-e3d5-4866-8b64-159b6c3a6114
transpose(A)

# ╔═╡ d9f9542f-8d4f-4c0c-b4ea-986eefc07636
A'  # complex conjugate followed by transpose

# ╔═╡ 17eeffee-701d-4251-aca7-308e456487da
let B = reshape(A, 2, 3), C = A'  # all share the same underlying data
	B[1, 2] = NaN
	C[1, 2] = NaN
	A, B, C
end

# ╔═╡ 493a6c95-3820-43aa-8e6c-939757aecf2b
M - I  # I is identity matrix

# ╔═╡ 6287eddc-9b35-489e-b584-8197c09cb228
let b = [1, 2, 3]
	x = inv(M) * b  # or M \ b
	M * x
end

# ╔═╡ 3f6fbfd0-b35a-4af9-86cd-55d7e4188301
let eig = eigen(M)
	λ = @show eig.values
	V = @show eig.vectors
	@assert M * V ≈ λ' .* V
end

# ╔═╡ bbd257e4-63ef-4024-bbe7-b57213d10e1f
let F = svd(M)
	U = @show F.U
	S = @show F.S
	V = @show F.V
	@assert U * U' ≈ V * V' ≈ I  # chained relations
	@assert U * Diagonal(S) * V' ≈ M
end

# ╔═╡ 8bc7e78b-ff6d-4553-b327-f03d21651121
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

# ╔═╡ fbd9a83b-17b4-47db-a46e-e7a9037b9090
@show_all let a = -6:3:6, b = [1, 2, 3], c = (4, 5, 6)
	length(a)
	maximum(a)
	max(b...)
	sum(b)
	reverse(b)

	zip(a, b, c) |> collect
	count(iseven, a)
	findfirst(iszero, a)
	map(-, a, b)
	mapreduce(-, +, a, b)  # reduce the result of map(-, a, b) by the + operator
	
	d = Dict(zip(b, c))
	
	push!(b, 5, 4)
	sort!(b)
	deleteat!(b, 1)
	
	d[1], d[2]
	d[4] = 7
	delete!(d, 1)
	keys(d), values(d)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractTrees = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
AbstractTrees = "~0.4.5"
PlutoUI = "~0.7.59"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "66aea074862d03f5685a7fafc697db126e68a254"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

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
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

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

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

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

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─0939489d-79d2-4c1a-9841-17d9ae448d94
# ╠═a025f4ac-9f39-4d05-9e0f-c12b9145d7c6
# ╟─b4cb3a82-d740-4d02-b0f4-f18ec9500b4f
# ╟─4efa23f3-e705-469e-8e82-fb6d0e4589a3
# ╠═52ab5184-2f0f-11ef-3034-8fd6a5c8a2cb
# ╠═50c86554-ff09-4e4a-94e8-0f30b83e8655
# ╠═0f63f358-310c-4475-a17b-6376ce26f903
# ╟─a8d9989e-79df-42c5-aaf2-7212ad26a9da
# ╠═3ae5a286-cc9d-4837-a6de-c79bad078df4
# ╠═662c94fb-a2b9-4970-86a8-5f952d118309
# ╟─86e687c7-4052-4d11-9ef9-7ac6b59cb8ae
# ╠═cdac9eca-48a6-44dd-9926-a1e0959c2c31
# ╠═5c64daca-361a-4c3b-92e0-b179c834a63e
# ╠═79dd50f1-bd99-4384-b691-4bdb73096161
# ╟─b541204e-3054-4504-b8f4-913209f19913
# ╟─efe4fd6a-b130-4f95-a95c-b0473022ffe9
# ╠═76f1b9df-46e4-4920-b62d-f6e802f9a8ec
# ╠═e68d2aa6-f69b-47ad-9319-44a91d678097
# ╟─12e18c54-1d43-4f23-89e1-f578f3f34cb0
# ╟─536ddaff-814b-4dd9-bbac-27008527f43c
# ╠═8af405f5-01c3-45e3-8451-3e3ac287466f
# ╠═a292b548-502b-455b-9ed8-15843b0930dc
# ╠═4ec99d1e-bbb8-4dde-9276-6532bf4eeb64
# ╟─ccd1d5e8-88b6-40af-a850-e16deb9718e9
# ╠═0c05213d-5390-40e0-8c92-676774067e28
# ╟─01e35e8d-cb99-45fb-8770-2e23f3ec7c7c
# ╟─8d8c7053-1a23-485f-90c5-2db999f7581d
# ╟─b5b168db-b896-41bb-afeb-08e328d7b28e
# ╠═8bed8fcc-1ea4-414f-a188-910b74632085
# ╠═40c36c39-b3b7-4c12-a116-7c0ddb079085
# ╟─4a851df6-3894-42a9-9acd-eb25a56f5535
# ╠═1396345b-8abf-48ac-8bfa-6c641a395c2c
# ╟─41f7af8e-28b2-4216-aac6-2827dda5e6db
# ╠═ef02cbb9-11af-49e9-a996-f2c44c9c1191
# ╟─c7bff4de-88ca-4264-bf83-4a2f08728395
# ╠═82b29e45-827c-44b9-9225-17ae863c34bd
# ╠═0fcdc4f2-f369-4d09-ac6c-c42d2a8172ce
# ╠═eb40f8e1-4e03-4099-bba0-29e0ea43ed79
# ╠═161587eb-5676-4154-9dff-abe0868efc03
# ╟─c827b4d5-67fe-48df-aad2-15af280b7050
# ╟─360d4228-a59a-4915-bf23-dd5537274d78
# ╟─94f505a0-c146-4ce6-8274-4d84edfd0abe
# ╠═0abdd55f-f7aa-4896-9f6d-f8c2ea638acf
# ╟─a338fd44-7ccb-4607-bdc7-01ada39f02b9
# ╟─8116c816-ab72-4415-94bf-a66ad7f52d2d
# ╟─bd7bed63-714c-417c-822c-2c07419d59db
# ╟─13104a6c-0eb7-42d7-961d-addc55f06588
# ╠═002bd083-00d2-4fd6-965f-9415d85f23f6
# ╠═e9f8aee3-aa16-446b-aeec-8d1aae6e7169
# ╠═18aab5fb-7add-4ada-b42e-2bc62968d6bc
# ╠═0c4a6998-8863-404e-96c2-952df70839ab
# ╠═a9561c08-2b07-4590-b901-d9cbd60355ee
# ╟─2e034e29-8755-43d5-b557-d247df23f50e
# ╠═e3f7a77a-8c9e-4f15-af47-551fd959b2a6
# ╠═0f2aff9d-778b-4a08-9c33-c1866279c686
# ╠═48e93319-299b-40b9-bbf9-09d18d683c9c
# ╠═fa1283d5-b3d5-46d4-a34c-4cddc32ab284
# ╠═322ea469-2961-46b0-a93c-20e2c8f94328
# ╠═ec5238e4-f445-491c-bd14-8e1aba59049f
# ╠═cfeb3928-cc2f-47a3-8a9b-e17eabd79a33
# ╟─f3b4eba4-5471-441e-b199-69fd07f528e2
# ╠═c6739f52-f87f-4bef-8c32-ce3ec4942342
# ╠═035f9794-43ea-4e19-860c-a66fd0ea1a14
# ╠═57f30a3c-7d28-4819-958a-bf1859d6947c
# ╠═024aa7d5-a569-4639-851f-b7d491855202
# ╠═f640df71-ae15-4b67-a30e-c806ea532a19
# ╠═76d2cfde-bdd8-4e45-83dd-92d3c651691f
# ╠═1e36bd1d-cb83-4e48-a5dc-f88bf04636ca
# ╠═b088c77f-9732-4c63-88f9-9bcd911e461c
# ╟─2e6521be-ff66-47a9-8c19-68216cb62f3d
# ╠═149a64ba-6d5b-4416-bc2d-8e1ae897c71d
# ╠═d00e9d96-59c7-4bd6-9667-340505d5ed5f
# ╠═8e8a900f-1d6c-4d65-afda-b03e64f3c9c8
# ╠═af5fffbd-baf5-46e4-b285-3a98a5d01e55
# ╟─7b6e1d43-c72c-4bd9-b493-838b05e845c4
# ╠═69283b2e-bd47-4c3c-890f-677b253183e7
# ╠═d7186b34-117c-4a11-8907-91766a038425
# ╠═7434577e-3147-4128-8f58-81ef081dd10a
# ╠═a2c92fca-fbab-4396-b472-a53d7a858abe
# ╠═0f3b3f22-89f3-491d-be29-57438d83f4cd
# ╠═2c0b579b-302c-458e-bfb0-75ce768de5bd
# ╠═28b55fda-da32-4b71-a18e-fabec0c7fb73
# ╠═b3321c01-db3d-42ed-9ea7-142e8773bc28
# ╠═760ff5fd-689b-4afe-9336-cc480fb6b486
# ╠═4f62d53f-11bb-4e53-b759-d6f49eec5cd4
# ╠═3cfce228-b634-4e31-b3f3-ddadb6c7a53d
# ╠═952db525-9d54-4b56-a09f-3014a9ca9293
# ╠═fbd9a83b-17b4-47db-a46e-e7a9037b9090
# ╠═6b3a83eb-e316-46b5-a097-233145ab1bcc
# ╠═d02b8c20-6e43-435c-ba9f-870b1bb5fae9
# ╠═b5eb64a4-6572-405f-bed4-7e483f6e50e5
# ╠═8bc03ce0-2fe3-45ca-9c1a-9bd2a98bc41e
# ╠═d1ca8fb0-580f-4625-aba3-dd18e054ee48
# ╠═9fc3a808-5a53-44e9-9f45-5939d9064c30
# ╠═1603ceb6-e8a8-486e-8bff-c721b57ab2eb
# ╠═8ea9ecaf-6d66-4e57-8606-e79fdc8415e5
# ╠═9bb81880-067c-4bde-a12f-c37eb4be2846
# ╠═12008adf-5162-484c-af6b-30b2d43f46b5
# ╠═8efda77f-e3d5-4866-8b64-159b6c3a6114
# ╠═d9f9542f-8d4f-4c0c-b4ea-986eefc07636
# ╠═a4e1bde7-2de3-4df9-8dc3-f25aafac7dfd
# ╠═65f92119-b389-491c-b809-fab91636c53a
# ╠═9cc9456e-fdac-4f56-89c4-e3ddf8a5f0af
# ╠═47aae1fe-5c76-4f47-ab94-d8c784c59c35
# ╠═6b95a054-c3f7-4777-bbcd-ccbd12741234
# ╠═2db6eff5-e28a-4594-a13b-3ed429d47140
# ╠═17eeffee-701d-4251-aca7-308e456487da
# ╟─26f43214-3b99-4c99-9512-398a28f9ae0a
# ╟─f942be94-a50f-4bd5-9987-ed0124531dd3
# ╠═b226106d-6f21-4d72-951c-c4d9d01cbbcb
# ╟─aa0c8fec-254b-4805-bf07-b1ce7266685c
# ╟─03c85588-2237-4d17-9755-bd3386f8e348
# ╠═24077fc9-4d06-4b80-91be-321a7bb0fe5c
# ╟─50cb4c19-1d76-4844-8bc7-bc564aa34ab8
# ╟─8615c4ca-7e2b-49fb-bb0f-078347a7c56b
# ╟─be7f3b8d-70e6-4ec3-a98f-07fbe17fb06a
# ╟─66cae8d2-8e20-4b1e-9dae-e120eee4d944
# ╠═5af22ae0-effd-4589-bd1f-d375299b6848
# ╠═493a6c95-3820-43aa-8e6c-939757aecf2b
# ╠═6287eddc-9b35-489e-b584-8197c09cb228
# ╠═5ee4f31b-ebae-4d8f-8ccc-6df671de6965
# ╠═3f6fbfd0-b35a-4af9-86cd-55d7e4188301
# ╠═bbd257e4-63ef-4024-bbe7-b57213d10e1f
# ╟─8bc7e78b-ff6d-4553-b327-f03d21651121
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
