### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a025f4ac-9f39-4d05-9e0f-c12b9145d7c6
begin
	using PlutoUI
	# import all names exported by "PlutoUI" into the global namespace
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

# ╔═╡ 3c74c07d-98a5-48b8-bf6c-2a25e85597d5
begin
	using Statistics
	names(Statistics)
end

# ╔═╡ 0939489d-79d2-4c1a-9841-17d9ae448d94
md"# An Introduction to Julia Programming"

# ╔═╡ b4cb3a82-d740-4d02-b0f4-f18ec9500b4f
md"""
!!! tip "Tip"
	Open the **Live Docs** at the bottom right of the page. When you click a symbol, you can read its documentation there. 
"""

# ╔═╡ 4efa23f3-e705-469e-8e82-fb6d0e4589a3
md"## Basic Calculation"

# ╔═╡ 52ab5184-2f0f-11ef-3034-8fd6a5c8a2cb
(1 + 2) * 3 ^ 2

# ╔═╡ 50c86554-ff09-4e4a-94e8-0f30b83e8655
@show 3+4 3*4 3/4 3÷4 4%3 3^4 3<4 3>=4 3==4 3!=4;

# ╔═╡ 1cd98952-cd47-4632-a71a-903f1809d6be
md"`@show` is a macro that prints expressions and their evaluated values. Read more about macros [here](https://docs.julialang.org/en/v1/manual/metaprogramming/#man-macros)."

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
z ≈ -1 ? print("Hurray!") : print("Why?")  # \approx <tab> => ≈

# ╔═╡ 9cde569d-7db4-4c06-8e03-346b32afaa16
md"## Compound expressions and binding"

# ╔═╡ f2496da6-a024-44eb-b1ee-6cd5e213a86a
md"Local binding with `let`"

# ╔═╡ 79dd50f1-bd99-4384-b691-4bdb73096161
let θ = rand(), z = exp(im * θ)  # let...end binds variables locally
	# θ is a random float number in [0, 1)
	@show θ abs(z) angle(z) cos(θ) real(z) sin(θ) imag(z)
end

# ╔═╡ 45737508-e741-445d-86ef-850ab9915039
md"Non-local binding via `begin`"

# ╔═╡ 88c296f4-51c7-4968-84d4-7d7d66288d8c
begin # after executing `a` and `b` will now be in the global scope. 
	a = 1
	b = 2
	a + b
end

# ╔═╡ 0d376fb5-3b55-4607-903b-1a1777f77215
a

# ╔═╡ 5dedb5f1-1e4e-4b47-9e28-46d9d901f6ca
md"begin..end can be treated as an expression: it has a value"

# ╔═╡ 636db2a4-c4f7-49ff-8cd6-9956e15e5f6e
three = begin
	c = 1
	d = 2
	c + d
end

# ╔═╡ b541204e-3054-4504-b8f4-913209f19913
md"## Control Structures"

# ╔═╡ 1423d00d-8d72-4d84-ad47-95131d8b4bad
md"**For Loop:**"

# ╔═╡ 6366c0dd-5ec4-435f-9b1e-27e5215f4cf9
for x in 1:10 # This is a range and we'll come back to it
	print(x)
end

# ╔═╡ 7c5c7303-474f-4214-a26e-ff98bcc1272b
md"**If expression block:**"

# ╔═╡ e94003d4-580e-454e-9330-b35bfe0bfce0
if 1 < 2
 	print("true")
elseif 2 < 5
	print("true alternative")
else
	print("false")
end

# ╔═╡ 70b0e880-be05-4170-98fc-9d0e2ff1df96
md"## Type structure"

# ╔═╡ 808d8684-e857-46b8-835f-6056079b1e77
md"Type signatures (declarations) in the form `binder :: type`"

# ╔═╡ 52f340d8-1242-4a1f-9bca-164816bd1896


# ╔═╡ 6d3abfce-8de4-45aa-9646-940fe329cf7a
md"Function with parameter and return types explicit"

# ╔═╡ d10c97a6-64c7-4adf-bf76-c25e19f7f215
# Function with parameter and return types explicit
function double(x :: Int) :: Int
	x * 2
end

# ╔═╡ 7e773d44-9823-4753-8a4f-ca31ec5afb85
md"We can query the type of an expression using `typeof`"

# ╔═╡ e68d2aa6-f69b-47ad-9319-44a91d678097
Tz = typeof(z)

# ╔═╡ 12e18c54-1d43-4f23-89e1-f578f3f34cb0
md"""ComplexF64 is a *composite* type as it is a collection of named fields. 

The curly brackets in `Complex{Float64}` means `Complex` is a parametric type, and what's inside the following braces (e.g. `Float64`) is its parameter."""

# ╔═╡ 559143b7-a4e1-4532-adde-523ba70e7d36
typeof(1+im)

# ╔═╡ 8af405f5-01c3-45e3-8451-3e3ac287466f
for s in fieldnames(Tz)
	println("z.", s, " = ", getfield(z, s))
end

# ╔═╡ a292b548-502b-455b-9ed8-15843b0930dc
fieldtypes(Tz)

# ╔═╡ 4ec99d1e-bbb8-4dde-9276-6532bf4eeb64
fieldnames(Float64)  # primitive type (no fields)

# ╔═╡ 79305e1d-a394-4247-b459-cd70a1d29213
md"The results above are tuples. Tuple is an `immutable` type, which means its items cannot be modified."

# ╔═╡ c88262d4-ba12-4955-9022-7724909231ee
let t = (1, 2, 3, 4)
	t[1] += 1
end

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

# ╔═╡ 52de688b-58c8-4aa4-8f92-4068488342e7
function identity(x)
	x
end

# ╔═╡ 7ae98bfb-d650-4bf2-b516-edc03ea68cde
identity(42)

# ╔═╡ 8b4fcefa-d53a-461a-a347-38c0acbf73ac
function early_return(x)
	return x
	42
end

# ╔═╡ b0e25850-9c6b-4898-8bcd-4a4cd4d2a1cb
early_return(100)

# ╔═╡ 8bed8fcc-1ea4-414f-a188-910b74632085
# arguments before the ; are positional, after are keyword
# there can be defaults in both categories
# anything without a default must be assigned when the function is called
# ... before the ; accepts any number of positional arguments
# ... after the ; accepts any keyword arguments
# the names args and kwargs are conventional for these extra arguments
# the string above the function definiction is documentation

"A function demonstrating the syntax of input arguments."
function func(a, b=0, args...; c, d=1, kwargs...)
	@show a b args c d kwargs
	println()
	return a
end

# ╔═╡ 40c36c39-b3b7-4c12-a116-7c0ddb079085
begin  # begin...end groups code into a block without creating a local scope
	func('a', 2, 3, 4, c=3, e=7)
	func(1, b=2, c=7)
end

# ╔═╡ 4a851df6-3894-42a9-9acd-eb25a56f5535
md"### Higher Order Functions"

# ╔═╡ 1396345b-8abf-48ac-8bfa-6c641a395c2c
begin
	inc = x -> x + 1  # anonymous function
	
	double(f) = x -> f(f(x))  # a shorthand of `function double(f) ... end`
	@show double(inc)(0)
	
	triple(f) = f ∘ f ∘ f  # \circ <tab> => ∘ (function composition)
	@show triple(inc)(0)

	@show double(triple)(double)(inc)(0)  # 2 ^ (3 ^ 2)
end

# ╔═╡ ef02cbb9-11af-49e9-a996-f2c44c9c1191
begin
	nfold(f, n) = reduce(∘, Iterators.repeated(f, n))  # generalize double and triple
	est_pi = nfold(x -> x + sin(x), 5)
	println(est_pi(1))
	md"(Optional) Estimate π using fixed point iteration:"
end

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

# Solution 1:
# fib(n) = n < 2 ? n : fib(n-1) + fib(n-2)

# Solution 2:
# function fib(n)
# 	@assert n >= 0
# 	n < 2 && return n
# 	dp = ones(n)
# 	for i = 3:n
# 		dp[i] = dp[i-1] + dp[i-2]
# 	end
# 	return dp[end]
# end

# ╔═╡ 0bcb6969-7e99-49ae-a57c-3c3b923f65f7
md"**Bonus point:** update your function to make the execution time of the following cell less than 1ms."

# ╔═╡ f3dfea15-4760-4294-badd-c2849426d53e
@time fib(42)

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

# ╔═╡ 0f2aff9d-778b-4a08-9c33-c1866279c686
begin
	abstract type AbstractNormal <: Distribution end
	(p::AbstractNormal)(x) = exp(-0.5((x-p.μ)/p.σ)^2) / (p.σ * √(2π))  # \sqrt => √
end

# ╔═╡ 47cd214a-ba2b-486f-b576-f2a583b50b7e
begin  # method overloading
	Statistics.mean(p::AbstractNormal) = p.μ
	Statistics.std(p::AbstractNormal) = p.σ
	Statistics.var(p::AbstractNormal) = p.σ ^ 2
end

# ╔═╡ 76d61e6d-16e8-440d-99f7-51a3775694b9
mean(1:10)

# ╔═╡ 9f56480f-52b2-4770-bf6e-9d7676756a87
methods(mean)

# ╔═╡ 0c371cea-44f9-4703-964f-13d1a9f55535
methodswith(AbstractNormal)

# ╔═╡ 48e93319-299b-40b9-bbf9-09d18d683c9c
struct NormalUntyped <: AbstractNormal
	μ
	σ
end

# ╔═╡ fa1283d5-b3d5-46d4-a34c-4cddc32ab284
fieldtypes(NormalUntyped)

# ╔═╡ 322ea469-2961-46b0-a93c-20e2c8f94328
p1 = NormalUntyped(0, 1)

# ╔═╡ cc45cdea-38c6-4c06-b62c-09a36559bfd6
@which mean(p1)

# ╔═╡ beb7b5f4-ee86-4130-aa61-d3f8498ff4ed
md"In multiple dispatch, Julia determines which method to call based on the numbers and types of input arguments."

# ╔═╡ ec5238e4-f445-491c-bd14-8e1aba59049f
p1(0)

# ╔═╡ f3b4eba4-5471-441e-b199-69fd07f528e2
md"A piece of Julia code is called 'type-stable' if all input and output variables have a concrete type, either by explicit declaration or by inference from the Julia compiler. Type-stable code will run much faster as the compiler can generate statically typed code and optimize it at compile-time."

# ╔═╡ cfeb3928-cc2f-47a3-8a9b-e17eabd79a33
@code_warntype p1(0)

# ╔═╡ c6739f52-f87f-4bef-8c32-ce3ec4942342
@code_llvm p1(0)

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

# ╔═╡ 76d2cfde-bdd8-4e45-83dd-92d3c651691f
struct NormalParametric{T} <: AbstractNormal
	μ :: T
	σ :: T
end

# ╔═╡ 1e36bd1d-cb83-4e48-a5dc-f88bf04636ca
p3 = NormalParametric(0f0, 5f-1)  # float32 version of 5e-1

# ╔═╡ 00ed2dc6-f770-49da-9eac-35042f437b6e
NormalParametric([0.0, 0.1], [0.5, 1.0])

# ╔═╡ b088c77f-9732-4c63-88f9-9bcd911e461c
@code_warntype p3(0)

# ╔═╡ 2e6521be-ff66-47a9-8c19-68216cb62f3d
md"We can see that the length of the LLVM bitcodes generated from a piece of type-stable Julia code is much shorter than its type-instable version. The following example will compare their performance."

# ╔═╡ 149a64ba-6d5b-4416-bc2d-8e1ae897c71d
function probability(P::Distribution, lo::Float64, hi::Float64; step=1e-6)
	step * sum(P(x) for x in lo:step:hi)
end

# ╔═╡ d00e9d96-59c7-4bd6-9667-340505d5ed5f
@time probability(p1, -1.96, 1.96)

# ╔═╡ 8e8a900f-1d6c-4d65-afda-b03e64f3c9c8
@time probability(p2, -1.96, 1.96)

# ╔═╡ af5fffbd-baf5-46e4-b285-3a98a5d01e55
@time probability(p3, -1.96, 1.96)

# ╔═╡ 7b6e1d43-c72c-4bd9-b493-838b05e845c4
md"## Collection Data Types"

# ╔═╡ aa08b116-025a-43cd-8f0d-74e035b9746d
md"**Vector:**"

# ╔═╡ 69283b2e-bd47-4c3c-890f-677b253183e7
v = [1, 2, 3, 4, 5]

# ╔═╡ a2c92fca-fbab-4396-b472-a53d7a858abe
typeof(v)

# ╔═╡ d7186b34-117c-4a11-8907-91766a038425
v[1]  # index starts from 1 in Juila

# ╔═╡ 0f3b3f22-89f3-491d-be29-57438d83f4cd
length(v)

# ╔═╡ 2c0b579b-302c-458e-bfb0-75ce768de5bd
v .* 2  # broadcasting

# ╔═╡ 176bb2e7-dde9-4696-ab01-eea38a1081b8
v ./ v

# ╔═╡ b3321c01-db3d-42ed-9ea7-142e8773bc28
sqrt.(v)

# ╔═╡ dbdbd2b6-5831-48da-a9a0-8052e96a5586
push!(v, 0)

# ╔═╡ 5e438baa-fc94-423a-a924-280405fa4255
deleteat!(v, 5)

# ╔═╡ 89b93f51-6e10-482e-85e2-9fc6ece8bf53
sort!(v)

# ╔═╡ 0f5a46ab-6108-4683-80e0-8f1acaec7c7f
md"Julia adopts [column major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)."

# ╔═╡ 8cc1e1ca-207e-4dc3-b860-2c5c2114a49a
[v; v]  # concatenate vertically

# ╔═╡ 90a98f2a-6d97-4697-a4a7-ab1cac19d9e1
vcat(-v, 2v)

# ╔═╡ 071a0163-3071-4398-bc46-d12c11bbcba0
hcat(v[1:3], v[1:2:end], v[end-1:-1:2])  # concatenate horizontally

# ╔═╡ 1cc93533-0edb-4db9-9016-4f52f3822b0a
v' * v  # inner product

# ╔═╡ b977a080-5e47-45fb-ad78-8080f93fa650
v * v'  # outer product

# ╔═╡ 4aa0d597-49b7-4e8f-807e-0181f6d75dae
md"**Range:**"

# ╔═╡ 760ff5fd-689b-4afe-9336-cc480fb6b486
let r = 1:2:5
	@show (3v)[r] r.start r.stop r.step
	collect(r)  # convert to array
end

# ╔═╡ ce603931-baa5-48aa-ba13-82b458962ddf
md"**Matrix:**"

# ╔═╡ 3cfce228-b634-4e31-b3f3-ddadb6c7a53d
Array{Int, 2}

# ╔═╡ 6b3a83eb-e316-46b5-a097-233145ab1bcc
[1 2 3
 5 6 4
 9 7 8]  # or [1 2 3; 5 6 4; 9 7 8]

# ╔═╡ 4f62d53f-11bb-4e53-b759-d6f49eec5cd4
let a = Array{Float64}(undef, 2, 3)  # initialize a 2x3 Matrix of Float64s
	@show a
	for i=1:2, j in 1:3  # equivalent to a nested loop (inner loop is on j)
		a[i, j] = i * j
	end
	a
end

# ╔═╡ 952db525-9d54-4b56-a09f-3014a9ca9293
[i * j for i in 1:2, j in 1:3]  # array comprehension

# ╔═╡ d02b8c20-6e43-435c-ba9f-870b1bb5fae9
zeros(3, 3)  # or ones

# ╔═╡ b5eb64a4-6572-405f-bed4-7e483f6e50e5
rand(2, 2, 2)

# ╔═╡ 8bc03ce0-2fe3-45ca-9c1a-9bd2a98bc41e
A = rand(ComplexF64, 3, 2)

# ╔═╡ b2d92744-576d-4611-af48-1ff6641a24e1
length(A)

# ╔═╡ d1ca8fb0-580f-4625-aba3-dd18e054ee48
size(A), size(A, 1)

# ╔═╡ 6036d669-f880-4852-86ca-bfc3f2ab52d2
A[4]

# ╔═╡ 9fc3a808-5a53-44e9-9f45-5939d9064c30
A[1:2:end, [2]]

# ╔═╡ 1603ceb6-e8a8-486e-8bff-c721b57ab2eb
reshape(A, :, 3)  # same as A.reshape(-1, 3) in Python

# ╔═╡ 8ea9ecaf-6d66-4e57-8606-e79fdc8415e5
[A; A]  # concat vertically (same as vcat(A, A))

# ╔═╡ 9bb81880-067c-4bde-a12f-c37eb4be2846
[A A]  # concat horizontally (same as hcat(A, A) and [A;; A])

# ╔═╡ 27be59f3-4a50-4518-bacc-6850025e7aa5
md"More on array concatenation at [https://docs.julialang.org/en/v1/manual/arrays/#man-array-concatenation](https://docs.julialang.org/en/v1/manual/arrays/#man-array-concatenation)."

# ╔═╡ 61f1ef4a-8457-4b39-aba5-e760070df95d
A .+ [1, 2, 3]

# ╔═╡ 3d8c4cc1-7c02-453f-a6dd-106b1390896a
A .+ [1 2]

# ╔═╡ 8efda77f-e3d5-4866-8b64-159b6c3a6114
transpose(A)

# ╔═╡ d9f9542f-8d4f-4c0c-b4ea-986eefc07636
A'  # adjoint: complex conjugate followed by transpose

# ╔═╡ 12008adf-5162-484c-af6b-30b2d43f46b5
sum(A, dims=2)  # sum along the 2nd axis

# ╔═╡ 65f92119-b389-491c-b809-fab91636c53a
mean(A)

# ╔═╡ 9cc9456e-fdac-4f56-89c4-e3ddf8a5f0af
mean(A, dims=1)

# ╔═╡ 17eeffee-701d-4251-aca7-308e456487da
let B = reshape(A, 2, 3), C = A', D = A[1:2,:]  # A,B,C share the same underlying data
	B[1, 2] = NaN
	C[1, 2] = NaN
	D[:] .= NaN  # sets all values of D to NaN
	A, B, C, D
end

# ╔═╡ fad551be-abbc-45c6-b08c-5e8d4ddccdb0
md"**Generic functions of iterable types (Polymorphism):**"

# ╔═╡ 26f43214-3b99-4c99-9512-398a28f9ae0a
md"""
!!! danger "Task"
	Generate a 1000×2 random matrix of float numbers from the normal distribution ``N(0, 1)`` using the `randn` function and assign it to `Q`.
"""

# ╔═╡ b226106d-6f21-4d72-951c-c4d9d01cbbcb
Q = missing  # replace missing with your answer

# ╔═╡ 820f0070-98b9-4bf6-a8db-65383e7c3c17
# plots will be generated after you finish the tasks
if !ismissing(Q)
	using Plots
	plt1 = scatter(Q[:, 1], Q[:, 2])
	plt2 = histogram(Q, bins=range(-3, 5.5, length=50))
	plot(plt1, plt2, layout=(2, 1), legend=false, size=(600, 800))
end

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
S = [2.0 -1.0; 1.2 0.6]

# ╔═╡ 50cb4c19-1d76-4844-8bc7-bc564aa34ab8
begin
	md"(Show this cell for a sample solution)"
	# Q .= S[2, :]' .* Q .+ S[1, :]'
	# OR
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

# ╔═╡ 493a6c95-3820-43aa-8e6c-939757aecf2b
M - I  # I is identity matrix

# ╔═╡ 2c379af2-73d9-4470-8f7f-9dafa789e951
M ^ -1 * M ≈ I  # M ^ -1 == inv(M)

# ╔═╡ 6287eddc-9b35-489e-b584-8197c09cb228
let b = [1, 2, 3]
	x = inv(M) * b  # or M \ b
	@show M * x
	M = [M rand(3)]  # size(M) = (3, 4)
	y = M \ b
	@show y
	M * y  # or pinv(M) * b (least squares)
end

# ╔═╡ 3f6fbfd0-b35a-4af9-86cd-55d7e4188301
let eig = eigen(M)
	λ = eig.values
	V = eig.vectors
	@assert M * V ≈ V * diagm(λ)  # diag(diagm(x)) == x
	λ, V
end

# ╔═╡ 2dde11e3-dcc7-416b-b351-bcf526f3deaa
svd(M).S .^ 2  # singular values of M = sqrt of eigen values of M'M

# ╔═╡ 5fbbf58e-2c28-4b80-b524-49f881258f46
eigen(M' * M).values |> reverse

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
@show_all let a = -6:6:6, b = [1, 2, 3], c = (4, 5, 6)
	length(a)
	maximum(a)
	sum(b)
	reverse(b)
	map(abs2, a)
	map(-, a, b)
	map(min, a, b, c)
	zip(a, b, c) |> collect  # `a |> f` (pipe operator) is equivalent to `f(a)`
	count(iseven, b)
	findfirst(iszero, a)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractTrees = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
AbstractTrees = "~0.4.5"
Plots = "~1.40.4"
PlutoUI = "~0.7.59"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "d5be2c1efeddaf18a4d70ea211137e85ab4d3f80"

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

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a2f1c8c668c8e3cb4cca4e57a8efdb09067bb3fd"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "71acdbf594aab5bbb2cec89b208c41b4c411e49f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.24.0"

[[deps.ChangesOfVariables]]
deps = ["InverseFunctions", "LinearAlgebra", "Test"]
git-tree-sha1 = "2fba81a302a7be671aefe194f0525ef231104e7f"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.8"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "4b270d6465eb21ae89b732182c20dc165f8bf9f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.25.0"

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

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "6cbbd4d241d7e6579ab354737f4dd95ca43946e1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

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
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ddda044ca260ee324c5fc07edb6d7cf3f0b9c350"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.5"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "278e5e0f820178e8a26df3184fcb2280717c79b1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.5+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "7c82e6a6cd34e9d935e9aa4051b66c6ff3af59ba"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.2+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "d1d712be3164d61d1fb98e7ce9bcbc6cc06b45ed"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.8"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

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

[[deps.InverseFunctions]]
deps = ["Dates", "Test"]
git-tree-sha1 = "e7cbed5032c4c397a6ac23d1493f3289e01231c4"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.14"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

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

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c84a835e1a09b289ffcd2271bf2a337bbdda6637"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.3+0"

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

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "e0b5cd21dc1b44ec6e64f351976f961e6f31d6c4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

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
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

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
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a028ee3cb5641cccc4c24e90c36b0a4f7707bdf5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.14+0"

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
version = "10.40.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "6e55c6841ce3411ccb3457ee52fc48cb698d6fb0"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.2.0"

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

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

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

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "a947ea21087caba0a798c5e494d0bb78e3a1a3a0"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.9"

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

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "Random"]
git-tree-sha1 = "dd260903fdabea27d9b6021689b3cd5401a57748"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.20.0"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

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

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "52ff2af32e591541550bd753c0da8b9bc92bb9d9"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.7+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

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

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

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
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e678132f07ddb5bfa46857f0d7620fb9be675d3b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

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
# ╟─0939489d-79d2-4c1a-9841-17d9ae448d94
# ╠═a025f4ac-9f39-4d05-9e0f-c12b9145d7c6
# ╟─b4cb3a82-d740-4d02-b0f4-f18ec9500b4f
# ╟─4efa23f3-e705-469e-8e82-fb6d0e4589a3
# ╠═52ab5184-2f0f-11ef-3034-8fd6a5c8a2cb
# ╠═50c86554-ff09-4e4a-94e8-0f30b83e8655
# ╟─1cd98952-cd47-4632-a71a-903f1809d6be
# ╠═0f63f358-310c-4475-a17b-6376ce26f903
# ╟─a8d9989e-79df-42c5-aaf2-7212ad26a9da
# ╠═3ae5a286-cc9d-4837-a6de-c79bad078df4
# ╠═662c94fb-a2b9-4970-86a8-5f952d118309
# ╟─86e687c7-4052-4d11-9ef9-7ac6b59cb8ae
# ╠═cdac9eca-48a6-44dd-9926-a1e0959c2c31
# ╠═5c64daca-361a-4c3b-92e0-b179c834a63e
# ╟─9cde569d-7db4-4c06-8e03-346b32afaa16
# ╠═f2496da6-a024-44eb-b1ee-6cd5e213a86a
# ╠═79dd50f1-bd99-4384-b691-4bdb73096161
# ╟─45737508-e741-445d-86ef-850ab9915039
# ╠═88c296f4-51c7-4968-84d4-7d7d66288d8c
# ╠═0d376fb5-3b55-4607-903b-1a1777f77215
# ╠═5dedb5f1-1e4e-4b47-9e28-46d9d901f6ca
# ╠═636db2a4-c4f7-49ff-8cd6-9956e15e5f6e
# ╟─b541204e-3054-4504-b8f4-913209f19913
# ╟─1423d00d-8d72-4d84-ad47-95131d8b4bad
# ╠═6366c0dd-5ec4-435f-9b1e-27e5215f4cf9
# ╟─7c5c7303-474f-4214-a26e-ff98bcc1272b
# ╠═e94003d4-580e-454e-9330-b35bfe0bfce0
# ╟─70b0e880-be05-4170-98fc-9d0e2ff1df96
# ╟─808d8684-e857-46b8-835f-6056079b1e77
# ╠═52f340d8-1242-4a1f-9bca-164816bd1896
# ╟─6d3abfce-8de4-45aa-9646-940fe329cf7a
# ╠═d10c97a6-64c7-4adf-bf76-c25e19f7f215
# ╟─7e773d44-9823-4753-8a4f-ca31ec5afb85
# ╠═e68d2aa6-f69b-47ad-9319-44a91d678097
# ╟─12e18c54-1d43-4f23-89e1-f578f3f34cb0
# ╠═559143b7-a4e1-4532-adde-523ba70e7d36
# ╠═8af405f5-01c3-45e3-8451-3e3ac287466f
# ╠═a292b548-502b-455b-9ed8-15843b0930dc
# ╠═4ec99d1e-bbb8-4dde-9276-6532bf4eeb64
# ╟─79305e1d-a394-4247-b459-cd70a1d29213
# ╠═c88262d4-ba12-4955-9022-7724909231ee
# ╟─ccd1d5e8-88b6-40af-a850-e16deb9718e9
# ╠═0c05213d-5390-40e0-8c92-676774067e28
# ╟─01e35e8d-cb99-45fb-8770-2e23f3ec7c7c
# ╟─8d8c7053-1a23-485f-90c5-2db999f7581d
# ╟─b5b168db-b896-41bb-afeb-08e328d7b28e
# ╠═52de688b-58c8-4aa4-8f92-4068488342e7
# ╠═7ae98bfb-d650-4bf2-b516-edc03ea68cde
# ╠═8b4fcefa-d53a-461a-a347-38c0acbf73ac
# ╠═b0e25850-9c6b-4898-8bcd-4a4cd4d2a1cb
# ╠═8bed8fcc-1ea4-414f-a188-910b74632085
# ╠═40c36c39-b3b7-4c12-a116-7c0ddb079085
# ╟─4a851df6-3894-42a9-9acd-eb25a56f5535
# ╠═1396345b-8abf-48ac-8bfa-6c641a395c2c
# ╟─ef02cbb9-11af-49e9-a996-f2c44c9c1191
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
# ╟─bd7bed63-714c-417c-822c-2c07419d59db
# ╟─0bcb6969-7e99-49ae-a57c-3c3b923f65f7
# ╠═f3dfea15-4760-4294-badd-c2849426d53e
# ╟─8116c816-ab72-4415-94bf-a66ad7f52d2d
# ╟─13104a6c-0eb7-42d7-961d-addc55f06588
# ╠═002bd083-00d2-4fd6-965f-9415d85f23f6
# ╠═e9f8aee3-aa16-446b-aeec-8d1aae6e7169
# ╠═18aab5fb-7add-4ada-b42e-2bc62968d6bc
# ╠═0c4a6998-8863-404e-96c2-952df70839ab
# ╠═a9561c08-2b07-4590-b901-d9cbd60355ee
# ╟─2e034e29-8755-43d5-b557-d247df23f50e
# ╠═e3f7a77a-8c9e-4f15-af47-551fd959b2a6
# ╠═0f2aff9d-778b-4a08-9c33-c1866279c686
# ╠═3c74c07d-98a5-48b8-bf6c-2a25e85597d5
# ╠═76d61e6d-16e8-440d-99f7-51a3775694b9
# ╠═47cd214a-ba2b-486f-b576-f2a583b50b7e
# ╠═9f56480f-52b2-4770-bf6e-9d7676756a87
# ╠═0c371cea-44f9-4703-964f-13d1a9f55535
# ╠═48e93319-299b-40b9-bbf9-09d18d683c9c
# ╠═fa1283d5-b3d5-46d4-a34c-4cddc32ab284
# ╠═322ea469-2961-46b0-a93c-20e2c8f94328
# ╠═cc45cdea-38c6-4c06-b62c-09a36559bfd6
# ╟─beb7b5f4-ee86-4130-aa61-d3f8498ff4ed
# ╠═ec5238e4-f445-491c-bd14-8e1aba59049f
# ╟─f3b4eba4-5471-441e-b199-69fd07f528e2
# ╠═cfeb3928-cc2f-47a3-8a9b-e17eabd79a33
# ╠═c6739f52-f87f-4bef-8c32-ce3ec4942342
# ╠═035f9794-43ea-4e19-860c-a66fd0ea1a14
# ╠═57f30a3c-7d28-4819-958a-bf1859d6947c
# ╠═024aa7d5-a569-4639-851f-b7d491855202
# ╠═f640df71-ae15-4b67-a30e-c806ea532a19
# ╠═76d2cfde-bdd8-4e45-83dd-92d3c651691f
# ╠═1e36bd1d-cb83-4e48-a5dc-f88bf04636ca
# ╠═00ed2dc6-f770-49da-9eac-35042f437b6e
# ╠═b088c77f-9732-4c63-88f9-9bcd911e461c
# ╟─2e6521be-ff66-47a9-8c19-68216cb62f3d
# ╠═149a64ba-6d5b-4416-bc2d-8e1ae897c71d
# ╠═d00e9d96-59c7-4bd6-9667-340505d5ed5f
# ╠═8e8a900f-1d6c-4d65-afda-b03e64f3c9c8
# ╠═af5fffbd-baf5-46e4-b285-3a98a5d01e55
# ╟─7b6e1d43-c72c-4bd9-b493-838b05e845c4
# ╟─aa08b116-025a-43cd-8f0d-74e035b9746d
# ╠═69283b2e-bd47-4c3c-890f-677b253183e7
# ╠═a2c92fca-fbab-4396-b472-a53d7a858abe
# ╠═d7186b34-117c-4a11-8907-91766a038425
# ╠═0f3b3f22-89f3-491d-be29-57438d83f4cd
# ╠═2c0b579b-302c-458e-bfb0-75ce768de5bd
# ╠═176bb2e7-dde9-4696-ab01-eea38a1081b8
# ╠═b3321c01-db3d-42ed-9ea7-142e8773bc28
# ╠═dbdbd2b6-5831-48da-a9a0-8052e96a5586
# ╠═5e438baa-fc94-423a-a924-280405fa4255
# ╠═89b93f51-6e10-482e-85e2-9fc6ece8bf53
# ╟─0f5a46ab-6108-4683-80e0-8f1acaec7c7f
# ╠═8cc1e1ca-207e-4dc3-b860-2c5c2114a49a
# ╠═90a98f2a-6d97-4697-a4a7-ab1cac19d9e1
# ╠═071a0163-3071-4398-bc46-d12c11bbcba0
# ╠═1cc93533-0edb-4db9-9016-4f52f3822b0a
# ╠═b977a080-5e47-45fb-ad78-8080f93fa650
# ╟─4aa0d597-49b7-4e8f-807e-0181f6d75dae
# ╠═760ff5fd-689b-4afe-9336-cc480fb6b486
# ╟─ce603931-baa5-48aa-ba13-82b458962ddf
# ╠═3cfce228-b634-4e31-b3f3-ddadb6c7a53d
# ╠═6b3a83eb-e316-46b5-a097-233145ab1bcc
# ╠═4f62d53f-11bb-4e53-b759-d6f49eec5cd4
# ╠═952db525-9d54-4b56-a09f-3014a9ca9293
# ╠═d02b8c20-6e43-435c-ba9f-870b1bb5fae9
# ╠═b5eb64a4-6572-405f-bed4-7e483f6e50e5
# ╠═8bc03ce0-2fe3-45ca-9c1a-9bd2a98bc41e
# ╠═b2d92744-576d-4611-af48-1ff6641a24e1
# ╠═d1ca8fb0-580f-4625-aba3-dd18e054ee48
# ╠═6036d669-f880-4852-86ca-bfc3f2ab52d2
# ╠═9fc3a808-5a53-44e9-9f45-5939d9064c30
# ╠═1603ceb6-e8a8-486e-8bff-c721b57ab2eb
# ╠═8ea9ecaf-6d66-4e57-8606-e79fdc8415e5
# ╠═9bb81880-067c-4bde-a12f-c37eb4be2846
# ╟─27be59f3-4a50-4518-bacc-6850025e7aa5
# ╠═61f1ef4a-8457-4b39-aba5-e760070df95d
# ╠═3d8c4cc1-7c02-453f-a6dd-106b1390896a
# ╠═8efda77f-e3d5-4866-8b64-159b6c3a6114
# ╠═d9f9542f-8d4f-4c0c-b4ea-986eefc07636
# ╠═12008adf-5162-484c-af6b-30b2d43f46b5
# ╠═65f92119-b389-491c-b809-fab91636c53a
# ╠═9cc9456e-fdac-4f56-89c4-e3ddf8a5f0af
# ╠═17eeffee-701d-4251-aca7-308e456487da
# ╟─fad551be-abbc-45c6-b08c-5e8d4ddccdb0
# ╠═fbd9a83b-17b4-47db-a46e-e7a9037b9090
# ╟─26f43214-3b99-4c99-9512-398a28f9ae0a
# ╠═b226106d-6f21-4d72-951c-c4d9d01cbbcb
# ╟─aa0c8fec-254b-4805-bf07-b1ce7266685c
# ╟─03c85588-2237-4d17-9755-bd3386f8e348
# ╠═24077fc9-4d06-4b80-91be-321a7bb0fe5c
# ╟─50cb4c19-1d76-4844-8bc7-bc564aa34ab8
# ╟─8615c4ca-7e2b-49fb-bb0f-078347a7c56b
# ╟─be7f3b8d-70e6-4ec3-a98f-07fbe17fb06a
# ╠═820f0070-98b9-4bf6-a8db-65383e7c3c17
# ╟─66cae8d2-8e20-4b1e-9dae-e120eee4d944
# ╠═5af22ae0-effd-4589-bd1f-d375299b6848
# ╠═493a6c95-3820-43aa-8e6c-939757aecf2b
# ╠═2c379af2-73d9-4470-8f7f-9dafa789e951
# ╠═6287eddc-9b35-489e-b584-8197c09cb228
# ╠═5ee4f31b-ebae-4d8f-8ccc-6df671de6965
# ╠═3f6fbfd0-b35a-4af9-86cd-55d7e4188301
# ╠═2dde11e3-dcc7-416b-b351-bcf526f3deaa
# ╠═5fbbf58e-2c28-4b80-b524-49f881258f46
# ╟─8bc7e78b-ff6d-4553-b327-f03d21651121
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
