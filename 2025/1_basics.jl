### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ a025f4ac-9f39-4d05-9e0f-c12b9145d7c6
# ╠═╡ show_logs = false
begin
	using PlutoUI
	# import all names exported by "PlutoUI" into the global namespace
	
	include("utils.jl")  # load local source file
	using .Utilities  # "utils.jl" has a module called "Utilities"
  
	TableOfContents(aside=true, depth=3)  # imported from PlutoUI
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

# ╔═╡ 128c2820-7676-4fcb-aa14-69b1478f9c68
2, 2.0 # integers, double precision

# ╔═╡ 5c64daca-361a-4c3b-92e0-b179c834a63e
z ≈ -1 ? print("Hurray!") : print("Why?")  # \approx <tab> => ≈

# ╔═╡ 9cde569d-7db4-4c06-8e03-346b32afaa16
md"## Compound Expressions and Local Scope"

# ╔═╡ f2496da6-a024-44eb-b1ee-6cd5e213a86a
md"**Local binding with `let`**"

# ╔═╡ 79dd50f1-bd99-4384-b691-4bdb73096161
let θ = rand(), z = exp(im * θ)  # let...end binds variables locally
	# θ is a random float number in [0, 1)
	@show θ abs(z) angle(z) cos(θ) real(z) sin(θ) imag(z)
end

# ╔═╡ 45737508-e741-445d-86ef-850ab9915039
md"**Non-local binding via `begin`**"

# ╔═╡ 5dedb5f1-1e4e-4b47-9e28-46d9d901f6ca
md"`begin...end` can be treated as an expression: it has a value"

# ╔═╡ b541204e-3054-4504-b8f4-913209f19913
md"## Control Structures"

# ╔═╡ 1423d00d-8d72-4d84-ad47-95131d8b4bad
md"**For Loop:**"

# ╔═╡ 6366c0dd-5ec4-435f-9b1e-27e5215f4cf9
for x in 1:10 # This is a range and we'll come back to it
	print(x)
end

# ╔═╡ e34871a1-501c-4fe6-b904-71e541ee3d81
md"**Sum of inverse squares**"

# ╔═╡ 7c5c7303-474f-4214-a26e-ff98bcc1272b
md"**Conditional Evaluation:**"

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

# ╔═╡ 43ca4fbd-f04d-4d94-af41-8272c605807f
md"### Concrete types"

# ╔═╡ e1735f0a-6d3d-41bc-81d3-5eead9d41f18
md"specify data structure, types of values (objects), `example: Float64`"

# ╔═╡ be99ab8d-1855-48e7-9b23-6e9021706569
md"### Abstract types"

# ╔═╡ eb0eed0a-11a1-4468-afc3-4690ec65ee50
md"Abstract types cannot be instantiated, `example: Number` says we can do operations like `+`, `-`, `*`, and `/` with corresponding values."

# ╔═╡ 09e2f1c0-633e-4a05-aa12-aa3189c1c854
md"Julia is built around types, high performance codes will make use of the type system"

# ╔═╡ 808d8684-e857-46b8-835f-6056079b1e77
md"Type signatures (declarations) in the form `binder :: type`"

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

# ╔═╡ 8af405f5-01c3-45e3-8451-3e3ac287466f
for s in fieldnames(Tz)
	println("z.", s, " = ", getfield(z, s))  # `println` prints a newline at the end
end

# ╔═╡ a292b548-502b-455b-9ed8-15843b0930dc
fieldtypes(Tz)

# ╔═╡ 4ec99d1e-bbb8-4dde-9276-6532bf4eeb64
fieldnames(Float64)  # primitive type (no fields)

# ╔═╡ 79305e1d-a394-4247-b459-cd70a1d29213
md"The results above are tuples. Tuple is an `immutable` type, which means its items cannot be modified."

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
begin
	md"(Show this cell for a sample solution)"
	# for T in fieldtypes(typeof(z))
	# 	println(fieldtypes(T) == () ? "primitive" : "composite")
	# end
end

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
"Call the `fib` function in the C library `libgmp`."
function fib_c(n)
    z = BigInt()
    @ccall "libgmp".__gmpz_fib_ui(z::Ref{BigInt}, n::Culong)::Cvoid
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

# ╔═╡ 48b447d2-0ec1-4d42-985d-84bb3ce4c759
Union{Int, Char}

# ╔═╡ 18aab5fb-7add-4ada-b42e-2bc62968d6bc
isabstracttype(Integer)

# ╔═╡ 0c4a6998-8863-404e-96c2-952df70839ab
isconcretetype(Int64)

# ╔═╡ 02dda798-9681-41fb-afc1-ba2e24e786e8
md"Inspecting the type tree"

# ╔═╡ 2e034e29-8755-43d5-b557-d247df23f50e
md"### Define Custom Types"

# ╔═╡ e3f7a77a-8c9e-4f15-af47-551fd959b2a6
abstract type Distribution end

# ╔═╡ beb7b5f4-ee86-4130-aa61-d3f8498ff4ed
md"In multiple dispatch, Julia determines which method to call based on the numbers and types of input arguments."

# ╔═╡ f3b4eba4-5471-441e-b199-69fd07f528e2
md"A piece of Julia code is called 'type-stable' if all input and output variables have a concrete type, either by explicit declaration or by inference from the Julia compiler. Type-stable code will run much faster as the compiler can generate statically typed code and optimize it at compile-time."

# ╔═╡ a4c7126d-57dd-4542-bcc4-d01cf657759a
md"Parametric types like `Complex{T}` allow parametric polymorphism."

# ╔═╡ 74c57fe8-e369-44f1-a51e-8365e4ffed5d
md"An advantage of parametric types is that a single piece of code can handle a variety of concrete types. This is called `generic programming`."

# ╔═╡ 2e6521be-ff66-47a9-8c19-68216cb62f3d
md"We can see that the length of the LLVM bitcodes generated from a piece of type-stable Julia code is much shorter than its type-instable version. The following example will compare their performance."

# ╔═╡ 149a64ba-6d5b-4416-bc2d-8e1ae897c71d
function probability(P::Distribution, lo::Float64, hi::Float64; dx=1e-6)
	sum(P(x) for x in lo:dx:hi) * dx
end

# ╔═╡ 7b6e1d43-c72c-4bd9-b493-838b05e845c4
md"## Collection Data Types"

# ╔═╡ 63eddb5a-960c-43c6-9425-5caa40f4802f
md"### Range"

# ╔═╡ 51c754f6-ba17-4936-8e1e-89899634e37d
md"### Array"

# ╔═╡ 4da224f7-7e68-425f-b575-877807efa884
[1, 2, 3]

# ╔═╡ 38f6649c-5daa-43f1-900f-98381b6d33fc
# explicit typing
Float64[1, 2, 3]

# ╔═╡ aa08b116-025a-43cd-8f0d-74e035b9746d
md"#### Vector"

# ╔═╡ 69283b2e-bd47-4c3c-890f-677b253183e7
v = [1, 2, 3, 4, 5]

# ╔═╡ 760ff5fd-689b-4afe-9336-cc480fb6b486
let r = 1:2:5
	@show v[r] r.start r.stop r.step
	collect(r)  # convert to array
end

# ╔═╡ a2c92fca-fbab-4396-b472-a53d7a858abe
typeof(v)

# ╔═╡ d7186b34-117c-4a11-8907-91766a038425
v[1]  # index starts from 1 in Juila

# ╔═╡ 0f3b3f22-89f3-491d-be29-57438d83f4cd
length(v)

# ╔═╡ b3321c01-db3d-42ed-9ea7-142e8773bc28
sqrt.(v)

# ╔═╡ dbdbd2b6-5831-48da-a9a0-8052e96a5586
push!(v, 0)

# ╔═╡ 88e5862d-0859-4ac7-b8a1-400ab4b10c18
insert!(v, 1, 6)

# ╔═╡ 89b93f51-6e10-482e-85e2-9fc6ece8bf53
sort!(v)

# ╔═╡ 5e438baa-fc94-423a-a924-280405fa4255
deleteat!(v, 1)

# ╔═╡ 0f5a46ab-6108-4683-80e0-8f1acaec7c7f
md"Julia adopts [column major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)."

# ╔═╡ 8cc1e1ca-207e-4dc3-b860-2c5c2114a49a
[v; v]  # concatenate vertically (same as vcat(v, v))

# ╔═╡ 90a98f2a-6d97-4697-a4a7-ab1cac19d9e1
[v v]  # concatenate horizontally (same as hcat(v, v))

# ╔═╡ 071a0163-3071-4398-bc46-d12c11bbcba0
hcat(v[1:3], v[1:2:end-1], v[end:-2:1])  # concatenate horizontally

# ╔═╡ ce603931-baa5-48aa-ba13-82b458962ddf
md"#### Matrix"

# ╔═╡ 3cfce228-b634-4e31-b3f3-ddadb6c7a53d
Array{Int, 2}

# ╔═╡ 6b3a83eb-e316-46b5-a097-233145ab1bcc
[1 2 3
 5 6 4
 9 7 8]  # or [1 2 3; 5 6 4; 9 7 8]

# ╔═╡ 4f62d53f-11bb-4e53-b759-d6f49eec5cd4
let a = Array{Float64}(undef, 2, 3)  # initialize a 2x3 Matrix of Float64s
	@show a
	for i in 1:2, j in 1:3  # equivalent to a nested loop (inner loop is on j)
		a[i, j] = i * j
	end
	a
end

# ╔═╡ 952db525-9d54-4b56-a09f-3014a9ca9293
[i * j for i in 1:2, j in 1:3]  # array comprehension

# ╔═╡ ae856b3a-795a-4f99-90d0-c5c9ffacc3e9
[(i, j) for i in 1:2 for j in 1:3]  # no extra dimensions

# ╔═╡ d02b8c20-6e43-435c-ba9f-870b1bb5fae9
zeros(3, 3)  # or ones

# ╔═╡ b5eb64a4-6572-405f-bed4-7e483f6e50e5
rand(2, 2, 2)

# ╔═╡ 9c670e29-f48a-4f4e-a9a1-425f76a1f006
rand('a':'z', 2, 4)  # random values from a range of characters

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
reshape(A, :, 3)  # same as A.reshape(-1, 3) in numpy

# ╔═╡ 8ea9ecaf-6d66-4e57-8606-e79fdc8415e5
[A; A]  # concat vertically (same as vcat(A, A))

# ╔═╡ 9bb81880-067c-4bde-a12f-c37eb4be2846
[A A]  # concat horizontally (same as hcat(A, A) and [A;; A])

# ╔═╡ 27be59f3-4a50-4518-bacc-6850025e7aa5
md"More on array concatenation at [https://docs.julialang.org/en/v1/manual/arrays/#man-array-concatenation](https://docs.julialang.org/en/v1/manual/arrays/#man-array-concatenation)."

# ╔═╡ 12008adf-5162-484c-af6b-30b2d43f46b5
sum(A, dims=2)  # sum along the 2nd axis

# ╔═╡ a5717c60-1abe-4164-a4c0-45708212f95d
B = copy(A)

# ╔═╡ aad8b0cc-4223-4309-bfee-f5c57e08e353
md"### Tuples"

# ╔═╡ dfaf45ea-ff3c-4b88-b033-a6443a57b598
md"A tuple is a fixed-length container which cvan hold any values, but cannot be modified, i.e. immutable, tuples are constructed with commas and parenthesis, they can be accessed via indexing."

# ╔═╡ e00f9ae1-3f93-41f4-bb5a-71a6cf8062b3
(3, 5)

# ╔═╡ 0b72c227-b7cf-4725-a4e6-baa5fec50e6a
m = (2.5, "Cambridge", 99)

# ╔═╡ 77467cf8-92a0-4c15-81e5-b26e4b1fc47f
m[1]

# ╔═╡ c7356050-5ff8-4d8d-af6b-7adf981ebf29
md"#### Named Tuples"

# ╔═╡ 5184e9de-f120-408f-927e-38dfe07d135a
md" A named tuple can be constructed"

# ╔═╡ db0de71e-d39c-4be3-a4f8-774f80ef54d1
n = (a = 3, b = 6)

# ╔═╡ 5749d084-2415-4046-a676-5b84e772795e
n[1]

# ╔═╡ 48840624-e055-4f56-baee-91ebe578dbf3
md"Fields of named tuples can be accessed by the name with a dot syntax `(n.a)`"

# ╔═╡ dfbfc391-1568-4e23-b49a-1dcc2b48aede
n.a

# ╔═╡ 34ea929c-4676-44c2-9dfd-2235a7b7414a
md"### Dictionary"

# ╔═╡ 52554079-c69e-41f3-a936-de8536f0c1b3
md"A dictionary is a collection of key-value pairs, each value can be accessed with its key. The key-valued pairs need not be of the same data type. A dictionary is more like an array, but the indices can be of any type while in an array, indices have to be integers only. Each key in a dictionary maps to a value."

# ╔═╡ 32d4c56f-99da-4e04-a2ea-8a741938d695
Dict1 = Dict() # creates an empty dictionary

# ╔═╡ 8b3266a8-23cb-4083-9048-bcffca178a75
Dict2 = Dict{String, Integer}("x" => 5, "y" => 10) # creates typed dictionary

# ╔═╡ e3ac2be4-6737-4924-ad75-a8339ac2be2c
Dict3 = Dict("x" => 5, "y" => 10, "x" => 15, "y" => 20) # creates untyped dictionary

# ╔═╡ 5892411a-514b-4123-8864-a8c8c1e65bbc
Dict4 = Dict("x" => 5, "y" => 10, "z" => "SummerSchool", 4 => 30) # creates a dictionary with mixed type keys

# ╔═╡ add58d1a-1462-4ba4-986f-d03021fd932d
Dict2["x"] # access dictionary values using keys

# ╔═╡ b190acf8-4abf-4790-a75d-602ecba34ce0
md"## Dispatch"

# ╔═╡ b4f5f60e-d064-45cc-b5c0-8ed5e3f8488e
md"Let us define a function that calculates the absolute value of a number, we want to calculate the absolute values of numbers -5.32 and 4.0 + 5.0i, we will see that the methods we employ depend on the type of the number. We can use `::` operator to annotate function arguments with types and define different methods."

# ╔═╡ b8ddb960-6d7f-44bf-949c-fa46bd2bc8c6
ourabs(a::Float64) = sign(a) * a

# ╔═╡ b6c9b6ae-9663-4170-999a-ffb87d468121
ourabs(z::ComplexF64) = sqrt(real(z * conj(z)))

# ╔═╡ 0c5d85fd-d81f-4df4-805c-860e787dbcc5
ourabs(a::Real) = sign(a) * a

# ╔═╡ 1fb11bbe-e970-41a4-88eb-09b269fd4127
ourabs(z::Complex) = sqrt(real(z * conj(z)))

# ╔═╡ 365933f6-6399-4bff-9332-85f6c7b20909
ourabs(-5.32)

# ╔═╡ 34f03b15-5990-4e21-b9cc-10ee092fbc33
methods(ourabs)

# ╔═╡ e102112f-b994-4088-8a33-4c2719fc7ef7
ourabs(-4)

# ╔═╡ 892ecca5-d7ff-4b37-a5aa-b2890914b331
md"Type annotations should be done as generic and at the same time specific as possible"

# ╔═╡ bc301f12-6a28-4106-aac1-fa4015be6f25
md"## Multiple Dispatch"

# ╔═╡ beac6982-09bf-4a27-b2e4-743bbcf7fb5f
md"When you call a generic function `g` for a given set of input arguments, which method gets executed? Julia chooses the most specific/specialized method by considering all the input argument types."

# ╔═╡ 4e70b378-9e33-40b2-9b63-a9d9bc1be5cc
begin
	g(a, b) = "a and b are anything"
	g(a::Number, b) = "a is a number, b is anything"
	g(a, b::Number) = "a is anything, b is a number"
	g(a::Number, b::Number) = "a and b are both numbers"
	g(a::AbstractFloat, b::AbstractFloat) = "a and b are both floats"
	g(a::Float32, b::Float32) = "a and b are both 32-bit floats"
end

# ╔═╡ 83a1561b-01b0-4a7b-bc83-2ea11a2ba28e
methods(g)

# ╔═╡ 84d392f4-40e0-4abd-a1f7-fc8b33796394
g(2.5, 5)

# ╔═╡ 9782436f-5cf6-45b8-a4b4-5477b6c20c80
g(2.5, 5.5)

# ╔═╡ bedc8c99-73f1-4360-8162-907084f93bf0
g(5, "ICCS")

# ╔═╡ 78215d5d-61b0-4dcc-ab96-0a795aa0d60d
md"We can check which particular method usedvia the `@which` macro"

# ╔═╡ 7a7a283b-ae93-4ead-9a71-9242d1b78390
@which g(2.5, 5.5)

# ╔═╡ fad551be-abbc-45c6-b08c-5e8d4ddccdb0
md"## Generic Functions of Iterables"

# ╔═╡ 26f43214-3b99-4c99-9512-398a28f9ae0a
md"""
!!! danger "Task"
	Generate a 2×1000 random matrix of float numbers from the normal distribution ``N(\mu, \sigma)`` where ``\mu`` is the first column of `S` and ``\sigma`` is the second column of `S`, and assign it to `Q`.
"""

# ╔═╡ 24077fc9-4d06-4b80-91be-321a7bb0fe5c
S = [2.0 1.2; -1.0 0.6]

# ╔═╡ b226106d-6f21-4d72-951c-c4d9d01cbbcb
Q = missing  # replace missing with your answer

# ╔═╡ aa0c8fec-254b-4805-bf07-b1ce7266685c
begin
	md"(Show this cell for a sample solution)"
	# p_ex = NormalParametric(S[:,1], S[:,2])
	# Q = sample(p_ex, 1000)
end

# ╔═╡ 8615c4ca-7e2b-49fb-bb0f-078347a7c56b
md"""
!!! danger "Task"
	Calculate the `mean` and `std` of each row of `Q` and concatenate them horizontally. Compare the result with `S`.
"""

# ╔═╡ be7f3b8d-70e6-4ec3-a98f-07fbe17fb06a
begin
	md"(Show this cell for a sample solution)"
	# [mean(Q, dims=2) std(Q, dims=2)]
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

# ╔═╡ 6d0ee428-b93b-4859-809d-a59b45728cf6
# list comprehensions
sum([1 / n^2 for n in 1:5000])

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

	function Base.:+(p1::T, p2::T) where {T<:AbstractNormal}  # operator overloading
		T(mean(p1) .+ mean(p2), √(var(p1) + var(p2)))
	end
end

# ╔═╡ 52ab5184-2f0f-11ef-3034-8fd6a5c8a2cb
(1 + 2) * 3 ^ 2

# ╔═╡ 50c86554-ff09-4e4a-94e8-0f30b83e8655
@show 3+4 3*4 3/4 3÷4 4%3 3^4 3<4 3>=4 3==4 3!=4;

# ╔═╡ cdac9eca-48a6-44dd-9926-a1e0959c2c31
(@show 0.1 + 0.2) == 0.3

# ╔═╡ 88c296f4-51c7-4968-84d4-7d7d66288d8c
begin # after executing `a` and `b` will now be in the global scope. 
	a = 1
	b = 2
	a + b
end

# ╔═╡ 0d376fb5-3b55-4607-903b-1a1777f77215
a

# ╔═╡ 636db2a4-c4f7-49ff-8cd6-9956e15e5f6e
three = begin
	c = 1
	d = 2
	c + d
end

# ╔═╡ 1cf649c7-6d3c-4dda-bdb6-ce53d178a033
let x = 0
	for n in 1:5000
		x += 1 / n^2
	end
	@show x
	@show x - π^2 / 6
end

# ╔═╡ 559143b7-a4e1-4532-adde-523ba70e7d36
typeof(1+im)

# ╔═╡ c88262d4-ba12-4955-9022-7724909231ee
let t = (1, 2, 3, 4)
	t[1] += 1
end

# ╔═╡ a77de155-4e6b-4985-a8d9-346417d5f83f
begin
	function j(x, y; z = 3)
		sqrt(x*x + y*y) + z
	end
end

# ╔═╡ 2001b208-3e8b-491e-91f8-7524140afca0
j(3, 4, z = 5)

# ╔═╡ 591d7e1c-c214-46ab-9402-a3a64881daa1
h(x, y) = sqrt(x^2 + y^2)

# ╔═╡ c0b8be06-85fc-457d-b4b8-1548fd714433
h(3, 4)

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
	md"(Optional) Estimate π using fixed point iteration (click the eye to show the cell):"
end

# ╔═╡ a3fe5049-4dcb-4071-9618-6b637b20fcc7
function sample(p::AbstractNormal, dims::Integer...)
	randn(size(p.μ)..., dims...) .* p.σ .+ p.μ
end

# ╔═╡ 76d61e6d-16e8-440d-99f7-51a3775694b9
mean(1:10)

# ╔═╡ 9f56480f-52b2-4770-bf6e-9d7676756a87
methods(mean)

# ╔═╡ 2c0b579b-302c-458e-bfb0-75ce768de5bd
v .+ 2  # broadcasting

# ╔═╡ 61f1ef4a-8457-4b39-aba5-e760070df95d
A .+ [1, 2, 3]

# ╔═╡ 3d8c4cc1-7c02-453f-a6dd-106b1390896a
A .+ [1 2]

# ╔═╡ 65f92119-b389-491c-b809-fab91636c53a
mean(A)

# ╔═╡ 9cc9456e-fdac-4f56-89c4-e3ddf8a5f0af
mean(A, dims=1)

# ╔═╡ 0b057b26-e5e8-4eb0-a01b-7fbc56497f6c
ourabs(4.0 + 5.0im)

# ╔═╡ 80caa1fe-0c8c-48a5-9bf8-9b818b70aa18
begin
	using Polynomials
	P(x) = Polynomial(x)
	p = [0, -3, 0, 5] # 5x^3 - 3x
	q = [1, 2, 3, 4]
	f = P(p) + P(q)
	@show f
	@show P(p + q)
	x = [0.0, 1.0, 2.0]
	f.(x)
end

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

# ╔═╡ b13074cb-0a3a-48b7-97ac-b9ef93fa184a
mean(p1)

# ╔═╡ eeb1f9c3-6342-4ff3-a731-77ec4a55ebd1
sample(p1, 5)

# ╔═╡ cc45cdea-38c6-4c06-b62c-09a36559bfd6
@which mean(p1)

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

# ╔═╡ ed7082dc-cd39-4488-842c-1f05968224bf
Normal(-1, 3) + Normal(3, 4)

# ╔═╡ 76d2cfde-bdd8-4e45-83dd-92d3c651691f
struct NormalParametric{T} <: AbstractNormal
	μ :: T
	σ :: T
end

# ╔═╡ 1e36bd1d-cb83-4e48-a5dc-f88bf04636ca
p3 = NormalParametric(0f0, 1f0)  # float32 versions of 0e0 and 1e0

# ╔═╡ b088c77f-9732-4c63-88f9-9bcd911e461c
@code_warntype p3(0)

# ╔═╡ af5fffbd-baf5-46e4-b285-3a98a5d01e55
@time probability(p3, -1.96, 1.96)

# ╔═╡ 00ed2dc6-f770-49da-9eac-35042f437b6e
p4 = NormalParametric([0.0, 0.1], [0.5, 1.0])

# ╔═╡ 0f7f260c-fbb2-4661-be71-86fe23a51d92
sample(p4, 5)

# ╔═╡ 176bb2e7-dde9-4696-ab01-eea38a1081b8
-v ./ 2v  # multiply a number and a variable by juxtaposition (higher priority)

# ╔═╡ 8efda77f-e3d5-4866-8b64-159b6c3a6114
transpose(A)

# ╔═╡ d9f9542f-8d4f-4c0c-b4ea-986eefc07636
A'  # adjoint: complex conjugate followed by transpose

# ╔═╡ 17eeffee-701d-4251-aca7-308e456487da
let C = reshape(B, 2, 3), D = B', E = B[1:2,:], F = @view B[1:2,:]
	# B,C,D,F share the same underlying data
	C[1, 2] = NaN
	D[1, 2] = NaN
	F[3] = NaN
	E[:] .= NaN  # sets all values of D to NaN
	B, C, D, E, F
end

# ╔═╡ 820f0070-98b9-4bf6-a8db-65383e7c3c17
if !ismissing(Q)
	using Plots
	plt1 = scatter(eachrow(Q)...)
	plt2 = histogram(Q', bins=range(-3, 5.5, length=50))
	plot(plt1, plt2, layout=(2, 1), legend=false, size=(600, 800))
else
	md"Some plots will be generated after you finish the tasks."
end

# ╔═╡ 859c21c8-74cc-4db1-9a35-4e75e4a4ab66
v ⋅ v == dot(v, v) == norm(v) ^ 2 == v' * v  # \cdot <tab> => ⋅

# ╔═╡ 493a6c95-3820-43aa-8e6c-939757aecf2b
M - I  # I is identity matrix

# ╔═╡ 2c379af2-73d9-4470-8f7f-9dafa789e951
inv(M) * M == M ^ -1 * M ≈ I

# ╔═╡ 6287eddc-9b35-489e-b584-8197c09cb228
let b = [1, 2, 3]
	x = inv(M) * b  # or M \ b
	@show M * x
	N = [M M]  # size(N) = (3, 6)
	y = N \ b  # (least squares)
	@show pinv(N) * b ≈ y
	N * y
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
	findfirst(iszero, a)
	count(iseven, b)
	map(abs2, a)
	sum(abs2, a)  # == sum(map(abs2, a))
	map(-, a, b)  # applies a binary operator to each pair in two sequences
	zip(a, b, c) |> collect  # `a |> f` (pipe operator) is equivalent to `f(a)`
	map(min, a, b, c)
end

# ╔═╡ 39d49206-d018-496a-a3fe-bc7fa41ae7a0
# Evaluate polynomials using matrix-vector multiplication
md" ### Polynomial evaluation"

# ╔═╡ a9054fcc-dfae-407c-b059-afcb78e368e7
plot(f, legend=:bottomright)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractTrees = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Polynomials = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
AbstractTrees = "~0.4.5"
Plots = "~1.40.4"
PlutoUI = "~0.7.59"
Polynomials = "~4.1.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "0f001f52d922d505d7d3bf2fbd54a369bd7ec4d2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "3a3dfb30697e96a440e4149c8c51bf32f818c0f3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.17.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "1828eb7275491981fa5f1752a5e126e8f26f8741"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.17"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "27299071cc29e409488ada41ec7643e0ab19091f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.17+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "35fbd0cefb04a516104b8e183ce0df11b70a3f1a"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ed5e9c58612c4e081aecdb6e1a479e18462e041e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

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
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "4f34eaabe49ecb3fb0d58d6015e32fd31a733199"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.8"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

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
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

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
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

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
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9216a80ff3682833ac4b733caa8c00390620ba5d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.0+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "275a9a6d85dc86c24d03d1837a0010226a96f540"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.3+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "28ea788b78009c695eb0d637587c81d26bdf0e36"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.14"

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
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "2b2127e64c1221b8204afe4eb71662b641f33b82"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.66"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "972089912ba299fba87671b025cd0da74f5f54f7"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.0"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
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
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "eb38d376097f47316fe089fc62cb7c6d85383a52"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "e1d5e16d0f65762396f9ca4644a5f4ddab8d452b"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

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
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "b81c5035922cc89c2d9523afc6c54be512411466"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.5"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

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
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d2282232f8a4d71f79e85dc4dd45e5b12a6297fb"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.23.1"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

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
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "XML2_jll"]
git-tree-sha1 = "53ab3e9c94f4343c68d5905565be63002e13ec8c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.23.1+1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "c5bf2dad6a03dfef57ea0a170a1fe493601603f2"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.5+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cd155272a3738da6db765745b89e466fa64d0830"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.49+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─0939489d-79d2-4c1a-9841-17d9ae448d94
# ╟─b4cb3a82-d740-4d02-b0f4-f18ec9500b4f
# ╠═a025f4ac-9f39-4d05-9e0f-c12b9145d7c6
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
# ╠═128c2820-7676-4fcb-aa14-69b1478f9c68
# ╠═5c64daca-361a-4c3b-92e0-b179c834a63e
# ╟─9cde569d-7db4-4c06-8e03-346b32afaa16
# ╟─f2496da6-a024-44eb-b1ee-6cd5e213a86a
# ╠═79dd50f1-bd99-4384-b691-4bdb73096161
# ╟─45737508-e741-445d-86ef-850ab9915039
# ╠═88c296f4-51c7-4968-84d4-7d7d66288d8c
# ╠═0d376fb5-3b55-4607-903b-1a1777f77215
# ╟─5dedb5f1-1e4e-4b47-9e28-46d9d901f6ca
# ╠═636db2a4-c4f7-49ff-8cd6-9956e15e5f6e
# ╠═b541204e-3054-4504-b8f4-913209f19913
# ╟─1423d00d-8d72-4d84-ad47-95131d8b4bad
# ╠═6366c0dd-5ec4-435f-9b1e-27e5215f4cf9
# ╟─e34871a1-501c-4fe6-b904-71e541ee3d81
# ╠═1cf649c7-6d3c-4dda-bdb6-ce53d178a033
# ╠═6d0ee428-b93b-4859-809d-a59b45728cf6
# ╟─7c5c7303-474f-4214-a26e-ff98bcc1272b
# ╠═e94003d4-580e-454e-9330-b35bfe0bfce0
# ╠═70b0e880-be05-4170-98fc-9d0e2ff1df96
# ╠═43ca4fbd-f04d-4d94-af41-8272c605807f
# ╟─e1735f0a-6d3d-41bc-81d3-5eead9d41f18
# ╠═be99ab8d-1855-48e7-9b23-6e9021706569
# ╠═eb0eed0a-11a1-4468-afc3-4690ec65ee50
# ╟─09e2f1c0-633e-4a05-aa12-aa3189c1c854
# ╠═808d8684-e857-46b8-835f-6056079b1e77
# ╠═6d3abfce-8de4-45aa-9646-940fe329cf7a
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
# ╠═a77de155-4e6b-4985-a8d9-346417d5f83f
# ╠═2001b208-3e8b-491e-91f8-7524140afca0
# ╠═591d7e1c-c214-46ab-9402-a3a64881daa1
# ╠═c0b8be06-85fc-457d-b4b8-1548fd714433
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
# ╠═48b447d2-0ec1-4d42-985d-84bb3ce4c759
# ╠═18aab5fb-7add-4ada-b42e-2bc62968d6bc
# ╠═0c4a6998-8863-404e-96c2-952df70839ab
# ╟─02dda798-9681-41fb-afc1-ba2e24e786e8
# ╠═a9561c08-2b07-4590-b901-d9cbd60355ee
# ╟─2e034e29-8755-43d5-b557-d247df23f50e
# ╠═e3f7a77a-8c9e-4f15-af47-551fd959b2a6
# ╠═0f2aff9d-778b-4a08-9c33-c1866279c686
# ╠═a3fe5049-4dcb-4071-9618-6b637b20fcc7
# ╠═3c74c07d-98a5-48b8-bf6c-2a25e85597d5
# ╠═76d61e6d-16e8-440d-99f7-51a3775694b9
# ╠═47cd214a-ba2b-486f-b576-f2a583b50b7e
# ╠═9f56480f-52b2-4770-bf6e-9d7676756a87
# ╠═0c371cea-44f9-4703-964f-13d1a9f55535
# ╠═48e93319-299b-40b9-bbf9-09d18d683c9c
# ╠═fa1283d5-b3d5-46d4-a34c-4cddc32ab284
# ╠═322ea469-2961-46b0-a93c-20e2c8f94328
# ╠═b13074cb-0a3a-48b7-97ac-b9ef93fa184a
# ╠═eeb1f9c3-6342-4ff3-a731-77ec4a55ebd1
# ╠═cc45cdea-38c6-4c06-b62c-09a36559bfd6
# ╟─beb7b5f4-ee86-4130-aa61-d3f8498ff4ed
# ╠═ec5238e4-f445-491c-bd14-8e1aba59049f
# ╟─f3b4eba4-5471-441e-b199-69fd07f528e2
# ╠═cfeb3928-cc2f-47a3-8a9b-e17eabd79a33
# ╠═c6739f52-f87f-4bef-8c32-ce3ec4942342
# ╠═035f9794-43ea-4e19-860c-a66fd0ea1a14
# ╠═57f30a3c-7d28-4819-958a-bf1859d6947c
# ╠═ed7082dc-cd39-4488-842c-1f05968224bf
# ╠═024aa7d5-a569-4639-851f-b7d491855202
# ╠═f640df71-ae15-4b67-a30e-c806ea532a19
# ╟─a4c7126d-57dd-4542-bcc4-d01cf657759a
# ╠═76d2cfde-bdd8-4e45-83dd-92d3c651691f
# ╠═1e36bd1d-cb83-4e48-a5dc-f88bf04636ca
# ╠═b088c77f-9732-4c63-88f9-9bcd911e461c
# ╟─74c57fe8-e369-44f1-a51e-8365e4ffed5d
# ╠═00ed2dc6-f770-49da-9eac-35042f437b6e
# ╠═0f7f260c-fbb2-4661-be71-86fe23a51d92
# ╟─2e6521be-ff66-47a9-8c19-68216cb62f3d
# ╠═149a64ba-6d5b-4416-bc2d-8e1ae897c71d
# ╠═d00e9d96-59c7-4bd6-9667-340505d5ed5f
# ╠═8e8a900f-1d6c-4d65-afda-b03e64f3c9c8
# ╠═af5fffbd-baf5-46e4-b285-3a98a5d01e55
# ╟─7b6e1d43-c72c-4bd9-b493-838b05e845c4
# ╟─63eddb5a-960c-43c6-9425-5caa40f4802f
# ╠═760ff5fd-689b-4afe-9336-cc480fb6b486
# ╠═51c754f6-ba17-4936-8e1e-89899634e37d
# ╠═4da224f7-7e68-425f-b575-877807efa884
# ╠═38f6649c-5daa-43f1-900f-98381b6d33fc
# ╟─aa08b116-025a-43cd-8f0d-74e035b9746d
# ╠═69283b2e-bd47-4c3c-890f-677b253183e7
# ╠═a2c92fca-fbab-4396-b472-a53d7a858abe
# ╠═d7186b34-117c-4a11-8907-91766a038425
# ╠═0f3b3f22-89f3-491d-be29-57438d83f4cd
# ╠═2c0b579b-302c-458e-bfb0-75ce768de5bd
# ╠═176bb2e7-dde9-4696-ab01-eea38a1081b8
# ╠═b3321c01-db3d-42ed-9ea7-142e8773bc28
# ╠═dbdbd2b6-5831-48da-a9a0-8052e96a5586
# ╠═88e5862d-0859-4ac7-b8a1-400ab4b10c18
# ╠═89b93f51-6e10-482e-85e2-9fc6ece8bf53
# ╠═5e438baa-fc94-423a-a924-280405fa4255
# ╟─0f5a46ab-6108-4683-80e0-8f1acaec7c7f
# ╠═8cc1e1ca-207e-4dc3-b860-2c5c2114a49a
# ╠═90a98f2a-6d97-4697-a4a7-ab1cac19d9e1
# ╠═071a0163-3071-4398-bc46-d12c11bbcba0
# ╟─ce603931-baa5-48aa-ba13-82b458962ddf
# ╠═3cfce228-b634-4e31-b3f3-ddadb6c7a53d
# ╠═6b3a83eb-e316-46b5-a097-233145ab1bcc
# ╠═4f62d53f-11bb-4e53-b759-d6f49eec5cd4
# ╠═952db525-9d54-4b56-a09f-3014a9ca9293
# ╠═ae856b3a-795a-4f99-90d0-c5c9ffacc3e9
# ╠═d02b8c20-6e43-435c-ba9f-870b1bb5fae9
# ╠═b5eb64a4-6572-405f-bed4-7e483f6e50e5
# ╠═9c670e29-f48a-4f4e-a9a1-425f76a1f006
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
# ╠═a5717c60-1abe-4164-a4c0-45708212f95d
# ╠═17eeffee-701d-4251-aca7-308e456487da
# ╠═aad8b0cc-4223-4309-bfee-f5c57e08e353
# ╟─dfaf45ea-ff3c-4b88-b033-a6443a57b598
# ╠═e00f9ae1-3f93-41f4-bb5a-71a6cf8062b3
# ╠═0b72c227-b7cf-4725-a4e6-baa5fec50e6a
# ╠═77467cf8-92a0-4c15-81e5-b26e4b1fc47f
# ╠═c7356050-5ff8-4d8d-af6b-7adf981ebf29
# ╟─5184e9de-f120-408f-927e-38dfe07d135a
# ╠═db0de71e-d39c-4be3-a4f8-774f80ef54d1
# ╠═5749d084-2415-4046-a676-5b84e772795e
# ╟─48840624-e055-4f56-baee-91ebe578dbf3
# ╠═dfbfc391-1568-4e23-b49a-1dcc2b48aede
# ╠═34ea929c-4676-44c2-9dfd-2235a7b7414a
# ╟─52554079-c69e-41f3-a936-de8536f0c1b3
# ╠═32d4c56f-99da-4e04-a2ea-8a741938d695
# ╠═8b3266a8-23cb-4083-9048-bcffca178a75
# ╠═e3ac2be4-6737-4924-ad75-a8339ac2be2c
# ╠═5892411a-514b-4123-8864-a8c8c1e65bbc
# ╠═add58d1a-1462-4ba4-986f-d03021fd932d
# ╟─b190acf8-4abf-4790-a75d-602ecba34ce0
# ╟─b4f5f60e-d064-45cc-b5c0-8ed5e3f8488e
# ╠═b8ddb960-6d7f-44bf-949c-fa46bd2bc8c6
# ╠═365933f6-6399-4bff-9332-85f6c7b20909
# ╠═0b057b26-e5e8-4eb0-a01b-7fbc56497f6c
# ╠═b6c9b6ae-9663-4170-999a-ffb87d468121
# ╠═34f03b15-5990-4e21-b9cc-10ee092fbc33
# ╠═0c5d85fd-d81f-4df4-805c-860e787dbcc5
# ╠═1fb11bbe-e970-41a4-88eb-09b269fd4127
# ╠═e102112f-b994-4088-8a33-4c2719fc7ef7
# ╠═892ecca5-d7ff-4b37-a5aa-b2890914b331
# ╠═bc301f12-6a28-4106-aac1-fa4015be6f25
# ╠═beac6982-09bf-4a27-b2e4-743bbcf7fb5f
# ╠═4e70b378-9e33-40b2-9b63-a9d9bc1be5cc
# ╠═83a1561b-01b0-4a7b-bc83-2ea11a2ba28e
# ╠═84d392f4-40e0-4abd-a1f7-fc8b33796394
# ╠═9782436f-5cf6-45b8-a4b4-5477b6c20c80
# ╠═bedc8c99-73f1-4360-8162-907084f93bf0
# ╠═78215d5d-61b0-4dcc-ab96-0a795aa0d60d
# ╠═7a7a283b-ae93-4ead-9a71-9242d1b78390
# ╠═fad551be-abbc-45c6-b08c-5e8d4ddccdb0
# ╠═fbd9a83b-17b4-47db-a46e-e7a9037b9090
# ╟─26f43214-3b99-4c99-9512-398a28f9ae0a
# ╠═24077fc9-4d06-4b80-91be-321a7bb0fe5c
# ╠═b226106d-6f21-4d72-951c-c4d9d01cbbcb
# ╟─aa0c8fec-254b-4805-bf07-b1ce7266685c
# ╟─8615c4ca-7e2b-49fb-bb0f-078347a7c56b
# ╟─be7f3b8d-70e6-4ec3-a98f-07fbe17fb06a
# ╟─820f0070-98b9-4bf6-a8db-65383e7c3c17
# ╟─66cae8d2-8e20-4b1e-9dae-e120eee4d944
# ╠═5af22ae0-effd-4589-bd1f-d375299b6848
# ╠═5ee4f31b-ebae-4d8f-8ccc-6df671de6965
# ╠═859c21c8-74cc-4db1-9a35-4e75e4a4ab66
# ╠═493a6c95-3820-43aa-8e6c-939757aecf2b
# ╠═2c379af2-73d9-4470-8f7f-9dafa789e951
# ╠═6287eddc-9b35-489e-b584-8197c09cb228
# ╠═3f6fbfd0-b35a-4af9-86cd-55d7e4188301
# ╠═2dde11e3-dcc7-416b-b351-bcf526f3deaa
# ╠═5fbbf58e-2c28-4b80-b524-49f881258f46
# ╟─8bc7e78b-ff6d-4553-b327-f03d21651121
# ╠═39d49206-d018-496a-a3fe-bc7fa41ae7a0
# ╠═80caa1fe-0c8c-48a5-9bf8-9b818b70aa18
# ╠═a9054fcc-dfae-407c-b059-afcb78e368e7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
