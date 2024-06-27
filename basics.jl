### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ a9561c08-2b07-4590-b901-d9cbd60355ee
begin
	using AbstractTrees
	AbstractTrees.children(d::DataType) = subtypes(d)
	print_tree(Number)
end

# ╔═╡ 52ab5184-2f0f-11ef-3034-8fd6a5c8a2cb
1 + 2 * 3

# ╔═╡ 0f63f358-310c-4475-a17b-6376ce26f903
@show log2(4) log(ℯ) log10(1e4) log(4, 1024) sqrt(4) exp(4) cos(0) acos(0);

# ╔═╡ 3ae5a286-cc9d-4837-a6de-c79bad078df4
z = exp(im * π)

# ╔═╡ b4de91e1-ef8a-44ae-ac31-ac99d0a041d2
z == -1, z ≈ -1  # isapprox

# ╔═╡ 79dd50f1-bd99-4384-b691-4bdb73096161
let θ = rand(), z = exp(im * θ)  # bind variables locally
	x, y = @show reim(z)
	x ^ 2 + y ^ 2 == abs(z) ^ 2
end

# ╔═╡ 13104a6c-0eb7-42d7-961d-addc55f06588
md"## Types and Fields"

# ╔═╡ b9ddc629-f680-4a71-8374-f3b01bb53890
typeof(z)  # type alias

# ╔═╡ bcacdf8c-695f-4590-b7d8-29d28086bd46
fieldtypes(typeof(z)), z.re, getfield(z, :im)

# ╔═╡ 45bd459d-024d-4c30-8ba0-acd3cd0e2eb3
typeof(:im)

# ╔═╡ 936bc73b-681e-46a5-b7a5-4bb071cc91a5
let T = Complex{Int64}
	@show T<:Complex T<:Number T<:Complex{<:Real} T<:Complex{Real}
	Dict(zip(fieldnames(T), fieldtypes(T)))
end

# ╔═╡ 002bd083-00d2-4fd6-965f-9415d85f23f6
subtypes(Integer), supertypes(Integer)

# ╔═╡ 18aab5fb-7add-4ada-b42e-2bc62968d6bc
isabstracttype(Integer)

# ╔═╡ 0c4a6998-8863-404e-96c2-952df70839ab
isconcretetype(Int)

# ╔═╡ 7b6e1d43-c72c-4bd9-b493-838b05e845c4
md"### Arrays"

# ╔═╡ 86242e6d-b229-4317-8f32-e92a4ad3de3e
[1, 2, 3] isa Vector{<:Number}

# ╔═╡ 510a8018-b160-42ea-9787-0bbb7bd890d2
Array{Int, 1}

# ╔═╡ 3cfce228-b634-4e31-b3f3-ddadb6c7a53d
Array{Int, 2}

# ╔═╡ 4f62d53f-11bb-4e53-b759-d6f49eec5cd4
let a = Array{Float64}(undef, 2, 3)  # new 2x3 Matrix of Float64s
	for i in 1:2, j in 1:3
		a[i, j] = i * j
	end
	a
end

# ╔═╡ 952db525-9d54-4b56-a09f-3014a9ca9293
[i * j for i in 1:2, j in 1:3]

# ╔═╡ 6b3a83eb-e316-46b5-a097-233145ab1bcc
[1 2 3
 5 6 4
 9 7 8]  # or [1 2 3; 5 6 4; 9 7 8]

# ╔═╡ d02b8c20-6e43-435c-ba9f-870b1bb5fae9
zeros(3, 3)

# ╔═╡ 8bc03ce0-2fe3-45ca-9c1a-9bd2a98bc41e
A = rand(Float64, (3, 4))

# ╔═╡ d1ca8fb0-580f-4625-aba3-dd18e054ee48
size(A), size(A, 1)

# ╔═╡ 8cbce37e-e384-47a3-b345-7f77699cfc8c
let B = @show similar(A)
	fill!(B, 3)
	B[2, :] .= 1
	B[10] = NaN
	B
end

# ╔═╡ 8ea9ecaf-6d66-4e57-8606-e79fdc8415e5
[A[:, 3:4]; A[[1,3], 1:2:end]]  # concat vertically

# ╔═╡ 2f512a32-8e03-4ef1-9a09-d5a388f06823
vcat(A[:, 3:4], A[[1,3], 1:2:end])

# ╔═╡ 12008adf-5162-484c-af6b-30b2d43f46b5
[sum(A .^ 2, dims=2) maximum(A, dims=2)]  # concat horizontally

# ╔═╡ 9cb3f794-5696-4c9d-adf1-5d1f31ae8c00
diff(cumsum(A, dims=2), dims=2) ≈ A[:, 2:end]

# ╔═╡ 17eeffee-701d-4251-aca7-308e456487da
let B = reshape(A, 2, 6)
	B[2, 3] = NaN
	C = @view B[1:2, 2:3]
	D = copy(C)
	i = @show findfirst(isnan.(A))
	A[i] = -1
	C, D
end

# ╔═╡ 5af22ae0-effd-4589-bd1f-d375299b6848
M = [i + j*im for i in 1:3, j in 1:3]

# ╔═╡ 0db34f03-95ea-4ed8-b652-abd8b3b200b8
begin
	using Statistics
	mean(M), mean(M; dims=1)
end

# ╔═╡ 5ee4f31b-ebae-4d8f-8ccc-6df671de6965
begin
	using LinearAlgebra
	rank(M), tr(M), det(M), diag(M)
end

# ╔═╡ 50c86554-ff09-4e4a-94e8-0f30b83e8655
@show 3+4 3*4 3/4 3÷4 4%3 3^4;

# ╔═╡ 1a2adde2-cb88-4877-8beb-edf9996b9d7e
M isa Array{<:Number, 2}

# ╔═╡ e3201408-e6b6-49be-b693-65d55b20be5f
M', transpose(M)

# ╔═╡ 83d0d182-c876-4aa2-a6f3-dfa92477bdcd
M ^ 2, exp(M)

# ╔═╡ 69d5b274-e06d-4bd7-b8a7-5127d1c02926
names(Statistics)

# ╔═╡ eb842ef0-5547-4a9e-8c5b-ff62b028b478
methods(mean)

# ╔═╡ a855b343-35e2-45ff-9957-9b6d3a2e7180
@which mean(M; dims=1)

# ╔═╡ 6287eddc-9b35-489e-b584-8197c09cb228
let b = [3, 2, 1]
	x = @show M \ b  # inv(M) * b
	M * x
end

# ╔═╡ 3f6fbfd0-b35a-4af9-86cd-55d7e4188301
let eig = eigen(M)
	@show typeof(eig)
	@show eig.values
	@show eig.vectors
	λ, V = eig
	M * V ≈ λ' .* V
end

# ╔═╡ 6bcb5989-0012-458c-a5a2-5f977fc781d6
md"## Functions"

# ╔═╡ 39a9ed81-ad29-45ef-a199-045a4634eee0
factorial(5)

# ╔═╡ 7996e940-12d0-4c90-b173-9f04b2ede3d0
factorial(32)

# ╔═╡ e9b56975-891a-4cf9-b4e6-7ff72fa4235b
let factorial(n) = n < 2 ? big(1) : n * factorial(n-1)
	@show factorial(32)
	@time factorial.(0:32)
end

# ╔═╡ e6e2109f-07f7-4bdb-a44b-075125de8cf1
let
	function factorial(n)
		if !(n isa Integer)
			throw(ValueError("input is not an integer"))
		elseif n < 0
			throw(ValueError("input cannot be negative"))
		else
			prod(1:big(n))
		end
	end
	@time factorial.(0:32)
end

# ╔═╡ 72bdde5a-a503-4cc4-bd06-de49794e53b5
let
	# arguments before the ; are positional, after are keyword
	# there can be defaults in both categories
	# anything without a default must be assigned when the function is called
	# ... before the ; accepts any number of positional arguments
	# ... after the ; accepts any keyword arguments
	# the names args and kwargs are conventional for these extra arguments
	function f(a, b=0, args...; c, d=1, kwargs...)
		@show a b args c d kwargs
	end
	f('a', 2, 3, 4; c=3, e=7)
	println()
	f(1; c=7)
end

# ╔═╡ abeca27b-685a-451c-9f18-39e5c2f87472
begin
	⟹(p::Bool, q::Bool) = !p | q  # \implies
	⟺(p::Bool, q::Bool) = (p ⟹ q) & (q ⟹ p)  # \iff
	bools = Set([true, false])
	# equivalence of contrapositive statements
	all((p ⟹ q) ⟺ (!q ⟹ !p) for p ∈ bools, q ∈ bools)
	# see https://github.com/JuliaLang/julia/blob/master/src/julia-parser.scm for the symbols that can be defined as infix binary operators
end

# ╔═╡ 7af106cf-3e7b-497d-95c1-b90c09b048e5
md"### Higher Order Functions"

# ╔═╡ 1396345b-8abf-48ac-8bfa-6c641a395c2c
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

# ╔═╡ 4c2d5a36-acaf-46ae-bb5a-a28d8ded855c
let f(g) = n -> n < 2 ? big(1) : n * g(n-1)
	partial_fact(i) = nfold(f, i)(x -> NaN)
	for i in 1:8
		println(partial_fact(i).(1:8))
	end
	Y_fact = (x -> f(y -> x(x)(y)))(x -> f(y -> x(x)(y)))  # Y combinator
	Y_fact.(1:8)  # f(f(f(f(...))))
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

# ╔═╡ a3ec0bc0-f187-4a1d-9ecd-bc3f1ebe5b22
@show_all let exp = :(exp(im * π))
	typeof(exp)
	eval(exp)
	print_tree(exp)
	exp.args[2] = :0
	print_tree(exp)
	eval(exp)
end

# ╔═╡ fbd9a83b-17b4-47db-a46e-e7a9037b9090
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
	push!(b, 5, 4)
	sort!(b)
	deleteat!(b, 1)
	accumulate(*, b)
	foldl(-, b)  # 2 - 3 - 4 - 5 (starting from the left)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractTrees = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
AbstractTrees = "~0.4.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "a6d346e1c03e1edc8ffa82b03d585d4f1f2eb498"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

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

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"
"""

# ╔═╡ Cell order:
# ╠═52ab5184-2f0f-11ef-3034-8fd6a5c8a2cb
# ╠═50c86554-ff09-4e4a-94e8-0f30b83e8655
# ╠═0f63f358-310c-4475-a17b-6376ce26f903
# ╠═3ae5a286-cc9d-4837-a6de-c79bad078df4
# ╠═b4de91e1-ef8a-44ae-ac31-ac99d0a041d2
# ╠═79dd50f1-bd99-4384-b691-4bdb73096161
# ╟─13104a6c-0eb7-42d7-961d-addc55f06588
# ╠═b9ddc629-f680-4a71-8374-f3b01bb53890
# ╠═bcacdf8c-695f-4590-b7d8-29d28086bd46
# ╠═45bd459d-024d-4c30-8ba0-acd3cd0e2eb3
# ╠═936bc73b-681e-46a5-b7a5-4bb071cc91a5
# ╠═002bd083-00d2-4fd6-965f-9415d85f23f6
# ╠═18aab5fb-7add-4ada-b42e-2bc62968d6bc
# ╠═0c4a6998-8863-404e-96c2-952df70839ab
# ╠═a9561c08-2b07-4590-b901-d9cbd60355ee
# ╠═a3ec0bc0-f187-4a1d-9ecd-bc3f1ebe5b22
# ╟─7b6e1d43-c72c-4bd9-b493-838b05e845c4
# ╠═86242e6d-b229-4317-8f32-e92a4ad3de3e
# ╠═510a8018-b160-42ea-9787-0bbb7bd890d2
# ╠═3cfce228-b634-4e31-b3f3-ddadb6c7a53d
# ╠═4f62d53f-11bb-4e53-b759-d6f49eec5cd4
# ╠═952db525-9d54-4b56-a09f-3014a9ca9293
# ╠═fbd9a83b-17b4-47db-a46e-e7a9037b9090
# ╠═6b3a83eb-e316-46b5-a097-233145ab1bcc
# ╠═d02b8c20-6e43-435c-ba9f-870b1bb5fae9
# ╠═8bc03ce0-2fe3-45ca-9c1a-9bd2a98bc41e
# ╠═d1ca8fb0-580f-4625-aba3-dd18e054ee48
# ╠═8cbce37e-e384-47a3-b345-7f77699cfc8c
# ╠═8ea9ecaf-6d66-4e57-8606-e79fdc8415e5
# ╠═2f512a32-8e03-4ef1-9a09-d5a388f06823
# ╠═12008adf-5162-484c-af6b-30b2d43f46b5
# ╠═9cb3f794-5696-4c9d-adf1-5d1f31ae8c00
# ╠═17eeffee-701d-4251-aca7-308e456487da
# ╠═5af22ae0-effd-4589-bd1f-d375299b6848
# ╠═1a2adde2-cb88-4877-8beb-edf9996b9d7e
# ╠═e3201408-e6b6-49be-b693-65d55b20be5f
# ╠═83d0d182-c876-4aa2-a6f3-dfa92477bdcd
# ╠═0db34f03-95ea-4ed8-b652-abd8b3b200b8
# ╠═69d5b274-e06d-4bd7-b8a7-5127d1c02926
# ╠═eb842ef0-5547-4a9e-8c5b-ff62b028b478
# ╠═a855b343-35e2-45ff-9957-9b6d3a2e7180
# ╠═6287eddc-9b35-489e-b584-8197c09cb228
# ╠═5ee4f31b-ebae-4d8f-8ccc-6df671de6965
# ╠═3f6fbfd0-b35a-4af9-86cd-55d7e4188301
# ╟─6bcb5989-0012-458c-a5a2-5f977fc781d6
# ╠═39a9ed81-ad29-45ef-a199-045a4634eee0
# ╠═7996e940-12d0-4c90-b173-9f04b2ede3d0
# ╠═e9b56975-891a-4cf9-b4e6-7ff72fa4235b
# ╠═e6e2109f-07f7-4bdb-a44b-075125de8cf1
# ╠═72bdde5a-a503-4cc4-bd06-de49794e53b5
# ╠═abeca27b-685a-451c-9f18-39e5c2f87472
# ╟─7af106cf-3e7b-497d-95c1-b90c09b048e5
# ╠═1396345b-8abf-48ac-8bfa-6c641a395c2c
# ╠═4c2d5a36-acaf-46ae-bb5a-a28d8ded855c
# ╟─8bc7e78b-ff6d-4553-b327-f03d21651121
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
