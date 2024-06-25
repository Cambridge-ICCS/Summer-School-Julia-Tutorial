### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 858d6813-dab2-4dd1-a899-03def1126ef0
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

# ╔═╡ 677a25e8-3242-11ef-3ab3-afbfdd010dc6
fib(n) = n < 2 ? big(n) : fib(n-2) + fib(n-1)

# ╔═╡ eb1a84cc-0a60-410b-ae92-7d9d730acbef
@time fib(32)

# ╔═╡ 3bc19fad-02b8-456d-909d-ef931f7dbbed
let f(g) = n -> n < 2 ? n : g(n-1) + g(n-2)
	nfold(f, n) = foldr(∘, fill(f, n))
	partial_fib(i) = nfold(f, i)(x -> NaN)
	for i in 1:8
		println(partial_fib(i).(1:8))
	end
	Y_fib = (x -> f(y -> x(x)(y)))(x -> f(y -> x(x)(y)))  # Y combinator
	Y_fib.(1:8)  # f(f(f(f(...))))
end

# ╔═╡ c3b6e79a-5a49-4b36-9199-f76073198834
begin
	@memoize mfib(n) = n < 2 ? big(n) : mfib(n-2) + mfib(n-1)
	@time @show fib.(1:32)
	@time @show mfib.(1:32)
end

# ╔═╡ 8c04ba16-4251-4bc1-a249-72a0f472f749
macroexpand(@__MODULE__, :(@memoize mfib(n) = n < 2 ? big(n) : mfib(n-2) + mfib(n-1)))

# ╔═╡ cc3649fe-5e86-4d9d-ab6e-a34e0f362b9d
md"A faster option: calling C function."

# ╔═╡ f4877904-acf8-457c-87db-df7ccd79b8e8
function fastfib(n)
    z = BigInt()
    ccall((:__gmpz_fib_ui, :libgmp), Cvoid, (Ref{BigInt}, Culong), z, n)
    return z
end

# ╔═╡ 6fa7cbdb-3f82-4ea2-b652-3b2ba2f1718d
@time fastfib.(1:32)

# ╔═╡ 8cc35c40-8a02-4d3c-990b-3fb3bc3c2b7b


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
MacroTools = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"

[compat]
MacroTools = "~0.5.13"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "a2180b5bf47049875ed29826fd5f189b00c803b5"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"
"""

# ╔═╡ Cell order:
# ╠═677a25e8-3242-11ef-3ab3-afbfdd010dc6
# ╠═eb1a84cc-0a60-410b-ae92-7d9d730acbef
# ╠═3bc19fad-02b8-456d-909d-ef931f7dbbed
# ╟─858d6813-dab2-4dd1-a899-03def1126ef0
# ╠═c3b6e79a-5a49-4b36-9199-f76073198834
# ╠═8c04ba16-4251-4bc1-a249-72a0f472f749
# ╟─cc3649fe-5e86-4d9d-ab6e-a34e0f362b9d
# ╠═f4877904-acf8-457c-87db-df7ccd79b8e8
# ╠═6fa7cbdb-3f82-4ea2-b652-3b2ba2f1718d
# ╠═8cc35c40-8a02-4d3c-990b-3fb3bc3c2b7b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
