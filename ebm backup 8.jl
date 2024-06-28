### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 1fbb5414-3483-11ef-2f91-affe69c5f577
begin
	using Plots, CSV, DataFrames, NaNStatistics, DifferentialEquations, ModelingToolkit
	import PlotlyJS
	plotlyjs()
end

# ╔═╡ 981f864e-da3f-4f27-996b-3172ae0fe76f


# ╔═╡ c5765cae-28fd-439e-b5ce-844b5637d2e4
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

# ╔═╡ d6ecc837-be55-4db7-bac7-8446e7a3cfa2
begin
	CO2_historical_data = subset(CO2_historical_data_raw, "Year" => y -> y .>= 1850)
	values = replace(Matrix(CO2_historical_data[:,2:end]), missing=>NaN)
	CO2_historical_data.CO2 = reshape(nanmean(values, dims=2), :)
	select!(CO2_historical_data, :Year, :CO2)
	first(CO2_historical_data, 5), last(CO2_historical_data, 5)
end

# ╔═╡ 8264a6cf-803c-4f37-b516-b82be1000a38
# Task 1: fit a polynomial to the Keeling curve
begin
	CO2(t) = features(t .- 1850) * CO2_params
	features(t) = hcat(ones(length(t)), t.^3)
	CO2_params = let
		t = CO2_historical_data[:, "Year"] .- 1850
		y = CO2_historical_data[:, "CO2"]
		X = features(t)
		p = X \ y  # least squares
	end
end

# ╔═╡ f539ddd6-041f-4b84-91c0-aca74131c960
begin
	years = 1850:2030
	let df = CO2_historical_data
		plot(df[:, "Year"] , df[:, "CO2"], 
			 label="Global atmospheric CO₂ concentration")
		plot!(years, CO2(years), label="Fitted curve", legend=:bottomright)
	end
	title!("CO₂ observations and fit")
end

# ╔═╡ c5d362b6-e0ed-4742-9ec6-1d1a9d03ac58
begin
	@parameters t α a S β γ C
	@variables Y(t) RC(t)

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

# ╔═╡ 70db5d4d-e346-476a-8bba-50070260abe5
@mtkbuild sys = ODESystem(eqs, t)

# ╔═╡ 4084d556-5673-444f-bb9f-e77a502799d5
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

# ╔═╡ 29f5d3a1-b805-4c90-b602-0d2de83d0152
begin
	temps = vcat(solve(prob).(30:175)...)
	plot(1880:2025, temps, lw=2, legend=:topleft,
		 label="Predicted Temperature from model")
	xlabel!("year")
	ylabel!("Temp °C")
end

# ╔═╡ b29d66d4-ef28-4de8-acc1-7d4fe607742b
begin
	T_url = "https://data.giss.nasa.gov/gistemp/graphs/graph_data/Global_Mean_Estimates_based_on_Land_and_Ocean_Data/graph.txt"
	s = read(download(T_url), String)
	io = replace(s, r" +" => " ") |> IOBuffer
	T_df = CSV.read(io, DataFrame, header=false, skipto=6);
	T_df = rename(T_df[:,1:2], :Column1=>:year, :Column2=>:temp)
	T_df.temp .+= 14.15
	T_df
end

# ╔═╡ 7f4af72c-9ca1-4b65-8557-9d59880ba261
plot!(T_df[:, :year], T_df[:, :temp], 
	  color=:black, label="NASA Observations", legend=:topleft)

# ╔═╡ a2bd2e6c-4908-472a-9cd7-b7bf22c313e2
md"""The reason why the predicted temperature is lower than the observation is probably that we have not taken into account other greenhouse gases and feedback factors such as water vapour."""

# ╔═╡ fc28c0f7-0182-40ac-863a-552ec1ce452e


# ╔═╡ Cell order:
# ╠═1fbb5414-3483-11ef-2f91-affe69c5f577
# ╠═c5765cae-28fd-439e-b5ce-844b5637d2e4
# ╠═d6ecc837-be55-4db7-bac7-8446e7a3cfa2
# ╠═8264a6cf-803c-4f37-b516-b82be1000a38
# ╠═f539ddd6-041f-4b84-91c0-aca74131c960
# ╠═c5d362b6-e0ed-4742-9ec6-1d1a9d03ac58
# ╠═70db5d4d-e346-476a-8bba-50070260abe5
# ╠═4084d556-5673-444f-bb9f-e77a502799d5
# ╠═29f5d3a1-b805-4c90-b602-0d2de83d0152
# ╠═b29d66d4-ef28-4de8-acc1-7d4fe607742b
# ╠═7f4af72c-9ca1-4b65-8557-9d59880ba261
# ╟─a2bd2e6c-4908-472a-9cd7-b7bf22c313e2
# ╠═981f864e-da3f-4f27-996b-3172ae0fe76f
# ╠═fc28c0f7-0182-40ac-863a-552ec1ce452e
