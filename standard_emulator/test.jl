@everywhere begin
using Turing
using Capse
using Lux
using NPZ
using PlanckLite
using BenchmarkTools
using Plots
using PlanckLite
using Zygote
using LinearAlgebra
using Pathfinder
using MicroCanonicalHMC
using Transducers
using PairPlots
using CairoMakie
using LaTeXStrings
using StatsPlots
include("utils.jl")
end

@everywhere begin


CℓTT_emu = Capse.load_emulator("/home/marcobonici/Desktop/work/papers/capse_paper/weights/weights_cosmopowerspace_10000/TT/", emu = Capse.SimpleChainsEmulator)

x = rand(6)
end

@everywhere begin
y1 = Capse.get_Cℓ(x, CℓTT_emu)

CℓTT_emu = Capse.load_emulator("/home/marcobonici/Desktop/work/papers/capse_paper/weights/weights_cosmopowerspace_10000/TT/", emu = Capse.LuxEmulator);

CℓTE_emu = Capse.load_emulator("/home/marcobonici/Desktop/work/papers/capse_paper/weights/weights_cosmopowerspace_10000/TE/", emu = Capse.LuxEmulator);

CℓEE_emu = Capse.load_emulator("/home/marcobonici/Desktop/work/papers/capse_paper/weights/weights_cosmopowerspace_10000/EE/", emu = Capse.LuxEmulator);

y2 = Capse.get_Cℓ(x, CℓTT_emu)

lsTT = 2:2508
lsTE = 2:1996
facTT=lsTT.*(lsTT.+1)./(2*π)
facTE=lsTE.*(lsTE.+1)./(2*π)

function call_emu_planck(θ, Emu_TT, Emu_TE, Emu_EE, facTT, facTE)
    return PlanckLite.bin_Cℓ(Capse.get_Cℓ(θ, Emu_TT)[1:2507]./facTT,
                            Capse.get_Cℓ(θ, Emu_TE)[1:1995]./facTE,
                            Capse.get_Cℓ(θ, Emu_EE)[1:1995]./facTE)
end

theory_planck(θ) = call_emu_planck(θ, CℓTT_emu, CℓTE_emu, CℓEE_emu, facTT, facTE)



Γ = sqrt(PlanckLite.cov)
iΓ = inv(Γ)
D = iΓ * PlanckLite.data;

@model function CMB_planck(D, iΓ)
    #prior on model parameters
    ln10As ~ Uniform(0.25, 0.35)
    ns     ~ Uniform(0.88, 1.06)
    h      ~ Uniform(0.60, 0.80)
    ωb     ~ Uniform(0.1985, 0.25)
    ωc     ~ Uniform(0.08, 0.20)
    τ      ~ Normal(0.0506, 0.0086)
    yₚ     ~ Normal(1.0, 0.0025)

    θ = [10*ln10As, ns, 100*h, ωb/10, ωc, τ]

    #compute theoretical prediction
    pred = iΓ * theory_planck(θ) ./(yₚ^2)
    #compute likelihood
    D ~ MvNormal(pred, I)

    return nothing
end

CMB_model_planck = CMB_planck(D, iΓ);
end
nsteps = 4000
nadapts = 500
nchains = 12

result_multi = multipathfinder(CMB_model_planck, 2000; nruns=12, executor=Transducers.PreferParallel())

init_params = collect.(eachrow(result_multi.draws_transformed.value[1:nchains, :, 1]));

chains_planck_nuts_fd = sample(CMB_model_planck, NUTS(nadapts, 0.75, adtype=AutoForwardDiff()), MCMCDistributed(), nsteps, nchains; init_params)

chains_planck_nuts_zy = sample(CMB_model_planck, NUTS(nadapts, 0.75, adtype=AutoZygote()), MCMCDistributed(), nsteps, nchains; init_params)


d = 7
nadapts = 5_000
nsteps = 60_000
spl = MCHMC(nadapts, 0.001; init_eps=0.05, L=sqrt(d), sigma=ones(d),
            adaptive=true, tune_L = false)

chains_planck_mchmc_fd = sample(CMB_model_planck, externalsampler(spl, adtype=AutoForwardDiff()), MCMCDistributed(),  nsteps, nchains; init_params = init_params[1])
chains_planck_mchmc_zy = sample(CMB_model_planck, externalsampler(spl, adtype=AutoZygote()), MCMCDistributed(),  nsteps, nchains; init_params = init_params[1])

plot_nuts_fd = PairPlots.Series(
    chains_planck_nuts_fd,
    label=L"\mathrm{NUTS}\quad\mathrm{ForwardDiff}",
    color=:red,
) => (
    PairPlots.Contour(
        sigmas=[1,2],
        color=(:red, 1.0),
        bandwidth=1,
        linewidth=3,
        linestyle=:dash
        # bandwidth is the smoothing
    ),
    PairPlots.MarginDensity(
        linewidth=3,
        bandwidth=1,
    ),
)

plot_nuts_zy = PairPlots.Series(
    chains_planck_nuts_zy,
    label=L"\mathrm{NUTS}\quad\mathrm{Zygote}",
    color=:orange,
) => (
    PairPlots.Contourf(
        sigmas=[1,2],
        color=(:orange, 0.2),
        bandwidth=1,
        # bandwidth is the smoothing
    ),
    PairPlots.MarginDensity(
        linewidth=3,
        bandwidth=1,
    ),
)

plot_mchmc_fd = PairPlots.Series(
    chains_planck_mchmc_fd,
    label=L"\mathrm{MCHMC}\quad\mathrm{ForwardDiff}",
    color=:green,
) => (
    PairPlots.Contourf(
        sigmas=[1,2],
        color=(:green, 0.2),
        bandwidth=3,
        # bandwidth is the smoothing
    ),
    PairPlots.MarginDensity(
        linewidth=3,
        bandwidth=1,
    ),
)

plot_mchmc_zy = PairPlots.Series(
    chains_planck_mchmc_zy,
    label=L"\mathrm{MCHMC}\quad\mathrm{Zygote}",
    color=:grey,
) => (
    PairPlots.Contourf(
        sigmas=[1,2],
        color=(:grey, 0.2),
        bandwidth=3,
        # bandwidth is the smoothing
    ),
    PairPlots.MarginDensity(
        linewidth=3,
        bandwidth=1,
    ),
)

fig = pairplot(plot_nuts_fd, plot_nuts_zy, plot_mchmc_fd, plot_mchmc_zy,
    #fullgrid=true,
    # Add LaTeX overrides for labels here!
    labels=Dict(
        :ln10As => L"\ln 10 A_s", # LaTeX
        :ns => L"n_s", # LaTeX
        :h => L"h", # LaTeX
        :ωb => L"\omega_b", # LaTeX
        :ωc => L"\omega_c", # LaTeX
        :τ => L"\tau", # LaTeX
        :yₚ => L"A_\mathrm{Planck}", # LaTeX
        :b => Makie.rich("long label α", color=:green)
    ),
    axis=(;
        h = (;
            ticks=(
                [0.65, 0.67, 0.69],
                [L"0.65", L"0.67", L"0.69"]
            ), lims=(;low=0.65-0.67*0.01, high=0.69+0.67*0.01)

        ),
        ns = (;
            ticks=(
                [0.955, 0.965, 0.975],
                [L"0.955", L"0.965", L"0.975"]
            ), lims=(;low=0.955-0.965*0.006, high=0.975+0.965*0.004)
        ),
        ln10As = (;
            ticks=(
                [0.3, 0.305, 0.31],
                [L"3.0", L"3.05", L"3.1"]
            ), lims=(;low=0.2975, high=0.3125)
        ),
        ωb = (;
            ticks=(
                [0.22, 0.225],
                [L"0.022", L"0.0225"]
            ), lims=(;low=0.215, high=0.23)
        ),
        ωc = (;
            ticks=(
                [0.12, 0.124],
                [L"0.12", L"0.124"]
            ), lims=(;low=0.116, high=0.125)
        ),
        τ = (;
            ticks=(
                [0.04, 0.06, 0.08],
                [L"0.04", L"0.06", L"0.08"]
            ), lims=(;low=0.02, high=0.1)
        ),
        yₚ = (;
            ticks=(
                [0.995, 1.005],
                [L"0.995", L"1.005"]
            ), lims=(;low=0.99, high=1.01)
        )

    )
)

rowgap!(fig.layout, 0)
colgap!(fig.layout, 0)
for i in 1:28
    ax  = fig.content[i]
    ax.spinewidth = 2
    ax.xtickwidth = 2
    ax.ytickwidth = 2
    ax.xlabelsize = 28
    ax.ylabelsize = 28
    ax.xticklabelsize = 18
    ax.yticklabelsize = 16
end




leg = fig.content[29]
leg.valign = :top
leg.halign = :center
leg.framevisible = false
leg.labelsize = 18
fig.layout[1,7] = leg
fig
CairoMakie.save("pippo.pdf", fig)
CairoMakie.save("pippo.png", fig)
fig
