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
    #label=L"\mathrm{NUTS}\quad\mathrm{ForwardDiff}",
    color=:red,  bottomleft=false, topright=true
) => (
    PairPlots.Contour(
        sigmas=[1,3],
        color=(:red, 1.0),
        bandwidth=2,
        linewidth=3,
        linestyle=:dash
        # bandwidth is the smoothing
    ),
    PairPlots.MarginDensity(
        linewidth=3,
        bandwidth=2,
    ),
)

plot_nuts_zy = PairPlots.Series(
    chains_planck_nuts_zy,
    #label=L"\mathrm{NUTS}\quad\mathrm{Zygote}",
    color=:orange,
) => (
    PairPlots.Contourf(
        sigmas=[1,3],
        color=(:orange, 0.2),
        bandwidth=2,
        # bandwidth is the smoothing
    ),
    PairPlots.MarginDensity(
        linewidth=3,
        bandwidth=2,
    ),
)

plot_mchmc_fd = PairPlots.Series(
    chains_planck_mchmc_fd,
    #label=L"\mathrm{MCHMC}\quad\mathrm{ForwardDiff}",
    color=:green,
) => (
    PairPlots.Contourf(
        sigmas=[1,3],
        color=(:green, 0.2),
        bandwidth=2,
        # bandwidth is the smoothing
    ),
    PairPlots.MarginDensity(
        linewidth=3,
        bandwidth=2,
    ),
)

plot_mchmc_zy = PairPlots.Series(
    chains_planck_mchmc_zy,
    #label=L"\mathrm{MCHMC}\quad\mathrm{Zygote}",
    color=:grey,
) => (
    PairPlots.Contourf(
        sigmas=[1,3],
        color=(:grey, 0.2),
        bandwidth=2,
        # bandwidth is the smoothing
    ),
    PairPlots.MarginDensity(
        linewidth=3,
        bandwidth=2,
    ),
)

chain_array = chains_planck_mchmc_zy.value.data
mean_array = [mean(chain_array[:,i,:]) for i in 1:7]
std_array = [std(chain_array[:,i,:]) for i in 1:7]
sigma_plot = 3.5

fig = pairplot(plot_nuts_fd, plot_nuts_zy, plot_mchmc_fd, plot_mchmc_zy,
    #fullgrid=true,
    # Add LaTeX overrides for labels here!
    labels=Dict(
        :ln10As => L"\log( 10^{10} A_\mathrm{s})", # LaTeX
        :ns => L"n_\mathrm{s}", # LaTeX
        :h => L"h", # LaTeX
        :ωb => L"\omega_\mathrm{b}", # LaTeX
        :ωc => L"\omega_\mathrm{c}", # LaTeX
        :τ => L"\tau", # LaTeX
        :yₚ => L"A_\mathrm{Planck}", # LaTeX)
    ),
    axis=(;
        h = (;
            ticks=(
                [0.66, 0.67, 0.68, 0.69],
                [L"0.66", L"0.67", L"0.68",  L"0.69"]
            ), lims=(;low=mean_array[3]-sigma_plot*std_array[3], high=mean_array[3]+sigma_plot*std_array[3])

        ),
        ns = (;
            ticks=(
                [0.955, 0.965, 0.975],
                [L"0.955", L"0.965", L"0.975"]
            ), lims=(;low=mean_array[2]-sigma_plot*std_array[2], high=mean_array[2]+sigma_plot*std_array[2])
        ),
        ln10As = (;
            ticks=(
                [0.3, 0.305, 0.31],
                [L"3.0", L"3.05", L"3.1"]
            ), lims=(;low=mean_array[1]-sigma_plot*std_array[1], high=mean_array[1]+sigma_plot*std_array[1])
        ),
        ωb = (;
            ticks=(
                [0.22, 0.225],
                [L"0.022", L"0.0225"]
            ), lims=(;low=mean_array[4]-sigma_plot*std_array[4], high=mean_array[4]+sigma_plot*std_array[4])
        ),
        ωc = (;
            ticks=(
                [0.12, 0.124],
                [L"0.12", L"0.124"]
            ), lims=(;low=mean_array[5]-sigma_plot*std_array[5], high=mean_array[5]+sigma_plot*std_array[5])
        ),
        τ = (;
            ticks=(
                [0.04, 0.06, 0.08],
                [L"0.04", L"0.06", L"0.08"]
            ), lims=(;low=mean_array[6]-sigma_plot*std_array[6], high=mean_array[6]+sigma_plot*std_array[6])
        ),
        yₚ = (;
            ticks=(
                [0.995, 1., 1.005],
                [L"0.995", L"1.0", L"1.005"]
            ), lims=(;low=mean_array[7]-sigma_plot*std_array[7], high=mean_array[7]+sigma_plot*std_array[7])
        )

    )
)

rowgap!(fig.layout, 0)
colgap!(fig.layout, 0)

for i in 1:49
    ax_top =
    ax  = fig.content[i]
    ax.spinewidth = 2.5
    ax.xtickwidth = 2.5
    ax.ytickwidth = 2.5
    ax.xlabelsize = 28
    ax.ylabelsize = 28
    ax.xticklabelsize = 18
    ax.yticklabelsize = 16
    ax.xtickalign = 1.0
    ax.ytickalign = 1.0
    ax.xticksvisible = true
    ax.yticksvisible = true
    ax.xticksize = 10
    ax.yticksize = 10
end

elem_1 = [PolyElement(color = :transparent, strokecolor = :red, strokewidth = 3, linestyle=:dash)]

elem_2 = [PolyElement(color = :orange, strokecolor = :transparent, strokewidth = 1)]

elem_3 = [PolyElement(color = :green, strokecolor = :transparent, strokewidth = 1)]

elem_4 = [PolyElement(color = :grey, strokecolor = :transparent, strokewidth = 1)]


Legend(fig[0, 1:7],
[elem_1, elem_2, elem_3, elem_4],
[L"\mathrm{NUTS}\,\,\mathrm{ForwardDiff}", L"\mathrm{NUTS}\,\,\mathrm{Zygote}", L"\mathrm{MCHMC}\,\,\mathrm{ForwardDiff}", L"\mathrm{MCHMC}\,\,\mathrm{Zygote}"],
patchsize = (35, 35), colgap = 30, orientation = :horizontal, labelsize = 25, framevisible = false)
CairoMakie.save("contour.pdf", fig)
CairoMakie.save("contour.png", fig)
fig
