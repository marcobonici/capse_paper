function transpose_samples_single_chain_HMC(sample)
    save_sample = zeros(length(sample[:,1]), 7)
    for i in 1:7
        save_sample[:,i] = sample[:,i]
    end
    return save_sample
end

function combine_chains(chn)
    n_chains = length(chn.value.data[1,1,:])
    final_chain = transpose_samples_single_chain_HMC(chn.value.data[:,:,1])
    for i in 2:n_chains
        final_chain = vcat(final_chain, transpose_samples_single_chain_HMC(chn.value.data[:,:,i]))
    end
    return final_chain
end

function extract_single(x, i::Int, nchains)
    planck_mchmc_ln10As = vec(vcat([Array(x[j][:,i]) for j in 1:nchains]))
    r=size(planck_mchmc_ln10As)[1]
    c=size(planck_mchmc_ln10As[1])[1]
    result = reshape(reduce(vcat,planck_mchmc_ln10As),(r*c))
    return result
end
