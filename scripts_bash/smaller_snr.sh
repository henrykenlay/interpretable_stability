seeds=100
r=0.1
snr=-10.0
all_graphs=(100_BA_3 100_ER_0.046 100_WS_4_0.1 100_kreg_3 100_knn_3 100_ERA_0.046_0.8 100_SBM_3 ENZYMES PROTEINS_full)
attacks=(pgd add remove addremove rewire)

# noisy signals
#for graph_string in ${all_graphs[@]}; do
#    python scripts/data/synthetic_noisy_signals.py --seeds $seeds --graph_string $graph_string --snr $snr &
#done
#wait

# perturbations
#for graph_string in ${all_graphs[@]}; do
#    for attack in ${attacks[@]}; do
#        python scripts/perturb.py --seeds $seeds --graph_string $graph_string --budget $r --attack $attack --snr $snr &
#    done
#done
#wait

#attacks=(robust pgd_$snr add remove addremove rewire)

# statistics graphs
#attacks=(add remove addremove rewire robust pgd_$snr)
#for graph_string in ${all_graphs[@]}; do
#    for attack in ${attacks[@]}; do
#        python scripts/statistics/statistics_graphs.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --attack $attack
#    done
#done

# statistics denoising
for graph_string in ${all_graphs[@]}; do
    python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --snr $snr
    for attack in ${attacks[@]}; do
        python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --attack $attack --snr $snr
    done
done
