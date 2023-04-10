#!/usr/bin/env bash
n=50
seeds=1
snr=0.0
r=0.1

# synthetic data (not including SBM)
er_p=$(python -c 'import sys; import numpy as np; n=float(sys.argv[1]); print(round(np.log(n)/n, 3))' $n)
#synthetic_graphs=($n\_BA_2 $n\_ER_$er_p $n\_WS_4_0.1 $n\_kreg_3 $n\_knn_3 $n\_ERA_$er_p\_0.8)
synthetic_graphs=($n\_BA_2)

echo ""
echo "Generating synthetic graphs"
python scripts/data/synthetic_graphs.py --seeds $seeds --graph_model BA --n $n --m 2
#python scripts/data/synthetic_graphs.py --seeds $seeds --graph_model ER --n $n --p $er_p
#python scripts/data/synthetic_graphs.py --seeds $seeds --graph_model WS --n $n --k 4 --p 0.1
#python scripts/data/synthetic_graphs.py --seeds $seeds --graph_model Kreg --n $n --k 3
#python scripts/data/synthetic_graphs.py --seeds $seeds --graph_model KNN --n $n --k 3
#python scripts/data/synthetic_graphs.py --seeds $seeds --graph_model ERA --n $n --p $er_p --cutoff 0.8 --max_iterations 50000
#
#echo ""
#echo "Generating synthetic signals"
for graph_string in ${synthetic_graphs[@]}; do
    python scripts/data/synthetic_clean_signals.py --seeds $seeds --graph_string $graph_string
    python scripts/data/synthetic_noisy_signals.py --seeds $seeds --graph_string $graph_string --snr $snr
done

# SBM data
#comm=3
#echo ""
#echo "Generating SBM graphs and signals"
#python scripts/data/sbm_graphs_and_signals.py --seeds $seeds --n $n --number_communities $comm
#python scripts/data/synthetic_noisy_signals.py --seeds $seeds --graph_string $n\_SBM\_$comm --snr $snr

# perturbations
#all_graphs=($n\_BA_2 $n\_ER_$er_p $n\_WS_4_0.1 $n\_kreg_3 $n\_knn_3 $n\_ERA_$er_p\_0.8 $n_SBM_3)
all_graphs=($n\_BA_2)
attacks=(add remove addremove rewire pgd robust)

for graph_string in "${all_graphs[@]}"; do
    echo $graph_string
    for attack in "${attacks[@]}"; do
        python scripts/perturb.py --seeds $seeds --graph_string $graph_string --budget $r --attack $attack
    done
done


# statistics
attacks=(add remove addremove rewire pgd_$snr robust)
for graph_string in "${all_graphs[@]}"; do
    echo $graph_string
    for attack in "${attacks[@]}"; do
        python scripts/statistics/statistics_graphs.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --attack $attack
        python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --attack $attack --snr $snr
    done
done
