#!/usr/bin/env bash

seeds="100"
n="100"
p=$(python -c 'import sys; import numpy as np; n=float(sys.argv[1]); print(round(np.log(n)/n, 3))' $n)
all_graphs=($n\_BA_3 $n\_ER_$p $n\_WS_4_0.1 $n\_kreg_3 $n\_knn_3 $n\_ERA_$p\_0.8 $n\_SBM_3 ENZYMES PROTEINS_full)
r=0.1
snr=0.0

for graph_string in ${all_graphs[@]}; do
    echo $r $graph_string
    python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --snr $snr
    python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --attack add --snr $snr
    python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --attack remove --snr $snr
    python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --attack addremove --snr $snr
    python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --attack rewire --snr $snr
    python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --attack robust --snr $snr
    python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string $graph_string --proportion_perturb $r --attack pgd_$snr --snr $snr
done
