#!/usr/bin/env bash

seeds=100
n=100
r=0.1
er_p=$(python -c 'import sys; import numpy as np; n=float(sys.argv[1]); print(round(np.log(n)/n, 3))' $n)
all_graphs=($n\_BA_3 $n\_ER_$er_p $n\_WS_4_0.1 $n\_kreg_3 $n\_knn_3 $n\_ERA_$er_p\_0.8 $n\_SBM_3 ENZYMES PROTEINS_full)
random_attacks=(add remove addremove rewire)

for graph_string in "${all_graphs[@]}"; do
    echo $graph_string
    for attack in "${random_attacks[@]}"; do
        python scripts/perturb.py --seeds $seeds --graph_string $graph_string --budget $r --attack $attack
    done
done
