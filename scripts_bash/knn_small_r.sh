#!/usr/bin/env bash
n=100
seeds=100
snr=0.0
r=0.02

echo "Perturbing the graphs"
attacks=(add remove addremove rewire pgd robust)
for attack in "${attacks[@]}"; do
  python scripts/perturb.py --seeds $seeds --graph_string 100_knn_3 --budget $r --attack $attack
done

echo "Graph statistics"
attacks=(add remove addremove rewire pgd_$snr robust)
for attack in "${attacks[@]}"; do
  python scripts/statistics/statistics_graphs.py --seeds $seeds --graph_string 100_knn_3 --proportion_perturb $r --attack $attack
done

echo "Denoising statistics"
for attack in "${attacks[@]}"; do
  python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string 100_knn_3 --proportion_perturb $r --snr $snr --attack $attack
done
