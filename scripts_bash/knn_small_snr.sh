#!/usr/bin/env bash
n=100
seeds=100
snr=-10.0
r=0.1

echo "Generating noisy signals"
python scripts/data/synthetic_noisy_signals.py --seeds $seeds --graph_string 100_knn_3 --snr $snr

echo "Perturbing the graphs"
python scripts/perturb.py --seeds $seeds --graph_string 100_knn_3 --budget $r --attack pgd --snr $snr

echo "Denoising statistics"
for attack in "${attacks[@]}"; do
  python scripts/statistics/statistics_denoising.py --seeds $seeds --graph_string 100_knn_3 --proportion_perturb $r --snr $snr --attack $attack
done
