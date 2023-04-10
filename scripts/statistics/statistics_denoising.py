"""Denoising example."""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm

from src.filters import LowPassFilter
from src.gsp import relative_error, signal_to_noise
from scripts.utils import parse_seeds
from scripts.utils import (load_clean_signal, load_graph,
                   load_noisy_signal, load_perturbed_graph)

warnings.filterwarnings('ignore', category=sp.SparseEfficiencyWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=str, default='1',
                    help='Number of experiments (100) or a range (100-200).')
parser.add_argument('--graph_string', type=str, default='100_BA_3',
                    help='String representing the graph')
parser.add_argument('--attack', type=str, default=None,
                    help='Type of attack. No attack if None')
parser.add_argument('--proportion_perturb', type=float, default=0.1,
                    help='Budget as a proportion of total edges.')
parser.add_argument('--snr', type=float, default=0.0,
                    help='SNR required (in dB).')
args = parser.parse_args()

denoising_statistics_dir = os.path.join('data', 'denoising_statistics', args.graph_string)
os.makedirs(denoising_statistics_dir, exist_ok=True)

for seed in tqdm(parse_seeds(args.seeds)):
    data = []

    # load graphs
    A = load_graph(args.graph_string, seed)
    if args.attack is None:
        Ap = A.copy()
    else:
        try:
            Ap = load_perturbed_graph(args.graph_string, args.proportion_perturb, args.attack, seed)
        except:  # happens for some pgd attacks that couldnt find a solution
            print(f'Failed seed {seed} for graph {args.graph_string} attack {args.attack} and budget {args.proportion_perturb}')
            continue

    # signals
    clean_signal = load_clean_signal(args.graph_string, seed)
    noisy_signal = load_noisy_signal(args.graph_string, args.snr, seed)
    noisy_signal_torch = torch.FloatTensor(noisy_signal).unsqueeze(1)

    # filter model
    graph_filter = LowPassFilter()

    # filter signal
    A = torch.FloatTensor(A.todense())
    filtered_signal = graph_filter(A, noisy_signal_torch).numpy()
    H = np.linalg.inv(graph_filter.H_inv.numpy())

    # perturbed filter signal
    Ap = torch.FloatTensor(Ap.todense())
    filtered_signal_p = graph_filter(Ap, noisy_signal_torch).numpy()
    Hp = np.linalg.inv(graph_filter.H_inv.numpy())

    # filtering statistics
    data.append(['snr_in', signal_to_noise(clean_signal, noisy_signal)])
    data.append(['snr_out_unperturbed', signal_to_noise(clean_signal, filtered_signal)])
    data.append(['snr_out_perturbed', signal_to_noise(clean_signal, filtered_signal_p)])
    data.append(['relative_recovery_error_unperturbed', relative_error(clean_signal, filtered_signal)])
    data.append(['relative_recovery_error_perturbed', relative_error(clean_signal, filtered_signal_p)])
    data.append(['filter_distance', np.linalg.norm(H-Hp, ord=2)])
    num = np.linalg.norm(filtered_signal-filtered_signal_p)
    den = np.linalg.norm(noisy_signal)
    data.append(['relative_output_distance', num/den])

    # save results
    df = pd.DataFrame(data, columns=['metric', 'value'])
    df.to_csv(os.path.join(denoising_statistics_dir, f'{args.proportion_perturb}_{args.snr}_{args.attack}_{seed}.csv'))
