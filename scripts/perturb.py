"""Generate perturbed graphs."""
import argparse
import os

import networkx as nx
from tqdm import tqdm

from src.perturb import randomly_perturb, random_rewire, perturb_robust, PGDAttack
from utils import load_graph, load_clean_signal, load_noisy_signal, get_device, parse_seeds, save_perturbed_graph
from src.gsp import relative_error
from src.filters import LowPassFilter


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=str, default='1',
                    help='Number of experiments (100) or a range (100-200).')
parser.add_argument('--graph_string', type=str, default='100_BA_3',
                    help='String representing the graph')
parser.add_argument('--attack', type=str, default='add', choices=['add', 'remove', 'addremove', 'rewire', 'robust',
                                                                  'pgd'], help='Dataset to use to attack.')
parser.add_argument('--samples', type=int, default=None, help='Sampling for fast robust')
parser.add_argument('--budget', type=float, default=0.05,
                    help='Budget as a proportion of total edges or the absolute budget. '
                         'Rewire is 4 operations (two removes and two adds).')
parser.add_argument('--snr', type=float, default=0.0, help='SNR required (in dB). (Required for pgd)')
parser.add_argument('--gpu', type=int, default=None, help='Which gpu to use if using pgd')
args = parser.parse_args()


target_dir = os.path.join('../data', 'perturbed_graphs', args.graph_string, str(args.budget), args.attack)
os.makedirs(os.path.join(target_dir), exist_ok=True)

for seed in tqdm(parse_seeds(args.seeds), desc=f'{args.budget}_{args.graph_string}_{args.attack}'):
    A = load_graph(args.graph_string, seed)
    G = nx.from_scipy_sparse_matrix(A)

    # calculate size of perturbation
    number_of_edges = G.number_of_edges()
    if args.budget >= 1:
        perturbations = int(args.budget)
    else:
        perturbations = int(args.budget * number_of_edges)

    # perturb graph
    if args.attack == 'remove':
        Gp = randomly_perturb(G, remove=perturbations)
    elif args.attack == 'add':
        Gp = randomly_perturb(G, add=perturbations)
    elif args.attack == 'addremove':
        Gp = randomly_perturb(G, add=int((perturbations+1)/2), remove=int(perturbations/2))
    elif args.attack == 'rewire':
        Gp = random_rewire(G, rewires=int(perturbations/4))
    elif args.attack == 'robust':
        Gp = perturb_robust(G, perturbations, args.samples)
    elif args.attack == 'pgd':
        # read clean and noisy signal
        clean_signal = load_clean_signal(args.graph_string, seed)
        noisy_signal = load_noisy_signal(args.graph_string, args.snr, seed)
        device = get_device(args.gpu)
        model = LowPassFilter()
        attacker = PGDAttack(model, relative_error, perturbations)
        Ap = attacker.pgd_attack(A, clean_signal, noisy_signal, allowed_attempts=50)
        if Ap is None:
            print(f'PGD failed {seed} for graph {args.graph_string} snr {args.snr} & budget {args.budget}')
            continue

    # networkx to scipy
    if args.attack != 'pgd':
        Ap = nx.adj_matrix(Gp)

    # sanity check
    observed_perturbations = ((A-Ap).count_nonzero() / 2)
    if args.attack == 'rewire':  # cant always do an exact number for rewiring
        assert observed_perturbations in [perturbations, perturbations-1, perturbations-2, perturbations-3]
    else:
        assert observed_perturbations == perturbations

    attack = args.attack
    if attack == 'pgd':
        attack = f'pgd_{args.snr}'

    save_perturbed_graph(Ap, args.graph_string, args.budget, attack, seed)
