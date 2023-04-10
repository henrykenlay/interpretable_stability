import argparse
from tqdm import tqdm
from src.graphs.sbm import equal_community_sizes, sample_sbm_graph, sample_sbm_signal
from scripts.utils import generate_graph_string, save_graph, save_clean_signal, parse_seeds

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=str, default='100', help='Number of experiments')
parser.add_argument('--n', type=int, help='Size of graphs')
parser.add_argument('--number_communities', type=int, help='Size of graphs')
parser.add_argument('--mu', type=int, default=2, help='Separation of communities')
parser.add_argument('--std', type=int, default=1, help='Noise standard deviation.')
args = parser.parse_args()


# use equally spaced communities for sbm graph
community_sizes = equal_community_sizes(args.n, args.number_communities)
for seed in tqdm(parse_seeds(args.seeds)):

    # generate data
    graph = sample_sbm_graph(community_sizes)
    signal = sample_sbm_signal(community_sizes, args.mu, args.std)
    assert graph.shape[0] == signal.shape[0]

    # save data
    graph_string = generate_graph_string('SBM', n=args.n, number_communities=args.number_communities)
    save_graph(graph, graph_string, seed)
    save_clean_signal(signal, graph_string, seed)
