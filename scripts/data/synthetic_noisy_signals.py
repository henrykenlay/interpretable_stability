"""Generate noisy signals"""
import argparse

from tqdm import tqdm

from src.gsp import addnoise
from scripts.utils import load_clean_signal, save_noisy_signal, parse_seeds

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=str, default='5',
                    help='Number of experiments (100) or a range (100-200).')
parser.add_argument('--graph_string', type=str, default='500_BA_2',
                    help='String representing the graph')
parser.add_argument('--snr', type=float, default=0.0,
                    help='SNR required (in dB).')
args = parser.parse_args()


def dB2ratio(snr_dB):
    """Convert a snr in dB to an untransformed ratio."""
    return 10**(snr_dB/10)


# directory for noisy signals
for seed in tqdm(parse_seeds(args.seeds), desc=f'{args.graph_string} ({args.snr}dB)'):
    clean_signal = load_clean_signal(args.graph_string, seed)
    snr = dB2ratio(args.snr)
    noisy_signal = addnoise(clean_signal, snr)
    save_noisy_signal(noisy_signal, args.snr, args.graph_string, seed)
