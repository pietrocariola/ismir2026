import argparse
import transformations as tf
import os    
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE

N_FILES = 10
PERP = 25

def main():
    parser = argparse.ArgumentParser(description="Generate Projs CLI")
    parser.add_argument("--embeds_path", type=str, help="Path to the embeddings")
    parser.add_argument("--datasets", type=str, nargs="+", help="Projections will be made for the specified datasets")
    parser.add_argument("--models", type=str, nargs="+", help="Projections will be made for the specified models")
    parser.add_argument("--transformations", type=str, nargs="+", help="Transformations to the sounds: pitch, time stretching, etc...")
    parser.add_argument("--output_path", type=str, help="Full path where the embeddings will be saved on")
    args = parser.parse_args()
    print(f"datasets: {args.datasets}")
    print(f"models: {args.models}")
    print(f"transformations: {args.transformations}")

    embeds_path = args.embeds_path
    datasets = args.datasets.lower()
    models = args.models.lower()
    transformations = args.transformations.lower()
    out_path = args.output_path

    tsne = TSNE(n_components=2, perplexity=PERP)

    df = pd.read

    if N_FILES > 0:
        files = files[:N_FILES]


if __name__ == "__main__":
    main()