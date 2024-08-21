"""
This ended up not being used...

"""



import os
os.chdir("..")

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from typing import List, Dict, Tuple, Union 

import numpy as np
import pandas as pd

import pickle
import json

from sklearn.manifold import TSNE

from moleculib.protein.datum import ProteinDatum
from moleculib.graphics.py3Dmol import plot_py3dmol, plot_py3dmol_grid


from helpers.database import populate_representations, whatis
from helpers.edges import connect_edges, CascadingEdges
from helpers.data_processing import LoadData, save_df, save_edges

from helpers.utils import CheckPDBs, aa_map, residue_map



## General useful functions

def load_data(folder, embeddings_file, sliced_proteins_file, tsne_file):

    # Load data from files
    dataloader = LoadData(folder, embeddings_file, sliced_proteins_file, tsne_file)
    dataloader.load_all()

    # Make objects
    if tsne_file is not None:
        reps, _ = populate_representations(dataloader.encoded_dataset, 
                                        dataloader.sliced_dataset, 
                                        dataloader.tsne_dataset)
    else:
        reps, _ = populate_representations(dataloader.encoded_dataset, dataloader.sliced_dataset)
        
    df = reps.to_dataframe()
    print(f"Loaded full dataset: {df.shape}")
    return df


def process_data(df):
    """Perform basic data processing. Drop Nones, and drop amino acid-level
        embeddings...
    """
    df_sample = df.dropna(subset=['datum']).reset_index(drop=True)  # drop nans
    df_sample = df_sample[df_sample['level'] != 0].reset_index(drop=True)  # drop level 0
    print(f"Shape of sample after drops: {df_sample.shape}")
    return df_sample


def make_edges(df, kernel, stride): 
    """Make edges and cascades."""
    # Compute edges
    edges_top_down, edges_bottom_up, n_misses = connect_edges(df, kernel, stride)
    make_cascades = CascadingEdges(edges_bottom_up)
    print(f"Misses: {n_misses}")

    return edges_top_down, edges_bottom_up, make_cascades


def compute_tsne(df: pd.DataFrame):
    """Given a dataframe (corresponding to our database), compute tsne coords and colors."""
    # encoded_dataset_tsne = defaultdict(lambda : defaultdict(lambda : defaultdict(dict)))

    # Get unique levels
    positions_lst = []
    colors_lst = []

    levels = sorted(df['level'].unique())
    for level in levels:
        df_for_level = df[df['level'] == level]
        level_data = np.array(df_for_level['scalar_rep'].values.tolist())

        print(f'computing position tsne for level {level}: {level_data.shape}')
        position = TSNE(n_components=2, perplexity=3, learning_rate='auto', init='random').fit_transform(level_data)
        print(f'computing color tsne for level {level}: {level_data.shape}')
        colors = TSNE(n_components=3, perplexity=3, learning_rate='auto', init='random').fit_transform(level_data)
        colors = (colors - colors.min())
        colors = (colors * 255 / colors.max()).astype(np.int32)
        colors = [f'rgb({r}, {g}, {b})' for r, g, b in colors]

        positions_lst.extend(position.tolist())
        colors_lst.extend(colors)

    return positions_lst, colors_lst




def main():

    FOLDER = "data/denim-energy-1008-embeddings"
    embeddings_file = "encoded_dataset.pkl"
    sliced_proteins_file = "sliced_dataset.pkl"
    # tsne_file = "encoded_dataset_tsne.json"

    df = load_data(FOLDER, embeddings_file=embeddings_file,
                sliced_proteins_file=sliced_proteins_file,
                tsne_file=None)

    df_sample = process_data(df)

    customloader = LoadData(FOLDER="data/custom-embeddings",
                            embeddings_file="encoded_dataset_custom.pkl",
                            sliced_proteins_file="sliced_dataset_custom.pkl")
    customloader.load_all()

    # Make objects
    custom_reps, _ = populate_representations(customloader.encoded_dataset, customloader.sliced_dataset)
    custom_df = custom_reps.to_dataframe()
    print(f"Loaded full dataset: {custom_df.shape}")

    # Process as before
    custom_sample = process_data(custom_df)

    new_df = pd.concat([df_sample, custom_sample], ignore_index=True)

    # Check that the indices are unique and increasing
    # print(new_df.index.is_monotonic_increasing)
    # print(new_df.index.is_unique)
    print(f"Shape of new_df: {new_df.shape}")

    # master_df = new_df.copy()
    master_df = new_df.sample(frac=0.005, random_state=42)

    print(f"Using df of shape: {master_df.shape}")

    ####### IMPORTANT #######

    positions, colors = compute_tsne(master_df)

    master_df['pos'] = positions
    master_df['color'] = colors

    ############################

    # full_positions, full_colors = compute_tsne(new_df)
    _, edges_bottom_up, _ = make_edges(master_df, kernel=5, stride=2)

    save_df(master_df, "df-master")
    save_edges(edges_bottom_up, "edges_bottom_up-master")

if __name__ == "__main__":
    main()
