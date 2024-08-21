"""compute_distance_matrices.py

Given a dataframe with different levels of hierarchy, compute the distance matrices for each level of hierarchy.
"""

import os
import pandas as pd
import scipy.sparse


from helpers.statistics import ComputeDistanceMatrix, Distributions


def main():
    """Do Everything."""
    
    # Import data
    path_to_data = "data/final/"
    df = pd.read_pickle(path_to_data + "master_dataframe.pkl")


    # Sample the DataFrame
    sampled_df = df.sample(frac=0.01)
    print(f"Sampled {len(sampled_df)} rows from the DataFrame.")

    # Instantiate Distributions object
    distributions = Distributions(sampled_df)

    # Compute distance matrices for all levels of hierarchy
    distributions()


    # Save all matrices to disk
    path = "data/final/"
    for level, matrix in distributions.distance_graphs.items():
        scipy.sparse.save_npz(path + f"level_{level}_distance_matrix.npz", matrix)
        print(f"Saved level {level} distance matrix to '{path}level_{level}_distance_matrix.npz'")


if __name__ == "__main__":
    main()
