"""
statistics.py

Contains functions for calculating statistics of the graph and the embeddings.
This mainly includes clustering algorithm support via KMeans and DBSCAN.
"""

import numpy as np
import pandas as pd
import sklearn.neighbors as skn
# import sklearn.cluster as skc
import sklearn.metrics as skm
import scipy.spatial.distance as ssd
import scipy.sparse as sp


def calculate_cosine_distances(df, cascades1, cascades2):
    """Calculate cosine distances between pairs of cascades based on their scalar representations.

    Args:
    df (pd.DataFrame): DataFrame containing the scalar representations.
    cascades1 (List[int]): List of indices for the first set of cascades.
    cascades2 (List[int]): List of indices for the second set of cascades.

    Returns:
    List[float]: List of cosine distances between the corresponding cascades.
    """
    if len(cascades1) != len(cascades2):
        raise ValueError("Both cascades lists must be of the same length.")

    distances = []
    for idx1, idx2 in zip(cascades1, cascades2):
        scalar_rep1 = df.loc[idx1, 'scalar_rep']
        scalar_rep2 = df.loc[idx2, 'scalar_rep']
        distance = ssd.cosine(scalar_rep1, scalar_rep2)
        distances.append(distance)

    return distances


class ComputeDistanceMatrix:
    """Computes a Distance Matrix, given a dataframe database and a level of hierarchy.

        Returns in csr matrix format.
    """
    def __init__(self, df: pd.DataFrame):
        """Initialize the dataframe"""
        self.df = df

    def __call__(self, level: int, return_df=False):
        """Compute the distance matrix for a given hierarchy level. Note if the
            given level has already been computed do not recompute: just return it.

            Return both the distance matrix and the DataFrame at that level.
        """
        level_df = self.df[(self.df['level'] == level)].reset_index(drop=False)
        if level_df.empty:
            raise ValueError(f"No data found for level {level}")
        scalars = np.stack(level_df['scalar_rep'].values)
        print(f"Computing at level {level} with scalars shape: {scalars.shape}")
        distances = skm.pairwise_distances(scalars, metric='cosine')
        if return_df:
            return sp.csr_matrix(distances), level_df
        return sp.csr_matrix(distances)

class Distributions:
    """Compute the distribution of distances for a given level of hierarchy"""
    def __init__(self, df: pd.DataFrame):
        """Initialize the dataframe"""
        self.df = df
        self.compute_distance_matrix = ComputeDistanceMatrix(df)

        self.distance_graphs = dict()

    def __call__(self):
        """Get distributions on all levels of hierarchy."""
        for level in self.df['level'].unique():
            self.distance_graphs[level] = self.compute_distance_matrix(level)

    @property
    def get_graph(self, level: int):
        """Return the graph for a given level of hierarchy"""
        return self.distance_graphs[level]






def get_column(df, pk=None, pdb_id=None, level=None, level_idx=None, column=None):
    """Get a column in a DataFrame given certain conditions.
        If no column is specified, return a view of the indexed
        DataFrame.
    """

    # Build out the indexing condition manually
    condition = True
    if pk is not None:
        if not isinstance(pk, list):
            pk = [pk]
        condition = (condition & (df['pk'].isin(pk)))
    if pdb_id is not None:
        if not isinstance(pdb_id, list):
            pdb_id = [pdb_id]
        condition = (condition & (df['PDBid'].isin(pdb_id)))
    if level is not None:
        if not isinstance(level, list):
            level = [level]
        condition = (condition & (df['level'].isin(level)))
    if level_idx is not None:
        if not isinstance(level_idx, list):
            level_idx = [level_idx]
        condition = (condition & (df['level_idx'].isin(level_idx)))

    if condition is True:
        condition = df.index  # return all rows
    if column is None:
        return df.loc[condition]
    return df.loc[condition, column].values



class RadiusNeighbors:
    """Class to build a similarity graph from DataFrame.

        Returns in csr matrix format.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Get the df where datum is not None
        self.df = self.df[self.df['datum'].notna()]

    def _get_scalars(self, **kwargs):
        """Return the scalar representations given a selection"""
        # Shape here is (N,), but return shape (N, M)
        sub_df = get_column(self.df, **kwargs)
        scalar_representations = sub_df['scalar_rep']
        return np.stack(scalar_representations), sub_df.reset_index(drop=False)

    def get_radius_neighbors(self, radius, sort=False, **kwargs):
        """Return the indices of the scalar representations that are within a certain radius
            of each other.
        """

        scalar_reps, sub_df = self._get_scalars(**kwargs)
        # print(f"Processing {len(scalar_reps)} scalar representations")
        print(f"Shape of scalar reps: {scalar_reps.shape}")
        graph = skn.radius_neighbors_graph(scalar_reps, radius, mode='distance', metric='cosine')
        if sort:
            graph = skn.sort_graph_by_row_values(graph, warn_when_not_sorted=False)
        return graph, sub_df

def sort_distance_graph(distance_matrix: sp.csr_matrix, start=0, end=None) -> list:
    """Takes a csr matrix and return a list of distance, (row, col) tuples
        in sorted order (smallest distance to largest)
        #ThankYouChatGPT4
    """
    # Extract the non-zero indices and data from the CSR matrix
    row_indices, col_indices = distance_matrix.nonzero()
    data = distance_matrix.data

    # Ensure each pair is unique by making row always the lesser index
    # This step assumes an undirected graph (symmetric distances)
    pairs = np.vstack([row_indices, col_indices]).T
    # Order pairs such that the first element is always less than the second
    ordered_pairs = np.sort(pairs, axis=1)
    # Remove duplicates and sort by distance
    unique_pairs, unique_indices = np.unique(ordered_pairs, axis=0, return_index=True)
    del unique_pairs
    if end is not None:
        sorted_indices = unique_indices[np.argsort(data[unique_indices])[start:end]]
    else:
        sorted_indices = unique_indices[np.argsort(data[unique_indices])][start:]

    # Create a list of tuples (distance, (row, col))
    sorted_distances_with_indices = [(data[idx], (row_indices[idx], col_indices[idx]))
                                     for idx in sorted_indices]

    return sorted_distances_with_indices


