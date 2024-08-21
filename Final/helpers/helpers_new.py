"""Helper functions for the protein representation database"""

# Standard
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, field

# Third-party
import numpy as np
import pandas as pd
from einops import rearrange
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
from sklearn.neighbors import radius_neighbors_graph, sort_graph_by_row_values
from sklearn.metrics import pairwise_distances
from Bio import Align
import matplotlib.pyplot as plt

# Local
from moleculib.protein.datum import ProteinDatum
from moleculib.graphics.py3Dmol import plot_py3dmol, plot_py3dmol_grid
from moleculib.protein.alphabet import all_residues

@dataclass
class Representation:
    """Representation object for a protein (sub)-structure. Corresponds to a row in the database."""

    pdb_id: str
    level: int
    level_idx: int
    scalar_rep: np.ndarray
    datum: ProteinDatum
    pos: Optional[np.ndarray] = field(default=None)
    color: Optional[str] = field(default=None)

@dataclass
class RepresentationDatabase:
    """Database of representation objects
    Distinguishes across different hierarchy levels
    """

    data: List[Representation] = None

    def __post_init__(self):
        self.data = [] if self.data is None else self.data

    def add_representation(self, representation: Representation):
        """Add a representation object."""
        self.data.append(representation)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame"""
        return pd.DataFrame([r.__dict__ for r in self.data])


def populate_representations(
    encoded_dataset: Dict[str, Dict[int, np.ndarray]], sliced_dataset,
    tsne_data: Optional[Dict[str, Dict[int, Dict]]] = None
) -> Tuple[RepresentationDatabase, Dict[str, Dict[int, Tuple[int, int]]]]:
    """Populate the representation database from a file."""
    reps = RepresentationDatabase()
    mismatches = defaultdict(dict)

    for pdb_id, levels in encoded_dataset.items():
        for level, embeddings in levels.items():
            protein_data = sliced_dataset[pdb_id][level]
            if tsne_data:
                coords = tsne_data[pdb_id][str(level)]['pos']
                colors = tsne_data[pdb_id][str(level)]['colors']
            n_embeddings, n_protein_data = len(embeddings), len(protein_data)

            if n_embeddings != n_protein_data:
                difference = n_embeddings - n_protein_data
                mismatches[pdb_id][level] = (n_embeddings, n_protein_data)
                protein_data.extend([None] * difference)

            for level_idx, scalar_rep in enumerate(embeddings):
                protein_datum = protein_data[level_idx]
                if tsne_data:
                    reps.add_representation(
                        Representation(pdb_id, level, level_idx, scalar_rep, protein_datum,
                                       coords[level_idx], colors[level_idx])
                    )
                else:
                    reps.add_representation(
                        Representation(pdb_id, level, level_idx, scalar_rep, protein_datum)
                    )

    return reps, mismatches


def get_column(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Fetch a column from the dataframe based on the specified conditions."""
    conditions = [
        df[col].isin([value]) if isinstance(value, list) else df[col] == value
        for col, value in kwargs.items()
        if value is not None
    ]
    condition = conditions[0] if conditions else True
    for cond in conditions[1:]:
        condition &= cond
    return df.loc[condition]


def get_scalars(df: pd.DataFrame, **kwargs):
    """Get the scalar representations from the database."""
    return get_column(df, **kwargs, column="scalar_rep")


def whatis(*objects):
    """Function to print the type and brief content of an object, or a list of objects."""
    for idx, obj in enumerate(objects):
        obj_type = type(obj).__name__
        obj_repr = repr(obj)  # Get a string representation of the object

        # Limit the length of the object representation for display
        max_length = 50
        display_repr = (
            (obj_repr[:max_length] + "...") if len(obj_repr) > max_length else obj_repr
        )

        if isinstance(obj, dict):
            print(
                f"Object {idx}: ({display_repr}) is a dictionary with length {len(obj)}"
            )
            continue

        shape_attr = getattr(obj, "shape", None)
        if shape_attr is not None:
            print(
                f"Object {idx}: ({display_repr}) is of type {obj_type} and has shape {obj.shape}"
            )
            continue

        # Handle list-like objects without a shape attribute by converting to NumPy array
        try:
            obj_as_array = np.array(obj)
            shape_description = f"in array form has shape {obj_as_array.shape}, dtype {obj_as_array.dtype}"
        except ValueError:
            shape_description = (
                "has inhomogeneous types or elements unsuitable for an array"
            )

        print(
            f"Object {idx}: ({display_repr}) is of type {obj_type} and {shape_description}"
        )


########### Edge Processing ###########

def connect_edges(df, kernel_size, stride):
    """Connect edges between nodes in the hierarchy based on the kernel size and stride."""
    n_misses = 0
    edges_top_down, edges_bottom_up = dict(), dict()
    grouped_by_pdb = df.groupby('pdb_id')

    # For each PDB...
    for pdb_id, pdb_group in grouped_by_pdb:
        unique_levels = sorted(pdb_group['level'].unique())

        # For each hierarchy level in the autoencoder...
        for level in unique_levels:
            lower_level, upper_level = level, level + 1
            lower_level_group = pdb_group[pdb_group['level'] == lower_level].sort_values(by='level_idx')
            upper_level_group = pdb_group[pdb_group['level'] == upper_level].sort_values(by='level_idx')
            num_lower_level = len(lower_level_group)
            for start in range(0, num_lower_level, stride):
                end = start + kernel_size
                lower_level_slice = lower_level_group.iloc[start:end]
                upper_level_node_index = start // stride
                if upper_level_node_index < len(upper_level_group):
                    upper_level_node = upper_level_group.iloc[upper_level_node_index]

                    # Key is index of upper node, value is list of indices for all lower nodes
                    edges_top_down[upper_level_node.name] = list(lower_level_slice.index)
                    edges_bottom_up.update(dict.fromkeys(lower_level_slice.index, upper_level_node.name))
                else:
                    n_misses += 1

    return edges_top_down, edges_bottom_up, n_misses

class CascadingEdges:
    """Initialize the CascadingEdges with a mapping from child to parent.

        Args:
        edges_bottom_up (Dict[int, int]): Dictionary mapping from child index to parent index.
    """
    def __init__(self, edges_bottom_up: Dict[int, int]):
        self.edges_bottom_up = edges_bottom_up

    def __call__(self, start_index: int, n_cascades: int = None, verbose=True) -> List[int]:
        """Cascades the edges to the top level by following parent links.

        Args:
        start_index (int): The starting primary key from which to begin cascading upward.
        n_cascades (int, optional): The number of cascades (levels to traverse upwards). 
        If None, continues until a top is reached.

        Returns:
        List[int]: List of primary keys traversed, up to the top or for `n_cascades` steps.
        """
        current_index = start_index
        cascades = [current_index]

        try:
            if n_cascades is not None:
                for _ in range(n_cascades):
                    current_index = self.edges_bottom_up[start_index]
                    cascades.append(current_index)
            else:
                while True:
                    current_index = self.edges_bottom_up[current_index]
                    cascades.append(current_index)
        except KeyError:
            if verbose:
                print(f"Stopped cascading at {current_index}: no further parent found.")

        return cascades


########### Distance Metrics (Vector Based) ###########

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
        distance = cosine(scalar_rep1, scalar_rep2)
        distances.append(distance)

    return distances


class ComputeDistanceMatrix:
    """Computes a Distance Matrix, given a dataframe database and a level of hierarchy.

        Returns in csr matrix format.
    """
    def __init__(self, df: pd.DataFrame):
        """Initialize the dataframe"""
        self.df = df
        self.df = self.df[self.df['datum'].notna()]

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
        distances = pairwise_distances(scalars, metric='cosine')
        if return_df:
            return csr_matrix(distances), level_df
        return csr_matrix(distances)


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
        graph = radius_neighbors_graph(scalar_reps, radius, mode='distance', metric='cosine')
        if sort:
            graph = sort_graph_by_row_values(graph, warn_when_not_sorted=False)
        return graph, sub_df

def sort_distance_graph(distance_matrix: csr_matrix, start=0, end=None):
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



########### Distance Metrics (Datum Based) ###########

class DistanceMapMetric:
    """Get structure-based distance score"""
    def __call__(self, datum1, datum2):
        backbone1 = datum1.atom_coord[..., 1, :]
        backbone2 = datum2.atom_coord[..., 1, :]

        mask1 = datum1.atom_mask[..., 1]
        mask2 = datum2.atom_mask[..., 1]

        if (mask1 != mask2).any():
            print('[WARNING!] Masks are mismatching')

        mask = mask1 & mask2

        def vector_map(x):
            return rearrange(x, 'i c -> i () c') - rearrange(x, 'j c -> () j c')
        def distance_map(x):
            return np.linalg.norm(x, axis=-1)
        cross_mask = rearrange(mask, 'i -> i ()') * rearrange(mask, 'j -> () j')
        loss = distance_map(vector_map(backbone1)) - distance_map(vector_map(backbone2))
        loss = loss ** 2
        loss = loss * cross_mask
        loss = loss.sum() / cross_mask.sum()
        return loss

class DistanceSeqMetric:
    """Get an alignment score, as well as hamming distance."""
    def __call__(self, datum1, datum2):
        seq1 = np.array(datum1.residue_token, np.int32)
        seq2 = np.array(datum2.residue_token, np.int32)

        # Do a Hamming distance
        hamming_distance = sum(seq1 != seq2)

        # Alignment
        aligner = Align.PairwiseAligner()
        alignments = aligner.align(seq1, seq2)
        best_alignment = alignments[0]
        alignment_score = best_alignment.score
        # print("Best alignment score:", alignment_score)

        return alignment_score, hamming_distance

########### Visualization ###########

# Plot protein datums via the primary key in the DataFrame

class PlotProteinDatum:
    """Given a list of indices from a DataFrame of protein representations, 
        plot the protein datum objects.
    """
    def __init__(self, df):
        self.df = df

    def __call__(self,
                 index1: Union[int, List[int]],
                 index2: Optional[Union[int, List[int]]] = None):
        """Plot the protein datum given a dataframe index or list of indices. Optionally plot a second datum."""
        if isinstance(index1, int):
            index1 = [index1]
        if index2 is not None:
            if isinstance(index2, int):
                index2 = [index2]
            if len(index1) != len(index2):
                raise ValueError("Both indices must be same length.")
            datum2 = self.df.iloc[index2]['datum'].values
        datum1 = self.df.iloc[index1]['datum'].values
        if index2 is not None:
            protein_plot = plot_py3dmol_grid([datum1, datum2])
        else:
            protein_plot = plot_py3dmol_grid([datum1])

        return protein_plot

def plot_similarity_histograms(lvl_data_list):
    """Given a list of `data` numpy arrays, plot a distribution
        of the values.
    """
    plt.figure(figsize=(15, 10))
    
    for i, lvl_data in enumerate(lvl_data_list, start=1):
        plt.subplot(3, 2, i)
        plt.hist(lvl_data, bins=30, alpha=0.75)
        plt.title(f"Histogram of Lvl {i-1} Similarity Distances")
        plt.xlabel("Similarity Distance")
        plt.ylabel("Count")
    
    plt.tight_layout()
    plt.show()

########### Pair Comparison ###########


class Comparison:
    """Compare a pair of lists of hierarchial (cascading) indices in the graph."""
    def __init__(self, df, us: List[int], vs: List[int], drop_na=True):
        if drop_na:
            self.df = df[df['datum'].notna()]
        else:
            self.df = df
        self.us = us
        self.vs = vs

        # Return attributes
        self.scores = dict(
            vector=list(),
            structure=list(),
            sequence=list()
        )

        # Data attributes
        self.u_datums: List[ProteinDatum] = []
        self.v_datums: List[ProteinDatum] = []
        self.u_seqs: List[str] = []
        self.v_seqs: List[str] = []
        for u, v in zip(us, vs):
            u_datum = self.df.loc[u, 'datum']
            v_datum = self.df.loc[v, 'datum']
            self.u_datums.append(u_datum)
            self.v_datums.append(v_datum)
            self.u_seqs.append(self._datum_to_sequence(u_datum))
            self.v_seqs.append(self._datum_to_sequence(v_datum))

        self.struct_metric = DistanceMapMetric()
        self.seq_metric = DistanceSeqMetric()

    def cascade_scores(self, return_scores=False):
        """Compute the scores for the cascades."""
        
        for i, (u, v) in enumerate(zip(self.us, self.vs)):
            datum1, datum2 = self.u_datums[i], self.v_datums[i]
            print(u, v)
            # Vector score (cosine distance)
            vec1 = self.df.loc[u, 'scalar_rep']
            vec2 = self.df.loc[v, 'scalar_rep']
            print(f"Shape of vec1: {vec1.shape}, vec2: {vec2.shape}")
            struct_map = self.struct_metric(datum1, datum2)
            seq_map = self.seq_metric(datum1, datum2)

            # Append scores
            self.scores['vector'].append(cosine(vec1, vec2))
            self.scores['structure'].append(struct_map)
            self.scores['sequence'].append(seq_map)  # (alignment, hamming distance)

        if return_scores:
            return self.scores['vector'][-1], struct_map, seq_map

    def _datum_to_sequence(self, datum):
        return [all_residues[token] for token in datum.residue_token]


def longest_common_subsequence_indices(seq, subseq):
    """Find the longest common subsequence (LCS) of two sequences and return.
        Standard dynamic programming, which we use here for amino acid
        subsequences. The function can take in residue tokens as well as strings.

        Returns the indices of the `seq` object that match the `subseq` object.
    """
    len1, len2 = len(seq), len(subseq)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)

    # Fill the dp array
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq[i - 1] == subseq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Find the LCS
    i, j = len1, len2
    lcs_indices = []

    while i > 0 and j > 0:
        if seq[i - 1] == subseq[j - 1]:
            lcs_indices.append(i - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_indices.reverse()
    return lcs_indices


def lcs_via_alignment(seq1, seq2):
    """Alternative way to get the longest common subsequence using alignment."""
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    seq1 = np.array(seq1, np.int32)
    seq2 = np.array(seq2, np.int32)

    alignments = aligner.align(seq1, seq2)

    best_alignment = alignments[0]
    aligned_seq1, _ = best_alignment.aligned

    # Extract the indices of the matching subsequence in seq1
    indices = []
    for block in aligned_seq1:
        indices.extend(range(block[0], block[1]))

    return indices



##### CASCADES #####

@dataclass
class Cascade:
    """A single cascade object stores information about a single protein
        cascade. A protein cascade is defined as the bottom-up hierarchical
        relationship of on protein representation and its parents.

        Handles: sequences, lengths, hierarchy levels, level idx in hierarchy,
            and indices for slicing parent compositions.

            Allows for displaying the relationship as well.
    """
    pdb_id: str = ""
    indices: List[int] = None
    residues: List[str] = None # these are the residue tokens
    sequences: List[str] = None
    lengths: List[int] = None
    levels: List[int] = None
    level_idxs: List[int] = None
    datums: List[ProteinDatum] = None
    cascade_df: pd.DataFrame = None

    def __len__(self):
        return len(self.indices)

    # Now pretty print the cascade
    def __str__(self):
        cascade_info = f"Cascade for {self.pdb_id} with {len(self)} levels.\n"
        cascade_info += f"Indices: {self.indices}\n"
        for level, level_idx, sequence in zip(self.levels, self.level_idxs, self.sequences):
            cascade_info += f"Sequence for level {level} at index {level_idx} (of length {len(sequence)}): {sequence}\n"
        return cascade_info
    
    def show_df(self):
        """Display the DataFrame filtered by the indices."""
        display(self.cascade_df)

    def plot(self):
        """Plot the cascade."""
        return plot_cascade(self)


def plot_cascade(cascade: Cascade):
    """On input a cascade object, plot the cascade with the indices highlighted.
        Assume that the first object in the cascade is the child, and the rest are parents.
        (So we color from index [1:])
    """
    view = plot_py3dmol_grid([cascade.datums])
    child_sequence = cascade.sequences[0]
    for i, sequence in enumerate(cascade.sequences[1:]):
        local_viewer = (0, i+1)
        local_indices = longest_common_subsequence_indices(seq=sequence, subseq=child_sequence)
        view.addStyle({'model': -1}, {"cartoon": {'color': 'white'}}, viewer=local_viewer)
        view.addStyle({'model': -1, 'resi': local_indices}, {"cartoon": {'color': 'spectrum'}}, viewer=local_viewer)
    return view
