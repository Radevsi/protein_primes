"""candidates.py"""

from typing import List
from dataclasses import dataclass
import numpy as np
import scipy.spatial.distance as ssd

from helpers.neighborhood import NeighborMetrics, MakeNeighborMetrics
from helpers.edges import CascadingEdges

# from helpers.cascades import


@dataclass
class DivergenceMetrics:
    """Store the metrics for determining divergence of vectors."""

    avg_distance: float
    max_distance: float
    pairwise_distances: np.array = None
    # (TODO): add more if needed


def vector_divergence(vectors: List[np.array]):
    """Calculate the divergence (arbitrarily defined) of a list of vectors."""

    distance_matrix = ssd.cdist(vectors, vectors, metric="cosine")
    nonzero_ids = distance_matrix.nonzero()
    distances = np.array(distance_matrix[nonzero_ids]).reshape(
        -1,
    )  # make one-dimensional
    avg_distance = np.mean(distances)
    max_distance = np.max(distances)
    # print(distances, avg_distance, max_distance)
    return DivergenceMetrics(avg_distance, max_distance)


@dataclass
class Candidate:
    """A Candidate is a query index in a dataframe, with its neighbors and metrics."""

    query_index: int
    neighbor_indices: List[int]
    neighbor_metrics: NeighborMetrics
    divergence_metrics: DivergenceMetrics

    def __str__(self):
        return f"""\
Candidate {self.query_index} with PDB id {self.query_pdb_id} \
has {len(self.neighbor_indices)} neighbors.\n\
# Sequence: {self.neighbor_metrics.query.sequences[0]}\n\
Neighbor PDB IDs: {self.neighbor_pdb_ids}\n\
Average Divergence: {self.divergence_metrics.avg_distance:.7f}
"""

    @property
    def query_pdb_id(self):
        """Property to get the PDB ID of the query index."""
        return self.neighbor_metrics.query.pdb_id

    @property
    def neighbor_pdb_ids(self):
        """Property to get the PDB IDs of the neighbor indices."""
        return [neighbor.pdb_id for neighbor in self.neighbor_metrics.neighbors]

    @property
    def query_sequence(self):
        """Property to get the sequence of the query index.
            If three_letter is True, return the three-letter amino acid sequence.
        """
        return self.neighbor_metrics.query.sequences_short[0]

    @property
    def neighbor_sequence(self):
        """Property to get the short sequences of the neighbor indices.
            If three_letter is True, return the three-letter amino acid sequence.
        """
        return [neighbor.sequences_short[0] for neighbor in self.neighbor_metrics.neighbors]


    def eval(self, divergence_threshold=None):
        """Evaluate the candidate for divergence."""
        return self.divergence_metrics.avg_distance > divergence_threshold


class MakeCandidate:
    """Make a candidate from a given query index."""

    def __init__(self, df, edges, candidate_index):
        self.df = df
        self.cascading_edges = CascadingEdges(edges)(candidate_index)

        # Get the level of the last index in the edges
        last_idx = self.cascading_edges[-1]
        self.top_level = df.iloc[last_idx]["level"]

        self.candidate_index = candidate_index
        self.make_neighbor_metrics = MakeNeighborMetrics(df, edges, candidate_index)
        # self.get_neighbors = GetNeighbors(df)
        # self.candidate = None

    def __call__(self, radius_threshold=None, n_neighbors_threshold=None):
        """Given thresholds, search for candidates that meet the criteria."""
        neighbor_metrics, distances, top_vectors = self.make_neighbor_metrics(
            radius=radius_threshold, n_neighbors=n_neighbors_threshold
        )
        if neighbor_metrics is None:
            return None
        neighbors = neighbor_metrics.neighbors
        neighbor_indices = [neighbor.indices[0] for neighbor in neighbors]
        try:
            top_vectors = np.stack(top_vectors)
        except ValueError as e:
            print(e)
            print([vector.shape for vector in top_vectors])
        divergence_metrics = vector_divergence(top_vectors)
        candidate = Candidate(
            self.candidate_index, neighbor_indices, neighbor_metrics, divergence_metrics
        )
        return candidate
