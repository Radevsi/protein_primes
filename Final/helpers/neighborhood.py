"""
neighborhood.py
Radius Nearest Neighbors
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


import numpy as np
import pandas as pd
import sklearn.neighbors as skn
import scipy.spatial.distance as ssd

from helpers.cascades import Cascade, MakeCascade, MakeMetricsPair, Metrics
from helpers.edges import CascadingEdges

# from helpers.metrics


class GetNeighbors:
    """Initialize with a dataframe, and then given an index, query for its
    neighbors. Supports both radius-based and n_neighbor-based search.
    """

    def __init__(self, df):
        self.df = df

    def _from_query(self, query_index, metric):
        """Get distances and indices from query"""
        level = self.df.loc[query_index, "level"]
        query_vec = [self.df.loc[query_index, "scalar_rep"]]

        level_df = self.df[self.df["level"] == level]
        reps_for_level = level_df["scalar_rep"].values.tolist()

        distances = ssd.cdist(query_vec, reps_for_level, metric=metric).flatten()
        indices = np.argsort(distances)

        # There should be a one-to-one correspondence between distances and indices
        return np.sort(distances), indices, level_df

    def __call__(
        self, query_index, radius=None, n_neighbors=None, metric="cosine"
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Allow both radius-based and n_neighbors-based queries. Only one should be specified, but
        will default to neighbors if both are given. If neither are given, will return all neighbors.
        """
        distances, indices, level_df = self._from_query(query_index, metric)
        # distances = np.array(distances)
        if n_neighbors is not None:
            return (
                distances[: n_neighbors + 1],
                level_df.iloc[indices[: n_neighbors + 1]],
            )
        elif radius is not None:
            indices_within_radius = [
                i for i, dist in enumerate(distances) if dist < radius
            ]
            return (
                distances[indices_within_radius],
                level_df.iloc[indices[indices_within_radius]],
            )
        else:
            return distances[indices], level_df.iloc[indices]


@dataclass
class NeighborMetrics:
    """A NeighborMetrics object stores a query and its neighbors. The query is
    stored as a cascade, and the neighbors as a list of cascades.

    The metrics are all computed relative to the query.

    """

    query: Cascade = None
    neighbors: List[Cascade] = None
    metrics: List[List[Metrics]] = None

    def __len__(self):
        """Return the number of neighbors."""
        return len(self.metrics)

    def __str__(self):

        # table_header = f"| {'PDB ID':^10} | {'Sequence':^20} |\n"
        # table_divider = f"|{'-'*12}|{'-'*22}|\n"
        # table_rows = table_header + table_divider
        # query_row = f"| {self.query.pdb_id:^10} | {self.query.sequences[0]:^20} |\n"
        # table_rows += query_row
        # for neighbor in self.neighbors:
        #     neighbor_row = f"| {neighbor.pdb_id:^10} | {neighbor.sequences[0]:^20} |\n"
        #     table_rows += neighbor_row
        # return table_rows

        neighbors_info = f"Showing Neighbors with {len(self)} comparisons.\n"
        neighbors_info += f"Query: {self.query}\n"
        for i, neighbor in enumerate(self.neighbors):
            neighbors_info += f"Neighbor {i}: {neighbor}\n"
        for i, metric in enumerate(self.metrics):
            neighbors_info += f"Metrics {i}: {metric}\n"
        return neighbors_info

    def plot(self, start_index=None, end_index=None):
        """Plot the cascades."""
        if start_index is not None and end_index is not None:
            return self.plot_neighbors(
                self.query,
                self.neighbors[start_index:end_index],
                self.metrics[start_index:end_index],
            )
        elif start_index is not None:
            end_index = start_index
            return self.plot_neighbors(
                self.query, self.neighbors[:end_index], self.metrics[:end_index]
            )
        return self.plot_neighbors(self.query, self.neighbors, self.metrics)

    @staticmethod
    def plot_neighbors(
        query: Cascade,
        neighbors: List[Cascade],
        metrics: Optional[List[List[Metrics]]] = None,
        return_indices=False,
    ):
        """Plot the cascades for the query and its neighbors."""

        view1, indices1 = MakeCascade.plot_cascade(query, return_indices=True)
        print(f"Query: {query.pdb_id}. part sequence: {query.sequences_short[0]}")
        view1.show()
        views = [view1]
        indices_lst = [indices1]
        for i, neighbor in enumerate(neighbors):
            print(
                f"Neighbor {i} at index {neighbor.indices[0]}, PDB ID: {neighbor.pdb_id}. part sequence: {neighbor.sequences_short[0]}",
                end=" -- ",
            )
            print(
                f"Alignment: {metrics[i][0].alignment}, RMSD: {metrics[i][0].distance:.4f}, cosine: {metrics[i][0].cosine:.6f}"
            )
            view, indices = MakeCascade.plot_cascade(neighbor, return_indices=True)
            print(f"Full sequence: {neighbor.sequences_short[-1]} -- part sequence at indices: {indices[-1]}")
            view.show()
            views.append(view)
            indices_lst.append(indices)
        # for view in views:
        #     view.show()

        if return_indices:
            return views, indices
        return


class MakeNeighborMetrics:
    """Make a NeighborMetrics object from a DataFrame and a single index.
    i.e., find the neighbors of a single cascade.
    """

    def __init__(self, df, edges, u):
        self.df = df
        self.cascading_edges = CascadingEdges(edges)
        self.u = u
        self.get_neighbors = GetNeighbors(df)

    def __call__(self, n_neighbors=None, radius=None) -> NeighborMetrics:
        u_cascades = self.cascading_edges(self.u)
        top_vectors = [self.df.iloc[u_cascades[-1]]["scalar_rep"]]
        neighbors = []
        metrics = []
        distances, neighbors_df = self.get_neighbors(
            self.u, n_neighbors=n_neighbors, radius=radius
        )
        query_neighbor_pair = None

        for v in neighbors_df.index:
            if v == self.u:
                continue
            query_neighbor_pair, v_vec = MakeMetricsPair(
                self.df, u_cascades, self.cascading_edges(v)
            )(return_v_vec=True)
            # print(f"Vec shape: {v_vec.shape}")
            top_vectors.append(v_vec)
            neighbors.append(query_neighbor_pair.cascade2)
            metrics.append(query_neighbor_pair.metrics)
        if query_neighbor_pair is None:
            # raise ValueError(f"No neighbors found for index: {self.u}.")
            return None, None, None
        return (
            NeighborMetrics(
                query=query_neighbor_pair.cascade1, neighbors=neighbors, metrics=metrics
            ),
            distances,
            top_vectors,
        )


class RadiusNeighbors:
    """Class to build a similarity graph from DataFrame.

    Returns in csr matrix format.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Get the df where datum is not None
        # self.df = self.df[self.df['datum'].notna()]

    # def _get_scalars(self, **kwargs):
    #     """Return the scalar representations given a selection"""
    #     # Shape here is (N,), but return shape (N, M)
    #     sub_df = get_column(self.df, **kwargs)
    #     scalar_representations = sub_df['scalar_rep']
    #     return np.stack(scalar_representations), sub_df.reset_index(drop=False)

    def get_radius_neighbors(self, radius, level, sort=False):
        """Return the indices of the scalar representations that are within a certain radius
        of each other.
        """
        level_df = self.df[(self.df["level"] == level)].reset_index(drop=False)
        scalar_reps = np.stack(level_df["scalar_rep"].values)
        # scalar_reps, sub_df = self._get_scalars(**kwargs)
        # print(f"Processing {len(scalar_reps)} scalar representations")
        print(f"Shape of scalar reps: {scalar_reps.shape}")
        graph = skn.radius_neighbors_graph(
            scalar_reps, radius, mode="distance", metric="cosine"
        )
        if sort:
            graph = skn.sort_graph_by_row_values(graph, warn_when_not_sorted=False)
        return graph, level_df
