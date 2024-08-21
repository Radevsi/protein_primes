"""
Cascades.py

Contains all funtionality related to tracing through a cascade in the graph.
"""

from dataclasses import dataclass
from typing import List
import scipy.spatial.distance as ssd


import pandas as pd
from IPython.display import display

from moleculib.graphics.py3Dmol import plot_py3dmol_grid
from moleculib.protein.datum import ProteinDatum
from moleculib.protein.alphabet import all_residues

from helpers.metrics import (
    longest_common_subsequence_indices,
    lcs_via_alignment,
    DistanceMapMetric,
    DistanceSeqMetric,
)

from helpers.utils import aa_map


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
    residues: List[str] = None  # these are the residue tokens
    sequences: List[List[str]] = None
    lengths: List[int] = None
    levels: List[int] = None
    level_idxs: List[int] = None
    datums: List[ProteinDatum] = None
    cascade_df: pd.DataFrame = None

    def __len__(self):
        return len(self.indices)

    @property
    def sequences_short(self):
        """Return the sequences with single-letter amino acids."""
        return aa_map(self.sequences)

    # Now pretty print the cascade
    def __str__(self):
        cascade_info = f"Cascade for {self.pdb_id} with {len(self)} levels.\n"
        cascade_info += f"Indices: {self.indices}\n"
        for level, level_idx, sequence in zip(
            self.levels, self.level_idxs, self.sequences
        ):
            cascade_info += f"Sequence for level {level} at index {level_idx} (of length {len(sequence)}): {sequence}\n"
        return cascade_info

    def show_df(self):
        """Display the DataFrame filtered by the indices."""
        display(self.cascade_df)

    def plot(self):
        """Plot the cascade."""
        return plot_cascade(self)


class MakeCascade:
    """Given a DataFrame and a list of indices, create a Cascade object."""

    def __init__(self, df, indices: List[int]):
        self.df = df
        self.indices = indices

    def __call__(self):
        datums = []
        residue_tokens = []
        sequences = []
        lengths = []
        levels = []
        level_idxs = []
        for i in self.indices:
            datum = self.df.iloc[i]["datum"]
            datums.append(datum)
            residue_tokens.append(datum.residue_token)
            sequences.append(self.datum_to_sequence(datum))
            lengths.append(len(datum))
            levels.append(self.df.iloc[i]["level"])
            level_idxs.append(self.df.iloc[i]["level_idx"])
        return Cascade(
            pdb_id=self._pdb_id,
            indices=self.indices,
            residues=residue_tokens,
            sequences=sequences,
            lengths=lengths,
            levels=levels,
            level_idxs=level_idxs,
            datums=datums,
            cascade_df=self.df.iloc[self.indices],
        )

    @property
    def datums(self):
        """Return the datums for the indices."""
        return self.df.iloc[self.indices]["datum"].values

    @property
    def _pdb_id(self):
        """Verify that the pdb id is the same for all datums."""
        pdb_ids = self.df.iloc[self.indices]["pdb_id"].values
        assert len(set(pdb_ids)) == 1, "PDB IDs are not the same."
        return pdb_ids[0]

    @staticmethod
    def datum_to_sequence(datum):
        """Given a datum object, return the sequence of the protein."""
        return [all_residues[token] for token in datum.residue_token]

    @staticmethod
    def plot_cascade(cascade: Cascade, return_indices=False):
        """On input a cascade object, plot the cascade with the indices highlighted.
        Assume that the first object in the cascade is the child, and the rest are parents.
        (So we color from index [1:])
        """
        view = plot_py3dmol_grid([cascade.datums])
        indices = []
        # Only rely on datum object
        # child_sequence = MakeCascade.datum_to_sequence(cascade.datums[0])
        for i, datum in enumerate(cascade.datums[1:]):
            local_viewer = (0, i + 1)
            # sequence = MakeCascade.datum_to_sequence(datum)
            # local_indices = longest_common_subsequence_indices(
            #     seq=sequence, subseq=child_sequence
            # )

            local_indices = lcs_via_alignment(
                datum.residue_token, cascade.datums[0].residue_token
            )

            indices.append(local_indices)
            view.addStyle(
                {"model": -1}, {"cartoon": {"color": "white"}}, viewer=local_viewer
            )
            view.addStyle(
                {"model": -1, "resi": local_indices},
                {"cartoon": {"color": "spectrum"}},
                viewer=local_viewer,
            )
        if return_indices:
            return view, indices
        return view

    @staticmethod
    def plot_datums(datums: List[ProteinDatum]):
        """A wrapper of `plot_cascade` for a list of datums."""
        return MakeCascade.plot_cascade(Cascade(datums=datums))


def plot_cascade(cascade: Cascade):
    """On input a cascade object, plot the cascade with the indices highlighted.
    Assume that the first object in the cascade is the child, and the rest are parents.
    (So we color from index [1:])
    """
    view = plot_py3dmol_grid([cascade.datums])
    child_sequence = cascade.sequences[0]
    for i, sequence in enumerate(cascade.sequences[1:]):
        local_viewer = (0, i + 1)
        local_indices = longest_common_subsequence_indices(
            seq=sequence, subseq=child_sequence
        )
        view.addStyle(
            {"model": -1}, {"cartoon": {"color": "white"}}, viewer=local_viewer
        )
        view.addStyle(
            {"model": -1, "resi": local_indices},
            {"cartoon": {"color": "spectrum"}},
            viewer=local_viewer,
        )
    return view


@dataclass
class Metrics:
    """The metrics object stores a single pairwise comparison
    between two ProteinDatum objects and their vector representations.

    On the raw datum side, supports structure-based distance map, sequential alignment score,
    and hamming distance.

    On the vector side, supports cosine distance.
    """

    distance: float
    alignment: float
    hamming: float
    cosine: float


@dataclass
class MetricsPair:
    """A metrics pair object stores information about a pair of cascades
    for comparison. A cascade pair is defined as the bottom-up hierarchical
    relationship of two protein representations and their parents.

    Processes two cascades and computes metrics between them
    """

    cascade1: Cascade = None
    cascade2: Cascade = None
    metrics: List[Metrics] = None

    def __len__(self):
        return len(self.metrics)

    # Now pretty print the cascade pair
    def __str__(self):
        pair_info = f"Cascade Pair with {len(self)} comparisons.\n"
        pair_info += f"Cascade 1: {self.cascade1}\n"
        pair_info += f"Cascade 2: {self.cascade2}\n"
        for i, metric in enumerate(self.metrics):
            pair_info += f"Metrics {i}: {metric}\n"
        return pair_info

    def plot(self):
        """Plot the cascades."""
        return self.plot_cascade_pair(self.cascade1, self.cascade2)

    @staticmethod
    def plot_cascade_pair(cascade1: Cascade, cascade2: Cascade, return_indices=False):
        """Plot the cascades."""
        if return_indices:
            view1, indices1 = MakeCascade.plot_cascade(
                cascade1, return_indices=return_indices
            )
            view2, indices2 = MakeCascade.plot_cascade(
                cascade2, return_indices=return_indices
            )
            return view1, view2, indices1, indices2
        else:
            view1 = MakeCascade.plot_cascade(cascade1)
            view2 = MakeCascade.plot_cascade(cascade2)
        view1.show()
        view2.show()


class MakeMetricsPair:
    """Make a CascadePair object from a DataFrame and two lists of indices."""

    def __init__(self, df, us: List[int], vs: List[int]):
        self.df = df
        self.us = us
        self.vs = vs

    def __call__(self, return_v_vec=False) -> MetricsPair:
        """If `return_v_vec` is provided, return `v_vec` of that level."""
        u_cascade = MakeCascade(self.df, self.us)()
        v_cascade = MakeCascade(self.df, self.vs)()
        metrics = []
        # return u_cascade, v_cascade

        for i, (u, v) in enumerate(zip(self.us, self.vs)):
            u_datum = u_cascade.datums[i]
            v_datum = v_cascade.datums[i]

            # min_length = min(len(u_datum), len(v_datum))
            # u_datum = u_datum[:min_length]
            # v_datum = v_datum[:min_length]

            u_vec = self.df.iloc[u]["scalar_rep"]
            v_vec = self.df.iloc[v]["scalar_rep"]
            if u_vec.shape != v_vec.shape:
                print(self.df.iloc[u], self.df.iloc[v])
                return
            struct_map = DistanceMapMetric()(u_datum, v_datum)
            alignment, hamming = DistanceSeqMetric()(u_datum, v_datum)
            cosine = ssd.cosine(u_vec, v_vec)
            metrics.append(
                Metrics(
                    distance=struct_map,
                    # distance=None,
                    alignment=alignment,
                    hamming=hamming,
                    cosine=cosine,
                )
            )
        if return_v_vec:
            return (
                MetricsPair(cascade1=u_cascade, cascade2=v_cascade, metrics=metrics),
                v_vec,
            )
        return MetricsPair(cascade1=u_cascade, cascade2=v_cascade, metrics=metrics)
