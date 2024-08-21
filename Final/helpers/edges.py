"""
edges.py

Contains functionality for connecting edges to form a hierarchial graph.
"""

from typing import Dict, List, Tuple
import pandas as pd

def connect_edges(
    df: pd.DataFrame, kernel_size: int, stride: int
) -> Tuple[Dict[int, List[int]], Dict[int, int], int]:
    """Connect edges between nodes in the hierarchy based on the kernel size and stride."""
    n_misses = 0
    edges_top_down, edges_bottom_up = dict(), dict()
    grouped_by_pdb = df.groupby("pdb_id")

    # For each PDB...
    for pdb_id, pdb_group in grouped_by_pdb:
        unique_levels = sorted(pdb_group["level"].unique())

        # For each hierarchy level in the autoencoder...
        for level in unique_levels:
            lower_level, upper_level = level, level + 1
            lower_level_group = pdb_group[
                pdb_group["level"] == lower_level
            ].sort_values(by="level_idx")
            upper_level_group = pdb_group[
                pdb_group["level"] == upper_level
            ].sort_values(by="level_idx")
            num_lower_level = len(lower_level_group)
            for start in range(0, num_lower_level, stride):
                end = start + kernel_size
                lower_level_slice = lower_level_group.iloc[start:end]
                upper_level_node_index = start // stride
                if upper_level_node_index < len(upper_level_group):
                    upper_level_node = upper_level_group.iloc[upper_level_node_index]

                    # Key is index of upper node, value is list of indices for all lower nodes
                    edges_top_down[upper_level_node.name] = list(
                        lower_level_slice.index
                    )
                    edges_bottom_up.update(
                        dict.fromkeys(lower_level_slice.index, upper_level_node.name)
                    )
                else:
                    n_misses += 1

    return edges_top_down, edges_bottom_up, n_misses


##################### Cascading Edges #####################

class CascadingEdges:
    """Initialize the CascadingEdges with a mapping from child to parent.

    Args:
    edges_bottom_up (Dict[int, int]): Dictionary mapping from child index to parent index.
    """

    def __init__(self, edges_bottom_up: Dict[int, int]):
        self.edges_bottom_up = edges_bottom_up

    def __call__(
        self, start_index: int, n_cascades: int = None, verbose=False
    ) -> List[int]:
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

        if len(cascades) == 1:
            print(f"Edge Warning: No parent found for {start_index}.")

        return cascades

