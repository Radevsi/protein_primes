"""
database.py

Contains functionality for treating a trained Ophiuchus model's embeddings as a database.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from moleculib.protein.datum import ProteinDatum

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
