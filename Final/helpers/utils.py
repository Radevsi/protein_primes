"""
Contains utility functions mostly for the cascades module,
but has some other utilities for general use.

"""

from typing import List

from moleculib.protein.alphabet import all_residues


# Dictionary map from 3-letter amino acids to single letter
d = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
    "PAD": "PAD",
    "MASK": "MASK",
    "UNK": "UNK",
}


def residue_map(tokens):
    """Map from residue tokens to three-letter amino acids"""
    return [all_residues[int(token)] for token in tokens]


def aa_map(sequences):
    """Map from three-letter amino acids to single letter"""
    if not isinstance(sequences, list):
        sequences = [sequences]
    shorts = []
    for sequence in sequences:
        # print(f"Sequence with type: {type(sequence)}, is {sequence}")
        short = "".join([d[aa] for aa in sequence])
        shorts.append(short)
    return shorts


class Idx2Datum:
    """Given a DataFrame, return the datum object given the index,
    or list of indices.
    """

    def __init__(self, df):
        self.df = df

    def __call__(self, *idxs):
        return self.df.loc[idxs, "datum"].values


class CheckPDBs:
    """Initialize with a list of PDB IDs to check for in the dataset."""

    def __init__(self, pdb_ids: List[str]):
        self.pdb_ids = pdb_ids

        self.common_keys = []
        self.missed_keys = []

    def df(self, df):
        """Compare with a pandas dataframe containing a 'pdb_id' column.
        Case insensitive, and allows for chain ids being appended to the
        pdb ids in the dataframe.
        """
        present_ids = df[df["pdb_id"].str.contains("|".join(self.pdb_ids), case=False)][
            "pdb_id"
        ].unique()
        self.common_keys = present_ids
        self.missed_keys = [
            pdb_id for pdb_id in self.pdb_ids if pdb_id not in present_ids
        ]
        return self.common_keys, self.missed_keys

    def ds(self, ds):
        """Compare with a dataset, which should be a dictionary. Proesses all
        pdbs as lowercase and again checks for string inclusion of the queried pdb ids
        inside the dataset keys.
        """
        pdbids_lower = [pdb_id.lower() for pdb_id in self.pdb_ids]
        dataset_keys = [key[:-1].lower() for key in ds.keys()]

        # Compare
        self.common_keys = [key for key in pdbids_lower if key in dataset_keys]
        self.missed_keys = [
            pdb_id for pdb_id in pdbids_lower if pdb_id not in dataset_keys
        ]

        return self.common_keys, self.missed_keys
