
"""Module for custom plotting functionality"""

from typing import List, Union, Optional
import matplotlib.pyplot as plt
from moleculib.graphics.py3Dmol import plot_py3dmol_grid


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