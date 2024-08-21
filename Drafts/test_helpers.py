"""Testing Module"""

import unittest
from helpers_new import populate_representations, get_column

class TestHelpers(unittest.TestCase):
    """Testing Class"""
    def __init__(self, tsne_data=None, encoded_dataset=None, sliced_dataset=None):
        super().__init__()
        self.tsne_data = tsne_data
        self.encoded_dataset = encoded_dataset  
        self.sliced_dataset = sliced_dataset
        self.df = None

    def setUp(self):
        reps, _ = populate_representations(self.encoded_dataset, self.sliced_dataset)
        self.df = reps.to_dataframe()

    def test_tsne_dataframe_match(self):
        """Test that the loaded tsne file and encoded dataset files have the same structure."""
        mismatches = {}
        for pdb_id in self.tsne_data:
            for level in self.tsne_data[pdb_id]:
                num_coords = len(self.tsne_data[pdb_id][level]['pos'])
                num_rows = len(get_column(self.df, pdb_id=pdb_id, level=int(level))) 
                if num_coords != num_rows:
                    if pdb_id not in mismatches:
                        mismatches[pdb_id] = {}
                    mismatches[pdb_id][level] = (num_coords, num_rows)
        self.assertEqual(len(mismatches), 0, f"Mismatches found: {mismatches}")

    def test_get_column(self):
        """Test the get_column function."""
        pdb_id = self.df['PDBid'].iloc[0]
        level = self.df['level'].iloc[0]
        filtered_df = get_column(self.df, pdb_id=pdb_id, level=level)
        self.assertTrue((filtered_df['PDBid'] == pdb_id).all())
        self.assertTrue((filtered_df['level'] == level).all())


if __name__ == '__main__':
    unittest.main()
    