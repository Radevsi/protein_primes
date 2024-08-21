"""Module to load and save data"""

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


import pickle
import json


class LoadData:
    """Load embeddings and sliced proteins and optional tsne files."""
    def __init__(self, FOLDER, embeddings_file, sliced_proteins_file, tsne_file=None):
        self.FOLDER = FOLDER
        self.embeddings_file = embeddings_file
        self.sliced_proteins_file = sliced_proteins_file
        self.tsne_file = tsne_file

    def load_embeddings(self):
        print(f"Loading embeddings from {self.FOLDER}/{self.embeddings_file}")
        with open(f"{self.FOLDER}/{self.embeddings_file}", "rb") as f:
            self.encoded_dataset = pickle.load(f)

    def load_sliced_proteins(self):
        print(f"Loading sliced proteins from {self.FOLDER}/{self.sliced_proteins_file}")
        with open(f"{self.FOLDER}/{self.sliced_proteins_file}", "rb") as f:
            self.sliced_dataset = pickle.load(f)

    def load_tsne(self):
        print(f"Loading tsne data from {self.FOLDER}/{self.tsne_file}")
        with open(f"{self.FOLDER}/{self.tsne_file}", "r") as f:
            self.tsne_dataset = json.load(f)

    def load_all(self):
        self.load_embeddings()
        self.load_sliced_proteins()
        if self.tsne_file is not None:
            self.load_tsne()

    def load_all_and_return(self):
        self.load_all()
        return self.encoded_dataset, self.sliced_dataset, self.tsne_data

# Code to save things to json files
def save_df(df, filename, folder='data/'):
    """Save a DataFrame of embedded data."""
    # Convert the DataFrame to a JSON string including the index
    df.to_json(f'{folder}/{filename}.json', orient='records')
    print(f"DataFrame saved as JSON to {folder}/{filename}.json")

def save_edges(edges_bottom_up, filename, folder='data/'):
    """Save an edges dictionary to a JSON file."""
    # Convert the edges_bottom_up dictionary to a list of tuples with integers
    edges_bottom_up_tuples = [(int(k), int(v)) for k, v in edges_bottom_up.items()]

    # Convert the list of tuples to JSON format
    edges_bottom_up_json = json.dumps(edges_bottom_up_tuples)

    # Save the JSON data to a file
    with open(f'{folder}/{filename}.json', 'w') as file:
        file.write(edges_bottom_up_json)

    print(f"edges_bottom_up has been saved to {folder}/{filename}.json")