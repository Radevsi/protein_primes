"""
Contains functions for calculating metrics on both object pairs, and
distributions over entire hierarchy levels.
"""

import numpy as np
from Bio import Align
from einops import rearrange


### Begin Distance Maps for single pair of ProteinDatum objects ###

class DistanceMapMetric:
    """Get structure-based distance score"""
    def __call__(self, datum1, datum2, verbose=False):

        min_length = min(len(datum1), len(datum2))
        datum1 = datum1[:min_length]
        datum2 = datum2[:min_length]

        backbone1 = datum1.atom_coord[..., 1, :]
        backbone2 = datum2.atom_coord[..., 1, :]

        mask1 = datum1.atom_mask[..., 1]
        mask2 = datum2.atom_mask[..., 1]

        if (mask1 != mask2).any():
            if verbose:
                print(f'[WARNING!] Masks are mismatching at {(mask1 != mask2).sum()} places')

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

        # mask1 = datum1.atom_mask[..., 1]
        # mask2 = datum2.atom_mask[..., 1]
        # mask = mask1 & mask2

        min_length = min(len(datum1), len(datum2))
        datum1 = datum1[:min_length]
        datum2 = datum2[:min_length]

        seq1 = np.array(datum1.residue_token, np.int32)
        seq2 = np.array(datum2.residue_token, np.int32)

        # Do a Hamming distance
        hamming_distance = sum(seq1 != seq2)
        if len(seq1) != len(seq2):
            raise ValueError("Sequences must be of equal length.")

        # Alignment
        aligner = Align.PairwiseAligner()
        alignments = aligner.align(seq1, seq2)
        best_alignment = alignments[0]
        alignment_score = best_alignment.score
        # print("Best alignment score:", alignment_score)

        return alignment_score, hamming_distance

### End Distance Maps for single pair of ProteinDatum objects ###



### BEGIN indexing functions ###

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

### END indexing functions ###
