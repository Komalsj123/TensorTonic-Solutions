import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
    """
    
    # Handle empty input
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=int)
    
    # Determine max length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0
    
    N = len(seqs)
    L = max_len
    
    # Initialize with pad_value
    result = np.full((N, L), pad_value, dtype=int)
    
    # Copy sequences with truncation if needed
    for i, seq in enumerate(seqs):
        trunc = seq[:L]          # truncate if longer
        result[i, :len(trunc)] = trunc
    
    return result