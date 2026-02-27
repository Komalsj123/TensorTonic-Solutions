import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    
    # Positions (T, 1)
    pos = np.arange(seq_len, dtype=float)[:, None]
    
    # Dimension indices (1, d_model)
    i = np.arange(d_model, dtype=float)[None, :]
    
    # Compute angle rates
    angle_rates = 1 / (base ** (2 * (i // 2) / d_model))
    
    angles = pos * angle_rates
    
    # Initialize PE
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # Even columns → sin
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    
    # Odd columns → cos
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe