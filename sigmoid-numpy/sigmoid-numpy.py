import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.asarray(x, dtype=float)   # Convert input to NumPy array
    return 1 / (1 + np.exp(-x))      # Vectorized sigmoid