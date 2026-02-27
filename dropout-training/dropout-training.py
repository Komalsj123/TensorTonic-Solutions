import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Training-mode dropout with inverted scaling.
    Returns (output, dropout_pattern)
    """

    x = np.asarray(x, dtype=float)

    if rng is None:
        rand = np.random.random(x.shape)
    else:
        rand = rng.random(x.shape)

    # scale factor for kept units
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0

    # pattern: 0 for dropped, scale for kept
    keep = rand < (1.0 - p)
    dropout_pattern = keep.astype(float) * scale

    # output
    output = x * dropout_pattern

    return output, dropout_pattern