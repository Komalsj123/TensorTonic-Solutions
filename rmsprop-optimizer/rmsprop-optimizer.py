import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    
    # Convert to numpy arrays
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    
    # Update running average of squared gradients
    new_s = beta * s + (1 - beta) * (g * g)
    
    # Update parameters
    new_w = w - lr * g / (np.sqrt(new_s) + eps)
    
    return new_w, new_s