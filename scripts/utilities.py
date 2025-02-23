import numpy as np

def pclip(p):
    """Clip recall probability to avoid numerical issues."""
    return p.clip(0.0001, 0.9999)

def hclip(h, min_half_life, max_half_life):
    """Clip half-life to a reasonable range."""
    return h.clip(min_half_life, max_half_life)

def mae(l1, l2):
    # mean average error
    return mean([abs(l1[i] - l2[i]) for i in range(len(l1))])

def mean(lst):
    # the average of a list
    return float(sum(lst))/len(lst)

def cap_y(prediction): 
    return np.clip(prediction, 0, 1)
