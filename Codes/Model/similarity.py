import numpy as np

def cosim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def pearson(a, b):
    return np.corrcoef(a, b)[0, 1]

def jacsim(a, b):
    return len(set(a).intersection(set(b))) / len(set(a).union(set(b)))