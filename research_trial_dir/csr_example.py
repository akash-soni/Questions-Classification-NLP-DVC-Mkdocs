import numpy as np
from scipy.sparse import csr_matrix

A = np.array([
    [1,0,0,0,0,1,0],
    [0,1,0,3,0,0,0],
    [0,0,0,0,1,0,2],

])

print(A)

s = csr_matrix(A)
print(s)
print(type(s))