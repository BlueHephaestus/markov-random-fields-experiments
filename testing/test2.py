import numpy as np

a = np.array([np.arange(5)]).transpose()
b = np.array([np.arange(5)])
c = a.dot(b)

"""
Given vector of 2d indices, we can get all the values at these indices in a 2d matrix.
"""
indices = [(0,0), (3,2), (1,4), (4,4)]
#print indices
print c
for pair in indices:
    print c[pair]
    #indices[pair_i] = i*5 + j

#print np.take(c, indices)
