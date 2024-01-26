import sys
import numpy as np
from numpy.linalg import cholesky

# Loading cov data
cov_data = np.loadtxt("../data/cov_lsst_y1")

# Parsing cov data
dv_length = 1560//2
cov = np.zeros((dv_length, dv_length))
for line in cov_data:
    i = int(line[0])
    j = int(line[1])
    if i < dv_length and j < dv_length:
        cov[i,j] = line[8] + line[9]
        cov[j,i] = cov[i,j]

L = cholesky(cov)

np.savetxt("./cholesky_L.txt", L)