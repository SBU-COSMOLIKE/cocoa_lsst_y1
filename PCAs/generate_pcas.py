import numpy as np
from numpy.linalg import svd

L = np.loadtxt("./cholesky_L.txt")

len_lhs = 30
diffs = np.zeros((1560//2, 30))
for i in range(len_lhs):
    cola_dv = np.loadtxt(f"./data_vectors/COLA_{i}.modelvector", max_rows=1560//2, unpack=True, usecols=1)
    ee2_dv = np.loadtxt(f"./data_vectors/EE2_{i}.modelvector", max_rows=1560//2, unpack=True, usecols=1)
    diff = ee2_dv - cola_dv
    diffs[:, i] = np.matmul(L, diff)

U, Sigma, V = svd(diffs)

np.savetxt("./PCs/U_matrix.txt", U)
np.savetxt("./PCs/V_matrix.txt", V)
np.savetxt("./PCs/Sigma_matrix.txt", Sigma)

for i in range(len_lhs):
    PC = U[:, i]
    new_PC = np.matmul(L, PC)
    np.savetxt(f"./PCs/PC_{i}.txt", new_PC)