import numpy as np

L = np.loadtxt("./cholesky_L.txt")

len_lhs = 30
diffs = np.zeros((1560/2, 30))
for i in range(len_lhs):
    cola_dv = np.loadtxt(f"./data_vectors/COLA_{i}.txt", max_rows=1560/2)
    ee2_dv = np.loadtxt(f"./data_vectors/EE2_{i}.txt", max_rows=1560/2)
    diff = ee2_dv - cola_dv
    diffs[:, i] = diff