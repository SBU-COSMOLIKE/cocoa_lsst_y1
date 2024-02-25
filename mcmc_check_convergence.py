import numpy as np
import os

for i in range(424,490):
	try:
		a = np.genfromtxt(f'./chains/MCMC{i}/MCMC{i}.progress', dtype=None, encoding='utf-8')
		r_minus_1 = float(a[-1][-2])
		if r_minus_1 < 0.005:
			print(f'Run {i} converged with R-1={r_minus_1}')
		else:
			print(f'Run {i} NOT converged with R-1={r_minus_1}')
	except FileNotFoundError:
		print(f'Run {i} hasnt began.')