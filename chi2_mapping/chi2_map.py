import numpy as np
from cobaya.yaml import yaml_load_file
from cobaya.input import update_info
from cobaya.run import run

#paths currently configured to run from the cocoa/Cocoa directory

info = yaml_load_file('./projects/lsst_y1/chi2_map.yaml')
#info['model']['emulation']['settings']['non_linear_emul'] = non_linear_emu

grid = np.loadtxt('./projects/lsst_y1/chi2_grid_omegam_As.txt')
params = ['omegam','As_1e9']
chi2_out = f'./projects/lsst_y1/chi2_2.txt'

for i in range(1):
	print('---------------------------------')
	for j in range(len(grid[0])):
		info['sampler']['evaluate']['override'][params[j]] = grid[i][j]
	print(f'Running Cosmology #{i}: omegam, As = {grid[i][0]}, {grid[i][1]}')
	info, sampler = run(info, force = True)
	chi2 = np.loadtxt('./projects/lsst_y1/chains/chi2map.1.txt', usecols=40)
	# f = open(chi2_out, "a")
	# f.write(f'{chi2} \n')
	# f.close()
	print(f"CHI2 FOR COSMOLOGY #{i} IS {chi2}")
	for j in range(len(grid[0])):
		info['sampler']['evaluate']['override'][params[j]] = grid[i][j]


