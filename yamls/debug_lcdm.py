codes = range(207, 286)

for code in codes:
	with open(f"MCMC{code}.yaml", "r") as f:
		lines = f.readlines()

	for i, line in enumerate(lines):
		if "As_1e9, ns, H0, omegab, omegam, w " in line:
			lines[i] = "            As_1e9, ns, H0, omegab, omegam\n"
		
	with open(f"MCMC{code}.yaml", "w") as f:
		f.writelines(lines)
