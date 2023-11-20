codes = range(206, 286)

for code in codes:
	with open(f"MCMC{code}.yaml", "r") as f:
		lines = f.readlines()

	for i, line in enumerate(lines):
		if "w: -1" in line: break
	
	lines = lines[:i+1] + ["  wa: 0\n"] + lines[i+1:]
		
	with open(f"MCMC{code}.yaml", "w") as f:
		f.writelines(lines)
	