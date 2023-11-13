codes = range(206, 286)

for code in codes:
	with open(f"MCMC{code}.yaml", "r") as f:
		lines = f.readlines()

	for i, line in enumerate(lines):
		if "num_refs:" in line:
			assert lines[i+1] == "\n"
			lines[i+1] = "    cola_emu_mode: LCDM\n"
			break
		
	with open(f"MCMC{code}.yaml", "w") as f:
		f.writelines(lines)
	