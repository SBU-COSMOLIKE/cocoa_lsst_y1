# MCMC codes for wCDM, EE2, masks M2, M3 and M4
codes = [2, 3, 178, 32, 33, 47, 48, 62, 63, 77, 78, 92, 93, 107, 108, 122, 123, 137, 138, 178, 181, 184, 190, 193, 196, 199, 202]

with open("./POST33.yaml", "r") as f:
    template = f.read()

for code in codes:
    print(f"Writing post-process yamls for chain {code}")
    data_file = None
    new_yaml = template.replace("./projects/lsst_y1/chains/MCMC33", f"./projects/lsst_y1/chains/MCMC{code}")
    with open(f"MCMC{code}.yaml", "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            if "data_file:" in line:
                data_file = line.split(": ")[1]
                break

    if data_file is None:
        print("ERROR: could not parse data_file")
        continue
            
    new_yaml = new_yaml.replace("LSST_Y1_M3_OM_HIGH_AS_HIGH.dataset", data_file)
    with open(f"./POST{code}.yaml", "w") as f:
        f.write(new_yaml)
    new_yaml = new_yaml.replace("suffix: COLA1", "suffix: COLA25")
    new_yaml = new_yaml.replace("num_refs: 1", "num_refs: 25")
    with open(f"./POST{code}_25.yaml", "w") as f:
        f.write(new_yaml)
