# MCMC codes for wCDM, EE2, masks M2, M3 and M4
codes = [2, 3, 178, 32, 33, 47, 48, 62, 63, 77, 78, 92, 93, 107, 108, 122, 123, 137, 138, 181, 184, 187, 190, 193, 196, 199, 202]

with open(f"./POST2.yaml", "r") as f:
    template = f.read()

for code in codes:
    print(f"Writing post-process yamls for chain {code}")
    
    new_yaml = template.replace("./projects/lsst_y1/chains/MCMC2", f"./projects/lsst_y1/chains/MCMC{code}")
    new_yaml = new_yaml.replace("suffix: COLA1", "suffix: COLAHIGH")
    new_yaml = new_yaml.replace("cola_precision: 1", "cola_precision: 2")
    
    data_file = None
    with open(f"MCMC{code}.yaml", "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            if "data_file:" in line:
                data_file = line.split(": ")[1]
                break

    if data_file is None:
        print("ERROR: could not parse data_file")
        continue
            
    new_yaml = new_yaml.replace("LSST_Y1_M2_EE2_NOISED.dataset", data_file)
    with open(f"./POST{code}_HIGH.yaml", "w") as f:
        f.write(new_yaml)
