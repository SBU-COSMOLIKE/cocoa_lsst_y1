import os
import sys
import numpy as np

lhs = np.loadtxt("./pcas_lhs_30.txt")
with open("./generate_dv_template.yaml", "r") as f:
    template = f.read()

for i, (Omega_m, Omega_b, ns, As, h) in enumerate(lhs):
    yaml = template
    yaml = yaml.replace("COLA_0.modelvector", f"EE2_{i}.modelvector")
    yaml = yaml.replace("As_1e9: 2.1", f"As_1e9: {As*1e9}")
    yaml = yaml.replace("ns: 0.96", f"ns: {ns}")
    yaml = yaml.replace("H0: 67.0", f"H0: {100*h}")
    yaml = yaml.replace("omegab: 0.049", f"omegab: {Omega_b}")
    yaml = yaml.replace("omegam: 0.319", f"omegam: {Omega_m}")
    yaml_file_path = f"./yamls/EE2_{i}.yaml"
    with open(yaml_file_path, "w") as f:
        f.write(yaml)