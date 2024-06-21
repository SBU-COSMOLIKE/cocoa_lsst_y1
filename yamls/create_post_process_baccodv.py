from yaml_utils import find_data_file

chains = range(700, 706)
template_yaml_name = "POST63_HIGH.yaml"
with open(template_yaml_name, "r") as f:
    template = f.read()

for chain in chains:
    # Create COLA500 yaml
    contents = template.replace("MCMC63", f"MCMC{chain}/MCMC{chain}")
    contents = contents.replace("suffix: COLAHIGH", "suffix: COLA500")
    data_file = find_data_file(chain)
    contents = contents.replace("data_file: LSST_Y1_M3_OM_HIGH_NS_HIGH.dataset", f"data_file: {data_file}")
    contents = contents.replace("non_linear_emul: 5", f"non_linear_emul: 6")
    contents = contents.replace("num_refs: 1", f"num_refs: 500")
    contents = contents.replace("cola_precision: 2", f"cola_precision: 1")
    yaml_name = f"POST{chain}_500.yaml"
    with open(yaml_name, "w") as f:
        f.write(contents)

    # Create COLAinf yaml
    contents = template.replace("MCMC63", f"MCMC{chain}/MCMC{chain}")
    contents = contents.replace("suffix: COLAHIGH", "suffix: COLAINF")
    data_file = find_data_file(chain)
    contents = contents.replace("data_file: LSST_Y1_M3_OM_HIGH_NS_HIGH.dataset", f"data_file: {data_file}")
    contents = contents.replace("non_linear_emul: 5", f"non_linear_emul: 8")
    contents = contents.replace("num_refs: 1", f"num_refs: 0")
    contents = contents.replace("cola_precision: 2", f"cola_precision: 1")
    yaml_name = f"POST{chain}_INF.yaml"
    with open(yaml_name, "w") as f:
        f.write(contents)