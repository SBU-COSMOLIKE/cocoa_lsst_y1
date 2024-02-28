codes = [362, 364, 410, 412, 474, 476]
template_file_path = "POST181_100.yaml"
with open(template_file_path, "r") as f:
    template = f.read()

def find_data_file(code):
    with open(f"MCMC{code}.yaml", "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        if "data_file:" in line:
            words = line.split()
            assert ".dataset" in words[-1], "This could be a bug"
            return words[-1]
    assert False, "Could not find dataset file"

for code in codes:
    for precision in ["default", "high"]:
        contents = template.replace("output: ./projects/lsst_y1/chains/MCMC181", f"output: ./projects/lsst_y1/chains/MCMC{code}/MCMC{code}")
        suffix = "COLA100" if precision == "default" else "COLA100_high"
        contents = contents.replace("suffix: COLA100", f"suffix: {suffix}")
        data_file = find_data_file(code)
        contents = contents.replace("data_file: LSST_Y1_M4_OM_HIGH_AS_HIGH.dataset", f"data_file: {data_file}")
        if precision == "high":
            contents = contents.replace("non_linear_emul: 5", "non_linear_emul: 6")
            contents = contents.replace("cola_precision: 1", "cola_precision: 2")
        file_name = f"POST{code}_100"
        if precision == "high": file_name += "_HIGH"
        file_name += ".yaml"
        with open(file_name, "w") as f:
            f.write(contents)