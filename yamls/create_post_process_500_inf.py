codes = [362, 363, 364, 365, 410, 411, 412, 413, 474, 475, 476, 477]
template_file_path = "POST362_100.yaml"
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
    for method in ['500', 'INF']:
        contents = template.replace("./projects/lsst_y1/chains/MCMC362/MCMC362", f"./projects/lsst_y1/chains/MCMC{code}/MCMC{code}")
        suffix = "COLA500" if method == "500" else "COLAINF"
        contents = contents.replace("suffix: COLA100", f"suffix: {suffix}")
        contents = contents.replace("data_file: LSST_Y1_M2_OM_HIGH_AS_HIGH_W_LOW.dataset", f"data_file: {find_data_file(code)}")
        emu_num = 6 if method == "500" else 8
        contents = contents.replace("non_linear_emul: 5", f"non_linear_emul: {emu_num}")
        num_refs = 500 if method == "500" else 0
        contents = contents.replace("num_refs: 100", f"num_refs: {num_refs}")
        new_yaml_path = f"POST{code}_{method}.yaml"
        with open(new_yaml_path, "w") as f:
            f.write(contents)