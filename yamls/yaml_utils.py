def find_data_file(code):
    with open(f"MCMC{code}.yaml", "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        if "data_file:" in line:
            words = line.split()
            assert ".dataset" in words[-1], "This could be a bug"
            return words[-1]
    assert False, "Could not find dataset file"