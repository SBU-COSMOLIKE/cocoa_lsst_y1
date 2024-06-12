indices = range(526, 538)
new_indices = range(712, 724)
for i, j in zip(indices, new_indices):
    with open(f"MCMC{i}.yaml", "r") as f:
        contents = f.read()

    contents = contents.replace(f"MCMC{i}", f"MCMC{j}")
    lines = contents.splitlines()
    for line in lines:
        if "data_file:" in line:
            dataset_file = line.split(": ")[1]
            break
    new_dataset_file = dataset_file.split(".")[0] + "_BACCO.dataset"
    contents = contents.replace(dataset_file, new_dataset_file)
    contents = contents.replace("non_linear_emul: 7", "non_linear_emul: 8")
    # contents = contents.replace("num_refs: 0", "num_refs: 500")
    with open(f"MCMC{j}.yaml", "w") as f:
        f.write(contents)