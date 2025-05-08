import os
dataset_files = [file for file in os.listdir() if "_M1.dataset" in file]

for dataset_file in dataset_files:
    with open(dataset_file, "r") as f: contents = f.read()
    for i in [2, 3, 4]:
        new_name = dataset_file.replace("_M1.dataset", f"_M{i}.dataset")
        new_contents = contents.replace("mask_file = lsst_y1_M1_GGLOLAP0.05.mask", f"mask_file = lsst_y1_M{i}_GGLOLAP0.05.mask")
        with open(new_name, "w") as f: f.write(new_contents)