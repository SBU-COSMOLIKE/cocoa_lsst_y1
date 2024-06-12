import shutil
import os
fids = ["AS_HIGH", "AS_LOW", "NS_HIGH", "NS_LOW"]
for fid in fids:
    original_name = f"CREATE_DV_OMEGAM_HIGH_{fid}_W_LOW.yaml"
    new_name = f"CREATE_DV_OMEGAM_HIGH_{fid}_W_LOW_BACCO.yaml"
    if os.path.isfile(original_name): shutil.copyfile(original_name, new_name)
    else: print(f"ERROR: could not find file for fid {fid}")
    with open(new_name, "r")  as f:
        contents = f.read()
    contents = contents.replace(f'print_datavector_file: "./projects/lsst_y1/data/OMEGAM_HIGH_{fid}_W_LOW_NO_NOISE.modelvector"', f'print_datavector_file: "./projects/lsst_y1/data/OMEGAM_HIGH_{fid}_W_LOW_BACCO_NO_NOISE.modelvector"')
    contents = contents.replace("non_linear_emul: 1", "non_linear_emul: 10")
    with open(new_name, "w") as f:
        f.write(contents)