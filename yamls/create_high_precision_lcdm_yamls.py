template_file = "MCMC209.yaml"

with open(template_file, "r") as f:
    template = f.read()

# Change in template:
# - output
# - emulator
# - num_refs
# - fiducial and mask
# - cola precision

with open("yamls.txt", "r") as f:
    lines = f.read().splitlines()
lines = [line for line in lines if line != ""]

for line in lines:
    code, rest = line.split(" - ")
    emu, mask, model, anchors, point = rest.split(" / ")

    if emu == "EE2":
        emu_code = 1
    elif emu == "COLA" or emu == "COLA high":
        emu_code = 5
    else:
        print(f"Unknown emulator {emu}")
        exit(1)
    
    if anchors == "--":
        num_anchors = 1
    elif anchors == "1 anchor":
        num_anchors = 1
    elif anchors == "25 anchors":
        num_anchors = 25
    else:
        print(f"Unsupported num of anchors emulator {anchors}")
        exit(1)

    if point == "EE2 ref":
        dataset_file = "LSST_Y1_M1_EE2_NOISED.dataset"
    elif point == "High Omega_m High As":
        dataset_file = "LSST_Y1_M1_OM_HIGH_AS_HIGH.dataset"
    elif point == "Low Omega_m High As":
        dataset_file = "LSST_Y1_M1_OM_LOW_AS_HIGH.dataset"
    elif point == "High Omega_m Low As":
        dataset_file = "LSST_Y1_M1_OM_HIGH_AS_LOW.dataset"
    elif point == "Low Omega_m Low As":
        dataset_file = "LSST_Y1_M1_OM_LOW_AS_LOW.dataset"
    elif point == "High Omega_m High ns":
        dataset_file = "LSST_Y1_M1_OM_HIGH_NS_HIGH.dataset"
    elif point == "Low Omega_m High ns":
        dataset_file = "LSST_Y1_M1_OM_LOW_NS_HIGH.dataset"
    elif point == "High Omega_m Low ns":
        dataset_file = "LSST_Y1_M1_OM_HIGH_NS_LOW.dataset"
    elif point == "Low Omega_m Low ns":
        dataset_file = "LSST_Y1_M1_OM_LOW_NS_LOW.dataset"
    else:
        print(f"Unsupported fiducial point {point}")
        exit(1)
    
    assert mask in ["M2", "M3", "M4"], f"Mask {mask} not a valid mask"

    dataset_file = dataset_file.replace("M1", mask)

    contents = template.replace("MCMC209", f"MCMC{code}")
    contents = contents.replace("non_linear_emul: 5", f"non_linear_emul: {emu_code}")
    contents = contents.replace("num_refs: 1", f"num_refs: {num_anchors}")
    contents = contents.replace("LSST_Y1_M2_OM_HIGH_AS_HIGH.dataset", dataset_file)
    contents = contents.replace("cola_precision: 1", "cola_precision: 2")

    with open(f"MCMC{code}.yaml", "w") as f:
        f.write(contents)