template_file_path = "MCMC356.yaml"
specifications_path = "../new_yamls.txt"

with open(template_file_path, "r") as f:
    template = f.read()

PC_block = """  LSST_COLA_Q1:
    prior:
      min: -100
      max: 100
    proposal: 2
    latex: Q1_\mathrm{COLA}^1
"""
PC_in_speed_blocking = ", LSST_COLA_Q1"

fiducials_lcdm = {
    f"{om_type} Omega_m {param_type} {param}": f"LSST_Y1_MX_OM_{om_type.upper()}_{param.upper()}_{param_type.upper()}" for om_type in ["High", "Low"] for param_type in ["High", "Low"] for param in ["As", "ns"]
}
fiducials_varying_w = {
    f"{om_type} Omega_m {param_type} {param} {w_type} w": f"LSST_Y1_MX_OM_{om_type.upper()}_{param.upper()}_{param_type.upper()}_W_{w_type.upper()}" for om_type in ["High", "Low"] for param_type in ["High", "Low"] for param in ["As", "ns"] for w_type in ["High", "Low"]
}
fiducials = {**fiducials_lcdm, **fiducials_varying_w}

with open(specifications_path, "r") as f:
    specs = f.read().splitlines()

for spec in specs:
    code = int(spec[:3])
    rest = spec[6:].split(" / ")
    emulator = rest[0]
    mask = rest[1]
    model = rest[2]
    anchors = rest[3]
    fiducial = rest[4]
    if len(rest) == 6:
        num_pcs = rest[5]

    assert emulator in ["EE2", "COLA", "COLA high"], f"Invalid emulator {emulator}"
    assert mask in ["M1", "M2", "M3", "M4", "M5"], f"Invalid mask {mask}"
    assert model in ["LCDM", "wCDM"], f"Invalid model {model}"
    assert anchors in ["--", "1 anchor", "25 anchors", "100 LCDM anchors"], f"Invalid number of anchors {anchors}"
    assert fiducial in fiducials, f"Invalid fiducial {fiducial}"

    print("Chain code:", code)
    print("Emulator:", emulator)
    print("Mask:", mask)
    print("Num anchors:", anchors)
    print("Fiducial:", fiducial)

    # Change the output to match the code
    new_contents = template.replace("output: ./projects/lsst_y1/chains/MCMC356/MCMC356", f"output: ./projects/lsst_y1/chains/MCMC{code}/MCMC{code}")
    # Change data file
    data_file_name = fiducials[fiducial].replace("MX", mask)
    new_contents = new_contents.replace("data_file: LSST_Y1_M2_OM_HIGH_AS_HIGH.dataset", f"data_file: {data_file_name}.dataset")
    # Change non_linear_emul
    if emulator == "EE2": new_contents = new_contents.replace("non_linear_emul: 5", f"non_linear_emul: 1")
    # Change cola_precision
    if emulator == "COLA high":
        new_contents = new_contents.replace("cola_precision: 1", f"cola_precision: 2")
        new_contents = new_contents.replace("num_refs: 100", "num_refs: 1")
    # Change num_refs
    if anchors != "--":
        num_anchors = int(anchors.split()[0])
        new_contents = new_contents.replace("num_refs: 100", f"num_refs: {num_anchors}")
    # Remove PC block if not marginalizing, also remove Q1 from blocking
    if len(rest) != 6:
        new_contents = new_contents.replace(PC_block, "")
        new_contents = new_contents.replace(PC_in_speed_blocking, "")
    new_yaml_name = f"MCMC{code}.yaml"
    with open(new_yaml_name, "w") as f:
        f.write(new_contents)