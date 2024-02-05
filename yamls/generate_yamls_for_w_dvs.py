for om_value in ["LOW", "HIGH"]:
    for param in ["AS", "NS"]:
        for param_value in ["LOW", "HIGH"]:
            file_path = f"CREATE_DV_OMEGAM_{om_value}_{param}_{param_value}.yaml"
            with open(file_path, "r") as f:
                contents = f.read()
            original_dv_file_path = f"./projects/lsst_y1/data/OMEGAM_{om_value}_{param}_{param_value}_NO_NOISE.modelvector"
            if original_dv_file_path not in contents:
                print(f"ERROR: for om {om_value}, {param} {param_value} the replacement of the dv file path doesn't work")
                exit(1)
            for w_value in ["LOW", "HIGH"]:
                w = -1.1 if w_value == "LOW" else -0.9
                new_dv_file_path = f"./projects/lsst_y1/data/OMEGAM_{om_value}_{param}_{param_value}_W_{w_value}_NO_NOISE.modelvector"
                new_contents = contents.replace(original_dv_file_path, new_dv_file_path)
                if "w: -1.0" not in new_contents:
                    print(f"ERROR: for om {om_value}, {param} {param_value} the replacement of the w fiducial value doesn't work")
                    exit(1)
                new_contents = new_contents.replace("w: -1.0", f"w: {w}")
                new_contents = new_contents.replace("w0pwa: -1.0", f"w0pwa: {w}")
                new_file_path = f"CREATE_DV_OMEGAM_{om_value}_{param}_{param_value}_W_{w_value}.yaml"
                print(f"INFO: Saving yaml {new_file_path}")
                with open(new_file_path, "w") as f:
                    f.write(new_contents)