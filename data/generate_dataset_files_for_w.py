params = ["AS", "NS"]
values = ["HIGH", "LOW"]
masks = ["M2", "M3", "M4"]
for param in params:
    for value in values:
        for mask in masks:
            dataset_file_name = f"LSST_Y1_{mask}_OM_HIGH_{param}_{value}.dataset"
            with open(dataset_file_name, "r") as f:
                contents = f.read()
            original_dv_file = f"OMEGAM_HIGH_{param}_{value}_NOISED.modelvector"
            for w_value in ["LOW", "HIGH"]:
                new_dataset_file_name = f"LSST_Y1_{mask}_OM_HIGH_{param}_{value}_W_{w_value}.dataset"
                dv_file = f"OMEGAM_HIGH_{param}_{value}_W_{w_value}_NOISED.modelvector"
                new_contents = contents.replace(original_dv_file, dv_file)
                if new_contents == contents:
                    print(f"ERROR: for {param} {value}, could not replace the data vector file")
                with open(new_dataset_file_name, "w") as f:
                    f.write(new_contents)