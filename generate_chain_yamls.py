"""
    COLA w0wa chain generation script.
    Author: João Rebouças
    Description: This script generates `.yaml` files for the chains listed in the documentation. This solution automates the generation of chains as well as promotes a documentation-first approach, where the most important aspect is the documentation of the chains.
"""
import os
import yaml

def diff_nested_dicts(dict1, dict2, label1, label2):
    """
    Compare two nested dictionaries and return the differences.
    """
    diff = {}
    for key in dict1.keys():
        if key not in dict2:
            diff[key] = dict1[key]
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            nested_diff = diff_nested_dicts(dict1[key], dict2[key], label1, label2)
            if nested_diff:
                diff[key] = nested_diff
        elif dict1[key] != dict2[key]:
            diff[key] = f"DIFF({label1} has {dict1[key]}, {label2} has {dict2[key]})" # (dict1[key], dict2[key])
    return diff

if __name__ == "__main__":
    import sys
    force = False
    if len(sys.argv) > 1 and sys.argv[1] == "-f":
        force = True

    # Open documentation file
    with open("CHAINS.md", "r") as f:
        lines = f.readlines()
    
    with open("TEMPLATE_MCMC.yaml", "r") as f:
        template = yaml.safe_load(f)

    for line in lines:
        if line.startswith("#"): continue
        fields = line[2:].split(" | ")
        index = int(fields[0])
        emulator = fields[1].strip()
        fid_cosmo = fields[2].strip()
        scale_cuts = fields[3].strip()
        try:
            extra_info = fields[4]
        except IndexError:
            extra_info = None
        
        # Generate yaml
        print(f"Generating chain {index} for {emulator} with fid_cosmo {fid_cosmo} and scale_cuts {scale_cuts} (extra_info: {extra_info})")
        chain = template.copy()
        chain["output"] = f"./projects/lsst_y1/chains/MCMC{index}/MCMC{index}"

        if emulator == "EE2":
            chain["likelihood"]["lsst_y1.lsst_y1_cosmic_shear"]["non_linear_emul"] = 1
        elif emulator == "COLA":
            chain["likelihood"]["lsst_y1.lsst_y1_cosmic_shear"]["non_linear_emul"] = 3
        else:
            raise Exception(f"Unknown emulator {emulator}")
        
        # NOTE: the files for Omega_m and As varying cosmologies are capitalized
        if fid_cosmo in ["EE2REF", "DESI2CMBPANPLUS"]:
            dataset_file = f"ee2_{fid_cosmo.lower()}_{scale_cuts}.dataset"
        else:
            dataset_file = f"ee2_{fid_cosmo}_{scale_cuts}.dataset"

        if not os.path.isfile(f"data/{dataset_file}"):
            raise Exception(f"Error in chain {index}: Dataset file {dataset_file} not found")
        chain["likelihood"]["lsst_y1.lsst_y1_cosmic_shear"]["data_file"] = dataset_file

        # NOTE(JR): Parsing of extra_info is going to be done in the future when extra_info is needed
        if extra_info is not None:
            raise Exception(f"Error in chain {index}: extra_info {extra_info} is not implemented yet")
        
        # Save yaml
        if force or not os.path.isfile(f"yamls/MCMC{index}.yaml"):
            with open(f"yamls/MCMC{index}.yaml", "w") as f:
                yaml.dump(chain, f, default_flow_style=False)
        else:
            # If yaml file exists and we are not forcing, compare the generated version with what's already there
            with open(f"yamls/MCMC{index}.yaml", "r") as f:
                existing_chain = yaml.safe_load(f)
            if existing_chain != chain:
                print(f"  - Error in chain {index}: Yaml already exists and is different from the generated one.")
                print(f"  - Difference: {diff_nested_dicts(chain, existing_chain, 'new', 'existing')}")
                print("Aborting.")
                exit(1)
            else:
                print(f"  - Chain {index} already generated. Continuing.")
                continue

        