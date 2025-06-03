"""
    COLA w0wa postprocess generation script.
    Author: João Rebouças
    Description: This script generates `.yaml` files for postprocessing the chains listed in the documentation. For each EE2 chain in the list, we postprocess them using COLA
"""
import os
import yaml
from copy import deepcopy
import sys

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
    if len(sys.argv) == 1:
        print("ERROR: you must provide chain indices to generate the postprocess")
        print("Example: python3 generate_postprocess.py 1-3,5-7,8,9,10 COLA")
        print("This would generate postprocess for chains 1, 2, 3, 5, 6, 7, 8, 9 and 10 using the COLA emulator")
        print("If the chain is not EE2, the postprocess is not generated.")
        exit(1)

    if len(sys.argv) == 2:
        print("ERROR: you must provide the emulator to postprocess the chains")
        print("Example: python3 generate_postprocess.py 1-3,5-7,8,9,10 COLA")
        print("This would generate postprocess for chains 1, 2, 3, 5, 6, 7, 8, 9 and 10 using the COLA emulator")
        print("If the chain is not EE2, the postprocess is not generated.")
        exit(1)

    # Parsing the indices
    inp = sys.argv[1].split(",")
    indices = []
    for entry in inp:
        if "-" in entry:
            low, high = list(map(int, entry.split("-")))
            for i in range(low, high+1): indices.append(i)
        else:
            indices.append(int(entry))
    
    # Parse the emulator
    emu = sys.argv[2]
    available_emus = {"COLA": 3, "EE2proj": 5, "EE2": 1}
    if emu not in available_emus:
        print(f"ERROR: emulator must be one of {list(available_emus.keys())} but you provided {emu}")
        exit(1)
    
    print(f"Generating postprocess yamls for chains {indices} using emulator {emu}")

    # Open documentation file
    with open("CHAINS.md", "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#"): continue
        fields = line[2:].split(" | ")
        index = int(fields[0])
        if index not in indices: continue
        emulator = fields[1].strip()
        fid_cosmo = fields[2].strip()
        scale_cuts = fields[3].strip()
        try:
            extra_info = fields[4]
        except IndexError:
            extra_info = None

        if emulator != "EE2":
            print(f"WARNING: requested postprocess of chain {index}, but this chain has the emulator {emulator}. Skipping...")
            continue

        if not os.path.exists(f"./yamls/MCMC{index}.yaml"):
            print(f"ERROR: Yaml for MCMC {index} does not exist, cannot generate postprocess file.")
            exit(1)
        
        with open(f"./yamls/MCMC{index}.yaml", "r") as f:
            mcmc_yaml = yaml.safe_load(f)
        
        assert mcmc_yaml["likelihood"]["lsst_y1.lsst_y1_cosmic_shear"]["non_linear_emul"] == 1, "The MCMC should be EE2"

        # NOTE: the `dict.copy` method returns shallow copies of a dictionary
        # Therefore, when I change manually the `non_linear_emul` field inside the `add` block,
        # it changes ALL occurrences of the `non_linear_emul` field, even in the `remove` block
        # and in the outside `likelihood` block, which may cause errors. The solution is to use `copy.deepcopy`. 
        postprocess_yaml = deepcopy(mcmc_yaml)
        postprocess_yaml.pop("sampler")
        postprocess_yaml["post"] = {
            "skip": 0.5,
            "thin": 15,
            "suffix": emu,
            "remove": {
                "theory": postprocess_yaml["theory"],
                "likelihood": deepcopy(mcmc_yaml["likelihood"]),
            },
            "add": {
                "theory": postprocess_yaml["theory"],
                "likelihood": deepcopy(mcmc_yaml["likelihood"]),
            },
        }
        postprocess_yaml["post"]["add"]["likelihood"]["lsst_y1.lsst_y1_cosmic_shear"]["non_linear_emul"] = available_emus[emu]

        with open(f"./yamls/POST{index}_{emu}.yaml", "w") as f:
            yaml.dump(postprocess_yaml, f, default_flow_style=False)
        