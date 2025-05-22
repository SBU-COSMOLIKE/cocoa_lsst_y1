"""
    COLA w0wa postprocess generation script.
    Author: João Rebouças
    Description: This script generates `.yaml` files for postprocessing the chains listed in the documentation. For each EE2 chain in the list, we postprocess them using COLA
"""
import os
import yaml
from copy import deepcopy

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
    # Open documentation file
    with open("CHAINS.md", "r") as f:
        lines = f.readlines()

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

        if emulator != "EE2": continue

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
            "suffix": "POSTCOLA",
            "remove": {
                "theory": postprocess_yaml["theory"],
                "likelihood": deepcopy(mcmc_yaml["likelihood"]),
            },
            "add": {
                "theory": postprocess_yaml["theory"],
                "likelihood": deepcopy(mcmc_yaml["likelihood"]),
            },
        }
        postprocess_yaml["post"]["add"]["likelihood"]["lsst_y1.lsst_y1_cosmic_shear"]["non_linear_emul"] = 3

        with open(f"./yamls/POST{index}_COLA.yaml", "w") as f:
            yaml.dump(postprocess_yaml, f, default_flow_style=False)
        