"""
Utility script to generate the same yamls for different fiducial data vectors automatically.
"""

import shutil
import os

offset = 60

def replace_on_file(file_path, expr, new_expr):
    n = int(file_path[4:6])
    original_n = n - offset
    with open(file_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if f"MCMC{original_n}" in line:
            lines[i] = line.replace(f"MCMC{original_n}", f"MCMC{n}")
        if expr in line:
            lines[i] = line.replace(expr, new_expr)
    with open(file_path, "w") as f:
        f.writelines(lines)

for i in range(31, 31+offset):
    shutil.copyfile(f"MCMC{i}.yaml", f"MCMC{i+offset}.yaml")
    replace_on_file(f"MCMC{i+offset}.yaml", "HIGH.dataset", "LOW.dataset")