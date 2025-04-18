import os

mask = "M1"
template_path = "ee2_ee2ref_M1.dataset"
with open(template_path, "r") as f: template = f.read()

modelvectors = [file for file in os.listdir() if file.endswith(".modelvector")]

for modelvector in modelvectors:
    contents = template.replace("ee2_ee2ref.modelvector", modelvector)
    new_name = modelvector.removesuffix(".modelvector") + "_M1.dataset"
    with open(new_name, "w") as f: f.write(contents)