import os

templates = sorted([file for file in os.listdir() if file[:4] == "POST" and "_" not in file])

for template in templates:
    code = template.split(".")[0][4:]
    new_file_name = f"POST{code}_100.yaml"
    with open(template, "r") as f:
        contents = f.read()
    new_contents = contents.replace("suffix: COLA1", "suffix: COLA100")
    new_contents = new_contents.replace("num_refs: 1", "num_refs: 100")
    with open(new_file_name, "w") as f:
        f.write(new_contents)