import sys
import numpy as np

# Loading cov data
print("Reading cov lsst_y1_cov")
cov_data = np.loadtxt("lsst_y1_cov")

if len(sys.argv) != 2:
    print(f"USAGE: {sys.argv[0]} <DATA_VECTOR>")
    print(f"    <DATA_VECTOR>: file to the data vector")
    exit(1)

data_vector_file = sys.argv[1]

# Loading data vector
print(f"Reading data vector {data_vector_file}")
try:
    data_vector = np.loadtxt(f"{data_vector_file}", usecols=(1,))
except FileNotFoundError:
    print(f"Could not load file {data_vector_file}")
    exit(1)

# Parsing cov data
dv_length = len(data_vector)
cov = np.zeros((dv_length, dv_length))
for line in cov_data:
    i = int(line[0])
    j = int(line[1])
    cov[i,j] = line[8] + line[9]
    cov[j,i] = cov[i,j]

# Generating noise realization
np.random.seed(1234) # Fix seed for reproducibility
noise_realization = np.random.multivariate_normal(mean=np.zeros((dv_length)), cov=cov)

# Adding noise realization to data vector
data_vector_with_noise = data_vector + noise_realization

# Saving noisy data vector with different name
# e.g. if the data vector is EE2_NO_NOISE.modelvector
# the new name will be EE2_NOISED.modelvector
new_name = data_vector_file.replace(".modelvector", "NOISED.modelvector")

assert new_name != data_vector_file, "Aborting operation: new file name is identical to original one."

# Saving noisy data vector to file
with open(new_name, "w") as f:
    for i, element in enumerate(data_vector_with_noise):
        f.write(f"{i} {element:.8e}\n")

print(f"Finished! Saved noisy data vector in {new_name}")