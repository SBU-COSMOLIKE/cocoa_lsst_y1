import numpy as np

# Loading cov data
print("Reading cov cov_lsst_y1")
cov_data = np.loadtxt("cov_lsst_y1")

# Loading data vector
print("Reading data vector EE2_FIDUCIAL.modelvector")
data_vector = np.loadtxt("EE2_FIDUCIAL.modelvector", usecols=(1,))

# Parsing cov data
dv_length = len(data_vector)
cov = np.zeros((dv_length, dv_length))
for line in cov_data:
    i = int(line[0])
    j = int(line[1])
    cov[i,j] = line[8] + line[9]
    cov[j,i] = cov[i,j]

# Generating noise realization
noise_realization = np.random.multivariate_normal(mean=np.zeros((dv_length)), cov=cov)

# Adding noise realization to data vector
data_vector_with_noise = data_vector + noise_realization

# Saving noisy data vector to file
with open("EE2_FIDUCIAL_NOISED.modelvector", "w") as f:
    for i, element in enumerate(data_vector_with_noise):
        f.write(f"{i} {element:.8e}\n")

print("Finished! Saved noisy data vector in EE2_FIDUCIAL_NOISED.modelvector")
