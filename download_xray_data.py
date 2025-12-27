import kagglehub

# Download latest version
print("Downloading X-ray dataset from Kaggle...")
path = kagglehub.dataset_download("prashant268/chest-xray-covid19-pneumonia")

print("Path to dataset files:", path)
