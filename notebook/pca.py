import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from time import time
from sklearn.decomposition import IncrementalPCA

# -------------------------
# 1) PARAMETERS
# -------------------------
data_dir = "/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet_v2-multi_class-colon_bowel/training_samples/labels"  # folder containing .nii.gz
pattern = "*.nii.gz"
n_components = 50  # how many eigencolons to compute
batch_size = 100  # for IncrementalPCA
n_jobs = 64  # parallel workers (-1 = all cores)
output_mean = "/home/yb107/cvpr2025/DukeDiffSeg/notebook/data/mean_colon_atlas.nii.gz"
output_stats = "/home/yb107/cvpr2025/DukeDiffSeg/notebook/data/eigencolon_stats.npz"
# -------------------------


# -------------------------
# 2) LOAD AND PREPROCESS
# -------------------------
def load_and_mask(fn):
    # load volume
    img = nib.load(fn)
    arr = img.get_fdata()  # float64
    # binary colon mask
    colon = (arr == 1).astype(np.float64)
    # flatten to 1D
    return colon.ravel()


# gather files
fns = sorted(glob.glob(os.path.join(data_dir, pattern)))
N = len(fns)
V = 96 * 96 * 96

print(f"Found {N} volumes of shape {96, 96, 96} → {V} voxels each")
start = time()
# load all masks in parallel
X = Parallel(n_jobs=n_jobs)(delayed(load_and_mask)(fn) for fn in fns)
X = np.vstack(X)  # shape (N, V)
print(f"Loaded {N} volumes → data matrix {X.shape} in {time() - start:.2f} seconds")

# -------------------------
# 3) COMPUTE MEAN ATLAS
# -------------------------
# compute mean across all volumes
print("Computing mean atlas...")
start = time()
mean_colon = X.mean(axis=0)  # shape (V,)
# reshape & save
# use header/affine from the first image
hdr = nib.load(fns[0])
mean_img = nib.Nifti1Image(mean_colon.reshape(96, 96, 96), hdr.affine, hdr.header)
nib.save(mean_img, output_mean)
print(f"Saved mean atlas to {output_mean} in {time() - start:.2f} seconds")

# -------------------------
# 4) RUN PCA (Incremental)
# -------------------------
start = time()
print(f"Running Incremental PCA with {n_components} components and batch size {batch_size}...")
ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

# fit in batches
for start in range(0, N, batch_size):
    ipca.partial_fit(X[start:min(start + batch_size, N)])

# project to get components
scores = ipca.transform(X)  # (N, n_components)
components = ipca.components_  # (n_components, V)
explained = ipca.explained_variance_  # (n_components,)
evr = ipca.explained_variance_ratio_  # (n_components,)

# save stats
np.savez(
    output_stats, explained_variance=explained, components=components, scores=scores, explained_variance_ratio=evr
)
print(f"Saved PCA stats to {output_stats} in {time() - start:.2f} seconds")

# # -------------------------
# # 5) PLOT EIGENVALUES
# # -------------------------
# plt.figure(figsize=(8, 5))
# plt.plot(np.arange(1, n_components + 1), explained, marker="o")
# plt.title("Scree Plot: Eigencolon Variances")
# plt.xlabel("Component #")
# plt.ylabel("Eigenvalue")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
