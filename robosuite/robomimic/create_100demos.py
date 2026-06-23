import h5py, numpy as np

src = "D:\\MLR_CS4\\robosuite\\all_pickplace_obs.hdf5"
dst = "D:\\MLR_CS4\\robosuite\\pickplace_100demos.hdf5"

f_src = h5py.File(src, "r")
f_dst = h5py.File(dst, "w")

demos = list(f_src["data"].keys())
np.random.seed(42)
np.random.shuffle(demos)
selected = demos[:100]

grp = f_dst.create_group("data")
for k, v in f_src["data"].attrs.items():
    grp.attrs[k] = v

for demo in selected:
    f_src.copy(f"data/{demo}", f_dst["data"])

print(f"Copied {len(selected)} demos to {dst}")

np.random.shuffle(selected)
n_train = 90
train = selected[:n_train]
valid = selected[n_train:]

mask = f_dst.create_group("mask")
mask.create_dataset("train", data=np.array(train, dtype="S"))
mask.create_dataset("valid", data=np.array(valid, dtype="S"))
print(f"Train: {len(train)}, Valid: {len(valid)}")

f_src.close()
f_dst.close()
print("Done!")
