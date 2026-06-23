import h5py, numpy as np
path = "D:\\MLR_CS4\\robosuite\\demo_data\\PickPlaceCan\\demo_best50.hdf5"
f = h5py.File(path, "a")
demos = list(f["data"].keys())
np.random.seed(42)
np.random.shuffle(demos)
if "mask" not in f: f.create_group("mask")
for n in [5, 10, 20]:
    key = f"demo{n}"
    if key in f["mask"]: del f["mask"][key]
    f["mask"].create_dataset(key, data=np.array(demos[:n], dtype="S"))
    print(f"Created split: {key} with {n} demos")
f.close()
print("Done!")
