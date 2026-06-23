"""Merge all PickPlaceCan demos (ours + teammates') into one HDF5 with train/val split."""

import h5py
import numpy as np
import os
import json

OUTPUT = os.path.expanduser("~/Desktop/MLR/Project4/datasets/all_pickplace_raw.hdf5")

SOURCES = [
    # Teammate's 47 demos
    os.path.expanduser("~/Desktop/MLR/Project4/datasets/teammate_pickplace_raw.hdf5"),
    # Our new gamepad demos (v1.5.2)
    os.path.expanduser("~/Desktop/MLR/Project4/robosuite/robosuite/models/assets/demonstrations_private/1777320626_2732935/demo.hdf5"),
    os.path.expanduser("~/Desktop/MLR/Project4/robosuite/robosuite/models/assets/demonstrations_private/1777321652_8750994/demo.hdf5"),
    os.path.expanduser("~/Desktop/MLR/Project4/robosuite/robosuite/models/assets/demonstrations_private/1777325355_774091/demo.hdf5"),
]

out = h5py.File(OUTPUT, "w")
grp = out.create_group("data")

demo_idx = 0
env_name = None
env_info = None

for src_path in SOURCES:
    if not os.path.exists(src_path):
        print(f"SKIP (not found): {src_path}")
        continue

    f = h5py.File(src_path, "r")
    demos = sorted([k for k in f["data"].keys() if k.startswith("demo")])
    print(f"{os.path.basename(src_path)}: {len(demos)} demos")

    if env_name is None:
        env_name = f["data"].attrs.get("env", None)
    if env_info is None:
        env_info = f["data"].attrs.get("env_info", None)

    for d in demos:
        demo_idx += 1
        src_grp = f["data"][d]
        dst_name = f"demo_{demo_idx}"
        dst_grp = grp.create_group(dst_name)

        for key in src_grp.keys():
            dst_grp.create_dataset(key, data=np.array(src_grp[key]))

        for attr_key, attr_val in src_grp.attrs.items():
            dst_grp.attrs[attr_key] = attr_val

    f.close()

print(f"\nTotal demos merged: {demo_idx}")

# metadata
import datetime
now = datetime.datetime.now()
grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
grp.attrs["repository_version"] = "1.5.2"
if env_name:
    grp.attrs["env"] = env_name
if env_info:
    grp.attrs["env_info"] = env_info

# 90/10 train/val split
num_demos = demo_idx
indices = list(range(1, num_demos + 1))
np.random.seed(42)
np.random.shuffle(indices)
n_train = int(0.9 * num_demos)
train_indices = sorted(indices[:n_train])
val_indices = sorted(indices[n_train:])

train_keys = [f"demo_{i}" for i in train_indices]
val_keys = [f"demo_{i}" for i in val_indices]

grp.attrs["train"] = json.dumps(train_keys)
grp.attrs["valid"] = json.dumps(val_keys)

print(f"Train: {len(train_keys)}, Val: {len(val_keys)}")

out.close()
print(f"Saved to {OUTPUT}")
