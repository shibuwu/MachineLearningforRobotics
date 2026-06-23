import h5py, numpy as np

src = "D:\\MLR_CS4\\robosuite\\all_pickplace_obs.hdf5"
dst = "D:\\MLR_CS4\\robosuite\\pickplace_best100.hdf5"

f_src = h5py.File(src, "r")
demos = list(f_src["data"].keys())

demo_lengths = []
for demo in demos:
    length = f_src["data"][demo]["actions"].shape[0]
    reward = np.sum(f_src["data"][demo]["rewards"][:])
    demo_lengths.append((demo, length, reward))

demo_lengths.sort(key=lambda x: x[1])

print("Top 10 shortest demos:")
for name, length, reward in demo_lengths[:10]:
    print(f"  {name}: {length} steps, reward={reward:.2f}")

print(f"\nBottom 10 longest demos:")
for name, length, reward in demo_lengths[-10:]:
    print(f"  {name}: {length} steps, reward={reward:.2f}")

best100 = [d[0] for d in demo_lengths[:100]]
print(f"\nSelected {len(best100)} best demos")
print(f"Avg steps: {np.mean([d[1] for d in demo_lengths[:100]]):.1f}")
print(f"Max steps: {max([d[1] for d in demo_lengths[:100]])}")
print(f"Min steps: {min([d[1] for d in demo_lengths[:100]])}")

f_dst_file = h5py.File(dst, "w")
grp = f_dst_file.create_group("data")
for k, v in f_src["data"].attrs.items():
    grp.attrs[k] = v

for demo in best100:
    f_src.copy(f"data/{demo}", f_dst_file["data"])

np.random.seed(42)
np.random.shuffle(best100)
n_train = 90
train = best100[:n_train]
valid = best100[n_train:]

mask = f_dst_file.create_group("mask")
mask.create_dataset("train", data=np.array(train, dtype="S"))
mask.create_dataset("valid", data=np.array(valid, dtype="S"))

print(f"\nTrain: {len(train)}, Valid: {len(valid)}")
print(f"Saved to {dst}")
f_src.close()
f_dst_file.close()
print("Done!")
