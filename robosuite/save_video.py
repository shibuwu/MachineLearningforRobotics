import h5py
import numpy as np
import robosuite as suite
import imageio

hdf5_path = r"D:\MLR_CS4\robosuite\demo_data\PickPlaceCan\1777060653_8015628\demo.hdf5"
output_path = r"D:\MLR_CS4\robosuite\pickplacecan_video.mp4"
n_demos = 5
camera = "agentview"

f = h5py.File(hdf5_path, "r")
demos = list(f["data"].keys())
print(f"Total demos: {len(demos)}")

env = suite.make(
    "PickPlaceCan",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=camera,
    camera_heights=512,
    camera_widths=512,
    ignore_done=True,
)

writer = imageio.get_writer(output_path, fps=20)

for demo_key in demos[:n_demos]:
    print(f"Rendering {demo_key}...")
    env.reset()
    initial_state = f["data"][demo_key]["states"][0]
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()
    actions = f["data"][demo_key]["actions"][:]
    for action in actions:
        obs, _, _, _ = env.step(action)
        frame = obs[f"{camera}_image"][::-1]
        writer.append_data(frame)

writer.close()
env.close()
f.close()
print(f"Video saved to: {output_path}")