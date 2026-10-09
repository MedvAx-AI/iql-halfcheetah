import hashlib
import json
import platform
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

env = gym.make("HalfCheetah-v5")
observation, _ = env.reset(seed=10000)
model = env.unwrapped.model
data = env.unwrapped.data
initial = np.concatenate([data.qpos.copy(), data.qvel.copy()])
actions = np.random.default_rng(10000).uniform(-1, 1, size=(1000, 6)).astype(np.float32)
arrays = [
    model.qpos0,
    model.body_mass,
    model.body_inertia,
    model.geom_size,
    model.geom_pos,
    model.geom_quat,
    model.actuator_gear,
    model.dof_damping,
]
model_hash = hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()
xml = Path(gym.__file__).parent / "envs/mujoco/assets/half_cheetah.xml"
result = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "mujoco": mujoco.__version__,
    "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
    "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
    "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
    "model_parameters_sha256": model_hash,
    "states": {},
    "rewards": {},
}
for step, action in enumerate(actions, 1):
    _, reward, _, _, _ = env.step(action)
    if step in [1, 10, 100, 1000]:
        result["states"][str(step)] = np.concatenate([data.qpos.copy(), data.qvel.copy()]).tolist()
        result["rewards"][str(step)] = reward
env.close()
print("SIMULATOR_PROBE_JSON " + json.dumps(result), flush=True)
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
