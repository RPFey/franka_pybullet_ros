# franka_pybullet_ros

This is a Pybullet simulation environment for the control and sensor perception of Franka Panda through ROS.

## Setup

```bash
mkdir pi-Physics && cd pi-Physics

# optional install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install pi0
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi 
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
uv pip install pybullet
cd ../

git clone -b pi0 https://github.com/RPFey/franka_pybullet_ros.git
git clone https://github.com/RPFey/grasp_episode.git

cd franka_pybullet_ros
uv pip install -e .

python pi0_sim.py --ep_file ../grasp_episode/seed4_ep2.json --ep_root ../grasp_episode
```

## Usage

Run 

```bash
python ros_example_physics.py --scene grasp_sdf_env/clutter.sdf

```
