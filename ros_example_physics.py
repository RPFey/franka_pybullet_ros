from env_ros import FrankaPandaEnvRosPhysics
import pybullet as p
import argparse

args = argparse.ArgumentParser()
args.add_argument("--scene", type=str, default="grasp_sdf_env/clutter.sdf")
options = args.parse_args()

# object_from_list -> frequency: 850 Hz

env = FrankaPandaEnvRosPhysics(connection_mode=p.GUI,
                               frequency=1000.,
                               controller='position',
                               include_gripper=True,
                               simple_model=True,
                               object_from_sdf=options.scene,
                               object_from_list=False)

env.simulation_loop()
