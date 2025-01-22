from env_ros import FrankaPandaEnvRosPhysics
import pybullet as p
import argparse

args = argparse.ArgumentParser()
args.add_argument("--scene", type=str, default="grasp_sdf_env/clutter.sdf")
args.add_argument("--remove_box", action="store_true", default=False)
args.add_argument("--load_list", action="store_true", default=False)
options = args.parse_args()
print(options.remove_box)

# object_from_list -> frequency: 850 Hz

if options.load_list:
    options.scene = None

env = FrankaPandaEnvRosPhysics(connection_mode=p.GUI,
                               frequency=1000.,
                               controller='position',
                               include_gripper=True,
                               simple_model=True,
                               object_from_sdf=options.scene,
                               remove_box=options.remove_box,
                               object_from_list=options.load_list)

env.simulation_loop()
