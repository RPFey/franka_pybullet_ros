from env_visual import FrankaPandaEnv
import pybullet as p

env = FrankaPandaEnv(connection_mode=p.GUI,
                     frequency=1000.,
                     include_gripper=True,
                     simple_model=False,
                     object_from_sdf=True,
                     object_from_list=False)

env.simulation_loop()
