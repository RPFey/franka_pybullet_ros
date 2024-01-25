from env_ros import FrankaPandaEnvRosVisual
import pybullet as p

env = FrankaPandaEnvRosVisual(connection_mode=p.GUI,
                              frequency=30.,
                              include_gripper=True,
                              simple_model=False,
                              object_from_sdf=True,
                              object_from_list=False)

env.simulation_loop()
