from env_ros import FrankaPandaEnvRosPhysics
import pybullet as p

# object_from_list -> frequency: 850 Hz

env = FrankaPandaEnvRosPhysics(connection_mode=p.DIRECT,
                               frequency=1000.,
                               controller='velocity',
                               include_gripper=True,
                               simple_model=True,
                               object_from_sdf=True,
                               object_from_list=False)

env.simulation_loop()
