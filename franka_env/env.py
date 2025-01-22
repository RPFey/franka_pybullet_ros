import os
import pkgutil
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as sciR

from franka_env.panda_robot import FrankaPanda # type: ignore
import pybullet_utils.bullet_client as bc
import numpy as np

dirname = os.path.dirname(__file__)
ycb_database = os.path.join(dirname, "../", "ycb_objects")

class FrankaPandaEnv:
    def __init__(self, connection_mode=p.GUI, frequency=1000., controller='position',
                 include_gripper=True, simple_model=False,
                 object_from_sdf=None, object_from_list=None, remove_box=False):

        self.bc = bc.BulletClient(connection_mode)
        egl = pkgutil.get_loader('eglRenderer')
        self.bc.loadPlugin(egl.get_filename(), "eglRendererPlugin")
        self.bc.configureDebugVisualizer(rgbBackground=[33. / 255., 90. / 255., 127. / 255.])
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_GUI, 0)
        self.frequency = frequency
        self.remove_box = remove_box
        self.bc.setTimeStep(1 / frequency)
        self.bc.setGravity(0, 0, -9.81)

        self.plane_id = None
        self.table_id = None
        self.object_id = []
        self.object_list = ["YcbBanana", "YcbPear", "YcbHammer", "YcbScissors", "YcbStrawberry", "YcbChipsCan",
                            "YcbCrackerBox", "YcbFoamBrick", "YcbGelatinBox", "YcbMasterChefCan", "YcbMediumClamp", 
                            "YcbMustardBottle", "YcbPottedMeatCan", "YcbPowerDrill", "YcbTennisBall", "YcbTomatoSoupCan"]

        if object_from_list:
            self.add_ycb_objects_from_list(self.object_list)
        elif object_from_sdf:
            self.add_ycb_objects_from_sdf(object_from_sdf)

        self.controller = controller
        self.include_gripper = include_gripper
        self.simple_model = simple_model
        self.object_from_list = object_from_list
        self.object_from_sdf = object_from_sdf
        self.panda_robot = FrankaPanda(self.bc, include_gripper=include_gripper, simple_model=simple_model)
        
        # camera parameters
        self.camera_width = 800
        self.camera_height = 800
        self.camera_fovx = np.pi / 2
        self.camera_fovy = np.pi / 2
        self.focal_x = (self.camera_width / 2) / (np.tan(self.camera_fovx / 2))
        self.focal_y = (self.camera_height / 2) / (np.tan(self.camera_fovy / 2))
        self.camera_near = 0.02
        self.camera_far = 5.00
        self.K = np.array([[self.focal_x, 0., self.camera_width / 2], [0., self.focal_y, self.camera_height / 2], [0., 0., 1.]])
        self.projection_matrix = np.array([
            [2 / self.camera_width * self.K[0, 0], 0, (self.camera_width - 2 * self.K[0, 2]) / self.camera_width, 0],
            [0, 2 / self.camera_height * self.K[1, 1], (2 * self.K[1, 2] - self.camera_height) / self.camera_height, 0],
            [0, 0, (self.camera_near + self.camera_far) / (self.camera_near - self.camera_far),
             2 * self.camera_near * self.camera_far / (self.camera_near - self.camera_far)],
            [0, 0, -1, 0]]).T

    def simulate_step(self):
        self.bc.stepSimulation()

    def simulation_loop(self):
        import time
        while True:
            start = time.time()
            self.simulate_step()
            print(1 / (time.time() - start))
        
        self.bc.disconnect()
        print("Simulation end")

    def add_urdf_object(self, filename):
        flags = self.bc.URDF_USE_INERTIA_FROM_FILE
        
        noise_x = np.random.uniform(-0.05, 0.05)
        noise_y = np.random.uniform(-0.05, 0.05)
        self.object_id.append(self.bc.loadURDF(filename, [0.5 + noise_x, 0.0 + noise_y, 0.4], flags=flags))

    def add_ycb_objects_from_list(self, object_list, object_num=10):
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)

        self.plane_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
                                         basePosition=[0, 0, -0.65], useFixedBase=True)
        self.table_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),
                                         basePosition=[0.5, 0, -0.65], useFixedBase=True)
        self.box_id = self.bc.loadURDF(os.path.join(ycb_database, "OpenBox/box.urdf"),
                                        basePosition=[0.36, -0.15, -0.02], useFixedBase=True)
        self.bc.setAdditionalSearchPath(os.path.join(dirname, "grasp_sdf_env"))
        obj_idx = np.arange(len(object_list))
        np.random.shuffle(obj_idx)
        obj_idx = obj_idx[:object_num]

        print("Loading objects from a list")
        for i in obj_idx:
            obj = object_list[i]
            self.bc.setAdditionalSearchPath(os.path.join(ycb_database,  obj))
            self.add_urdf_object(os.path.join(ycb_database,  obj,  "model.urdf"))
            self.wait_for_objects_to_rest()
                
        if self.remove_box:
            self.bc.removeBody(self.box_id)
        
        np.set_printoptions(precision=4, suppress=True)
        for i, body in zip(obj_idx, self.object_id):
            pos, quat = self.bc.getBasePositionAndOrientation(body)
            pos = np.array(pos)
            euler = sciR.from_quat(quat).as_euler('xyz')
            print(object_list[i], pos, euler)
        np.set_printoptions(precision=8, suppress=False)
        
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)

    def add_ycb_objects_from_sdf(self, object_sdf):
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)

        self.plane_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
                                         basePosition=[0, 0, -0.65], useFixedBase=True)
        self.table_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),
                                         basePosition=[0.5, 0, -0.65], useFixedBase=True)
        
        print("Loading objects from a SDF")
        self.bc.setAdditionalSearchPath(os.path.join(dirname, "grasp_sdf_env"))
        self.bc.setAdditionalSearchPath(ycb_database)
        object_id_temp = self.bc.loadSDF(object_sdf)
        for id in object_id_temp:
            pos, orn = self.bc.getBasePositionAndOrientation(id)
            pos = (pos[0] + 0.9, pos[1] - 0.15, pos[2])
            self.bc.resetBasePositionAndOrientation(id, pos, orn)
            self.wait_for_objects_to_rest()
        
        self.object_id += object_id_temp
        # box should be the first object in sdf
        self.box_id = object_id_temp[0]
        if self.remove_box:
            self.bc.removeBody(self.box_id)
        
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)
    
    def remove_objects(self):
        self.bc.removeBody(self.plane_id)
        self.bc.removeBody(self.table_id)
        for obj in self.object_id:
            self.bc.removeBody(obj)
        self.object_id = []
    
    def reset(self):
        # Reset the simulation
        # self.bc.resetSimulation()
        self.bc.setGravity(0, 0, -9.81)
        self.bc.setTimeStep(1 / self.frequency)

        self.panda_robot.reset_state()
        self.panda_robot.open_gripper()
        
        self.remove_objects()
        
        # Reload objects
        self.object_id = []
        if self.object_from_list:
            self.add_ycb_objects_from_list(self.object_list)
        elif self.object_from_sdf:
            self.add_ycb_objects_from_sdf(self.object_from_sdf)
            
    def wait_for_objects_to_rest(self, tol=0.01):
        step = 0
        
        objects_resting = False
        while not objects_resting and step < 200:
            # simulate a quarter of a second
            for _ in range(100):
                self.bc.stepSimulation()
            
            # check whether all objects are resting
            objects_resting = True
            for body in self.object_id:
                vel = self.bc.getBaseVelocity(body)
                if np.linalg.norm(vel) > tol:
                    objects_resting = False
                    break
            
            step += 1
